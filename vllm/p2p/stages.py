# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1：把 A 的静态通道（channels.json）拆成 B-1 的分段引擎清单（stages.json）。**纯 Python。**

    python3 vllm/p2p/stages.py --channels p2p_channels/channels.json --out p2p_channels/stages.json
    python3 vllm/p2p/stages.py … --only gsm8k-0,gsm8k-1          # 只要这几条通道的后段（及其前段）
    python3 vllm/p2p/launch.py --channels p2p_channels/stages.json start all    # 一样的 launch

每条通道「前段节点 → 后段节点们」拆成：

* 一个**前段引擎** `front-<节点>`：只建层 [0, L₀)，全专家（不分 task，所有请求共用）；单卡；
* 一个**后段引擎** `back-<通道>`：层 [L₀, L)，驻留集、各节点的切层（`VLLM_PP_LAYER_PARTITION`
  取通道原来的后几项）、显存比例、`--max-num-seqs` 都与 A 的那条通道相同；PP 跨原来的后段节点。

每个引擎的首节点上跑 `vllm.p2p.stage_agent`（EngineCore + 给协调器的 ZMQ 入口），其余节点跑同一程序的
headless 模式。生成的条目与 channels.json 同构（name / nodes / api / commands），所以
`launch.py --channels stages.json start|status|logs|errors|stop|gpu` 原样可用；进程按
`--served-model-name <引擎名>` 认。

`--hf-overrides` 直接内联进命令（不经文件，节点上不用同步 overrides）。
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Mapping, Sequence

__all__ = ["stages_from_channels", "agent_command", "main"]

ARCH = "Qwen3MoePartialForCausalLM"


def _flag(cmd: str, name: str, default: str | None = None) -> str | None:
    toks = shlex.split(cmd.split(" --hf-overrides ", 1)[0])
    for i, t in enumerate(toks):
        if t == name and i + 1 < len(toks):
            return toks[i + 1]
    return default


def _venv(cmd: str) -> str:
    for tok in shlex.split(cmd.split(" --hf-overrides ", 1)[0]):
        if "=" in tok and not tok.startswith("/"):
            continue
        for exe in ("/bin/vllm", "/bin/python", "/bin/python3"):
            if tok.endswith(exe):
                return tok[: -len(exe)]
        break
    raise ValueError(f"认不出命令里的 venv：{cmd[:120]}")


def agent_command(*, python: str, name: str, model_dir: str, role: str, layers: Sequence[int],
                  task: str | None, nodes: Sequence[str], rank: int, master: str,
                  master_port: int, partition: Sequence[int] | None, dtype: str,
                  max_model_len: int, gpu_mem: float, max_num_seqs: int | None,
                  attention_backend: str | None, port: int, overrides: Mapping,
                  eager: bool = False) -> str:
    args = [python, "-m", "vllm.p2p.stage_agent",
            "--served-model-name", name, "--model", model_dir, "--role", role,
            "--layers", f"{int(layers[0])},{int(layers[1])}"]
    if task:
        args += ["--task", task]
    args += ["--pipeline-parallel-size", str(len(nodes))]
    if len(nodes) > 1:
        args += ["--nnodes", str(len(nodes)), "--node-rank", str(rank),
                 "--master-addr", master, "--master-port", str(master_port)]
    else:
        args += ["--node-rank", "0"]
    args += ["--dtype", dtype, "--max-model-len", str(max_model_len),
             "--gpu-memory-utilization", f"{gpu_mem:g}"]
    if max_num_seqs:
        args += ["--max-num-seqs", str(max_num_seqs)]
    if attention_backend:
        args += ["--attention-backend", attention_backend]
    if eager:
        args.append("--enforce-eager")
    args += ["--port", str(port)]
    cmd = " ".join(shlex.quote(a) for a in args)
    if partition and len(nodes) > 1:
        cmd = f"VLLM_PP_LAYER_PARTITION={','.join(str(int(x)) for x in partition)} {cmd}"
    # --hf-overrides 放最后（launch.venv_of / _flag 按它切）；内联 JSON
    return f"{cmd} --hf-overrides {shlex.quote(json.dumps(overrides, separators=(',', ':')))}"


def stages_from_channels(cj: Mapping, *, only: Sequence[str] | None = None,
                         front_max_num_seqs: int = 64, front_gpu_mem: float | None = None,
                         port: int = 8300, master_port: int = 29600,
                         overrides_of=None, eager: bool = False) -> dict:
    """channels.json 的内容 → stages.json 的内容。`overrides_of(channel) -> dict` 读通道的
    hf_overrides（默认读 channel["overrides"] 指的文件）。"""
    l0, n_layers, model_dir = int(cj["l0"]), int(cj["n_layers"]), cj["model_dir"]
    chans = [c for c in cj["channels"] if not only or c["name"] in only]
    if only:
        miss = sorted(set(only) - {c["name"] for c in chans})
        if miss:
            raise ValueError(f"channels.json 里没有通道 {miss}")
    if not chans:
        raise ValueError("一条通道都没有")
    if overrides_of is None:
        def overrides_of(c):
            return json.loads(Path(c["overrides"]).read_text(encoding="utf-8"))

    fronts: dict[str, dict] = {}
    backs: list[dict] = []
    for c in chans:
        if "commands" not in c:
            raise ValueError(f"{c['name']} 没有启动命令 —— 生成 channels.json 时要给 layout.py 传 --hosts")
        head_cmd = c["commands"][0]["cmd"]
        venv = _venv(head_cmd)
        python = f"{venv}/bin/python"
        mml = int(_flag(head_cmd, "--max-model-len", "4096"))
        hosts = {st["node"]: st["host"] for st in c["commands"]}
        ov = overrides_of(c)
        part = list(c["partition"])
        if part[0] != l0:
            raise ValueError(f"{c['name']} 的第一级是 {part[0]} 层，与 L₀={l0} 不符 —— 前段必须正好是前 L₀ 层")
        gm = dict(c.get("gpu_mem") or {})

        # ---- 前段：每个前段节点一个，所有通道共用 ----
        fnode = c["nodes"][0]
        if fnode not in fronts:
            fname = f"front-{fnode}"
            fov = {"architectures": [ARCH], "p2p_layer_experts": {}, "p2p_channel": fname,
                   "p2p_miss_policy": ov.get("p2p_miss_policy", "local_topk")}
            if ov.get("p2p_stats") is not None:
                fov["p2p_stats"] = ov["p2p_stats"]
            fgm = front_gpu_mem if front_gpu_mem is not None else gm.get(fnode, 0.9)
            fronts[fnode] = {
                "name": fname, "role": "front", "task": None, "nodes": [fnode],
                "layers": [0, l0], "partition": [l0], "dtype": c["dtype"],
                "gpu_mem": {fnode: fgm}, "max_num_seqs": front_max_num_seqs,
                "api": f"http://{hosts[fnode]}:{port}/v1", "zmq": f"tcp://{hosts[fnode]}:{port + 1}",
                "from_channels": [c["name"]],
                "commands": [{"node": fnode, "host": hosts[fnode], "cmd": agent_command(
                    python=python, name=fname, model_dir=model_dir, role="front",
                    layers=(0, l0), task=None, nodes=[fnode], rank=0, master=hosts[fnode],
                    master_port=master_port, partition=None, dtype=c["dtype"],
                    max_model_len=mml, gpu_mem=fgm, max_num_seqs=front_max_num_seqs,
                    attention_backend=c.get("attention_backend"), port=port, overrides=fov,
                    eager=eager)}],
            }
        else:
            fronts[fnode]["from_channels"].append(c["name"])

        # ---- 后段：每条通道一个 ----
        bnodes = list(c["nodes"][1:])
        bname = f"back-{c['name']}"
        bov = dict(ov, p2p_channel=bname)
        bov.pop("p2p_stage", None)
        master = hosts[bnodes[0]]
        backs.append({
            "name": bname, "role": "back", "task": c["task"], "nodes": bnodes,
            "layers": [l0, n_layers], "partition": part[1:], "dtype": c["dtype"],
            "gpu_mem": {v: gm.get(v, 0.9) for v in bnodes},
            "max_num_seqs": c.get("max_num_seqs"),
            "api": f"http://{master}:{port}/v1", "zmq": f"tcp://{master}:{port + 1}",
            "from_channels": [c["name"]],
            "commands": [{"node": v, "host": hosts[v], "cmd": agent_command(
                python=python, name=bname, model_dir=model_dir, role="back",
                layers=(l0, n_layers), task=c["task"], nodes=bnodes, rank=k, master=master,
                master_port=master_port, partition=part[1:], dtype=c["dtype"],
                max_model_len=mml, gpu_mem=gm.get(v, 0.9), max_num_seqs=c.get("max_num_seqs"),
                attention_backend=c.get("attention_backend"), port=port, overrides=bov,
                eager=eager)} for k, v in enumerate(bnodes)],
        })

    fl = list(fronts.values())
    return {"kind": "tiered", "plan": cj.get("plan"), "l0": l0, "n_layers": n_layers,
            "n_experts": cj.get("n_experts"), "model_dir": model_dir,
            "fronts": [f["name"] for f in fl], "backs": [b["name"] for b in backs],
            # 叫 channels 是为了 launch.py / bench.py 原样能读
            "channels": fl + backs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--channels", default="/home/ubuntu/vllm-p2p/p2p_channels/channels.json")
    ap.add_argument("--out", type=Path, default=None, help="默认 = channels.json 旁边的 stages.json")
    ap.add_argument("--only", default=None, help="只要这些通道（逗号分隔）的后段与它们的前段")
    ap.add_argument("--front-max-num-seqs", type=int, default=64,
                    help="前段同时容纳的请求数：一条请求从进来到结束都占着前段的一个位置")
    ap.add_argument("--front-gpu-mem", type=float, default=None,
                    help="前段的显存比例（默认沿用通道里前段节点的；B 不跑识别服务时可给 0.9）")
    ap.add_argument("--port", type=int, default=8300, help="代理的 HTTP 端口（ZMQ = 它 + 1）")
    ap.add_argument("--master-port", type=int, default=29600)
    ap.add_argument("--eager", action="store_true", help="所有分段引擎 --enforce-eager（排错用）")
    a = ap.parse_args(argv)
    src = Path(a.channels)
    cj = json.loads(src.read_text(encoding="utf-8"))
    only = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else None
    try:
        st = stages_from_channels(cj, only=only, front_max_num_seqs=a.front_max_num_seqs,
                                  front_gpu_mem=a.front_gpu_mem, port=a.port,
                                  master_port=a.master_port, eager=a.eager)
    except (ValueError, OSError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    out = a.out or src.with_name("stages.json")
    out.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(st['fronts'])} 个前段 {st['fronts']}，{len(st['backs'])} 个后段：")
    for c in st["channels"]:
        print(f"  {c['name']:<20} {c['role']:<5} task={c['task'] or '—':<10} 层 {c['layers']}  "
              f"节点 {'+'.join(c['nodes'])}  PP 切层 {c['partition']}  显存 {c['gpu_mem']}  "
              f"max_num_seqs {c['max_num_seqs']}  {c['zmq']}")
    print(f"已写 {out}\n拉起：python3 vllm/p2p/launch.py --channels {out} start all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
