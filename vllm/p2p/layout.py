# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""p2p 部署清单 → vLLM 静态通道（P2P 移植阶段 A）。**纯 Python，不 import torch / vllm。**

    # 控制机上不装 vLLM 也能跑（按文件路径直接执行；本文件不依赖包内其它模块）
    python3 vllm/p2p/layout.py --plan ../p2p-framework/qwen3-30b/plan_es90_fill.json \\
        --hosts ../p2p-framework/task/hosts.txt \\
        --model-dir /home/ubuntu/p2p-framework/weights-q3-30b --out p2p_channels
    # 装了 vLLM 的环境里也可以
    python -m vllm.p2p.layout --plan … --hosts …

一条通道 = 本框架的一条前段 + 静态配给它的一条后段 = **一个 vLLM 流水线实例**：

    N01:1–8  →  N03:9–21  →  N05:22–48        VLLM_PP_LAYER_PARTITION=8,13,27
    node_rank 0  node_rank 1   node_rank 2       （单卡节点：node_rank = PP 级号）

这里产出三样东西，全部由清单**推出来**、不另做决定：

1. 每条通道的节点顺序与切层（`Channel.stages` / `partition`）；
2. 逐层驻留专家 → `--hf-overrides`（`vllm/model_executor/models/qwen3_moe_partial.py` 读它）；
3. 逐节点的 `vllm serve` 命令。

精度与注意力后端**全部通道统一**（默认 fp16 + TRITON_ATTN）：vLLM 在 sm75（T4 / RTX6000）上
不许 bf16、也用不了 FlashAttention，而对照实验要求五条通道、以及本框架的对照组
（`DTYPE=float16`）跑同一精度、同一套 kernel。`--dtype auto` 可退回「按卡型逐通道判」。

配对（哪条前段接哪条后段）与 p2p-framework `p2pmoe/deploy/control.py` 的静态模式同一口径：给了 `--wiring`
就按它（格式与 `--save-wiring` 导出的相同），否则用清单里存的组合矩阵按 t50 贪心。

层号口径：清单与本文件内部都是 **1-based**（规划器口径）；写进 hf_overrides 的是
**0-based**（vLLM / checkpoint 的 `model.layers.{i}`）。只在 `Channel.hf_overrides`
这一处转换 —— 搞错会静默把专家集合装到错的层上。
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "ARCH", "Stage", "Channel", "HostInfo", "LayoutError",
    "load_plan", "pair_static", "channels_from_plan", "parse_hosts",
    "parse_layer_experts", "localize_expert_name", "expert_key", "channel_dtype",
    "parse_gpu_mem_node",
]

ARCH = "Qwen3MoePartialForCausalLM"
"""部分专家模型的架构名（`vllm/model_executor/models/registry.py` 里登记的就是它）。"""

# 不支持 bf16 的卡（算力 < 8.0）。vLLM 0.14 在这些卡上直接拒绝 bf16
# （platforms/cuda.py: check_if_supports_dtype），而一个流水线实例只有一种 dtype ——
# 通道里只要有一台在这张表里，整条通道就得用 fp16。
SM75_OR_OLDER = ("T4", "RTX6000", "RTX 6000", "QUADRO", "V100", "P100", "P40", "T1000",
                 "RTX2080", "RTX 2080", "TITAN RTX")


class LayoutError(ValueError):
    """清单 / 配对 / hosts 自相矛盾。在生成命令之前就报，不留到节点上。"""


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stage:
    """流水线的一级 = 一台节点。"""

    node: str
    role: str
    """"front" 或 "back"。"""
    lo: int
    hi: int
    """本级的层区间，闭区间、1-based。"""
    layer_experts: Mapping[int, tuple[int, ...]]
    """1-based 层号 → 驻留专家（已排序）。"""

    @property
    def n_layers(self) -> int:
        return self.hi - self.lo + 1


@dataclass(frozen=True)
class Channel:
    name: str
    """如 gsm8k-0 —— task 加该 task 的序号。"""
    task: str
    front: str
    """前段的段 id（F0 …）。"""
    back: str
    """后段的段 id（Bgsm8k0 …）。"""
    stages: tuple[Stage, ...]
    n_layers: int
    n_experts: int

    @property
    def nodes(self) -> list[str]:
        return [s.node for s in self.stages]

    @property
    def partition(self) -> list[int]:
        return [s.n_layers for s in self.stages]

    @property
    def partition_env(self) -> str:
        return ",".join(str(x) for x in self.partition)

    def layer_experts(self) -> dict[int, tuple[int, ...]]:
        """整条通道的 1-based 层 → 驻留专家。"""
        out: dict[int, tuple[int, ...]] = {}
        for s in self.stages:
            out.update(s.layer_experts)
        return out

    def partial_layers(self) -> dict[int, tuple[int, ...]]:
        """只列**没装全**的层（1-based）。装全的层按原生 Qwen3-MoE 构造。"""
        return {l: es for l, es in self.layer_experts().items() if len(es) < self.n_experts}

    def hf_overrides(self, *, miss_policy: str = "local_topk",
                     stats: Mapping | None = None) -> dict:
        """给 `vllm serve --hf-overrides` 的 dict。键是 **0-based** 层号的字符串。

        `stats`：开 A5 统计钩子（`vllm/p2p/stats.py`），如 `{"lr_layers": [0, …, 7]}`。
        不给 = 不开，模型前向与没有钩子时逐位相同。"""
        d = {
            "architectures": [ARCH],
            "p2p_layer_experts": {str(l - 1): list(es)
                                  for l, es in sorted(self.partial_layers().items())},
            "p2p_miss_policy": miss_policy,
            "p2p_channel": self.name,
        }
        if stats is not None:
            d["p2p_stats"] = dict(stats)
        return d

    def resident_fraction(self) -> float:
        le = self.layer_experts()
        return sum(len(v) for v in le.values()) / (len(le) * self.n_experts)


# --------------------------------------------------------------------------- #
def load_plan(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("nodes", "segments", "l0"):
        if k not in d:
            raise LayoutError(f"{path} 不像部署清单：缺 {k!r}")
    return d


_MODEL_TAG = re.compile(r"(\d+)L-d(\d+)-E(\d+)")


def _model_shape(plan: Mapping, n_layers: int | None, n_experts: int | None) -> tuple[int, int]:
    """清单的 model 字段形如 "48L-d2048-E128"。显式给了就以显式为准，并核对。"""
    m = _MODEL_TAG.search(str(plan.get("model", "")))
    L = int(m.group(1)) if m else None
    E = int(m.group(3)) if m else None
    if n_layers is not None and L is not None and n_layers != L:
        raise LayoutError(f"清单是 {L} 层的模型，参数给的是 {n_layers} 层")
    if n_experts is not None and E is not None and n_experts != E:
        raise LayoutError(f"清单是每层 {E} 个专家，参数给的是 {n_experts}")
    L = n_layers if n_layers is not None else L
    E = n_experts if n_experts is not None else E
    if L is None or E is None:
        raise LayoutError("清单的 model 字段里读不出层数/专家数 —— 用 --n-layers / --n-experts 给")
    return L, E


def pair_static(plan: Mapping, wiring: Mapping | None = None) -> dict[str, tuple[str, str]]:
    """前段段 id → (后段段 id, task)。

    `wiring` 是 `control.py --save-wiring` 的格式 {"pairs": [{"front", "back", "task"?}]}，
    front/back 可以写段 id，也可以写节点 id。不给就按清单里的组合矩阵、t50 升序贪心 ——
    与 `control.py` 在 `--load-plan` 而无 `--wiring` 时的行为逐字一致。
    """
    segs = plan["segments"]
    by_node = {v: sid for sid, info in segs.items() for v in info["nodes"]}

    def sid_of(name: str, want: str) -> str:
        sid = name if name in segs else by_node.get(name)
        if sid is None:
            raise LayoutError(f"wiring 里的 {name!r} 既不是段 id 也不是清单里的节点")
        role = segs[sid]["role"]
        if (want == "front") != (role == "front"):
            raise LayoutError(f"wiring 里的 {name!r} 解析成 {sid}（{role}），不是{want}")
        return sid

    out: dict[str, tuple[str, str]] = {}
    if wiring is not None:
        pairs = wiring.get("pairs", wiring if isinstance(wiring, list) else None)
        if not pairs:
            raise LayoutError("wiring 里没有 pairs")
        used: set[str] = set()
        for i, it in enumerate(pairs):
            f, b = sid_of(str(it["front"]), "front"), sid_of(str(it["back"]), "back")
            task = segs[b]["role"].split(":", 1)[1]
            if it.get("task") and it["task"] != task:
                raise LayoutError(f"wiring 第 {i} 条写 task={it['task']}，但 {b} 是按 {task} 建的")
            if f in out or b in used:
                raise LayoutError(f"wiring 第 {i} 条：{f} 或 {b} 被用了两次 —— 静态通道一对一")
            out[f] = (b, task)
            used.add(b)
        return out

    pairings = plan.get("pairings") or []
    if not pairings:
        raise LayoutError("清单里没有组合矩阵（pairings），无法自动配对 —— 请给 --wiring")
    used_f: set[str] = set()
    used_b: set[str] = set()
    for pr in sorted(pairings, key=lambda x: float(x["t50"])):
        f, b = pr["front"], pr["back"]
        if f in used_f or b in used_b:
            continue
        out[f] = (b, pr["task"])
        used_f.add(f)
        used_b.add(b)
    return out


def channels_from_plan(
    plan: Mapping,
    wired: Mapping[str, tuple[str, str]],
    *,
    n_layers: int | None = None,
    n_experts: int | None = None,
) -> list[Channel]:
    """把清单 + 配对翻成通道。会核对：层区间首尾相接、覆盖 1..L、专家 id 合法。"""
    L, E = _model_shape(plan, n_layers, n_experts)
    plans = {p["node"]: p for p in plan["nodes"]}
    segs = plan["segments"]

    def stages_of(sid: str, role: str) -> list[Stage]:
        out = []
        for v in segs[sid]["nodes"]:
            p = plans.get(v)
            if p is None:
                raise LayoutError(f"段 {sid} 的节点 {v} 在清单的 nodes 里没有计划")
            le: dict[int, tuple[int, ...]] = {}
            for ld in p["layers"]:
                es = tuple(sorted(int(e) for e in ld["experts"]))
                if not es:
                    raise LayoutError(f"{v} 第 {ld['layer']} 层的驻留专家集为空")
                if len(set(es)) != len(es):
                    raise LayoutError(f"{v} 第 {ld['layer']} 层的专家 id 有重复")
                if es[0] < 0 or es[-1] >= E:
                    raise LayoutError(f"{v} 第 {ld['layer']} 层的专家 id 越界（0..{E - 1}）")
                le[int(ld["layer"])] = es
            ls = sorted(le)
            if ls != list(range(ls[0], ls[-1] + 1)):
                raise LayoutError(f"{v} 的层不连续：{ls}")
            out.append(Stage(node=v, role=role, lo=ls[0], hi=ls[-1], layer_experts=le))
        return out

    counter: dict[str, int] = {}
    chans: list[Channel] = []
    # 按 task、再按后段 id 排，让通道编号稳定（gsm8k-0 永远是 Bgsm8k0 那条，若它被配上）
    for f, (b, task) in sorted(wired.items(), key=lambda kv: (kv[1][1], kv[1][0])):
        if segs[f]["role"] != "front":
            raise LayoutError(f"{f} 不是前段")
        st = stages_of(f, "front") + stages_of(b, "back")
        nxt = 1
        for s in st:
            if s.lo != nxt:
                raise LayoutError(
                    f"通道 {f}×{b}：{s.node} 从第 {s.lo} 层开始，但上一级只到第 {nxt - 1} 层"
                    f" —— 层没接上（漏层或重叠）")
            nxt = s.hi + 1
        if nxt - 1 != L:
            raise LayoutError(f"通道 {f}×{b} 只覆盖到第 {nxt - 1} 层，模型有 {L} 层")
        k = counter.get(task, 0)
        counter[task] = k + 1
        chans.append(Channel(name=f"{task}-{k}", task=task, front=f, back=b,
                             stages=tuple(st), n_layers=L, n_experts=E))
    used = [n for c in chans for n in c.nodes]
    dup = sorted({n for n in used if used.count(n) > 1})
    if dup:
        raise LayoutError(f"节点 {dup} 出现在多条通道里 —— 一台卡只能是一个实例的一级")
    return chans


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HostInfo:
    node: str
    host: str
    gpu: str = ""
    """hosts.txt 行尾注释里的第一个词（「# A30 24GB」→ A30）。只用来判 dtype。"""


def parse_hosts(path: str | Path) -> dict[str, HostInfo]:
    """`task/hosts.txt`：`节点id  地址[:端口]  [附加参数]  [# 卡型 …]`。

    端口是 p2p agent 的，vLLM 用不上，只取地址。
    """
    out: dict[str, HostInfo] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        body, _, comment = raw.partition("#")
        parts = body.split()
        if len(parts) < 2:
            continue
        host = parts[1].rsplit(":", 1)[0] if parts[1].count(":") == 1 else parts[1]
        gpu = comment.split()[0] if comment.split() else ""
        out[parts[0]] = HostInfo(node=parts[0], host=host, gpu=gpu)
    return out


def channel_dtype(ch: Channel, hosts: Mapping[str, HostInfo],
                  sm75: Sequence[str] = ()) -> tuple[str, list[str]]:
    """→ (dtype, 迫使降到 fp16 的节点)。`sm75` 可额外点名（hosts 里没写卡型时）。"""
    old = []
    for v in ch.nodes:
        g = (hosts.get(v).gpu if hosts.get(v) else "").upper().replace("-", "")
        if v in sm75 or any(g.startswith(x.replace(" ", "").replace("-", ""))
                            for x in SM75_OR_OLDER):
            old.append(v)
    return ("half" if old else "bfloat16"), old


# --------------------------------------------------------------------------- #
# 模型侧用的两个纯函数（放这里是为了不装 vLLM 也能测）
# --------------------------------------------------------------------------- #
def parse_layer_experts(raw: Mapping | None, n_layers: int,
                        n_experts: int) -> dict[int, tuple[int, ...]]:
    """hf_overrides 里的 `p2p_layer_experts` → {0-based 层: 驻留专家（升序）}。

    没列出的层表示全装。这里严格核对 —— 写错的后果是装错专家，而那**不会报错**，
    只会让输出悄悄变差。
    """
    out: dict[int, tuple[int, ...]] = {}
    for k, v in (raw or {}).items():
        try:
            l = int(k)
        except (TypeError, ValueError):
            raise LayoutError(f"p2p_layer_experts 的键 {k!r} 不是层号") from None
        if not 0 <= l < n_layers:
            raise LayoutError(f"p2p_layer_experts 的层号 {l} 越界（0-based，0..{n_layers - 1}）")
        es = sorted(int(e) for e in v)
        if not es:
            raise LayoutError(f"p2p_layer_experts[{l}] 为空 —— 至少要驻留一个专家")
        if len(set(es)) != len(es):
            raise LayoutError(f"p2p_layer_experts[{l}] 有重复的专家 id")
        if es[0] < 0 or es[-1] >= n_experts:
            raise LayoutError(f"p2p_layer_experts[{l}] 的专家 id 越界（0..{n_experts - 1}）")
        out[l] = tuple(es)
    return out


# 前缀可有可无：checkpoint 里是 model.layers.…，vLLM 的 AutoWeightsLoader 往下传时会剥成 layers.…
_EXPERT_KEY = re.compile(r"^((?:.*\.)?layers\.(\d+)\.mlp\.experts\.)(\d+)(\..*)$")


_EXPERT_PROJ = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.")


def expert_key(name: str) -> tuple[int, int, str] | None:
    """专家权重名 → (层, 专家号, gate_proj/up_proj/down_proj)；不是专家权重返回 None。

    用 `search` 而不是 `match`：checkpoint 里的名字带 `model.` 前缀，vLLM 往下传时又可能
    剥掉 —— 两种都要认。（插件第一版用了 `match`，带前缀的名字一个都没认出来，
    逐专家核对因此把所有专家都报成了缺失。）
    """
    m = _EXPERT_PROJ.search(name)
    return None if m is None else (int(m.group(1)), int(m.group(2)), m.group(3))


def localize_expert_name(name: str,
                         layer_experts: Mapping[int, Sequence[int]]) -> str | None:
    """checkpoint 里的专家权重名：全局专家号 → 本层的本地号；不驻留则返回 None（跳过）。

        model.layers.20.mlp.experts.37.up_proj.weight
        └ 第 20 层驻留 [3, 12, 37, …] → 37 是第 2 个 → …experts.2.up_proj.weight

    非专家权重、以及装全的层，名字原样返回。本地号按驻留集**升序**排 —— 模型构造
    `FusedMoE` 时用的是同一个顺序（见 `qwen3_moe_partial.py`）。
    """
    m = _EXPERT_KEY.match(name)
    if m is None:
        return name
    layer, e = int(m.group(2)), int(m.group(3))
    es = layer_experts.get(layer)
    if es is None:
        return name
    try:
        local = list(es).index(e)
    except ValueError:
        return None
    return f"{m.group(1)}{local}{m.group(4)}"


# --------------------------------------------------------------------------- #
def serve_commands(
    ch: Channel,
    hosts: Mapping[str, HostInfo],
    *,
    model_dir: str,
    overrides_path: str,
    dtype: str,
    vllm_bin: str = "vllm",
    port: int = 8000,
    master_port: int = 29500,
    max_model_len: int = 4096,
    gpu_mem: float = 0.90,
    gpu_mem_node: Mapping[str, float] | None = None,
    attention_backend: str | None = "TRITON_ATTN",
    prefix_caching: bool = False,
    max_num_seqs: int | None = None,
    extra: Sequence[str] = (),
) -> list[tuple[str, str]]:
    """逐节点的启动命令 → [(节点, 命令)]，按 node_rank 排。

    node_rank 0（前段节点）起 API 服务器，其余 `--headless`。三台的参数除 node_rank、
    headless 与逐节点的显存比例（`gpu_mem_node`）外必须一字不差 —— 所以由同一处生成。

    `gpu_mem_node`：个别节点要给别的进程让显存时用（如 A4 的识别服务与前段 vLLM 同卡）。
    显存比例是**每台各自**生效的（每台 worker 按本机的 `--gpu-memory-utilization` 算 KV），
    所以只改那一台、不影响其余各级。

    `prefix_caching` 默认关：本框架没有前缀缓存（每条 prompt 整段 prefill），开了的话同一
    模板前缀会被复用，TTFT 不可比；A5 的统计也只能看到真正算过的 token。
    """
    missing = [v for v in ch.nodes if v not in hosts]
    if missing:
        raise LayoutError(f"hosts 里没有 {missing} 的地址")
    head = hosts[ch.nodes[0]].host
    out = []
    for rank, v in enumerate(ch.nodes):
        args = [
            vllm_bin, "serve", model_dir,
            "--served-model-name", ch.name,
            "--pipeline-parallel-size", str(len(ch.stages)),
            "--nnodes", str(len(ch.stages)), "--node-rank", str(rank),
            "--master-addr", head, "--master-port", str(master_port),
            "--dtype", dtype,
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", f"{(gpu_mem_node or {}).get(v, gpu_mem):g}",
            *([] if prefix_caching else ["--no-enable-prefix-caching"]),
            *(["--max-num-seqs", str(max_num_seqs)] if max_num_seqs else []),
            *extra,
        ]
        if attention_backend:
            # 默认所有通道都用 TRITON_ATTN：sm75 上 FlashAttention 不可用（要 sm80），而
            # 对照实验要求五条通道跑同一套 kernel —— A30 的通道也不例外。
            args += ["--attention-backend", attention_backend]
        if rank == 0:
            args += ["--host", "0.0.0.0", "--port", str(port)]
        else:
            args += ["--headless"]
        cmd = " ".join(shlex.quote(a) for a in args)
        env = f"VLLM_PP_LAYER_PARTITION={ch.partition_env}"
        cmd = f'{env} {cmd} --hf-overrides "$(cat {shlex.quote(overrides_path)})"'
        out.append((v, cmd))
    return out


def parse_gpu_mem_node(items: Sequence[str]) -> dict[str, float]:
    """["N01=0.5", "N09=0.6"] → {"N01": 0.5, "N09": 0.6}。"""
    out: dict[str, float] = {}
    for it in items:
        node, eq, val = str(it).partition("=")
        try:
            f = float(val)
        except ValueError:
            f = -1.0
        if not eq or not node.strip() or not 0.0 < f <= 1.0:
            raise LayoutError(f"--gpu-mem-node {it!r} 要写成 节点=比例（0 < 比例 ≤ 1），如 N01=0.5")
        out[node.strip()] = f
    return out


def describe(ch: Channel, dtype: str = "", forced: Sequence[str] = ()) -> str:
    segs = "  →  ".join(
        f"{s.node}:{s.lo}–{s.hi}"
        + ("" if s.role == "front" and all(len(e) == ch.n_experts
                                           for e in s.layer_experts.values())
           else f"（层均 {sum(len(e) for e in s.layer_experts.values()) / s.n_layers:.1f}"
                f"/{ch.n_experts}）")
        for s in ch.stages)
    tail = f"  dtype={dtype}" + (f"（因 {','.join(forced)} 不支持 bf16）" if forced else "")
    return (f"{ch.name:<12} {ch.front}×{ch.back}  PP={ch.partition_env}\n"
            f"    {segs}\n    驻留 {ch.resident_fraction():.1%} 的专家{tail if dtype else ''}")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, type=Path, help="部署清单 JSON")
    ap.add_argument("--wiring", type=Path, default=None,
                    help="前后段配对（control.py --save-wiring 的格式）；不给就按 t50 贪心")
    ap.add_argument("--hosts", type=Path, default=None, help="节点 id → 地址（task/hosts.txt）")
    ap.add_argument("--model-dir", default="$WEIGHTS",
                    help="各节点上的 checkpoint 目录（同一路径；可以是 fetch 拉下的部分 checkpoint）")
    ap.add_argument("--out", type=Path, default=Path("p2p_channels"),
                    help="写出 <通道>.overrides.json、channels.json、<通道>.sh")
    ap.add_argument("--remote-out", default=None,
                    help="节点上读 overrides 的路径前缀（默认 = --out，假设代码目录同步过去）")
    ap.add_argument("--miss-policy", default="local_topk", choices=("local_topk",))
    ap.add_argument("--sm75", default="", help="额外点名不支持 bf16 的节点，逗号分隔")
    ap.add_argument("--dtype", default="half", choices=("half", "bfloat16", "auto"),
                    help="默认 half：**所有通道统一 fp16**，与本框架的对照组（DTYPE=float16）"
                         "同一精度才公平。auto：按卡型判，有 sm75 卡的通道 half、其余 bfloat16")
    ap.add_argument("--attention-backend", default="TRITON_ATTN",
                    help="所有通道统一的注意力后端（默认 TRITON_ATTN，sm75 唯一能用的）；"
                         "给空串则交给 vLLM 自选（各通道可能不同）")
    ap.add_argument("--vllm-bin", default="vllm")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--master-port", type=int, default=29500)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--gpu-mem-node", action="append", default=[], metavar="节点=比例",
                    help="逐节点覆盖 --gpu-mem，可重复，如 --gpu-mem-node N01=0.5"
                         "（A4 的识别服务与 N01 的前段同卡时给它让显存）")
    ap.add_argument("--prefix-caching", action="store_true",
                    help="开 vLLM 的前缀缓存（默认关：本框架没有前缀缓存，开了 TTFT 不可比）")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="每条通道同时调度的请求数上限（vLLM 默认 256）。显存压得很紧的末级"
                         "（N05/N07/N08/N13）在采样器预热（按这个数造假请求）时可能 OOM，降到 128 能省出几百 MB")
    ap.add_argument("--stats", action="store_true",
                    help="开 A5 统计钩子：逐请求的全量路由 top-k、后段 miss、前 L₀ 层 LR 特征，"
                         "落到各节点的 $VLLM_P2P_STATS_DIR（默认 /tmp/p2p_stats）")
    ap.add_argument("--lr-layers", default=None,
                    help="--stats 时按 body token 聚合哪些层（0-based，逗号分隔）；默认前 L₀ 层")
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-experts", type=int, default=None)
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    wiring = json.loads(args.wiring.read_text(encoding="utf-8")) if args.wiring else None
    try:
        chans = channels_from_plan(plan, pair_static(plan, wiring),
                                   n_layers=args.n_layers, n_experts=args.n_experts)
    except LayoutError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    hosts = parse_hosts(args.hosts) if args.hosts else {}
    sm75 = [x.strip() for x in args.sm75.split(",") if x.strip()]
    try:
        gmn = parse_gpu_mem_node(args.gpu_mem_node)
    except LayoutError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2
    known = {v for ch in chans for v in ch.nodes}
    stray = sorted(set(gmn) - known)
    if stray:
        print(f"✗ --gpu-mem-node 里的 {stray} 不在任何通道里（有的是 {sorted(known)}）",
              file=sys.stderr)
        return 2
    stats = None
    if args.stats:
        lr = ([int(x) for x in args.lr_layers.split(",") if x.strip()]
              if args.lr_layers else list(range(int(plan["l0"]))))
        stats = {"lr_layers": lr}

    args.out.mkdir(parents=True, exist_ok=True)
    remote = (args.remote_out or str(args.out)).rstrip("/")
    summary = []
    print(f"{len(chans)} 条通道（L₀={plan['l0']}，{'按 --wiring' if wiring else '按 t50 贪心配对'}）")
    for ch in chans:
        auto, old = channel_dtype(ch, hosts, sm75)
        if args.dtype == "auto":
            dtype, why = auto, old
        else:
            if args.dtype == "bfloat16" and old:
                print(f"✗ {ch.name}：{old} 不支持 bf16，不能 --dtype bfloat16",
                      file=sys.stderr)
                return 2
            dtype, why = args.dtype, []
        print(describe(ch, dtype, why))
        ov = args.out / f"{ch.name}.overrides.json"
        ov.write_text(json.dumps(ch.hf_overrides(miss_policy=args.miss_policy, stats=stats),
                                 separators=(",", ":")), encoding="utf-8")
        entry = {"name": ch.name, "task": ch.task, "front": ch.front, "back": ch.back,
                 "nodes": ch.nodes, "partition": ch.partition, "dtype": dtype,
                 "attention_backend": args.attention_backend or None,
                 "port": args.port, "overrides": str(ov),
                 "gpu_mem": {v: gmn.get(v, args.gpu_mem) for v in ch.nodes},
                 "prefix_caching": bool(args.prefix_caching),
                 "stats": stats,
                 "stages": [{"node": s.node, "role": s.role, "layers": [s.lo, s.hi],
                             "n_experts": round(sum(len(e) for e in s.layer_experts.values())
                                                / s.n_layers, 1)} for s in ch.stages]}
        if hosts:
            cmds = serve_commands(ch, hosts, model_dir=args.model_dir,
                                  overrides_path=f"{remote}/{ov.name}", dtype=dtype,
                                  vllm_bin=args.vllm_bin, port=args.port,
                                  master_port=args.master_port,
                                  max_model_len=args.max_model_len, gpu_mem=args.gpu_mem,
                                  gpu_mem_node=gmn,
                                  attention_backend=args.attention_backend or None,
                                  prefix_caching=args.prefix_caching,
                                  max_num_seqs=args.max_num_seqs)
            entry["api"] = f"http://{hosts[ch.nodes[0]].host}:{args.port}/v1"
            entry["commands"] = [{"node": v, "host": hosts[v].host, "cmd": c} for v, c in cmds]
            sh = args.out / f"{ch.name}.sh"
            sh.write_text(
                "# 逐节点执行（node_rank 0 最后起也行 —— 它会等其余各级连上）\n"
                + "".join(f"\n# {v} @ {hosts[v].host}  (node_rank {i})\n{c}\n"
                          for i, (v, c) in enumerate(cmds)), encoding="utf-8")
            print(f"    API {entry['api']}   命令见 {sh}")
        summary.append(entry)
    (args.out / "channels.json").write_text(
        json.dumps({"plan": str(args.plan), "l0": plan["l0"],
                    "n_layers": chans[0].n_layers if chans else None,
                    "n_experts": chans[0].n_experts if chans else None,
                    "model_dir": args.model_dir, "channels": summary},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写 {args.out}/channels.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
