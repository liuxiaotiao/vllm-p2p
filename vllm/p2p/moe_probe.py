# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""用 vLLM 自己采专家激活画像（GLM-4.7-Flash 等新模型用；Qwen3-30B 那份是 HF 上的 moe_prefill_probe 采的）。

产物与 ``moe_prefill_probe.py`` 的输出目录**同格式**，p2p-framework 下游的工具原样能读：

* ``{task}_prefill_persample.npz`` —— 逐样本 ``counts_full / wsum_full / tokens_full``（整条 prompt）、
  ``counts_body / wsum_body / tokens_body``（只数 body，LR 识别器的口径）、
  ``counts_decode / wsum_decode / tokens_decode``（模型自己生成的 decode 段），
  形状 ``[样本, 层, 专家]``（层 0-based，稠密层全 0）；``sample_idx``、``n_prompt_tokens``、``n_generated``；
* ``{task}_prefill_meta.json`` —— 模型、层数、专家数、top-k、稠密层、采集参数；
* ``{task}_bodies.jsonl`` —— 用到的样本（``sample_idx``、``body``），``lr_tool tokens/cases`` 要它。

之后：``examples/probe_to_data.py``（规划器的 CSV）、``moe_probe.py sets``（逐层 90% 驻留集，
与 expert_sets_90 同格式）、``p2pmoe.deploy.lr_tool train``（训 LR 识别器）。

怎么采
------
起一条**全专家**的静态通道（部分专家架构、不裁任何层），开画像统计；把训练样本按 task 选择器的
口径（任务指令模板 + 题目、对话模板、``enable_thinking=False``）编成 token 发过去，贪心生成
``--max-tokens`` 个 token；结束后从各级节点收回逐请求的统计、拼成上面的文件::

    # 1. 全专家方案（p2p-framework 的手动放置，只出清单：一条通道、前段只是第一级）
    python3 -m p2pmoe.deploy.run --spec probe_spec.json --plan-only --skip-model-check --model-dir $WEIGHTS \\
        --save-plan plan_probe.json
    # 2. 通道：画像统计（所有 MoE 层的 body 计数 + 整条 prompt / decode 计数）
    python3 vllm/p2p/layout.py --plan plan_probe.json --hosts task/hosts.txt \\
        --model-dir $WEIGHTS --out p2p_channels_probe --stats --stats-profile
    python3 vllm/p2p/launch.py --channels p2p_channels_probe/channels.json start general-0
    # 3. 采（bodies 取自 Qwen 那次 probe 的同一批样本：两个模型用同一批题）
    python3 vllm/p2p/moe_probe.py run --channels p2p_channels_probe/channels.json \\
        --bodies-dir $PROBE --model-dir $WEIGHTS --tasks gsm8k mbpp no_robots \\
        --max-tokens 256 --out probe_glm
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

if __package__ in (None, ""):                       # 按文件路径运行：把源码树放上 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

__all__ = ["records_to_arrays", "expert_sets", "main"]


# --------------------------------------------------------------------------- #
# 记录 → 数组
# --------------------------------------------------------------------------- #
def _dense_wire(w: Mapping, E: int) -> tuple[np.ndarray, np.ndarray, int]:
    """``LayerStat.to_wire`` 的稀疏格式 {i, c, w, t} → (count[E], mass[E], tokens)。"""
    c = np.zeros(E, np.int64)
    m = np.zeros(E, np.float64)
    c[w["i"]] = w["c"]
    m[w["i"]] = w["w"]
    return c, m, int(w.get("t", 0))


def records_to_arrays(recs: Sequence[Mapping], *, n_layers: int, n_experts: int) -> dict:
    """同一条请求在各级的统计记录（画像模式）→ 一个样本的各数组（层 0-based，没记到的层是 0）。"""
    L, E = n_layers, n_experts
    out = {k: np.zeros((L, E), np.int64) for k in ("counts_full", "counts_body", "counts_decode")}
    out.update({k: np.zeros((L, E), np.float64) for k in ("wsum_full", "wsum_body", "wsum_decode")})
    tok = {"full": 0, "body": 0, "decode": 0}
    n_prompt = n_seen = 0
    for r in recs:
        prof = r.get("profile") or {}
        if not prof:
            raise ValueError(f"{r.get('p2p_id')} 第 {r.get('pp_rank')} 级的记录里没有 profile —— "
                             "通道要用 layout.py --stats --stats-profile 起")
        for phase, key in (("prefill", "full"), ("decode", "decode")):
            ph = prof.get(phase) or {}
            tok[key] = max(tok[key], int(ph.get("t", 0)))
            for l1, w in (ph.get("layers") or {}).items():
                out[f"counts_{key}"][int(l1) - 1] = w["c"]
                out[f"wsum_{key}"][int(l1) - 1] = w["w"]
        for l1, w in (r.get("lr_layers") or {}).items():
            c, m, t = _dense_wire(w, E)
            out["counts_body"][int(l1) - 1] = c
            out["wsum_body"][int(l1) - 1] = m
            tok["body"] = max(tok["body"], t)
        n_prompt = max(n_prompt, int(r.get("n_prompt", 0)))
        n_seen = max(n_seen, int(r.get("n_seen", 0)))
    out.update(tokens_full=tok["full"], tokens_body=tok["body"], tokens_decode=tok["decode"],
               n_prompt_tokens=n_prompt, n_generated=max(n_seen - n_prompt, 0) + (1 if n_seen else 0))
    return out


def write_probe(out_dir: Path, task: str, rows: Sequence[tuple[int, str, Mapping]], *,
                meta: Mapping) -> Path:
    """rows = [(sample_idx, body, records_to_arrays 的结果)] → {task}_prefill_persample.npz 等。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda x: x[0])
    keys3 = ("counts_full", "wsum_full", "counts_body", "wsum_body", "counts_decode", "wsum_decode")
    arr = {k: np.stack([r[2][k] for r in rows]).astype(np.int32 if k.startswith("counts") else np.float32)
           for k in keys3}
    for k in ("tokens_full", "tokens_body", "tokens_decode", "n_prompt_tokens", "n_generated"):
        arr[k] = np.array([int(r[2][k]) for r in rows], np.int64)
    arr["sample_idx"] = np.array([int(r[0]) for r in rows], np.int64)
    f = out_dir / f"{task}_prefill_persample.npz"
    np.savez_compressed(f, **arr)
    (out_dir / f"{task}_prefill_meta.json").write_text(
        json.dumps(dict(meta, task=task, n_samples=len(rows)), ensure_ascii=False, indent=1),
        encoding="utf-8")
    (out_dir / f"{task}_bodies.jsonl").write_text(
        "".join(json.dumps({"sample_idx": int(i), "body": b}, ensure_ascii=False) + "\n"
                for i, b, _ in rows), encoding="utf-8")
    return f


# --------------------------------------------------------------------------- #
# 驻留集：逐层覆盖 q 的最少专家（与 expert_sets_90 同一条规则）
# --------------------------------------------------------------------------- #
def cover_set(counts: np.ndarray, q: float, k_min: int) -> list[int]:
    """一层的计数 → 按计数降序取，直到覆盖 q 的激活量；不少于 k_min 个。没有激活（稠密层）→ []。"""
    tot = float(counts.sum())
    if tot <= 0:
        return []
    order = np.argsort(-counts, kind="stable")
    cum = np.cumsum(counts[order]) / tot
    n = int(np.searchsorted(cum, q - 1e-12) + 1)
    n = max(n, int(k_min))
    return sorted(int(e) for e in order[:n])


def fit_indices(n: int, frac: float, seed: int) -> np.ndarray:
    """拟合用的那部分样本（其余留给评估）。frac ≥ 1 = 全部。"""
    if frac >= 1:
        return np.arange(n)
    m = max(1, int(math.ceil(n * frac)))
    return np.sort(np.random.default_rng(seed).choice(n, size=m, replace=False))


def expert_sets(probe_dir: Path, tasks: Sequence[str], *, q: float = 0.9, k_min: int | None = None,
                fit_frac: float = 0.2, seed: int = 42) -> dict:
    """{task}_prefill_persample.npz → expert_sets_90 同格式的 JSON（prefill 用 counts_full、
    decode 用 counts_decode，逐层各取覆盖 q 的最少专家，再取并集）。"""
    out: dict = {"threshold": q, "fit": f"{fit_frac:.0%} 样本（seed {seed}）", "tasks": {}}
    for u in tasks:
        meta_f = probe_dir / f"{u}_prefill_meta.json"
        meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
        kmin = int(k_min if k_min is not None else meta.get("top_k") or 1)
        with np.load(probe_dir / f"{u}_prefill_persample.npz") as z:
            idx = fit_indices(z["counts_full"].shape[0], fit_frac, seed)
            pre = z["counts_full"][idx].astype(np.int64).sum(0)
            dec = (z["counts_decode"][idx].astype(np.int64).sum(0)
                   if "counts_decode" in z.files else np.zeros_like(pre))
        L = pre.shape[0]
        ps = [cover_set(pre[l], q, kmin) for l in range(L)]
        ds = [cover_set(dec[l], q, kmin) for l in range(L)]
        un = [sorted(set(a) | set(b)) for a, b in zip(ps, ds)]
        out["tasks"][u] = {"source": meta.get("source", "vllm moe_probe"), "n_fit": int(len(idx)),
                           "union": un, "prefill": ps, "decode": ds}
        out.setdefault("top_k_min", kmin)
        out.setdefault("dense_layers", meta.get("dense_layers", []))
    out["rule"] = (f"每层：prefill（整条 prompt）计数取覆盖 {q:.0%} 激活量的最少专家（不少于 top-k）；"
                   "decode 同理；两者取并集。稠密层（没有路由）为空列表")
    return out


# --------------------------------------------------------------------------- #
# run：发请求、收统计
# --------------------------------------------------------------------------- #
def _post(url: str, data: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(data).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _body_str(ranges) -> str:
    return ",".join(f"{int(a)}:{int(b)}" for a, b in ranges)


def cmd_run(a) -> int:
    from vllm.p2p.bench import use_p2p_framework
    from vllm.p2p import launch as _launch
    from vllm.p2p.bench import stats_path

    use_p2p_framework(a.p2p_framework)
    from p2pmoe.deploy.lr_tool import train_segments
    from p2pmoe.runtime.text import TextIO

    cj = json.loads(Path(a.channels).read_text(encoding="utf-8"))
    chans = [c for c in cj["channels"] if not a.channel or c["name"] == a.channel]
    if len(chans) != 1:
        print(f"✗ 要正好一条通道（用 --channel 选）；有 {[c['name'] for c in cj['channels']]}",
              file=sys.stderr)
        return 2
    ch = chans[0]
    if not (ch.get("stats") or {}).get("profile"):
        print("✗ 这条通道没开画像统计 —— layout.py 要加 --stats --stats-profile", file=sys.stderr)
        return 2
    api = ch["api"].rstrip("/")
    model = _get(f"{api}/models")["data"][0]["id"]
    tio = TextIO.from_model_dir(a.model_dir, chat=True)
    tio.template_kw["enable_thinking"] = False
    L, E = int(cj["n_layers"]), int(cj["n_experts"])
    cfg = json.loads((Path(a.model_dir) / "config.json").read_text(encoding="utf-8"))
    top_k = int(cfg.get("num_experts_per_tok") or 0)

    # 样本
    jobs = []
    for u in a.tasks:
        f = Path(a.bodies_dir) / f"{u}_bodies.jsonl"
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        if a.limit:
            rows = rows[:a.limit]
        for r in rows:
            segs, _ = train_segments(u, r["body"])
            bp = tio.encode_body(segs)
            jobs.append((u, int(r["sample_idx"]), r["body"], bp.ids, bp.body))
    print(f"{len(jobs)} 条（{', '.join(f'{u} {sum(j[0] == u for j in jobs)}' for u in a.tasks)}），"
          f"通道 {ch['name']} @ {api}，每条生成 {a.max_tokens} 个 token，并发 {a.concurrency}")

    # 统计文件先清空
    hosts = [(st["node"], st["host"]) for st in ch["commands"]]
    for k, (node, host) in enumerate(hosts):
        _launch._ssh(host, a.user, f"mkdir -p {shlex.quote(a.stats_dir)} && : > "
                     f"{shlex.quote(stats_path(ch['name'], k, a.stats_dir))}", timeout=30)

    done = [0]
    lock = threading.Lock()
    t0 = time.time()

    def one(j):
        u, i, _, ids, body = j
        _post(f"{api}/completions", {
            "model": model, "prompt": ids, "max_tokens": a.max_tokens, "temperature": 0.0,
            "skip_special_tokens": True,
            "vllm_xargs": {"p2p_id": f"{u}:{i}", "p2p_body": _body_str(body)}}, a.req_timeout)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(jobs):
                print(f"  {done[0]}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(a.concurrency) as ex:
        list(ex.map(one, jobs))
    # 最后一批的「结束」要等下一步调度才送到 worker —— 再发一条短的，把它们送出去
    _post(f"{api}/completions", {"model": model, "prompt": [jobs[0][3][0]], "max_tokens": 1,
                                 "temperature": 0.0}, 60)
    time.sleep(3.0)

    got: dict[str, list[dict]] = defaultdict(list)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "raw_stats.jsonl").open("w", encoding="utf-8") as fraw:
        for k, (node, host) in enumerate(hosts):
            r = _launch._ssh(host, a.user, f"cat {shlex.quote(stats_path(ch['name'], k, a.stats_dir))}"
                             " 2>/dev/null || true", timeout=600)
            for ln in r.stdout.splitlines():
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if d.get("p2p_id"):
                    got[d["p2p_id"]].append(d)
                    fraw.write(ln + "\n")
    return _convert(got, jobs, out, L=L, E=E, top_k=top_k, cj=cj, a=a, model=model)


def _convert(got: Mapping[str, list], jobs, out: Path, *, L: int, E: int, top_k: int, cj: Mapping,
             a, model: str) -> int:
    n_stage = max((len(v) for v in got.values()), default=0)
    by_task: dict[str, list] = defaultdict(list)
    miss = 0
    for u, i, body, _, _ in jobs:
        recs = got.get(f"{u}:{i}") or []
        if len(recs) < n_stage or not recs:
            miss += 1
            continue
        by_task[u].append((i, body, records_to_arrays(recs, n_layers=L, n_experts=E)))
    meta = {"model": model, "source": "vllm moe_probe", "n_layers": L, "n_experts": E, "top_k": top_k,
            "dense_layers": list(cj.get("dense_layers") or []), "family": cj.get("family"),
            "max_tokens": getattr(a, "max_tokens", None), "enable_thinking": False,
            "variants": ["full", "body", "decode"]}
    for u, rows in by_task.items():
        f = write_probe(out, u, rows, meta=meta)
        a0 = rows[0][2]
        print(f"  {u:<10} {len(rows):>5} 条 → {f.name}（第一条：prompt {a0['n_prompt_tokens']} token、"
              f"body {a0['tokens_body']}、生成 {a0['n_generated']}）")
    if miss:
        print(f"  ⚠ {miss} 条没收全各级记录（{n_stage} 级），没写进去")
    print(f"已写 {out}/  下一步：python3 vllm/p2p/moe_probe.py sets --probe-dir {out} --out "
          f"{out}/expert_sets_90_union.json")
    return 0


def cmd_convert(a) -> int:
    got: dict[str, list[dict]] = defaultdict(list)
    for ln in Path(a.raw).read_text(encoding="utf-8").splitlines():
        if ln.strip():
            d = json.loads(ln)
            got[d["p2p_id"]].append(d)
    jobs = []
    for u in a.tasks:
        f = Path(a.bodies_dir) / f"{u}_bodies.jsonl"
        for x in f.read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                jobs.append((u, int(r["sample_idx"]), r["body"], None, None))
    cj = json.loads(Path(a.channels).read_text(encoding="utf-8")) if a.channels else {}
    return _convert(got, jobs, Path(a.out), L=a.n_layers or int(cj["n_layers"]),
                    E=a.n_experts or int(cj["n_experts"]), top_k=a.top_k, cj=cj, a=a,
                    model=a.model_name)


def cmd_sets(a) -> int:
    d = Path(a.probe_dir)
    tasks = a.tasks or sorted(p.name[:-len("_prefill_persample.npz")]
                              for p in d.glob("*_prefill_persample.npz"))
    res = expert_sets(d, tasks, q=a.q, k_min=a.k_min, fit_frac=a.fit_frac, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    for u, t in res["tasks"].items():
        sz = [len(x) for x in t["union"] if x]
        print(f"  {u:<10} 拟合 {t['n_fit']} 条；MoE 层驻留 {min(sz)}–{max(sz)} 个，均 "
              f"{sum(sz) / len(sz):.1f}")
    print(f"已写 {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="发训练样本、收统计、写 probe 目录")
    r.add_argument("--channels", required=True)
    r.add_argument("--channel", default=None)
    r.add_argument("--bodies-dir", required=True, help="有 {task}_bodies.jsonl 的目录（Qwen 那次 probe 的）")
    r.add_argument("--model-dir", required=True, help="控制机上的 tokenizer / config 目录")
    r.add_argument("--tasks", nargs="+", required=True)
    r.add_argument("--limit", type=int, default=0, help="每个 task 最多几条（0 = 全部）")
    r.add_argument("--max-tokens", type=int, default=256, help="decode 画像：每条生成几个 token")
    r.add_argument("--concurrency", type=int, default=8)
    r.add_argument("--req-timeout", type=float, default=600)
    r.add_argument("--stats-dir", default="/tmp/p2p_stats")
    r.add_argument("--user", default="ubuntu")
    r.add_argument("--p2p-framework", default=None)
    r.add_argument("--out", required=True)
    c = sub.add_parser("convert", help="已收回的 raw_stats.jsonl → probe 目录（run 的后半段）")
    c.add_argument("--raw", required=True)
    c.add_argument("--bodies-dir", required=True)
    c.add_argument("--tasks", nargs="+", required=True)
    c.add_argument("--channels", default=None)
    c.add_argument("--n-layers", type=int, default=None)
    c.add_argument("--n-experts", type=int, default=None)
    c.add_argument("--top-k", type=int, default=0)
    c.add_argument("--model-name", default="model")
    c.add_argument("--out", required=True)
    s = sub.add_parser("sets", help="probe 目录 → 逐层驻留集（expert_sets_90 同格式）")
    s.add_argument("--probe-dir", required=True)
    s.add_argument("--tasks", nargs="+", default=None)
    s.add_argument("--q", type=float, default=0.9, help="覆盖率（默认 0.9）")
    s.add_argument("--k-min", type=int, default=None, help="每层至少几个（默认 top-k）")
    s.add_argument("--fit-frac", type=float, default=0.2, help="用多少样本拟合（默认 20%%，同 Qwen 那份）")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"run": cmd_run, "convert": cmd_convert, "sets": cmd_sets}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
