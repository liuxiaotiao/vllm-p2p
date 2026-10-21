# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1：分段结果的开销拆解，可选与 A 的结果按通道并排。**纯标准库，按文件路径运行。**

    python3 vllm/p2p/tiered_stats.py --b results/b1_gsm8k_inflight8.json [--a results/a3_gsm8k_inflight8.json]

按（前段节点, 后段引擎）分组：条数、TTFT / TPOT 中位数，以及分段网关逐请求报的
「前段 prefill」「绑定」「前段腿」（后段出 token → 前段 hidden 回到协调器）「后段腿」
（hidden 送出 → 下一个 token 回来）。给了 --a 时再列 A 同一条通道（后段 back-X ↔ 通道 X）的 TPOT。
每 token 的时间 ≈ 前段腿 + 后段腿；两条腿里都含两次跨机、一次调度、一次前向。

开头先打整体与按 task 的逐请求统计（`request_stats`：TTFT / TPOT 的均值 ± 标准差，每条请求的 token/s 取均值）；
`bench.py` 跑完也打同一张表、写进结果的 `run.per_request`。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def _med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(st.median(xs), 1) if xs else None


def _tpot(r):
    n = int(r.get("n_generated") or 0)
    return (r["total_ms"] - r["prefill_ms"]) / (n - 1) if n > 1 else None


def _ms(xs):
    """→ {n, mean, std}（样本标准差；只有 1 个值时 std = 0）。"""
    xs = [float(x) for x in xs if isinstance(x, (int, float))]
    if not xs:
        return {"n": 0, "mean": None, "std": None}
    return {"n": len(xs), "mean": round(st.fmean(xs), 3),
            "std": round(st.stdev(xs), 3) if len(xs) > 1 else 0.0}


def _tok_s(r):
    """一条请求自己的速度：生成 token 数 / 这条从发出到收完的时间（含 TTFT）。"""
    n = int(r.get("n_generated") or 0)
    t = float(r.get("total_ms") or 0)
    return n / (t / 1e3) if n > 0 and t > 0 else None


def _decode_tok_s(r):
    """只看 decode 段：1000 / TPOT。"""
    tp = _tpot(r)
    return 1e3 / tp if tp and tp > 0 else None


def request_stats(rows) -> dict:
    """逐请求的 TTFT / TPOT 均值与标准差、每条请求的 token/s 取均值；整体 + 按 task。"""
    def one(rs):
        return {"n": len(rs),
                "ttft_ms": _ms([r.get("prefill_ms") for r in rs]),
                "tpot_ms": _ms([_tpot(r) for r in rs]),
                "tok_per_s": _ms([_tok_s(r) for r in rs]),
                "decode_tok_per_s": _ms([_decode_tok_s(r) for r in rs])}
    by = defaultdict(list)
    for r in rows:
        by[r.get("true_task") or "?"].append(r)
    return {"all": one(list(rows)), "by_task": {u: one(rs) for u, rs in sorted(by.items())},
            "note": "均值 ± 样本标准差；tok_per_s = 每条请求 生成 token / 总时延（含 TTFT），"
                    "decode_tok_per_s = 1000 / TPOT，都是先逐条算再对所有请求取均值"}


def format_request_stats(rs: dict) -> list[str]:
    def f(d, digits=1):
        if not d or d.get("mean") is None:
            return "—"
        return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"
    head = f"  {'':<12}{'条数':>5}{'TTFT ms':>20}{'TPOT ms':>18}{'token/s':>16}{'decode token/s':>18}"
    out = ["逐请求统计（均值 ± 标准差）", head]
    rows = [("全部", rs["all"])] + [(u, v) for u, v in rs["by_task"].items()]
    for name, v in rows:
        out.append(f"  {name:<12}{v['n']:>5}{f(v['ttft_ms'], 0):>20}{f(v['tpot_ms']):>18}"
                   f"{f(v['tok_per_s']):>16}{f(v['decode_tok_per_s']):>18}")
    out.append("  （token/s = 每条请求 生成 token / 总时延，含首 token；decode token/s = 1000 / TPOT；"
               "都是逐条算再对所有请求取均值）")
    return out


def summarise_b(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r.get("front") or "?", r.get("channel") or "?")].append(r)
    out = []
    for (f, b), rs in sorted(g.items()):
        eng = [r.get("engine") or {} for r in rs]
        out.append({"front": f, "back": b, "n": len(rs),
                    "ttft": _med([r["prefill_ms"] for r in rs]),
                    "tpot": _med([_tpot(r) for r in rs]),
                    "front_prefill": _med([e.get("front_prefill_ms") for e in eng]),
                    "bind": _med([e.get("bind_ms") for e in eng]),
                    "front_leg": _med([e.get("front_leg_ms_p50") for e in eng]),
                    "back_leg": _med([e.get("back_leg_ms_p50") for e in eng]),
                    "front_eng": _med([e.get("front_engine_ms_p50") for e in eng]),
                    "back_eng": _med([e.get("back_engine_ms_p50") for e in eng])})
    return out


def summarise_a(rows):
    g = defaultdict(list)
    for r in rows:
        g[r.get("channel") or "?"].append(r)
    return {c: {"n": len(rs), "ttft": _med([r["prefill_ms"] for r in rs]),
                "tpot": _med([_tpot(r) for r in rs])} for c, rs in g.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--b", required=True, type=Path, help="分段网关的结果")
    ap.add_argument("--a", type=Path, default=None, help="A 的结果（同一批 case）")
    a = ap.parse_args(argv)
    rb = json.loads(a.b.read_text(encoding="utf-8"))["requests"]
    ra = json.loads(a.a.read_text(encoding="utf-8"))["requests"] if a.a else []
    A = summarise_a(ra)
    for ln in format_request_stats(request_stats(rb)):
        print(ln)
    if ra:
        print(f"（对照 {a.a.name}）")
        for ln in format_request_stats(request_stats(ra))[2:-1]:
            print(ln)
    print()

    def f(x):
        return "—" if x is None else f"{x:g}"
    print(f"{'前段':<8}{'后段':<16}{'条数':>4}{'TTFT':>8}{'TPOT':>8}{'前段prefill':>12}{'绑定':>7}"
          f"{'前段腿':>8}{'(引擎内)':>9}{'后段腿':>8}{'(引擎内)':>9}   {'A 同通道 TPOT / TTFT':>22}")
    for s in summarise_b(rb):
        ac = A.get(s["back"].removeprefix("back-"))
        tail = f"{f(ac['tpot'])} / {f(ac['ttft'])}（{ac['n']} 条）" if ac else ""
        print(f"{s['front']:<8}{s['back']:<16}{s['n']:>4}{f(s['ttft']):>8}{f(s['tpot']):>8}"
              f"{f(s['front_prefill']):>12}{f(s['bind']):>7}{f(s['front_leg']):>8}"
              f"{f(s['front_eng']):>9}{f(s['back_leg']):>8}{f(s['back_eng']):>9}   {tail:>22}")
    print("（单位 ms；TPOT = (总时延 − 首 token) / (生成数 − 1)；腿是每条请求的中位数再取中位数；\n"
          " 「引擎内」= 代理收到这条请求的输入到送出它的输出（排队 + 调度 + 前向），腿 − 引擎内 ≈ 网络 + 协调器）")
    ident = summarise_ident(rb)
    if ident:
        print(f"\n前段出口识别（lr 模式）：绑对 task {ident['ok']}/{ident['n']}"
              f"（{100 * ident['ok'] / ident['n']:.1f}%）；LR 耗时中位数 {f(ident['lr_ms'])} ms；"
              f"区间 {ident['zones']}")
        tasks = sorted({t for pair in ident["matrix"] for t in pair})
        head = "真实 \\ 绑到"
        print(f"  {head:<14}" + "".join(f"{t:>12}" for t in tasks))
        for u in tasks:
            if any(k[0] == u for k in ident["matrix"]):
                print(f"  {u:<14}" + "".join(f"{ident['matrix'].get((u, t), 0) or '·':>12}"
                                             for t in tasks))
        if ident["wrong"]:
            print(f"  绑错的（前 10 条）：{ident['wrong'][:10]}")
    return 0


def summarise_ident(rows):
    """lr 模式的结果：真实 task × 绑到的 task、准确率、LR 耗时、置信区间分布。"""
    rs = [r for r in rows if r.get("correct") is not None and r.get("true_task")]
    if not rs:
        return None
    m = defaultdict(int)
    zones = defaultdict(int)
    for r in rs:
        m[(r["true_task"], r.get("task") or "?")] += 1
        zones[r.get("zone") or "?"] += 1
    return {"n": len(rs), "ok": sum(bool(r["correct"]) for r in rs), "matrix": dict(m),
            "zones": dict(zones),
            "lr_ms": _med([(r.get("identify") or {}).get("compute_ms") for r in rs]),
            "wrong": [f"{r['req']}:{r['true_task']}→{r.get('task')}" for r in rs
                      if not r["correct"]]}


if __name__ == "__main__":
    raise SystemExit(main())
