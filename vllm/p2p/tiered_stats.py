# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1：分段结果的开销拆解，可选与 A 的结果按通道并排。**纯标准库，按文件路径运行。**

    python3 vllm/p2p/tiered_stats.py --b results/b1_gsm8k_inflight8.json [--a results/a3_gsm8k_inflight8.json]

按（前段节点, 后段引擎）分组：条数、TTFT / TPOT 中位数，以及分段网关逐请求报的
「前段 prefill」「绑定」「前段腿」（后段出 token → 前段 hidden 回到协调器）「后段腿」
（hidden 送出 → 下一个 token 回来）。给了 --a 时再列 A 同一条通道（后段 back-X ↔ 通道 X）的 TPOT。
每 token 的时间 ≈ 前段腿 + 后段腿；两条腿里都含两次跨机、一次调度、一次前向。
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
