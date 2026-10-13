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
                    "back_leg": _med([e.get("back_leg_ms_p50") for e in eng])})
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
          f"{'前段腿':>8}{'后段腿':>8}   {'A 同通道 TPOT / TTFT':>22}")
    for s in summarise_b(rb):
        ac = A.get(s["back"].removeprefix("back-"))
        tail = f"{f(ac['tpot'])} / {f(ac['ttft'])}（{ac['n']} 条）" if ac else ""
        print(f"{s['front']:<8}{s['back']:<16}{s['n']:>4}{f(s['ttft']):>8}{f(s['tpot']):>8}"
              f"{f(s['front_prefill']):>12}{f(s['bind']):>7}{f(s['front_leg']):>8}"
              f"{f(s['back_leg']):>8}   {tail:>22}")
    print("（单位 ms；TPOT = (总时延 − 首 token) / (生成数 − 1)；腿是每条请求的中位数再取中位数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
