# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""多轮汇总：同一个实验跑了几轮 bench，把每轮的指标算出来，再给均值 ± 标准差。**按文件路径运行。**

    python3 vllm/p2p/rounds.py results/glm47_theory_1f_shadow_r*.json
    python3 vllm/p2p/rounds.py --group shadow='results/glm47_theory_1f_shadow_r*.json' \\
                               --group lr='results/glm47_theory_1f_lr_r*.json' --out results/rounds.json

指标都直接从每轮的 `requests` 逐条记录算（不依赖各版本 run 字段的写法），口径：
* 墙钟 = 最后一条完成 − 第一条提交；吞吐 = 生成 token 数 / 墙钟；
* TTFT = 提交到第一个 token（`prefill_ms`，含在前段等后段槽位的时间）；
* TPOT = 每条请求相邻 token 间隔的中位数（`per_token_p50_ms`），再取各条的 p50；
* 识别准确率：lr 轮看 `correct`，shadow 轮看 `shadow.task` 与真实 task 是否一致；
* 逐 task：TTFT p50、TPOT p50、端到端 p50 与条数。
只用标准库。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

__all__ = ["round_metrics", "aggregate", "main"]


def _pct(xs: Sequence[float], q: float) -> float | None:
    xs = sorted(float(x) for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _acc(rows: Sequence[Mapping]) -> float | None:
    ok = [bool(r["correct"]) for r in rows if r.get("correct") is not None]
    if ok:
        return sum(ok) / len(ok)
    sh = [r["shadow"]["task"] == r.get("true_task") for r in rows
          if isinstance(r.get("shadow"), Mapping) and "task" in r["shadow"]]
    return sum(sh) / len(sh) if sh else None


def round_metrics(res: Mapping) -> dict:
    """一轮 bench 结果（results/*.json 的内容）→ 平铺的指标字典。"""
    rows = list(res.get("requests") or [])
    if not rows:
        raise ValueError("这份结果里没有 requests")
    t0 = min(float(r["t_submit_s"]) for r in rows)
    t1 = max(float(r["t_done_s"]) for r in rows)
    wall = max(t1 - t0, 1e-9)
    out_tok = sum(int(r.get("n_generated") or 0) for r in rows)
    ttft = [r["prefill_ms"] for r in rows]
    tpot = [r["per_token_p50_ms"] for r in rows]
    e2e = [r["total_ms"] for r in rows]
    m = {
        "n": len(rows), "failed": len(res.get("failed") or []),
        "wall_s": wall, "out_tok_per_s": out_tok / wall, "req_per_min": len(rows) / wall * 60,
        "ttft_p50_ms": _pct(ttft, .5), "ttft_p90_ms": _pct(ttft, .9),
        "tpot_p50_ms": _pct(tpot, .5), "tpot_p90_ms": _pct(tpot, .9),
        "e2e_p50_ms": _pct(e2e, .5), "e2e_p90_ms": _pct(e2e, .9),
        "wait_front_mean_ms": st.fmean(float(r.get("wait_front_ms") or 0) for r in rows),
        "accuracy": _acc(rows),
    }
    by = defaultdict(list)
    for r in rows:
        by[r.get("true_task") or "?"].append(r)
    for u, rs in sorted(by.items()):
        m[f"{u}.n"] = len(rs)
        m[f"{u}.ttft_p50_ms"] = _pct([r["prefill_ms"] for r in rs], .5)
        m[f"{u}.tpot_p50_ms"] = _pct([r["per_token_p50_ms"] for r in rs], .5)
        m[f"{u}.e2e_p50_ms"] = _pct([r["total_ms"] for r in rs], .5)
        m[f"{u}.accuracy"] = _acc(rs)
    return m


def aggregate(per_round: Sequence[Mapping]) -> dict:
    """各轮指标 → {指标: {mean, std, min, max, n}}（只对各轮都有数的指标）。"""
    keys = list(dict.fromkeys(k for m in per_round for k in m))
    agg = {}
    for k in keys:
        xs = [float(m[k]) for m in per_round if m.get(k) is not None]
        if not xs:
            continue
        agg[k] = {"mean": st.fmean(xs), "std": st.stdev(xs) if len(xs) > 1 else 0.0,
                  "min": min(xs), "max": max(xs), "n": len(xs)}
    return agg


def _fmt(a: Mapping | None, k: str) -> str:
    if a is None:
        return "—"
    if k.endswith("accuracy"):
        return f"{a['mean'] * 100:.2f}±{a['std'] * 100:.2f}%"
    if k.endswith(".n") or k in ("n", "failed"):
        return f"{a['mean']:.0f}"
    return f"{a['mean']:.1f}±{a['std']:.1f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="同一个实验的各轮结果（不给 --group 时用）")
    ap.add_argument("--group", action="append", default=[], metavar="名字=通配",
                    help="一组实验，可重复，如 --group lr='results/x_lr_r*.json'（通配要加引号）")
    ap.add_argument("--out", type=Path, default=None, help="把各轮指标与汇总写成 JSON")
    a = ap.parse_args(argv)
    groups: dict[str, list[str]] = {}
    for g in a.group:
        name, _, pat = g.partition("=")
        if not pat:
            ap.error(f"--group {g!r} 要写成 名字=通配")
        groups[name] = sorted(glob.glob(pat))
    if a.files:
        groups.setdefault("runs", []).extend(sorted(a.files))
    if not groups:
        ap.error("给结果文件，或 --group")
    result = {}
    for name, files in groups.items():
        if not files:
            print(f"✗ {name}：一个文件都没匹配上", file=sys.stderr)
            return 2
        per = []
        for f in files:
            m = round_metrics(json.loads(Path(f).read_text(encoding="utf-8")))
            m["file"] = f
            per.append(m)
        result[name] = {"rounds": per, "summary": aggregate(
            [{k: v for k, v in m.items() if k != "file"} for m in per])}
        print(f"{name}：{len(files)} 轮  " + "  ".join(Path(f).name for f in files))

    names = list(result)
    keys = list(dict.fromkeys(k for n in names for k in result[n]["summary"]))
    w = max(len(k) for k in keys) + 2
    print("\n" + "指标".ljust(w - 2) + "".join(f"{n:>22}" for n in names) + "   （均值±标准差）")
    for k in keys:
        print(k.ljust(w) + "".join(f"{_fmt(result[n]['summary'].get(k), k):>22}" for n in names))
    if a.out:
        a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
