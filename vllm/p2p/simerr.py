# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""模拟 vs 真机的误差：读 rounds.py 写的对照汇总（每个方案一份，组名里有 real 与若干 sim_*），
逐方案、逐指标给相对误差，再给各模拟版本的平均绝对误差，以及方案排序是否与真机一致。
**按文件路径运行。**

    python3 vllm/p2p/simerr.py results/sim/cmp_theory_bf.json results/sim/cmp_alg_bf.json \\
                               results/sim/cmp_alg_ff.json --out results/sim/simerr.json

方案名取文件名里 cmp_ 之后的部分（或写成 名字=文件）。相对误差 = (模拟均值 − 真机均值) / 真机均值，
正数 = 模拟偏大（时延偏慢 / 吞吐偏高）。只用标准库。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

__all__ = ["KEY_METRICS", "plan_errors", "rank", "main"]

# (指标, 说明, 越大越好?)
KEY_METRICS = (
    ("out_tok_per_s", "吞吐 tok/s", True),
    ("req_per_min", "请求/分钟", True),
    ("ttft_p50_ms", "TTFT p50", False),
    ("ttft_p90_ms", "TTFT p90", False),
    ("tpot_p50_ms", "TPOT p50", False),
    ("e2e_p50_ms", "端到端 p50", False),
    ("front_prefill_p50_ms", "前段 prefill p50", False),
)
TASK_METRICS = ("ttft_p50_ms", "tpot_p50_ms", "e2e_p50_ms")


def _mean(summary: Mapping, k: str) -> float | None:
    v = summary.get(k)
    return None if v is None else float(v["mean"])


def _err(sim: float | None, real: float | None) -> float | None:
    if sim is None or real is None or real == 0:
        return None
    return (sim - real) / real


def plan_errors(cmp: Mapping, real: str = "real") -> dict:
    """一份 rounds 对照汇总 → {sim组: {指标: {real, sim, err}}}（含逐 task 的几项）。"""
    if real not in cmp:
        raise ValueError(f"汇总里没有 {real!r} 组（有 {sorted(cmp)}）")
    rs = cmp[real]["summary"]
    tasks = sorted({k.split(".")[0] for k in rs if "." in k})
    keys = [k for k, _, _ in KEY_METRICS] + [f"{t}.{m}" for t in tasks for m in TASK_METRICS]
    out = {}
    for g, d in cmp.items():
        if g == real:
            continue
        ss = d["summary"]
        out[g] = {k: {"real": _mean(rs, k), "sim": _mean(ss, k), "err": _err(_mean(ss, k), _mean(rs, k))}
                  for k in keys if _mean(rs, k) is not None}
    return out


def rank(vals: Mapping[str, float | None], higher_better: bool) -> list[str]:
    ok = {k: v for k, v in vals.items() if v is not None}
    return sorted(ok, key=lambda k: -ok[k] if higher_better else ok[k])


def _pct(e: float | None) -> str:
    return "—" if e is None else f"{e * 100:+.1f}%"


def _num(v: float | None) -> str:
    return "—" if v is None else (f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmps", nargs="+", help="rounds.py --out 写的对照汇总（每个方案一份），可写 名字=文件")
    ap.add_argument("--real", default="real", help="真机那一组的组名")
    ap.add_argument("--tasks", action="store_true", help="也列逐 task 的 TTFT / TPOT / 端到端")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    plans: dict[str, dict] = {}
    for it in a.cmps:
        name, sep, f = it.partition("=")
        if not sep:
            f, name = it, Path(it).stem.removeprefix("cmp_")
        try:
            plans[name] = plan_errors(json.loads(Path(f).read_text(encoding="utf-8")), a.real)
        except (OSError, ValueError) as e:
            print(f"✗ {f}：{e}", file=sys.stderr)
            return 2
    sims = list(dict.fromkeys(g for p in plans.values() for g in p))
    label = {k: lab for k, lab, _ in KEY_METRICS}

    # 1. 逐方案：真机值、各模拟值与相对误差
    for pn, pe in plans.items():
        print(f"\n######## {pn}")
        print(f"{'指标':<22}{'真机':>10}" + "".join(f"{g:>14}{'误差':>9}" for g in sims))
        keys = [k for k in next(iter(pe.values()))
                if "." not in k or a.tasks]
        for k in keys:
            real_v = next((pe[g][k]["real"] for g in sims if k in pe.get(g, {})), None)
            row = f"{label.get(k, k):<22}{_num(real_v):>10}"
            for g in sims:
                c = pe.get(g, {}).get(k) or {}
                row += f"{_num(c.get('sim')):>14}{_pct(c.get('err')):>9}"
            print(row)

    # 2. 各模拟版本：每个指标在全部方案上的平均绝对误差
    print("\n######## 平均绝对误差（全部方案）")
    print(f"{'指标':<22}" + "".join(f"{g:>16}" for g in sims))
    mae: dict[str, dict] = {g: {} for g in sims}
    for k, lab, _ in KEY_METRICS:
        row = f"{lab:<22}"
        for g in sims:
            es = [abs(pe[g][k]["err"]) for pe in plans.values()
                  if g in pe and k in pe[g] and pe[g][k]["err"] is not None]
            mae[g][k] = sum(es) / len(es) if es else None
            row += f"{('—' if mae[g][k] is None else f'{mae[g][k] * 100:.1f}%'):>16}"
        print(row)
    row = f"{'以上合计（平均）':<22}"
    for g in sims:
        vs = [v for v in mae[g].values() if v is not None]
        tot = sum(vs) / len(vs) if vs else None
        mae[g]["_all"] = tot
        row += f"{('—' if tot is None else f'{tot * 100:.1f}%'):>16}"
    print(row)

    # 3. 方案排序：模拟能不能排对
    ranks = {}
    if len(plans) > 1:
        print("\n######## 方案排序（左 = 最好）")
        for k, lab, hb in KEY_METRICS:
            rv = {pn: next((pe[g][k]["real"] for g in sims if k in pe.get(g, {})), None)
                  for pn, pe in plans.items()}
            r_real = rank(rv, hb)
            line = f"{lab:<22}真机 {' > '.join(r_real) or '—'}"
            ranks[k] = {"real": r_real}
            for g in sims:
                r_sim = rank({pn: (pe.get(g, {}).get(k) or {}).get("sim") for pn, pe in plans.items()}, hb)
                ranks[k][g] = r_sim
                line += f"   {g} {'✓' if r_sim == r_real else '✗ ' + ' > '.join(r_sim)}"
            print(line)
    if a.out:
        a.out.write_text(json.dumps({"plans": plans, "mae": mae, "ranks": ranks},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
