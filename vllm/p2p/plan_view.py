# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""看 p2p-framework 规划器算出的分配方案（plan JSON）。**按文件路径运行，不要 torch。**

    python3 vllm/p2p/plan_view.py --plan <plan.json>
    python3 vllm/p2p/plan_view.py --channels p2p_channels_new/channels.json          # 用 layout 记下的那份 plan
    python3 vllm/p2p/plan_view.py --channels … --stages p2p_channels_new/stages.json \\
        --steps results/steps_n150/*.jsonl                                            # 再对上实测每步

打印四张表（markdown，`--md` 另存）：

* 段：前段 / 后段（哪类 task）、由哪些节点组成、每个节点管哪几层；规划器预计的每步计算 / 段内传输 / 合计 ms；
  给了 `--stages` 时标出它在 vLLM 部署里叫什么（front-N03、back-gsm8k-0 …），给了 `--steps` 时并上实测的
  「纯 decode、批 1」每步 p50 —— 规划器的预计与实测对照；
* 节点：角色、所属段、层范围、每层驻留多少专家（最少 / 平均 / 最多，占全部专家的比例）、权重与 KV 显存；
* 配对：前段 → 后段（task）、前向 / 回环两条路径经过的节点、规划器预计的 t50 等；
* 规划器自己报的违规（violations）、警告与备用前段。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Mapping, Sequence


def _model_experts(plan: Mapping) -> int | None:
    m = str(plan.get("model") or "")
    for part in m.split("-"):
        if part.startswith("E") and part[1:].isdigit():
            return int(part[1:])
    return None


def _ranges(splits) -> str:
    return " + ".join(f"{a}–{b}" for a, b in splits)


def segment_rows(plan: Mapping, stages: Mapping | None = None,
                 step_b1: Mapping[str, float] | None = None) -> list[dict]:
    """每个段一行；stages / step_b1 给了就对上 vLLM 引擎名与实测批 1 每步 p50。"""
    by_nodes = {}
    for e in (stages or {}).get("channels") or []:
        by_nodes[tuple(e.get("nodes") or ())] = e["name"]
    rows = []
    for sid, s in (plan.get("segments") or {}).items():
        nodes = list(s.get("nodes") or [])
        eng = by_nodes.get(tuple(nodes))
        rows.append({
            "segment": sid, "role": "前段" if str(s.get("role", "")).startswith("front") else "后段",
            "task": s.get("task") or "—", "nodes": nodes,
            "layers": " / ".join(f"{n}:{a}–{b}" for n, (a, b) in zip(nodes, s.get("splits") or [])),
            "compute_ms": s.get("compute_ms"), "hop_ms": s.get("hop_ms"), "delay_ms": s.get("delay_ms"),
            "engine": eng, "measured_b1_ms": (step_b1 or {}).get(eng) if eng else None,
        })
    return rows


def node_rows(plan: Mapping) -> list[dict]:
    n_exp = _model_experts(plan)
    rows = []
    for n in plan.get("nodes") or []:
        counts = [len(ld.get("experts") or []) for ld in n.get("layers") or []]
        rows.append({
            "node": n["node"], "role": n.get("role"), "segment": n.get("segment"),
            "layers": "{}–{}".format(*n["layer_range"]) if n.get("layer_range") else "—",
            "n_layers": len(counts),
            "experts_min": min(counts) if counts else None,
            "experts_mean": round(st.fmean(counts), 1) if counts else None,
            "experts_max": max(counts) if counts else None,
            "resident": (round(st.fmean(counts) / n_exp, 3) if counts and n_exp else None),
            "weight_gb": n.get("weight_gb"), "kv_gb": n.get("kv_gb"), "total_gb": n.get("total_gb"),
        })
    return rows


def step_b1_from_traces(files: Sequence[str]) -> dict[str, float]:
    """步级记录 → 每个引擎「纯 decode、批 1」的每步 p50（ms）。"""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import steptrace  # noqa: E402  按文件路径运行时同目录 import

    out = {}
    for name, rows in steptrace.load([Path(f) for f in files]).items():
        s = steptrace.summarise(rows)
        v = ((s.get("decode_ms_by_batch") or {}).get("1") or {}).get("p50")
        if v is not None:
            out[name] = float(v)
    return out


def _f(x, nd=1):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render(plan: Mapping, *, stages: Mapping | None = None, step_b1: Mapping[str, float] | None = None,
           source: str = "") -> list[str]:
    n_exp = _model_experts(plan)
    out = [f"# 分配方案 {source}".rstrip(), "",
           f"模型 {plan.get('model')}；前段层数 L₀ = {plan.get('l0')}；节点 {len(plan.get('nodes') or [])} 个、"
           f"段 {len(plan.get('segments') or {})} 个、配对 {len(plan.get('pairings') or [])} 条", ""]
    segs = segment_rows(plan, stages, step_b1)
    show_eng = any(r["engine"] for r in segs)
    show_meas = any(r["measured_b1_ms"] is not None for r in segs)
    head = ["段", "角色", "task", "节点与层", "预计计算 ms", "预计段内传输 ms", "预计合计 ms"]
    if show_eng:
        head.append("vLLM 引擎")
    if show_meas:
        head += ["实测批 1 每步 p50 ms", "实测 / 预计合计"]
    out += ["## 段", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in segs:
        cells = [r["segment"], r["role"], r["task"], r["layers"], _f(r["compute_ms"]), _f(r["hop_ms"]),
                 _f(r["delay_ms"])]
        if show_eng:
            cells.append(r["engine"] or "—")
        if show_meas:
            m, d = r["measured_b1_ms"], r["delay_ms"]
            cells += [_f(m), f"{m / d:.2f}×" if m and d else "—"]
        out.append("| " + " | ".join(cells) + " |")
    out += ["", "（层号按规划器的口径；预计 = 规划器按拓扑与每层耗时算的一步；实测批 1 = 步级记录里只算 1 条 decode 的步）", ""]

    head = ["节点", "角色", "段", "层", "层数", "每层专家 最少 / 平均 / 最多"
            + (f"（共 {n_exp}）" if n_exp else ""), "驻留比例", "权重 GB", "KV GB", "合计 GB"]
    out += ["## 节点", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in node_rows(plan):
        out.append("| " + " | ".join([
            r["node"], str(r["role"]), str(r["segment"]), r["layers"], str(r["n_layers"]),
            f"{_f(r['experts_min'])} / {_f(r['experts_mean'])} / {_f(r['experts_max'])}",
            f"{100 * r['resident']:.0f}%" if r["resident"] is not None else "—",
            _f(r["weight_gb"], 2), _f(r["kv_gb"], 2), _f(r["total_gb"], 2)]) + " |")
    out.append("")

    pr = plan.get("pairings") or []
    if pr:
        out += ["## 配对（前段 → 后段）", "", "| 前段 | 后段 | task | 前向路径 | 回环路径 | 预计 t50 ms | w p50 / p95 |",
                "|---|---|---|---|---|---|---|"]
        for p in pr:
            out.append(f"| {p.get('front')} | {p.get('back')} | {p.get('task')} | "
                       f"{' → '.join(p.get('forward') or [])} | {' → '.join(p.get('loop') or [])} | "
                       f"{_f(p.get('t50'))} | {_f(p.get('w_p50'))} / {_f(p.get('w_p95'))} |")
        out.append("")
    for key, title in (("violations", "违规"), ("warnings", "警告"), ("standby_fronts", "备用前段")):
        v = plan.get(key)
        if v:
            out += [f"## {title}", ""] + [f"* {json.dumps(x, ensure_ascii=False)}" for x in v] + [""]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--plan", help="规划器的 plan JSON")
    src.add_argument("--channels", help="layout.py 生成的 channels.json（里面记着用的哪份 plan）")
    ap.add_argument("--stages", default=None, help="stages.json：标出每个段在 vLLM 部署里的引擎名")
    ap.add_argument("--steps", nargs="*", default=None, help="步级记录（steptrace collect 收回的 *.jsonl）")
    ap.add_argument("--md", default=None, help="另存成这个 .md")
    a = ap.parse_args(argv)
    plan_path = a.plan
    if a.channels:
        cj = json.loads(Path(a.channels).read_text(encoding="utf-8"))
        plan_path = cj.get("plan")
        if not plan_path or not Path(plan_path).exists():
            raise SystemExit(f"{a.channels} 里记的 plan 是 {plan_path!r}，这台机器上找不到 —— 直接用 --plan 给路径")
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    stages = json.loads(Path(a.stages).read_text(encoding="utf-8")) if a.stages else None
    step_b1 = step_b1_from_traces(a.steps) if a.steps else None
    lines = render(plan, stages=stages, step_b1=step_b1, source=f"（{plan_path}）")
    print("\n".join(lines))
    if a.md:
        Path(a.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
