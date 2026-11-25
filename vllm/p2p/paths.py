# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""同一个方案跑的几轮 bench：逐个样本核对每轮走的路径是否一样。**按文件路径运行。**

    python3 vllm/p2p/paths.py results/glm47_ff_lr_r[0-9]*.json
    python3 vllm/p2p/paths.py --group alg_ff='results/glm47_ff_lr_r[0-9]*.json' \\
                              --group alg_bf='results/glm47_alg_1f_lr_r[0-9]*.json' --out results/paths.json

样本按 case_id 对齐（没有就用 真实 task + prompt）。每个样本比这几项：
* 识别出的 task（lr 轮决定去哪类后段；shadow 轮比的是记下的判断）；
* 后段：哪条后段通道。同一类有几条后段时（gsm8k-0 / gsm8k-1），`--bind least-free` 按当时谁空挑，
  轮与轮之间换一条是正常的 —— 单独记作「同类换副本」，不算路径不同；
* 前段节点；
* 输出：贪心解码，生成的 token 数与文本应当逐字一样（不一样说明数值有抖动，值得查）。
只用标准库。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

__all__ = ["sample_key", "path_of", "compare_rounds", "main"]


def sample_key(r: Mapping) -> str:
    if r.get("case_id"):
        return str(r["case_id"])
    h = hashlib.sha1(str(r.get("prompt") or "").encode()).hexdigest()[:12]
    return f"{r.get('true_task')}:{h}"


def _task_of_back(ch: str | None) -> str | None:
    """back-gsm8k-1 → gsm8k；back-no_robots-0 → no_robots。"""
    if not ch:
        return None
    m = re.match(r"^back-(.+?)(?:-\d+)?$", ch)
    return m.group(1) if m else ch


def path_of(r: Mapping) -> dict:
    sh = r.get("shadow") if isinstance(r.get("shadow"), Mapping) else None
    return {"task": r.get("task") if r.get("correct") is not None or not sh else sh.get("task"),
            "routed_task": r.get("task"),
            "back": r.get("channel"), "back_task": _task_of_back(r.get("channel")),
            "front": r.get("front") or r.get("front_engine"),
            "n_gen": r.get("n_generated"), "text": r.get("text")}


def compare_rounds(docs: Sequence[tuple[str, Mapping]]) -> dict:
    """docs = [(文件名, 一轮结果)…] → 逐样本核对的结果。"""
    rounds = []
    for name, d in docs:
        rows = {}
        for r in d.get("requests") or []:
            rows.setdefault(sample_key(r), r)
        rounds.append((name, rows))
    keys_all = set().union(*(set(rows) for _, rows in rounds))
    common = sorted(set.intersection(*(set(rows) for _, rows in rounds)))
    missing = {name: sorted(keys_all - set(rows)) for name, rows in rounds if keys_all - set(rows)}
    cnt = Counter()
    diffs = []
    for k in common:
        ps = [path_of(rows[k]) for _, rows in rounds]
        d = {}
        for fld, lab in (("task", "识别"), ("back_task", "后段类"), ("front", "前段"),
                         ("n_gen", "生成长度"), ("text", "输出文本")):
            vals = [p[fld] for p in ps]
            if len(set(map(str, vals))) > 1:
                cnt[lab] += 1
                d[lab] = vals if fld != "text" else [(v or "")[:60] for v in vals]
        backs = [p["back"] for p in ps]
        if len(set(backs)) > 1 and "后段类" not in d:
            cnt["同类换副本"] += 1
            d["同类换副本"] = backs
        if not d:
            cnt["完全一致"] += 1
        else:
            diffs.append({"sample": k, "true_task": rounds[0][1][k].get("true_task"), **d})
    by_task = Counter(rounds[0][1][k].get("true_task") for k in common)
    return {"rounds": [n for n, _ in rounds], "n_common": len(common), "missing": missing,
            "counts": dict(cnt), "by_task": dict(by_task), "diffs": diffs}


def _report(name: str, res: Mapping, show: int) -> list[str]:
    c = res["counts"]
    n = res["n_common"]
    path_bad = sum(1 for d in res["diffs"] if "识别" in d or "后段类" in d or "前段" in d)
    out_bad = sum(1 for d in res["diffs"] if "生成长度" in d or "输出文本" in d)
    lines = [f"== {name}：{len(res['rounds'])} 轮，{n} 个样本各轮都有（{dict(res['by_task'])}）",
             f"   路径（识别 / 后段类 / 前段）全一样：{n - path_bad}/{n}"
             + (f"   ✗ {path_bad} 个不一样" if path_bad else "   ✓"),
             f"   输出（长度 + 文本）全一样：{n - out_bad}/{n}" + (f"   ✗ {out_bad} 个不一样" if out_bad else "   ✓"),
             f"   同类换副本（least-free 挑的，正常）：{c.get('同类换副本', 0)}",
             "   各项不一样的样本数：" + ("，".join(f"{k} {v}" for k, v in c.items() if k != "完全一致") or "无")]
    for nm, ks in res["missing"].items():
        lines.append(f"   ⚠ {nm} 少了 {len(ks)} 个样本：{ks[:5]}{'…' if len(ks) > 5 else ''}")
    shown = [d for d in res["diffs"] if set(d) - {"sample", "true_task", "同类换副本"}][:show]
    for d in shown:
        det = "；".join(f"{k}={v}" for k, v in d.items() if k not in ("sample", "true_task", "同类换副本"))
        lines.append(f"     {d['sample']}（{d['true_task']}）：{det}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="同一个方案的各轮结果（不给 --group 时用）")
    ap.add_argument("--group", action="append", default=[], metavar="名字=通配",
                    help="一个方案的各轮，可重复（通配加引号，用 _r[0-9]* 免得匹配到汇总文件）")
    ap.add_argument("--show", type=int, default=10, help="每个方案最多列几个不一致的样本")
    ap.add_argument("--out", type=Path, default=None, help="把逐样本的差异写成 JSON")
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
    result, rc = {}, 0
    for name, files in groups.items():
        docs = []
        for f in files:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            if d.get("requests"):
                docs.append((Path(f).name, d))
            else:
                print(f"  跳过 {f}：不是一轮 bench 结果", file=sys.stderr)
        if len(docs) < 2:
            print(f"✗ {name}：至少要两轮（匹配到 {len(docs)} 份）", file=sys.stderr)
            rc = 2
            continue
        res = compare_rounds(docs)
        result[name] = res
        print("\n".join(_report(name, res, a.show)))
    if a.out:
        a.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写 {a.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
