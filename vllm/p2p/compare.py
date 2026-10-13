# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2 验收：同一批测试样本，p2p-framework 与 vLLM 的贪心输出与时延逐条对比。**按文件路径运行。**

    python3 vllm/p2p/compare.py \\
        --a /home/ubuntu/p2p-framework/results/e1_fp16.json   --a-name p2p \\
        --b /home/ubuntu/vllm-p2p/results/a3_shadow.json      --b-name vllm \\
        --model-dir /home/ubuntu/p2p-framework/weights-q3-30b [--prefix 64] [--out cmp.json]

两份都是 `results/*.json` 格式（p2p-framework 的 `--save-results` 与 `vllm/p2p/bench.py` 同构）。
按 `case_id` 配对（TestSamples/cases.jsonl 每条都有 id）；没有 id 时退到 (task, prompt)。

输出的判据（VLLM_MIGRATION.md A2）：两边都 fp16 时，**贪心前 64 token 完全一致的比例 ≥ 95%**。
文本按同一个 tokenizer 重新分词再比 token —— 两边都是 skip_special 解码出的文本，重新分词得到的
就是原 token 序列（byte-level BPE 对它自己的输出是可逆的；少数在边界上重分词不同的会表现为
很靠后的分叉，不影响前 64 的判据）。另报：共同前缀长度、整段相同的比例、逐 task 的时延对比。

只要标准库 + tokenizers（vllm-p2p 的 venv 里有）+ p2p-framework 源码目录。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

try:
    from identifier import use_p2p_framework                     # 同目录
except ImportError:                                               # pragma: no cover
    from vllm.p2p.identifier import use_p2p_framework

__all__ = ["pair_rows", "common_prefix", "compare", "main"]


def _key(r: Mapping) -> tuple:
    return ("id", r["case_id"]) if r.get("case_id") else ("p", r.get("true_task"), r.get("prompt"))


def pair_rows(a: Sequence[Mapping], b: Sequence[Mapping]) -> tuple[list[tuple[dict, dict]], int, int]:
    """→ (配上的 (a 行, b 行), a 里没配上的条数, b 里没配上的条数)。同一个 key 多次出现时按顺序配。"""
    pool: dict[tuple, list[dict]] = defaultdict(list)
    for r in b:
        pool[_key(r)].append(dict(r))
    out = []
    miss_a = 0
    for r in a:
        q = pool.get(_key(r))
        if q:
            out.append((dict(r), q.pop(0)))
        else:
            miss_a += 1
    return out, miss_a, sum(len(v) for v in pool.values())


def common_prefix(x: Sequence[int], y: Sequence[int]) -> int:
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 1) if xs else None


def _tpot(r: Mapping) -> float | None:
    n = int(r.get("n_generated") or 0)
    if n < 2:
        return None
    return (r["total_ms"] - r["prefill_ms"]) / (n - 1)


def compare(pairs: Sequence[tuple[dict, dict]], encode, *, prefix: int = 64) -> dict:
    by: dict[str, list] = defaultdict(list)
    for ra, rb in pairs:
        ta, tb = encode(ra.get("text") or ""), encode(rb.get("text") or "")
        k = min(prefix, len(ta), len(tb))
        cp = common_prefix(ta, tb)
        # 前 prefix 个 token 一致：两边都至少有 prefix 个且前 prefix 相同；或者两边都更短但整段相同
        ok = (cp >= prefix) or (ta == tb)
        by[ra.get("true_task") or "?"].append({
            "case": ra.get("case_id") or ra.get("req"), "ok": ok, "cp": cp, "k": k,
            "same_text": (ra.get("text") or "") == (rb.get("text") or ""),
            "len_a": len(ta), "len_b": len(tb),
            "lat_a": ra.get("total_ms"), "lat_b": rb.get("total_ms"),
            "ttft_a": ra.get("prefill_ms"), "ttft_b": rb.get("prefill_ms"),
            "tpot_a": _tpot(ra), "tpot_b": _tpot(rb),
            "snip_a": (ra.get("text") or "")[:160], "snip_b": (rb.get("text") or "")[:160],
        })

    def agg(rows):
        n = len(rows)
        return {
            "n": n,
            f"prefix{prefix}_agree": round(sum(r["ok"] for r in rows) / n, 4) if n else None,
            "same_text": round(sum(r["same_text"] for r in rows) / n, 4) if n else None,
            "common_prefix_median": _med([r["cp"] for r in rows]),
            "latency_ms_p50": [_med([r["lat_a"] for r in rows]), _med([r["lat_b"] for r in rows])],
            "ttft_ms_p50": [_med([r["ttft_a"] for r in rows]), _med([r["ttft_b"] for r in rows])],
            "tpot_ms_p50": [_med([r["tpot_a"] for r in rows]), _med([r["tpot_b"] for r in rows])],
            "gen_len_median": [_med([r["len_a"] for r in rows]), _med([r["len_b"] for r in rows])],
        }

    allr = [r for rs in by.values() for r in rs]
    return {"prefix": prefix, "all": agg(allr),
            "by_task": {u: agg(rs) for u, rs in sorted(by.items())},
            "diverged": sorted((r for r in allr if not r["ok"]), key=lambda r: r["cp"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True, type=Path, help="一份 results/*.json（如 p2p-framework E1）")
    ap.add_argument("--b", required=True, type=Path, help="另一份（如 vLLM A3）")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--model-dir", required=True, help="有 tokenizer.json 的目录")
    ap.add_argument("--p2p-framework", default=None)
    ap.add_argument("--prefix", type=int, default=64)
    ap.add_argument("--show", type=int, default=8, help="列出前几条分叉的样本")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    use_p2p_framework(a.p2p_framework)
    from p2pmoe.runtime.text import Tokenizer

    tok = Tokenizer.from_model_dir(a.model_dir)
    ra = json.loads(a.a.read_text(encoding="utf-8"))["requests"]
    rb = json.loads(a.b.read_text(encoding="utf-8"))["requests"]
    pairs, ma, mb = pair_rows(ra, rb)
    if not pairs:
        print("✗ 一条都配不上 —— 两份结果不是同一个测试集？（按 case_id，没有就按 task+prompt）")
        return 2
    res = compare(pairs, lambda s: tok.encode(s), prefix=a.prefix)
    A, Bn = a.a_name, a.b_name
    print(f"配上 {len(pairs)} 条（{A} 多 {ma} 条、{Bn} 多 {mb} 条没配上）；判据：贪心前 {a.prefix} token 一致")
    hdr = (f"  {'task':<11}{'条数':>5}{'前缀一致':>9}{'全文相同':>9}{'公共前缀':>9}"
           f"   {'时延p50 ms':>18}   {'首token p50 ms':>18}   {'每token p50 ms':>18}")
    print(hdr)
    print(f"  {'':<11}{'':>5}{'':>9}{'':>9}{'(中位)':>9}   {A + ' / ' + Bn:>18}   "
          f"{A + ' / ' + Bn:>18}   {A + ' / ' + Bn:>18}")

    def row(name, g):
        def two(x):
            return f"{x[0]} / {x[1]}"
        print(f"  {name:<11}{g['n']:>5}{g[f'prefix{a.prefix}_agree']:>9.1%}{g['same_text']:>9.1%}"
              f"{g['common_prefix_median']:>9}   {two(g['latency_ms_p50']):>18}   "
              f"{two(g['ttft_ms_p50']):>18}   {two(g['tpot_ms_p50']):>18}")
    for u, g in res["by_task"].items():
        row(u, g)
    row("全部", res["all"])
    agree = res["all"][f"prefix{a.prefix}_agree"]
    print(f"\n  A2 验收（≥ 95%）：{'✓' if agree >= 0.95 else '✗'} {agree:.1%}")
    if res["diverged"]:
        print(f"\n  分叉最早的 {min(a.show, len(res['diverged']))} 条（第几个 token 开始不同）：")
        for r in res["diverged"][:a.show]:
            print(f"    {r['case']}  第 {r['cp']} 个 token（{A} {r['len_a']} / {Bn} {r['len_b']} token）")
            print(f"      {A}: {r['snip_a'][:110]!r}")
            print(f"      {Bn}: {r['snip_b'][:110]!r}")
    if a.out:
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  明细 → {a.out}")
    return 0 if agree >= 0.95 else 1


if __name__ == "__main__":
    sys.exit(main())
