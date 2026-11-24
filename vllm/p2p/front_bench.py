# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""前段（前 L₀ 层、全专家）实测：并发下的单步、prefill、固定开销、真实运行里的占用率。

单前段方案（一个前段装 Qwen3-30B-A3B 的前 2 层、所有流量从它分发）里，前段的代价是
「并发数 × 单步时间」再加上 prefill 期间别的请求全部停下的那一段。这里把这几样分开量。
入口在 `layer_bench.py` 里（`front` / `front-cluster` / `front-occ`），也可以直接
`python -m vllm.p2p.front_bench run|cluster|occ`。

1. 并发下的单步（`run`，每种卡一台）
   起一个**真的前段引擎**（与部署同一个 `Qwen3MoePartialForCausalLM`、`p2p_stage.role=front`、
   `layers=[0, L₀]`、全专家、`p2p_stats.emit_lr`；默认随机权重，`--real-weights` 读 checkpoint），
   开步级记录（`VLLM_P2P_STEP_TRACE`）。c = 1、2、4、6 条请求同时在飞，每条：发 prompt →
   等前段交出 prompt 的 hidden → 「后段」（这里用 sleep 代替，`--back-ms`）→ `p2p_append` 下一个
   token → 等前段交出这一位的 hidden → …，与真部署里前段收到的东西完全一样。
   每个 token 拿客户端的两个时刻（发 token、收到 hidden）去对步级记录里算它的那一步，拆成：

       时延 = 排队（发出 → 那一步开始：进程间转发 + 等前一步算完 + 调度）
            + 服务（那一步的用时：调度 + 前向 + 输出整理）
            + 回传（那一步结束 → 客户端收到）

   「等待」= 时延 − 服务。每档报均值与 p95，另报批大小、前段忙的比例。
   `--back-ms` 是逗号分隔的几种「后段」：`0` = 后段不花时间（token 一回来马上送下一个，量纯的
   batch=c 单步）；`20:40` = 每次 sleep 20–40 ms 均匀随机（各条请求错开，量真实的排队）。
   被 prefill 挡住的 token（它那一步里有别的请求在 prefill）单独计，不混进 decode 的数。
2. prefill：单条 128 / 512 / 1024 token（`--prefill`），各 `--repeats` 次；再在后台有
   `--prefill-bg` 条 decode 时插 prefill，看挡住的那些 decode token 多等了多少。
3. 固定开销：
   * 层数相减：同一张卡再起 L₀ 层（不开统计）、1 层（不开统计）两个引擎，量 batch=c 的单步 →
     每层多少 ms、与层数无关的固定部分多少 ms；
   * 统计回传：开 / 不开 `p2p_stats` 的同一套 prefill、decode 相减；
   * 词嵌入、hidden 与统计的 GPU→CPU 拷贝：引擎都停了以后在本进程里用 CUDA event 单独量；
   * 选择器打分：给了 `--lr-model` 且找得到 p2p-framework 时，拿引擎交出的真 `p2p_feat` 跑
     网关同一个 `classify`（网关在控制机上跑它，CPU 的数以控制机为准；`occ` 里有真实运行的值）。
4. 真实运行的占用率（`occ`，控制机上、只用标准库）：拿一次 continuous 的结果 JSON 和
   `steptrace.py collect` 收回来的步级记录，算每个引擎忙的比例（其中 prefill 占多少）、
   每个 decode token 经历的那一步平均多久（服务），对比网关记的每位置「引擎内毫秒」
   （`front_engine_ms_p50`：代理收到 token → 交出 hidden）→ 排队 ≈ 引擎内 − 服务，以及它占 TPOT 的比例。

例::

    # 一台节点（卡要空着；vllm-p2p 的 venv）
    python -m vllm.p2p.layer_bench front --node N13 --out /tmp/front_bench_N13.json
    # 控制机：每种卡一台（hosts 行尾注释写卡型），或 --nodes N13,N10,1.5
    python3 vllm/p2p/layer_bench.py front-cluster --hosts $HOSTS --nodes one-per-gpu --out-dir results/front_bench
    # 控制机：真实运行的占用率
    python3 vllm/p2p/layer_bench.py front-occ --result results/b1f_pf6_n150.json \\
        --steps results/steps_b1f_pf6_n150/*.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import itertools
import json
import math
import os
import random
import shlex
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = ["parse_back_ms", "match_tokens", "summarise_tokens", "summarise_steps", "prefill_rows",
           "fixed_overhead", "occupancy", "drive_decode", "drive_prefill", "main"]

ARCH = "Qwen3MoePartialForCausalLM"
TRACE_ENV = "VLLM_P2P_STEP_TRACE"
DEFAULT_CONC = "1,2,4,6"
DEFAULT_PREFILL = "128,512,1024"
DEFAULT_BACK = "0,20:40"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "StrictHostKeyChecking=accept-new"]
_SEQ = itertools.count()                 # 请求 id 后缀：同一个引擎上各段的 id 不能重（输出按 id 认）


# --------------------------------------------------------------------------- #
# 纯函数（控制机上也能跑；单测直接测）
# --------------------------------------------------------------------------- #
def _stats(xs: Sequence[float]) -> dict:
    xs = sorted(float(x) for x in xs if x is not None and not math.isnan(float(x)))
    if not xs:
        return {"n": 0}
    k95 = min(len(xs) - 1, max(0, math.ceil(0.95 * len(xs)) - 1))
    return {"n": len(xs), "mean": round(st.mean(xs), 3), "p50": round(xs[len(xs) // 2], 3),
            "p95": round(xs[k95], 3), "max": round(xs[-1], 3)}


def parse_back_ms(spec: str) -> list[tuple[float, float]]:
    """「0,20:40」→ [(0, 0), (20, 40)]（毫秒，均匀随机的上下界）。"""
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition(":")
        lo_f, hi_f = float(lo), float(hi or lo)
        if lo_f < 0 or hi_f < lo_f:
            raise ValueError(f"--back-ms {part!r}：要 下:上（ms，0 ≤ 下 ≤ 上）")
        out.append((lo_f, hi_f))
    if not out:
        raise ValueError("--back-ms 是空的")
    return out


def back_label(b: tuple[float, float]) -> str:
    return f"{b[0]:g}" if b[0] == b[1] else f"{b[0]:g}:{b[1]:g}"


def _ints(spec: str) -> list[int]:
    out = [int(x) for x in str(spec).split(",") if x.strip()]
    if not out or min(out) <= 0:
        raise ValueError(f"{spec!r} 要是逗号分隔的正整数")
    return out


def _end(s: Mapping) -> float:
    return float(s["t"]) + float(s["ms"]) / 1e3


def match_tokens(tokens: Sequence[Mapping], steps: Sequence[Mapping], *,
                 slack_s: float = 0.002) -> list[dict]:
    """每个 token（客户端 `t_in` 发出 token、`t_out` 收到这一位的 hidden，都是 time.time()）
    → 算它的那一步：`t_out` 之前最后结束的一步（引擎是一步一步串行的，结果在那一步结束后才发出）。
    那一步开始得比 `t_in` 还早（超过 slack）= 对不上（记录缺了），`step=None`。"""
    ss = sorted(steps, key=_end)
    ends = [_end(s) for s in ss]
    out = []
    for tk in tokens:
        t_in, t_out = float(tk["t_in"]), float(tk["t_out"])
        k = bisect.bisect_right(ends, t_out + 1e-4) - 1
        row = dict(tk, lat_ms=round((t_out - t_in) * 1e3, 3))
        if k < 0 or float(ss[k]["t"]) < t_in - slack_s:
            row["step"] = None
        else:
            s = ss[k]
            row.update(step=k, svc_ms=float(s["ms"]),
                       queue_ms=round((float(s["t"]) - t_in) * 1e3, 3),
                       out_ms=round((t_out - ends[k]) * 1e3, 3),
                       wait_ms=round((t_out - t_in) * 1e3 - float(s["ms"]), 3),
                       batch=int(s.get("n", 0)), mixed=bool(s.get("npf")))
        out.append(row)
    return out


def summarise_tokens(rows: Sequence[Mapping]) -> dict:
    """match_tokens 的结果 → 时延 / 服务 / 等待（= 时延 − 服务）/ 排队 / 回传，均值与 p95。
    `clean` 只算没被 prefill 挡住的 token；`mixed` 是被挡住的。"""
    ok = [r for r in rows if r.get("step") is not None]
    clean = [r for r in ok if not r.get("mixed")]
    mixed = [r for r in ok if r.get("mixed")]

    def part(rs):
        if not rs:
            return {"n": 0}
        return {"n": len(rs), "lat": _stats([r["lat_ms"] for r in rs]),
                "svc": _stats([r["svc_ms"] for r in rs]), "wait": _stats([r["wait_ms"] for r in rs]),
                "queue": _stats([r["queue_ms"] for r in rs]), "out": _stats([r["out_ms"] for r in rs]),
                "batch_mean": round(st.mean(r["batch"] for r in rs), 3)}
    return {"tokens": len(rows), "unmatched": len(rows) - len(ok), "clean": part(clean),
            "mixed": part(mixed)}


def summarise_steps(steps: Sequence[Mapping], t0: float | None = None,
                    t1: float | None = None) -> dict:
    """步级记录 → 步数、忙的比例（其中 prefill 步占多少）、纯 decode 步按批大小的用时、
    每个 decode token 平均经历多长的一步（`svc_per_token_ms`，按 token 加权，含被 prefill 拖长的步）。"""
    rs = sorted((r for r in steps if (t0 is None or _end(r) >= t0) and (t1 is None or r["t"] <= t1)),
                key=lambda r: r["t"])
    if not rs:
        return {"steps": 0}
    # 给了窗口就按窗口算（窗口里闲着的时间也算进分母）；没给就用记录自己的首尾
    a = rs[0]["t"] if t0 is None else t0
    b = _end(rs[-1]) if t1 is None else t1
    span = max(b - a, 1e-9)
    busy = sum(r["ms"] for r in rs) / 1e3
    pf_busy = sum(r["ms"] for r in rs if r.get("npf")) / 1e3
    by_b: dict[int, list[float]] = {}
    for r in rs:
        if not r.get("npf"):
            by_b.setdefault(int(r["n"]), []).append(float(r["ms"]))
    ndec = [(max(int(r.get("n", 0)) - int(r.get("npf", 0)), 0), float(r["ms"])) for r in rs]
    tok = sum(n for n, _ in ndec)
    return {
        "steps": len(rs), "span_s": round(span, 3),
        "busy_frac": round(min(busy / span, 1.0), 4),
        "prefill_busy_frac": round(min(pf_busy / span, 1.0), 4),
        "prefill_steps": sum(1 for r in rs if r.get("npf")),
        "batch_mean": round(st.mean(int(r["n"]) for r in rs), 3),
        "decode_ms_by_batch": {str(k): _stats(v) for k, v in sorted(by_b.items())},
        "decode_tokens": tok,
        "svc_per_token_ms": round(sum(n * ms for n, ms in ndec) / tok, 3) if tok else None,
        "steps_per_s": round(len(rs) / span, 2),
    }


def prefill_rows(prefills: Sequence[Mapping], steps: Sequence[Mapping]) -> list[dict]:
    """每次 prefill（`t_add` 发出、`t_done` prompt 的 hidden 全部收到）→ 期间有 prefill 的那几步
    用时之和（`svc_ms`，长 prompt 被切块时是几步）、端到端 `lat_ms`、那几步里同批的 decode 条数。"""
    out = []
    for p in prefills:
        a, b = float(p["t_add"]), float(p["t_done"])
        hit = [s for s in steps if s.get("npf") and float(s["t"]) >= a - 0.002 and _end(s) <= b + 1e-4]
        out.append(dict(p, lat_ms=round((b - a) * 1e3, 3),
                        svc_ms=round(sum(float(s["ms"]) for s in hit), 3) if hit else None,
                        n_steps=len(hit),
                        decode_in_batch=sum(int(s.get("n", 0)) - int(s.get("npf", 0)) for s in hit)))
    return out


def fixed_overhead(main_ms: Mapping[str, float], nostats_ms: Mapping[str, float] | None,
                   l1_ms: Mapping[str, float] | None, layers: int) -> dict:
    """各档 batch 的纯单步（back=0 的服务 p50）：main = L₀ 层开统计、nostats = L₀ 层不开、
    l1 = 1 层不开。→ 每层 ms、固定 ms（与层数无关：调度、词嵌入、输出整理…）、统计 ms。"""
    out = {}
    for c, m in main_ms.items():
        row: dict[str, Any] = {"step_ms": m}
        ns = (nostats_ms or {}).get(c)
        l1 = (l1_ms or {}).get(c)
        if ns is not None:
            row["stats_ms"] = round(m - ns, 3)
        if ns is not None and l1 is not None and layers > 1:
            per = (ns - l1) / (layers - 1)
            row["per_layer_ms"] = round(per, 3)
            row["layers_ms"] = round(per * layers, 3)
            row["fixed_ms"] = round(l1 - per, 3)
        out[c] = row
    return out


# --------------------------------------------------------------------------- #
# 驱动前段引擎（真引擎、测试里的假引擎都是 tiered.Engine）
# --------------------------------------------------------------------------- #
class _Tap:
    """读前段引擎的输出：每条请求的 hidden 交到了哪一位；等某一位交出时叫醒等的人。"""

    def __init__(self, eng):
        self.eng = eng
        self.end: dict[str, int] = {}
        self.hist: dict[str, list[tuple[int, float]]] = {}   # 每次交出：(交到第几位, 收到时刻)
        self.feat: dict[str, Any] = {}
        self.wait: dict[str, tuple[int, asyncio.Future]] = {}
        self.err: BaseException | None = None
        self.task: asyncio.Task | None = None

    def start(self) -> "_Tap":
        self.task = asyncio.get_running_loop().create_task(self._loop())
        return self

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except (asyncio.CancelledError, Exception):      # noqa: BLE001
                pass

    async def _loop(self) -> None:
        try:
            while True:
                outs = await self.eng.outputs()
                now = time.time()
                for o in outs.outputs:
                    h = getattr(o, "p2p_hidden", None)
                    if h is None:
                        continue
                    rid = o.request_id
                    ft = getattr(o, "p2p_feat", None)
                    if ft is not None:
                        self.feat.setdefault(rid, ft)
                    e = int(o.p2p_pos) + int(h.shape[0])
                    if e > self.end.get(rid, 0):
                        self.end[rid] = e
                        self.hist.setdefault(rid, []).append((e, now))
                    w = self.wait.get(rid)
                    if w is not None and self.end[rid] >= w[0] and not w[1].done():
                        w[1].set_result(now)
        except asyncio.CancelledError:
            raise
        except BaseException as e:                           # noqa: BLE001 — 引擎死了
            self.err = e
            for _, f in self.wait.values():
                if not f.done():
                    f.set_exception(e)

    async def until(self, rid: str, pos: int, timeout: float) -> float:
        if self.err is not None:
            raise RuntimeError(f"前段引擎出错：{self.err}")
        if self.end.get(rid, 0) >= pos:          # 等的人还没挂上就已经到了：用当时收到的时刻
            return next(t for e, t in self.hist[rid] if e >= pos)
        fut = asyncio.get_running_loop().create_future()
        self.wait[rid] = (pos, fut)
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self.wait.pop(rid, None)


def _request(rid: str, prompt: list[int], max_tokens: int):
    from vllm.p2p.tiered import _core_request

    # 与网关一样带 p2p_id、p2p_body（整条 prompt 都算 body）：前段的识别统计按它数
    return _core_request(rid, prompt=prompt, embeds=None, max_tokens=max_tokens, eos=None, logprobs=0,
                         ignore_eos=True, stop_ids=(), arrival=time.time(),
                         extra_args={"p2p_id": rid, "p2p_body": f"0:{len(prompt)}"})


async def _one_request(eng, tap: _Tap, rid: str, prompt: list[int], *, n_tokens: int | None,
                       back: tuple[float, float], rng: random.Random, vocab: int, timeout: float,
                       tokens: list, prefills: list, stop: asyncio.Event | None = None) -> None:
    P = len(prompt)
    rid = f"{rid}#{next(_SEQ)}"
    t_add = time.time()
    await eng.add(_request(rid, prompt, max_tokens=(n_tokens or 4096) + 8))
    t_done = await tap.until(rid, P, timeout)
    prefills.append({"rid": rid, "P": P, "t_add": t_add, "t_done": t_done})
    i = 0
    try:
        while (n_tokens is None or i < n_tokens) and not (stop is not None and stop.is_set()):
            d = rng.uniform(*back) / 1e3
            if d > 0:
                await asyncio.sleep(d)
            t_in = time.time()
            await eng.fire("p2p_append", [[rid, [rng.randrange(100, max(vocab - 100, 101))]]])
            t_out = await tap.until(rid, P + i + 1, timeout)
            tokens.append({"rid": rid, "i": i, "t_in": t_in, "t_out": t_out})
            i += 1
    finally:
        await eng.abort([rid])


async def drive_decode(eng, tap: _Tap, *, c: int, n_tokens: int, reqs_per_slot: int, prompt_len: int,
                       back: tuple[float, float], vocab: int, seed: int, tag: str,
                       timeout: float = 120.0) -> dict:
    """c 条请求同时在飞（每个「槽」先后跑 reqs_per_slot 条，每条 prompt 之后 n_tokens 个 token）。"""
    tokens: list = []
    prefills: list = []

    async def slot(k):
        rng = random.Random(seed * 1000 + k)
        await asyncio.sleep(k * max(back[1], 2.0) / 1e3 / max(c, 1))     # 错开一点，别全挤在第一步
        for j in range(reqs_per_slot):
            prompt = [rng.randrange(100, max(vocab - 100, 101)) for _ in range(prompt_len)]
            await _one_request(eng, tap, f"{tag}-c{c}-{k}-{j}", prompt, n_tokens=n_tokens, back=back,
                               rng=rng, vocab=vocab, timeout=timeout, tokens=tokens, prefills=prefills)

    t0 = time.time()
    await asyncio.gather(*(slot(k) for k in range(c)))
    return {"c": c, "back": list(back), "t0": t0, "t1": time.time(), "tokens": tokens,
            "prefills": prefills}


async def drive_prefill(eng, tap: _Tap, *, lens: Sequence[int], repeats: int, bg: int,
                        bg_prompt: int, back: tuple[float, float], vocab: int, seed: int, tag: str,
                        gap_s: float = 0.05, timeout: float = 120.0, bg_tokens: int = 512) -> dict:
    """单条 prefill（各长度 1 次预热 + repeats 次）；bg > 0 时后台同时有 bg 条一直在 decode。"""
    rng = random.Random(seed)
    tokens: list = []
    bg_prefills: list = []
    prefills: list = []
    stop = asyncio.Event()

    async def bg_slot(k):
        r = random.Random(seed * 7919 + k)
        while not stop.is_set():                    # 一条做到 bg_tokens 个就换一条（别超 max_model_len）
            prompt = [r.randrange(100, max(vocab - 100, 101)) for _ in range(bg_prompt)]
            await _one_request(eng, tap, f"{tag}-bg{k}", prompt, n_tokens=bg_tokens, back=back, rng=r,
                               vocab=vocab, timeout=timeout, tokens=tokens, prefills=bg_prefills,
                               stop=stop)

    bgs = [asyncio.get_running_loop().create_task(bg_slot(k)) for k in range(bg)]
    try:
        if bg:
            while len(bg_prefills) < bg:                    # 后台的都过了 prompt 再开始插
                await asyncio.sleep(0.01)
                for t in bgs:
                    if t.done() and t.exception():
                        raise t.exception()
            await asyncio.sleep(0.2)
        t0 = time.time()
        for P in lens:
            for r in range(repeats + 1):
                rid = f"{tag}-pf{P}-{r}#{next(_SEQ)}"
                prompt = [rng.randrange(100, max(vocab - 100, 101)) for _ in range(P)]
                t_add = time.time()
                await eng.add(_request(rid, prompt, max_tokens=8))
                t_done = await tap.until(rid, P, timeout)
                await eng.abort([rid])
                if r > 0:                                    # 第 0 次是这个长度的预热
                    prefills.append({"rid": rid, "P": P, "t_add": t_add, "t_done": t_done})
                await asyncio.sleep(gap_s)
        t1 = time.time()
    finally:
        stop.set()
        await asyncio.gather(*bgs, return_exceptions=True)
    return {"bg": bg, "back": list(back), "t0": t0, "t1": t1, "prefills": prefills,
            "tokens": [t for t in tokens if t0 <= t["t_in"] <= t1]}


# --------------------------------------------------------------------------- #
# 一个引擎上的全套（真引擎与假引擎共用；steps_fn 返回步级记录）
# --------------------------------------------------------------------------- #
async def bench_engine(eng, steps_fn: Callable, *, conc: Sequence[int], backs: Sequence[tuple],
                       n_tokens: int, reqs_per_slot: int, prompt_len: int, prefill: Sequence[int],
                       repeats: int, prefill_bg: int, vocab: int, seed: int = 0, tag: str = "fb",
                       timeout: float = 120.0, log: Callable[[str], None] = print,
                       warmup: bool = True) -> dict:
    """→ {decode: {back: {c: 汇总}}, prefill: {P: 汇总}, prefill_bg: …, feats: [...]}"""
    tap = _Tap(eng).start()
    out: dict[str, Any] = {"decode": {}, "prefill": {}, "prefill_bg": None}
    try:
        if warmup:                                           # 各个 batch 形状先走一遍
            await drive_decode(eng, tap, c=max(conc), n_tokens=4, reqs_per_slot=1,
                               prompt_len=prompt_len, back=(0.0, 0.0), vocab=vocab, seed=seed + 99,
                               tag=f"{tag}w", timeout=timeout)
        for b in backs:
            row = out["decode"][back_label(b)] = {}
            for c in conc:
                d = await drive_decode(eng, tap, c=c, n_tokens=n_tokens, reqs_per_slot=reqs_per_slot,
                                       prompt_len=prompt_len, back=b, vocab=vocab, seed=seed + c,
                                       tag=tag, timeout=timeout)
                steps = await steps_fn()
                tk = summarise_tokens(match_tokens(d["tokens"], steps))
                sp = summarise_steps(steps, d["t0"], d["t1"])
                row[str(c)] = {**tk, "engine": sp,
                               "front_tok_per_s": round(len(d["tokens"]) / max(d["t1"] - d["t0"], 1e-9), 1)}
                cl = tk["clean"]
                if cl.get("n"):
                    log(f"    后段 {back_label(b):>6} ms  c={c}  服务 {cl['svc']['mean']:.2f}/"
                        f"{cl['svc']['p95']:.2f}  等待 {cl['wait']['mean']:.2f}/{cl['wait']['p95']:.2f}  "
                        f"时延 {cl['lat']['mean']:.2f}/{cl['lat']['p95']:.2f} ms（均值/p95）"
                        f"  批 {cl['batch_mean']:.2f}  忙 {sp.get('busy_frac', 0) * 100:.0f}%")
        if prefill:
            d = await drive_prefill(eng, tap, lens=prefill, repeats=repeats, bg=0, bg_prompt=prompt_len,
                                    back=backs[-1], vocab=vocab, seed=seed + 500, tag=f"{tag}p",
                                    timeout=timeout)
            rows = prefill_rows(d["prefills"], await steps_fn())
            for P in prefill:
                rs = [r for r in rows if r["P"] == P]
                out["prefill"][str(P)] = {"svc": _stats([r["svc_ms"] for r in rs]),
                                          "lat": _stats([r["lat_ms"] for r in rs]),
                                          "steps": _stats([r["n_steps"] for r in rs])}
                s = out["prefill"][str(P)]
                if s["svc"].get("n"):
                    log(f"    prefill P={P:<5} 那一步 {s['svc']['mean']:.2f} ms（p95 {s['svc']['p95']:.2f}）"
                        f"  端到端 {s['lat']['mean']:.2f} ms")
            if prefill_bg:
                d = await drive_prefill(eng, tap, lens=prefill, repeats=repeats, bg=prefill_bg,
                                        bg_prompt=prompt_len, back=backs[-1], vocab=vocab,
                                        seed=seed + 700, tag=f"{tag}q", timeout=timeout)
                steps = await steps_fn()
                rows = prefill_rows(d["prefills"], steps)
                tk = summarise_tokens(match_tokens(d["tokens"], steps))
                per = {}
                for P in prefill:
                    rs = [r for r in rows if r["P"] == P]
                    per[str(P)] = {"svc": _stats([r["svc_ms"] for r in rs]),
                                   "lat": _stats([r["lat_ms"] for r in rs])}
                out["prefill_bg"] = {"bg": prefill_bg, "back": back_label(backs[-1]), "prefill": per,
                                     "decode": tk}
                m, cl = tk["mixed"], tk["clean"]
                if m.get("n") and cl.get("n"):
                    log(f"    后台 {prefill_bg} 条 decode 时插 prefill：被挡的 token {m['n']} 个，时延 "
                        f"{m['lat']['mean']:.2f}/{m['lat']['p95']:.2f} ms vs 没被挡的 "
                        f"{cl['lat']['mean']:.2f}/{cl['lat']['p95']:.2f} ms")
        out["feats"] = list(tap.feat.values())
    finally:
        await tap.close()
    return out


# --------------------------------------------------------------------------- #
# run：本机 GPU（vllm-p2p 的 venv）
# --------------------------------------------------------------------------- #
def _front_engine(model_dir: str, *, layers: int, stats: bool, a, trace: Path, mml: int):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.p2p.tiered import Engine
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.core_client import EngineCoreClient
    from vllm.v1.executor import Executor

    from vllm.p2p import models
    cfg_json = models.load_config(model_dir)
    fam = models.family_of(cfg_json)
    # 与 stages.py 给前段的 hf_overrides 相同（全专家、前段出口识别统计）
    ov: dict = {"architectures": [fam.arch], "p2p_layer_experts": {}, "p2p_channel": "front-bench",
                "p2p_stage": {"role": "front", "layers": [0, int(layers)]}}
    dense: frozenset = models.dense_layers(cfg_json) if cfg_json is not None else frozenset()
    if fam.name == "glm4_moe_lite" and not a.real_weights:
        # 随机权重时把 GLM 的稠密第 0 层也换成 MoE 层：L₀ 层 / 1 层相减求「每层」要求层是同一种
        ov.update(first_k_dense_replace=0, num_nextn_predict_layers=0)
        dense = frozenset()
    if stats:
        ov["p2p_stats"] = {"lr_layers": [l for l in range(int(layers)) if l not in dense],
                           "emit_lr": True, "write": False, "prefill_only": True}
    kw = dict(model=model_dir, dtype=a.dtype, hf_overrides=ov, skip_tokenizer_init=True,
              enable_prompt_embeds=False, async_scheduling=False, enable_prefix_caching=False,
              gpu_memory_utilization=a.gpu_mem, max_model_len=mml, max_num_seqs=a.max_num_seqs,
              enforce_eager=a.eager, seed=0, load_format="auto" if a.real_weights else "dummy")
    backend = a.attention_backend or fam.attention_backend
    if backend != "auto":
        kw["attention_backend"] = backend
    cfg = EngineArgs(**kw).create_engine_config(UsageContext.ENGINE_CONTEXT)
    trace.parent.mkdir(parents=True, exist_ok=True)
    if trace.exists():
        trace.unlink()
    old = os.environ.get(TRACE_ENV)
    os.environ[TRACE_ENV] = str(trace)                       # EngineCore 子进程继承
    try:
        client = EngineCoreClient.make_async_mp_client(cfg, Executor.get_class(cfg), log_stats=False)
    finally:
        if old is None:
            os.environ.pop(TRACE_ENV, None)
        else:
            os.environ[TRACE_ENV] = old
    return Engine("front", "front", client)


def _trace_reader(eng, path: Path) -> Callable:
    async def steps():
        await eng.client.call_utility_async("p2p_trace_flush")
        if not path.is_file():
            return []
        rows = []
        for ln in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
        return rows
    return steps


def micro_overheads(*, hidden: int, vocab: int, n_experts: int, top_k: int, layers: int, dtype: str,
                    ns: Sequence[int], repeats: int = 200) -> dict:
    """词嵌入（F.embedding）与 GPU→CPU 拷贝（hidden；统计的 top-k 下标 int16 + 权重 fp32）。"""
    import torch
    import torch.nn.functional as F

    dt = {"half": torch.float16, "float16": torch.float16, "bfloat16": torch.bfloat16,
          "float32": torch.float32}.get(dtype, torch.float16)
    dev = torch.device("cuda")
    table = torch.empty(vocab, hidden, dtype=dt, device=dev).normal_()
    out = {}
    for n in ns:
        ids = torch.randint(0, vocab, (n,), device=dev)
        for _ in range(10):
            F.embedding(ids, table)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(repeats):
            F.embedding(ids, table)
        e1.record()
        torch.cuda.synchronize()
        emb = e0.elapsed_time(e1) / repeats

        def d2h(*ts):
            for _ in range(5):
                [t.cpu() for t in ts]
            xs = []
            for _ in range(max(repeats // 4, 10)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                [t.cpu() for t in ts]
                xs.append((time.perf_counter() - t0) * 1e3)
            return round(st.median(xs), 4)
        h = torch.randn(n, hidden, dtype=dt, device=dev)
        ti = torch.zeros(n, layers, top_k, dtype=torch.int16, device=dev)
        tw = torch.zeros(n, layers, top_k, dtype=torch.float32, device=dev)
        out[str(n)] = {"embed_ms": round(emb, 4), "hidden_d2h_ms": d2h(h), "stats_d2h_ms": d2h(ti, tw)}
    del table
    torch.cuda.empty_cache()
    return out


def selector_timing(feats: Sequence, *, lr_model: str, layers: int, n_experts: int, tasks: Sequence[str],
                    p2p_framework: str | None, repeats: int = 20) -> dict:
    """网关同一个 classify（tiered_gateway.make_lr_classify）跑引擎交出的真 p2p_feat。"""
    from vllm.p2p.identifier import use_p2p_framework

    use_p2p_framework(p2p_framework)
    from vllm.p2p.tiered_gateway import make_lr_classify

    classify, _ = make_lr_classify(lr_model=lr_model, l0=layers, n_experts=n_experts,
                                   tasks={t: 1.0 for t in tasks})
    xs = []
    for ft in feats:
        info = {"n_prompt": None, "body": None}
        classify(ft, info)                                   # 第一次（缓存、懒加载）不计
        for _ in range(repeats):
            t0 = time.perf_counter()
            classify(ft, info)
            xs.append((time.perf_counter() - t0) * 1e3)
    return {"model": lr_model, "feats": len(feats), "ms": _stats(xs)}


def _gpu_info() -> dict:
    try:
        q = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=name,memory.total,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        name, tot, used = [x.strip() for x in q.stdout.strip().split(",")[:3]]
        return {"gpu": name, "mem_total_mb": float(tot), "mem_used_mb": float(used)}
    except Exception as e:                                   # noqa: BLE001
        return {"gpu": "?", "error": f"nvidia-smi: {e}"}


def _wait_gpu_free(max_used_mb: float, timeout_s: float = 60.0) -> None:
    """上一个引擎关了以后等显存真的还回来，再起下一个（不然 gpu_memory_utilization 的检查过不去）。"""
    t0 = time.monotonic()
    time.sleep(2.0)
    while time.monotonic() - t0 < timeout_s:
        if _gpu_info().get("mem_used_mb", 0) <= max_used_mb:
            return
        time.sleep(1.0)


def _prepare_dummy_dir(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "generation_config.json"):
        if (src / name).is_file():
            (dst / name).write_bytes((src / name).read_bytes())
    return dst


async def _run_async(a) -> dict:
    cfg = json.loads((Path(a.model_dir) / "config.json").read_text(encoding="utf-8"))
    vocab, hidden = int(cfg["vocab_size"]), int(cfg["hidden_size"])
    from vllm.p2p import models
    E, K = models.n_experts_of(cfg), models.top_k_of(cfg)
    a.attention_backend = a.attention_backend or models.family_of(cfg).attention_backend
    conc, prefill = _ints(a.conc), _ints(a.prefill) if a.prefill else []
    backs = parse_back_ms(a.back_ms)
    mml = max([*prefill, a.prompt_len]) + max(a.tokens, 512) + 256   # 后台 decode 每条最多 512 个
    L = int(a.layers)
    td = Path(tempfile.mkdtemp(prefix="front_bench_"))
    model_dir = a.model_dir if a.real_weights else str(_prepare_dummy_dir(Path(a.model_dir), td / "model"))
    plan = [("main", L, not a.no_stats)]
    if not a.no_fixed:
        if not a.no_stats:
            plan.append(("nostats", L, False))
        if L > 1:
            plan.append(("L1", 1, False))
    res: dict[str, Any] = {"engines": {}}
    feats: list = []
    for name, layers, stats in plan:
        trace = td / f"{name}.jsonl"
        t0 = time.perf_counter()
        print(f"  → 引擎 {name}：前 {layers} 层，统计{'开' if stats else '关'} …", flush=True)
        eng = _front_engine(model_dir, layers=layers, stats=stats, a=a, trace=trace, mml=mml)
        print(f"    起好 {time.perf_counter() - t0:.0f}s", flush=True)
        main = name == "main"
        try:
            r = await bench_engine(
                eng, _trace_reader(eng, trace), conc=conc,
                # 对照引擎只量纯单步（后段 0 ms）与单条 prefill，省时间
                backs=backs if main else [(0.0, 0.0)],
                n_tokens=a.tokens, reqs_per_slot=a.reqs_per_slot if main else 1,
                prompt_len=a.prompt_len, prefill=prefill, repeats=a.repeats,
                prefill_bg=a.prefill_bg if main else 0, vocab=vocab, seed=a.seed, tag=name,
                timeout=a.timeout, log=lambda s: print(s, flush=True))
        finally:
            eng.client.shutdown()
        if main:
            feats = r.pop("feats", [])
        else:
            r.pop("feats", None)
        res["engines"][name] = {"layers": layers, "stats": stats, **r}
        _wait_gpu_free(a.max_used_mb)
    # 固定开销：纯单步（后段 0 ms）的服务 p50
    def svc(name):
        e = res["engines"].get(name)
        if not e:
            return None
        row = e["decode"].get("0") or {}
        return {c: v["clean"]["svc"]["p50"] for c, v in row.items() if v["clean"].get("n")}

    def pf(name):
        e = res["engines"].get(name)
        return {P: v["svc"].get("p50") for P, v in (e or {}).get("prefill", {}).items()} if e else None
    if svc("main") is not None:
        res["fixed"] = {"decode": fixed_overhead(svc("main"), svc("nostats"), svc("L1"), L),
                        "prefill": fixed_overhead(pf("main") or {}, pf("nostats"), pf("L1"), L)}
    if not a.no_micro:
        try:
            ns = sorted({*conc, *prefill})
            res["micro"] = micro_overheads(hidden=hidden, vocab=vocab, n_experts=E, top_k=K, layers=L,
                                           dtype=a.dtype, ns=ns)
        except Exception as e:                               # noqa: BLE001
            res["micro"] = {"error": f"{type(e).__name__}: {e}"}
    if a.lr_model:
        try:
            res["selector"] = selector_timing(feats, lr_model=a.lr_model, layers=L, n_experts=E,
                                              tasks=[t for t in a.lr_tasks.split(",") if t],
                                              p2p_framework=a.p2p_framework)
        except BaseException as e:                           # noqa: BLE001 — 没有 p2p-framework 也照样出结果
            res["selector"] = {"error": f"{type(e).__name__}: {e}"}
    res["n_feats"] = len(feats)
    return res


def cmd_run(a) -> int:
    info = _gpu_info()
    if info.get("mem_used_mb", 0) > a.max_used_mb and not a.force:
        print(f"✗ 卡上已经占着 {info['mem_used_mb']:.0f} MiB（> {a.max_used_mb}）—— 有别的引擎在跑，"
              "量出来的数不准。先停掉（launch.py … stop all），或 --force", file=sys.stderr)
        return 3
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    node = a.node or os.environ.get("P2P_NODE") or os.uname().nodename
    print(f"[front_bench] {node}：{info.get('gpu')}，前 {a.layers} 层，并发 {a.conc}，后段 {a.back_ms} ms，"
          f"prefill {a.prefill}，{'真权重' if a.real_weights else '随机权重'}，"
          f"{'eager' if a.eager else 'CUDA Graph'}", flush=True)
    res = asyncio.run(_run_async(a))
    try:
        import vllm
        ver = getattr(vllm, "__version__", "?")
    except Exception:                                        # noqa: BLE001
        ver = "?"
    res = {"node": node, "host": os.uname().nodename, **info, "vllm": ver, "dtype": a.dtype,
           "attention": a.attention_backend, "cuda_graph": not a.eager, "layers": int(a.layers),
           "conc": _ints(a.conc), "back_ms": a.back_ms, "tokens": a.tokens,
           "reqs_per_slot": a.reqs_per_slot, "prompt_len": a.prompt_len,
           "real_weights": bool(a.real_weights), **res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print_run(res)
    print(f"已写 {a.out}")
    return 0


# --------------------------------------------------------------------------- #
# 打印
# --------------------------------------------------------------------------- #
def _f(x, fmt="{:.2f}"):
    return "—" if x is None else fmt.format(x)




def _gd(d, *ks):
    for k in ks:
        if not isinstance(d, Mapping):
            return None
        d = d.get(k)
    return d


def _mp(s: Mapping | None, fmt="{:.2f}") -> str:
    """{mean, p95} → 「均值/p95」。"""
    if not s or not s.get("n"):
        return "—"
    return f"{fmt.format(s['mean'])}/{fmt.format(s['p95'])}"


def print_run(res: Mapping) -> None:
    eng = (res.get("engines") or {}).get("main") or {}
    print(f"\n== {res.get('node')}（{res.get('gpu')}）前 {res.get('layers')} 层 · "
          f"{'真权重' if res.get('real_weights') else '随机权重'} · "
          f"{'CUDA Graph' if res.get('cuda_graph') else 'eager'} · {res.get('dtype')} ==")
    print("1. 并发下的单步：每个 token 前段这一步（ms，均值/p95；不含被 prefill 挡住的 token）")
    print(f"  {'后段ms':>7} {'c':>2}  {'服务':>12} {'等待':>12} {'其中排队':>12} {'回传':>10} "
          f"{'时延':>12} {'批':>5} {'忙':>5} {'tok/s':>7} {'n':>5}")
    for b, row in (eng.get("decode") or {}).items():
        for c, v in row.items():
            cl = v.get("clean") or {}
            print(f"  {b:>7} {c:>2}  {_mp(cl.get('svc')):>12} {_mp(cl.get('wait')):>12} "
                  f"{_mp(cl.get('queue')):>12} {_mp(cl.get('out')):>10} {_mp(cl.get('lat')):>12} "
                  f"{_f(cl.get('batch_mean')):>5} {_f(_gd(v, 'engine', 'busy_frac'), '{:.0%}'):>5} "
                  f"{_f(v.get('front_tok_per_s'), '{:.0f}'):>7} {cl.get('n', 0):>5}")
    if eng.get("prefill"):
        print("2. prefill（单条、前段没有别的请求；ms）")
        print(f"  {'P':>6}  {'那一步 均值/p95':>16} {'端到端 均值':>10} {'步数':>4}")
        for P, v in eng["prefill"].items():
            print(f"  {P:>6}  {_mp(v.get('svc')):>16} {_f(_gd(v, 'lat', 'mean')):>10} "
                  f"{_f(_gd(v, 'steps', 'mean'), '{:.0f}'):>4}")
    pb = eng.get("prefill_bg")
    if pb:
        m, cl = _gd(pb, "decode", "mixed") or {}, _gd(pb, "decode", "clean") or {}
        print(f"  后台 {pb['bg']} 条 decode（后段 {pb['back']} ms）时插 prefill：")
        for P, v in pb["prefill"].items():
            print(f"    P={P:<5} 那一步（含同批 decode）{_mp(v.get('svc'))} ms")
        print(f"    被挡住的 decode token {m.get('n', 0)} 个：时延 {_mp(m.get('lat'))} ms，等待 "
              f"{_mp(m.get('wait'))}；没被挡的 {_mp(cl.get('lat'))} ms")
    fx = res.get("fixed") or {}
    if fx.get("decode"):
        print("3. 固定开销（纯单步 p50，后段 0 ms；层数相减 = 不开统计的 L₀ 层 − 1 层）")
        print(f"  {'c':>4}  {'单步':>7} {'= 层计算':>8} {'+ 固定':>7} {'+ 统计':>7}   {'每层':>6}")
        for c, v in fx["decode"].items():
            print(f"  {c:>4}  {_f(v.get('step_ms')):>7} {_f(v.get('layers_ms')):>8} {_f(v.get('fixed_ms')):>7} "
                  f"{_f(v.get('stats_ms')):>7}   {_f(v.get('per_layer_ms')):>6}")
        for P, v in (fx.get("prefill") or {}).items():
            print(f"  P={P:<5}{_f(v.get('step_ms')):>7} {_f(v.get('layers_ms')):>8} {_f(v.get('fixed_ms')):>7} "
                  f"{_f(v.get('stats_ms')):>7}   {_f(v.get('per_layer_ms')):>6}")
    mi = res.get("micro") or {}
    if mi and "error" not in mi:
        print("  单独量（CUDA event / 同步拷贝，ms）：")
        print(f"  {'token':>6} {'词嵌入':>8} {'hidden 拷回':>10} {'统计拷回':>8}")
        for n, v in mi.items():
            print(f"  {n:>6} {_f(v['embed_ms'], '{:.4f}'):>8} {_f(v['hidden_d2h_ms'], '{:.4f}'):>10} "
                  f"{_f(v['stats_d2h_ms'], '{:.4f}'):>8}")
    elif mi:
        print(f"  单独量失败：{mi['error']}")
    sel = res.get("selector")
    if sel:
        if "error" in sel:
            print(f"  选择器打分：没量（{sel['error']}）")
        else:
            print(f"  选择器打分（本机 CPU，{sel['feats']} 条真 p2p_feat）：{_mp(sel.get('ms'), '{:.3f}')} ms")


# --------------------------------------------------------------------------- #
# occ：真实运行的占用率（控制机，只用标准库）
# --------------------------------------------------------------------------- #
def load_steps(paths: Sequence[str | Path]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in paths:
        p = Path(p)
        rows = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
        out[p.stem] = rows
    return out


def _tpot(r: Mapping) -> float | None:
    n = int(r.get("n_generated") or 0)
    return float(r["decode_ms"]) / (n - 1) if n > 1 and r.get("decode_ms") is not None else None


def occupancy(result: Mapping, traces: Mapping[str, Sequence[Mapping]], *,
              t0: float | None = None, t1: float | None = None) -> dict:
    """continuous 的结果 + 各引擎的步级记录 → 每个引擎：忙的比例（其中 prefill）、每个 decode token
    经历的一步多久（服务）、网关记的每位置引擎内毫秒（p50 的均值）、排队 ≈ 引擎内 − 服务、占 TPOT 多少。
    窗口默认用结果里的 run.t0_wall / t1_wall（新版 bench 才有）；没有就用各引擎记录自己的首尾。"""
    run = result.get("run") or {}
    t0 = t0 if t0 is not None else run.get("t0_wall")
    t1 = t1 if t1 is not None else run.get("t1_wall")
    rows = list(result.get("requests") or [])
    tp_all = [x for x in map(_tpot, rows) if x is not None]
    sel = [_gd(r, "identify", "compute_ms") for r in rows]
    out: dict[str, Any] = {"window": [t0, t1] if t0 is not None else None, "requests": len(rows),
                           "tpot": _stats(tp_all), "selector_ms": _stats([x for x in sel if x is not None]),
                           "engines": {}}
    fronts = [n for n in traces if n.startswith("front")]
    for name, steps in traces.items():
        role = "front" if name.startswith("front") else "back"
        sp = summarise_steps(steps, t0, t1)
        mine = [r for r in rows if _gd(r, "engine", role) == name]
        if not mine and role == "front" and len(fronts) == 1:
            mine = [r for r in rows if r.get("engine")]          # 只有一个前段：所有请求都过它
        key = f"{role}_engine_ms_p50"
        E = [float(r["engine"][key]) for r in mine if _gd(r, "engine", key) is not None]
        tp = [x for x in map(_tpot, mine) if x is not None]
        row: dict[str, Any] = {"role": role, **sp, "requests": len(mine), "engine_ms": _stats(E),
                               "tpot": _stats(tp)}
        svc = sp.get("svc_per_token_ms")
        if E and svc is not None:
            e = st.mean(E)
            row["queue_ms"] = round(e - svc, 3)
            if tp:
                tpm = st.median(tp)
                row["engine_share_tpot"] = round(e / tpm, 4)
                row["svc_share_tpot"] = round(svc / tpm, 4)
                row["queue_share_tpot"] = round((e - svc) / tpm, 4)
        out["engines"][name] = row
    return out


def print_occ(occ: Mapping) -> None:
    w = occ.get("window")
    print(f"\n== 真实运行的占用率（{occ['requests']} 条；窗口 "
          f"{'run.t0_wall–t1_wall' if w else '各引擎记录的首尾（结果里没有 run.t0_wall，含预热）'}）==")
    print(f"  TPOT（每条请求）{_mp(occ.get('tpot'), '{:.1f}')} ms；选择器打分（网关记的 compute_ms）"
          f"{_mp(occ.get('selector_ms'), '{:.3f}')} ms")
    print(f"  {'引擎':<20} {'忙':>5} {'其中pf':>6} {'批':>5} {'每token服务':>10} {'引擎内':>8} {'排队≈':>7} "
          f"{'TPOT':>7} {'排队/TPOT':>9} {'引擎内/TPOT':>11} {'条':>4}")
    for name, v in sorted(occ["engines"].items(), key=lambda kv: (kv[1]["role"] != "front", kv[0])):
        if not v.get("steps"):
            print(f"  {name:<20} （窗口里没有步）")
            continue
        print(f"  {name:<20} {_f(v.get('busy_frac'), '{:.0%}'):>5} {_f(v.get('prefill_busy_frac'), '{:.0%}'):>6} "
              f"{_f(v.get('batch_mean')):>5} {_f(v.get('svc_per_token_ms')):>10} "
              f"{_f(_gd(v, 'engine_ms', 'mean')):>8} {_f(v.get('queue_ms')):>7} {_f(_gd(v, 'tpot', 'p50'), '{:.1f}'):>7} "
              f"{_f(v.get('queue_share_tpot'), '{:.1%}'):>9} {_f(v.get('engine_share_tpot'), '{:.1%}'):>11} "
              f"{v.get('requests', 0):>4}")
    print("  忙 = 步用时之和 / 窗口；其中pf = 带 prefill 的步占的那部分；每token服务 = 每个 decode token 所在那一步的平均用时；")
    print("  引擎内 = 网关记的每位置「代理收到 → 交出」p50（每条请求一个，取均值）；排队 ≈ 引擎内 − 每token服务（含代理↔引擎转发）")


def cmd_occ(a) -> int:
    result = json.loads(Path(a.result).read_text(encoding="utf-8"))
    traces = load_steps(a.steps)
    occ = occupancy(result, traces, t0=a.t0, t1=a.t1)
    print_occ(occ)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(occ, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已写 {a.out}")
    return 0


# --------------------------------------------------------------------------- #
# cluster：每种卡一台，ssh 过去跑 run
# --------------------------------------------------------------------------- #
RUN_OPTS = ("model_dir", "layers", "conc", "back_ms", "tokens", "reqs_per_slot", "prompt_len", "prefill",
            "repeats", "prefill_bg", "gpu_mem", "max_num_seqs", "dtype", "attention_backend", "seed",
            "timeout", "lr_model", "lr_tasks", "p2p_framework", "max_used_mb")
RUN_FLAGS = ("real_weights", "eager", "no_stats", "no_fixed", "no_micro", "force")


def run_argv(a, node: str, out: str) -> list[str]:
    args = ["run", "--node", node, "--out", out]
    for k in RUN_OPTS:
        v = getattr(a, k, None)
        if v is not None and v != "":
            args += [f"--{k.replace('_', '-')}", str(v)]
    args += [f"--{k.replace('_', '-')}" for k in RUN_FLAGS if getattr(a, k, False)]
    return args


def remote_command(node: str, a) -> str:
    return (f"cd {shlex.quote(a.repo)} && {shlex.quote(a.python)} -m vllm.p2p.front_bench "
            + " ".join(shlex.quote(x) for x in run_argv(a, node, f"/tmp/front_bench_{node}.json")))


def _layer_bench():
    if __package__:
        from vllm.p2p import layer_bench
    else:                                                    # 按文件路径跑（控制机，不 import vllm）
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import layer_bench                                   # type: ignore[no-redef]
    return layer_bench


def cmd_cluster(a) -> int:
    lb = _layer_bench()
    hosts = lb.parse_hosts(a.hosts)
    nodes = lb.pick_nodes(hosts, a.nodes)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[front_bench cluster] {len(nodes)} 台：" +
          "，".join(f"{n}（{hosts[n][1] or '?'}）" for n in nodes), flush=True)
    ok: dict[str, bool] = {}
    lock = threading.Lock()

    def one(n):
        host = hosts[n][0]
        t0 = time.perf_counter()
        try:
            p = subprocess.run([*SSH, f"{a.user}@{host}", remote_command(n, a)], capture_output=True,
                               text=True, timeout=a.ssh_timeout)
        except subprocess.TimeoutExpired:
            with lock:
                print(f"✗ {n}：{a.ssh_timeout:.0f}s 没跑完", flush=True)
                ok[n] = False
            return
        log = out_dir / f"{n}.log"
        log.write_text(p.stdout + p.stderr, encoding="utf-8")
        got = subprocess.run([*SSH, f"{a.user}@{host}", f"cat /tmp/front_bench_{n}.json"],
                             capture_output=True, text=True, timeout=60)
        with lock:
            if p.returncode != 0 or got.returncode != 0:
                tail = "\n    ".join((p.stdout + p.stderr).strip().splitlines()[-10:])
                print(f"✗ {n}（{hosts[n][1]}）失败，退出码 {p.returncode}；日志 {log}\n    {tail}", flush=True)
                ok[n] = False
                return
            (out_dir / f"{n}.json").write_text(got.stdout, encoding="utf-8")
            print(f"✓ {n}（{json.loads(got.stdout).get('gpu')}）{time.perf_counter() - t0:.0f}s", flush=True)
            ok[n] = True

    ths = [threading.Thread(target=one, args=(n,)) for n in nodes]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    good = [n for n in nodes if ok.get(n)]
    results = [json.loads((out_dir / f"{n}.json").read_text(encoding="utf-8")) for n in good]
    for r in results:
        print_run(r)
    if len(results) > 1:
        print_compare(results)
    print(f"\n{len(good)}/{len(nodes)} 台成功，每台的结果在 {out_dir}/<节点>.json，日志 <节点>.log")
    return 0 if len(good) == len(nodes) else 2


def print_compare(results: Sequence[Mapping]) -> None:
    """几种卡并排：每档并发的服务 / 时延（均值），prefill，固定开销。"""
    print("\n== 卡型对比（ms，均值）==")
    names = [f"{r.get('node')}·{_layer_bench().gpu_key(r.get('gpu', ''))}" for r in results]
    w = max(14, *(len(n) for n in names)) + 1
    backs = list((_gd(results[0], "engines", "main", "decode") or {}).keys())
    for b in backs:
        print(f"  后段 {b} ms：每 token 服务 / 等待 / 时延")
        print(f"  {'c':>4} " + "".join(f"{n:>{w + 8}}" for n in names))
        cs = list((_gd(results[0], "engines", "main", "decode", b) or {}).keys())
        for c in cs:
            cells = []
            for r in results:
                cl = _gd(r, "engines", "main", "decode", b, c, "clean") or {}
                cells.append(f"{_f(_gd(cl, 'svc', 'mean'))}/{_f(_gd(cl, 'wait', 'mean'))}/"
                             f"{_f(_gd(cl, 'lat', 'mean'))}")
            print(f"  {c:>4} " + "".join(f"{x:>{w + 8}}" for x in cells))
    pfs = list((_gd(results[0], "engines", "main", "prefill") or {}).keys())
    if pfs:
        print("  prefill 那一步（单条）")
        for P in pfs:
            print(f"  {P:>4} " + "".join(f"{_f(_gd(r, 'engines', 'main', 'prefill', P, 'svc', 'mean')):>{w + 8}}"
                                          for r in results))
    if all(_gd(r, "fixed", "decode") for r in results):
        print("  纯单步 c=1 拆分：层计算 + 固定 + 统计")
        cells = []
        for r in results:
            v = _gd(r, "fixed", "decode", "1") or {}
            cells.append(f"{_f(v.get('layers_ms'))}+{_f(v.get('fixed_ms'))}+{_f(v.get('stats_ms'))}")
        print("     " + "".join(f"{x:>{w + 8}}" for x in cells))


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def add_run_args(ap, *, cluster: bool = False) -> None:
    ap.add_argument("--model-dir", default="/home/ubuntu/p2p-framework/weights-q3-30b",
                    help="取 config.json 的目录（--real-weights 时也读权重）")
    ap.add_argument("--layers", default="2", help="前段几层（L₀）")
    ap.add_argument("--conc", default=DEFAULT_CONC, help="同时在飞几条（逗号分隔）")
    ap.add_argument("--back-ms", default=DEFAULT_BACK,
                    help="「后段」花多久再送下一个 token：0 = 马上（纯单步）；20:40 = 均匀随机（排队）；逗号分隔几种")
    ap.add_argument("--tokens", type=int, default=64, help="每条请求 decode 几个 token")
    ap.add_argument("--reqs-per-slot", type=int, default=2, help="每个并发槽先后跑几条请求")
    ap.add_argument("--prompt-len", type=int, default=128, help="decode 那几条请求的 prompt 长度")
    ap.add_argument("--prefill", default=DEFAULT_PREFILL, help="prefill 长度（逗号分隔；空 = 不量）")
    ap.add_argument("--repeats", type=int, default=5, help="每个 prefill 长度量几次（另有 1 次预热）")
    ap.add_argument("--prefill-bg", type=int, default=3, help="插 prefill 时后台几条 decode（0 = 不量）")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--dtype", default="half")
    ap.add_argument("--attention-backend", default=None,
                    help="默认按模型族（Qwen 系 TRITON_ATTN、GLM-4.7-Flash TRITON_MLA）；auto = vLLM 自选")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120.0, help="等一位 hidden 最多几秒")
    ap.add_argument("--real-weights", action="store_true", help="读 checkpoint（默认随机权重，起得快）")
    ap.add_argument("--eager", action="store_true", help="不用 CUDA Graph（部署时是用的）")
    ap.add_argument("--no-stats", action="store_true", help="主引擎不开 p2p_stats（部署时前段是开的）")
    ap.add_argument("--no-fixed", action="store_true", help="不起对照引擎（不开统计的 L₀ 层、1 层）")
    ap.add_argument("--no-micro", action="store_true", help="不单独量词嵌入、拷贝")
    ap.add_argument("--lr-model", default=None,
                    help="选择器文件：拿引擎交出的 p2p_feat 量打分（要能找到 p2p-framework）")
    ap.add_argument("--lr-tasks", default="gsm8k,mbpp,no_robots")
    ap.add_argument("--p2p-framework", default=None)
    ap.add_argument("--max-used-mb", type=float, default=1024, help="卡上已占超过这么多就拒绝量")
    ap.add_argument("--force", action="store_true", help="卡上有别的进程也照量")
    if cluster:
        ap.add_argument("--hosts", required=True, help="task/hosts.txt（行尾注释写卡型）")
        ap.add_argument("--nodes", default="one-per-gpu",
                        help="one-per-gpu（默认，每种卡型第一台）/ all / 逗号分隔的节点")
        ap.add_argument("--out-dir", default="results/front_bench")
        ap.add_argument("--repo", default="/home/ubuntu/vllm-p2p")
        ap.add_argument("--python", default="/home/ubuntu/venvs/vllm-p2p/bin/python")
        ap.add_argument("--user", default="ubuntu")
        ap.add_argument("--ssh-timeout", type=float, default=3600)
    else:
        ap.add_argument("--node", default=None, help="节点名（默认主机名）")
        ap.add_argument("--out", required=True)


def add_occ_args(ap) -> None:
    ap.add_argument("--result", required=True, help="bench.py continuous 的结果 JSON")
    ap.add_argument("--steps", nargs="+", required=True,
                    help="steptrace.py collect 收回来的 <引擎名>.jsonl（前段、后段都可以给）")
    ap.add_argument("--t0", type=float, default=None, help="窗口开始（time.time()；默认 run.t0_wall）")
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--out", default=None, help="另存 JSON")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run", help="本机 GPU：并发单步、prefill、固定开销"))
    add_run_args(sub.add_parser("cluster", help="控制机：ssh 到每种卡一台跑 run"), cluster=True)
    add_occ_args(sub.add_parser("occ", help="控制机：真实运行的占用率"))
    a = ap.parse_args(argv)
    return {"run": cmd_run, "cluster": cmd_cluster, "occ": cmd_occ}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
