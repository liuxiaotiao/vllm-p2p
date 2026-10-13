# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B-1 分段引擎的协调器 + B0 原型（1 个前段 + 2 个同 task 后段，单卡、微型 checkpoint）。

    # vllm-0.14.0 目录下、装好本源码的环境里（与 parity 同一个）
    python -m vllm.p2p.tiered                         # fp16 + TRITON_ATTN，8 并发
    python -m vllm.p2p.tiered --dtype float32         # 近乎逐位的对拍
    python -m vllm.p2p.tiered --requests 32 --concurrency 16 --max-tokens 128 --out b0.json

一条请求的一生（协议见 `vllm/p2p/stage.py`）：

    1. 进前段引擎（token prompt）。前段 prefill，交出 prompt 每个位置的残差流 hidden。
    2. prompt 的 hidden 收齐 → **绑定**一个后段（同 task 的后段里挑负载最轻的；B1 换成 LR）
       → 以 prompt_embeds 进那个后段。后段 prefill，采样出第 1 个 token。
    3. 循环：后段的 token ──p2p_append──▶ 前段算这一位置的 hidden ──p2p_feed──▶ 后段采下一个。
    4. 后段结束（EOS / 长度）→ 前段那条 abort。

绑定后不再换（换绑不在本阶段范围）；同一个前段、同一 task 的请求可以落到不同的后段。

B0 验什么
---------
* **正确性**：两个后段默认装**不同**的专家子集；每种驻留集各起一个「整模型」引擎（同一份驻留集的
  Qwen3MoePartial、不分段），每条分段结果对着**它被绑到的那个后段**的整模型逐条比 token 与 logprob。判据同 parity：公共前缀上 max |Δlogprob| ≤ tol，分叉处必须是
  近似平票（整模型自己的 top-k 里两个选择的 logprob 差 ≤ 2·tol）。
* **开销**：同并发下 TTFT / TPOT 与整模型对比；每个 decode 位置拆成「前段腿」（后段出 token →
  前段 hidden 回到协调器）与「后段腿」（hidden 送出 → 下一个 token 回来）。
* **按请求绑定**：两个后段各分到多少条。

数据路径（原型，后面再优化）：前段 worker → 前段 EngineCore 输出 → 协调器 → 后段 EngineCore
utility → SchedulerOutput → 后段 worker。每个 decode 位置一来一回两次 ZMQ。
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import random
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from vllm.p2p.stage import RowStream, pack_rows

__all__ = ["Engine", "TieredCoordinator", "MonoRunner", "compare_runs", "main"]


# --------------------------------------------------------------------------- #
# 一个引擎（EngineCore 的 AsyncMPClient）的薄封装
# --------------------------------------------------------------------------- #
@dataclass
class Engine:
    name: str
    role: str | None                 # "front" / "back" / None（整模型）
    client: Any                      # vllm.v1.engine.core_client.AsyncMPClient（测试里是假的）
    task: str | None = None          # 后段服务的 task（绑定时只在同 task 的后段里挑）
    inflight: int = 0
    n_bound: int = 0
    errors: list = field(default_factory=list)
    on_error: Callable[[str], None] | None = None

    async def add(self, req) -> None:
        await self.client.add_request_async(req)

    async def abort(self, ids: list[str]) -> None:
        if ids:
            await self.client.abort_requests_async(ids)

    async def outputs(self):
        return await self.client.get_output_async()

    async def fire(self, method: str, *args) -> None:
        """发一个 utility 调用、**不等结果**（等的话要等对面这一步算完，整条流水线就串行了）。
        失败记进 errors，由协调器统一抛。"""
        fire = getattr(self.client, "p2p_fire", None)
        if fire is not None:                         # 测试用的假引擎
            fut = await fire(method, *args)
        else:
            fut = await _fire_utility(self.client, method, *args)

        def done(f):
            if not f.cancelled() and f.exception() is not None:
                msg = f"{self.name}.{method}: {f.exception()}"
                self.errors.append(msg)
                if self.on_error is not None:
                    self.on_error(msg)
        fut.add_done_callback(done)


async def _fire_utility(client, method: str, *args):
    """`AsyncMPClient._call_utility_async` 去掉最后的 `await future`。"""
    from vllm.v1.engine import EngineCoreRequestType

    call_id = uuid.uuid1().int >> 64
    fut = asyncio.get_running_loop().create_future()
    client.utility_results[call_id] = fut
    msg = (EngineCoreRequestType.UTILITY.value,
           *client.encoder.encode((client.client_index, call_id, method, args)))
    await client._send_input_message(msg, client.core_engine, args)
    client._ensure_output_queue_task()
    return fut


def _core_request(rid: str, *, prompt: list[int] | None, embeds: torch.Tensor | None,
                  max_tokens: int, eos: int | None, logprobs: int, ignore_eos: bool,
                  stop_ids: Sequence[int], arrival: float):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, detokenize=False,
                        logprobs=logprobs or None, ignore_eos=ignore_eos,
                        stop_token_ids=list(stop_ids) or None)
    return EngineCoreRequest(
        request_id=rid, prompt_token_ids=prompt, mm_features=None, sampling_params=sp,
        pooling_params=None, eos_token_id=eos, arrival_time=arrival, lora_request=None,
        cache_salt=None, data_parallel_rank=None, prompt_embeds=embeds)


def _lp_rows(lp) -> list[dict[int, float]]:
    """EngineCoreOutput.new_logprobs（LogprobsLists）→ 每个新 token 一个 {token: logprob}。"""
    if lp is None:
        return []
    ids, vals = lp.logprob_token_ids, lp.logprobs
    return [{int(t): float(v) for t, v in zip(ids[j], vals[j])} for j in range(len(ids))]


# --------------------------------------------------------------------------- #
# 协调器
# --------------------------------------------------------------------------- #
class _Req:
    __slots__ = ("rid", "prompt", "n_prompt", "max_tokens", "task", "back", "stream", "tokens",
                 "lps", "t_arrive", "t_front", "t_bound", "t_first", "t_last", "t_tok", "t_hid",
                 "finish", "done")

    def __init__(self, rid, prompt, max_tokens, task, loop):
        self.rid, self.prompt, self.max_tokens, self.task = rid, prompt, max_tokens, task
        self.n_prompt = len(prompt)
        self.back: Engine | None = None
        self.stream = RowStream()
        self.tokens: list[int] = []
        self.lps: list[dict] = []
        self.t_arrive = time.perf_counter()
        self.t_front = self.t_bound = self.t_first = self.t_last = None
        self.t_tok: list[float] = []         # 第 k 个 token 回到协调器的时刻
        self.t_hid: list[float] = []         # 第 k 个 token 的前段 hidden 回到协调器的时刻
        self.finish = None
        self.done: asyncio.Future = loop.create_future()

    def result(self) -> dict:
        n = len(self.tokens)
        ttft = (self.t_first - self.t_arrive) * 1e3 if self.t_first else None
        tpot = ((self.t_last - self.t_first) * 1e3 / (n - 1)) if n > 1 else None
        legs_f = [(h - t) * 1e3 for t, h in zip(self.t_tok, self.t_hid)]
        legs_b = [(t - h) * 1e3 for h, t in zip(self.t_hid, self.t_tok[1:])]
        return {
            "rid": self.rid, "back": self.back.name if self.back else None, "task": self.task,
            "n_prompt": self.n_prompt, "tokens": self.tokens, "logprobs": self.lps,
            "finish": self.finish, "ttft_ms": ttft, "tpot_ms": tpot,
            "front_prefill_ms": (self.t_front - self.t_arrive) * 1e3 if self.t_front else None,
            "front_leg_ms": legs_f, "back_leg_ms": legs_b,
        }


class TieredCoordinator:
    """一个前段 + 若干后段。`generate()` 可以并发调用。

    bind：`least`（同 task 后段里在飞请求最少的，平手轮流）/ `rr`（轮流）/ 一个函数
    `f(req_info: dict, candidates: list[Engine]) -> Engine`（B1 的 LR 绑定从这里接）。"""

    def __init__(self, front: Engine, backs: Sequence[Engine], *, bind: str | Callable = "least",
                 eos: int | None = None, stop_ids: Sequence[int] = (), ignore_eos: bool = False,
                 logprobs: int = 0):
        if not backs:
            raise ValueError("至少要一个后段")
        self.front, self.backs = front, list(backs)
        self.bind, self.eos, self.stop_ids = bind, eos, tuple(stop_ids)
        self.ignore_eos, self.logprobs = ignore_eos, logprobs
        self.reqs: dict[str, _Req] = {}
        self._rr: dict[Any, itertools.count] = defaultdict(itertools.count)
        self._tasks: list[asyncio.Task] = []
        self._ids = itertools.count()
        for e in (front, *self.backs):
            e.on_error = lambda msg: self._fail_all(RuntimeError(f"分段引擎 utility 调用失败：{msg}"))

    # ---- 生命周期 ----
    def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(self._front_loop(), name="p2p-front")]
        self._tasks += [loop.create_task(self._back_loop(b), name=f"p2p-{b.name}")
                        for b in self.backs]

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    def _fail_all(self, exc: BaseException) -> None:
        for r in list(self.reqs.values()):
            if not r.done.done():
                r.done.set_exception(exc)

    def _check_errors(self) -> None:
        errs = [e for x in (self.front, *self.backs) for e in x.errors]
        if errs:
            raise RuntimeError("分段引擎 utility 调用失败：\n  " + "\n  ".join(errs[:5]))

    # ---- 请求 ----
    async def generate(self, prompt: list[int], max_tokens: int, *, task: str | None = None,
                       rid: str | None = None, timeout: float | None = None) -> dict:
        self._check_errors()
        rid = rid or f"t{next(self._ids)}"
        r = self.reqs[rid] = _Req(rid, list(prompt), max_tokens, task, asyncio.get_running_loop())
        self.front.inflight += 1
        # 前段这条永远不会自己结束（不采样、不查停止条件）：max_tokens 只是 Request 的必填项
        await self.front.add(_core_request(
            rid, prompt=list(prompt), embeds=None, max_tokens=max_tokens + 1, eos=None,
            logprobs=0, ignore_eos=True, stop_ids=(), arrival=time.time()))
        try:
            return await asyncio.wait_for(asyncio.shield(r.done), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            why = self.describe(rid)
            await self.front.abort([rid])
            if r.back is not None:
                await r.back.abort([rid])
            if isinstance(e, asyncio.CancelledError):
                raise
            raise TimeoutError(f"{rid} 超时：{why}") from None
        finally:
            self.reqs.pop(rid, None)

    def describe(self, rid: str) -> str:
        r = self.reqs.get(rid)
        if r is None:
            return "（已结束）"
        return (f"back={r.back.name if r.back else None} hidden 收到 {r.stream.next} 位"
                f"（prompt {r.n_prompt}） token {len(r.tokens)} 个 hidden 回 {len(r.t_hid)} 次")

    def _candidates(self, task: str | None) -> list[Engine]:
        same = [b for b in self.backs if task is not None and b.task == task]
        return same or self.backs

    def _pick(self, r: _Req) -> Engine:
        cands = self._candidates(r.task)
        if callable(self.bind):
            return self.bind({"rid": r.rid, "task": r.task, "n_prompt": r.n_prompt}, cands)
        k = next(self._rr[r.task])
        if self.bind == "rr":
            return cands[k % len(cands)]
        low = min(b.inflight for b in cands)
        least = [b for b in cands if b.inflight == low]
        return least[k % len(least)]

    async def _bind(self, r: _Req) -> None:
        b = r.back = self._pick(r)
        b.inflight += 1
        b.n_bound += 1
        r.t_bound = time.perf_counter()
        await b.add(_core_request(
            r.rid, prompt=None, embeds=r.stream.take(), max_tokens=r.max_tokens, eos=self.eos,
            logprobs=self.logprobs, ignore_eos=self.ignore_eos, stop_ids=self.stop_ids,
            arrival=time.time()))

    # ---- 两个方向的搬运 ----
    async def _front_loop(self) -> None:
        try:
            while True:
                outs = await self.front.outputs()
                feeds: dict[str, tuple[Engine, list]] = {}
                now = time.perf_counter()
                for o in outs.outputs:
                    r = self.reqs.get(o.request_id)
                    if r is None or o.p2p_hidden is None:
                        continue
                    got = r.stream.add(o.p2p_pos, o.p2p_hidden)
                    if got is None:
                        continue          # 前段被抢占后重算，送来的是旧位置
                    start, rows = got
                    if r.back is None:
                        r.stream.keep(rows)
                        if r.stream.next >= r.n_prompt:
                            r.t_front = now
                            await self._bind(r)
                    else:
                        r.t_hid.append(now)
                        b = r.back
                        feeds.setdefault(b.name, (b, []))[1].append(pack_rows(r.rid, start, rows))
                for b, items in feeds.values():
                    await b.fire("p2p_feed", items)
        except asyncio.CancelledError:
            raise
        except BaseException as e:      # noqa: BLE001 — 引擎死了：让所有在等的请求都知道
            self._fail_all(e)
            raise

    async def _back_loop(self, b: Engine) -> None:
        try:
            while True:
                outs = await b.outputs()
                appends, aborts = [], []
                now = time.perf_counter()
                for o in outs.outputs:
                    r = self.reqs.get(o.request_id)
                    if r is None:
                        continue
                    if o.new_token_ids:
                        r.tokens.extend(int(t) for t in o.new_token_ids)
                        r.lps.extend(_lp_rows(o.new_logprobs))
                        r.t_first = r.t_first or now
                        r.t_last = now
                        r.t_tok.extend([now] * len(o.new_token_ids))
                    if o.finish_reason is not None:
                        r.finish = str(o.finish_reason)
                        b.inflight -= 1
                        self.front.inflight -= 1
                        aborts.append(r.rid)
                        if not r.done.done():
                            r.done.set_result(r.result())
                    elif o.new_token_ids:
                        appends.append([r.rid, [int(t) for t in o.new_token_ids]])
                if appends:
                    await self.front.fire("p2p_append", appends)
                if aborts:
                    await self.front.abort(aborts)
        except asyncio.CancelledError:
            raise
        except BaseException as e:      # noqa: BLE001
            self._fail_all(e)
            raise


class MonoRunner:
    """整模型引擎上的同一套请求（对拍与开销的参照）。"""

    def __init__(self, eng: Engine, *, eos=None, stop_ids=(), ignore_eos=False, logprobs=0):
        self.eng, self.eos, self.stop_ids = eng, eos, tuple(stop_ids)
        self.ignore_eos, self.logprobs = ignore_eos, logprobs
        self.reqs: dict[str, _Req] = {}
        self._task: asyncio.Task | None = None
        self._ids = itertools.count()

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def generate(self, prompt, max_tokens, *, task=None, rid=None, timeout=None) -> dict:
        rid = rid or f"m{next(self._ids)}"
        r = self.reqs[rid] = _Req(rid, list(prompt), max_tokens, task, asyncio.get_running_loop())
        await self.eng.add(_core_request(
            rid, prompt=list(prompt), embeds=None, max_tokens=max_tokens, eos=self.eos,
            logprobs=self.logprobs, ignore_eos=self.ignore_eos, stop_ids=self.stop_ids,
            arrival=time.time()))
        try:
            return await asyncio.wait_for(asyncio.shield(r.done), timeout)
        finally:
            self.reqs.pop(rid, None)

    async def _loop(self) -> None:
        try:
            while True:
                outs = await self.eng.outputs()
                now = time.perf_counter()
                for o in outs.outputs:
                    r = self.reqs.get(o.request_id)
                    if r is None:
                        continue
                    if o.new_token_ids:
                        r.tokens.extend(int(t) for t in o.new_token_ids)
                        r.lps.extend(_lp_rows(o.new_logprobs))
                        r.t_first = r.t_first or now
                        r.t_last = now
                    if o.finish_reason is not None:
                        r.finish = str(o.finish_reason)
                        if not r.done.done():
                            r.done.set_result(r.result())
        except asyncio.CancelledError:
            raise
        except BaseException as e:      # noqa: BLE001
            for r in self.reqs.values():
                if not r.done.done():
                    r.done.set_exception(e)
            raise


# --------------------------------------------------------------------------- #
# 对拍
# --------------------------------------------------------------------------- #
def compare_runs(ref: Sequence[dict], got: Sequence[dict], *, tol: float) -> dict:
    """逐条比较（同序）。ref 是整模型，got 是分段。"""
    rows = []
    for a, b in zip(ref, got):
        ta, tb = a["tokens"], b["tokens"]
        cp = 0
        while cp < min(len(ta), len(tb)) and ta[cp] == tb[cp]:
            cp += 1
        dl = [abs(a["logprobs"][i][ta[i]] - b["logprobs"][i][tb[i]])
              for i in range(cp) if i < len(a["logprobs"]) and i < len(b["logprobs"])
              and ta[i] in a["logprobs"][i] and tb[i] in b["logprobs"][i]]
        tie = None
        if cp < min(len(ta), len(tb)) and cp < len(a["logprobs"]):
            top = a["logprobs"][cp]
            tie = (top[ta[cp]] - top[tb[cp]]) if tb[cp] in top else float("inf")
        rows.append({"rid": b["rid"], "back": b.get("back"), "same": ta == tb,
                     "common_prefix": cp, "len": [len(ta), len(tb)],
                     "max_lp_diff": max(dl) if dl else 0.0, "tie_gap": tie})
    n = len(rows)
    ok_lp = all(r["max_lp_diff"] <= tol for r in rows)
    ok_tie = all(r["tie_gap"] is None or r["tie_gap"] <= 2 * tol for r in rows)
    return {"n": n, "same": sum(r["same"] for r in rows),
            "max_lp_diff": max((r["max_lp_diff"] for r in rows), default=0.0),
            "diverged": [r for r in rows if not r["same"]], "ok": ok_lp and ok_tie,
            "ok_lp": ok_lp, "ok_tie": ok_tie, "rows": rows}


def _p(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2)


def summarise(res: Sequence[dict], wall_s: float) -> dict:
    toks = sum(len(r["tokens"]) for r in res)
    fl = [x for r in res for x in r.get("front_leg_ms", ())]
    bl = [x for r in res for x in r.get("back_leg_ms", ())]
    return {
        "n": len(res), "wall_s": round(wall_s, 3), "tok_per_s": round(toks / wall_s, 1),
        "ttft_ms_p50": _p([r["ttft_ms"] for r in res], .5),
        "ttft_ms_p90": _p([r["ttft_ms"] for r in res], .9),
        "tpot_ms_p50": _p([r["tpot_ms"] for r in res], .5),
        "tpot_ms_p90": _p([r["tpot_ms"] for r in res], .9),
        "front_prefill_ms_p50": _p([r.get("front_prefill_ms") for r in res], .5),
        "front_leg_ms_p50": _p(fl, .5), "back_leg_ms_p50": _p(bl, .5),
        "by_back": _count(r.get("back") for r in res),
    }


def _count(xs) -> dict:
    out: dict = defaultdict(int)
    for x in xs:
        out[str(x)] += 1
    return dict(sorted(out.items()))


async def closed_loop(gen, prompts: Sequence[list[int]], *, max_tokens: int, concurrency: int,
                      tasks: Sequence[str | None] | None = None, timeout: float = 300.0,
                      tag: str = "r"):
    """固定并发：同时最多 concurrency 条在飞；结果按 prompts 的顺序返回。
    请求号 `<tag><i>`：每一轮用不同的 tag（同一个号在引擎里结束后再复用虽然合法，但前一轮的
    迟到输出会被认到新请求上）。"""
    sem = asyncio.Semaphore(concurrency)
    out: list[dict | None] = [None] * len(prompts)

    async def one(i):
        async with sem:
            out[i] = await gen(prompts[i], max_tokens,
                               task=tasks[i] if tasks else None, rid=f"{tag}{i}",
                               timeout=timeout)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(len(prompts))))
    return out, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# B0 原型
# --------------------------------------------------------------------------- #
def tiny_config(n_layers: int, dtype: str) -> dict:
    from vllm.p2p.parity import PARITY_CFG

    torch_dtype = {"half": "float16", "float16": "float16", "bfloat16": "bfloat16",
                   "float32": "float32", "float": "float32"}[dtype]
    return dict(PARITY_CFG, num_hidden_layers=n_layers, torch_dtype=torch_dtype)


def back_residents(n_layers: int, split: int, n_experts: int, keep: int, seed: int) -> dict:
    """后段各层一份随机驻留集（前段全装 —— 与 p2p-framework 的部署一致）。"""
    rng = random.Random(seed)
    return {str(l): sorted(rng.sample(range(n_experts), keep)) for l in range(split, n_layers)}


def make_engine(name: str, model: str, *, role: str | None, layers: tuple[int, int] | None,
                layer_experts: dict, args, task: str | None = None) -> Engine:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.p2p.layout import ARCH
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.core_client import EngineCoreClient
    from vllm.v1.executor import Executor

    ov: dict = {"architectures": [ARCH], "p2p_layer_experts": layer_experts}
    if role:
        ov["p2p_stage"] = {"role": role, "layers": list(layers)}
    kw = dict(model=model, dtype=args.dtype, hf_overrides=ov, skip_tokenizer_init=True,
              enable_prompt_embeds=(role == "back"), async_scheduling=False,
              enable_prefix_caching=False, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              enforce_eager=args.eager, seed=0)
    if args.attention_backend:
        kw["attention_backend"] = args.attention_backend
    if args.kv_bytes:
        kw["kv_cache_memory_bytes"] = int(args.kv_bytes)
    cfg = EngineArgs(**kw).create_engine_config(UsageContext.ENGINE_CONTEXT)
    t0 = time.perf_counter()
    client = EngineCoreClient.make_async_mp_client(cfg, Executor.get_class(cfg), log_stats=False)
    print(f"  {name:<6} {role or 'mono':<5} 层 {list(layers) if layers else 'all'}  "
          f"起好了（{time.perf_counter() - t0:.1f}s）", flush=True)
    return Engine(name, role, client, task=task)


def _prompts(n: int, vocab: int, lo: int, hi: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(3, vocab) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


async def _run(args) -> int:
    wd = Path(args.workdir or tempfile.mkdtemp(prefix="p2p_tiered_"))
    model = args.model
    n_layers, n_experts, vocab, eos = args.n_layers, 16, 512, 2
    if model is None:
        from vllm.p2p.parity import _use_p2p_framework

        _use_p2p_framework(args.p2p_framework)
        from p2pmoe.sim.fake_checkpoint import write_fake_checkpoint

        ckpt = wd / f"ckpt_L{n_layers}_{args.dtype}"
        if not (ckpt / "config.json").exists():
            write_fake_checkpoint(ckpt, tiny_config(n_layers, args.dtype), seed=11)
        model = str(ckpt)
    else:
        cfg = json.loads((Path(model) / "config.json").read_text())
        n_layers, n_experts = int(cfg["num_hidden_layers"]), int(cfg["num_experts"])
        vocab, eos = int(cfg["vocab_size"]), cfg.get("eos_token_id")
        eos = eos[0] if isinstance(eos, list) else eos
    split = args.split
    if not 0 < split < n_layers:
        raise SystemExit(f"--split 要在 (0, {n_layers}) 里")
    # 每个后段一份驻留集（默认两条后段装不同的子集；前段全装 —— 与 p2p-framework 的部署一致）
    les = [({} if args.full_experts else
            back_residents(n_layers, split, n_experts, args.keep_experts,
                           seed=5 if args.same_experts else 5 + i))
           for i in range(args.n_back)]
    distinct = sorted({json.dumps(x, sort_keys=True) for x in les})
    mono_of = [distinct.index(json.dumps(x, sort_keys=True)) for x in les]   # 后段 i → 整模型 j
    print(f"checkpoint: {model}\n  {n_layers} 层，前段 [0, {split})、后段 [{split}, {n_layers})；"
          f"{args.n_back} 个后段，每层驻留 "
          f"{'全部' if args.full_experts else args.keep_experts}/{n_experts} 个专家"
          f"（{len(distinct)} 种驻留集）；dtype={args.dtype} "
          f"attention={args.attention_backend or 'auto'}{' eager' if args.eager else ''}")

    prompts = _prompts(args.requests, vocab, args.prompt_min, args.prompt_max, seed=3)
    tol = args.tol if args.tol is not None else (1e-3 if args.dtype in ("float32", "float")
                                                 else 0.02)
    common = dict(eos=eos, ignore_eos=not args.honor_eos, logprobs=5)
    report: dict = {"model": model, "split": split, "n_layers": n_layers, "dtype": args.dtype,
                    "requests": args.requests, "concurrency": args.concurrency,
                    "max_tokens": args.max_tokens, "layer_experts": les, "bind": args.bind}

    print("起引擎（逐个起：每个都要在其余几个已占的显存之外做 profile）…", flush=True)
    engines: list[Engine] = []
    try:
        for j, x in enumerate(distinct):
            engines.append(make_engine(f"mono{j}", model, role=None, layers=None,
                                       layer_experts=json.loads(x), args=args))
        monos = list(engines)
        front = make_engine("front", model, role="front", layers=(0, split), layer_experts={},
                            args=args)
        engines.append(front)
        backs = []
        for i in range(args.n_back):
            backs.append(make_engine(f"back{i}", model, role="back", layers=(split, n_layers),
                                     layer_experts=les[i], args=args, task="t0"))
            engines.append(backs[-1])

        mrs = [MonoRunner(m, **common) for m in monos]
        for m in mrs:
            m.start()
        co = TieredCoordinator(front, backs, bind=args.bind, **common)
        co.start()
        # 预热：各一条，排除首轮分配
        for m in mrs:
            await m.generate(prompts[0], 4, rid="warm-m", timeout=args.timeout)
        await co.generate(prompts[0], 4, rid="warm-t", timeout=args.timeout)
        for b in backs:
            b.n_bound = 0

        phases = sorted({1, args.concurrency})
        refs_c1: list = []
        for c in phases:
            print(f"\n并发 {c}：整模型 ×{len(mrs)} …", flush=True)
            refs, w_ref = [], None
            for j, m in enumerate(mrs):
                ref, w = await closed_loop(m.generate, prompts, max_tokens=args.max_tokens,
                                           concurrency=c, timeout=args.timeout, tag=f"c{c}-m{j}-")
                refs.append(ref)
                w_ref = w if w_ref is None else w_ref       # 时延参照取第一个整模型
            print(f"并发 {c}：分段（1 前段 + {len(backs)} 后段，bind={args.bind}）…", flush=True)
            got, w_got = await closed_loop(co.generate, prompts, max_tokens=args.max_tokens,
                                           concurrency=c, timeout=args.timeout, tag=f"c{c}-t")
            # 每条分段结果对「它被绑到的那个后段」的整模型
            ref_for = [refs[mono_of[int(r["back"].removeprefix("back"))]][i]
                       for i, r in enumerate(got)]
            cmp = compare_runs(ref_for, got, tol=tol)
            # 噪声底：同一个整模型引擎，并发 1 与并发 c 之间的差（批的组成不同 → kernel 的分块 /
            # 累加顺序不同；fp16 下随机小模型的 logits 很平，这个差不小）。并发 > 1 时分段对
            # 整模型的差只要求与噪声底同量级（≤ 3× 底，且不低于 tol），同 parity 的判据。
            if c == 1:
                refs_c1 = refs
                floor = None
            else:
                floors = [compare_runs(a, b, tol=tol) for a, b in zip(refs_c1, refs)]
                floor = {"max_lp_diff": max(f["max_lp_diff"] for f in floors),
                         "same": sum(f["same"] for f in floors),
                         "n": sum(f["n"] for f in floors)}
                lim = max(tol, 3 * floor["max_lp_diff"])
                cmp["ok_lp"] = cmp["max_lp_diff"] <= lim
                cmp["ok"] = cmp["ok_lp"] and cmp["ok_tie"]
                cmp["lp_limit"] = lim
            differ = (sum(a["tokens"] != b["tokens"] for a, b in zip(refs[0], refs[1]))
                      if len(refs) > 1 else None)
            sm, sg = summarise(refs[0], w_ref), summarise(got, w_got)
            report[f"c{c}"] = {"mono": sm, "tiered": sg, "compare":
                               {k: v for k, v in cmp.items() if k != "rows"},
                               "rows": cmp["rows"], "mono_sets_differ": differ,
                               "noise_floor": floor}
            _print_phase(c, sm, sg, cmp, cmp.get("lp_limit", tol))
            if floor is not None:
                print(f"  噪声底（同一整模型，并发 1 vs 并发 {c}）：{floor['same']}/{floor['n']} 条逐 token "
                      f"相同，公共前缀上 max|Δlogprob| = {floor['max_lp_diff']:.2e} → 分段的上限取 "
                      f"max(tol, 3×底) = {cmp['lp_limit']:.2e}")
            if differ is not None:
                print(f"  两种驻留集的整模型输出不同的有 {differ}/{len(prompts)} 条"
                      f"（>0 说明「绑到哪个后段」确实决定了输出，上面的对拍是对着各自的那个比的）")
        ok = all(report[f"c{c}"]["compare"]["ok"] for c in phases)
        bound = {b.name: b.n_bound for b in backs}
        spread = len(backs) == 1 or all(v > 0 for v in bound.values())
        print(f"\n按请求绑定：{bound}  {'✓ 每个后段都分到了' if spread else '✗ 有后段没分到'}")
        print(f"B0 正确性：{'✓' if ok else '✗'}（tol={tol}）")
        report["bound"], report["ok"] = bound, ok and spread
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
            print(f"明细 → {args.out}")
        await co.close()
        for m in mrs:
            await m.close()
        return 0 if report["ok"] else 1
    finally:
        for e in engines:
            try:
                e.client.shutdown()
            except Exception:   # noqa: BLE001
                pass


def _print_phase(c, sm, sg, cmp, tol) -> None:
    def f(x):
        return "—" if x is None else f"{x:g}"
    print(f"  {'':<8}{'TTFT p50':>10}{'TTFT p90':>10}{'TPOT p50':>10}{'TPOT p90':>10}"
          f"{'tok/s':>9}{'墙钟 s':>9}")
    for name, s in (("整模型", sm), ("分段", sg)):
        print(f"  {name:<8}{f(s['ttft_ms_p50']):>10}{f(s['ttft_ms_p90']):>10}"
              f"{f(s['tpot_ms_p50']):>10}{f(s['tpot_ms_p90']):>10}{f(s['tok_per_s']):>9}"
              f"{f(s['wall_s']):>9}")
    print(f"  分段每个 decode 位置：前段腿 p50 {f(sg['front_leg_ms_p50'])} ms、"
          f"后段腿 p50 {f(sg['back_leg_ms_p50'])} ms；前段 prefill p50 "
          f"{f(sg['front_prefill_ms_p50'])} ms；分到 {sg['by_back']}")
    print(f"  对拍：{cmp['same']}/{cmp['n']} 条逐 token 相同；公共前缀上 max|Δlogprob| = "
          f"{cmp['max_lp_diff']:.2e}（≤ {tol} {'✓' if cmp['ok_lp'] else '✗'}）；"
          f"分叉处都是近似平票 {'✓' if cmp['ok_tie'] else '✗'}")
    for r in cmp["diverged"][:4]:
        gap = r["tie_gap"]
        print(f"    {r['rid']}（{r['back']}）第 {r['common_prefix']} 个 token 分叉，"
              f"整模型自己的两个候选差 {('%.4f' % gap) if gap is not None else '—'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=None, help="现成的 checkpoint；不给 = 生成微型随机 checkpoint")
    ap.add_argument("--p2p-framework", default=None, help="微型 checkpoint 生成器在那边")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--n-layers", type=int, default=8, help="微型 checkpoint 的层数")
    ap.add_argument("--split", type=int, default=4, help="前段层数 L0（前段 [0, L0)）")
    ap.add_argument("--n-back", type=int, default=2, help="同 task 后段的个数")
    ap.add_argument("--keep-experts", type=int, default=8, help="后段每层驻留几个专家（共 16）")
    ap.add_argument("--full-experts", action="store_true", help="后段也全装")
    ap.add_argument("--same-experts", action="store_true",
                    help="两个后段装同一份子集（默认各装不同的，各对各自的整模型）")
    ap.add_argument("--bind", default="least", choices=("least", "rr"))
    ap.add_argument("--dtype", default="half", choices=("half", "float16", "bfloat16", "float32"))
    ap.add_argument("--attention-backend", default="TRITON_ATTN")
    ap.add_argument("--gpu-mem", type=float, default=0.12,
                    help="每个引擎的显存比例（默认 5 个引擎：2 整模型 + 1 前段 + 2 后段）")
    ap.add_argument("--kv-bytes", type=float, default=None, help="直接给每个引擎的 KV 字节数")
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt-min", type=int, default=8)
    ap.add_argument("--prompt-max", type=int, default=96)
    ap.add_argument("--honor-eos", action="store_true", help="遇到 EOS 就停（默认生成满 max-tokens）")
    ap.add_argument("--tol", type=float, default=None, help="logprob 容差（默认 fp16 0.02、fp32 1e-3）")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    # 所有引擎都在这台机器上（每个 world_size=1）：分布式初始化一律走回环。
    # 不设的话 vLLM 自己挑地址（可能是公网 IPv6、可能是 hostname 解析出的 127.0.1.1），
    # 在有防火墙 / 多网卡的节点上会卡在 TCPStore 连接上（10-13 实测）。已设的不覆盖。
    #
    # 引擎进程用 spawn 起，不用 vLLM 默认的 fork：本进程先用 torch 在 CPU 上生成了微型
    # checkpoint，torch 的 OpenMP 线程池已经起来了；fork 出来的子进程里第一次走到并行的
    # CPU 算子（runner 初始化时 torch.zeros 一个 ~1 MB 的 pinned 缓冲）就永远等不到那些
    # 线程 —— 10-13 实测卡在 CpuGpuBuffer.__init__（py-spy）。
    for k, v in (("VLLM_HOST_IP", "127.0.0.1"), ("GLOO_SOCKET_IFNAME", "lo"),
                 ("NCCL_SOCKET_IFNAME", "lo"), ("VLLM_WORKER_MULTIPROC_METHOD", "spawn")):
        os.environ.setdefault(k, v)
    return asyncio.run(_run(a))


if __name__ == "__main__":
    sys.exit(main())
