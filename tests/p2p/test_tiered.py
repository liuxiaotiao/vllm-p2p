# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B-1 分段引擎：**真的 vLLM 调度器** + 真的协调器（vllm/p2p/tiered.py），只把 GPU 上的那一步
换成一个确定性的假 runner。不需要 GPU。

假模型的结构与真模型同构（因果、逐位置）：

    前段  h[p] = [tok[p], p, (tok[0] + … + tok[p]) mod 1000, 0, …]   —— 只依赖 ≤ p 的 token
    后段  next(p) = (Σ_{q ≤ p} 3·h[q][0] + h[q][2]) mod 509 + 3        —— 依赖 ≤ p 的全部输入

于是「整模型」的贪心输出可以直接算出来；分段跑出来的必须逐 token 一致。钉住的是：

* 后段只排已经收到 hidden 的位置（假 runner 读到没送来的位置就抛 IndexError）；
* 前段 prefill 完就停住、协调器 append 一个 token 才往前一位；append 的 token 随
  SchedulerOutput 到了 runner（假前段 runner 不知道 token 就算不了）；
* KV 很紧：两边都会被抢占、从头重算 —— 前段重送的旧位置要去重，后段重算要用存下的 hidden；
* 后段遇 EOS 结束 → 前段那条被 abort（前段调度器最后是空的）；
* 同一 task 的请求分到了两个后段。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
W = 256                      # PARITY_CFG 的 hidden_size
V = 509
EOS = 11


def feat(t: list[int], p: int) -> torch.Tensor:
    row = torch.zeros(W, dtype=torch.float16)
    row[0], row[1], row[2] = t[p], p % 1000, sum(t[: p + 1]) % 1000
    return row


def next_tok(rows: torch.Tensor) -> int:
    return int((rows[:, 0].double() * 3 + rows[:, 2].double()).sum().item()) % V + 3


def mono(prompt: list[int], max_tokens: int, eos: int | None) -> list[int]:
    t, out = list(prompt), []
    for _ in range(max_tokens):
        nt = next_tok(torch.stack([feat(t, p) for p in range(len(t))]))
        out.append(nt)
        t.append(nt)
        if nt == eos:
            break
    return out


# --------------------------------------------------------------------------- #
# 真调度器
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def configs(tmp_path_factory):
    os.environ["PYTHONPATH"] = f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.p2p.parity import PARITY_CFG

    d = tmp_path_factory.mktemp("ckpt")
    (d / "config.json").write_text(json.dumps(dict(PARITY_CFG, torch_dtype="float16")))
    out = {}
    for role, layers in (("front", [0, 2]), ("back", [2, 4]), ("mono", None)):
        ov = {"architectures": ["Qwen3MoePartialForCausalLM"]}
        if layers:
            ov["p2p_stage"] = {"role": role, "layers": layers}
        mc = ModelConfig(model=str(d), dtype="float16", seed=0, skip_tokenizer_init=True,
                         max_model_len=256, enable_prompt_embeds=(role == "back"),
                         hf_overrides=ov)
        sc = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=64, max_model_len=256,
                             enable_chunked_prefill=True, is_encoder_decoder=False,
                             async_scheduling=False)
        cc = CacheConfig(block_size=16, gpu_memory_utilization=0.9, cache_dtype="auto",
                         enable_prefix_caching=False)
        out[role] = VllmConfig(model_config=mc, scheduler_config=sc, cache_config=cc,
                               device_config=DeviceConfig("cpu"))
    return out


def make_sched(vc, num_blocks: int):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    kv = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["layer"], FullAttentionSpec(block_size=16, num_kv_heads=1,
                                                      head_size=1, dtype=torch.float32))])
    vc.cache_config.num_gpu_blocks = num_blocks
    return Scheduler(vllm_config=vc, kv_cache_config=kv, log_stats=False,
                     structured_output_manager=StructuredOutputManager(vc), block_size=16)


# --------------------------------------------------------------------------- #
# 假 runner（与 gpu_model_runner 里 P2P 那几处同一套记账）
# --------------------------------------------------------------------------- #
def _computed(so) -> dict[str, int]:
    c = {nr.req_id: nr.num_computed_tokens for nr in so.scheduled_new_reqs}
    c.update(zip(so.scheduled_cached_reqs.req_ids, so.scheduled_cached_reqs.num_computed_tokens))
    return c


class FrontRunner:
    def __init__(self):
        self.toks: dict[str, list[int]] = {}

    def step(self, so):
        from vllm.v1.outputs import ModelRunnerOutput

        for rid in so.finished_req_ids:
            self.toks.pop(rid, None)
        for nr in so.scheduled_new_reqs:
            self.toks[nr.req_id] = list(nr.prompt_token_ids)
        cr = so.scheduled_cached_reqs
        assert len(cr.new_token_ids) == len(cr.req_ids), "前段要随调度结果收到输入 token"
        for i, rid in enumerate(cr.req_ids):
            new, known = cr.new_token_ids[i], self.toks[rid]
            extra = cr.num_computed_tokens[i] + len(new) - len(known)
            if extra > 0:
                known.extend(new[-extra:])
        comp, ids, hid = _computed(so), list(so.num_scheduled_tokens), {}
        for rid in ids:
            n, s0, t = so.num_scheduled_tokens[rid], comp[rid], self.toks[rid]
            assert s0 + n <= len(t), f"前段排到了还不知道 token 的位置 {rid}"
            hid[rid] = (s0, torch.stack([feat(t, p) for p in range(s0, s0 + n)]))
        return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                 p2p_hidden=hid)


class BackRunner:
    def __init__(self):
        from vllm.p2p.stage import RowBuffer

        self.RowBuffer = RowBuffer
        self.inp: dict = {}
        self.n_tok: dict[str, int] = {}
        self.max_seen = 0

    def step(self, so):
        from vllm.v1.outputs import ModelRunnerOutput

        for rid in so.finished_req_ids:
            self.inp.pop(rid, None)
            self.n_tok.pop(rid, None)
        for nr in so.scheduled_new_reqs:
            assert nr.prompt_token_ids is None and nr.prompt_embeds is not None
            b = self.inp[nr.req_id] = self.RowBuffer(W, torch.float16)
            b.put(0, nr.prompt_embeds)
            self.n_tok[nr.req_id] = nr.prompt_embeds.shape[0]
        for rid, items in (so.p2p_feed or {}).items():
            for pos, rows in items:
                if rid in self.inp:
                    assert pos >= self.n_tok[rid] - 1, "hidden 送早了 / 送错位置"
                    self.inp[rid].put(pos, rows)
        comp, ids, sampled = _computed(so), list(so.num_scheduled_tokens), []
        for rid in ids:
            n, s0 = so.num_scheduled_tokens[rid], comp[rid]
            rows = self.inp[rid].get(0, s0 + n)     # 没送到的位置被排上 → IndexError
            if s0 + n >= self.n_tok[rid]:
                sampled.append([next_tok(rows)])
                self.n_tok[rid] += 1
            else:
                sampled.append([])                  # 分块 / 重算还没到末尾：不采样
        return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                 sampled_token_ids=sampled)


class FakeCore:
    """EngineCore + AsyncMPClient 的替身：同一个事件循环里一步一步跑真调度器。"""

    def __init__(self, sched, runner, seed: int):
        self.sched, self.runner = sched, runner
        self.q: asyncio.Queue = asyncio.Queue()
        self.wake = asyncio.Event()
        self.rng = random.Random(seed)
        self.task = None
        self.aborted: list[str] = []
        self.preempted = 0

    def _ensure(self):
        if self.task is None:
            self.task = asyncio.get_running_loop().create_task(self._loop())

    async def add_request_async(self, req):
        from vllm.v1.request import Request

        self.sched.add_request(Request.from_engine_core_request(req, None))
        self.wake.set()
        self._ensure()

    async def abort_requests_async(self, ids):
        from vllm.v1.request import RequestStatus

        self.aborted += ids
        self.sched.finish_requests(ids, RequestStatus.FINISHED_ABORTED)

    async def p2p_fire(self, method, *args):
        fut = asyncio.get_running_loop().create_future()
        try:
            fut.set_result(getattr(self.sched, method)(*args))
        except Exception as e:    # noqa: BLE001
            fut.set_exception(e)
        self.wake.set()
        return fut

    async def get_output_async(self):
        self._ensure()
        out = await self.q.get()
        if isinstance(out, BaseException):
            raise out
        return out

    async def _loop(self):
        try:
            while True:
                if not self.sched.has_requests():
                    self.wake.clear()
                    await self.wake.wait()
                so = self.sched.schedule()
                self.preempted += len(so.preempted_req_ids or ())
                outs = self.sched.update_from_output(so, self.runner.step(so))
                for eco in outs.values():
                    if eco.outputs:
                        self.q.put_nowait(eco)
                if so.total_num_scheduled_tokens == 0:
                    self.wake.clear()
                    try:
                        await asyncio.wait_for(self.wake.wait(), 0.05)
                    except asyncio.TimeoutError:
                        pass
                else:
                    await asyncio.sleep(self.rng.random() * 0.002)   # 打乱两边的步调
        except BaseException as e:   # noqa: BLE001
            self.q.put_nowait(e)
            raise

    def close(self):
        if self.task:
            self.task.cancel()

    shutdown = close


def _run(configs, *, n_req, conc, max_tokens, front_blocks, back_blocks, eos, n_back=2):
    from vllm.p2p.tiered import Engine, TieredCoordinator, closed_loop

    rng = random.Random(0)
    prompts = [[rng.randrange(3, V) for _ in range(rng.randint(3, 40))] for _ in range(n_req)]

    async def go():
        front = FakeCore(make_sched(configs["front"], front_blocks), FrontRunner(), 1)
        backs = [FakeCore(make_sched(configs["back"], back_blocks), BackRunner(), 2 + i)
                 for i in range(n_back)]
        co = TieredCoordinator(Engine("front", "front", front),
                               [Engine(f"back{i}", "back", b, task="t") for i, b in
                                enumerate(backs)],
                               eos=eos, ignore_eos=eos is None)
        co.start()
        try:
            res, _ = await closed_loop(co.generate, prompts, max_tokens=max_tokens,
                                       concurrency=conc, tasks=["t"] * n_req, timeout=30)
            await asyncio.sleep(0.05)                  # 让最后的 abort 落地
            return res, front, backs, co
        finally:
            await co.close()
            for c in (front, *backs):
                c.close()

    return prompts, asyncio.run(go())


def test_tiered_matches_monolithic_under_preemption(configs) -> None:
    prompts, (res, front, backs, co) = _run(
        configs, n_req=12, conc=8, max_tokens=24, front_blocks=10, back_blocks=6, eos=None)
    for p, r in zip(prompts, res):
        assert r["tokens"] == mono(p, 24, None), r["rid"]
        assert r["finish"] == "length"
        # 每个 decode 位置都走了一圈：token → 前段 hidden → 下一个 token
        assert len(r["front_leg_ms"]) == 23 and len(r["back_leg_ms"]) == 23
    assert front.preempted > 0 and sum(b.preempted for b in backs) > 0, \
        "KV 给得这么紧，两边都该被抢占过（否则这条测试没测到重算）"
    bound = [x.n_bound for x in co.backs]
    assert sum(bound) == 12 and min(bound) > 0               # 同 task 分到了两个后段
    assert {r["back"] for r in res} == {"back0", "back1"}
    assert not front.sched.requests and not any(b.sched.requests for b in backs)
    assert sorted(front.aborted) == sorted(r["rid"] for r in res)   # 前段那条都被收掉了


def test_tiered_eos_stops_back_and_aborts_front(configs) -> None:
    # 找一个会在中途出 EOS 的种子：让第一条 prompt 的整模型输出里含 EOS
    prompts, (res, front, backs, _) = _run(
        configs, n_req=8, conc=4, max_tokens=40, front_blocks=64, back_blocks=64, eos=EOS)
    stopped = 0
    for p, r in zip(prompts, res):
        want = mono(p, 40, EOS)
        assert r["tokens"] == want
        if want[-1] == EOS and len(want) < 40:
            stopped += 1
            assert r["finish"] == "stop"
    assert not front.sched.requests                          # 不管怎么结束，前段都收干净了
    if stopped == 0:
        pytest.skip("这组随机 prompt 里没有中途出 EOS 的（不影响上面的逐 token 对拍）")


def test_back_scheduler_gates_on_fed_positions(configs) -> None:
    """单看后段调度器：prompt 之后的位置，没 feed 就不排。"""
    from vllm.p2p.stage import pack_rows
    from vllm.p2p.tiered import _core_request
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    s = make_sched(configs["back"], 32)
    req = _core_request("a", prompt=None, embeds=torch.zeros(5, W, dtype=torch.float16),
                        max_tokens=8, eos=None, logprobs=0, ignore_eos=True, stop_ids=(),
                        arrival=0.0)
    s.add_request(Request.from_engine_core_request(req, None))
    so = s.schedule()
    assert so.num_scheduled_tokens == {"a": 5}                 # prompt 整段可排
    s.update_from_output(so, ModelRunnerOutput(req_ids=["a"], req_id_to_index={"a": 0},
                                               sampled_token_ids=[[42]]))
    assert s.schedule().num_scheduled_tokens == {}             # 有了 token 42，但 hidden 没来
    s.p2p_feed([pack_rows("a", 5, torch.ones(1, W, dtype=torch.float16))])
    so = s.schedule()
    assert so.num_scheduled_tokens == {"a": 1}
    (pos, rows), = so.p2p_feed["a"]
    assert pos == 5 and rows.shape == (1, W) and rows.dtype == torch.float16
    assert s.schedule().p2p_feed is None                       # 只随一次调度结果送下去
    with pytest.raises(RuntimeError, match="前段"):
        s.p2p_append([["a", [1]]])
    assert s.p2p_feed([pack_rows("gone", 9, torch.ones(1, W, dtype=torch.float16))]) == 0


class MonoRunner_:
    """整模型的假 runner：自己缓存采样出的 token（与真 runner 一样）。"""

    def __init__(self):
        self.toks: dict[str, list[int]] = {}

    def step(self, so):
        from vllm.v1.outputs import ModelRunnerOutput

        for rid in so.finished_req_ids:
            self.toks.pop(rid, None)
        for nr in so.scheduled_new_reqs:
            self.toks[nr.req_id] = list(nr.prompt_token_ids)
        comp, ids, sampled = _computed(so), list(so.num_scheduled_tokens), []
        for rid in ids:
            n, s0, t = so.num_scheduled_tokens[rid], comp[rid], self.toks[rid]
            if s0 + n >= len(t):
                nt = next_tok(torch.stack([feat(t, p) for p in range(len(t))]))
                sampled.append([nt])
                t.append(nt)
            else:
                sampled.append([])
        return ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                 sampled_token_ids=sampled)


def test_mono_runner_and_compare_runs(configs) -> None:
    """B0 脚本的另一半：整模型参照（MonoRunner）+ 对拍；不开 logprobs 时只比 token。"""
    from vllm.p2p.tiered import Engine, MonoRunner, closed_loop, compare_runs, summarise

    rng = random.Random(1)
    prompts = [[rng.randrange(3, V) for _ in range(rng.randint(3, 30))] for _ in range(6)]

    async def go():
        core = FakeCore(make_sched(configs["mono"], 64), MonoRunner_(), 9)
        mr = MonoRunner(Engine("mono", None, core), ignore_eos=True)
        mr.start()
        try:
            return await closed_loop(mr.generate, prompts, max_tokens=12, concurrency=3,
                                     timeout=30, tag="m")
        finally:
            await mr.close()
            core.close()

    res, wall = asyncio.run(go())
    assert [r["tokens"] for r in res] == [mono(p, 12, None) for p in prompts]
    assert [r["rid"] for r in res] == [f"m{i}" for i in range(6)]
    for r in res:                                  # compare_runs 要 logprobs：补一份确定的
        r["logprobs"] = [{t: -0.5} for t in r["tokens"]]
    c = compare_runs(res, res, tol=0.02)
    assert c["ok"] and c["same"] == 6
    sm = summarise(res, wall)
    assert sm["n"] == 6 and sm["tpot_ms_p50"] is not None and sm["front_leg_ms_p50"] is None


def test_b0_script_end_to_end_with_fake_engines(configs, tmp_path, monkeypatch, capsys) -> None:
    """`python -m vllm.p2p.tiered` 的编排（起引擎、预热、两轮并发、逐条对各自的整模型、报告），
    引擎换成假的。驻留集不同但假模型不看专家 → 两个整模型输出相同，对拍必须全过。"""
    from vllm.p2p import tiered

    made = []

    def fake_make_engine(name, model, *, role, layers, layer_experts, args, task=None):
        runner = {"front": FrontRunner, "back": BackRunner, None: MonoRunner_}[role]()
        made.append((name, role, layers, json.dumps(layer_experts, sort_keys=True)))
        return tiered.Engine(name, role, FakeCore(make_sched(configs[role or "mono"], 64),
                                                  runner, len(made)), task=task)

    monkeypatch.setattr(tiered, "make_engine", fake_make_engine)
    ckpt = Path(configs["mono"].model_config.model)
    out = tmp_path / "b0.json"
    rc = tiered.main(["--model", str(ckpt), "--split", "2", "--requests", "6",
                      "--concurrency", "3", "--max-tokens", "10", "--prompt-max", "30",
                      "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0, text
    rep = json.loads(out.read_text())
    assert rep["ok"] and set(rep) >= {"c1", "c3", "bound"}
    assert rep["c3"]["compare"]["same"] == 6 and rep["c3"]["mono_sets_differ"] == 0
    assert min(rep["bound"].values()) > 0
    names = [m[0] for m in made]
    assert names == ["mono0", "mono1", "front", "back0", "back1"]
    les = {m[0]: m[3] for m in made}
    assert les["mono0"] != les["mono1"] and {les["back0"], les["back1"]} == {les["mono0"],
                                                                            les["mono1"]}
    assert [m[2] for m in made if m[1] == "back"] == [(2, 4), (2, 4)]
    assert "B0 正确性：✓" in text
