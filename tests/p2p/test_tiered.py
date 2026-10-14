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


# --------------------------------------------------------------------------- #
# B1：同一套，但每个引擎隔着真的 ZMQ（remote.AgentBridge ↔ RemoteClient），
# 两个前段 + 两个后段，再在前面挂分段网关（tiered_gateway.make_app）走 HTTP/SSE
# --------------------------------------------------------------------------- #
class _ToyTok:
    """decode(ids) → 每个 id 一个字母（够测增量解码与拼接）。"""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i % 26) for i in ids if not (skip_special_tokens and i == EOS))


@pytest.mark.parametrize("direct", [False, True], ids=["via-coordinator", "direct"])
def test_remote_agents_and_gateway_end_to_end(configs, direct) -> None:
    pytest.importorskip("zmq")
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    from vllm.p2p.remote import AgentBridge, RemoteClient
    from vllm.p2p.tiered import Engine, TieredCoordinator
    from vllm.p2p.tiered_gateway import IncrementalDetok, make_app

    rng = random.Random(7)
    prompts = [[rng.randrange(3, V) for _ in range(rng.randint(3, 30))] for _ in range(10)]
    MAXT = 12

    async def go():
        cores, bridges, tasks, clients = [], [], [], []
        spec = [("front-A", "front", None), ("front-B", "front", None),
                ("back-t-0", "back", "t"), ("back-t-1", "back", "t")]
        port0 = 25000 + random.Random().randrange(2000)
        for k, (name, role, task) in enumerate(spec):
            core = FakeCore(make_sched(configs[role], 64),
                            FrontRunner() if role == "front" else BackRunner(), 30 + k)
            core._ensure()
            br = AgentBridge(core, f"tcp://127.0.0.1:{port0 + k}",
                             {"name": name, "role": role, "task": task, "layers": [0, 2],
                              "pp": 1, "max_num_seqs": 16})
            cores.append(core)
            bridges.append(br)
            tasks.append(asyncio.get_running_loop().create_task(br.run()))
        engines = []
        for k, (name, role, task) in enumerate(spec):
            cl = RemoteClient(f"tcp://127.0.0.1:{port0 + k}")
            info = await cl.connect(timeout=10, direct=direct)
            assert info["name"] == name and info["role"] == role
            clients.append(cl)
            engines.append(Engine(name, role, cl, task=task))
        co = TieredCoordinator(engines[:2], engines[2:], eos=EOS, ignore_eos=False,
                               direct=direct)
        co.start()
        stages = {n: {"nodes": [f"N-{n}"]} for n, _, _ in spec}
        app = make_app(co, stages=stages, tokenizer=_ToyTok(), max_model_len=256)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            async def one(i, p):
                r = await client.post("/v1/completions", json={
                    "prompt": p, "max_tokens": MAXT, "temperature": 0.0, "stream": True,
                    "p2p_task": "t", "p2p_id": f"q{i}", "stream_options": {"include_usage": True}})
                assert r.status == 200, await r.text()
                text, usage, fin, met = "", None, None, None
                async for line in r.content:
                    line = line.strip()
                    if not line.startswith(b"data:") or line == b"data: [DONE]":
                        continue
                    d = json.loads(line[5:])
                    assert "error" not in d, d
                    for c in d.get("choices") or []:
                        text += c["text"]
                        fin = c["finish_reason"] or fin
                    usage = d.get("usage") or usage
                    met = d.get("p2p_metrics") or met
                return {"text": text, "usage": usage, "finish": fin, "metrics": met,
                        "channel": r.headers.get("X-P2P-Channel"),
                        "front": r.headers.get("X-P2P-Front")}

            res = await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
            nb = await client.post("/v1/completions", json={
                "prompt": prompts[0], "max_tokens": MAXT, "temperature": 0.0,
                "p2p_task": "t", "p2p_id": "nostream"})
            nonstream = await nb.json()
            nob = await client.post("/v1/completions", json={
                "prompt": prompts[0], "max_tokens": 4, "temperature": 0.0, "p2p_task": "zzz"})
            assert nob.status == 503 and "没有后段" in (await nob.json())["error"]["message"]
            recs = (await (await client.get("/p2p/records")).json())["records"]
            chans = await (await client.get("/p2p/channels")).json()
            await asyncio.sleep(0.1)
            n_direct = sum(br.n_direct for br in bridges)
            assert (n_direct > 0) == direct           # 直连时前后段代理之间真的互发了
            assert not any(br.routes or br.held for br in bridges)   # 路由都收干净了
            return res, nonstream, recs, chans, cores
        finally:
            await client.close()
            await co.close()
            for t in tasks:
                t.cancel()
            for c in cores:
                c.close()
            for cl in clients:
                cl.shutdown()

    res, nonstream, recs, chans, cores = asyncio.run(go())
    tok = _ToyTok()
    for p, r in zip(prompts, res):
        want = mono(p, MAXT, EOS)
        assert r["text"] == tok.decode(want), (r, want)
        assert r["usage"]["completion_tokens"] == len(want)
        assert r["finish"] == ("stop" if want[-1] == EOS else "length")
        assert r["channel"] in ("back-t-0", "back-t-1") and r["front"] in ("front-A", "front-B")
        assert r["metrics"]["front_prefill_ms"] is not None
        # 代理量的「引擎内」耗时随输出带回来了（前段：append → hidden；后段：feed → token）
        assert r["metrics"]["front_engine_ms_p50"] is not None
        assert r["metrics"]["back_engine_ms_p50"] is not None
    assert {r["front"] for r in res} == {"front-A", "front-B"}          # 前段池：两个都用上了
    assert {r["channel"] for r in res} == {"back-t-0", "back-t-1"}      # 同 task 两个后段
    assert nonstream["choices"][0]["text"] == tok.decode(mono(prompts[0], MAXT, EOS))
    by = {r["id"]: r for r in recs}
    assert by["q0"]["front_nodes"] and by["q0"]["back_nodes"] and by["q0"]["status"] == 200
    assert [c["name"] for c in chans["channels"]] == ["back-t-0", "back-t-1"]
    assert {f["name"] for f in chans["fronts"]} == {"front-A", "front-B"}
    assert not any(c.sched.requests for c in cores)                    # 两端都收干净了

    # 增量解码：多字节字符没凑齐时不吐、凑齐了一次吐出
    class _Bytes:
        def decode(self, ids, skip_special_tokens=True):
            return bytes(ids).decode("utf-8", errors="replace")
    d = IncrementalDetok(_Bytes())
    han = list("中".encode())
    assert d.push([ord("a")]) == "a" and d.push(han[:2]) == "" and d.push(han[2:]) == "中"
    assert d.flush() == "" and d.text == "a中"


# --------------------------------------------------------------------------- #
# 前段出口识别：前段在 prompt 算完那一步交出「特征」，协调器判 task、按它绑后段
# --------------------------------------------------------------------------- #
def body_class(prompt: list[int], body) -> int:
    """假识别器的「真值」：body token 之和的奇偶。"""
    return sum(prompt[p] for a, b in body for p in range(a, b)) % 2


class LRFrontRunner(FrontRunner):
    """FrontRunner + 统计钩子的替身：prompt 算完的那一步交一次 p2p_feat（只数 body token，
    body 从请求的 extra_args 里读 —— 钉住 body 一路传到了 worker）。"""

    def __init__(self):
        super().__init__()
        self.meta: dict[str, tuple[int, list]] = {}

    def step(self, so):
        from vllm.p2p.stats import parse_body

        for nr in so.scheduled_new_reqs:
            if nr.req_id not in self.meta:
                xa = nr.sampling_params.extra_args or {}
                self.meta[nr.req_id] = (len(nr.prompt_token_ids), parse_body(xa.get("p2p_body")))
        out = super().step(so)
        comp, feats = _computed(so), {}
        for rid, n in so.num_scheduled_tokens.items():
            n_prompt, body = self.meta.get(rid, (0, None))
            if body is not None and comp[rid] < n_prompt <= comp[rid] + n:
                t = self.toks[rid]
                feats[rid] = {"cls": body_class(t[:n_prompt], body), "n": n_prompt}
        for rid in so.finished_req_ids:
            self.meta.pop(rid, None)
        out.p2p_feat = feats or None
        return out


def _toy_classify(feat, info):
    if "cls" not in feat:
        raise KeyError("没有 cls")
    return {"task": f"t{feat['cls']}", "label": f"t{feat['cls']}", "conf": 1.0, "zone": "commit",
            "n_prompt": info["n_prompt"]}


@pytest.mark.parametrize("mode", ["lr", "shadow"])
def test_front_exit_identification_end_to_end(configs, mode) -> None:
    """2 前段 + 后段 t0×2、t1×1，全隔着 ZMQ（直连）+ 网关：lr 模式按前段交出的特征绑
    （请求自带的 p2p_task 故意一半是错的，绑定不受影响、只记对错）；shadow 按标注绑、只记识别。"""
    pytest.importorskip("zmq")
    pytest.importorskip("aiohttp")
    from aiohttp.test_utils import TestClient, TestServer

    from vllm.p2p.remote import AgentBridge, RemoteClient
    from vllm.p2p.tiered import Engine, TieredCoordinator
    from vllm.p2p.tiered_gateway import make_app

    rng = random.Random(11)
    prompts = [[rng.randrange(3, V) for _ in range(rng.randint(6, 40))] for _ in range(12)]
    bodies = [[[1, len(p) - 2]] for p in prompts]
    truth = [f"t{body_class(p, b)}" for p, b in zip(prompts, bodies)]
    assert set(truth) == {"t0", "t1"}
    labels = [t if i % 2 else "t9" for i, t in enumerate(truth)]   # lr：一半标签是错的
    if mode == "shadow":
        labels = truth
    MAXT = 8

    async def go():
        cores, bridges, tasks, clients, engines = [], [], [], [], []
        spec = [("front-A", "front", None), ("front-B", "front", None),
                ("back-t0-0", "back", "t0"), ("back-t0-1", "back", "t0"),
                ("back-t1-0", "back", "t1")]
        port0 = 27000 + random.Random().randrange(2000)
        for k, (name, role, task) in enumerate(spec):
            core = FakeCore(make_sched(configs[role], 64),
                            LRFrontRunner() if role == "front" else BackRunner(), 50 + k)
            core._ensure()
            br = AgentBridge(core, f"tcp://127.0.0.1:{port0 + k}",
                             {"name": name, "role": role, "task": task, "layers": [0, 2],
                              "pp": 1, "max_num_seqs": 16})
            cores.append(core)
            bridges.append(br)
            tasks.append(asyncio.get_running_loop().create_task(br.run()))
        for k, (name, role, task) in enumerate(spec):
            cl = RemoteClient(f"tcp://127.0.0.1:{port0 + k}")
            await cl.connect(timeout=10, direct=True)
            clients.append(cl)
            engines.append(Engine(name, role, cl, task=task))
        co = TieredCoordinator(engines[:2], engines[2:], eos=EOS, direct=True,
                               classify=_toy_classify, route="lr" if mode == "lr" else "task")
        co.start()
        app = make_app(co, stages={n: {"nodes": [f"N-{n}"]} for n, _, _ in spec},
                       tokenizer=_ToyTok(), mode=mode, max_model_len=256, needs_body=True,
                       identifier="toy")
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            async def one(i):
                r = await client.post("/v1/completions", json={
                    "prompt": prompts[i], "max_tokens": MAXT, "temperature": 0.0,
                    "p2p_task": labels[i], "p2p_id": f"q{i}", "p2p_body": bodies[i]})
                assert r.status == 200, await r.text()
                d = await r.json()
                return {"text": d["choices"][0]["text"], "channel": r.headers["X-P2P-Channel"],
                        "task": r.headers["X-P2P-Task"]}

            res = await asyncio.gather(*(one(i) for i in range(len(prompts))))
            nobody = await client.post("/v1/completions", json={
                "prompt": prompts[0], "max_tokens": 2, "temperature": 0.0, "p2p_task": "t0"})
            recs = {r["id"]: r for r in
                    (await (await client.get("/p2p/records")).json())["records"]}
            await asyncio.sleep(0.1)
            assert not any(br.routes or br.held for br in bridges)
            return res, nobody.status, recs, cores
        finally:
            await client.close()
            await co.close()
            for t in tasks:
                t.cancel()
            for c in cores:
                c.close()
            for cl in clients:
                cl.shutdown()

    res, nobody, recs, cores = asyncio.run(go())
    tok = _ToyTok()
    for i, (p, r) in enumerate(zip(prompts, res)):
        assert r["text"] == tok.decode(mono(p, MAXT, EOS))
        # 两种模式下绑定的都是真 task 的后段：lr 靠识别（标签一半是错的），shadow 靠标签
        assert r["task"] == truth[i] and r["channel"].startswith(f"back-{truth[i]}-")
        rec = recs[f"q{i}"]
        assert rec["true_task"] == labels[i] and rec["task"] == truth[i]
        v = rec["verdict"] if mode == "lr" else rec["shadow"]
        assert v["task"] == truth[i] and v["compute_ms"] is not None
        assert (rec["verdict"] is None) == (mode == "shadow")
    assert {r["channel"] for r in res if r["task"] == "t0"} == {"back-t0-0", "back-t0-1"}
    # lr：识别器要 body，没带 body 直接 400；shadow 照常按标签跑、识别记下「没法判」
    assert nobody == (400 if mode == "lr" else 200)
    assert not any(c.sched.requests for c in cores)


def test_front_exit_identify_failure_does_not_peek_at_label() -> None:
    """识别抛错（或前段没交特征）：lr 模式下不偷看请求自带的 task，在全部后段里挑。"""
    from vllm.p2p.tiered import Engine, TieredCoordinator, _Req

    backs = [Engine("b0", "back", None, task="t0"), Engine("b1", "back", None, task="t1")]
    co = TieredCoordinator(Engine("f", "front", None), backs, classify=_toy_classify, route="lr")
    loop = asyncio.new_event_loop()
    try:
        r = _Req("x", [1, 2, 3], 4, "t1", loop)
        r.front = co.front
        co._identify(r)
        assert r.task is None and "前段没有交出" in r.info["verdict"]["error"]
        r.feat = {"bogus": 1}
        co._identify(r)
        assert r.task is None and "KeyError" in r.info["verdict"]["error"]
        r.feat = {"cls": 1}
        co._identify(r)
        assert r.task == "t1" and co._pick(r).name == "b1"
    finally:
        loop.close()
    with pytest.raises(ValueError):
        TieredCoordinator(Engine("f", "front", None), backs, route="lr")


def test_lr_classify_reads_route_stats_wire() -> None:
    """真的选择器文件 + 统计钩子交出的格式：前 L₀ 层齐了能判（不退先验报错），缺层时退先验并说明。"""
    fw = ROOT.parent / "p2p-framework"
    npz = fw / "selector" / "selector_body_L8_params.npz"
    if not npz.is_file():
        pytest.skip("旁边没有 p2p-framework 的选择器文件")
    import sys

    import numpy as np

    sys.path.insert(0, str(fw))
    from vllm.p2p.stats import RouteStats
    from vllm.p2p.tiered_gateway import make_lr_classify

    E, K, L0 = 128, 8, 8
    classify, clf = make_lr_classify(lr_model=str(npz), l0=L0, n_experts=E,
                                     tasks={"gsm8k": 1.0, "mbpp": 1.0})
    st = RouteStats(n_experts=E, top_k=K, norm_topk=True, lr_layers=range(L0), channel="f",
                    emit_lr=True, write=False, prefill_only=True)
    for l in range(L0):
        st.register_layer(l, None)
    st.allocate(64, "cpu")
    lg = torch.from_numpy(np.random.default_rng(0).standard_normal((20, E)).astype(np.float32))
    for slot in range(L0):
        st.capture(slot, lg)
    from types import SimpleNamespace
    reqs = {"r": SimpleNamespace(num_prompt_tokens=20, sampling_params=SimpleNamespace(
        extra_args={"p2p_id": "r", "p2p_body": "3:17"}))}
    st.on_step(["r"], [20], np.arange(20), reqs)
    feat = st.take_lr("r")
    assert sorted(int(k) for k in feat) == list(range(1, L0 + 1))
    v = classify(feat, {"n_prompt": 20, "body": [[3, 17]]})
    assert v["task"] in ("gsm8k", "mbpp") and not v.get("error") and v["n_body"] == 14
    assert v["label"] and v["zone"] in ("commit", "observe", "prior")
    del feat["8"]
    v2 = classify(feat, {"n_prompt": 20, "body": [[3, 17]]})
    assert v2["zone"] == "prior" and v2.get("error")


def test_tiered_stats_groups_by_path(tmp_path, capsys) -> None:
    from vllm.p2p import tiered_stats

    def row(front, ch, n, total, ttft, fl=3.0, bl=9.0):
        return {"front": front, "channel": ch, "n_generated": n, "total_ms": total,
                "prefill_ms": ttft, "engine": {"front_prefill_ms": 50, "bind_ms": 1,
                                                "front_leg_ms_p50": fl, "back_leg_ms_p50": bl}}
    b = [row("N01", "back-g-0", 11, 200, 100), row("N10", "back-g-0", 11, 300, 100),
         row("N01", "back-g-0", 11, 220, 120)]
    a = [{"channel": "g-0", "n_generated": 11, "total_ms": 150, "prefill_ms": 50}]
    (tmp_path / "b.json").write_text(json.dumps({"requests": b}))
    (tmp_path / "a.json").write_text(json.dumps({"requests": a}))
    assert tiered_stats.main(["--b", str(tmp_path / "b.json"), "--a", str(tmp_path / "a.json")]) == 0
    s = {(x["front"], x["back"]): x for x in tiered_stats.summarise_b(b)}
    assert s[("N01", "back-g-0")]["n"] == 2 and s[("N01", "back-g-0")]["tpot"] == 10.0
    assert s[("N10", "back-g-0")]["tpot"] == 20.0
    assert "10 / 50（1 条）" in capsys.readouterr().out


def test_front_pick_speed_aware() -> None:
    from vllm.p2p.tiered import Engine, TieredCoordinator

    fast, slow = Engine("fast", "front", None), Engine("slow", "front", None)
    co = TieredCoordinator([fast, slow], [Engine("b", "back", None, task="t")])
    # 还没数据：轮流
    assert {co._pick_front().name for _ in range(4)} == {"fast", "slow"}
    fast.observe(6.0)
    slow.observe(28.0)
    picks = []
    for _ in range(6):                     # 每挑一个就给它记一条在飞
        f = co._pick_front()
        f.inflight += 1
        picks.append(f.name)
    # 慢 4.7 倍：前 4 条都给快的（(4+1)×6=30 > (0+1)×28），第 5 条才轮到慢的
    assert picks[:4] == ["fast"] * 4 and "slow" in picks
    co2 = TieredCoordinator([fast, slow], [Engine("b", "back", None, task="t")], front_pick="least")
    fast.inflight = slow.inflight = 0
    assert {co2._pick_front().name for _ in range(4)} == {"fast", "slow"}
