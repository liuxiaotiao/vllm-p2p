# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 / A4 端到端：识别服务（真的跑 p2p-framework 的 TorchSegmentModel + LR）→ 网关 → 假的
vLLM 通道（按 vLLM 的 SSE 格式流式回包）→ bench 落出的结果能被 `deploy_15.sh report` 读。

要 torch、aiohttp、tokenizers 与旁边的 p2p-framework；缺哪个跳过。不需要 GPU，不需要 vLLM。
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("tokenizers")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
FW = ROOT.parent / "p2p-framework"
if not (FW / "p2pmoe").is_dir():
    pytest.skip("旁边没有 p2p-framework", allow_module_level=True)
sys.path.insert(0, str(FW))
sys.path.insert(0, str(ROOT / "vllm" / "p2p"))          # 与「按文件路径运行」同一种 import

import bench as B          # noqa: E402
import gateway as G        # noqa: E402
import identifier as I     # noqa: E402

TASKS = ["gsm8k", "mbpp", "no_robots"]
L0 = 2


# --------------------------------------------------------------------------- #
class Servers:
    """在一个后台事件循环里跑若干 aiohttp app，端口 0 自动分配。"""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.th = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.th.start()
        self.runners = []

    def add(self, app) -> int:
        from aiohttp import web

        async def go():
            r = web.AppRunner(app)
            await r.setup()
            s = web.TCPSite(r, "127.0.0.1", 0)
            await s.start()
            self.runners.append(r)
            return s._server.sockets[0].getsockname()[1]
        return asyncio.run_coroutine_threadsafe(go(), self.loop).result(30)

    def close(self):
        async def stop():
            for r in self.runners:
                await r.cleanup()
        asyncio.run_coroutine_threadsafe(stop(), self.loop).result(30)
        self.loop.call_soon_threadsafe(self.loop.stop)


def fake_vllm(name: str, seen: list):
    """按 vLLM 0.14 /v1/completions 的流式格式回包；记下收到的请求体。"""
    from aiohttp import web

    async def comp(request):
        d = await request.json()
        seen.append((name, d, dict(request.headers)))
        assert d["model"] == name
        assert not any(k.startswith("p2p_") for k in d), d.keys()
        n = min(int(d["max_tokens"]), 3 + len(d["prompt"]) % 4)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for j in range(n):
            await asyncio.sleep(0.005)
            fin = None if j < n - 1 else ("length" if n == d["max_tokens"] else "stop")
            ch = {"id": "cmpl-x", "choices": [{"index": 0, "text": f"t{j} ",
                                               "finish_reason": fin, "stop_reason": None}]}
            so = d.get("stream_options") or {}
            if so.get("include_usage") and so.get("continuous_usage_stats"):   # 与 vLLM 同：每块带累计数
                ch["usage"] = {"prompt_tokens": len(d["prompt"]), "completion_tokens": j + 1,
                               "total_tokens": len(d["prompt"]) + j + 1}
            await resp.write(f"data: {json.dumps(ch)}\n\n".encode())
        usage = {"id": "cmpl-x", "choices": [],
                 "usage": {"prompt_tokens": len(d["prompt"]), "completion_tokens": n,
                           "total_tokens": len(d["prompt"]) + n}}
        if d.get("vllm_xargs", {}).get("p2p_id"):
            usage["p2p_metrics"] = {"queued_ms": 1.5, "prefill_ms": 3.0, "decode_ms": 9.0,
                                    "ttft_ms": 4.5, "num_generation_tokens": n}
        await resp.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        await resp.write_eof()
        return resp

    async def metrics(_):
        return web.Response(text="vllm:num_preemptions_total 0\n"
                                 "vllm:request_queue_time_seconds_sum 0.5\n"
                                 "vllm:request_queue_time_seconds_count 4\n")

    app = web.Application()
    app.add_routes([web.post("/v1/completions", comp), web.get("/metrics", metrics)])
    return app


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from p2pmoe.sim.fake_checkpoint import (TINY_QWEN3_MOE, write_fake_checkpoint,
                                            write_fake_tokenizer)

    d = tmp_path_factory.mktemp("a3a4")
    ckpt = d / "ckpt"
    cfg = dict(TINY_QWEN3_MOE, torch_dtype="float32")
    write_fake_checkpoint(ckpt, cfg, seed=3)
    write_fake_tokenizer(ckpt, vocab_size=int(cfg["vocab_size"]))
    (ckpt / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))
    E = int(cfg["num_experts"])
    rng = np.random.default_rng(0)
    lr = d / "lr.json"
    lr.write_text(json.dumps({"classes": TASKS, "coef": rng.standard_normal((3, L0 * E)).tolist(),
                              "intercept": [0.0, 0.0, 0.0],
                              "features": {"tokens": "body", "normalize": "per_token"}}))
    ident = I.Identifier(model_dir=str(ckpt), lr_model=str(lr),
                         tasks={u: 1.0 for u in TASKS}, l0=L0, device="cpu", dtype="float32",
                         tau_hi=0.55)
    cases = d / "cases.jsonl"
    rows = []
    for i in range(9):
        u = TASKS[i % 3]
        rows.append({"id": f"{u}:{i}", "task": u, "segments": [[f"Task {u}: ", True],
                                                               [f"the quick brown fox {i} " * (i + 1), False]]})
    cases.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    srv = Servers()
    seen: list = []
    names = [("gsm8k-0", "gsm8k"), ("gsm8k-1", "gsm8k"), ("mbpp-0", "mbpp"),
             ("no_robots-0", "no_robots")]
    chans = []
    for nm, u in names:
        port = srv.add(fake_vllm(nm, seen))
        chans.append({"name": nm, "task": u, "nodes": [f"F{nm}", f"B{nm}a", f"B{nm}b"],
                      "partition": [2, 1, 1], "api": f"http://127.0.0.1:{port}/v1",
                      "commands": []})
    cj = d / "channels.json"
    cj.write_text(json.dumps({"plan": "fake.json", "l0": L0, "n_layers": 4, "n_experts": E,
                              "channels": chans}))
    iport = srv.add(I.make_app(ident))
    yield {"dir": d, "ckpt": ckpt, "lr": lr, "ident": ident, "cases": cases, "srv": srv,
           "seen": seen, "channels": cj, "ident_url": f"http://127.0.0.1:{iport}", "E": E}
    srv.close()


def gateway(world, mode: str) -> str:
    router = G.Router(G.load_channels(world["channels"]))
    port = world["srv"].add(G.make_app(router, mode=mode, identifier=world["ident_url"],
                                       ident_layers=True))
    return f"http://127.0.0.1:{port}"


def run_bench(world, gw: str, out: Path, *extra: str) -> dict:
    rc = B.main(["--gateway", gw, "--channels", str(world["channels"]),
                 "--cases", str(world["cases"]), "--model-dir", str(world["ckpt"]),
                 "--p2p-framework", str(FW), "--max-tokens", "6", "--inflight", "3",
                 "--warmup", "1", "--out", str(out), *extra])
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
def test_identifier_counts_only_body_tokens(world) -> None:
    from p2pmoe.runtime.text import TextIO

    tio = TextIO.from_model_dir(world["ckpt"], chat=True)
    tio.template_kw["enable_thinking"] = False
    bp = tio.encode_body([("Task gsm8k: ", True), ("the quick brown fox", False)])
    out = world["ident"].identify(bp.ids, bp.body, want_layers=True)
    assert out["task"] in TASKS and out["zone"] in ("commit", "observe", "prior")
    assert out["n_body"] == bp.n_body < len(bp.ids)
    k = 2                                     # TINY_QWEN3_MOE 的 top_k
    for l in ("1", "2"):
        assert out["layers"][l]["t"] == bp.n_body
        assert sum(out["layers"][l]["c"]) == bp.n_body * k
    with pytest.raises(ValueError, match="body"):
        world["ident"].identify(bp.ids, None)


def test_router_least_inflight_and_round_robin() -> None:
    cs = [G.ChannelInfo("a-0", "a", "http://x"), G.ChannelInfo("a-1", "a", "http://y"),
          G.ChannelInfo("b-0", "b", "http://z")]
    r = G.Router(cs)
    first = r.pick("a")
    first.inflight += 1
    assert r.pick("a") is not first                  # 取在途少的
    cs[0].inflight = cs[1].inflight = 2
    got = {r.pick("a").name for _ in range(4)}
    assert got == {"a-0", "a-1"}                     # 相同时轮转
    with pytest.raises(G.NoChannel, match="一条通道都没有"):
        r.pick("c")
    fwd, ext = G.strip_p2p({"prompt": [1], "p2p_task": "a", "p2p_body": [[0, 1]], "n": 1})
    assert ext == {"p2p_task": "a", "p2p_body": [[0, 1]]} and fwd == {"prompt": [1], "n": 1}
    x = G.with_xargs({"vllm_xargs": {"foo": 1}}, "id7", [[0, 3], [5, 6]])
    assert x["vllm_xargs"] == {"foo": 1, "p2p_id": "id7", "p2p_body": "0:3,5:6"}


def test_oracle_end_to_end_and_report(world, tmp_path) -> None:
    gw = gateway(world, "oracle")
    world["seen"].clear()
    out = run_bench(world, gw, tmp_path / "e3.json")
    rs = out["requests"]
    assert len(rs) == 9 and not out["failed"]
    assert out["mode"] == "static" and out["gateway_mode"] == "oracle"
    for r in rs:
        assert r["task"] == r["true_task"] and r["channel"].startswith(r["true_task"])
        assert r["queue_back_ms"] == 1.5 and r["n_generated"] >= 3
        assert r["prefill_ms"] <= r["total_ms"] and r["correct"] is None
    # 同 task 两条通道都分到了请求（取在途少的）
    assert {r["channel"] for r in rs if r["true_task"] == "gsm8k"} == {"gsm8k-0", "gsm8k-1"}
    # vLLM 收到的是 token id、body 区间进了 vllm_xargs、请求 id 进了头
    reqs = [d for _, d, _ in world["seen"] if d["vllm_xargs"]["p2p_id"].startswith("req")]
    assert all(isinstance(d["prompt"], list) and d["temperature"] == 0.0 for d in reqs)
    assert all(re.fullmatch(r"(\d+:\d+,?)+", d["vllm_xargs"]["p2p_body"]) for d in reqs)
    run = out["run"]
    assert run["n_requests"] == 9 and run["tpot_ms"]["n"] > 0 and run["accuracy"] is None
    assert set(run["by_task"]) == set(TASKS)
    em = out["channels"]["gsm8k-0"]["engine_metrics"]      # 假服务器的计数器不动 → 差为 0
    assert em["vllm:request_queue_time_seconds_count"] == 0 and "queue_ms_mean" not in em
    # p2p-framework 的 `deploy_15.sh report` 直接读得了
    sh = (FW / "deploy_15.sh").read_text(encoding="utf-8")
    m = re.search(r"cmd_report\(\) \{.*?<<'PYEOF'\n(.*?)\nPYEOF", sh, re.S)
    assert m, "deploy_15.sh 里找不到 report 的脚本"
    script = tmp_path / "report.py"
    script.write_text(m.group(1), encoding="utf-8")
    p = subprocess.run([sys.executable, str(script), str(tmp_path / "e3.json")],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "连续性测试" in p.stdout


def test_any_mode_ignores_task_for_full_expert_channels(world, tmp_path) -> None:
    """全专家部署（如 plan_full_2ch）：两条通道都能服务任何 task。any 模式不看 p2p_task、不要识别器，
    在全部通道里挑在途最少的；bench 不因「这个 task 没有通道」跳过 case。"""
    cs = [G.ChannelInfo("a-0", "a", "http://x"), G.ChannelInfo("b-0", "b", "http://y")]
    r = G.Router(cs)
    first = r.pick_any()
    first.inflight += 1
    assert r.pick_any() is not first
    cs[0].inflight = cs[1].inflight = 3
    assert {r.pick_any().name for _ in range(4)} == {"a-0", "b-0"}

    full = json.loads(world["channels"].read_text(encoding="utf-8"))
    full["channels"] = [c for c in full["channels"] if c["name"] in ("gsm8k-0", "mbpp-0")]
    cj = tmp_path / "channels_full2.json"
    cj.write_text(json.dumps(full))
    router = G.Router(G.load_channels(cj))
    port = world["srv"].add(G.make_app(router, mode="any"))
    rc = B.main(["--gateway", f"http://127.0.0.1:{port}", "--channels", str(cj),
                 "--cases", str(world["cases"]), "--model-dir", str(world["ckpt"]),
                 "--p2p-framework", str(FW), "--max-tokens", "6", "--warmup", "1",
                 "--out", str(tmp_path / "any.json")])
    assert rc == 0
    out = json.loads((tmp_path / "any.json").read_text(encoding="utf-8"))
    rs = out["requests"]
    assert len(rs) == 9 and not out["failed"] and out["gateway_mode"] == "any"
    assert {r["true_task"] for r in rs} == set(TASKS)                    # no_robots 也跑了
    assert {r["channel"] for r in rs} == {"gsm8k-0", "mbpp-0"}           # 两条都用上
    assert any(r["true_task"] != r["channel"].rsplit("-", 1)[0] for r in rs)   # 不按 task 分
    assert out["run"]["accuracy"] is None and out["concurrency"] == 2    # 在途默认 = 通道数
    from vllm.p2p.tiered_stats import continuous_report
    txt = "\n".join(continuous_report(out))
    assert "识别：" not in txt and "no_robots" in txt


def test_bench_marks_at_generated_lengths(world, tmp_path) -> None:
    """--marks：每条请求生成到第 k 个 token 时打点（靠 continuous_usage_stats 的累计数），
    没生成到 k 的请求没有这一档；汇总里多一张打点表。"""
    gw = gateway(world, "oracle")
    out = run_bench(world, gw, tmp_path / "marks.json", "--marks", "2,4,64")
    rs = out["requests"]
    assert out["run"]["marks"] == [2, 4, 64]
    for r in rs:
        m = r["marks"]
        assert "2" in m and "64" not in m                         # 至少 3 个 token；最多 6 个
        assert ("4" in m) == (r["n_generated"] >= 4)
        a = m["2"]
        assert a["n"] == 2 and 0 < a["since_first_ms"] <= a["t_ms"] <= r["total_ms"] + 1
        assert a["tpot_ms"] == pytest.approx(a["since_first_ms"], rel=1e-3)
        assert a["tok_per_s"] == pytest.approx(2 / (a["t_ms"] / 1e3), rel=1e-2)
        if "4" in m:
            assert m["4"]["t_ms"] >= a["t_ms"] and m["4"]["tpot_ms"] == pytest.approx(
                m["4"]["since_first_ms"] / 3, rel=1e-3)
    ms = out["run"]["marks_stats"]
    assert ms["all"]["2"]["n"] == len(rs) and ms["all"]["4"]["n"] == sum(r["n_generated"] >= 4 for r in rs)
    from vllm.p2p.tiered_stats import continuous_report
    txt = "\n".join(continuous_report(out))
    assert "生成长度打点" in txt and "k=4" in txt


def test_lr_mode_routes_by_identifier(world, tmp_path) -> None:
    gw = gateway(world, "lr")
    out = run_bench(world, gw, tmp_path / "e4.json")
    rs = out["requests"]
    assert len(rs) == 9 and out["mode"] == "dynamic"
    for r in rs:
        assert r["zone"] in ("commit", "observe", "prior") and r["raw_scores"]
        assert r["channel"].startswith(r["task"])                # 按识别结果路由
        assert r["correct"] == (r["task"] == r["true_task"])
        assert r["queue_ms"] > 0 and r["identify"]["compute_ms"] > 0
    acc = out["run"]["accuracy"]
    assert acc is not None and acc["n"] == 9
    # 与直接调识别器的结论逐条一致
    from p2pmoe.runtime.text import TextIO
    tio = TextIO.from_model_dir(world["ckpt"], chat=True)
    tio.template_kw["enable_thinking"] = False
    cases = [json.loads(x) for x in world["cases"].read_text().splitlines()]
    by_case = {r["case_id"]: r for r in rs}
    for c in cases:
        bp = tio.encode_body([tuple(s) for s in c["segments"]])
        assert world["ident"].identify(bp.ids, bp.body)["task"] == by_case[c["id"]]["task"]


def test_shadow_mode_records_but_does_not_route(world, tmp_path) -> None:
    gw = gateway(world, "shadow")
    out = run_bench(world, gw, tmp_path / "shadow.json")
    rs = out["requests"]
    assert all(r["task"] == r["true_task"] for r in rs)          # 路由按标注
    assert all(r.get("shadow") and "raw" in r["shadow"] for r in rs)
    assert out["summary"]["shadow_accuracy"] is not None


def test_unknown_task_fails_fast(world) -> None:
    import urllib.error
    import urllib.request

    gw = gateway(world, "oracle")
    req = urllib.request.Request(f"{gw}/v1/completions", method="POST",
                                 data=json.dumps({"prompt": [1, 2], "max_tokens": 2,
                                                  "p2p_task": "math"}).encode(),
                                 headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=10)
    assert ei.value.code == 503
    assert "一条通道都没有" in json.loads(ei.value.read())["error"]["message"]


# --------------------------------------------------------------------------- #
def test_small_helpers() -> None:
    assert B.parse_sse_line(b"data: [DONE]\n") is None
    assert B.parse_sse_line(b": keepalive\n") is None
    assert B.parse_sse_line(b'data: {"a": 1}\n') == {"a": 1}
    assert B.stop_reason_of("length", None) == "max_tokens"
    assert B.stop_reason_of("stop", None) == "eos"
    assert B.stop_reason_of("stop", 151645) == "eos"
    assert B.stop_reason_of("stop", "\n\n") == "stop_string"
    s = B.summarise_smi("2026/10/13 05:00:00.1, 50, 20000, 24576, 100.5\n"
                        "2026/10/13 05:00:01.1, 70, 21000, 24576, 120.5\nbad line\n")
    assert s["gpu_util_mean"] == 60.0 and s["mem_used_max_mb"] == 21000 and s["n_samples"] == 2
    assert s["compute_util"] == 0.0
    assert B.summarise_smi("t, 584, 1, 2, 3\n")["error"]
    p0 = B.parse_prom('vllm:request_queue_time_seconds_sum{model_name="x"} 1.0\n'
                      'vllm:request_queue_time_seconds_count{model_name="x"} 2\n# HELP x\n')
    p1 = B.parse_prom('vllm:request_queue_time_seconds_sum{model_name="x"} 3.0\n'
                      'vllm:request_queue_time_seconds_count{model_name="x"} 4\n')
    assert B.metrics_delta(p0, p1)["queue_ms_mean"] == 1000.0
    rows = [{"req": "r1"}, {"req": "r2"}]
    st = {"r1": [{"pp_rank": 1, "layers": [9, 21], "n_seen": 10,
                  "prefill": {"n_token_layer": 26, "miss_token_layer": 2, "miss_mass": 0.1},
                  "decode": {"n_token_layer": 13, "miss_token_layer": 1, "miss_mass": 0.05},
                  "decode_miss_seq": [1]},
                 {"pp_rank": 0, "layers": [1, 8], "n_seen": 10,
                  "prefill": {"n_token_layer": 16, "miss_token_layer": 0, "miss_mass": 0.0},
                  "decode": {"n_token_layer": 8, "miss_token_layer": 0, "miss_mass": 0.0},
                  "decode_miss_seq": [0], "lr_layers": {"1": {"i": [], "c": [], "w": [],
                                                              "t": 0, "e": 4}}}]}
    assert B.join_stats(rows, st) == 1
    assert rows[0]["stats"]["miss_rate"] == round(3 / 63, 4)
    assert [s["pp_rank"] for s in rows[0]["stats"]["stages"]] == [0, 1]
    assert "_front_layers" in rows[0] and "stats" not in rows[1]


def test_compare_two_runs(world, tmp_path) -> None:
    """A2 验收脚本：两份结果按 case_id 配对、按 token 比前缀；改掉一条的输出要被抓到。"""
    import compare as C

    gw = gateway(world, "oracle")
    a = run_bench(world, gw, tmp_path / "a.json")
    b = json.loads(json.dumps(a))
    b["requests"] = list(reversed(b["requests"]))              # 顺序不同也要配得上
    b["requests"][0]["text"] = "完全不同的输出 " + b["requests"][0]["text"]
    (tmp_path / "b.json").write_text(json.dumps(b), encoding="utf-8")
    rc = C.main(["--a", str(tmp_path / "a.json"), "--b", str(tmp_path / "b.json"),
                 "--model-dir", str(world["ckpt"]), "--p2p-framework", str(FW),
                 "--out", str(tmp_path / "cmp.json")])
    res = json.loads((tmp_path / "cmp.json").read_text(encoding="utf-8"))
    assert res["all"]["n"] == 9
    assert res["all"]["prefix64_agree"] == round(8 / 9, 4) and rc == 1      # 8/9 < 95%
    assert len(res["diverged"]) == 1 and res["diverged"][0]["cp"] == 0
    pairs, ma, mb = C.pair_rows(a["requests"], a["requests"][:5])
    assert len(pairs) == 5 and ma == 4 and mb == 0
    assert C.common_prefix([1, 2, 3], [1, 2, 4]) == 2
