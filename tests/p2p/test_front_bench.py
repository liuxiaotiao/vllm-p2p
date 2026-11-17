# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""前段实测（vllm/p2p/front_bench.py）：拆分的纯函数、真调度器 + 假前段 runner 上的整套流程、
真实运行占用率、命令行。不需要 GPU。"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.p2p import front_bench as fb

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# 纯函数
# --------------------------------------------------------------------------- #
def test_parse_back_ms() -> None:
    assert fb.parse_back_ms("0,20:40") == [(0.0, 0.0), (20.0, 40.0)]
    assert fb.back_label((20.0, 40.0)) == "20:40" and fb.back_label((0.0, 0.0)) == "0"
    with pytest.raises(ValueError):
        fb.parse_back_ms("40:20")


def test_match_tokens_splits_queue_service_out() -> None:
    T = 1000.0
    steps = [  # 两步串行：第一步 10 ms（有 prefill），第二步 5 ms，批 2
        {"t": T + 0.000, "ms": 10.0, "n": 2, "npf": 1, "pf": 100},
        {"t": T + 0.012, "ms": 5.0, "n": 2, "npf": 0, "pf": 0},
    ]
    toks = [
        {"t_in": T - 0.001, "t_out": T + 0.0105},     # 第一步：排队 1、服务 10、回传 0.5（被 prefill 挡）
        {"t_in": T + 0.003, "t_out": T + 0.0175},     # 第一步算着时到的 → 第二步：排队 9、服务 5、回传 0.5
        {"t_in": T + 0.020, "t_out": T + 0.0176},     # 发出比那一步开始还晚 → 对不上
    ]
    rows = fb.match_tokens(toks, steps)
    a, b, c = rows
    assert a["step"] == 0 and a["mixed"] and a["svc_ms"] == 10.0
    assert a["queue_ms"] == pytest.approx(1.0) and a["out_ms"] == pytest.approx(0.5)
    assert b["step"] == 1 and not b["mixed"] and b["batch"] == 2
    assert b["queue_ms"] == pytest.approx(9.0) and b["wait_ms"] == pytest.approx(14.5 - 5.0)
    assert b["lat_ms"] == pytest.approx(b["queue_ms"] + b["svc_ms"] + b["out_ms"], abs=1e-6)
    assert c["step"] is None
    sm = fb.summarise_tokens(rows)
    assert sm["unmatched"] == 1 and sm["clean"]["n"] == 1 and sm["mixed"]["n"] == 1
    assert sm["clean"]["svc"]["mean"] == 5.0 and sm["clean"]["wait"]["p95"] == pytest.approx(9.5)


def test_summarise_steps_and_prefill_rows() -> None:
    steps = [{"t": 0.0, "ms": 100.0, "n": 1, "npf": 1, "pf": 512},
             {"t": 0.2, "ms": 10.0, "n": 2, "npf": 0, "pf": 0},
             {"t": 0.3, "ms": 20.0, "n": 3, "npf": 1, "pf": 64}]
    sp = fb.summarise_steps(steps)
    assert sp["steps"] == 3 and sp["span_s"] == pytest.approx(0.32)
    assert sp["busy_frac"] == pytest.approx(0.13 / 0.32, abs=1e-4)
    assert sp["prefill_busy_frac"] == pytest.approx(0.12 / 0.32, abs=1e-4)
    # decode token：第 2 步 2 个（10 ms）、第 3 步 2 个（20 ms，被 prefill 拖长）
    assert sp["decode_tokens"] == 4 and sp["svc_per_token_ms"] == pytest.approx(15.0)
    assert sp["decode_ms_by_batch"] == {"2": {"n": 1, "mean": 10.0, "p50": 10.0, "p95": 10.0, "max": 10.0}}
    assert fb.summarise_steps(steps, t0=0.15)["steps"] == 2
    pr = fb.prefill_rows([{"P": 512, "t_add": -0.001, "t_done": 0.1005}], steps)
    assert pr[0]["svc_ms"] == 100.0 and pr[0]["n_steps"] == 1 and pr[0]["lat_ms"] == pytest.approx(101.5)


def test_fixed_overhead_layer_difference() -> None:
    fx = fb.fixed_overhead({"1": 8.0, "4": 10.0}, {"1": 7.5, "4": 9.0}, {"1": 4.5, "4": 5.0}, layers=2)
    assert fx["1"] == {"step_ms": 8.0, "stats_ms": 0.5, "per_layer_ms": 3.0, "layers_ms": 6.0, "fixed_ms": 1.5}
    assert fx["4"]["per_layer_ms"] == 4.0 and fx["4"]["fixed_ms"] == 1.0
    assert fb.fixed_overhead({"1": 8.0}, None, None, layers=2) == {"1": {"step_ms": 8.0}}


# --------------------------------------------------------------------------- #
# 真调度器 + 假前段 runner：整套流程
# --------------------------------------------------------------------------- #
def test_bench_engine_on_real_scheduler() -> None:
    pytest.importorskip("torch")
    from tests.p2p.test_tiered import V, FakeCore, FrontRunner, make_sched
    from vllm.p2p.tiered import Engine

    configs = _configs()
    steps: list[dict] = []

    class Timed(FrontRunner):
        """每步「GPU」花 3 ms（阻塞事件循环，像真引擎一样别的都得等），记步级记录。"""

        def step(self, so):
            t, p = time.time(), time.perf_counter()
            out = super().step(so)
            time.sleep(0.003)
            nst = so.num_scheduled_tokens
            pf = [int(v) for v in nst.values() if int(v) > 1]
            steps.append({"t": t, "ms": (time.perf_counter() - p) * 1e3, "n": len(nst),
                          "npf": len(pf), "pf": sum(pf)})
            return out

    async def go():
        core = FakeCore(make_sched(configs["front"], 256), Timed(), seed=1)
        eng = Engine("front", "front", core)

        async def steps_fn():
            return list(steps)
        try:
            return await fb.bench_engine(
                eng, steps_fn, conc=[1, 4], backs=[(0.0, 0.0), (2.0, 6.0)], n_tokens=6,
                reqs_per_slot=2, prompt_len=16, prefill=[16, 48], repeats=2, prefill_bg=2, vocab=V,
                timeout=20, log=lambda s: None)
        finally:
            core.close()

    res = asyncio.run(go())
    d0 = res["decode"]["0"]
    c1, c4 = d0["1"], d0["4"]
    assert c1["tokens"] == 12 and c1["unmatched"] == 0
    assert c1["clean"]["batch_mean"] == pytest.approx(1.0)
    assert c1["clean"]["svc"]["mean"] >= 3.0
    assert c4["tokens"] == 48 and c4["unmatched"] == 0 and c4["clean"]["batch_mean"] > 1.5
    for v in (c1, c4):
        cl = v["clean"]
        assert cl["lat"]["mean"] == pytest.approx(cl["svc"]["mean"] + cl["wait"]["mean"], abs=0.01)
        assert cl["queue"]["mean"] > -0.5 and cl["out"]["mean"] >= -0.2
        assert 0 < v["engine"]["busy_frac"] <= 1
    assert set(res["decode"]) == {"0", "2:6"} and res["decode"]["2:6"]["4"]["unmatched"] == 0
    for P in ("16", "48"):
        assert res["prefill"][P]["svc"]["n"] == 2 and res["prefill"][P]["svc"]["mean"] >= 3.0
    pb = res["prefill_bg"]
    assert pb["bg"] == 2 and pb["decode"]["tokens"] > 0 and pb["prefill"]["48"]["svc"]["n"] == 2


def _configs():
    import os

    from tests.p2p import test_tiered as tt

    if not hasattr(_configs, "v"):
        os.environ["PYTHONPATH"] = f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
        from vllm.p2p.parity import PARITY_CFG
        import tempfile

        d = Path(tempfile.mkdtemp(prefix="fb_ckpt_"))
        (d / "config.json").write_text(json.dumps(dict(PARITY_CFG, torch_dtype="float16")))
        mc = ModelConfig(model=str(d), dtype="float16", seed=0, skip_tokenizer_init=True, max_model_len=256,
                         hf_overrides={"architectures": ["Qwen3MoePartialForCausalLM"],
                                       "p2p_stage": {"role": "front", "layers": [0, 2]}})
        sc = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=64, max_model_len=256,
                             enable_chunked_prefill=True, is_encoder_decoder=False, async_scheduling=False)
        cc = CacheConfig(block_size=16, gpu_memory_utilization=0.9, cache_dtype="auto",
                         enable_prefix_caching=False)
        _configs.v = {"front": VllmConfig(model_config=mc, scheduler_config=sc, cache_config=cc,
                                          device_config=DeviceConfig("cpu"))}
        assert tt.W == PARITY_CFG["hidden_size"]
    return _configs.v


def test_engine_core_trace_flush(tmp_path) -> None:
    from vllm.p2p.steptrace import StepTrace
    from vllm.v1.engine.core import EngineCore

    tr = StepTrace(tmp_path / "f.jsonl", flush_every=1000, flush_s=1e9)
    tr.begin("k", SimpleNamespace(num_scheduled_tokens={"a": 1}, total_num_scheduled_tokens=1))
    tr.end("k")
    assert not (tmp_path / "f.jsonl").exists()
    assert EngineCore.p2p_trace_flush(SimpleNamespace(_p2p_trace=tr)) == str(tmp_path / "f.jsonl")
    assert len((tmp_path / "f.jsonl").read_text().splitlines()) == 1
    assert EngineCore.p2p_trace_flush(SimpleNamespace(_p2p_trace=None)) is None


# --------------------------------------------------------------------------- #
# occ：真实运行的占用率
# --------------------------------------------------------------------------- #
def _occ_inputs():
    # 前段：窗口 [0, 1] 秒里 40 步 × 5 ms 纯 decode（批 3）+ 2 步 × 50 ms prefill → 忙 30%
    front = [{"t": 0.01 * k, "ms": 5.0, "n": 3, "npf": 0, "pf": 0} for k in range(40)]
    front += [{"t": 0.5, "ms": 50.0, "n": 1, "npf": 1, "pf": 300}, {"t": 0.7, "ms": 50.0, "n": 1, "npf": 1, "pf": 300}]
    back = [{"t": 0.02 * k, "ms": 15.0, "n": 2, "npf": 0, "pf": 0} for k in range(50)]
    rows = [{"n_generated": 11, "decode_ms": 400.0, "identify": {"compute_ms": 0.3},
             "engine": {"front": "front-N13", "back": "back-x-0", "front_engine_ms_p50": 8.0,
                        "back_engine_ms_p50": 20.0}},
            {"n_generated": 11, "decode_ms": 400.0, "identify": {"compute_ms": 0.5},
             "engine": {"front": "front-N13", "back": "back-x-0", "front_engine_ms_p50": 10.0,
                        "back_engine_ms_p50": 22.0}}]
    return {"requests": rows, "run": {"t0_wall": 0.0, "t1_wall": 1.0}}, {"front-N13": front, "back-x-0": back}


def test_occupancy() -> None:
    result, traces = _occ_inputs()
    occ = fb.occupancy(result, traces)
    f = occ["engines"]["front-N13"]
    assert f["role"] == "front" and f["requests"] == 2
    assert f["busy_frac"] == pytest.approx(0.30, abs=1e-3)
    assert f["prefill_busy_frac"] == pytest.approx(0.10, abs=1e-3)
    assert f["svc_per_token_ms"] == pytest.approx(5.0)
    assert f["queue_ms"] == pytest.approx(9.0 - 5.0)                  # 引擎内均值 9 − 服务 5
    assert f["queue_share_tpot"] == pytest.approx(4.0 / 40.0)          # TPOT = 400 / 10
    assert f["engine_share_tpot"] == pytest.approx(9.0 / 40.0)
    b = occ["engines"]["back-x-0"]
    assert b["role"] == "back" and b["queue_ms"] == pytest.approx(21.0 - 15.0)
    assert occ["selector_ms"]["n"] == 2 and occ["tpot"]["p50"] == 40.0
    # 没有 run.t0_wall：各引擎用自己记录的首尾；只有一个前段时名字对不上也算全部请求
    result2 = {"requests": [dict(r, engine=dict(r["engine"], front="front-other"))
                            for r in result["requests"]]}
    o2 = fb.occupancy(result2, traces)
    assert o2["window"] is None and o2["engines"]["front-N13"]["requests"] == 2


def test_front_occ_cli_through_layer_bench_file(tmp_path) -> None:
    """控制机上按文件路径跑 layer_bench.py front-occ（不 import vllm）。"""
    result, traces = _occ_inputs()
    (tmp_path / "r.json").write_text(json.dumps(result))
    for n, rows in traces.items():
        (tmp_path / f"{n}.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    code = ("import sys, runpy; sys.modules['vllm'] = None; sys.argv = ['layer_bench.py'] + sys.argv[1:];"
            f"runpy.run_path({str(ROOT / 'vllm/p2p/layer_bench.py')!r}, run_name='__main__')")
    p = subprocess.run([sys.executable, "-c", code, "front-occ", "--result", str(tmp_path / "r.json"),
                        "--steps", str(tmp_path / "front-N13.jsonl"), str(tmp_path / "back-x-0.jsonl"),
                        "--out", str(tmp_path / "occ.json")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "front-N13" in p.stdout and "30%" in p.stdout
    assert json.loads((tmp_path / "occ.json").read_text())["engines"]["front-N13"]["queue_ms"] == 4.0


def test_cluster_remote_command_passes_options() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    fb.add_run_args(ap, cluster=True)
    a = ap.parse_args(["--hosts", "h.txt", "--conc", "1,6", "--back-ms", "0", "--eager",
                       "--lr-model", "/x/sel.npz"])
    cmd = fb.remote_command("N13", a)
    assert cmd.startswith("cd /home/ubuntu/vllm-p2p && /home/ubuntu/venvs/vllm-p2p/bin/python -m vllm.p2p.front_bench run")
    for s in ("--node N13", "--out /tmp/front_bench_N13.json", "--conc 1,6", "--back-ms 0", "--eager",
              "--lr-model /x/sel.npz", "--prefill 128,512,1024"):
        assert s in cmd, s
    # 选项与 run 的解析器对得上
    rp = argparse.ArgumentParser()
    fb.add_run_args(rp)
    got = rp.parse_args(fb.run_argv(a, "N13", "/tmp/o.json")[1:])
    assert got.conc == "1,6" and got.eager and got.lr_model == "/x/sel.npz" and not got.real_weights


def test_print_run_smoke(capsys) -> None:
    cl = {"n": 3, "lat": {"n": 3, "mean": 9.0, "p95": 12.0}, "svc": {"n": 3, "mean": 7.0, "p95": 7.5},
          "wait": {"n": 3, "mean": 2.0, "p95": 4.5}, "queue": {"n": 3, "mean": 1.5, "p95": 4.0},
          "out": {"n": 3, "mean": 0.5, "p95": 0.6}, "batch_mean": 1.0}
    res = {"node": "N13", "gpu": "Tesla T4", "layers": 2, "cuda_graph": True, "dtype": "half",
           "engines": {"main": {"decode": {"0": {"1": {"clean": cl, "engine": {"busy_frac": 0.5},
                                                     "front_tok_per_s": 100}}},
                                "prefill": {"128": {"svc": {"n": 2, "mean": 60.0, "p95": 61.0},
                                                    "lat": {"n": 2, "mean": 62.0}, "steps": {"n": 2, "mean": 1}}}}},
           "fixed": {"decode": {"1": {"step_ms": 7.0, "layers_ms": 5.0, "fixed_ms": 1.5, "stats_ms": 0.5,
                                      "per_layer_ms": 2.5}}},
           "micro": {"1": {"embed_ms": 0.01, "hidden_d2h_ms": 0.02, "stats_d2h_ms": 0.02}},
           "selector": {"feats": 3, "ms": {"n": 60, "mean": 0.2, "p95": 0.3}}}
    fb.print_run(res)
    fb.print_compare([res, dict(res, node="1.5", gpu="NVIDIA A30")])
    out = capsys.readouterr().out
    assert "7.00/7.50" in out and "60.00/61.00" in out and "5.00" in out and "0.200/0.300" in out
    assert "N13·T4" in out and "1.5·A30" in out
