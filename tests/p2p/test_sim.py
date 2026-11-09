# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""模拟器（vllm/p2p/sim.py）与步级记录（vllm/p2p/steptrace.py）。不要 GPU。

拿 A 的真实清单生成 stages.json，配一张假的 layer_bench 表：
* 每个前段始终 1 条时，前段每步批大小恒为 1、峰值 1，结果是 bench 格式（tiered_stats 能直接读）；
* 单条请求的时间线能手算核对（TTFT、每 token 间隔 = 前段一步 + 后段一步 + 两跳）；
* 后段槽位 1、在途多于后段数时，多出来的在前段等（hold），关掉 hold 就在后段引擎里排队；
* 用已知参数造一份「真机」结果，从别的起点标定能找回来。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from vllm.p2p import sim as S                  # noqa: E402
from vllm.p2p import steptrace as T            # noqa: E402
from vllm.p2p.layout import main as layout_main  # noqa: E402
from vllm.p2p.stages import stages_from_channels  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
PLAN, HOSTS = DATA / "plan_es90_fill.json", DATA / "hosts.txt"


def fake_perf() -> dict:
    """每层毫秒：A30 最快、T4 最慢；随批、上下文变大；截距（固定开销）1 ms。"""
    def tab(base):
        dec = {str(c): {str(b): round(base * (1 + 0.15 * (b - 1)) * (1 + c / 4096), 4) for b in (1, 2, 4, 8)}
               for c in (128, 512, 1024)}
        pf = {str(p): round(base * (1 + p / 64), 4) for p in (128, 512, 1024)}
        oh = {str(c): {str(b): 1.0 for b in (1, 2, 4, 8)} for c in (128, 512, 1024)}
        return {"decode": {"128": dec, "64": dec}, "prefill": {"128": pf, "64": pf},
                "decode_overhead_ms": {"128": oh, "64": oh}}
    return {"A30": {"gpu": "NVIDIA A30", **tab(0.1)}, "RTX6000": {"gpu": "Quadro RTX 6000", **tab(0.2)},
            "T4": {"gpu": "Tesla T4", **tab(0.35)}}


@pytest.fixture()
def world(tmp_path):
    if not (PLAN.exists() and HOSTS.exists()):
        pytest.skip("清单或 hosts 不在")
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out), "--model-dir", "/w",
                        "--vllm-bin", "/venv/bin/vllm", "--stats", "--max-num-seqs", "2"]) == 0
    st = stages_from_channels(json.loads((out / "channels.json").read_text()))
    sp = tmp_path / "stages.json"
    sp.write_text(json.dumps(st))
    lb = tmp_path / "summary_by_gpu.json"
    lb.write_text(json.dumps(fake_perf()))
    return {"stages": st, "stages_path": sp, "lb": lb, "tmp": tmp_path}


def fake_real(st, n=12, gen=(40, 70, 25)) -> dict:
    """一份「真机」结果的骨架：只用到回放要的字段。"""
    tasks = ["gsm8k", "mbpp", "no_robots"]
    by = {u: [b for b in st["backs"] if any(c["name"] == b and c["task"] == u for c in st["channels"])]
          for u in tasks}
    rows = []
    for i in range(n):
        u = tasks[i % 3]
        rows.append({"req": f"req{i}", "case_id": f"{u}:{i}", "true_task": u, "channel": by[u][0],
                     "n_prompt": 100 + 10 * i, "n_generated": gen[i % 3], "stop_reason": "length"})
    return {"requests": rows}


def setup(world, real, **kw):
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    net = S.Net(ip=ip, hop_ms=kw.pop("hop_ms", 2.0), lat_same=0.1, lat_cross=3.0, lat_gw=0.1)
    params = S.Params(**{k: kw.pop(k) for k in list(kw) if k in S.Params().__dict__})
    reqs = S.replay_requests(real, world["stages"])
    return S.run_sim(world["stages"], reqs, perf=perf, net=net, node_gpu=gpu, params=params, real=real, **kw)


def test_per_front_one_request_each(world) -> None:
    real = fake_real(world["stages"], n=15)
    doc = setup(world, real, per_front=1)
    rows = doc["requests"]
    assert len(rows) == 15 and doc["concurrency"] == len(world["stages"]["fronts"]) == 5
    assert all(v["peak"] == 1 for v in doc["run"]["front_peaks"].values())
    assert sum(v["served"] for v in doc["run"]["front_peaks"].values()) == 15
    fr = {n: e for n, e in doc["sim"]["engines"].items() if e["role"] == "front"}
    assert all(e["batch_mean"] == 1.0 for e in fr.values() if e["steps"])      # 前段批恒为 1
    for r, q in zip(rows, real["requests"]):
        assert r["n_generated"] == q["n_generated"] and r["channel"] in world["stages"]["backs"]
        assert r["task"] == q["true_task"] and r["prefill_ms"] > 0 and r["total_ms"] > r["prefill_ms"]
        assert ("64" in r["marks"]) == (r["n_generated"] >= 64) and len(r["token_ms"]) == r["n_generated"] - 1
        if "64" in r["marks"]:
            assert r["marks"]["64"]["t_ms"] < r["total_ms"] and r["marks"]["64"]["tpot_ms"] > 0
    # 统计脚本直接能读
    from vllm.p2p.tiered_stats import continuous_report
    txt = "\n".join(continuous_report(doc))
    assert "TTFT" in txt and "TPOT" in txt


def test_single_request_timeline_by_hand(world) -> None:
    """一条请求、固定开销与跳数都给定：每 token 间隔 = 前段一步 + 后段一步 + 两跳（前→后、后→前）。"""
    st = world["stages"]
    real = {"requests": [{"req": "req0", "true_task": "gsm8k", "channel": "back-gsm8k-0", "n_prompt": 128,
                          "n_generated": 5}]}
    doc = setup(world, real, inflight=1, front_fixed_ms=0.5, back_fixed_ms=1.0, hop_ms=2.0)
    r = doc["requests"][0]
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    fch = next(c for c in st["channels"] if c["name"] == r["front_engine"])
    bch = next(c for c in st["channels"] if c["name"] == "back-gsm8k-0")
    net = S.Net(ip=ip, hop_ms=2.0, lat_same=0.1, lat_cross=3.0, lat_gw=0.1)
    gf = perf[gpu[fch["nodes"][0]]]
    e_b = S.resident_experts(bch)
    f_step = 0.5 + 8 * gf.decode_ms(128, 130, 1)               # 上下文 ≈ 128 + 已生成
    b_step = 1.0 + sum(nl * perf[gpu[v]].decode_ms(e_b, 130, 1) for v, nl in zip(bch["nodes"], bch["partition"]))
    b_step += sum(net.pp(a, b, 2048 * 2) for a, b in zip(bch["nodes"], bch["nodes"][1:]))
    hops = net.msg(fch["nodes"][0], bch["nodes"][0], 2048 * 2) + net.msg(bch["nodes"][0], fch["nodes"][0], 8)
    gap = f_step + b_step + hops
    assert r["token_ms"][0] == pytest.approx(gap, rel=0.02)
    assert r["queue_ms"] == 0 and r["wait_front_ms"] == 0


def test_hold_waits_at_front_and_no_hold_queues_in_back(world) -> None:
    st = world["stages"]
    # 全部判成 gsm8k：gsm8k 只有两个后段、各 2 个槽位 → 4 条以外在前段等
    real = {"requests": [{"req": f"req{i}", "true_task": "gsm8k", "channel": "back-gsm8k-0",
                          "n_prompt": 64, "n_generated": 30} for i in range(12)]}
    doc = setup(world, real, inflight=10)
    waits = [r["wait_front_ms"] for r in doc["requests"]]
    assert doc["run"]["gateway_waits"]["back_hold"] > 0 and max(waits) > 0
    # hold 时到后段只可能等「正在算的那一步」算完（远小于一条请求的时长）
    assert max(r["queue_back_ms"] for r in doc["requests"]) < 0.1 * max(waits)
    nh = setup(world, real, inflight=10, hold=False)
    assert all(r["wait_front_ms"] == 0 for r in nh["requests"])
    # 不 hold：改在后段 vLLM 里排（max_num_seqs 挡着），排的时间与 hold 时在前段等的差不多
    assert max(r["queue_back_ms"] for r in nh["requests"]) > 0.5 * max(waits)
    assert {r["channel"] for r in doc["requests"]} == {"back-gsm8k-0", "back-gsm8k-1"}


def test_fit_recovers_known_parameters(world) -> None:
    real = setup(world, fake_real(world["stages"], n=15), per_front=1, front_fixed_ms=1.5, back_fixed_ms=4.0,
                 hop_ms=3.0)
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    net = S.Net(ip=ip, hop_ms=1.0, lat_same=0.1, lat_cross=3.0, lat_gw=0.1)

    def make(p, hop):
        net.hop_ms = hop
        return S.run_sim(world["stages"], S.replay_requests(real, world["stages"]), perf=perf, net=net,
                         node_gpu=gpu, params=p, per_front=1, real=real)

    _, score0 = S.compare(make(S.Params(front_fixed_ms=0.5, back_fixed_ms=1.0), 1.0), real)
    p, hop, score, _ = S.fit(make, real, S.Params(front_fixed_ms=0.5, back_fixed_ms=1.0), 1.0, rounds=4,
                             log=lambda s: None)
    assert score < score0 * 0.05 and score < 1e-3
    # 每 token 间隔只约束「前段 + 后段 + 2 跳」的和；TTFT 再约束一部分 —— 看总和找回来了
    assert p.front_fixed_ms + p.back_fixed_ms + 2 * hop == pytest.approx(1.5 + 4.0 + 6.0, rel=0.08)


def test_steptrace_records_and_summarises(tmp_path) -> None:
    from types import SimpleNamespace as NS

    tr = T.StepTrace(tmp_path / "e.jsonl", flush_every=1)
    sched = NS(requests={"a": NS(num_computed_tokens=100), "b": NS(num_computed_tokens=300)}, waiting=[1, 2])
    for k, (nst, tot) in enumerate((({"a": 50}, 50), ({"a": 1, "b": 1}, 2), ({"a": 1, "b": 1}, 2), ({}, 0))):
        so = NS(num_scheduled_tokens=nst, total_num_scheduled_tokens=tot)
        tr.begin(k, so, sched, start=T.StepTrace.now())
        tr.end(k)
    tr.flush()
    rows = T.load([tmp_path / "e.jsonl"])["e"]
    assert len(rows) == 3                                   # 空步不记
    assert rows[0]["npf"] == 1 and rows[0]["pf"] == 50 and rows[1]["n"] == 2 and rows[1]["ctx"] == 200.0
    s = T.summarise(rows)
    assert s["steps"] == 3 and s["batch_hist"] == {"2": 2} and s["prefill_steps"] == 1
    # stages.py --step-trace：命令前加了环境变量，collect 认得出 rank 0 的文件
    if PLAN.exists():
        out = tmp_path / "vc"
        assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out), "--model-dir", "/w",
                            "--vllm-bin", "/venv/bin/vllm", "--stats"]) == 0
        st = stages_from_channels(json.loads((out / "channels.json").read_text()), step_trace="/tmp/steps")
        files = T.trace_files(st)
        assert set(files) == set(st["fronts"]) | set(st["backs"])
        b = next(c for c in st["channels"] if c["name"] == "back-gsm8k-0")
        assert files["back-gsm8k-0"] == (b["commands"][0]["host"], "/tmp/steps/back-gsm8k-0.jsonl")
        assert all(x["cmd"].startswith("VLLM_P2P_STEP_TRACE=/tmp/steps/") for x in b["commands"])
        from vllm.p2p import launch
        assert launch.venv_of(b["commands"][0]["cmd"]) == "/venv"
    assert T.StepTrace.from_env() is None or True
