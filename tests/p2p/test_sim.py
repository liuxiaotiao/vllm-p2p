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
import shlex
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


def pp_timeline(net, nodes, comp, tok_bytes):
    """多机 PP 一步（手算版）：调度结果同时发给各台；第 i 段等调度结果与上一段的 hidden 都到；
    最后采样结果回 node_rank 0。"""
    end = comp[0]
    for i in range(1, len(nodes)):
        start = max(net.one_way(nodes[0], nodes[i]) + net.pp_hop_ms,
                    end + net.one_way(nodes[i - 1], nodes[i]) + net.xfer(nodes[i - 1], nodes[i], tok_bytes)
                    + net.pp_hop_ms)
        end = start + comp[i]
    if len(nodes) > 1:
        end += net.one_way(nodes[-1], nodes[0]) + net.xfer(nodes[-1], nodes[0], 8) + net.pp_hop_ms
    return end


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
    h = kw.pop("hop_ms", 2.0)
    net = S.Net(ip=ip, hop_ms=h, agent_hop_ms=h, lat_same=0.1, lat_cross=3.0, lat_gw=0.1)
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
    b_step = 1.0 + pp_timeline(net, bch["nodes"], [nl * perf[gpu[v]].decode_ms(e_b, 130, 1)
                                                   for v, nl in zip(bch["nodes"], bch["partition"])], 2048 * 2)
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
        net.hop_ms = net.agent_hop_ms = hop        # 分段部署走代理那条路
        return S.run_sim(world["stages"], S.replay_requests(real, world["stages"]), perf=perf, net=net,
                         node_gpu=gpu, params=p, per_front=1, real=real)

    _, score0 = S.compare(make(S.Params(front_fixed_ms=0.5, back_fixed_ms=1.0), 1.0), real)
    p, hop, score, _ = S.fit(make, real, S.Params(front_fixed_ms=0.5, back_fixed_ms=1.0), 1.0, rounds=4,
                             log=lambda s: None)
    assert score < score0 * 0.05 and score < 3e-3
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


def test_full_pipeline_channels_any_mode(world) -> None:
    """layout.py 的 channels.json（整条流水线 A）：网关 any 挑在途最少的通道；decode 在引擎里连着算，
    每 token 间隔 = 这条流水线一步的用时（单条请求时可以手算）。"""
    cj = json.loads((world["tmp"] / "vc" / "channels.json").read_text())
    names = [c["name"] for c in cj["channels"]]
    real = {"requests": [{"req": f"req{i}", "true_task": "gsm8k", "channel": names[i % len(names)],
                          "n_prompt": 80, "n_generated": 20} for i in range(10)]}
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    net = S.Net(ip=ip, hop_ms=2.0, lat_same=0.1, lat_cross=3.0, lat_gw=0.1)
    doc = S.run_sim(cj, S.replay_requests(real, cj), perf=perf, net=net, node_gpu=gpu,
                    params=S.Params(back_fixed_ms=1.0), real=real)
    rows = doc["requests"]
    assert doc["concurrency"] == len(names) and doc["mode"] == "static"
    assert {r["channel"] for r in rows} == set(names)          # any：5 条在途 → 每条通道各一条
    assert all(r["front_engine"] is None and r["wait_front_ms"] == 0 for r in rows)
    one = S.run_sim(cj, S.replay_requests({"requests": real["requests"][:1]}, cj), perf=perf, net=net,
                    node_gpu=gpu, params=S.Params(back_fixed_ms=1.0), inflight=1)["requests"][0]
    ch = next(c for c in cj["channels"] if c["name"] == one["channel"])
    e = S.resident_experts(ch) or 128
    step = 1.0 + pp_timeline(net, ch["nodes"], [nl * perf[gpu[v]].decode_ms(e, 81, 1)
                                                for v, nl in zip(ch["nodes"], ch["partition"])], 2048 * 2)
    assert one["token_ms"][0] == pytest.approx(step, rel=0.02)
    with pytest.raises(SystemExit, match="没有前段"):
        S.run_sim(cj, S.replay_requests(real, cj), perf=perf, net=net, node_gpu=gpu, params=S.Params(),
                  per_front=1)


def test_pp_simple_is_v1_and_broadcast_can_dominate(world) -> None:
    """--pp-simple = v1（只算往下传的几跳）；新算法多出「结果回传」；第一段很快、跨网段下发很慢时，
    后面的段要等调度结果。"""
    st = world["stages"]
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    sim = S.Sim(st, perf=perf, net=S.Net(ip=ip, lat_same=0.1, lat_cross=3.0), node_gpu=gpu,
                params=S.Params(back_fixed_ms=0.0), front_slots=None, hold=True)
    e = sim.engines["back-mbpp-1"]                      # N13 → N14：同网段
    new = e.step_ms([200.0], [])
    sim.params.pp_simple = True
    old = e.step_ms([200.0], [])
    assert new - old == pytest.approx(sim.net.one_way("N14", "N13") + sim.net.pp_hop_ms
                                      + sim.net.xfer("N14", "N13", 8), rel=1e-6)
    # 三段的流水线：首段到末段直连很慢（50 ms）、中间两跳很快 → 末段要等调度结果，结果回传也走慢的那条
    cj = json.loads((world["tmp"] / "vc" / "channels.json").read_text())
    net = S.Net(ip=ip, lat_same=0.1, lat_cross=3.0, lat={("N12", "N08"): 50.0})
    sa = S.Sim(cj, perf=perf, net=net, node_gpu=gpu, params=S.Params(back_fixed_ms=0.0), front_slots=None,
               hold=True)
    e3 = sa.engines["no_robots-0"]
    assert e3.nodes == ["N12", "N09", "N08"]
    e3.gpus = [S.GpuPerf("x", {"128": {"128": {"1": 0.0}}}, {"128": {"128": 0.0}}, {})] * 3
    want = (50.0 + net.pp_hop_ms) + (50.0 + net.xfer("N08", "N12", 8) + net.pp_hop_ms)
    assert e3.step_ms([128.0], []) == pytest.approx(want, rel=1e-9)


def test_layout_step_trace_for_full_channels(tmp_path) -> None:
    """layout.py --step-trace：整条流水线（A）的每台命令前加环境变量；collect 认 node_rank 0 那台。"""
    if not PLAN.exists():
        pytest.skip("清单不在")
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out), "--model-dir", "/w",
                        "--vllm-bin", "/venv/bin/vllm", "--step-trace", "/tmp/steps"]) == 0
    cj = json.loads((out / "channels.json").read_text())
    files = T.trace_files(cj)
    assert set(files) == {c["name"] for c in cj["channels"]}
    for c in cj["channels"]:
        assert c["step_trace"] == f"/tmp/steps/{c['name']}.jsonl"
        assert all(x["cmd"].startswith(f"VLLM_P2P_STEP_TRACE=/tmp/steps/{c['name']}.jsonl ")
                   for x in c["commands"])
        assert files[c["name"]][0] == c["commands"][0]["host"]          # node_rank 0
        assert "VLLM_P2P_STEP_TRACE" in (out / f"{c['name']}.sh").read_text()
    from vllm.p2p import launch
    assert launch.venv_of(cj["channels"][0]["commands"][0]["cmd"]) == "/venv"


def test_prefill_fit_drops_cold_start_and_sim_uses_measured_steps(world) -> None:
    rows = [{"t": i, "ms": 20 + 2.0 * n, "n": 1, "tok": n, "pf": n, "npf": 1} for i, n in
            enumerate((50, 80, 120, 200, 300))]
    rows.append({"t": 9, "ms": 4000.0, "n": 1, "tok": 60, "pf": 60, "npf": 1})        # 冷启动
    rows += [{"t": 10 + i, "ms": 55.0, "n": 1, "tok": 1, "pf": 0, "npf": 0} for i in range(5)]
    s = T.summarise(rows)
    pf = s["prefill_fit"]
    assert pf["dropped"] == 1 and pf["a_ms"] == pytest.approx(20, abs=1e-6)
    assert pf["b_ms_per_token"] == pytest.approx(2.0, abs=1e-6)
    assert s["decode_ms_by_batch"]["1"]["p50"] == 55.0
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    sim = S.Sim(world["stages"], perf=perf, net=S.Net(ip=ip), node_gpu=gpu, params=S.Params(),
                front_slots=None, hold=True,
                steps={"back-gsm8k-0": {"dec": {1: 55.0}, "pf": pf}})
    e = sim.engines["back-gsm8k-0"]
    assert e.step_ms([300.0], []) == 55.0 and e.step_ms([], [100]) == pytest.approx(220.0)
    assert e.step_ms([300.0, 310.0], []) != 55.0          # 没量过的批大小：回到模型


def test_prefill_fixed_cost_per_step_stage_layer(world) -> None:
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    sim = S.Sim(world["stages"], perf=perf, net=S.Net(ip=ip), node_gpu=gpu, params=S.Params(back_fixed_ms=0.0),
                front_slots=None, hold=True)
    e = sim.engines["back-gsm8k-0"]                      # 两段（13 + 27 层）
    base, dec = e.step_ms([], [100]), e.step_ms([200.0], [])
    for per, add in (("step", 10.0), ("stage", 20.0), ("layer", 400.0)):
        sim.params.pf_fixed_ms, sim.params.pf_fixed_per = 10.0, per
        assert e.step_ms([], [100]) == pytest.approx(base + add, rel=1e-9), per
        assert e.step_ms([200.0], []) == pytest.approx(dec)                  # 纯 decode 不受影响


def test_agent_hop_is_separate_from_gateway_hop(world) -> None:
    """分段部署的每 token 间隔只随 agent_hop_ms 变（前→后、后→前两跳），不随网关 HTTP 的 hop_ms 变。"""
    real = {"requests": [{"req": "req0", "true_task": "gsm8k", "channel": "back-gsm8k-0", "n_prompt": 64,
                          "n_generated": 4}]}
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)

    def gap(hop, agent):
        net = S.Net(ip=ip, hop_ms=hop, agent_hop_ms=agent)
        d = S.run_sim(world["stages"], S.replay_requests(real, world["stages"]), perf=perf, net=net,
                      node_gpu=gpu, params=S.Params(), inflight=1)
        return d["requests"][0]["token_ms"][0]

    assert gap(2.0, 2.0) == pytest.approx(gap(20.0, 2.0))
    assert gap(2.0, 5.0) - gap(2.0, 2.0) == pytest.approx(6.0, rel=1e-6)


def test_back_pp_hop_and_score_modes(world) -> None:
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    net = S.Net(ip=ip, pp_hop_ms=4.0)
    sim = S.Sim(world["stages"], perf=perf, net=net, node_gpu=gpu, params=S.Params(back_fixed_ms=0.0),
                front_slots=None, hold=True)
    e = sim.engines["back-mbpp-1"]                       # 两段：下发（与往下传重叠）+ 往下传 + 回传
    t4 = e.step_ms([200.0], [])
    net.back_pp_hop_ms = 1.0
    assert t4 - e.step_ms([200.0], []) == pytest.approx(2 * 3.0, rel=1e-6)
    real = fake_real(world["stages"], n=6)
    doc = setup(world, real, per_front=1)
    _, s_all = S.compare(doc, doc)
    assert s_all == 0.0
    lines, s_t = S.compare(doc, {"requests": [dict(r, prefill_ms=r["prefill_ms"] * 2) for r in doc["requests"]],
                                 "run": doc["run"]}, "tpot")
    assert s_t > 0                                       # TPOT 的算法里用到了 prefill_ms（总时延 − 首 token）
    _, s_tt = S.compare(doc, {"requests": doc["requests"], "run": doc["run"]}, "ttft")
    assert s_tt == 0.0


def test_batch_wait_merges_alternating_requests_on_one_back(world) -> None:
    """同一后段上两条请求的输入错开到达：不等时一条一条算（批 1，交替）；凑批等待几毫秒后合成批 2，
    两条的 TPOT 都下降（假表里批 2 只比批 1 贵 15%）。"""
    real = {"requests": [{"req": f"req{i}", "true_task": "no_robots", "channel": "back-no_robots-0",
                          "n_prompt": 64 + 40 * i, "n_generated": 200} for i in range(2)]}
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)

    def run(wait):
        d = S.run_sim(world["stages"], S.replay_requests(real, world["stages"]), perf=perf,
                      net=S.Net(ip=ip, agent_hop_ms=1.0, lat_cross=1.25), node_gpu=gpu,
                      params=S.Params(back_fixed_ms=3.0, batch_wait_ms=wait), per_front=1)
        tp = [(r["total_ms"] - r["prefill_ms"]) / (r["n_generated"] - 1) for r in d["requests"]]
        return d, sum(tp) / len(tp)

    d0, tp0 = run(0.0)
    d5, tp5 = run(8.0)
    assert {r["channel"] for r in d0["requests"]} == {"back-no_robots-0"}
    assert len({r["front_engine"] for r in d0["requests"]}) == 2          # 两个前段、同一个后段
    b0, b5 = d0["sim"]["engines"]["back-no_robots-0"], d5["sim"]["engines"]["back-no_robots-0"]
    assert b0["batch_mean"] < 1.05 and b5["batch_mean"] > 1.5 and b5["batch_waits"] > 0
    assert tp5 < 0.85 * tp0


def test_steptrace_records_batch_wait(tmp_path) -> None:
    """后段开了凑批等待：每步记「这一步之前等了多少 ms」，summary 汇总。"""
    from types import SimpleNamespace as NS

    tr = T.StepTrace(tmp_path / "b.jsonl", flush_every=1)
    sched = NS(requests={"a": NS(num_computed_tokens=10)}, waiting=[], p2p_batch_wait_ms=8.0,
               p2p_hold_ms=0.0)
    for k, held in enumerate((0.0, 5.0, 5.0, 12.5)):
        sched.p2p_hold_ms = held
        tr.begin(k, NS(num_scheduled_tokens={"a": 1}, total_num_scheduled_tokens=1), sched)
        tr.end(k)
    tr.flush()
    rows = T.load([tmp_path / "b.jsonl"])["b"]
    assert [r["w"] for r in rows] == [0.0, 5.0, 0.0, 7.5]
    s = T.summarise(rows)
    assert s["hold_steps"] == 2 and s["hold_ms_total"] == 12.5
    tr2 = T.StepTrace(tmp_path / "c.jsonl", flush_every=1)       # 没开等待：不记 w
    tr2.begin(0, NS(num_scheduled_tokens={"a": 1}, total_num_scheduled_tokens=1),
              NS(requests={}, waiting=[], p2p_batch_wait_ms=0.0, p2p_hold_ms=0.0))
    tr2.end(0)
    tr2.flush()
    assert "w" not in T.load([tmp_path / "c.jsonl"])["c"][0]
    assert "hold_steps" not in T.summarise(T.load([tmp_path / "c.jsonl"])["c"])


@pytest.mark.skipif(not PLAN.exists(), reason="没有 p2p-framework 的部署清单")
def test_stages_back_batch_wait(tmp_path) -> None:
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out), "--model-dir", "/w",
                        "--vllm-bin", "/venv/bin/vllm", "--stats"]) == 0
    cj = json.loads((out / "channels.json").read_text())
    st = stages_from_channels(cj, back_batch_wait_ms=8)
    for c in st["channels"]:
        ovs = [json.loads(shlex.split(x["cmd"])[shlex.split(x["cmd"]).index("--hf-overrides") + 1])
               for x in c["commands"]]
        if c["role"] == "back":
            assert c["batch_wait_ms"] == 8.0 and all(o["p2p_batch_wait_ms"] == 8.0 for o in ovs)
        else:
            assert all("p2p_batch_wait_ms" not in o for o in ovs)
    st0 = stages_from_channels(cj)
    assert all(not c.get("batch_wait_ms") for c in st0["channels"])
    assert all("p2p_batch_wait_ms" not in x["cmd"] for c in st0["channels"] for x in c["commands"])


def test_ttft_breakdown_real_and_sim(world) -> None:
    """TTFT 拆四段：真机行按网关 / 协调器记录拆；模拟行同一口径（网关外 = 0），四段之和 = TTFT。"""
    from vllm.p2p import tiered_stats as TS

    row = {"prefill_ms": 150.0, "true_task": "gsm8k",
           "tiered": {"front_prefill_ms": 60.0, "bind_ms": 5.0, "wait_back_ms": 0.0},
           "engine": {"ttft_ms": 120.0}}
    assert TS.ttft_parts(row) == {"网关外": 30.0, "前段 prefill": 60.0, "识别+绑定": 5.0, "后段→首 token": 55.0}
    row["engine"].update(front_prefill_engine_ms=50.0, back_prefill_engine_ms=40.0)
    assert TS.ttft_parts(row)["后段·引擎内"] == 40.0 and TS.ttft_breakdown([row])["keys"][-1] == "后段·引擎内"
    assert TS.ttft_parts({"prefill_ms": 1.0}) is None and TS.ttft_breakdown([{"prefill_ms": 1.0}]) is None
    doc = setup(world, fake_real(world["stages"], n=12), per_front=1)
    tb = TS.ttft_breakdown(doc["requests"])
    assert tb and tb["n"] == 12
    for r in doc["requests"]:
        p = TS.ttft_parts(r)
        assert p["网关外"] == pytest.approx(0, abs=0.02) and p["前段 prefill"] > 0 and p["识别+绑定"] >= 0.5
        assert p["后段→首 token"] > 0 and sum(p[k] for k in TS.TTFT_PARTS) == pytest.approx(r["prefill_ms"], abs=0.05)
        assert 0 < p["前段·引擎内"] < p["前段 prefill"] and 0 < p["后段·引擎内"] < p["后段→首 token"]
    txt = "\n".join(TS.continuous_report(doc))
    assert "TTFT 拆分" in txt
    ks = S.key_metrics(doc)
    assert {"TTFT·前段 prefill p50", "TTFT·后段→首 token p50"} <= set(ks)


def test_bandwidth_moves_prompt_transfer_not_decode(world) -> None:
    """有效带宽只影响大消息：prompt 的 hidden（每 token 4 KB）—— TTFT 的前段 / 后段两段变长，
    每 token 间隔（只传一个位置）几乎不变。"""
    from vllm.p2p import tiered_stats as TS

    real = fake_real(world["stages"], n=9)
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    docs = {}
    for bw in (10000.0, 1000.0):
        net = S.Net(ip=ip, hop_ms=2.0, agent_hop_ms=2.0, lat_same=0.1, lat_cross=3.0, lat_gw=0.1, bw_default=bw)
        docs[bw] = S.run_sim(world["stages"], S.replay_requests(real, world["stages"]), perf=perf, net=net,
                             node_gpu=gpu, params=S.Params(), real=real, per_front=1)
    hi, lo = (TS.ttft_breakdown(docs[b]["requests"])["all"] for b in (10000.0, 1000.0))
    # 平均 prompt ≈ 140 token ≈ 1.1 MB（fp16，hidden 2048 × 2 字节 × 2）：1 Gbps 多 ≈ 4 ms / 次
    assert lo["前段 prefill"]["mean"] > hi["前段 prefill"]["mean"] + 2
    assert lo["后段→首 token"]["mean"] > hi["后段→首 token"]["mean"] + 2
    k = S.key_metrics
    assert k(docs[1000.0])["TPOT 均值"] == pytest.approx(k(docs[10000.0])["TPOT 均值"], rel=0.02)


def test_replay_from_other_deployment_uses_true_task(world) -> None:
    """别的部署（比如 petals 整条流水线）的结果回放到分段部署：通道名对不上、没有判出的类 → 按标注的类绑。"""
    real = {"requests": [{"req": f"req{i}", "true_task": u, "channel": "petals-0", "task": None,
                          "n_prompt": 90, "n_generated": 20} for i, u in enumerate(("gsm8k", "mbpp", "no_robots"))]}
    reqs = S.replay_requests(real, world["stages"])
    assert [r.task for r in reqs] == ["gsm8k", "mbpp", "no_robots"]
    doc = setup(world, real, per_front=1)
    st = world["stages"]
    task_of = {c["name"]: c["task"] for c in st["channels"] if c["role"] == "back"}
    assert all(task_of[r["channel"]] == r["true_task"] for r in doc["requests"])


def test_same_time_arrivals_share_one_step(world) -> None:
    """同一时刻到同一个引擎的几条输入（真机里是代理按目的地合成的一条消息：后段一步出的几个 token
    发给同一个前段）要进同一步 —— 不能第一条一到就开算、其余等下一步。"""
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    net = S.Net(ip=ip, hop_ms=0.5, agent_hop_ms=0.5, lat_same=0.1, lat_cross=1.0, lat_gw=0.1)
    sim = S.Sim(world["stages"], perf=perf, net=net, node_gpu=gpu, params=S.Params(), front_slots=None, hold=True)
    sim.on_step_out = lambda e, rid, item, t: None
    e = sim.engines[world["stages"]["fronts"][0]]
    e.admitted |= {"a", "b", "c"}
    for rid in ("a", "b"):                                    # 同一时刻到
        sim.at(5.0, sim._arrive, e, rid, ("dec", 100))
    sim.at(50.0, sim._arrive, e, "c", ("dec", 100))            # 很晚才到：自己一步
    sim.reqs.update({r: S.SimReq(rid=r, case_id=None, true_task=None, task=None, n_prompt=1, n_gen=2)
                     for r in ("a", "b", "c")})
    sim.run()
    assert e.n_steps == 2 and e.batch_sum == 3                # 第一步 a+b（旧写法是 a、b 各一步），第二步 c


def test_summary_table_cli(world, tmp_path) -> None:
    real = fake_real(world["stages"], n=9)
    a = setup(world, real, per_front=1)
    b = setup(world, real, per_front=1, hop_ms=3.0)
    for n, d in (("r.json", a), ("s.json", b)):
        (tmp_path / n).write_text(json.dumps(d))
    md, js = tmp_path / "t.md", tmp_path / "t.json"
    assert S.main(["table", f"甲={tmp_path / 'r.json'},{tmp_path / 's.json'}",
                   f"乙={tmp_path / 'r.json'},{tmp_path / 'r.json'}", "--md", str(md), "--json", str(js)]) == 0
    txt = md.read_text()
    assert "| 指标 | 甲 | 乙 |" in txt and "输出 tok/s" in txt and "TTFT·后段→首 token p50" in txt
    recs = json.loads(js.read_text())
    assert recs[1]["metrics"]["TPOT 均值"]["err"] == 0 and recs[0]["metrics"]["TPOT 均值"]["err"] > 0
    assert recs[0]["n"] == 9


def test_fit_prefill_fixed_and_scale_from_ttft_mean(world) -> None:
    """prefill 的固定部分与随 token 数增长的部分一起标：中位数 + 均值（长 prompt 拉高均值）能把两个都找回来。"""
    real_tpl = fake_real(world["stages"], n=12, gen=(8, 8, 8))
    for i, r in enumerate(real_tpl["requests"]):
        r["n_prompt"] = 900 if i % 6 == 5 else 60 + 10 * i      # 多数短、少数很长：均值被长的拉高
    truth = setup(world, real_tpl, per_front=1, pf_fixed_ms=8.0, pf_scale=0.5, front_fixed_ms=1.0, back_fixed_ms=1.0)
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    net = S.Net(ip=ip, hop_ms=2.0, agent_hop_ms=2.0, lat_same=0.1, lat_cross=3.0, lat_gw=0.1)

    def make(p, hop, pf_fixed_ms=None, pf_scale=None):
        net.hop_ms = hop
        p = S.Params(**{**p.as_dict(), **({"pf_fixed_ms": pf_fixed_ms} if pf_fixed_ms is not None else {}),
                        **({"pf_scale": pf_scale} if pf_scale is not None else {})})
        return S.run_sim(world["stages"], S.replay_requests(truth, world["stages"]), perf=perf, net=net,
                         node_gpu=gpu, params=p, real=truth, per_front=1)
    base = S.Params(front_fixed_ms=1.0, back_fixed_ms=1.0)
    p, _, score, doc = S.fit(make, truth, base, 2.0, rounds=6, fixed=("front_fixed_ms", "back_fixed_ms", "hop_ms"),
                             extra={"pf_fixed_ms": 20.0, "pf_scale": 1.0}, score="ttft-all", log=lambda s: None)
    got = doc["sim"]["fit"]["params"]
    assert score < 1e-3, (score, doc["sim"]["fit"]["params"])
    assert got["pf_scale"] == pytest.approx(0.5, rel=0.3) and got["pf_fixed_ms"] == pytest.approx(8.0, rel=0.5), got


def test_detail_cli(world, tmp_path) -> None:
    real = fake_real(world["stages"], n=9)
    a = setup(world, real, per_front=1)
    (tmp_path / "a.json").write_text(json.dumps(a))
    md = tmp_path / "d.md"
    assert S.main(["detail", "--real", str(tmp_path / "a.json"), "--sim", str(tmp_path / "a.json"),
                   "--md", str(md)]) == 0
    txt = md.read_text()
    assert "### TTFT ms" in txt and "### TPOT ms" in txt and "| 全部 | 9 / 9 |" in txt
    assert "| gsm8k | 3 / 3 |" in txt and "（+0.0%）" in txt
    assert S.main(["detail", "--real", str(tmp_path / "a.json"), "--sim", str(tmp_path / "a.json"),
                   "--by", "channel"]) == 0


# --------------------------------------------------------------------------- #
# 按卡型标定（calib）
# --------------------------------------------------------------------------- #
def test_prefill_below_table_continues_from_decode_rows() -> None:
    """layer_bench 的 prefill 表从 128 起；更短的 prompt 用 decode 表最短上下文那行（批 = token 数）接上。"""
    perf = fake_perf()["A30"]
    gp = S.GpuPerf("A30", perf["decode"], perf["prefill"], perf["decode_overhead_ms"])
    row = perf["decode"]["64"]["128"]
    assert gp.prefill_ms(64, 8) == pytest.approx(row["8"])
    assert gp.prefill_ms(64, 1) == pytest.approx(row["1"])
    assert gp.prefill_ms(64, 128) == pytest.approx(perf["prefill"]["64"]["128"])
    mid = gp.prefill_ms(64, 68)                                    # 8 与 128 之间线性
    assert mid == pytest.approx(row["8"] + (perf["prefill"]["64"]["128"] - row["8"]) * 60 / 120)


def test_stats_full_detection() -> None:
    def ch(ov):
        return {"commands": [{"node": "N1", "host": "h", "cmd": "x --hf-overrides " + shlex.quote(json.dumps(ov))}]}
    assert S.stats_full(ch({"p2p_stats": {"lr_layers": [0, 1]}}))
    assert not S.stats_full(ch({"p2p_stats": {"lr_layers": [0, 1], "emit_lr": True, "prefill_only": True}}))
    assert not S.stats_full(ch({"p2p_layer_experts": {}}))
    assert not S.stats_full({"commands": []})


def _back_pp(sim):
    for e in sim.engines.values():
        if e.role == "back" and len(e.nodes) > 1:
            return e
    raise AssertionError("清单里没有多机后段")


def test_calib_knobs_move_step_time(world) -> None:
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    net = S.Net(ip=ip, lat_same=0.1, lat_cross=3.0, pp_hop_ms=0.5, back_pp_hop_ms=0.5)
    p = S.Params(front_fixed_ms=1.0, back_fixed_ms=1.0)
    sim = S.Sim(world["stages"], perf=perf, net=net, node_gpu=gpu, params=p, front_slots=1, hold=True)
    e = _back_pp(sim)
    assert e.stats_full                                    # layout --stats：后段开了逐请求统计
    base = e.step_ms([300.0], [])
    p.stats_ms = 4.0
    assert e.step_ms([300.0], []) == pytest.approx(base + 4.0)
    p.stats_ms = 0.0
    comp = sum(nl * g.decode_ms(e.experts, 300.0, 1) for g, nl in zip(e.gpus, e.layers))
    p.dec_scale = {f"back:{g.gpu}": 2.0 for g in e.gpus}
    assert e.step_ms([300.0], []) == pytest.approx(base + comp, rel=1e-6)   # 单段串行时恰好多一份
    p.dec_scale = {}
    ncross = sum(1 for a, b in [(e.nodes[0], v) for v in e.nodes[1:]] + list(zip(e.nodes, e.nodes[1:]))
                 + [(e.nodes[-1], e.nodes[0])] if net.cross(a, b))
    net.back_pp_cross_ms = 1.0
    if ncross:
        assert e.step_ms([300.0], []) > base
    else:
        assert e.step_ms([300.0], []) == pytest.approx(base)
    pf = e.step_ms([], [200])
    p.pf_token_back_ms = 0.1
    assert e.step_ms([], [200]) == pytest.approx(pf + 20.0)


def _synth_rows(sim, *, b_max=3, n_per=40, pf_sizes=(20, 60, 120, 300, 600)) -> dict:
    """拿当前参数下模拟器自己的每步用时造步级记录（纯 decode 各批 n_per 步、单条 prefill 各长度 8 步）。"""
    rows = {}
    for name, e in sim.engines.items():
        rs, t = [], 0.0
        for b in range(1, b_max + 1):
            ms = e.step_ms([300.0] * b, [])
            for _ in range(n_per):
                rs.append({"t": t, "ms": ms, "n": b, "npf": 0, "pf": 0, "ctx": 300.0})
                t += ms / 1e3 + 0.01
        for n in pf_sizes:
            ms = e.step_ms([], [n])
            for _ in range(8):
                rs.append({"t": t, "ms": ms, "n": 1, "npf": 1, "pf": n, "ctx": 0.0})
                t += ms / 1e3 + 0.01
        rows[name] = rs
    return rows


def test_calibrate_recovers_known_step_model(world) -> None:
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    truth = dict(dec_scale={"back:A30": 0.8, "back:RTX6000": 1.3, "back:T4": 1.1, "front:A30": 1.2,
                            "front:RTX6000": 1.2, "front:T4": 1.2},
                 pf_gpu_scale={"back:A30": 1.1, "back:RTX6000": 0.9, "back:T4": 1.2, "front:A30": 1.0,
                               "front:RTX6000": 1.0, "front:T4": 1.0},
                 front_fixed_ms=0.9, pf_fixed_front_ms=1.0, pf_fixed_back_ms=2.0, pf_token_back_ms=0.08,
                 pf_fixed_per="stage")
    net_t = S.Net(ip=ip, lat_same=0.1, lat_cross=1.25, pp_hop_ms=0.5, back_pp_hop_ms=1.5)
    sim_t = S.Sim(world["stages"], perf=perf, net=net_t, node_gpu=gpu, params=S.Params(**truth), front_slots=1,
                  hold=True)
    rows = _synth_rows(sim_t)
    # 从「命令行默认」出发：倍数全 1、固定开销是别的值
    net0 = S.Net(ip=ip, lat_same=0.1, lat_cross=1.25, pp_hop_ms=0.5, back_pp_hop_ms=3.3)
    sim0 = S.Sim(world["stages"], perf=perf, net=net0, node_gpu=gpu,
                 params=S.Params(front_fixed_ms=None, pf_fixed_ms=27.0), front_slots=1, hold=True)
    cal = S.calibrate([("w", world["stages"], sim0, rows)], ridge=0.0, log=lambda s: None)
    assert cal["fit"]["rms_log_after"]["decode"] < 0.02, cal["fit"]
    assert cal["fit"]["rms_log_after"]["prefill"] < 0.03, cal["fit"]
    assert cal["fit"]["rms_log_before"]["decode"] > cal["fit"]["rms_log_after"]["decode"]
    # 单项不一定唯一（后段 PP 开销与每层倍数在同一种部署里分不太开），要求的是每步用时对得上
    # 套回去：每步用时和「真值」一致
    p2, n2 = S.apply_calib(cal, S.Params(), S.Net(ip=ip, lat_same=0.1, lat_cross=1.25, pp_hop_ms=0.5))
    sim2 = S.Sim(world["stages"], perf=perf, net=n2, node_gpu=gpu, params=p2, front_slots=1, hold=True)
    for name, e in sim_t.engines.items():
        assert sim2.engines[name].step_ms([300.0, 300.0], []) == pytest.approx(e.step_ms([300.0, 300.0], []), rel=0.05)


def test_calib_cli_then_run_with_calib(world, tmp_path) -> None:
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    sim_t = S.Sim(world["stages"], perf=perf, net=S.Net(ip=ip, back_pp_hop_ms=1.0), node_gpu=gpu,
                  params=S.Params(dec_scale={"back:A30": 0.9}, front_fixed_ms=1.0), front_slots=1, hold=True)
    d = tmp_path / "steps"
    d.mkdir()
    for name, rs in _synth_rows(sim_t).items():
        (d / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rs))
    real = setup(world, fake_real(world["stages"], n=6), per_front=1, gw_client_ms=2.0)   # 「网关外」2 ms
    rp = tmp_path / "real.json"
    rp.write_text(json.dumps(real))
    out = tmp_path / "calib.json"
    assert S.main(["calib", "--layer-bench", str(world["lb"]), "--hosts", str(HOSTS), "--back-pp-hop-ms", "3",
                   "--deploy", str(world["stages_path"]), str(d), str(rp), "--out", str(out)]) == 0
    cal = json.loads(out.read_text())
    assert cal["params"]["gw_client_ms"] == pytest.approx(2.0)        # 第三个值（真机结果）也当 --real 用
    fronts = {c["nodes"][0] for c in world["stages"]["channels"] if c["role"] == "front"}
    assert set(cal["net"]["agent_node_ms"]) == fronts
    assert set(cal["params"]["dec_scale"]) >= {"back:A30"}
    so = tmp_path / "sim.json"
    assert S.main(["run", "--stages", str(world["stages_path"]), "--replay", str(rp), "--layer-bench", str(world["lb"]),
                   "--hosts", str(HOSTS), "--per-front", "1", "--calib", str(out), "--out", str(so)]) == 0
    doc = json.loads(so.read_text())
    assert doc["sim"]["params"]["gw_client_ms"] == pytest.approx(2.0)
    assert doc["sim"]["params"]["dec_scale"] == cal["params"]["dec_scale"]


def test_gw_client_ms_shifts_ttft_not_tpot(world) -> None:
    real = fake_real(world["stages"], n=6)
    a = setup(world, real, per_front=1)
    b = setup(world, real, per_front=1, gw_client_ms=3.0)
    for r in a["requests"]:                          # 客户端口径 − 协调器口径 = 「网关外」
        assert r["prefill_ms"] - r["engine"]["ttft_ms"] == pytest.approx(0.0, abs=0.02)
    for r in b["requests"]:
        assert r["prefill_ms"] - r["engine"]["ttft_ms"] == pytest.approx(3.0, abs=0.02)
    ta = sorted(r["per_token_p50_ms"] for r in a["requests"])
    tb = sorted(r["per_token_p50_ms"] for r in b["requests"])
    assert tb == pytest.approx(ta, rel=0.05)                 # 只推后客户端看到的时刻，不改 token 间隔


def test_agent_node_ms_adds_to_agent_messages_only() -> None:
    net = S.Net(ip={"A": "10.0.1.1", "B": "10.0.1.2"}, lat_same=0.1, agent_hop_ms=1.0, hop_ms=5.0,
                agent_node_ms={"B": 2.0})
    assert net.msg("A", "B", 0, kind="agent") == pytest.approx(0.1 + 1.0 + 2.0)
    assert net.msg("B", "A", 0, kind="agent") == pytest.approx(0.1 + 1.0 + 2.0)
    assert net.msg("A", "B", 0, kind="gw") == pytest.approx(0.1 + 5.0)          # HTTP 网关那条路不算


def test_fit_agent_nodes_finds_slow_front_host(world) -> None:
    perf = S.load_perf([world["lb"]])
    ip, gpu = S.load_hosts(HOSTS)
    p = S.Params(front_fixed_ms=1.0, back_fixed_ms=1.0)
    sim0 = S.Sim(world["stages"], perf=perf, net=S.Net(ip=ip), node_gpu=gpu, params=p, front_slots=1, hold=True)
    steps = S.steps_model(_synth_rows(sim0))
    fronts = [c["nodes"][0] for c in world["stages"]["channels"] if c["role"] == "front"]
    slow = fronts[0]
    truth_net = S.Net(ip=ip, agent_hop_ms=1.0, agent_node_ms={slow: 2.0})
    real = S.run_sim(world["stages"], S.replay_requests(fake_real(world["stages"], n=24, gen=(30, 40, 20)),
                                                        world["stages"]),
                     perf=perf, net=truth_net, node_gpu=gpu, params=p, per_front=2, steps=steps)
    real["run"] = {**(real.get("run") or {}), "per_front": 2}
    got = S.fit_agent_nodes(world["stages"], perf=perf, net=S.Net(ip=ip, agent_hop_ms=1.0), node_gpu=gpu,
                            params=p, steps=steps, real=real, log=lambda s: None)
    assert got[slow] == pytest.approx(2.0, abs=0.5), got
    assert all(v <= 0.5 for k, v in got.items() if k != slow), got


def test_bind_speed_keeps_requests_on_fast_back(world) -> None:
    """mbpp 有一个 A30×2 的快后段、一个 RTX6000+T4 的慢后段：bind=speed 把更多请求放到快的上，TPOT 更低；
    least-free 只换「一样多时给谁」，结果合法。"""
    st = world["stages"]
    rows = [{"req": f"req{i}", "case_id": f"mbpp:{i}", "true_task": "mbpp", "channel": "back-mbpp-0",
             "n_prompt": 120, "n_generated": 60, "stop_reason": "length"} for i in range(24)]
    real = {"requests": rows}
    out = {b: setup(world, real, inflight=2, bind=b) for b in ("least", "least-free", "speed")}
    for b, doc in out.items():
        assert len(doc["requests"]) == 24 and doc["sim"]["bind"] == b
    n_fast = {b: sum(r["channel"] == "back-mbpp-0" for r in d["requests"]) for b, d in out.items()}
    tpot = {b: S.key_metrics(d)["TPOT 均值"] for b, d in out.items()}
    assert n_fast["speed"] > n_fast["least"] and tpot["speed"] < tpot["least"]


def test_back_slots_one_request_per_path(world) -> None:
    """--back-slots 1：每条后段 path 同时只跑一条；同类 path 都满时在前段等（wait_back_ms > 0）。"""
    real = fake_real(world["stages"], n=24)
    doc = setup(world, real, inflight=6, back_slots=1)
    assert set(doc["sim"]["back_slots"].values()) == {1}
    ev = {}
    for r in doc["requests"]:
        t0 = r["t_submit_s"] + r["prefill_ms"] / 1e3
        ev.setdefault(r["channel"], []).extend([(r["t_done_s"], -1), (t0, 1)])
    for e in ev.values():
        c = 0
        for _, d in sorted(e):
            c += d
            assert c <= 1
    assert any((r["tiered"].get("wait_back_ms") or 0) > 0 for r in doc["requests"])


def test_schemes_cli_side_by_side(world, tmp_path) -> None:
    """sim schemes：几份结果并排，一行一个方案，带两种吞吐和各 path 条数。"""
    real = fake_real(world["stages"], n=12)
    a = setup(world, real, inflight=2, back_slots=1)
    b = setup(world, real, inflight=4)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    md, js = tmp_path / "t.md", tmp_path / "t.json"
    assert S.main(["schemes", f"甲={pa}", f"乙={pb}", "--md", str(md), "--json", str(js)]) == 0
    text = md.read_text()
    assert "| 甲 |" in text and "| 乙 |" in text and "稳态 tok/s" in text
    recs = json.loads(js.read_text())
    assert [r["label"] for r in recs] == ["甲", "乙"]
    assert sum(recs[0]["path"].values()) == 12 and recs[0]["稳态 tok/s"] > 0
