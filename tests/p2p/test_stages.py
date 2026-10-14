# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1：channels.json → stages.json（vllm/p2p/stages.py），以及它与 launch.py / stage_agent 的衔接。

拿 A 的真实清单（tests/p2p/data）生成通道，再拆成分段引擎，钉住：前段 = 前 L₀ 层、每个前段节点一个；
后段 = 其余层、驻留集与切层沿用通道；命令能被 launch 认出（进程名、venv、node_rank 0 最后起）、
能被 stage_agent 原样解析成同一组引擎参数（各节点逐字一致）。
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from vllm.p2p import launch, stages
from vllm.p2p.layout import main as layout_main

DATA = Path(__file__).resolve().parent / "data"
PLAN, HOSTS = DATA / "plan_es90_fill.json", DATA / "hosts.txt"


@pytest.fixture()
def cj(tmp_path):
    if not (PLAN.exists() and HOSTS.exists()):
        pytest.skip("清单或 hosts 不在")
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out),
                        "--model-dir", "/w", "--vllm-bin", "/venv/bin/vllm", "--stats",
                        "--max-num-seqs", "8", "--max-num-seqs-channel", "no_robots-0=32"]) == 0
    return out / "channels.json"


def test_split_channels_into_fronts_and_backs(cj, tmp_path) -> None:
    out = tmp_path / "stages.json"
    assert stages.main(["--channels", str(cj), "--out", str(out), "--front-gpu-mem", "0.9"]) == 0
    st = json.loads(out.read_text())
    src = json.loads(cj.read_text())
    by = {c["name"]: c for c in st["channels"]}
    assert len(st["fronts"]) == len(src["channels"]) == len(st["backs"])     # 五条通道、五个前段
    for c in src["channels"]:
        f, b = by[f"front-{c['nodes'][0]}"], by[f"back-{c['name']}"]
        assert f["layers"] == [0, 8] and f["nodes"] == c["nodes"][:1] and f["task"] is None
        assert b["layers"] == [8, 48] and b["nodes"] == c["nodes"][1:] and b["task"] == c["task"]
        assert b["partition"] == c["partition"][1:] and f["gpu_mem"] == {c["nodes"][0]: 0.9}
        assert b["max_num_seqs"] == c["max_num_seqs"]
        # 后段的驻留集与通道一字不差；前段不带驻留集（全装）；p2p_stage 由代理加，不在命令里
        cov = json.loads(Path(c["overrides"]).read_text())
        for e, want_le in ((b, cov["p2p_layer_experts"]), (f, {})):
            for st_cmd in e["commands"]:
                ov = json.loads(shlex.split(st_cmd["cmd"].split(" --hf-overrides ", 1)[1])[0])
                assert ov["p2p_layer_experts"] == want_le and "p2p_stage" not in ov
                assert ov["p2p_channel"] == e["name"] and "p2p_stats" in ov
    assert by["back-no_robots-0"]["max_num_seqs"] == 32
    # 前段默认开前段出口识别的统计：前 L₀ 层、交给协调器、不落盘、只统计 prefill
    fcmd = by[st["fronts"][0]]["commands"][0]["cmd"]
    fov = json.loads(shlex.split(fcmd.split(" --hf-overrides ", 1)[1])[0])
    assert fov["p2p_stats"] == {"lr_layers": list(range(8)), "emit_lr": True, "write": False,
                                "prefill_only": True}
    st2 = stages.stages_from_channels(src, front_lr=False)
    fcmd2 = {c["name"]: c for c in st2["channels"]}[st2["fronts"][0]]["commands"][0]["cmd"]
    fov2 = json.loads(shlex.split(fcmd2.split(" --hf-overrides ", 1)[1])[0])
    assert "emit_lr" not in fov2.get("p2p_stats", {})


def test_commands_fit_launch_and_agent(cj) -> None:
    from vllm.p2p import stage_agent

    st = stages.stages_from_channels(json.loads(cj.read_text()), only=["gsm8k-0"])
    assert st["fronts"] == ["front-N01"] and st["backs"] == ["back-gsm8k-0"]
    back = {c["name"]: c for c in st["channels"]}["back-gsm8k-0"]
    parsed = []
    for k, x in enumerate(back["commands"]):
        cmd = x["cmd"]
        assert launch.venv_of(cmd) == "/venv"
        import re
        assert re.search(launch.match_pattern("back-gsm8k-0"), cmd)
        assert not re.search(launch.match_pattern("back-gsm8k-0"), cmd.replace("back-gsm8k-0", "back-gsm8k-00"))
        assert cmd.startswith("VLLM_PP_LAYER_PARTITION=13,27 ")
        assert f"--node-rank {k} " in cmd
        toks = shlex.split(cmd)
        argv = toks[toks.index("vllm.p2p.stage_agent") + 1:]
        a = stage_agent.build_args(argv)
        assert a.role == "back" and a.layers == "8,48" and a.nnodes == 2 and a.node_rank == k
        assert a.master_addr == back["commands"][0]["host"]
        parsed.append({k2: v for k2, v in vars(a).items() if k2 != "node_rank"})
    assert parsed[0] == parsed[1]          # 各节点除 node_rank 外参数逐字一致
    # launch 先起 headless、最后起 node_rank 0
    order = sorted(back["commands"], key=lambda s: "--node-rank 0 " in s["cmd"])
    assert "--node-rank 1 " in order[0]["cmd"]
    front = {c["name"]: c for c in st["channels"]}["front-N01"]["commands"][0]["cmd"]
    assert "--node-rank 0 " in front and "VLLM_PP_LAYER_PARTITION" not in front
    assert "--nnodes" not in front


def test_engine_args_from_agent_flags(cj) -> None:
    pytest.importorskip("torch")
    from vllm.p2p import stage_agent

    st = stages.stages_from_channels(json.loads(cj.read_text()), only=["gsm8k-0"])
    back = {c["name"]: c for c in st["channels"]}["back-gsm8k-0"]["commands"][0]["cmd"]
    toks = shlex.split(back)
    ea = stage_agent.build_engine_args(stage_agent.build_args(toks[toks.index("vllm.p2p.stage_agent") + 1:]))
    assert ea.hf_overrides["p2p_stage"] == {"role": "back", "layers": [8, 48]}
    assert ea.enable_prompt_embeds and ea.async_scheduling is False and not ea.enable_prefix_caching
    assert ea.pipeline_parallel_size == 2 and ea.nnodes == 2 and ea.skip_tokenizer_init
    assert ea.served_model_name == ["back-gsm8k-0"] and ea.max_num_seqs == 8
    assert ea.distributed_executor_backend == "mp"


def test_only_unknown_channel_is_an_error(cj) -> None:
    with pytest.raises(ValueError, match="没有通道"):
        stages.stages_from_channels(json.loads(cj.read_text()), only=["nope"])


def test_compare_same_channel_pairs() -> None:
    from vllm.p2p.compare import same_channel

    a = [{"case_id": "x", "channel": "gsm8k-0"}, {"case_id": "y", "channel": "gsm8k-1"}]
    b = [{"case_id": "x", "channel": "back-gsm8k-0"}, {"case_id": "y", "channel": "back-gsm8k-0"}]
    kept, dropped = same_channel(list(zip(a, b)))
    assert [x["case_id"] for x, _ in kept] == ["x"] and dropped == 1
