# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P 移植：静态通道的清单翻译（`vllm/p2p/layout.py`）。只用标准库与 numpy。

钉住的是那些写错了**不会报错、只会让输出悄悄变差**的地方：
层号 0/1-based、全局→本地专家号、切层与清单对不上、两条通道共用一台卡。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from vllm.p2p.layout import (
    ARCH,
    LayoutError,
    channel_dtype,
    channels_from_plan,
    expert_key,
    load_plan,
    localize_expert_name,
    main as layout_main,
    pair_static,
    parse_hosts,
    parse_layer_experts,
    serve_commands,
)

# 取自 p2p-framework（qwen3-30b/plan_es90_fill.json、task/hosts.txt），拷一份作测试数据，
# 免得 vLLM 的测试依赖另一个仓库的目录结构
DATA = Path(__file__).resolve().parent / "data"
PLAN = DATA / "plan_es90_fill.json"
HOSTS = DATA / "hosts.txt"

# VLLM_MIGRATION.md 第 3 节那张表 —— 与 p2p-framework sel_static4 的静态配对一致
EXPECTED = {
    "gsm8k-0": (["N01", "N03", "N05"], [8, 13, 27]),
    "gsm8k-1": (["N10", "N04", "N02"], [8, 16, 24]),
    "mbpp-0": (["N11", "N06", "N07"], [8, 14, 26]),
    "mbpp-1": (["N15", "N13", "N14"], [8, 27, 13]),
    "no_robots-0": (["N12", "N09", "N08"], [8, 6, 34]),
}


@pytest.fixture(scope="module")
def plan():
    if not PLAN.exists():
        pytest.skip(f"{PLAN} 不在")
    return load_plan(PLAN)


@pytest.fixture(scope="module")
def chans(plan):
    return {c.name: c for c in channels_from_plan(plan, pair_static(plan))}


# --------------------------------------------------------------------------- #
# 清单 → 通道
# --------------------------------------------------------------------------- #
def test_greedy_pairing_reproduces_the_static_channels(chans) -> None:
    assert {n: (c.nodes, c.partition) for n, c in chans.items()} == EXPECTED


def test_pairing_matches_control_py_greedy(plan) -> None:
    """与 control.py 在 --load-plan 无 --wiring 时的贪心逐字同一口径：t50 升序、一对一。"""
    wired = pair_static(plan)
    used_f, used_b, ref = set(), set(), {}
    for pr in sorted(plan["pairings"], key=lambda x: x["t50"]):
        if pr["front"] in used_f or pr["back"] in used_b:
            continue
        ref[pr["front"]] = (pr["back"], pr["task"])
        used_f.add(pr["front"])
        used_b.add(pr["back"])
    assert wired == ref


def test_every_channel_covers_all_layers_exactly_once(chans) -> None:
    for c in chans.values():
        assert sum(c.partition) == c.n_layers == 48
        assert sorted(c.layer_experts()) == list(range(1, 49))


def test_hf_overrides_are_zero_based_and_match_the_manifest(plan, chans) -> None:
    """清单是 1-based，hf_overrides 是 0-based —— 只在一处转，这里钉住那一处。"""
    nodes = {p["node"]: p for p in plan["nodes"]}
    c = chans["gsm8k-0"]
    ov = c.hf_overrides()
    assert ov["architectures"] == [ARCH]
    le = ov["p2p_layer_experts"]
    # 前段 1..8 全装 → 不出现；后段从 1-based 第 9 层 = 0-based "8" 开始
    assert min(int(k) for k in le) == 8 and "7" not in le
    n03 = {ld["layer"]: ld["experts"] for ld in nodes["N03"]["layers"]}
    assert le["8"] == sorted(n03[9])
    assert le[str(21 - 1)] == sorted(n03[21])
    n05 = {ld["layer"]: ld["experts"] for ld in nodes["N05"]["layers"]}
    assert le["47"] == sorted(n05[48])
    json.dumps(ov)          # 能原样写进 --hf-overrides


def test_full_layers_are_left_out_of_overrides(chans) -> None:
    for c in chans.values():
        le = c.hf_overrides()["p2p_layer_experts"]
        assert all(len(v) < 128 for v in le.values())
        assert all(int(k) >= 8 for k in le)            # 前段 8 层都是全装


def test_gap_in_layers_is_caught(plan) -> None:
    bad = copy.deepcopy(plan)
    n03 = next(p for p in bad["nodes"] if p["node"] == "N03")
    n03["layers"] = [ld for ld in n03["layers"] if ld["layer"] != 21]   # 漏掉第 21 层
    with pytest.raises(LayoutError, match="接上|不连续"):
        channels_from_plan(bad, pair_static(bad))


def test_expert_id_out_of_range_is_caught(plan) -> None:
    bad = copy.deepcopy(plan)
    n03 = next(p for p in bad["nodes"] if p["node"] == "N03")
    n03["layers"][0]["experts"] = [0, 128]
    with pytest.raises(LayoutError, match="越界"):
        channels_from_plan(bad, pair_static(bad))


def test_wiring_by_node_id_and_mismatched_task(plan) -> None:
    w = {"pairs": [{"front": "N01", "back": "N06"}]}            # 节点 id 也认
    (c,) = channels_from_plan(plan, pair_static(plan, w))
    assert c.task == "mbpp" and c.nodes == ["N01", "N06", "N07"]
    with pytest.raises(LayoutError, match="task"):
        pair_static(plan, {"pairs": [{"front": "F0", "back": "Bmbpp0", "task": "gsm8k"}]})
    with pytest.raises(LayoutError, match="两次"):
        pair_static(plan, {"pairs": [{"front": "F0", "back": "Bmbpp0"},
                                     {"front": "F1", "back": "Bmbpp0"}]})


def test_model_shape_mismatch_is_caught(plan) -> None:
    with pytest.raises(LayoutError, match="层"):
        channels_from_plan(plan, pair_static(plan), n_layers=36)


# --------------------------------------------------------------------------- #
# dtype 与启动命令
# --------------------------------------------------------------------------- #
def test_auto_dtype_follows_card_type(chans) -> None:
    """`--dtype auto` 的口径：有 sm75 卡的通道 half。实验默认不用它（见下一条）。"""
    if not HOSTS.exists():
        pytest.skip("task/hosts.txt 不在")
    hosts = parse_hosts(HOSTS)
    got = {n: channel_dtype(c, hosts)[0] for n, c in chans.items()}
    # 只有 gsm8k-0 三台全是 A30
    assert got == {"gsm8k-0": "bfloat16", "gsm8k-1": "half", "mbpp-0": "half",
                   "mbpp-1": "half", "no_robots-0": "half"}
    assert channel_dtype(chans["gsm8k-0"], hosts, sm75=["N03"]) == ("half", ["N03"])


def test_default_is_fp16_and_one_attention_backend_everywhere(tmp_path) -> None:
    """对照实验的口径：**五条通道全部 fp16、同一个注意力后端**，A30 那条也不例外。

    只有 gsm8k-0 三台全是 A30、本可以 bf16 + FlashAttention —— 让它特殊，它的数字就和
    另外四条（以及 p2p-framework 的 DTYPE=float16 对照组）不可比。"""
    if not (PLAN.exists() and HOSTS.exists()):
        pytest.skip("清单或 hosts 不在")
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out)]) == 0
    chs = json.loads((out / "channels.json").read_text(encoding="utf-8"))["channels"]
    assert {c["dtype"] for c in chs} == {"half"}
    assert {c["attention_backend"] for c in chs} == {"TRITON_ATTN"}
    for c in chs:
        for cmd in c["commands"]:
            assert "--dtype half" in cmd["cmd"]
            assert "--attention-backend TRITON_ATTN" in cmd["cmd"]
    # 显式要 bf16：含 sm75 卡的通道必须被拒，而不是悄悄降级
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out),
                        "--dtype", "bfloat16"]) == 2


def test_serve_commands_differ_only_in_rank(chans) -> None:
    if not HOSTS.exists():
        pytest.skip("task/hosts.txt 不在")
    hosts = parse_hosts(HOSTS)
    c = chans["mbpp-1"]
    cmds = serve_commands(c, hosts, model_dir="/w", overrides_path="vc/mbpp-1.json",
                          dtype="half")
    assert [v for v, _ in cmds] == c.nodes
    assert all("VLLM_PP_LAYER_PARTITION=8,27,13 " in x for _, x in cmds)
    assert all("--master-addr 192.168.4.4" in x for _, x in cmds)     # 前段节点
    assert all("--attention-backend TRITON_ATTN" in x for _, x in cmds)
    assert "--headless" not in cmds[0][1] and "--port 8000" in cmds[0][1]
    assert all("--headless" in x for _, x in cmds[1:])

    def norm(s):
        s = s.replace("--headless", "").replace("--host 0.0.0.0 --port 8000", "")
        for r in range(3):
            s = s.replace(f"--node-rank {r}", "")
        return " ".join(s.split())
    assert len({norm(x) for _, x in cmds}) == 1, "除 node_rank / headless 外三台必须一字不差"


def test_parse_hosts_reads_gpu_from_comment(tmp_path) -> None:
    p = tmp_path / "h.txt"
    p.write_text("# 注释行\nN01  10.0.0.1:9101   # A30 24GB\nN02 10.0.0.2 --mem-mb 1 # T4\n\n",
                 encoding="utf-8")
    h = parse_hosts(p)
    assert h["N01"].host == "10.0.0.1" and h["N01"].gpu == "A30"
    assert h["N02"].host == "10.0.0.2" and h["N02"].gpu == "T4"


# --------------------------------------------------------------------------- #
# 模型侧的两个纯函数
# --------------------------------------------------------------------------- #
LE = {8: (3, 12, 37, 100), 9: (0, 1)}


@pytest.mark.parametrize("name,want", [
    ("model.layers.8.mlp.experts.37.up_proj.weight", "model.layers.8.mlp.experts.2.up_proj.weight"),
    ("layers.8.mlp.experts.3.down_proj.weight", "layers.8.mlp.experts.0.down_proj.weight"),
    ("model.layers.8.mlp.experts.38.up_proj.weight", None),           # 不驻留 → 跳过
    ("model.layers.7.mlp.experts.38.up_proj.weight",                  # 全装层：原样
     "model.layers.7.mlp.experts.38.up_proj.weight"),
    ("model.layers.8.mlp.gate.weight", "model.layers.8.mlp.gate.weight"),   # 路由器：原样
    ("model.layers.8.self_attn.q_proj.weight", "model.layers.8.self_attn.q_proj.weight"),
    ("model.layers.18.mlp.experts.1.up_proj.weight",                  # 18 ≠ 8
     "model.layers.18.mlp.experts.1.up_proj.weight"),
])
def test_localize_expert_name(name, want) -> None:
    assert localize_expert_name(name, LE) == want


def test_local_ids_follow_ascending_resident_order(chans) -> None:
    """本地号 i ↔ 全局号 resident[i]。模型构造 FusedMoE 用的是同一个升序。"""
    le0 = {int(k): tuple(v) for k, v in
           chans["no_robots-0"].hf_overrides()["p2p_layer_experts"].items()}
    for l, es in le0.items():
        assert list(es) == sorted(es)
        for i, e in enumerate(es):
            got = localize_expert_name(f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight", le0)
            assert got == f"model.layers.{l}.mlp.experts.{i}.gate_proj.weight"


def test_parse_layer_experts_validates() -> None:
    assert parse_layer_experts({"3": [5, 1]}, 4, 8) == {3: (1, 5)}
    assert parse_layer_experts(None, 4, 8) == {}
    for bad, msg in [({"4": [0]}, "越界"), ({"0": []}, "为空"), ({"0": [1, 1]}, "重复"),
                     ({"0": [8]}, "越界"), ({"x": [0]}, "层号")]:
        with pytest.raises(LayoutError, match=msg):
            parse_layer_experts(bad, 4, 8)


# --------------------------------------------------------------------------- #
# 部分专家模型的路由与 p2p-framework local_topk 的等价性（模块文档里那段证明的数值版）
# --------------------------------------------------------------------------- #
def _softmax(x):
    z = np.exp(x - x.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


@pytest.mark.parametrize("n_res,k", [(5, 2), (40, 8), (3, 8), (128, 8)])
def test_restricted_softmax_topk_equals_local_topk(n_res, k) -> None:
    rng = np.random.default_rng(n_res * 10 + k)
    E, T = 128, 64
    logits = rng.standard_normal((T, E)) * 3
    res = np.sort(rng.choice(E, n_res, replace=False))
    kk = min(k, n_res)

    # p2p-framework torch_model._choose 的 local_topk：全量 softmax → 非驻留置 −∞ → top-k → 重归一
    p = _softmax(logits)
    masked = np.full_like(p, -np.inf)
    masked[:, res] = p[:, res]
    ref_i = np.argsort(-masked, axis=1)[:, :kk]
    ref_w = np.take_along_axis(masked, ref_i, 1)
    ref_w /= ref_w.sum(1, keepdims=True)

    # vLLM 部分专家模型：取列 → softmax → top-k → 重归一（FusedMoE, renormalize=True）
    q = _softmax(logits[:, res])
    loc_i = np.argsort(-q, axis=1)[:, :kk]
    loc_w = np.take_along_axis(q, loc_i, 1)
    loc_w /= loc_w.sum(1, keepdims=True)

    assert np.array_equal(res[loc_i], ref_i)          # 选的是同一批全局专家
    assert np.allclose(loc_w, ref_w, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name,want", [
    ("model.layers.0.mlp.experts.0.gate_proj.weight", (0, 0, "gate_proj")),    # 带 model. 前缀
    ("layers.8.mlp.experts.37.up_proj.weight", (8, 37, "up_proj")),            # 剥了前缀
    ("model.layers.47.mlp.experts.127.down_proj.weight", (47, 127, "down_proj")),
    ("model.layers.8.mlp.gate.weight", None),                                  # 路由器
    ("model.layers.8.self_attn.q_proj.weight", None),
    ("model.embed_tokens.weight", None),
])
def test_expert_key(name, want) -> None:
    assert expert_key(name) == want


def test_expert_key_sees_every_localized_expert() -> None:
    """模型的逐专家核对就是「本地化后的名字 → expert_key」。全集走一遍，一个都不能漏 ——
    第一版用 re.match，带 model. 前缀的名字全漏了，核对把所有专家报成缺失。"""
    le = {2: (1, 4, 6), 3: (0, 7)}
    seen = set()
    for l in range(4):
        for e in range(8):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                new = localize_expert_name(f"model.layers.{l}.mlp.experts.{e}.{proj}.weight", le)
                if new is not None:
                    seen.add(expert_key(new))
    want = {(l, i, p) for l in range(4) for i in range(len(le.get(l, range(8))))
            for p in ("gate_proj", "up_proj", "down_proj")}
    assert seen == want


def test_gpu_mem_node_stats_and_prefix_caching(tmp_path) -> None:
    """A4：识别服务与 N01 的前段同卡时只降 N01 的显存比例；A5：--stats 进 hf_overrides；
    前缀缓存默认关（本框架没有前缀缓存，开了 TTFT 不可比）。"""
    if not (PLAN.exists() and HOSTS.exists()):
        pytest.skip("清单或 hosts 不在")
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out),
                        "--gpu-mem", "0.95", "--gpu-mem-node", "N01=0.5", "--stats"]) == 0
    cj = json.loads((out / "channels.json").read_text(encoding="utf-8"))
    assert cj["n_experts"] == 128 and cj["n_layers"] == 48
    g0 = next(c for c in cj["channels"] if c["name"] == "gsm8k-0")
    assert g0["gpu_mem"] == {"N01": 0.5, "N03": 0.95, "N05": 0.95}
    cmds = {x["node"]: x["cmd"] for x in g0["commands"]}
    assert "--gpu-memory-utilization 0.5 " in cmds["N01"]
    assert "--gpu-memory-utilization 0.95 " in cmds["N03"]
    assert all("--no-enable-prefix-caching" in c for c in cmds.values())
    ov = json.loads((out / "gsm8k-0.overrides.json").read_text(encoding="utf-8"))
    assert ov["p2p_stats"] == {"lr_layers": list(range(8))}
    # 其它通道不受 N01 的影响
    g1 = next(c for c in cj["channels"] if c["name"] == "gsm8k-1")
    assert set(g1["gpu_mem"].values()) == {0.95}
    # 写错节点名 / 比例要报错
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out),
                        "--gpu-mem-node", "N99=0.5"]) == 2
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out),
                        "--gpu-mem-node", "N01=1.5"]) == 2
    # 不给 --stats 时 overrides 里没有它；--prefix-caching 时不加关闭开关
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out),
                        "--prefix-caching"]) == 0
    ov = json.loads((out / "gsm8k-0.overrides.json").read_text(encoding="utf-8"))
    assert "p2p_stats" not in ov
    cj = json.loads((out / "channels.json").read_text(encoding="utf-8"))
    assert not any("--no-enable-prefix-caching" in x["cmd"]
                   for c in cj["channels"] for x in c["commands"])


def test_max_num_seqs_is_passed_to_every_rank(tmp_path) -> None:
    if not (PLAN.exists() and HOSTS.exists()):
        pytest.skip("清单或 hosts 不在")
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out),
                        "--max-num-seqs", "128"]) == 0
    cj = json.loads((out / "channels.json").read_text(encoding="utf-8"))
    assert all("--max-num-seqs 128 " in x["cmd"] for c in cj["channels"] for x in c["commands"])
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out)]) == 0
    cj = json.loads((out / "channels.json").read_text(encoding="utf-8"))
    assert not any("--max-num-seqs" in x["cmd"] for c in cj["channels"] for x in c["commands"])


def test_max_num_seqs_per_channel_override(tmp_path) -> None:
    if not (PLAN.exists() and HOSTS.exists()):
        pytest.skip("清单或 hosts 不在")
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out),
                        "--max-num-seqs", "8",
                        "--max-num-seqs-channel", "no_robots-0=32"]) == 0
    cj = json.loads((out / "channels.json").read_text(encoding="utf-8"))
    by = {c["name"]: c for c in cj["channels"]}
    assert by["no_robots-0"]["max_num_seqs"] == 32 and by["gsm8k-0"]["max_num_seqs"] == 8
    assert all("--max-num-seqs 32 " in x["cmd"] for x in by["no_robots-0"]["commands"])
    assert all("--max-num-seqs 8 " in x["cmd"] for x in by["gsm8k-0"]["commands"])
    # 写错通道名 / 格式：直接报错，不悄悄忽略
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out),
                        "--max-num-seqs-channel", "no_robot=32"]) == 2
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out),
                        "--max-num-seqs-channel", "no_robots-0"]) == 2


def test_relative_out_becomes_absolute_in_remote_commands(tmp_path, monkeypatch) -> None:
    """--out 给相对路径时，命令里读 overrides 的路径必须是绝对的（ssh 后的工作目录是 ~）。"""
    if not (PLAN.exists() and HOSTS.exists()):
        pytest.skip("清单或 hosts 不在")
    monkeypatch.chdir(tmp_path)
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", "vc"]) == 0
    cj = json.loads((tmp_path / "vc" / "channels.json").read_text(encoding="utf-8"))
    want = f'$(cat {tmp_path / "vc"}/'
    assert all(want in x["cmd"] for c in cj["channels"] for x in c["commands"])


def test_manual_plan_without_pairings_or_model_shape(tmp_path) -> None:
    """p2p-framework `deploy/manual.py` 的清单（如 plan_full_2ch）：没有组合矩阵、model 字段是模型名。
    layout 按段名编号配对（F{i} ↔ B<task>{i}），层数 / 专家数从 --model-dir 的 config.json 读。"""
    import sys

    fw = Path(__file__).resolve().parents[3] / "p2p-framework"
    if not (fw / "p2pmoe" / "deploy" / "manual.py").is_file():
        pytest.skip("旁边没有 p2p-framework")
    sys.path.insert(0, str(fw))
    from p2pmoe.deploy.manual import ManualSpec, build_manual_manifest
    from p2pmoe.planner.hf_config import PRESETS, model_spec_from_hf

    from vllm.p2p import layout as LY

    cfg = dict(PRESETS["qwen3-30b-a3b"])
    spec_m, _ = model_spec_from_hf(cfg, name="Qwen3-30B-A3B")
    lay = ManualSpec.from_dict({"l0": 8, "channels": [
        {"front": "N01", "back": [{"node": "N02", "layers": [9, 28]}, {"node": "N03", "layers": [29, 48]}]},
        {"front": "N04", "back": [{"node": "N05", "layers": [9, 28]}, {"node": "N06", "layers": [29, 48]}]},
    ]}, n_layers=48)
    man, wired = build_manual_manifest(lay, spec_m, model_name="Qwen3-30B-A3B")
    plan = json.loads(man.to_json())
    assert not plan["pairings"]
    assert LY.manual_wiring(plan) == wired == {"F0": ("Bgeneral0", "general"), "F1": ("Bgeneral1", "general")}

    p = tmp_path / "plan_full_2ch.json"
    p.write_text(json.dumps(plan))
    hosts = tmp_path / "hosts.txt"
    hosts.write_text("".join(f"N0{i} 192.168.1.{i + 1}:9101 # A30\n" for i in range(1, 7)))
    mdir = tmp_path / "w"
    mdir.mkdir()
    (mdir / "config.json").write_text(json.dumps({"num_hidden_layers": 48, "num_experts": 128}))
    out = tmp_path / "vc"
    assert LY.main(["--plan", str(p), "--hosts", str(hosts), "--out", str(out),
                    "--model-dir", str(mdir), "--vllm-bin", "/venv/bin/vllm", "--max-num-seqs", "8"]) == 0
    cj = json.loads((out / "channels.json").read_text())
    assert [(c["name"], c["nodes"], c["partition"]) for c in cj["channels"]] == [
        ("general-0", ["N01", "N02", "N03"], [8, 20, 20]), ("general-1", ["N04", "N05", "N06"], [8, 20, 20])]
    ov = json.loads((out / "general-0.overrides.json").read_text())
    assert not ov.get("p2p_layer_experts")                       # 全专家：不列驻留集
    # 配对信息不全（编号对不上）时仍然要求 --wiring
    bad = json.loads(json.dumps(plan))
    bad["segments"]["Bgeneral7"] = bad["segments"].pop("Bgeneral1")
    assert LY.manual_wiring(bad) is None
    with pytest.raises(LY.LayoutError, match="--wiring"):
        LY.pair_static(bad)
