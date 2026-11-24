# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""p2p 工具链的模型族（`vllm/p2p/models.py`）与它在 layout / stages / layer_bench 里的走向。

钉住 GLM-4.7-Flash 那几处写错了**不报错**的：稠密第 0 层进了 p2p_layer_experts（模型构造会拒）、
LR 特征算了没有路由的层（特征维对不上 LR 模型）、注意力后端还是 TRITON_ATTN（MLA 模型选不到）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm.p2p import models as M
from vllm.p2p.layout import main as layout_main
from vllm.p2p.stages import stages_from_channels

GLM_CFG = {"architectures": ["Glm4MoeLiteForCausalLM"], "model_type": "glm4_moe_lite",
           "num_hidden_layers": 47, "hidden_size": 2048, "n_routed_experts": 64,
           "num_experts_per_tok": 4, "first_k_dense_replace": 1, "n_shared_experts": 1,
           "vocab_size": 154880}
QWEN_CFG = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
            "num_hidden_layers": 48, "hidden_size": 2048, "num_experts": 128,
            "num_experts_per_tok": 8, "decoder_sparse_step": 1, "mlp_only_layers": []}
NEXT_CFG = {"architectures": ["Qwen3NextForCausalLM"], "model_type": "qwen3_next",
            "num_hidden_layers": 48, "hidden_size": 2048, "num_experts": 512,
            "num_experts_per_tok": 10, "decoder_sparse_step": 1, "mlp_only_layers": []}


def test_family_lookup() -> None:
    assert M.family_of(GLM_CFG).arch == "Glm4MoeLitePartialForCausalLM"
    assert M.family_of(QWEN_CFG).arch == "Qwen3MoePartialForCausalLM"
    assert M.family_of(NEXT_CFG).name == "qwen3_next"
    assert M.family_of(None) is M.DEFAULT_FAMILY
    assert M.family_of({"num_hidden_layers": 48, "num_experts": 128}) is M.DEFAULT_FAMILY
    # 只认架构名（没有 model_type）也行
    assert M.family_of({"architectures": ["Glm4MoeLitePartialForCausalLM"]}).name == "glm4_moe_lite"
    with pytest.raises(ValueError, match="不认识"):
        M.family_of({"model_type": "llama"})
    assert M.family_of(GLM_CFG).attention_backend == "TRITON_MLA"
    assert M.family_of(GLM_CFG).scoring == "sigmoid"


def test_shape_helpers() -> None:
    assert M.n_experts_of(GLM_CFG) == 64 and M.top_k_of(GLM_CFG) == 4
    assert M.n_experts_of(QWEN_CFG) == 128 and M.n_experts_of(NEXT_CFG) == 512
    assert M.dense_layers(GLM_CFG) == {0}
    assert M.dense_layers(QWEN_CFG) == frozenset()
    assert M.dense_layers(dict(QWEN_CFG, mlp_only_layers=[3])) == {3}
    assert M.dense_layers(dict(QWEN_CFG, decoder_sparse_step=2)) == {l for l in range(48) if l % 2 == 0}
    assert M.dense_layers(dict(GLM_CFG, first_k_dense_replace=3)) == {0, 1, 2}


# --------------------------------------------------------------------------- #
def _glm_plan(dense_experts) -> dict:
    """两条通道：F0(N01: 1–4) → Bgsm8k0(N02: 5–25, N03: 26–47)，F1(N04) → Bmbpp1(N05, N06)。"""
    def layers(lo, hi, n):
        return [{"layer": l, "experts": (dense_experts if l == 1 else list(range(n)))}
                for l in range(lo, hi + 1)]
    nodes = [{"node": "N01", "layers": layers(1, 4, 64)}, {"node": "N04", "layers": layers(1, 4, 64)},
             {"node": "N02", "layers": layers(5, 25, 40)}, {"node": "N03", "layers": layers(26, 47, 48)},
             {"node": "N05", "layers": layers(5, 30, 32)}, {"node": "N06", "layers": layers(31, 47, 64)}]
    segs = {"F0": {"role": "front", "nodes": ["N01"]}, "F1": {"role": "front", "nodes": ["N04"]},
            "Bgsm8k0": {"role": "back:gsm8k", "nodes": ["N02", "N03"]},
            "Bmbpp1": {"role": "back:mbpp", "nodes": ["N05", "N06"]}}
    return {"model": "GLM-4.7-Flash", "l0": 4, "nodes": nodes, "segments": segs}


@pytest.mark.parametrize("dense_experts", [[], list(range(64))])
def test_layout_glm(tmp_path: Path, dense_experts) -> None:
    """稠密层在清单里写不写专家都行；overrides 不列它，LR 层跳过它，后端是 TRITON_MLA。"""
    mdir = tmp_path / "glm"
    mdir.mkdir()
    (mdir / "config.json").write_text(json.dumps(GLM_CFG))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_glm_plan(dense_experts)))
    hosts = tmp_path / "hosts.txt"
    hosts.write_text("".join(f"N0{i} 10.0.1.{i}:9101 # T4\n" for i in range(1, 7)))
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(plan), "--hosts", str(hosts), "--out", str(out),
                        "--model-dir", str(mdir), "--stats", "--vllm-bin", "/venv/bin/vllm"]) == 0
    cj = json.loads((out / "channels.json").read_text())
    assert cj["family"] == "glm4_moe_lite" and cj["n_experts"] == 64 and cj["n_layers"] == 47
    assert cj["dense_layers"] == [0]
    c0 = next(c for c in cj["channels"] if c["name"] == "gsm8k-0")
    assert c0["attention_backend"] == "TRITON_MLA"
    assert "--attention-backend TRITON_MLA" in c0["commands"][0]["cmd"]
    assert c0["stages"][0]["n_experts"] == 64.0                   # 前段：只数 MoE 层
    ov = json.loads((out / "gsm8k-0.overrides.json").read_text())
    assert ov["architectures"] == ["Glm4MoeLitePartialForCausalLM"]
    le = ov["p2p_layer_experts"]
    assert "0" not in le                                          # 稠密层不列
    assert set(le) == {str(l) for l in range(4, 47)}              # 后段 5–47（1-based）全是部分专家
    assert le["4"] == list(range(40)) and le["46"] == list(range(48))
    assert ov["p2p_stats"]["lr_layers"] == [1, 2, 3]              # 前 L₀=4 层去掉稠密的第 0 层

    st = stages_from_channels(cj)
    front = next(e for e in st["channels"] if e["role"] == "front")
    fov = json.loads(front["commands"][0]["cmd"].split("--hf-overrides ", 1)[1].strip("'"))
    assert fov["architectures"] == ["Glm4MoeLitePartialForCausalLM"]
    assert fov["p2p_stats"]["lr_layers"] == [1, 2, 3]
    assert "--attention-backend TRITON_MLA" in front["commands"][0]["cmd"]
    assert st["family"] == "glm4_moe_lite" and st["dense_layers"] == [0]


def test_layout_rejects_lr_on_dense_layer(tmp_path: Path) -> None:
    mdir = tmp_path / "glm"
    mdir.mkdir()
    (mdir / "config.json").write_text(json.dumps(GLM_CFG))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_glm_plan([])))
    assert layout_main(["--plan", str(plan), "--out", str(tmp_path / "vc"), "--model-dir", str(mdir),
                        "--stats", "--lr-layers", "0,1,2"]) == 2


def test_layout_family_flag_without_config(tmp_path: Path) -> None:
    """控制机上没有权重目录（--model-dir 是 $WEIGHTS）：用 --model-family 指定，稠密层按族的默认。"""
    p = _glm_plan([])
    p["model"] = "47L-d2048-E64"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(p))
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(plan), "--out", str(out), "--model-family", "glm4_moe_lite"]) == 0
    cj = json.loads((out / "channels.json").read_text())
    assert cj["dense_layers"] == [0] and cj["channels"][0]["attention_backend"] == "TRITON_MLA"
    ov = json.loads((out / "gsm8k-0.overrides.json").read_text())
    assert "0" not in ov["p2p_layer_experts"]


def test_layout_family_mismatch(tmp_path: Path) -> None:
    mdir = tmp_path / "glm"
    mdir.mkdir()
    (mdir / "config.json").write_text(json.dumps(GLM_CFG))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(_glm_plan([])))
    assert layout_main(["--plan", str(plan), "--out", str(tmp_path / "vc"), "--model-dir", str(mdir),
                        "--model-family", "qwen3_moe"]) == 2


def test_layer_bench_overrides_glm() -> None:
    from vllm.p2p.layer_bench import _overrides, family_defaults
    ov = _overrides(2, 32, cfg=GLM_CFG)
    assert ov["architectures"] == ["Glm4MoeLitePartialForCausalLM"]
    assert ov["first_k_dense_replace"] == 0 and ov["num_hidden_layers"] == 2
    assert ov["p2p_layer_experts"] == {"0": list(range(32)), "1": list(range(32))}
    assert "p2p_layer_experts" not in _overrides(6, 64, cfg=GLM_CFG)
    assert family_defaults(GLM_CFG) == ([64, 32], "TRITON_MLA")
    assert family_defaults(QWEN_CFG) == ([128, 64], "TRITON_ATTN")
    # 老调用方式（Qwen，显式总专家数）不变
    assert _overrides(2, 64, 128)["architectures"] == ["Qwen3MoePartialForCausalLM"]


def test_layout_family_from_plan_shape(tmp_path: Path) -> None:
    """控制机上没有 config、也没给 --model-family：按清单的 "47L-d2048-E64" 认出 GLM。"""
    p = _glm_plan([])
    p["model"] = "47L-d2048-E64"
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(p))
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(plan), "--out", str(out)]) == 0
    cj = json.loads((out / "channels.json").read_text())
    assert cj["family"] == "glm4_moe_lite" and cj["dense_layers"] == [0]
    assert M.family_by_shape(48, 128).name == "qwen3_moe" and M.family_by_shape(10, 10) is None


def test_layout_qwen3_next(tmp_path: Path) -> None:
    mdir = tmp_path / "next"
    mdir.mkdir()
    (mdir / "config.json").write_text(json.dumps(NEXT_CFG))
    p = _glm_plan([])
    nodes = []
    for nd in p["nodes"]:
        lo, hi = nd["layers"][0]["layer"], nd["layers"][-1]["layer"]
        hi = 48 if hi == 47 else hi
        nodes.append({"node": nd["node"], "layers": [{"layer": l, "experts": list(range(512 if lo == 1 else 100))}
                                                    for l in range(lo, hi + 1)]})
    p["nodes"] = nodes
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(p))
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(plan), "--out", str(out), "--model-dir", str(mdir), "--stats"]) == 0
    cj = json.loads((out / "channels.json").read_text())
    assert cj["family"] == "qwen3_next" and cj["dense_layers"] == [] and cj["n_experts"] == 512
    assert cj["channels"][0]["attention_backend"] == "TRITON_ATTN"
    ov = json.loads((out / "gsm8k-0.overrides.json").read_text())
    assert ov["architectures"] == ["Qwen3NextPartialForCausalLM"]
    assert "0" not in ov["p2p_layer_experts"] and len(ov["p2p_layer_experts"]["10"]) == 100
    assert ov["p2p_stats"]["lr_layers"] == [0, 1, 2, 3]
    assert layout_main(["--plan", str(plan), "--out", str(out), "--model-dir", str(mdir),
                        "--prefix-caching"]) == 2
