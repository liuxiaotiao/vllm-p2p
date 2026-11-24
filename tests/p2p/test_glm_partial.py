# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4.7-Flash 的部分专家 / 分段模型（`glm4_moe_lite_partial.py`）。

真的在 CPU 上建 vLLM 的模块：要把 vLLM 的平台强制成 CPU，所以放在子进程里跑
（`cpu_scripts/`），不影响同一个 pytest 进程里的别的测试。"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent / "cpu_scripts"


def _run(script: str, *args: str) -> dict:
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, str(SCRIPTS / script), *args], capture_output=True,
                       text=True, timeout=600, env=env, cwd=str(ROOT))
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        if "No module named" in p.stderr and "vllm" not in p.stderr.split("No module named")[-1][:20]:
            pytest.skip(p.stderr.strip().splitlines()[-1])
        raise AssertionError(f"{script} 失败：\n{p.stderr[-3000:]}")
    return json.loads(lines[-1])


@pytest.mark.parametrize("seed", ["0", "1"])
def test_glm_block_routing_matches_reference(seed) -> None:
    """原生 GLM MoE 块 = 手写参考（sigmoid、按 s+bias 挑、权重 s 重归一 × 1.8、加共享专家）；
    部分专家块 = 「非驻留置 −∞」的参考；真实路由落在驻留集里的 token 与全装逐位相同。"""
    r = _run("glm_partial_block.py", seed)
    assert r["full_vs_ref"] < 1e-4 and r["part_vs_masked_ref"] < 1e-4
    assert r["part_vs_full_on_covered"] == 0.0
    assert r["part_vs_masked_native"] < 1e-4          # parity_glm.py 的参考就靠这个等价
    assert r["local_bias_ok"] and r["scale"] == pytest.approx(1.8)


def test_glm_partial_model_stage_and_loading() -> None:
    r = _run("glm_partial_model.py")
    assert r["back_layers"] == ["PPMissingLayer", "PPMissingLayer",
                                "Glm4MoeLiteDecoderLayer", "Glm4MoeLiteDecoderLayer"]
    assert r["back_embed"] == "ZeroEmbed" and r["back_head"] == "ParallelLMHead"
    assert r["back_partial"] == [True, [1, 4, 6], [0, 7]]
    assert r["back_expert_rows_ok"] and r["back_bias_ok"] and r["back_full_gate_ok"]
    assert r["front_layers"][:2] == ["Glm4MoeLiteDecoderLayer"] * 2
    assert r["front_norm"] == "AddResidual" and r["front_head"] == "PPMissingLayer"
    assert r["front_dense0"] == "Glm4MoeLiteMLP" and r["front_moe1_full"]
    assert r["missing_raises"] and r["dense_listed_raises"]


def test_sigmoid_bias_topk_equals_masked_full_routing() -> None:
    """驻留子集上的 sigmoid+bias top-k = 全量打分把不驻留的置 −∞ 再挑（换回全局号后逐个相等）。"""
    pytest.importorskip("vllm.model_executor.models.p2p_partial")
    from vllm.model_executor.models.p2p_partial import sigmoid_bias_topk
    g = torch.Generator().manual_seed(3)
    logits, bias = torch.randn(50, 16, generator=g), torch.randn(16, generator=g) * 0.3
    R = torch.tensor([0, 3, 4, 9, 11, 15])
    w, ids = sigmoid_bias_topk(logits[:, R], bias[R], 4, True)
    s = torch.sigmoid(logits)
    choice = (s + bias).masked_fill(~torch.isin(torch.arange(16), R), float("-inf"))
    ref_ids = choice.topk(4, -1)[1]
    ref_w = s.gather(-1, ref_ids)
    ref_w = ref_w / ref_w.sum(-1, keepdim=True)
    got = {(t, int(R[i])): float(w[t, j]) for t in range(50) for j, i in enumerate(ids[t])}
    ref = {(t, int(e)): float(ref_w[t, j]) for t in range(50) for j, e in enumerate(ref_ids[t])}
    assert got.keys() == ref.keys()
    assert max(abs(got[k] - ref[k]) for k in got) < 1e-6
    assert w.dtype == torch.float32 and ids.dtype == torch.int32
    w1, ids1 = sigmoid_bias_topk(logits[:, :2], bias[:2], 4, False)      # 驻留比 k 少：取全部
    assert ids1.shape == (50, 2) and torch.allclose(w1, torch.sigmoid(logits[:, :2]).gather(-1, ids1.long()))


def test_route_stats_sigmoid_scoring() -> None:
    from vllm.p2p.stats import RouteStats
    st = RouteStats(n_experts=6, top_k=2, norm_topk=True, scoring="sigmoid")
    st.register_layer(1, None)
    st.allocate(8, "cpu")
    logits = torch.tensor([[0.0, 3.0, 2.0, -1.0, 0.5, 2.5]])
    bias = torch.tensor([0.0, -0.9, 0.0, 2.0, 0.0, 0.0])
    st.capture(0, logits, bias)
    s = torch.sigmoid(logits[0])
    ids = sorted(int(x) for x in st.topi[0, 0])
    assert ids == [3, 5]                               # 按 s+bias 挑：3（bias 大）、5
    w = {int(i): float(x) for i, x in zip(st.topi[0, 0], st.topw[0, 0])}
    assert w[3] == pytest.approx(float(s[3] / (s[3] + s[5])))  # 权重不带 bias
    with pytest.raises(ValueError):
        RouteStats(n_experts=6, top_k=2, norm_topk=True, scoring="relu")
    soft = RouteStats(n_experts=6, top_k=2, norm_topk=False)
    soft.register_layer(1, None)
    soft.allocate(8, "cpu")
    soft.capture(0, logits)
    assert sorted(int(x) for x in soft.topi[0, 0]) == [1, 5]
    assert np.isclose(float(soft.topw[0, 0].sum()),
                      float(torch.softmax(logits[0], -1)[[1, 5]].sum()))


def test_parity_glm_checkpoints_differ_only_in_bias(tmp_path) -> None:
    """parity_glm.py 的参考 checkpoint 只改了部分层不驻留专家的纠偏 bias，其余逐位相同。"""
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file

    from vllm.p2p import parity_glm as G
    ck, mk = G.write_checkpoints(tmp_path, "half")
    a, b = load_file(str(ck / "model.safetensors")), load_file(str(mk / "model.safetensors"))
    assert set(a) == set(b)
    diff = sorted(k for k in a if not torch.equal(a[k], b[k]))
    assert diff == [f"model.layers.{l}.mlp.gate.e_score_correction_bias" for l in sorted(G.PARTIAL)]
    for l, keep in G.PARTIAL.items():
        bb = b[f"model.layers.{l}.mlp.gate.e_score_correction_bias"]
        assert all((bb[e] == -1e4) == (e not in keep) for e in range(16))
    assert all(len(v) >= G.PARITY_CFG["num_experts_per_tok"] for v in G.PARTIAL.values())
    cfg = json.loads((ck / "config.json").read_text())
    assert "rope_theta" not in cfg and cfg["rope_parameters"]["rope_theta"] == 1e6
