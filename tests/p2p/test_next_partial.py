# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-Next 的部分专家 / 分段模型（`qwen3_next_partial.py`）。CPU 上真的建 vLLM 的模块（子进程里，
平台强制成 CPU）。GPU 上的逐 token 对拍是 `python -m vllm.p2p.parity --family qwen3_next`。"""

import pytest

from tests.p2p.test_glm_partial import _run


@pytest.fixture(scope="module")
def r():
    return _run("next_partial.py")


def test_next_block_routing(r) -> None:
    """原生块 = 手写参考（softmax、top-k、重归一 + sigmoid 门 × 共享专家）；部分专家块 = 「非驻留置
    −∞」的参考；真实路由落在驻留集里的 token 与全装逐位相同。"""
    assert r["full_vs_ref"] < 1e-4 and r["part_vs_masked_ref"] < 1e-4
    assert r["part_vs_full_on_covered"] == 0.0 and r["n_local"] == 3


def test_next_stage_and_loading(r) -> None:
    assert r["layer_types"] == ["linear_attention"] * 3 + ["full_attention"]
    assert r["back_layers"] == ["PPMissingLayer"] * 2 + ["Qwen3NextDecoderLayer"] * 2
    assert r["back_embed"] == "ZeroEmbed" and r["back_head"] == "ParallelLMHead"
    assert r["back_resident"] == [[1, 4, 6], [0, 7]] and r["back_rows_ok"]
    assert r["back_shared_gate_ok"] and r["back_full_router"] == [8, 64]
    assert r["back_mixers"] == [True, True]            # 第 2 层线性注意力、第 3 层全注意力都在
    assert r["front_layers"] == ["Qwen3NextDecoderLayer"] * 2 + ["PPMissingLayer"] * 2
    assert r["front_norm"] == "AddResidual" and r["front_head"] == "PPMissingLayer"
    assert r["front_full"] and r["missing_raises"]
