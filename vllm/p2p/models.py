# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""p2p 工具链认得的模型族。**纯 Python，不 import torch / vllm**（layout.py 在控制机上也要能用）。

每个模型族说清楚四件事：

* 部分专家 + 分段的架构名（``--hf-overrides`` 里的 ``architectures``）与原生架构名；
* config.json 里专家数、top-k 叫什么（Qwen 系 ``num_experts``，GLM / DeepSeek 系 ``n_routed_experts``）；
* 哪些层没有专家（稠密层）—— ``p2p_layer_experts`` 不能列它们，逐专家核对、统计也跳过；
* 默认的注意力后端：要能在 Turing（T4 / RTX 6000，没有 FlashAttention）上跑、所有通道同一套 kernel。

    qwen3_moe      Qwen3-30B-A3B           TRITON_ATTN    48 层 × 128 专家，top-8
    glm4_moe_lite  GLM-4.7-Flash           TRITON_MLA     47 层（第 0 层稠密）× 64 路由专家 + 1 共享，top-4
    qwen3_next     Qwen3-Next-80B-A3B      TRITON_ATTN    48 层（3/4 线性注意力）× 512 专家 + 1 共享，top-10
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

__all__ = ["Family", "FAMILIES", "family_of", "family_by_name", "family_by_shape", "load_config",
           "n_experts_of", "top_k_of", "dense_layers", "DEFAULT_FAMILY", "KNOWN_SHAPES"]


@dataclass(frozen=True)
class Family:
    name: str
    arch: str                     # 部分专家 + 分段（本仓库登记的）
    stock_arch: str               # vLLM 原生
    model_types: tuple[str, ...]
    experts_key: str
    attention_backend: str
    scoring: str                  # 路由打分：softmax / sigmoid（统计口径）
    shared_expert: bool
    note: str = ""
    default_dense: tuple[int, ...] = ()   # 读不到 config.json 时用的稠密层（0-based）


FAMILIES: dict[str, Family] = {f.name: f for f in (
    Family("qwen3_moe", "Qwen3MoePartialForCausalLM", "Qwen3MoeForCausalLM", ("qwen3_moe",),
           "num_experts", "TRITON_ATTN", "softmax", False, "Qwen3-30B-A3B"),
    Family("glm4_moe_lite", "Glm4MoeLitePartialForCausalLM", "Glm4MoeLiteForCausalLM",
           ("glm4_moe_lite",), "n_routed_experts", "TRITON_MLA", "sigmoid", True,
           "GLM-4.7-Flash（MLA 注意力；Turing 上 prefill 走纯 torch）", (0,)),
    Family("qwen3_next", "Qwen3NextPartialForCausalLM", "Qwen3NextForCausalLM", ("qwen3_next",),
           "num_experts", "TRITON_ATTN", "softmax", True,
           "Qwen3-Next-80B-A3B（3/4 层是 Gated DeltaNet 线性注意力）"),
)}
DEFAULT_FAMILY = FAMILIES["qwen3_moe"]

# 规划器清单的 model 字段（"47L-d2048-E64"）→ 模型族：控制机上读不到 config.json 时靠它认，
# 免得把 GLM 的清单当成 Qwen 起（架构名错了 vLLM 照样会起，只是权重名对不上）
KNOWN_SHAPES: dict[tuple[int, int], str] = {(48, 128): "qwen3_moe", (47, 64): "glm4_moe_lite",
                                            (48, 512): "qwen3_next"}


def family_by_shape(n_layers: int | None, n_experts: int | None) -> Family | None:
    name = KNOWN_SHAPES.get((int(n_layers or 0), int(n_experts or 0)))
    return FAMILIES[name] if name else None


def family_by_name(name: str) -> Family:
    if name not in FAMILIES:
        raise ValueError(f"不认识的模型族 {name!r}；认得的有 {sorted(FAMILIES)}")
    return FAMILIES[name]


def family_of(cfg: Mapping | None) -> Family:
    """按 config.json 的 model_type（其次 architectures）认模型族；给了却认不出来时报错（不猜）。
    ``cfg`` 为 None（没有 config.json 可读）或两者都没写时返回默认的 Qwen3-MoE —— 与加这个模块之前的行为相同。"""
    if cfg is None:
        return DEFAULT_FAMILY
    mt = str(cfg.get("model_type") or "")
    for f in FAMILIES.values():
        if mt in f.model_types:
            return f
    archs = set(cfg.get("architectures") or ())
    for f in FAMILIES.values():
        if archs & {f.arch, f.stock_arch}:
            return f
    if not mt and not archs:
        return DEFAULT_FAMILY          # 只有形状字段的精简 config（测试、手写的）：按老行为当 Qwen3-MoE
    raise ValueError(f"p2p 工具链不认识这个模型（model_type={mt!r}，architectures="
                     f"{sorted(archs)}）；认得的有 {[f.note or f.name for f in FAMILIES.values()]}")


def load_config(model_dir: str | Path | None) -> dict | None:
    """读 ``<model_dir>/config.json``；目录不在本机（如 ``$WEIGHTS``）或没有这个文件时返回 None。"""
    if not model_dir:
        return None
    p = Path(str(model_dir)) / "config.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def n_experts_of(cfg: Mapping) -> int:
    for k in ("num_experts", "n_routed_experts"):
        if cfg.get(k):
            return int(cfg[k])
    raise ValueError("config 里没有 num_experts / n_routed_experts")


def top_k_of(cfg: Mapping) -> int:
    return int(cfg["num_experts_per_tok"])


def dense_layers(cfg: Mapping) -> frozenset[int]:
    """没有路由专家的层（0-based）。

    * Qwen3-MoE / Qwen3-Next：``mlp_only_layers`` 里的层，以及 ``(l + 1) % decoder_sparse_step != 0`` 的层；
    * GLM / DeepSeek：前 ``first_k_dense_replace`` 层，以及 ``l % moe_layer_freq != 0`` 的层。"""
    L = int(cfg["num_hidden_layers"])
    f = family_of(cfg)
    out: set[int] = set()
    if f.name == "glm4_moe_lite":
        k = int(cfg.get("first_k_dense_replace", 0) or 0)
        freq = int(cfg.get("moe_layer_freq", 1) or 1)
        out = {l for l in range(L) if l < k or l % freq != 0}
    else:
        out = {int(x) for x in (cfg.get("mlp_only_layers") or ())}
        step = int(cfg.get("decoder_sparse_step", 1) or 1)
        out |= {l for l in range(L) if (l + 1) % step != 0}
    return frozenset(out)
