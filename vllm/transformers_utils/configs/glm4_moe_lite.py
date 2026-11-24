# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4.7-Flash（``model_type: glm4_moe_lite``）的 config。

vLLM 0.15 起直接用 transformers 的 ``Glm4MoeLiteConfig``，但那要 transformers ≥ 5；
本仓库（vLLM 0.14）配的 transformers 4.5x 不认识这个 model_type。结构上它就是
DeepSeek-V3 那一套（MLA 注意力 + 带纠偏的 sigmoid 路由 + 共享专家 + 头几层稠密），
所以这里直接继承 ``DeepseekV3Config``，只做两件事：

1. 默认值换成 GLM-4.7-Flash 的（47 层、64 个路由专家、top-4、1 个共享专家……）；
2. 认 transformers 5 风格的 ``rope_parameters``：GLM-4.7-Flash 的 config.json 里没有
   顶层 ``rope_theta``，θ 在 ``rope_parameters.rope_theta``（1e6）。不搬出来的话，
   vLLM 的 ``patch_rope_parameters`` 会用 DeepseekV3Config 的默认 10000 覆盖它 ——
   位置编码全错、模型照样能跑但输出是乱的。
"""

from __future__ import annotations

from typing import Any

from transformers import DeepseekV3Config


class Glm4MoeLiteConfig(DeepseekV3Config):
    model_type = "glm4_moe_lite"

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 2048,
        intermediate_size: int = 10240,
        moe_intermediate_size: int = 1536,
        num_hidden_layers: int = 47,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 20,
        n_shared_experts: int = 1,
        n_routed_experts: int = 64,
        routed_scaling_factor: float = 1.8,
        kv_lora_rank: int = 512,
        q_lora_rank: int | None = 768,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        qk_nope_head_dim: int = 192,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int = 4,
        first_k_dense_replace: int = 1,
        norm_topk_prob: bool = True,
        max_position_embeddings: int = 202752,
        rms_norm_eps: float = 1e-5,
        rope_theta: float | None = None,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rope_interleave: bool = True,
        pad_token_id: int | None = 154820,
        bos_token_id: int = 0,
        eos_token_id: int | list[int] = (154820, 154827, 154829),
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        rp = dict(rope_parameters) if isinstance(rope_parameters, dict) else None
        if rope_theta is None:
            rope_theta = float((rp or {}).get("rope_theta", 1_000_000.0))
        if rope_scaling is None and rp and rp.get("rope_type", rp.get("type", "default")) != "default":
            # 非 default 的（yarn 等）照原样交给 vLLM；θ 已经单独搬出来了
            rope_scaling = {k: v for k, v in rp.items() if k != "rope_theta"}
        if rp is not None and "partial_rotary_factor" in rp:
            kwargs.setdefault("partial_rotary_factor", rp["partial_rotary_factor"])
        if isinstance(eos_token_id, tuple):
            eos_token_id = list(eos_token_id)
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            moe_intermediate_size=moe_intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            n_shared_experts=n_shared_experts,
            n_routed_experts=n_routed_experts,
            routed_scaling_factor=routed_scaling_factor,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            n_group=n_group,
            topk_group=topk_group,
            num_experts_per_tok=num_experts_per_tok,
            first_k_dense_replace=first_k_dense_replace,
            norm_topk_prob=norm_topk_prob,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_interleave=rope_interleave,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # transformers 4.x 会把没认出来的 rope_parameters 原样挂成属性；它若还在，
        # vLLM 的 patch 会在它上面写 θ —— 先去掉，让 vLLM 从 rope_theta / rope_scaling 重建
        if hasattr(self, "rope_parameters") and rp is not None:
            try:
                delattr(self, "rope_parameters")
            except AttributeError:
                pass
