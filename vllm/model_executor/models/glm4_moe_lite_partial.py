# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4.7-Flash（``Glm4MoeLiteForCausalLM``）**每层只装指定的那些专家**、可分前后段 ——
``qwen3_moe_partial.py`` 在 GLM 上的对应物。用法相同：

    vllm serve <ckpt> --hf-overrides '{"architectures": ["Glm4MoeLitePartialForCausalLM"],
                                       "p2p_layer_experts": {"5": [2, 3, 9, …], …},
                                       "p2p_stage": {"role": "back", "layers": [2, 47]}}'

与 Qwen3-MoE 版不同的地方
-------------------------
**路由是 sigmoid + 纠偏 bias。** 原生 ``Glm4MoE``：``s = sigmoid(gate(x))``，按 ``s + bias``
（``e_score_correction_bias``，只管挑谁）选 top-4，权重取被选中者的 ``s``、重归一，最后整体乘
``routed_scaling_factor``（1.8）。GLM-4.7-Flash 的 ``n_group = topk_group = 1``，分组不起作用。

驻留子集上的 local_topk：只在驻留者里按 ``s + bias`` 挑 top-k、权重重归一。sigmoid 是逐元素的，
所以这与「全量打分、非驻留置 −∞ 再挑」逐个相同；驻留集覆盖了真实路由时与全装一致（不同
kernel 的累加顺序除外）。这里不用 FusedMoE 自带的分组路由（它的融合 kernel 对专家数有要求、
又绑着全量的 bias），而是把 ``p2p_partial.sigmoid_bias_topk`` 作为 ``custom_routing_function``
交给 FusedMoE，bias 用本层驻留者那几个（``local_bias``，权重加载完从全量 bias 里取出来）。

**共享专家恒装。** 每层 1 个共享专家，不参与裁剪 —— 它属于这一层的固定成分，不属于哪个 task。
后段每个节点都装（部分 checkpoint 也要带上 ``mlp.shared_experts.*``）。

**第 0 层是稠密的**（``first_k_dense_replace = 1``）。``p2p_layer_experts`` 不能列它；
逐专家核对、统计钩子都跳过它。

**MTP 层**（第 47 层，``num_nextn_predict_layers = 1``）原生实现就跳过，这里也一样。

**注意力是 MLA。** 不能用 ``--attention-backend TRITON_ATTN``（那是 Qwen 用的）：A30 上自动选
``TRITON_MLA``；Turing（T4 / RTX 6000）上 decode 也走 ``TRITON_MLA``，prefill 走纯 torch 回退
（``vllm/v1/attention/backends/mla/torch_prefill.py``，那些卡没有 FlashAttention）。
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.models import glm4_moe_lite as _g
from vllm.model_executor.models import p2p_partial as _pp
from vllm.model_executor.models.glm4_moe import Glm4MoeMLP
from vllm.model_executor.models.utils import PPMissingLayer, extract_layer_index
from vllm.p2p import stage as _p2p_stage
from vllm.p2p import stats as _p2p_stats
from vllm.p2p.layout import LayoutError

logger = init_logger(__name__)

__all__ = ["Glm4MoeLitePartialForCausalLM", "PartialGlm4MoeLite"]


def _check_grouping(config) -> None:
    ng, tg = int(getattr(config, "n_group", 1) or 1), int(getattr(config, "topk_group", 1) or 1)
    if ng != 1 or tg != 1:
        raise ValueError(f"部分专家层只支持 n_group = topk_group = 1（这个模型是 {ng} / {tg}）："
                         "分组路由在驻留子集上没有明确的含义")


class PartialGlm4MoeLite(_g.Glm4MoeLite):
    """一层 MoE。层号在驻留表里 → 只建 n 个路由专家（共享专家照建）；不在 → 完全走原生构造。

    继承原生类，好让 ``Glm4MoeLiteForCausalLM.set_moe_parameters`` 里的
    ``isinstance(layer.mlp, Glm4MoeLite)`` 照常成立。"""

    def __init__(self, config, quant_config=None, prefix: str = "", enable_eplb: bool = False):
        layer = extract_layer_index(prefix)
        resident = _pp.layer_experts_of(config).get(layer)
        st = _p2p_stats.get_collector()
        E = int(config.n_routed_experts)
        if resident is None:
            super().__init__(config=config, quant_config=quant_config, prefix=prefix,
                             enable_eplb=enable_eplb)
            self.resident: tuple[int, ...] | None = None
            self.p2p_slot: int | None = None
            if st is not None:
                if self.enable_eplb or self.ep_size != 1:
                    raise ValueError("统计钩子不支持 EP / EPLB")
                self.p2p_slot = st.register_layer(layer, None)
                self.register_buffer("resident_idx", torch.arange(E, dtype=torch.long),
                                     persistent=False)
            return

        nn.Module.__init__(self)
        _check_grouping(config)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        if self.ep_size != 1 or enable_eplb:
            raise ValueError("部分专家层不支持专家并行 / EPLB（它们会再切一遍专家）")
        if quant_config is not None:
            raise ValueError("部分专家层暂不支持量化 checkpoint")
        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

        self.resident = tuple(resident)
        n = len(self.resident)
        self.n_routed_experts = E
        self.n_shared_experts = config.n_shared_experts
        self.enable_eplb = False
        self.n_redundant_experts = 0
        self.n_logical_experts = E
        self.n_physical_experts = E
        self.n_local_physical_experts = n
        self.physical_expert_start = 0
        self.physical_expert_end = n

        # 路由器保持全量 E 维（fp32，与原生相同）：统计口径永远是「路由真正想要谁」
        self.gate = nn.Linear(config.hidden_size, E, bias=False, dtype=torch.float32)
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(E, dtype=torch.float32))
        # 本地号 i ↔ 全局号 resident[i]（升序）—— 与 layout.localize_expert_name 同一个顺序
        self.register_buffer("resident_idx", torch.tensor(self.resident, dtype=torch.long),
                             persistent=False)
        # 驻留者的纠偏 bias：权重加载完由 sync_local_bias 从全量 bias 里取（地址不变，CUDA Graph 安全）
        self.register_buffer("local_bias", torch.zeros(n, dtype=torch.float32), persistent=False)

        if config.n_shared_experts is not None:
            self.shared_experts = Glm4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=None,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        # 对 FusedMoE 来说这一层就是一个 n 专家的 MoE；路由由 custom_routing_function 做
        # （scoring_func 留默认：FusedMoE 只在分组路由里用它，非分组时还会拒收 sigmoid）
        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            num_experts=n,
            top_k=min(int(config.num_experts_per_tok), n),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=bool(config.norm_topk_prob),
            quant_config=None,
            use_grouped_topk=False,
            custom_routing_function=self._route,
            prefix=f"{prefix}.experts",
            routed_scaling_factor=1.0,           # 与原生一样在外面乘
            router_logits_dtype=torch.float32,
        )
        self.p2p_slot = st.register_layer(layer, self.resident) if st is not None else None

    def _route(self, hidden_states: torch.Tensor, gating_output: torch.Tensor, topk: int,
               renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
        return _pp.sigmoid_bias_topk(gating_output, self.local_bias, topk, renormalize)

    @torch.no_grad()
    def sync_local_bias(self) -> None:
        if self.resident is not None:
            self.local_bias.copy_(
                self.gate.e_score_correction_bias.detach().float().index_select(
                    0, self.resident_idx.to(self.gate.e_score_correction_bias.device)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.resident is None and self.p2p_slot is None:
            return super().forward(hidden_states)
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states.to(dtype=torch.float32))      # [T, E] 全量
        if self.p2p_slot is not None:
            local_logits = torch.ops.vllm.p2p_route_select_bias(
                router_logits, self.resident_idx, self.gate.e_score_correction_bias,
                self.p2p_slot)
        else:
            local_logits = router_logits.index_select(-1, self.resident_idx)
        fused_moe_out = self.experts(hidden_states=hidden_states, router_logits=local_logits)
        if self.shared_experts is not None:
            shared_output, final_hidden_states = fused_moe_out
            final_hidden_states = (final_hidden_states * self.routed_scaling_factor
                                   + shared_output)
        else:
            final_hidden_states = fused_moe_out * self.routed_scaling_factor
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)


class _PartialBlocks:
    """构造期间把 glm4_moe_lite 模块里的 MoE 块换成部分专家版（DecoderLayer 按模块全局名取它）。"""

    def __enter__(self):
        self.orig = _g.Glm4MoeLite
        _g.Glm4MoeLite = PartialGlm4MoeLite
        return self

    def __exit__(self, *exc):
        _g.Glm4MoeLite = self.orig
        return False


class Glm4MoeLitePartialForCausalLM(_g.Glm4MoeLiteForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        policy = getattr(config, "p2p_miss_policy", "local_topk")
        if policy != "local_topk":
            raise ValueError(f"p2p_miss_policy={policy!r} 暂不支持；只有 local_topk")
        try:
            le = _pp.layer_experts_of(config)
        except LayoutError as e:
            raise ValueError(f"hf_overrides 的 p2p_layer_experts 有误：{e}") from e
        dense = [l for l in le if l < int(config.first_k_dense_replace)]
        if dense:
            raise ValueError(f"p2p_layer_experts 列了稠密层 {dense}（GLM 前 "
                             f"{config.first_k_dense_replace} 层没有专家）")
        try:
            stage = _p2p_stage.stage_of(config)
        except ValueError as e:
            raise ValueError(f"hf_overrides 的 p2p_stage 有误：{e}") from e
        st = _pp.make_route_stats(config, n_experts=int(config.n_routed_experts),
                                  top_k=int(config.num_experts_per_tok),
                                  norm_topk=bool(config.norm_topk_prob), scoring="sigmoid")
        _p2p_stats.set_collector(st)
        with _PartialBlocks(), _pp.stage_layers(_g, stage):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.p2p_layer_experts = le
        self.p2p_stage = stage
        _pp.allocate_route_stats(st, vllm_config)
        _pp.trim_for_stage(self, config, stage, vllm_config)
        if stage is not None:
            logger.info("p2p 分段引擎（GLM-4.7-Flash）：%s，本段全局层 [%d, %d)，本 PP 级 [%d, %d)",
                        stage.role, stage.lo, stage.hi, self.model.start_layer,
                        self.model.end_layer)
        here = range(self.model.start_layer, self.model.end_layer)
        part = [l for l in here if l in le]
        if part:
            avg = sum(len(le[l]) for l in part) / len(part)
            logger.info("p2p 部分专家（GLM）：层 [%d, %d)，其中 %d 层只装子集（层均 %.1f/%d 个路由专家）",
                        here.start, here.stop, len(part), avg, int(config.n_routed_experts))

    def set_moe_parameters(self):
        """原生版找不到 MoE 层就报错；分段 / PP 下某一级可能只有第 0 层（稠密），那就没有。"""
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, _g.Glm4MoeLite):
                example = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        if example is not None:
            self.extract_moe_parameters(example)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if isinstance(self.lm_head, PPMissingLayer):
            return None
        return super().compute_logits(hidden_states)

    def _is_moe(self, l: int) -> bool:
        layer = self.model.layers[l]
        return not isinstance(layer, PPMissingLayer) and isinstance(layer.mlp, _g.Glm4MoeLite)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        le = self.p2p_layer_experts
        lo, hi = self.model.start_layer, self.model.end_layer
        seen = _pp.new_seen()
        skip = _pp.stage_skip_prefixes(self.p2p_stage)
        loaded = super().load_weights(_pp.localized_weights(weights, le, lo, hi, skip, seen))
        _pp.check_resident_weights(lo, hi, le, seen, int(self.config.n_routed_experts),
                                   self._is_moe)
        for l in range(lo, hi):
            if self._is_moe(l) and isinstance(self.model.layers[l].mlp, PartialGlm4MoeLite):
                self.model.layers[l].mlp.sync_local_bias()
        return loaded
