# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-Next（``Qwen3NextForCausalLM``）**每层只装指定的那些专家**、可分前后段 ——
``qwen3_moe_partial.py`` 在 Qwen3-Next 上的对应物。用法相同：

    vllm serve <ckpt> --hf-overrides '{"architectures": ["Qwen3NextPartialForCausalLM"],
                                       "p2p_layer_experts": {"12": [2, 3, 9, …], …},
                                       "p2p_stage": {"role": "back", "layers": [12, 48]}}'

与 Qwen3-MoE 版相同的地方
-------------------------
**路由是 softmax → top-k → 重归一**（``norm_topk_prob``），所以驻留子集上的 local_topk
（只把驻留者那几列打分交给 FusedMoE）与「全量 softmax、非驻留置 −∞、top-k、重归一」逐个相同 ——
理由见 ``qwen3_moe_partial.py`` 的模块文档。统计钩子用同一个自定义算子 ``vllm.p2p_route_select``。

不同的地方
----------
**512 个专家、top-10。** 只是数大了，做法不变。

**共享专家恒装，且带门。** 每层 1 个共享专家（``shared_expert``，中间维 512），输出再乘
``sigmoid(shared_expert_gate(x))``。它不参与裁剪：后段每个节点都装（部分 checkpoint 也要带上
``mlp.shared_expert.*`` 与 ``mlp.shared_expert_gate.*``）。

**混合层。** 每 4 层里 3 层是 Gated DeltaNet（线性注意力，``linear_attn.*``，状态定长：一个
[头, k_dim, v_dim] 的矩阵 + conv 窗口），第 4 层是带输出门的全注意力（``self_attn.*``）。
分段 / PP 时每一级只建自己的层，vLLM 按本级实际建出的层分配 KV 页与线性注意力的状态槽
（层外的是 ``PPMissingLayer``，不登记）。前段交出去的仍是残差流 h + r（final norm 换成
``AddResidual``），后段从那里接着算 —— 线性注意力的状态在各自的层里、由各自的引擎维护，
不需要跨段传。

**不支持前缀缓存**（原生 Qwen3-Next 就不支持；分段引擎本来也关着）。
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models import p2p_partial as _pp
from vllm.model_executor.models import qwen3_moe_partial as _qp  # noqa: F401  登记 vllm.p2p_route_select
from vllm.model_executor.models import qwen3_next as _n
from vllm.model_executor.models.utils import PPMissingLayer, extract_layer_index
from vllm.p2p import stage as _p2p_stage
from vllm.p2p import stats as _p2p_stats
from vllm.p2p.layout import LayoutError

logger = init_logger(__name__)

__all__ = ["Qwen3NextPartialForCausalLM", "PartialQwen3NextSparseMoeBlock", "dense_layers_of"]


def dense_layers_of(config) -> set[int]:
    """没有路由专家的层（0-based）：``mlp_only_layers`` 与 ``decoder_sparse_step`` 之外的层。
    Qwen3-Next-80B 两者都没有（48 层全是 MoE）。"""
    L = int(config.num_hidden_layers)
    out = {int(x) for x in (getattr(config, "mlp_only_layers", None) or ())}
    step = int(getattr(config, "decoder_sparse_step", 1) or 1)
    out |= {l for l in range(L) if (l + 1) % step != 0}
    if int(getattr(config, "num_experts", 0) or 0) <= 0:
        out = set(range(L))
    return out


class PartialQwen3NextSparseMoeBlock(_n.Qwen3NextSparseMoeBlock):
    """一层 MoE。层号在驻留表里 → 只建 n 个路由专家（共享专家与它的门照建）；不在 → 原生构造。

    继承原生类，好让 ``set_moe_parameters`` / ``update_physical_experts_metadata`` 里的
    ``isinstance(layer.mlp, Qwen3NextSparseMoeBlock)`` 照常成立。"""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        layer = extract_layer_index(prefix)
        resident = _pp.layer_experts_of(config).get(layer)
        st = _p2p_stats.get_collector()
        E = int(config.num_experts)
        if resident is None:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self.resident: tuple[int, ...] | None = None
            self.p2p_slot: int | None = None
            if st is not None:
                if self.is_sequence_parallel or self.enable_eplb or self.ep_size != 1:
                    raise ValueError("统计钩子不支持 sequence-parallel MoE / EP / EPLB")
                self.p2p_slot = st.register_layer(layer, None)
                self.register_buffer("resident_idx", torch.arange(E, dtype=torch.long),
                                     persistent=False)
            return

        nn.Module.__init__(self)
        parallel_config = vllm_config.parallel_config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        if self.ep_size != 1 or parallel_config.enable_eplb:
            raise ValueError("部分专家层不支持专家并行 / EPLB（它们会再切一遍专家）")
        if parallel_config.use_sequence_parallel_moe:
            raise ValueError("部分专家层不支持 sequence-parallel MoE")
        if vllm_config.quant_config is not None:
            raise ValueError("部分专家层暂不支持量化 checkpoint")

        self.is_sequence_parallel = False
        self.enable_eplb = False
        self.resident = tuple(resident)
        n = len(self.resident)
        self.n_routed_experts = E
        self.n_logical_experts = E
        self.n_redundant_experts = 0
        self.n_physical_experts = E
        self.n_local_physical_experts = n
        self.physical_expert_start = 0
        self.physical_expert_end = n

        # 路由器保持全量 E 维：统计口径永远是「路由真正想要谁」
        self.gate = ReplicatedLinear(config.hidden_size, E, bias=False, quant_config=None,
                                     prefix=f"{prefix}.gate")
        self.shared_expert_gate = ReplicatedLinear(config.hidden_size, 1, bias=False,
                                                   quant_config=None,
                                                   prefix=f"{prefix}.shared_expert_gate")
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = _n.Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=None,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert = None
        # 对 FusedMoE 来说这一层就是一个 n 专家的 MoE。renormalize 恒为 True：local_topk 的定义
        # 里就有重归一（Qwen3-Next 的 norm_topk_prob 本来也是 True）
        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            num_experts=n,
            top_k=min(int(config.num_experts_per_tok), n),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=True,
            quant_config=None,
            prefix=f"{prefix}.experts",
            routing_method_type=RoutingMethodType.Renormalize,
        )
        # 本地号 i ↔ 全局号 resident[i]（升序）—— 与 layout.localize_expert_name 同一个顺序
        self.register_buffer("resident_idx", torch.tensor(self.resident, dtype=torch.long),
                             persistent=False)
        self.p2p_slot = st.register_layer(layer, self.resident) if st is not None else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.resident is None and self.p2p_slot is None:
            return super().forward(hidden_states)
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits, _ = self.gate(hidden_states)                     # [T, E] 全量
        if self.p2p_slot is None:
            local_logits = router_logits.index_select(-1, self.resident_idx)
        else:
            local_logits = torch.ops.vllm.p2p_route_select(
                router_logits, self.resident_idx, self.p2p_slot)
        out = self.experts(hidden_states=hidden_states, router_logits=local_logits)
        if self.shared_expert is not None:
            out = out[0] + out[1]
        if self.tp_size > 1:
            out = self.experts.maybe_all_reduce_tensor_model_parallel(out)
        return out.view(orig_shape)


class _PartialBlocks:
    """构造期间把 qwen3_next 模块里的 MoE 块换成部分专家版（DecoderLayer 按模块全局名取它）。"""

    def __enter__(self):
        self.orig = _n.Qwen3NextSparseMoeBlock
        _n.Qwen3NextSparseMoeBlock = PartialQwen3NextSparseMoeBlock
        return self

    def __exit__(self, *exc):
        _n.Qwen3NextSparseMoeBlock = self.orig
        return False


class Qwen3NextPartialForCausalLM(_n.Qwen3NextForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        policy = getattr(config, "p2p_miss_policy", "local_topk")
        if policy != "local_topk":
            raise ValueError(f"p2p_miss_policy={policy!r} 暂不支持；只有 local_topk")
        try:
            le = _pp.layer_experts_of(config)
        except LayoutError as e:
            raise ValueError(f"hf_overrides 的 p2p_layer_experts 有误：{e}") from e
        dense = sorted(set(le) & dense_layers_of(config))
        if dense:
            raise ValueError(f"p2p_layer_experts 列了没有路由专家的层 {dense}")
        try:
            stage = _p2p_stage.stage_of(config)
        except ValueError as e:
            raise ValueError(f"hf_overrides 的 p2p_stage 有误：{e}") from e
        st = _pp.make_route_stats(config, n_experts=int(config.num_experts),
                                  top_k=int(config.num_experts_per_tok),
                                  norm_topk=bool(getattr(config, "norm_topk_prob", True)),
                                  scoring="softmax")
        _p2p_stats.set_collector(st)
        with _PartialBlocks(), _pp.stage_layers(_n, stage):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.p2p_layer_experts = le
        self.p2p_stage = stage
        _pp.allocate_route_stats(st, vllm_config)
        _pp.trim_for_stage(self, config, stage, vllm_config)
        if stage is not None:
            logger.info("p2p 分段引擎（Qwen3-Next）：%s，本段全局层 [%d, %d)，本 PP 级 [%d, %d)",
                        stage.role, stage.lo, stage.hi, self.model.start_layer,
                        self.model.end_layer)
        here = range(self.model.start_layer, self.model.end_layer)
        part = [l for l in here if l in le]
        if part:
            avg = sum(len(le[l]) for l in part) / len(part)
            logger.info("p2p 部分专家（Qwen3-Next）：层 [%d, %d)，其中 %d 层只装子集（层均 %.1f/%d 个路由专家）",
                        here.start, here.stop, len(part), avg, int(config.num_experts))

    def set_moe_parameters(self):
        """原生版找不到 MoE 层就报错；分段 / PP 下某一级可能一层 MoE 都没有（层全在别处）。"""
        self.expert_weights = []
        self.moe_layers = []
        example = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, _n.Qwen3NextSparseMoeBlock):
                example = layer.mlp
                self.moe_layers.append(layer.mlp.experts)
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        E = int(self.config.num_experts)
        self.num_logical_experts = E
        self.num_routed_experts = E
        self.num_physical_experts = E
        self.num_redundant_experts = 0
        self.num_local_physical_experts = (example.n_local_physical_experts
                                           if example is not None else E)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if isinstance(self.lm_head, PPMissingLayer):
            return None
        return super().compute_logits(hidden_states)

    def _is_moe(self, l: int) -> bool:
        layer = self.model.layers[l]
        return (not isinstance(layer, PPMissingLayer)
                and isinstance(layer.mlp, _n.Qwen3NextSparseMoeBlock))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        le = self.p2p_layer_experts
        lo, hi = self.model.start_layer, self.model.end_layer
        seen = _pp.new_seen()
        skip = _pp.stage_skip_prefixes(self.p2p_stage) + ("mtp.",)
        loaded = super().load_weights(_pp.localized_weights(weights, le, lo, hi, skip, seen))
        _pp.check_resident_weights(lo, hi, le, seen, int(self.config.num_experts), self._is_moe)
        return loaded
