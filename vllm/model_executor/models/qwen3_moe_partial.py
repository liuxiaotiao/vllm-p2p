# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-MoE 的 vLLM 实现，**每层只装指定的那些专家** —— p2p-framework `PartialExpertMoEBlock`
在 vLLM 0.14 上的对应物。

    vllm serve <ckpt> --hf-overrides '{"architectures": ["Qwen3MoePartialForCausalLM"],
                                       "p2p_layer_experts": {"8": [2, 3, 5, …], …}}'

配置（经 `--hf-overrides` 写进 HF config，所有 PP 级都读得到）：

* `p2p_layer_experts`：{0-based 层号字符串: 驻留专家 id 列表}。**没列出的层 = 全装**，
  走 vLLM 原生的 `Qwen3MoeSparseMoeBlock`，与官方实现逐位相同。
* `p2p_miss_policy`：目前只支持 `local_topk`（p2p-framework 的实验一直用它）。

为什么不用 EP 的 expert_map
---------------------------
`FusedMoE.__init__` 只在开启专家并行时才让 `local_num_experts` 小于全局数，权重又是
**在构造时**按 `local_num_experts` 分配的。单卡上 expert_map 为 None、权重按 128 个
分配 —— 事后再改 map 来不及，40 层全专家 24GB 装不下。所以这里每层的 `FusedMoE`
**直接以驻留专家数构造**：对它来说，这一层就是一个只有 n 个专家的 MoE。

路由：local_topk 的等价性
-------------------------
路由器 `mlp.gate` 仍是全量 E 维（统计要看全量路由，见 A5）。前向时把它的输出按驻留集
**取列**，再交给 `FusedMoE`（softmax → top-k → 重归一，`renormalize=True`）。
p2p-framework 的 local_topk 是：全量 softmax → 非驻留置 −∞ → top-k → 重归一。两者相同：

* 子集上的 softmax = 全量 softmax 限制到子集再除以子集总和 —— 单调、保比例；
* 所以 top-k 选出的是同一批专家，重归一后的权重也逐个相等。

于是「驻留集覆盖实际路由时与全装逐位一致」这条性质在这里同样成立（不同 kernel 的
累加顺序除外）。驻留专家少于 top_k 时取 min(top_k, n)，与 p2p-framework 相同。

PP 各级只建自己用得到的东西
---------------------------
原生 Qwen3-MoE 在**每一级**都建 embed_tokens、norm、lm_head。而 `p2pmoe.deploy.fetch`
拉下来的部分 checkpoint 里，前段节点没有 lm_head、后段节点没有 embed_tokens ——
vLLM 的加载器是严格的（`default_loader.py`：每个参数都要有权重），会直接报错。
这里按 Llama 的做法把用不到的换成 `PPMissingLayer`（加载器遇到它就跳过）。

逐专家核对
----------
严格检查是**按参数**的：`experts.w13_weight` 是 n 个专家堆在一起的一个参数，只要其中
一个专家加载过，它就算「已加载」。某台节点的部分 checkpoint 若缺了某个驻留专家，那一行
会是未初始化的显存 —— 不报错，只是输出悄悄变差。所以 `load_weights` 自己逐
(层, 专家, gate/up/down) 核对，缺一个就报。
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models import qwen3_moe as _q
from vllm.model_executor.models.utils import PPMissingLayer, extract_layer_index
from vllm.p2p.layout import LayoutError, localize_expert_name, parse_layer_experts

logger = init_logger(__name__)

__all__ = ["Qwen3MoePartialForCausalLM", "PartialQwen3MoeSparseMoeBlock",
           "SUPPORTED_MISS_POLICIES"]

SUPPORTED_MISS_POLICIES = ("local_topk",)

# 本级加载到的专家权重名（已换成本地号）→ (层, 本地专家号, gate/up/down)，逐专家核对用
_EXPERT = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.")
# TODO(drop / drop_noscale)：全量 softmax 取 top-k、缺失的权重置 0（drop_noscale）或
# 再重归一（drop）。走 FusedMoE 的 custom_routing_function，把全局专家号映射成本地号、
# 缺失的映到任一本地专家且权重为 0。实验没用到，先不做。


_PARSED: dict[int, dict[int, tuple[int, ...]]] = {}


def _layer_experts_of(config) -> dict[int, tuple[int, ...]]:
    """从 HF config 读逐层驻留集（0-based）。每个 config 对象只解析一次。

    缓存放在模块里而不是挂到 config 上：vLLM 会序列化 / 哈希 HF config（编译缓存的键），
    往上挂私有字段可能改变哈希或序列化结果。"""
    key = id(config)
    if key not in _PARSED:
        _PARSED[key] = parse_layer_experts(getattr(config, "p2p_layer_experts", None),
                                           int(config.num_hidden_layers),
                                           int(config.num_experts))
    return _PARSED[key]


# --------------------------------------------------------------------------- #
class PartialQwen3MoeSparseMoeBlock(_q.Qwen3MoeSparseMoeBlock):
    """一层 MoE。层号在驻留表里 → 只建 n 个专家；不在 → 完全走原生构造。

    继承原生类是为了让 `Qwen3MoeForCausalLM.__init__` 里的
    `isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)` 照常成立（它靠这个找 MoE 层）。
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        layer = extract_layer_index(prefix)
        resident = _layer_experts_of(config).get(layer)
        if resident is None:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            self.resident: tuple[int, ...] | None = None
            return

        nn.Module.__init__(self)
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        # 下面这几种并行方式都会再切一遍专家 / token，与「本层只有这 n 个」的语义冲突
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = _q.get_ep_group().device_group
        self.ep_rank = _q.get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        if self.ep_size != 1 or parallel_config.enable_eplb:
            raise ValueError("部分专家层不支持专家并行 / EPLB（它们会再切一遍专家）")
        if parallel_config.use_sequence_parallel_moe:
            raise ValueError("部分专家层不支持 sequence-parallel MoE")
        if quant_config is not None:
            raise ValueError("部分专家层暂不支持量化 checkpoint")

        self.is_sequence_parallel = False
        self.enable_eplb = False
        self.resident = tuple(resident)
        n = len(self.resident)
        E = int(config.num_experts)
        self.n_routed_experts = E
        self.n_logical_experts = E
        self.n_redundant_experts = 0
        self.n_physical_experts = E
        self.n_local_physical_experts = n
        self.physical_expert_start = 0
        self.physical_expert_end = n

        # 对 FusedMoE 来说这一层就是一个 n 专家的 MoE：权重只分配 n 份。
        # renormalize 恒为 True —— local_topk 的定义里就有重归一（见模块文档）。
        self.experts = FusedMoE(
            num_experts=n,
            top_k=min(int(config.num_experts_per_tok), n),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=True,
            renormalize=True,
            quant_config=None,
            prefix=f"{prefix}.experts",
            routing_method_type=RoutingMethodType.Renormalize,
        )
        # 路由器保持全量 E 维：统计口径永远是「路由真正想要谁」
        self.gate = ReplicatedLinear(
            config.hidden_size, E, bias=False, quant_config=None, prefix=f"{prefix}.gate",
        )
        # 本地号 i ↔ 全局号 resident[i]（升序）。权重加载用的是同一个顺序
        # （layout.localize_expert_name），两处必须一致。
        self.register_buffer(
            "resident_idx", torch.tensor(self.resident, dtype=torch.long), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.resident is None:
            return super().forward(hidden_states)
        assert hidden_states.dim() <= 2
        is_input_1d = hidden_states.dim() == 1
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        router_logits, _ = self.gate(hidden_states)                 # [T, E] 全量
        local_logits = router_logits.index_select(-1, self.resident_idx)   # [T, n]
        out = self.experts(hidden_states=hidden_states, router_logits=local_logits)
        return out.squeeze(0) if is_input_1d else out


@contextmanager
def _partial_blocks():
    """构造期间把 qwen3_moe 模块里的 MoE 块换成部分专家版。

    `Qwen3MoeDecoderLayer.__init__` 按**模块全局名**取 `Qwen3MoeSparseMoeBlock`，所以
    换掉这个名字就能在权重分配之前接管 —— 不必复制一整份 DecoderLayer / Model 的代码
    （复制的话 vLLM 升级时两份会悄悄分叉）。只在本模型的构造期间换，退出即恢复，
    同一进程里的原生 Qwen3-MoE 不受影响。
    """
    orig = _q.Qwen3MoeSparseMoeBlock
    _q.Qwen3MoeSparseMoeBlock = PartialQwen3MoeSparseMoeBlock
    try:
        yield
    finally:
        _q.Qwen3MoeSparseMoeBlock = orig


# --------------------------------------------------------------------------- #
class Qwen3MoePartialForCausalLM(_q.Qwen3MoeForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        policy = getattr(config, "p2p_miss_policy", "local_topk")
        if policy not in SUPPORTED_MISS_POLICIES:
            raise ValueError(f"p2p_miss_policy={policy!r} 暂不支持；可选 {SUPPORTED_MISS_POLICIES}")
        try:
            self_le = _layer_experts_of(config)
        except LayoutError as e:
            raise ValueError(f"hf_overrides 的 p2p_layer_experts 有误：{e}") from e
        with _partial_blocks():
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.p2p_layer_experts = self_le
        self.p2p_miss_policy = policy

        # PP：只建这一级用得到的（见模块文档「PP 各级只建自己用得到的东西」）
        pp = get_pp_group()
        tied = bool(config.tie_word_embeddings)
        if not tied:
            if not pp.is_first_rank:
                self.model.embed_tokens = PPMissingLayer()
            if not pp.is_last_rank:
                self.lm_head = PPMissingLayer()
        if not pp.is_last_rank:
            self.model.norm = PPMissingLayer()

        here = range(self.model.start_layer, self.model.end_layer)
        part = [l for l in here if l in self_le]
        if part:
            avg = sum(len(self_le[l]) for l in part) / len(part)
            logger.info(
                "p2p 部分专家：%s 层 [%d, %d)，其中 %d 层只装子集（层均 %.1f/%d 个专家），"
                "miss_policy=%s", getattr(config, "p2p_channel", ""), here.start, here.stop,
                len(part), avg, int(config.num_experts), policy)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if isinstance(self.lm_head, PPMissingLayer):
            return None
        return super().compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        le = self.p2p_layer_experts
        lo, hi = self.model.start_layer, self.model.end_layer
        seen: dict[int, set[tuple[int, str]]] = defaultdict(set)

        def localized():
            for name, w in weights:
                new = localize_expert_name(name, le)
                if new is None:
                    continue              # 不驻留的专家：不读进显存
                m = _EXPERT.match(new)
                if m is not None and lo <= int(m.group(1)) < hi:
                    seen[int(m.group(1))].add((int(m.group(2)), m.group(3)))
                yield new, w

        loaded = super().load_weights(localized())

        # 逐专家核对：本级每个 MoE 层的每个本地专家都要有 gate/up/down 三份
        bad = []
        for l in range(lo, hi):
            if not isinstance(self.model.layers[l].mlp, _q.Qwen3MoeSparseMoeBlock):
                continue                  # mlp_only_layers：稠密层，没有专家
            n = len(le[l]) if l in le else int(self.config.num_experts)
            want = {(i, s) for i in range(n) for s in ("gate_proj", "up_proj", "down_proj")}
            miss = want - seen.get(l, set())
            if miss:
                gl = (lambda i: le[l][i]) if l in le else (lambda i: i)
                ex = sorted({gl(i) for i, _ in miss})
                bad.append(f"第 {l} 层（0-based）缺 {len(miss)} 份专家权重，全局专家号 {ex[:8]}"
                           + ("…" if len(ex) > 8 else ""))
        if bad:
            raise ValueError(
                "checkpoint 里缺驻留专家的权重 —— 不拦的话那几行是未初始化的显存，"
                "输出会悄悄变差：\n  " + "\n  ".join(bad[:6])
                + "\n这台的部分 checkpoint 多半是按别的驻留集拉的：用同一份清单重跑 "
                  "p2pmoe.deploy.fetch")
        return loaded

