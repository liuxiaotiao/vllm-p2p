# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""几个「部分专家 + 分段」模型共用的零件（GLM-4.7-Flash、Qwen3-Next；Qwen3-MoE 那版早于它们，
自带一份同样的逻辑，见 ``qwen3_moe_partial.py``，那边的模块文档讲了每一件为什么这么做）。

* ``layer_experts_of``：从 HF config 读逐层驻留集（经 ``--hf-overrides`` 的 ``p2p_layer_experts``）；
* ``sigmoid_bias_topk``：sigmoid 打分 + 纠偏 bias 只用来挑专家（GLM / DeepSeek-V3 的路由）在
  驻留子集上的 top-k —— 与「全量路由把不驻留的置 −∞」逐个相同（sigmoid 是逐元素的，
  子集上挑 top-k 就是全量里挑驻留者的 top-k；权重取不带 bias 的 sigmoid、重归一）；
* ``p2p_route_select_bias``：自定义算子，顺带把全量 top-k 写进统计缓冲区（带 bias 的版本）；
* ``stage_layers``：构造期间换掉某个模型模块里的 ``make_layers``，只建本段（再按 PP 切）的层；
* ``trim_for_stage``：PP / 分段下不建用不到的 embed / norm / lm_head（前段交出残差流，后段零 embed）；
* ``localized_weights`` / ``check_resident_weights``：权重名换成本地专家号、逐专家核对缺失。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager

import torch
from torch import nn

from vllm.distributed import get_pp_group
from vllm.model_executor.models.utils import PPMissingLayer, maybe_offload_to_cpu
from vllm.p2p import stage as _p2p_stage
from vllm.p2p import stats as _p2p_stats
from vllm.p2p.layout import expert_key, localize_expert_name, parse_layer_experts
from vllm.utils.torch_utils import direct_register_custom_op

__all__ = ["layer_experts_of", "sigmoid_bias_topk", "stage_layers", "trim_for_stage",
           "localized_weights", "check_resident_weights", "make_route_stats",
           "allocate_route_stats", "n_routed_experts_of"]


# --------------------------------------------------------------------------- #
def n_routed_experts_of(config) -> int:
    """路由专家数：Qwen 系叫 num_experts，GLM / DeepSeek 叫 n_routed_experts。"""
    for k in ("num_experts", "n_routed_experts"):
        v = getattr(config, k, None)
        if v:
            return int(v)
    raise ValueError("config 里没有 num_experts / n_routed_experts")


_PARSED: dict[int, dict[int, tuple[int, ...]]] = {}


def layer_experts_of(config) -> dict[int, tuple[int, ...]]:
    """逐层驻留集（0-based）；没列出的层 = 全装。每个 config 对象只解析一次（缓存不挂在 config 上：
    vLLM 会哈希 / 序列化 config）。"""
    key = id(config)
    if key not in _PARSED:
        _PARSED[key] = parse_layer_experts(getattr(config, "p2p_layer_experts", None),
                                           int(config.num_hidden_layers),
                                           n_routed_experts_of(config))
    return _PARSED[key]


# --------------------------------------------------------------------------- #
def sigmoid_bias_topk(gating_output: torch.Tensor, bias: torch.Tensor | None, topk: int,
                      renormalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """GLM-4 / DeepSeek-V3（n_group = 1）的路由，在给定的这些专家上：

        s = sigmoid(logits)；按 s + bias 挑 top-k；权重 = 被挑中者的 s（不带 bias）；可选重归一。

    与 vLLM ``grouped_topk``（一个组）的 torch 实现同一套运算，fp32。缩放因子不在这里乘 ——
    模型在外面乘（与原生 ``Glm4MoE`` 相同）。返回 (权重 fp32, 本地专家号 int32)。"""
    scores = gating_output.float().sigmoid()
    choice = scores if bias is None else scores + bias.float().unsqueeze(0)
    k = min(int(topk), scores.shape[-1])
    ids = torch.topk(choice, k=k, dim=-1, sorted=False)[1]
    w = scores.gather(-1, ids)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return w.to(torch.float32), ids.to(torch.int32)


def p2p_route_select_bias(router_logits: torch.Tensor, resident_idx: torch.Tensor,
                          bias: torch.Tensor, slot: int) -> torch.Tensor:
    """返回 ``router_logits[:, resident_idx]``，顺带把全量 top-k（sigmoid + bias 口径）记进统计缓冲区。
    做成自定义算子的理由同 ``vllm.p2p_route_select``：副作用对 torch.compile 不可见、CUDA Graph 里随图重放。"""
    st = _p2p_stats.get_collector()
    if st is not None and slot >= 0:
        st.capture(slot, router_logits, bias)
    return router_logits.index_select(-1, resident_idx)


def _p2p_route_select_bias_fake(router_logits: torch.Tensor, resident_idx: torch.Tensor,
                                bias: torch.Tensor, slot: int) -> torch.Tensor:
    return router_logits.new_empty((router_logits.shape[0], resident_idx.shape[0]))


direct_register_custom_op(
    op_name="p2p_route_select_bias",
    op_func=p2p_route_select_bias,
    mutates_args=[],
    fake_impl=_p2p_route_select_bias_fake,
)


# --------------------------------------------------------------------------- #
@contextmanager
def stage_layers(module, stage: "_p2p_stage.StageSpec | None"):
    """构造期间换掉 ``module.make_layers``：只建本段（再按 PP 切）的层，段外是 ``PPMissingLayer``
    （vLLM 的 forward、权重加载、KV / 状态缓存分配都认它）。层号保持全局，权重按原名加载。"""
    if stage is None:
        yield
        return
    orig = module.make_layers

    def make_layers(num_hidden_layers, layer_fn, prefix):
        pp = get_pp_group()
        s, e = _p2p_stage.stage_layer_range(stage, pp.rank_in_group, pp.world_size)
        modules = nn.ModuleList(
            [PPMissingLayer() for _ in range(s)]
            + [maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{i}")) for i in range(s, e)]
            + [PPMissingLayer() for _ in range(e, num_hidden_layers)])
        return s, e, modules

    module.make_layers = make_layers
    try:
        yield
    finally:
        module.make_layers = orig


def trim_for_stage(lm: nn.Module, config, stage, vllm_config) -> None:
    """``lm`` 是 *ForCausalLM（有 ``.model.embed_tokens / .model.norm / .lm_head``）。

    PP：只建这一级用得到的（部分 checkpoint 里前段没有 lm_head、后段没有 embed —— vLLM 的
    加载器是严格的）。分段：前段的 final norm 换成 ``AddResidual``（交出 h + r）、不建 lm_head；
    后段第一级的 embed 换成 ``ZeroEmbed``（每个位置都由前段的 hidden 喂入）。"""
    pp = get_pp_group()
    tied = bool(getattr(config, "tie_word_embeddings", False))
    if not tied:
        if not pp.is_first_rank:
            lm.model.embed_tokens = PPMissingLayer()
        if not pp.is_last_rank:
            lm.lm_head = PPMissingLayer()
    if not pp.is_last_rank:
        lm.model.norm = PPMissingLayer()
    if stage is not None and stage.role == "front":
        if pp.is_last_rank:
            lm.model.norm = _p2p_stage.AddResidual()
        lm.lm_head = PPMissingLayer()
    if stage is not None and stage.role == "back" and pp.is_first_rank:
        if tied:
            raise ValueError("后段不能用 tie_word_embeddings 的模型（lm_head 借的是 embed 权重）")
        lm.model.embed_tokens = _p2p_stage.ZeroEmbed(
            config.hidden_size, vllm_config.model_config.dtype)


def stage_skip_prefixes(stage) -> tuple[str, ...]:
    """分段引擎换掉了的模块：checkpoint 里有，本段不收。"""
    if stage is None:
        return ()
    return ("model.norm.", "lm_head.") if stage.role == "front" else ("model.embed_tokens.",)


# --------------------------------------------------------------------------- #
def localized_weights(weights: Iterable[tuple[str, torch.Tensor]],
                      le: Mapping[int, tuple[int, ...]], lo: int, hi: int,
                      skip: tuple[str, ...], seen: dict[int, set]):
    """全局专家号 → 本层本地号；不驻留的专家不读；顺带记下 [lo, hi) 里每层收到了哪些 (本地号, 投影)。"""
    for name, w in weights:
        if skip and name.startswith(skip):
            continue
        new = localize_expert_name(name, le)
        if new is None:
            continue
        k = expert_key(new)
        if k is not None and lo <= k[0] < hi:
            seen[k[0]].add((k[1], k[2]))
        yield new, w


def check_resident_weights(lo: int, hi: int, le: Mapping[int, tuple[int, ...]],
                           seen: Mapping[int, set], n_experts: int,
                           is_moe: Callable[[int], bool]) -> None:
    """本级每个 MoE 层的每个本地专家都要有 gate/up/down 三份，缺一个就报（不拦的话那几行是
    未初始化的显存 —— 不报错，只是输出悄悄变差）。"""
    bad = []
    for l in range(lo, hi):
        if not is_moe(l):
            continue
        n = len(le[l]) if l in le else n_experts
        want = {(i, s) for i in range(n) for s in ("gate_proj", "up_proj", "down_proj")}
        miss = want - seen.get(l, set())
        if miss:
            gl = (lambda i: le[l][i]) if l in le else (lambda i: i)
            ex = sorted({gl(i) for i, _ in miss})
            bad.append(f"第 {l} 层（0-based）缺 {len(miss)} 份专家权重，全局专家号 {ex[:8]}"
                       + ("…" if len(ex) > 8 else ""))
    if bad:
        raise ValueError(
            "checkpoint 里缺驻留专家的权重 —— 不拦的话那几行是未初始化的显存，输出会悄悄变差：\n  "
            + "\n  ".join(bad[:6])
            + "\n这台的部分 checkpoint 多半是按别的驻留集拉的：用同一份清单重跑 p2pmoe.deploy.fetch")


def new_seen() -> dict[int, set]:
    return defaultdict(set)


# --------------------------------------------------------------------------- #
def make_route_stats(config, *, n_experts: int, top_k: int, norm_topk: bool,
                     scoring: str) -> "_p2p_stats.RouteStats | None":
    """hf_overrides 里有 ``p2p_stats`` 才建（A5 统计钩子 / B-1 前段的 LR 特征）。"""
    raw = getattr(config, "p2p_stats", None)
    if not raw:
        return None
    raw = raw if isinstance(raw, dict) else {}
    return _p2p_stats.RouteStats(
        n_experts=n_experts, top_k=top_k, norm_topk=norm_topk,
        lr_layers=raw.get("lr_layers", ()), channel=getattr(config, "p2p_channel", ""),
        pp_rank=get_pp_group().rank_in_group,
        emit_lr=bool(raw.get("emit_lr", False)), write=bool(raw.get("write", True)),
        prefill_only=bool(raw.get("prefill_only", False)), scoring=scoring,
        profile=bool(raw.get("profile", False)))


def allocate_route_stats(st, vllm_config) -> None:
    """模型构造完、CUDA Graph 捕获之前：统计缓冲区地址从此固定。"""
    if st is None:
        return
    sc = vllm_config.scheduler_config
    cc = vllm_config.compilation_config
    cap = max(int(sc.max_num_batched_tokens),
              int(getattr(cc, "max_cudagraph_capture_size", None) or 0)) + 16
    dev = (torch.device("cuda", torch.cuda.current_device())
           if torch.cuda.is_available() else torch.device("cpu"))
    st.allocate(cap, dev)
