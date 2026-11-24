# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA prefill 的纯 torch 回退 —— 给没有 FlashAttention 的 GPU（Turing：T4、RTX 6000）用。

MLA 的 decode 走 Triton（``TritonMLAImpl``），哪张卡都行；但 prefill（新 token 之间、新 token 对
已缓存上下文）在 vLLM 里只有 FlashAttention / FlashInfer / cuDNN / TRT-LLM 几条路，它们都要
sm80 以上。GLM-4.7-Flash 是 MLA 模型，要把它的层放到 Turing 卡上就得有一条不挑卡的 prefill。

这里逐条序列、按查询分块算标准注意力（fp32 累加）：

    s = q·kᵀ·scale（causal 时把「未来」置 −∞）→ lse = logsumexp(s) → o = softmax(s)·v

返回的形状与 FlashAttention varlen 一致：``out`` 为 ``[T, H, Dv]``、``lse`` 为 ``[H, T]``（fp32），
所以 MLA 里 ``merge_attn_states`` 那套分块上下文合并原样可用。``v`` 不需要补齐到 q 的维度。

慢（每条序列一次 python 循环、不融合），但 prefill 在本方案里只占小头（decode 才是大头），
而且只在 Turing 卡上启用。显存：每块 ``H × q_block × Lk`` 个 fp32。
"""

from __future__ import annotations

import torch

__all__ = ["torch_varlen_attention"]

_Q_BLOCK = 256


def _lens(cu: torch.Tensor | list[int]) -> list[int]:
    return [int(x) for x in (cu.tolist() if isinstance(cu, torch.Tensor) else cu)]


def torch_varlen_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                           cu_seqlens_q, cu_seqlens_k, softmax_scale: float, causal: bool,
                           return_softmax_lse: bool = False, q_block: int = _Q_BLOCK):
    """q:[Tq,H,D] k:[Tk,H,D] v:[Tk,H,Dv]，按 cu_seqlens 切成若干条序列。

    causal：第 i 个查询（序列内）能看到的 key 是前 ``Lk − Lq + i + 1`` 个 —— 与 FlashAttention
    「右下对齐」的 causal 相同（新 token 对自己时 Lk = Lq，就是普通的下三角）。"""
    cq, ck = _lens(cu_seqlens_q), _lens(cu_seqlens_k)
    if len(cq) != len(ck):
        raise ValueError(f"cu_seqlens_q 与 cu_seqlens_k 条数不同：{len(cq)} vs {len(ck)}")
    T, H = q.shape[0], q.shape[1]
    Dv = v.shape[-1]
    out = q.new_zeros((T, H, Dv))
    lse = torch.full((H, T), float("-inf"), dtype=torch.float32, device=q.device)
    for b in range(len(cq) - 1):
        q0, q1, k0, k1 = cq[b], cq[b + 1], ck[b], ck[b + 1]
        Lq, Lk = q1 - q0, k1 - k0
        if Lq == 0:
            continue
        if Lk == 0:
            continue                                   # 没有可看的 key：输出 0、lse −∞（合并时权重为 0）
        kb = k[k0:k1].float().permute(1, 2, 0)         # [H, D, Lk]
        vb = v[k0:k1].float().permute(1, 0, 2)         # [H, Lk, Dv]
        for s0 in range(0, Lq, q_block):
            s1 = min(Lq, s0 + q_block)
            qb = q[q0 + s0:q0 + s1].float().permute(1, 0, 2)     # [H, n, D]
            s = torch.matmul(qb, kb) * softmax_scale             # [H, n, Lk]
            if causal:
                qi = torch.arange(s0, s1, device=q.device).unsqueeze(1)        # 序列内的查询号
                kj = torch.arange(Lk, device=q.device).unsqueeze(0)
                s = s.masked_fill(kj > qi + (Lk - Lq), float("-inf"))
            l = torch.logsumexp(s, dim=-1)                        # [H, n]
            p = torch.exp(s - l.unsqueeze(-1))
            p = torch.nan_to_num(p, nan=0.0)                      # 整行都被遮住时 l = −∞
            o = torch.matmul(p, vb)                               # [H, n, Dv]
            out[q0 + s0:q0 + s1] = o.permute(1, 0, 2).to(out.dtype)
            lse[:, q0 + s0:q0 + s1] = l
    if return_softmax_lse:
        return out, lse
    return out
