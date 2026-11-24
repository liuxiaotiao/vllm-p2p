# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA prefill 的纯 torch 回退（Turing 卡用）：与逐条手算的标准注意力一致；分块上下文按 lse 合并后
等于一次看全部 key；输出 / lse 的形状与 FlashAttention varlen 相同。"""

import importlib.util
from pathlib import Path

import torch

_P = Path(__file__).resolve().parents[2] / "vllm/v1/attention/backends/mla/torch_prefill.py"
_spec = importlib.util.spec_from_file_location("torch_prefill", _P)
TP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TP)


def _ref(q, k, v, scale, causal):
    s = torch.einsum("qhd,khd->hqk", q.double(), k.double()) * scale
    if causal:
        Lq, Lk = q.shape[0], k.shape[0]
        m = torch.arange(Lk)[None, :] > torch.arange(Lq)[:, None] + (Lk - Lq)
        s = s.masked_fill(m, float("-inf"))
    lse = torch.logsumexp(s, -1)
    o = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), v.double())
    return o, lse


def test_matches_reference_causal_varlen_with_blocks() -> None:
    g = torch.Generator().manual_seed(0)
    H, D, Dv = 3, 24, 16
    lens = [5, 1, 37]
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    q = torch.randn(cu[-1], H, D, generator=g)
    k = torch.randn(cu[-1], H, D, generator=g)
    v = torch.randn(cu[-1], H, Dv, generator=g)
    out, lse = TP.torch_varlen_attention(q, k, v, cu_seqlens_q=torch.tensor(cu), cu_seqlens_k=cu,
                                         softmax_scale=0.3, causal=True, return_softmax_lse=True,
                                         q_block=8)
    assert out.shape == (cu[-1], H, Dv) and lse.shape == (H, cu[-1]) and lse.dtype == torch.float32
    for b in range(len(lens)):
        a, z = cu[b], cu[b + 1]
        o, l = _ref(q[a:z], k[a:z], v[a:z], 0.3, True)
        assert torch.allclose(out[a:z].double(), o, atol=1e-5)
        assert torch.allclose(lse[:, a:z].double(), l, atol=1e-5)
    assert torch.equal(TP.torch_varlen_attention(q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                                 softmax_scale=0.3, causal=True), out)


def test_context_chunks_merged_by_lse_equal_full_attention() -> None:
    """新 token 对上下文（不遮）分两块算、按 lse 合并，再并上新 token 之间（causal），
    等于一次看「上下文 + 新 token」的 causal 注意力 —— MLA 分块 prefill 正是这么拼的。"""
    g = torch.Generator().manual_seed(1)
    H, D, Dv, C, N = 2, 16, 8, 20, 6
    qn = torch.randn(N, H, D, generator=g)
    kc, vc = torch.randn(C, H, D, generator=g), torch.randn(C, H, Dv, generator=g)
    kn, vn = torch.randn(N, H, D, generator=g), torch.randn(N, H, Dv, generator=g)
    parts = []
    for a, z in ((0, 12), (12, C)):
        parts.append(TP.torch_varlen_attention(qn, kc[a:z], vc[a:z], cu_seqlens_q=[0, N],
                                               cu_seqlens_k=[0, z - a], softmax_scale=0.25,
                                               causal=False, return_softmax_lse=True))
    parts.append(TP.torch_varlen_attention(qn, kn, vn, cu_seqlens_q=[0, N], cu_seqlens_k=[0, N],
                                           softmax_scale=0.25, causal=True,
                                           return_softmax_lse=True))
    lse = torch.logsumexp(torch.stack([l for _, l in parts]), 0)               # [H, N]
    out = sum(o * torch.exp(l - lse).T.unsqueeze(-1) for o, l in parts)
    full_o, full_l = _ref(qn, torch.cat([kc, kn]), torch.cat([vc, vn]), 0.25, True)
    assert torch.allclose(out.double(), full_o, atol=1e-5)
    assert torch.allclose(lse.double(), full_l, atol=1e-5)


def test_empty_sequences_are_safe() -> None:
    q = torch.randn(3, 1, 4)
    out, lse = TP.torch_varlen_attention(q, q, q, cu_seqlens_q=[0, 0, 3], cu_seqlens_k=[0, 2, 2],
                                         softmax_scale=1.0, causal=False, return_softmax_lse=True)
    assert torch.equal(out, torch.zeros_like(out)) and torch.isinf(lse).all()
