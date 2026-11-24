# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""在 CPU 上真的建 GLM-4.7-Flash 的 MoE 块（原生 / 部分专家）并前向，核对路由语义。
由 tests/p2p/test_glm_partial.py 在子进程里跑（要把 vLLM 的平台强制成 CPU，不能污染别的测试）。
打印一行 JSON。"""

import json
import os
import socket
import sys

import torch

import vllm.platforms as _plat
from vllm.platforms.cpu import CpuPlatform

_plat._current_platform = CpuPlatform()

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
init_distributed_environment(world_size=1, rank=0, local_rank=0,
                             distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo")
initialize_model_parallel(1, 1)

from vllm.model_executor.models import glm4_moe_lite_partial as P  # noqa: E402
from vllm.transformers_utils.configs.glm4_moe_lite import Glm4MoeLiteConfig  # noqa: E402

E, K, H, I = 8, 2, 64, 32
R = [1, 4, 6]
cfg = Glm4MoeLiteConfig(num_hidden_layers=4, hidden_size=H, intermediate_size=128,
                        moe_intermediate_size=I, n_routed_experts=E, num_experts_per_tok=K,
                        num_attention_heads=4, num_key_value_heads=4, q_lora_rank=32,
                        kv_lora_rank=16, qk_nope_head_dim=8, qk_rope_head_dim=8, v_head_dim=8,
                        vocab_size=100, routed_scaling_factor=1.8)
cfg.p2p_layer_experts = {"2": R}
vc = VllmConfig()
torch.manual_seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
with set_current_vllm_config(vc):
    full = P.PartialGlm4MoeLite(cfg, prefix="model.layers.1.mlp")      # 第 1 层：全装（原生构造）
    part = P.PartialGlm4MoeLite(cfg, prefix="model.layers.2.mlp")      # 第 2 层：只装 R
for p in full.parameters():
    p.data.normal_(0, 0.2)
with torch.no_grad():
    full.gate.e_score_correction_bias.normal_(0, 0.3)
    part.gate.weight.copy_(full.gate.weight)
    part.gate.e_score_correction_bias.copy_(full.gate.e_score_correction_bias)
    idx = torch.tensor(R)
    part.experts.w13_weight.copy_(full.experts.w13_weight[idx])
    part.experts.w2_weight.copy_(full.experts.w2_weight[idx])
    part.shared_experts.gate_up_proj.weight.copy_(full.shared_experts.gate_up_proj.weight)
    part.shared_experts.down_proj.weight.copy_(full.shared_experts.down_proj.weight)
w13 = full.experts.w13_weight.detach().clone()
w2 = full.experts.w2_weight.detach().clone()
sh_gu = full.shared_experts.gate_up_proj.weight.detach().clone()
sh_d = full.shared_experts.down_proj.weight.detach().clone()
gate_w = full.gate.weight.detach().clone()
bias = full.gate.e_score_correction_bias.detach().clone()


def reference(x, allowed):
    """手写的 GLM MoE：s = sigmoid(gate(x))，按 s+bias 在 allowed 里挑 top-k，权重 = s 重归一 × 1.8，加共享专家。"""
    s = torch.sigmoid(x.float() @ gate_w.float().T)
    choice = (s + bias).masked_fill(~allowed, float("-inf"))
    ids = choice.topk(min(K, int(allowed.sum())), dim=-1)[1]
    w = s.gather(-1, ids)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            g, u = (w13[e].float() @ x[t].float()).chunk(2)
            out[t] += w[t, j] * (w2[e].float() @ (torch.nn.functional.silu(g) * u))
    g, u = (sh_gu.float() @ x.float().T).T.chunk(2, dim=-1)
    shared = (torch.nn.functional.silu(g) * u) @ sh_d.float().T
    return out * 1.8 + shared, ids


x = torch.randn(16, H)
with set_current_vllm_config(vc), set_forward_context(None, vc):
    for mod in (full, part):
        for sm in mod.modules():
            qm = getattr(sm, "quant_method", None)
            if qm is not None and hasattr(qm, "process_weights_after_loading"):
                qm.process_weights_after_loading(sm)
    part.sync_local_bias()
    y_full = full(x.clone())
    y_part = part(x.clone())
    # parity_glm.py 的参考：原生块、把不驻留专家的纠偏 bias 改成 −1e4 —— 应当与部分专家块相同
    with torch.no_grad():
        nb = full.gate.e_score_correction_bias
        keep = torch.zeros(E, dtype=torch.bool)
        keep[R] = True
        nb.copy_(torch.where(keep, nb, torch.full_like(nb, -1e4)))
    y_masked = full(x.clone())
all_ok = torch.ones(E, dtype=torch.bool)
in_r = torch.zeros(E, dtype=torch.bool)
in_r[R] = True
ref_full, ids_full = reference(x, all_ok)
ref_part, _ = reference(x, in_r)
covered = [all(int(e) in R for e in row) for row in ids_full]
print(json.dumps({
    "full_vs_ref": float((y_full.float() - ref_full).abs().max()),
    "part_vs_masked_ref": float((y_part.float() - ref_part).abs().max()),
    "covered_tokens": sum(covered),
    "part_vs_full_on_covered": max([float((y_part[t].float() - y_full[t].float()).abs().max())
                                    for t in range(16) if covered[t]] or [0.0]),
    "local_bias_ok": bool(torch.equal(part.local_bias, bias[R])),
    "part_vs_masked_native": float((y_part.float() - y_masked.float()).abs().max()),
    "scale": float(part.routed_scaling_factor),
}))
