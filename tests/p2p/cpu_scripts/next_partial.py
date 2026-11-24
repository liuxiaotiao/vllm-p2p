# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""在 CPU 上建 Qwen3-Next 的部分专家模型：MoE 块的路由语义（原生 / 部分专家 / 手写参考），
整模型的切层、PP/分段裁剪、专家本地化、共享专家与它的门、缺专家时报错。
由 tests/p2p/test_next_partial.py 在子进程里跑，打印一行 JSON。"""

import json
import pathlib
import socket
import sys
import tempfile

import torch

import vllm.platforms as _plat
from vllm.platforms.cpu import CpuPlatform

_plat._current_platform = CpuPlatform()

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
init_distributed_environment(world_size=1, rank=0, local_rank=0,
                             distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo")
initialize_model_parallel(1, 1)

from vllm.model_executor.models import qwen3_next_partial as P  # noqa: E402
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig  # noqa: E402

L, H, E, K, I, SI, V = 4, 64, 8, 2, 32, 32, 100
R2, R3 = [1, 4, 6], [0, 7]
d = pathlib.Path(tempfile.mkdtemp())
cfg = Qwen3NextConfig(num_hidden_layers=L, hidden_size=H, intermediate_size=128,
                      moe_intermediate_size=I, shared_expert_intermediate_size=SI, num_experts=E,
                      num_experts_per_tok=K, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, linear_num_value_heads=4, linear_num_key_heads=2,
                      linear_value_head_dim=16, linear_key_head_dim=16, linear_conv_kernel_dim=4,
                      vocab_size=V, full_attention_interval=4, norm_topk_prob=True,
                      decoder_sparse_step=1, mlp_only_layers=[])
cfg.architectures = ["Qwen3NextForCausalLM"]
cfg.save_pretrained(d)
out = {"layer_types": list(cfg.layer_types)}


def build(stage=None, le=None, dtype="float32"):
    ov = {"architectures": ["Qwen3NextPartialForCausalLM"]}
    if stage:
        ov["p2p_stage"] = stage
    if le:
        ov["p2p_layer_experts"] = le
    mc = ModelConfig(model=str(d), tokenizer=None, skip_tokenizer_init=True, dtype=dtype,
                     hf_overrides=ov)
    vc = VllmConfig(model_config=mc)
    return vc


def finish(mod):
    for sm in mod.modules():
        qm = getattr(sm, "quant_method", None)
        if qm is not None and hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(sm)


# ---------------------------------------------------------------- 1. MoE 块的数值
torch.manual_seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
vc = build(le={"2": R2})
with set_current_vllm_config(vc):
    full = P.PartialQwen3NextSparseMoeBlock(vc, prefix="model.layers.1.mlp")
    part = P.PartialQwen3NextSparseMoeBlock(vc, prefix="model.layers.2.mlp")
for p in full.parameters():
    p.data.normal_(0, 0.2)
with torch.no_grad():
    idx = torch.tensor(R2)
    part.gate.weight.copy_(full.gate.weight)
    part.shared_expert_gate.weight.copy_(full.shared_expert_gate.weight)
    part.experts.w13_weight.copy_(full.experts.w13_weight[idx])
    part.experts.w2_weight.copy_(full.experts.w2_weight[idx])
    part.shared_expert.gate_up_proj.weight.copy_(full.shared_expert.gate_up_proj.weight)
    part.shared_expert.down_proj.weight.copy_(full.shared_expert.down_proj.weight)
w13, w2 = full.experts.w13_weight.detach().clone(), full.experts.w2_weight.detach().clone()
sh_gu = full.shared_expert.gate_up_proj.weight.detach().clone()
sh_d = full.shared_expert.down_proj.weight.detach().clone()
sh_g = full.shared_expert_gate.weight.detach().clone()
gate_w = full.gate.weight.detach().clone()


def reference(x, allowed):
    """手写：全量 softmax，非驻留置 −∞，top-k，重归一；加 sigmoid(门) × 共享专家。"""
    logits = (x.float() @ gate_w.float().T).masked_fill(~allowed, float("-inf"))
    p = torch.softmax(logits, -1)
    w, ids = p.topk(min(K, int(allowed.sum())), dim=-1)
    w = w / w.sum(-1, keepdim=True)
    y = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            g, u = (w13[e].float() @ x[t].float()).chunk(2)
            y[t] += w[t, j] * (w2[e].float() @ (torch.nn.functional.silu(g) * u))
    g, u = (sh_gu.float() @ x.float().T).T.chunk(2, dim=-1)
    shared = ((torch.nn.functional.silu(g) * u) @ sh_d.float().T) * torch.sigmoid(x.float() @ sh_g.float().T)
    return y + shared, ids


x = torch.randn(16, H)
with set_current_vllm_config(vc), set_forward_context(None, vc):
    finish(full)
    finish(part)
    y_full, y_part = full(x.clone()), part(x.clone())
all_ok = torch.ones(E, dtype=torch.bool)
in_r = torch.zeros(E, dtype=torch.bool)
in_r[R2] = True
ref_full, ids_full = reference(x, all_ok)
ref_part, _ = reference(x, in_r)
covered = [all(int(e) in R2 for e in row) for row in ids_full]
out.update(full_vs_ref=float((y_full.float() - ref_full).abs().max()),
           part_vs_masked_ref=float((y_part.float() - ref_part).abs().max()),
           covered=sum(covered),
           part_vs_full_on_covered=max([float((y_part[t] - y_full[t]).abs().max())
                                        for t in range(16) if covered[t]] or [0.0]),
           n_local=int(part.experts.w13_weight.shape[0]))

# ---------------------------------------------------------------- 2. 整模型：切层、加载、核对
g = torch.Generator().manual_seed(0)


def r(*shape):
    return torch.randn(*shape, generator=g)


kd, vd, kh, vh = 16, 16, 2, 4
ckpt = {"model.embed_tokens.weight": r(V, H), "model.norm.weight": r(H), "lm_head.weight": r(V, H)}
for l in range(L):
    p = f"model.layers.{l}."
    ckpt.update({p + "input_layernorm.weight": r(H), p + "post_attention_layernorm.weight": r(H)})
    if cfg.layer_types[l] == "linear_attention":
        ckpt.update({p + "linear_attn.conv1d.weight": r(kd * kh * 2 + vd * vh, 1, 4),
                     p + "linear_attn.A_log": r(vh), p + "linear_attn.dt_bias": r(vh),
                     p + "linear_attn.in_proj_qkvz.weight": r(kd * kh * 2 + vd * vh * 2, H),
                     p + "linear_attn.in_proj_ba.weight": r(vh * 2, H),
                     p + "linear_attn.norm.weight": r(vd),
                     p + "linear_attn.out_proj.weight": r(H, vd * vh)})
    else:
        ckpt.update({p + "self_attn.q_proj.weight": r(4 * 16 * 2, H),
                     p + "self_attn.k_proj.weight": r(2 * 16, H),
                     p + "self_attn.v_proj.weight": r(2 * 16, H),
                     p + "self_attn.o_proj.weight": r(H, 4 * 16),
                     p + "self_attn.q_norm.weight": r(16), p + "self_attn.k_norm.weight": r(16)})
    ckpt.update({p + "mlp.gate.weight": r(E, H), p + "mlp.shared_expert_gate.weight": r(1, H),
                 p + "mlp.shared_expert.gate_proj.weight": r(SI, H),
                 p + "mlp.shared_expert.up_proj.weight": r(SI, H),
                 p + "mlp.shared_expert.down_proj.weight": r(H, SI)})
    for e in range(E):
        ckpt.update({p + f"mlp.experts.{e}.gate_proj.weight": r(I, H) + e,
                     p + f"mlp.experts.{e}.up_proj.weight": r(I, H),
                     p + f"mlp.experts.{e}.down_proj.weight": r(H, I)})


def model(stage, le):
    vc = build(stage, le)
    with set_current_vllm_config(vc):
        return P.Qwen3NextPartialForCausalLM(vllm_config=vc), vc


back, vc = model({"role": "back", "layers": [2, 4]}, {"2": R2, "3": R3})
with set_current_vllm_config(vc):
    back.load_weights(iter(ckpt.items()))
m2, m3 = back.model.layers[2].mlp, back.model.layers[3].mlp
w13b = m2.experts.w13_weight
out.update(
    back_layers=[type(x).__name__ for x in back.model.layers],
    back_embed=type(back.model.embed_tokens).__name__, back_head=type(back.lm_head).__name__,
    back_resident=[m2.resident, m3.resident],
    back_rows_ok=all(torch.equal(w13b[i, :I], ckpt[f"model.layers.2.mlp.experts.{e}.gate_proj.weight"])
                     for i, e in enumerate(R2)),
    back_shared_gate_ok=bool(torch.equal(m3.shared_expert_gate.weight,
                                         ckpt["model.layers.3.mlp.shared_expert_gate.weight"])),
    back_full_router=list(m2.gate.weight.shape),
    back_mixers=[hasattr(back.model.layers[2], "linear_attn"), hasattr(back.model.layers[3], "self_attn")],
)
front, vc = model({"role": "front", "layers": [0, 2]}, None)
with set_current_vllm_config(vc):
    front.load_weights(iter(ckpt.items()))
out.update(front_layers=[type(x).__name__ for x in front.model.layers],
           front_norm=type(front.model.norm).__name__, front_head=type(front.lm_head).__name__,
           front_full=front.model.layers[1].mlp.resident is None)

broken = {k: v for k, v in ckpt.items() if "layers.3.mlp.experts.7." not in k}
b2, vc = model({"role": "back", "layers": [2, 4]}, {"2": R2, "3": R3})
try:
    with set_current_vllm_config(vc):
        b2.load_weights(iter(broken.items()))
    out["missing_raises"] = False
except ValueError as e:
    out["missing_raises"] = "全局专家号 [7]" in str(e)
print(json.dumps(out, ensure_ascii=False))
