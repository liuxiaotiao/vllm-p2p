# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""在 CPU 上建整个 Glm4MoeLitePartialForCausalLM（前段 / 后段）并按 checkpoint 名加载一份假权重：
核对切层、PP/分段裁剪、专家本地化、纠偏 bias 同步、缺专家时报错。MLA 在 CPU 上没有后端，
这里用 VLLM_MLA_DISABLE=1 建普通注意力 —— 本脚本只管结构与加载，前向数值在 glm_partial_block.py。
由 tests/p2p/test_glm_partial.py 在子进程里跑，打印一行 JSON。"""

import json
import os
import pathlib
import socket
import tempfile

os.environ["VLLM_MLA_DISABLE"] = "1"

import torch  # noqa: E402

import vllm.platforms as _plat  # noqa: E402
from vllm.platforms.cpu import CpuPlatform  # noqa: E402

_plat._current_platform = CpuPlatform()

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
init_distributed_environment(world_size=1, rank=0, local_rank=0,
                             distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo")
initialize_model_parallel(1, 1)

from vllm.model_executor.models.glm4_moe_lite_partial import (  # noqa: E402
    Glm4MoeLitePartialForCausalLM, PartialGlm4MoeLite)
from vllm.transformers_utils.configs.glm4_moe_lite import Glm4MoeLiteConfig  # noqa: E402

L, H, E, I, DI, V = 4, 64, 8, 32, 128, 100
NH, NOPE, ROPE, VD, QL, KVL = 4, 8, 8, 8, 32, 16
d = pathlib.Path(tempfile.mkdtemp())
cfg = Glm4MoeLiteConfig(num_hidden_layers=L, hidden_size=H, intermediate_size=DI,
                        moe_intermediate_size=I, n_routed_experts=E, num_experts_per_tok=2,
                        num_attention_heads=NH, num_key_value_heads=NH, q_lora_rank=QL,
                        kv_lora_rank=KVL, qk_nope_head_dim=NOPE, qk_rope_head_dim=ROPE,
                        v_head_dim=VD, vocab_size=V, num_nextn_predict_layers=0)
cfg.architectures = ["Glm4MoeLiteForCausalLM"]
cfg.save_pretrained(d)

g = torch.Generator().manual_seed(0)


def r(*shape):
    return torch.randn(*shape, generator=g)


ckpt = {"model.embed_tokens.weight": r(V, H), "model.norm.weight": r(H), "lm_head.weight": r(V, H)}
for l in range(L):
    p = f"model.layers.{l}."
    ckpt.update({p + "input_layernorm.weight": r(H), p + "post_attention_layernorm.weight": r(H),
                 p + "self_attn.q_a_proj.weight": r(QL, H), p + "self_attn.q_a_layernorm.weight": r(QL),
                 p + "self_attn.q_b_proj.weight": r(NH * (NOPE + ROPE), QL),
                 p + "self_attn.kv_a_proj_with_mqa.weight": r(KVL + ROPE, H),
                 p + "self_attn.kv_a_layernorm.weight": r(KVL),
                 p + "self_attn.kv_b_proj.weight": r(NH * (NOPE + VD), KVL),
                 p + "self_attn.o_proj.weight": r(H, NH * VD)})
    if l == 0:
        ckpt.update({p + "mlp.gate_proj.weight": r(DI, H), p + "mlp.up_proj.weight": r(DI, H),
                     p + "mlp.down_proj.weight": r(H, DI)})
        continue
    ckpt.update({p + "mlp.gate.weight": r(E, H), p + "mlp.gate.e_score_correction_bias": r(E),
                 p + "mlp.shared_experts.gate_proj.weight": r(I, H),
                 p + "mlp.shared_experts.up_proj.weight": r(I, H),
                 p + "mlp.shared_experts.down_proj.weight": r(H, I)})
    for e in range(E):
        ckpt.update({p + f"mlp.experts.{e}.gate_proj.weight": r(I, H) + e,
                     p + f"mlp.experts.{e}.up_proj.weight": r(I, H),
                     p + f"mlp.experts.{e}.down_proj.weight": r(H, I)})

R2, R3 = [1, 4, 6], [0, 7]


def build(stage, le=None):
    ov = {"architectures": ["Glm4MoeLitePartialForCausalLM"], "p2p_stage": stage}
    if le:
        ov["p2p_layer_experts"] = le
    mc = ModelConfig(model=str(d), tokenizer=None, skip_tokenizer_init=True, dtype="float32",
                     hf_overrides=ov)
    vc = VllmConfig(model_config=mc)
    with set_current_vllm_config(vc):
        return Glm4MoeLitePartialForCausalLM(vllm_config=vc), vc


out = {}
back, vc = build({"role": "back", "layers": [2, 4]}, {"2": R2, "3": R3})
with set_current_vllm_config(vc):
    back.load_weights(iter(ckpt.items()))
m2, m3 = back.model.layers[2].mlp, back.model.layers[3].mlp
w13 = m2.experts.w13_weight
out["back_layers"] = [type(x).__name__ for x in back.model.layers]
out["back_embed"] = type(back.model.embed_tokens).__name__
out["back_head"] = type(back.lm_head).__name__
out["back_partial"] = [isinstance(m2, PartialGlm4MoeLite), m2.resident, m3.resident]
out["back_expert_rows_ok"] = all(
    torch.equal(w13[i, :I], ckpt[f"model.layers.2.mlp.experts.{e}.gate_proj.weight"])
    for i, e in enumerate(R2))
out["back_bias_ok"] = bool(torch.equal(m3.local_bias,
                                       ckpt["model.layers.3.mlp.gate.e_score_correction_bias"][R3]))
out["back_full_gate_ok"] = bool(torch.equal(m2.gate.weight, ckpt["model.layers.2.mlp.gate.weight"]))

front, vc = build({"role": "front", "layers": [0, 2]})
with set_current_vllm_config(vc):
    front.load_weights(iter(ckpt.items()))
out["front_layers"] = [type(x).__name__ for x in front.model.layers]
out["front_norm"] = type(front.model.norm).__name__
out["front_head"] = type(front.lm_head).__name__
out["front_dense0"] = type(front.model.layers[0].mlp).__name__
out["front_moe1_full"] = front.model.layers[1].mlp.resident is None

broken = {k: v for k, v in ckpt.items() if "layers.3.mlp.experts.7." not in k}
b2, vc = build({"role": "back", "layers": [2, 4]}, {"2": R2, "3": R3})
try:
    with set_current_vllm_config(vc):
        b2.load_weights(iter(broken.items()))
    out["missing_raises"] = False
except ValueError as e:
    out["missing_raises"] = "全局专家号 [7]" in str(e)
try:
    build({"role": "back", "layers": [2, 4]}, {"0": [1, 2]})
    out["dense_listed_raises"] = False
except ValueError as e:
    out["dense_listed_raises"] = "稠密层" in str(e)
print(json.dumps(out, ensure_ascii=False))
