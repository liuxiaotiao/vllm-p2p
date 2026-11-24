# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-4.7-Flash 的单卡对拍：部分专家模型 vs vLLM 原生（P2P 移植，GLM 版的 A1 验收）。

    # vllm-0.14.0 目录下，装好本源码的那个环境里；A30 与 T4 各跑一次
    python -m vllm.p2p.parity_glm                   # fp16 + TRITON_MLA（实验口径）
    python -m vllm.p2p.parity --family glm4_moe_lite   # 同上（parity.py 转过来）

与 Qwen 版（``parity.py``）不同，这里**不需要 p2p-framework**：GLM 的路由是「sigmoid 打分 + 纠偏 bias
只用来挑专家」，挑选与权重都逐专家独立 —— 所以「只装子集 S」在数学上**正好等于**「原生模型、
把 S 以外专家的纠偏 bias 改成 −1e4」（那些专家永远挑不中；被挑中者的权重仍是不带 bias 的 sigmoid、
再重归一）。于是参考实现就是 vLLM 原生 GLM 跑在一份改过 bias 的 checkpoint 上，前提是每层驻留数 ≥ top-k
（少于 top-k 时部分专家模型只挑 n 个，原生仍会挑 k 个，两者不再等价 —— 本脚本的子集都 ≥ top-k）。

在一个**真实 key 命名**的微型 GLM-4.7-Flash checkpoint 上跑（第 0 层稠密、MLA 维度与真模型相同的
kv_lora 512 / rope 64，好让 Triton MLA 的 kernel 走真模型那条路；config 用 transformers 5 风格的
``rope_parameters``，顺带验 θ 的换算）：

    stock            vLLM 原生 Glm4MoeLiteForCausalLM
    partial-full     Glm4MoeLitePartialForCausalLM，不裁任何层 —— 必须与 stock **逐 token、逐 logprob 相同**
    partial-sub      第 2/3 层只装子集 —— 要验的东西
    partial-stats    partial-sub + 统计钩子 —— 输出必须与 partial-sub 逐位相同，每条请求落一行
    masked (tf)      原生 + 改过 bias 的 checkpoint，沿 partial-sub 生成的序列做一次 prefill（teacher forcing）
    stock (tf)       原生，沿 stock 自己生成的序列做 prefill —— decode 与 prefill 两条路（MLA 的
                     吸收 / 不吸收）之间的差 = **噪声底**
    torch-prefill    （仅 sm80+）原生 + ``VLLM_P2P_MLA_TORCH_PREFILL=1``，同 stock (tf) —— Turing 上用的
                     纯 torch prefill 与 FlashAttention 的差

判据同 Qwen 版：partial-sub 的逐步 logprob 与 masked (tf) 之差 ≤ 3× 噪声底 + 容差；
partial-sub 必须**不同于** stock（否则 hf_overrides 没生效）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ARCH = "Glm4MoeLitePartialForCausalLM"
STOCK = "Glm4MoeLiteForCausalLM"

# 层数、宽度都小；MLA 的 kv_lora / rope 与真模型相同（Triton MLA decode 的块大小按它们写的）
PARITY_CFG = {
    "architectures": [STOCK],
    "model_type": "glm4_moe_lite",
    "num_hidden_layers": 4,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": 128,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 192,
    "qk_rope_head_dim": 64,
    "v_head_dim": 256,
    "n_routed_experts": 16,
    "n_shared_experts": 1,
    "num_experts_per_tok": 4,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.8,
    "first_k_dense_replace": 1,
    "num_nextn_predict_layers": 0,
    "vocab_size": 512,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
    "rope_interleave": True,
    "hidden_act": "silu",
    "attention_bias": False,
    "tie_word_embeddings": False,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "pad_token_id": 0,
}

# 0-based 层 → 驻留专家（都 ≥ top-k = 4，见模块说明）。第 0 层稠密，第 1 层全装
PARTIAL = {2: [1, 4, 6, 9, 11, 14], 3: [0, 5, 7, 13]}
STATS_LR_LAYERS = [1, 2]
STATS_BODY = "1:5"

PROMPTS = [
    [5, 17, 99, 230, 41, 8, 300, 12],
    [400, 3, 3, 3, 77, 501, 260, 33, 90, 120, 7, 16, 44, 200, 311, 15],
    [9, 250],
    [i * 13 % 509 + 3 for i in range(48)],
]


# --------------------------------------------------------------------------- #
# 微型 checkpoint
# --------------------------------------------------------------------------- #
def write_checkpoint(d: Path, cfg: dict, *, dtype: str, seed: int = 11,
                     mask: dict[int, list[int]] | None = None) -> None:
    """按 GLM-4.7-Flash 的权重名写一份随机 checkpoint（单个 model.safetensors）。
    ``mask``：{0-based 层: 驻留专家}，把其余专家的纠偏 bias 改成 −1e4（参考 checkpoint）。"""
    import torch
    from safetensors.torch import save_file

    dt = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    g = torch.Generator().manual_seed(seed)

    def r(*shape, std=0.05):
        return (torch.randn(*shape, generator=g) * std).to(dt)

    def norm(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(dt)

    L, H, V = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["vocab_size"]
    NH, NOPE, ROPE, VD = (cfg["num_attention_heads"], cfg["qk_nope_head_dim"],
                          cfg["qk_rope_head_dim"], cfg["v_head_dim"])
    QL, KVL = cfg["q_lora_rank"], cfg["kv_lora_rank"]
    E, I, DI = cfg["n_routed_experts"], cfg["moe_intermediate_size"], cfg["intermediate_size"]
    w = {"model.embed_tokens.weight": r(V, H, std=0.5), "model.norm.weight": norm(H),
         "lm_head.weight": r(V, H)}
    for l in range(L):
        p = f"model.layers.{l}."
        w.update({p + "input_layernorm.weight": norm(H), p + "post_attention_layernorm.weight": norm(H),
                  p + "self_attn.q_a_proj.weight": r(QL, H), p + "self_attn.q_a_layernorm.weight": norm(QL),
                  p + "self_attn.q_b_proj.weight": r(NH * (NOPE + ROPE), QL),
                  p + "self_attn.kv_a_proj_with_mqa.weight": r(KVL + ROPE, H),
                  p + "self_attn.kv_a_layernorm.weight": norm(KVL),
                  p + "self_attn.kv_b_proj.weight": r(NH * (NOPE + VD), KVL),
                  p + "self_attn.o_proj.weight": r(H, NH * VD)})
        if l < cfg["first_k_dense_replace"]:
            w.update({p + "mlp.gate_proj.weight": r(DI, H), p + "mlp.up_proj.weight": r(DI, H),
                      p + "mlp.down_proj.weight": r(H, DI)})
            continue
        bias = 0.1 * torch.randn(E, generator=g)
        if mask and l in mask:
            keep = torch.zeros(E, dtype=torch.bool)
            keep[list(mask[l])] = True
            bias = torch.where(keep, bias, torch.full_like(bias, -1e4))
        w.update({p + "mlp.gate.weight": (torch.randn(E, H, generator=g) * 0.1).to(dt),
                  p + "mlp.gate.e_score_correction_bias": bias.float(),
                  p + "mlp.shared_experts.gate_proj.weight": r(I, H),
                  p + "mlp.shared_experts.up_proj.weight": r(I, H),
                  p + "mlp.shared_experts.down_proj.weight": r(H, I)})
        for e in range(E):
            w.update({p + f"mlp.experts.{e}.gate_proj.weight": r(I, H),
                      p + f"mlp.experts.{e}.up_proj.weight": r(I, H),
                      p + f"mlp.experts.{e}.down_proj.weight": r(H, I)})
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in w.items()}, str(d / "model.safetensors"),
              metadata={"format": "pt"})
    (d / "config.json").write_text(json.dumps(dict(cfg, torch_dtype=str(dt).split(".")[-1]),
                                              indent=1), encoding="utf-8")
    (d / "generation_config.json").write_text(json.dumps({"bos_token_id": 0, "eos_token_id": 2}),
                                              encoding="utf-8")


def write_checkpoints(wd: Path, dtype: str) -> tuple[Path, Path]:
    """同一个种子写两份：原样的、改过 bias 的（其余权重逐位相同）。"""
    ckpt, masked = wd / "ckpt", wd / "ckpt_masked"
    if not (ckpt / "config.json").exists():
        write_checkpoint(ckpt, PARITY_CFG, dtype=dtype)
    if not (masked / "config.json").exists():
        write_checkpoint(masked, PARITY_CFG, dtype=dtype, mask=PARTIAL)
    return ckpt, masked


# --------------------------------------------------------------------------- #
# 子进程：跑一个 vLLM 变体
# --------------------------------------------------------------------------- #
def _child(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-dir", default="")
    ap.add_argument("--attention-backend", default="")
    ap.add_argument("--teacher", default="", help="JSON [[prompt, gen], …]：沿 prompt+gen 做 prefill")
    a = ap.parse_args(argv)
    if a.stats_dir:
        os.environ["VLLM_P2P_STATS_DIR"] = a.stats_dir

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(model=a.ckpt, dtype=a.dtype, skip_tokenizer_init=True, max_model_len=512,
              gpu_memory_utilization=0.25, seed=0, max_num_seqs=8, enforce_eager=a.eager,
              enable_prefix_caching=False)
    if a.attention_backend:
        kw["attention_backend"] = a.attention_backend
    if a.overrides:
        kw["hf_overrides"] = json.loads(a.overrides)
    llm = LLM(**kw)
    arch = llm.llm_engine.model_config.architectures
    res = []
    if a.teacher:
        pairs = json.loads(a.teacher)
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=5, detokenize=False)
        outs = llm.generate([TokensPrompt(prompt_token_ids=list(p) + list(g)) for p, g in pairs], sp)
        for (p, g), o in zip(pairs, outs):
            steps = []
            for j, t in enumerate(g):
                lp = o.prompt_logprobs[len(p) + j]
                steps.append({"chosen": int(t), "lp": float(lp[t].logprob),
                              "top": {int(k): float(v.logprob) for k, v in lp.items()
                                      if v.rank is not None and v.rank <= 5}})
            res.append({"tokens": [int(t) for t in g], "steps": steps})
    else:
        extra = ({"p2p_body": STATS_BODY} if a.stats_dir else None)
        sps = [SamplingParams(temperature=0.0, max_tokens=a.tokens, logprobs=5, ignore_eos=True,
                              detokenize=False,
                              extra_args=(dict(extra, p2p_id=f"p{i}") if extra else None))
               for i in range(len(PROMPTS))]
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in PROMPTS], sps)
        if a.stats_dir:
            # 最后一批的「结束」要等下一步调度才送到 worker —— 再跑一条，把它们送出去
            llm.generate([TokensPrompt(prompt_token_ids=[3, 4, 5])],
                         SamplingParams(temperature=0.0, max_tokens=1, detokenize=False))
        for o in outs:
            c = o.outputs[0]
            steps = [{"chosen": int(t), "lp": float(lp[t].logprob),
                      "top": {int(k): float(v.logprob) for k, v in lp.items()}}
                     for t, lp in zip(c.token_ids, c.logprobs or [])]
            res.append({"tokens": [int(t) for t in c.token_ids], "steps": steps})
    Path(a.out).write_text(json.dumps({"arch": arch, "prompts": res}), encoding="utf-8")
    return 0


def _run(ckpt: Path, a, out: Path, *, overrides: dict | None = None, stats_dir: Path | None = None,
         teacher: list | None = None, env: dict | None = None) -> dict:
    cmd = [sys.executable, "-m", "vllm.p2p.parity_glm", "_vllm", "--ckpt", str(ckpt),
           "--dtype", a.dtype, "--tokens", str(a.tokens), "--out", str(out)]
    if a.attention_backend:
        cmd += ["--attention-backend", a.attention_backend]
    if stats_dir is not None:
        cmd += ["--stats-dir", str(stats_dir)]
    if overrides is not None:
        cmd += ["--overrides", json.dumps(overrides)]
    if teacher is not None:
        cmd += ["--teacher", json.dumps(teacher)]
    if a.eager:
        cmd.append("--eager")
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-40:])
        raise RuntimeError(f"vLLM 子进程失败（{out.stem}）：\n{tail}")
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
def cmp_steps(a: list[dict], b: list[dict]) -> dict:
    """a、b 是同一条序列上逐步的 logprob（同一 chosen token）。"""
    assert [s["chosen"] for s in a] == [s["chosen"] for s in b]
    d = [abs(x["lp"] - y["lp"]) for x, y in zip(a, b)]
    ov = [len(set(x["top"]) & set(y["top"])) / max(min(len(x["top"]), len(y["top"])), 1)
          for x, y in zip(a, b)]
    return {"max_lp_diff": max(d) if d else 0.0, "mean_lp_diff": sum(d) / len(d) if d else 0.0,
            "top5_overlap": sum(ov) / len(ov) if ov else 1.0}


def first_divergence(x: list[int], y: list[int]) -> int | None:
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None


def _capability() -> tuple[int, int]:
    try:
        import torch
        return torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    except Exception:                                        # noqa: BLE001
        return (0, 0)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_vllm":
        return _child(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dtype", default="half", choices=("half", "bfloat16"))
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--eager", action="store_true", help="关掉 CUDA Graph / torch.compile")
    ap.add_argument("--attention-backend", default="TRITON_MLA",
                    help="与 layout.py 给 GLM 的通道同一后端；空串 = vLLM 自选")
    ap.add_argument("--tol", type=float, default=None,
                    help="logprob 绝对容差（默认 bf16 0.05、fp16 0.02）")
    a = ap.parse_args(argv)
    tol = a.tol if a.tol is not None else (0.05 if a.dtype == "bfloat16" else 0.02)
    wd = a.workdir or Path(tempfile.mkdtemp(prefix="p2p_parity_glm_"))
    ckpt, masked = write_checkpoints(wd, a.dtype)
    cap = _capability()
    print(f"checkpoint: {ckpt}（参考：{masked}）  dtype={a.dtype}  attention={a.attention_backend or 'auto'}"
          f"  tokens={a.tokens}  sm{cap[0]}{cap[1]}{'  (eager)' if a.eager else ''}")

    full_ov = {"architectures": [ARCH], "p2p_layer_experts": {}}
    part_ov = {"architectures": [ARCH], "p2p_layer_experts": {str(k): v for k, v in PARTIAL.items()}}
    stats_ov = dict(part_ov, p2p_stats={"lr_layers": STATS_LR_LAYERS}, p2p_channel="parity-glm")
    stats_dir = wd / "stats"
    if stats_dir.exists():
        for f in stats_dir.glob("*.jsonl"):
            f.unlink()
    runs: dict[str, dict] = {}
    for name, ov in (("stock", None), ("partial-full", full_ov), ("partial-sub", part_ov),
                     ("partial-stats", stats_ov)):
        print(f"  vLLM {name} …", flush=True)
        runs[name] = _run(ckpt, a, wd / f"{name}.json", overrides=ov,
                          stats_dir=stats_dir if name == "partial-stats" else None)
        print(f"    架构 {runs[name]['arch']}")
    pairs_stock = [[p, r["tokens"]] for p, r in zip(PROMPTS, runs["stock"]["prompts"])]
    pairs_part = [[p, r["tokens"]] for p, r in zip(PROMPTS, runs["partial-sub"]["prompts"])]
    print("  vLLM stock (tf) …", flush=True)
    runs["stock-tf"] = _run(ckpt, a, wd / "stock-tf.json", teacher=pairs_stock)
    print("  vLLM masked (tf) …", flush=True)
    runs["masked-tf"] = _run(masked, a, wd / "masked-tf.json", teacher=pairs_part)
    if cap >= (8, 0):
        print("  vLLM stock (tf, torch prefill) …", flush=True)
        runs["torch-prefill-tf"] = _run(ckpt, a, wd / "torch-prefill-tf.json", teacher=pairs_stock,
                                        env={"VLLM_P2P_MLA_TORCH_PREFILL": "1"})

    checks: list[tuple[str, bool, str]] = []
    rep: dict = {"dtype": a.dtype, "tol": tol, "capability": list(cap), "prompts": []}
    P = lambda n: runs[n]["prompts"]                                     # noqa: E731

    # 1. 全装 ≡ 原生
    same_tok = all(x["tokens"] == y["tokens"] for x, y in zip(P("stock"), P("partial-full")))
    d_pf = (max(cmp_steps(x["steps"], y["steps"])["max_lp_diff"]
                for x, y in zip(P("stock"), P("partial-full"))) if same_tok else math.inf)
    checks.append(("部分专家模型全装 ≡ vLLM 原生（逐 token）", same_tok and d_pf <= 1e-5,
                   f"max |Δlogprob| = {d_pf:.2e}"))
    checks.append(("部分专家模型确实被选用", runs["partial-sub"]["arch"] == [ARCH],
                   f"architectures = {runs['partial-sub']['arch']}"))

    # 2. 噪声底（decode vs prefill）；3. 部分专家 vs 改 bias 的原生
    floor, part, differs = [], [], False
    for i, p in enumerate(PROMPTS):
        st, pp = P("stock")[i], P("partial-sub")[i]
        c_floor = cmp_steps(st["steps"], P("stock-tf")[i]["steps"])
        c_part = cmp_steps(pp["steps"], P("masked-tf")[i]["steps"])
        dv = first_divergence(st["tokens"], pp["tokens"])
        if dv is not None or any(abs(x["lp"] - y["lp"]) > 1e-3 for x, y in zip(st["steps"], pp["steps"])):
            differs = True
        floor.append(c_floor)
        part.append(c_part)
        row = {"len": len(p), "floor": c_floor, "partial": c_part, "stock_vs_partial_diverge_at": dv}
        if "torch-prefill-tf" in runs:
            row["torch_prefill"] = cmp_steps(P("stock-tf")[i]["steps"], P("torch-prefill-tf")[i]["steps"])
        rep["prompts"].append(row)
        print(f"  prompt{i} (len {len(p):2d})  噪声底 max|Δ|={c_floor['max_lp_diff']:.4f}"
              f"  部分专家 max|Δ|={c_part['max_lp_diff']:.4f}  top5 重合 {c_part['top5_overlap']:.0%}"
              f"  与全装分叉于 {dv if dv is not None else '—'}"
              + (f"  torch prefill max|Δ|={row['torch_prefill']['max_lp_diff']:.4f}"
                 if "torch_prefill" in row else ""))
    fl = max(c["max_lp_diff"] for c in floor)
    pt = max(c["max_lp_diff"] for c in part)
    checks.append(("部分专家生效（输出不同于全装）", differs,
                   "裁掉专家后 logprob / 输出应当变化；没变说明 hf_overrides 没生效"))
    checks.append(("部分专家 ≈ 原生 + 改 bias（数学上等价）", pt <= 3 * fl + tol,
                   f"max |Δlogprob| {pt:.4f}（噪声底 {fl:.4f}，阈值 3×底+{tol:g}）"))
    checks.append(("top-5 重合", min(c["top5_overlap"] for c in part) >= 0.8,
                   f"最低 {min(c['top5_overlap'] for c in part):.0%}"))
    if "torch-prefill-tf" in runs:
        tp = max(r["torch_prefill"]["max_lp_diff"] for r in rep["prompts"])
        checks.append(("纯 torch MLA prefill ≈ FlashAttention（Turing 上用的那条）", tp <= fl + tol,
                       f"max |Δlogprob| {tp:.4f}（阈值 噪声底 {fl:.4f} + {tol:g}）"))
    else:
        print("  （这张卡没有 FlashAttention：prefill 本来就走纯 torch，噪声底里已经含着它）")

    # 4. 统计钩子
    same = all(x["tokens"] == y["tokens"] and cmp_steps(x["steps"], y["steps"])["max_lp_diff"] <= 1e-5
               for x, y in zip(P("partial-sub"), P("partial-stats")))
    checks.append(("开统计钩子不改输出（≡ partial-sub）", same, "逐 token、逐 logprob"))
    recs = {}
    for f in sorted(stats_dir.glob("*.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            d = json.loads(ln)
            if d.get("p2p_id"):
                recs[d["p2p_id"]] = d
    got = all(f"p{i}" in recs for i in range(len(PROMPTS)))
    checks.append(("钩子为每条请求落了一行", got, f"{len(recs)}/{len(PROMPTS)} 条（{stats_dir}）"))
    if got:
        keys_ok = all(set(recs[f"p{i}"].get("lr_layers", {})) == {str(l + 1) for l in STATS_LR_LAYERS}
                      for i in range(len(PROMPTS)))
        miss = sum(r["prefill"]["miss_token_layer"] + r["decode"]["miss_token_layer"] for r in recs.values())
        checks.append(("钩子的 LR 层 = 配置的 MoE 层", keys_ok, f"{STATS_LR_LAYERS}（0-based）"))
        checks.append(("钩子数到了后段 miss（sigmoid + bias 口径）", miss > 0, f"miss (token,层) = {miss}"))

    print()
    ok = True
    for name, good, why in checks:
        ok &= good
        print(f"  {'✓' if good else '✗'} {name}：{why}")
    rep["checks"] = [{"name": n, "ok": g, "detail": w} for n, g, w in checks]
    (wd / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'通过' if ok else '未通过'}  —— 明细 {wd / 'report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
