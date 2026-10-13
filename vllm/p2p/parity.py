# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""单卡对拍：部分专家模型 vs vLLM 原生 vs p2p-framework 的 torch 执行层（P2P 移植 A1 验收）。

    # vllm-0.14.0 目录下，装好本源码的那个环境里（安装方式见 vllm/p2p/__init__.py）
    python -m vllm.p2p.parity                       # fp16 + TRITON_ATTN（实验口径）
    python -m vllm.p2p.parity --dtype bfloat16      # 仅 A30，做参照
    python -m vllm.p2p.parity --p2p-framework /path/to/p2p-framework

参考实现与微型 checkpoint 生成器都来自 p2p-framework（`p2pmoe.runtime.torch_model`、
`p2pmoe.sim.fake_checkpoint`）—— 它就是要对齐的标准，不在 vLLM 里另写一份。默认找
vllm-0.14.0 旁边的 `p2p-framework/`，或者用 `--p2p-framework` / 环境变量 `P2P_FRAMEWORK` 指定。

在一个**真实 key 命名**的微型 Qwen3-MoE checkpoint 上跑，维度放大到 vLLM 的 kernel 吃得下。
五个变体：

    stock          vLLM 原生 Qwen3MoeForCausalLM
    partial-full   Qwen3MoePartialForCausalLM，但不裁任何层 —— 必须与 stock **逐 token、逐 logprob 相同**
    partial-sub    Qwen3MoePartialForCausalLM，第 2/3 层只装子集 —— A1 要验的东西
    ref-full       p2p-framework TorchSegmentModel，全装
    ref-sub        p2p-framework TorchSegmentModel，同样的子集，miss_policy=local_topk
    partial-stats  partial-sub + A5 统计钩子（p2p_stats）—— 输出必须与 partial-sub **逐位相同**，
                   钩子记下的逐请求 miss 与 body 计数要与 ref-sub 自己数的一致

判据（为什么不直接比贪心输出）：随机权重的 logits 很平，top-1 与 top-2 间距小，两套不同的
kernel（融合 MoE / paged attention vs 逐专家 matmul）累加顺序不同，argmax 翻一次整条
轨迹就分叉 —— 那测的是舍入噪声，不是正确性。所以主判据是 **teacher forcing**：参考实现
沿 vLLM 生成的那条序列做一次 prefill，逐步比较「vLLM 选中的 token」的 logprob。

* stock vs ref-full 的差距 = 两套实现之间的**噪声底**；
* partial-sub vs ref-sub 的差距必须与噪声底同量级（≤ 3× 底 + 容差）；
* partial-sub 必须**不同于** stock —— 否则说明 hf_overrides 根本没生效（静默失效）。

每个 vLLM 变体在独立子进程里起（一个进程里先后建多个 LLM 不可靠，显存也不一定还得回来）。
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

from vllm.p2p.layout import ARCH


def _use_p2p_framework(path: str | None) -> Path:
    """把 p2p-framework 放上 sys.path —— 参考实现与微型 checkpoint 生成器在那边。"""
    cand = [Path(p) for p in (path, os.environ.get("P2P_FRAMEWORK")) if p]
    cand.append(Path(__file__).resolve().parents[3] / "p2p-framework")
    for c in cand:
        if (c / "p2pmoe" / "runtime" / "torch_model.py").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    raise SystemExit("找不到 p2p-framework（参考实现在那边）。用 --p2p-framework 或 "
                     f"P2P_FRAMEWORK 指定；试过 {[str(c) for c in cand]}")

# 比 TINY_QWEN3_MOE 大一号：head_dim 64、hidden 256 —— paged attention / 融合 MoE 的
# kernel 对过小的维度不一定有实现
PARITY_CFG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "num_hidden_layers": 4,
    "num_experts": 16,
    "num_experts_per_tok": 4,
    "hidden_size": 256,
    "moe_intermediate_size": 128,
    "intermediate_size": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 512,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "norm_topk_prob": True,
    "tie_word_embeddings": False,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "hidden_act": "silu",
    "attention_bias": False,
    "bos_token_id": 0,
    "eos_token_id": 2,
}

# 0-based 层 → 驻留专家。第 3 层只有 3 个 < top_k=4：走 min(top_k, n) 那条分支
PARTIAL = {2: [1, 4, 6, 9, 11, 14], 3: [0, 7, 13]}
# A5 钩子对拍：前两层按 body 计数（LR 特征的口径），body = prompt 的第 1..4 个 token
STATS_LR_LAYERS = [0, 1]
STATS_BODY = "1:5"

PROMPTS = [
    [5, 17, 99, 230, 41, 8, 300, 12],
    [400, 3, 3, 3, 77, 501, 260, 33, 90, 120, 7, 16, 44, 200, 311, 15],
    [9, 250],
    [i * 13 % 509 + 3 for i in range(48)],
]


# --------------------------------------------------------------------------- #
# 子进程：跑一个 vLLM 变体
# --------------------------------------------------------------------------- #
def _vllm_child(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats-dir", default="")
    a = ap.parse_args(argv)
    if a.stats_dir:
        os.environ["VLLM_P2P_STATS_DIR"] = a.stats_dir      # worker 进程继承

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    kw = dict(model=a.ckpt, dtype=a.dtype, skip_tokenizer_init=True, max_model_len=512,
              gpu_memory_utilization=0.25, seed=0, max_num_seqs=8, enforce_eager=a.eager)
    if a.overrides:
        kw["hf_overrides"] = json.loads(a.overrides)
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=a.tokens, logprobs=5, ignore_eos=True,
                        detokenize=False)
    if a.stats_dir:
        # 每条带上 p2p_id 与 body 区间（与网关转发时一样经 extra_args 进 worker）
        sps = [SamplingParams(temperature=0.0, max_tokens=a.tokens, logprobs=5, ignore_eos=True,
                              detokenize=False,
                              extra_args={"p2p_id": f"p{i}", "p2p_body": STATS_BODY})
               for i in range(len(PROMPTS))]
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in PROMPTS], sps)
        # 最后一批的「结束」要等下一步调度才送到 worker —— 再跑一条，把它们送出去
        llm.generate([TokensPrompt(prompt_token_ids=[3, 4, 5])],
                     SamplingParams(temperature=0.0, max_tokens=1, detokenize=False))
    else:
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in PROMPTS], sp)
    res = []
    for o in outs:
        c = o.outputs[0]
        steps = []
        for t, lp in zip(c.token_ids, c.logprobs or []):
            steps.append({"chosen": int(t), "lp": float(lp[t].logprob),
                          "top": {int(k): float(v.logprob) for k, v in lp.items()}})
        res.append({"tokens": [int(t) for t in c.token_ids], "steps": steps})
    arch = llm.llm_engine.model_config.architectures
    Path(a.out).write_text(json.dumps({"arch": arch, "prompts": res}), encoding="utf-8")
    return 0


def _run_vllm(ckpt: Path, dtype: str, overrides: dict | None, tokens: int, eager: bool,
              out: Path, stats_dir: Path | None = None) -> dict:
    cmd = [sys.executable, "-m", "vllm.p2p.parity", "_vllm", "--ckpt", str(ckpt),
           "--dtype", dtype, "--tokens", str(tokens), "--out", str(out)]
    if stats_dir is not None:
        cmd += ["--stats-dir", str(stats_dir)]
    if overrides is not None:
        cmd += ["--overrides", json.dumps(overrides)]
    if eager:
        cmd.append("--eager")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-40:])
        raise RuntimeError(f"vLLM 子进程失败（{' '.join(cmd[3:5])}…）：\n{tail}")
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# 参考实现：p2p-framework 的 torch 执行层
# --------------------------------------------------------------------------- #
class _Ref:
    def __init__(self, ckpt: Path, layer_experts_0b: dict[int, list[int]], device: str):
        import torch

        from p2pmoe.runtime.torch_model import TorchModelConfig, TorchSegmentModel
        from p2pmoe.runtime.weights import KeyPlan, SelectiveLoader, WeightIndex, qwen_moe_keys

        cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
        mc = TorchModelConfig.from_hf(cfg)
        E = mc.n_experts
        le = {l + 1: list(layer_experts_0b.get(l, range(E))) for l in range(mc.n_layers)}
        keys = qwen_moe_keys(KeyPlan(layer_experts=le, with_embed=True, with_lm_head=True))
        tensors, rep = SelectiveLoader(WeightIndex(ckpt)).load(keys, device=device,
                                                              dtype=mc.torch_dtype)
        assert not rep.missing, rep.missing[:3]
        self.m = TorchSegmentModel(mc, le, tensors, device=device, miss_policy="local_topk")
        self.torch = torch

    def stats(self, ids: list[int], body: tuple[int, int]) -> tuple[dict, int, int]:
        """沿 ids 做一次 prefill，返回 (body 位置上的逐层 LayerStat, miss (token,层) 数, token×层)。"""
        import numpy as np

        torch = self.torch
        m = np.zeros(len(ids), dtype=bool)
        m[body[0]:min(body[1], len(ids))] = True
        req = f"s{len(ids)}-{ids[0]}"
        with torch.no_grad():
            _, st = self.m.forward(req, self.m.embed_tokens(ids), count_mask=m)
        self.m.drop_kv(req)
        return st.layers, st.miss_token_layer, st.n_token_layer

    def teacher_forced(self, prompt: list[int], gen: list[int]) -> list[dict]:
        """沿 prompt+gen 做一次 prefill，返回生成的每一步的 log_softmax（top-5 + 选中 token）。"""
        torch = self.torch
        ids = list(prompt) + list(gen)
        req = f"r{len(ids)}-{ids[0]}"
        with torch.no_grad():
            h, _ = self.m.forward(req, self.m.embed_tokens(ids))
            lg = self.m.logits(h).float()
            lp = torch.log_softmax(lg, dim=-1)
        self.m.drop_kv(req)
        out = []
        for j, t in enumerate(gen):
            row = lp[len(prompt) - 1 + j]
            v, i = torch.topk(row, 5)
            out.append({"chosen": int(t), "lp": float(row[t]),
                        "top": {int(a): float(b) for a, b in zip(i.tolist(), v.tolist())},
                        "argmax": int(i[0])})
        return out


# --------------------------------------------------------------------------- #
def _cmp_steps(a: list[dict], b: list[dict]) -> dict:
    """a、b 是同一条序列上逐步的 logprob（同一 chosen token）。"""
    assert [s["chosen"] for s in a] == [s["chosen"] for s in b]
    d = [abs(x["lp"] - y["lp"]) for x, y in zip(a, b)]
    ov = [len(set(x["top"]) & set(y["top"])) / max(len(x["top"]), 1) for x, y in zip(a, b)]
    return {"max_lp_diff": max(d) if d else 0.0, "mean_lp_diff": sum(d) / len(d) if d else 0.0,
            "top5_overlap": sum(ov) / len(ov) if ov else 1.0}


def _first_divergence(x: list[int], y: list[int]) -> int | None:
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_vllm":
        return _vllm_child(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dtype", default="half", choices=("half", "bfloat16"),
                    help="默认 half：对照实验统一 fp16（VLLM_MIGRATION.md 第 3 节）")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--eager", action="store_true", help="关掉 CUDA Graph / torch.compile")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--p2p-framework", default=None,
                    help="p2p-framework 目录（参考实现在那边）；默认环境变量 P2P_FRAMEWORK，"
                         "再默认 vllm-0.14.0 旁边的 p2p-framework/")
    ap.add_argument("--attention-backend", default="TRITON_ATTN",
                    help="与 layout.py 生成的通道同一后端；空串 = vLLM 自选")
    ap.add_argument("--tol", type=float, default=None,
                    help="logprob 绝对容差（默认 bf16 0.05、fp16 0.02）")
    a = ap.parse_args(argv)
    p2p = _use_p2p_framework(a.p2p_framework)
    tol = a.tol if a.tol is not None else (0.05 if a.dtype == "bfloat16" else 0.02)
    if a.attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = a.attention_backend   # 子进程继承

    wd = a.workdir or Path(tempfile.mkdtemp(prefix="p2p_parity_"))
    ckpt = wd / "ckpt"
    if not (ckpt / "config.json").exists():
        from p2pmoe.sim.fake_checkpoint import write_fake_checkpoint

        cfg = dict(PARITY_CFG, torch_dtype="bfloat16" if a.dtype == "bfloat16" else "float16")
        write_fake_checkpoint(ckpt, cfg, seed=11)
    print(f"参考实现: {p2p}")
    print(f"checkpoint: {ckpt}  dtype={a.dtype}  attention={a.attention_backend or 'auto'}"
          f"  tokens={a.tokens}"
          f"{'  (eager)' if a.eager else ''}")

    full_ov = {"architectures": [ARCH], "p2p_layer_experts": {}}
    part_ov = {"architectures": [ARCH],
               "p2p_layer_experts": {str(k): v for k, v in PARTIAL.items()}}
    stats_ov = dict(part_ov, p2p_stats={"lr_layers": STATS_LR_LAYERS}, p2p_channel="parity")
    stats_dir = wd / "stats"
    if stats_dir.exists():
        for f in stats_dir.glob("*.jsonl"):
            f.unlink()
    runs = {}
    for name, ov in (("stock", None), ("partial-full", full_ov), ("partial-sub", part_ov),
                     ("partial-stats", stats_ov)):
        print(f"  vLLM {name} …", flush=True)
        runs[name] = _run_vllm(ckpt, a.dtype, ov, a.tokens, a.eager, wd / f"{name}.json",
                               stats_dir if name == "partial-stats" else None)
        print(f"    架构 {runs[name]['arch']}")

    ref_full = _Ref(ckpt, {}, a.device)
    ref_part = _Ref(ckpt, PARTIAL, a.device)

    checks: list[tuple[str, bool, str]] = []
    rep: dict = {"dtype": a.dtype, "tol": tol, "prompts": []}

    # 1. 部分专家模型全装 ≡ 原生
    same_tok = all(x["tokens"] == y["tokens"] for x, y in
                   zip(runs["stock"]["prompts"], runs["partial-full"]["prompts"]))
    d_pf = max((_cmp_steps(x["steps"], y["steps"])["max_lp_diff"] for x, y in
                zip(runs["stock"]["prompts"], runs["partial-full"]["prompts"])), default=0.0) \
        if same_tok else math.inf
    checks.append(("部分专家模型全装 ≡ vLLM 原生（逐 token）", same_tok and d_pf <= 1e-5,
                   f"max |Δlogprob| = {d_pf:.2e}"))
    checks.append(("部分专家模型确实被选用", runs["partial-sub"]["arch"] == [ARCH],
                   f"architectures = {runs['partial-sub']['arch']}"))

    # 2. 噪声底：原生 vLLM vs p2p-framework 全装；3. 部分专家：vLLM vs p2p-framework
    floor, part, differs = [], [], False
    for i, p in enumerate(PROMPTS):
        st, pp = runs["stock"]["prompts"][i], runs["partial-sub"]["prompts"][i]
        c_floor = _cmp_steps(st["steps"], ref_full.teacher_forced(p, st["tokens"]))
        c_part = _cmp_steps(pp["steps"], ref_part.teacher_forced(p, pp["tokens"]))
        dv = _first_divergence(st["tokens"], pp["tokens"])
        if dv is not None or any(abs(x["lp"] - y["lp"]) > 1e-3
                                 for x, y in zip(st["steps"], pp["steps"])):
            differs = True
        floor.append(c_floor)
        part.append(c_part)
        rep["prompts"].append({"len": len(p), "floor": c_floor, "partial": c_part,
                               "stock_vs_partial_diverge_at": dv})
        print(f"  prompt{i} (len {len(p):2d})  噪声底 max|Δ|={c_floor['max_lp_diff']:.4f}"
              f"  部分专家 max|Δ|={c_part['max_lp_diff']:.4f}"
              f"  top5 重合 {c_part['top5_overlap']:.0%}"
              f"  与全装分叉于 {dv if dv is not None else '—'}")

    fl = max(c["max_lp_diff"] for c in floor)
    pt = max(c["max_lp_diff"] for c in part)
    checks.append(("部分专家生效（输出不同于全装）", differs,
                   "裁掉专家后 logprob / 输出应当变化；没变说明 hf_overrides 没生效"))
    checks.append(("部分专家：vLLM ≈ p2p-framework local_topk", pt <= 3 * fl + tol,
                   f"max |Δlogprob| {pt:.4f}（噪声底 {fl:.4f}，阈值 3×底+{tol:g}）"))
    checks.append(("top-5 重合", min(c["top5_overlap"] for c in part) >= 0.8,
                   f"最低 {min(c['top5_overlap'] for c in part):.0%}"))

    # 4. A5 钩子：不改输出；记下的统计与参考实现自己数的一致
    same = all(x["tokens"] == y["tokens"] and _cmp_steps(x["steps"], y["steps"])["max_lp_diff"]
               <= 1e-5 for x, y in zip(runs["partial-sub"]["prompts"],
                                       runs["partial-stats"]["prompts"]))
    checks.append(("开统计钩子不改输出（≡ partial-sub）", same, "逐 token、逐 logprob"))
    recs = {}
    for f in sorted(stats_dir.glob("*.jsonl")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            d = json.loads(ln)
            if d.get("p2p_id"):
                recs[d["p2p_id"]] = d
    got_all = all(f"p{i}" in recs for i in range(len(PROMPTS)))
    checks.append(("钩子为每条请求落了一行", got_all, f"{len(recs)}/{len(PROMPTS)} 条（{stats_dir}）"))
    if got_all:
        import numpy as np

        from p2pmoe.runtime.model import LayerStat

        b0, b1 = (int(x) for x in STATS_BODY.split(":"))
        eq = tot = 0
        dm = []
        for i, p in enumerate(PROMPTS):
            gen = runs["partial-stats"]["prompts"][i]["tokens"]
            seen = list(p) + list(gen[:-1])          # 最后一个生成的 token 没有再进模型
            lay, miss, ntl = ref_part.stats(seen, (b0, b1))
            r = recs[f"p{i}"]
            hook_miss = r["prefill"]["miss_token_layer"] + r["decode"]["miss_token_layer"]
            hook_ntl = r["prefill"]["n_token_layer"] + r["decode"]["n_token_layer"]
            dm.append(abs(hook_miss - miss) / max(ntl, 1))
            for l0b in STATS_LR_LAYERS:
                h = LayerStat.from_wire(r["lr_layers"][str(l0b + 1)])
                tot += 1
                eq += bool(np.array_equal(h.count, lay[l0b + 1].count))
            if hook_ntl != ntl or r["n_seen"] != len(seen):
                dm.append(1.0)
        checks.append(("钩子的 body 逐层计数 = 参考实现", eq >= 0.9 * tot,
                       f"{eq}/{tot} 层完全相同（随机权重上 fp 舍入偶尔会让并列的 top-k 换位）"))
        checks.append(("钩子的 miss = 参考实现", max(dm) <= 0.02,
                       f"max |Δmiss 率| = {max(dm):.4f}"))

    print()
    ok = True
    for name, good, why in checks:
        ok &= good
        print(f"  {'✓' if good else '✗'} {name}：{why}")
    rep["checks"] = [{"name": n, "ok": g, "detail": w} for n, g, w in checks]
    (wd / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(f"\n{'通过' if ok else '未通过'}  —— 明细 {wd / 'report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
