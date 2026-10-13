# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A4 · 前置识别服务：前 L₀ 层（全专家）跑一遍 prefill → LR 判 task。**按文件路径运行。**

    # 在一台前段节点上（它本地有层 1..L₀ 的全专家权重与词嵌入），用 vllm-p2p 的 venv：
    /home/ubuntu/venvs/vllm-p2p/bin/python /home/ubuntu/vllm-p2p/vllm/p2p/identifier.py \\
        --p2p-framework /home/ubuntu/p2p-framework \\
        --model-dir /home/ubuntu/p2p-framework/weights-q3-30b \\
        --lr-model /home/ubuntu/p2p-framework/selector/selector_body_L8_params.npz \\
        --tasks gsm8k=1,mbpp=1,no_robots=1 --l0 8 --device cuda:0 --port 8100

    （一般用 `python3 vllm/p2p/launch.py ident start …` 远程拉起，见 launch.py。）

为什么不重写：识别这一步**直接复用 p2p-framework**（`TorchSegmentModel` 层 1..L₀ +
`LRClassifier.predict_stats`），它在线上验证过 150/150。网关把 prompt 的 token id 与 body 区间
发过来（bench 用 p2p-framework 的 `TextIO.encode_body` 构造，与选择器训练时同一口径），
这里只做三件事：嵌入 → 前 L₀ 层前向（只数 body token 的逐层 top-k）→ LR。KV 用完即丢。

这就是实验矩阵里 E4 的「不解耦的代价」：前 L₀ 层对每条请求要多算一遍（之后通道的前段
还会再算一遍）。GPU 上一次只算一条（串行），排队时间单独报出来。

显存：层 1..8 全专家 + 嵌入 ≈ 10 GiB（fp16）。和那台的前段 vLLM 同卡时，给 vLLM 让出位置：
`layout.py --gpu-mem-node N01=0.5`（N01 的 KV 仍比整条通道最紧的末级多，通道容量不变）。

不 import vllm —— 只要 torch、safetensors、numpy、aiohttp（vllm-p2p 的 venv 里都有）与
p2p-framework 的源码目录。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

__all__ = ["Identifier", "parse_tasks", "main"]


def use_p2p_framework(path: str | None) -> Path:
    """把 p2p-framework 的源码目录放上 sys.path（不要求 pip install）。"""
    cands = [path, os.environ.get("P2P_FRAMEWORK"),
             str(Path(__file__).resolve().parents[3] / "p2p-framework"),
             "/home/ubuntu/p2p-framework"]
    for c in cands:
        if c and (Path(c) / "p2pmoe" / "__init__.py").is_file():
            p = str(Path(c).resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
            return Path(p)
    raise SystemExit(f"找不到 p2p-framework（要有 p2pmoe/）；试过 {[c for c in cands if c]} —— "
                     "用 --p2p-framework 指定")


def parse_tasks(s: str) -> dict[str, float]:
    """"gsm8k=1,mbpp=1" → {"gsm8k": 1.0, "mbpp": 1.0}（与 control.py --tasks 同一写法）。"""
    out: dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = float(v) if v else 1.0
    if not out:
        raise ValueError("--tasks 是空的")
    return out


def ranges_to_mask(ranges: Sequence[Sequence[int]], T: int):
    import numpy as np

    m = np.zeros(T, dtype=bool)
    for a, b in ranges:
        a, b = int(a), int(b)
        if not 0 <= a <= b <= T:
            raise ValueError(f"body 区间 [{a},{b}) 超出 prompt 长度 {T}")
        m[a:b] = True
    return m


def verdict_wire(v) -> dict:
    """与 p2p-framework `node._verdict_wire` 同格式（网关、bench、report 都按它读）。"""
    d = {"task": v.task, "conf": round(float(v.confidence), 4), "zone": v.zone,
         "scores": {k: round(float(x), 4) for k, x in (v.scores or {}).items()}}
    if v.raw is not None:
        d["raw"] = {k: round(float(x), 4) for k, x in v.raw.items()}
    if getattr(v, "error", None):
        d["error"] = v.error
    if getattr(v, "note", None):
        d["note"] = v.note
    return d


# --------------------------------------------------------------------------- #
class Identifier:
    """层 1..L₀ 全专家 + LR。`identify()` 线程安全（内部串行）。"""

    def __init__(self, *, model_dir: str, lr_model: str, tasks: Mapping[str, float], l0: int,
                 device: str = "cuda:0", dtype: str = "float16",
                 lr_classes: str | None = None, lr_features: str | None = None,
                 lr_feature_fn: str | None = None, tau_hi: float | None = None,
                 tau_lo: float = 0.40, tensors: Mapping | None = None,
                 hf_config: Mapping | None = None):
        from p2pmoe.runtime.lr_classifier import LRClassifier
        from p2pmoe.runtime.torch_model import TorchModelConfig, TorchSegmentModel

        cfg = dict(hf_config) if hf_config is not None else json.loads(
            (Path(model_dir) / "config.json").read_text(encoding="utf-8"))
        cfg["torch_dtype"] = dtype                       # 与通道同精度（全部 fp16）
        self.mcfg = mcfg = TorchModelConfig.from_hf(cfg)
        self.l0 = int(l0)
        E = mcfg.n_experts
        le = {l: list(range(E)) for l in range(1, self.l0 + 1)}
        t0 = time.perf_counter()
        if tensors is None:
            from p2pmoe.runtime.weights import (KeyPlan, SelectiveLoader, WeightIndex,
                                                qwen_moe_keys)

            keys = qwen_moe_keys(KeyPlan(layer_experts=le, with_embed=True),
                                 tie_word_embeddings=mcfg.tie_word_embeddings)
            tensors, rep = SelectiveLoader(WeightIndex(model_dir)).load(
                keys, device=device, dtype=mcfg.torch_dtype)
            if rep.missing:
                raise SystemExit(
                    f"{model_dir} 缺 {len(rep.missing)} 个张量（首个 {rep.missing[0]}）—— "
                    f"识别服务要层 1..{self.l0} 的**全部**专家与词嵌入，只有前段节点本地有；"
                    "换一台前段节点，或用 p2pmoe.deploy.fetch 给这台拉一份前段权重")
            self.loaded_gb = round(rep.bytes_loaded / 1e9, 3)
        else:
            self.loaded_gb = None
        self.model = TorchSegmentModel(mcfg, le, tensors, device=device,
                                       miss_policy="local_topk")
        self.load_s = round(time.perf_counter() - t0, 1)
        self.clf = LRClassifier.load(lr_model, l0=self.l0, n_experts=E, priors=dict(tasks),
                                     class_map=lr_classes, features=lr_features,
                                     feature_fn=lr_feature_fn, tau_hi=tau_hi, tau_lo=tau_lo)
        self.device = device
        self._lock = threading.Lock()
        self._n = 0

    def describe(self) -> dict:
        return {"l0": self.l0, "n_experts": self.mcfg.n_experts, "dtype": self.mcfg.dtype,
                "device": self.device, "loaded_gb": self.loaded_gb, "load_s": self.load_s,
                "classifier": self.clf.describe(), "needs_body": self.clf.needs_body,
                "tasks": self.clf.priors, "served": self._n}

    def identify(self, ids: Sequence[int], body: Sequence[Sequence[int]] | None = None,
                 *, want_layers: bool = False) -> dict:
        import torch

        from p2pmoe.runtime.lr_classifier import verdict_label

        ids = [int(x) for x in ids]
        if not ids:
            raise ValueError("prompt 是空的")
        if self.clf.needs_body and body is None:
            raise ValueError("识别器只数 body token —— 请求里要带 body 区间（p2p_body）")
        mask = ranges_to_mask(body, len(ids)) if body is not None else None
        t_wait = time.perf_counter()
        with self._lock:
            t0 = time.perf_counter()
            self._n += 1
            req = f"id{self._n}"
            try:
                with torch.inference_mode():
                    x = self.model.embed_tokens(ids)
                    _, st = self.model.forward(req, x, count_mask=mask)
            finally:
                self.model.drop_kv(req)
            t1 = time.perf_counter()
        v = self.clf.predict_stats(st)
        out = {**verdict_wire(v), "label": verdict_label(v),
               "n_prompt": len(ids),
               "n_body": int(mask.sum()) if mask is not None else None,
               "queue_ms": round((t0 - t_wait) * 1000, 3),
               "compute_ms": round((t1 - t0) * 1000, 3)}
        if want_layers:
            out["layers"] = {str(l): s.to_wire() for l, s in sorted(st.layers.items())}
        return out


# --------------------------------------------------------------------------- #
def make_app(ident: Identifier):
    from aiohttp import web

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ident")

    async def identify(request: web.Request) -> web.Response:
        t_recv = time.perf_counter()
        try:
            d = await request.json()
            ids = d["ids"]
            body = d.get("body")
        except Exception as e:                                  # noqa: BLE001
            return web.json_response({"error": f"请求要是 JSON，带 ids（与可选的 body）：{e}"},
                                     status=400)
        loop = asyncio.get_running_loop()
        try:
            out = await loop.run_in_executor(
                pool, lambda: ident.identify(ids, body, want_layers=bool(d.get("layers"))))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:                                  # noqa: BLE001
            return web.json_response({"error": f"{type(e).__name__}: {e}"}, status=500)
        out["server_ms"] = round((time.perf_counter() - t_recv) * 1000, 3)
        return web.json_response(out)

    async def health(_):
        return web.json_response({"ok": True})

    async def info(_):
        return web.json_response(ident.describe())

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.add_routes([web.post("/identify", identify), web.get("/health", health),
                    web.get("/info", info)])
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p2p-identifier", action="store_true",
                    help=argparse.SUPPRESS)  # launch.py 用它在 pgrep 里认出本进程
    ap.add_argument("--p2p-framework", default=None)
    ap.add_argument("--model-dir", required=True,
                    help="有层 1..L₀ 全专家与词嵌入的 checkpoint 目录（前段节点的那份就行）")
    ap.add_argument("--lr-model", required=True)
    ap.add_argument("--tasks", required=True, help="部署的 task 与先验，如 gsm8k=1,mbpp=1,no_robots=1")
    ap.add_argument("--l0", type=int, required=True)
    ap.add_argument("--lr-classes", default=None)
    ap.add_argument("--lr-features", default=None)
    ap.add_argument("--lr-feature-fn", default=None)
    ap.add_argument("--tau-hi", type=float, default=None)
    ap.add_argument("--tau-lo", type=float, default=0.40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args(argv)

    use_p2p_framework(a.p2p_framework)
    ident = Identifier(model_dir=a.model_dir, lr_model=a.lr_model, tasks=parse_tasks(a.tasks),
                       l0=a.l0, device=a.device, dtype=a.dtype, lr_classes=a.lr_classes,
                       lr_features=a.lr_features, lr_feature_fn=a.lr_feature_fn,
                       tau_hi=a.tau_hi, tau_lo=a.tau_lo)
    print(f"识别服务：{json.dumps(ident.describe(), ensure_ascii=False)}", flush=True)
    from aiohttp import web

    print(f"监听 {a.host}:{a.port}（POST /identify，GET /health，GET /info）", flush=True)
    web.run_app(make_app(ident), host=a.host, port=a.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
