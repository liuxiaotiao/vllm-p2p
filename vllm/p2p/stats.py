# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 · vLLM 内的逐请求路由统计（worker 进程里跑）。

p2p-framework 的节点在每层前向时顺带记两样东西（`runtime/torch_model.py`「统计口径」）：

* **全量路由**的 top-k（路由器对全部 E 个专家打分，不管本地驻留了谁）——
  前段拿它在 body token 上逐层计数，就是 LR 识别器的特征（`LayerStat`）；
* 后段的 **miss**：top-k 里有不驻留的专家的 (token, 层) 个数、缺掉的门控质量 ——
  通道二（换绑）的输入。

这里在 vLLM 上做同一件事，分两半：

1. **模型里**（`qwen3_moe_partial.py`，CUDA Graph / torch.compile 里也成立）：每个 MoE 层把
   全量 softmax 的 top-k（id 与重归一后的权重）写进一块**固定地址**的显存缓冲区
   `[最大 token 数, 本级 MoE 层数, k]`。写入走自定义算子（`torch.ops.vllm.p2p_route_select`），
   对编译器是黑盒、在 CUDA Graph 里随图重放，不改变前向的数值。
2. **model runner 里**（`gpu_model_runner.py` 两处钩子）：每步前向之后把这一步的
   `[:本步 token 数]` 搬回 CPU（**每步一次同步**），按 `input_batch` 的请求顺序与
   各 token 的位置（`positions`）切给各请求，累加；请求结束时写一行 JSONL。

每个 PP 级只统计自己那几层，各写各的文件：`$VLLM_P2P_STATS_DIR/<通道>_pp<级号>.jsonl`
（默认目录 /tmp/p2p_stats）。压测完由 `vllm/p2p/bench.py` 收回来按请求 id 拼起来。

请求 id：网关把自己的 id 放进 `vllm_xargs.p2p_id`（→ `SamplingParams.extra_args`），
body token 区间放进 `vllm_xargs.p2p_body`（"a:b,c:d"，prompt 里的 token 下标）。
没给 body 时按整条 prompt 计（`masked=False`），与 p2p-framework 不开 count_body 时一致。

开销：只在 hf_overrides 里有 `p2p_stats` 时才启用；启用后每步多一次 GPU→CPU 拷贝
（decode 时几 KB，prefill 时几 MB）与一次同步。量时延的正式实验可以不开 —— 两者的
前向数值相同（单测与 parity 里核对过）。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = ["RouteStats", "get_collector", "set_collector", "parse_body", "STATS_DIR_ENV"]

STATS_DIR_ENV = "VLLM_P2P_STATS_DIR"
DEFAULT_DIR = "/tmp/p2p_stats"

_COLLECTOR: "RouteStats | None" = None


def get_collector() -> "RouteStats | None":
    return _COLLECTOR


def set_collector(c: "RouteStats | None") -> None:
    global _COLLECTOR
    _COLLECTOR = c


def parse_body(raw) -> list[tuple[int, int]] | None:
    """`vllm_xargs.p2p_body`："3:40,52:60" 或 [[3, 40], [52, 60]] → [(3, 40), (52, 60)]。"""
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        out = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            a, _, b = part.partition(":")
            out.append((int(a), int(b)))
        return out
    return [(int(a), int(b)) for a, b in raw]


def body_string(ranges: Sequence[Sequence[int]] | None) -> str:
    return ",".join(f"{int(a)}:{int(b)}" for a, b in (ranges or ()))


# --------------------------------------------------------------------------- #
@dataclass
class _Acc:
    """一条请求在本级的累计量。"""

    p2p_id: str | None
    n_prompt: int
    body: list[tuple[int, int]] | None
    upto: int = 0
    """已经统计过的位置上界。抢占后重算的位置（< upto）不重复计。"""
    t_first: float = 0.0
    t_last: float = 0.0
    n_steps: int = 0
    # miss（本级各层合计；逐层另存）
    pre_tok_layer: int = 0
    pre_miss_tl: int = 0
    pre_miss_mass: float = 0.0
    dec_tok_layer: int = 0
    dec_miss_tl: int = 0
    dec_miss_mass: float = 0.0
    layer_miss_pre: np.ndarray | None = None
    layer_miss_dec: np.ndarray | None = None
    dec_miss_seq: list[int] = field(default_factory=list)
    """逐个 decode token：本级有几层出现了 miss（通道二的滑窗就是在它上面开的）。"""
    lr_count: dict[int, np.ndarray] = field(default_factory=dict)
    lr_mass: dict[int, np.ndarray] = field(default_factory=dict)
    lr_tok: dict[int, int] = field(default_factory=dict)


class RouteStats:
    """一个 worker 进程里的单例。由模型在构造时建（`Qwen3MoePartialForCausalLM`）。"""

    def __init__(self, *, n_experts: int, top_k: int, norm_topk: bool,
                 lr_layers: Sequence[int] = (), channel: str = "", pp_rank: int = 0,
                 out_dir: str | None = None):
        self.E = int(n_experts)
        self.k = int(top_k)
        self.norm_topk = bool(norm_topk)
        self.lr_layers = sorted({int(x) for x in lr_layers})
        self.channel = channel or "vllm"
        self.pp_rank = int(pp_rank)
        self.layers: list[int] = []
        """本级的 MoE 层（0-based，按 slot 顺序 = 构造顺序 = 层号升序）。"""
        self._resident: list[np.ndarray] = []
        self.topi = None   # torch [N, L, k] int16（设备上）
        self.topw = None   # torch [N, L, k] float32
        self.capacity = 0
        self._acc: dict[str, _Acc] = {}
        d = Path(out_dir or os.environ.get(STATS_DIR_ENV) or DEFAULT_DIR)
        self.path = d / f"{self.channel}_pp{self.pp_rank}.jsonl"
        self.n_written = 0

    # -- 构造期：模型逐层登记 ------------------------------------------------ #
    def register_layer(self, layer: int, resident: Sequence[int] | None) -> int:
        """返回该层的 slot。resident=None 表示全装。"""
        mask = np.zeros(self.E, dtype=bool)
        if resident is None:
            mask[:] = True
        else:
            mask[np.asarray(list(resident), dtype=np.int64)] = True
        self.layers.append(int(layer))
        self._resident.append(mask)
        return len(self.layers) - 1

    def allocate(self, capacity: int, device) -> None:
        """模型构造完、CUDA Graph 捕获之前调：缓冲区地址从此固定。"""
        import torch

        L = max(1, len(self.layers))
        self.capacity = int(capacity)
        self.topi = torch.zeros((self.capacity, L, self.k), dtype=torch.int16, device=device)
        self.topw = torch.zeros((self.capacity, L, self.k), dtype=torch.float32, device=device)
        self._res_table = np.stack(self._resident) if self._resident else np.ones((1, self.E), bool)

    # -- 前向里（自定义算子的实现调它）--------------------------------------- #
    def capture(self, slot: int, router_logits) -> None:
        import torch

        if self.topi is None:
            return
        probs = torch.softmax(router_logits.float(), dim=-1)
        w, i = torch.topk(probs, self.k, dim=-1)
        if self.norm_topk:
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-20)
        n = min(int(router_logits.shape[0]), self.capacity)
        self.topi[:n, slot].copy_(i[:n].to(torch.int16))
        self.topw[:n, slot].copy_(w[:n])

    # -- runner：每步一次 ----------------------------------------------------- #
    def on_step(self, req_ids: Sequence[str], num_scheduled: Sequence[int],
                positions: np.ndarray, requests: Mapping) -> None:
        """一步前向之后调。

        `req_ids` / `num_scheduled` 按 `input_batch` 的顺序（本步 token 就是按这个顺序连续排的），
        `positions` 是本步每个 token 在其请求里的位置（runner 的 `self.positions.np[:T]`），
        `requests` 是 runner 的 `self.requests`（req_id → CachedRequestState）。"""
        T = int(np.sum(num_scheduled))
        if T <= 0 or self.topi is None or not self.layers:
            return
        if T > self.capacity:
            # 不该发生（缓冲区按 max_num_batched_tokens 分配）；宁可丢这一步也不越界
            return
        ids = self.topi[:T].cpu().numpy().astype(np.int64)     # 唯一一次同步
        ws = self.topw[:T].cpu().numpy()
        now = time.time()
        off = 0
        for rid, n in zip(req_ids, num_scheduled):
            n = int(n)
            if n <= 0:
                continue
            st = requests.get(rid)
            acc = self._acc.get(rid)
            if acc is None:
                ea = (getattr(getattr(st, "sampling_params", None), "extra_args", None)
                      or {}) if st is not None else {}
                n_prompt = int(getattr(st, "num_prompt_tokens", 0) or 0)
                acc = self._acc[rid] = _Acc(p2p_id=ea.get("p2p_id"), n_prompt=n_prompt,
                                            body=parse_body(ea.get("p2p_body")),
                                            t_first=now)
                L = len(self.layers)
                acc.layer_miss_pre = np.zeros(L, dtype=np.int64)
                acc.layer_miss_dec = np.zeros(L, dtype=np.int64)
            self._account(acc, ids[off:off + n], ws[off:off + n],
                          np.asarray(positions[off:off + n], dtype=np.int64))
            acc.t_last = now
            acc.n_steps += 1
            off += n

    def _account(self, acc: _Acc, ids: np.ndarray, ws: np.ndarray, pos: np.ndarray) -> None:
        keep = pos >= acc.upto
        if not keep.any():
            return
        ids, ws, pos = ids[keep], ws[keep], pos[keep]
        acc.upto = int(pos.max()) + 1
        L = len(self.layers)
        here = self._res_table[np.arange(L)[None, :, None], ids]        # [n, L, k]
        miss_any = ~here.all(-1)                                         # [n, L]
        miss_mass = (ws * ~here).sum(-1)                                 # [n, L]
        pre = pos < acc.n_prompt
        dec = ~pre
        if pre.any():
            acc.pre_tok_layer += int(pre.sum()) * L
            acc.pre_miss_tl += int(miss_any[pre].sum())
            acc.pre_miss_mass += float(miss_mass[pre].sum())
            acc.layer_miss_pre += miss_any[pre].sum(0)
            self._lr(acc, ids[pre], ws[pre], pos[pre])
        if dec.any():
            acc.dec_tok_layer += int(dec.sum()) * L
            acc.dec_miss_tl += int(miss_any[dec].sum())
            acc.dec_miss_mass += float(miss_mass[dec].sum())
            acc.layer_miss_dec += miss_any[dec].sum(0)
            acc.dec_miss_seq.extend(int(x) for x in miss_any[dec].sum(1))

    def _lr(self, acc: _Acc, ids: np.ndarray, ws: np.ndarray, pos: np.ndarray) -> None:
        """前 L₀ 层：只数 body token 的 top-k 次数与门控质量（= LayerStat）。"""
        if not self.lr_layers:
            return
        if acc.body is not None:
            m = np.zeros(pos.shape[0], dtype=bool)
            for a, b in acc.body:
                m |= (pos >= a) & (pos < b)
            ids, ws = ids[m], ws[m]
        if ids.shape[0] == 0:
            # 本步没有 body token，但层要登记上（n_tokens=0），否则缺层
            for li, l in enumerate(self.layers):
                if l in self.lr_layers:
                    acc.lr_count.setdefault(l, np.zeros(self.E, np.int64))
                    acc.lr_mass.setdefault(l, np.zeros(self.E, np.float64))
                    acc.lr_tok.setdefault(l, 0)
            return
        for li, l in enumerate(self.layers):
            if l not in self.lr_layers:
                continue
            cnt = acc.lr_count.setdefault(l, np.zeros(self.E, np.int64))
            mas = acc.lr_mass.setdefault(l, np.zeros(self.E, np.float64))
            np.add.at(cnt, ids[:, li, :].reshape(-1), 1)
            np.add.at(mas, ids[:, li, :].reshape(-1), ws[:, li, :].reshape(-1).astype(np.float64))
            acc.lr_tok[l] = acc.lr_tok.get(l, 0) + int(ids.shape[0])

    # -- runner：请求结束 ----------------------------------------------------- #
    def finish(self, req_ids) -> None:
        rows = []
        for rid in req_ids:
            acc = self._acc.pop(rid, None)
            if acc is not None:
                rows.append(self.record(rid, acc))
        if not rows:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, separators=(",", ":")) + "\n")
            self.n_written += len(rows)
        except OSError:
            pass                       # 统计落不了盘不能拖垮推理

    def record(self, rid: str, acc: _Acc) -> dict:
        masked = acc.body is not None
        lr = {}
        for l in sorted(acc.lr_count):
            c, w = acc.lr_count[l], acc.lr_mass[l]
            nz = np.flatnonzero(c)
            # 与 p2p-framework `LayerStat.to_wire` 同格式；键是 **1-based**（规划器口径）
            lr[str(l + 1)] = {"i": nz.tolist(), "c": c[nz].astype(int).tolist(),
                              "w": [round(float(x), 6) for x in w[nz]],
                              "t": int(acc.lr_tok.get(l, 0)), "e": self.E,
                              **({"m": 1} if masked else {})}
        return {
            "req_id": rid, "p2p_id": acc.p2p_id, "channel": self.channel,
            "pp_rank": self.pp_rank, "layers": [l + 1 for l in self.layers],
            "n_prompt": acc.n_prompt, "n_seen": acc.upto, "n_steps": acc.n_steps,
            "t_first": round(acc.t_first, 6), "t_last": round(acc.t_last, 6),
            "prefill": {"n_token_layer": acc.pre_tok_layer, "miss_token_layer": acc.pre_miss_tl,
                        "miss_mass": round(acc.pre_miss_mass, 6)},
            "decode": {"n_token_layer": acc.dec_tok_layer, "miss_token_layer": acc.dec_miss_tl,
                       "miss_mass": round(acc.dec_miss_mass, 6)},
            "layer_miss": {"prefill": acc.layer_miss_pre.tolist(),
                           "decode": acc.layer_miss_dec.tolist()},
            "decode_miss_seq": acc.dec_miss_seq,
            **({"lr_layers": lr, "body": [list(x) for x in acc.body] if masked else None}
               if lr else {}),
        }
