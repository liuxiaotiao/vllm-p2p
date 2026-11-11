# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1 分段服务的离散事件模拟器（v1）：粒度到「每个分段引擎的每一步」。

模拟的是 `tiered_gateway.py` + 分段引擎这套真系统在 bench 压测下的样子，输出与 bench.py **同一种
结果 JSON**，所以 `tiered_stats.py --b sim.json --a real.json` 直接出对比表。

    # 1. 照一次真机结果回放（请求顺序、每条的 prompt 长度 / 生成长度 / LR 判成哪类都取自它）
    python -m vllm.p2p.sim run --stages p2p_channels/stages.json --replay results/b1_lr_pf1_x.json \\
        --layer-bench results/layer_bench/summary_by_gpu.json --hosts <hosts.txt> --per-front 1 \\
        --out results/sim_pf1_x.json
    python3 vllm/p2p/tiered_stats.py --b results/sim_pf1_x.json --a results/b1_lr_pf1_x.json
    # 2. 标定三个自由参数（前段 / 后段每步固定开销、每跳软件开销），拿真机结果拟合
    python -m vllm.p2p.sim fit  …同上… --out results/sim_fit_x.json
    # 3. 对比任意两份结果（关键指标 + 相对误差）
    python -m vllm.p2p.sim compare --sim results/sim_pf1_x.json --real results/b1_lr_pf1_x.json

模型
----
**请求**：网关挑前段（`TieredCoordinator.assign_front`，`--front-slots` 满了在网关排队）→ 前段 prefill
→ prompt 的 hidden 交给网关、LR 判类（回放真机判的类）→ `bind_decision`（hold：这类后段坐满就在前段
等，后段做完一条时 `drain_plan` 按先来先到接上）→ 后段 prefill 出第 1 个 token → decode 循环：
后段出 token → 直连发给前段 → 前段算一步 → hidden 直连发给后段 → 后段算一步 … → 生成够长度结束。
**挑前段 / 挑后段 / 等待队列用的就是真网关那几个同步函数**（tiered.py），不是另写一份。

**引擎**：照 vLLM 的调度循环 —— 空闲时一有输入就开一步；一步取「输入已经到了」的请求（已在跑的
decode 优先，新请求 prefill 受 max_num_seqs 与每步 token 上限约束）；算的时候新到的输入等下一步。
一步用时 = 每步固定开销 + Σ各 PP 段 [层数 × 每层耗时] + 段间转发。每层耗时查 layer_bench：
decode[专家数][上下文 C][批 B]（B 向上补到量过的档位 —— CUDA Graph 也是这么补的；C 线性插值、
超出范围线性外推），prefill[专家数][P]。给了 `--steps`（steptrace 收回来的步级记录）时，
纯 decode 步直接用那个引擎实测的「这个批大小的每步 p50」。

**网络**：每条消息 = 单向时延 + 字节 / 带宽 + 每跳软件开销（`--hop-ms`）。单向时延与带宽取
`--topo` 的 edges；没有拓扑就按 IP 同不同 /24 网段取 `--lat-same` / `--lat-cross`。

**整条流水线（A）**：给的是 layout.py 的 channels.json（不是 stages.json）时，每条通道是一个从第 0 层
到最后一层的 vLLM 引擎（PP 多机），网关按 gateway.py 的 Router 挑通道（`--a-mode any` = 全部通道里
在途最少的，`oracle` = 按标注的类）；decode 循环全在引擎里，不经过网关。挑通道用的也是真网关的
`gateway.Router`。这种部署的每步固定开销用 `--back-fixed-ms`（含 embedding、lm_head、采样）。

**多机 PP 的一步**：调度器在 node_rank 0，每步先把调度结果发给其余各台；第 i 段等调度结果与上一段的
hidden 都到了才算；最后一段采样的 token 回到 node_rank 0 才算这一步完（`--pp-simple` 回到 v1 只算
往下传的几跳）。时延按 `--topo` / 网段估，每次 PP 传输另加 `--pp-hop-ms`。

v1 的简化（与真机比对时心里要有数）：几批之间不重叠（不模拟 vLLM 的 batch queue；批大小 1 时
没有区别）；不模拟 KV 容量与抢占（槽位小时不会发生）；识别结果、生成长度回放真机。
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import re
import shlex
import statistics as stt
import sys
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_MARKS = (64, 128, 256, 512, 1024, 2048)


# --------------------------------------------------------------------------- #
# 每层耗时表（layer_bench）
# --------------------------------------------------------------------------- #
def gpu_key(name: str) -> str:
    s = str(name or "").upper()
    for p in ("NVIDIA ", "TESLA ", "QUADRO ", "GEFORCE "):
        s = s.replace(p, "")
    return re.sub(r"[\s\-_]", "", s)


def _interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """分段线性插值；两端之外用最近两点线性外推（不小于 0）。"""
    if len(xs) == 1:
        return float(ys[0])
    i = bisect_left(xs, x)
    if i < len(xs) and xs[i] == x:
        return float(ys[i])
    if i == 0:
        i = 1
    elif i >= len(xs):
        i = len(xs) - 1
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return max(0.0, y0 + (y1 - y0) * (x - x0) / (x1 - x0))


@dataclass
class GpuPerf:
    """一种卡的 layer_bench 表：decode[E][C][B]、prefill[E][P]、decode_overhead_ms[E][C][B]（每层毫秒）。"""
    gpu: str
    decode: dict
    prefill: dict
    overhead: dict

    def _e(self, table: Mapping, experts: int) -> Mapping:
        cache = self.__dict__.setdefault("_ecache", {})
        if (id(table), experts) in cache:
            return cache[(id(table), experts)]
        cache[(id(table), experts)] = self._e_uncached(table, experts)
        return cache[(id(table), experts)]

    def _e_uncached(self, table: Mapping, experts: int) -> Mapping:
        ks = sorted(int(k) for k in table if table[k])
        if not ks:
            raise ValueError(f"{self.gpu} 的 layer_bench 表是空的")
        e = min(ks, key=lambda k: (abs(k - experts), -k))
        return table[str(e)]

    def _by_batch(self, tab: Mapping, batch: int) -> tuple[list[int], list[float]]:
        """固定批大小时各上下文长度的每层耗时（缓存：每步都要查）。"""
        cache = self.__dict__.setdefault("_cache", {})
        key = (id(tab), batch)
        if key not in cache:
            cs = sorted(int(c) for c in tab)
            vals = []
            for c in cs:
                row = tab[str(c)]
                bs = sorted(int(b) for b in row)
                up = [b for b in bs if b >= batch]
                if up:                               # 补到量过的档位（CUDA Graph 同样补齐）
                    vals.append(float(row[str(up[0])]))
                else:                                # 超过最大档：按最后两档外推
                    vals.append(_interp(bs, [row[str(b)] for b in bs], batch))
            cache[key] = (cs, vals)
        return cache[key]

    def _cb(self, tab: Mapping, ctx: float, batch: int) -> float:
        cs, vals = self._by_batch(tab, batch)
        return _interp(cs, vals, ctx)

    def decode_ms(self, experts: int, ctx: float, batch: int) -> float:
        return self._cb(self._e(self.decode, experts), ctx, batch)

    def overhead_ms(self, experts: int, ctx: float, batch: int) -> float:
        try:
            return self._cb(self._e(self.overhead, experts), ctx, batch)
        except ValueError:
            return 0.0

    def prefill_ms(self, experts: int, n: int) -> float:
        tab = self._e(self.prefill, experts)
        ps = sorted(int(p) for p in tab)
        return _interp(ps, [tab[str(p)] for p in ps], n)


def load_perf(paths: Sequence[str | Path]) -> dict[str, GpuPerf]:
    """layer_bench 的 summary_by_gpu.json（{卡型: {...}}）或若干台的单机结果（同卡型取均值）。"""
    by: dict[str, list[Mapping]] = defaultdict(list)
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if "decode" in d and isinstance(d.get("decode"), dict) and ("gpu" in d or "node" in d):
            by[gpu_key(d.get("gpu", ""))].append(d)
        else:
            for g, v in d.items():
                if isinstance(v, dict) and "decode" in v:
                    by[gpu_key(v.get("gpu") or g)].append(v)
    if not by:
        raise SystemExit(f"{list(map(str, paths))} 里没有 layer_bench 的 decode 表")

    def mean_tree(objs):
        objs = [o for o in objs if o is not None]
        if not objs:
            return {}
        if isinstance(objs[0], Mapping):
            keys = set().union(*[o.keys() for o in objs if isinstance(o, Mapping)])
            return {k: mean_tree([o.get(k) for o in objs if isinstance(o, Mapping)]) for k in keys}
        nums = [float(o) for o in objs if isinstance(o, (int, float))]
        return sum(nums) / len(nums) if nums else None

    return {g: GpuPerf(g, mean_tree([r.get("decode") for r in rs]), mean_tree([r.get("prefill") for r in rs]),
                       mean_tree([r.get("decode_overhead_ms") for r in rs])) for g, rs in by.items()}


# --------------------------------------------------------------------------- #
# 网络
# --------------------------------------------------------------------------- #
@dataclass
class Net:
    """节点间单向时延（ms）与带宽（Mbps）。没拓扑时按 IP 网段估。"""
    ip: dict[str, str]
    lat: dict[tuple[str, str], float] = field(default_factory=dict)
    bw: dict[tuple[str, str], float] = field(default_factory=dict)
    lat_same: float = 0.1
    lat_cross: float = 3.15
    bw_default: float = 10000.0
    lat_gw: float = 0.1                    # 网关 ↔ 节点（没给 --gateway-node 时）
    gateway: str | None = None
    hop_ms: float = 2.0                    # 网关 HTTP 那条路的每跳开销（整条流水线：gateway.py → vLLM API 服务器）
    agent_hop_ms: float = 2.0              # 分段部署：协调器 ↔ 代理、前段 ↔ 后段直连（ZMQ + 代理 + EngineCore）每跳开销
    pp_hop_ms: float = 0.3                 # 多机 PP 每次传输（段间 hidden / 调度下发 / 结果回传）的固定开销
    back_pp_hop_ms: float | None = None    # 分段部署后段的；None = 同 pp_hop_ms

    def one_way(self, a: str | None, b: str | None) -> float:
        if a is None or b is None:
            return self.lat_gw
        if a == b:
            return 0.0
        if (a, b) in self.lat:
            return self.lat[(a, b)]
        if (b, a) in self.lat:
            return self.lat[(b, a)]
        ia, ib = self.ip.get(a, ""), self.ip.get(b, "")
        same = ia.rsplit(".", 1)[0] == ib.rsplit(".", 1)[0] if ia and ib else True
        return self.lat_same if same else self.lat_cross

    def xfer(self, a: str | None, b: str | None, nbytes: float) -> float:
        bw = self.bw.get((a, b)) or self.bw.get((b, a)) or self.bw_default
        return nbytes * 8 / (bw * 1e3)                      # Mbps → bit/ms = bw×1e3

    def msg(self, a: str | None, b: str | None, nbytes: float = 0.0, kind: str = "gw") -> float:
        """一条消息：单向 + 传输 + 软件开销。a/b 为 None = 网关。kind="gw" 走 HTTP 网关那条路
        （hop_ms），"agent" 走分段部署的代理（agent_hop_ms）—— 两条路的软件栈不同，开销分开标定。"""
        a = self.gateway if a is None else a
        b = self.gateway if b is None else b
        return self.one_way(a, b) + self.xfer(a, b, nbytes) + (self.agent_hop_ms if kind == "agent" else self.hop_ms)

    def pp(self, a: str, b: str, nbytes: float, hop: float | None = None) -> float:
        return self.one_way(a, b) + self.xfer(a, b, nbytes) + (self.pp_hop_ms if hop is None else hop)

    def ctrl(self, a: str, b: str, hop: float | None = None) -> float:
        """多机 PP：node_rank 0 把这一步的调度结果发给 b（小消息）。"""
        return self.one_way(a, b) + (self.pp_hop_ms if hop is None else hop)

    def pp_hop_for(self, role: str) -> float:
        """分段部署的后段（stage_agent 驱动）与整条流水线（vllm serve）的 PP 每次传输开销分开标定。"""
        if role == "back" and self.back_pp_hop_ms is not None:
            return self.back_pp_hop_ms
        return self.pp_hop_ms


def load_topo(path: str | Path | None) -> tuple[dict, dict, dict]:
    """拓扑 JSON → (单向时延, 带宽, 节点卡型)。"""
    if not path:
        return {}, {}, {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    lat, bw = {}, {}
    for e in d.get("edges") or []:
        lat[(e["u"], e["v"])] = float(e.get("propagation_delay", 0.0))
        if e.get("bandwidth"):
            bw[(e["u"], e["v"])] = float(e["bandwidth"])
    roles = {n["id"]: gpu_key(n.get("role", "")) for n in d.get("nodes") or []}
    return lat, bw, roles


def load_hosts(path: str | Path | None) -> tuple[dict, dict]:
    """hosts.txt：`节点 地址[:端口] # 卡型 …` → (IP, 卡型)。"""
    ip, gpu = {}, {}
    if not path:
        return ip, gpu
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        body, _, comment = raw.partition("#")
        parts = body.split()
        if len(parts) < 2:
            continue
        ip[parts[0]] = parts[1].rsplit(":", 1)[0] if parts[1].count(":") == 1 else parts[1]
        if comment.split():
            gpu[parts[0]] = gpu_key(comment.split()[0])
    return ip, gpu


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    front_fixed_ms: float | None = None      # 前段每步固定开销；None = layer_bench 截距 × 0.5（无 lm_head）
    back_fixed_ms: float | None = None       # 后段每步固定开销；None = layer_bench 截距（含 lm_head、采样）
    lr_ms: float = 2.0                       # 网关上跑 LR 的时间
    client_ms: float = 1.0                   # bench 收到完成到发出下一条
    max_batched_tokens: int = 8192
    hidden: int = 2048                       # hidden 维度（每位置 hidden × 2 字节）
    pp_simple: bool = False                  # True = v1 的多机 PP 算法（不算调度下发与结果回传）
    # 含 prefill 的步额外的固定开销（layer_bench 的逐层斜率量不到：prefill 不走 CUDA Graph 等）。
    # pf_fixed_per：step = 每步一次；stage = 每个 PP 段一次；layer = 每层一次
    pf_fixed_ms: float = 0.0
    pf_fixed_per: str = "stage"
    # 凑批等待（假设实验，真引擎还没有）：引擎空闲、手上的 decode 输入比「在它这儿跑的请求」少时，
    # 最多再等这么久让其余请求的输入也到齐，再一起算一步。0 = 不等（= 现在的 vLLM）
    batch_wait_ms: float = 0.0
    batch_wait_role: str = "back"            # back / front / all

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class SimEngine:
    def __init__(self, ch: Mapping, perf: Mapping[str, GpuPerf], node_gpu: Mapping[str, str],
                 experts: int, sim: "Sim", steps: Mapping | None = None):
        self.name, self.role, self.task = ch["name"], ch["role"], ch.get("task")
        self.nodes = list(ch["nodes"])
        self.layers = [int(x) for x in ch["partition"]]
        self.max_seqs = int(ch.get("max_num_seqs") or 256)
        self.gpus = []
        for v in self.nodes:
            g = node_gpu.get(v)
            if g is None:
                raise SystemExit(f"不知道节点 {v} 是什么卡 —— 给 --hosts（注释写卡型）或 --topo（role）")
            if g not in perf:
                raise SystemExit(f"layer_bench 里没有卡型 {g}（节点 {v}）；有 {sorted(perf)}")
            self.gpus.append(perf[g])
        self.experts = int(experts)
        self.sim = sim
        # 实测（steptrace）：{"dec": {批大小: 每步毫秒}, "pf": {a_ms, b_ms_per_token | ms_per_token_p50}}
        self.steps = steps or {}
        self.ready: dict[str, tuple] = {}        # rid → ("pf", n) | ("dec", ctx)
        self.admitted: set[str] = set()
        self.busy = False
        self._in_done = False
        self._kick_due = False
        self.t_ready: dict[str, float] = {}
        # 统计
        self.n_steps = 0
        self.busy_ms = 0.0
        self.batch_sum = 0
        self.t_first_step: float | None = None
        self.t_last_end = 0.0

    @property
    def node0(self) -> str:
        return self.nodes[0]

    def fixed_ms(self, ctx: float, batch: int) -> float:
        p = self.sim.params
        v = p.front_fixed_ms if self.role == "front" else p.back_fixed_ms
        if v is not None:
            return float(v)
        g = self.gpus[0] if self.role == "front" else self.gpus[-1]
        oh = g.overhead_ms(self.experts, ctx, batch)
        return oh * (0.5 if self.role == "front" else 1.0)

    def step_ms(self, dec_ctx: Sequence[float], pf_tokens: Sequence[int]) -> float:
        """一步的用时。多机 PP 时按 vLLM 的实际过程排时间线：

        1. 调度器在 node_rank 0 上：把这一步的调度结果同时发给其余各台（`Net.ctrl`）；
        2. 第 i 段要等「调度结果到了」和「上一段的 hidden 到了」都满足才开算；
        3. 最后一段采样出的 token 要回到 node_rank 0（EngineCore 在那儿出结果）。

        `Params.pp_simple` = v1 的算法（各段串行相加，只算往下传的那几跳）。"""
        b = len(dec_ctx)
        ctx = (sum(dec_ctx) / b) if b else 0.0
        dec_meas = self.steps.get("dec") or {}
        if b and not pf_tokens and b in dec_meas:
            return float(dec_meas[b])
        pfm = self.steps.get("pf")
        if pf_tokens and not b and pfm:                   # 只有 prefill 的步：用实测的线性拟合
            n = sum(pf_tokens)
            if "a_ms" in pfm:
                return float(pfm["a_ms"]) + float(pfm["b_ms_per_token"]) * n
            return float(pfm["ms_per_token_p50"]) * n
        fixed = self.fixed_ms(ctx, max(b, 1))
        net = self.sim.net
        tok = b + sum(pf_tokens)
        H = self.sim.params.hidden * 2
        comp = []
        P = self.sim.params
        for g, nl in zip(self.gpus, self.layers):
            c = nl * g.decode_ms(self.experts, ctx, b) if b else 0.0
            c += sum(nl * g.prefill_ms(self.experts, n) for n in pf_tokens)
            if pf_tokens and P.pf_fixed_ms:
                c += P.pf_fixed_ms * (nl if P.pf_fixed_per == "layer" else 1 if P.pf_fixed_per == "stage" else 0)
            comp.append(c)
        if pf_tokens and P.pf_fixed_ms and P.pf_fixed_per == "step":
            fixed += P.pf_fixed_ms
        if self.sim.params.pp_simple or len(self.nodes) == 1:
            t = fixed + sum(comp)
            for i in range(len(self.nodes) - 1):
                t += net.pp(self.nodes[i], self.nodes[i + 1], tok * H, hop=net.pp_hop_for(self.role))
            return t
        n0 = self.nodes[0]
        end = comp[0]
        for i in range(1, len(self.nodes)):
            ph = net.pp_hop_for(self.role)
            start = max(net.ctrl(n0, self.nodes[i], hop=ph),
                        end + net.pp(self.nodes[i - 1], self.nodes[i], tok * H, hop=ph))
            end = start + comp[i]
        end += net.pp(self.nodes[-1], n0, max(b, 1) * 8, hop=net.pp_hop_for(self.role))   # 采样结果回 node_rank 0
        return fixed + end

    # ---- 调度 ----
    def put(self, rid: str, item: tuple, now: float) -> None:
        self.ready[rid] = item
        self.t_ready.setdefault(rid, now)
        if not self._in_done and not self._kick_due:
            # 不马上开算：排一个同一时刻的事件（排在已经排好的同时刻到达之后）。同一时刻到的几条输入
            # 是真机里代理按目的地合成的一条消息（后段一步出的几个 token 发给同一个前段，前段一步的
            # 几个 hidden 发给同一个后段），要进同一步。一步结束时逐条放回的，等全放完再开下一步（_done）
            self._kick_due = True
            self.sim.at(now, self._kick_now)

    def _kick_now(self, now: float) -> None:
        self._kick_due = False
        self.kick(now)

    def _should_wait(self, now: float) -> bool:
        """凑批等待：只等「已经在这个引擎上跑、但这一步的输入还在路上」的请求；最多等 batch_wait_ms
        （从引擎能开算的时刻算起）。到点由一个定时事件再 kick 一次。"""
        P = self.sim.params
        if P.batch_wait_ms <= 0 or (P.batch_wait_role != "all" and P.batch_wait_role != self.role):
            return False
        if any(it[0] == "pf" for it in self.ready.values()):
            return False
        missing = [r for r in self.admitted if r not in self.ready]
        if not missing:
            return False
        # 从「引擎能开算」的时刻算起：手上最早的输入到达、上一步结束，两者取晚的。两条请求交替时，
        # 后到的那条的输入早在上一步算的时候就到了 —— 从它到达算起，窗口在上一步里就过完了，永远合不上
        first = min(self.t_ready.get(r, now) for r in self.ready)
        deadline = max(first, self.t_last_end) + P.batch_wait_ms
        if now + 1e-9 >= deadline:
            return False
        if getattr(self, "_wake_at", None) != deadline:
            self._wake_at = deadline
            self.sim.at(deadline, lambda t: self.kick(t))
        return True

    def kick(self, now: float) -> None:
        if self.busy or not self.ready:
            return
        if self._should_wait(now):
            self.n_waits = getattr(self, "n_waits", 0) + 1
            return
        dec, pf = [], []
        tok = 0
        for rid, it in list(self.ready.items()):
            if it[0] == "dec" and rid in self.admitted:
                dec.append((rid, it))
                tok += 1
        for rid, it in list(self.ready.items()):
            if it[0] != "pf":
                continue
            if rid not in self.admitted and len(self.admitted) >= self.max_seqs:
                continue
            if tok + it[1] > self.sim.params.max_batched_tokens and (dec or pf):
                continue
            self.admitted.add(rid)
            pf.append((rid, it))
            tok += it[1]
        if not dec and not pf:
            return
        for rid, _ in dec + pf:
            del self.ready[rid]
        ms = self.step_ms([it[1] for _, it in dec], [it[1] for _, it in pf])
        self.busy = True
        self.n_steps += 1
        self.busy_ms += ms
        self.batch_sum += len(dec) + len(pf)
        if self.t_first_step is None:
            self.t_first_step = now
        for rid, it in pf:
            self.sim.reqs[rid].queue_engine_ms[self.role] = now - self.t_ready.pop(rid, now)
        for rid, _ in dec:
            self.t_ready.pop(rid, None)
        if self.role == "front" and dec:
            self.sim.co_eng[self.name].observe(ms)
        self.sim.at(now + ms, self._done, dec, pf, ms)

    def _done(self, now: float, dec, pf, ms) -> None:
        self.busy = False
        self.t_last_end = now
        self._in_done = True
        try:
            for rid, it in pf + dec:
                self.sim.on_step_out(self, rid, it, now)
        finally:
            self._in_done = False
        self.kick(now)

    def release(self, rid: str, now: float | None = None) -> None:
        self.admitted.discard(rid)
        self.ready.pop(rid, None)
        self.t_ready.pop(rid, None)
        if now is not None:              # 可能有人在凑批等它：它走了就不必再等
            self.kick(now)


# --------------------------------------------------------------------------- #
# 请求与整套系统
# --------------------------------------------------------------------------- #
@dataclass
class SimReq:
    rid: str
    case_id: str | None
    true_task: str | None
    task: str | None                 # 识别后用来绑定的类（回放真机）
    n_prompt: int
    n_gen: int
    stop_reason: str = "length"
    front: Any = None                # tiered.Engine（策略用）
    back: Any = None
    identified: bool = False
    t_wait: float | None = None
    t_bound: float | None = None
    t_submit: float = 0.0
    t_gw_wait: float = 0.0           # 在网关等空闲前段
    t_first: float | None = None
    t_done: float | None = None
    t_front_done: float | None = None
    t_front_seen: float | None = None   # 前段 prefill 的输出到网关（协调器）的时刻
    t_first_gw: float | None = None     # 第一个 token 到网关的时刻
    t_in: dict = field(default_factory=dict)   # 角色 → prompt 到这个引擎的时刻（引擎内毫秒的起点）
    t_back_out0: float | None = None    # 后段出第 1 个 token 的时刻
    toks: list = field(default_factory=list)       # 每个 token 到客户端的时刻
    n_out: int = 0
    queue_engine_ms: dict = field(default_factory=dict)


class Sim:
    def __init__(self, stages: Mapping, *, perf: Mapping[str, GpuPerf], net: Net, node_gpu: Mapping,
                 params: Params, front_slots: int | None, hold: bool, front_pick: str = "least",
                 bind: str = "least", steps: Mapping[str, Mapping] | None = None,
                 n_experts: int | None = None, a_mode: str = "any"):
        self.params, self.net = params, net
        self.now = 0.0
        self._q: list = []
        self._seq = itertools.count()
        self.reqs: dict[str, SimReq] = {}
        chans = {c["name"]: c for c in stages["channels"]}
        ne = int(n_experts or stages.get("n_experts") or 128)
        steps = steps or {}
        self.engines: dict[str, SimEngine] = {}
        self.gw_queue: list[SimReq] = []
        self.on_finish: Callable[[SimReq, float], None] = lambda r, t: None
        self.full = stages.get("kind") != "tiered"
        self.kind = "gw" if self.full else "agent"
        if self.full:
            # 整条流水线（A）：每条通道一个引擎；挑通道用 gateway.Router（与真网关同一份）
            from vllm.p2p.gateway import ChannelInfo, Router
            for c in stages["channels"]:
                cc = dict(c, role="full", partition=c.get("partition") or [])
                self.engines[c["name"]] = SimEngine(cc, perf, node_gpu, resident_experts(c) or ne, self,
                                                    steps.get(c["name"]))
            self.router = Router([ChannelInfo(name=c["name"], task=c.get("task") or "", api="sim",
                                              nodes=list(c["nodes"])) for c in stages["channels"]])
            self.a_mode = a_mode
            self.co = None
            return
        from vllm.p2p.tiered import Engine, TieredCoordinator

        for name in [*stages["fronts"], *stages["backs"]]:
            c = chans[name]
            e = ne if c["role"] == "front" else (resident_experts(c) or ne)
            self.engines[name] = SimEngine(c, perf, node_gpu, e, self, steps.get(name))
        self.co_eng = {n: Engine(n, chans[n]["role"], None, task=chans[n].get("task"),
                                 slots=(int(chans[n]["max_num_seqs"]) if hold and chans[n]["role"] == "back"
                                        and chans[n].get("max_num_seqs") else None))
                       for n in self.engines}
        self.co = TieredCoordinator([self.co_eng[n] for n in stages["fronts"]],
                                    [self.co_eng[n] for n in stages["backs"]], bind=bind,
                                    front_pick=front_pick, hold=hold, front_slots=front_slots)
        self.co.clock = lambda: self.now

    # ---- 事件 ----
    def at(self, t: float, fn: Callable, *args) -> None:
        heapq.heappush(self._q, (t, next(self._seq), fn, args))

    def run(self, until: float = math.inf) -> None:
        while self._q:
            t, _, fn, args = heapq.heappop(self._q)
            if t > until:
                break
            self.now = t
            fn(t, *args)

    # ---- 网关 ----
    def submit(self, r: SimReq, t: float) -> None:
        self.reqs[r.rid] = r
        r.t_submit = t
        if self.full:
            ch = self.router.pick_any() if self.a_mode == "any" else self.router.pick(r.task)
            ch.inflight += 1
            ch.served += 1
            r.back = ch
            e = self.engines[ch.name]
            self.at(t + self.net.msg(None, e.node0, r.n_prompt * 4, kind=self.kind), self._arrive, e, r.rid, ("pf", r.n_prompt))
            return
        if self.co.front_slots and not self.co._front_has_room():
            self.co.n_front_wait += 1
            self.gw_queue.append(r)
            return
        self._to_front(r, t)

    def _to_front(self, r: SimReq, t: float) -> None:
        f = self.co.assign_front()
        r.front = f
        r.t_gw_wait = t - r.t_submit
        e = self.engines[f.name]
        self.at(t + self.net.msg(None, e.node0, r.n_prompt * 4, kind=self.kind), self._arrive, e, r.rid, ("pf", r.n_prompt))

    def _arrive(self, t: float, e: SimEngine, rid: str, item: tuple) -> None:
        r = self.reqs.get(rid)
        if r is not None and r.t_done is None:
            if item[0] == "pf":
                r.t_in.setdefault(e.role, t)
            e.put(rid, item, t)

    def _identified(self, t: float, r: SimReq) -> None:
        r.identified = True
        b = self.co.bind_decision(r)
        if b is not None:
            self._send_prompt(r, b, t)

    def _send_prompt(self, r: SimReq, b, t: float) -> None:
        e = self.engines[b.name]
        self.at(t + self.net.msg(None, e.node0, r.n_prompt * self.params.hidden * 2, kind=self.kind),
                self._arrive, e, r.rid, ("pf", r.n_prompt))

    def _finish(self, t: float, r: SimReq) -> None:
        r.t_done = t
        if self.full:
            r.back.inflight -= 1
            self.engines[r.back.name].release(r.rid, t)
            self.on_finish(r, t)
            return
        r.back.inflight -= 1
        r.front.inflight -= 1
        self.engines[r.front.name].release(r.rid, t)
        self.engines[r.back.name].release(r.rid, t)
        for w, b in self.co.drain_plan():
            self._send_prompt(w, b, t)
        while self.gw_queue and self.co._front_has_room():
            self._to_front(self.gw_queue.pop(0), t)
        self.on_finish(r, t)

    # ---- 引擎出结果 ----
    def on_step_out(self, e: SimEngine, rid: str, item: tuple, t: float) -> None:
        r = self.reqs[rid]
        H = self.params.hidden * 2
        if e.role == "front":
            if item[0] == "pf":
                r.t_front_done = t
                r.t_front_seen = t + self.net.msg(e.node0, None, r.n_prompt * H, kind=self.kind)
                self.at(r.t_front_seen + self.params.lr_ms, self._identified, r)
            else:
                be = self.engines[r.back.name]
                self.at(t + self.net.msg(e.node0, be.node0, H, kind=self.kind), self._arrive, be, rid, ("dec", item[1]))
            return
        # 后段 / 整条流水线：出一个 token
        r.n_out += 1
        t_cli = t + self.net.msg(e.node0, None, 8, kind=self.kind)
        r.toks.append(t_cli)
        if r.n_out == 1:
            r.t_back_out0 = t
            r.t_first = t_cli
            r.t_first_gw = t_cli          # 模拟里网关 = 客户端：网关外这一段只有这条消息
        if r.n_out >= r.n_gen:
            self.at(t_cli, lambda tt, rr=r: self._finish(tt, rr))
            return
        if self.full:                     # decode 循环在引擎里：下一步就能算
            e.put(rid, ("dec", r.n_prompt + r.n_out - 1), t)
            return
        fe = self.engines[r.front.name]
        ctx = r.n_prompt + r.n_out - 1
        self.at(t + self.net.msg(e.node0, fe.node0, 8, kind=self.kind), self._arrive, fe, rid, ("dec", ctx))


def resident_experts(ch: Mapping) -> int | None:
    """后段命令里（或 A 通道的 overrides 文件里）的 p2p_layer_experts → 每层驻留专家数的均值。"""
    if ch.get("overrides") and Path(ch["overrides"]).is_file():
        try:
            le = json.loads(Path(ch["overrides"]).read_text(encoding="utf-8")).get("p2p_layer_experts") or {}
            sizes = [len(v) for v in le.values() if v]
            if sizes:
                return round(sum(sizes) / len(sizes))
        except ValueError:
            pass
    for c in ch.get("commands") or []:
        if " --hf-overrides " not in c["cmd"]:
            continue
        try:
            ov = json.loads(shlex.split(c["cmd"].split(" --hf-overrides ", 1)[1])[0])
        except (ValueError, IndexError):
            continue
        le = ov.get("p2p_layer_experts") or {}
        sizes = [len(v) for v in le.values() if v]
        if sizes:
            return round(sum(sizes) / len(sizes))
    return None


# --------------------------------------------------------------------------- #
# 回放、客户端、结果
# --------------------------------------------------------------------------- #
def replay_requests(real: Mapping, stages: Mapping, *, n: int | None = None) -> list[SimReq]:
    """真机结果 → 同样顺序的请求（prompt 长度、生成长度、绑到的类都照真机）。"""
    chans = {c["name"]: c for c in stages["channels"]}
    rows = sorted(real.get("requests") or [], key=lambda r: _req_index(r.get("req")))
    if not rows:
        raise SystemExit("回放文件里没有 requests")
    out = []
    for i in range(n or len(rows)):
        r = rows[i % len(rows)]
        ch = chans.get(r.get("channel") or "")
        # 绑定用的类：回放文件的通道在这份部署里 → 它的 task；否则（比如拿 petals 的结果回放到分段部署）
        # 用那次判出的类，再没有就用标注的类
        task = (ch or {}).get("task") or r.get("task") or r.get("true_task")
        out.append(SimReq(rid=f"req{i}", case_id=r.get("case_id"), true_task=r.get("true_task"),
                          task=task, n_prompt=int(r.get("n_prompt") or 1),
                          n_gen=max(1, int(r.get("n_generated") or 1)),
                          stop_reason=r.get("stop_reason") or "length"))
    return out


def _req_index(rid: Any) -> int:
    m = re.search(r"(\d+)$", str(rid or ""))
    return int(m.group(1)) if m else 0


def run_sim(stages: Mapping, reqs: Sequence[SimReq], *, perf, net: Net, node_gpu, params: Params,
            per_front: int = 0, inflight: int = 0, front_slots: int = 0, hold: bool = True,
            front_pick: str = "least", steps=None, marks: Sequence[int] = DEFAULT_MARKS,
            real: Mapping | None = None, label: str = "", a_mode: str = "any") -> dict:
    full = stages.get("kind") != "tiered"
    if per_front:
        if full:
            raise SystemExit("整条流水线（A）没有前段：用 --inflight（比如 = 通道数，每条流水线一条）")
        inflight, front_slots = per_front * len(stages["fronts"]), per_front
    inflight = inflight or len(stages["channels"] if full else stages["backs"])
    sim = Sim(stages, perf=perf, net=net, node_gpu=node_gpu, params=params,
              front_slots=front_slots or None, hold=hold, front_pick=front_pick, steps=steps, a_mode=a_mode)
    queue = list(reqs)
    nxt = iter(range(len(queue)))

    def release(t: float) -> None:
        i = next(nxt, None)
        if i is not None:
            sim.submit(queue[i], t)

    sim.on_finish = lambda r, t: sim.at(t + params.client_ms, lambda tt: release(tt))
    for _ in range(min(inflight, len(queue))):
        release(0.0)
    sim.run()
    done = [r for r in queue if r.t_done is not None]
    if len(done) < len(queue):
        raise RuntimeError(f"模拟没跑完：{len(done)}/{len(queue)}（有请求卡住 —— 槽位 / 前段设置不一致？）")
    return build_output(sim, queue, stages, inflight=inflight, per_front=per_front, marks=marks,
                        real=real, label=label, front_slots=front_slots, hold=hold)


def build_rows(sim: Sim, reqs: Sequence[SimReq], marks: Sequence[int]) -> list[dict]:
    rows = []
    for r in reqs:
        fe, be = (sim.engines[r.front.name] if r.front is not None else None), sim.engines[r.back.name]
        total = r.t_done - r.t_submit
        ttft = r.t_first - r.t_submit
        gaps = [b - a for a, b in zip(r.toks, r.toks[1:])]
        mk = {}
        for k in marks:
            if len(r.toks) >= k:
                t = r.toks[k - 1]
                e, sf = t - r.t_submit, t - r.t_first
                mk[str(k)] = {"n": k, "t_ms": round(e, 2), "since_first_ms": round(sf, 2),
                              "tpot_ms": round(sf / (k - 1), 3) if k > 1 and sf else None,
                              "tok_per_s": round(k / (e / 1e3), 3) if e > 0 else None,
                              "decode_tok_per_s": round((k - 1) / (sf / 1e3), 3) if k > 1 and sf else None}
        wait_back = (r.t_bound - r.t_wait) if r.t_wait is not None and r.t_bound is not None else 0.0
        rows.append({
            "req": r.rid, "prompt": None, "text": "", "stop_reason": r.stop_reason,
            "true_task": r.true_task, "task": r.task,
            "correct": (None if r.true_task is None else bool(r.task == r.true_task)),
            "confidence": 0.0, "zone": "", "rebinds": 0, "alarms": 0,
            "front": "+".join(fe.nodes) if fe else None, "back": "+".join(be.nodes),
            "front_engine": fe.name if fe else None,
            **({"tiered": {"front_prefill_ms": round((r.t_front_seen or r.t_front_done or 0) - r.t_submit, 2),
                           "bind_ms": (round(r.t_bound - r.t_front_seen, 2)
                                       if r.t_bound is not None and r.t_front_seen is not None else None),
                           "wait_back_ms": round(wait_back, 2)}} if fe else {}),
            "wait_front_ms": round(wait_back, 2),
            "channel": be.name, "n_prompt": r.n_prompt, "n_generated": r.n_out,
            "total_ms": round(total, 2), "queue_ms": round(r.t_gw_wait, 2),
            "queue_back_ms": round(r.queue_engine_ms.get("back", 0.0), 2),
            "prefill_ms": round(ttft, 2), "decode_ms": round(total - ttft, 2),
            "per_token_p50_ms": round(stt.median(gaps), 2) if gaps else 0.0,
            "compute_ms": 0.0, "network_and_overhead_ms": 0.0, "traces_complete": False,
            "utilisation": 0.0, "token_ms": [round(x, 2) for x in gaps], "marks": mk,
            "t_submit_s": round(r.t_submit / 1e3, 3), "t_done_s": round(r.t_done / 1e3, 3),
            **({"case_id": r.case_id} if r.case_id else {}),
            "engine": ({"ttft_ms": round(r.t_first_gw - r.t_submit, 2),
                        "front_prefill_engine_ms": (round(r.t_front_done - r.t_in["front"], 2)
                                                    if r.t_front_done is not None and "front" in r.t_in else None),
                        "back_prefill_engine_ms": (round(r.t_back_out0 - r.t_in["back"], 2)
                                                   if r.t_back_out0 is not None and "back" in r.t_in else None)}
                       if fe and r.t_first_gw is not None else None),
            "identify": None, "nodes": [], "missing_traces": [],
        })
    return rows


def _stats_mod():
    try:
        from vllm.p2p import tiered_stats as ts
    except ImportError:                                     # pragma: no cover
        import tiered_stats as ts
    return ts


def run_metrics(rows: Sequence[Mapping], wall_s: float, inflight: int) -> dict:
    """与 p2p-framework continuous.run_metrics 同口径（有就直接用它）。"""
    try:
        try:
            from p2pmoe.deploy.continuous import run_metrics as rm
        except ImportError:
            from vllm.p2p.identifier import use_p2p_framework
            use_p2p_framework(None)
            from p2pmoe.deploy.continuous import run_metrics as rm
        return rm(rows, wall_s, inflight=inflight)
    except (ImportError, SystemExit):
        pass
    ts = _stats_mod()
    out_tok = sum(int(r["n_generated"]) for r in rows)
    in_tok = sum(int(r["n_prompt"]) for r in rows)
    by = defaultdict(list)
    for r in rows:
        by[r.get("true_task") or "?"].append(r)
    return {"n_requests": len(rows), "n_timeout": 0, "inflight": inflight, "wall_s": round(wall_s, 2),
            "output_tokens": out_tok, "prompt_tokens": in_tok,
            "throughput": {"output_tok_per_s": round(out_tok / wall_s, 3) if wall_s else None,
                           "total_tok_per_s": round((out_tok + in_tok) / wall_s, 3) if wall_s else None,
                           "req_per_min": round(len(rows) / wall_s * 60, 3) if wall_s else None},
            "by_task": {u: {"n": len(rs)} for u, rs in by.items()},
            "stop_reasons": {}, "accuracy": None,
            "per_request_fallback": ts.request_stats(rows)}


def build_output(sim: Sim, reqs, stages, *, inflight, per_front, marks, real, label,
                 front_slots, hold) -> dict:
    ts = _stats_mod()
    rows = build_rows(sim, reqs, marks)
    t0 = min(r.t_submit for r in reqs)
    t1 = max(r.t_done for r in reqs)
    wall = (t1 - t0) / 1e3
    out = {
        "model": f"sim · {Path(str(stages.get('plan') or '')).name}",
        "backend": "sim", "mode": "static" if sim.full else "dynamic",
        "gateway_mode": sim.a_mode if sim.full else ("lr" if real else "oracle"),
        "label": label or "sim", "tokens": max(r.n_gen for r in reqs), "concurrency": inflight,
        "channels": {n: {"nodes": e.nodes, "partition": e.layers} for n, e in sim.engines.items()
                     if e.role != "front"},
        "requests": rows, "failed": [],
    }
    run = run_metrics(rows, wall, inflight)
    run.update(order="replay", max_tokens=out["tokens"], marks=list(marks), offline=False,
               per_front=per_front or None)
    if sim.co is not None:
        run["front_peaks"] = {f.name: {"peak": f.peak, "served": f.n_served} for f in sim.co.fronts}
        run["gateway_waits"] = {"front_slot": sim.co.n_front_wait, "back_hold": sim.co.n_held}
    run["per_request"] = ts.request_stats(rows)
    run["marks_stats"] = ts.mark_stats(rows)
    run["segment_stats"] = ts.segment_stats(rows)
    run["cumulative_stats"] = ts.cumulative_stats(rows)
    out["run"] = run
    out["sim"] = {
        "params": sim.params.as_dict(),
        "net": {"hop_ms": sim.net.hop_ms, "agent_hop_ms": sim.net.agent_hop_ms, "pp_hop_ms": sim.net.pp_hop_ms,
                "back_pp_hop_ms": sim.net.back_pp_hop_ms, "lat_same": sim.net.lat_same,
                "lat_cross": sim.net.lat_cross, "lat_gw": sim.net.lat_gw, "gateway": sim.net.gateway},
        "front_slots": front_slots or None, "hold": hold,
        "engines": {n: {"role": e.role, "task": e.task, "nodes": e.nodes, "gpus": [g.gpu for g in e.gpus],
                        "experts": e.experts, "steps": e.n_steps,
                        "fixed_ms_b1": round(e.fixed_ms(512.0, 1), 4),
                        "busy_frac": round(e.busy_ms / (t1 - t0), 4) if t1 > t0 else None,
                        "batch_mean": round(e.batch_sum / e.n_steps, 3) if e.n_steps else None,
                        "batch_waits": getattr(e, "n_waits", 0),
                        "measured_steps": {k: bool(v) for k, v in e.steps.items()}}
                    for n, e in sim.engines.items()},
    }
    return out


# --------------------------------------------------------------------------- #
# 比对与标定
# --------------------------------------------------------------------------- #
def _tpot(r: Mapping) -> float | None:
    n = int(r.get("n_generated") or 0)
    if n <= 1:
        return None
    return (float(r["total_ms"]) - float(r["prefill_ms"])) / (n - 1)


def key_metrics(doc: Mapping) -> dict:
    rows = doc.get("requests") or []
    tp = [x for x in (_tpot(r) for r in rows) if x is not None]
    tt = [float(r["prefill_ms"]) for r in rows]
    out = {"TTFT 均值": stt.mean(tt) if tt else None, "TTFT p50": stt.median(tt) if tt else None,
           "TPOT 均值": stt.mean(tp) if tp else None, "TPOT p50": stt.median(tp) if tp else None,
           "TPOT p90": (sorted(tp)[int(0.9 * (len(tp) - 1))] if tp else None),
           "输出 tok/s": ((doc.get("run") or {}).get("throughput") or {}).get("output_tok_per_s"),
           "墙钟 s": (doc.get("run") or {}).get("wall_s")}
    by = defaultdict(list)
    for r in rows:
        v = _tpot(r)
        if v is not None:
            by[r.get("channel") or "?"].append(v)
    for ch, vs in sorted(by.items()):
        out[f"TPOT 均值 {ch}"] = stt.mean(vs)
    tb = _stats_mod().ttft_breakdown(rows)            # 分段网关：TTFT 四段的 p50（只显示，不进误差分）
    if tb:
        for k, v in tb["all"].items():
            out[f"TTFT·{k} p50"] = v["p50"]
    return out


def compare(sim: Mapping, real: Mapping, score: str = "all") -> tuple[list[str], float]:
    """关键指标逐项比；返回 (表格行, 误差分 = 各项相对误差平方的均值，后段分项按条数加权)。
    score：all = TTFT p50 + TPOT + 吞吐；tpot = 只看 TPOT（含各后段）；ttft = 只看 TTFT p50。"""
    ks, kr = key_metrics(sim), key_metrics(real)
    lines = [f"{'指标':<28}{'真机':>12}{'模拟':>12}{'相对误差':>10}"]
    errs, ws = [], []
    n_by = defaultdict(int)
    for r in real.get("requests") or []:
        n_by[r.get("channel") or "?"] += 1
    for k in kr:
        a, b = kr[k], ks.get(k)
        if a is None or b is None:
            continue
        e = (b - a) / a if a else 0.0
        lines.append(f"{k:<28}{a:>12.2f}{b:>12.2f}{100 * e:>9.1f}%")
        # TTFT 用中位数打分：均值会被引擎刚起来时那几条（几秒）拉偏
        use = {"all": ("TTFT p50", "TPOT 均值", "TPOT p90", "输出 tok/s"), "tpot": ("TPOT 均值", "TPOT p90"),
               "ttft": ("TTFT p50",)}[score]
        if k in use:
            errs.append(e * e)
            ws.append(2.0)
        elif k.startswith("TPOT 均值 ") and score != "ttft":
            errs.append(e * e)
            ws.append(n_by.get(k[len("TPOT 均值 "):], 1) / max(len(real.get("requests") or [1]), 1) * 4)
    score = sum(e * w for e, w in zip(errs, ws)) / sum(ws) if ws else math.inf
    return lines, score


TABLE_KEYS = ("输出 tok/s", "TPOT 均值", "TPOT p90", "TTFT p50", "TTFT 均值")


def summary_table(runs: Sequence[tuple[str, Mapping, Mapping]]) -> tuple[list[str], list[dict]]:
    """若干组（标签, 真机结果, 模拟结果）→ 汇总表（markdown 行）+ 每组的数（JSON 用）。
    每组：条数、墙钟、平均生成长度，关键指标与各后段 TPOT、TTFT 各段 p50 的 真机 / 模拟 / 误差。"""
    recs = []
    for label, real, sim in runs:
        kr, ks = key_metrics(real), key_metrics(sim)
        rows = real.get("requests") or []
        keys = [k for k in kr if k in TABLE_KEYS or k.startswith("TPOT 均值 ") or k.startswith("TTFT·")]
        keys = [k for k in TABLE_KEYS if k in kr] + [k for k in keys if k not in TABLE_KEYS]
        m = {}
        for k in keys:
            a, b = kr.get(k), ks.get(k)
            if a is None or b is None:
                continue
            m[k] = {"real": round(a, 3), "sim": round(b, 3), "err": round((b - a) / a, 4) if a else None}
        recs.append({"label": label, "n": len(rows), "wall_s": (real.get("run") or {}).get("wall_s"),
                     "gen_mean": (round(stt.mean(int(r.get("n_generated") or 0) for r in rows), 1) if rows else None),
                     "metrics": m})
    keys = []
    for r in recs:
        keys += [k for k in r["metrics"] if k not in keys]
    out = ["| 指标 | " + " | ".join(r["label"] for r in recs) + " |",
           "|---|" + "---|" * len(recs),
           "| 条数 / 墙钟 s / 平均生成长度 | " + " | ".join(
               f"{r['n']} / {r['wall_s'] or 0:.0f} / {r['gen_mean']}" for r in recs) + " |"]
    for k in keys:
        cells = []
        for r in recs:
            v = r["metrics"].get(k)
            if v is None:
                cells.append("—")
                continue
            e = "—" if v["err"] is None else f"{100 * v['err']:+.1f}%"
            cells.append(f"{v['real']:.1f} / {v['sim']:.1f}（{e}）")
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("每格：真机 / 模拟（模拟相对真机的误差）。TTFT·xx = TTFT 拆分各段的 p50（ms）。")
    return out, recs


def fit(make: Callable[..., dict], real: Mapping, params: Params, hop_ms: float, *,
        rounds: int = 3, fixed: Sequence[str] = (), log: Callable[[str], None] = print,
        extra: Mapping[str, float] | None = None, score: str = "all") -> tuple[Params, float, float, dict]:
    """坐标下降：依次调 front_fixed_ms / back_fixed_ms / hop_ms（以及 `extra` 里的，比如 pp_hop_ms ——
    以关键字传给 make），每次在当前值的几档倍数里挑误差分最小的。各参数最后的取值也写进
    返回结果的 `doc["sim"]["fit"]`。"""
    extra = dict(extra or {})
    cur = {"front_fixed_ms": params.front_fixed_ms, "back_fixed_ms": params.back_fixed_ms, "hop_ms": hop_ms,
           **extra}
    if cur["front_fixed_ms"] is None or cur["back_fixed_ms"] is None:
        # None = 用 layer_bench 截距（各卡型不同）—— 标定从它在模拟里的平均取值出发
        eng = make(params, hop_ms)["sim"]["engines"]
        for k, role in (("front_fixed_ms", "front"), ("back_fixed_ms", "back")):
            if cur[k] is None:
                vs = [v["fixed_ms_b1"] for v in eng.values()
                      if v["role"] == role or (role == "back" and v["role"] == "full")]
                cur[k] = round(sum(vs) / len(vs), 4) if vs else 1.0

    def score_of(c) -> tuple[float, dict]:
        p = Params(**{**params.as_dict(), "front_fixed_ms": c["front_fixed_ms"],
                      "back_fixed_ms": c["back_fixed_ms"]})
        doc = make(p, c["hop_ms"], **{k: c[k] for k in extra})
        return compare(doc, real, score)[1], doc

    best, best_doc = score_of(cur)
    log(f"  起点 {cur} → 误差分 {best:.4f}")
    for rd in range(rounds):
        for k in ("front_fixed_ms", "back_fixed_ms", "hop_ms", *extra):
            if k in fixed:
                continue
            base = cur[k] if cur[k] > 0 else 0.5
            for m in (0.0, 0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0):
                c = dict(cur, **{k: round(base * m, 4)})
                s, doc = score_of(c)
                if s < best - 1e-9:
                    best, best_doc, cur = s, doc, c
            log(f"  第 {rd + 1} 轮 {k:<15} → {cur[k]:.3f}   误差分 {best:.4f}")
    p = Params(**{**params.as_dict(), "front_fixed_ms": cur["front_fixed_ms"],
                  "back_fixed_ms": cur["back_fixed_ms"]})
    best_doc.setdefault("sim", {})["fit"] = {"params": cur, "score": best}
    return p, cur["hop_ms"], best, best_doc


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def _setup(a) -> tuple:
    stages = json.loads(Path(a.stages).read_text(encoding="utf-8"))
    if stages.get("kind") != "tiered" and not all("partition" in c for c in stages.get("channels") or [{}]):
        raise SystemExit(f"{a.stages} 既不是分段部署的 stages.json，也不是 layout.py 的 channels.json")
    ip, gpu_h = load_hosts(a.hosts)
    for c in stages["channels"]:
        for x in c.get("commands") or []:
            ip.setdefault(x["node"], x["host"])
    lat, bw, gpu_t = load_topo(a.topo)
    node_gpu = {**gpu_t, **gpu_h}
    for kv in a.gpu or []:
        k, v = kv.split("=", 1)
        node_gpu[k] = gpu_key(v)
    perf = load_perf(a.layer_bench)
    net = Net(ip=ip, lat=lat, bw=bw, lat_same=a.lat_same, lat_cross=a.lat_cross, lat_gw=a.lat_gw,
              gateway=a.gateway_node, hop_ms=a.hop_ms, agent_hop_ms=a.agent_hop_ms, pp_hop_ms=a.pp_hop_ms,
              back_pp_hop_ms=a.back_pp_hop_ms, bw_default=a.bw_mbps)
    hidden = a.hidden
    if a.model_dir and (Path(a.model_dir) / "config.json").exists():
        hidden = int(json.loads((Path(a.model_dir) / "config.json").read_text())["hidden_size"])
    params = Params(front_fixed_ms=a.front_fixed_ms, back_fixed_ms=a.back_fixed_ms, lr_ms=a.lr_ms,
                    client_ms=a.client_ms, max_batched_tokens=a.max_batched_tokens, hidden=hidden,
                    pp_simple=a.pp_simple, pf_fixed_ms=a.pf_fixed_ms, pf_fixed_per=a.pf_fixed_per,
                    batch_wait_ms=a.batch_wait_ms, batch_wait_role=a.batch_wait_role)
    steps = None
    if a.steps:
        from vllm.p2p import steptrace
        steps = {}
        for name, rows in steptrace.load(a.steps).items():
            s = steptrace.summarise(rows)
            steps[name] = {"dec": {int(b): v["p50"] for b, v in (s.get("decode_ms_by_batch") or {}).items()},
                           "pf": s.get("prefill_fit")}
    real = json.loads(Path(a.replay).read_text(encoding="utf-8"))
    reqs_fn = lambda: replay_requests(real, stages, n=a.requests or None)   # noqa: E731
    return stages, node_gpu, perf, net, params, steps, real, reqs_fn


def _common(ap) -> None:
    ap.add_argument("--stages", required=True,
                    help="分段部署的 stages.json；或整条流水线（A）的 channels.json（layout.py 生成）")
    ap.add_argument("--a-mode", choices=("any", "oracle"), default="any",
                    help="整条流水线（A）时网关怎么挑通道（同 gateway.py --mode）")
    ap.add_argument("--replay", required=True,
                    help="一次真机结果（bench 的 --out）：请求顺序、prompt / 生成长度、绑到哪类都照它")
    ap.add_argument("--layer-bench", nargs="+", required=True,
                    help="layer_bench 的 summary_by_gpu.json，或各台的单机结果（同卡型取均值）")
    ap.add_argument("--hosts", default=None, help="hosts.txt（节点 IP；注释的第一个词当卡型）")
    ap.add_argument("--topo", default=None, help="拓扑 JSON（edges 的单向时延 / 带宽、nodes 的 role 当卡型）")
    ap.add_argument("--gpu", action="append", help="手动指定卡型：N13=RTX6000（可多次）")
    ap.add_argument("--model-dir", default=None, help="读 config.json 的 hidden_size（默认 --hidden）")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--per-front", type=int, default=0, help="每个前段始终 N 条（同 bench --per-front）")
    ap.add_argument("--inflight", type=int, default=0, help="闭环总在途（不给 --per-front 时）")
    ap.add_argument("--front-slots", type=int, default=0, help="同网关 --front-slots（--per-front 时自动）")
    ap.add_argument("--no-hold", action="store_true", help="同网关 --no-hold")
    ap.add_argument("--front-pick", choices=("least", "speed"), default="least")
    ap.add_argument("--requests", type=int, default=0, help="请求数（默认 = 回放文件的条数，多了循环用）")
    ap.add_argument("--front-fixed-ms", type=float, default=None,
                    help="前段每步固定开销（默认 layer_bench 截距 × 0.5）")
    ap.add_argument("--back-fixed-ms", type=float, default=None, help="后段每步固定开销（默认 layer_bench 截距）")
    ap.add_argument("--hop-ms", type=float, default=2.0,
                    help="整条流水线（A）：网关 HTTP 那条路（gateway.py → vLLM API 服务器）每跳的软件开销")
    ap.add_argument("--agent-hop-ms", type=float, default=2.0,
                    help="分段部署：协调器 ↔ 代理、前段 ↔ 后段直连每跳的软件开销（O1 实测每 token 来回约 4 ms）")
    ap.add_argument("--pp-hop-ms", type=float, default=0.3,
                    help="多机 PP 每次传输（段间 hidden、调度下发、结果回传）的固定开销")
    ap.add_argument("--back-pp-hop-ms", type=float, default=None,
                    help="分段部署后段（stage_agent 驱动）的 PP 每次传输开销；默认同 --pp-hop-ms")
    ap.add_argument("--batch-wait-ms", type=float, default=0.0,
                    help="假设实验：引擎空闲时最多再等这么久，让同一引擎上其余请求的输入到齐后一起算一步"
                         "（O3 的凑批窗口；真 vLLM 还没有，默认 0 = 不等）")
    ap.add_argument("--batch-wait-role", choices=("back", "front", "all"), default="back")
    ap.add_argument("--pf-fixed-ms", type=float, default=0.0,
                    help="含 prefill 的步额外的固定开销（layer_bench 的逐层斜率量不到的那部分）")
    ap.add_argument("--pf-fixed-per", choices=("step", "stage", "layer"), default="stage",
                    help="--pf-fixed-ms 按什么计：每步一次 / 每个 PP 段一次 / 每层一次")
    ap.add_argument("--pp-simple", action="store_true",
                    help="多机 PP 用 v1 的算法：只算往下传的几跳，不算调度下发与结果回传")
    ap.add_argument("--lat-same", type=float, default=0.1, help="同网段单向时延 ms（没 --topo 时）")
    ap.add_argument("--lat-cross", type=float, default=3.15, help="跨网段单向时延 ms（没 --topo 时）")
    ap.add_argument("--lat-gw", type=float, default=0.1, help="网关 ↔ 节点单向时延（没 --gateway-node 时）")
    ap.add_argument("--bw-mbps", type=float, default=10000.0,
                    help="节点间 / 网关 ↔ 节点的有效带宽（Mbps；--topo 里没写的边都用它）。只影响大消息："
                         "prompt 的 hidden（每 token 4 KB）、PP 的 prefill hidden")
    ap.add_argument("--gateway-node", default=None, help="网关在哪个节点（如 N09），网关消息按它算时延")
    ap.add_argument("--lr-ms", type=float, default=2.0)
    ap.add_argument("--client-ms", type=float, default=1.0)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--steps", nargs="*", default=None,
                    help="steptrace 收回来的步级记录（<引擎名>.jsonl）：纯 decode 步用实测每步 p50")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _common(sub.add_parser("run", help="跑一次模拟，写 bench 格式的结果"))
    f = sub.add_parser("fit", help="拿 --replay 的真机结果标定 前段 / 后段每步固定开销、每跳开销")
    _common(f)
    f.add_argument("--rounds", type=int, default=3)
    f.add_argument("--fit-back-pp", action="store_true",
                   help="也标定 --back-pp-hop-ms（分段部署后段的 PP 每次传输开销）")
    f.add_argument("--score", choices=("all", "tpot", "ttft"), default="all",
                   help="标定看哪些指标：all = TTFT 中位数 + TPOT + 吞吐；tpot / ttft = 只看那一项"
                        "（只调影响 TPOT 的参数时用 tpot，免得被 TTFT 里别的误差带偏）")
    f.add_argument("--fit-agent", action="store_true",
                   help="也标定 --agent-hop-ms（分段部署代理那条路每跳的开销）")
    f.add_argument("--fit-pf", action="store_true",
                   help="也标定 --pf-fixed-ms（含 prefill 的步的固定开销；按 --pf-fixed-per 计）。"
                        "不要和 --steps 一起用（那时 prefill 步用的是实测值）")
    f.add_argument("--fit-pp", action="store_true",
                   help="也标定 --pp-hop-ms（多机 PP 每次传输的固定开销）。同一深度的流水线下它和每步固定开销"
                        "分不开 —— 有步级记录（--steps）或不同深度的部署一起看时再开")
    f.add_argument("--fix", default="", help="不调的参数（逗号分隔：front_fixed_ms,back_fixed_ms,hop_ms）")
    c = sub.add_parser("compare", help="比两份结果的关键指标")
    c.add_argument("--sim", required=True)
    c.add_argument("--real", required=True)
    t = sub.add_parser("table", help="几组「真机 + 模拟」并排出一张汇总表（markdown，可另存 JSON）")
    t.add_argument("runs", nargs="+", metavar="标签=真机.json,模拟.json")
    t.add_argument("--md", default=None, help="表另存到这个 .md")
    t.add_argument("--json", default=None, help="数另存到这个 .json")
    a = ap.parse_args(argv)

    if a.cmd == "table":
        runs = []
        for spec in a.runs:
            label, _, files = spec.partition("=")
            rp, _, sp = files.partition(",")
            if not (label and rp and sp):
                raise SystemExit(f"写成 标签=真机.json,模拟.json，给的是 {spec!r}")
            runs.append((label, json.loads(Path(rp).read_text(encoding="utf-8")),
                         json.loads(Path(sp).read_text(encoding="utf-8"))))
        lines, recs = summary_table(runs)
        print("\n".join(lines))
        if a.md:
            Path(a.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        if a.json:
            Path(a.json).write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
        return 0

    if a.cmd == "compare":
        lines, score = compare(json.loads(Path(a.sim).read_text(encoding="utf-8")),
                               json.loads(Path(a.real).read_text(encoding="utf-8")))
        print("\n".join(lines))
        print(f"误差分（加权相对误差平方均值）：{score:.4f}")
        return 0

    stages, node_gpu, perf, net, params, steps, real, reqs_fn = _setup(a)

    def make(p: Params, hop: float, pp_hop_ms: float | None = None,
             pf_fixed_ms: float | None = None, agent_hop_ms: float | None = None,
             back_pp_hop_ms: float | None = None) -> dict:
        net.hop_ms = hop
        if back_pp_hop_ms is not None:
            net.back_pp_hop_ms = back_pp_hop_ms
        if agent_hop_ms is not None:
            net.agent_hop_ms = agent_hop_ms
        if pp_hop_ms is not None:
            net.pp_hop_ms = pp_hop_ms
        if pf_fixed_ms is not None:
            p = Params(**{**p.as_dict(), "pf_fixed_ms": pf_fixed_ms})
        return run_sim(stages, reqs_fn(), perf=perf, net=net, node_gpu=node_gpu, params=p,
                       per_front=a.per_front, inflight=a.inflight, front_slots=a.front_slots,
                       hold=not a.no_hold, front_pick=a.front_pick, steps=steps, real=real,
                       label=a.label or f"sim {a.cmd}", a_mode=a.a_mode)

    if a.cmd == "fit":
        print("标定（坐标下降）：")
        fixed = [x.strip() for x in a.fix.split(",") if x.strip()]
        if stages.get("kind") != "tiered":
            fixed.append("front_fixed_ms")            # 整条流水线没有前段
        extra = {}
        if a.fit_pp:
            extra["pp_hop_ms"] = a.pp_hop_ms
        if a.fit_pf:
            extra["pf_fixed_ms"] = a.pf_fixed_ms or 10.0
        if a.fit_agent:
            extra["agent_hop_ms"] = a.agent_hop_ms
        if a.fit_back_pp:
            extra["back_pp_hop_ms"] = a.back_pp_hop_ms if a.back_pp_hop_ms is not None else a.pp_hop_ms
        p, hop, score, doc = fit(make, real, params, a.hop_ms, rounds=a.rounds, fixed=fixed,
                                 extra=extra or None, score=a.score)
        fp = doc["sim"]["fit"]["params"]
        pp = "".join(f" --{k.replace('_', '-')} {fp[k]:g}" for k in extra)
        if a.fit_pf:
            pp += f" --pf-fixed-per {a.pf_fixed_per}"
        print(f"结果：--front-fixed-ms {p.front_fixed_ms:g} --back-fixed-ms {p.back_fixed_ms:g} "
              f"--hop-ms {hop:g}{pp}（误差分 {score:.4f}）")
    else:
        doc = make(params, a.hop_ms)
    lines, score = compare(doc, real)
    print("\n".join(lines))
    print(f"误差分：{score:.4f}")
    eng = doc["sim"]["engines"]
    print("引擎（模拟）：" + "；".join(f"{n} 忙 {100 * (v['busy_frac'] or 0):.0f}% 批 {v['batch_mean']}"
                                      for n, v in eng.items()))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已写 {a.out}（python3 vllm/p2p/tiered_stats.py --b {a.out} --a {a.replay}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
