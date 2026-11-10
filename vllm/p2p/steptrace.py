# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""分段引擎的步级记录：每一步（一次前向）什么时候开始、算了多久、批里几条请求、几个 token。

给模拟器（`vllm/p2p/sim.py`）标定「一步多久」用的真值，也能直接看引擎的忙闲与批大小。

开关：引擎进程的环境变量 `VLLM_P2P_STEP_TRACE=<文件路径>`（或一个已存在的目录 → 目录下
`<pid>.jsonl`）。`stages.py --step-trace <目录>` 会给每个分段引擎的启动命令加上
`VLLM_P2P_STEP_TRACE=<目录>/<引擎名>.jsonl`。不设 = 什么都不做（`StepTrace.from_env()` 返回 None）。

每行一步::

    {"t": 开始时刻（time.time()，秒）, "ms": 用时, "n": 请求数, "tok": token 数,
     "pf": prefill token 数（这一步里 >1 个 token 的请求的 token 和）, "npf": 这种请求几条,
     "ctx": 批里请求已算 token 数的均值, "q": 这一步开始时调度器里等着的请求数}

「用时」= 调度 + 前向 + 采样 + 更新输出（EngineCore.step 一次）。PP 多机时（batch queue）是
这一批从调度到取回结果的时间，几批可能重叠。

收集与汇总（控制节点上，只用标准库）::

    python3 vllm/p2p/steptrace.py collect --stages p2p_channels/stages.json --dir /tmp/p2p_steps \\
        --out results/steps_<TAG>
    python3 vllm/p2p/steptrace.py summary results/steps_<TAG>/*.jsonl
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ENV = "VLLM_P2P_STEP_TRACE"


class StepTrace:
    """引擎进程里用：`begin(key, sched_out, scheduler)` 在调度出一批时调，`end(key)` 在这批结果
    处理完时调。写文件是攒一批再写，不在每步里做 I/O。"""

    def __init__(self, path: str | Path, flush_every: int = 256, flush_s: float = 2.0):
        p = Path(path)
        if p.is_dir():
            p = p / f"{os.getpid()}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.path = p
        self._buf: list[str] = []
        self._open: dict[Any, dict] = {}
        self._lock = threading.Lock()
        self._flush_every, self._flush_s = int(flush_every), float(flush_s)
        self._last_flush = time.monotonic()
        atexit.register(self.flush)

    @classmethod
    def from_env(cls) -> "StepTrace | None":
        v = os.environ.get(ENV)
        if not v:
            return None
        try:
            return cls(v)
        except OSError as e:                     # 记录失败不能拖垮引擎
            print(f"[p2p steptrace] 打不开 {v}：{e}，不记录", file=sys.stderr)
            return None

    @staticmethod
    def describe(sched_out, scheduler=None) -> dict:
        """从 SchedulerOutput 取批的形状（只读字段，不改任何东西）。"""
        nst: Mapping[str, int] = getattr(sched_out, "num_scheduled_tokens", None) or {}
        pf = [int(v) for v in nst.values() if int(v) > 1]
        rec = {"n": len(nst), "tok": int(getattr(sched_out, "total_num_scheduled_tokens", 0) or 0),
               "pf": sum(pf), "npf": len(pf)}
        reqs = getattr(scheduler, "requests", None)
        if reqs:
            ctx = [int(getattr(reqs[r], "num_computed_tokens", 0)) for r in nst if r in reqs]
            rec["ctx"] = round(sum(ctx) / len(ctx), 1) if ctx else 0.0
            waiting = getattr(scheduler, "waiting", None)
            try:
                rec["q"] = len(waiting) if waiting is not None else None
            except TypeError:
                rec["q"] = None
        return rec

    @staticmethod
    def now() -> tuple[float, float]:
        return time.time(), time.perf_counter()

    def begin(self, key, sched_out, scheduler=None, start: tuple[float, float] | None = None) -> None:
        """`start` = 调度之前 `now()` 的返回值（这一步的用时含调度）；不给就从现在算。"""
        try:
            if int(getattr(sched_out, "total_num_scheduled_tokens", 0) or 0) <= 0:
                return
            rec = self.describe(sched_out, scheduler)
            rec["t"], rec["_p"] = start if start is not None else self.now()
            self._open[key] = rec
        except Exception:                                    # noqa: BLE001
            pass

    def end(self, key) -> None:
        rec = self._open.pop(key, None)
        if rec is None:
            return
        rec["ms"] = round((time.perf_counter() - rec.pop("_p")) * 1e3, 3)
        rec["t"] = round(rec["t"], 6)
        with self._lock:
            self._buf.append(json.dumps(rec, separators=(",", ":")))
            if (len(self._buf) >= self._flush_every
                    or time.monotonic() - self._last_flush >= self._flush_s):
                self._flush_locked()

    def _flush_locked(self) -> None:
        if self._buf:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write("\n".join(self._buf) + "\n")
            except OSError:
                pass
            self._buf.clear()
        self._last_flush = time.monotonic()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()


# --------------------------------------------------------------------------- #
# 控制节点：收集、汇总（只用标准库）
# --------------------------------------------------------------------------- #
def trace_files(stages: Mapping) -> dict[str, tuple[str, str]]:
    """stages.json → {引擎名: (rank 0 所在的 host, 记录文件路径)}；没开记录的引擎不在里面。"""
    out = {}
    for c in stages.get("channels") or []:
        for cmd in c.get("commands") or []:
            toks = shlex.split(cmd["cmd"].split(" --hf-overrides ", 1)[0])
            path = next((t.split("=", 1)[1] for t in toks if t.startswith(ENV + "=")), None)
            rank0 = "--node-rank" not in toks or toks[toks.index("--node-rank") + 1] == "0"
            if path and rank0:
                out[c["name"]] = (cmd["host"], path)
    return out


def load(paths: Sequence[str | Path]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in paths:
        p = Path(p)
        rows = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    continue
        rows.sort(key=lambda r: r["t"])
        out[p.stem] = rows
    return out


def summarise(rows: Sequence[Mapping], *, t0: float | None = None, t1: float | None = None) -> dict:
    """一个引擎的记录 → 步数、忙的比例、批大小分布、按批大小的每步用时（纯 decode 步）。"""
    rs = [r for r in rows if (t0 is None or r["t"] >= t0) and (t1 is None or r["t"] <= t1)]
    if not rs:
        return {"steps": 0}
    span = (rs[-1]["t"] + rs[-1]["ms"] / 1e3) - rs[0]["t"]
    busy = sum(r["ms"] for r in rs) / 1e3
    by_b: dict[int, list[float]] = {}
    for r in rs:
        if not r.get("npf"):
            by_b.setdefault(int(r["n"]), []).append(float(r["ms"]))
    pf = [r for r in rs if r.get("npf")]
    return {
        "steps": len(rs), "span_s": round(span, 3),
        "busy_frac": round(busy / span, 4) if span > 0 else None,
        "batch_mean": round(st.mean(r["n"] for r in rs), 3),
        "batch_hist": {str(b): len(v) for b, v in sorted(by_b.items())},
        "decode_ms_by_batch": {str(b): {"n": len(v), "p50": round(st.median(v), 3),
                                        "mean": round(st.mean(v), 3)} for b, v in sorted(by_b.items())},
        "prefill_steps": len(pf),
        "prefill_ms_per_token": (round(sum(r["ms"] for r in pf) / max(sum(r["pf"] for r in pf), 1), 4)
                                 if pf else None),
        "prefill_fit": prefill_fit(pf),
        "ctx_mean": round(st.mean(r.get("ctx", 0.0) for r in rs), 1),
    }


def prefill_fit(pf: Sequence[Mapping]) -> dict | None:
    """只有 prefill 的步：每步用时 ≈ a + b × prefill token 数（最小二乘）。先去掉冷启动：
    每 token 用时超过中位数 3 倍的步不算（引擎刚起来时前几次 prefill 慢好几倍）。"""
    pts = [(float(r["pf"]), float(r["ms"])) for r in pf if not (r.get("n", 0) - r.get("npf", 0)) and r["pf"] > 0]
    if not pts:
        return None
    per = sorted(ms / n for n, ms in pts)
    med = per[len(per) // 2]
    keep = [(n, ms) for n, ms in pts if ms / n <= 3 * med]
    out = {"n": len(keep), "dropped": len(pts) - len(keep), "ms_per_token_p50": round(med, 4)}
    if len(keep) >= 3 and len({n for n, _ in keep}) >= 2:
        mx = sum(n for n, _ in keep) / len(keep)
        my = sum(ms for _, ms in keep) / len(keep)
        sxx = sum((n - mx) ** 2 for n, _ in keep)
        b = sum((n - mx) * (ms - my) for n, ms in keep) / sxx
        a = my - b * mx
        if b > 0 and a >= 0:
            out.update(a_ms=round(a, 3), b_ms_per_token=round(b, 4))
    return out


def cmd_collect(a) -> int:
    stages = json.loads(Path(a.stages).read_text(encoding="utf-8"))
    files = trace_files(stages)
    if not files:
        print("stages.json 里没有开步级记录的引擎（stages.py 要加 --step-trace <目录>）", file=sys.stderr)
        return 2
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, (host, path) in sorted(files.items()):
        dst = out / f"{name}.jsonl"
        cmd = ["scp", "-q", "-o", "BatchMode=yes", f"{a.user}@{host}:{path}", str(dst)]
        rc = subprocess.run(cmd, check=False).returncode
        print(f"  {'✓' if rc == 0 else '✗'} {name:<22} {host}:{path}")
        bad += rc != 0
    print(f"已收到 {out}/（{len(files) - bad}/{len(files)}）\n汇总：python3 {sys.argv[0]} summary {out}/*.jsonl")
    return 1 if bad else 0


def cmd_summary(a) -> int:
    data = load(a.files)
    for name, rows in sorted(data.items()):
        s = summarise(rows, t0=a.t0, t1=a.t1)
        if not s["steps"]:
            print(f"{name}: 没有记录")
            continue
        dec = "  ".join(f"B{b}:{v['p50']:.2f}ms×{v['n']}" for b, v in s["decode_ms_by_batch"].items())
        print(f"{name}: {s['steps']} 步 / {s['span_s']:.0f}s，忙 {100 * (s['busy_frac'] or 0):.1f}%，"
              f"平均批 {s['batch_mean']:.2f}，平均上下文 {s['ctx_mean']:.0f}；prefill {s['prefill_steps']} 步"
              + (f"（{s['prefill_ms_per_token']:.3f} ms/token）" if s["prefill_ms_per_token"] else ""))
        print(f"    纯 decode 每步 p50：{dec}")
        pfit = s.get("prefill_fit")
        if pfit:
            lin = (f"{pfit['a_ms']:.1f} ms + {pfit['b_ms_per_token']:.3f} ms × token" if "a_ms" in pfit
                   else f"{pfit['ms_per_token_p50']:.3f} ms/token（中位数）")
            print(f"    prefill 步：{lin}（{pfit['n']} 步；去掉冷启动 {pfit['dropped']} 步）")
    if a.json:
        Path(a.json).write_text(json.dumps({k: summarise(v, t0=a.t0, t1=a.t1) for k, v in data.items()},
                                           ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="把各引擎 rank 0 节点上的记录 scp 回来")
    c.add_argument("--stages", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--user", default="ubuntu")
    c.add_argument("--dir", default=None, help="（只作说明；路径以 stages.json 命令里的为准）")
    s = sub.add_parser("summary", help="各引擎：步数、忙的比例、批大小、按批大小的每步用时")
    s.add_argument("files", nargs="+")
    s.add_argument("--t0", type=float, default=None, help="只看这个 time.time() 之后的")
    s.add_argument("--t1", type=float, default=None)
    s.add_argument("--json", default=None, help="汇总另存 JSON")
    a = ap.parse_args(argv)
    return cmd_collect(a) if a.cmd == "collect" else cmd_summary(a)


if __name__ == "__main__":
    raise SystemExit(main())
