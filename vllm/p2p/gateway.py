# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 / A4 · 按 task 分发的薄网关（OpenAI 兼容，流式透传）。**按文件路径运行。**

    # 控制机上，vllm-p2p 的 venv（要 aiohttp）
    python vllm/p2p/gateway.py --channels p2p_channels/channels.json --mode oracle --port 9000
    python vllm/p2p/gateway.py --channels … --mode lr     --identifier http://192.168.1.2:8100
    python vllm/p2p/gateway.py --channels … --mode shadow --identifier http://192.168.1.2:8100

三种模式（实验矩阵 E3 / E4）：

* `oracle`：请求自带 task（`p2p_task`），直接转发 —— E3；
* `lr`：先问识别服务（`identifier.py`，前 L₀ 层 + LR），按它判的 task 转发 —— E4，
  「不解耦的代价」：识别耗时（含排队）记在 `route_ms` 里，算进这条请求的首 token 时延；
* `shadow`：按 `p2p_task` 转发，同时**旁路**问识别服务、只记结果不影响路由 ——
  与 p2p-framework 静态模式下的旁路 LR 同一口径，用来在 E3 里顺带量识别准确率。

同 task 有多条通道时取**在途数最少**的那条（相同时轮转）。某个 task 一条通道都没有时
立刻返回 503 并说明原因（与 p2p-framework 协调器「容量 0 立即失败」一致），不排队。

请求里认的扩展字段（转发前全部剥掉）：

    p2p_task  真实 / oracle 的 task            p2p_id    请求 id（不给就现编）
    p2p_body  body token 区间 [[a, b], …]（prompt 的 token 下标；LR 只数这些位置）

转发时把 `p2p_id` 与 body 区间放进 `vllm_xargs`（→ vLLM 的 `SamplingParams.extra_args`）：
A5 的统计钩子按它认请求、按它数 body；vLLM 也据此在最后的 usage 块里附上引擎侧的逐请求
时间（`p2p_metrics`）。同一个 id 还放进 `X-Request-Id` 头。

每条请求的路由记录可以 `GET /p2p/records` 取回（bench 用它拼结果），`GET /p2p/channels`
看各通道在途数。
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

__all__ = ["Router", "ChannelInfo", "load_channels", "body_string", "make_app", "main"]

MODES = ("oracle", "lr", "shadow")


@dataclass
class ChannelInfo:
    name: str
    task: str
    api: str
    """形如 http://192.168.1.2:8000/v1"""
    nodes: list[str] = field(default_factory=list)
    inflight: int = 0
    served: int = 0


def load_channels(path: str | Path) -> list[ChannelInfo]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for c in d["channels"]:
        if not c.get("api"):
            raise SystemExit(f"{path} 里 {c['name']} 没有 api 地址 —— 生成时要给 layout.py 传 --hosts")
        out.append(ChannelInfo(name=c["name"], task=c["task"], api=c["api"].rstrip("/"),
                               nodes=list(c.get("nodes", []))))
    if not out:
        raise SystemExit(f"{path} 里一条通道都没有")
    return out


def body_string(ranges: Sequence[Sequence[int]] | None) -> str:
    """[[3, 40], [52, 60]] → "3:40,52:60"（vllm_xargs 只收标量或标量列表）。"""
    return ",".join(f"{int(a)}:{int(b)}" for a, b in (ranges or ()))


class NoChannel(Exception):
    pass


class Router:
    """task → 通道。纯逻辑（不碰网络），单测直接测它。"""

    def __init__(self, channels: Sequence[ChannelInfo]):
        self.channels = list(channels)
        self.by_task: dict[str, list[ChannelInfo]] = {}
        for c in self.channels:
            self.by_task.setdefault(c.task, []).append(c)
        self._rr = {u: itertools.count() for u in self.by_task}
        """每个 task 各自轮转（共用一个计数器的话，别的 task 的请求会让它总落在同一条上）。"""

    @property
    def tasks(self) -> list[str]:
        return sorted(self.by_task)

    def pick(self, task: str) -> ChannelInfo:
        cs = self.by_task.get(task)
        if not cs:
            have = {u: len(v) for u, v in sorted(self.by_task.items())}
            raise NoChannel(f"task {task!r} 一条通道都没有（现有 {have}）—— 不是拥塞，"
                            "是部署时就没给它建通道，排队也等不来")
        low = min(c.inflight for c in cs)
        cands = [c for c in cs if c.inflight == low]
        return cands[next(self._rr[task]) % len(cands)]

    def state(self) -> list[dict]:
        return [asdict(c) for c in self.channels]


def strip_p2p(body: Mapping) -> tuple[dict, dict]:
    """→ (转发给 vLLM 的请求体, 剥下来的 p2p_* 字段)。"""
    fwd, ext = {}, {}
    for k, v in body.items():
        (ext if k.startswith("p2p_") else fwd)[k] = v
    return fwd, ext


def with_xargs(fwd: dict, p2p_id: str, body_ranges) -> dict:
    xa = dict(fwd.get("vllm_xargs") or {})
    xa["p2p_id"] = p2p_id
    if body_ranges:
        xa["p2p_body"] = body_string(body_ranges)
    out = dict(fwd)
    out["vllm_xargs"] = xa
    return out


# --------------------------------------------------------------------------- #
def make_app(router: Router, *, mode: str, identifier: str | None = None,
             ident_layers: bool = False, log_path: Path | None = None):
    import aiohttp
    from aiohttp import web

    if mode not in MODES:
        raise ValueError(f"mode={mode!r}；可选 {MODES}")
    if mode in ("lr", "shadow") and not identifier:
        raise ValueError(f"--mode {mode} 要给 --identifier（识别服务地址）")
    records: list[dict] = []
    shadow_tasks: set[asyncio.Task] = set()
    state: dict = {}

    async def on_startup(app):
        state["sess"] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
            connector=aiohttp.TCPConnector(limit=0))

    async def on_cleanup(app):
        await state["sess"].close()

    def log(rec: dict) -> None:
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    async def identify(ids, body) -> dict:
        t0 = time.perf_counter()
        async with state["sess"].post(f"{identifier.rstrip('/')}/identify",
                                      json={"ids": ids, "body": body,
                                            "layers": ident_layers}) as r:
            d = await r.json()
            if r.status != 200:
                raise RuntimeError(d.get("error") or f"识别服务返回 {r.status}")
        d["rtt_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        return d

    def bad(status: int, msg: str, rec: dict | None = None) -> web.Response:
        if rec is not None:
            rec.update(status=status, error=msg, t_done=time.time())
            log(rec)
        return web.json_response({"error": {"message": msg, "code": status}}, status=status,
                                 dumps=lambda o: json.dumps(o, ensure_ascii=False))

    async def completions(request: web.Request, path: str) -> web.StreamResponse:
        t_recv_wall, t_recv = time.time(), time.perf_counter()
        try:
            body = await request.json()
        except Exception as e:                                    # noqa: BLE001
            return bad(400, f"请求体不是 JSON：{e}")
        fwd, ext = strip_p2p(body)
        pid = str(ext.get("p2p_id") or request.headers.get("X-Request-Id")
                  or f"gw-{uuid.uuid4().hex[:12]}")
        true_task = ext.get("p2p_task")
        prange = ext.get("p2p_body")
        rec = {"id": pid, "mode": mode, "path": path, "true_task": true_task,
               "t_recv": t_recv_wall, "task": None, "channel": None, "verdict": None,
               "shadow": None, "route_ms": 0.0, "status": None}
        records.append(rec)
        ids = fwd.get("prompt")
        is_ids = isinstance(ids, list) and ids and all(isinstance(x, int) for x in ids)

        if mode in ("oracle", "shadow"):
            if not true_task:
                return bad(400, f"{mode} 模式要在请求里给 p2p_task", rec)
            task = str(true_task)
            if mode == "shadow":
                if path != "completions" or not is_ids:
                    rec["shadow"] = {"error": "旁路识别要 /v1/completions 且 prompt 是 token id"}
                else:
                    async def run_shadow():
                        try:
                            rec["shadow"] = await identify(ids, prange)
                        except Exception as e:                    # noqa: BLE001
                            rec["shadow"] = {"error": f"{type(e).__name__}: {e}"}
                    t = asyncio.create_task(run_shadow())
                    shadow_tasks.add(t)
                    t.add_done_callback(shadow_tasks.discard)
        else:  # lr
            if path != "completions" or not is_ids:
                return bad(400, "lr 模式要走 /v1/completions，prompt 给 token id 列表"
                                "（bench 用 p2p-framework 的 encode_body 构造，与选择器训练时同一口径）",
                           rec)
            try:
                v = await identify(ids, prange)
            except Exception as e:                                # noqa: BLE001
                return bad(502, f"识别服务出错：{type(e).__name__}: {e}", rec)
            rec["verdict"] = v
            task = v["task"]
        rec["task"] = task
        try:
            ch = router.pick(task)
        except NoChannel as e:
            return bad(503, str(e), rec)
        rec["channel"] = ch.name
        rec["route_ms"] = round((time.perf_counter() - t_recv) * 1000, 3)
        out = with_xargs(dict(fwd, model=ch.name), pid, prange)
        hdr = {"X-P2P-Id": pid, "X-P2P-Channel": ch.name, "X-P2P-Task": task,
               "X-P2P-Route-Ms": str(rec["route_ms"])}
        if rec["verdict"] is not None:
            hdr["X-P2P-LR"] = json.dumps({k: rec["verdict"].get(k) for k in
                                          ("task", "label", "conf", "zone")},
                                         ensure_ascii=True, separators=(",", ":"))
        ch.inflight += 1
        resp = None
        try:
            async with state["sess"].post(f"{ch.api}/{path.replace('_', '/')}", json=out,
                                          headers={"X-Request-Id": pid}) as up:
                rec["status"] = up.status
                if up.status != 200 or not out.get("stream"):
                    data = await up.read()
                    rec["t_first_byte"] = rec["t_done"] = time.time()
                    ctype = up.headers.get("Content-Type", "application/json")
                    return web.Response(body=data, status=up.status, headers=hdr,
                                        content_type=ctype.split(";")[0])
                resp = web.StreamResponse(status=200, headers={
                    **hdr, "Content-Type": up.headers.get("Content-Type", "text/event-stream")})
                await resp.prepare(request)
                first = True
                async for chunk in up.content.iter_any():
                    if first:
                        rec["t_first_byte"] = time.time()
                        first = False
                    await resp.write(chunk)
                await resp.write_eof()
                rec["t_done"] = time.time()
                return resp
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            if resp is None:
                rec["status"] = 502
                return bad(502, f"转发到 {ch.name}（{ch.api}）失败：{rec['error']}", None)
            return resp
        finally:
            ch.inflight -= 1
            ch.served += 1
            log(rec)

    async def v1_completions(request):
        return await completions(request, "completions")

    async def v1_chat(request):
        return await completions(request, "chat_completions")

    async def get_records(request: web.Request):
        since = int(request.query.get("since", 0))
        # 旁路识别可能还没回来：等一会儿（最多 30s）再交
        if shadow_tasks and request.query.get("wait", "1") != "0":
            await asyncio.wait(set(shadow_tasks), timeout=30)
        return web.json_response({"mode": mode, "n": len(records), "records": records[since:]})

    async def get_channels(_):
        return web.json_response({"mode": mode, "identifier": identifier,
                                  "tasks": router.tasks, "channels": router.state()})

    async def health(_):
        return web.json_response({"ok": True, "mode": mode})

    async def reset(_):
        records.clear()
        return web.json_response({"ok": True})

    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.add_routes([
        web.post("/v1/completions", v1_completions),
        web.post("/v1/chat/completions", v1_chat),
        web.get("/p2p/records", get_records),
        web.post("/p2p/reset", reset),
        web.get("/p2p/channels", get_channels),
        web.get("/health", health),
    ])
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p2p-gateway", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--channels", default="/home/ubuntu/vllm-p2p/p2p_channels/channels.json")
    ap.add_argument("--mode", choices=MODES, default="oracle")
    ap.add_argument("--identifier", default=None, help="识别服务地址，如 http://192.168.1.2:8100")
    ap.add_argument("--ident-layers", action="store_true",
                    help="让识别服务把逐层计数也带回来（A5 核对钩子用）")
    ap.add_argument("--only", default=None, help="只用这些通道（逗号分隔）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--log", type=Path, default=None, help="逐请求路由记录另写一份 JSONL")
    a = ap.parse_args(argv)

    chans = load_channels(a.channels)
    if a.only:
        want = {x.strip() for x in a.only.split(",") if x.strip()}
        chans = [c for c in chans if c.name in want]
        if not chans:
            raise SystemExit(f"--only {a.only} 一条都没对上")
    router = Router(chans)
    from aiohttp import web

    print(f"网关 {a.mode}：{len(chans)} 条通道 "
          f"{ {u: [c.name for c in cs] for u, cs in sorted(router.by_task.items())} }"
          + (f"，识别服务 {a.identifier}" if a.identifier else ""), flush=True)
    print(f"监听 {a.host}:{a.port}", flush=True)
    web.run_app(make_app(router, mode=a.mode, identifier=a.identifier,
                         ident_layers=a.ident_layers, log_path=a.log),
                host=a.host, port=a.port, print=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
