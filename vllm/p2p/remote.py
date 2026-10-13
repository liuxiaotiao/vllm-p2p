# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B1：协调器与**别的机器上**的分段引擎之间的线路（ZMQ DEALER ↔ ROUTER，vLLM 自己的 msgpack）。

每个分段引擎（前段 / 后段）的首节点上跑一个代理 `stage_agent.py`：它在本机起 vLLM 的
EngineCore（`AsyncMPClient`，PP 多机时 worker 在其余节点上以 headless 起），再把 EngineCore 的
四个入口原样搬到网络上。协调器这边的 `RemoteClient` 提供与 `AsyncMPClient` 相同的那几个方法，
所以 `tiered.TieredCoordinator` 不用改就能接远端引擎。

帧格式（每条消息一个 multipart）：

    协调器 → 代理   [H]                         握手（代理回 [I, info json]）
                    [A, *EngineCoreRequest]     add_request
                    [X, *list[str]]             abort
                    [U, *(method, args)]        utility，不等结果（失败时代理回 [E, …]）
    代理 → 协调器   [O, *EngineCoreOutputs]     引擎的每一批输出
                    [I, info json]  [E, 错误文本]

张量走 vLLM 的零拷贝辅助帧（`MsgpackEncoder` 把大张量放进单独的帧）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

__all__ = ["RemoteClient", "AgentBridge", "K_HELLO", "K_ADD", "K_ABORT", "K_UTIL",
           "K_OUT", "K_INFO", "K_ERR"]

K_HELLO, K_ADD, K_ABORT, K_UTIL = b"H", b"A", b"X", b"U"
K_OUT, K_INFO, K_ERR = b"O", b"I", b"E"


class RemoteClient:
    """协调器侧：长得像 `AsyncMPClient` 的远端分段引擎。"""

    def __init__(self, address: str, *, ctx=None):
        import zmq
        import zmq.asyncio

        from vllm.v1.engine import EngineCoreOutputs
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

        self.address = address
        self._own_ctx = ctx is None
        self.ctx = ctx or zmq.asyncio.Context()
        self.sock = self.ctx.socket(zmq.DEALER)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(address)
        self.enc = MsgpackEncoder()
        self.dec_out = MsgpackDecoder(EngineCoreOutputs)
        self.dec_any = MsgpackDecoder()
        self.info: dict = {}
        self._pending_info: asyncio.Future | None = None

    async def connect(self, timeout: float = 30.0) -> dict:
        """握手：拿到代理报的本段信息（角色、层、hidden 维度…）。"""
        await self.sock.send_multipart([K_HELLO])
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            left = deadline - asyncio.get_running_loop().time()
            if left <= 0:
                raise TimeoutError(f"{self.address} {timeout:.0f}s 内没有回握手 —— 代理没起或端口不通")
            frames = await asyncio.wait_for(self.sock.recv_multipart(copy=False), left)
            kind = bytes(frames[0])
            if kind == K_INFO:
                self.info = json.loads(bytes(frames[1]))
                return self.info
            if kind == K_ERR:
                raise RuntimeError(f"{self.address}：{bytes(frames[1]).decode('utf-8', 'replace')}")
            # 握手前到的引擎输出：上一个协调器留下的，丢掉

    async def _send(self, kind: bytes, obj: Any) -> None:
        bufs = self.enc.encode(obj)
        await self.sock.send_multipart([kind, *bufs], copy=False)

    async def add_request_async(self, req) -> None:
        await self._send(K_ADD, req)

    async def abort_requests_async(self, ids: list[str]) -> None:
        if ids:
            await self._send(K_ABORT, list(ids))

    async def p2p_fire(self, method: str, *args):
        await self._send(K_UTIL, (method, list(args)))
        fut = asyncio.get_running_loop().create_future()
        fut.set_result(None)            # 失败由代理以 [E] 异步报回（get_output_async 里抛）
        return fut

    async def get_output_async(self):
        while True:
            frames = await self.sock.recv_multipart(copy=False)
            kind = bytes(frames[0])
            if kind == K_OUT:
                return self.dec_out.decode(frames[1:])
            if kind == K_ERR:
                raise RuntimeError(f"分段引擎 {self.info.get('name') or self.address} 报错："
                                   f"{bytes(frames[1]).decode('utf-8', 'replace')}")
            if kind == K_INFO:
                self.info = json.loads(bytes(frames[1]))

    def shutdown(self) -> None:
        self.sock.close(0)
        if self._own_ctx:
            self.ctx.term()


class AgentBridge:
    """代理侧：ROUTER socket ↔ 本机 EngineCore（`AsyncMPClient`）。

    只认最近一个发过消息的协调器（同一时刻只有一个协调器连着一个分段引擎）。"""

    def __init__(self, client, bind: str, info: dict):
        import zmq
        import zmq.asyncio

        from vllm.v1.engine import EngineCoreRequest
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

        self.client, self.info = client, info
        self.ctx = zmq.asyncio.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.ROUTER_HANDOVER, 1)      # 协调器重连（同 identity）也认
        self.sock.bind(bind)
        self.enc = MsgpackEncoder()
        self.dec_req = MsgpackDecoder(EngineCoreRequest)
        self.dec_any = MsgpackDecoder()
        self.peer: bytes | None = None
        self.n_in = self.n_out = 0
        self.dead: str | None = None

    async def _err(self, msg: str) -> None:
        if self.peer is not None:
            await self.sock.send_multipart([self.peer, K_ERR, msg.encode("utf-8")])

    async def _inbound(self) -> None:
        from vllm.p2p.tiered import _fire_utility

        while True:
            frames = await self.sock.recv_multipart(copy=False)
            self.peer = bytes(frames[0])
            kind = bytes(frames[1])
            body = frames[2:]
            self.n_in += 1
            try:
                if kind == K_HELLO:
                    await self.sock.send_multipart(
                        [self.peer, K_INFO, json.dumps(self.info).encode("utf-8")])
                elif kind == K_ADD:
                    await self.client.add_request_async(self.dec_req.decode(body))
                elif kind == K_ABORT:
                    await self.client.abort_requests_async(list(self.dec_any.decode(body)))
                elif kind == K_UTIL:
                    method, args = self.dec_any.decode(body)
                    fire = getattr(self.client, "p2p_fire", None)     # 测试用的假引擎
                    fut = await (fire(method, *args) if fire is not None
                                 else _fire_utility(self.client, method, *args))
                    peer = self.peer

                    def done(f, method=method, peer=peer):
                        if not f.cancelled() and f.exception() is not None:
                            asyncio.ensure_future(self.sock.send_multipart(
                                [peer, K_ERR, f"{method}: {f.exception()}".encode("utf-8")]))
                    fut.add_done_callback(done)
                else:
                    await self._err(f"不认识的消息类型 {kind!r}")
            except Exception as e:                           # noqa: BLE001
                await self._err(f"处理 {kind!r} 失败：{type(e).__name__}: {e}")

    async def _outbound(self) -> None:
        while True:
            try:
                outs = await self.client.get_output_async()
            except Exception as e:                           # noqa: BLE001 — 引擎死了
                self.dead = f"{type(e).__name__}: {e}"
                await self._err(f"EngineCore 退出：{self.dead}")
                raise
            if self.peer is None or not outs.outputs:
                continue
            self.n_out += 1
            await self.sock.send_multipart([self.peer, K_OUT, *self.enc.encode(outs)], copy=False)

    async def run(self) -> None:
        await asyncio.gather(self._inbound(), self._outbound())
