# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""拉起 / 查看 / 停止 vLLM 静态通道（P2P 移植阶段 A2）。**纯 Python，控制机上按路径直接跑。**

    python3 vllm/p2p/launch.py start  gsm8k-0          # 按 channels.json 在各级节点上后台启动
    python3 vllm/p2p/launch.py status gsm8k-0          # 进程在不在、API 健康检查通没通、KV 容量
    python3 vllm/p2p/launch.py logs   gsm8k-0 [-n 40]  # 各级日志的最后几行
    python3 vllm/p2p/launch.py stop   gsm8k-0          # 停掉
    python3 vllm/p2p/launch.py status all              # 所有通道

`channels.json` 由 `vllm/p2p/layout.py --hosts … --out DIR` 生成（默认读
/home/ubuntu/vllm-p2p/p2p_channels/channels.json，`--channels` 可改）。各级的启动命令、
地址都在里面，这里只负责 ssh 过去执行 —— 命令本身一个字不改，保证与 layout 生成的一致。

日志在各节点的 /tmp/p2p_vllm_<通道>_<节点>.log。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_CHANNELS = "/home/ubuntu/vllm-p2p/p2p_channels/channels.json"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "StrictHostKeyChecking=accept-new"]


def log_path(channel: str, node: str) -> str:
    return f"/tmp/p2p_vllm_{channel}_{node}.log"


def match_pattern(channel: str) -> str:
    """pgrep / pkill 用的**正则**：只认本通道的 vllm serve 进程。

    写成 `[-]-served-model-name …` 而不是字面量：ssh 执行 `pgrep -f '<模式>'` 的那个 shell，
    自己的命令行里就带着这串字符 —— 字面量会匹配到它，于是「没在跑」也报成在跑。
    方括号写法匹配真正的 `--served-model-name`，却匹配不到写着 `[-]-served…` 的那条命令行。
    """
    return f"[-]-served-model-name {channel}( |$)"


def remote_start(cmd: str, channel: str, node: str, env: dict[str, str]) -> str:
    """远端执行的 shell：后台起 vllm serve，输出写日志，ssh 断开后进程不退。

    `cmd` 原样放进 `bash -c`（里面有 `$(cat overrides.json)`，要由远端 bash 展开）。
    `cmd` 以 `VLLM_PP_LAYER_PARTITION=… vllm serve …` 开头 —— `exec VAR=值 命令` 在 bash 里
    不成立（VAR=值 会被当成命令名），所以用 `exec env …`：进程直接替换成 vllm，pgrep 只见一个。
    """
    exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in sorted(env.items()))
    log = log_path(channel, node)
    inner = f"{exports}exec env {cmd}"
    # 注意是 `cd /tmp;` 不是 `cd /tmp &&`：`A && B &` 会把整个 `A && B` 放进一个后台子 shell，
    # 它一直占着 ssh 的输出管道，ssh 就要等到 vllm 退出才返回（实测卡死过）。
    return (f"cd /tmp || exit 1; nohup bash -c {shlex.quote(inner)} > {shlex.quote(log)} 2>&1 "
            f"< /dev/null & echo $!")


def remote_running(channel: str) -> str:
    return f"pgrep -f -- {shlex.quote(match_pattern(channel))} || true"


def remote_stop(channel: str) -> str:
    pat = shlex.quote(match_pattern(channel))
    return (f"pkill -TERM -f -- {pat}; for i in $(seq 20); do pgrep -f -- {pat} >/dev/null || exit 0; "
            f"sleep 1; done; pkill -KILL -f -- {pat}; true")


# --------------------------------------------------------------------------- #
def _ssh(host: str, user: str, script: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run([*SSH, f"{user}@{host}", script], capture_output=True, text=True,
                          timeout=timeout)


def _health(api: str, timeout: float = 3.0) -> bool:
    base = api.rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _load(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"找不到 {p} —— 先用 vllm/p2p/layout.py --hosts … --out {p.parent} 生成")
    chans = json.loads(p.read_text(encoding="utf-8"))["channels"]
    if not all("commands" in c for c in chans):
        raise SystemExit(f"{p} 里没有启动命令 —— 生成时要给 layout.py 传 --hosts")
    return chans


def _pick(chans: list[dict], name: str) -> list[dict]:
    if name == "all":
        return chans
    got = [c for c in chans if c["name"] == name]
    if not got:
        raise SystemExit(f"没有通道 {name}；有的是 {[c['name'] for c in chans]}")
    return got


def _kv_lines(host: str, user: str, channel: str, node: str) -> list[str]:
    r = _ssh(host, user, f"grep -E 'KV cache memory|Maximum concurrency|GPU KV cache size' "
                         f"{shlex.quote(log_path(channel, node))} 2>/dev/null | tail -3")
    return [l.strip() for l in r.stdout.splitlines() if l.strip()]


def cmd_start(c: dict, user: str, env: dict[str, str], wait: float) -> int:
    name = c["name"]
    for st in c["commands"]:
        r = _ssh(st["host"], user, remote_running(name))
        if r.stdout.strip():
            print(f"  ✗ {st['node']} 上 {name} 已经在跑（pid {r.stdout.split()}）—— 先 stop")
            return 1
    # 先起 headless 的各级，最后起 node_rank 0（它起 API 服务、等各级连上）
    for st in sorted(c["commands"], key=lambda s: "--node-rank 0 " in s["cmd"]):
        r = _ssh(st["host"], user, remote_start(st["cmd"], name, st["node"], env))
        if r.returncode != 0:
            print(f"  ✗ {st['node']}@{st['host']} 启动失败：{r.stderr.strip()[:300]}")
            return 1
        print(f"  → {st['node']}@{st['host']} 已启动 pid {r.stdout.strip()}，日志 {log_path(name, st['node'])}")
    if wait <= 0:
        return 0
    print(f"  等 API 就绪（最多 {wait:.0f}s）：{c['api']}")
    t0 = time.time()
    while time.time() - t0 < wait:
        if _health(c["api"]):
            print(f"  ✓ {name} 就绪（{time.time() - t0:.0f}s）")
            head = c["commands"][0]
            for l in _kv_lines(head["host"], user, name, head["node"]):
                print(f"      {l[-160:]}")
            return 0
        for st in c["commands"]:
            if not _ssh(st["host"], user, remote_running(name)).stdout.strip():
                print(f"  ✗ {st['node']} 上的进程退出了 —— 看日志：")
                print(_ssh(st["host"], user,
                           f"tail -25 {shlex.quote(log_path(name, st['node']))}").stdout)
                return 1
        time.sleep(10)
    print(f"  ✗ {wait:.0f}s 内没就绪 —— python3 {sys.argv[0]} logs {name}")
    return 1


def cmd_status(c: dict, user: str) -> int:
    name = c["name"]
    up = _health(c["api"])
    print(f"{name:12s} API {'✓ 就绪' if up else '✗ 不通'}  {c['api']}")
    bad = 0
    for st in c["commands"]:
        r = _ssh(st["host"], user, remote_running(name))
        pids = r.stdout.split()
        print(f"    {st['node']:4s} {st['host']:15s} {'运行中 pid ' + ','.join(pids) if pids else '没有进程'}")
        bad += not pids
    if up:
        head = c["commands"][0]
        for l in _kv_lines(head["host"], user, name, head["node"]):
            print(f"      {l[-160:]}")
    return 0 if up and not bad else 1


def cmd_logs(c: dict, user: str, n: int) -> int:
    for st in c["commands"]:
        print(f"===== {c['name']} / {st['node']} @ {st['host']}  {log_path(c['name'], st['node'])}")
        r = _ssh(st["host"], user, f"tail -n {int(n)} {shlex.quote(log_path(c['name'], st['node']))}")
        print(r.stdout or r.stderr)
    return 0


def cmd_stop(c: dict, user: str) -> int:
    for st in c["commands"]:
        _ssh(st["host"], user, remote_stop(c["name"]), timeout=60)
        print(f"  ■ {st['node']}@{st['host']} 已停")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("start", "status", "logs", "stop"))
    ap.add_argument("channel", help="通道名（如 gsm8k-0），或 all")
    ap.add_argument("--channels", default=DEFAULT_CHANNELS, help="layout.py 生成的 channels.json")
    ap.add_argument("--user", default="ubuntu", help="ssh 用户")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="额外环境变量，如 NCCL_SOCKET_IFNAME=eth0（多网卡时 NCCL 选错网卡会卡在初始化）")
    ap.add_argument("--wait", type=float, default=900, help="start 后等 API 就绪的秒数（0 = 不等）")
    ap.add_argument("-n", type=int, default=40, help="logs 显示的行数")
    a = ap.parse_args(argv)
    env = dict(kv.split("=", 1) for kv in a.env)
    rc = 0
    for c in _pick(_load(a.channels), a.channel):
        if a.action == "start":
            print(f"== 启动 {c['name']}（{' → '.join(c['nodes'])}，PP={','.join(map(str, c['partition']))}，"
                  f"dtype={c['dtype']}）")
            rc |= cmd_start(c, a.user, env, a.wait)
        elif a.action == "status":
            rc |= cmd_status(c, a.user)
        elif a.action == "logs":
            rc |= cmd_logs(c, a.user, a.n)
        else:
            print(f"== 停止 {c['name']}")
            rc |= cmd_stop(c, a.user)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
