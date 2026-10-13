# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P 移植：`vllm/p2p/launch.py` 生成的远端 shell（不用 ssh，直接在本机 bash 里跑）。

钉住的是两个实际踩过的坑：
* `cd /tmp && nohup … &` 会让 ssh 卡到 vllm 退出 —— start 必须立即返回；
* `pgrep -f` 用字面量会匹配到执行查询的 shell 自己 —— 必须只找到真正的进程。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import pytest

from vllm.p2p import launch as L

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or sys.platform == "win32",
                                reason="需要 bash / pgrep")

# 不在本文件里拼出完整字面量，免得 pgrep 匹配到 pytest 进程自己
NAME = "p2ptest" + "-0"
FLAG = "--served" + "-model-name"


def _sh(s: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", s], capture_output=True, text=True, timeout=30)


def test_start_returns_and_passes_env_and_overrides(tmp_path) -> None:
    (tmp_path / "ov.json").write_text('{"architectures": ["X"]}', encoding="utf-8")
    got = tmp_path / "got"
    fake = (f"VLLM_PP_LAYER_PARTITION=8,13,27 {sys.executable} -c 'import os,sys,time; "
            f"open(\"{got}\",\"w\").write(os.environ[\"VLLM_PP_LAYER_PARTITION\"]+\"|\""
            f"+os.environ.get(\"EXTRA\",\"\")+\"|\"+sys.argv[-1]); time.sleep(60)' "
            f"{FLAG} {NAME} --hf-overrides \"$(cat {tmp_path / 'ov.json'})\"")
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                              FLAG, "p2ptest" + "-1", "--x"])
    try:
        t = time.time()
        r = _sh(L.remote_start(fake, NAME, "N01", {"EXTRA": "e1"}))
        assert time.time() - t < 5, "start 没有立即返回 —— ssh 会卡到 vllm 退出"
        pid = r.stdout.strip()
        for _ in range(50):
            if got.exists() and got.read_text():
                break
            time.sleep(0.1)
        assert got.read_text() == '8,13,27|e1|{"architectures": ["X"]}'
        assert _sh(L.remote_running(NAME)).stdout.split() == [pid]
        assert _sh(L.remote_running("p2ptest")).stdout.split() == []      # 前缀不误匹配
        _sh(L.remote_stop(NAME))
        assert _sh(L.remote_running(NAME)).stdout.split() == []
        assert other.poll() is None, "停一条通道把别的通道也杀了"
    finally:
        other.kill()
        _sh(L.remote_stop(NAME))
        if os.path.exists(L.log_path(NAME, "N01")):
            os.remove(L.log_path(NAME, "N01"))


def test_match_pattern_does_not_match_its_own_query() -> None:
    import re
    pat = L.match_pattern(NAME)
    assert re.search(pat, f"vllm serve /w {FLAG} {NAME} --pipeline-parallel-size 3")
    assert re.search(pat, f"vllm serve /w {FLAG} {NAME}")
    assert not re.search(pat, f"vllm serve /w {FLAG} {NAME}0 --x")
    assert not re.search(pat, L.remote_running(NAME))     # 执行查询的 shell 自己的命令行
