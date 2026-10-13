# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""p2p_install.sh 的逐步检查：每一步「已经就绪了吗」。只用标准库 + pip 自带的 packaging。

    python -I vllm/p2p/envcheck.py torch                         # torch 2.9.1 能 import（含 CUDA 库）
    python -I vllm/p2p/envcheck.py reqs requirements/build.txt   # 这份 requirements 都满足了吗
    python -I vllm/p2p/envcheck.py vllm /home/ubuntu/vllm-p2p    # vLLM 以可编辑方式装的是这个目录、
                                                                 # 编译好的扩展在、依赖齐全

退出码 0 = 就绪（跳过这一步），1 = 缺东西（打印缺了什么），2 = 用法错。
用 `python -I` 跑：不把本文件所在目录放进 sys.path，也不读用户 site —— 检查的就是 venv 本身。
"""

from __future__ import annotations

import importlib.metadata as md
import json
import sys
from pathlib import Path

try:                                     # venv 里一定有 pip；pip 自带 packaging
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
except ImportError:                      # pragma: no cover
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

TORCH = {"torch": "2.9.1", "torchvision": "0.24.1", "torchaudio": "2.9.1"}
EXTENSIONS = ("vllm/_C.abi3.so", "vllm/_moe_C.abi3.so")


def _installed() -> dict[str, str]:
    out = {}
    for d in md.distributions():
        name = d.metadata.get("Name")
        if name:
            out[canonicalize_name(name)] = d.version
    return out


def _unsatisfied(reqs, have: dict[str, str]) -> list[str]:
    bad = []
    for raw in reqs:
        raw = raw.split("#", 1)[0].strip()
        if not raw or raw.startswith("-"):
            continue
        try:
            r = Requirement(raw)
        except Exception:
            continue
        if r.marker is not None and not r.marker.evaluate({"extra": ""}):
            continue
        v = have.get(canonicalize_name(r.name))
        if v is None:
            bad.append(f"{r.name}（没装）")
        elif r.specifier and not r.specifier.contains(v, prereleases=True):
            bad.append(f"{r.name} {v}（要 {r.specifier}）")
    return bad


def check_torch() -> list[str]:
    have = _installed()
    bad = [f"{n} {have.get(n, '没装')}（要 {v}）" for n, v in TORCH.items() if have.get(n) != v]
    if bad:
        return bad
    try:
        import torch  # noqa: F401  —— 这一步会加载 libcublas / libcusparseLt 等 CUDA 库
        import torch._C  # noqa: F401
    except Exception as e:
        return [f"import torch 失败：{type(e).__name__}: {e}"[:300]]
    return []


def check_reqs(path: str) -> list[str]:
    p = Path(path)
    if not p.is_file():
        return [f"{p} 不存在"]
    return _unsatisfied(p.read_text(encoding="utf-8").splitlines(), _installed())


def check_vllm(src: str) -> list[str]:
    src_p = Path(src).resolve()
    try:
        d = md.distribution("vllm")
    except md.PackageNotFoundError:
        return ["vllm 没装"]
    bad = []
    raw = d.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    url = info.get("url", "")
    editable = bool(info.get("dir_info", {}).get("editable"))
    if not (editable and url.startswith("file://")
            and Path(url[len("file://"):]).resolve() == src_p):
        bad.append(f"vllm 不是从 {src_p} 可编辑安装的（direct_url={url or '无'}, editable={editable}）")
    for so in EXTENSIONS:
        if not (src_p / so).is_file():
            bad.append(f"缺编译好的扩展 {so}（预编译那一步没跑完）")
    bad += _unsatisfied(d.requires or [], _installed())
    return bad


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    what, rest = argv[0], argv[1:]
    if what == "torch":
        bad = check_torch()
    elif what == "reqs" and len(rest) == 1:
        bad = check_reqs(rest[0])
    elif what == "vllm" and len(rest) == 1:
        bad = check_vllm(rest[0])
    else:
        print(__doc__)
        return 2
    for b in bad[:15]:
        print(f"    缺：{b}")
    if len(bad) > 15:
        print(f"    …… 共 {len(bad)} 项")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
