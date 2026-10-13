#!/usr/bin/env bash
# P2P 移植：在一台节点上安装本目录（修改后的 vLLM 0.14.0）。每台节点执行一次。
#
#     bash /home/ubuntu/vllm-p2p/p2p_install.sh
#
# 做的事：用 moe 环境的 Python 解释器建一个**完全独立**的 venv（不复用 moe 的任何包，moe 本身不改），
# 装 torch 2.9.1（连同它的 CUDA 12 库），再以「可编辑」方式安装本目录 —— Python 代码直接用本目录的
# （含部分专家模型、vllm/p2p/），只从官方 0.14.0 wheel 里取编译好的 CUDA 扩展（.so），不在节点上编译。
#
# 为什么不复用 moe 里已有的 CUDA 库（--system-site-packages）：torch 找 libcublas / libcusparseLt 等
# 靠的是写在 .so 里的相对路径 $ORIGIN/../../nvidia/*/lib —— 必须和 torch 装在同一个 site-packages。
# venv 里的 torch 2.9.1 找不到 moe 目录下的库（ImportError: libcusparseLt.so.0），实测踩过。
#
# 可覆盖的变量（一般不用改）：
#   SRC      本目录（修改后的 vLLM）          默认 /home/ubuntu/vllm-p2p
#   MOE_PY   moe 环境的 python                默认 /home/ubuntu/anaconda3/envs/moe/bin/python
#   VENV     新建的 venv                      默认 /home/ubuntu/venvs/vllm-p2p
#   WHEEL    已经下好的官方 wheel（可选）      给了就不再联网下载
set -euo pipefail

SRC=${SRC:-/home/ubuntu/vllm-p2p}
MOE_PY=${MOE_PY:-/home/ubuntu/anaconda3/envs/moe/bin/python}
VENV=${VENV:-/home/ubuntu/venvs/vllm-p2p}
WHEEL=${WHEEL:-}
TMPW=$(mktemp -d /tmp/vllm-wheel.XXXXXX)
trap 'rm -rf "$TMPW"' EXIT

say() { printf '\n== %s\n' "$*"; }
die() { printf '✗ %s\n' "$*" >&2; exit 1; }

say "0/6 预检（$(hostname)）"
[ -f "$SRC/vllm/model_executor/models/qwen3_moe_partial.py" ] \
  || die "$SRC 里没有 qwen3_moe_partial.py —— 不是修改后的 vLLM，或者还没同步过来"
[ -x "$MOE_PY" ] || die "找不到 moe 的 python：$MOE_PY（用 MOE_PY=… 指定）"
pyv=$("$MOE_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$pyv" in 3.10|3.11|3.12|3.13) echo "  moe Python $pyv ✓" ;;
  *) die "moe 是 Python $pyv，vLLM 0.14 要 3.10–3.13" ;; esac
glibc=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0)
awk -v v="$glibc" 'BEGIN{split(v,a,"."); exit !(a[1]>2 || (a[1]==2 && a[2]>=31))}' \
  || die "glibc $glibc < 2.31，官方 0.14.0 wheel（manylinux_2_31）装不上"
echo "  glibc $glibc ✓"
d=$(dirname "$VENV"); while [ ! -d "$d" ]; do d=$(dirname "$d"); done   # 第一次跑时 venvs/ 还不存在
free_gb=$(df -Pk "$d" | awk 'NR==2{printf "%d", $4/1048576}')
[ "${free_gb:-0}" -ge 12 ] || die "$d 只剩 ${free_gb}GB，至少要 12GB（装完约 10GB）"
echo "  磁盘剩 ${free_gb}GB ✓"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU /'

say "1/6 建独立 venv（只借 moe 的 Python 解释器，不读 moe 的包）"
mkdir -p "$(dirname "$VENV")"
if [ -f "$VENV/pyvenv.cfg" ] && grep -qiE '^include-system-site-packages *= *true' "$VENV/pyvenv.cfg"; then
  echo "  $VENV 是旧的 --system-site-packages 版本（会找不到 CUDA 库）—— 删掉重建"
  rm -rf "$VENV"
fi
[ -x "$VENV/bin/python" ] || "$MOE_PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
[ "$(python -c 'import sys; print(sys.prefix)')" = "$VENV" ] || die "venv 没激活成功"
export PIP_NO_CACHE_DIR=1
pip install -q --upgrade pip

say "2/6 torch 2.9.1 + CUDA 12 库（vLLM 0.14 钉死这个版本；moe 里的 2.9.0 不动）"
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1

say "3/6 构建工具（关了构建隔离，所以要自己装）"
cd "$SRC"
pip install -r requirements/build.txt

say "4/6 官方 0.14.0 wheel —— 只用来取编译好的 .so"
if [ -z "$WHEEL" ]; then
  pip download vllm==0.14.0 --no-deps -d "$TMPW"
  WHEEL=$(ls "$TMPW"/vllm-0.14.0-*x86_64.whl)
fi
[ -f "$WHEEL" ] || die "wheel 不存在：$WHEEL"
echo "  $WHEEL"

say "5/6 可编辑安装本目录"
VLLM_VERSION_OVERRIDE=0.14.0 \
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION="$WHEEL" \
  pip install --no-build-isolation -e .

say "6/6 验证"
cd /
python - <<'PY'
import sys
import torch, vllm
from vllm import ModelRegistry
print(f"  torch {torch.__version__}  ←  {torch.__file__}")
print(f"  vllm  {vllm.__version__}  ←  {vllm.__file__}")
assert torch.__version__.startswith("2.9.1"), "用到的不是 venv 里的 torch 2.9.1"
assert "/site-packages/" not in vllm.__file__, "import 到的不是本目录的 vllm"
assert torch.__file__.startswith(sys.prefix), "torch 不是 venv 里的那份"
assert "Qwen3MoePartialForCausalLM" in ModelRegistry.get_supported_archs(), "部分专家模型没注册上"
import vllm._C  # noqa: F401  —— 编译好的扩展能加载
import vllm._moe_C  # noqa: F401
print("  Qwen3MoePartialForCausalLM 已注册，vllm._C 可加载 ✓")
print(f"  CUDA 可用: {torch.cuda.is_available()}" + (f"，{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
PY
echo "  moe 的 torch 仍是 $("$MOE_PY" -c 'import torch; print(torch.__version__)')（应为 2.9.0）"
echo
echo "装好了。以后用：source $VENV/bin/activate"
echo "远程起服务时用绝对路径：$VENV/bin/vllm（ssh 非交互 shell 不会激活 venv）"
