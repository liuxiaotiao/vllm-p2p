# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P2P MoE 移植：把 p2p-framework 的「双段放置 + 只装部分专家」搬到 vLLM 上。

计划与验收标准见 p2p-framework 的 `VLLM_MIGRATION.md`。本源码树里的改动：

* `vllm/model_executor/models/qwen3_moe_partial.py` —— `Qwen3MoePartialForCausalLM`：
  每层只为驻留专家分配 / 加载权重，路由等价于 p2p-framework 的 `local_topk`；
  已登记在 `vllm/model_executor/models/registry.py`（测试侧 `tests/models/registry.py`）。
* `vllm/p2p/layout.py` —— p2p 部署清单 → 每条静态通道的切层、逐层专家（`--hf-overrides`）
  与逐节点 `vllm serve` 命令。纯 Python，控制机上不装 vLLM 也能按文件路径直接跑。
* `vllm/p2p/parity.py` —— 单卡对拍（vLLM 原生 / 部分专家模型 / p2p-framework 参考实现）。
* `tests/p2p/` —— `layout.py` 的单测。

在集群上从这份源码安装 —— 建在 p2p-framework 的 moe 环境之上的 venv（复用 moe 里已有的
CUDA 12 库，约省 4.4 GiB；moe 本身一个文件都不会被改）::

    MOE=/home/ubuntu/anaconda3/envs/moe/bin/python          # = qwen3-30b/env.sh 的 NODE_PY
    $MOE -m venv --system-site-packages ~/venvs/vllm014
    source ~/venvs/vllm014/bin/activate
    pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
    pip install --no-cache-dir -r requirements/build.txt     # 在 vllm-0.14.0 目录下
    pip download vllm==0.14.0 --no-deps -d /tmp/w
    VLLM_VERSION_OVERRIDE=0.14.0 VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=$(ls /tmp/w/vllm-0.14.0-*.whl) \
        pip install --no-cache-dir --no-build-isolation -e .
    rm -rf /tmp/w

三个参数各有原因：

* `VLLM_VERSION_OVERRIDE` —— 本目录不是 git 仓库，setuptools-scm 取不到版本号会直接报错；
* `VLLM_USE_PRECOMPILED` + `VLLM_PRECOMPILED_WHEEL_LOCATION` —— 复用官方 0.14.0 wheel 里编好的
  CUDA 扩展（只改了 Python 文件，扩展与源码匹配），不在节点上编译；
* `--no-build-isolation` —— 构建依赖里有 `torch==2.9.1`，隔离构建会在临时目录里再下一整套
  torch + CUDA 库（约 4 GiB）。先装好 torch 再关掉隔离就不会。

venv 单向依赖 moe：**不要在 moe 里升级 / 卸载 nvidia-*-cu12**，那是 venv 在用的。
`pip install` 时看到 “Not uninstalling torch at …/envs/moe/…, outside environment” 是正常的
（venv 里另装 2.9.1 覆盖，moe 的 2.9.0 不动）。ssh 非交互 shell 不会 activate venv，
所以 `vllm/p2p/layout.py` 生成命令时用 `--vllm-bin ~/venvs/vllm014/bin/vllm`。

本包刻意不在 `__init__` 里 import 任何东西 —— `layout.py` 要能在没有 torch 的控制机上跑。
"""
