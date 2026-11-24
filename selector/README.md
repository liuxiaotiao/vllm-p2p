# task 选择器（前段出口 LR）

网关（`vllm/p2p/tiered_gateway.py --mode lr|shadow`）、A4 识别服务、`bench.py --lr-model` 都从这里取。
`--lr-model` 写文件名即可（`.npz` 可省）；不写时网关按 stages.json 的模型族取默认
（见 `vllm/p2p/selectors.py`）。选择器旁边的 `<名字>.features.json` 会被自动当作特征口径用
（没给 `--lr-features` 时）。

| 文件 | 模型 | 特征 |
|---|---|---|
| `selector_body_L2_cmw_params.npz` | Qwen3-30B-A3B（默认） | 前 2 层 counts 份额 + meanw，只数 body |
| `selector_body_L8_params.npz` | Qwen3-30B-A3B（A4 识别服务） | 前 8 层 counts 份额，只数 body |
| `task_clf_glm47flash_L1L2_meanw.npz` + `.features.json` | GLM-4.7-Flash（默认） | 第 1、2 层 meanw，整条 prompt |

## GLM-4.7-Flash：`task_clf_glm47flash_L1L2_meanw`

* 文件是 `{W[4,128], b[4], tasks, n_layers=2, n_experts=64, feature_mode=meanw}`，`W·x + b` 直接作用在
  原始特征上（没有标准化），概率取 softmax。4 类：gsm8k / mbpp / humaneval / no_robots。
* 文件里没写层号与 token 口径，由 `.features.json` 给：**第 1、2 层**（0-based；第 0 层是稠密层，
  HF probe 的 slot 0、1 就是这两层）、**整条 prompt**（不是 body）、每层 64 维
  meanw = wsum / counts（没选中记 0），层优先拼接。
* 门控权重 = 重归一后的 sigmoid × routed_scaling_factor 1.8（真正乘到专家输出上的系数，
  HF probe 的 wsum 就是这个口径；vLLM 前段的统计钩子已经乘好）。
* 11-24 用 `verify/glm47flash_stats/prefill_GLM-4.7-Flash_w` 核对：整条 prompt + 第 1、2 层 + ×1.8
  整体 99.7%（参考 20% 上 100%）；去掉 ×1.8 只有 88.9%，换成 body 54.7%，换成第 2、3 层 ≤ 28%。
