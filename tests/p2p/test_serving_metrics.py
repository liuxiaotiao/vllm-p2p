# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3：vLLM 在 usage 块里附带的逐请求引擎时间（serving_completion._p2p_metrics）。

要能 import vLLM 的 OpenAI 服务层（装好的环境里才有）；import 不了就跳过。"""

from __future__ import annotations

import pytest


def test_p2p_metrics_from_request_state_stats() -> None:
    try:
        from vllm.entrypoints.openai.serving_completion import _p2p_metrics
        from vllm.v1.metrics.stats import RequestStateStats
    except Exception as e:                                        # noqa: BLE001
        pytest.skip(f"vLLM 服务层 import 不了：{e}")
    m = RequestStateStats(num_generation_tokens=7, arrival_time=1000.0, queued_ts=10.0,
                          scheduled_ts=10.25, first_token_ts=10.5, last_token_ts=11.5,
                          first_token_latency=0.6)
    d = _p2p_metrics(m)
    assert d["queued_ms"] == 250.0 and d["prefill_ms"] == 250.0 and d["decode_ms"] == 1000.0
    assert d["ttft_ms"] == 600.0 and d["num_generation_tokens"] == 7
    # 没开 log_stats 的引擎：时间戳都是 0 → None，而不是算出负数
    z = _p2p_metrics(RequestStateStats())
    assert z["queued_ms"] is None and z["ttft_ms"] is None
    assert _p2p_metrics(None) is None
