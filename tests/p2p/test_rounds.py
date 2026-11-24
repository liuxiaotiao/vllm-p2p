# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

from vllm.p2p import rounds as R


def _res(scale, lr=True):
    rows = []
    for i, (u, ok) in enumerate([("gsm8k", True), ("gsm8k", False), ("mbpp", True), ("mbpp", True)]):
        r = {"true_task": u, "t_submit_s": i * 1.0, "t_done_s": i * 1.0 + 10 * scale,
             "n_generated": 100, "prefill_ms": 100.0 * scale * (i + 1),
             "per_token_p50_ms": 50.0 * scale, "total_ms": 1000.0 * scale, "wait_front_ms": 0}
        if lr:
            r["correct"] = ok
        else:
            r["correct"] = None
            r["shadow"] = {"task": u if ok else "mbpp"}
        rows.append(r)
    return {"requests": rows, "failed": []}


def test_round_metrics_and_aggregate(tmp_path, capsys):
    m = R.round_metrics(_res(1.0))
    assert m["n"] == 4 and abs(m["wall_s"] - 13.0) < 1e-9
    assert abs(m["out_tok_per_s"] - 400 / 13) < 1e-9
    assert m["accuracy"] == 0.75 and m["gsm8k.accuracy"] == 0.5 and m["mbpp.accuracy"] == 1.0
    assert m["ttft_p50_ms"] == 250.0 and m["tpot_p50_ms"] == 50.0
    assert R.round_metrics(_res(1.0, lr=False))["accuracy"] == 0.75   # shadow 口径
    agg = R.aggregate([R.round_metrics(_res(1.0)), R.round_metrics(_res(2.0))])
    assert agg["tpot_p50_ms"]["mean"] == 75.0 and agg["tpot_p50_ms"]["n"] == 2
    for k, s in (("a", 1.0), ("b", 2.0)):
        (tmp_path / f"x_{k}.json").write_text(json.dumps(_res(s)))
    out = tmp_path / "agg.json"
    assert R.main(["--group", f"g={tmp_path}/x_*.json", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert len(d["g"]["rounds"]) == 2 and "tpot_p50_ms" in d["g"]["summary"]
    assert "75.0±35.4" in capsys.readouterr().out
