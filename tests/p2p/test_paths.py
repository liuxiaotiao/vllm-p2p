# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

from vllm.p2p import paths as P


def _row(cid, u, task, back, text="ok", n=5, front="N03"):
    return {"case_id": cid, "true_task": u, "task": task, "correct": task == u, "channel": back,
            "front": front, "n_generated": n, "text": text}


def test_compare_rounds_separates_replica_swaps_from_path_changes(tmp_path, capsys):
    r1 = {"requests": [_row("g1", "gsm8k", "gsm8k", "back-gsm8k-0"), _row("g2", "gsm8k", "gsm8k", "back-gsm8k-1"),
                       _row("m1", "mbpp", "mbpp", "back-mbpp-0"), _row("n1", "no_robots", "no_robots", "back-no_robots-0")]}
    r2 = {"requests": [_row("g1", "gsm8k", "gsm8k", "back-gsm8k-1"),                 # 同类换副本：正常
                       _row("g2", "gsm8k", "gsm8k", "back-gsm8k-1"),
                       _row("m1", "mbpp", "gsm8k", "back-gsm8k-0"),                  # 识别变了 → 路径变了
                       _row("n1", "no_robots", "no_robots", "back-no_robots-0", text="different")]}
    res = P.compare_rounds([("r1", r1), ("r2", r2)])
    assert res["n_common"] == 4 and res["counts"]["完全一致"] == 1
    assert res["counts"]["同类换副本"] == 1 and res["counts"]["识别"] == 1 and res["counts"]["后段类"] == 1
    assert res["counts"]["输出文本"] == 1
    assert P._task_of_back("back-no_robots-1") == "no_robots"
    for k, d in (("r1", r1), ("r2", r2)):
        (tmp_path / f"x_{k}.json").write_text(json.dumps(d))
    (tmp_path / "x_rounds.json").write_text("{}")
    assert P.main(["--group", f"g={tmp_path}/x_r*.json", "--out", str(tmp_path / "o.json")]) == 0
    out = capsys.readouterr().out
    assert "路径（识别 / 后段类 / 前段）全一样：3/4" in out and "m1（mbpp）" in out
