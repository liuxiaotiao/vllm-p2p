# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

from vllm.p2p import simerr as E


def _s(**kv):
    return {"summary": {k: {"mean": v, "std": 0.0} for k, v in kv.items()}}


def test_plan_errors_mae_and_ranking(tmp_path, capsys):
    a = {"real": _s(out_tok_per_s=100.0, tpot_p50_ms=50.0, **{"gsm8k.tpot_p50_ms": 40.0}),
         "sim_profile": _s(out_tok_per_s=110.0, tpot_p50_ms=45.0, **{"gsm8k.tpot_p50_ms": 44.0}),
         "sim_theory": _s(out_tok_per_s=150.0, tpot_p50_ms=30.0, **{"gsm8k.tpot_p50_ms": 20.0})}
    b = {"real": _s(out_tok_per_s=80.0, tpot_p50_ms=60.0),
         "sim_profile": _s(out_tok_per_s=84.0, tpot_p50_ms=63.0),
         "sim_theory": _s(out_tok_per_s=160.0, tpot_p50_ms=25.0)}
    pe = E.plan_errors(a)
    assert abs(pe["sim_profile"]["out_tok_per_s"]["err"] - 0.10) < 1e-12
    assert abs(pe["sim_theory"]["gsm8k.tpot_p50_ms"]["err"] + 0.5) < 1e-12
    for n, d in (("A", a), ("B", b)):
        (tmp_path / f"cmp_{n}.json").write_text(json.dumps(d))
    out = tmp_path / "e.json"
    assert E.main([str(tmp_path / "cmp_A.json"), str(tmp_path / "cmp_B.json"), "--tasks", "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert abs(d["mae"]["sim_profile"]["out_tok_per_s"] - 0.075) < 1e-12          # (10% + 5%) / 2
    assert d["ranks"]["out_tok_per_s"]["real"] == ["A", "B"]
    assert d["ranks"]["out_tok_per_s"]["sim_profile"] == ["A", "B"]
    assert d["ranks"]["out_tok_per_s"]["sim_theory"] == ["B", "A"]                # 理论版排反了
    txt = capsys.readouterr().out
    assert "sim_theory ✗ B > A" in txt and "+10.0%" in txt
