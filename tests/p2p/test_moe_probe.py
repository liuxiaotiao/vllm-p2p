# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""moe_probe.py：画像记录 → moe_prefill_probe 同格式的 npz、逐层 90% 驻留集。只要 numpy。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vllm.p2p.moe_probe import cover_set, expert_sets, fit_indices, records_to_arrays, write_probe

E, L = 8, 4                       # 第 0 层稠密（两级：0–1、2–3）


def _rec(rank, layers, rng, body_t=3):
    prof = {}
    for ph, t in (("prefill", 6), ("decode", 4)):
        prof[ph] = {"t": t, "layers": {str(l + 1): {"c": rng.integers(0, 5, E).tolist(),
                                                    "w": rng.random(E).round(4).tolist()}
                                       for l in layers}}
    lr = {}
    for l in layers:
        c = rng.integers(0, 3, E)
        nz = np.flatnonzero(c)
        lr[str(l + 1)] = {"i": nz.tolist(), "c": c[nz].tolist(), "w": [0.5] * len(nz), "t": body_t,
                          "e": E, "m": 1}
    return {"p2p_id": "gsm8k:7", "pp_rank": rank, "layers": [l + 1 for l in layers], "n_prompt": 6,
            "n_seen": 9, "profile": prof, "lr_layers": lr}


def test_records_to_arrays_places_layers_and_keeps_dense_zero() -> None:
    rng = np.random.default_rng(0)
    r0, r1 = _rec(0, [1], rng), _rec(1, [2, 3], rng)
    a = records_to_arrays([r1, r0], n_layers=L, n_experts=E)
    assert a["counts_full"].shape == (L, E)
    assert not a["counts_full"][0].any() and not a["counts_body"][0].any()        # 稠密层
    assert a["counts_full"][2].tolist() == r1["profile"]["prefill"]["layers"]["3"]["c"]
    assert a["counts_decode"][1].tolist() == r0["profile"]["decode"]["layers"]["2"]["c"]
    w = r1["lr_layers"]["4"]
    got = np.zeros(E, int)
    got[w["i"]] = w["c"]
    assert a["counts_body"][3].tolist() == got.tolist()
    assert a["tokens_full"] == 6 and a["tokens_decode"] == 4 and a["tokens_body"] == 3
    assert a["n_prompt_tokens"] == 6 and a["n_generated"] == 4        # n_seen 9 = 6 + 4 − 1
    with pytest.raises(ValueError, match="stats-profile"):
        records_to_arrays([{"p2p_id": "x", "pp_rank": 0}], n_layers=L, n_experts=E)


def test_write_probe_format(tmp_path) -> None:
    rng = np.random.default_rng(1)
    rows = [(i, f"body {i}", records_to_arrays([_rec(0, [1], rng), _rec(1, [2, 3], rng)],
                                               n_layers=L, n_experts=E)) for i in (5, 2, 9)]
    write_probe(tmp_path, "gsm8k", rows, meta={"top_k": 2, "dense_layers": [0]})
    with np.load(tmp_path / "gsm8k_prefill_persample.npz") as z:
        assert z["counts_full"].shape == (3, L, E) and z["sample_idx"].tolist() == [2, 5, 9]
        for k in ("counts_body", "wsum_body", "tokens_body", "counts_decode", "n_prompt_tokens"):
            assert k in z.files
    meta = json.loads((tmp_path / "gsm8k_prefill_meta.json").read_text())
    assert meta["n_samples"] == 3 and meta["dense_layers"] == [0]
    bodies = [json.loads(x) for x in (tmp_path / "gsm8k_bodies.jsonl").read_text().splitlines()]
    assert [b["sample_idx"] for b in bodies] == [2, 5, 9] and bodies[0]["body"] == "body 2"


def test_cover_set_and_expert_sets(tmp_path) -> None:
    assert cover_set(np.array([50, 30, 10, 5, 5, 0]), 0.9, 1) == [0, 1, 2]
    assert cover_set(np.array([50, 30, 10, 5, 5, 0]), 0.9, 4) == [0, 1, 2, 3]
    assert cover_set(np.zeros(6), 0.9, 2) == []
    assert len(fit_indices(10, 0.2, 42)) == 2 and len(fit_indices(10, 1.0, 0)) == 10

    rng = np.random.default_rng(2)
    rows = [(i, "b", records_to_arrays([_rec(0, [1], rng), _rec(1, [2, 3], rng)], n_layers=L,
                                       n_experts=E)) for i in range(10)]
    write_probe(tmp_path, "mbpp", rows, meta={"top_k": 2, "dense_layers": [0]})
    res = expert_sets(tmp_path, ["mbpp"], q=0.9, fit_frac=1.0)
    t = res["tasks"]["mbpp"]
    assert t["union"][0] == [] and all(len(x) >= 2 for x in t["union"][1:])
    with np.load(tmp_path / "mbpp_prefill_persample.npz") as z:
        pre = z["counts_full"].astype(np.int64).sum(0)
    assert t["prefill"][2] == cover_set(pre[2], 0.9, 2)
    assert set(t["prefill"][3]) | set(t["decode"][3]) == set(t["union"][3])
    assert res["dense_layers"] == [0] and res["threshold"] == 0.9
