# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 统计钩子（vllm/p2p/stats.py）：逐请求切分、miss、body 计数、落盘格式。

不需要 GPU，也不需要装好的 vLLM（stats.py 只依赖 numpy 与 torch）。拿暴力算的结果对。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("p2p_stats_mod", ROOT / "vllm/p2p/stats.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["p2p_stats_mod"] = S          # dataclass 要在 sys.modules 里找到所属模块
_spec.loader.exec_module(S)

E, K = 16, 4
LAYERS = {3: None, 4: [1, 4, 6, 9, 11, 14], 5: [0, 7, 13]}    # 0-based 层 → 驻留（None=全装）


def make(tmp_path, lr_layers=(3, 4)) -> "S.RouteStats":
    st = S.RouteStats(n_experts=E, top_k=K, norm_topk=True, lr_layers=lr_layers,
                      channel="t", pp_rank=1, out_dir=str(tmp_path))
    for l, res in LAYERS.items():
        st.register_layer(l, res)
    st.allocate(256, "cpu")
    return st


def req(n_prompt, p2p_id, body=None):
    ea = {"p2p_id": p2p_id}
    if body is not None:
        ea["p2p_body"] = body
    return SimpleNamespace(num_prompt_tokens=n_prompt,
                           sampling_params=SimpleNamespace(extra_args=ea))


def step(st, logits: dict[str, np.ndarray], positions: dict[str, np.ndarray], requests):
    """模拟一步：logits[rid] 形如 [n, L, E]，按 rid 顺序拼成 batch 写进缓冲区。"""
    rids = list(logits)
    big = np.concatenate([logits[r] for r in rids], axis=0)            # [T, L, E]
    for slot in range(len(LAYERS)):
        st.capture(slot, torch.from_numpy(big[:, slot, :]))
    st.on_step(rids, [logits[r].shape[0] for r in rids],
               np.concatenate([positions[r] for r in rids]), requests)


def brute(logit_rows: np.ndarray):
    """[n, L, E] → (topi [n,L,k], topw [n,L,k])，与 stats.capture 同一口径。"""
    t = torch.from_numpy(logit_rows).float()
    p = torch.softmax(t, -1)
    w, i = torch.topk(p, K, -1)
    w = w / w.sum(-1, keepdim=True)
    return i.numpy(), w.numpy()


def test_split_miss_body_and_dump(tmp_path) -> None:
    rng = np.random.default_rng(0)
    st = make(tmp_path)
    L = len(LAYERS)
    reqs = {"a-x": req(6, "A", "1:4"), "b-y": req(3, "B")}
    la = rng.standard_normal((9, L, E)).astype(np.float32)     # A：6 prompt + 3 decode
    lb = rng.standard_normal((5, L, E)).astype(np.float32)     # B：3 prompt + 2 decode
    # 第 1 步：A 分块 prefill 前 4 个、B 整段 prefill
    step(st, {"a-x": la[:4], "b-y": lb[:3]},
         {"a-x": np.arange(0, 4), "b-y": np.arange(0, 3)}, reqs)
    # 第 2 步：A 剩下 2 个 prompt + 第一个 decode 输入；B 一个 decode
    step(st, {"a-x": la[4:7], "b-y": lb[3:4]},
         {"a-x": np.arange(4, 7), "b-y": np.arange(3, 4)}, reqs)
    # 第 3 步：A 被抢占后从头重算（位置 0..7）—— 已统计过的位置不能重复计
    step(st, {"a-x": la[:8], "b-y": lb[4:5]},
         {"a-x": np.arange(0, 8), "b-y": np.arange(4, 5)}, reqs)
    st.finish(["a-x", "b-y"])

    rows = {json.loads(x)["p2p_id"]: json.loads(x)
            for x in (tmp_path / "t_pp1.jsonl").read_text().splitlines()}
    assert set(rows) == {"A", "B"}
    res = [np.ones(E, bool) if v is None else np.isin(np.arange(E), v) for v in LAYERS.values()]

    for rid, lg, n_prompt, seen in (("A", la, 6, 8), ("B", lb, 3, 5)):
        r = rows[rid]
        ti, tw = brute(lg[:seen])
        here = np.stack([res[l][ti[:, l, :]] for l in range(L)], axis=1)      # [n, L, k]
        miss_any = ~here.all(-1)
        mass = (tw * ~here).sum(-1)
        pre = np.arange(seen) < n_prompt
        assert r["n_seen"] == seen and r["n_prompt"] == n_prompt
        assert r["layers"] == [l + 1 for l in LAYERS]                     # 1-based
        assert r["prefill"]["n_token_layer"] == int(pre.sum()) * L
        assert r["prefill"]["miss_token_layer"] == int(miss_any[pre].sum())
        assert r["decode"]["miss_token_layer"] == int(miss_any[~pre].sum())
        assert r["prefill"]["miss_mass"] == pytest.approx(float(mass[pre].sum()), abs=1e-5)
        assert r["decode_miss_seq"] == [int(x) for x in miss_any[~pre].sum(1)]
        assert r["layer_miss"]["prefill"] == miss_any[pre].sum(0).tolist()
        # 全装的那层（slot 0）永远不 miss
        assert r["layer_miss"]["prefill"][0] == 0 and r["layer_miss"]["decode"][0] == 0

        # LR：只数 body（A 是 1:4，B 没给 → 整条 prompt）、只数 prompt 位置
        body = np.zeros(seen, bool)
        if rid == "A":
            body[1:4] = True
        else:
            body[:n_prompt] = True
        body &= pre
        for slot, l in enumerate(LAYERS):
            key = str(l + 1)
            if l not in (3, 4):
                assert key not in r["lr_layers"]
                continue
            w = r["lr_layers"][key]
            cnt = np.zeros(E, int)
            np.add.at(cnt, ti[body, slot, :].reshape(-1), 1)
            got = np.zeros(E, int)
            got[w["i"]] = w["c"]
            assert (got == cnt).all()
            assert w["t"] == int(body.sum())
            assert bool(w.get("m")) == (rid == "A")


def test_lr_layers_match_p2p_layerstat_wire(tmp_path) -> None:
    """落盘的逐层统计能被 p2p-framework 的 LayerStat.from_wire 读回来（lr_tool check 要读）。"""
    fw = ROOT.parent / "p2p-framework"
    if not (fw / "p2pmoe").is_dir():
        pytest.skip("旁边没有 p2p-framework")
    sys.path.insert(0, str(fw))
    from p2pmoe.runtime.model import LayerStat

    st = make(tmp_path)
    lg = np.random.default_rng(1).standard_normal((5, len(LAYERS), E)).astype(np.float32)
    reqs = {"r": req(5, "R", "0:5")}
    step(st, {"r": lg}, {"r": np.arange(5)}, reqs)
    st.finish(["r"])
    r = json.loads((tmp_path / "t_pp1.jsonl").read_text())
    ls = LayerStat.from_wire(r["lr_layers"]["4"])
    assert ls.masked and ls.n_tokens == 5 and int(ls.count.sum()) == 5 * K
    assert ls.mass.sum() == pytest.approx(5.0, abs=1e-4)          # 每个 token 的 top-k 权重和 = 1


def test_unknown_request_and_empty_step_are_harmless(tmp_path) -> None:
    st = make(tmp_path)
    st.on_step([], [], np.zeros(0, np.int64), {})
    lg = np.random.default_rng(2).standard_normal((2, len(LAYERS), E)).astype(np.float32)
    step(st, {"x": lg}, {"x": np.arange(2)}, {})                   # runner 里查不到这条
    st.finish(["x", "never-seen"])
    r = json.loads((tmp_path / "t_pp1.jsonl").read_text())
    assert r["p2p_id"] is None and r["n_seen"] == 2


def test_parse_body() -> None:
    assert S.parse_body("1:4,7:9") == [(1, 4), (7, 9)]
    assert S.parse_body([[0, 3]]) == [(0, 3)]
    assert S.parse_body("") is None and S.parse_body(None) is None
    assert S.body_string([(1, 4), (7, 9)]) == "1:4,7:9"
