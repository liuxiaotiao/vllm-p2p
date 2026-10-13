# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B-1 分段引擎的公共件（vllm/p2p/stage.py）与协调器的对拍函数。只要 torch（CPU）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vllm.p2p import stage as S  # noqa: E402


def test_stage_of_parses_and_validates() -> None:
    assert S.stage_of({"num_hidden_layers": 48}) is None
    f = S.stage_of({"num_hidden_layers": 48, "p2p_stage": {"role": "front", "layers": [0, 8]}})
    assert f == S.StageSpec("front", 0, 8) and f.n_layers == 8
    b = S.stage_of(SimpleNamespace(num_hidden_layers=48,
                                   p2p_stage={"role": "back", "layers": [8, 48]}))
    assert b == S.StageSpec("back", 8, 48)
    assert S.role_of(SimpleNamespace(num_hidden_layers=4)) is None
    bad = [
        ({"role": "middle", "layers": [0, 8]}, "role"),
        ({"role": "front", "layers": [2, 8]}, "第 0 层"),         # 前段要从 0 开始（embed 在前段）
        ({"role": "back", "layers": [8, 40]}, "最后一层"),        # 后段要到最后（lm_head 在后段）
        ({"role": "back", "layers": [8, 50]}, "超出"),
        ({"role": "front", "layers": [0, 0]}, "空"),
        ({"role": "front", "layers": "0-8"}, "[lo, hi]"),
    ]
    for raw, msg in bad:
        with pytest.raises(ValueError, match=msg.replace("[", r"\[").replace("]", r"\]")):
            S.stage_of({"num_hidden_layers": 48, "p2p_stage": raw})


def test_stage_layer_range_splits_the_stage_not_the_model(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    back = S.StageSpec("back", 8, 48)
    parts = [S.stage_layer_range(back, r, 3) for r in range(3)]
    # 40 层切 3 级：与 get_pp_indices(40, r, 3) 相同、再整体平移 8
    assert parts[0][0] == 8 and parts[-1][1] == 48
    assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))
    assert sum(e - s for s, e in parts) == 40
    assert S.stage_layer_range(S.StageSpec("front", 0, 8), 0, 1) == (0, 8)
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "20,10,10")    # 指的是**本段**的划分
    import vllm.envs as envs
    if hasattr(envs, "disable_envs_cache"):
        envs.disable_envs_cache()
    assert [S.stage_layer_range(back, r, 3) for r in range(3)] == [(8, 28), (28, 38), (38, 48)]


def test_add_residual_and_zero_embed() -> None:
    h, r = torch.randn(3, 8), torch.randn(3, 8)
    out, res = S.AddResidual()(h, r)
    assert res is None and torch.equal(out, h + r)
    out, _ = S.AddResidual()(h, None)
    assert out is h
    z = S.ZeroEmbed(16, torch.float16)(torch.tensor([[1, 2, 3]]))
    assert z.shape == (1, 3, 16) and z.dtype == torch.float16 and not z.any()
    assert list(S.ZeroEmbed(4, torch.float16).parameters()) == []      # 不占权重、不读 checkpoint


def test_row_buffer_grows_and_refuses_holes() -> None:
    b = S.RowBuffer(4, torch.float16, cap=2)
    rows = torch.arange(20, dtype=torch.float16).view(5, 4)
    b.put(0, rows[:1])
    b.put(1, rows[1:5])                  # 超出容量 → 翻倍增长，旧数据保留
    assert b.n == 5 and torch.equal(b.get(0, 5), rows)
    b.put(2, rows[2:4])                  # 重写已有位置（抢占后重送）可以
    assert b.n == 5
    with pytest.raises(ValueError, match="空洞"):
        b.put(7, rows[:1])
    with pytest.raises(IndexError):
        b.get(3, 6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_pack_unpack_roundtrip(dtype) -> None:
    rows = torch.randn(3, 16).to(dtype)
    item = S.pack_rows("req-1", 41, rows)
    assert isinstance(item[-1], bytes)   # utility 参数没有类型信息：只能传原生类型
    rid, pos, back = S.unpack_rows(item, 16)
    assert (rid, pos) == ("req-1", 41) and back.dtype == dtype and torch.equal(back, rows)
    with pytest.raises(ValueError, match="不符"):
        S.unpack_rows(item, 8)


def test_row_stream_dedupes_resent_positions() -> None:
    s = S.RowStream()
    r = torch.arange(10, dtype=torch.float32).view(10, 1)
    assert s.add(0, r[:4])[0] == 0                      # 分块 prefill 第 1 块
    st, new = s.add(2, r[2:7])                          # 抢占后重算：前 2 行是旧的
    assert st == 4 and torch.equal(new, r[4:7]) and s.next == 7
    assert s.add(0, r[:7]) is None                      # 全旧
    with pytest.raises(ValueError, match="跳了位置"):
        s.add(9, r[9:])
    s.keep(r[:2])
    s.keep(r[2:3])
    assert torch.equal(s.take(), r[:3]) and s._parts == []


def test_compare_runs_flags_real_divergence_but_not_ties() -> None:
    from vllm.p2p.tiered import compare_runs

    def run(tokens, lps):
        return {"rid": "x", "tokens": tokens, "logprobs": lps}

    ref = [run([5, 6, 7], [{5: -0.1, 9: -2}, {6: -0.5, 8: -0.51}, {7: -0.2}])]
    same = [run([5, 6, 7], [{5: -0.105}, {6: -0.5}, {7: -0.2}])]
    tie = [run([5, 8, 1], [{5: -0.1}, {8: -0.5}, {1: -1}])]       # 第 1 位平票（差 0.01）
    far = [run([9, 6, 7], [{9: -2.0}, {6: -0.5}, {7: -0.2}])]     # 第 0 位差 1.9：真分叉
    a = compare_runs(ref, same, tol=0.02)
    assert a["ok"] and a["same"] == 1 and a["max_lp_diff"] == pytest.approx(0.005)
    b = compare_runs(ref, tie, tol=0.02)
    assert b["ok"] and b["same"] == 0 and b["rows"][0]["common_prefix"] == 1
    c = compare_runs(ref, far, tol=0.02)
    assert not c["ok"] and not c["ok_tie"]


def test_stage_layers_builds_only_this_stage(monkeypatch) -> None:
    """模型构造期间 `make_layers` 被换掉：只建本段（再按 PP 切）的层，层号保持全局。"""
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    import vllm.envs as envs
    if hasattr(envs, "disable_envs_cache"):
        envs.disable_envs_cache()
    m = pytest.importorskip("vllm.model_executor.models.qwen3_moe_partial")
    from vllm.model_executor.models import qwen3_moe as q
    from vllm.model_executor.models.utils import PPMissingLayer

    built = []

    def layer_fn(prefix):
        built.append(prefix)
        return torch.nn.Linear(1, 1)

    for rank, want in ((0, (8, 28)), (1, (28, 48))):
        monkeypatch.setattr(m, "get_pp_group",
                            lambda r=rank: SimpleNamespace(rank_in_group=r, world_size=2))
        built.clear()
        orig = q.make_layers
        with m._stage_layers(S.StageSpec("back", 8, 48)):
            s, e, mods = q.make_layers(48, layer_fn, "model.layers")
        assert q.make_layers is orig                                 # 退出即恢复
        assert (s, e) == want and len(mods) == 48
        assert built == [f"model.layers.{i}" for i in range(*want)]   # 全局层号 → 权重按原名加载
        assert all(isinstance(mods[i], PPMissingLayer) for i in range(48) if not s <= i < e)
    with m._stage_layers(None):
        assert q.make_layers is orig


def test_wire_formats_survive_vllm_serialization() -> None:
    """两条线上格式走一遍 vLLM 自己的 msgpack 编解码（与 ZMQ 收发同一套）：
    前段的 hidden 挂在 EngineCoreOutput 上（有类型 → 张量解得回来，大的走零拷贝辅助帧）；
    协调器 → 后段的 utility 参数没有类型（→ 只能是 bytes）。"""
    from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    from vllm.p2p.tiered import _core_request

    enc = MsgpackEncoder()
    for n in (1, 300):                                 # 小的内联、大的走辅助帧
        h = torch.randn(n, 256).half()
        frames = enc.encode(EngineCoreOutputs(outputs=[
            EngineCoreOutput(request_id="a", new_token_ids=[], p2p_pos=7, p2p_hidden=h)]))
        o = MsgpackDecoder(EngineCoreOutputs).decode(frames).outputs[0]
        assert o.p2p_pos == 7 and o.new_token_ids == [] and o.finish_reason is None
        assert torch.equal(o.p2p_hidden, h)
    plain = MsgpackDecoder(EngineCoreOutputs).decode(enc.encode(EngineCoreOutputs(outputs=[
        EngineCoreOutput(request_id="b", new_token_ids=[3])]))).outputs[0]
    assert plain.p2p_hidden is None and plain.p2p_pos == -1    # 普通引擎的输出不受影响

    rows = torch.randn(2, 256).half()
    frames = enc.encode((0, 99, "p2p_feed", ([S.pack_rows("a", 12, rows)],)))
    _, call_id, method, (items,) = MsgpackDecoder().decode(frames)
    rid, pos, back = S.unpack_rows(items[0], 256)
    assert (call_id, method, rid, pos) == (99, "p2p_feed", "a", 12) and torch.equal(back, rows)

    emb = torch.randn(40, 256).half()                  # 后段请求：prompt 以 embedding 送入
    req = _core_request("a", prompt=None, embeds=emb, max_tokens=8, eos=2, logprobs=5,
                        ignore_eos=False, stop_ids=(), arrival=0.0)
    got = MsgpackDecoder(EngineCoreRequest).decode(enc.encode(req))
    assert got.prompt_token_ids is None and torch.equal(got.prompt_embeds, emb)
    assert got.sampling_params.temperature == 0.0 and got.sampling_params.logprobs == 5
