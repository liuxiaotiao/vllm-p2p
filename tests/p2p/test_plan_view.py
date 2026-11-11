# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""plan_view：规划器的分配方案 → 段 / 节点 / 配对表，对上 vLLM 引擎名与实测每步。"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vllm" / "p2p"))
import plan_view as PV  # noqa: E402
from layout import main as layout_main  # noqa: E402
from stages import stages_from_channels  # noqa: E402

DATA = Path(__file__).parent / "data"
PLAN, HOSTS = DATA / "plan_es90_fill.json", DATA / "hosts.txt"


def test_plan_view_tables_and_measured(tmp_path) -> None:
    out = tmp_path / "vc"
    assert layout_main(["--plan", str(PLAN), "--hosts", str(HOSTS), "--out", str(out), "--model-dir", "/w",
                        "--vllm-bin", "/venv/bin/vllm"]) == 0
    st = stages_from_channels(json.loads((out / "channels.json").read_text()))
    back = next(c for c in st["channels"] if c["role"] == "back")
    tr = tmp_path / f"{back['name']}.jsonl"
    tr.write_text("".join(json.dumps({"t": i * 0.02, "ms": 15.0 + (i % 3), "n": 1, "tok": 1, "pf": 0, "npf": 0,
                                      "ctx": 100, "q": 0}) + "\n" for i in range(9)))
    (tmp_path / "stages.json").write_text(json.dumps(st))
    md = tmp_path / "p.md"
    assert PV.main(["--channels", str(out / "channels.json"), "--stages", str(tmp_path / "stages.json"),
                    "--steps", str(tr), "--md", str(md)]) == 0
    txt = md.read_text()
    assert "## 段" in txt and "## 节点" in txt and "## 配对" in txt and "共 128" in txt
    assert back["name"] in txt and "| 16.0 |" in txt                       # 实测批 1 p50
    plan = json.loads(PLAN.read_text())
    rows = PV.segment_rows(plan, st, {back["name"]: 16.0})
    hit = [r for r in rows if r["engine"] == back["name"]]
    assert len(hit) == 1 and hit[0]["measured_b1_ms"] == 16.0 and hit[0]["role"] == "后段"
    nodes = PV.node_rows(plan)
    assert len(nodes) == len(plan["nodes"]) and all(0 < r["resident"] <= 1 for r in nodes)
    with pytest.raises(SystemExit, match="找不到"):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"plan": "/nope.json", "channels": []}))
        PV.main(["--channels", str(bad)])
