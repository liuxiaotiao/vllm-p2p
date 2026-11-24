# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""选择器放在仓库的 selector/ 下：按名字 / 模型族找到，旁边的 .features.json 自动当特征口径。"""

from pathlib import Path

import pytest

from vllm.p2p import lr_selectors as S


def test_glm_default_and_sidecar() -> None:
    p = S.default_for("glm4_moe_lite")
    assert p.name == "task_clf_glm47flash_L1L2_meanw.npz" and p.parent == S.SELECTOR_DIR
    assert S.resolve("task_clf_glm47flash_L1L2_meanw") == p
    f = S.sidecar_features(p)
    assert f is not None and f.name == "task_clf_glm47flash_L1L2_meanw.features.json"


def test_resolve_paths(tmp_path: Path) -> None:
    x = tmp_path / "my.npz"
    x.write_bytes(b"")
    assert S.resolve(x) == x and S.sidecar_features(x) is None
    (tmp_path / "my.features.json").write_text("{}")
    assert S.sidecar_features(x) == tmp_path / "my.features.json"
    with pytest.raises(FileNotFoundError, match="找不到选择器"):
        S.resolve("no_such_selector")
    with pytest.raises(FileNotFoundError, match="没有默认"):
        S.default_for("llama")
