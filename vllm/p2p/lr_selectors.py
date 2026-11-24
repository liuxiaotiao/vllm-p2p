# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""task 选择器（前段出口 LR）放在本仓库的 ``selector/`` 目录下，按模型族取默认。纯 Python。

    selector/
        selector_body_L2_cmw_params.npz            Qwen3-30B：前 2 层 counts 份额 + meanw，只数 body
        selector_body_L8_params.npz                Qwen3-30B：前 8 层 counts 份额（A4 识别服务）
        task_clf_glm47flash_L1L2_meanw.npz         GLM-4.7-Flash：第 1、2 层 meanw，整条 prompt
        task_clf_glm47flash_L1L2_meanw.features.json   ↑ 的特征口径（文件里没写层号与 token 口径）

``--lr-model`` 可以写完整路径，也可以只写 ``selector/`` 下的文件名（``.npz`` 可省）；不写就按
stages.json 的模型族取默认。选择器旁边若有同名的 ``<名字>.features.json``，没给 ``--lr-features``
时自动用它 —— 选择器文件自己说不清特征口径（层号、body / 整条 prompt）时就靠它。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["SELECTOR_DIR", "DEFAULTS", "resolve", "default_for", "sidecar_features"]

SELECTOR_DIR = Path(__file__).resolve().parents[2] / "selector"

DEFAULTS = {
    "qwen3_moe": "selector_body_L2_cmw_params.npz",
    "glm4_moe_lite": "task_clf_glm47flash_L1L2_meanw.npz",
}


def resolve(name_or_path: str | Path) -> Path:
    """完整路径原样返回（要存在）；否则在 ``selector/`` 下找（可省 ``.npz``）。"""
    p = Path(str(name_or_path)).expanduser()
    if p.is_file():
        return p
    for c in (SELECTOR_DIR / p.name, SELECTOR_DIR / f"{p.name}.npz"):
        if c.is_file():
            return c
    have = sorted(x.name for x in SELECTOR_DIR.glob("*") if x.suffix in (".npz", ".joblib", ".json")
                  and not x.name.endswith(".features.json")) if SELECTOR_DIR.is_dir() else []
    raise FileNotFoundError(f"找不到选择器 {name_or_path!r}（{SELECTOR_DIR} 里有 {have}）")


def default_for(family: str | None) -> Path:
    name = DEFAULTS.get(str(family or "qwen3_moe"))
    if name is None:
        raise FileNotFoundError(f"模型族 {family!r} 没有默认的选择器 —— 用 --lr-model 指定")
    return resolve(name)


def sidecar_features(model: str | Path) -> Path | None:
    """``<选择器去掉扩展名>.features.json``，没有就 None。"""
    p = Path(str(model))
    stem = p.name[: -len(p.suffix)] if p.suffix else p.name
    f = p.with_name(f"{stem}.features.json")
    return f if f.is_file() else None
