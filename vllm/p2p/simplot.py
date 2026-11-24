# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""模拟 vs 真机：画成统计图（每张 PNG + PDF）。

    python3 vllm/p2p/simplot.py \\
        --run "三前段=results/b1_o3c_s1_pf1_n150.json,results/simv3_3f.json" \\
        --run "A30单前段=results/ffa95_s1_pf6_n150.json,results/simv3_a30.json" \\
        --out results/figs [--lang en]

每个 --run 是「标签=真机结果,模拟结果」（模拟是 `sim run --replay 真机结果` 的输出：同一批请求）。出的图：

1. ``1_metrics``   各方案的关键指标误差（模拟相对真机 %），灰带 = ±5%。
2. ``2_<指标>``     每个方案一列：上 = 逐请求分布（密度，高斯核估计），下 = 累积分布（ECDF）。
                    标 KS 距离 D（两条累积分布最大的纵向差）与 W1/均值（把一个分布搬成另一个平均要挪多少，
                    相对真机均值）。指标：TPOT、TTFT、端到端时延、单请求解码 token/s。
3. ``3_qq``         分位数对分位数：真机第 q 分位 vs 模拟第 q 分位（q = 1%…99%），越贴对角线越像。
4. ``4_err_gauss``  逐请求配对误差：同一条请求、真机与模拟走的是同一条后段时，TPOT 的相对误差直方图，
                    叠拟合的正态分布 N(μ, σ)；标 μ、σ、落在 ±10% 内的比例、参与配对的条数。

只用 numpy 和 matplotlib；中文字体找不到时自动改用英文标签（或 --lang en）。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# 默认调色板（dataviz 参考实例，已过色盲检查）：真机 = 蓝实线，模拟 = 橙虚线；方案按固定顺序取色
REAL, SIM = "#2a78d6", "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#ecebe5"

T = {
    "zh": {"real": "真机", "sim": "模拟", "err": "模拟相对真机的误差（%）", "band": "±5%",
           "tpot": "TPOT（ms / token）", "ttft": "TTFT（ms，对数轴）", "e2e": "端到端时延（s，对数轴）",
           "toks": "单请求解码速度（token/s）", "density": "密度", "ecdf": "累积比例",
           "qq_x": "真机分位数", "qq_y": "模拟分位数", "pair": "逐请求配对 TPOT 相对误差（%）",
           "count": "请求数", "pooled": "全部方案合并", "metrics_title": "关键指标：模拟相对真机的误差",
           "dist_title": "逐请求分布：真机 vs 模拟", "qq_title": "分位数对比（Q-Q）",
           "gauss_title": "逐请求配对误差与正态拟合（同一条请求、走同一条后段）",
           "in10": "±10% 内", "n": "配对", "of": "条，共",
           "m": {"steady": "稳态吞吐", "full": "全程吞吐", "tpot_mean": "TPOT 均值", "tpot_p50": "TPOT p50",
                 "tpot_p90": "TPOT p90", "ttft_p50": "TTFT p50", "ttft_mean": "TTFT 均值",
                 "e2e_mean": "端到端时延均值", "task": "{t} 速度"}},
    "en": {"real": "Real", "sim": "Simulated", "err": "Simulation error vs. real (%)", "band": "±5%",
           "tpot": "TPOT (ms / token)", "ttft": "TTFT (ms, log)", "e2e": "End-to-end latency (s, log)",
           "toks": "Per-request decode speed (token/s)", "density": "Density", "ecdf": "Cumulative share",
           "qq_x": "Real quantile", "qq_y": "Simulated quantile", "pair": "Paired per-request TPOT error (%)",
           "count": "Requests", "pooled": "All schemes pooled", "metrics_title": "Key metrics: simulation error",
           "dist_title": "Per-request distributions: real vs. simulated", "qq_title": "Quantile-quantile",
           "gauss_title": "Paired per-request error with normal fit (same request, same back path)",
           "in10": "within ±10%", "n": "paired", "of": "of",
           "m": {"steady": "Steady throughput", "full": "Whole-run throughput", "tpot_mean": "TPOT mean",
                 "tpot_p50": "TPOT p50", "tpot_p90": "TPOT p90", "ttft_p50": "TTFT p50",
                 "ttft_mean": "TTFT mean", "e2e_mean": "E2E latency mean", "task": "{t} speed"}},
}


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def _tpot(r) -> float | None:
    n = int(r.get("n_generated") or 0)
    return (float(r["total_ms"]) - float(r["prefill_ms"])) / (n - 1) if n > 1 else None


def per_request(doc) -> dict[str, dict]:
    out = {}
    for r in doc.get("requests") or []:
        tp = _tpot(r)
        out[str(r.get("req"))] = {
            "tpot": tp, "ttft": float(r["prefill_ms"]), "e2e": float(r["total_ms"]) / 1e3,
            "toks": (1000.0 / tp) if tp else None, "channel": r.get("channel"),
            "task": r.get("true_task") or r.get("task"), "n": int(r.get("n_generated") or 0)}
    return out


def _q(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def aggregates(doc) -> dict[str, float]:
    """方案级指标（与 sim.py 的 scheme_metrics / key_metrics 同一口径）。"""
    rows = doc.get("requests") or []
    tp = [x for x in (_tpot(r) for r in rows) if x is not None]
    tt = [float(r["prefill_ms"]) for r in rows]
    times, tok, dec = [], {}, {}
    for r in rows:
        k = r.get("true_task") or r.get("task") or "?"
        tok[k] = tok.get(k, 0) + max(0, int(r.get("n_generated") or 0) - 1)
        dec[k] = dec.get(k, 0.0) + float(r.get("decode_ms") or 0) / 1e3
        t = float(r["t_submit_s"]) + float(r["prefill_ms"]) / 1e3
        times.append(t)
        for x in r.get("token_ms") or []:
            t += float(x) / 1e3
            times.append(t)
    t_end = max(float(r["t_done_s"]) for r in rows)
    t_sub = max(float(r["t_submit_s"]) for r in rows)
    out = {"steady": sum(1 for t in times if t <= t_sub) / t_sub, "full": len(times) / t_end,
           "tpot_mean": float(np.mean(tp)), "tpot_p50": _q(tp, .5), "tpot_p90": _q(tp, .9),
           "ttft_p50": _q(tt, .5), "ttft_mean": float(np.mean(tt)),
           "e2e_mean": float(np.mean([float(r["total_ms"]) for r in rows]))}
    for k in sorted(tok):
        if dec[k] > 0:
            out[f"task:{k}"] = tok[k] / dec[k]
    return out


def ks_w1(a, b) -> tuple[float, float]:
    """两组样本的 KS 距离 D 与 Wasserstein-1 距离。"""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    d = float(np.max(np.abs(fa - fb)))
    qs = np.linspace(0, 1, 2001)[1:-1]
    w1 = float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))
    return d, w1


def kde(x, grid, log=False):
    """高斯核密度（Silverman 带宽）；log=True 时在对数轴上估计。"""
    x = np.asarray(x, float)
    g = np.asarray(grid, float)
    if log:
        x, g = np.log10(x), np.log10(g)
    n = len(x)
    sd = np.std(x, ddof=1) if n > 1 else 1.0
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.34 if n > 3 else sd
    h = 0.9 * max(min(sd, iqr) if iqr > 0 else sd, 1e-9) * n ** (-0.2)
    z = (g[:, None] - x[None, :]) / h
    return np.exp(-0.5 * z * z).sum(1) / (n * h * math.sqrt(2 * math.pi))


# --------------------------------------------------------------------------- #
# 画图
# --------------------------------------------------------------------------- #
def _setup(lang: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    if lang == "zh":
        names = {f.name for f in fm.fontManager.ttflist}
        cjk = next((n for n in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC", "WenQuanYi Zen Hei",
                                "PingFang SC", "Microsoft YaHei", "SimHei", "Heiti SC") if n in names), None)
        if cjk is None:
            print("没找到中文字体，图上改用英文标签（装 fonts-noto-cjk 后可用中文）")
            lang = "en"
        else:
            plt.rcParams["font.sans-serif"] = [cjk, "DejaVu Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif", "axes.unicode_minus": False, "font.size": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "legend.frameon": False, "lines.linewidth": 2.0})
    return plt, T[lang]


def _save(fig, out: Path, name: str) -> list[Path]:
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    return paths


def _log_ticks(ax, which: str, lo: float, hi: float) -> None:
    """对数轴的刻度：写成普通数字（100、1000…，不用 10ⁿ 和 2×10⁰ 这类）；跨度不到 100 倍时再加 2、5 两档，
    免得整个轴只有一个刻度。"""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    subs = (1.0, 2.0, 5.0) if hi / max(lo, 1e-12) < 100 else (1.0,)
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in ((ax.xaxis, ax.yaxis) if which == "both" else (ax.xaxis,)):
        axis.set_major_locator(LogLocator(base=10, subs=subs))
        axis.set_major_formatter(fmt)
        axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        axis.set_minor_formatter(NullFormatter())


def fig_metrics(plt, L, runs, out):
    keys = ["steady", "full", "tpot_mean", "tpot_p50", "tpot_p90", "ttft_p50", "ttft_mean", "e2e_mean"]
    tasks = sorted({k for _, r, s in runs for k in aggregates(r) if k.startswith("task:")})
    keys += tasks
    names = [L["m"].get(k) or L["m"]["task"].format(t=k[5:]) for k in keys]
    err = {lab: {} for lab, _, _ in runs}
    for lab, r, s in runs:
        ar, as_ = aggregates(r), aggregates(s)
        for k in keys:
            if ar.get(k) and as_.get(k) is not None:
                err[lab][k] = (as_[k] / ar[k] - 1) * 100
    fig, ax = plt.subplots(figsize=(max(7.2, 1.55 * len(runs) + 1.6), 0.42 * len(keys) + 1.4))
    ax.axvspan(-5, 5, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    n = len(runs)
    off = np.linspace(-0.27, 0.27, n) if n > 1 else [0.0]
    for i, (lab, _, _) in enumerate(runs):
        ys = [j + off[i] for j, k in enumerate(keys) if k in err[lab]]
        xs = [err[lab][k] for k in keys if k in err[lab]]
        ax.scatter(xs, ys, s=42, color=SERIES[i % len(SERIES)], edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=lab)
    ax.set_yticks(range(len(keys)), names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    lim = max(15.0, max((abs(v) for e in err.values() for v in e.values()), default=0) * 1.15)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(L["err"])
    ax.text(4.6, len(keys) - 0.45, L["band"], color=MUTED, ha="right", va="center", fontsize=8)
    ax.set_title(L["metrics_title"], loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=min(n, 6), handletextpad=0.3, columnspacing=1.0,
              borderaxespad=0.2)
    return _save(fig, out, "1_metrics"), err


def fig_dist(plt, L, runs, out, metric):
    log = metric in ("ttft", "e2e")
    n = len(runs)
    fig, axs = plt.subplots(2, n, figsize=(3.0 * n + 0.6, 5.4), squeeze=False)
    stats = {}
    for i, (lab, r, s) in enumerate(runs):
        a = [v[metric] for v in per_request(r).values() if v[metric]]
        b = [v[metric] for v in per_request(s).values() if v[metric]]
        d, w1 = ks_w1(a, b)
        stats[lab] = {"ks_d": d, "w1_rel": w1 / float(np.mean(a)), "n_real": len(a), "n_sim": len(b)}
        lo, hi = min(min(a), min(b)), max(max(a), max(b))
        if log:
            grid = np.logspace(math.log10(lo * 0.8), math.log10(hi * 1.2), 300)
        else:
            pad = (hi - lo) * 0.08
            grid = np.linspace(max(lo - pad, 0), hi + pad, 300)
        top, bot = axs[0][i], axs[1][i]
        for xs, col, ls, name in ((a, REAL, "-", L["real"]), (b, SIM, "--", L["sim"])):
            dens = kde(xs, grid, log=log)
            top.plot(grid, dens, color=col, ls=ls, label=name)
            if ls == "-":
                top.fill_between(grid, dens, color=col, alpha=0.10, lw=0)
            xs_ = np.sort(xs)
            bot.step(xs_, np.arange(1, len(xs_) + 1) / len(xs_), where="post", color=col, ls=ls, label=name)
        for ax in (top, bot):
            if log:
                ax.set_xscale("log")
                _log_ticks(ax, "x", lo, hi)
        top.set_title(lab, loc="left", pad=18)
        top.set_yticks([])
        bot.set_ylim(0, 1.02)
        bot.set_xlabel(L[metric])
        top.text(0, 1.01, f"KS D = {d:.3f}   W1/μ = {w1 / np.mean(a) * 100:.1f}%", transform=top.transAxes,
                 ha="left", va="bottom", fontsize=8, color=INK2)
        if i == 0:
            top.set_ylabel(L["density"])
            bot.set_ylabel(L["ecdf"])
    axs[0][0].legend(loc="upper left", bbox_to_anchor=(0, 1.0))
    fig.suptitle(f"{L['dist_title']} · {L[metric].split('（')[0].split(' (')[0]}", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, out, f"2_{metric}"), stats


def fig_qq(plt, L, runs, out):
    metrics = ["tpot", "e2e"]
    n = len(runs)
    fig, axs = plt.subplots(len(metrics), n, figsize=(2.9 * n + 0.6, 2.9 * len(metrics) + 0.4), squeeze=False)
    qs = np.linspace(0.01, 0.99, 99)
    for j, m in enumerate(metrics):
        for i, (lab, r, s) in enumerate(runs):
            a = np.array([v[m] for v in per_request(r).values() if v[m]])
            b = np.array([v[m] for v in per_request(s).values() if v[m]])
            qa, qb = np.quantile(a, qs), np.quantile(b, qs)
            ax = axs[j][i]
            lo, hi = min(qa.min(), qb.min()), max(qa.max(), qb.max())
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=":", zorder=1)
            ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1], color=BAND, lw=0, zorder=0)
            ax.scatter(qa, qb, s=12, color=SERIES[i % len(SERIES)], zorder=3, linewidth=0)
            if m == "e2e":
                ax.set_xscale("log")
                ax.set_yscale("log")
                _log_ticks(ax, "both", lo, hi)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if j == 0:
                ax.set_title(lab, loc="left")
            ax.set_xlabel(f"{L['qq_x']} · {L[m]}", fontsize=8)
            if i == 0:
                ax.set_ylabel(L["qq_y"])
    fig.suptitle(f"{L['qq_title']}{'（灰带 = ±10%）' if L is T['zh'] else ' (band = ±10%)'}", x=0.01, ha="left",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out, "3_qq")


def fig_gauss(plt, L, runs, out, min_tokens=8):
    groups, pooled = [], []
    for lab, r, s in runs:
        R, S = per_request(r), per_request(s)
        e = [(S[k]["tpot"] / R[k]["tpot"] - 1) * 100 for k in R
             if k in S and R[k]["tpot"] and S[k]["tpot"] and R[k]["channel"] == S[k]["channel"]
             and R[k]["n"] > min_tokens]
        groups.append((lab, e, len(R)))
        pooled += e
    groups.append((L["pooled"], pooled, sum(g[2] for g in groups)))
    n = len(groups)
    fig, axs = plt.subplots(1, n, figsize=(2.9 * n + 0.4, 3.4), squeeze=False)
    stats = {}
    lim = 40
    bins = np.arange(-lim, lim + 0.1, 4)
    for i, (lab, e, total) in enumerate(groups):
        ax = axs[0][i]
        e = np.asarray(e, float)
        col = SERIES[i % len(SERIES)] if i < n - 1 else INK2
        mu, sd = float(np.mean(e)), float(np.std(e, ddof=1))
        within = float(np.mean(np.abs(e) <= 10))
        stats[lab] = {"mu": mu, "sigma": sd, "within10": within, "n": int(len(e)), "total": total}
        ax.axvspan(-10, 10, color=BAND, lw=0, zorder=0)
        ax.hist(np.clip(e, -lim, lim), bins=bins, color=col, alpha=0.85, edgecolor=SURFACE, linewidth=1.2,
                zorder=2)
        xs = np.linspace(-lim, lim, 400)
        ax.plot(xs, len(e) * 4 * np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)),
                color=INK, lw=1.5, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_xlim(-lim, lim)
        ax.set_title(lab, loc="left", pad=30)
        ax.set_xlabel(L["pair"], fontsize=8)
        if i == 0:
            ax.set_ylabel(L["count"])
        ax.text(0, 1.01, f"μ = {mu:+.1f}%   σ = {sd:.1f}%\n{L['in10']} {within * 100:.0f}%   "
                         f"{L['n']} {len(e)} {L['of']} {total}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=INK2)
    fig.suptitle(L["gauss_title"], x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, out, "4_err_gauss"), stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="标签=真机.json,模拟.json")
    ap.add_argument("--out", default="results/figs")
    ap.add_argument("--lang", choices=("zh", "en"), default="zh")
    a = ap.parse_args(argv)
    runs = []
    for spec in a.run:
        lab, _, files = spec.partition("=")
        rp, _, sp = files.partition(",")
        if not (lab and rp and sp):
            raise SystemExit(f"--run 写成 标签=真机.json,模拟.json，给的是 {spec!r}")
        runs.append((lab, json.loads(Path(rp).read_text(encoding="utf-8")),
                     json.loads(Path(sp).read_text(encoding="utf-8"))))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt, L = _setup(a.lang)
    summary = {}
    files, summary["metrics_err_pct"] = fig_metrics(plt, L, runs, out)
    for m in ("tpot", "ttft", "e2e", "toks"):
        f, summary[f"dist_{m}"] = fig_dist(plt, L, runs, out, m)
        files += f
    files += fig_qq(plt, L, runs, out)
    f, summary["paired_tpot_err"] = fig_gauss(plt, L, runs, out)
    files += f
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    plt.close("all")
    for p in files:
        if p.suffix == ".png":
            print(f"已写 {p}（另有 .pdf）")
    print(f"已写 {out / 'summary.json'}（图上的数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
