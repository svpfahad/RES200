"""Generate the data figures for the QLattice symbolic-pKa manuscript.

Every figure is built from recorded run artifacts under sota_pka/runs (and the new
E7/E9/E10 result CSVs). No invented numbers. Outputs 300-dpi PNG + vector PDF to
figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/Users/fahad/Downloads/RES200")
RUNS = REPO / "sota_pka" / "runs"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans", "legend.frameon": False,
})
# Colorblind-safe (Wong)
C = {"symbolic": "#0072B2", "distilled": "#56B4E9", "gbm": "#D55E00",
     "rf": "#E69F00", "elastic": "#999999", "yrand": "#000000", "ceiling": "#009E73"}


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png/.pdf")


def _gbm(task):
    d = "res200_op2_full_descriptors_wsl_full" if task == "acidic" else "res200_basic_op2_full_descriptors"
    df = pd.read_csv(RUNS / d / "metrics.csv")
    return {r["model"]: (int(r["n_features"]), float(r["test_r2"])) for _, r in df.iterrows()}


def _symbolic(task):
    out = {}
    for mode in ("direct", "distilled"):
        p = RUNS / f"qlattice_{task}_op2" / f"summary_{task}_{mode}.json"
        s = json.loads(p.read_text())
        out[mode] = (int(s["chosen_top_k"]), float(s["test_metrics"]["r2"]))
    return out


# --------------------------------------------------------------------------- #
# Fig 1 — accuracy vs interpretability tradeoff
# --------------------------------------------------------------------------- #
def fig_tradeoff():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, task in zip(axes, ("acidic", "basic")):
        g = _gbm(task); s = _symbolic(task)
        gbm_models = [("lightgbm", "LightGBM"), ("xgboost", "XGBoost"),
                      ("catboost", "CatBoost"), ("random_forest", "Random forest")]
        gbm_r2 = []
        nf_gbm = None
        for key, lab in gbm_models:
            if key in g:
                nf, r2 = g[key]
                nf_gbm = nf
                gbm_r2.append(r2)
                ax.scatter(nf, r2, s=70, c=C["gbm"], marker="o", zorder=3,
                           edgecolor="white", linewidth=0.6)
        # single cluster label instead of overlapping per-point text
        if gbm_r2 and nf_gbm is not None:
            ax.annotate(f"GBM ensembles\nR²={min(gbm_r2):.2f}–{max(gbm_r2):.2f}\n({nf_gbm} descriptors)",
                        (nf_gbm, max(gbm_r2)), textcoords="offset points", xytext=(-12, 8),
                        ha="right", va="bottom", fontsize=8.5, color=C["gbm"], fontweight="bold")
        if "elasticnet" in g and g["elasticnet"][1] > -2:
            nf, r2 = g["elasticnet"]
            ax.scatter(nf, r2, s=70, c=C["elastic"], marker="s", zorder=3, edgecolor="white")
            ax.annotate("ElasticNet", (nf, r2), textcoords="offset points", xytext=(-6, -12),
                        ha="right", fontsize=8, color=C["elastic"])
        nf, r2 = s["direct"]
        ax.scatter(nf, r2, s=130, c=C["symbolic"], marker="*", zorder=4,
                   edgecolor="white", linewidth=0.6, label="QLattice (direct)")
        ax.annotate(f"QLattice\n{nf} feats, R²={r2:.2f}", (nf, r2),
                    textcoords="offset points", xytext=(10, -2), ha="left",
                    fontsize=8.5, color=C["symbolic"], fontweight="bold")
        nf, r2 = s["distilled"]
        ax.scatter(nf, r2, s=90, c=C["distilled"], marker="D", zorder=4, edgecolor="white")
        ax.annotate("distilled", (nf, r2), textcoords="offset points", xytext=(10, -10),
                    ha="left", fontsize=8, color=C["distilled"])
        ax.set_xscale("log")
        ax.set_xlabel("Number of input descriptors (log scale)")
        ax.set_title(f"{task.capitalize()} pK$_a$", fontsize=12)
        ax.set_xlim(4, 4000)
        ax.set_ylim(-0.05, 0.95)
    axes[0].set_ylabel("Held-out test R²")
    fig.suptitle("Accuracy–interpretability tradeoff: compact symbolic formulas vs gradient-boosted ensembles",
                 fontsize=11.5, y=1.02)
    _save(fig, "fig1_tradeoff")


# --------------------------------------------------------------------------- #
# Fig — external generalization: rank signal vs calibration (R²)
# --------------------------------------------------------------------------- #
def fig_external_rank_vs_calib():
    boot = pd.read_csv(RUNS / "uncertainty_calibration" / "bootstrap_ci.csv")
    pooled = boot[boot["scope"] == "external_pooled"]
    models = [("qlattice", "QLattice", C["symbolic"]), ("lightgbm", "LightGBM", C["gbm"]),
              ("random_forest", "Random forest", C["rf"]), ("y_random_lightgbm", "y-random", C["yrand"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, task in zip(axes, ("acidic", "basic")):
        sub = pooled[pooled["task"] == task]
        x = np.arange(len(models)); w = 0.38
        sp, sp_lo, sp_hi, r2, r2_lo, r2_hi, cols, labs = [], [], [], [], [], [], [], []
        for key, lab, col in models:
            r = sub[sub["model"] == key]
            if r.empty:
                continue
            r = r.iloc[0]
            sp.append(r["spearman"]); sp_lo.append(r["spearman"] - r["spearman_lo"]); sp_hi.append(r["spearman_hi"] - r["spearman"])
            r2.append(max(r["r2"], -0.5)); r2_lo.append(0); r2_hi.append(0)
            cols.append(col); labs.append(lab)
        xx = np.arange(len(labs))
        ax.bar(xx - w/2, sp, w, yerr=[sp_lo, sp_hi], capsize=3, color=cols, alpha=0.95, label="Spearman ρ (rank)")
        ax.bar(xx + w/2, r2, w, color=cols, alpha=0.45, hatch="//", label="R² (calibration)")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xx); ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{task.capitalize()} pK$_a$ — pooled external", fontsize=12)
        ax.set_ylim(-0.55, 1.0)
    axes[0].set_ylabel("Score")
    # one shared legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor="#777", label="Spearman ρ (rank order)"),
           Patch(facecolor="#777", alpha=0.45, hatch="//", label="R² (absolute calibration)")]
    axes[1].legend(handles=leg, loc="upper right", fontsize=8.5)
    fig.suptitle("External blind sets: symbolic models keep above-chance rank signal but lose calibration",
                 fontsize=11.5, y=1.02)
    _save(fig, "fig_external_rank_vs_calib")


# --------------------------------------------------------------------------- #
# Fig — OOD calibration decomposition
# --------------------------------------------------------------------------- #
def fig_ood_calibration():
    rec = pd.read_csv(RUNS / "uncertainty_calibration" / "ood_recalibration.csv")
    pooled = rec[rec["scope"] == "external_pooled"]
    models = [("qlattice", "QLattice"), ("lightgbm", "LightGBM"), ("random_forest", "Random forest")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, task in zip(axes, ("acidic", "basic")):
        sub = pooled[pooled["task"] == task]
        labs = [l for k, l in models if not sub[sub["model"] == k].empty]
        keys = [k for k, l in models if not sub[sub["model"] == k].empty]
        xx = np.arange(len(keys)); w = 0.26
        raw = [sub[sub["model"] == k].iloc[0]["r2_raw"] for k in keys]
        recal = [sub[sub["model"] == k].iloc[0]["r2_recal_cv"] for k in keys]
        ceil = [sub[sub["model"] == k].iloc[0]["r2_affine_ceiling"] for k in keys]
        ax.bar(xx - w, raw, w, color=C["gbm"], label="R² raw (as scored)")
        ax.bar(xx, recal, w, color=C["symbolic"], label="R² after OOF affine recalibration")
        ax.bar(xx + w, ceil, w, color=C["ceiling"], alpha=0.6, label="R² affine ceiling (Pearson²)")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=9)
        ax.set_title(f"{task.capitalize()} pK$_a$ — pooled external", fontsize=12)
        for i, v in enumerate(raw):
            ax.annotate(f"{v:.2f}", (i - w, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=7.5)
        for i, v in enumerate(recal):
            ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=7.5)
    axes[0].set_ylabel("R²")
    axes[0].set_ylim(-0.6, 0.8)
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Out-of-distribution failure decomposed: basic-pK$_a$ symbolic deficit is mostly recoverable miscalibration",
                 fontsize=11, y=1.02)
    _save(fig, "fig_ood_calibration")


# --------------------------------------------------------------------------- #
# Fig — nested-CV stability band (reads whatever folds are done)
# --------------------------------------------------------------------------- #
def fig_nested_cv():
    p = RUNS / "nested_cv" / "results.jsonl"
    if not p.exists():
        print("  [skip] nested_cv results not yet present")
        return
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip() and "outer_r2" in json.loads(l)]
    if not recs:
        print("  [skip] no completed nested-CV folds yet")
        return
    boot = pd.read_csv(RUNS / "uncertainty_calibration" / "bootstrap_ci.csv")
    indist = boot[boot["scope"] == "in_distribution"].set_index("task")
    recorded = {"acidic": 0.448, "basic": 0.470}
    tasks = ["acidic", "basic"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for i, task in enumerate(tasks):
        rs = [r["outer_r2"] for r in recs if r["task"] == task]
        if not rs:
            continue
        xs = np.full(len(rs), i) + np.linspace(-0.06, 0.06, len(rs))
        ax.scatter(xs, rs, s=45, color=C["symbolic"], zorder=3, label="nested-CV outer fold" if i == 0 else None)
        m = np.mean(rs)
        ax.hlines(m, i - 0.18, i + 0.18, color=C["symbolic"], lw=2,
                  label="nested-CV mean" if i == 0 else None)
        if task in indist.index:
            row = indist.loc[task]
            ax.errorbar(i + 0.22, row["r2"], yerr=[[row["r2"] - row["r2_lo"]], [row["r2_hi"] - row["r2"]]],
                        fmt="s", color=C["gbm"], capsize=4, ms=7,
                        label="single held-out test ± bootstrap 95% CI" if i == 0 else None)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"{t.capitalize()} pK$_a$" for t in tasks])
    ax.set_ylabel("Test R²")
    ax.set_ylim(0.30, 0.60)
    ax.set_title(f"Symbolic-model R² is stable (nested CV, {len(recs)} outer folds)", fontsize=11.5)
    ax.legend(fontsize=8.5, loc="lower center")
    _save(fig, "fig_nested_cv_stability")


if __name__ == "__main__":
    print("Generating manuscript figures...")
    fig_tradeoff()
    fig_external_rank_vs_calib()
    fig_ood_calibration()
    fig_nested_cv()
    print("Done.")
