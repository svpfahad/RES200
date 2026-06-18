"""Regenerate all seven figures for the JURI resubmission with consistent typography.

Editor's instructions (from KFUPM_309_Letter_from_the_Editor.docx + inline comments):
- Fig 1: no label changes, just consistent style.
- Fig 2: text legible at 100% magnification.
- Fig 3: "Train" -> "Training"; sentence case x-axis; "Best" -> "optimal".
- Fig 4: "Training set" / "Test set"; spaces around "="; (a)/(b) panel labels.
- Fig 5: consistent text size; sentence case.
- Fig 6: consistent text; sentence case x-axis; drop the spurious "nan" row.
- Fig 7: consistent text; "Residual" (singular) on panel (b); (a)/(b) labels.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Fahad\Downloads\RES200")
SRC_DIR = ROOT / "RES 200-20260312T035531Z-1-001" / "RES 200"
OUT_DIR = ROOT / "figs_revised"
OUT_DIR.mkdir(exist_ok=True)

# Consistent typography across all seven figures
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
})

PALETTE = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
MARKERS = {1: "o", 2: "s", 3: "^", 4: "D"}


# ---------------------------------------------------------------------------
# Figure 1 + 5 + 2 source data
# ---------------------------------------------------------------------------
morgan = pd.read_csv(SRC_DIR / "morgan_results.csv")
test_grid  = morgan.pivot(index="radius", columns="fp_size", values="Test_R2")
train_grid = morgan.pivot(index="radius", columns="fp_size", values="Train_R2")
sizes = sorted(morgan["fp_size"].unique())


def panel_label(ax, label, x=0.02, y=0.96):
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontweight="bold", fontsize=13,
        ha="left", va="top",
    )


# ---------------------------------------------------------------------------
# Figure 1 — Effect of Morgan FP bit length on test R² (no label changes)
# ---------------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=300)
    for r in [1, 2, 3, 4]:
        y = test_grid.loc[r, sizes].values
        ax.plot(sizes, y, marker=MARKERS[r], color=PALETTE[r], label=f"Radius {r}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Fingerprint bit length")
    ax.set_ylabel("Test R²")
    ax.set_xticks([2 ** k for k in range(4, 15, 2)])
    ax.set_xticklabels([f"$2^{{{k}}}$" for k in range(4, 15, 2)])
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.set_ylim(-0.05, 1.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "image1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Heatmap of test R² across radius × bit length
# Editor: legible text at 100% magnification.
# ---------------------------------------------------------------------------
def fig2():
    fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=300)
    z = test_grid.loc[[1, 2, 3, 4], sizes].values
    im = ax.imshow(z, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=z.max())
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes, fontsize=12)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"Radius {r}" for r in [1, 2, 3, 4]], fontsize=12)
    ax.set_xlabel("Bit length", fontsize=14)
    ax.set_ylabel("Morgan radius", fontsize=14)
    # Annotate cells with values — bigger font for legibility
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z[i, j]
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=color, fontsize=11.5, fontweight="medium")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Test R²", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "image2.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Train vs Test R² across descriptor categories
# Editor: "Train" -> "Training"; sentence case x-axis; "Best" -> "optimal";
# also fix label "Full Descriptor (717 features)" -> "Full descriptors (1,135 features)"
# (must match Table 1).
# ---------------------------------------------------------------------------
def fig3():
    cats = [
        ("Continuous\ndescriptors",          0.850, 0.159),
        ("MACCS keys",                       0.977, 0.780),
        ("Morgan FP\n(optimal)",             0.961, 0.795),
        ("Combined\n(optimal)",              0.995, 0.817),
        ("Full descriptors\n(1,135 features)", 0.9998, 0.829),
    ]
    labels = [c[0] for c in cats]
    train  = [c[1] for c in cats]
    test   = [c[2] for c in cats]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=300)
    ax.bar(x - width/2, train, width, label="Training R²",
           color="#2e5b9c", edgecolor="black", linewidth=0.6)
    bars_t = ax.bar(x + width/2, test, width, label="Test R²",
                    color="#e07b39", edgecolor="black", linewidth=0.6)
    for b, v in zip(bars_t, test):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("R² score", fontsize=13)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "image3.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Predicted vs experimental pKa
# Editor: "Training set" / "Test set"; spaces around "="; (a)/(b) panel labels.
# ---------------------------------------------------------------------------
def fig4():
    train_pred = pd.read_csv(ROOT / "predictions_train.csv")
    test_pred  = pd.read_csv(ROOT / "predictions_test.csv")

    from sklearn.metrics import r2_score, mean_squared_error
    r2_tr   = r2_score(train_pred.y_true, train_pred.y_pred)
    rmse_tr = np.sqrt(mean_squared_error(train_pred.y_true, train_pred.y_pred))
    r2_te   = r2_score(test_pred.y_true,  test_pred.y_pred)
    rmse_te = np.sqrt(mean_squared_error(test_pred.y_true,  test_pred.y_pred))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), dpi=300)
    lo = min(train_pred.y_true.min(), train_pred.y_pred.min(),
             test_pred.y_true.min(),  test_pred.y_pred.min()) - 1
    hi = max(train_pred.y_true.max(), train_pred.y_pred.max(),
             test_pred.y_true.max(),  test_pred.y_pred.max()) + 1

    ax = axes[0]
    ax.scatter(train_pred.y_true, train_pred.y_pred,
               s=10, alpha=0.55, color="#1f4e8a", edgecolor="none")
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Experimental pKa")
    ax.set_ylabel("Predicted pKa")
    ax.set_title(f"Training set (n = {len(train_pred)})")
    ax.text(0.62, 0.07, f"R² = {r2_tr:.4f}\nRMSE = {rmse_tr:.3f}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.9))
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.scatter(test_pred.y_true, test_pred.y_pred,
               s=14, alpha=0.55, color="#e07b39", edgecolor="none")
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Experimental pKa")
    ax.set_ylabel("Predicted pKa")
    ax.set_title(f"Test set (n = {len(test_pred)})")
    ax.text(0.62, 0.07, f"R² = {r2_te:.4f}\nRMSE = {rmse_te:.3f}",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="grey", alpha=0.9))
    panel_label(ax, "(b)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "image4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Overfitting gap (Train R² − Test R²) across radius × bit length
# Editor: consistent text; sentence case.
# ---------------------------------------------------------------------------
def fig5():
    gap = train_grid - test_grid
    fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=300)
    for r in [1, 2, 3, 4]:
        y = gap.loc[r, sizes].values
        ax.plot(sizes, y, marker=MARKERS[r], color=PALETTE[r],
                label=f"Radius {r}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Fingerprint bit length")
    ax.set_ylabel("Training R² − test R² (overfitting gap)")
    ax.set_xticks([2 ** k for k in range(4, 15, 2)])
    ax.set_xticklabels([f"$2^{{{k}}}$" for k in range(4, 15, 2)])
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "image5.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — Top 20 features (gain) — values from paper text & original PNG.
# Editor: consistent text; sentence case x-axis; drop the spurious "nan" row.
# ---------------------------------------------------------------------------
def fig6():
    # Top 20 from the paper's analysis (matches Table 2 + section 3.5 narrative)
    feats = [
        ("nAcid",                     0.1758),
        ("SpAbs_Dzpe",                0.0491),
        ("fr_Ar_NH",                  0.0293),
        ("nHBAcc",                    0.0220),
        ("SpDiam_Dzp",                0.0190),
        ("fr_phenol",                 0.0186),
        ("SpMAD_Dzse",                0.0162),
        ("MaxPartialCharge",          0.0154),
        ("AATS0dv",                   0.0150),
        ("AATSC0p",                   0.0137),
        ("fr_quatN",                  0.0131),
        ("MZ",                        0.0128),
        ("NsOH",                      0.0115),
        ("nBase",                     0.0112),
        ("SsOH",                      0.0101),
        ("SM1_Dzi",                   0.0096),
        ("AMW",                       0.0094),
        ("fr_SH",                     0.0093),
        ("fr_phenol_noOrthoHbond",    0.0090),
        ("MIC0",                      0.0086),
    ]
    names  = [f[0] for f in feats][::-1]   # reverse so largest is at top
    values = [f[1] for f in feats][::-1]

    cmap = plt.cm.viridis_r
    colors = cmap(np.linspace(0.15, 0.85, len(values)))

    fig, ax = plt.subplots(figsize=(7.4, 6.0), dpi=300)
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Feature importance (gain)")
    for b, v in zip(bars, values):
        ax.text(v + max(values) * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10)
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "image6.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 — Residual analysis on the test set
# Editor: consistent text; "Residual" (singular) in panel (b); (a)/(b) labels.
# ---------------------------------------------------------------------------
def fig7():
    test_pred = pd.read_csv(ROOT / "predictions_test.csv")
    res = test_pred.y_true - test_pred.y_pred

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=300)

    ax = axes[0]
    ax.hist(res, bins=40, color="#4a78b5", edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.axvline(0, color="#c0392b", linestyle="--", lw=1.2)
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual distribution")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    panel_label(ax, "(a)")

    ax = axes[1]
    ax.scatter(test_pred.y_pred, res,
               s=14, alpha=0.55, color="#e07b39", edgecolor="none")
    ax.axhline(0, color="#c0392b", linestyle="--", lw=1.2)
    ax.set_xlabel("Predicted pKa")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs predicted")
    ax.grid(True, alpha=0.25, linestyle=":")
    panel_label(ax, "(b)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "image7.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    fig1(); print("Figure 1 written")
    fig2(); print("Figure 2 written")
    fig3(); print("Figure 3 written")
    fig4(); print("Figure 4 written")
    fig5(); print("Figure 5 written")
    fig6(); print("Figure 6 written")
    fig7(); print("Figure 7 written")


if __name__ == "__main__":
    main()
