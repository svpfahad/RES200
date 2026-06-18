"""End-to-end uncertainty + AD evaluation for one task, with paper artifacts.

Pipeline (all leakage-safe — calibration and AD come from training data only,
the op2 test set is scored once):

  1. bootstrap XGBoost ensemble on proper-train → test point predictions (mean)
     and a per-sample disagreement std;
  2. a full-train point model for the headline R²/RMSE comparison to the paper;
  3. split / normalized / CQR prediction intervals at several nominal levels;
  4. kNN-distance and Tanimoto applicability domains;
  5. coverage / sharpness / calibration tables, and the in-vs-out-of-AD
     stratification that motivates reporting an AD;
  6. CSVs under ``runs/uq_xgb/<task>/`` and figures under ``paper_assets/uq/``.

Run via the CLI: ``python -m sota_pka.uq.cli run --task acidic``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluate import regression_metrics
from .applicability import ADResult, KNNDistanceAD, TanimotoAD
from .baseline import fit_ensemble, fit_point_model
from .conformal import Intervals, cqr_intervals, normalized_conformal, split_conformal
from .data_splits import UQSplit, load_split


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


RUNS_ROOT = _repo_root() / "sota_pka" / "runs" / "uq_xgb"
FIG_ROOT = _repo_root() / "sota_pka" / "paper_assets" / "uq"


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _coverage_rows(intervals: dict[tuple[str, float], Intervals], y_test: np.ndarray) -> list[dict]:
    rows = []
    for (method, alpha), iv in intervals.items():
        cov = iv.covered(y_test)
        w = iv.width
        rows.append({
            "method": method,
            "alpha": alpha,
            "nominal_coverage": round(1.0 - alpha, 3),
            "empirical_coverage": float(cov.mean()),
            "mean_width": float(w.mean()),
            "median_width": float(np.median(w)),
            "n": int(len(y_test)),
        })
    return rows


def _ad_summary_row(ad: ADResult, err: np.ndarray, y_true: np.ndarray, pred: np.ndarray) -> dict:
    inn, out = ad.in_domain, ~ad.in_domain
    def _m(mask):
        if not mask.any():
            return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
        return {"n": int(mask.sum()),
                "mae": float(err[mask].mean()),
                "rmse": float(np.sqrt(np.mean((y_true[mask] - pred[mask]) ** 2))),
                "r2": float(regression_metrics(y_true[mask], pred[mask])["r2"]) if mask.sum() > 1 else np.nan}
    mi, mo = _m(inn), _m(out)
    return {
        "ad_method": ad.name,
        "threshold": ad.threshold,
        "frac_in_domain": ad.frac_in,
        "n_in": mi["n"], "mae_in": mi["mae"], "rmse_in": mi["rmse"], "r2_in": mi["r2"],
        "n_out": mo["n"], "mae_out": mo["mae"], "rmse_out": mo["rmse"], "r2_out": mo["r2"],
        "mae_ratio_out_in": (mo["mae"] / mi["mae"]) if mi["mae"] else np.nan,
    }


def _ad_coverage_rows(ad: ADResult, intervals: dict[tuple[str, float], Intervals],
                      y_test: np.ndarray, alpha: float) -> list[dict]:
    inn, out = ad.in_domain, ~ad.in_domain
    rows = []
    for (method, a), iv in intervals.items():
        if a != alpha:
            continue
        cov = iv.covered(y_test)
        w = iv.width
        rows.append({
            "ad_method": ad.name, "interval_method": method, "alpha": alpha,
            "nominal": round(1 - alpha, 3),
            "coverage_in": float(cov[inn].mean()) if inn.any() else np.nan,
            "coverage_out": float(cov[out].mean()) if out.any() else np.nan,
            "width_in": float(w[inn].mean()) if inn.any() else np.nan,
            "width_out": float(w[out].mean()) if out.any() else np.nan,
        })
    return rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _figures(task: str, intervals: dict, y_test: np.ndarray, ens_std: np.ndarray,
             err: np.ndarray, ads: list[ADResult], cov_df: pd.DataFrame, fig_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # (1) Calibration curve: empirical vs nominal coverage per method.
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0.7, 1.0], [0.7, 1.0], "k--", lw=1, label="ideal")
    for method, g in cov_df.groupby("method"):
        g = g.sort_values("nominal_coverage")
        ax.plot(g["nominal_coverage"], g["empirical_coverage"], "o-", label=method)
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title(f"{task}: interval calibration"); ax.legend(fontsize=8)
    fig.tight_layout(); p = fig_dir / f"{task}_calibration.png"; fig.savefig(p, dpi=150); plt.close(fig)
    saved.append(str(p))

    # (2) Interval width distributions (alpha=0.1).
    a = 0.1
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    data, labels = [], []
    for (method, al), iv in intervals.items():
        if al == a:
            data.append(iv.width); labels.append(method)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("interval width (pKa units)")
    ax.set_title(f"{task}: sharpness @ 90% nominal")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
    fig.tight_layout(); p = fig_dir / f"{task}_width.png"; fig.savefig(p, dpi=150); plt.close(fig)
    saved.append(str(p))

    # (3) AD score vs |error|, in/out coloured (one panel per AD method).
    fig, axes = plt.subplots(1, len(ads), figsize=(4.6 * len(ads), 4.0), squeeze=False)
    for ax, ad in zip(axes[0], ads):
        m = ~np.isnan(ad.score)
        inn = ad.in_domain & m
        out = (~ad.in_domain) & m
        ax.scatter(ad.score[inn], err[inn], s=10, alpha=0.5, label="in-domain")
        ax.scatter(ad.score[out], err[out], s=12, alpha=0.6, color="crimson", label="out-of-domain")
        ax.axvline(ad.threshold, color="k", ls="--", lw=1)
        ax.set_xlabel(f"{ad.name} score"); ax.set_ylabel("|error| (pKa)")
        ax.set_title(f"{task}: {ad.name} AD"); ax.legend(fontsize=8)
    fig.tight_layout(); p = fig_dir / f"{task}_ad_error.png"; fig.savefig(p, dpi=150); plt.close(fig)
    saved.append(str(p))

    return saved


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class UQReport:
    task: str
    predictive: pd.DataFrame
    coverage: pd.DataFrame
    ad_summary: pd.DataFrame
    ad_coverage: pd.DataFrame
    predictions: pd.DataFrame
    figures: list[str]
    meta: dict


def run_task(
    task: str,
    alphas: tuple[float, ...] = (0.05, 0.1, 0.2),
    n_members: int = 20,
    seed: int = 42,
    calib_frac: float = 0.2,
    make_figures: bool = True,
) -> UQReport:
    split: UQSplit = load_split(task, calib_frac=calib_frac, seed=seed)
    yc = split.y_calib.to_numpy(float)
    yt = split.y_test.to_numpy(float)

    # 1. ensemble (conformal point model + difficulty std)
    ens = fit_ensemble(split.x_proper, split.y_proper, n_members=n_members, base_seed=1000 + seed)
    ep_c, ep_t = ens.predict(split.x_calib), ens.predict(split.x_test)
    err = np.abs(yt - ep_t.mean)

    # 2. full-train point model (headline comparison to the paper)
    pm_full = fit_point_model(split.x_train_full, split.y_train_full, seed=seed)
    pred_full = pm_full.predict(split.x_test)

    predictive = pd.DataFrame([
        {"model": "xgb_ensemble_mean(proper-train)", **regression_metrics(yt, ep_t.mean)},
        {"model": "xgb_point(full-train)", **regression_metrics(yt, pred_full)},
    ])

    # 3. intervals
    intervals: dict[tuple[str, float], Intervals] = {}
    for a in alphas:
        intervals[("split_conformal", a)] = split_conformal(yc, ep_c.mean, ep_t.mean, a)
        intervals[("normalized_conformal", a)] = normalized_conformal(
            yc, ep_c.mean, ep_c.std, ep_t.mean, ep_t.std, a)
        intervals[("cqr", a)] = cqr_intervals(
            split.x_proper, split.y_proper, split.x_calib, yc, split.x_test, a, seed=seed)
    coverage = pd.DataFrame(_coverage_rows(intervals, yt)).sort_values(
        ["method", "nominal_coverage"]).reset_index(drop=True)

    # 4. applicability domains (kNN on natively-measured features only — see
    #    data_splits: 0-filled columns would push every test point out of domain)
    ad_cols = split.ad_feature_names
    ad_knn = KNNDistanceAD().fit(split.x_train_full[ad_cols]).evaluate(split.x_test[ad_cols])
    ad_tan = TanimotoAD().fit(split.smiles_train_full).evaluate(split.smiles_test)
    ads = [ad_knn, ad_tan]

    ad_summary = pd.DataFrame([_ad_summary_row(ad, err, yt, ep_t.mean) for ad in ads])
    ad_cov_rows = []
    for ad in ads:
        ad_cov_rows += _ad_coverage_rows(ad, intervals, yt, alpha=0.1)
    ad_coverage = pd.DataFrame(ad_cov_rows)

    # 5. per-compound predictions table
    pred_tbl = pd.DataFrame({
        "smiles": split.smiles_test.to_numpy(),
        "y_true": yt,
        "y_pred_ensemble": ep_t.mean,
        "ensemble_std": ep_t.std,
        "y_pred_full": pred_full,
        "abs_error": err,
        "ad_knn_score": ad_knn.score, "ad_knn_in": ad_knn.in_domain,
        "ad_tanimoto_score": ad_tan.score, "ad_tanimoto_in": ad_tan.in_domain,
    })
    for name, a in [("split_conformal", 0.1), ("normalized_conformal", 0.1), ("cqr", 0.1)]:
        iv = intervals[(name, a)]
        pred_tbl[f"{name}_lower"] = iv.lower
        pred_tbl[f"{name}_upper"] = iv.upper

    # 6. persist
    out_dir = RUNS_ROOT / task
    out_dir.mkdir(parents=True, exist_ok=True)
    predictive.to_csv(out_dir / "metrics_predictive.csv", index=False)
    coverage.to_csv(out_dir / "coverage.csv", index=False)
    ad_summary.to_csv(out_dir / "ad_summary.csv", index=False)
    ad_coverage.to_csv(out_dir / "ad_coverage.csv", index=False)
    pred_tbl.to_csv(out_dir / "predictions_test.csv", index=False)

    figures = []
    if make_figures:
        figures = _figures(task, intervals, yt, ep_t.std, err, ads, coverage, FIG_ROOT)

    meta = {**split.meta, "n_members": n_members, "alphas": list(alphas),
            "ensemble_test_r2": float(regression_metrics(yt, ep_t.mean)["r2"]),
            "out_dir": str(out_dir)}
    return UQReport(task, predictive, coverage, ad_summary, ad_coverage, pred_tbl, figures, meta)
