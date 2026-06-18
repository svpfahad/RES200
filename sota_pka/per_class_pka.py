"""Per-class symbolic pKa modelling (Hammett/Taft-style segmentation).

Route each molecule to its ionizable-group class, fit a compact leakage-safe
symbolic equation per major class, fall back to a global symbolic model for rare
classes, then assemble predictions over the FULL held-out test set so the overall
R2/RMSE is directly comparable to the global single-equation model.

Rationale (from EDA): functional-group identity alone explains R2~0.40; within a
class the pKa spread is small, so a per-class substituent equation is both easier
to fit and more interpretable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .qlattice_search import regression_metrics, run_qlattice_experiment

logging.getLogger("feyn").setLevel(logging.WARNING)

META_COLS = ["smiles", "class"]

PER_CLASS_GRID = dict(
    top_k_grid=(4, 6, 8),
    max_complexity=20,
    criterion="bic",
    cv_folds=4,
    coarse_epochs=30,
    refine_epochs=80,
    refit_seeds=(0, 1),
    corr_threshold=0.95,
    threads=14,
)


def _features_only(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df.drop(columns=[c for c in META_COLS if c in df.columns], errors="ignore")


def _fit_and_predict(train_df, test_df, output_dir, label, target, grid, teacher_factory=None):
    """Run one leakage-safe experiment; return (summary, predictions aligned to test_df.index)."""
    mode = "distilled" if teacher_factory else "direct"
    summary = run_qlattice_experiment(
        _features_only(train_df, target), _features_only(test_df, target),
        output_dir=output_dir, target=target, mode=mode,
        teacher_factory=teacher_factory, label=label, **grid,
    )
    pred_csv = Path(output_dir) / f"predictions_qlattice_{label}.csv"
    y_pred = pd.read_csv(pred_csv)["y_pred"].to_numpy()
    if len(y_pred) != len(test_df):
        raise RuntimeError(f"{label}: prediction count {len(y_pred)} != test rows {len(test_df)}")
    return summary, pd.Series(y_pred, index=test_df.index)


def run_per_class(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path,
    task: str,
    target: str = "pKa",
    min_train: int = 60,
    min_test: int = 12,
    grid: dict | None = None,
    teacher_factory=None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = grid or PER_CLASS_GRID

    train_counts = train["class"].value_counts()
    test_counts = test["class"].value_counts()
    major = [c for c in train_counts.index
             if train_counts.get(c, 0) >= min_train and test_counts.get(c, 0) >= min_test]

    test_pred = pd.Series(np.nan, index=test.index, dtype=float)
    class_rows: list[dict] = []
    class_means = train.groupby("class")[target].mean()

    # ---- per major class ----
    for c in major:
        tr_c = train[train["class"] == c]
        te_c = test[test["class"] == c]
        label = f"{task}_{c}"
        try:
            summary, preds = _fit_and_predict(tr_c, te_c, output_dir / f"class_{c}", label, target, grid, teacher_factory)
            test_pred.loc[preds.index] = preds.to_numpy()
            m = summary["test_metrics"]
            class_rows.append({
                "class": c, "n_train": int(len(tr_c)), "n_test": int(len(te_c)),
                "test_r2": m["r2"], "test_rmse": m["rmse"], "test_mae": m["mae"],
                "k": summary["chosen_top_k"], "complexity": summary["sympy_ops"],
                "formula": summary["formula"], "source": "per_class",
            })
        except Exception as exc:  # noqa: BLE001 — fall back to class mean
            mu = float(class_means.get(c, train[target].mean()))
            test_pred.loc[te_c.index] = mu
            class_rows.append({"class": c, "n_train": int(len(tr_c)), "n_test": int(len(te_c)),
                               "test_r2": None, "test_rmse": None, "test_mae": None,
                               "k": 0, "complexity": 0, "formula": f"{target} = {mu:.3f} (class mean; SR failed: {exc})",
                               "source": "class_mean_fallback"})

    # ---- rare classes: one global symbolic fallback over all training rows ----
    minor_mask = test_pred.isna()
    if minor_mask.any():
        test_minor = test[minor_mask]
        try:
            summary, preds = _fit_and_predict(train, test_minor, output_dir / "fallback_global",
                                              f"{task}_fallback", target, grid, teacher_factory)
            test_pred.loc[preds.index] = preds.to_numpy()
            class_rows.append({"class": "<rare:global_fallback>", "n_train": int(len(train)),
                               "n_test": int(minor_mask.sum()), "test_r2": summary["test_metrics"]["r2"],
                               "test_rmse": summary["test_metrics"]["rmse"], "test_mae": summary["test_metrics"]["mae"],
                               "k": summary["chosen_top_k"], "complexity": summary["sympy_ops"],
                               "formula": summary["formula"], "source": "global_fallback"})
        except Exception:
            # last resort: overall training mean
            mu = float(train[target].mean())
            test_pred.loc[minor_mask] = mu
            class_rows.append({"class": "<rare:mean_fallback>", "n_train": int(len(train)),
                               "n_test": int(minor_mask.sum()), "formula": f"{target} = {mu:.3f}",
                               "source": "mean_fallback"})

    # ---- overall metrics on the FULL test set ----
    y_true = test[target].to_numpy(dtype=float)
    y_pred = test_pred.to_numpy(dtype=float)
    overall = regression_metrics(y_true, y_pred)

    # class-mean-only baseline (route to training class mean) for reference
    cm_pred = test["class"].map(class_means).fillna(train[target].mean()).to_numpy()
    cm_overall = regression_metrics(y_true, cm_pred)

    result = {
        "task": task,
        "overall_test_metrics": overall,
        "class_mean_baseline_test": cm_overall,
        "major_classes": major,
        "per_class": class_rows,
        "n_test_total": int(len(test)),
    }
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "class": test["class"].to_numpy()}).to_csv(
        output_dir / f"predictions_per_class_{task}.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / f"per_class_table_{task}.csv", index=False)
    (output_dir / f"summary_per_class_{task}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
