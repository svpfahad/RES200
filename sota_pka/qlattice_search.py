"""Leakage-safe QLattice symbolic-regression search for pKa modeling.

This module replaces the old test-set-selected grid search. Hyperparameters are
chosen by cross-validation on the training data only; the held-out test split is
evaluated exactly once with the chosen model.

Two modes:
  * ``direct``    - QLattice fits the experimental pKa labels.
  * ``distilled`` - QLattice fits the out-of-fold predictions of a strong
                    gradient-boosting teacher (knowledge distillation into a
                    compact symbolic surrogate). Accuracy is still reported
                    against the true experimental pKa; fidelity to the teacher
                    is reported separately.

Design choices that matter for publishability:
  * Correlation pruning (|Pearson r| > threshold) before ranking, so the
    symbolic search sees a de-correlated, stable feature pool.
  * Feature ranking with a LightGBM teacher on the TRAINING data only.
  * Feature names are sanitised to ``x0, x1, ...`` for sympy/Feyn and the real
    descriptor names are substituted back into the reported formula.
  * The final model is selected from the refit pool by training-data BIC
    (information criterion), never by test performance.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

logging.getLogger("feyn").setLevel(logging.WARNING)

TARGET_DEFAULT = "pKa"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Feature preparation
# --------------------------------------------------------------------------- #
def prepare_feature_frame(
    train: pd.DataFrame, test: pd.DataFrame, target: str = TARGET_DEFAULT
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Align numeric features, median-impute (train statistics), drop zero-variance."""
    y_train = pd.to_numeric(train[target], errors="coerce")
    y_test = pd.to_numeric(test[target], errors="coerce")

    x_train = train.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number]).copy()
    x_test = test.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number]).copy()
    x_train = x_train.loc[:, ~x_train.columns.duplicated()]
    x_test = x_test.loc[:, ~x_test.columns.duplicated()]

    common = [c for c in x_train.columns if c in x_test.columns]
    x_train, x_test = x_train[common], x_test[common]

    x_train = x_train.replace([np.inf, -np.inf], np.nan)
    x_test = x_test.replace([np.inf, -np.inf], np.nan)

    x_train = x_train.dropna(axis=1, how="all")
    x_test = x_test.reindex(columns=x_train.columns)

    medians = x_train.median(numeric_only=True)
    x_train = x_train.fillna(medians).fillna(0.0)
    x_test = x_test.fillna(medians).fillna(0.0)

    keep = x_train.std(axis=0) > 1e-12
    x_train = x_train.loc[:, keep]
    x_test = x_test[x_train.columns]

    mask_tr = y_train.notna().to_numpy()
    mask_te = y_test.notna().to_numpy()
    return (
        x_train.loc[mask_tr].reset_index(drop=True),
        x_test.loc[mask_te].reset_index(drop=True),
        y_train[mask_tr].reset_index(drop=True),
        y_test[mask_te].reset_index(drop=True),
    )


def prune_correlated(x: pd.DataFrame, y: pd.Series, threshold: float = 0.95) -> list[str]:
    """Greedy correlation pruning: keep the feature most correlated with y from
    each cluster of mutually |r|>threshold features."""
    corr = x.corr().abs()
    y_corr = x.apply(lambda c: abs(np.corrcoef(c, y)[0, 1]) if c.std() > 0 else 0.0)
    order = list(y_corr.sort_values(ascending=False).index)
    kept: list[str] = []
    dropped: set[str] = set()
    for col in order:
        if col in dropped:
            continue
        kept.append(col)
        redundant = corr.index[corr[col] > threshold]
        for other in redundant:
            if other != col and other not in kept:
                dropped.add(other)
    return kept


def rank_features(x: pd.DataFrame, y: pd.Series, n: int, seed: int = 0) -> tuple[list[str], pd.Series]:
    """Rank features by LightGBM importance (falls back to RandomForest)."""
    try:
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(x, y)
        importances = pd.Series(model.feature_importances_, index=x.columns, dtype=float)
    except Exception:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        model.fit(x, y)
        importances = pd.Series(model.feature_importances_, index=x.columns, dtype=float)
    importances = importances.sort_values(ascending=False)
    return importances.head(n).index.tolist(), importances


def _safe_names(cols: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    safe = {col: f"x{i}" for i, col in enumerate(cols)}
    inverse = {v: k for k, v in safe.items()}
    return safe, inverse


# --------------------------------------------------------------------------- #
# QLattice helpers
# --------------------------------------------------------------------------- #
def fit_qlattice(frame: pd.DataFrame, target: str, seed: int, epochs: int, max_complexity: int, criterion: str, threads: int):
    import feyn

    ql = feyn.QLattice(random_seed=seed)
    return ql.auto_run(
        frame,
        output_name=target,
        kind="regression",
        n_epochs=epochs,
        max_complexity=max_complexity,
        criterion=criterion,
        threads=threads,
    )


def model_formula(model, inverse: dict[str, str], signif: int = 4) -> tuple[str, int | None]:
    """Return (human-readable formula with real names, sympy op count)."""
    import sympy

    try:
        expr = model.sympify(signif=signif)
        try:
            expr = expr.as_expr()
        except Exception:
            pass
        text = str(expr)
        for safe in sorted(inverse, key=len, reverse=True):
            text = re.sub(rf"\b{safe}\b", inverse[safe], text)
        try:
            ops = int(sympy.count_ops(expr))
        except Exception:
            ops = None
        return text, ops
    except Exception:
        return str(model), None


def _train_bic(model, frame: pd.DataFrame, y_true: np.ndarray) -> float:
    """Schwarz BIC computed on the training frame; p = edges as free params."""
    pred = np.asarray(model.predict(frame), dtype=float)
    n = len(y_true)
    sse = float(np.sum((y_true - pred) ** 2))
    sse = max(sse, 1e-12)
    p = int(getattr(model, "edge_count", len(getattr(model, "features", []))) or 1)
    return n * math.log(sse / n) + p * math.log(n)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run_qlattice_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str | Path,
    target: str = TARGET_DEFAULT,
    mode: str = "direct",
    top_k_grid: tuple[int, ...] = (8, 12, 20),
    max_complexity: int = 30,
    criterion: str = "bic",
    cv_folds: int = 4,
    coarse_epochs: int = 40,
    refine_epochs: int = 120,
    refit_seeds: tuple[int, ...] = (0, 1, 2),
    corr_threshold: float = 0.95,
    threads: int = 8,
    teacher_factory: Callable[[], object] | None = None,
    label: str | None = None,
) -> dict:
    """Run one leakage-safe QLattice experiment and write artifacts.

    Returns a JSON-serialisable summary dict.
    """
    t_start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = label or mode

    x_train, x_test, y_train, y_test = prepare_feature_frame(train, test, target)

    kept = prune_correlated(x_train, y_train, corr_threshold)
    x_train, x_test = x_train[kept], x_test[kept]

    max_k = min(max(top_k_grid), x_train.shape[1])
    ranked, importances = rank_features(x_train, y_train, n=max_k)

    # Fit target: real labels (direct) or teacher OOF predictions (distilled).
    fit_target = y_train.to_numpy(dtype=float)
    teacher_test_pred = None
    if mode == "distilled":
        if teacher_factory is None:
            raise ValueError("distilled mode requires teacher_factory")
        teacher = teacher_factory()
        oof = cross_val_predict(teacher, x_train[ranked], y_train, cv=cv_folds, n_jobs=-1)
        fit_target = np.asarray(oof, dtype=float)
        teacher = teacher_factory()
        teacher.fit(x_train[ranked], y_train)
        teacher_test_pred = np.asarray(teacher.predict(x_test[ranked]), dtype=float)

    # ----- CV selection over top_k (evaluated vs TRUE pKa on the held-out fold) -----
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
    cv_rows: list[dict] = []
    pareto: list[dict] = []
    for k in top_k_grid:
        k = min(k, len(ranked))
        feats = ranked[:k]
        safe, inverse = _safe_names(feats)
        fold_best: list[float] = []
        for fold, (tri, vai) in enumerate(kf.split(x_train)):
            f_train = x_train.iloc[tri][feats].rename(columns=safe).copy()
            f_train[target] = fit_target[tri]
            f_val = x_train.iloc[vai][feats].rename(columns=safe)
            y_val_true = y_train.iloc[vai].to_numpy(dtype=float)
            models = fit_qlattice(f_train, target, seed=fold, epochs=coarse_epochs,
                                  max_complexity=max_complexity, criterion=criterion, threads=threads)
            best_r2 = -np.inf
            for mdl in models:
                pred = np.asarray(mdl.predict(f_val), dtype=float)
                r2 = r2_score(y_val_true, pred)
                _, ops = model_formula(mdl, inverse)
                pareto.append({"top_k": k, "complexity": ops, "edge_count": int(getattr(mdl, "edge_count", 0) or 0), "fold": fold, "val_r2": float(r2)})
                best_r2 = max(best_r2, r2)
            fold_best.append(float(best_r2))
        cv_rows.append({
            "top_k": k,
            "cv_r2_mean": float(np.mean(fold_best)),
            "cv_r2_std": float(np.std(fold_best)),
            "folds": fold_best,
        })

    cv_df = pd.DataFrame(cv_rows)
    best_idx = cv_df["cv_r2_mean"].idxmax()
    best_mean = float(cv_df.loc[best_idx, "cv_r2_mean"])
    best_se = float(cv_df.loc[best_idx, "cv_r2_std"]) / math.sqrt(cv_folds)
    eligible = cv_df[cv_df["cv_r2_mean"] >= best_mean - best_se].sort_values("top_k")
    chosen_k = int(eligible.iloc[0]["top_k"])

    # ----- Refit on full train (chosen_k), multiple seeds, more epochs -----
    feats = ranked[:chosen_k]
    safe, inverse = _safe_names(feats)
    full = x_train[feats].rename(columns=safe).copy()
    full[target] = fit_target
    y_train_true = y_train.to_numpy(dtype=float)

    candidates = []
    for seed in refit_seeds:
        models = fit_qlattice(full, target, seed=seed, epochs=refine_epochs,
                              max_complexity=max_complexity, criterion=criterion, threads=threads)
        for mdl in models[:3]:  # keep the top few per seed for BIC selection
            candidates.append(mdl)

    # Final model selected by TRAIN BIC (leakage-safe; balances fit and complexity).
    final = min(candidates, key=lambda m: _train_bic(m, full, y_train_true))

    # ----- Single test evaluation -----
    test_frame = x_test[feats].rename(columns=safe)
    test_pred = np.asarray(final.predict(test_frame), dtype=float)
    train_pred = np.asarray(final.predict(full), dtype=float)

    formula, ops = model_formula(final, inverse)
    test_metrics = regression_metrics(y_test.to_numpy(), test_pred)
    train_metrics = regression_metrics(y_train_true, train_pred)

    summary: dict = {
        "label": label,
        "mode": mode,
        "criterion": criterion,
        "max_complexity": max_complexity,
        "corr_threshold": corr_threshold,
        "n_features_pruned_pool": int(len(kept)),
        "chosen_top_k": chosen_k,
        "selected_features": feats,
        "edge_count": int(getattr(final, "edge_count", 0) or 0),
        "sympy_ops": ops,
        "formula": formula,
        "cv_selection": cv_rows,
        "cv_best_mean_r2": best_mean,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    if mode == "distilled" and teacher_test_pred is not None:
        summary["teacher_test_metrics"] = regression_metrics(y_test.to_numpy(), teacher_test_pred)
        summary["distill_fidelity_r2_test"] = float(r2_score(teacher_test_pred, test_pred))

    # ----- Artifacts -----
    pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": test_pred, "mode": mode}).to_csv(
        output_dir / f"predictions_qlattice_{label}.csv", index=False
    )
    cv_df.assign(folds=cv_df["folds"].map(json.dumps)).to_csv(output_dir / f"cv_selection_{label}.csv", index=False)
    pd.DataFrame(pareto).to_csv(output_dir / f"pareto_{label}.csv", index=False)
    importances.rename("importance").to_csv(output_dir / f"feature_importance_{label}.csv")
    (output_dir / f"formula_{label}.txt").write_text(formula + "\n", encoding="utf-8")
    (output_dir / f"summary_{label}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
