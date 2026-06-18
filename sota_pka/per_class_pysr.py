"""Deep PySR symbolic regression for pKa — per-class and global.

PySR (Julia SymbolicRegression.jl) runs a multi-objective evolutionary search and
typically finds stronger accuracy-vs-complexity tradeoffs than QLattice. This
module runs it leakage-safe: train-only feature selection, equation chosen by
PySR's training-data score (loss + parsimony), held-out test scored once.

Equations are kept in ORIGINAL descriptor units (no standardisation) with a
rational + square + tanh operator basis, which suits pKa (already a log quantity).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .qlattice_search import prepare_feature_frame, prune_correlated, rank_features, regression_metrics

META_COLS = ["smiles", "class"]

PROCS = int(os.environ.get("PYSR_PROCS", "14"))

HEAVY = dict(
    niterations=120,
    populations=28,
    population_size=40,
    ncycles_per_iteration=400,
    maxsize=28,
    top_k=12,
    parsimony=0.0,
    adaptive_parsimony_scaling=1000.0,
    weight_optimize=0.001,
)
SMOKE = dict(niterations=20, populations=8, population_size=30, ncycles_per_iteration=300,
             maxsize=20, top_k=8, parsimony=0.0, adaptive_parsimony_scaling=1000.0, weight_optimize=0.001)


def _features_only(df, target):
    return df.drop(columns=[c for c in META_COLS if c in df.columns], errors="ignore")


def fit_pysr_one(train_df, test_df, output_dir, label, target="pKa", cfg=None):
    """Fit PySR on one (train, test) frame; return (summary, test_pred Series)."""
    from pysr import PySRRegressor

    cfg = cfg or HEAVY
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = prepare_feature_frame(_features_only(train_df, target),
                                                             _features_only(test_df, target), target)
    kept = prune_correlated(x_train, y_train, 0.95)
    x_train, x_test = x_train[kept], x_test[kept]
    feats, _ = rank_features(x_train, y_train, n=min(cfg["top_k"], x_train.shape[1]))
    # PySR needs identifier-safe variable names
    safe = {f: f"f{i}" for i, f in enumerate(feats)}
    inv = {v: k for k, v in safe.items()}

    model = PySRRegressor(
        niterations=cfg["niterations"],
        populations=cfg["populations"],
        population_size=cfg["population_size"],
        ncycles_per_iteration=cfg["ncycles_per_iteration"],
        maxsize=cfg["maxsize"],
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "cube", "tanh"],
        model_selection="best",
        parsimony=cfg["parsimony"],
        adaptive_parsimony_scaling=cfg["adaptive_parsimony_scaling"],
        weight_optimize=cfg["weight_optimize"],
        parallelism="serial",
        random_state=0,
        deterministic=False,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
        delete_tempfiles=True,
    )
    Xtr = x_train[feats].rename(columns=safe)
    Xte = x_test[feats].rename(columns=safe)
    model.fit(Xtr.to_numpy(), y_train.to_numpy(), variable_names=list(Xtr.columns))

    test_pred = np.asarray(model.predict(Xte.to_numpy()), dtype=float)
    train_pred = np.asarray(model.predict(Xtr.to_numpy()), dtype=float)
    best = model.get_best()
    eq = str(best["equation"])
    for s in sorted(inv, key=len, reverse=True):
        eq = eq.replace(s, inv[s])

    # Pareto front + oracle (best-on-test, diagnostic only, NOT selected)
    eqs = model.equations_.copy()
    oracle_r2 = None
    try:
        preds_all = [np.asarray(model.predict(Xte.to_numpy(), index=i), dtype=float) for i in eqs.index]
        oracle_r2 = max(regression_metrics(y_test.to_numpy(), p)["r2"] for p in preds_all)
    except Exception:
        pass
    eqs.to_csv(output_dir / f"pareto_pysr_{label}.csv", index=False)

    summary = {
        "label": label, "engine": "pysr",
        "selected_features": feats,
        "complexity": int(best["complexity"]),
        "loss": float(best["loss"]),
        "formula": eq,
        "train_metrics": regression_metrics(y_train.to_numpy(), train_pred),
        "test_metrics": regression_metrics(y_test.to_numpy(), test_pred),
        "oracle_best_on_test_r2": oracle_r2,
    }
    pd.DataFrame({"y_true": y_test.to_numpy(), "y_pred": test_pred}).to_csv(
        output_dir / f"predictions_pysr_{label}.csv", index=False)
    (output_dir / f"summary_pysr_{label}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, pd.Series(test_pred, index=test_df.index)


def run_per_class_pysr(train, test, output_dir, task, target="pKa", min_train=60, min_test=12, cfg=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_counts = train["class"].value_counts()
    test_counts = test["class"].value_counts()
    major = [c for c in train_counts.index
             if train_counts.get(c, 0) >= min_train and test_counts.get(c, 0) >= min_test]

    test_pred = pd.Series(np.nan, index=test.index, dtype=float)
    class_rows = []
    class_means = train.groupby("class")[target].mean()

    for c in major:
        tr_c, te_c = train[train["class"] == c], test[test["class"] == c]
        try:
            summ, preds = fit_pysr_one(tr_c, te_c, output_dir / f"class_{c}", f"{task}_{c}", target, cfg)
            test_pred.loc[preds.index] = preds.to_numpy()
            m = summ["test_metrics"]
            class_rows.append({"class": c, "n_train": len(tr_c), "n_test": len(te_c), "test_r2": m["r2"],
                               "test_rmse": m["rmse"], "test_mae": m["mae"], "complexity": summ["complexity"],
                               "formula": summ["formula"], "oracle_r2": summ["oracle_best_on_test_r2"],
                               "source": "per_class_pysr"})
        except Exception as exc:  # noqa: BLE001
            mu = float(class_means.get(c, train[target].mean()))
            test_pred.loc[te_c.index] = mu
            class_rows.append({"class": c, "n_train": len(tr_c), "n_test": len(te_c), "source": "class_mean_fallback",
                               "formula": f"{target}={mu:.3f} (PySR failed: {exc})"})

    minor = test_pred.isna()
    if minor.any():
        try:
            summ, preds = fit_pysr_one(train, test[minor], output_dir / "fallback_global", f"{task}_fallback", target, cfg)
            test_pred.loc[preds.index] = preds.to_numpy()
            class_rows.append({"class": "<rare:global_fallback>", "n_train": len(train), "n_test": int(minor.sum()),
                               "test_r2": summ["test_metrics"]["r2"], "complexity": summ["complexity"],
                               "formula": summ["formula"], "source": "global_fallback"})
        except Exception:
            test_pred.loc[minor] = float(train[target].mean())

    y_true = test[target].to_numpy(dtype=float)
    overall = regression_metrics(y_true, test_pred.to_numpy(dtype=float))
    cm = test["class"].map(class_means).fillna(train[target].mean()).to_numpy()
    result = {"task": task, "engine": "pysr", "overall_test_metrics": overall,
              "class_mean_baseline_test": regression_metrics(y_true, cm), "major_classes": major,
              "per_class": class_rows, "n_test_total": int(len(test))}
    pd.DataFrame({"y_true": y_true, "y_pred": test_pred.to_numpy(), "class": test["class"].to_numpy()}).to_csv(
        output_dir / f"predictions_per_class_pysr_{task}.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / f"per_class_pysr_table_{task}.csv", index=False)
    (output_dir / f"summary_per_class_pysr_{task}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
