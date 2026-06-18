"""Orchestrate the full leakage-safe QLattice study.

Runs, for each task (acidic, basic) and each mode (direct, distilled):
  * leakage-safe CV-selected QLattice symbolic regression,
  * a QLattice Y-randomization control (direct mode),
and writes a combined summary plus a benchmark table merging the GBM baselines
with the symbolic models.

Usage (from RES200 root, with the macOS venv):
    .venv_mac/bin/python -m sota_pka.run_qlattice_all
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .qlattice_search import (
    fit_qlattice,
    prepare_feature_frame,
    prune_correlated,
    rank_features,
    regression_metrics,
    run_qlattice_experiment,
    _safe_names,
)

logging.getLogger("feyn").setLevel(logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "RES 200-20260312T035531Z-1-001" / "RES 200"
RUNS = Path(__file__).resolve().parent / "runs"

TASKS = {
    "acidic": {
        "train": DATA / "train_descriptors_op2.csv",
        "test": DATA / "test_descriptors_op2.csv",
        "baseline_metrics": RUNS / "res200_op2_full_descriptors_wsl_full" / "metrics.csv",
    },
    "basic": {
        "train": DATA / "train_descriptors_basic_op2.csv",
        "test": DATA / "test_descriptors_basic_op2.csv",
        "baseline_metrics": RUNS / "res200_basic_op2_full_descriptors" / "metrics.csv",
    },
}

# Search configuration (sized for an overnight-safe local CPU run).
GRID = dict(
    top_k_grid=(8, 12, 20),
    max_complexity=30,
    criterion="bic",
    cv_folds=4,
    coarse_epochs=35,
    refine_epochs=100,
    refit_seeds=(0, 1, 2),
    corr_threshold=0.95,
    threads=14,
)


def lgbm_teacher_factory():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(n_estimators=600, learning_rate=0.03, random_state=42, n_jobs=-1, verbose=-1)


def qlattice_y_randomization(train, test, output_dir, chosen_k, target="pKa"):
    """Fit QLattice on label-shuffled training data; expect R2 ~ 0 on test."""
    x_train, x_test, y_train, y_test = prepare_feature_frame(train, test, target)
    kept = prune_correlated(x_train, y_train, GRID["corr_threshold"])
    x_train, x_test = x_train[kept], x_test[kept]
    ranked, _ = rank_features(x_train, y_train, n=min(chosen_k, x_train.shape[1]))
    feats = ranked[:chosen_k]
    safe, _ = _safe_names(feats)
    rng = np.random.default_rng(0)
    y_shuffled = rng.permutation(y_train.to_numpy(dtype=float))
    frame = x_train[feats].rename(columns=safe).copy()
    frame[target] = y_shuffled
    models = fit_qlattice(frame, target, seed=0, epochs=GRID["refine_epochs"],
                          max_complexity=GRID["max_complexity"], criterion=GRID["criterion"],
                          threads=GRID["threads"])
    test_frame = x_test[feats].rename(columns=safe)
    pred = np.asarray(models[0].predict(test_frame), dtype=float)
    metrics = regression_metrics(y_test.to_numpy(), pred)
    out = {"control": "y_randomization_qlattice", "chosen_k": chosen_k, **metrics}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "y_randomization_qlattice.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    summaries: list[dict] = []
    for task, cfg in TASKS.items():
        if not Path(cfg["train"]).exists() or not Path(cfg["test"]).exists():
            print(f"[skip] {task}: descriptor files missing ({cfg['train']})")
            continue
        train = pd.read_csv(cfg["train"])
        test = pd.read_csv(cfg["test"])
        run_dir = RUNS / f"qlattice_{task}_op2"
        chosen_k_direct = None
        for mode in ("direct", "distilled"):
            print(f"\n===== {task} / {mode} =====", flush=True)
            try:
                summary = run_qlattice_experiment(
                    train, test, output_dir=run_dir, mode=mode,
                    teacher_factory=lgbm_teacher_factory if mode == "distilled" else None,
                    label=f"{task}_{mode}", **GRID,
                )
                summary["task"] = task
                summaries.append(summary)
                if mode == "direct":
                    chosen_k_direct = summary["chosen_top_k"]
                print(f"  -> test R2={summary['test_metrics']['r2']:.4f} "
                      f"RMSE={summary['test_metrics']['rmse']:.4f} k={summary['chosen_top_k']} "
                      f"ops={summary['sympy_ops']} ({summary['elapsed_sec']}s)", flush=True)
                # Persist incrementally so a later crash keeps earlier results.
                (RUNS / "qlattice_combined_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                import traceback
                print(f"  !! {task}/{mode} FAILED: {exc}", flush=True)
                traceback.print_exc()

        # Y-randomization control (direct features).
        print(f"\n----- {task} / y-randomization -----", flush=True)
        try:
            yr = qlattice_y_randomization(train, test, run_dir, chosen_k_direct or 12)
            print(f"  -> y-rand test R2={yr['r2']:.4f} (expect ~0)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {task}/y-randomization FAILED: {exc}", flush=True)

    (RUNS / "qlattice_combined_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    # Combined benchmark rows.
    rows = []
    for s in summaries:
        rows.append({
            "task": s["task"], "model": f"qlattice_{s['mode']}",
            "test_r2": s["test_metrics"]["r2"], "test_rmse": s["test_metrics"]["rmse"],
            "test_mae": s["test_metrics"]["mae"], "n_features": s["chosen_top_k"],
            "complexity_ops": s["sympy_ops"], "edge_count": s["edge_count"],
        })
    pd.DataFrame(rows).to_csv(RUNS / "qlattice_benchmark_rows.csv", index=False)
    print("\nWrote combined summary + benchmark rows to", RUNS)


if __name__ == "__main__":
    main()
