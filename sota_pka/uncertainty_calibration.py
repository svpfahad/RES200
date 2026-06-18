"""E7 — Uncertainty bands + out-of-distribution (OOD) calibration analysis.

Everything here is computed from *already-recorded predictions* — no model is
re-fit, so there is zero leakage risk and the numbers are exactly the recorded
models' behavior. Two questions:

  (1) Uncertainty band on the reported metrics. The single held-out test gives a
      point R^2; we attach a nonparametric bootstrap 95% CI (resample the stored
      (y_true, y_pred) pairs). For the external sets — which are small — this CI
      is the honest statement of how much the headline numbers can move.

  (2) OOD calibration decomposition. The E6 audit showed the symbolic models
      carry above-chance *rank* signal externally (Spearman 0.27-0.58) but poor
      R^2 (negative for basic). Is that a fixable *miscalibration* (wrong
      scale/offset) or a genuine loss of predictive association?  We answer it by
      fitting a 2-parameter affine map  y ~ a*y_pred + b  with K-fold
      cross-validation *within each external set* (strictly out-of-fold, so the
      recalibrated R^2 is honest), and compare:
         R2_raw           — as scored by the OP2-trained model
         R2_recal (CV)    — after the OOF affine correction
         R2_affine_ceiling= Pearson r^2 — the in-sample upper bound of any affine map
         Spearman         — invariant to a monotone affine map (sanity anchor)
      If R2_recal >> R2_raw the external failure is dominated by miscalibration
      (recoverable from a few external anchors); the residual gap from
      R2_affine_ceiling to the in-distribution R^2 is irreducible loss of linear
      association. This separates "needs recalibration" from "lost signal".

Run:
    .venv_mac/bin/python -m sota_pka.uncertainty_calibration
    .venv_mac/bin/python -m sota_pka.uncertainty_calibration --boot 20000
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


RUNS = _repo_root() / "sota_pka" / "runs"
EXT_PRED = RUNS / "external_holdout" / "predictions"
OUT_DIR = RUNS / "uncertainty_calibration"

# In-distribution symbolic test predictions + their recorded CV selection.
INDIST = {
    "acidic": {
        "pred": RUNS / "qlattice_acidic_op2" / "predictions_qlattice_acidic_direct.csv",
        "summary": RUNS / "qlattice_acidic_op2" / "summary_acidic_direct.json",
    },
    "basic": {
        "pred": RUNS / "qlattice_basic_op2" / "predictions_qlattice_basic_direct.csv",
        "summary": RUNS / "qlattice_basic_op2" / "summary_basic_direct.json",
    },
}

# External sets grouped by task (neutral representation is the valid one).
EXT_SETS = {
    "acidic": ["novartis_acidic", "AvLiLuMoVe_123_acidic", "SAMPL7_acidic"],
    "basic": ["novartis_basic", "AvLiLuMoVe_123_basic"],
}
MODELS = ["qlattice", "lightgbm", "random_forest", "y_random_lightgbm"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    out = {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    out["spearman"] = float(spearmanr(y_true, y_pred).statistic) if len(y_true) > 2 else float("nan")
    out["pearson"] = float(pearsonr(y_true, y_pred).statistic) if len(y_true) > 2 else float("nan")
    return out


def bootstrap_ci(
    y_true: np.ndarray, y_pred: np.ndarray, b: int = 10000, seed: int = 0, alpha: float = 0.05
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CI for r2/rmse/mae/spearman by resampling pairs."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ("r2", "rmse", "mae", "spearman")}
    for _ in range(b):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if np.std(yt) < 1e-12:  # degenerate resample
            continue
        acc["r2"].append(r2_score(yt, yp))
        acc["rmse"].append(math.sqrt(mean_squared_error(yt, yp)))
        acc["mae"].append(mean_absolute_error(yt, yp))
        acc["spearman"].append(spearmanr(yt, yp).statistic if np.std(yp) > 1e-12 else 0.0)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for k, v in acc.items() if v}


def cv_affine_recalibration(
    y_true: np.ndarray, y_pred: np.ndarray, seed: int = 0
) -> dict[str, float]:
    """Out-of-fold affine recalibration y ~ a*y_pred + b.

    K = 5 for n >= 40, else leave-one-out (so tiny sets still get an honest OOF
    estimate). Returns raw vs recalibrated R^2, the affine ceiling (Pearson r^2),
    and the mean fitted (a, b)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    k = 5 if n >= 40 else n  # LOO for small sets
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan)
    a_list, b_list = [], []
    for tri, tei in kf.split(y_pred):
        xt, yt = y_pred[tri], y_true[tri]
        if np.std(xt) < 1e-12:
            a, b = 0.0, float(np.mean(yt))
        else:
            a, b = np.polyfit(xt, yt, 1)
        oof[tei] = a * y_pred[tei] + b
        a_list.append(float(a))
        b_list.append(float(b))
    pear = pearsonr(y_true, y_pred).statistic if np.std(y_pred) > 1e-12 else 0.0
    return {
        "n": int(n),
        "k_folds": int(k),
        "r2_raw": float(r2_score(y_true, y_pred)),
        "r2_recal_cv": float(r2_score(y_true, oof)),
        "r2_affine_ceiling": float(pear ** 2),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "a_mean": float(np.mean(a_list)),
        "b_mean": float(np.mean(b_list)),
    }


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def _load_pred(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df["y_true"].to_numpy(float), df["y_pred"].to_numpy(float)


def _load_pooled(task: str, model: str, rep: str = "neutral") -> tuple[np.ndarray, np.ndarray]:
    yts, yps = [], []
    for name in EXT_SETS[task]:
        p = EXT_PRED / f"{name}__{rep}__{model}.csv"
        if p.exists():
            yt, yp = _load_pred(p)
            yts.append(yt)
            yps.append(yp)
    return np.concatenate(yts), np.concatenate(yps)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(prog="uncertainty_calibration")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    boot_rows: list[dict] = []
    recal_rows: list[dict] = []

    # ---- (1) In-distribution uncertainty band on the symbolic test R^2 ----
    indist_lines = []
    for task, cfg in INDIST.items():
        yt, yp = _load_pred(cfg["pred"])
        m = _metrics(yt, yp)
        ci = bootstrap_ci(yt, yp, b=args.boot, seed=args.seed)
        summ = json.loads(Path(cfg["summary"]).read_text())
        chosen_k = summ["chosen_top_k"]
        cv_sel = {r["top_k"]: r for r in summ["cv_selection"]}[chosen_k]
        cv_mean, cv_std = cv_sel["cv_r2_mean"], cv_sel["cv_r2_std"]
        cv_se = cv_std / math.sqrt(len(cv_sel["folds"]))
        boot_rows.append({
            "scope": "in_distribution", "task": task, "model": "qlattice", "n": m["n"],
            "r2": m["r2"], "r2_lo": ci["r2"][0], "r2_hi": ci["r2"][1],
            "rmse": m["rmse"], "rmse_lo": ci["rmse"][0], "rmse_hi": ci["rmse"][1],
            "spearman": m["spearman"], "spearman_lo": ci["spearman"][0], "spearman_hi": ci["spearman"][1],
            "cv_r2_mean": cv_mean, "cv_r2_se": cv_se,
        })
        indist_lines.append(
            f"| {task} | {m['n']} | {m['r2']:.3f} | [{ci['r2'][0]:.3f}, {ci['r2'][1]:.3f}] | "
            f"{m['rmse']:.3f} | [{ci['rmse'][0]:.3f}, {ci['rmse'][1]:.3f}] | "
            f"{cv_mean:.3f} ± {1.96*cv_se:.3f} |"
        )

    # ---- (2) External: bootstrap CI + OOD affine recalibration ----
    for task in EXT_SETS:
        # per-set
        for name in EXT_SETS[task]:
            for model in MODELS:
                p = EXT_PRED / f"{name}__neutral__{model}.csv"
                if not p.exists():
                    continue
                yt, yp = _load_pred(p)
                m = _metrics(yt, yp)
                ci = bootstrap_ci(yt, yp, b=args.boot, seed=args.seed)
                boot_rows.append({
                    "scope": "external_set", "task": task, "set": name, "model": model, "n": m["n"],
                    "r2": m["r2"], "r2_lo": ci["r2"][0], "r2_hi": ci["r2"][1],
                    "rmse": m["rmse"], "rmse_lo": ci["rmse"][0], "rmse_hi": ci["rmse"][1],
                    "spearman": m["spearman"], "spearman_lo": ci["spearman"][0], "spearman_hi": ci["spearman"][1],
                })
                if model in ("qlattice", "lightgbm", "random_forest"):
                    r = cv_affine_recalibration(yt, yp, seed=args.seed)
                    r.update({"scope": "external_set", "task": task, "set": name, "model": model})
                    recal_rows.append(r)
        # pooled per task
        for model in MODELS:
            yt, yp = _load_pooled(task, model)
            m = _metrics(yt, yp)
            ci = bootstrap_ci(yt, yp, b=args.boot, seed=args.seed)
            boot_rows.append({
                "scope": "external_pooled", "task": task, "set": "POOLED", "model": model, "n": m["n"],
                "r2": m["r2"], "r2_lo": ci["r2"][0], "r2_hi": ci["r2"][1],
                "rmse": m["rmse"], "rmse_lo": ci["rmse"][0], "rmse_hi": ci["rmse"][1],
                "spearman": m["spearman"], "spearman_lo": ci["spearman"][0], "spearman_hi": ci["spearman"][1],
            })
            if model in ("qlattice", "lightgbm", "random_forest"):
                r = cv_affine_recalibration(yt, yp, seed=args.seed)
                r.update({"scope": "external_pooled", "task": task, "set": "POOLED", "model": model})
                recal_rows.append(r)

    boot_df = pd.DataFrame(boot_rows)
    recal_df = pd.DataFrame(recal_rows)
    boot_df.to_csv(OUT_DIR / "bootstrap_ci.csv", index=False)
    recal_df.to_csv(OUT_DIR / "ood_recalibration.csv", index=False)

    # ---- Report ----
    L: list[str] = ["# E7 — Uncertainty bands & OOD calibration decomposition\n"]
    L.append("*Computed from recorded predictions only (no refit). Bootstrap = "
             f"{args.boot} resamples, percentile 95% CI.*\n")
    L.append("## In-distribution symbolic test R² (with bootstrap 95% CI + CV band)\n")
    L.append("| task | n | test R² | R² 95% CI | RMSE | RMSE 95% CI | CV R² (mean ± 1.96·SE) |")
    L.append("|---|---:|---:|---|---:|---|---|")
    L.extend(indist_lines)

    L.append("\n## External OOD calibration decomposition (symbolic vs ceiling)\n")
    L.append("`r2_raw` = OP2-model as-is · `r2_recal_cv` = after out-of-fold affine "
             "correction · `r2_affine_ceiling` = Pearson r² (best any affine map can do) "
             "· Spearman is affine-invariant.\n")
    L.append("| task | set | model | n | R²_raw | R²_recal(CV) | R²_affine_ceiling | Spearman | a | b |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    order = {"qlattice": 0, "lightgbm": 1, "random_forest": 2}
    for _, r in recal_df.sort_values(
        ["task", "scope", "set", "model"], key=lambda s: s.map(lambda v: order.get(v, v) if isinstance(v, str) else v)
    ).iterrows():
        L.append(
            f"| {r['task']} | {r['set']} | {r['model']} | {int(r['n'])} | "
            f"{r['r2_raw']:.3f} | {r['r2_recal_cv']:.3f} | {r['r2_affine_ceiling']:.3f} | "
            f"{r['spearman']:.3f} | {r['a_mean']:.3f} | {r['b_mean']:.3f} |"
        )

    report = "\n".join(L) + "\n"
    (OUT_DIR / "RESULTS_uncertainty_calibration.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {OUT_DIR/'bootstrap_ci.csv'}, {OUT_DIR/'ood_recalibration.csv'}, and report.")


if __name__ == "__main__":
    main()
