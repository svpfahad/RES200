"""Assemble the best leakage-safe per-class symbolic pKa model.

For each functional-group class, use its QLattice equation only when the
equation's cross-validated R2 clears a guard (else fall back to the class mean),
and clip every equation's predictions to that class's observed pKa range (symbolic
formulas can extrapolate to absurd values out-of-domain). Predictions are
assembled over the full held-out test set.

Reports: class-mean baseline, raw per-class, clipped per-class, and the final
clipped+guarded model — plus the PySR per-class assembly for comparison.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .qlattice_search import regression_metrics

PROC = Path(__file__).resolve().parent / "data" / "processed"
RUNS = Path(__file__).resolve().parent / "runs"
GUARD = 0.20  # trust an equation only if it explains >=20% of within-class CV variance
MARGIN = 1.0


def synthesize_task(task: str, run_subdir: str | None = None, data_suffix: str = "") -> dict:
    run_subdir = run_subdir or f"per_class_{task}_op2"
    train = pd.read_csv(PROC / f"{task}_train_classed{data_suffix}.csv", usecols=["class", "pKa"])
    test = pd.read_csv(PROC / f"{task}_test_classed{data_suffix}.csv", usecols=["class", "pKa"])
    cmeans = train.groupby("class")["pKa"].mean()
    rng = train.groupby("class")["pKa"].agg(["min", "max"])

    base = test["class"].map(cmeans).fillna(train["pKa"].mean()).to_numpy(dtype=float)
    raw = base.copy()
    clipped = base.copy()
    final = base.copy()  # clipped + CV-guarded

    rows = []
    qdir = RUNS / run_subdir
    for cdir in sorted(glob.glob(str(qdir / "class_*"))):
        c = Path(cdir).name.replace("class_", "")
        sfile = Path(cdir) / f"summary_{task}_{c}.json"
        pfile = Path(cdir) / f"predictions_qlattice_{task}_{c}.csv"
        if not sfile.exists() or not pfile.exists():
            continue
        s = json.loads(sfile.read_text())
        cv = s.get("cv_best_mean_r2")
        te_c = test[test["class"] == c]
        pred = pd.read_csv(pfile)["y_pred"].to_numpy(dtype=float)
        if len(pred) != len(te_c):
            continue
        lo, hi = rng.loc[c, "min"] - MARGIN, rng.loc[c, "max"] + MARGIN
        pred_clip = np.clip(pred, lo, hi)
        raw[te_c.index] = pred
        clipped[te_c.index] = pred_clip
        use_eq = cv is not None and cv >= GUARD
        if use_eq:
            final[te_c.index] = pred_clip
        rows.append({"class": c, "n_test": len(te_c), "cv_r2": cv,
                     "test_r2_raw": s["test_metrics"]["r2"],
                     "test_r2_clipped": regression_metrics(te_c["pKa"], pred_clip)["r2"],
                     "used": "equation" if use_eq else "class_mean",
                     "complexity": s.get("sympy_ops"), "formula": s.get("formula")})

    yt = test["pKa"].to_numpy(dtype=float)
    result = {
        "task": task,
        "class_mean_baseline": regression_metrics(yt, base),
        "per_class_raw": regression_metrics(yt, raw),
        "per_class_clipped": regression_metrics(yt, clipped),
        "per_class_clipped_guarded": regression_metrics(yt, final),
        "per_class_table": rows,
    }
    # PySR comparison (already assembled, no clipping there)
    pj = RUNS / f"pysr_per_class_{task}_op2" / f"summary_per_class_pysr_{task}.json"
    if pj.exists():
        result["pysr_per_class"] = json.loads(pj.read_text())["overall_test_metrics"]

    result["run_subdir"] = run_subdir
    (RUNS / f"synthesis_{run_subdir}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({"y_true": yt, "y_pred": final, "class": test["class"]}).to_csv(
        RUNS / f"predictions_best_{run_subdir}.csv", index=False)
    return result


def main() -> None:
    allres = []
    for task in ("acidic", "basic"):
        r = synthesize_task(task)
        allres.append(r)
        print(f"\n===== {task.upper()} =====")
        print(f"  class-mean baseline       R2={r['class_mean_baseline']['r2']:.4f}")
        print(f"  per-class (raw)           R2={r['per_class_raw']['r2']:.4f}")
        print(f"  per-class (clipped)       R2={r['per_class_clipped']['r2']:.4f}")
        print(f"  per-class (clip+guard) ** R2={r['per_class_clipped_guarded']['r2']:.4f} "
              f"RMSE={r['per_class_clipped_guarded']['rmse']:.4f}")
        if "pysr_per_class" in r:
            print(f"  [PySR per-class           R2={r['pysr_per_class']['r2']:.4f}]")
        print(f"  {'class':18s} {'n':>4s} {'cv':>6s} {'raw':>6s} {'clip':>6s}  used")
        for row in r["per_class_table"]:
            print(f"  {row['class']:18s} {row['n_test']:4d} {row['cv_r2']:6.3f} "
                  f"{row['test_r2_raw']:6.2f} {row['test_r2_clipped']:6.3f}  {row['used']}")
    (RUNS / "synthesis_combined.json").write_text(json.dumps(allres, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
