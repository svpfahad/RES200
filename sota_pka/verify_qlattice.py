"""Adversarial verification of the QLattice study.

Independent re-checks (nothing trusts the run's own numbers):
  1. Recompute R2/RMSE/MAE from every predictions_*.csv with sklearn.
  2. Confirm held-out test sizes (acidic=765, basic=843).
  3. Duplicate-molecule leakage audit between train/test descriptor matrices.
  4. Y-randomization controls are ~0.
  5. Formula faithfulness: evaluate the *published* sympy formula on the test
     descriptors and confirm it reproduces the saved model predictions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sympy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sota_pka.qlattice_search import prepare_feature_frame

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "RES 200-20260312T035531Z-1-001" / "RES 200"
RUNS = Path(__file__).resolve().parent / "runs"
FILES = {
    "acidic": (DATA / "train_descriptors_op2.csv", DATA / "test_descriptors_op2.csv", 765),
    "basic": (DATA / "train_descriptors_basic_op2.csv", DATA / "test_descriptors_basic_op2.csv", 843),
}

ok = []
warn = []


def check(cond, msg):
    (ok if cond else warn).append(("PASS" if cond else "FAIL") + " | " + msg)


def metrics(yt, yp):
    return r2_score(yt, yp), float(np.sqrt(mean_squared_error(yt, yp))), mean_absolute_error(yt, yp)


# 1 + 2: recompute metrics and test sizes -----------------------------------
combined = json.loads((RUNS / "qlattice_combined_summary.json").read_text())
for s in combined:
    task, mode = s["task"], s["mode"]
    pred = pd.read_csv(RUNS / f"qlattice_{task}_op2" / f"predictions_qlattice_{task}_{mode}.csv")
    r2, rmse, mae = metrics(pred["y_true"], pred["y_pred"])
    rep = s["test_metrics"]
    check(abs(r2 - rep["r2"]) < 1e-6, f"{task}/{mode} R2 recompute {r2:.4f} == reported {rep['r2']:.4f}")
    check(abs(rmse - rep["rmse"]) < 1e-6, f"{task}/{mode} RMSE recompute {rmse:.4f} == reported {rep['rmse']:.4f}")
    check(len(pred) == FILES[task][2], f"{task}/{mode} test n={len(pred)} == expected {FILES[task][2]}")

# 3: duplicate-molecule leakage between train/test ---------------------------
for task, (tr, te, _) in FILES.items():
    a = pd.read_csv(tr).loc[:, lambda d: ~d.columns.duplicated()]
    b = pd.read_csv(te).loc[:, lambda d: ~d.columns.duplicated()]
    cols = [c for c in a.columns if c in set(b.columns) and c != "pKa"]

    def keyset(df):
        arr = np.round(df[cols].to_numpy(dtype=float), 4)
        return {hash(row.tobytes()) for row in arr}

    overlap = keyset(a) & keyset(b)
    check(len(overlap) == 0, f"{task}: train/test descriptor-row overlap = {len(overlap)} (expect 0)")

# 4: y-randomization ~0 ------------------------------------------------------
for task in FILES:
    p = RUNS / f"qlattice_{task}_op2" / "y_randomization_qlattice.json"
    if p.exists():
        yr = json.loads(p.read_text())
        check(abs(yr["r2"]) < 0.05, f"{task}: y-randomization R2={yr['r2']:.4f} (~0 expected)")

# 5: formula faithfulness ----------------------------------------------------
for s in combined:
    if s["mode"] != "direct":
        continue
    task = s["task"]
    tr, te, _ = FILES[task]
    train, test = pd.read_csv(tr), pd.read_csv(te)
    x_train, x_test, y_train, y_test = prepare_feature_frame(train, test)
    feats = s["selected_features"]
    missing = [f for f in feats if f not in x_test.columns]
    if missing:
        warn.append(f"FAIL | {task}: features missing from test frame: {missing}")
        continue
    formula = s["formula"]
    # Replace each (possibly hyphenated) descriptor name with a clean symbol.
    sym = {f: f"v{i}" for i, f in enumerate(feats)}
    disp = formula
    for f in sorted(feats, key=len, reverse=True):
        disp = disp.replace(f, sym[f])
    # any leftover hyphen that is not a minus between numbers/identifiers is a problem
    try:
        expr = sympy.sympify(disp, locals={v: sympy.Symbol(v) for v in sym.values()})
        symbols = [sympy.Symbol(sym[f]) for f in feats]
        fn = sympy.lambdify(symbols, expr, modules=["numpy"])
        X = [x_test[f].to_numpy(dtype=float) for f in feats]
        formula_pred = np.asarray(fn(*X), dtype=float)
        saved = pd.read_csv(RUNS / f"qlattice_{task}_op2" / f"predictions_qlattice_{task}_direct.csv")["y_pred"].to_numpy()
        m = min(len(formula_pred), len(saved))
        faith = r2_score(saved[:m], formula_pred[:m])
        check(faith > 0.99, f"{task}: published-formula vs model-prediction R2={faith:.4f} (>0.99 = faithful)")
    except Exception as exc:  # noqa: BLE001
        warn.append(f"FAIL | {task}: formula could not be evaluated symbolically: {exc}")

print("\n".join(ok))
print("\n".join(warn) if warn else "\n(no failures)")
print(f"\n{len(ok)} passed, {len(warn)} failed/flagged")
