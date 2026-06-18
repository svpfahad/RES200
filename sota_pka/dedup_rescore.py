"""E9b — Re-score the external holdout with train-overlapping molecules removed.

The dedup audit (E9) found exact-structure train/external overlap only in
AvLiLuMoVe (2 molecules basic, 0 acidic) plus a few same-skeleton analogs. This
recomputes the external metrics with those molecules excluded, to confirm the E6
conclusions are not driven by contamination. It is feyn-free: the scoring step
preserves row order, so stored (y_true, y_pred) rows align 1:1 with the surviving
raw external rows (verified: 97/97 and 26/26). We rebuild each row's neutralized
canonical SMILES + InChIKey-connectivity, mask out train-overlapping rows, and
recompute on the stored predictions. Two exclusion levels:
  * exact      — drop rows whose neutral canonical SMILES is in OP2-train.
  * connectivity — also drop rows sharing an OP2-train 2D skeleton (strict).

Run:
    .venv_mac/bin/python -m sota_pka.dedup_rescore
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .external_holdout import _ext_root, _op2_dir, _standardizers, neutralize

RDLogger.DisableLog("rdApp.*")

PRED = Path(__file__).resolve().parents[1] / "sota_pka" / "runs" / "external_holdout" / "predictions"
OUT_DIR = Path(__file__).resolve().parents[1] / "sota_pka" / "runs" / "dedup_audit"
MODELS = ["qlattice", "lightgbm", "random_forest"]
EXT_SETS = {
    "acidic": ["novartis_acidic", "AvLiLuMoVe_123_acidic", "SAMPL7_acidic"],
    "basic": ["novartis_basic", "AvLiLuMoVe_123_basic"],
}


def _metrics(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    return {
        "n": int(len(yt)),
        "r2": float(r2_score(yt, yp)),
        "rmse": float(math.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "spearman": float(spearmanr(yt, yp).statistic),
        "pearson": float(pearsonr(yt, yp).statistic),
    }


def _train_keys(task: str):
    unch, frag = _standardizers()
    tr = pd.read_csv(_op2_dir() / f"Opt2_{task}_tr.csv")
    canon, conn = set(), set()
    for s in tr["OriginalSmiles"]:
        neu = neutralize(s, unch, frag)
        if neu is None:
            continue
        canon.add(neu)
        m = Chem.MolFromSmiles(neu)
        if m is not None:
            ik = Chem.MolToInchiKey(m)
            if ik:
                conn.add(ik.split("-")[0])
    return canon, conn


def _row_keys(name: str):
    """Per surviving-row (neutral canonical SMILES, connectivity), in scoring order."""
    unch, frag = _standardizers()
    raw = pd.read_csv(_ext_root() / f"{name}.csv")
    smi = raw["smiles"].astype(str)
    pka = pd.to_numeric(raw["pKa"], errors="coerce")
    canon, conn = [], []
    for s, p in zip(smi, pka):
        neu = neutralize(s, unch, frag)
        if neu is None or pd.isna(p):
            continue
        m = Chem.MolFromSmiles(neu)
        ik = Chem.MolToInchiKey(m).split("-")[0] if m is not None else None
        canon.append(neu)
        conn.append(ik)
    return canon, conn


def main() -> None:
    rows = []
    pooled = {lvl: {t: {m: ([], []) for m in MODELS} for t in EXT_SETS} for lvl in ("full", "exact", "conn")}

    for task, sets in EXT_SETS.items():
        tr_canon, tr_conn = _train_keys(task)
        for name in sets:
            canon, conn = _row_keys(name)
            n = len(canon)
            mask_exact = np.array([c not in tr_canon for c in canon])
            mask_conn = np.array([(c not in tr_canon) and (k not in tr_conn) for c, k in zip(canon, conn)])
            n_drop_exact = int((~mask_exact).sum())
            n_drop_conn = int((~mask_conn).sum())
            for model in MODELS:
                p = PRED / f"{name}__neutral__{model}.csv"
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                assert len(df) == n, f"alignment broke for {name}: {len(df)} vs {n}"
                yt, yp = df["y_true"].to_numpy(float), df["y_pred"].to_numpy(float)
                for lvl, mask in (("full", np.ones(n, bool)), ("exact", mask_exact), ("conn", mask_conn)):
                    mm = _metrics(yt[mask], yp[mask])
                    rows.append({"task": task, "set": name, "model": model, "level": lvl,
                                 "n_dropped": {"full": 0, "exact": n_drop_exact, "conn": n_drop_conn}[lvl], **mm})
                    pl_t, pl_p = pooled[lvl][task][model]
                    pl_t.append(yt[mask]); pl_p.append(yp[mask])

    # pooled per task/level
    for lvl in ("full", "exact", "conn"):
        for task in EXT_SETS:
            for model in MODELS:
                yts, yps = pooled[lvl][task][model]
                if not yts:
                    continue
                yt, yp = np.concatenate(yts), np.concatenate(yps)
                rows.append({"task": task, "set": "POOLED", "model": model, "level": lvl,
                             "n_dropped": "", **_metrics(yt, yp)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "dedup_rescore_metrics.csv", index=False)

    # Compact report: only sets that actually changed + pooled
    L = ["# E9b — External metrics with train-overlapping molecules removed\n"]
    L.append("Feyn-free recompute on stored predictions (row order verified 1:1). "
             "`full` = E6 as-reported · `exact` = drop exact-structure train overlaps · "
             "`conn` = also drop same-2D-skeleton analogs.\n")
    for task in EXT_SETS:
        L.append(f"\n## {task} — QLattice (symbolic) and ceiling\n")
        L.append("| set | model | level | n | dropped | R² | RMSE | Spearman |")
        L.append("|---|---|---|---:|---:|---:|---:|---:|")
        sub = df[(df["task"] == task)]
        for set_name in EXT_SETS[task] + ["POOLED"]:
            for model in MODELS:
                for lvl in ("full", "exact", "conn"):
                    r = sub[(sub["set"] == set_name) & (sub["model"] == model) & (sub["level"] == lvl)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    # only print exact/conn rows when they differ from full (dropped>0) or pooled
                    if lvl != "full" and set_name != "POOLED" and r["n_dropped"] in (0, "0", 0.0):
                        continue
                    L.append(f"| {set_name} | {model} | {lvl} | {int(r['n'])} | {r['n_dropped']} | "
                             f"{r['r2']:.3f} | {r['rmse']:.3f} | {r['spearman']:.3f} |")
    report = "\n".join(L) + "\n"
    (OUT_DIR / "RESULTS_dedup_rescore.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {OUT_DIR/'dedup_rescore_metrics.csv'} and report.")


if __name__ == "__main__":
    main()
