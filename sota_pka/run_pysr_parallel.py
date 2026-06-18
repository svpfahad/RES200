"""Parallel PySR campaign: one serial-PySR process per (task, class) fit.

Avoids Julia Distributed entirely (which errors on this setup). Class-level
parallelism via a spawn ProcessPool saturates the cores robustly. The global fit
predicts all test rows and doubles as the rare-class fallback.

    .venv_mac/bin/python -m sota_pka.run_pysr_parallel
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROC = Path(__file__).resolve().parent / "data" / "processed"
RUNS = Path(__file__).resolve().parent / "runs"
MIN_TRAIN, MIN_TEST = 60, 12
MAX_WORKERS = 8


def _pysr_job(task: str, cls: str) -> dict:
    """Child process: fit one serial-PySR model."""
    import warnings as _w
    _w.filterwarnings("ignore")
    from .per_class_pysr import HEAVY, fit_pysr_one

    train = pd.read_csv(PROC / f"{task}_train_classed.csv")
    test = pd.read_csv(PROC / f"{task}_test_classed.csv")
    if cls == "__global__":
        tr, te = train, test
        out = RUNS / f"pysr_global_{task}_op2"
        label = f"{task}_global"
    else:
        tr, te = train[train["class"] == cls], test[test["class"] == cls]
        out = RUNS / f"pysr_per_class_{task}_op2" / f"class_{cls}"
        label = f"{task}_{cls}"
    try:
        summ, _ = fit_pysr_one(tr, te, out, label, cfg=HEAVY)
        return {"task": task, "cls": cls, "ok": True, "r2": summ["test_metrics"]["r2"],
                "rmse": summ["test_metrics"]["rmse"], "complexity": summ["complexity"],
                "formula": summ["formula"], "pred_path": str(out / f"predictions_pysr_{label}.csv")}
    except Exception as exc:  # noqa: BLE001
        return {"task": task, "cls": cls, "ok": False, "err": str(exc)}


def _build_jobs() -> list[tuple[str, str]]:
    jobs = []
    for task in ("acidic", "basic"):
        train = pd.read_csv(PROC / f"{task}_train_classed.csv", usecols=["class"])
        test = pd.read_csv(PROC / f"{task}_test_classed.csv", usecols=["class"])
        tc, vc = train["class"].value_counts(), test["class"].value_counts()
        jobs.append((task, "__global__"))
        for c in tc.index:
            if tc.get(c, 0) >= MIN_TRAIN and vc.get(c, 0) >= MIN_TEST:
                jobs.append((task, c))
    return jobs


def _assemble(task: str, results: list[dict]) -> dict:
    from .qlattice_search import regression_metrics

    train = pd.read_csv(PROC / f"{task}_train_classed.csv", usecols=["class", "pKa"])
    test = pd.read_csv(PROC / f"{task}_test_classed.csv", usecols=["class", "pKa"])
    test_pred = pd.Series(np.nan, index=test.index, dtype=float)

    g = next((r for r in results if r["task"] == task and r["cls"] == "__global__" and r["ok"]), None)
    if g:
        gp = pd.read_csv(g["pred_path"])["y_pred"].to_numpy()
        if len(gp) == len(test):
            test_pred[:] = gp

    class_rows = []
    for r in results:
        if r["task"] != task or r["cls"] == "__global__" or not r["ok"]:
            continue
        te_c = test[test["class"] == r["cls"]]
        pp = pd.read_csv(r["pred_path"])["y_pred"].to_numpy()
        if len(pp) == len(te_c):
            test_pred.loc[te_c.index] = pp
        class_rows.append({"class": r["cls"], "n_test": len(te_c), "test_r2": r["r2"],
                           "test_rmse": r["rmse"], "complexity": r["complexity"], "formula": r["formula"]})

    if test_pred.isna().any():  # global failed -> class mean
        cm = train.groupby("class")["pKa"].mean()
        fill = test.loc[test_pred.isna(), "class"].map(cm).fillna(train["pKa"].mean())
        test_pred.loc[test_pred.isna()] = fill.to_numpy()

    y_true = test["pKa"].to_numpy(dtype=float)
    overall = regression_metrics(y_true, test_pred.to_numpy(dtype=float))
    out_dir = RUNS / f"pysr_per_class_{task}_op2"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_true, "y_pred": test_pred.to_numpy(), "class": test["class"]}).to_csv(
        out_dir / f"predictions_per_class_pysr_{task}.csv", index=False)
    result = {"task": task, "engine": "pysr_parallel", "overall_test_metrics": overall,
              "global_test_r2": (g["r2"] if g else None), "per_class": class_rows}
    (out_dir / f"summary_per_class_pysr_{task}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    jobs = _build_jobs()
    print(f"PySR parallel: {len(jobs)} jobs, {MAX_WORKERS} workers", flush=True)
    results = []
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx) as ex:
        futs = {ex.submit(_pysr_job, t, c): (t, c) for t, c in jobs}
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = f"{r['task']}/{r['cls']}"
            if r["ok"]:
                print(f"  done {tag:28s} R2={r['r2']:.3f} RMSE={r['rmse']:.3f} cmplx={r['complexity']}", flush=True)
            else:
                print(f"  FAIL {tag:28s} {r['err'][:80]}", flush=True)
            (RUNS / "pysr_parallel_raw.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    summaries = [_assemble(task, results) for task in ("acidic", "basic")]
    (RUNS / "pysr_parallel_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    for s in summaries:
        o = s["overall_test_metrics"]
        print(f"\n[{s['task']}] PySR per-class OVERALL R2={o['r2']:.4f} RMSE={o['rmse']:.4f} "
              f"(global {s['global_test_r2']})", flush=True)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
