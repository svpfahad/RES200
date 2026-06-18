"""Deep PySR campaign: per-class + global symbolic regression, both tasks.

Set Julia thread count high to use the machine fully. Run from RES200 root:
    .venv_mac/bin/python -m sota_pka.run_pysr_all
"""
from __future__ import annotations

import os

# Must be set before juliacall/pysr import (happens lazily inside per_class_pysr).
os.environ.setdefault("PYTHON_JULIACALL_THREADS", os.environ.get("PYSR_THREADS", "16"))

import json
from pathlib import Path

import pandas as pd

from .per_class_pysr import HEAVY, fit_pysr_one, run_per_class_pysr

PROC = Path(__file__).resolve().parent / "data" / "processed"
RUNS = Path(__file__).resolve().parent / "runs"


def main() -> None:
    summaries = []
    for task in ("acidic", "basic"):
        train = pd.read_csv(PROC / f"{task}_train_classed.csv")
        test = pd.read_csv(PROC / f"{task}_test_classed.csv")

        print(f"\n===== PySR PER-CLASS {task} =====", flush=True)
        pc = run_per_class_pysr(train, test, RUNS / f"pysr_per_class_{task}_op2", task, cfg=HEAVY)
        o = pc["overall_test_metrics"]
        print(f"  per-class overall R2={o['r2']:.4f} RMSE={o['rmse']:.4f} "
              f"(class-mean {pc['class_mean_baseline_test']['r2']:.4f})", flush=True)

        print(f"\n===== PySR GLOBAL {task} =====", flush=True)
        g, _ = fit_pysr_one(train, test, RUNS / f"pysr_global_{task}_op2", f"{task}_global", cfg=HEAVY)
        print(f"  global R2={g['test_metrics']['r2']:.4f} RMSE={g['test_metrics']['rmse']:.4f} "
              f"complexity={g['complexity']} (oracle {g['oracle_best_on_test_r2']})", flush=True)

        summaries.append({"task": task, "per_class": pc, "global": g})
        (RUNS / "pysr_combined_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
