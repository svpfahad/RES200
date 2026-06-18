"""Run per-class symbolic pKa modelling for acidic and basic tasks."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .per_class_pka import run_per_class

PROC = Path(__file__).resolve().parent / "data" / "processed"
RUNS = Path(__file__).resolve().parent / "runs"


def main() -> None:
    summaries = []
    for task in ("acidic", "basic"):
        train = pd.read_csv(PROC / f"{task}_train_classed.csv")
        test = pd.read_csv(PROC / f"{task}_test_classed.csv")
        print(f"\n===== PER-CLASS {task} (train {len(train)}, test {len(test)}) =====", flush=True)
        res = run_per_class(train, test, RUNS / f"per_class_{task}_op2", task)
        o = res["overall_test_metrics"]
        b = res["class_mean_baseline_test"]
        print(f"  overall test R2={o['r2']:.4f} RMSE={o['rmse']:.4f} MAE={o['mae']:.4f}", flush=True)
        print(f"  class-mean baseline R2={b['r2']:.4f} | major classes: {res['major_classes']}", flush=True)
        summaries.append(res)
        (RUNS / "per_class_combined_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\ndone")


if __name__ == "__main__":
    main()
