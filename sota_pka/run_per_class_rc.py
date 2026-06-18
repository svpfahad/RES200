"""Per-class symbolic models on reaction-center features.

Two variants per task:
  * rcfull  — all descriptors + RC features (max accuracy)
  * rconly  — only the 33 reaction-center features (max interpretability;
              equations in named physical quantities)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .per_class_pka import run_per_class
from .reaction_center import RC_KEYS

PROC = Path(__file__).resolve().parent / "data" / "processed"
RUNS = Path(__file__).resolve().parent / "runs"
META = ["smiles", "class", "pKa"]


def main() -> None:
    summaries = []
    # variant-outer so both RC-only results (the priority deliverable) land first
    for variant in ("rconly", "rcfull"):
        for task in ("acidic", "basic"):
            train = pd.read_csv(PROC / f"{task}_train_classed_rc.csv")
            test = pd.read_csv(PROC / f"{task}_test_classed_rc.csv")
            rc_cols = [c for c in RC_KEYS if c in train.columns]
            cols = (META + rc_cols) if variant == "rconly" else None
            tr = train[cols] if cols else train
            te = test[cols] if cols else test
            print(f"\n===== {variant} {task} ({tr.shape[1]} cols) =====", flush=True)
            res = run_per_class(tr, te, RUNS / f"per_class_{variant}_{task}_op2", task)
            o = res["overall_test_metrics"]
            print(f"  {variant} {task}: overall R2={o['r2']:.4f} RMSE={o['rmse']:.4f} "
                  f"(class-mean {res['class_mean_baseline_test']['r2']:.4f})", flush=True)
            res["variant"] = variant
            summaries.append(res)
            (RUNS / "per_class_rc_combined_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\ndone")


if __name__ == "__main__":
    main()
