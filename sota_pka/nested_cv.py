"""E8 — Repeated nested cross-validation: unbiased uncertainty band on the
symbolic pKa R².

The recorded pipeline reports a single held-out OP2-test R² (acidic 0.448,
basic 0.470) plus an inner 4-fold CV used for top_k selection. Both are point-ish
estimates. Nested CV gives an *unbiased-of-selection* generalization distribution:

  outer KFold over OP2-train  →  for each outer fold, run the ENTIRE leakage-safe
  pipeline (correlation prune → LightGBM-importance rank → inner 4-fold CV top_k
  selection with the 1-SE rule → multi-seed refit → BIC pick) on the outer-train
  only, then score once on the held-out outer fold.

Selection never sees the outer fold, so each outer-fold R² is an honest
generalization estimate; their spread is the uncertainty band. The OP2 *official*
test split is deliberately untouched here — this is a within-train estimator that
complements the single official-test number.

Resumable: each completed (task, repeat, fold) is appended to results.jsonl and
skipped on re-run.

Run:
    .venv_mac/bin/python -m sota_pka.nested_cv --dry-run          # show fold sizes, fit nothing
    .venv_mac/bin/python -m sota_pka.nested_cv                    # 5 outer folds x 1 repeat / task
    .venv_mac/bin/python -m sota_pka.nested_cv --folds 5 --repeats 2
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .qlattice_search import run_qlattice_experiment

TARGET = "pKa"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _op2_dir() -> Path:
    return _repo_root() / "RES 200-20260312T035531Z-1-001" / "RES 200"


TASKS = {
    "acidic": _op2_dir() / "train_descriptors_op2.csv",
    "basic": _op2_dir() / "train_descriptors_basic_op2.csv",
}
OUT_DIR = _repo_root() / "sota_pka" / "runs" / "nested_cv"
RESULTS = OUT_DIR / "results.jsonl"


def _done() -> set[tuple[str, int, int]]:
    if not RESULTS.exists():
        return set()
    seen = set()
    for line in RESULTS.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        seen.add((d["task"], int(d["repeat"]), int(d["fold"])))
    return seen


def _append(rec: dict) -> None:
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(prog="nested_cv")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = _done()
    for task in args.tasks:
        df = pd.read_csv(TASKS[task])
        df = df[pd.to_numeric(df[TARGET], errors="coerce").notna()].reset_index(drop=True)
        n = len(df)
        for rep in range(args.repeats):
            kf = KFold(n_splits=args.folds, shuffle=True, random_state=rep)
            for fold, (tri, tei) in enumerate(kf.split(df)):
                tag = (task, rep, fold)
                if tag in done:
                    print(f"[skip] {task} rep{rep} fold{fold} (already done)")
                    continue
                outer_train = df.iloc[tri].reset_index(drop=True)
                outer_val = df.iloc[tei].reset_index(drop=True)
                if args.dry_run:
                    print(f"[dry] {task} rep{rep} fold{fold}: train={len(outer_train)} val={len(outer_val)} "
                          f"(total {n})")
                    continue
                fold_dir = OUT_DIR / f"{task}_r{rep}_f{fold}"
                print(f"\n=== {task} rep{rep} fold{fold}: train={len(outer_train)} val={len(outer_val)} ===",
                      flush=True)
                try:
                    summ = run_qlattice_experiment(
                        outer_train, outer_val, output_dir=fold_dir,
                        mode="direct", label=f"{task}_r{rep}_f{fold}",
                    )
                    rec = {
                        "task": task, "repeat": rep, "fold": fold,
                        "n_train": len(outer_train), "n_val": len(outer_val),
                        "outer_r2": summ["test_metrics"]["r2"],
                        "outer_rmse": summ["test_metrics"]["rmse"],
                        "outer_mae": summ["test_metrics"]["mae"],
                        "chosen_top_k": summ["chosen_top_k"],
                        "edge_count": summ["edge_count"],
                        "elapsed_sec": summ["elapsed_sec"],
                    }
                    _append(rec)
                    print(f"  -> outer R²={rec['outer_r2']:.4f} k={rec['chosen_top_k']} "
                          f"({rec['elapsed_sec']}s)", flush=True)
                except Exception as exc:  # keep the long job alive; log and continue
                    _append({"task": task, "repeat": rep, "fold": fold, "error": repr(exc)})
                    print(f"  !! FAILED {task} rep{rep} fold{fold}: {exc!r}", flush=True)

    if args.dry_run:
        return

    # ---- Aggregate ----
    recs = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    ok = [r for r in recs if "outer_r2" in r]
    lines = ["# E8 — Repeated nested-CV uncertainty band on symbolic R²\n"]
    lines.append("| task | folds | nested R² mean | std | 95% band (mean ± 1.96·SE) | "
                 "recorded single-test R² | chosen_k (mode) |")
    lines.append("|---|---:|---:|---:|---|---:|---:|")
    recorded = {"acidic": 0.448, "basic": 0.470}
    summary = {}
    for task in args.tasks:
        rs = [r["outer_r2"] for r in ok if r["task"] == task]
        if not rs:
            continue
        arr = np.array(rs, float)
        se = arr.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else float("nan")
        ks = [r["chosen_top_k"] for r in ok if r["task"] == task]
        kmode = max(set(ks), key=ks.count) if ks else None
        summary[task] = {
            "n_folds": len(arr), "mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "se": float(se), "k_mode": kmode, "folds": rs,
        }
        lines.append(f"| {task} | {len(arr)} | {arr.mean():.3f} | {arr.std(ddof=1):.3f} | "
                     f"[{arr.mean()-1.96*se:.3f}, {arr.mean()+1.96*se:.3f}] | "
                     f"{recorded.get(task, float('nan')):.3f} | {kmode} |")
    (OUT_DIR / "nested_cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = "\n".join(lines) + "\n"
    (OUT_DIR / "RESULTS_nested_cv.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
