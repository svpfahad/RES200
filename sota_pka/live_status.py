"""Snapshot of all running symbolic-regression work (read-only).

Run once for a snapshot, or loop it:
    while true; do clear; .venv_mac/bin/python -m sota_pka.live_status; sleep 10; done
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def main() -> None:
    print("=" * 78)
    print("LIVE STATUS — per-class & global symbolic regression")
    print("=" * 78)

    # per-class component summaries that have completed so far
    rows = []
    for p in sorted(glob.glob(str(RUNS / "**" / "summary_*.json"), recursive=True)):
        s = _load(p)
        if not s or "test_metrics" not in s:
            continue
        engine = s.get("engine", "qlattice")
        label = s.get("label", Path(p).stem.replace("summary_", ""))
        m = s["test_metrics"]
        comp = s.get("sympy_ops") or s.get("complexity")
        rows.append((engine, label, m["r2"], m["rmse"], comp))
    if rows:
        print(f"\nCompleted equations: {len(rows)}")
        print(f"  {'engine':9s} {'label':30s} {'R2':>7s} {'RMSE':>7s} {'cmplx':>6s}")
        for engine, label, r2, rmse, comp in rows:
            print(f"  {engine:9s} {label[:30]:30s} {r2:7.3f} {rmse:7.3f} {str(comp):>6s}")

    # overall task-level results if a driver finished a task
    for name, pat in [("QLattice per-class", "per_class_*_op2/summary_per_class_*.json"),
                      ("PySR per-class", "pysr_per_class_*_op2/summary_per_class_pysr_*.json")]:
        for p in sorted(glob.glob(str(RUNS / pat))):
            s = _load(p)
            if s and "overall_test_metrics" in s:
                o = s["overall_test_metrics"]
                b = s["class_mean_baseline_test"]["r2"]
                print(f"\n[{name}] {s['task']}: OVERALL R2={o['r2']:.4f} RMSE={o['rmse']:.4f} "
                      f"(class-mean baseline {b:.3f})")

    # driver log tails
    for log in ["per_class_driver.log", "pysr_driver.log"]:
        lp = RUNS / log
        if lp.exists():
            tail = lp.read_text().splitlines()[-4:]
            print(f"\n--- {log} (tail) ---")
            for line in tail:
                print("  " + line[:100])
    print()


if __name__ == "__main__":
    main()
