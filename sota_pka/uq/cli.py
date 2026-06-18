"""CLI for the uncertainty + applicability-domain XGBoost track.

    .venv_mac/bin/python -m sota_pka.uq.cli run --task acidic
    .venv_mac/bin/python -m sota_pka.uq.cli run --task all --members 20

Writes per-task CSVs to ``sota_pka/runs/uq_xgb/<task>/``, figures to
``sota_pka/paper_assets/uq/``, and a combined ``RESULTS_uq.md`` summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .evaluate_uq import FIG_ROOT, RUNS_ROOT, UQReport, run_task

TASKS = ["acidic", "basic"]


def _fmt(df: pd.DataFrame) -> str:
    """Minimal GitHub-markdown table (avoids a `tabulate` dependency)."""
    def cell(v):
        if isinstance(v, float):
            return "nan" if pd.isna(v) else f"{v:.3f}"
        return str(v)
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _write_results_doc(reports: list[UQReport]) -> Path:
    L: list[str] = ["# Uncertainty-aware & Applicability-Domain pKa Prediction (XGBoost)\n"]
    L.append("Leakage-safe: intervals calibrated on a held-out calibration slice of the "
             "training data; applicability domains fit on training data only; the op2 test "
             "set is scored once. CPU-only (XGBoost `hist`, no GPU path on Apple Silicon).\n")
    for r in reports:
        m = r.meta
        L.append(f"\n## {r.task.capitalize()} task\n")
        L.append(f"- splits: proper-train **{m['n_proper']}**, calibration **{m['n_calib']}**, "
                 f"test **{m['n_test']}**; features **{m['n_features']}**; "
                 f"ensemble members **{m['n_members']}**; SMILES coverage test "
                 f"**{m['smiles_coverage_test']:.3f}**.\n")
        L.append("**Predictive performance (op2 test):**\n")
        L.append(_fmt(r.predictive) + "\n")
        L.append("**Interval coverage & sharpness:**\n")
        L.append(_fmt(r.coverage) + "\n")
        L.append("**Applicability domain — error inside vs outside:**\n")
        L.append(_fmt(r.ad_summary) + "\n")
        L.append("**Coverage inside vs outside AD (@ 90% nominal):**\n")
        L.append(_fmt(r.ad_coverage) + "\n")
        if r.figures:
            L.append("Figures: " + ", ".join(f"`{Path(f).name}`" for f in r.figures) + "\n")
    doc = FIG_ROOT / "RESULTS_uq.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("\n".join(L), encoding="utf-8")
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(prog="sota_pka.uq.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the UQ+AD pipeline")
    run.add_argument("--task", choices=TASKS + ["all"], default="all")
    run.add_argument("--members", type=int, default=20, help="ensemble size")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--calib-frac", type=float, default=0.2)
    run.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    run.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if args.cmd == "run":
        tasks = TASKS if args.task == "all" else [args.task]
        reports = []
        for task in tasks:
            print(f"\n=== {task} ===")
            r = run_task(task, alphas=tuple(args.alphas), n_members=args.members,
                         seed=args.seed, calib_frac=args.calib_frac,
                         make_figures=not args.no_figures)
            reports.append(r)
            print(r.predictive.to_string(index=False))
            print(r.ad_summary[["ad_method", "frac_in_domain", "mae_in", "mae_out",
                                "mae_ratio_out_in"]].to_string(index=False))
        doc = _write_results_doc(reports)
        print(f"\nWrote results doc → {doc}")
        print(f"CSVs → {RUNS_ROOT}/<task>/   figures → {FIG_ROOT}/")


if __name__ == "__main__":
    main()
