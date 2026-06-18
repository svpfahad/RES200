"""E10 — Paired significance test for the symbolic-vs-ceiling accuracy gap.

The benchmark claims the symbolic model is out-accuracied by the GBM ceiling. A
reviewer will ask whether that gap is statistically significant rather than noise.
Because every model is scored on the *same* molecules in the *same* order, the
per-molecule absolute errors are paired, so a Wilcoxon signed-rank test on
|error_symbolic| - |error_ceiling| is the right tool (no normality assumption).

Feyn-free: operates on the stored, row-aligned external prediction CSVs.

Run:
    .venv_mac/bin/python -m sota_pka.significance
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PRED = Path(__file__).resolve().parents[1] / "sota_pka" / "runs" / "external_holdout" / "predictions"
OUT_DIR = Path(__file__).resolve().parents[1] / "sota_pka" / "runs" / "external_holdout"
EXT_SETS = {
    "acidic": ["novartis_acidic", "AvLiLuMoVe_123_acidic", "SAMPL7_acidic"],
    "basic": ["novartis_basic", "AvLiLuMoVe_123_basic"],
}


def _abs_err(name: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(PRED / f"{name}__neutral__{model}.csv")
    return df["y_true"].to_numpy(float), np.abs(df["y_true"].to_numpy(float) - df["y_pred"].to_numpy(float))


def _paired(name_list, model_a: str, model_b: str):
    """Pool |err| for model_a, model_b over the given sets (aligned per row)."""
    a, b = [], []
    for name in name_list:
        yt_a, ea = _abs_err(name, model_a)
        yt_b, eb = _abs_err(name, model_b)
        assert np.allclose(yt_a, yt_b), f"misaligned y_true for {name}"
        a.append(ea); b.append(eb)
    return np.concatenate(a), np.concatenate(b)


def main() -> None:
    rows = []
    for task, sets in EXT_SETS.items():
        for ceiling in ("lightgbm", "random_forest"):
            ea, eb = _paired(sets, "qlattice", ceiling)
            diff = ea - eb  # >0 means symbolic worse
            try:
                stat, p = wilcoxon(ea, eb, zero_method="wilcox")
            except ValueError:
                stat, p = float("nan"), float("nan")
            rows.append({
                "task": task, "comparison": f"qlattice vs {ceiling}", "n_pairs": int(len(ea)),
                "median_abs_err_symbolic": float(np.median(ea)),
                f"median_abs_err_{ceiling}": float(np.median(eb)),
                "median_diff(sym-ceil)": float(np.median(diff)),
                "wilcoxon_stat": float(stat), "p_value": float(p),
                "ceiling_better_frac": float(np.mean(diff > 0)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "significance_symbolic_vs_ceiling.csv", index=False)

    L = ["# E10 — Paired Wilcoxon: symbolic vs ceiling (pooled external, neutral)\n"]
    L.append("Per-molecule paired |error|. p < 0.05 ⇒ the accuracy gap is significant. "
             "`ceiling_better_frac` = fraction of molecules where the ceiling has smaller |error|.\n")
    L.append("| task | comparison | n | median \\|err\\| symbolic | median \\|err\\| ceiling | "
             "median Δ(sym−ceil) | Wilcoxon p | ceiling better |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ceil = r["comparison"].split("vs ")[1]
        L.append(f"| {r['task']} | {r['comparison']} | {r['n_pairs']} | "
                 f"{r['median_abs_err_symbolic']:.3f} | {r[f'median_abs_err_{ceil}']:.3f} | "
                 f"{r['median_diff(sym-ceil)']:+.3f} | {r['p_value']:.2e} | "
                 f"{100*r['ceiling_better_frac']:.0f}% |")
    report = "\n".join(L) + "\n"
    (OUT_DIR / "RESULTS_significance.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {OUT_DIR/'significance_symbolic_vs_ceiling.csv'} and report.")


if __name__ == "__main__":
    main()
