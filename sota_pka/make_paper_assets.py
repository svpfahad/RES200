from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .evaluate import summarize_prediction_file


def make_metric_table(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pred_path in sorted(run_dir.glob("predictions_*.csv")):
        metrics = summarize_prediction_file(pred_path)
        rows.append({"artifact": pred_path.name, **metrics})
    table = pd.DataFrame(rows).sort_values("rmse") if rows else pd.DataFrame()
    table.to_csv(output_dir / "benchmark_table.csv", index=False)
    return table


def make_prediction_plots(run_dir: Path, output_dir: Path) -> None:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pred_path in sorted(run_dir.glob("predictions_*.csv")):
        predictions = pd.read_csv(pred_path)
        if not {"y_true", "y_pred"}.issubset(predictions.columns):
            continue
        fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=200)
        ax.scatter(predictions["y_true"], predictions["y_pred"], s=14, alpha=0.55)
        lo = min(predictions["y_true"].min(), predictions["y_pred"].min())
        hi = max(predictions["y_true"].max(), predictions["y_pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlabel("Experimental pKa")
        ax.set_ylabel("Predicted pKa")
        ax.set_title(pred_path.stem.replace("predictions_", ""))
        fig.tight_layout()
        fig.savefig(output_dir / f"{pred_path.stem}_scatter.png")
        plt.close(fig)
