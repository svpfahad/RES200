from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def summarize_prediction_file(path: Path) -> dict[str, float]:
    predictions = pd.read_csv(path)
    missing = {"y_true", "y_pred"} - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    return regression_metrics(predictions["y_true"], predictions["y_pred"])


def write_predictions(path: Path, y_true, y_pred, metadata: dict[str, object] | None = None) -> None:
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    if metadata:
        for key, value in metadata.items():
            frame[key] = value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
