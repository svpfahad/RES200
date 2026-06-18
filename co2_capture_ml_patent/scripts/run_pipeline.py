from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from co2patent.pipeline import (  # noqa: E402
    infer_group,
    infer_target,
    load_table,
    save_train_result,
    train_best_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local CO2 solvent screening models.")
    parser.add_argument("--data", required=True, help="CSV/XLSX data file.")
    parser.add_argument("--sheet", default=None, help="Excel sheet index/name. Default: first sheet.")
    parser.add_argument("--target", default=None, help="Target column. Recommended to pass explicitly.")
    parser.add_argument("--group", default=None, help="Solvent identity/group column for leakage-safe split.")
    parser.add_argument("--out", default="runs/demo", help="Output directory.")
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=None,
        help="Feature columns to remove before training. Useful for ablations.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--calibration-size", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.10, help="Miscoverage rate. 0.10 gives ~90%% intervals.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_sheet(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_table(args.data, sheet=parse_sheet(args.sheet))
    target = infer_target(df, args.target)
    group = infer_group(df, args.group)
    result, leaderboard, predictions = train_best_model(
        df=df,
        target=target,
        group=group,
        exclude_columns=args.exclude_columns,
        test_size=args.test_size,
        calibration_size=args.calibration_size,
        alpha=args.alpha,
        random_state=args.seed,
    )

    save_train_result(result, out)
    leaderboard.to_csv(out / "leaderboard.csv", index=False)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "best_model": result.model_name,
                "metrics": result.metrics,
                "calibration_qhat": result.calibration_qhat,
                "target": target,
                "group": group,
                "excluded_columns": args.exclude_columns or [],
                "features": result.feature_columns,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Best model: {result.model_name}")
    print(json.dumps(result.metrics, indent=2))
    print(f"Saved outputs to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
