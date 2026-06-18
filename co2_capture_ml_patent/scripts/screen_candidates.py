from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from co2patent.pipeline import load_model_bundle, load_table, screen_candidates  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank CO2 solvent candidates with confidence gates.")
    parser.add_argument("--model", required=True, help="Path to model.joblib from run_pipeline.py.")
    parser.add_argument("--candidates", required=True, help="CSV/XLSX candidate file.")
    parser.add_argument("--sheet", default=None, help="Excel sheet index/name. Default: first sheet.")
    parser.add_argument("--out", required=True, help="Output CSV for ranked candidates.")
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
    bundle = load_model_bundle(args.model)
    candidates = load_table(args.candidates, sheet=parse_sheet(args.sheet))
    ranked = screen_candidates(bundle, candidates)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out, index=False)
    print(f"Saved ranked candidates to: {out}")
    print(ranked.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

