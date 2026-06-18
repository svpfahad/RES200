from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "data/processed/zenodo_3251643_co2capacity.csv"
DEFAULT_OUTPUT = "data/processed/zenodo_candidate_examples.csv"


FEATURE_COLUMNS = [
    "candidate_id",
    "solvent_id",
    "temperature_k",
    "pressure_bar",
    "cation_code",
    "anion_code",
    "ln_pressure_bar",
    "cation_smiles",
    "anion_smiles",
]


def make_candidates(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    base = df.dropna(subset=["cation_smiles", "anion_smiles"]).copy()
    if base.empty:
        base = df.copy()

    quantiles = [0.20, 0.50, 0.80]
    rows = []
    for idx, quantile in enumerate(quantiles, start=1):
        target_value = base["co2_solubility_mol_frac"].quantile(quantile)
        selected = base.iloc[(base["co2_solubility_mol_frac"] - target_value).abs().argsort().iloc[0]].copy()
        selected["candidate_id"] = f"zenodo_observed_q{int(quantile * 100)}"
        rows.append(selected)

    ood = rows[-1].copy()
    ood["candidate_id"] = "deliberate_ood_high_pressure"
    ood["solvent_id"] = "C9999_A9999"
    ood["cation_code"] = "C9999"
    ood["anion_code"] = "A9999"
    ood["temperature_k"] = 500.0
    ood["pressure_bar"] = 1500.0
    ood["ln_pressure_bar"] = math.log(1500.0)
    ood["cation_smiles"] = "C[N+](C)(C)C"
    ood["anion_smiles"] = "[Cl-]"
    rows.append(ood)

    candidates = pd.DataFrame(rows)
    candidates = candidates[FEATURE_COLUMNS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create small Zenodo candidate-screen examples.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = make_candidates(Path(args.input), Path(args.out))
    print(f"Saved {len(frame)} candidates to: {args.out}")
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

