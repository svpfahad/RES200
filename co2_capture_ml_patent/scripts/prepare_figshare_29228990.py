from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = "data/raw/figshare_29228990/ef5c01345_si_001.xlsx"
DEFAULT_OUTPUT = "data/processed/figshare_29228990_unique_ils.csv"


RENAME_MAP = {
    "IL No.": "il_no",
    "Unique IL": "solvent_id",
    "Ln(CO2 solubility-experimental)": "ln_co2_solubility_exp",
    "Ln(Pressure (MPa))": "ln_pressure_mpa",
    "Temp (K)": "temperature_k",
    "Ref.": "reference",
}


def clean_column(name: object) -> str:
    text = str(name).strip()
    if text in RENAME_MAP:
        return RENAME_MAP[text]
    text = text.replace("*", "_anion")
    text = text.replace("[", "_").replace("]", "")
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "unnamed"


def prepare(input_path: Path, output_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(input_path, sheet_name="S4-Experimental data_Unique ILs", header=2)
    raw = raw.dropna(axis=1, how="all")
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    raw = raw.rename(columns={column: clean_column(column) for column in raw.columns})

    required = {"il_no", "solvent_id", "ln_co2_solubility_exp", "ln_pressure_mpa", "temperature_k"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing expected columns from S4 sheet: {missing}")

    raw["il_no"] = raw["il_no"].ffill()
    raw["solvent_id"] = raw["solvent_id"].ffill()
    raw = raw.dropna(subset=["solvent_id", "ln_co2_solubility_exp", "ln_pressure_mpa", "temperature_k"])

    for column in raw.columns:
        if column not in {"il_no", "solvent_id", "reference"}:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw["pressure_mpa"] = np.exp(raw["ln_pressure_mpa"])
    raw["co2_solubility_mol_frac"] = np.exp(raw["ln_co2_solubility_exp"])
    raw["source_sheet"] = "S4-Experimental data_Unique ILs"
    raw["source_file"] = str(input_path)

    prior_model_columns = ["swr", "mlp", "rbf", "rf", "lsboost"]
    raw = raw.drop(columns=[c for c in prior_model_columns if c in raw.columns])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_path, index=False)
    return raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the exact experimental-target Figshare S4 sheet.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = prepare(Path(args.input), Path(args.out))
    print(f"Saved {len(frame)} prepared rows to: {args.out}")
    print(f"Unique solvents: {frame['solvent_id'].nunique()}")
    print("Target columns: ln_co2_solubility_exp, co2_solubility_mol_frac")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
