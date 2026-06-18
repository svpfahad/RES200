from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = "data/raw/zenodo_3251643/CO2CAPACITY.txt"
DEFAULT_SMILES = "data/raw/zenodo_3251643/CA.smi"
DEFAULT_OUTPUT = "data/processed/zenodo_3251643_co2capacity.csv"


def read_smiles_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ion_code", "smiles"])

    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        first, second = parts[0], parts[1]
        if first.startswith(("C", "A")) and "_" not in first:
            code, smiles = first, second
        else:
            smiles, code = first, second
        rows.append({"ion_code": code, "smiles": smiles})

    return pd.DataFrame(rows).drop_duplicates("ion_code")


def prepare(input_path: Path, smiles_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, sep=r"\s+")
    expected = {"CATION_ANION", "TEMP", "PRESSURE", "xCO2"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected CO2CAPACITY columns: {sorted(missing)}")

    prepared = df.rename(
        columns={
            "CATION_ANION": "solvent_id",
            "TEMP": "temperature_k",
            "PRESSURE": "pressure_bar",
            "xCO2": "co2_solubility_mol_frac",
        }
    )
    numeric_columns = ["temperature_k", "pressure_bar", "co2_solubility_mol_frac"]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=numeric_columns)
    prepared = prepared[
        (prepared["temperature_k"] > 0)
        & (prepared["pressure_bar"] > 0)
        & (prepared["co2_solubility_mol_frac"] > 0)
    ].copy()

    parts = prepared["solvent_id"].astype(str).str.split("_", n=1, expand=True)
    prepared["cation_code"] = parts[0]
    prepared["anion_code"] = parts[1]
    prepared["ln_pressure_bar"] = np.log(prepared["pressure_bar"].astype(float))
    prepared["ln_co2_solubility"] = np.log(prepared["co2_solubility_mol_frac"].astype(float))

    smiles = read_smiles_map(smiles_path)
    if not smiles.empty:
        cation_smiles = smiles.rename(columns={"ion_code": "cation_code", "smiles": "cation_smiles"})
        anion_smiles = smiles.rename(columns={"ion_code": "anion_code", "smiles": "anion_smiles"})
        prepared = prepared.merge(cation_smiles, on="cation_code", how="left")
        prepared = prepared.merge(anion_smiles, on="anion_code", how="left")

    prepared["source_record"] = "zenodo_3251643"
    prepared["source_file"] = str(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Zenodo 3251643 CO2 capacity data.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--smiles", default=DEFAULT_SMILES)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = prepare(Path(args.input), Path(args.smiles), Path(args.out))
    print(f"Saved {len(frame)} prepared rows to: {args.out}")
    print(f"Unique solvents: {frame['solvent_id'].nunique()}")
    print("Target columns: co2_solubility_mol_frac, ln_co2_solubility")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
