from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _require_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Fragments, MACCSkeys
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except Exception as exc:
        raise RuntimeError("RDKit is required for chemistry featurization") from exc
    return Chem, Descriptors, Fragments, MACCSkeys, GetMorganGenerator


def featurize_smiles_table(rows: pd.DataFrame, output_path: Path, radius: int = 2, n_bits: int = 2048) -> pd.DataFrame:
    """Generate RDKit descriptors, fragment counts, Morgan bits, and MACCS keys."""
    Chem, Descriptors, Fragments, MACCSkeys, GetMorganGenerator = _require_rdkit()
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    descriptor_funcs = Descriptors.descList
    fragment_funcs = [(name, func) for name, func in Fragments.__dict__.items() if name.startswith("fr_") and callable(func)]
    records: list[dict[str, object]] = []
    for row in rows.itertuples(index=False):
        smiles = getattr(row, "canonical_smiles", "") or getattr(row, "smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        record = {"pKa": getattr(row, "pka"), "smiles": smiles, "inchi_key": getattr(row, "inchi_key"), "task": getattr(row, "task")}
        for name, func in descriptor_funcs:
            try:
                record[name] = func(mol)
            except Exception:
                record[name] = np.nan
        for name, func in fragment_funcs:
            try:
                record[name] = func(mol)
            except Exception:
                record[name] = np.nan
        fp = generator.GetFingerprint(mol)
        for idx, value in enumerate(fp):
            record[f"morgan_r{radius}_{n_bits}_{idx}"] = int(value)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        for idx, value in enumerate(maccs):
            record[f"maccs_{idx}"] = int(value)
        records.append(record)
    features = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return features


def featurize_op_split(
    csv_path: Path,
    output_path: Path,
    smiles_col: str = "OriginalSmiles",
    target: str = "pKa",
    strip_salts: bool = True,
    keep_smiles: bool = False,
) -> pd.DataFrame:
    """Replicate the RES200 notebook descriptor pipeline for one split CSV.

    RDKit ``Descriptors.descList`` + RDKit ``Fragments`` (fr_*) + Mordred 2D
    (``ignore_3D=True``), optional salt removal, numeric coercion, and dropping of
    columns that are entirely missing. The output frame is ``pKa`` followed by the
    descriptor columns. Zero-variance/alignment/imputation are handled downstream
    in the modeling step so train and test stay independent here.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Fragments
    from rdkit.Chem.SaltRemover import SaltRemover

    raw = pd.read_csv(csv_path)
    raw[target] = pd.to_numeric(raw[target], errors="coerce")
    raw = raw[raw[smiles_col].notna() & raw[target].notna()].reset_index(drop=True)

    remover = SaltRemover() if strip_salts else None

    def to_mol(smiles: object):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        if remover is not None:
            try:
                mol = remover.StripMol(mol)
            except Exception:
                pass
        return mol

    raw["_mol"] = raw[smiles_col].map(to_mol)
    valid = raw[raw["_mol"].notna()].reset_index(drop=True)
    mols = valid["_mol"].tolist()

    descriptor_funcs = list(Descriptors.descList)
    fragment_funcs = [(n, f) for n, f in Fragments.__dict__.items() if n.startswith("fr_") and callable(f)]

    def per_mol(mol) -> dict:
        record: dict[str, object] = {}
        for name, func in descriptor_funcs:
            try:
                record[name] = func(mol)
            except Exception:
                record[name] = np.nan
        for name, func in fragment_funcs:
            try:
                record[name] = func(mol)
            except Exception:
                record[name] = np.nan
        return record

    rdkit_df = pd.DataFrame([per_mol(m) for m in mols])

    # Mordred 2D (mordredcommunity); degrade gracefully if unavailable.
    try:
        from mordred import Calculator, descriptors as mordred_descriptors

        calc = Calculator(mordred_descriptors, ignore_3D=True)
        # nproc=1 avoids multiprocessing-spawn failures across launch contexts.
        mordred_df = calc.pandas(mols, nproc=1, quiet=True)
        mordred_df = mordred_df.apply(pd.to_numeric, errors="coerce")
        mordred_df = mordred_df.reset_index(drop=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[featurize] Mordred unavailable, using RDKit-only features: {exc}")
        mordred_df = pd.DataFrame(index=range(len(mols)))

    rdkit_df = rdkit_df.reset_index(drop=True)
    full = pd.concat([rdkit_df, mordred_df], axis=1)
    full = full.loc[:, ~full.columns.duplicated()]
    full = full.apply(pd.to_numeric, errors="coerce")
    full = full.dropna(axis=1, how="all")

    lead = [target]
    lead_frame = valid[[target]].reset_index(drop=True)
    if keep_smiles:
        lead_frame = pd.concat(
            [valid[[smiles_col]].reset_index(drop=True).rename(columns={smiles_col: "smiles"}), lead_frame],
            axis=1,
        )
        lead = ["smiles", target]
    out = pd.concat([lead_frame, full], axis=1)
    out = out[lead + [c for c in out.columns if c not in lead]]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out
