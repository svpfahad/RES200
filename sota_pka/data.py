from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

SCHEMA_COLUMNS = [
    "smiles",
    "canonical_smiles",
    "inchi_key",
    "pka",
    "task",
    "source",
    "temperature",
    "method",
    "split",
    "notes",
]


def _optional_rdkit():
    try:
        from rdkit import Chem
    except Exception:
        return None
    return Chem


def canonicalize_smiles(smiles: object) -> str:
    text = "" if pd.isna(smiles) else str(smiles).strip()
    if not text:
        return ""
    chem = _optional_rdkit()
    if chem is None:
        return text
    mol = chem.MolFromSmiles(text)
    if mol is None:
        return text
    return chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def inchi_key_from_smiles(smiles: object) -> str:
    text = "" if pd.isna(smiles) else str(smiles).strip()
    if not text:
        return ""
    chem = _optional_rdkit()
    if chem is None:
        return ""
    mol = chem.MolFromSmiles(text)
    if mol is None:
        return ""
    try:
        return chem.MolToInchiKey(mol)
    except Exception:
        return ""


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([""] * len(df), index=df.index)


def _normalize_task(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if "acid" in text:
        return "acidic"
    if "base" in text or "basic" in text:
        return "basic"
    return "unknown"


def normalize_res200_rows(raw: pd.DataFrame, source: str, split: str) -> pd.DataFrame:
    """Normalize one RES200/DataWarrior-style table to the project schema."""
    smiles = _first_existing(raw, ["OriginalSmiles", "SMILES", "smiles"])
    pka = pd.to_numeric(_first_existing(raw, ["pKa", "pka_value", "pka"]), errors="coerce")
    task = _first_existing(raw, ["basicOrAcidic", "acidity_label", "task"]).map(_normalize_task)
    temperature = pd.to_numeric(_first_existing(raw, ["temp", "T", "temperature"]), errors="coerce")
    method = _first_existing(raw, ["method", "Method"]).fillna("").astype(str)
    provided_inchi = _first_existing(raw, ["InChI Key_QSARr", "InChIKey", "inchi_key"]).fillna("").astype(str)

    normalized = pd.DataFrame(
        {
            "smiles": smiles.fillna("").astype(str).str.strip(),
            "pka": pka,
            "task": task,
            "source": source,
            "temperature": temperature,
            "method": method,
            "split": split,
        }
    )
    normalized["canonical_smiles"] = normalized["smiles"].map(canonicalize_smiles)
    normalized["inchi_key"] = provided_inchi.str.strip()
    missing_inchi = normalized["inchi_key"].eq("")
    if missing_inchi.any():
        normalized.loc[missing_inchi, "inchi_key"] = normalized.loc[missing_inchi, "canonical_smiles"].map(
            inchi_key_from_smiles
        )
    normalized["notes"] = ""
    normalized = normalized[SCHEMA_COLUMNS]
    normalized = normalized[normalized["smiles"].ne("") & normalized["pka"].notna()].copy()
    return normalized.reset_index(drop=True)


def _normalize_iupac_task(row: pd.Series) -> str:
    label = "" if pd.isna(row.get("acidity_label", "")) else str(row.get("acidity_label", "")).strip().upper()
    pka_type = "" if pd.isna(row.get("pka_type", "")) else str(row.get("pka_type", "")).strip().lower()
    if label == "AH" or "pkah" in pka_type:
        return "basic"
    if label in {"HA", "H"} or "pka" in pka_type:
        return "acidic"
    return "unknown"


def normalize_iupac_rows(raw: pd.DataFrame, source: str = "IUPAC_pKa", split: str = "external") -> pd.DataFrame:
    """Normalize the high-confidence IUPAC pKa table to the project schema."""
    smiles = _first_existing(raw, ["SMILES", "smiles"])
    pka = pd.to_numeric(_first_existing(raw, ["pka_value", "pKa", "pka"]), errors="coerce")
    temperature = pd.to_numeric(_first_existing(raw, ["T", "temperature", "original_T"]), errors="coerce")
    method = _first_existing(raw, ["method"]).fillna("").astype(str)
    normalized = pd.DataFrame(
        {
            "smiles": smiles.fillna("").astype(str).str.strip(),
            "pka": pka,
            "task": raw.apply(_normalize_iupac_task, axis=1),
            "source": source,
            "temperature": temperature,
            "method": method,
            "split": split,
        }
    )
    normalized["canonical_smiles"] = normalized["smiles"].map(canonicalize_smiles)
    normalized["inchi_key"] = normalized["canonical_smiles"].map(inchi_key_from_smiles)
    assessment = _first_existing(raw, ["assessment"]).fillna("").astype(str).str.strip()
    normalized["notes"] = assessment.map(lambda value: f"assessment={value}" if value else "")
    normalized = normalized[SCHEMA_COLUMNS]
    normalized = normalized[normalized["smiles"].ne("") & normalized["pka"].notna()].copy()
    return normalized.reset_index(drop=True)


def deduplicate_measurements(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep the first measurement per task and structure key."""
    data = rows.copy()
    data["_structure_key"] = data["inchi_key"].where(data["inchi_key"].fillna("").ne(""), data["canonical_smiles"])
    data = data[data["_structure_key"].fillna("").ne("")]
    data = data.drop_duplicates(subset=["task", "_structure_key"], keep="first")
    return data.drop(columns=["_structure_key"]).reset_index(drop=True)


def assert_no_split_leakage(
    rows: pd.DataFrame,
    train_splits: set[str] | None = None,
    test_splits: set[str] | None = None,
) -> None:
    train_splits = train_splits or {"train"}
    test_splits = test_splits or {"test", "external"}
    data = rows.copy()
    data["_structure_key"] = data["inchi_key"].where(data["inchi_key"].fillna("").ne(""), data["canonical_smiles"])
    train_keys = set(data.loc[data["split"].isin(train_splits), "_structure_key"].dropna())
    test_keys = set(data.loc[data["split"].isin(test_splits), "_structure_key"].dropna())
    overlap = sorted(k for k in train_keys & test_keys if k)
    if overlap:
        sample = ", ".join(overlap[:5])
        raise ValueError(f"Train/test split leakage detected for {len(overlap)} structures: {sample}")


def purge_split_leakage(
    rows: pd.DataFrame,
    train_splits: set[str] | None = None,
    test_splits: set[str] | None = None,
) -> pd.DataFrame:
    """Remove held-out rows whose structures already appear in training rows."""
    train_splits = train_splits or {"train"}
    test_splits = test_splits or {"test", "external"}
    data = rows.copy()
    data["_structure_key"] = data["inchi_key"].where(data["inchi_key"].fillna("").ne(""), data["canonical_smiles"])
    train_keys = set(data.loc[data["split"].isin(train_splits), "_structure_key"].dropna())
    leaked_heldout = data["split"].isin(test_splits) & data["_structure_key"].isin(train_keys)
    return data.loc[~leaked_heldout].drop(columns=["_structure_key"]).reset_index(drop=True)


def dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_res200_op_splits(data_root: Path) -> pd.DataFrame:
    """Load all current OP1/OP2/OP3 acidic/basic split CSVs if present."""
    data_root = Path(data_root)
    frames: list[pd.DataFrame] = []
    for op in ["Opt1", "Opt2", "Opt3"]:
        for task in ["acidic", "basic"]:
            for split_name, split in [("tr", "train"), ("tst", "test")]:
                path = data_root / f"{op}_{task}_{split_name}.csv"
                if not path.exists():
                    continue
                raw = pd.read_csv(path)
                frames.append(normalize_res200_rows(raw, source=f"RES200_{op}_{task}", split=split))
    if not frames:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def prepare_res200_dataset(data_root: Path, output_path: Path) -> pd.DataFrame:
    rows = load_res200_op_splits(data_root)
    rows = deduplicate_measurements(rows)
    rows = purge_split_leakage(rows)
    assert_no_split_leakage(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)
    return rows


def prepare_res200_plus_iupac_dataset(data_root: Path, iupac_file: Path, output_path: Path) -> pd.DataFrame:
    res200 = load_res200_op_splits(data_root)
    iupac = normalize_iupac_rows(pd.read_csv(iupac_file), source="IUPAC_pKa", split="external")
    rows = pd.concat([res200, iupac], ignore_index=True)
    rows = deduplicate_measurements(rows)
    rows = purge_split_leakage(rows)
    assert_no_split_leakage(rows, train_splits={"train"}, test_splits={"test", "external"})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False)
    return rows
