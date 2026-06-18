"""Data loading, SMILES recovery, and the train/calibration/test split.

The op2 descriptor matrices (``train_descriptors_op2.csv`` etc.) are pure numeric
RDKit+Mordred features plus a ``pKa`` column — they carry **no SMILES**, which the
Tanimoto applicability domain needs. We recover SMILES from the raw ``Opt2_*``
splits (which have ``OriginalSmiles``):

* **basic** task — the descriptor matrix is row-aligned with the raw split
  (verified: identical row counts and a 1.000 pKa-sequence match), so SMILES come
  straight from row order.
* **acidic** task — the matrix was reordered/filtered relative to the raw split,
  so we join on a near-unique composite key recomputed from each raw SMILES:
  ``(pKa, MolWt, MaxEStateIndex, MinEStateIndex)`` (~99.9% coverage). Rows that
  stay ambiguous keep ``smiles=None`` and are simply skipped by the Tanimoto AD;
  the kNN AD and the model use every row regardless.

The calibration set required by split conformal prediction is carved out of the
**training** file only — the op2 held-out test set is never touched until the
final evaluation. Feature columns are defined by the training data alone
(``align_numeric_features``), so there is no test-set leakage into the feature
space.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..train_baselines import align_numeric_features

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


DATA_DIR = _repo_root() / "RES 200-20260312T035531Z-1-001" / "RES 200"

# (descriptor matrix, raw split with OriginalSmiles) per task/split.
_FILES = {
    "acidic": {
        "train": ("train_descriptors_op2.csv", "Opt2_acidic_tr.csv"),
        "test": ("test_descriptors_op2.csv", "Opt2_acidic_tst.csv"),
    },
    "basic": {
        "train": ("train_descriptors_basic_op2.csv", "Opt2_basic_tr.csv"),
        "test": ("test_descriptors_basic_op2.csv", "Opt2_basic_tst.csv"),
    },
}

TARGET = "pKa"
# How SMILES are recovered per task (see module docstring).
_SMILES_MODE = {"basic": "row_order", "acidic": "key_join"}
_KEY_COLS = ["MolWt", "MaxEStateIndex", "MinEStateIndex"]  # + pKa, recomputed from SMILES


# --------------------------------------------------------------------------- #
# SMILES recovery
# --------------------------------------------------------------------------- #
def _recompute_keys(smiles: pd.Series) -> pd.DataFrame:
    """RDKit (pKa-free) composite key for each raw SMILES, matching descriptor cols."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.EState.EState import MaxEStateIndex, MinEStateIndex

    rows = []
    for s in smiles.astype(str):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            rows.append((np.nan, np.nan, np.nan))
            continue
        try:
            rows.append(
                (
                    round(Descriptors.MolWt(mol), 3),
                    round(MaxEStateIndex(mol), 3),
                    round(MinEStateIndex(mol), 3),
                )
            )
        except Exception:
            rows.append((np.nan, np.nan, np.nan))
    return pd.DataFrame(rows, columns=_KEY_COLS, index=smiles.index)


def _recover_smiles(task: str, which: str, desc: pd.DataFrame) -> pd.Series:
    """Return a SMILES Series aligned to ``desc.index`` (None where unrecoverable)."""
    desc_file, raw_file = _FILES[task][which]
    raw = pd.read_csv(DATA_DIR / raw_file)
    raw_smiles = raw["OriginalSmiles"].astype(str)

    if _SMILES_MODE[task] == "row_order":
        if len(raw) != len(desc):
            raise ValueError(
                f"{task}/{which}: row-order SMILES recovery needs equal lengths "
                f"(descriptors={len(desc)}, raw={len(raw)})."
            )
        return pd.Series(raw_smiles.to_numpy(), index=desc.index, name="smiles")

    # key_join (acidic): build a 4-key (pKa + 3 descriptors) and merge.
    raw_pka = pd.to_numeric(raw[TARGET], errors="coerce").round(3)
    raw_keys = _recompute_keys(raw_smiles)
    raw_keys[TARGET] = raw_pka.to_numpy()
    raw_keys["smiles"] = raw_smiles.to_numpy()
    raw_keys = raw_keys.dropna(subset=_KEY_COLS)
    # Drop keys that map to >1 distinct SMILES so we never attach a wrong structure.
    key_full = _KEY_COLS + [TARGET]
    counts = raw_keys.groupby(key_full)["smiles"].transform("nunique")
    raw_keys = raw_keys[counts == 1].drop_duplicates(subset=key_full)

    left = desc[[TARGET] + _KEY_COLS].copy()
    left[_KEY_COLS] = left[_KEY_COLS].round(3)
    left[TARGET] = pd.to_numeric(left[TARGET], errors="coerce").round(3)
    merged = left.merge(raw_keys[key_full + ["smiles"]], on=key_full, how="left")
    out = pd.Series(merged["smiles"].to_numpy(), index=desc.index, name="smiles")
    return out


# --------------------------------------------------------------------------- #
# Split bundle
# --------------------------------------------------------------------------- #
@dataclass
class UQSplit:
    """A leakage-safe proper-train / calibration / test bundle for one task."""

    task: str
    feature_names: list[str]
    ad_feature_names: list[str]  # features natively measured in BOTH train & test
    x_proper: pd.DataFrame
    y_proper: pd.Series
    x_calib: pd.DataFrame
    y_calib: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    smiles_proper: pd.Series
    smiles_calib: pd.Series
    smiles_test: pd.Series
    meta: dict = field(default_factory=dict)

    @property
    def x_train_full(self) -> pd.DataFrame:
        """Proper-train ∪ calibration (the original op2 training rows)."""
        return pd.concat([self.x_proper, self.x_calib])

    @property
    def y_train_full(self) -> pd.Series:
        return pd.concat([self.y_proper, self.y_calib])

    @property
    def smiles_train_full(self) -> pd.Series:
        return pd.concat([self.smiles_proper, self.smiles_calib])


def load_split(task: str, calib_frac: float = 0.2, seed: int = 42) -> UQSplit:
    """Build the proper-train/calibration/test split for ``task``.

    Feature columns are defined by the **training** matrix (test reindexed onto
    them, missing filled with 0) via ``align_numeric_features`` — leakage-safe.
    The calibration set is a seeded random slice of the training rows; the op2
    test set is held out untouched.
    """
    if task not in _FILES:
        raise ValueError(f"Unknown task {task!r}; expected one of {list(_FILES)}.")
    if not 0.0 < calib_frac < 0.9:
        raise ValueError(f"calib_frac must be in (0, 0.9); got {calib_frac}.")

    train_file, _ = _FILES[task]["train"]
    test_file, _ = _FILES[task]["test"]
    train_desc = pd.read_csv(DATA_DIR / train_file)
    test_desc = pd.read_csv(DATA_DIR / test_file)

    # Recover SMILES on the raw descriptor frames (before column alignment drops
    # nothing structural, but we key off MolWt/EState which must still be present).
    smiles_train = _recover_smiles(task, "train", train_desc)
    smiles_test = _recover_smiles(task, "test", test_desc)

    x_train, x_test, y_train, y_test = align_numeric_features(train_desc, test_desc, TARGET)
    feature_names = list(x_train.columns)
    # The applicability domain must not use columns that were 0-filled in the test
    # set (present in train, absent in test) — that offset would push every test
    # compound out of domain. Restrict AD features to those natively measured in
    # both raw matrices. The *model* still uses the full aligned feature set.
    test_native = set(test_desc.select_dtypes(include=[np.number]).columns)
    ad_feature_names = [c for c in feature_names if c in test_native]

    # Carve calibration out of the training rows only.
    idx = np.arange(len(x_train))
    proper_idx, calib_idx = train_test_split(idx, test_size=calib_frac, random_state=seed)
    proper_pos, calib_pos = x_train.index[proper_idx], x_train.index[calib_idx]

    smiles_train_cov = float(smiles_train.notna().mean())
    smiles_test_cov = float(smiles_test.notna().mean())

    return UQSplit(
        task=task,
        feature_names=feature_names,
        ad_feature_names=ad_feature_names,
        x_proper=x_train.loc[proper_pos],
        y_proper=y_train.loc[proper_pos],
        x_calib=x_train.loc[calib_pos],
        y_calib=y_train.loc[calib_pos],
        x_test=x_test,
        y_test=y_test,
        smiles_proper=smiles_train.reindex(proper_pos),
        smiles_calib=smiles_train.reindex(calib_pos),
        smiles_test=smiles_test.reindex(x_test.index),
        meta={
            "n_proper": len(proper_pos),
            "n_calib": len(calib_pos),
            "n_test": len(x_test),
            "n_features": len(feature_names),
            "calib_frac": calib_frac,
            "seed": seed,
            "smiles_coverage_train": smiles_train_cov,
            "smiles_coverage_test": smiles_test_cov,
        },
    )
