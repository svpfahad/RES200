"""Build SMILES- and class-tagged descriptor datasets for per-class pKa modelling.

For each task (acidic, basic) and split (train, test), recompute descriptors with
SMILES retained, attach the functional-group class, and write
``data/processed/{task}_{split}_classed.csv`` with columns:

    smiles, class, pKa, <descriptors...>
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .classify import classify_series
from .featurize import featurize_op_split

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "RES 200-20260312T035531Z-1-001" / "RES 200"
OUT = Path(__file__).resolve().parent / "data" / "processed"

SPLITS = {
    ("acidic", "train"): DATA / "Opt2_acidic_tr.csv",
    ("acidic", "test"): DATA / "Opt2_acidic_tst.csv",
    ("basic", "train"): DATA / "Opt2_basic_tr.csv",
    ("basic", "test"): DATA / "Opt2_basic_tst.csv",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for (task, split), raw in SPLITS.items():
        tmp = OUT / f"_tmp_{task}_{split}.csv"
        df = featurize_op_split(raw, tmp, smiles_col="OriginalSmiles", target="pKa", keep_smiles=True)
        df.insert(1, "class", classify_series(df["smiles"]))
        out_path = OUT / f"{task}_{split}_classed.csv"
        df.to_csv(out_path, index=False)
        tmp.unlink(missing_ok=True)
        counts = df["class"].value_counts()
        print(f"{task}/{split}: {df.shape[0]} rows x {df.shape[1]} cols | classes: "
              + ", ".join(f"{k}={v}" for k, v in counts.head(8).items()), flush=True)
    print("done")


if __name__ == "__main__":
    main()
