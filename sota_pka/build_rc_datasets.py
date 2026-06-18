"""Append reaction-center features to the classed datasets -> *_classed_rc.csv."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
from .reaction_center import add_rc_columns

PROC = Path(__file__).resolve().parent / "data" / "processed"


def main() -> None:
    for task in ("acidic", "basic"):
        for split in ("train", "test"):
            df = pd.read_csv(PROC / f"{task}_{split}_classed.csv")
            out = add_rc_columns(df)
            out.to_csv(PROC / f"{task}_{split}_classed_rc.csv", index=False)
            print(f"{task}/{split}: {out.shape[0]} rows x {out.shape[1]} cols "
                  f"(+{out.shape[1]-df.shape[1]} rc features)", flush=True)
    print("done")


if __name__ == "__main__":
    main()
