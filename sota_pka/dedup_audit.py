"""E9 — Train/external deduplication audit (E6 validity check).

The external-holdout audit (E6) is only valid if the external benchmark molecules
were NOT in the OP2 training set. This module canonicalizes both sides in the SAME
neutralized representation E6 actually scored in (RDKit Uncharger + largest-fragment
parent, then canonical SMILES) and reports the overlap, per task.

Two identity levels:
  * canonical neutral SMILES  — exact structure match (the molecule E6 fed the model).
  * InChIKey connectivity block (first 14 chars) — looser 2D-skeleton match that also
    catches tautomer/stereo variants of the same compound.

Also reports OP2 train-vs-test overlap as a leakage sanity anchor (should be ~0).

Run:
    .venv_mac/bin/python -m sota_pka.dedup_audit
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from .external_holdout import _standardizers, neutralize

RDLogger.DisableLog("rdApp.*")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _op2_dir() -> Path:
    return _repo_root() / "RES 200-20260312T035531Z-1-001" / "RES 200"


def _ext_root() -> Path:
    return _repo_root() / "sota_pka" / "data" / "raw_external" / "SAT4pKa" / "data" / "test"


OUT_DIR = _repo_root() / "sota_pka" / "runs" / "dedup_audit"

OP2_SPLITS = {
    ("acidic", "train"): _op2_dir() / "Opt2_acidic_tr.csv",
    ("acidic", "test"): _op2_dir() / "Opt2_acidic_tst.csv",
    ("basic", "train"): _op2_dir() / "Opt2_basic_tr.csv",
    ("basic", "test"): _op2_dir() / "Opt2_basic_tst.csv",
}
EXTERNAL = {
    "acidic": ["novartis_acidic", "AvLiLuMoVe_123_acidic", "SAMPL7_acidic"],
    "basic": ["novartis_basic", "AvLiLuMoVe_123_basic"],
}


def _keys(smiles_iter) -> tuple[set[str], set[str], dict[str, str]]:
    """Return (canonical-neutral-SMILES set, InChIKey-connectivity set, canon->raw map)."""
    uncharger, fragment_chooser = _standardizers()
    canon: set[str] = set()
    conn: set[str] = set()
    canon_to_raw: dict[str, str] = {}
    for raw in smiles_iter:
        neu = neutralize(raw, uncharger, fragment_chooser)
        if neu is None:
            continue
        canon.add(neu)
        canon_to_raw.setdefault(neu, str(raw))
        mol = Chem.MolFromSmiles(neu)
        if mol is not None:
            try:
                ik = Chem.MolToInchiKey(mol)
                if ik:
                    conn.add(ik.split("-")[0])
            except Exception:
                pass
    return canon, conn, canon_to_raw


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # OP2 splits → key sets
    op2_keys: dict[tuple[str, str], tuple[set[str], set[str], dict]] = {}
    for (task, split), path in OP2_SPLITS.items():
        df = pd.read_csv(path)
        op2_keys[(task, split)] = _keys(df["OriginalSmiles"])
        print(f"OP2 {task}/{split}: {len(df)} rows -> {len(op2_keys[(task, split)][0])} unique neutral structures")

    rows: list[dict] = []
    overlaps_detail: dict[str, list[str]] = {}

    # External vs OP2-train (task-matched) + vs OP2-test (context)
    for task, sets in EXTERNAL.items():
        tr_canon, tr_conn, _ = op2_keys[(task, "train")]
        te_canon, te_conn, _ = op2_keys[(task, "test")]
        for name in sets:
            df = pd.read_csv(_ext_root() / f"{name}.csv")
            ex_canon, ex_conn, ex_map = _keys(df["smiles"])
            ov_canon = sorted(ex_canon & tr_canon)
            ov_conn = sorted(ex_conn & tr_conn)
            ov_canon_te = sorted(ex_canon & te_canon)
            rows.append({
                "external_set": name, "task": task,
                "n_external_unique": len(ex_canon),
                "overlap_train_canonical": len(ov_canon),
                "overlap_train_connectivity": len(ov_conn),
                "overlap_test_canonical": len(ov_canon_te),
                "pct_train_canonical": round(100 * len(ov_canon) / max(len(ex_canon), 1), 2),
            })
            if ov_canon:
                overlaps_detail[name] = [ex_map.get(c, c) for c in ov_canon]
            print(f"  {name:24s} n={len(ex_canon):3d}  train-overlap canon={len(ov_canon)} "
                  f"conn={len(ov_conn)}  test-overlap canon={len(ov_canon_te)}")

    # OP2 train vs test leakage anchor
    leak = []
    for task in ("acidic", "basic"):
        tr_canon = op2_keys[(task, "train")][0]
        te_canon = op2_keys[(task, "test")][0]
        n_ov = len(tr_canon & te_canon)
        leak.append({"task": task, "train_test_canonical_overlap": n_ov,
                     "n_train_unique": len(tr_canon), "n_test_unique": len(te_canon)})
        print(f"OP2 {task} train/test canonical overlap: {n_ov}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(OUT_DIR / "dedup_external_vs_train.csv", index=False)
    pd.DataFrame(leak).to_csv(OUT_DIR / "dedup_op2_train_test.csv", index=False)
    (OUT_DIR / "overlap_detail.json").write_text(json.dumps(overlaps_detail, indent=2), encoding="utf-8")

    # Report
    L = ["# E9 — Train/external deduplication audit (E6 validity)\n"]
    L.append("Both sides canonicalized in the neutralized representation E6 scored in.\n")
    L.append("## External vs OP2-train overlap (task-matched)\n")
    L.append("| external set | task | n (unique) | overlap w/ train (canonical) | % | overlap (connectivity) | overlap w/ OP2-test |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        L.append(f"| {r['external_set']} | {r['task']} | {r['n_external_unique']} | "
                 f"{r['overlap_train_canonical']} | {r['pct_train_canonical']:.1f}% | "
                 f"{r['overlap_train_connectivity']} | {r['overlap_test_canonical']} |")
    L.append("\n## OP2 train/test leakage anchor (should be ~0)\n")
    L.append("| task | n train | n test | train∩test (canonical) |")
    L.append("|---|---:|---:|---:|")
    for r in leak:
        L.append(f"| {r['task']} | {r['n_train_unique']} | {r['n_test_unique']} | {r['train_test_canonical_overlap']} |")
    tot_canon = sum(r["overlap_train_canonical"] for r in rows)
    L.append(f"\n**Verdict:** total exact-structure external↔train overlap = **{tot_canon}** molecule(s) "
             f"across all external sets. "
             + ("E6 external sets are clean — no training contamination." if tot_canon == 0
                else "See `overlap_detail.json`; exclude overlapping molecules and re-report E6."))
    report = "\n".join(L) + "\n"
    (OUT_DIR / "RESULTS_dedup_audit.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
