"""Functional-group class router for pKa modelling.

Each molecule is assigned to the ionizable-group class that most determines its
pKa, using a priority-ordered SMARTS list (first match wins). This enables
per-class symbolic equations in the spirit of Hammett/Taft linear free-energy
relationships, where a fixed reaction centre carries a small substituent
correction.
"""
from __future__ import annotations

import pandas as pd

# Priority order matters: strongest / most pKa-determining centre first.
PATTERNS: list[tuple[str, str]] = [
    ("sulfonic_acid", "S(=O)(=O)[OX2H1]"),
    ("carboxylic_acid", "[CX3](=O)[OX2H1]"),
    ("phenol", "c[OX2H1]"),
    ("thiol", "[#6][SX2H1]"),
    ("hydroxamic_imide_NH", "[NX3H1](C=O)C=O"),
    ("guanidine", "[NX3][CX3](=[NX2,NX3+])[NX3]"),
    ("amidine", "[CX3](=[NX2,NX3+])[NX3]"),
    ("imidazole_like", "c1cnc[nH]1"),
    ("aromatic_N", "[n;!$([n][OH])]"),
    ("aniline", "c[NX3;H2,H1,H0;!$(NC=O)]"),
    ("prim_amine", "[NX3;H2][CX4]"),
    ("sec_amine", "[NX3;H1]([CX4])[CX4]"),
    ("tert_amine", "[NX3;H0]([CX4])([CX4])[CX4]"),
    ("alcohol", "[CX4][OX2H1]"),
]


def _compiled():
    from rdkit import Chem

    return [(name, Chem.MolFromSmarts(smarts)) for name, smarts in PATTERNS]


def classify_smiles(smiles: object, compiled=None) -> str:
    from rdkit import Chem

    compiled = compiled or _compiled()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "unparseable"
    for name, patt in compiled:
        if patt is not None and mol.HasSubstructMatch(patt):
            return name
    return "other"


def classify_series(smiles: pd.Series) -> pd.Series:
    compiled = _compiled()
    return smiles.map(lambda s: classify_smiles(s, compiled))
