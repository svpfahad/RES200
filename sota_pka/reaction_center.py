"""Reaction-center (local) descriptors for pKa — the Hammett/Taft signal.

Within a functional-group class, pKa is governed by the electronic environment of
the protonation site. Whole-molecule descriptors blur this; here we compute
features localised on the ionizable atom and its 1-3 bond neighbourhood:
Gasteiger charges, E-state, electronegativity, and electron-withdrawing/-donating
counts at increasing radius. These operationalise substituent effects directly.

`rc_features(smiles, cls)` returns a dict of ``rc_*`` features.
`add_rc_columns(df)` appends them to a classed dataframe (needs `smiles`,`class`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .classify import PATTERNS

_EN = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "P": 2.19, "S": 2.58,
       "Cl": 3.16, "Br": 2.96, "I": 2.66, "B": 2.04, "Si": 1.90}
_HETERO = {"O", "N", "S"}
_HALOGEN = {"F", "Cl", "Br", "I"}

RC_KEYS = [
    "rc_q_center", "rc_estate_center", "rc_en_center", "rc_center_aromatic", "rc_center_inring",
    "rc_center_h", "rc_center_degree", "rc_center_formalq",
    "rc_q_r1_sum", "rc_q_r1_mean", "rc_q_r1_min", "rc_q_r1_max",
    "rc_q_r2_sum", "rc_q_r2_mean", "rc_q_r3_sum",
    "rc_estate_r1_sum", "rc_estate_r2_sum",
    "rc_en_r1_sum", "rc_en_r1_max",
    "rc_nhetero_r3", "rc_nhalogen_r3", "rc_ncarbonyl_r3", "rc_naromatic_r3", "rc_nEWG_r3", "rc_nEDG_r3",
    "rc_min_dist_carbonyl", "rc_n_ionizable_sites",
    # distance-decayed whole-molecule electronic sums (topological Hammett analog)
    "rc_decay_en", "rc_decay_q", "rc_decay_ewg", "rc_decay_edg", "rc_decay_halogen", "rc_decay_nitro",
]


def _compiled():
    from rdkit import Chem
    return [(n, Chem.MolFromSmarts(s)) for n, s in PATTERNS]


def _shells(mol, center: int, max_radius: int = 3):
    """BFS shells of atom indices around `center` (excludes center)."""
    seen = {center}
    frontier = {center}
    shells = []
    for _ in range(max_radius):
        nxt = set()
        for a in frontier:
            for nb in mol.GetAtomWithIdx(a).GetNeighbors():
                j = nb.GetIdx()
                if j not in seen:
                    nxt.add(j)
                    seen.add(j)
        shells.append(nxt)
        frontier = nxt
        if not nxt:
            break
    while len(shells) < max_radius:
        shells.append(set())
    return shells


def rc_features(smiles: object, cls: str, compiled=None) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.EState import EStateIndices

    out = {k: np.nan for k in RC_KEYS}
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return out
    compiled = compiled or _compiled()
    try:
        AllChem.ComputeGasteigerCharges(mol)
        estate = EStateIndices(mol)
    except Exception:
        return out

    def q(i):
        try:
            v = float(mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge"))
            return v if np.isfinite(v) else 0.0
        except Exception:
            return 0.0

    # candidate centre atoms: heteroatoms in any class-SMARTS match
    candidates = []
    for name, patt in compiled:
        if patt is None:
            continue
        for match in mol.GetSubstructMatches(patt):
            for idx in match:
                if mol.GetAtomWithIdx(idx).GetSymbol() in _HETERO:
                    candidates.append(idx)
    candidates = list(dict.fromkeys(candidates))
    out["rc_n_ionizable_sites"] = float(len(candidates))
    if not candidates:
        # fall back to most polarised heteroatom overall
        candidates = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in _HETERO]
    if not candidates:
        return out
    center = max(candidates, key=lambda i: abs(q(i)))
    catom = mol.GetAtomWithIdx(center)

    out["rc_q_center"] = q(center)
    out["rc_estate_center"] = float(estate[center])
    out["rc_en_center"] = _EN.get(catom.GetSymbol(), 2.5)
    out["rc_center_aromatic"] = float(catom.GetIsAromatic())
    out["rc_center_inring"] = float(catom.IsInRing())
    out["rc_center_h"] = float(catom.GetTotalNumHs())
    out["rc_center_degree"] = float(catom.GetDegree())
    out["rc_center_formalq"] = float(catom.GetFormalCharge())

    shells = _shells(mol, center, 3)
    r1, r2, r3 = shells[0], shells[1], shells[2]
    q1 = [q(i) for i in r1]
    out["rc_q_r1_sum"] = float(sum(q1)) if q1 else 0.0
    out["rc_q_r1_mean"] = float(np.mean(q1)) if q1 else 0.0
    out["rc_q_r1_min"] = float(min(q1)) if q1 else 0.0
    out["rc_q_r1_max"] = float(max(q1)) if q1 else 0.0
    out["rc_q_r2_sum"] = float(sum(q(i) for i in r2))
    out["rc_q_r2_mean"] = float(np.mean([q(i) for i in r2])) if r2 else 0.0
    out["rc_q_r3_sum"] = float(sum(q(i) for i in r3))
    out["rc_estate_r1_sum"] = float(sum(estate[i] for i in r1))
    out["rc_estate_r2_sum"] = float(sum(estate[i] for i in r2))
    en1 = [_EN.get(mol.GetAtomWithIdx(i).GetSymbol(), 2.5) for i in r1]
    out["rc_en_r1_sum"] = float(sum(en1)) if en1 else 0.0
    out["rc_en_r1_max"] = float(max(en1)) if en1 else 0.0

    local = r1 | r2 | r3
    nhet = nhal = ncarbonyl = naro = newg = nedg = 0
    for i in local:
        a = mol.GetAtomWithIdx(i)
        sym = a.GetSymbol()
        if sym in _HETERO:
            nhet += 1
        if sym in _HALOGEN:
            nhal += 1
        if a.GetIsAromatic():
            naro += 1
        # carbonyl carbon
        is_carbonyl = sym == "C" and any(b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(a).GetSymbol() == "O"
                                         for b in a.GetBonds())
        if is_carbonyl:
            ncarbonyl += 1
        # crude EWG/EDG by atom type
        if sym in _HALOGEN or is_carbonyl or (sym == "N" and a.GetIsAromatic()) or a.GetFormalCharge() > 0:
            newg += 1
        elif sym == "C" and not a.GetIsAromatic() and a.GetTotalNumHs() > 0:
            nedg += 1
    out["rc_nhetero_r3"] = float(nhet)
    out["rc_nhalogen_r3"] = float(nhal)
    out["rc_ncarbonyl_r3"] = float(ncarbonyl)
    out["rc_naromatic_r3"] = float(naro)
    out["rc_nEWG_r3"] = float(newg)
    out["rc_nEDG_r3"] = float(nedg)

    # graph distance from centre to nearest carbonyl carbon
    from rdkit.Chem import GetDistanceMatrix
    dm = GetDistanceMatrix(mol)
    carbonyls = [a.GetIdx() for a in mol.GetAtoms()
                 if a.GetSymbol() == "C" and any(b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(a).GetSymbol() == "O"
                                                 for b in a.GetBonds())]
    out["rc_min_dist_carbonyl"] = float(min((dm[center][c] for c in carbonyls), default=20.0))

    # distance-decayed electronic-effect sums over the whole molecule. Captures
    # aromatic substituent effects (e.g. para-nitro) that local radius-3 misses.
    nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
    nitro_atoms = {m[0] for m in mol.GetSubstructMatches(nitro)} if nitro else set()
    dec_en = dec_q = dec_ewg = dec_edg = dec_hal = dec_nitro = 0.0
    for a in mol.GetAtoms():
        i = a.GetIdx()
        if i == center:
            continue
        w = 1.0 / (dm[center][i] + 1.0)
        sym = a.GetSymbol()
        dec_en += _EN.get(sym, 2.5) * w
        dec_q += q(i) * w
        if sym in _HALOGEN:
            dec_hal += w
        if i in nitro_atoms:
            dec_nitro += w
        is_carbonyl = sym == "C" and any(b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(a).GetSymbol() == "O"
                                         for b in a.GetBonds())
        if sym in _HALOGEN or is_carbonyl or (sym == "N" and a.GetIsAromatic()) or a.GetFormalCharge() > 0:
            dec_ewg += w
        elif sym == "C" and not a.GetIsAromatic() and a.GetTotalNumHs() > 0:
            dec_edg += w
    out["rc_decay_en"] = dec_en
    out["rc_decay_q"] = dec_q
    out["rc_decay_ewg"] = dec_ewg
    out["rc_decay_edg"] = dec_edg
    out["rc_decay_halogen"] = dec_hal
    out["rc_decay_nitro"] = dec_nitro
    return out


def add_rc_columns(df: pd.DataFrame, smiles_col: str = "smiles", class_col: str = "class") -> pd.DataFrame:
    compiled = _compiled()
    recs = [rc_features(s, c, compiled) for s, c in zip(df[smiles_col], df[class_col])]
    rc = pd.DataFrame(recs, columns=RC_KEYS)
    return pd.concat([df.reset_index(drop=True), rc.reset_index(drop=True)], axis=1)
