"""Applicability-domain (AD) definitions for the pKa models.

Two complementary, train-derived AD definitions. Both fit *only* on the training
data and set their threshold from the training set's own neighbour distribution,
so flagging a test compound never uses test labels or test structures.

* **kNN-distance AD** (descriptor space) — standardize features with a
  train-fit scaler, then take each compound's mean Euclidean distance to its
  ``k`` nearest training neighbours. A test compound is **out-of-domain** if that
  distance exceeds the ``q``-th percentile (default 95) of the training set's own
  leave-one-out kNN distances. Captures "far from any training point in feature
  space".

* **Tanimoto-similarity AD** (structure space) — Morgan/ECFP fingerprints; each
  compound's **maximum** Tanimoto similarity to the training set. A test compound
  is out-of-domain if that max similarity is **below** the ``q``-th percentile
  (default 5) of the training set's own nearest-neighbour similarities. Captures
  "structurally unlike anything seen in training" and is the chemically intuitive
  complement to the descriptor-space distance.

Reporting both lets the paper show cross-method agreement on which test
compounds are reliable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class ADResult:
    """Per-compound AD scores and in/out flags for a target set."""

    name: str
    score: np.ndarray          # distance (kNN) or similarity (Tanimoto)
    in_domain: np.ndarray      # bool; NaN-scored compounds are False (unknown→out)
    threshold: float
    higher_is_inside: bool     # Tanimoto: True; kNN distance: False
    meta: dict = field(default_factory=dict)

    @property
    def frac_in(self) -> float:
        return float(np.mean(self.in_domain))


# --------------------------------------------------------------------------- #
# kNN-distance AD (descriptor space)
# --------------------------------------------------------------------------- #
@dataclass
class KNNDistanceAD:
    k: int = 5
    percentile: float = 95.0
    scaler: StandardScaler = field(default=None)
    nn: NearestNeighbors = field(default=None)
    medians: pd.Series = field(default=None)
    threshold: float = field(default=np.nan)

    def fit(self, x_train: pd.DataFrame) -> "KNNDistanceAD":
        self.medians = x_train.median(numeric_only=True)
        xt = self._prep(x_train)
        self.scaler = StandardScaler().fit(xt)
        xs = self.scaler.transform(xt)
        self.nn = NearestNeighbors(n_neighbors=self.k + 1, n_jobs=-1).fit(xs)
        # Leave-one-out: drop the self-neighbour (column 0) for the train threshold.
        d, _ = self.nn.kneighbors(xs)
        train_scores = d[:, 1:].mean(axis=1)
        self.threshold = float(np.percentile(train_scores, self.percentile))
        self._train_scores = train_scores
        return self

    def _prep(self, x: pd.DataFrame) -> np.ndarray:
        return x.fillna(self.medians).replace([np.inf, -np.inf], np.nan).fillna(self.medians).to_numpy(float)

    def score(self, x: pd.DataFrame) -> np.ndarray:
        xs = self.scaler.transform(self._prep(x))
        d, _ = self.nn.kneighbors(xs)  # k+1 neighbours; for a *new* set none is self
        return d[:, : self.k].mean(axis=1)

    def evaluate(self, x: pd.DataFrame, name: str = "knn_distance") -> ADResult:
        sc = self.score(x)
        return ADResult(
            name=name,
            score=sc,
            in_domain=sc <= self.threshold,
            threshold=self.threshold,
            higher_is_inside=False,
            meta={"k": self.k, "percentile": self.percentile,
                  "train_score_median": float(np.median(self._train_scores))},
        )


# --------------------------------------------------------------------------- #
# Tanimoto-similarity AD (structure space)
# --------------------------------------------------------------------------- #
def _morgan_fps(smiles: pd.Series, radius: int = 2, n_bits: int = 2048):
    """Morgan fingerprints; returns (fps, valid_mask) aligned to ``smiles``."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, valid = [], []
    for s in smiles:
        mol = None if s is None or (isinstance(s, float) and np.isnan(s)) else Chem.MolFromSmiles(str(s))
        if mol is None:
            valid.append(False)
            continue
        fps.append(gen.GetFingerprint(mol))
        valid.append(True)
    return fps, np.array(valid, bool)


@dataclass
class TanimotoAD:
    radius: int = 2
    n_bits: int = 2048
    percentile: float = 5.0
    train_fps: list = field(default=None)
    threshold: float = field(default=np.nan)

    def fit(self, train_smiles: pd.Series) -> "TanimotoAD":
        from rdkit import DataStructs

        self.train_fps, _ = _morgan_fps(train_smiles, self.radius, self.n_bits)
        # Leave-one-out nearest-neighbour similarity within train.
        nn_sims = []
        for i, fp in enumerate(self.train_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fp, self.train_fps)
            sims[i] = -1.0  # exclude self
            nn_sims.append(max(sims))
        self.threshold = float(np.percentile(nn_sims, self.percentile))
        self._train_nn_sims = np.array(nn_sims)
        return self

    def score(self, smiles: pd.Series) -> np.ndarray:
        from rdkit import DataStructs

        out = np.full(len(smiles), np.nan)
        fps, valid = _morgan_fps(smiles, self.radius, self.n_bits)
        j = 0
        for i, ok in enumerate(valid):
            if not ok:
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fps[j], self.train_fps)
            out[i] = max(sims) if sims else np.nan
            j += 1
        return out

    def evaluate(self, smiles: pd.Series, name: str = "tanimoto") -> ADResult:
        sc = self.score(smiles)
        in_dom = np.where(np.isnan(sc), False, sc >= self.threshold)
        return ADResult(
            name=name,
            score=sc,
            in_domain=in_dom.astype(bool),
            threshold=self.threshold,
            higher_is_inside=True,
            meta={"radius": self.radius, "n_bits": self.n_bits, "percentile": self.percentile,
                  "n_unscored": int(np.isnan(sc).sum()),
                  "train_nn_sim_median": float(np.median(self._train_nn_sims))},
        )
