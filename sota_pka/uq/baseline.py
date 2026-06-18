"""Leakage-safe XGBoost baseline + a bootstrap ensemble.

The point model is XGBoost with the same hyperparameters the parent
``train_baselines`` uses for the paper's headline model, but pinned to the
Apple-Silicon-optimal CPU path: ``tree_method="hist"``, ``device="cpu"``,
``n_jobs=-1`` (there is no GPU path for XGBoost on macOS/Metal).

The ensemble trains ``n_members`` XGBoost models on bootstrap resamples of the
*proper-training* set, each with a different seed. It gives two things:

* a point prediction = ensemble **mean** (more stable than a single fit), and
* a per-sample **std** across members = a cheap difficulty / disagreement signal,
  which ``conformal.py`` uses to make locally-adaptive (normalized) intervals and
  which correlates with being outside the applicability domain.

Members are trained in parallel across cores (``joblib``), each with the inner
XGBoost thread count set to 1 to avoid oversubscription — on an 18-core M5 Pro
that means ~18 members fit concurrently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class XGBConfig:
    """XGBoost hyperparameters (mirrors train_baselines, CPU-hist pinned)."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0

    def kwargs(self, seed: int, n_jobs: int) -> dict:
        return dict(
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=seed,
            n_jobs=n_jobs,
        )


def _make_regressor(seed: int, cfg: XGBConfig, n_jobs: int):
    from xgboost import XGBRegressor

    return XGBRegressor(**cfg.kwargs(seed=seed, n_jobs=n_jobs))


def _impute_fit(x: pd.DataFrame) -> pd.Series:
    """Column medians for NaN imputation (train-derived; matches the parent pipeline)."""
    return x.median(numeric_only=True)


def _impute_apply(x: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    return x.fillna(medians).replace([np.inf, -np.inf], np.nan).fillna(medians).to_numpy(float)


# --------------------------------------------------------------------------- #
# Single point model
# --------------------------------------------------------------------------- #
@dataclass
class PointModel:
    model: object
    medians: pd.Series
    cfg: XGBConfig

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(_impute_apply(x, self.medians))


def fit_point_model(
    x: pd.DataFrame, y: pd.Series, cfg: XGBConfig | None = None, seed: int = 42, n_jobs: int = -1
) -> PointModel:
    cfg = cfg or XGBConfig()
    medians = _impute_fit(x)
    model = _make_regressor(seed, cfg, n_jobs)
    model.fit(_impute_apply(x, medians), y.to_numpy(float))
    return PointModel(model=model, medians=medians, cfg=cfg)


# --------------------------------------------------------------------------- #
# Bootstrap ensemble
# --------------------------------------------------------------------------- #
@dataclass
class EnsemblePrediction:
    mean: np.ndarray
    std: np.ndarray
    members: np.ndarray  # shape (n_members, n_samples)


@dataclass
class Ensemble:
    models: list
    medians: pd.Series
    cfg: XGBConfig
    seeds: list[int] = field(default_factory=list)

    def predict(self, x: pd.DataFrame) -> EnsemblePrediction:
        xa = _impute_apply(x, self.medians)
        members = np.vstack([m.predict(xa) for m in self.models])
        return EnsemblePrediction(
            mean=members.mean(axis=0),
            std=members.std(axis=0, ddof=1) if len(self.models) > 1 else np.zeros(members.shape[1]),
            members=members,
        )


def _fit_member(seed: int, xa: np.ndarray, y: np.ndarray, cfg: XGBConfig):
    """Fit one ensemble member on a bootstrap resample (inner n_jobs=1)."""
    from xgboost import XGBRegressor

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), len(y))  # bootstrap with replacement
    model = XGBRegressor(**cfg.kwargs(seed=seed, n_jobs=1))
    model.fit(xa[idx], y[idx])
    return model


def fit_ensemble(
    x: pd.DataFrame,
    y: pd.Series,
    n_members: int = 20,
    cfg: XGBConfig | None = None,
    base_seed: int = 1000,
    n_jobs: int = -1,
) -> Ensemble:
    """Train ``n_members`` bootstrap XGBoost models in parallel across cores."""
    cfg = cfg or XGBConfig()
    medians = _impute_fit(x)
    xa = _impute_apply(x, medians)
    ya = y.to_numpy(float)
    seeds = [base_seed + i for i in range(n_members)]
    # Thread backend: XGBoost releases the GIL during fit, so members train
    # concurrently without pickling the (large) feature matrix to subprocesses.
    models = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_member)(s, xa, ya, cfg) for s in seeds
    )
    return Ensemble(models=models, medians=medians, cfg=cfg, seeds=seeds)
