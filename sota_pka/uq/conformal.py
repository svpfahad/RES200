"""Distribution-free prediction intervals for the XGBoost pKa models.

Three interval methods, all calibrated on the held-out calibration slice carved
by ``data_splits`` (never on the test set), so each carries a finite-sample
marginal-coverage guarantee:

1. **Split conformal** — constant-width ``ŷ ± q̂`` where ``q̂`` is the
   conformalized quantile of absolute calibration residuals. Simple, exact
   marginal coverage, but the same width everywhere.

2. **Normalized (locally-adaptive) conformal** — nonconformity score
   ``|resid| / (σ(x) + β)`` scaled by a per-sample difficulty ``σ(x)`` (the
   ensemble std from ``baseline``). Intervals widen where the ensemble disagrees,
   so they track the applicability domain while keeping marginal coverage.

3. **Conformalized quantile regression (CQR)** — XGBoost ``reg:quantileerror``
   lower/upper quantile models, then a conformal correction on the calibration
   residuals so coverage is guaranteed even if the raw quantiles are miscalibrated
   (Romano et al., 2019). Gives asymmetric, adaptive intervals.

The conformal quantile uses the finite-sample level
``⌈(n+1)(1-α)⌉ / n`` (Lei et al., 2018) so small calibration sets still get
honest, slightly-conservative coverage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baseline import XGBConfig, _impute_apply, _impute_fit


# --------------------------------------------------------------------------- #
# Core conformal quantile
# --------------------------------------------------------------------------- #
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of nonconformity ``scores`` at level 1-α.

    The split-conformal quantile is the ``⌈(n+1)(1-α)⌉``-th smallest score
    (1-indexed order statistic; Lei et al., 2018). We compute it directly rather
    than via ``np.quantile`` to avoid interpolation-basis ambiguity.
    """
    scores = np.sort(np.asarray(scores, float))
    n = len(scores)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:  # too few calibration points for this α → widest score (full coverage)
        return float(scores[-1])
    return float(scores[k - 1])


@dataclass
class Intervals:
    """Prediction intervals on a target set."""

    method: str
    alpha: float
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def covered(self, y_true: np.ndarray) -> np.ndarray:
        y = np.asarray(y_true, float)
        return (y >= self.lower) & (y <= self.upper)


# --------------------------------------------------------------------------- #
# 1. Split conformal
# --------------------------------------------------------------------------- #
def split_conformal(
    y_calib: np.ndarray,
    pred_calib: np.ndarray,
    pred_test: np.ndarray,
    alpha: float = 0.1,
) -> Intervals:
    resid = np.abs(np.asarray(y_calib, float) - np.asarray(pred_calib, float))
    q = conformal_quantile(resid, alpha)
    pred_test = np.asarray(pred_test, float)
    return Intervals(
        method="split_conformal",
        alpha=alpha,
        point=pred_test,
        lower=pred_test - q,
        upper=pred_test + q,
        meta={"q_halfwidth": q, "n_calib": int(len(resid))},
    )


# --------------------------------------------------------------------------- #
# 2. Normalized (locally-adaptive) conformal
# --------------------------------------------------------------------------- #
def normalized_conformal(
    y_calib: np.ndarray,
    pred_calib: np.ndarray,
    sigma_calib: np.ndarray,
    pred_test: np.ndarray,
    sigma_test: np.ndarray,
    alpha: float = 0.1,
    beta: float | None = None,
) -> Intervals:
    """Locally-adaptive intervals scaled by a difficulty estimate ``σ(x)``.

    ``beta`` floors σ so a few zero-disagreement points don't collapse the
    interval; defaults to the median calibration σ.
    """
    y_calib = np.asarray(y_calib, float)
    pred_calib = np.asarray(pred_calib, float)
    sigma_calib = np.asarray(sigma_calib, float)
    sigma_test = np.asarray(sigma_test, float)
    pred_test = np.asarray(pred_test, float)
    if beta is None:
        beta = float(np.median(sigma_calib))
    scores = np.abs(y_calib - pred_calib) / (sigma_calib + beta)
    q = conformal_quantile(scores, alpha)
    halfwidth = q * (sigma_test + beta)
    return Intervals(
        method="normalized_conformal",
        alpha=alpha,
        point=pred_test,
        lower=pred_test - halfwidth,
        upper=pred_test + halfwidth,
        meta={"q_scaled": q, "beta": beta, "n_calib": int(len(scores))},
    )


# --------------------------------------------------------------------------- #
# 3. Conformalized quantile regression (CQR)
# --------------------------------------------------------------------------- #
def _fit_quantile_model(xa: np.ndarray, y: np.ndarray, q: float, cfg: XGBConfig, seed: int):
    from xgboost import XGBRegressor

    params = cfg.kwargs(seed=seed, n_jobs=-1)
    params.update(objective="reg:quantileerror", quantile_alpha=q)
    model = XGBRegressor(**params)
    model.fit(xa, y)
    return model


def cqr_intervals(
    x_proper: pd.DataFrame,
    y_proper: pd.Series,
    x_calib: pd.DataFrame,
    y_calib: np.ndarray,
    x_test: pd.DataFrame,
    alpha: float = 0.1,
    cfg: XGBConfig | None = None,
    seed: int = 42,
) -> Intervals:
    """Conformalized quantile regression (Romano et al., 2019)."""
    cfg = cfg or XGBConfig()
    medians = _impute_fit(x_proper)
    xa_p = _impute_apply(x_proper, medians)
    xa_c = _impute_apply(x_calib, medians)
    xa_t = _impute_apply(x_test, medians)
    yp = y_proper.to_numpy(float)
    y_calib = np.asarray(y_calib, float)

    lo_model = _fit_quantile_model(xa_p, yp, alpha / 2.0, cfg, seed)
    hi_model = _fit_quantile_model(xa_p, yp, 1.0 - alpha / 2.0, cfg, seed + 1)

    q_lo_c, q_hi_c = lo_model.predict(xa_c), hi_model.predict(xa_c)
    # CQR conformity score: how far y lies outside the predicted band.
    scores = np.maximum(q_lo_c - y_calib, y_calib - q_hi_c)
    e = conformal_quantile(scores, alpha)

    q_lo_t, q_hi_t = lo_model.predict(xa_t), hi_model.predict(xa_t)
    lower, upper = q_lo_t - e, q_hi_t + e
    point = 0.5 * (q_lo_t + q_hi_t)
    return Intervals(
        method="cqr",
        alpha=alpha,
        point=point,
        lower=lower,
        upper=upper,
        meta={"e_correction": float(e), "n_calib": int(len(scores))},
    )
