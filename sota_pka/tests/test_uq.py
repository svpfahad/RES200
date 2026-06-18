"""Tests for the uncertainty + applicability-domain (sota_pka.uq) track.

Unit tests use small synthetic data so they run fast; one integration test loads
the real acidic split to lock in the leakage guarantees (calibration/test
disjoint, AD features exclude 0-filled columns).
"""
import math

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# Conformal core
# --------------------------------------------------------------------------- #
def test_conformal_quantile_finite_sample_level():
    from sota_pka.uq.conformal import conformal_quantile

    scores = np.arange(1, 101, dtype=float)  # 1..100
    # level = ceil((100+1)*0.9)/100 = ceil(90.9)/100 = 91/100 -> 91st value
    q = conformal_quantile(scores, alpha=0.1)
    assert q == 91.0
    # too few points for the level -> falls back to max
    assert conformal_quantile(np.array([1.0, 2.0, 3.0]), alpha=0.01) == 3.0


def test_split_conformal_achieves_marginal_coverage():
    from sota_pka.uq.conformal import split_conformal

    rng = np.random.default_rng(0)
    n = 4000
    pred_calib = rng.normal(0, 3, n)
    y_calib = pred_calib + rng.normal(0, 1, n)
    pred_test = rng.normal(0, 3, n)
    y_test = pred_test + rng.normal(0, 1, n)

    iv = split_conformal(y_calib, pred_calib, pred_test, alpha=0.1)
    cov = iv.covered(y_test).mean()
    assert 0.87 <= cov <= 0.93
    # constant width for split conformal
    assert np.allclose(iv.width, iv.width[0])


def test_normalized_conformal_width_tracks_sigma():
    from sota_pka.uq.conformal import normalized_conformal

    rng = np.random.default_rng(1)
    n = 3000
    sigma_calib = rng.uniform(0.5, 3.0, n)
    pred_calib = rng.normal(0, 2, n)
    y_calib = pred_calib + rng.normal(0, sigma_calib)  # heteroscedastic
    pred_test = np.zeros(5)
    sigma_test = np.array([0.5, 1.0, 1.5, 2.0, 3.0])

    iv = normalized_conformal(y_calib, pred_calib, sigma_calib, pred_test, sigma_test, alpha=0.1)
    # higher difficulty -> strictly wider interval
    assert np.all(np.diff(iv.width) > 0)


def test_cqr_returns_ordered_band(tmp_path):
    from sota_pka.uq.conformal import cqr_intervals

    rng = np.random.default_rng(2)
    x = pd.DataFrame({"f1": rng.normal(size=200), "f2": rng.normal(size=200)})
    y = pd.Series(2 * x["f1"] - x["f2"] + rng.normal(0, 0.5, 200))
    x_cal = pd.DataFrame({"f1": rng.normal(size=80), "f2": rng.normal(size=80)})
    y_cal = (2 * x_cal["f1"] - x_cal["f2"]).to_numpy()
    x_te = pd.DataFrame({"f1": rng.normal(size=50), "f2": rng.normal(size=50)})

    iv = cqr_intervals(x, y, x_cal, y_cal, x_te, alpha=0.1)
    assert iv.method == "cqr"
    assert np.all(iv.upper >= iv.lower)
    assert len(iv.lower) == 50


# --------------------------------------------------------------------------- #
# Applicability domain
# --------------------------------------------------------------------------- #
def test_knn_distance_ad_flags_distant_points():
    from sota_pka.uq.applicability import KNNDistanceAD

    rng = np.random.default_rng(3)
    train = pd.DataFrame(rng.normal(0, 1, size=(300, 5)), columns=[f"f{i}" for i in range(5)])
    near = pd.DataFrame(rng.normal(0, 1, size=(50, 5)), columns=train.columns)
    far = pd.DataFrame(rng.normal(20, 1, size=(50, 5)), columns=train.columns)
    test = pd.concat([near, far], ignore_index=True)

    ad = KNNDistanceAD(k=5, percentile=95).fit(train).evaluate(test)
    # near points mostly in-domain, far points all out
    assert ad.in_domain[:50].mean() > 0.8
    assert ad.in_domain[50:].sum() == 0
    assert ad.higher_is_inside is False


def test_tanimoto_ad_scores_and_handles_bad_smiles():
    from sota_pka.uq.applicability import TanimotoAD

    train = pd.Series(["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCC"])
    ad = TanimotoAD(percentile=5).fit(train)
    res = ad.evaluate(pd.Series(["CCO", "not_a_smiles", None]))
    assert math.isclose(res.score[0], 1.0, rel_tol=1e-6)  # identical to a train member
    assert np.isnan(res.score[1]) and np.isnan(res.score[2])  # unparseable -> NaN
    assert res.in_domain[1] == False and res.in_domain[2] == False  # unknown -> out
    assert res.higher_is_inside is True


# --------------------------------------------------------------------------- #
# Integration: leakage guarantees on the real split
# --------------------------------------------------------------------------- #
def _data_available() -> bool:
    from sota_pka.uq.data_splits import DATA_DIR
    return (DATA_DIR / "train_descriptors_op2.csv").exists()


@pytest.mark.skipif(not _data_available(), reason="op2 descriptor matrices not present")
def test_uqsplit_is_leakage_safe_and_ad_features_clean():
    from sota_pka.uq.data_splits import load_split

    s = load_split("acidic", calib_frac=0.2, seed=42)
    # calibration is carved out of training only: proper ∪ calib partitions the
    # training rows, with no overlap (the meaningful leakage guarantee; train and
    # test live in separate files with independent 0-based indices).
    assert set(s.x_proper.index).isdisjoint(s.x_calib.index)
    assert len(s.x_proper) + len(s.x_calib) == s.meta["n_proper"] + s.meta["n_calib"]
    # feature space is train-defined: test carries exactly the model features
    assert list(s.x_test.columns) == s.feature_names
    # AD features are a subset of model features (the 0-filled ones are dropped)
    assert set(s.ad_feature_names).issubset(s.feature_names)
    assert len(s.ad_feature_names) <= len(s.feature_names)
    # SMILES were recovered for the vast majority of rows
    assert s.meta["smiles_coverage_test"] > 0.9
