# Uncertainty-aware & Applicability-Domain pKa Prediction (XGBoost)

A self-contained extension of the RES200 XGBoost pKa work. It keeps the original
full-descriptor XGBoost model as the baseline and adds two things every QSPR
model needs but the original paper lacked: **calibrated per-prediction
uncertainty** and an explicit **applicability domain (AD)**.

Everything is leakage-safe and CPU-only (XGBoost `hist`; there is no GPU path for
XGBoost on Apple Silicon), and the whole study runs in ~2 minutes on a laptop.

## What it does

| Component | Method | Module |
|---|---|---|
| Baseline | Full RDKit+Mordred XGBoost (the paper's headline model) + a 20-member bootstrap ensemble | `baseline.py` |
| Uncertainty | **Split conformal**, **normalized (locally-adaptive) conformal**, **conformalized quantile regression (CQR)** | `conformal.py` |
| Applicability domain | **kNN distance** (descriptor space) + **Tanimoto similarity** (Morgan fingerprints) | `applicability.py` |
| Evaluation | coverage, sharpness, calibration curves, inside-vs-outside-AD stratification | `evaluate_uq.py` |

All prediction intervals are calibrated on a held-out **calibration** slice
(20%) carved from the *training* set; the op2 **test** set is scored exactly
once. The kNN AD uses only features natively measured in both train and test
(0-filled columns are excluded so they cannot push every test point out of
domain). Both AD thresholds are derived from the training set's own
nearest-neighbour distribution.

## How to run

```bash
# both tasks, 20-member ensemble (default)
.venv_mac/bin/python -m sota_pka.uq.cli run --task all

# one task, custom ensemble size / nominal levels
.venv_mac/bin/python -m sota_pka.uq.cli run --task acidic --members 30 --alphas 0.1 0.2

.venv_mac/bin/python -m pytest sota_pka/tests/test_uq.py -q   # unit + integration tests
```

Outputs:
- `sota_pka/runs/uq_xgb/<task>/` — `metrics_predictive.csv`, `coverage.csv`,
  `ad_summary.csv`, `ad_coverage.csv`, `predictions_test.csv` (per-compound
  prediction, ensemble std, AD scores/flags, and the three interval bands).
- `sota_pka/paper_assets/uq/` — `RESULTS_uq.md` and figures
  (`<task>_calibration.png`, `<task>_width.png`, `<task>_ad_error.png`).

## Key results (op2 held-out test)

**Predictive (reproduces the paper):** acidic full-train XGBoost R² **0.826**,
RMSE **1.42**; basic R² **0.842**, RMSE **1.29**.

**Interval calibration** — empirical coverage tracks the nominal level (acidic):

| method | 80% | 90% | 95% | median width @90% |
|---|---|---|---|---|
| split conformal | 0.805 | 0.906 | 0.949 | 5.05 |
| normalized conformal | 0.816 | 0.914 | 0.950 | **4.20** |
| CQR | 0.808 | 0.897 | 0.932 | 7.55 |

Normalized conformal holds coverage while being **sharper** than split conformal
(tighter median width) by widening only where the ensemble disagrees.

**Applicability domain** — out-of-domain compounds are markedly less accurate,
and marginal intervals undercover there (the justification for an AD):

| task | AD | % in-domain | MAE in | MAE out | out/in |
|---|---|---|---|---|---|
| acidic | kNN | 95.3% | 1.00 | 1.60 | 1.60× |
| acidic | Tanimoto | 94.9% | 1.00 | 1.59 | 1.59× |
| basic | kNN | 94.0% | 0.89 | 1.58 | 1.79× |
| basic | Tanimoto | 94.4% | 0.87 | 1.94 | **2.24×** |

At 90% nominal, split-conformal coverage on basic drops from ~0.89 **inside** the
AD to ~0.62–0.75 **outside** it — so flagging out-of-domain predictions is what
keeps the reported reliability honest.

## Design notes

- **SMILES recovery** (needed for Tanimoto, since descriptor matrices carry no
  SMILES): row-order for *basic* (verified 1:1), a `(pKa, MolWt, MaxEStateIndex,
  MinEStateIndex)` composite-key join for *acidic* (~99.9% coverage); unmatched
  rows are skipped by Tanimoto only.
- **Apple-Silicon performance**: `tree_method="hist"`, `device="cpu"`,
  `n_jobs=-1`; ensemble members train concurrently via a joblib thread backend
  (XGBoost releases the GIL during fit) — ~8.6× speedup on the 18-core M5 Pro.
  The cost is dominated by the 1,100–1,700-feature model, not parallel overhead.
