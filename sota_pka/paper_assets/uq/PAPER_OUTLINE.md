# Paper outline — Uncertainty-aware & Applicability-Domain pKa Prediction

*Target: undergraduate / student research journal (e.g. JURI). Builds directly on
"XGBoost-Based Prediction of pKa Values from Molecular Descriptors" (Atwi &
Al-Khater).*

## Working title
**Knowing When to Trust a pKa Model: Conformal Prediction Intervals and an
Applicability Domain for XGBoost Descriptor Models**

## Abstract (≈150 words)
The prior XGBoost model predicts acidic/basic pKa from RDKit+Mordred descriptors
at R² ≈ 0.83 but returns a single number with no confidence and no statement of
where it applies. We add (i) distribution-free **conformal prediction intervals**
(split, locally-adaptive/normalized, and conformalized quantile regression) that
achieve their nominal coverage on a held-out test set, and (ii) two
**applicability-domain** definitions (kNN distance in descriptor space; Tanimoto
similarity on Morgan fingerprints). Out-of-domain compounds show ~1.6–2.2× higher
error and are systematically under-covered by marginal intervals, so the AD
flags exactly the predictions that should not be trusted. The result is a
drop-in, laptop-scale, open-source reliability layer for descriptor-based pKa
models.

## 1. Introduction
- pKa matters (solubility, ADMET, reaction design); descriptor + GBM models are
  accurate and cheap.
- Gap: point predictions with no uncertainty and no domain of validity — a known
  weakness of the prior paper (high train R², train–test gap).
- Contribution: a reliability layer — calibrated intervals + AD — that is
  distribution-free, reproducible, and runs on a personal laptop.

## 2. Data and baseline
- Mansouri/DataWarrior acidic & basic pKa, op2 splits (acidic 2.3k train / 765
  test; basic 2.5k train / 843 test); full RDKit+Mordred descriptors (~1.1–1.7k).
- Baseline XGBoost (`hist`, depth 6, 300 trees) reproduces the paper: acidic
  R² 0.826 / RMSE 1.42, basic R² 0.842 / RMSE 1.29.
- **Split for honest UQ:** proper-train (80%) / calibration (20%) / held-out test.

## 3. Methods
### 3.1 Uncertainty
- **Split conformal** — constant-width band from calibration residual quantile;
  finite-sample marginal-coverage guarantee.
- **Normalized conformal** — residuals scaled by a difficulty estimate σ(x) (a
  20-member bootstrap-XGBoost disagreement std) → adaptive width.
- **CQR** — XGBoost quantile-regression band with a conformal correction.
### 3.2 Applicability domain
- **kNN distance** in standardized descriptor space; threshold = 95th percentile
  of training leave-one-out distances.
- **Tanimoto similarity** (Morgan r=2, 2048 bits); threshold = 5th percentile of
  training nearest-neighbour similarity.
### 3.3 Evaluation
- Coverage vs nominal (80/90/95%), interval width (sharpness), calibration curve.
- Error & coverage **inside vs outside** the AD. Strict single test evaluation.

## 4. Results and discussion
- **Calibration:** all three methods reach nominal coverage; normalized conformal
  is sharpest at equal coverage (Fig. calibration, Fig. width).
- **AD value:** out-of-domain MAE 1.6–2.2× higher; marginal coverage drops from
  ~0.9 inside to ~0.62–0.75 outside (Fig. ad_error; Table ad_coverage).
- **Method trade-offs:** split = simplest & guaranteed; normalized = best
  sharpness/adaptivity; CQR = asymmetric but widest here.
- Two AD definitions agree on ~95% of compounds → robust flagging.

## 5. Limitations and future work
- Marginal (not conditional) coverage; Mondrian/AD-conditional conformal next.
- pKa values are wide-range and the descriptor model plateaus at R² ≈ 0.83 —
  the UQ layer reports that ceiling honestly rather than removing it.
- External-set transfer and protonation-state standardization (cf. the symbolic
  track's E6 audit) are natural follow-ups.

## 6. Reproducibility
One command (`python -m sota_pka.uq.cli run --task all`) regenerates every table
and figure; open data, open-source stack, CPU-only, ~2 min on a laptop. Code,
unit tests, and per-compound predictions released.

## Figures / tables to include
- Fig 1 — workflow (descriptors → XGBoost ensemble → conformal + AD).
- Fig 2 — calibration curves (`<task>_calibration.png`).
- Fig 3 — interval sharpness (`<task>_width.png`).
- Fig 4 — AD score vs |error|, in/out coloured (`<task>_ad_error.png`).
- Table 1 — predictive metrics vs original paper (`metrics_predictive.csv`).
- Table 2 — coverage & width by method & level (`coverage.csv`).
- Table 3 — error & coverage inside vs outside AD (`ad_summary.csv`, `ad_coverage.csv`).
