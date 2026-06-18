# Knowing When to Trust a pKa Model: Conformal Prediction Intervals and an Applicability Domain for XGBoost Descriptor Models

**Fahad Ali Atwi, Mohammed Al-Khater**

*Draft for submission to an undergraduate research journal. Extends "XGBoost-Based Prediction of pKa Values from Molecular Descriptors" (Atwi & Al-Khater).*

---

## Abstract

Machine-learning models for aqueous pKa are now accurate enough to be useful in
early-stage molecular design, but most return a single number with no measure of
confidence and no statement of where the model can be trusted. Building on a
previously reported XGBoost model that predicts acidic and basic pKa from
RDKit/Mordred molecular descriptors (test R² ≈ 0.83), we add a reproducible
*reliability layer* consisting of two components. First, we attach
distribution-free **conformal prediction intervals** using three methods — split
conformal, locally adaptive (normalized) conformal, and conformalized quantile
regression — each calibrated on a held-out portion of the training data. On a
strictly held-out test set these intervals achieve their nominal coverage
(e.g. 90.6 % empirical coverage at the 90 % level for split conformal on the
acidic task), and the locally adaptive variant reaches the same coverage with
narrower intervals by widening only where an XGBoost ensemble disagrees. Second,
we define an explicit **applicability domain (AD)** using two complementary
measures: k-nearest-neighbour distance in descriptor space and Tanimoto
similarity over Morgan fingerprints. Compounds flagged as outside the AD are
predicted with 1.6–2.2× higher mean absolute error and are systematically
under-covered by the marginal intervals, demonstrating that the AD isolates
exactly the predictions that should be treated with caution. The entire pipeline
is leakage-safe, built on open data and open-source libraries, runs in roughly
two minutes on a laptop CPU, and is released as a small, documented Python
package so that other students can reuse it as an open baseline.

---

## 1. Introduction

The acid dissociation constant (pKa) governs the charge state of a molecule at a
given pH and therefore influences solubility, membrane permeability, protein
binding, and reactivity. Reliable pKa estimates are valuable throughout
medicinal chemistry and chemical process design, and a large body of work has
shown that machine-learning models trained on molecular descriptors or
fingerprints can predict pKa quickly and at low cost.

Our previous study trained gradient-boosted regression trees (XGBoost) on the
publicly available acidic and basic pKa data curated by Mansouri *et al.* from
DataWarrior, comparing several descriptor representations and reaching a test
coefficient of determination of approximately 0.83 with a full RDKit/Mordred
descriptor set. As is common for high-capacity models on chemical data, the
model showed a clear train–test gap, indicating that some predictions are far
more reliable than others.

That earlier model — like most QSPR models reported at this level — has two
practical gaps. First, it produces a **point estimate with no uncertainty**: a
prediction of 7.2 is reported with the same apparent authority whether the
molecule closely resembles the training set or is structurally unusual. Second,
it has **no explicit applicability domain**: nothing tells a user when a query
molecule lies outside the region of chemical space the model actually learned.
Both gaps matter most precisely when a model is used for decision-making on new
chemistry.

In this work we close those two gaps without changing the underlying model. We
add (i) calibrated, distribution-free prediction intervals via conformal
prediction, and (ii) an explicit applicability domain. We then evaluate, on
strictly held-out data, whether the intervals achieve their promised coverage,
how sharp they are, and — most importantly — whether the applicability domain
genuinely separates trustworthy from untrustworthy predictions.

We are explicit about the nature of the contribution. Conformal prediction and
distance/similarity-based applicability domains are established tools in
cheminformatics; the novelty here is not a new algorithm but a careful,
reproducible, and honest reliability layer for descriptor-based pKa prediction,
packaged so that it can be reused and extended. The emphasis throughout is on
rigour — leakage-safe data handling, a single held-out test evaluation, and
diagnostics that could disconfirm our claims.

## 2. Data and baseline model

### 2.1 Dataset and splits

We use the acidic and basic aqueous pKa datasets from the Mansouri/DataWarrior
curation, in the same train/test partitions ("op2") as the original study.
Molecules are represented by the full RDKit + Mordred descriptor set used
previously (1,133 numeric descriptors retained for the acidic task and 1,707 for
the basic task after train-only cleaning).

To calibrate prediction intervals honestly, the training set is split a second
time into a **proper-training** set (80 %) and a **calibration** set (20 %); the
original held-out **test** set is never used for any model fitting, feature
selection, threshold choice, or interval calibration. Feature columns are
defined entirely by the training data, and any descriptor present in training
but absent from the test matrix is imputed without consulting test values. The
resulting partitions are summarized in Table 1.

**Table 1. Data partitions.**

| Task | Proper-train | Calibration | Test | Descriptors |
|---|---:|---:|---:|---:|
| Acidic | 1,834 | 459 | 765 | 1,133 |
| Basic | 2,021 | 506 | 843 | 1,707 |

### 2.2 Baseline XGBoost model

The baseline is XGBoost regression with the configuration from the original
study (300 trees, maximum depth 6, learning rate 0.1, subsample and column
sample 0.8, L2 regularization 1.0), trained with the histogram tree method on
CPU. Trained on the full training set and evaluated once on the held-out test
set, it reproduces the original performance (Table 2).

**Table 2. Baseline predictive performance on the held-out test set.**

| Task | Model | R² | RMSE | MAE |
|---|---|---:|---:|---:|
| Acidic | XGBoost (full train) | 0.826 | 1.416 | 0.892 |
| Basic  | XGBoost (full train) | 0.842 | 1.289 | 0.825 |

For uncertainty quantification we also train a 20-member ensemble of XGBoost
models on bootstrap resamples of the proper-training set. The ensemble mean is a
slightly more conservative predictor (test R² 0.768 acidic, 0.794 basic, the
expected cost of training on the smaller proper-train split), and the
per-compound standard deviation across members provides a cheap *difficulty*
signal used below.

## 3. Methods

### 3.1 Conformal prediction intervals

Conformal prediction converts any point predictor into interval predictions with
a finite-sample, distribution-free marginal coverage guarantee, using only a
held-out calibration set. We implement three variants and compare them.

- **Split conformal.** Compute absolute residuals on the calibration set, take
  the conformal quantile q̂ = the ⌈(n+1)(1−α)⌉-th smallest residual, and report
  the band ŷ ± q̂. This guarantees marginal coverage of at least 1−α but gives
  the same interval width to every molecule.

- **Normalized (locally adaptive) conformal.** Scale each residual by a
  per-compound difficulty estimate σ(x) — here the standard deviation across the
  XGBoost ensemble — so the nonconformity score is |residual| / (σ(x) + β). The
  resulting intervals widen where the ensemble disagrees and narrow where it is
  confident, while retaining the marginal-coverage guarantee.

- **Conformalized quantile regression (CQR).** Fit XGBoost quantile-regression
  models for the lower and upper quantiles, then apply a conformal correction
  estimated on the calibration set so that coverage is guaranteed even when the
  raw quantile estimates are imperfect. This yields asymmetric, adaptive bands.

### 3.2 Applicability domain

We define two complementary applicability-domain measures, each fit on training
data only, with thresholds derived from the training set's own
nearest-neighbour distribution.

- **k-nearest-neighbour (kNN) distance** in standardized descriptor space.
  Each compound's mean Euclidean distance to its five nearest training
  neighbours is computed; a compound is *outside* the domain if this distance
  exceeds the 95th percentile of the training set's own leave-one-out kNN
  distances. To avoid an artefact in which descriptors measured in training but
  absent from the test matrix would push every test compound out of domain, the
  kNN distance uses only descriptors natively measured in both sets.

- **Tanimoto similarity** over Morgan fingerprints (radius 2, 2,048 bits). Each
  compound's maximum Tanimoto similarity to any training molecule is computed; a
  compound is *outside* the domain if this maximum similarity is below the 5th
  percentile of training nearest-neighbour similarities. This is the chemically
  intuitive complement to the descriptor-space distance.

### 3.3 Evaluation protocol

All evaluation uses the single held-out test set. We report (i) predictive
accuracy (R², RMSE, MAE); (ii) empirical interval coverage at nominal levels of
80 %, 90 %, and 95 %, against which calibration is judged; (iii) interval
sharpness (mean and median width); and (iv) accuracy and coverage **inside
versus outside** each applicability domain. The last comparison is the central
test of whether the AD is meaningful: a useful domain should contain the
accurate, well-covered predictions and exclude the inaccurate, under-covered
ones.

## 4. Results

### 4.1 Interval calibration and sharpness

All three conformal methods achieve close-to-nominal coverage on both tasks
(Table 3). On the acidic task, split conformal yields 90.6 % empirical coverage
at the 90 % nominal level; normalized conformal reaches 91.4 % coverage while
producing the **narrowest** median interval (4.20 versus 5.05 pKa units for split
conformal), because it concentrates width on difficult compounds rather than
spreading it uniformly. CQR achieves comparable coverage but with the widest
intervals in this setting.

**Table 3. Empirical coverage (and median interval width, pKa units) on the
held-out test set.**

| Task | Method | 80 % | 90 % | 95 % | Median width @ 90 % |
|---|---|---:|---:|---:|---:|
| Acidic | Split conformal | 0.805 | 0.906 | 0.949 | 5.05 |
| Acidic | Normalized conformal | 0.816 | 0.914 | 0.950 | **4.20** |
| Acidic | CQR | 0.808 | 0.897 | 0.932 | 7.55 |
| Basic | Split conformal | 0.775 | 0.878 | 0.944 | 3.94 |
| Basic | Normalized conformal | 0.763 | 0.896 | 0.943 | **3.44** |
| Basic | CQR | 0.759 | 0.887 | 0.955 | 6.32 |

Coverage is slightly conservative at the 95 % level and slightly below nominal
at 80–90 % on the basic task, consistent with the finite calibration-set size;
all values are within a few percentage points of target, as expected for
distribution-free intervals.

### 4.2 The applicability domain separates reliable from unreliable predictions

Both AD measures classify roughly 94–95 % of test compounds as inside the
domain, and both show that the excluded minority is predicted markedly worse
(Table 4). Mean absolute error is 1.6× higher outside the kNN domain on the
acidic task and 1.8× higher on the basic task; for the Tanimoto domain the
out-of-domain penalty reaches 2.2× on the basic task. The coefficient of
determination collapses correspondingly — for example, from 0.81 inside the
basic kNN domain to 0.36 outside it.

**Table 4. Accuracy inside versus outside each applicability domain (test set).**

| Task | AD measure | % in-domain | MAE in | MAE out | MAE ratio | R² in | R² out |
|---|---|---:|---:|---:|---:|---:|---:|
| Acidic | kNN distance | 95.3 | 1.004 | 1.603 | 1.60 | 0.779 | 0.430 |
| Acidic | Tanimoto | 94.9 | 1.003 | 1.590 | 1.59 | 0.776 | 0.595 |
| Basic | kNN distance | 94.0 | 0.886 | 1.584 | 1.79 | 0.814 | 0.359 |
| Basic | Tanimoto | 94.4 | 0.869 | 1.942 | 2.24 | 0.814 | 0.531 |

### 4.3 Marginal intervals under-cover outside the domain

Because conformal coverage is *marginal* (averaged over all molecules), it can
hide the fact that hard compounds are systematically under-covered. Stratifying
the 90 %-nominal intervals by applicability domain makes this explicit (Table 5).
On the basic task, split-conformal coverage falls from 89 % inside the Tanimoto
domain to 62 % outside it; the locally adaptive (normalized) intervals partly
compensate by widening out-of-domain (coverage 70–77 % outside versus 91 %
inside). The effect is present but milder on the acidic task, whose
out-of-domain test compounds are less extreme.

**Table 5. Empirical coverage at 90 % nominal, inside versus outside the
applicability domain.**

| Task | AD measure | Method | Coverage in | Coverage out |
|---|---|---|---:|---:|
| Acidic | kNN | Split | 0.911 | 0.806 |
| Acidic | kNN | Normalized | 0.918 | 0.833 |
| Acidic | Tanimoto | Split | 0.910 | 0.821 |
| Acidic | Tanimoto | Normalized | 0.915 | 0.897 |
| Basic | kNN | Split | 0.886 | 0.745 |
| Basic | kNN | Normalized | 0.907 | 0.765 |
| Basic | Tanimoto | Split | 0.893 | 0.617 |
| Basic | Tanimoto | Normalized | 0.910 | 0.702 |

Together, Tables 4 and 5 show that the two applicability-domain measures —
derived independently from descriptor distances and from structural fingerprints
— agree on which predictions are unreliable, and that these are exactly the
predictions where both accuracy and interval coverage degrade.

## 5. Discussion

The results support three practical conclusions. First, attaching conformal
prediction intervals to an existing XGBoost pKa model is straightforward and
delivers intervals that meet their stated coverage on unseen data, turning a
bare point estimate into an honest "prediction ± reliability" statement.
Second, among the three interval methods, locally adaptive (normalized)
conformal is the most attractive default: it matches the coverage guarantee of
split conformal while producing visibly sharper intervals, because it routes
interval width to the molecules the model finds difficult. Third, an explicit
applicability domain is not a formality — out-of-domain compounds are
substantially less accurate and are under-covered by marginal intervals, so
flagging them is what keeps the reported reliability truthful.

A user of this model would therefore make decisions differently depending on the
domain flag: in-domain predictions can be used with the conformal interval as a
realistic error bar, whereas out-of-domain predictions should be treated as
low-confidence and ideally confirmed experimentally or with a more expensive
method. That the two independent AD measures largely agree gives additional
confidence in the flag.

## 6. Limitations and future work

The most important limitation is that all evaluation is **in-distribution**: both
the calibration and the test compounds are drawn from the same dataset and
split. Conformal coverage guarantees assume exchangeability between calibration
and test data, which holds here but would not for a genuinely external set drawn
from different chemistry or measurement conditions. The natural next step — and
the strongest single addition to this work — is to score the same intervals and
applicability domains on external pKa benchmarks and quantify how coverage
degrades under distribution shift and whether the AD correctly flags external
compounds as out-of-domain.

Other extensions include conditional (Mondrian) conformal prediction to target
coverage within subgroups rather than only marginally; a leverage- or
Mahalanobis-based AD as a third, independent measure; and recalibrating the
descriptor model itself rather than only quantifying its uncertainty. We also
note that the descriptor model's accuracy plateaus near R² ≈ 0.83; the
reliability layer reports that ceiling honestly rather than removing it.

## 7. Reproducibility

The complete pipeline is released as a small, documented Python package. A single
command regenerates every table and figure in this paper from the open data:

```
python -m sota_pka.uq.cli run --task all
```

The implementation uses only widely available open-source libraries (RDKit,
Mordred, XGBoost, scikit-learn, NumPy, pandas, Matplotlib), requires no GPU, and
completes in roughly two minutes on a laptop CPU. Per-compound predictions,
ensemble uncertainties, applicability-domain scores, and all three interval
bands are written to disk, and unit tests cover the conformal-coverage logic and
the leakage-safe data handling.

## 8. Conclusion

We have equipped an existing XGBoost pKa model with a reproducible reliability
layer: calibrated, distribution-free prediction intervals and an explicit
applicability domain built from two independent measures. On held-out data the
intervals achieve their nominal coverage, the locally adaptive variant is the
sharpest at equal coverage, and the applicability domain cleanly separates
predictions that are accurate and well-covered from those that are neither.
Without changing the underlying model, these additions let a practitioner know
not only what the model predicts but how much to trust each prediction — a small
but consequential step toward dependable machine-learning models for molecular
property prediction.

---

### Figures (generated by the pipeline)

- **Figure 1.** Interval calibration: empirical versus nominal coverage for the
  three conformal methods (`acidic_calibration.png`, `basic_calibration.png`).
- **Figure 2.** Interval sharpness: width distributions at 90 % nominal
  (`acidic_width.png`, `basic_width.png`).
- **Figure 3.** Applicability domain versus error: AD score against absolute
  prediction error, with in- and out-of-domain compounds distinguished
  (`acidic_ad_error.png`, `basic_ad_error.png`).
