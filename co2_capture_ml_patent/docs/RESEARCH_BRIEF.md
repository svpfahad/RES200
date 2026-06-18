# Research Brief

Date: 2026-05-12

Working title: Confidence-Gated ML System for Screening CO2 Capture Solvents Under Saudi Operating Conditions

## Bottom Line

The strongest local-PC ML patent track is not generic property prediction. The stronger direction is a computer-implemented decision system that uses public CO2 capture solvent data to rank solvent candidates only when predictions are reliable enough, and that downgrades or abstains when candidates are outside the model's applicability domain.

This gives the work a technical effect beyond "use ML to predict solubility":

- lower risk of false-positive solvent recommendations;
- explicit operating-window handling through temperature, pressure, and optional gas/process constraints;
- calibrated prediction intervals;
- out-of-domain detection;
- decision output suitable for experimental triage.

## Why This Direction

1. Saudi industrial fit is real.

Aramco has publicly described CO2 direct-air-capture testing in Saudi Arabia and a Jubail CCS hub. That makes carbon-capture material and solvent screening a Saudi-relevant application area.

2. Public data exists.

The Figshare CO2/ionic-liquid dataset reports 16,480 experimental data points across 296 ionic liquids, with 103 cation structures and 78 anion structures. It is licensed CC BY-NC 4.0, so it is suitable for local research and prototype evidence, but commercial or patent-filing use should be reviewed.

The Zenodo `Ionic liquid Properties` dataset is better for the first benchmark because it provides explicit CO2 capacity targets plus cation/anion SMILES under CC BY 4.0. The prepared local file currently contains 10,865 valid rows and 216 solvent identities after filtering nonpositive pressure or xCO2 values.

3. The prior art is crowded but not closed.

Many papers already predict CO2 solubility in ionic liquids with ANN, SVM, RF, boosting, deep learning, COSMO-RS descriptors, and other ML methods. A patent claim saying "train ML on descriptors to predict CO2 solubility" is weak. A narrower decision system with leakage-safe validation, conformal confidence, applicability-domain gating, and operating-window scoring is a better target.

4. It is local-PC feasible.

The core models are tabular models: Ridge, Random Forest, HistGradientBoosting, XGBoost if installed, and optional CatBoost/LightGBM later. These run on CPU with public datasets at the 10k to 20k row scale.

## Recommended Research Question

Can a leakage-safe, confidence-gated ML decision engine rank CO2 capture solvent candidates more safely than a point-prediction model by combining:

- solvent-held-out validation;
- calibrated prediction intervals;
- applicability-domain detection;
- operating-window scoring; and
- explainable reason codes?

## Proposed Technical Contribution

Input:

- solvent candidate identifier;
- cation/anion or descriptor columns;
- temperature and pressure;
- optional constraints such as solvent family, cost, toxicity flag, viscosity proxy, or Saudi climate operating profile.

Processing:

1. Normalize the tabular dataset.
2. Split by solvent identity to prevent leakage.
3. Train baseline and ensemble ML regressors.
4. Calibrate prediction intervals with conformal residuals.
5. Build an applicability-domain profile from training data.
6. Score candidates by predicted CO2 uptake, interval width, out-of-domain penalties, and user constraints.

Output:

- predicted CO2 solubility or related target;
- lower and upper prediction interval;
- confidence grade;
- domain status;
- rank score;
- reason codes;
- decision label: `test`, `watch`, `needs_data`, or `reject`.

## What To Avoid

- Do not claim generic XGBoost, Random Forest, neural networks, descriptors, or fingerprints.
- Do not claim discovery of a specific solvent composition unless experiments support it.
- Do not publish diagrams, claim language, or candidate rankings before filing review.
- Do not rely on random row splits as the main evidence; they can leak solvent identity across train and test.

## First Evidence Milestones

1. Reproduce a baseline on public data.
2. Compare random row split vs solvent-held-out split.
3. Show that confidence-gated ranking reduces bad high-confidence recommendations.
4. Show calibrated interval coverage near 90%.
5. Show candidate examples where the system abstains because the candidate is outside training domain.
6. Build a claim chart against close papers and patents.
