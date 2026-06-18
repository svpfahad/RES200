# Experiment Plan

Goal: produce patent-review evidence for a local-PC CO2 capture solvent screening decision engine.

## Phase 1: Data Acquisition

Primary target:

- Zenodo 3251643 Ionic liquid Properties CO2 capacity file. This is the current main benchmark because it has explicit target values and CC BY 4.0 licensing.
- Figshare article `29228990`: CO2 solubility in diverse ionic liquids.

Secondary targets:

- 2020 Chemical Engineering Science dataset if accessible.
- 2024/2025 open supplementary datasets from Scientific Reports or ACS Figshare.
- Small focused IL mixture datasets for external sanity checks.

License check:

- Record license/source for every file.
- Keep non-commercial data separate from any commercial demo.
- Do not redistribute downloaded raw files outside private research.

## Phase 2: Data Normalization

Required columns:

- target CO2 capture property, such as mole fraction, solubility, Henry constant, or loading;
- temperature;
- pressure;
- solvent identity.

Preferred columns:

- cation;
- anion;
- cation family;
- anion family;
- descriptor columns.

If cation and anion are available, compute:

```text
solvent_id = cation + "|" + anion
```

## Phase 3: Leakage-Safe Splits

Run at least three evaluations:

1. Random row split: only a sanity check.
2. Solvent-held-out split: primary benchmark.
3. Family-held-out split: stronger extrapolation benchmark if family labels exist.

## Phase 4: Models

Baseline models:

- Dummy mean predictor;
- Ridge regression;
- Random Forest;
- HistGradientBoosting;
- XGBoost if installed.

Optional later:

- LightGBM;
- CatBoost;
- graph or molecular models if SMILES are available.

## Phase 5: Uncertainty And Domain Gating

Minimum:

- conformal residual interval;
- numeric min/max range checks;
- unseen categorical level checks;
- rank penalty for interval width and out-of-domain status.

Better:

- normalized conformal intervals with an error model;
- nearest-neighbor distance in descriptor space;
- family-conditioned calibration.

## Phase 6: Decision Engine

For every candidate, output:

- predicted value;
- lower and upper interval;
- interval width;
- confidence grade;
- domain status;
- rank score;
- reason codes;
- decision label.

Decision labels:

- `test`: strong candidate, in domain, tight enough interval;
- `watch`: moderate confidence;
- `needs_data`: promising but unreliable or out of domain;
- `reject`: low predicted value or poor constraints.

## Phase 7: Patent Evidence Package

Create:

- system architecture diagram;
- data-flow diagram;
- model benchmark table;
- interval-coverage table;
- candidate-ranking examples;
- ablation table: point ranking vs confidence-gated ranking;
- prior-art claim chart;
- revised invention disclosure.
