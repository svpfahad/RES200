# Invention Disclosure Draft

Working title:

Confidence-Gated Machine-Learning Decision System for Screening CO2 Capture Solvents

## Field

Computer-implemented chemical screening, carbon capture, solvent selection, molecular property prediction, and machine-learning decision support.

## Problem

CO2 capture solvent screening is expensive because many candidate solvents and operating conditions can be tested. Existing ML approaches can predict CO2 solubility, but point predictions can be overconfident for candidates outside the training domain. Random train/test splits can also overstate performance when the same solvent appears in both training and test rows at different temperatures or pressures.

## Proposed Solution

A local computer-implemented system receives solvent candidate features and operating conditions, predicts CO2 capture performance, calibrates a prediction interval, checks applicability domain, and outputs a ranked decision recommendation.

Core steps:

1. Load public experimental CO2 solvent data.
2. Create a solvent identity key from a provided solvent ID or from cation and anion fields.
3. Split training and evaluation data by solvent identity.
4. Train one or more regression models to predict a CO2 capture target.
5. Use held-out calibration residuals to compute conformal prediction intervals.
6. Build a training-domain profile from numeric feature ranges and categorical feature levels.
7. For each candidate, predict performance and interval width.
8. Penalize candidates with wide intervals, out-of-domain feature values, or constraint violations.
9. Output rank score, prediction interval, confidence grade, domain status, reason codes, and decision label.

## Potential Novel Elements To Test

- Solvent-held-out model training and evaluation as a required gate before ranking.
- Candidate ranking using both predicted performance and calibrated uncertainty.
- Applicability-domain rule set that converts unseen cations/anions or outside-range process conditions into decision labels.
- Saudi operating-window presets, such as higher ambient temperatures or industrial pressure windows, used as candidate scoring conditions.
- Automatic `needs_data` recommendation when a candidate is promising but outside the model domain.

## Example Output

| candidate_id | prediction | interval | domain_status | confidence_grade | decision |
|---|---:|---|---|---|---|
| IL_A_313K_10bar | 0.41 | 0.32 to 0.50 | in_domain | A | test |
| IL_B_333K_40bar | 0.62 | 0.21 to 1.03 | edge | C | needs_data |
| IL_C_358K_80bar | 0.70 | 0.10 to 1.30 | out_of_domain | D | needs_data |

## Evidence Needed Before Filing Review

- Reproducible public-data benchmark.
- Solvent-held-out results.
- Interval coverage and interval-width tables.
- Candidate-ranking ablation: point prediction vs confidence-gated ranking.
- Out-of-domain refusal examples.
- Prior-art claim chart.

## Disclosure Risk

Keep this disclosure private until reviewed. Public disclosure can damage patent rights in some jurisdictions.

