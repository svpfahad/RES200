# Claim Strategy

This is technical claim planning, not legal advice.

## Stronger Claim Lane

A computer-implemented method for screening CO2 capture solvents, comprising:

1. receiving solvent candidate data and operating condition data;
2. training a regression model using experimental CO2 capture data split by solvent identity;
3. calibrating prediction intervals using a calibration set;
4. determining applicability-domain status from training-domain statistics;
5. generating a candidate rank score from predicted performance, interval width, domain status, and operating constraints;
6. outputting a decision label that can abstain from ranking or flag the candidate as needing data.

## Dependent Claim Ideas

- Solvent identity is generated from cation and anion identifiers.
- Operating conditions include temperature and pressure.
- The prediction interval is produced by conformal calibration of residuals.
- Applicability-domain status includes unseen cation, unseen anion, and numeric condition range checks.
- The rank score penalizes uncertainty and domain violations.
- The system outputs reason codes.
- Saudi operating-window presets are applied to rank candidates under regional deployment conditions.
- The system recommends experiments for promising candidates with insufficient domain support.

## Weak Claim Lane To Avoid

- "Use machine learning to predict CO2 solubility."
- "Use XGBoost or Random Forest for ionic liquids."
- "Use descriptors for cations and anions."
- "Use uncertainty in screening" without a concrete gating and decision workflow.
- "Discover a solvent" without lab evidence.

## Evidence That Makes The Claim Stronger

- Held-out solvent families are harder than random splits.
- Confidence-gated ranking improves candidate triage vs point-prediction ranking.
- The system correctly marks extrapolative candidates as `needs_data`.
- Reason codes are consistent and reproducible.
- The system runs locally from public datasets without cloud services or proprietary data.

