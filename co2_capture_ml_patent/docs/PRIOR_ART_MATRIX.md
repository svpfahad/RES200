# Prior Art Matrix

This is a technical prior-art triage, not a legal freedom-to-operate opinion.

| Source | What It Covers | Risk | How We Differentiate |
|---|---|---:|---|
| Prediction of CO2 solubility in ionic liquids using machine learning methods, Chemical Engineering Science 2020 | 10,116 CO2 solubility data points; ANN/SVM group-contribution models | High | Avoid claiming CO2 solubility prediction from IL structure and process variables alone |
| Prediction of CO2 solubility in Ionic liquids for CO2 capture using deep learning models, Scientific Reports 2024 | Deep-learning prediction of CO2 solubility for CO2 capture | High | Claim decision-grade reliability, not a deep learning architecture |
| Figshare 29228990 / Energy & Fuels supplement | 16,480 experimental CO2 solubility points in 296 ILs; descriptor-based ML models | High | Use as evidence data; do not claim the dataset, descriptors, or LSBoost-style modeling |
| Screening Environmentally Benign Ionic Liquids for CO2 Absorption Using Representation Uncertainty-Based Machine Learning, EST Letters 2024 | Uncertainty-based ML screening of environmentally benign ILs | High | Narrow to conformal interval calibration, solvent-held-out leakage controls, Saudi operating-window scoring, and decision output |
| CPSign conformal prediction for cheminformatics, J. Cheminformatics 2024 | Conformal prediction intervals for cheminformatics | Medium | Use conformal prediction as a component; invention is the CO2 solvent screening decision workflow |
| WO2024187266A1 | ML monitoring of amine-based carbon-capture solvent deterioration during plant operation | Medium | We are screening/ranking new candidates from public property data, not monitoring solvent degradation in a plant loop |
| CN116230115B | Phase-change absorbent screening using ML and quantum chemistry | High | Avoid broad absorbent screening claims; focus on confidence-gated candidate triage with calibrated intervals and operating-window decision rules |
| Generic molecular ML patents | Molecular descriptors, graph models, uncertainty, and candidate screening | High | Need concrete application, data controls, technical outputs, and measurable improvement |

## Patentability Guardrails

Saudi patent law requires novelty, inventive step, and industrial applicability, while mathematical methods are excluded as inventions. The claim must therefore be framed as a technical system or method for solvent-screening decisions, not as a mathematical model by itself.

## Initial Claim Gap

The most defensible gap to test:

> A local computer-implemented system that ranks CO2 capture solvent candidates under specified operating conditions by combining a solvent-identity leakage-safe trained model, conformal prediction intervals, applicability-domain checks, and decision rules that abstain from or downgrade unreliable candidates.

The gap is weak unless experiments show that the confidence gate changes decisions in a useful way compared with point-prediction ranking.

