# Attorney Review Brief

Status: attorney-review draft, not legal advice.

## One-Sentence Summary

A local computer-implemented system screens CO2 capture solvent candidates by predicting capture performance, calibrating prediction intervals on chemically held-out data, detecting out-of-domain chemistry and operating conditions, and outputting a ranked or abstained decision with reason codes for experimental triage.

## Technical Problem

Machine-learning models for CO2 solvent and ionic-liquid screening can appear accurate under random row splits because the same solvent identity can appear in both training and test records at different pressures or temperatures. When a genuinely new cation, anion, solvent family, or operating window is presented, point predictions can be overconfident and can cause poor experimental selection.

## Proposed Technical Solution

The system:

1. receives experimental solvent records and candidate solvent records;
2. forms solvent, cation, and anion identity information and molecular descriptors;
3. trains one or more regression models for a CO2 capture response;
4. validates the model under solvent-held-out or ion-family-held-out splits;
5. calibrates a prediction interval from held-out calibration residuals;
6. builds an applicability-domain profile from training features;
7. scores candidates using predicted response, interval width, domain status, and operating constraints;
8. suppresses recommendations for out-of-domain candidates by assigning `needs_data`;
9. outputs prediction interval, confidence grade, domain status, reason codes, rank score, and decision label.

## Local Evidence

Dataset: Zenodo record 3251643, `CO2CAPACITY.txt` and `CA.smi`, prepared into 10,865 valid records and 216 solvent identities.

Evidence summary:

| run | split | best model | R2 | MAE | RMSE | coverage | interval width |
|---|---|---:|---:|---:|---:|---:|---:|
| `zenodo_random_full` | random row | random forest | 0.9402 | 0.1636 | 0.2985 | 0.9146 | 0.8980 |
| `zenodo_solvent_holdout_full` | solvent held out | XGBoost | 0.3226 | 0.4465 | 0.8874 | 0.8596 | 1.4720 |
| `zenodo_solvent_holdout_structure_only` | solvent held out, no ion codes | XGBoost | 0.2529 | 0.5214 | 0.9319 | 0.8910 | 1.8151 |
| `zenodo_cation_holdout_no_cation_code` | cation held out | hist gradient boosting | 0.5241 | 0.4211 | 0.8754 | 0.9342 | 2.9450 |
| `zenodo_anion_holdout_no_anion_code` | anion held out | XGBoost | 0.7901 | 0.3633 | 0.5953 | 0.9452 | 2.3826 |

Candidate screen:

- In-domain observed median candidate: `test`.
- Unseen cation candidate: `needs_data`.
- Deliberate high-pressure unseen candidate: `needs_data`.
- Low-performing in-domain candidate: `reject`.

## Why This Is Not Just "Use ML"

The strongest claim lane is not a model architecture. It is a decision system that converts uncertainty and domain validity into an experimental action. The practical output is not just a predicted value; it is a controlled go/no-go recommendation with abstention and reason codes.

## Closest Prior-Art Risks

Highest-risk prior art:

- Zhong et al., 2024, representation uncertainty for environmentally benign ionic-liquid CO2 absorption screening.
- Norinder et al., 2014, conformal prediction in cheminformatics.
- CPSign, 2024, conformal prediction tooling for cheminformatics.
- ML/QSPR papers predicting CO2 solubility in ionic liquids, including 2017, 2024, and 2025 works.
- General solvent-screening patents using ML and uncertainty in process optimization.

Main differentiation:

- use of solvent-held-out or ion-family-held-out validation as a screening gate;
- formal calibrated intervals tied to decision suppression;
- explicit out-of-domain refusal for unseen cation/anion/process regimes;
- local/offline workflow using public or private records;
- auditable reason codes for experimental triage in CO2 capture solvent selection.

## U.S. Filing Strategy

Preferred first step: file a detailed U.S. provisional before any public disclosure. Include the specification, flowcharts, evidence tables, code-level examples, and candidate-ranking outputs. Then refine claims for nonprovisional/PCT filing within 12 months.

Eligibility framing:

- avoid claiming math, ranking, or information display in isolation;
- claim a practical chemical screening process;
- include optional outputs that generate an experimental test queue or test protocol;
- include a system configured to perform the method;
- include non-transitory computer-readable medium claims, not "signal" or "carrier wave" claims.

## Saudi Filing Strategy

Preferred Saudi framing: a computer-implemented industrial process for screening CO2 capture solvents, not a mathematical method. Saudi law identifies patentable inventions as products, industrial processes, or related subject matter when new, involving an innovative step, and industrially applicable. The Saudi law also excludes discoveries, scientific theories, mathematical methods, pure mental/business rules, and other categories. The specification should emphasize industrial use, chemical process relevance, and technical decision control.

Saudi-specific package items likely needed:

- Arabic specification/claims/abstract or compliant translation path;
- Saudi agent if applicant resides outside the Kingdom;
- applicant/inventor details;
- assignment evidence if applicant is not inventor;
- priority documents if claiming foreign priority;
- disclosure history documents;
- power of attorney for Saudi agent.

## Counsel Questions

1. Should the first filing be U.S. provisional, Saudi national, or PCT-first?
2. Is there any public disclosure that already occurred?
3. Are all inventors named correctly?
4. Does any university, employer, grant, or contractor own rights?
5. Should claims include apparatus/test-control steps or remain software/process only?
6. Should any candidate solvent compositions be claimed after further experimental testing?
