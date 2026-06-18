# Patent Package README

Working title: Confidence-Gated Machine-Learning System for Screening CO2 Capture Solvents

Status: attorney-review draft, not legal advice.

This folder contains a draft patent filing package for a computer-implemented chemical screening invention. The package is designed so a U.S. patent attorney and a Saudi patent agent can quickly review, revise, and file from the same technical disclosure.

## Core Invention

The invention is a local/offline machine-learning workflow that screens candidate solvents for CO2 capture by combining:

1. leakage-aware model training and validation using solvent-held-out or ion-family-held-out splits;
2. calibrated prediction intervals;
3. applicability-domain checks for unseen chemistry and out-of-range process conditions;
4. a rank score that penalizes uncertainty and extrapolation;
5. a decision output that can abstain from recommending a candidate and instead flag it as needing experimental data.

The patentable focus should be the technical decision workflow and evidence-backed refusal logic, not generic use of ML, XGBoost, random forests, descriptors, or CO2 solubility prediction.

## Files

- `01_ATTORNEY_REVIEW_BRIEF.md`: executive technical/legal brief.
- `02_SPECIFICATION_DRAFT.md`: draft specification usable as a U.S. provisional base and as a Saudi specification source after attorney revision and Arabic translation.
- `03_CLAIMS_DRAFT.md`: draft claim set for attorney revision.
- `04_DRAWING_PACKET.md`: drawing list and figure descriptions.
- `05_PRIOR_ART_AND_IDS_CANDIDATES.md`: closest prior art and U.S. IDS candidate list.
- `06_US_FILING_CHECKLIST.md`: U.S. provisional/nonprovisional checklist.
- `07_SAIP_FILING_CHECKLIST.md`: Saudi filing checklist.
- `08_EMAILS_TO_SEND.md`: email drafts for patent counsel / university or company IP office.
- `09_INVENTOR_FACTS_NEEDED.md`: facts still needed before filing.

## Evidence Files

Primary local evidence is in:

- `runs/evidence_suite/summary.md`
- `runs/evidence_suite/summary.csv`
- `runs/evidence_suite/candidate_ranking.csv`
- `runs/evidence_suite/zenodo_solvent_holdout_full/`
- `runs/evidence_suite/zenodo_solvent_holdout_structure_only/`

Reproducible command:

```powershell
python scripts/run_evidence_suite.py
```

Verification command:

```powershell
python -m py_compile src/co2patent/pipeline.py scripts/run_pipeline.py scripts/run_evidence_suite.py scripts/screen_candidates.py
python -m pytest tests -q
```

## Immediate Attorney Instruction

Ask counsel to review:

1. whether to file a U.S. provisional immediately before any disclosure;
2. whether to file Saudi national directly or via PCT/Paris route;
3. whether claim 1 should be method-first, system-first, or computer-readable-medium-first;
4. whether optional apparatus/LIMS/test scheduling language is sufficiently supported;
5. whether any ownership, university, employer, or funding obligation affects filing.
