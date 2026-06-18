# Current Results

Date: 2026-05-12

## Local Smoke Run

Command:

```powershell
python scripts/run_pipeline.py --data data/sample/demo_co2_solubility.csv --target co2_solubility_mol_frac --group solvent_id --out runs/demo
python scripts/screen_candidates.py --model runs/demo/model.joblib --candidates data/sample/candidate_screen.csv --out runs/demo/candidate_ranking.csv
```

Result:

- The local code path works.
- Reliable candidates are sorted above extrapolative candidates.
- Out-of-domain candidates are labeled `needs_data`.
- This sample is synthetic and must not be used as scientific evidence.

## Figshare 29228990 Real-Data Run

Commands:

```powershell
python scripts/download_figshare.py --article-id 29228990 --out data/raw/figshare_29228990
python scripts/prepare_figshare_29228990.py
python scripts/run_pipeline.py --data data/processed/figshare_29228990_unique_ils.csv --target ln_co2_solubility_exp --group solvent_id --out runs/figshare_s4_strict
```

Prepared data:

- Source file: `data/raw/figshare_29228990/ef5c01345_si_001.xlsx`
- Prepared file: `data/processed/figshare_29228990_unique_ils.csv`
- Rows with explicit experimental target: 308
- Unique solvent identities: 7
- Target: `ln_co2_solubility_exp`

Strict solvent-held-out benchmark:

| model | R2 | MAE | RMSE |
|---|---:|---:|---:|
| xgboost | 0.4606 | 0.3012 | 0.3644 |
| random_forest | 0.3094 | 0.3471 | 0.4123 |
| ridge | 0.2725 | 0.3539 | 0.4232 |
| hist_gradient_boosting | -0.1092 | 0.4334 | 0.5225 |
| dummy_mean | -6.4382 | 1.2588 | 1.3531 |

Conformal interval check:

- Calibration qhat: 1.5624 log-solubility units.
- Test interval coverage: 1.000 on 84 held-out rows.
- Interpretation: coverage is conservative because the verified-target dataset is small and the calibration split has few solvent identities.

## Evidence Status

This is a working local implementation and a first public-data proof of workflow, not yet patent-ready evidence.

Main limitation:

- The downloaded workbook reports a 16,480-row dataset, but the large `S2` and `S3` sheets expose model predictions and ARD values rather than a clean experimental target column. The explicit experimental target is available in the smaller `S4` sheet only.

Next evidence task:

- Obtain or verify the full experimental target column for the 16,480-row dataset, or add another public dataset with explicit target values.
- Then rerun solvent-held-out and family-held-out benchmarks, interval calibration, and candidate-ranking ablations.

## Zenodo 3251643 Main Evidence Run

Commands:

```powershell
python scripts/download_zenodo_record.py --record-id 3251643 --out data/raw/zenodo_3251643 --files CO2CAPACITY.txt CA.smi
python scripts/prepare_zenodo_3251643.py
python scripts/run_pipeline.py --data data/processed/zenodo_3251643_co2capacity.csv --target ln_co2_solubility --group solvent_id --out runs/zenodo_co2_strict
python scripts/run_pipeline.py --data data/processed/zenodo_3251643_co2capacity.csv --target ln_co2_solubility --group NONE --out runs/zenodo_co2_random
python scripts/run_evidence_suite.py
python scripts/make_zenodo_candidate_examples.py
python scripts/screen_candidates.py --model runs/zenodo_co2_strict/model.joblib --candidates data/processed/zenodo_candidate_examples.csv --out runs/zenodo_co2_strict/candidate_ranking.csv
```

Prepared data:

- Source record: Zenodo `3251643`, `Ionic liquid Properties`.
- Local raw files: `data/raw/zenodo_3251643/CO2CAPACITY.txt`, `data/raw/zenodo_3251643/CA.smi`.
- Prepared file: `data/processed/zenodo_3251643_co2capacity.csv`.
- Valid rows after filtering nonpositive pressure or xCO2 values: 10,865.
- Unique solvent identities: 216.
- Target: `ln_co2_solubility`.

Strict solvent-held-out benchmark:

| model | R2 | MAE | RMSE |
|---|---:|---:|---:|
| xgboost | 0.3322 | 0.4362 | 0.8810 |
| hist_gradient_boosting | 0.2942 | 0.4316 | 0.9058 |
| random_forest | 0.2024 | 0.4579 | 0.9628 |
| dummy_mean | -0.0328 | 0.8743 | 1.0956 |
| ridge | -0.2148 | 0.6970 | 1.1883 |

Strict split uncertainty:

- Calibration qhat: 0.6939 log-solubility units.
- Test interval coverage: 0.8496 on 2,707 rows.
- Mean interval width: 1.3877.

True random-row benchmark:

| best model | R2 | MAE | RMSE | interval coverage | mean interval width |
|---|---:|---:|---:|---:|---:|
| random_forest | 0.9441 | 0.1566 | 0.2885 | 0.9139 | 0.8394 |

Interpretation:

- Random row splitting gives very high apparent accuracy because the same solvent identities can appear in both training and test rows.
- Solvent-held-out splitting is much harder and is the relevant patent-evidence split for screening new solvents.
- The large gap between random and solvent-held-out results supports the need for the proposed confidence-gated decision system.

Candidate screen:

- Output: `runs/zenodo_co2_strict/candidate_ranking.csv`.
- In-domain candidate: `test`.
- Deliberate high-pressure unseen candidate: `needs_data`.
- Held-out candidate with unseen cation code/SMILES: `needs_data`.

## Expanded Evidence Suite

Output folder: `runs/evidence_suite`.

| run | group | excluded columns | best model | R2 | MAE | RMSE | interval coverage | mean interval width | test rows |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `zenodo_random_full` | none | none | random forest | 0.9402 | 0.1636 | 0.2985 | 0.9146 | 0.8980 | 2,717 |
| `zenodo_solvent_holdout_full` | solvent id | none | XGBoost | 0.3226 | 0.4465 | 0.8874 | 0.8596 | 1.4720 | 2,707 |
| `zenodo_solvent_holdout_structure_only` | solvent id | cation code, anion code | XGBoost | 0.2529 | 0.5214 | 0.9319 | 0.8910 | 1.8151 | 2,707 |
| `zenodo_cation_holdout_no_cation_code` | cation code | cation code | hist gradient boosting | 0.5241 | 0.4211 | 0.8754 | 0.9342 | 2.9450 | 2,479 |
| `zenodo_anion_holdout_no_anion_code` | anion code | anion code | XGBoost | 0.7901 | 0.3633 | 0.5953 | 0.9452 | 2.3826 | 5,947 |

Candidate screen from `runs/evidence_suite/candidate_ranking.csv`:

- `zenodo_observed_q50`: in domain, confidence B, decision `test`.
- `zenodo_observed_q80`: unseen cation, decision `needs_data`.
- `deliberate_ood_high_pressure`: outside training operating range and unseen ions, decision `needs_data`.
- `zenodo_observed_q20`: in domain, confidence B, decision `reject`.

Interpretation:

- Random splitting remains much easier than chemically held-out screening.
- The strongest patent evidence is the reliability gate: the system identifies unsupported candidates and avoids presenting them as recommended tests.
- The evidence does not support claiming a universal accuracy improvement over all prior ML models.

## Patent Package Status

- Attorney-review package generated in `patent_package/`.
- Draft specification generated.
- Draft claims generated.
- U.S. and Saudi filing checklists generated.
- Prior-art and IDS candidate list generated.
- Attorney and IP-office email drafts generated.

Evidence status:

- This is now a real public-data, local-PC evidence package.
- The strict generalization performance is not yet strong enough to claim superior prediction accuracy.
- The patent angle should remain the reliability-gated decision workflow, not raw point-prediction accuracy.
