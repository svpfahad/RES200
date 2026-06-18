# CO2 Capture ML Patent Track

Purpose: build a local-PC, public-data ML research package aimed at a defensible patent direction, not just a paper.

Working invention direction:

> A confidence-gated machine-learning decision system for screening CO2 capture solvent candidates under specified operating conditions, using leakage-safe validation, calibrated prediction intervals, applicability-domain checks, and decision-grade candidate ranking.

This is technical research support, not legal advice. Do not publicly disclose claim details, figures, or results before a patent attorney or university IP office reviews them.

## What Is Included

- `docs/RESEARCH_BRIEF.md`: broad research rationale and source-backed direction.
- `docs/PRIOR_ART_MATRIX.md`: early prior-art map and risk assessment.
- `docs/INVENTION_DISCLOSURE_DRAFT.md`: draft invention disclosure.
- `docs/CLAIM_STRATEGY.md`: possible claim lanes and what to avoid.
- `docs/EXPERIMENT_PLAN.md`: complete local-PC experiment plan.
- `docs/CURRENT_RESULTS.md`: current demo and first public-data run results.
- `patent_package/`: attorney-review patent package with specification, draft claims, drawings, filing checklists, prior-art/IDS candidates, and email drafts.
- `patent_package_docx/`: Word exports of the patent package, including a combined package.
- `deliverables/co2_capture_ml_patent_attorney_package.zip`: zip file ready to attach to an attorney email.
- `configs/public_sources.csv`: public data and prior-art source manifest.
- `src/co2patent/`: reusable Python pipeline code.
- `scripts/run_pipeline.py`: trains/evaluates local ML models with conformal intervals.
- `scripts/screen_candidates.py`: ranks candidate solvents with confidence and domain flags.
- `scripts/run_evidence_suite.py`: runs the repeatable patent-evidence benchmark suite.
- `scripts/download_figshare.py`: downloads public Figshare article files when internet access is available.
- `scripts/download_zenodo_record.py`: downloads selected Zenodo record files.
- `data/sample/`: tiny synthetic sample data for smoke tests only.

## Quick Smoke Test

Run from this folder:

```powershell
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --data data/sample/demo_co2_solubility.csv --target co2_solubility_mol_frac --group solvent_id --out runs/demo
python scripts/screen_candidates.py --model runs/demo/model.joblib --candidates data/sample/candidate_screen.csv --out runs/demo/candidate_ranking.csv
```

The sample data is synthetic and exists only to prove the software path works. Patent evidence must be generated from public experimental data listed in `configs/public_sources.csv`.

## Public Data Workflow

### Figshare/ACS Supplement

1. Download public data:

```powershell
python scripts/download_figshare.py --article-id 29228990 --out data/raw/figshare_29228990
```

2. Inspect the downloaded spreadsheet/CSV columns.
3. Prepare the exact experimental-target sheet from the ACS/Figshare workbook:

```powershell
python scripts/prepare_figshare_29228990.py
```

The workbook's large `S2` and `S3` sheets expose model predictions and ARD columns, not a clean experimental target column. Do not use those sheets as independent training targets unless the original target column is obtained or a reconstruction method is verified. The preparation script uses `S4-Experimental data_Unique ILs`, which contains the explicit experimental target.

4. Run the same pipeline, explicitly setting target and group columns:

```powershell
python scripts/run_pipeline.py --data data/processed/figshare_29228990_unique_ils.csv --target ln_co2_solubility_exp --group solvent_id --out runs/figshare_s4_strict
```

If a dataset has cation and anion columns but no single solvent ID, create one with `cation + "|" + anion` before final experiments. The strict test split must hold out solvent identities, not random rows from the same solvent.

### Zenodo Ionic Liquid Properties CO2 Dataset

This is the better first evidence dataset because it has an explicit target column and a permissive CC BY 4.0 license:

```powershell
python scripts/download_zenodo_record.py --record-id 3251643 --out data/raw/zenodo_3251643 --files CO2CAPACITY.txt CA.smi
python scripts/prepare_zenodo_3251643.py
python scripts/run_pipeline.py --data data/processed/zenodo_3251643_co2capacity.csv --target ln_co2_solubility --group solvent_id --out runs/zenodo_co2_strict
```

For the full repeatable evidence suite:

```powershell
python scripts/run_evidence_suite.py
```

Outputs are saved in `runs/evidence_suite/`, including `summary.md`, `summary.csv`, model bundles, test predictions, and `candidate_ranking.csv`.

## Patent-Relevant Evidence Targets

- Random split benchmark: useful only as a sanity check.
- Solvent-held-out benchmark: required.
- Family-held-out benchmark: stronger evidence.
- Conformal interval coverage near the requested confidence level.
- Higher uncertainty for held-out or out-of-domain solvents.
- Decision engine refusing or downgrading low-confidence candidates.
- Claim chart comparing at least 10 close papers/patents.

## Attorney Package

Start with:

```text
patent_package/00_PACKAGE_README.md
patent_package/01_ATTORNEY_REVIEW_BRIEF.md
patent_package/02_SPECIFICATION_DRAFT.md
patent_package/03_CLAIMS_DRAFT.md
```

Word exports can be regenerated with:

```powershell
python scripts/export_patent_package_docx.py
```

Do not send enabling details publicly until a patent attorney confirms the filing plan.
