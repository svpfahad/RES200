# Pipeline & Operational Reference

Detailed operational reference for RES200, moved out of `CLAUDE.md` so the
always-loaded context stays lean. Read this on demand.

## Environment

**Always use `.venv_mac`** (uv, Python 3.12). The system Python 3.14 and
`sota_pka/.venv_wsl` do not work on macOS.

```bash
# Activate or prefix every command with:
.venv_mac/bin/python -m sota_pka.cli <subcommand>
```

## Running the SOTA Pipeline

All commands run from the repo root (`RES200/`).

```bash
# Featurize a raw OP2 split (SMILES → RDKit/Mordred descriptors)
.venv_mac/bin/python -m sota_pka.cli featurize_op \
  --input "RES 200-20260312T035531Z-1-001/RES 200/Opt2_acidic_tr.csv" \
  --output "RES 200-20260312T035531Z-1-001/RES 200/train_descriptors_op2.csv"

# Single leakage-safe QLattice experiment
.venv_mac/bin/python -m sota_pka.cli qlattice_experiment \
  --train "<path>/train_descriptors_op2.csv" \
  --test "<path>/test_descriptors_op2.csv" \
  --output-dir sota_pka/runs/qlattice_acidic_op2 --mode direct --label acidic_direct

# Full study (acidic+basic × direct+distilled + y-randomization)
.venv_mac/bin/python -m sota_pka.run_qlattice_all

# Post-run: reports, verification, results doc
.venv_mac/bin/python -m sota_pka.qlattice_report     # figures + benchmark tables → paper_assets/
.venv_mac/bin/python -m sota_pka.verify_qlattice      # 18 adversarial validation checks
.venv_mac/bin/python -m sota_pka.make_results_doc     # paper_assets/RESULTS_qlattice.md

# E6 — external-holdout generalization audit (SAT4pKa external sets)
.venv_mac/bin/python -m sota_pka.external_holdout                       # full: refit + score all sets, neutral+raw
.venv_mac/bin/python -m sota_pka.external_holdout --sets AvLiLuMoVe_123_acidic --reps neutral   # smoke
.venv_mac/bin/python -m sota_pka.external_summary                       # rank metrics + pooled doc (no re-fit)
```

## Running Tests

```bash
.venv_mac/bin/python -m pytest sota_pka/tests/ -v

# Single test
.venv_mac/bin/python -m pytest sota_pka/tests/test_core_pipeline.py::test_metrics_are_recomputed_from_prediction_file -v
```

## `sota_pka/` Architecture

The package is a leakage-safe ML pipeline for symbolic pKa modeling:

- **`data.py`** — standardizes raw OP2/IUPAC CSVs to a canonical 10-column schema; deduplication and split-leakage checks.
- **`featurize.py`** — RDKit + Mordred descriptor generation with salt stripping (`featurize_op_split`).
- **`train_baselines.py`** — trains ElasticNet / RF / XGBoost / LightGBM / CatBoost; `align_numeric_features` fills missing test columns with zeros.
- **`qlattice_search.py`** — the canonical leakage-safe QLattice pipeline: correlation pruning → LightGBM-importance ranking (train only) → name sanitisation → 4-fold CV with 1-SE parsimony rule → refit on full train → **single** held-out test eval. Use `run_qlattice_experiment`, not the legacy `train_qlattice.run_qlattice_grid`.
- **`evaluate.py`** — `regression_metrics`, `summarize_prediction_file`.
- **`qlattice_report.py`** — parity/residual/Pareto/Williams plots + benchmark tables.
- **`verify_qlattice.py`** — 18 adversarial checks (y-randomization, applicability domain, formula faithfulness, leakage audit).
- **`cli.py`** — `argparse` entry point (`python -m sota_pka.cli <subcommand>`).
- **`run_qlattice_all.py`** — orchestrates the full acidic+basic × direct+distilled study.

Artifacts land in:
- `sota_pka/runs/<experiment_name>/` — metrics CSV, prediction CSVs, model pickle, status JSON.
- `sota_pka/paper_assets/` — auto-generated figures, tables, `RESULTS_qlattice.md`.

## Data Files

Raw OP2 splits live in `RES 200-20260312T035531Z-1-001/RES 200/`:
- Acidic: `Opt2_acidic_tr.csv` (2,321 rows) / `Opt2_acidic_tst.csv` (774 rows)
- Basic: `Opt2_basic_tr.csv` (2,527 rows) / `Opt2_basic_tst.csv` (843 rows)
- Descriptor matrices: `train_descriptors_op2.csv` (2,293 × 1,135+1), `test_descriptors_op2.csv`

Target column: `pKa`. SMILES column: `OriginalSmiles`.

## Key Results (do not overwrite without re-running)

| Model | Task | Test R² | RMSE |
|---|---|---|---|
| QLattice direct | Acidic | 0.448 | 2.52 |
| QLattice direct | Basic | 0.470 | 2.37 |
| LightGBM (ceiling) | Acidic/Basic | ~0.836 | — |
| XGBoost full desc. | Combined | 0.830 | 1.40 |

Distillation (QLattice fitting LightGBM OOF) hurts performance — reportable negative result.

### E6 — external-holdout generalization (neutralized SAT4pKa sets, pooled per task)

| Model | Task | Ext R² | Ext RMSE | Spearman ρ | (in-dist OP2-test R²) |
|---|---|---|---|---|---|
| QLattice (symbolic) | Acidic (n=158) | 0.336 | 2.15 | 0.575 | 0.448 |
| LightGBM (ceiling) | Acidic (n=158) | 0.490 | 1.88 | 0.707 | 0.832 |
| QLattice (symbolic) | Basic (n=265) | −0.377 | 2.64 | 0.417 | 0.452 |
| LightGBM (ceiling) | Basic (n=265) | 0.673 | 1.29 | 0.817 | 0.837 |
| y-random (control) | both | <0 | — | 0.05–0.17 | ≈0 |

H3 partially rejected: symbolic formulas transfer **weak-to-moderate above-chance rank signal** (ρ 0.27–0.58 across 4/5 sets) but are **miscalibrated out-of-distribution** (low/negative R², calibration slopes 0.26–0.76) and are out-ranked by the GBM externally. **Representation is first-order:** scoring *raw ionized* SMILES collapses every model (R² −2…−62) → external SMILES must be neutralized to the training convention. Full tables: `sota_pka/runs/external_holdout/RESULTS_external_holdout.md`. Do not overwrite without re-running `external_holdout` + `external_summary`.

## Research Positioning

Do not claim QLattice outperforms XGBoost. The paper angle is **interpretable symbolic pKa modeling**: compact formulas, chemically inspectable descriptors (partial-charge/electronic terms dominate), and honest accuracy-interpretability tradeoff.

Remaining work for submission: novelty gate (V1); repeated/nested CV for uncertainty bands; an out-of-distribution calibration analysis for the symbolic model (its main external weakness — see E6). External holdout (E6) is **done** via SAT4pKa sets; SAMPL6/7-minimal was dropped (no in-repo labels / tiny N).
