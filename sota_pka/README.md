# SOTA pKa Modeling With QLattice Interpretation

This project is the reproducible implementation track for the RES200 pKa work.
It keeps accuracy-first modeling separate from the older manuscript notebooks and
uses QLattice as the symbolic interpretability layer.

## Layout

- `data/raw_external/`: public downloaded datasets and cloned source exports.
- `data/processed/`: standardized schema and feature matrices.
- `runs/`: metrics, predictions, fitted model artifacts, and status files.
- `models/`: promoted final models.
- `paper_assets/`: tables, figures, formulas, and source manifests.
- `configs/`: environment, run, and public-source manifests.

## Standard Schema

All raw datasets are normalized to:

`smiles, canonical_smiles, inchi_key, pka, task, source, temperature, method, split, notes`

`task` is `acidic` or `basic`. Rows are deduplicated by task plus InChIKey when
available, falling back to canonical SMILES.

## First Local Baseline

From `C:\Users\Fahad\Downloads\RES200`:

```powershell
python -m sota_pka.cli train_baselines `
  --train "RES 200-20260312T035531Z-1-001\RES 200\train_descriptors_op2.csv" `
  --test "RES 200-20260312T035531Z-1-001\RES 200\test_descriptors_op2.csv" `
  --output-dir "sota_pka\runs\res200_op2_full_descriptors" `
  --models elasticnet random_forest xgboost
```

Then regenerate paper assets:

```powershell
python -m sota_pka.cli make_paper_assets `
  --run-dir "sota_pka\runs\res200_op2_full_descriptors" `
  --output-dir "sota_pka\paper_assets\res200_op2_full_descriptors"
```

## WSL GPU Environment

The full SOTA track should run inside WSL2 Ubuntu because this PC exposes the
RTX 5070 to WSL and Linux is the most reliable path for Chemprop, SAT4pKa,
Uni-pKa, QupKake, PyTorch Geometric, and xTB.

Use `configs/environment-wsl.yml` as the target package manifest. If conda/mamba
is unavailable, create a Python 3.11 venv and install equivalent packages with
`pip`, keeping exact versions in `runs/environment_freeze.txt`.

Current local environment: `sota_pka/.venv_wsl` has the core tabular,
cheminformatics, QLattice, Chemprop, PyTorch, and PyG packages installed. PyTorch
detects the RTX 5070 as `sm_120`, but the stable CUDA 12.4 wheel may not execute
GPU kernels for this architecture; use a CUDA 12.8/12.9 nightly wheel if graph
training fails with a kernel-compatibility error.

## Leakage-Safe QLattice Symbolic Track (macOS)

Run everything with the macOS venv `../.venv_mac` (uv, Python 3.12; system Python
3.14 and `.venv_wsl` do not work — see project memory). From the RES200 root:

```bash
# 1. (basic task only) generate descriptors to match the acidic file
.venv_mac/bin/python -m sota_pka.cli featurize_op \
  --input "RES 200-20260312T035531Z-1-001/RES 200/Opt2_basic_tr.csv" \
  --output "RES 200-20260312T035531Z-1-001/RES 200/train_descriptors_basic_op2.csv"

# 2. one leakage-safe experiment (CV-selected, single test eval)
.venv_mac/bin/python -m sota_pka.cli qlattice_experiment \
  --train ".../train_descriptors_op2.csv" --test ".../test_descriptors_op2.csv" \
  --output-dir sota_pka/runs/qlattice_acidic_op2 --mode direct --label acidic_direct

# 3. full study: acidic+basic x direct+distilled + y-randomization
.venv_mac/bin/python -m sota_pka.run_qlattice_all

# 4. assets, verification, results doc
.venv_mac/bin/python -m sota_pka.qlattice_report      # parity/residual/pareto/Williams + benchmark tables
.venv_mac/bin/python -m sota_pka.verify_qlattice      # 18 adversarial checks
.venv_mac/bin/python -m sota_pka.make_results_doc     # paper_assets/RESULTS_qlattice.md
```

Method: correlation pruning -> LightGBM-importance ranking (train only) ->
name sanitisation -> 4-fold CV selection (1-SE parsimony rule) -> refit on full
train -> **single** held-out test evaluation. `direct` fits experimental pKa;
`distilled` fits a LightGBM teacher's out-of-fold predictions. The old
`train_qlattice.run_qlattice_grid` (test-set-selected) is retained only for
reference — prefer `qlattice_search.run_qlattice_experiment`.

## SOTA Positioning

The project may only claim broader state-of-the-art performance after external
holdout comparisons beat citable open-source baselines. Until then, the paper
position is: strong open-source pKa benchmark plus QLattice symbolic
interpretability/distillation.
