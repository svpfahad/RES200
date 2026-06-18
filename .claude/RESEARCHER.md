# RESEARCHER.md — RES200 Project Fact Sheet

This is the project-specific fact sheet read by the reusable autonomous research agent (`/research` command and the `experimentalist` / `novelty-assessor` subagents). It tells the agent *what this project is* and *exactly how to run it*. The agent's reusable methodology lives in `~/.claude/`; the live state lives in `./RESEARCH_LOG.md`.

## Research Domain

**Computational cheminformatics — interpretable machine learning for molecular property prediction, specifically aqueous pKa.** The scientific question: can symbolic regression (QLattice) produce compact, chemically inspectable pKa equations that approach black-box accuracy, and what is the honest accuracy-interpretability tradeoff against strong cheminformatics baselines?

Sub-area keywords for literature search: `pKa prediction`, `symbolic regression chemistry`, `QSPR`, `interpretable molecular property prediction`, `QLattice`, `SAMPL pKa challenge`, `applicability domain`.

## Positioning (do not overclaim)

- **Do NOT** claim QLattice out-accuracies XGBoost/LightGBM. It does not — the GBMs are the accuracy ceiling.
- **The defensible contribution** is: a curated acidic/basic pKa benchmark, leakage-safe symbolic models, chemical interpretation of the selected descriptors (partial-charge/electronic terms dominate — physically correct for protonation), and an honest accuracy-vs-complexity tradeoff. The failed distillation is a legitimate reportable negative result.

## Environment

Always use the macOS venv. System Python 3.14 and `sota_pka/.venv_wsl` do **not** work on macOS.

```bash
.venv_mac/bin/python -m sota_pka.<module> ...
```

All commands run from the repo root: `/Users/fahad/Downloads/RES200`.

## Pipeline Commands (use verbatim)

```bash
# Featurize a raw SMILES+pKa split → RDKit/Mordred descriptors
.venv_mac/bin/python -m sota_pka.cli featurize_op \
  --input "<raw_split>.csv" --output "<descriptors>.csv"

# One leakage-safe QLattice experiment (CV-selected, single test eval)
.venv_mac/bin/python -m sota_pka.cli qlattice_experiment \
  --train "<train_desc>.csv" --test "<test_desc>.csv" \
  --output-dir sota_pka/runs/<name> --mode direct --label <label>

# Baselines (the ≥3 baselines the experimentalist must include)
.venv_mac/bin/python -m sota_pka.cli train_baselines \
  --train "<train_desc>.csv" --test "<test_desc>.csv" \
  --output-dir sota_pka/runs/<name> --models elasticnet random_forest lightgbm

# y-randomization control
.venv_mac/bin/python -m sota_pka.cli y_randomization \
  --train "<train_desc>.csv" --test "<test_desc>.csv" --output-dir sota_pka/runs/<name>

# Full study (acidic+basic × direct+distilled + y-rand)
.venv_mac/bin/python -m sota_pka.run_qlattice_all

# Reports / validation / results doc
.venv_mac/bin/python -m sota_pka.qlattice_report     # figures + benchmark tables
.venv_mac/bin/python -m sota_pka.verify_qlattice     # 18 adversarial validation checks
.venv_mac/bin/python -m sota_pka.make_results_doc    # paper_assets/RESULTS_qlattice.md
```

**Metric recomputation** (experimentalist must use this, not printed summaries): `sota_pka/evaluate.py` → `regression_metrics`, `summarize_prediction_file`.

**Leakage-safe rule:** `qlattice_search.py` does correlation pruning → LightGBM-importance ranking (train only) → 4-fold CV with a 1-SE parsimony rule → refit → **single** test eval. Never bypass this; never select on the test set.

## Data

| What | Path |
|---|---|
| Raw OP2 splits | `RES 200-20260312T035531Z-1-001/RES 200/Opt2_{acidic,basic}_{tr,tst}.csv` |
| Descriptor matrices | `RES 200-20260312T035531Z-1-001/RES 200/{train,test}_descriptors_op2.csv` |
| External holdouts (raw, **not yet featurized**) | `sota_pka/data/raw_external/{SAMPL6_pKa_minimal,SAMPL7_pKa_minimal,SAT4pKa,QupKake,Uni-pKa}/` |
| Run artifacts | `sota_pka/runs/<name>/` (metrics.csv, predictions_*.csv, summary_*.json, formula_*.txt) |
| Paper figures/tables | `sota_pka/paper_assets/` (parity/residual/Williams plots, `RESULTS_qlattice.md`) |

Target column: `pKa`. SMILES column: `OriginalSmiles`. Task labels: `acidic` / `basic`.

## Current Results (snapshot — see RESEARCH_LOG.md for live state)

| Model | Task | Test R² | RMSE |
|---|---|---|---|
| QLattice direct (8 descr, 49 ops) | Acidic | 0.448 | 2.52 |
| QLattice direct (12 descr, 44 ops) | Basic | 0.470 | 2.37 |
| LightGBM (ceiling) | Acidic/Basic | ~0.836 / 0.837 | — |
| Distillation (QLattice on GBM OOF) | both | 0.33–0.35 | — (worse → negative result) |

Validation: 18/18 checks pass (y-rand R²≈0, ~96% in applicability domain, formulas reproduce predictions at R²=1.000, zero train/test leakage).

## Paper Targets

- **Working title:** *Interpretable Symbolic Regression for Molecular pKa Prediction Using QLattice-Guided Descriptor Discovery*
- **Target venues** (priority): *Journal of Cheminformatics*, *JCIM*, *Molecules* (open-access).
- **Writing engine:** the `claude-scientific-writer-main/` skills are available this session — `scientific-writing`, `citation-management`, `peer-review`, `venue-templates`, `literature-review`, `hypothesis-generation`, `scholar-evaluation`, `scientific-critical-thinking`.

## Tooling Notes / Fallbacks

- `research-lookup` and some skills need **Perplexity/Parallel API keys**. If unset, fall back to: `WebSearch`, Hugging Face `paper_search` MCP, and the free arXiv/PubMed/Semantic-Scholar backends inside `literature-review`.
- This repo is **not a git repo** — no commits/PRs expected.

## Open Work (drives the next research moves)

1. **External holdout — DONE (E6, 2026-06-13).** Scored QLattice + GBM on neutralized SAT4pKa sets (Novartis/AvLiLuMoVe/SAMPL7). Symbolic transfers partial above-chance rank signal (Spearman 0.27–0.58) but is miscalibrated OOD; GBM generalizes better. Driver `sota_pka/external_holdout.py`, summary `sota_pka/external_summary.py`, results `sota_pka/runs/external_holdout/RESULTS_external_holdout.md`. SAMPL6/7-minimal dropped (no in-repo labels / tiny N).
2. **Novelty gate (V1) — NEXT.** Run `novelty-assessor` on the E6-augmented framing.
3. **Repeated / nested CV** — uncertainty band on the symbolic R².
4. **OOD calibration** — the symbolic model's main external weakness; try a simple recalibration (linear/Platt-style) on a held-out external slice and report Spearman alongside R².
5. **Accuracy-optimized vs parsimony variant** — draw the full tradeoff curve.
6. **Optional** — site-localized / fragment-state descriptors to raise the symbolic ceiling above ~0.47.
