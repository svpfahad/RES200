# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

RES200 is a research codebase with two tracks:

1. **Manuscript track** — a finalized JURI paper. Key files: the `*_FINAL.docx`
   documents, `verify_final.py`, `generate_figures.py`.
2. **Modeling track** — a leakage-safe ML package in `sota_pka/` (the active track).

It is driven by an autonomous research agent: run `/research` to start or resume.
Reusable agent logic (the `/research` command and the `experimentalist` +
`novelty-assessor` subagents) lives in `~/.claude/` and wires into the
scientific-writing skills under `claude-scientific-writer-main/`.

## Where things live

- **Operational reference** (environment, pipeline commands, tests, architecture,
  data files, recorded results) → `docs/PIPELINE.md`.
- **Project facts** (domain, data paths, target venues) → `.claude/RESEARCHER.md`.
- **Live state** (current phase, next action, experiment log, paper status) →
  `RESEARCH_LOG.md`, the single source of truth across sessions.

## Environment

Always use `.venv_mac` (uv, Python 3.12); the system Python and `.venv_wsl` do not
work on macOS. Prefix commands with `.venv_mac/bin/python`. See `docs/PIPELINE.md`
for the full command list.

## Ground rules

- Don't overwrite the recorded results tables (in `docs/PIPELINE.md`) without
  re-running the pipeline.
- Preserve the package's leakage-safe discipline: train-only feature selection,
  a single held-out test evaluation, and the adversarial validation checks.
- Positioning: favor interpretable models with an honest
  accuracy/interpretability tradeoff — don't overclaim against the
  gradient-boosted baseline.
