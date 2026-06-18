"""Uncertainty-aware & applicability-domain pKa prediction (XGBoost track).

A self-contained extension of the RES200 work: it reuses the leakage-safe data
plumbing from the parent ``sota_pka`` package (the op2 RDKit+Mordred descriptor
matrices and ``train_baselines``/``evaluate`` helpers) but is its own modeling
line — XGBoost baselines wrapped with conformal prediction intervals and two
applicability-domain (AD) definitions.

Modules
-------
- ``data_splits``  : load op2 matrices, recover SMILES, carve a calibration set.
- ``baseline``     : leakage-safe XGBoost + a seed/bootstrap ensemble (std → AD).
- ``conformal``    : split / normalized / quantile prediction intervals.
- ``applicability``: kNN-distance and Tanimoto-similarity AD.
- ``evaluate_uq``  : coverage / sharpness / calibration + inside-vs-outside-AD.
- ``cli``          : ``python -m sota_pka.uq.cli run --task acidic`` driver.

Everything is CPU-only (no GPU path exists for XGBoost on Apple Silicon) and
tuned for a multicore laptop: ``tree_method="hist"`` + ``n_jobs=-1`` and parallel
ensemble training.
"""
