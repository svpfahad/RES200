# E7 — Uncertainty bands & OOD calibration decomposition

*Computed from recorded predictions only (no refit). Bootstrap = 10000 resamples, percentile 95% CI.*

## In-distribution symbolic test R² (with bootstrap 95% CI + CV band)

| task | n | test R² | R² 95% CI | RMSE | RMSE 95% CI | CV R² (mean ± 1.96·SE) |
|---|---:|---:|---|---:|---|---|
| acidic | 765 | 0.448 | [0.395, 0.500] | 2.519 | [2.343, 2.696] | 0.453 ± 0.037 |
| basic | 843 | 0.470 | [0.404, 0.530] | 2.365 | [2.182, 2.558] | 0.419 ± 0.055 |

## External OOD calibration decomposition (symbolic vs ceiling)

`r2_raw` = OP2-model as-is · `r2_recal_cv` = after out-of-fold affine correction · `r2_affine_ceiling` = Pearson r² (best any affine map can do) · Spearman is affine-invariant.

| task | set | model | n | R²_raw | R²_recal(CV) | R²_affine_ceiling | Spearman | a | b |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| acidic | POOLED | qlattice | 158 | 0.336 | 0.337 | 0.356 | 0.575 | 0.913 | 0.854 |
| acidic | POOLED | lightgbm | 158 | 0.490 | 0.490 | 0.499 | 0.707 | 0.882 | 0.695 |
| acidic | POOLED | random_forest | 158 | 0.438 | 0.472 | 0.490 | 0.770 | 0.784 | 1.583 |
| acidic | AvLiLuMoVe_123_acidic | qlattice | 26 | 0.131 | -0.143 | 0.251 | 0.577 | 0.647 | 1.267 |
| acidic | AvLiLuMoVe_123_acidic | lightgbm | 26 | -3.021 | -0.123 | 0.029 | 0.404 | 0.096 | 3.495 |
| acidic | AvLiLuMoVe_123_acidic | random_forest | 26 | -3.222 | -0.197 | 0.022 | 0.405 | 0.080 | 3.606 |
| acidic | SAMPL7_acidic | qlattice | 20 | -0.298 | -0.207 | 0.019 | 0.117 | 0.496 | 4.894 |
| acidic | SAMPL7_acidic | lightgbm | 20 | -0.228 | -0.084 | 0.077 | 0.242 | 0.547 | 4.467 |
| acidic | SAMPL7_acidic | random_forest | 20 | 0.076 | 0.134 | 0.259 | 0.599 | 1.799 | -5.212 |
| acidic | novartis_acidic | qlattice | 112 | 0.222 | 0.219 | 0.262 | 0.429 | 0.758 | 1.753 |
| acidic | novartis_acidic | lightgbm | 112 | 0.533 | 0.516 | 0.544 | 0.706 | 0.886 | 0.604 |
| acidic | novartis_acidic | random_forest | 112 | 0.415 | 0.425 | 0.487 | 0.731 | 0.752 | 1.798 |
| basic | POOLED | qlattice | 265 | -0.377 | 0.165 | 0.177 | 0.417 | 0.598 | 3.779 |
| basic | POOLED | lightgbm | 265 | 0.673 | 0.672 | 0.675 | 0.817 | 0.953 | 0.402 |
| basic | POOLED | random_forest | 265 | 0.657 | 0.673 | 0.676 | 0.795 | 0.994 | 0.345 |
| basic | AvLiLuMoVe_123_basic | qlattice | 97 | -3.647 | 0.064 | 0.106 | 0.270 | 0.259 | 7.073 |
| basic | AvLiLuMoVe_123_basic | lightgbm | 97 | 0.399 | 0.386 | 0.402 | 0.675 | 0.945 | 0.544 |
| basic | AvLiLuMoVe_123_basic | random_forest | 97 | 0.179 | 0.274 | 0.296 | 0.569 | 0.882 | 1.452 |
| basic | novartis_basic | qlattice | 168 | -0.219 | 0.073 | 0.100 | 0.301 | 0.526 | 3.479 |
| basic | novartis_basic | lightgbm | 168 | 0.578 | 0.575 | 0.584 | 0.723 | 0.920 | 0.558 |
| basic | novartis_basic | random_forest | 168 | 0.608 | 0.607 | 0.620 | 0.709 | 0.940 | 0.565 |
