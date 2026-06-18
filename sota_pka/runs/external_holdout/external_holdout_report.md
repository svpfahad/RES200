# E6 — External-holdout generalization (H3)


## Task: acidic

- Recorded OP2-test R² (QLattice): **0.448**
- Reproduced OP2-test R² (QLattice): **0.448** (RMSE 2.52)
- OP2-test R² (lightgbm): 0.832
- OP2-test R² (elasticnet): -0.149
- OP2-test R² (random_forest): 0.776
- OP2-test R² (y-random LGBM): -0.136

| set | rep | model | n | R² | RMSE | MAE | ΔR² vs OP2-test |
|---|---|---|---:|---:|---:|---:|---:|
| novartis_acidic | neutral | qlattice | 112 | 0.222 | 2.25 | 1.75 | +0.226 |
| novartis_acidic | neutral | lightgbm | 112 | 0.533 | 1.74 | 1.28 | +0.300 |
| novartis_acidic | neutral | elasticnet | 112 | -0.318 | 2.93 | 2.21 | +0.169 |
| novartis_acidic | neutral | random_forest | 112 | 0.415 | 1.95 | 1.33 | +0.360 |
| novartis_acidic | neutral | y_random_lightgbm | 112 | -0.174 | 2.76 | 2.38 | +0.038 |
| novartis_acidic | raw | qlattice | 112 | -0.897 | 3.51 | 2.93 | +1.345 |
| novartis_acidic | raw | lightgbm | 112 | -2.031 | 4.44 | 3.49 | +2.863 |
| novartis_acidic | raw | elasticnet | 112 | -3.155 | 5.20 | 4.20 | +3.006 |
| novartis_acidic | raw | random_forest | 112 | -1.644 | 4.15 | 3.24 | +2.420 |
| novartis_acidic | raw | y_random_lightgbm | 112 | -0.112 | 2.69 | 2.30 | -0.024 |
| AvLiLuMoVe_123_acidic | neutral | qlattice | 26 | 0.131 | 0.84 | 0.64 | +0.317 |
| AvLiLuMoVe_123_acidic | neutral | lightgbm | 26 | -3.021 | 1.80 | 1.11 | +3.853 |
| AvLiLuMoVe_123_acidic | neutral | elasticnet | 26 | -15.082 | 3.59 | 2.39 | +14.933 |
| AvLiLuMoVe_123_acidic | neutral | random_forest | 26 | -3.222 | 1.84 | 1.07 | +3.998 |
| AvLiLuMoVe_123_acidic | neutral | y_random_lightgbm | 26 | -11.074 | 3.11 | 2.70 | +10.938 |
| AvLiLuMoVe_123_acidic | raw | qlattice | 26 | -20.855 | 4.19 | 3.59 | +21.303 |
| AvLiLuMoVe_123_acidic | raw | lightgbm | 26 | -46.706 | 6.19 | 5.95 | +47.538 |
| AvLiLuMoVe_123_acidic | raw | elasticnet | 26 | -62.157 | 7.12 | 6.74 | +62.008 |
| AvLiLuMoVe_123_acidic | raw | random_forest | 26 | -36.497 | 5.49 | 5.22 | +37.273 |
| AvLiLuMoVe_123_acidic | raw | y_random_lightgbm | 26 | -7.637 | 2.63 | 2.25 | +7.500 |
| SAMPL7_acidic | neutral | qlattice | 20 | -0.298 | 2.68 | 2.38 | +0.746 |
| SAMPL7_acidic | neutral | lightgbm | 20 | -0.228 | 2.60 | 2.08 | +1.060 |
| SAMPL7_acidic | neutral | elasticnet | 20 | -3.744 | 5.12 | 3.54 | +3.594 |
| SAMPL7_acidic | neutral | random_forest | 20 | 0.076 | 2.26 | 2.07 | +0.700 |
| SAMPL7_acidic | neutral | y_random_lightgbm | 20 | -0.803 | 3.15 | 2.66 | +0.666 |
| SAMPL7_acidic | raw | qlattice | 20 | -0.298 | 2.68 | 2.38 | +0.746 |
| SAMPL7_acidic | raw | lightgbm | 20 | -0.228 | 2.60 | 2.08 | +1.060 |
| SAMPL7_acidic | raw | elasticnet | 20 | -3.744 | 5.12 | 3.54 | +3.594 |
| SAMPL7_acidic | raw | random_forest | 20 | 0.076 | 2.26 | 2.07 | +0.700 |
| SAMPL7_acidic | raw | y_random_lightgbm | 20 | -0.803 | 3.15 | 2.66 | +0.666 |


## Task: basic

- Recorded OP2-test R² (QLattice): **0.470**
- Reproduced OP2-test R² (QLattice): **0.452** (RMSE 2.40)
- OP2-test R² (lightgbm): 0.837
- OP2-test R² (elasticnet): -51.892
- OP2-test R² (random_forest): 0.801
- OP2-test R² (y-random LGBM): -0.156

| set | rep | model | n | R² | RMSE | MAE | ΔR² vs OP2-test |
|---|---|---|---:|---:|---:|---:|---:|
| novartis_basic | neutral | qlattice | 168 | -0.219 | 2.34 | 1.88 | +0.671 |
| novartis_basic | neutral | lightgbm | 168 | 0.578 | 1.38 | 1.09 | +0.258 |
| novartis_basic | neutral | elasticnet | 168 | -808.682 | 60.34 | 60.20 | +756.789 |
| novartis_basic | neutral | random_forest | 168 | 0.608 | 1.33 | 1.05 | +0.193 |
| novartis_basic | neutral | y_random_lightgbm | 168 | -0.170 | 2.29 | 1.90 | +0.015 |
| novartis_basic | raw | qlattice | 168 | -0.345 | 2.46 | 1.95 | +0.797 |
| novartis_basic | raw | lightgbm | 168 | -0.233 | 2.35 | 1.84 | +1.069 |
| novartis_basic | raw | elasticnet | 168 | -165.212 | 27.34 | 26.89 | +113.320 |
| novartis_basic | raw | random_forest | 168 | -0.092 | 2.22 | 1.76 | +0.893 |
| novartis_basic | raw | y_random_lightgbm | 168 | 0.007 | 2.11 | 1.73 | -0.163 |
| AvLiLuMoVe_123_basic | neutral | qlattice | 97 | -3.647 | 3.08 | 2.61 | +4.099 |
| AvLiLuMoVe_123_basic | neutral | lightgbm | 97 | 0.399 | 1.11 | 0.76 | +0.438 |
| AvLiLuMoVe_123_basic | neutral | elasticnet | 97 | -3636.759 | 86.28 | 86.22 | +3584.866 |
| AvLiLuMoVe_123_basic | neutral | random_forest | 97 | 0.179 | 1.30 | 1.02 | +0.622 |
| AvLiLuMoVe_123_basic | neutral | y_random_lightgbm | 97 | -2.699 | 2.75 | 2.38 | +2.543 |
| AvLiLuMoVe_123_basic | raw | qlattice | 97 | -9.230 | 4.58 | 3.63 | +9.682 |
| AvLiLuMoVe_123_basic | raw | lightgbm | 97 | -5.380 | 3.61 | 3.37 | +6.216 |
| AvLiLuMoVe_123_basic | raw | elasticnet | 97 | -1491.070 | 55.26 | 55.10 | +1439.178 |
| AvLiLuMoVe_123_basic | raw | random_forest | 97 | -6.585 | 3.94 | 3.68 | +7.385 |
| AvLiLuMoVe_123_basic | raw | y_random_lightgbm | 97 | -1.737 | 2.37 | 2.06 | +1.581 |
