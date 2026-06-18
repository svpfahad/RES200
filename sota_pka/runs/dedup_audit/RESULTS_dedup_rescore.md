# E9b — External metrics with train-overlapping molecules removed

Feyn-free recompute on stored predictions (row order verified 1:1). `full` = E6 as-reported · `exact` = drop exact-structure train overlaps · `conn` = also drop same-2D-skeleton analogs.


## acidic — QLattice (symbolic) and ceiling

| set | model | level | n | dropped | R² | RMSE | Spearman |
|---|---|---|---:|---:|---:|---:|---:|
| novartis_acidic | qlattice | full | 112 | 0 | 0.222 | 2.250 | 0.429 |
| novartis_acidic | lightgbm | full | 112 | 0 | 0.533 | 1.744 | 0.706 |
| novartis_acidic | random_forest | full | 112 | 0 | 0.415 | 1.950 | 0.731 |
| AvLiLuMoVe_123_acidic | qlattice | full | 26 | 0 | 0.131 | 0.836 | 0.577 |
| AvLiLuMoVe_123_acidic | qlattice | conn | 21 | 5 | 0.015 | 0.915 | 0.481 |
| AvLiLuMoVe_123_acidic | lightgbm | full | 26 | 0 | -3.021 | 1.797 | 0.404 |
| AvLiLuMoVe_123_acidic | lightgbm | conn | 21 | 5 | -3.695 | 1.998 | 0.420 |
| AvLiLuMoVe_123_acidic | random_forest | full | 26 | 0 | -3.222 | 1.842 | 0.405 |
| AvLiLuMoVe_123_acidic | random_forest | conn | 21 | 5 | -3.937 | 2.049 | 0.381 |
| SAMPL7_acidic | qlattice | full | 20 | 0 | -0.298 | 2.676 | 0.117 |
| SAMPL7_acidic | lightgbm | full | 20 | 0 | -0.228 | 2.603 | 0.242 |
| SAMPL7_acidic | random_forest | full | 20 | 0 | 0.076 | 2.258 | 0.599 |
| POOLED | qlattice | full | 158 |  | 0.336 | 2.147 | 0.575 |
| POOLED | qlattice | exact | 158 |  | 0.336 | 2.147 | 0.575 |
| POOLED | qlattice | conn | 153 |  | 0.326 | 2.181 | 0.559 |
| POOLED | lightgbm | full | 158 |  | 0.490 | 1.883 | 0.707 |
| POOLED | lightgbm | exact | 158 |  | 0.490 | 1.883 | 0.707 |
| POOLED | lightgbm | conn | 153 |  | 0.481 | 1.913 | 0.699 |
| POOLED | random_forest | full | 158 |  | 0.438 | 1.975 | 0.770 |
| POOLED | random_forest | exact | 158 |  | 0.438 | 1.975 | 0.770 |
| POOLED | random_forest | conn | 153 |  | 0.429 | 2.007 | 0.763 |

## basic — QLattice (symbolic) and ceiling

| set | model | level | n | dropped | R² | RMSE | Spearman |
|---|---|---|---:|---:|---:|---:|---:|
| novartis_basic | qlattice | full | 168 | 0 | -0.219 | 2.341 | 0.301 |
| novartis_basic | lightgbm | full | 168 | 0 | 0.578 | 1.377 | 0.723 |
| novartis_basic | random_forest | full | 168 | 0 | 0.608 | 1.328 | 0.709 |
| AvLiLuMoVe_123_basic | qlattice | full | 97 | 0 | -3.647 | 3.084 | 0.270 |
| AvLiLuMoVe_123_basic | qlattice | exact | 95 | 2 | -3.758 | 3.093 | 0.248 |
| AvLiLuMoVe_123_basic | qlattice | conn | 86 | 11 | -3.769 | 3.163 | 0.187 |
| AvLiLuMoVe_123_basic | lightgbm | full | 97 | 0 | 0.399 | 1.109 | 0.675 |
| AvLiLuMoVe_123_basic | lightgbm | exact | 95 | 2 | 0.379 | 1.118 | 0.671 |
| AvLiLuMoVe_123_basic | lightgbm | conn | 86 | 11 | 0.366 | 1.153 | 0.630 |
| AvLiLuMoVe_123_basic | random_forest | full | 97 | 0 | 0.179 | 1.296 | 0.569 |
| AvLiLuMoVe_123_basic | random_forest | exact | 95 | 2 | 0.148 | 1.309 | 0.556 |
| AvLiLuMoVe_123_basic | random_forest | conn | 86 | 11 | 0.134 | 1.348 | 0.505 |
| POOLED | qlattice | full | 265 |  | -0.377 | 2.637 | 0.417 |
| POOLED | qlattice | exact | 263 |  | -0.389 | 2.638 | 0.410 |
| POOLED | qlattice | conn | 254 |  | -0.410 | 2.648 | 0.381 |
| POOLED | lightgbm | full | 265 |  | 0.673 | 1.286 | 0.817 |
| POOLED | lightgbm | exact | 263 |  | 0.668 | 1.290 | 0.815 |
| POOLED | lightgbm | conn | 254 |  | 0.657 | 1.306 | 0.807 |
| POOLED | random_forest | full | 265 |  | 0.657 | 1.316 | 0.795 |
| POOLED | random_forest | exact | 263 |  | 0.651 | 1.321 | 0.792 |
| POOLED | random_forest | conn | 254 |  | 0.642 | 1.335 | 0.783 |
