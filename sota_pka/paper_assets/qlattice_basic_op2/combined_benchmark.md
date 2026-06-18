### Basic pKa — model comparison (OP2 held-out test)

| Model | Type | Test R² | Test RMSE | Test MAE | # feats | Complexity |
|---|---|---|---|---|---|---|
| lightgbm | GBM baseline | 0.837 | 1.313 | 0.847 | 1707 | — |
| xgboost | GBM baseline | 0.834 | 1.322 | 0.853 | 1707 | — |
| catboost | GBM baseline | 0.813 | 1.406 | 0.935 | 1707 | — |
| random_forest | GBM baseline | 0.801 | 1.450 | 0.928 | 1707 | — |
| QLattice (direct) | symbolic | 0.470 | 2.365 | 1.759 | 12 | 44 |
| QLattice (distilled) | symbolic | 0.331 | 2.657 | 2.083 | 8 | 30 |
| elasticnet | GBM baseline | -51.892 | 23.621 | 22.128 | 1707 | — |
