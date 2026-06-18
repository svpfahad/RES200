### Acidic pKa — model comparison (OP2 held-out test)

| Model | Type | Test R² | Test RMSE | Test MAE | # feats | Complexity |
|---|---|---|---|---|---|---|
| lightgbm | GBM baseline | 0.836 | 1.375 | 0.868 | 1133 | — |
| xgboost | GBM baseline | 0.829 | 1.402 | 0.874 | 1133 | — |
| catboost | GBM baseline | 0.810 | 1.479 | 0.972 | 1133 | — |
| random_forest | GBM baseline | 0.776 | 1.604 | 1.001 | 1133 | — |
| QLattice (direct) | symbolic | 0.448 | 2.519 | 1.826 | 8 | 49 |
| QLattice (distilled) | symbolic | 0.350 | 2.734 | 2.037 | 8 | 39 |
| elasticnet | GBM baseline | -0.149 | 3.635 | 1.510 | 1133 | — |
