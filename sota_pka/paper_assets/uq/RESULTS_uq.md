# Uncertainty-aware & Applicability-Domain pKa Prediction (XGBoost)

Leakage-safe: intervals calibrated on a held-out calibration slice of the training data; applicability domains fit on training data only; the op2 test set is scored once. CPU-only (XGBoost `hist`, no GPU path on Apple Silicon).


## Acidic task

- splits: proper-train **1834**, calibration **459**, test **765**; features **1133**; ensemble members **20**; SMILES coverage test **0.997**.

**Predictive performance (op2 test):**

| model | n | r2 | rmse | mae |
| --- | --- | --- | --- | --- |
| xgb_ensemble_mean(proper-train) | 765 | 0.768 | 1.634 | 1.033 |
| xgb_point(full-train) | 765 | 0.826 | 1.416 | 0.892 |

**Interval coverage & sharpness:**

| method | alpha | nominal_coverage | empirical_coverage | mean_width | median_width | n |
| --- | --- | --- | --- | --- | --- | --- |
| cqr | 0.200 | 0.800 | 0.808 | 4.474 | 4.583 | 765 |
| cqr | 0.100 | 0.900 | 0.897 | 6.843 | 7.549 | 765 |
| cqr | 0.050 | 0.950 | 0.932 | 9.825 | 9.815 | 765 |
| normalized_conformal | 0.200 | 0.800 | 0.816 | 3.011 | 2.888 | 765 |
| normalized_conformal | 0.100 | 0.900 | 0.914 | 4.375 | 4.196 | 765 |
| normalized_conformal | 0.050 | 0.950 | 0.950 | 5.582 | 5.354 | 765 |
| split_conformal | 0.200 | 0.800 | 0.805 | 3.443 | 3.443 | 765 |
| split_conformal | 0.100 | 0.900 | 0.906 | 5.054 | 5.054 | 765 |
| split_conformal | 0.050 | 0.950 | 0.949 | 6.887 | 6.887 | 765 |

**Applicability domain — error inside vs outside:**

| ad_method | threshold | frac_in_domain | n_in | mae_in | rmse_in | r2_in | n_out | mae_out | rmse_out | r2_out | mae_ratio_out_in |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn_distance | 34.276 | 0.953 | 729 | 1.004 | 1.599 | 0.779 | 36 | 1.603 | 2.227 | 0.430 | 1.596 |
| tanimoto | 0.320 | 0.949 | 726 | 1.003 | 1.610 | 0.776 | 39 | 1.590 | 2.031 | 0.595 | 1.586 |

**Coverage inside vs outside AD (@ 90% nominal):**

| ad_method | interval_method | alpha | nominal | coverage_in | coverage_out | width_in | width_out |
| --- | --- | --- | --- | --- | --- | --- | --- |
| knn_distance | split_conformal | 0.100 | 0.900 | 0.911 | 0.806 | 5.054 | 5.054 |
| knn_distance | normalized_conformal | 0.100 | 0.900 | 0.918 | 0.833 | 4.345 | 4.970 |
| knn_distance | cqr | 0.100 | 0.900 | 0.897 | 0.889 | 6.851 | 6.673 |
| tanimoto | split_conformal | 0.100 | 0.900 | 0.910 | 0.821 | 5.054 | 5.054 |
| tanimoto | normalized_conformal | 0.100 | 0.900 | 0.915 | 0.897 | 4.329 | 5.234 |
| tanimoto | cqr | 0.100 | 0.900 | 0.899 | 0.846 | 6.802 | 7.600 |

Figures: `acidic_calibration.png`, `acidic_width.png`, `acidic_ad_error.png`


## Basic task

- splits: proper-train **2021**, calibration **506**, test **843**; features **1707**; ensemble members **20**; SMILES coverage test **1.000**.

**Predictive performance (op2 test):**

| model | n | r2 | rmse | mae |
| --- | --- | --- | --- | --- |
| xgb_ensemble_mean(proper-train) | 843 | 0.794 | 1.476 | 0.929 |
| xgb_point(full-train) | 843 | 0.842 | 1.289 | 0.825 |

**Interval coverage & sharpness:**

| method | alpha | nominal_coverage | empirical_coverage | mean_width | median_width | n |
| --- | --- | --- | --- | --- | --- | --- |
| cqr | 0.200 | 0.800 | 0.759 | 4.038 | 3.809 | 843 |
| cqr | 0.100 | 0.900 | 0.887 | 6.535 | 6.317 | 843 |
| cqr | 0.050 | 0.950 | 0.955 | 10.154 | 10.094 | 843 |
| normalized_conformal | 0.200 | 0.800 | 0.763 | 2.432 | 2.273 | 843 |
| normalized_conformal | 0.100 | 0.900 | 0.896 | 3.685 | 3.443 | 843 |
| normalized_conformal | 0.050 | 0.950 | 0.943 | 4.866 | 4.547 | 843 |
| split_conformal | 0.200 | 0.800 | 0.775 | 2.670 | 2.670 | 843 |
| split_conformal | 0.100 | 0.900 | 0.878 | 3.942 | 3.942 | 843 |
| split_conformal | 0.050 | 0.950 | 0.944 | 5.826 | 5.826 | 843 |

**Applicability domain — error inside vs outside:**

| ad_method | threshold | frac_in_domain | n_in | mae_in | rmse_in | r2_in | n_out | mae_out | rmse_out | r2_out | mae_ratio_out_in |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| knn_distance | 42.022 | 0.940 | 792 | 0.886 | 1.411 | 0.814 | 51 | 1.584 | 2.252 | 0.359 | 1.787 |
| tanimoto | 0.333 | 0.944 | 796 | 0.869 | 1.385 | 0.814 | 47 | 1.942 | 2.563 | 0.531 | 2.235 |

**Coverage inside vs outside AD (@ 90% nominal):**

| ad_method | interval_method | alpha | nominal | coverage_in | coverage_out | width_in | width_out |
| --- | --- | --- | --- | --- | --- | --- | --- |
| knn_distance | split_conformal | 0.100 | 0.900 | 0.886 | 0.745 | 3.942 | 3.942 |
| knn_distance | normalized_conformal | 0.100 | 0.900 | 0.904 | 0.765 | 3.650 | 4.233 |
| knn_distance | cqr | 0.100 | 0.900 | 0.895 | 0.765 | 6.555 | 6.219 |
| tanimoto | split_conformal | 0.100 | 0.900 | 0.893 | 0.617 | 3.942 | 3.942 |
| tanimoto | normalized_conformal | 0.100 | 0.900 | 0.907 | 0.702 | 3.640 | 4.448 |
| tanimoto | cqr | 0.100 | 0.900 | 0.902 | 0.638 | 6.529 | 6.634 |

Figures: `basic_calibration.png`, `basic_width.png`, `basic_ad_error.png`
