# E10 — Paired Wilcoxon: symbolic vs ceiling (pooled external, neutral)

Per-molecule paired |error|. p < 0.05 ⇒ the accuracy gap is significant. `ceiling_better_frac` = fraction of molecules where the ceiling has smaller |error|.

| task | comparison | n | median \|err\| symbolic | median \|err\| ceiling | median Δ(sym−ceil) | Wilcoxon p | ceiling better |
|---|---|---:|---:|---:|---:|---:|---:|
| acidic | qlattice vs lightgbm | 158 | 1.217 | 0.876 | +0.219 | 9.75e-04 | 63% |
| acidic | qlattice vs random_forest | 158 | 1.217 | 0.873 | +0.281 | 2.31e-04 | 66% |
| basic | qlattice vs lightgbm | 265 | 1.845 | 0.734 | +1.154 | 1.06e-21 | 77% |
| basic | qlattice vs random_forest | 265 | 1.845 | 0.883 | +0.974 | 3.22e-22 | 76% |
