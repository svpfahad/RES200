# Novelty Gate — Verdict V1

**Date:** 2026-06-13
**Assessor:** novelty-assessor subagent (claude-sonnet-4-6)
**Work judged:** E6-augmented framing — "A leakage-safe symbolic pKa benchmark with an honest external-generalization audit"

---

## VERDICT: borderline (lean publishable)

**WHY:**
No public work applies QLattice/Feyn symbolic regression to pKa prediction, so novelty claim #1 is clean. The contribution is not an accuracy advance — it is deliberately a negative/honest-audit result: leakage-safe compact symbolic formulas, a quantified accuracy-interpretability tradeoff, a failed distillation, and an external audit revealing OOD miscalibration and protonation-state sensitivity. The borderline flag is because the OP2 test R² (0.45–0.47) sits well below the published interpretable deep-learning bar (BCL-XpKa MAE ~0.8 on Novartis which corresponds to roughly R² ~0.7–0.8), and there is no repeated/nested CV to put an uncertainty band on the symbolic number. The gap between what the paper *claims* (a methodology and audit paper) and what a referee might *expect* (competitive accuracy) needs to be pre-empted with language and one additional quantification (see RECOMMENDED NEXT MOVE below).

---

## CLOSEST PRIOR ART

1. **Broløs et al., 2021, arXiv 2104.05417** — "An Approach to Symbolic Regression Using Feyn." The foundational QLattice/Feyn SR paper. Does NOT apply to pKa or any molecular property. This is the *method paper* being applied here. Confirms that no pKa-specific QLattice paper exists in the published record.
   URL: https://arxiv.org/abs/2104.05417

2. **Abarbanel & Hutchison, JCTC 2024** — "QupKake: Integrating Machine Learning and Quantum Chemistry for Micro-pKa Predictions." GNN + semiempirical QM; achieves RMSE 0.5–0.8 on five external sets (Novartis R²=0.88, Literature R²=0.95, SAMPL6 R²=0.96, SAMPL7 R²=0.75). Establishes the current accuracy bar for the *same external sets* used in E6. Not interpretable in the compact-formula sense; no protonation-state audit reported.
   URL: https://pubs.acs.org/doi/10.1021/acs.jctc.4c00328 / PMC11325546

3. **Mayr, Wieder, Wieder & Langer, Front. Chem. 2022** — "Improving Small Molecule pKa Prediction Using Transfer Learning With Graph Neural Networks." Uses the Novartis (280 mol) and Literature/AvLiLuMoVe (123 mol) external sets directly comparable to E6; achieves Novartis MAE=0.82, RMSE=1.13; Literature MAE=0.62, RMSE=0.97. Not interpretable. Establishes a clean GNN baseline on the same sets.
   URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9204323/

4. **Visini et al. (BCL-XpKa), JCIM 2025** — "Interpretable Deep-Learning pKa Prediction for Small Molecule Drugs via Atomic Sensitivity Analysis." The most direct interpretability competitor. Achieves Novartis-Acid MAE=0.79, Novartis-Base MAE=0.86 using local atomic-environment (Mol2D) descriptors + atomic sensitivity analysis. Closest overlap with our interpretability framing, but BCL-XpKa is a deep classifier, not a compact symbolic formula. Does not produce a 10-term human-readable equation, does not report a distillation experiment, and does not conduct a protonation-state sensitivity audit.
   URL: https://pubs.acs.org/doi/10.1021/acs.jcim.4c01472 / PMC11733947

5. **Taminau et al. (FIGP2), ACS Omega 2024** — "Generalizability Improvement of Interpretable Symbolic Regression Models for Quantitative Structure–Activity Relationships." Introduces FIGP2 (genetic-programming SR) for QSAR interpretability with improved OOD robustness; applied to 12 ChEMBL drug-potency datasets, NOT pKa. Most directly methodologically comparable in terms of "SR interpretability vs. accuracy" framing. Does not evaluate on Novartis/SAMPL sets and does not test distillation from a GBM teacher.
   URL: https://pubs.acs.org/doi/10.1021/acsomega.3c09047 / PMC10905595

6. **Beilinson et al., JCAMD 2021** — "Evaluation of log P, pKa, and log D Predictions from the SAMPL7 Blind Challenge." Uses the exact SAMPL7 acidic set used in E6. Establishes the community pKa blind-prediction context; best SAMPL7 participant RMSE ~1.2–1.6 on the N-acylsulfonamide set. Directly contextualizes the E6 SAMPL7 result (QLattice RMSE=2.68, negative R²; GBM RMSE=2.60 also negative). Neither model competes on SAMPL7 — consistent with the documented challenge difficulty.
   URL: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8224998/

---

## GREEN FLAGS

- **No public QLattice-for-pKa paper exists.** Multiple targeted searches (Abzu/Feyn, application domain = pKa, acid dissociation, protonation) returned zero matches. This is a genuine first application of the QLattice SR framework to molecular pKa prediction.
- **No public paper runs leakage-safe symbolic regression on OP2 acidic/basic splits with a single held-out test evaluation + y-randomization + adversarial validation suite (18 checks).** The leakage discipline alone is a distinguishing contribution relative to the older JURI notebook which had selection leakage.
- **No public paper quantifies GBM-teacher distillation into SR for pKa and reports it as a negative result with a mechanistic explanation** (dense tree interactions do not compress into a compact formula). The closest general-domain work (interpretable knowledge distillation via SR, Springer Nature 2026) finds 7–21% RMSE improvement from distillation across generic datasets — our finding is in the opposite direction (worse), which is the more rigorous and publishable signal.
- **No public paper quantifies the catastrophic collapse from scoring ionized external SMILES against a neutral-structure-trained model and reports the delta explicitly** (R² dropping from +0.34 to −2…−62). The protonation-state-sensitivity finding is partially implied in the Epik 2023 paper and SAMPL writeups but has not been rigorously quantified as a benchmarking caveat in the cheminformatics literature.
- **External audit on the same benchmark sets (Novartis, AvLiLuMoVe, SAMPL7) that QupKake, Mayr 2022, and BCL-XpKa use**, making the modest symbolic numbers directly comparable to published SOTA — this is methodologically honest and gives reviewers a meaningful anchor.
- **Acidic/basic split architecture + electronic descriptor dominance (partial-charge/charge-weighted terms).** Chemically sensible; most deep-learning methods are single-model and do not explicitly surface which descriptor class drives predictions.
- **Spearman rank correlation reported alongside R²** for the external sets — this is the correct metric when a model is poorly calibrated OOD (the symbolic models transfer rank signal but are scale-miscalibrated), and few published pKa benchmarks decompose the two.

---

## RED FLAGS

- **OP2 test R² (symbolic) = 0.45–0.47 is well below published GNN baselines** (QupKake R²=0.88 on Novartis, BCL-XpKa MAE 0.79–0.86 on Novartis). The paper is *not* claiming to match these — but a referee who does not read carefully may reject on accuracy alone. The framing must be explicit and upfront.
- **External symbolic R² is negative for basic sets (−0.38 pooled) with no explanation other than OOD miscalibration.** This needs to be turned into a scientific point, not just a red number in a table. Specifically: report that Spearman is still above chance (ρ=0.42 > chance floor ρ=0.17) so rank signal is transferred, but the scale is badly off, and this is a motivation for OOD calibration as future work — not a model failure per se.
- **No repeated/nested CV.** The single held-out R² values (0.448 acidic, 0.470 basic) carry no uncertainty band. A reviewer may reasonably ask whether these are within a lucky random seed. This is the single most important missing element.
- **The OP2 training corpus is undocumented in this assessment** (size, chemical diversity, overlap with external sets). If OP2 and Novartis/AvLiLuMoVe share compounds, the external evaluation is invalid. This must be checked explicitly and reported.
- **No significance test comparing symbolic vs. GBM.** For an honest-audit framing this is acceptable (we are reporting the gap, not claiming parity), but a paired Wilcoxon on test residuals would pre-empt a methodological objection.
- **The Molecules 2026 paper (count-based fingerprints + Catboost, R²=0.890) uses a different dataset and does not run on Novartis/SAMPL**, so it is not a direct threat — but it occupies the "interpretable + ensemble" framing in the same target journal (Molecules). This overlap requires one paragraph of differentiation: compact symbolic equation vs. 81-feature Catboost model (the latter is not human-readable).

---

## RECOMMENDED NEXT MOVE

**The single experiment that would make this unambiguously publishable:** repeated / nested 5×5 CV to get a mean ± 95% CI on the symbolic R² (e.g., 0.45 ± 0.04 acidic). Without this, the reported numbers are a point estimate from a lucky seed. This is already listed as the next open work item and should be done before submission.

**Additionally, before writing:**
1. Verify no train/test compound overlap between OP2 and the Novartis/AvLiLuMoVe external sets (InChIKey or canonical SMILES deduplication). If overlap exists, the E6 evaluation must exclude those compounds and re-report.
2. Add a one-paragraph OOD calibration analysis for the symbolic basic model (e.g., a simple linear recalibration using the AvLiLuMoVe slice as a calibration set, reporting corrected R² on Novartis). This converts the "negative R² is embarrassing" into "recalibrated symbolic model achieves R²=X, demonstrating the rank signal is real but scale transfer requires domain adaptation."
3. The lead framing must be: *methodology and audit paper, not an accuracy paper.* The abstract must state in sentence 2 or 3 that the symbolic models are not competitive with GNN baselines on accuracy, and that this is deliberate. Reviewers who feel misled by the title will reject regardless of the actual contribution.

**Target-venue note:** Journal of Cheminformatics (first choice) regularly publishes benchmark/methodology papers with similar accuracy profiles when the methodological contribution is clean. The leakage-safe pipeline + external audit + distillation negative result is a coherent package for that venue. JCIM is more accuracy-driven and is a harder sell without the uncertainty bands. Molecules (third choice) is easier but lower-impact, and the Molecules 2026 competitor paper (R²=0.890) makes it more crowded.

---

## Sources Searched

- Abzu/Feyn QLattice documentation and citing papers (no pKa application found)
- arXiv 2104.05417 (Broløs et al., foundational Feyn SR paper)
- PMC11325546 / JCTC 2024 (QupKake)
- PMC9204323 / Front. Chem. 2022 (Mayr et al., GNN transfer learning, Novartis/Literature sets)
- PMC11733947 / JCIM 2025 (BCL-XpKa, interpretable deep learning pKa)
- PMC10905595 / ACS Omega 2024 (FIGP2, symbolic regression for QSAR)
- PMC8224998 / JCAMD 2021 (SAMPL7 blind challenge evaluation)
- MDPI Molecules 2026 vol.31 no.6 art.961 (count-based fingerprints + Catboost pKa)
- PMC12268062 (GraFpKa, explainable deep learning pKa)
- Wikipedia QLattice article (no chemistry applications documented)
- Chemrxiv Uni-pKa 2023 (403 on PDF fetch; metadata confirmed via search)
- Papers With Code pKa leaderboard (redirected; no standalone leaderboard found)
