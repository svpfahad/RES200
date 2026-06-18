Dear Dr. Zahid Bhat,

Thank you for the opportunity to revise our manuscript JURI-00032-2025-01, entitled "XGBoost-Based Prediction of pKa Values Using Molecular Fingerprints and Computed Descriptors: A Comparative Study." We sincerely appreciate the reviewer's thorough evaluation and constructive feedback, which has significantly improved the quality of our work.

The manuscript has been substantially rewritten to address all concerns raised during the review process. Below, we provide a point-by-point response to each comment.

---

**Reviewer Comment 1 — Reference inconsistencies:**
*"The citation order is incorrect (e.g., jumping from reference [1] to [6], then to [3], and back to [2]). Please carefully revise and reorder the references according to their appearance in the text."*

**Response:** The entire reference list has been reordered to follow the order of first appearance in the text. All 15 references now appear in strict sequential order (1, 2, 3, … 15), consistent with the Nature citation style required by the JURI template. Every in-text citation has been verified to match the renumbered reference list.

---

**Reviewer Comment 2 — Over-reliance on a single source:**
*"Reference [1] is cited more than 12 times. A well-balanced introduction and methodology section should incorporate insights from a broader range of literature."*

**Response:** The literature base has been substantially expanded. The revised manuscript now includes 15 references (up from the original count), drawing from diverse sources covering molecular fingerprints (Rogers & Hahn, 2010), MACCS keys (Durant et al., 2002), descriptor calculators (Moriwaki et al., 2018), gradient boosting best practices (Boldini et al., 2023), QSAR validation guidelines (Tropsha, 2010), DFT-based pKa methods (Lawler et al., 2021; Sanchez et al., 2024), graph neural network approaches (Pan et al., 2021), and multi-solvent prediction (Yang et al., 2020). The Mansouri et al. reference is now cited 7 times (reduced from 12+), and only where directly relevant — specifically when referring to the original dataset curation and benchmark results.

---

**Reviewer Comment 3 — Structure of the introduction and conclusion:**
*"Avoid using section-like titles within the introduction. Instead, write a coherent narrative that logically presents the background, motivation, and objectives. The same applies to the conclusion."*

**Response:** The introduction has been completely rewritten as a continuous, flowing narrative without any internal subheadings. It progresses logically from (i) the importance of pKa in medicinal chemistry, (ii) limitations of experimental and quantum-mechanical methods, (iii) the emergence of machine learning approaches with emphasis on XGBoost, (iv) the role of molecular representations, (v) prior work and the identified research gap, to (vi) the specific objectives of the present study. The conclusion has likewise been restructured as a concise, integrative summary that reflects the overall findings without bullet points or sub-sections.

---

**Reviewer Comment 4 — Formatting issues:**
*"Several words throughout the manuscript are unnecessarily written in bold. Please remove these and maintain a consistent, formal formatting style."*

**Response:** All unnecessary bold formatting has been removed from the body text. Bold is now used only for section headings, table/figure caption labels (e.g., "Figure 1.", "Table 1."), and the keyword label, in accordance with the JURI template. The manuscript maintains consistent Times New Roman formatting throughout.

---

**Reviewer Comment 5 — Section and subsection numbering:**
*"The manuscript contains multiple errors in the section numbering. Please revise to ensure a coherent and consistent structure."*

**Response:** The section structure has been revised to follow a clean, consistent hierarchy: Abstract, Introduction, Methodology (with subsections: Dataset, Morgan Circular Fingerprints, MACCS Structural Keys, Physicochemical Descriptors, Combined Descriptor Vectors, Extended Descriptor Analysis, XGBoost Model Training and Evaluation), Results and Discussion (with subsections for each analysis), Conclusions, Acknowledgements, and References. All numbering errors have been corrected.

---

**Reviewer Comment 6 — Figures and tables:**
*"The numbering of figures and tables is inconsistent, and some titles are missing. For instance, Table 1 and Figure 2 (which should be Figure 1) lack captions. Please revise all figures and tables accordingly."*

**Response:** All figures and tables have been renumbered sequentially and provided with complete, descriptive captions. The revised manuscript contains seven figures (Figures 1–7) and two tables (Tables 1–2), each with a full caption describing the content. Specifically: Figure 1 (Morgan fingerprint performance), Figure 2 (model comparison), Figure 3 (predicted vs. experimental pKa), Figure 4 (feature importance), Figure 5 (overfitting analysis), Figure 6 (residual analysis), and Figure 7 (heatmap). Table 1 summarizes performance across descriptor categories and Table 2 lists the top ten features by importance.

---

**Reviewer Comment 7 — Feature importance analysis:**
*"Although highlighted in the methodology, this analysis is not discussed or presented in the results. Please address this omission."*

**Response:** A dedicated "Feature Importance Analysis" subsection has been added to the Results and Discussion section. This includes Figure 4 (a horizontal bar chart of the top 20 features by XGBoost gain metric) and Table 2 (the top 10 features with importance scores and chemical descriptions). The discussion provides chemical interpretation of the most influential descriptors, including nAcid (number of acidic groups, importance = 0.176), SpAbs_Dzpe (spectral Barysz electronegativity, 0.049), fr_Ar_NH (aromatic amine count, 0.029), and others, explaining their relevance to acid–base equilibria.

---

**Reviewer Comment 8 — Prediction results (predicted vs. actual figures):**
*"Figures comparing predicted vs actual values are missing. These are essential for evaluating the performance of your model and should be included."*

**Response:** Figure 3 has been added, presenting scatter plots of predicted versus experimental pKa values for the full-descriptor XGBoost model on both the training set (n = 2,293; R² = 0.9998) and test set (n = 765; R² = 0.829). The dashed identity line is included for reference, and the plots demonstrate good generalization with no systematic bias across the pKa range. Additionally, Figure 6 presents a residual analysis (histogram and residuals-versus-predicted plot) to further characterize prediction errors.

---

**Reviewer Comment 9 — Writing style:**
*"The manuscript should maintain a formal academic tone. Avoid using informal expressions or pronouns such as 'we'. Revise the text to reflect the appropriate style for a research publication."*

**Response:** The entire manuscript has been rewritten using formal passive voice. A systematic check confirmed zero instances of the pronoun "we" in the revised text. All informal expressions have been replaced with formal academic language appropriate for a research publication.

---

**Additional improvements beyond reviewer comments:**

- ORCID IDs have been provided for all authors, as requested by the Editor.
- An extended analysis using 1,135 Mordred and RDKit descriptors has been added, demonstrating improved prediction accuracy (test R² = 0.829, RMSE = 1.404) beyond the fingerprint-based models.
- An overfitting analysis section with a dedicated figure (Figure 5) has been included to address the train–test performance gap.
- A "Comparison with Literature" subsection contextualizes the results against published benchmarks.

We believe these revisions comprehensively address all concerns raised by the reviewer and substantially improve the quality, rigor, and presentation of the manuscript. We are grateful for the constructive feedback and look forward to your decision.

Respectfully,

Fahad Atwi, Moayad Alnammi, Mohammed Al-Khater
King Fahd University of Petroleum & Minerals, Dhahran, Saudi Arabia
