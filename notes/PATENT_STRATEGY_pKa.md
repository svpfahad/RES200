# Patent Strategy Note: RES200 pKa Prediction

Date: 2026-05-12

Purpose: shift the RES200 work from publication-only research into a patent-oriented invention track.

Important: this is a technical strategy note, not legal advice. A patent attorney or university IP office should review before any filing or public disclosure.

## Current Position

The current project has strong evidence for ML-based pKa prediction, but the broad idea of "predict pKa from molecular descriptors using ML" is already crowded.

Local evidence:

- LightGBM on the OP2 full descriptor test split: R2 = 0.8356, RMSE = 1.3746, MAE = 0.8681.
- XGBoost on the same split: R2 = 0.8289, RMSE = 1.4024, MAE = 0.8736.
- Current QLattice symbolic runs are interpretable but much less accurate: best R2 = 0.5476, RMSE = 2.2806.
- Y-randomization control for LightGBM gives negative R2, which supports that the model is learning real signal rather than random target structure.

Patent implication: do not try to patent a generic XGBoost, LightGBM, Random Forest, descriptor, fingerprint, or QLattice pKa model. That will be easy to reject as obvious or already disclosed.

## Recommended Shift

The strongest patent direction is a specific computer-implemented pKa decision system, not a standalone model.

Best current claim lane:

> A leakage-safe, environment-aware, interpretable pKa prediction system that trains a high-accuracy descriptor/graph model, distills chemically meaningful symbolic rules, estimates applicability domain and uncertainty, and outputs a pKa value plus a decision recommendation for molecule screening under specified use conditions.

The word "environment-aware" matters. Standard aqueous pKa prediction is crowded. A stronger, Saudi-relevant angle is pKa prediction under industrially relevant environments:

- high salinity water and brines,
- elevated temperature,
- mixed solvents,
- CO2 capture solvent screening,
- desalination and brine treatment chemistry,
- petrochemical or oilfield additive screening,
- reactive extraction and separation workflows.

This turns the project from "predict a molecule property" into "select a molecule or process condition for a real industrial use case."

## Preliminary Prior Art Map

| Prior art area | What already exists | Risk to us | How to avoid the crowded area |
|---|---|---:|---|
| Descriptor-based pKa QSAR | Open-source pKa models using descriptors, fingerprints, Random Forest, XGBoost, and related ML approaches | High | Do not claim descriptors plus ML alone |
| Graph neural pKa models | MolGpKa, SAT4pKa, QupKake, Uni-pKa, and other graph/transformer/free-energy models | High | Do not claim graph pKa prediction alone |
| QM plus ML pKa | QupKake uses GFN2-xTB/QM features plus GNNs for micro-pKa; other recent work uses semiempirical or solvation calculations | High | Use QM features only as one optional input, not the invention core |
| Thermodynamics-aware pKa | Uni-pKa models protonation ensembles and free energies for thermodynamic consistency | High | Do not claim protonation ensemble/free-energy pKa generally |
| ML in non-aqueous/arbitrary solvents | Recent papers already address pKa prediction in non-aqueous solvents and arbitrary solvent environments | Medium to high | Narrow to measurable workflow improvements: industrial brine/temperature/solvent correction, confidence gating, and decision output |
| Patent literature around molecular ML | Broad patent families cover graph/quantum representations, thermodynamic ensembles, molecular descriptors, ligand screening, and molecular property prediction | High | Claims need concrete technical steps, data controls, output controls, and a specific application |

## Potential Invention Disclosure

Working title:

Environment-Aware Interpretable System for Predicting pKa and Screening Ionizable Organic Compounds

Problem:

Standard pKa models often predict a single aqueous pKa value without robust applicability-domain warnings, without leakage-safe validation, and without environment-specific correction for industrial conditions such as salinity, temperature, or mixed solvent composition.

Proposed solution:

1. Receive a molecule structure and optional condition vector:
   - SMILES, InChI, or molecular graph.
   - Acidic/basic task label or ionization-site candidates.
   - Environment fields such as temperature, ionic strength, solvent fraction, salinity, pH range, or process use case.

2. Normalize and audit the input:
   - Canonicalize structure.
   - Deduplicate by InChIKey or canonical SMILES.
   - Separate acidic/basic or micro/macro pKa tasks.
   - Prevent train/test/external leakage at structure level.

3. Generate multi-layer representations:
   - RDKit/Mordred descriptors.
   - Morgan/MACCS fingerprints.
   - Ionization-site and fragment indicators.
   - Optional QM, solvation, or environment descriptors.

4. Train an accuracy model:
   - Use LightGBM/XGBoost/CatBoost or a graph model as the high-accuracy predictor.
   - Use train-only feature filtering and seed-repeated validation.
   - Store metrics, predictions, residuals, and leakage audit outputs.

5. Train an interpretable symbolic layer:
   - Select top-K features only from training data.
   - Fit symbolic equations or rule sets that approximate either experimental pKa or the high-accuracy model.
   - Penalize formulas that violate chemical expectations or exceed complexity thresholds.

6. Compute reliability outputs:
   - Applicability-domain score from structural similarity and descriptor-space distance.
   - Prediction interval or uncertainty.
   - Warning flags for out-of-domain molecules, unsupported ionization classes, or unstable formula behavior.

7. Output a decision object:
   - Predicted pKa.
   - Acid/base class or likely ionization site.
   - Interpretable formula/rule explanation.
   - Applicability-domain and confidence flags.
   - Screening recommendation for the selected industrial use case.

## Claim Angles

Strongest claim angle:

- A method that combines environment-conditioned pKa prediction with applicability-domain gating and symbolic explanation to select compounds or operating conditions for a specified industrial chemistry workflow.

Medium claim angles:

- A leakage-safe pipeline that generates acidic/basic pKa predictors and symbolic surrogate equations from a controlled descriptor feature space.
- A confidence-gated pKa screening system that rejects or routes out-of-domain structures before process selection.
- A method for correcting aqueous pKa predictions using condition descriptors for salinity, temperature, or solvent composition, then producing a screening decision.

Weak claim angles:

- XGBoost predicts pKa from RDKit/Mordred descriptors.
- QLattice predicts pKa.
- Combining fingerprints and descriptors for pKa.
- Reporting feature importance for pKa.

## What We Need Next

1. Disclosure freeze
   - Confirm whether the JURI paper has been published, posted, presented, or otherwise publicly disclosed.
   - Record exact dates for manuscript submission, revision submission, conference use, GitHub uploads, posters, and any public sharing.

2. Patentability search phase 2
   - Search claims, not only abstracts.
   - Target CPC classes G16C20/70, G16C20/50, G06N20/00, G06N3/04, and pKa/acidity/free-energy/protonation terms.
   - Build a claim chart against the strongest patents and papers.

3. Technical prototype
   - Add an explicit condition vector to the pipeline, even if the first version only handles "standard aqueous" as a baseline.
   - Add applicability-domain scoring.
   - Add symbolic surrogate extraction from the best accuracy model.
   - Generate a structured JSON output with pKa, confidence, formula, and recommendation.

4. Differentiating experiment
   - Best option: collect or curate pKa data under non-standard conditions relevant to KSA industry, such as brine, mixed solvents, elevated temperature, or CO2 capture candidate molecules.
   - Minimum option: demonstrate that the pipeline produces better rejection/confidence behavior than ordinary pKa prediction on external holdouts.

5. Attorney package
   - One-page invention summary.
   - Architecture diagram.
   - Prior-art matrix.
   - Example input/output.
   - Experimental evidence table.
   - Proposed claim set.

## Go/No-Go

Go only if we can claim at least one of these:

- environment-conditioned prediction beyond standard aqueous pKa,
- confidence-gated screening that improves decision reliability,
- symbolic/chemical explanation tied to a real screening decision,
- a specific industrial workflow where pKa prediction changes compound or condition selection,
- a validated leakage-safe pipeline with documented applicability-domain controls.

No-go for patent filing if the invention remains only:

- "a model trained on descriptors to predict pKa,"
- "a QLattice formula for pKa,"
- "a comparison of ML models for pKa,"
- "a research paper with better benchmark metrics."

## Preliminary Source List

- Baltruschat and Czodrowski, Machine learning meets pKa, F1000Research, 2020.
- Mansouri et al., Open-source QSAR models for pKa prediction using multiple machine learning approaches, Journal of Cheminformatics, 2019.
- Wu et al., Machine learning methods for pKa prediction of small molecules: Advances and challenges, Drug Discovery Today, 2022.
- Pan et al., MolGpKa: a web server for small molecule pKa prediction using a graph-convolutional neural network, JCIM, 2021.
- Abarbanel and Hutchison, QupKake: Integrating Machine Learning and Quantum Chemistry for Micro-pKa Predictions, JCTC, 2024.
- Luo et al., Bridging Machine Learning and Thermodynamics for Accurate pKa Prediction, JACS Au, 2024.
- Qiu et al., Graph transformer based transfer learning for aqueous pKa prediction of organic small molecules, Chemical Engineering Science, 2024.
- Suzuki and Mori, Acidity Prediction in Arbitrary Solvents: Machine Learning Based on Semiempirical Molecular Orbital Calculation, J. Phys. Chem. A, 2025.
- Zheng et al., pKa prediction in non-aqueous solvents, Journal of Computational Chemistry, 2025.
- US20220383992A1 / US12211592B2, Machine learning based methods of analysing drug-like molecules.
- US20250061979A1, Predicting molecule properties using graph neural network.
- SE547814C2 / WO2024151196A1, Ligand candidate screen and prediction based on molecular descriptors.
