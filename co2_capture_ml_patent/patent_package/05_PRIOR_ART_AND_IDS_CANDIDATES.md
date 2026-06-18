# Prior Art And IDS Candidates

Status: attorney-review draft, not legal advice. U.S. counsel should decide what to submit in an IDS.

## Closest Prior Art

| Ref | Citation | Why It Matters | Claim Strategy |
|---|---|---|---|
| A | Zhong et al., "Screening Environmentally Benign Ionic Liquids for CO2 Absorption Using Representation Uncertainty-Based Machine Learning," Environmental Science & Technology Letters, 2024, DOI: 10.1021/acs.estlett.4c00524 | Closest scientific reference. Uses uncertainty across representations for IL CO2 absorption screening. | Do not claim "uncertainty for IL CO2 screening" broadly. Focus on calibrated interval, held-out chemistry gate, OOD refusal, local workflow, and decision labels. |
| B | Norinder et al., "Introducing Conformal Prediction in Predictive Modeling," Journal of Chemical Information and Modeling, 2014, DOI: 10.1021/ci5001168 | Foundational conformal prediction in cheminformatics and applicability-domain context. | Do not claim conformal prediction itself. Claim domain-specific integration into CO2 capture solvent triage. |
| C | Arvidsson McShane et al., "CPSign: conformal prediction for cheminformatics modeling," Journal of Cheminformatics, 2024, DOI: 10.1186/s13321-024-00870-9 | Tooling prior art for conformal intervals in cheminformatics. | Emphasize capture-solvent workflow, decision suppression, and evidence tied to solvent-held-out validation. |
| D | Bastami et al., "Predicting CO2 Solubility in Diverse Ionic Liquids: A Data-Driven Approach Using Machine Learning Algorithms," Energy & Fuels, 2025, DOI: 10.1021/acs.energyfuels.5c01345 | Large modern QSPR/ML dataset and models for CO2 solubility in ILs. | Avoid broad prediction/model/descriptors claims. |
| E | Venkatraman et al., "The Ionic Liquid Property Explorer: An Extensive Library of Task-Specific Solvents," Data, 2019, DOI: 10.3390/data4020088; Zenodo 3251643 | Public data source used in this project; IL property dataset with CO2 capacity and SMILES. | Cite as data source and background, not invention. |
| F | "Predicting CO2 capture of ionic liquids using machine learning," Journal of CO2 Utilization, 2017, DOI: 10.1016/j.jcou.2017.06.012 | Early ML prediction of IL CO2 capture. | Avoid broad ML-based screening claims. |
| G | "Prediction of CO2 solubility in Ionic liquids for CO2 capture using deep learning models," Scientific Reports, 2024 | Deep learning for IL CO2 solubility. | Differentiate with calibrated decision workflow and OOD abstention. |
| H | Qin et al., "Predicting the solubility of CO2 and N2 in ionic liquids based on COSMO-RS and machine learning," Frontiers in Chemistry, 2024, DOI: 10.3389/fchem.2024.1480468 | COSMO-RS plus ML for gas solubility in ILs. | Avoid descriptor/model accuracy focus. |
| I | WO2025073762A1, "Solvent screening method" | Patent publication involving ML solvent screening, uncertainty, and process parameters. | Keep claims limited to CO2 capture solvent screening and calibrated OOD decision suppression. |
| J | US20230122168A1, "Conformal Inference for Optimization" | Patent publication applying conformal intervals to optimization of biological sequences. | Avoid broad conformal optimization. Claim solvent/CO2-specific data structures and decisions. |
| K | US20260042050 / WO2024187266A1, "Amine-based carbon capture solvent degradation monitoring" | Carbon-capture ML monitoring/threshold prior art. | Keep claims on pre-deployment screening and candidate triage, not degradation monitoring. |

## IDS Candidate Table

Counsel should review whether to include these items in a U.S. Information Disclosure Statement:

1. Zhong et al., Environmental Science & Technology Letters, 2024, DOI: 10.1021/acs.estlett.4c00524.
2. Norinder et al., Journal of Chemical Information and Modeling, 2014, DOI: 10.1021/ci5001168.
3. Arvidsson McShane et al., Journal of Cheminformatics, 2024, DOI: 10.1186/s13321-024-00870-9.
4. Bastami et al., Energy & Fuels, 2025, DOI: 10.1021/acs.energyfuels.5c01345.
5. Venkatraman, Evjen, and Lethesh, Zenodo record 3251643, DOI: 10.5281/zenodo.3251643.
6. Venkatraman et al., Data, 2019, DOI: 10.3390/data4020088.
7. Predicting CO2 capture of ionic liquids using machine learning, Journal of CO2 Utilization, 2017, DOI: 10.1016/j.jcou.2017.06.012.
8. Prediction of CO2 solubility in Ionic liquids for CO2 capture using deep learning models, Scientific Reports, 2024.
9. Qin et al., Frontiers in Chemistry, 2024, DOI: 10.3389/fchem.2024.1480468.
10. WO2025073762A1, Solvent screening method.
11. US20230122168A1, Conformal Inference for Optimization.
12. US20260042050 or related WO publication, Amine-based carbon capture solvent degradation monitoring.

## Remaining Novel Claim Space

The likely remaining claim space is a specific workflow in which the system:

- requires chemically held-out validation before candidate ranking;
- calibrates intervals on held-out chemistry/process regimes;
- converts interval width and domain status into a rank penalty;
- suppresses positive recommendation for OOD candidates;
- outputs reason-coded decisions for experiment planning;
- runs locally without exposing candidate chemistry to cloud services;
- generates a concrete experimental queue for CO2 capture testing.
