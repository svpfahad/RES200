# Source Manifest

Access date: 2026-05-12.

## Data Sources

0. Zenodo 3251643, "Ionic liquid Properties"
   - URL: https://zenodo.org/records/3251643
   - DOI: 10.5281/zenodo.3251643
   - Reported scope: experimental data for 12 ionic-liquid properties including CO2 capacity; cation and anion structures are supplied in SMILES format.
   - License: CC BY 4.0.
   - Local files used: `CO2CAPACITY.txt`, `CA.smi`
   - Prepared file: `data/processed/zenodo_3251643_co2capacity.csv`
   - Current prepared scope: 10,865 valid rows after filtering nonpositive pressure or xCO2 values; 216 solvent identities.
   - Use: main current benchmark dataset because it has an explicit target column.

1. Figshare 29228990, "Predicting CO2 Solubility in Diverse Ionic Liquids: A Data-Driven Approach Using Machine Learning Algorithms"
   - URL: https://figshare.com/articles/dataset/Predicting_CO_sub_2_sub_Solubility_in_Diverse_Ionic_Liquids_A_Data-Driven_Approach_Using_Machine_Learning_Algorithms/29228990
   - Reported scope: 16,480 experimental CO2 solubility data points, 296 ionic liquids, 103 cations, 78 anions.
   - License: CC BY-NC 4.0.
   - Use: primary research dataset candidate.
   - Local inspection note: the downloaded workbook exposes large train/test sheets with model predictions and ARD columns, while the exact experimental target is explicit in the smaller `S4-Experimental data_Unique ILs` sheet. Use `scripts/prepare_figshare_29228990.py` for verified target training until the full target column is obtained.

2. Chemical Engineering Science 2020, "Prediction of CO2 solubility in ionic liquids using machine learning methods"
   - URL: https://www.sciencedirect.com/science/article/pii/S0009250920302840
   - Reported scope: 10,116 CO2 solubility measurements.
   - Use: close prior art and possible dataset lead.

## Prior-Art Sources

1. Scientific Reports 2024, "Prediction of CO2 solubility in Ionic liquids for CO2 capture using deep learning models"
   - URL: https://www.nature.com/articles/s41598-024-65499-y

2. EST Letters 2024, "Screening Environmentally Benign Ionic Liquids for CO2 Absorption Using Representation Uncertainty-Based Machine Learning"
   - URL: https://pubmed.ncbi.nlm.nih.gov/39554598/

3. CPSign 2024, conformal prediction for cheminformatics
   - URL: https://link.springer.com/article/10.1186/s13321-024-00870-9

4. WO2024187266A1, amine-based carbon capture solvent degradation monitoring
   - URL: https://patents.google.com/patent/WO2024187266A1/en

5. CN116230115B, phase-change absorbent screening based on machine learning and quantum chemistry
   - URL: https://patents.google.com/patent/CN116230115B/en

## Saudi Context Sources

1. Aramco DAC test unit announcement
   - URL: https://www.aramco.com/en/news-media/news/2025/saudi-aramco-launches-the-first-direct-air-capture-and-carbon-dioxide

2. Aramco CCUS page
   - URL: https://www.aramco.com/en/making-a-difference/planet/carbon-capture-utilization-and-storage

3. Saudi patent law via WIPO Lex
   - URL: https://www.wipo.int/wipolex/en/legislation/details/3596
