# Specification Draft

Status: attorney-review draft, not legal advice.

## Title

Confidence-Gated Solvent Screening System

## Field

The disclosure relates to computer-implemented chemical screening, carbon capture, solvent selection, molecular property prediction, applicability-domain assessment, calibrated uncertainty estimation, and experimental triage of CO2 capture solvent candidates.

## Background

Carbon dioxide capture processes require solvents or solvent systems that provide suitable absorption performance under practical operating conditions. Ionic liquids and related solvents can be varied by changing cations, anions, substituents, mixtures, and operating conditions. Experimental testing of all possible candidates is expensive and slow.

Machine-learning models have been used to predict CO2 solubility or absorption performance from molecular descriptors and operating conditions. However, a model may provide a confident point prediction for a candidate that is outside the chemical or process domain represented in the training data. A random row train/test split may also report high accuracy when records for the same solvent identity appear in both the training and test sets at different temperatures or pressures. Such evaluation can overstate the model's ability to screen new solvents.

There is a need for a solvent screening system that does not merely output a point prediction, but evaluates whether the prediction is supported by the model domain and by calibrated uncertainty, and that can abstain from recommending a candidate when additional experimental data is required.

## Summary

In one embodiment, a computer-implemented method screens candidate solvents for CO2 capture. The method receives training records including solvent identity information, molecular representation information, operating condition information, and measured CO2 capture response values. The method forms a chemically aware evaluation split, trains one or more regression models, calibrates prediction intervals from calibration residuals, builds an applicability-domain profile from the training data, and evaluates candidate solvent records.

For each candidate, the method predicts a CO2 capture response, determines a prediction interval, determines an applicability-domain status, computes a rank score that penalizes uncertainty and domain violations, and outputs a decision label. The decision label can include `test`, `watch`, `reject`, or `needs_data`. In some embodiments, an out-of-domain candidate is not ranked as a viable candidate even if the point prediction is high.

In another embodiment, the system outputs reason codes identifying why a candidate was accepted, rejected, or routed for data generation. Example reason codes include unseen cation, unseen anion, temperature outside training range, pressure outside training range, descriptor outside training range, interval too wide, or candidate within training domain.

In another embodiment, the output is used to form an experimental test queue, a laboratory information management system record, or an experimental protocol specifying candidate solvent, temperature, pressure, and priority.

## Brief Description of Drawings

FIG. 1 illustrates an overall confidence-gated solvent screening workflow.

FIG. 2 illustrates model training using solvent-held-out or ion-family-held-out validation and conformal calibration.

FIG. 3 illustrates applicability-domain assessment using numeric range checks and categorical unseen-level checks.

FIG. 4 illustrates candidate scoring and decision labeling.

FIG. 5 illustrates an example local computer system configured to execute the workflow.

FIG. 6 illustrates an optional experimental test queue generated from candidate decisions.

## Detailed Description

### Definitions

The term "solvent" includes a pure solvent, ionic liquid, deep eutectic solvent, amine-containing solvent, solvent blend, or other liquid composition considered for CO2 capture.

The term "CO2 capture response" includes one or more measured or transformed values associated with CO2 absorption or solubility, including mole fraction, loading, capacity, Henry-law constant, selectivity, rate, viscosity-adjusted capture metric, or a logarithmic transform of any of the foregoing.

The term "candidate record" means a set of data describing a candidate solvent under one or more operating conditions. A candidate record can include molecular structures, cation identifiers, anion identifiers, SMILES strings, descriptors, temperature, pressure, gas composition, water content, impurity condition, process unit identifier, or other process attributes.

The term "applicability-domain status" means a classification or score indicating whether a candidate is inside, near the edge of, or outside the domain represented by training data.

The term "prediction interval" means an interval associated with a prediction, including a conformal interval, quantile interval, Bayesian interval, ensemble interval, or calibrated residual interval.

### Training Records

Training records may be obtained from public datasets, internal experimental datasets, published literature, laboratory information systems, or combinations thereof. Each training record may include:

- solvent identity;
- cation and anion identity for ionic liquids;
- one or more molecular representations;
- operating conditions, including temperature and pressure;
- measured CO2 capture response;
- source information or reference identifier.

The system may transform pressure using a logarithm and may transform solubility or capacity using a logarithm. The system may compute descriptors from SMILES strings. In one embodiment, lightweight descriptors include string length, branch count, ring digit count, charge symbol counts, aromatic atom counts, bond counts, atom token counts, heteroatom counts, and a formal-charge proxy. In other embodiments, the descriptors include fingerprints, group-contribution descriptors, sigma-profile descriptors, quantum chemical descriptors, graph embeddings, image embeddings, or learned molecular representations.

### Leakage-Aware Evaluation

The system forms a training set, calibration set, and test set. In one embodiment, records are split by solvent identity so that solvent identities in the test set do not appear in the training set. In another embodiment, records are split by cation family, anion family, scaffold, molecular cluster, process window, source publication, or combinations thereof.

The system can require that a candidate model satisfy a held-out chemistry evaluation gate before the model is used for candidate ranking. The gate can include one or more thresholds for error, interval coverage, interval width, or calibration quality.

### Model Training

The system trains one or more supervised regression models to predict the CO2 capture response. Example models include linear models, random forests, gradient boosted trees, histogram gradient boosting, Gaussian process models, neural networks, graph neural networks, support vector regression, and ensembles thereof.

The system selects a model based on one or more metrics computed on a held-out test set, such as root mean squared error, mean absolute error, coefficient of determination, interval coverage, interval width, or a combined utility metric.

### Prediction Interval Calibration

The system calibrates prediction intervals using a calibration set. In one embodiment, the system computes absolute residuals between calibration predictions and measured calibration responses, selects a residual quantile based on a desired miscoverage rate, and forms a prediction interval around a candidate prediction.

The calibration set may be split by solvent identity or by ion family. This prevents the interval from being calibrated only on records that are chemically too similar to training records.

### Applicability-Domain Profile

The system builds a profile of the training domain. For numeric features, the profile can include minimum, maximum, median, interquartile range, quantiles, or distance thresholds. For categorical features, the profile can include the observed cation identifiers, anion identifiers, solvent families, scaffold labels, source labels, process-unit labels, or other categories.

For a candidate record, the system compares candidate features to the training profile. If one or more categorical values are unseen, or if one or more numeric features are outside a training range or margin, the candidate may be classified as edge or out of domain.

### Candidate Scoring

For a candidate, the system computes:

- a predicted response value;
- lower and upper prediction interval values;
- interval width;
- applicability-domain status;
- confidence grade;
- one or more reason codes;
- rank score;
- decision label.

In one embodiment, the rank score is the predicted response minus an uncertainty penalty and a domain penalty. The uncertainty penalty may be proportional to interval width. The domain penalty may increase when the candidate is edge of domain or out of domain. In some embodiments, the candidate is suppressed from positive recommendation when out of domain regardless of the point prediction.

### Decision Labels

The decision label can include:

- `test`: candidate is sufficiently supported and is prioritized for experimental testing;
- `watch`: candidate may be considered but has lower score or confidence;
- `reject`: candidate is not recommended based on predicted performance or rank score;
- `needs_data`: candidate requires additional experimental data before recommendation.

The `needs_data` label is an active abstention output. It prevents the system from presenting unsupported high point predictions as recommended solvent candidates.

### Local/Offline Operation

In one embodiment, the system executes on a local personal computer or workstation without transmitting candidate structures or process data to a cloud service. This can protect confidential chemical structures, plant-specific operating conditions, and unpublished experimental records. The system may run using public data, private laboratory data, or combinations thereof.

### Optional Experimental Queue

In one embodiment, the system outputs an experimental queue. The queue can identify candidate solvent, temperature, pressure, gas composition, priority, confidence grade, and reason codes. The queue may be exported as a CSV, JSON, laboratory information management system entry, or instrument control instruction. In another embodiment, the system controls or configures a capture test apparatus according to the queue.

### Example 1: Public-Data Implementation

A public ionic-liquid property dataset is downloaded. CO2 capacity records and cation/anion SMILES structures are prepared into a table containing solvent identity, temperature, pressure, logarithmic pressure, logarithmic CO2 solubility, cation code, anion code, and cation/anion SMILES.

The implementation trains models on 10,865 valid records representing 216 solvent identities. A random row split produces high apparent accuracy, while a solvent-held-out split produces lower accuracy and wider uncertainty. This demonstrates that random row performance is not sufficient for screening new solvent identities.

### Example 2: Evidence Suite

An evidence suite trains and evaluates multiple configurations:

| run | split | best model | R2 | MAE | RMSE | coverage | interval width |
|---|---|---:|---:|---:|---:|---:|---:|
| random row | random | random forest | 0.9402 | 0.1636 | 0.2985 | 0.9146 | 0.8980 |
| solvent holdout | solvent identity | XGBoost | 0.3226 | 0.4465 | 0.8874 | 0.8596 | 1.4720 |
| structure-only solvent holdout | solvent identity, no ion codes | XGBoost | 0.2529 | 0.5214 | 0.9319 | 0.8910 | 1.8151 |
| cation holdout | cation family | hist gradient boosting | 0.5241 | 0.4211 | 0.8754 | 0.9342 | 2.9450 |
| anion holdout | anion family | XGBoost | 0.7901 | 0.3633 | 0.5953 | 0.9452 | 2.3826 |

### Example 3: Candidate Decisions

Four candidates are screened by the solvent-held-out full model. An in-domain candidate is labeled `test`. An unseen cation candidate is labeled `needs_data`. A candidate with high pressure, high temperature, unseen cation, and unseen anion is labeled `needs_data`. A low-performing in-domain candidate is labeled `reject`.

### Industrial Applicability

The disclosed system can be used in carbon capture research, solvent development, industrial gas treatment, petroleum and gas processing, direct air capture, flue gas capture, syngas treatment, laboratory screening, and process-development services. The method can reduce unsupported experimental recommendations and can route uncertain candidates to data generation before deployment.

## Abstract

A computer-implemented solvent screening system receives experimental CO2 capture records and candidate solvent records, trains a regression model using chemically held-out validation, calibrates prediction intervals, builds an applicability-domain profile, and scores candidate solvents using predicted performance, interval width, domain status, and operating constraints. For each candidate, the system outputs a prediction interval, confidence grade, domain status, reason codes, rank score, and decision label. Out-of-domain candidates can be withheld from positive recommendation and labeled as requiring additional data. The system can run locally and can generate an experimental queue for testing CO2 capture solvents.
