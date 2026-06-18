# Claims Draft

Status: attorney-review draft, not legal advice. Claims should be revised by patent counsel before filing.

## Method Claims

1. A computer-implemented method for screening candidate solvents for carbon dioxide capture, the method comprising:
   receiving, by one or more processors, training records comprising solvent identity information, molecular representation information, operating condition information, and measured carbon dioxide capture response values;
   partitioning the training records into at least a training subset and a held-out subset according to a chemical grouping selected from solvent identity, cation identity, anion identity, solvent family, molecular scaffold, or combinations thereof;
   training, using the training subset, one or more regression models to predict a carbon dioxide capture response from the molecular representation information and the operating condition information;
   calibrating a prediction interval using residuals computed from predictions for calibration records having chemical groupings held out from the training subset;
   generating an applicability-domain profile from the training subset, the applicability-domain profile comprising numeric training ranges and observed categorical chemistry values;
   receiving a candidate solvent record comprising candidate molecular representation information and candidate operating condition information;
   predicting, using a selected regression model, a candidate carbon dioxide capture response for the candidate solvent record;
   determining a candidate prediction interval from the calibrated prediction interval;
   determining an applicability-domain status for the candidate solvent record by comparing the candidate solvent record with the applicability-domain profile;
   computing a candidate rank score from at least the candidate carbon dioxide capture response, a width of the candidate prediction interval, and the applicability-domain status; and
   outputting a decision label for the candidate solvent record, wherein the decision label withholds a positive testing recommendation when the applicability-domain status is out of domain.

2. The method of claim 1, wherein the solvent identity information comprises a cation identifier and an anion identifier for an ionic liquid.

3. The method of claim 1, wherein the molecular representation information comprises one or more SMILES strings.

4. The method of claim 3, further comprising deriving descriptor values from the one or more SMILES strings.

5. The method of claim 4, wherein the descriptor values comprise at least one of atom-token counts, heteroatom counts, charge counts, ring counts, branch counts, bond counts, aromatic atom counts, molecular fingerprints, group-contribution descriptors, sigma-profile descriptors, quantum chemical descriptors, graph embeddings, or learned molecular embeddings.

6. The method of claim 1, wherein the operating condition information comprises temperature and pressure.

7. The method of claim 1, wherein the measured carbon dioxide capture response values comprise carbon dioxide solubility, carbon dioxide loading, carbon dioxide capacity, carbon dioxide selectivity, absorption rate, Henry-law constant, a viscosity-adjusted capture metric, or a logarithmic transform thereof.

8. The method of claim 1, wherein calibrating the prediction interval comprises computing absolute residuals on the calibration records and selecting a residual quantile corresponding to a desired miscoverage rate.

9. The method of claim 1, wherein the applicability-domain status is selected from in domain, edge of domain, and out of domain.

10. The method of claim 1, wherein determining the applicability-domain status comprises identifying an unseen cation identifier, an unseen anion identifier, or an operating condition outside a numeric training range.

11. The method of claim 1, wherein computing the candidate rank score comprises subtracting an uncertainty penalty based on the width of the candidate prediction interval.

12. The method of claim 1, wherein computing the candidate rank score comprises subtracting a domain penalty when the applicability-domain status is edge of domain or out of domain.

13. The method of claim 1, wherein the decision label is selected from test, watch, reject, and needs data.

14. The method of claim 13, wherein the needs data decision label is assigned when the candidate solvent record includes an unseen chemistry value or an operating condition outside a training range.

15. The method of claim 1, further comprising outputting reason codes identifying one or more causes of the decision label.

16. The method of claim 15, wherein the reason codes identify at least one of unseen cation, unseen anion, temperature outside training range, pressure outside training range, descriptor outside training range, interval width exceeding a threshold, or candidate within training domain.

17. The method of claim 1, further comprising generating an experimental test queue comprising a candidate solvent identifier, operating condition values, the candidate prediction interval, the applicability-domain status, the candidate rank score, and the decision label.

18. The method of claim 17, further comprising transmitting the experimental test queue to a laboratory information management system or a capture test apparatus.

19. The method of claim 1, wherein the one or more processors are included in a local computer that executes the method without transmitting the candidate molecular representation information to a cloud service.

20. The method of claim 1, wherein the candidate solvent comprises an ionic liquid, deep eutectic solvent, amine-containing solvent, solvent blend, or physical solvent.

## System Claims

21. A system for screening carbon dioxide capture solvents, comprising:
   one or more processors; and
   one or more non-transitory computer-readable storage media storing instructions that, when executed by the one or more processors, cause the system to perform the method of any of claims 1 to 20.

22. The system of claim 21, further comprising a local data store containing public experimental solvent records and private experimental solvent records.

23. The system of claim 21, wherein the system is configured to display, for each candidate solvent record, a prediction interval, confidence grade, domain status, rank score, decision label, and reason code.

## Computer-Readable Medium Claims

24. One or more non-transitory computer-readable media storing instructions that, when executed by one or more processors, cause the one or more processors to perform the method of any of claims 1 to 20.

## Experimental Queue Claims

25. A computer-implemented method for generating experimental tests for carbon dioxide capture solvent candidates, comprising:
   receiving ranked candidate solvent records generated by the method of claim 1;
   excluding from a positive testing queue any candidate solvent record assigned an out-of-domain status;
   assigning an experimental priority to each remaining candidate solvent record based on the candidate rank score and the candidate prediction interval; and
   generating a test protocol specifying solvent identity, temperature, pressure, and priority for at least one candidate solvent.

## Claim Notes For Counsel

- Claim 1 is intentionally process-focused and tied to chemical screening.
- Consider whether claim 18 should stay in the first filing. It adds practical-application weight but should be supported by examples.
- Consider a narrower independent claim focused on ionic liquids and cation/anion OOD gating.
- Consider a separate independent claim focused on local/offline confidentiality if commercial value supports it.
- Do not claim conformal prediction, ML, descriptors, or CO2 solubility prediction broadly.
