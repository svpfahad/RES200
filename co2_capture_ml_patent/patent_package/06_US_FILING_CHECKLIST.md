# U.S. Filing Checklist

Status: attorney-review draft, not legal advice.

## Recommended First Filing

File a detailed U.S. provisional before any public disclosure. The provisional should include the full technical disclosure, examples, flowcharts, code/pseudocode, evidence tables, and candidate-ranking outputs. A provisional is not examined, does not require formal claims, oath/declaration, or IDS, and lasts 12 months. A nonprovisional must be filed within the 12-month pendency period to claim benefit.

## U.S. Provisional Package

Minimum package for counsel:

- title;
- inventor names and contact information;
- applicant/owner information;
- detailed specification;
- drawing packet;
- experimental evidence summary;
- source code archive or appendix if counsel wants it;
- public-data source list;
- prior-art list;
- disclosure history;
- ownership/assignment facts;
- filing fee/entity status information.

## U.S. Nonprovisional Package

USPTO nonprovisional utility applications require a specification including description and claim(s), drawings when necessary, oath/declaration, and filing/search/examination fees. The specification should include at least one claim and an abstract. The abstract should be a single paragraph and no longer than 150 words.

Package:

- transmittal form or letter;
- fees;
- application data sheet;
- specification;
- claims;
- abstract;
- drawings;
- inventor oath/declaration;
- assignment documents if applicable;
- IDS materials;
- sequence listing only if applicable.

## Section 101 / Eligibility Notes

The application should avoid presenting the invention as only a mathematical method, algorithm, ranking scheme, or generic computer implementation. The practical application should be stated as a technical chemical screening process that controls whether candidates are recommended, rejected, or routed to experimental data generation.

Strong eligibility facts to include:

- candidates are chemical solvent records for CO2 capture;
- the system reduces unsupported experimental recommendations;
- output can generate a laboratory test queue or protocol;
- OOD candidates are withheld from recommendation;
- workflow is tied to solvent/cation/anion/process-domain constraints.

## Section 112 / Enablement Notes

Include enough detail for a skilled person to implement:

- data schema;
- target variables;
- feature generation;
- split rules;
- model examples;
- calibration method;
- domain profile method;
- rank score;
- decision thresholds;
- output schema;
- examples with actual results.

Also disclose the best mode known to the inventors at filing, including the preferred data handling, feature generation, split strategy, model configuration, calibration strategy, domain checks, scoring rule, and decision thresholds.

If any broad chemical genus or solvent class is claimed, counsel should decide whether the present evidence supports that breadth. Broad composition claims usually need representative species, structural features, functional correlations, and experimental support. The current strongest support is for a screening workflow, not for ownership of a broad solvent genus.

## IDS / Duty Of Disclosure Notes

Counsel should review the prior-art list and decide what to submit in an IDS. U.S. rules impose a duty of candor and good faith, including disclosure of known information material to patentability during prosecution. IDS timing and content are procedural and should be handled by counsel.

Build an IDS packet for counsel from:

- papers and patents listed in `05_PRIOR_ART_AND_IDS_CANDIDATES.md`;
- public datasets and code used in the evidence suite;
- inventor disclosures, posters, preprints, demos, or public repositories;
- foreign search reports if any later counterpart application is filed;
- anything inconsistent with an argument of novelty, non-obviousness, enablement, or inventorship.

Do not file an IDS in a provisional application. Provide the prior-art packet to counsel so it can be handled in the later nonprovisional or PCT/national phase strategy.

## Inventorship And AI-Assisted Work

Name only human inventors who contributed to conception of the claimed invention. AI tools may assist drafting, coding, or experimentation, but counsel should identify the human contributors who conceived the technical screening workflow, uncertainty-gating logic, chemical-domain gating, scoring/decision rules, and any experimental embodiments.

Keep a dated record of:

- who conceived each feature;
- when the feature was first implemented;
- who selected the technical problem and claim direction;
- who contributed data, experimental design, or candidate chemistry;
- whether any employer, university, sponsor, or contractor has rights.

## Public Disclosure Risk

The U.S. has limited inventor-originated grace-period protections, but foreign rights can be lost by public disclosure before filing. Treat the draft specification, claims, figures, evidence tables, candidate lists, and code as confidential until counsel confirms the filing plan.

## Filing Questions For Counsel

1. Provisional now, then PCT/nonprovisional within 12 months?
2. Should claims include LIMS/test apparatus integration?
3. Should code be included as appendix or kept as supporting material only?
4. Should candidate solvents be kept as trade secret until wet-lab validation?
5. Should we file before sending any paper, poster, GitHub repo, investor deck, or email with enabling details?
6. Should a PCT application be filed within 12 months of a U.S. provisional to preserve Saudi and other foreign options?
