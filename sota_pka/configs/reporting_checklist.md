# Reporting Checklist

- Record dataset file hashes before and after processing.
- Report row counts by source, task, split, and deduplication stage.
- Prove no train/test/external overlap by InChIKey or canonical SMILES.
- Recompute all metrics from saved prediction CSVs.
- Run at least five seeds for final models.
- Include Y-randomization control metrics.
- Separate acidic and basic pKa model results.
- Mark any QupKake/Uni-pKa/SAT4pKa result as external comparator unless retrained under the same split protocol.
- Do not use the phrase "state of the art" unless external benchmark results support it.
