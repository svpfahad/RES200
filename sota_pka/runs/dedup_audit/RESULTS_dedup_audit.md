# E9 — Train/external deduplication audit (E6 validity)

Both sides canonicalized in the neutralized representation E6 scored in.

## External vs OP2-train overlap (task-matched)

| external set | task | n (unique) | overlap w/ train (canonical) | % | overlap (connectivity) | overlap w/ OP2-test |
|---|---|---:|---:|---:|---:|---:|
| novartis_acidic | acidic | 112 | 0 | 0.0% | 0 | 0 |
| AvLiLuMoVe_123_acidic | acidic | 26 | 0 | 0.0% | 3 | 0 |
| SAMPL7_acidic | acidic | 20 | 0 | 0.0% | 0 | 0 |
| novartis_basic | basic | 168 | 0 | 0.0% | 0 | 0 |
| AvLiLuMoVe_123_basic | basic | 97 | 2 | 2.1% | 10 | 0 |

## OP2 train/test leakage anchor (should be ~0)

| task | n train | n test | train∩test (canonical) |
|---|---:|---:|---:|
| acidic | 2320 | 774 | 1 |
| basic | 2527 | 843 | 0 |

**Verdict:** total exact-structure external↔train overlap = **2** molecule(s) across all external sets. See `overlap_detail.json`; exclude overlapping molecules and re-report E6.
