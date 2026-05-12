# Exp P paired McNemar — 2026-05-12 11:26:11

| pair | description | n_paired | A_only | B_only | χ² | p | favored |
|---|---|---|---|---|---|---|---|
| P1 vs P0 | F4 same-benchmark anchor: F4 vs BF16 dense | 74 | 29 | 45 | 3.04 | 0.0812 | B |
| P2 vs P0 | F9 same-benchmark anchor: F9 vs BF16 dense | 16 | 7 | 9 | 0.06 | 0.8026 | B |
| P3 vs P4 | Quest vs Random at top-25% sparse | 10 | 4 | 6 | 0.10 | 0.7518 | B |
| P5 vs P3 | Oracle (budget-matched) headroom over Quest | 7 | 3 | 4 | 0.00 | 1.0000 | B |
| P5_only vs P3 | Oracle (needle-only) vs Quest at top-25% — does needle alone suffice? | 6 | 3 | 3 | 0.17 | 0.6831 | tie |
| P6 vs P2 | FormatBook (Quest top-50%) vs dense F9 | 19 | 9 | 10 | 0.00 | 1.0000 | B |
| P6 vs P6R | FormatBook Quest vs Random — does Quest selection matter? | 16 | 11 | 5 | 1.56 | 0.2113 | A |
| P6O vs P6 | FormatBook Oracle headroom over Quest | 12 | 6 | 6 | 0.08 | 0.7728 | tie |
| P6 vs P1 | FormatBook vs dense F4 | 79 | 46 | 33 | 1.82 | 0.1770 | A |
| P2b vs P2 | J12 (INT8 sidecode) vs F9 dense — INT8 confound check | 24 | 9 | 15 | 1.04 | 0.3074 | B |
| P3b vs P4b | Quest vs Random at top-50% (stretch) | 5 | 2 | 3 | 0.00 | 1.0000 | B |
