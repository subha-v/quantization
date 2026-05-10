# Exp J paired McNemar — Stage 1

`(a, b)` = (alternative, baseline). χ² = `(b_only - a_only)² / (a_only + b_only)`. p ≈ from χ²₁.

| label | a | b | n | acc(a) | acc(b) | a_only | b_only | both | neither | χ² | p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tt_vs_generic | J4_Outlier8_TT_128f | J3_F8_128f | 64 | 0.703 | 0.719 | 1 | 2 | 44 | 17 | 0.333 | nan |
| tv_vs_generic | J5_Outlier8_TV_128f | J3_F8_128f | 64 | 0.719 | 0.719 | 2 | 2 | 44 | 16 | 0.000 | nan |
| tt_tv_vs_generic | J6_Outlier8_TT_TV_128f | J3_F8_128f | 64 | 0.719 | 0.719 | 2 | 2 | 44 | 16 | 0.000 | nan |
| balanced_vs_generic | J7_Outlier8_BAL_128f | J3_F8_128f | 64 | 0.734 | 0.719 | 1 | 0 | 46 | 17 | 1.000 | nan |
| pivot_vs_generic | J8_Outlier8_PIVOT_128f | J3_F8_128f | 64 | 0.734 | 0.719 | 2 | 1 | 45 | 16 | 0.333 | nan |
| tt_tv_vs_f9 | J6_Outlier8_TT_TV_128f | J2_F9_128f | 64 | 0.719 | 0.703 | 3 | 2 | 43 | 16 | 0.200 | nan |
| la_50pct_vs_f9 | J9_LA_TT_TV_50pct_128f | J2_F9_128f | 64 | 0.734 | 0.703 | 2 | 0 | 45 | 17 | 2.000 | nan |
| la_75pct_vs_f9 | J11_LA_TT_TV_75pct_128f | J2_F9_128f | 64 | 0.719 | 0.703 | 1 | 0 | 45 | 18 | 1.000 | nan |
| int8side_vs_bf16side | J12_F9_INT8side_128f | J2_F9_128f | 64 | 0.703 | 0.703 | 0 | 0 | 45 | 19 | nan | nan |
| int6side_vs_bf16side | J13_F9_INT6side_128f | J2_F9_128f | 64 | 0.531 | 0.703 | 3 | 14 | 31 | 16 | 7.118 | nan |
| tt_tv_int8side_vs_f9 | J14_TT_TV_INT8side_128f | J2_F9_128f | 64 | 0.719 | 0.703 | 1 | 0 | 45 | 18 | 1.000 | nan |
| f9_reproduces_seed2 | J2_F9_128f | J1_F4_128f | 64 | 0.703 | 0.688 | 4 | 3 | 41 | 16 | 0.143 | nan |
| f8_reproduces_seed2 | J3_F8_128f | J1_F4_128f | 64 | 0.719 | 0.688 | 6 | 4 | 40 | 14 | 0.400 | nan |
