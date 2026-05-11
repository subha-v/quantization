# Exp J paired McNemar — Stage 3

`(a, b)` = (alternative, baseline). χ² = `(b_only - a_only)² / (a_only + b_only)`. p ≈ from χ²₁.

| label | a | b | n | acc(a) | acc(b) | a_only | b_only | both | neither | χ² | p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tt_vs_generic | J4_Outlier8_TT_128f | J3_F8_128f | 200 | 0.695 | 0.695 | 6 | 6 | 133 | 55 | 0.000 | nan |
| tv_vs_generic | J5_Outlier8_TV_128f | J3_F8_128f | 200 | 0.690 | 0.695 | 7 | 8 | 131 | 54 | 0.067 | nan |
| tt_tv_vs_generic | J6_Outlier8_TT_TV_128f | J3_F8_128f | 200 | 0.690 | 0.695 | 6 | 7 | 132 | 55 | 0.077 | nan |
| balanced_vs_generic | J7_Outlier8_BAL_128f | J3_F8_128f | 200 | 0.725 | 0.695 | 7 | 1 | 138 | 54 | 4.500 | nan |
| pivot_vs_generic | J8_Outlier8_PIVOT_128f | J3_F8_128f | 200 | 0.700 | 0.695 | 7 | 6 | 133 | 54 | 0.077 | nan |
| tt_tv_vs_f9 | J6_Outlier8_TT_TV_128f | J2_F9_128f | 200 | 0.690 | 0.695 | 5 | 6 | 133 | 56 | 0.091 | nan |
| la_50pct_vs_f9 | J9_LA_TT_TV_50pct_128f | J2_F9_128f | 200 | 0.680 | 0.695 | 4 | 7 | 132 | 57 | 0.818 | nan |
| la_75pct_vs_f9 | J11_LA_TT_TV_75pct_128f | J2_F9_128f | 200 | 0.705 | 0.695 | 2 | 0 | 139 | 59 | 2.000 | nan |
| int8side_vs_bf16side | J12_F9_INT8side_128f | J2_F9_128f | 200 | 0.695 | 0.695 | 1 | 1 | 138 | 60 | 0.000 | nan |
| int6side_vs_bf16side | J13_F9_INT6side_128f | J2_F9_128f | — | — | — | — | — | — | — | — | — |
| tt_tv_int8side_vs_f9 | J14_TT_TV_INT8side_128f | J2_F9_128f | 200 | 0.680 | 0.695 | 3 | 6 | 133 | 58 | 1.000 | nan |
| random_vs_generic | J15_Outlier8_RANDOM_128f | J3_F8_128f | 200 | 0.650 | 0.695 | 10 | 19 | 120 | 51 | 2.793 | nan |
| balanced_beats_random | J7_Outlier8_BAL_128f | J15_Outlier8_RANDOM_128f | 200 | 0.725 | 0.650 | 22 | 7 | 123 | 48 | 7.759 | nan |
| pivot_beats_random | J8_Outlier8_PIVOT_128f | J15_Outlier8_RANDOM_128f | 200 | 0.700 | 0.650 | 19 | 9 | 121 | 51 | 3.571 | nan |
| tt_tv_beats_random | J6_Outlier8_TT_TV_128f | J15_Outlier8_RANDOM_128f | 200 | 0.690 | 0.650 | 17 | 9 | 121 | 53 | 2.462 | nan |
| random_LA_vs_f9 | J16_LA_RANDOM_50pct_128f | J2_F9_128f | 200 | 0.680 | 0.695 | 2 | 5 | 134 | 59 | 1.286 | nan |
| LA_TT_TV_beats_random_LA | J9_LA_TT_TV_50pct_128f | J16_LA_RANDOM_50pct_128f | 200 | 0.680 | 0.680 | 6 | 6 | 130 | 58 | 0.000 | nan |
| pivot_err_vs_pivot_energy | J17_Outlier8_PIVOT_ERR_128f | J8_Outlier8_PIVOT_128f | 200 | 0.690 | 0.700 | 3 | 5 | 135 | 57 | 0.500 | nan |
| pivot_err_vs_generic | J17_Outlier8_PIVOT_ERR_128f | J3_F8_128f | 200 | 0.690 | 0.695 | 6 | 7 | 132 | 55 | 0.077 | nan |
| pivot_err_vs_f9 | J17_Outlier8_PIVOT_ERR_128f | J2_F9_128f | 200 | 0.690 | 0.695 | 3 | 4 | 135 | 58 | 0.143 | nan |
| f9_reproduces_seed2 | J2_F9_128f | J1_F4_128f | 200 | 0.695 | 0.645 | 14 | 4 | 125 | 57 | 5.556 | nan |
| f8_reproduces_seed2 | J3_F8_128f | J1_F4_128f | 200 | 0.695 | 0.645 | 18 | 8 | 121 | 53 | 3.846 | nan |
