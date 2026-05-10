# Exp I paired McNemar — Stage 1

`(a, b)` = (alternative, baseline). χ² = `(b_only - a_only)² / (a_only + b_only)`. p ≈ from χ²₁.

| label | a | b | n_paired | acc(a) | acc(b) | a_only | b_only | both | neither | χ² | p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tempwin_vs_modality_split_only_128f | I3_TempWin2_128f | I4_TextVisualSplit_128f | 64 | 0.547 | 0.562 | 4 | 5 | 31 | 24 | 0.111 | nan |
| modality_aware_vs_blind_128f | I3_TempWin2_128f | I5_TokenBlock4_128f | 64 | 0.547 | 0.578 | 5 | 7 | 30 | 22 | 0.333 | nan |
| windowcount_2_vs_4_128f | I3_TempWin2_128f | I6_TempWin4_128f | 64 | 0.547 | 0.578 | 3 | 5 | 32 | 24 | 0.500 | nan |
| vidkv_v_addition_128f | I3_TempWin2_128f | I7_TempWin2_VidKVV_128f | 64 | 0.547 | 0.516 | 3 | 1 | 32 | 28 | 1.000 | nan |
| outlier8_addition_128f | I3_TempWin2_128f | I8_TempWin2_Outlier8_128f | 64 | 0.547 | 0.562 | 3 | 4 | 32 | 25 | 0.143 | nan |
| tempwin2_vs_f4_128f | I3_TempWin2_128f | I1_F4_128f | 64 | 0.547 | 0.641 | 3 | 9 | 32 | 20 | 3.000 | nan |
| tempwin2_vs_f9_128f | I3_TempWin2_128f | I2_F9_128f | 64 | 0.547 | 0.609 | 2 | 6 | 33 | 23 | 2.000 | nan |
| modality_aware_vs_blind_256f | I11_TempWin4_256f | I12_TokenBlock6_256f | 57 | 0.614 | 0.544 | 8 | 4 | 27 | 18 | 1.333 | nan |
| outlier8_addition_256f | I11_TempWin4_256f | I13_TempWin4_Outlier8_256f | 57 | 0.614 | 0.561 | 4 | 1 | 31 | 21 | 1.800 | nan |
| vidkv_v_addition_256f | I11_TempWin4_256f | I14_TempWin4_VidKVV_256f | 57 | 0.614 | 0.579 | 4 | 2 | 31 | 20 | 0.667 | nan |
| tempwin4_vs_f4_256f | I11_TempWin4_256f | I9_F4_256f | 57 | 0.614 | 0.526 | 10 | 5 | 25 | 17 | 1.667 | nan |
| tempwin4_vs_f9_256f | I11_TempWin4_256f | I10_F9_256f | 57 | 0.614 | 0.509 | 7 | 1 | 28 | 21 | 4.500 | nan |
| duration_vs_random_hybrid | I15_F9MidElseTempWin | I16_F9RandomMatched | 64 | 0.578 | 0.547 | 3 | 1 | 34 | 26 | 1.000 | nan |
| hybrid_vs_tempwin_only | I15_F9MidElseTempWin | I3_TempWin2_128f | 64 | 0.578 | 0.547 | 2 | 0 | 35 | 27 | 2.000 | nan |
| hybrid_vs_f9_only | I15_F9MidElseTempWin | I2_F9_128f | 64 | 0.578 | 0.609 | 2 | 4 | 35 | 23 | 0.667 | nan |
