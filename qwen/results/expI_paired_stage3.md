# Exp I paired McNemar — Stage 3

`(a, b)` = (alternative, baseline). χ² = `(b_only - a_only)² / (a_only + b_only)`. p ≈ from χ²₁.

| label | a | b | n_paired | acc(a) | acc(b) | a_only | b_only | both | neither | χ² | p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tempwin_vs_modality_split_only_128f | I3_TempWin2_128f | I4_TextVisualSplit_128f | 200 | 0.560 | 0.570 | 14 | 16 | 98 | 72 | 0.133 | nan |
| modality_aware_vs_blind_128f | I3_TempWin2_128f | I5_TokenBlock4_128f | 200 | 0.560 | 0.555 | 16 | 15 | 96 | 73 | 0.032 | nan |
| windowcount_2_vs_4_128f | I3_TempWin2_128f | I6_TempWin4_128f | 200 | 0.560 | 0.560 | 10 | 10 | 102 | 78 | 0.000 | nan |
| vidkv_v_addition_128f | I3_TempWin2_128f | I7_TempWin2_VidKVV_128f | — | — | — | — | — | — | — | — | — |
| outlier8_addition_128f | I3_TempWin2_128f | I8_TempWin2_Outlier8_128f | 200 | 0.560 | 0.585 | 9 | 14 | 103 | 74 | 1.087 | nan |
| tempwin2_vs_f4_128f | I3_TempWin2_128f | I1_F4_128f | 200 | 0.560 | 0.570 | 18 | 20 | 94 | 68 | 0.105 | nan |
| tempwin2_vs_f9_128f | I3_TempWin2_128f | I2_F9_128f | 200 | 0.560 | 0.605 | 9 | 18 | 103 | 70 | 3.000 | nan |
| modality_aware_vs_blind_256f | I11_TempWin4_256f | I12_TokenBlock6_256f | — | — | — | — | — | — | — | — | — |
| outlier8_addition_256f | I11_TempWin4_256f | I13_TempWin4_Outlier8_256f | — | — | — | — | — | — | — | — | — |
| vidkv_v_addition_256f | I11_TempWin4_256f | I14_TempWin4_VidKVV_256f | — | — | — | — | — | — | — | — | — |
| tempwin4_vs_f4_256f | I11_TempWin4_256f | I9_F4_256f | 183 | 0.557 | 0.563 | 19 | 20 | 83 | 61 | 0.026 | nan |
| tempwin4_vs_f9_256f | I11_TempWin4_256f | I10_F9_256f | 183 | 0.557 | 0.541 | 12 | 9 | 90 | 72 | 0.429 | nan |
| duration_vs_random_hybrid | I15_F9MidElseTempWin | I16_F9RandomMatched | 200 | 0.575 | 0.560 | 7 | 4 | 108 | 81 | 0.818 | nan |
| hybrid_vs_tempwin_only | I15_F9MidElseTempWin | I3_TempWin2_128f | 200 | 0.575 | 0.560 | 4 | 1 | 111 | 84 | 1.800 | nan |
| hybrid_vs_f9_only | I15_F9MidElseTempWin | I2_F9_128f | 200 | 0.575 | 0.605 | 8 | 14 | 107 | 71 | 1.636 | nan |
