# Exp K paired McNemar — seed=1

`(a, b)` = (alternative, baseline). χ² = `(b_only - a_only)² / (a_only + b_only)`.

| label | a | b | n | acc(a) | acc(b) | a_only | b_only | both | neither | χ² | p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced_vs_generic | K6_Bal2pb_BF16side_128f | K4_F8_BF16side_128f | 200 | 0.575 | 0.575 | 10 | 10 | 105 | 75 | 0.000 | nan |
| balanced_vs_random | K6_Bal2pb_BF16side_128f | K5_Random8_BF16side_128f | 200 | 0.575 | 0.585 | 14 | 16 | 101 | 69 | 0.133 | nan |
| crossmodal_vs_balanced_random | K6_Bal2pb_BF16side_128f | K10_BalRandomPos_BF16side_128f | 200 | 0.575 | 0.565 | 18 | 16 | 97 | 69 | 0.118 | nan |
| balanced_int8_vs_bf16 | K7_Bal2pb_INT8side_128f | K6_Bal2pb_BF16side_128f | 200 | 0.575 | 0.575 | 9 | 9 | 106 | 76 | 0.000 | nan |
| f9_int8_vs_bf16 | K3_F9_INT8side_128f | K2_F9_BF16side_128f | 200 | 0.600 | 0.615 | 3 | 6 | 117 | 74 | 1.000 | nan |
| top1pb_vs_top2pb | K8_Bal1pb_BF16side_128f | K6_Bal2pb_BF16side_128f | 200 | 0.595 | 0.575 | 13 | 9 | 106 | 72 | 0.727 | nan |
| top3pb_vs_top2pb | K9_Bal3pb_BF16side_128f | K6_Bal2pb_BF16side_128f | 200 | 0.630 | 0.575 | 13 | 2 | 113 | 72 | 8.067 | nan |
| K7_vs_F9_pareto | K7_Bal2pb_INT8side_128f | K2_F9_BF16side_128f | 200 | 0.575 | 0.615 | 10 | 18 | 105 | 67 | 2.286 | nan |
| K6_vs_F9 | K6_Bal2pb_BF16side_128f | K2_F9_BF16side_128f | 200 | 0.575 | 0.615 | 6 | 14 | 109 | 71 | 3.200 | nan |
| f9_reproduces | K2_F9_BF16side_128f | K1_F4_128f | 200 | 0.615 | 0.570 | 19 | 10 | 104 | 67 | 2.793 | nan |
| pivot_vs_generic | K11_Pivot8_BF16side_128f | K4_F8_BF16side_128f | 200 | 0.610 | 0.575 | 13 | 6 | 109 | 72 | 2.579 | nan |
| pivot_vs_random | K11_Pivot8_BF16side_128f | K5_Random8_BF16side_128f | 200 | 0.610 | 0.585 | 16 | 11 | 106 | 67 | 0.926 | nan |
