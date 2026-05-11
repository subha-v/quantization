# Exp K paired McNemar — seed=1

`(a, b)` = (alternative, baseline). χ² = `(b_only - a_only)² / (a_only + b_only)`.

| label | a | b | n | acc(a) | acc(b) | a_only | b_only | both | neither | χ² | p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced_vs_generic | K6_Bal2pb_BF16side_128f | K4_F8_BF16side_128f | 200 | 0.560 | 0.570 | 6 | 8 | 106 | 80 | 0.286 | nan |
| balanced_vs_random | K6_Bal2pb_BF16side_128f | K5_Random8_BF16side_128f | 200 | 0.560 | 0.590 | 11 | 17 | 101 | 71 | 1.286 | nan |
| crossmodal_vs_balanced_random | K6_Bal2pb_BF16side_128f | K10_BalRandomPos_BF16side_128f | 200 | 0.560 | 0.570 | 14 | 16 | 98 | 72 | 0.133 | nan |
| balanced_int8_vs_bf16 | K7_Bal2pb_INT8side_128f | K6_Bal2pb_BF16side_128f | 200 | 0.580 | 0.560 | 10 | 6 | 106 | 78 | 1.000 | nan |
| f9_int8_vs_bf16 | K3_F9_INT8side_128f | K2_F9_BF16side_128f | 200 | 0.605 | 0.595 | 7 | 5 | 114 | 74 | 0.333 | nan |
| top1pb_vs_top2pb | K8_Bal1pb_BF16side_128f | K6_Bal2pb_BF16side_128f | 200 | 0.585 | 0.560 | 11 | 6 | 106 | 77 | 1.471 | nan |
| top3pb_vs_top2pb | K9_Bal3pb_BF16side_128f | K6_Bal2pb_BF16side_128f | 200 | 0.590 | 0.560 | 10 | 4 | 108 | 78 | 2.571 | nan |
| K7_vs_F9_pareto | K7_Bal2pb_INT8side_128f | K2_F9_BF16side_128f | 200 | 0.580 | 0.595 | 10 | 13 | 106 | 71 | 0.391 | nan |
| K6_vs_F9 | K6_Bal2pb_BF16side_128f | K2_F9_BF16side_128f | 200 | 0.560 | 0.595 | 8 | 15 | 104 | 73 | 2.130 | nan |
| f9_reproduces | K2_F9_BF16side_128f | K1_F4_128f | 200 | 0.595 | 0.570 | 15 | 10 | 104 | 71 | 1.000 | nan |
| pivot_vs_generic | K11_Pivot8_BF16side_128f | K4_F8_BF16side_128f | 200 | 0.600 | 0.570 | 12 | 6 | 108 | 74 | 2.000 | nan |
| pivot_vs_random | K11_Pivot8_BF16side_128f | K5_Random8_BF16side_128f | 200 | 0.600 | 0.590 | 15 | 13 | 105 | 67 | 0.143 | nan |
