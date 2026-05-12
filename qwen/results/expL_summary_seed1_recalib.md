# Exp K summary — seed=1 (n=200)

| Condition | acc | 95% CI | n | rel_kv_mem | avg_kv_bits | verdict |
|---|---:|---|---:|---:|---:|---|
| K0_BF16_128f | 0.615 | [0.550, 0.680] | 200 | 2.000 | 16.000 | anchor |
| K1_F4_128f | 0.570 | [0.505, 0.635] | 200 | 0.500 | 4.000 | anchor |
| K2_F9_BF16side_128f | 0.615 | [0.550, 0.680] | 200 | 0.594 | 4.750 | anchor |
| K3_F9_INT8side_128f | 0.600 | [0.530, 0.665] | 200 | 0.531 | 4.250 | borderline |
| K4_F8_BF16side_128f | 0.575 | [0.510, 0.640] | 200 | 0.547 | 4.375 | anchor |
| K5_Random8_BF16side_128f | 0.585 | [0.515, 0.655] | 200 | 0.547 | 4.375 | control_random |
| K6_Bal2pb_BF16side_128f | 0.575 | [0.510, 0.645] | 200 | 0.547 | 4.375 | matches_baseline |
| K7_Bal2pb_INT8side_128f | 0.575 | [0.510, 0.645] | 200 | 0.516 | 4.125 | borderline |
| K8_Bal1pb_BF16side_128f | 0.595 | [0.525, 0.660] | 200 | 0.523 | 4.188 | matches_K6 |
| K9_Bal3pb_BF16side_128f | 0.630 | [0.565, 0.695] | 200 | 0.570 | 4.562 | beats_K6 |
| K10_BalRandomPos_BF16side_128f | 0.565 | [0.495, 0.630] | 200 | 0.547 | 4.375 | control_ties_K6 |
| K11_Pivot8_BF16side_128f | 0.610 | [0.540, 0.675] | 200 | 0.547 | 4.375 | replicates_pivot_win |
