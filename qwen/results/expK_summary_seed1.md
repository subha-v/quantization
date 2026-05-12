# Exp K summary — seed=1 (n=200)

| Condition | acc | 95% CI | n | rel_kv_mem | avg_kv_bits | verdict |
|---|---:|---|---:|---:|---:|---|
| K0_BF16_128f | 0.615 | [0.550, 0.680] | 200 | 2.000 | 16.000 | anchor |
| K1_F4_128f | 0.570 | [0.505, 0.635] | 200 | 0.500 | 4.000 | anchor |
| K2_F9_BF16side_128f | 0.595 | [0.525, 0.660] | 200 | 0.594 | 4.750 | anchor |
| K3_F9_INT8side_128f | 0.605 | [0.540, 0.670] | 200 | 0.531 | 4.250 | pareto_winner |
| K4_F8_BF16side_128f | 0.570 | [0.505, 0.635] | 200 | 0.547 | 4.375 | anchor |
| K5_Random8_BF16side_128f | 0.590 | [0.520, 0.655] | 200 | 0.547 | 4.375 | control_random |
| K6_Bal2pb_BF16side_128f | 0.560 | [0.495, 0.630] | 200 | 0.547 | 4.375 | matches_baseline |
| K7_Bal2pb_INT8side_128f | 0.580 | [0.515, 0.645] | 200 | 0.516 | 4.125 | borderline |
| K8_Bal1pb_BF16side_128f | 0.585 | [0.520, 0.650] | 200 | 0.523 | 4.188 | matches_K6 |
| K9_Bal3pb_BF16side_128f | 0.590 | [0.525, 0.655] | 200 | 0.570 | 4.562 | matches_K6 |
| K10_BalRandomPos_BF16side_128f | 0.570 | [0.500, 0.640] | 200 | 0.547 | 4.375 | control_ties_K6 |
| K11_Pivot8_BF16side_128f | 0.600 | [0.535, 0.665] | 200 | 0.547 | 4.375 | replicates_pivot_win |
