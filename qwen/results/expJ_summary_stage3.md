# Exp J summary — Stage 3

_n items per condition (max) = 200; 95% CI bootstrap n_boot=2000_

| Condition | acc | 95% CI | n | rel_kv_mem | avg_kv_bits | verdict |
|---|---:|---|---:|---:|---:|---|
| J0_BF16_128f | 0.705 | [0.640, 0.765] | 200 | 2.000 | 16.000 | anchor |
| J1_F4_128f | 0.645 | [0.580, 0.710] | 200 | 0.500 | 4.000 | anchor |
| J2_F9_128f | 0.695 | [0.630, 0.760] | 200 | 0.594 | 4.750 | anchor |
| J3_F8_128f | 0.695 | [0.630, 0.755] | 200 | 0.547 | 4.375 | anchor |
| J4_Outlier8_TT_128f | 0.695 | [0.630, 0.755] | 200 | 0.547 | 4.375 | pareto_winner |
| J5_Outlier8_TV_128f | 0.690 | [0.625, 0.750] | 200 | 0.547 | 4.375 | pareto_winner |
| J6_Outlier8_TT_TV_128f | 0.690 | [0.625, 0.750] | 200 | 0.547 | 4.375 | pareto_winner |
| J7_Outlier8_BAL_128f | 0.725 | [0.660, 0.785] | 200 | 0.547 | 4.375 | paper_strong |
| J8_Outlier8_PIVOT_128f | 0.700 | [0.635, 0.760] | 200 | 0.547 | 4.375 | pareto_winner |
| J9_LA_TT_TV_50pct_128f | 0.680 | [0.615, 0.745] | 200 | 0.547 | 4.375 | borderline |
| J10_LA_ALL_50pct_128f | 0.675 | [0.610, 0.735] | 200 | 0.547 | 4.375 | borderline |
| J11_LA_TT_TV_75pct_128f | 0.705 | [0.640, 0.765] | 200 | 0.570 | 4.562 | pareto_winner |
| J12_F9_INT8side_128f | 0.695 | [0.630, 0.755] | 200 | 0.531 | 4.250 | pareto_winner |
| J14_TT_TV_INT8side_128f | 0.680 | [0.615, 0.745] | 200 | 0.531 | 4.250 | borderline |
| J15_Outlier8_RANDOM_128f | 0.650 | [0.585, 0.715] | 200 | 0.547 | 4.375 | control_random |
| J16_LA_RANDOM_50pct_128f | 0.680 | [0.615, 0.745] | 200 | 0.547 | 4.375 | control_random |
| J17_Outlier8_PIVOT_ERR_128f | 0.690 | [0.625, 0.750] | 200 | 0.547 | 4.375 | pareto_winner |
