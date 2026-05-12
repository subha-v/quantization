# Exp J summary — Stage 1

_n items per condition (max) = 64; 95% CI bootstrap n_boot=2000_

| Condition | acc | 95% CI | n | rel_kv_mem | avg_kv_bits | verdict |
|---|---:|---|---:|---:|---:|---|
| J0_BF16_128f | 0.703 | [0.594, 0.812] | 64 | 2.000 | 16.000 | anchor |
| J1_F4_128f | 0.688 | [0.578, 0.797] | 64 | 0.500 | 4.000 | anchor |
| J2_F9_128f | 0.703 | [0.594, 0.812] | 64 | 0.594 | 4.750 | anchor |
| J3_F8_128f | 0.719 | [0.609, 0.828] | 64 | 0.547 | 4.375 | anchor |
| J4_Outlier8_TT_128f | 0.703 | [0.594, 0.812] | 64 | 0.547 | 4.375 | pareto_winner |
| J5_Outlier8_TV_128f | 0.719 | [0.609, 0.828] | 64 | 0.547 | 4.375 | pareto_winner |
| J6_Outlier8_TT_TV_128f | 0.719 | [0.609, 0.828] | 64 | 0.547 | 4.375 | pareto_winner |
| J7_Outlier8_BAL_128f | 0.734 | [0.625, 0.844] | 64 | 0.547 | 4.375 | pareto_winner |
| J8_Outlier8_PIVOT_128f | 0.734 | [0.625, 0.844] | 64 | 0.547 | 4.375 | pareto_winner |
| J9_LA_TT_TV_50pct_128f | 0.734 | [0.625, 0.844] | 64 | 0.547 | 4.375 | pareto_winner |
| J10_LA_ALL_50pct_128f | 0.703 | [0.594, 0.812] | 64 | 0.547 | 4.375 | pareto_winner |
| J11_LA_TT_TV_75pct_128f | 0.719 | [0.609, 0.828] | 64 | 0.570 | 4.562 | pareto_winner |
| J12_F9_INT8side_128f | 0.703 | [0.594, 0.812] | 64 | 0.531 | 4.250 | pareto_winner |
| J13_F9_INT6side_128f | 0.531 | [0.422, 0.656] | 64 | 0.516 | 4.125 | kill |
| J14_TT_TV_INT8side_128f | 0.719 | [0.609, 0.828] | 64 | 0.531 | 4.250 | pareto_winner |
