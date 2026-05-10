# Exp J verdict matrix — Stage 1

- Anchors J0/J1 always carry verdict `anchor`.
- Cross-modal selection (J4–J8): judged vs J3 (generic top-8).
- Layer-adaptive (J9–J11) and sidecode (J12–J14): judged vs J2 F9.
- `pareto_winner`: matches J2 F9 within 1pp at strictly fewer KV bits.

| Condition | acc | rel_kv_mem | avg_kv_bits | verdict |
|---|---:|---:|---:|---|
| J0_BF16_128f | 0.703 | 2.000 | 16.000 | anchor |
| J1_F4_128f | 0.688 | 0.500 | 4.000 | anchor |
| J2_F9_128f | 0.703 | 0.594 | 4.750 | anchor |
| J3_F8_128f | 0.719 | 0.547 | 4.375 | anchor |
| J4_Outlier8_TT_128f | 0.703 | 0.547 | 4.375 | pareto_winner |
| J5_Outlier8_TV_128f | 0.719 | 0.547 | 4.375 | pareto_winner |
| J6_Outlier8_TT_TV_128f | 0.719 | 0.547 | 4.375 | pareto_winner |
| J7_Outlier8_BAL_128f | 0.734 | 0.547 | 4.375 | pareto_winner |
| J8_Outlier8_PIVOT_128f | 0.734 | 0.547 | 4.375 | pareto_winner |
| J9_LA_TT_TV_50pct_128f | 0.734 | 0.547 | 4.375 | pareto_winner |
| J10_LA_ALL_50pct_128f | 0.703 | 0.547 | 4.375 | pareto_winner |
| J11_LA_TT_TV_75pct_128f | 0.719 | 0.570 | 4.562 | pareto_winner |
| J12_F9_INT8side_128f | 0.703 | 0.531 | 4.250 | pareto_winner |
| J13_F9_INT6side_128f | 0.531 | 0.516 | 4.125 | kill |
| J14_TT_TV_INT8side_128f | 0.719 | 0.531 | 4.250 | pareto_winner |

**paper_strong / pareto_winner**: ['J4_Outlier8_TT_128f', 'J5_Outlier8_TV_128f', 'J6_Outlier8_TT_TV_128f', 'J7_Outlier8_BAL_128f', 'J8_Outlier8_PIVOT_128f', 'J9_LA_TT_TV_50pct_128f', 'J10_LA_ALL_50pct_128f', 'J11_LA_TT_TV_75pct_128f', 'J12_F9_INT8side_128f', 'J14_TT_TV_INT8side_128f']
**promote_n200 (incl. above)**: ['J4_Outlier8_TT_128f', 'J5_Outlier8_TV_128f', 'J6_Outlier8_TT_TV_128f', 'J7_Outlier8_BAL_128f', 'J8_Outlier8_PIVOT_128f', 'J9_LA_TT_TV_50pct_128f', 'J10_LA_ALL_50pct_128f', 'J11_LA_TT_TV_75pct_128f', 'J12_F9_INT8side_128f', 'J14_TT_TV_INT8side_128f']
**borderline**:  []
**kill**:        ['J13_F9_INT6side_128f']
