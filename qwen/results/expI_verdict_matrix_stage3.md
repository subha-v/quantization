# Exp I verdict matrix — Stage 3

- Anchors I0, I1, I9 always carry verdict `anchor`.
- Mechanism controls I4 / I5 / I12 are expected to LOSE to their TempWin counterpart — `borderline` is the success case for the temporal-locality hypothesis.
- I3 / I11 are judged vs F4 baseline at the same frame tier.
- Add-on variants (I6 / I7 / I8 / I13 / I14) and the hybrid I15 are judged vs their respective TempWin anchors / source.

| Condition | acc | rel_kv_mem | avg_kv_bits | verdict |
|---|---:|---:|---:|---|
| I0_BF16_128f | 0.615 | 2.000 | 16.000 | anchor |
| I1_F4_128f | 0.570 | 0.500 | 4.000 | anchor |
| I2_F9_128f | 0.605 | 0.594 | 4.750 | borderline |
| I3_TempWin2_128f | 0.560 | 0.500 | 4.000 | promote_n200 |
| I4_TextVisualSplit_128f | 0.570 | 0.500 | 4.000 | promote_n200 |
| I5_TokenBlock4_128f | 0.555 | 0.500 | 4.000 | promote_n200 |
| I6_TempWin4_128f | 0.560 | 0.500 | 4.000 | promote_n200 |
| I8_TempWin2_Outlier8_128f | 0.585 | 0.547 | 4.375 | promote_n200 |
| I9_F4_256f | 0.563 | 1.000 | 4.000 | anchor |
| I10_F9_256f | 0.541 | 1.188 | 4.750 | borderline |
| I11_TempWin4_256f | 0.557 | 1.000 | 4.000 | promote_n200 |
| I15_F9MidElseTempWin | 0.575 | 0.500 | 4.000 | borderline |
| I16_F9RandomMatched | 0.560 | 0.500 | 4.000 | control |

**paper_strong**: []
**promote_n200**: ['I3_TempWin2_128f', 'I4_TextVisualSplit_128f', 'I5_TokenBlock4_128f', 'I6_TempWin4_128f', 'I8_TempWin2_Outlier8_128f', 'I11_TempWin4_256f']
**borderline**:  ['I2_F9_128f', 'I10_F9_256f', 'I15_F9MidElseTempWin']
**kill**:        []
