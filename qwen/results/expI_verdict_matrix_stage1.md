# Exp I verdict matrix — Stage 1

- Anchors I0, I1, I9 always carry verdict `anchor`.
- Mechanism controls I4 / I5 / I12 are expected to LOSE to their TempWin counterpart — `borderline` is the success case for the temporal-locality hypothesis.
- I3 / I11 are judged vs F4 baseline at the same frame tier.
- Add-on variants (I6 / I7 / I8 / I13 / I14) and the hybrid I15 are judged vs their respective TempWin anchors / source.

| Condition | acc | rel_kv_mem | avg_kv_bits | verdict |
|---|---:|---:|---:|---|
| I0_BF16_128f | 0.594 | 2.000 | 16.000 | anchor |
| I1_F4_128f | 0.641 | 0.500 | 4.000 | anchor |
| I2_F9_128f | 0.609 | 0.594 | 4.750 | borderline |
| I3_TempWin2_128f | 0.547 | 0.500 | 4.000 | kill |
| I4_TextVisualSplit_128f | 0.562 | 0.500 | 4.000 | promote_n200 |
| I5_TokenBlock4_128f | 0.578 | 0.500 | 4.000 | promote_paper_strong |
| I6_TempWin4_128f | 0.578 | 0.500 | 4.000 | promote_paper_strong |
| I7_TempWin2_VidKVV_128f | 0.516 | 0.500 | 4.000 | borderline |
| I8_TempWin2_Outlier8_128f | 0.562 | 0.547 | 4.375 | promote_n200 |
| I9_F4_256f | 0.526 | 1.000 | 4.000 | anchor |
| I10_F9_256f | 0.509 | 1.188 | 4.750 | borderline |
| I11_TempWin4_256f | 0.614 | 1.000 | 4.000 | promote_paper_strong |
| I12_TokenBlock6_256f | 0.544 | 1.000 | 4.000 | borderline |
| I13_TempWin4_Outlier8_256f | 0.561 | 1.094 | 4.375 | kill |
| I14_TempWin4_VidKVV_256f | 0.579 | 1.000 | 4.000 | borderline |
| I15_F9MidElseTempWin | 0.578 | 0.500 | 4.000 | borderline |
| I16_F9RandomMatched | 0.547 | 0.500 | 4.000 | control |

**paper_strong**: ['I5_TokenBlock4_128f', 'I6_TempWin4_128f', 'I11_TempWin4_256f']
**promote_n200**: ['I4_TextVisualSplit_128f', 'I5_TokenBlock4_128f', 'I6_TempWin4_128f', 'I8_TempWin2_Outlier8_128f', 'I11_TempWin4_256f']
**borderline**:  ['I2_F9_128f', 'I7_TempWin2_VidKVV_128f', 'I10_F9_256f', 'I12_TokenBlock6_256f', 'I14_TempWin4_VidKVV_256f', 'I15_F9MidElseTempWin']
**kill**:        ['I3_TempWin2_128f', 'I13_TempWin4_Outlier8_256f']
