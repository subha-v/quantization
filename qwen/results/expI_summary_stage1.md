# Exp I summary — Stage 1

_n items = 61; 95% CI bootstrap n_boot=2000_

| Condition | n_frames | acc | 95% CI | n | rel_kv_mem | avg_kv_bits | verdict |
|---|---:|---:|---|---:|---:|---:|---|
| I0_BF16_128f | 128 | 0.594 | [0.484, 0.703] | 64 | 2.000 | 16.000 | anchor |
| I1_F4_128f | 128 | 0.641 | [0.516, 0.750] | 64 | 0.500 | 4.000 | anchor |
| I2_F9_128f | 128 | 0.609 | [0.500, 0.719] | 64 | 0.594 | 4.750 | borderline |
| I3_TempWin2_128f | 128 | 0.547 | [0.422, 0.657] | 64 | 0.500 | 4.000 | kill |
| I4_TextVisualSplit_128f | 128 | 0.562 | [0.453, 0.688] | 64 | 0.500 | 4.000 | promote_n200 |
| I5_TokenBlock4_128f | 128 | 0.578 | [0.453, 0.688] | 64 | 0.500 | 4.000 | promote_paper_strong |
| I6_TempWin4_128f | 128 | 0.578 | [0.453, 0.688] | 64 | 0.500 | 4.000 | promote_paper_strong |
| I7_TempWin2_VidKVV_128f | 128 | 0.516 | [0.406, 0.625] | 64 | 0.500 | 4.000 | borderline |
| I8_TempWin2_Outlier8_128f | 128 | 0.562 | [0.453, 0.672] | 64 | 0.547 | 4.375 | promote_n200 |
| I9_F4_256f | 256 | 0.526 | [0.403, 0.667] | 57 | 1.000 | 4.000 | anchor |
| I10_F9_256f | 256 | 0.509 | [0.386, 0.632] | 57 | 1.188 | 4.750 | borderline |
| I11_TempWin4_256f | 256 | 0.614 | [0.491, 0.737] | 57 | 1.000 | 4.000 | promote_paper_strong |
| I12_TokenBlock6_256f | 256 | 0.544 | [0.404, 0.667] | 57 | 1.000 | 4.000 | borderline |
| I13_TempWin4_Outlier8_256f | 256 | 0.561 | [0.421, 0.684] | 57 | 1.094 | 4.375 | kill |
| I14_TempWin4_VidKVV_256f | 256 | 0.579 | [0.439, 0.702] | 57 | 1.000 | 4.000 | borderline |
| I15_F9MidElseTempWin | 128 | 0.578 | [0.453, 0.703] | 64 | 0.500 | 4.000 | borderline |
| I16_F9RandomMatched | 128 | 0.547 | [0.422, 0.656] | 64 | 0.500 | 4.000 | control |
