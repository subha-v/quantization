# Experiment D1 stratified by D0 evidence label

n_d1_rows: 2800, with_label: 2800, missing_d0: 0

Cell format: `acc (n)`. Pattern to expect on **localized** items: D1.5a > D1.6a (top-1 vs random-1) at matched budget.

| Condition | localized | global | distributed | attention_not_causal | unlabeled |
|---|---|---|---|---|---|
| `D1_3_TextBF16_VisInt4_VInt4` | 0.818 (11) | 0.526 (19) | 0.500 (2) | 0.510 (49) | 0.269 (119) |
| `D1_4_TextInt4_VisBF16_VInt4` | 0.091 (11) | 0.158 (19) | 0.000 (2) | 0.265 (49) | 0.210 (119) |
| `D1_5a_TextBF16_Top1VisBF16_VInt4` | 0.909 (11) | 0.632 (19) | 0.500 (2) | 0.592 (49) | 0.261 (119) |
| `D1_5a_mh_TextBF16_Top1MaxheadVisBF16_VInt4` | 0.909 (11) | 0.632 (19) | 0.500 (2) | 0.612 (49) | 0.269 (119) |
| `D1_5b_TextBF16_Top2VisBF16_VInt4` | 1.000 (11) | 0.579 (19) | 1.000 (2) | 0.633 (49) | 0.261 (119) |
| `D1_5b_mh_TextBF16_Top2MaxheadVisBF16_VInt4` | 0.909 (11) | 0.526 (19) | 0.500 (2) | 0.592 (49) | 0.277 (119) |
| `D1_6a_TextBF16_Rand1VisBF16_VInt4_seed0` | 0.909 (11) | 0.632 (19) | 0.500 (2) | 0.612 (49) | 0.261 (119) |
| `D1_6a_TextBF16_Rand1VisBF16_VInt4_seed1` | 0.909 (11) | 0.579 (19) | 0.500 (2) | 0.571 (49) | 0.269 (119) |
| `D1_6a_TextBF16_Rand1VisBF16_VInt4_seed2` | 1.000 (11) | 0.632 (19) | 0.500 (2) | 0.633 (49) | 0.244 (119) |
| `D1_6b_TextBF16_Rand2VisBF16_VInt4_seed0` | 0.818 (11) | 0.579 (19) | 0.500 (2) | 0.653 (49) | 0.244 (119) |
| `D1_6b_TextBF16_Rand2VisBF16_VInt4_seed1` | 0.818 (11) | 0.526 (19) | 0.500 (2) | 0.653 (49) | 0.277 (119) |
| `D1_6b_TextBF16_Rand2VisBF16_VInt4_seed2` | 0.909 (11) | 0.632 (19) | 0.500 (2) | 0.612 (49) | 0.286 (119) |
| `D1_7a_TextBF16_UniformMidVisBF16_VInt4` | 1.000 (11) | 0.579 (19) | 0.500 (2) | 0.551 (49) | 0.244 (119) |
| `D1_7b_TextBF16_Uniform2VisBF16_VInt4` | 1.000 (11) | 0.632 (19) | 0.500 (2) | 0.551 (49) | 0.235 (119) |

## Headline pairs on **localized** items

| Pair | acc(left) | acc(right) | Δ |
|---|---:|---:|---:|
| `D1_5a_TextBF16_Top1VisBF16_VInt4` vs `D1_6a_TextBF16_Rand1VisBF16_VInt4_seed0` | 0.909 | 0.909 | +0.0 pp |
| `D1_5b_TextBF16_Top2VisBF16_VInt4` vs `D1_6b_TextBF16_Rand2VisBF16_VInt4_seed0` | 1.000 | 0.818 | +18.2 pp |
| `D1_5a_TextBF16_Top1VisBF16_VInt4` vs `D1_7a_TextBF16_UniformMidVisBF16_VInt4` | 0.909 | 1.000 | -9.1 pp |
| `D1_5b_TextBF16_Top2VisBF16_VInt4` vs `D1_7b_TextBF16_Uniform2VisBF16_VInt4` | 1.000 | 1.000 | +0.0 pp |
| `D1_5a_mh_TextBF16_Top1MaxheadVisBF16_VInt4` vs `D1_6a_TextBF16_Rand1VisBF16_VInt4_seed0` | 0.909 | 0.909 | +0.0 pp |
| `D1_5b_mh_TextBF16_Top2MaxheadVisBF16_VInt4` vs `D1_6b_TextBF16_Rand2VisBF16_VInt4_seed0` | 0.909 | 0.818 | +9.1 pp |
| `D1_5a_mh_TextBF16_Top1MaxheadVisBF16_VInt4` vs `D1_5a_TextBF16_Top1VisBF16_VInt4` | 0.909 | 0.909 | +0.0 pp |
| `D1_4_TextInt4_VisBF16_VInt4` vs `D1_3_TextBF16_VisInt4_VInt4` | 0.091 | 0.818 | -72.7 pp |
