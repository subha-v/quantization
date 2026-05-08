# Experiment D1 — Cross-modal K/V quantization

n_rows: 2800, n_conditions: 14

| Condition | n | acc | 95% CI | avg KV bits | mean margin |
|---|---:|---:|---|---:|---:|
| D1_3_TextBF16_VisInt4_VInt4 | 200 | 0.385 | [0.315, 0.455] | 4.15 | -0.418 |
| D1_4_TextInt4_VisBF16_VInt4 | 200 | 0.210 | [0.155, 0.265] | 9.85 | -1.609 |
| D1_5a_TextBF16_Top1VisBF16_VInt4 | 200 | 0.415 | [0.345, 0.485] | 4.88 | -0.383 |
| D1_5a_mh_TextBF16_Top1MaxheadVisBF16_VInt4 | 200 | 0.425 | [0.360, 0.490] | 4.88 | -0.412 |
| D1_5b_TextBF16_Top2VisBF16_VInt4 | 200 | 0.430 | [0.365, 0.500] | 5.61 | -0.238 |
| D1_5b_mh_TextBF16_Top2MaxheadVisBF16_VInt4 | 200 | 0.415 | [0.350, 0.485] | 5.61 | -0.305 |
| D1_6a_TextBF16_Rand1VisBF16_VInt4_seed0 | 200 | 0.420 | [0.350, 0.490] | 4.88 | -0.357 |
| D1_6a_TextBF16_Rand1VisBF16_VInt4_seed1 | 200 | 0.410 | [0.345, 0.480] | 4.88 | -0.322 |
| D1_6a_TextBF16_Rand1VisBF16_VInt4_seed2 | 200 | 0.420 | [0.350, 0.495] | 4.88 | -0.386 |
| D1_6b_TextBF16_Rand2VisBF16_VInt4_seed0 | 200 | 0.410 | [0.340, 0.480] | 5.61 | -0.315 |
| D1_6b_TextBF16_Rand2VisBF16_VInt4_seed1 | 200 | 0.425 | [0.355, 0.495] | 5.61 | -0.305 |
| D1_6b_TextBF16_Rand2VisBF16_VInt4_seed2 | 200 | 0.435 | [0.365, 0.505] | 5.61 | -0.301 |
| D1_7a_TextBF16_UniformMidVisBF16_VInt4 | 200 | 0.395 | [0.330, 0.465] | 4.88 | -0.359 |
| D1_7b_TextBF16_Uniform2VisBF16_VInt4 | 200 | 0.395 | [0.325, 0.465] | 5.61 | -0.331 |

## BF16-correct preservation

| Condition | n_bf16_correct | preserved (bf16_correct AND pred==correct) | rate |
|---|---:|---:|---:|
| D1_3_TextBF16_VisInt4_VInt4 | 100 | 55 | 0.550 |
| D1_4_TextInt4_VisBF16_VInt4 | 100 | 19 | 0.190 |
| D1_5a_TextBF16_Top1VisBF16_VInt4 | 100 | 62 | 0.620 |
| D1_5a_mh_TextBF16_Top1MaxheadVisBF16_VInt4 | 100 | 63 | 0.630 |
| D1_5b_TextBF16_Top2VisBF16_VInt4 | 100 | 65 | 0.650 |
| D1_5b_mh_TextBF16_Top2MaxheadVisBF16_VInt4 | 100 | 60 | 0.600 |
| D1_6a_TextBF16_Rand1VisBF16_VInt4_seed0 | 100 | 63 | 0.630 |
| D1_6a_TextBF16_Rand1VisBF16_VInt4_seed1 | 100 | 60 | 0.600 |
| D1_6a_TextBF16_Rand1VisBF16_VInt4_seed2 | 100 | 65 | 0.650 |
| D1_6b_TextBF16_Rand2VisBF16_VInt4_seed0 | 100 | 63 | 0.630 |
| D1_6b_TextBF16_Rand2VisBF16_VInt4_seed1 | 100 | 62 | 0.620 |
| D1_6b_TextBF16_Rand2VisBF16_VInt4_seed2 | 100 | 63 | 0.630 |
| D1_7a_TextBF16_UniformMidVisBF16_VInt4 | 100 | 60 | 0.600 |
| D1_7b_TextBF16_Uniform2VisBF16_VInt4 | 100 | 61 | 0.610 |
