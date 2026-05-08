# Experiment E1 — Text-K slice ablation

n_rows: 2200, n_conditions: 11

| Condition | n | acc | 95% CI | mean margin | median n_text_protected | avg KV bits |
|---|---:|---:|---|---:|---:|---:|
| **E1.1 = D1.3 (all text-K BF16)** [reused] | 200 | 0.385 | [0.315, 0.455] | -0.418 | ~140 | 4.15 |
| E1_10_KResidTopTextK | 200 | 0.200 | [0.145, 0.255] | -1.087 | 20 | 4.02 |
| E1_2_HeaderOnly | 200 | 0.215 | [0.160, 0.275] | -0.663 | 14 | 4.01 |
| E1_3_QuestionOnly | 200 | 0.175 | [0.125, 0.225] | -1.131 | 50 | 4.05 |
| E1_4_OptionsOnly | 200 | 0.290 | [0.225, 0.355] | -0.874 | 40 | 4.06 |
| E1_5_InstrAnsPrefix | 200 | 0.225 | [0.170, 0.285] | -1.294 | 22 | 4.02 |
| E1_6_QuestionOptions | 200 | 0.270 | [0.215, 0.330] | -0.970 | 91 | 4.11 |
| E1_7_OptionsAnsPrefix | 200 | 0.185 | [0.130, 0.235] | -0.972 | 45 | 4.06 |
| E1_8_QuestionOptionsAnsPrefix | 200 | 0.205 | [0.150, 0.260] | -0.843 | 96 | 4.12 |
| E1_9_RandomTextK_seed0 | 200 | 0.220 | [0.165, 0.280] | -1.041 | 20 | 4.02 |
| E1_9_RandomTextK_seed1 | 200 | 0.215 | [0.160, 0.275] | -1.249 | 20 | 4.02 |
| E1_9_RandomTextK_seed2 | 200 | 0.210 | [0.155, 0.270] | -1.258 | 20 | 4.02 |

## BF16-correct preservation (paired on the BF16-correct subset)

| Condition | n_bf16_correct | preserved | rate |
|---|---:|---:|---:|
| E1_10_KResidTopTextK | 100 | 24 | 0.240 |
| E1_2_HeaderOnly | 100 | 22 | 0.220 |
| E1_3_QuestionOnly | 100 | 18 | 0.180 |
| E1_4_OptionsOnly | 100 | 36 | 0.360 |
| E1_5_InstrAnsPrefix | 100 | 19 | 0.190 |
| E1_6_QuestionOptions | 100 | 33 | 0.330 |
| E1_7_OptionsAnsPrefix | 100 | 22 | 0.220 |
| E1_8_QuestionOptionsAnsPrefix | 100 | 24 | 0.240 |
| E1_9_RandomTextK_seed0 | 100 | 23 | 0.230 |
| E1_9_RandomTextK_seed1 | 100 | 19 | 0.190 |
| E1_9_RandomTextK_seed2 | 100 | 28 | 0.280 |
