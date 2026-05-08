# Experiment E1 — Pair analysis & verdict matrix


## E1.x vs E1.1 (all text-K BF16) — what fraction of D1.3's rescue does this slice deliver?

| Condition | acc | n_text_protected | acc - E1.1 | tokens / E1.1 tokens |
|---|---:|---:|---:|---:|
| E1_10_KResidTopTextK | 0.200 | 20 | -18.5 pp | 0.14 |
| E1_2_HeaderOnly | 0.215 | 14 | -17.0 pp | 0.10 |
| E1_3_QuestionOnly | 0.175 | 50 | -21.0 pp | 0.36 |
| E1_4_OptionsOnly | 0.290 | 40 | -9.5 pp | 0.29 |
| E1_5_InstrAnsPrefix | 0.225 | 22 | -16.0 pp | 0.16 |
| E1_6_QuestionOptions | 0.270 | 91 | -11.5 pp | 0.65 |
| E1_7_OptionsAnsPrefix | 0.185 | 45 | -20.0 pp | 0.32 |
| E1_8_QuestionOptionsAnsPrefix | 0.205 | 96 | -18.0 pp | 0.69 |
| E1_9_RandomTextK_seed0 | 0.220 | 20 | -16.5 pp | 0.14 |
| E1_9_RandomTextK_seed1 | 0.215 | 20 | -17.0 pp | 0.14 |
| E1_9_RandomTextK_seed2 | 0.210 | 20 | -17.5 pp | 0.14 |

## Per-bucket accuracy

| Condition | short | mid | long | very_long |
|---|---:|---:|---:|---:|
| `E1_10_KResidTopTextK` | 0.152 (n=33) | 0.182 (n=33) | 0.254 (n=67) | 0.179 (n=67) |
| `E1_2_HeaderOnly` | 0.273 (n=33) | 0.212 (n=33) | 0.209 (n=67) | 0.194 (n=67) |
| `E1_3_QuestionOnly` | 0.182 (n=33) | 0.182 (n=33) | 0.224 (n=67) | 0.119 (n=67) |
| `E1_4_OptionsOnly` | 0.212 (n=33) | 0.455 (n=33) | 0.254 (n=67) | 0.284 (n=67) |
| `E1_5_InstrAnsPrefix` | 0.273 (n=33) | 0.212 (n=33) | 0.224 (n=67) | 0.209 (n=67) |
| `E1_6_QuestionOptions` | 0.273 (n=33) | 0.364 (n=33) | 0.239 (n=67) | 0.254 (n=67) |
| `E1_7_OptionsAnsPrefix` | 0.212 (n=33) | 0.273 (n=33) | 0.164 (n=67) | 0.149 (n=67) |
| `E1_8_QuestionOptionsAnsPrefix` | 0.152 (n=33) | 0.273 (n=33) | 0.254 (n=67) | 0.149 (n=67) |
| `E1_9_RandomTextK_seed0` | 0.242 (n=33) | 0.182 (n=33) | 0.269 (n=67) | 0.179 (n=67) |
| `E1_9_RandomTextK_seed1` | 0.242 (n=33) | 0.152 (n=33) | 0.239 (n=67) | 0.209 (n=67) |
| `E1_9_RandomTextK_seed2` | 0.242 (n=33) | 0.242 (n=33) | 0.254 (n=67) | 0.134 (n=67) |

## Verdict: smallest sufficient text-K subset

Reference: E1.1 (all text-K BF16) acc = 0.385, n_text_protected ≈ 140.

80% of E1.1 acc threshold = 0.308; <50% tokens = 70.


| Condition | acc | n_tokens | meets 80% acc | <50% tokens | sufficient? |
|---|---:|---:|:-:|:-:|:-:|
| E1_10_KResidTopTextK | 0.200 | 20 | no | yes | no |
| E1_2_HeaderOnly | 0.215 | 14 | no | yes | no |
| E1_3_QuestionOnly | 0.175 | 50 | no | yes | no |
| E1_4_OptionsOnly | 0.290 | 40 | no | yes | no |
| E1_5_InstrAnsPrefix | 0.225 | 22 | no | yes | no |
| E1_6_QuestionOptions | 0.270 | 91 | no | no | no |
| E1_7_OptionsAnsPrefix | 0.185 | 45 | no | yes | no |
| E1_8_QuestionOptionsAnsPrefix | 0.205 | 96 | no | no | no |
| E1_9_RandomTextK_seed0 | 0.220 | 20 | no | yes | no |
| E1_9_RandomTextK_seed1 | 0.215 | 20 | no | yes | no |
| E1_9_RandomTextK_seed2 | 0.210 | 20 | no | yes | no |
