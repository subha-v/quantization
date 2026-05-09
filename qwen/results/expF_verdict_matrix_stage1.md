# Exp F Verdict Matrix (stage 1)

Decision rules:

- `kill`         : acc_upper_CI <= 0.27
- `borderline`   : 0.27 < acc_mean <= 0.34
- `promote_n100` : 0.34 < acc_mean < 0.40
- `promote_n200` : 0.40 <= acc_mean < 0.45
- `paper_strong` : 0.45 <= acc_mean
- `anchor`       : F0..F3 (reference, no decision)

## Verdict by condition

| Condition | acc | CI | bf16-pres | Δmargin vs F1 | Verdict |
|---|---:|---|---:|---:|:-:|
| `F0_BF16` | 0.672 | [0.547, 0.781] | 1.000 | +1.897 | **anchor** |
| `F1_UniformInt4` | 0.188 | [0.094, 0.281] | 0.186 | +0.000 | **anchor** |
| `F2_TextBF16_VisInt4` | 0.500 | [0.375, 0.625] | 0.628 | +0.791 | **anchor** |
| `F3_AllKBF16_VInt4` | 0.672 | [0.547, 0.781] | 1.000 | +1.942 | **anchor** |
| `F4_KIVI_PerChannelSeq` | 0.656 | [0.547, 0.766] | 0.884 | +1.676 | **paper_strong** |
| `F5_KIVI_TextVisualSplit` | 0.594 | [0.469, 0.719] | 0.860 | +1.538 | **paper_strong** |
| `F6_KIVI_RoleSplit` | 0.656 | [0.531, 0.766] | 0.884 | +1.626 | **paper_strong** |
| `F7_KIVI_P99_5` | 0.562 | [0.438, 0.672] | 0.767 | +1.358 | **paper_strong** |
| `F8_KIVI_Outlier8` | 0.641 | [0.516, 0.750] | 0.930 | +2.013 | **paper_strong** |
| `F9_KIVI_Outlier16` | 0.672 | [0.547, 0.781] | 0.977 | +1.957 | **paper_strong** |
| `F10_ScoreCal_Generic` | 0.281 | [0.172, 0.391] | 0.326 | -0.261 | **borderline** |
| `F11_ScoreCal_Block_TTHeavy` | 0.172 | [0.094, 0.266] | 0.163 | -0.468 | **kill** |
| `F12_ScoreCal_Block_Balanced` | 0.188 | [0.094, 0.281] | 0.186 | -0.430 | **borderline** |
| `F13_ScoreCal_TextOnly` | 0.172 | [0.078, 0.266] | 0.163 | -1.526 | **kill** |

## Promotion plan

**Promote to n=100 (Stage 2):** none
**Promote to n=200 (Stage 3):** ['F4_KIVI_PerChannelSeq', 'F5_KIVI_TextVisualSplit', 'F6_KIVI_RoleSplit', 'F7_KIVI_P99_5', 'F8_KIVI_Outlier8', 'F9_KIVI_Outlier16']
**Borderline (manual review of Δmargin + bf16-pres):** ['F10_ScoreCal_Generic', 'F12_ScoreCal_Block_Balanced']
