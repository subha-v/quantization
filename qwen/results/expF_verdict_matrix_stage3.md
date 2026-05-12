# Exp F Verdict Matrix (stage 3)

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
| `F0_BF16` | 0.565 | [0.495, 0.635] | 1.000 | +1.727 | **anchor** |
| `F1_UniformInt4` | 0.210 | [0.155, 0.265] | 0.204 | +0.000 | **anchor** |
| `F2_TextBF16_VisInt4` | 0.385 | [0.315, 0.455] | 0.566 | +0.388 | **anchor** |
| `F3_AllKBF16_VInt4` | 0.550 | [0.480, 0.620] | 0.965 | +1.699 | **anchor** |
| `F4_KIVI_PerChannelSeq` | 0.545 | [0.475, 0.615] | 0.841 | +1.308 | **paper_strong** |
| `F5_KIVI_TextVisualSplit` | 0.510 | [0.440, 0.585] | 0.850 | +1.397 | **paper_strong** |
| `F6_KIVI_RoleSplit` | 0.525 | [0.455, 0.595] | 0.841 | +1.513 | **paper_strong** |
| `F7_KIVI_P99_5` | 0.540 | [0.470, 0.610] | 0.832 | +1.400 | **paper_strong** |
| `F8_KIVI_Outlier8` | 0.540 | [0.475, 0.610] | 0.894 | +1.644 | **paper_strong** |
| `F9_KIVI_Outlier16` | 0.560 | [0.495, 0.630] | 0.929 | +1.682 | **paper_strong** |

## Promotion plan

**Promote to n=100 (Stage 2):** none
**Promote to n=200 (Stage 3):** ['F4_KIVI_PerChannelSeq', 'F5_KIVI_TextVisualSplit', 'F6_KIVI_RoleSplit', 'F7_KIVI_P99_5', 'F8_KIVI_Outlier8', 'F9_KIVI_Outlier16']
**Borderline (manual review of Δmargin + bf16-pres):** none
