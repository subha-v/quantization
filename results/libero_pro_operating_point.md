# LIBERO-PRO operating point sweep

Per-cell FP16 success rate (50 trials each) on LIBERO-PRO position perturbation.

Goal: pick `D*` per suite closest to FP16 ≈ 70% SR — the regime where W4 has
headroom to degrade. Step 3 uses these `D*` values for the focused expC subset.


## Per-cell results

| Suite | Axis | Magnitude | n | SR | 95% CI | gap to 70% |
|-------|------|-----------|---|------:|-------|-----------:|
| Object | x | 0.1 | 50 | 0.880 | [0.780, 0.960] | 0.180 |
| Object | x | 0.2 | 50 | 0.500 | [0.360, 0.640] | 0.200 |
| Object | x | 0.3 | 50 | 0.480 | [0.340, 0.620] | 0.220 |

## Recommended D* per suite (target FP16 ≈ 70%)

| Suite | D* | SR at D* | rule |
|-------|----|----------:|------|
| Object | x0.1 | 0.880 | saturated; consider extending sweep to higher magnitudes |

## Files

- Per-trial rows written to: `results/libero_pro_operating_point.jsonl`
