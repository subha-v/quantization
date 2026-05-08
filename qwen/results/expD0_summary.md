# Experiment D0 — Evidence-window diagnostic

n_items: 200

## Evidence label distribution

| Label | n | % |
|---|---:|---:|
| localized | 11 | 5.5% |
| global | 19 | 9.5% |
| distributed | 2 | 1.0% |
| attention_not_causal | 49 | 24.5% |
| unlabeled | 119 | 59.5% |

## Per-condition accuracy (frame-restriction conditions)

| Condition | n | acc | 95% CI |
|---|---:|---:|---|
| D0.1 Full-64 BF16 | 200 | 0.500 | [0.430, 0.570] |
| D0.2 Uniform-16 BF16 | 200 | 0.500 | [0.435, 0.570] |
| D0.3 Top-1-window-only | 200 | 0.425 | [0.360, 0.490] |
| D0.4 Top-2-windows-only | 200 | 0.520 | [0.450, 0.590] |
| D0.5 Top-1-window-removed | 200 | 0.560 | [0.495, 0.630] |
| D0.6 Random-window-removed (3 seeds pooled) | 600 | 0.548 | — |

## Mean answer margin

| Condition | mean | std | n |
|---|---:|---:|---:|
| Full-64 | 0.646 | 2.673 | 200 |
| Uniform-16 | 0.326 | 2.305 | 200 |
| Top-1-only | -0.097 | 2.224 | 200 |
| Top-2-only | 0.473 | 2.525 | 200 |
| Top-1-removed | 0.675 | 2.665 | 200 |

## EvidenceCausalGap by duration bucket

| Bucket | n | median EvidenceCausalGap | IQR |
|---|---:|---:|---|
| short | 33 | -0.040 | [-0.278, 0.238] |
| mid | 33 | -0.040 | [-0.317, 0.198] |
| long | 67 | 0.040 | [-0.198, 0.278] |
| very_long | 67 | 0.040 | [-0.198, 0.159] |

## Visual mass total (mean over items)

| Pool | mean | std | min | max |
|---|---:|---:|---:|---:|
| all-layer | 0.0620 | 0.0081 | 0.0449 | 0.0912 |
| mid-layer | 0.0511 | 0.0071 | 0.0351 | 0.0724 |
