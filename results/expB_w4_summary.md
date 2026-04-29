# ExpB W4-First — Online Mixed-Precision Quantization Summary

_n rollouts = 2300_

## Overall success rate (95% bootstrap CI, n_boot=10k)

| Condition | n | success rate | 95% CI | avg bits |
|---|---:|---:|---|---:|
| FP16 | 100 | 0.940 | [0.890, 0.980] | 16.00 |
| W4-Floor | 100 | 0.940 | [0.890, 0.980] | 4.00 |
| W4-Static-Sched | 100 | 0.940 | [0.890, 0.980] | 4.00 |
| Random-W4 | 100 | 0.980 | [0.950, 1.000] | 8.82 |
| AttnEntropy-W4 | 100 | 0.960 | [0.920, 0.990] | 8.82 |
| S1-Bin-W4 | 100 | 0.950 | [0.900, 0.990] | 9.13 |
| S2-Bin-W4 | 100 | 0.970 | [0.930, 1.000] | 8.82 |
| S3-Bin-W4-l1h7-top1 | 100 | 0.970 | [0.930, 1.000] | 7.90 |
| S3-Bin-W4-l9h2-ent | 100 | 0.980 | [0.950, 1.000] | 5.05 |
| S3-Bin-W4-l12h2-ent | 100 | 0.970 | [0.930, 1.000] | 4.42 |
| S1-Tern-W4 | 100 | 0.790 | [0.710, 0.870] | 4.57 |
| S2-Tern-W4 | 100 | 0.740 | [0.650, 0.820] | 4.19 |
| S3-Tern-W4-l12h2 | 100 | 0.950 | [0.900, 0.990] | 3.58 |
| Random-Tern-W4 | 100 | 0.790 | [0.710, 0.870] | 4.19 |
| AttnEntropy-W4-top | 100 | 0.950 | [0.900, 0.990] | 8.82 |
| S1-Bin-W4-top | 100 | 0.980 | [0.950, 1.000] | 8.90 |
| S2-Bin-W4-top | 100 | 0.950 | [0.900, 0.990] | 8.82 |
| S3-Bin-W4-l12h2-ent-top | 100 | 0.980 | [0.950, 1.000] | 6.49 |
| S1-Tern-W4-top | 100 | 0.710 | [0.620, 0.800] | 4.45 |
| S2-Tern-W4-top | 100 | 0.640 | [0.540, 0.730] | 4.19 |
| S3-Tern-W4-l12h2-top | 100 | 0.970 | [0.930, 1.000] | 5.20 |
| S3-Bin-W4-l1h7-bottom | 100 | 0.980 | [0.950, 1.000] | 9.32 |
| S3-Tern-W4-l1h7-bottom | 100 | 0.910 | [0.850, 0.960] | 4.43 |

## Per-suite success rate

| Condition | Long | Object |
|---|---:|---:|
| FP16 | 0.880 [0.78,0.96] (n=50) | 1.000 [1.00,1.00] (n=50) |
| W4-Floor | 0.920 [0.84,0.98] (n=50) | 0.960 [0.90,1.00] (n=50) |
| W4-Static-Sched | 0.920 [0.84,0.98] (n=50) | 0.960 [0.90,1.00] (n=50) |
| Random-W4 | 0.960 [0.90,1.00] (n=50) | 1.000 [1.00,1.00] (n=50) |
| AttnEntropy-W4 | 0.920 [0.84,0.98] (n=50) | 1.000 [1.00,1.00] (n=50) |
| S1-Bin-W4 | 0.900 [0.80,0.98] (n=50) | 1.000 [1.00,1.00] (n=50) |
| S2-Bin-W4 | 0.940 [0.86,1.00] (n=50) | 1.000 [1.00,1.00] (n=50) |
| S3-Bin-W4-l1h7-top1 | 0.940 [0.86,1.00] (n=50) | 1.000 [1.00,1.00] (n=50) |
| S3-Bin-W4-l9h2-ent | 0.960 [0.90,1.00] (n=50) | 1.000 [1.00,1.00] (n=50) |
| S3-Bin-W4-l12h2-ent | 0.940 [0.86,1.00] (n=50) | 1.000 [1.00,1.00] (n=50) |
| S1-Tern-W4 | 0.680 [0.56,0.82] (n=50) | 0.900 [0.80,0.98] (n=50) |
| S2-Tern-W4 | 0.560 [0.42,0.70] (n=50) | 0.920 [0.84,0.98] (n=50) |
| S3-Tern-W4-l12h2 | 0.940 [0.86,1.00] (n=50) | 0.960 [0.90,1.00] (n=50) |
| Random-Tern-W4 | 0.680 [0.56,0.80] (n=50) | 0.900 [0.80,0.98] (n=50) |
| AttnEntropy-W4-top | 0.940 [0.86,1.00] (n=50) | 0.960 [0.90,1.00] (n=50) |
| S1-Bin-W4-top | 0.960 [0.90,1.00] (n=50) | 1.000 [1.00,1.00] (n=50) |
| S2-Bin-W4-top | 0.920 [0.84,0.98] (n=50) | 0.980 [0.94,1.00] (n=50) |
| S3-Bin-W4-l12h2-ent-top | 0.960 [0.90,1.00] (n=50) | 1.000 [1.00,1.00] (n=50) |
| S1-Tern-W4-top | 0.580 [0.44,0.72] (n=50) | 0.840 [0.74,0.94] (n=50) |
| S2-Tern-W4-top | 0.520 [0.38,0.66] (n=50) | 0.760 [0.64,0.86] (n=50) |
| S3-Tern-W4-l12h2-top | 0.940 [0.86,1.00] (n=50) | 1.000 [1.00,1.00] (n=50) |
| S3-Bin-W4-l1h7-bottom | 0.960 [0.90,1.00] (n=50) | 1.000 [1.00,1.00] (n=50) |
| S3-Tern-W4-l1h7-bottom | 0.880 [0.78,0.96] (n=50) | 0.940 [0.86,1.00] (n=50) |

## Hypothesis matrix (matched-pair signed deltas)

Each row computes SR(A) − SR(B) over trials present in BOTH conditions.
Positive = A wins. Matched seeds cancel intrinsic trial difficulty.

| Tag | A | B | n_matched | SR(A) − SR(B) | Question |
|---|---|---|---:|---:|---|
| HW0 | W4-Floor | FP16 | 100 | +0.000 | Is W4 alone good enough? (defines whether FP16-rescue is meaningful) |
| HW1 | S1-Bin-W4 | Random-W4 | 100 | -0.030 | Does the lag-1 mechanism work at W4? (bottom dir) |
| HW2 | S3-Bin-W4-l12h2-ent | S1-Bin-W4 | 100 | +0.020 | Does intra-pass beat lag-1 at W4? (bottom dir) |
| HW3a | S3-Bin-W4-l1h7-top1 | S3-Bin-W4-l12h2-ent | 100 | +0.000 | Earlier-layer cheap-pass viable at W4? |
| HW3b | S3-Bin-W4-l9h2-ent | S3-Bin-W4-l12h2-ent | 100 | +0.010 | Mid-layer alt viable? |
| HW4 | S1-Tern-W4 | W4-Floor | 100 | -0.150 | Sub-W4 average preserves SR vs uniform W4? |
| HW5 | S2-Bin-W4 | S1-Bin-W4 | 100 | +0.020 | No-lag advantage at W4? |
| HW6 | AttnEntropy-W4 | Random-W4 | 100 | -0.020 | Oracle direction validation (bottom dir, W2 default) |
| HW9a | AttnEntropy-W4-top | Random-W4 | 100 | -0.030 | Top-direction oracle vs random — does flipped direction work? |
| HW9b | AttnEntropy-W4-top | AttnEntropy-W4 | 100 | -0.010 | Top vs bottom direction oracle — which is right at W4? |
| HW9c | S1-Bin-W4-top | S1-Bin-W4 | 100 | +0.030 | Top vs bottom lag-1 — which is right at W4? |
| HW9d | S3-Bin-W4-l12h2-ent-top | S3-Bin-W4-l12h2-ent | 100 | +0.010 | Top vs bottom intra-pass at l12h2 |
| HW9e | S2-Bin-W4-top | S2-Bin-W4 | 100 | -0.020 | Top vs bottom speculative |
| HW10a | S1-Tern-W4-top | S1-Tern-W4 | 100 | -0.080 | Top vs bottom lag-1 ternary — does flip rescue S1-Tern? |
| HW10b | S2-Tern-W4-top | S2-Tern-W4 | 100 | -0.100 | Top vs bottom speculative ternary — does flip rescue S2-Tern? |
| HW10c | S3-Tern-W4-l12h2-top | S3-Tern-W4-l12h2 | 100 | +0.020 | Top vs bottom intra-pass ternary at l12h2 |
| HW10d | S1-Tern-W4-top | W4-Floor | 100 | -0.230 | Direction-flipped ternary: does it match Floor at sub-W4 bits? |
| HW10e | S3-Tern-W4-l12h2-top | W4-Floor | 100 | +0.030 | Intra-pass top-dir ternary: even better than bottom? |
| HW11a | S3-Bin-W4-l1h7-bottom | S3-Bin-W4-l1h7-top1 | 100 | +0.010 | l1h7 bottom (W4-correct) vs top (W2-default) |
| HW11b | S3-Bin-W4-l1h7-bottom | S3-Bin-W4-l12h2-ent | 100 | +0.010 | l1h7 bottom vs l12h2 bottom — earlier layer better? |
| HW11c | S3-Tern-W4-l1h7-bottom | S3-Tern-W4-l12h2 | 100 | -0.040 | l1h7 bottom ternary vs l12h2 bottom ternary |
| HW11d | S3-Tern-W4-l1h7-bottom | W4-Floor | 100 | -0.030 | l1h7 ternary vs W4-Floor (cheap-pass Pareto test) |

## HW7 — D2-W4 transfer (per-trial Spearman ρ)

Per-trial ρ between l12h2-entropy on W4-pass and ‖a_FP − a_W4‖² per cycle.
If |median ρ| > 0.15, the D2 mechanism transfers cleanly from W2 to W4.

_n=100 trials. median ρ = 0.115, mean ρ = 0.120, P(|ρ| > 0.15) = 0.57, min/max = -0.464/0.684._

| quantile | ρ |
|---|---:|
| p10 | -0.208 |
| p25 | -0.053 |
| p50 | 0.115 |
| p75 | 0.283 |
| p90 | 0.416 |

### HW7-extension — alternative probes

| probe tag | n trials | median ρ | mean ρ | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| l1h7-top1 | 100 | -0.167 | -0.155 | -0.330 | +0.048 |
| l9h2-ent | 100 | -0.031 | -0.023 | -0.197 | +0.127 |
| l12h2-ent | 100 | +0.115 | +0.120 | -0.053 | +0.283 |
| l3h4-top5 | 100 | -0.016 | -0.014 | -0.228 | +0.133 |
| l17h4-top1 | 100 | -0.009 | +0.016 | -0.124 | +0.167 |
