# ExpB W4-First — Online Mixed-Precision Quantization Summary

_n rollouts = 1000_

## Overall success rate (95% bootstrap CI, n_boot=10k)

| Condition | n | success rate | 95% CI | avg bits |
|---|---:|---:|---|---:|
| FP16 | 200 | 0.420 | [0.355, 0.485] | 16.00 |
| W4-Floor | 200 | 0.460 | [0.390, 0.530] | 4.00 |
| Random-W4 | 200 | 0.445 | [0.380, 0.515] | 9.99 |
| AttnEntropy-W4 | 200 | 0.465 | [0.395, 0.535] | 9.99 |
| S3-Tern-W4-l12h2 | 200 | 0.485 | [0.415, 0.555] | 3.52 |

## Per-suite success rate

| Condition | Object |
|---|---:|
| FP16 | 0.420 [0.35,0.48] (n=200) |
| W4-Floor | 0.460 [0.39,0.53] (n=200) |
| Random-W4 | 0.445 [0.38,0.52] (n=200) |
| AttnEntropy-W4 | 0.465 [0.40,0.54] (n=200) |
| S3-Tern-W4-l12h2 | 0.485 [0.41,0.56] (n=200) |

## Hypothesis matrix (matched-pair signed deltas)

Each row computes SR(A) − SR(B) over trials present in BOTH conditions.
Positive = A wins. Matched seeds cancel intrinsic trial difficulty.

| Tag | A | B | n_matched | SR(A) − SR(B) | Question |
|---|---|---|---:|---:|---|
| HW0 | W4-Floor | FP16 | 200 | +0.040 | Is W4 alone good enough? (defines whether FP16-rescue is meaningful) |
| HW6 | AttnEntropy-W4 | Random-W4 | 200 | +0.020 | Oracle direction validation (bottom dir, W2 default) |

## HW7 — D2-W4 transfer (per-trial Spearman ρ)

Per-trial ρ between l12h2-entropy on W4-pass and ‖a_FP − a_W4‖² per cycle.
If |median ρ| > 0.15, the D2 mechanism transfers cleanly from W2 to W4.

_n=200 trials. median ρ = 0.143, mean ρ = 0.133, P(|ρ| > 0.15) = 0.62, min/max = -0.485/0.678._

| quantile | ρ |
|---|---:|
| p10 | -0.178 |
| p25 | -0.041 |
| p50 | 0.143 |
| p75 | 0.304 |
| p90 | 0.437 |

### HW7-extension — alternative probes

| probe tag | n trials | median ρ | mean ρ | p25 | p75 |
|---|---:|---:|---:|---:|---:|
| l1h7-top1 | 200 | -0.153 | -0.177 | -0.399 | +0.004 |
| l9h2-ent | 200 | +0.099 | +0.105 | -0.093 | +0.331 |
| l12h2-ent | 200 | +0.143 | +0.133 | -0.041 | +0.304 |
| l3h4-top5 | 200 | -0.010 | -0.004 | -0.156 | +0.141 |
| l17h4-top1 | 200 | -0.039 | -0.037 | -0.174 | +0.115 |
