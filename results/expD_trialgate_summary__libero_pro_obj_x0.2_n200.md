# ExpD Trial-Gate Analysis — Phase A (libero_pro_obj_x0.2_n200)

_n trials = 200_

## Stage-1 detector AUC (LOOCV ridge LR, 95% bootstrap CI)

| Target | Features | n_feats | AUC | 95% CI | Brier |
|---|---|---:|---:|---|---:|
| `y_w4_fail` | attn | 90 | 0.834 | [0.769, 0.893] | 0.157 |
| `y_w4_fail` | chunks | 45 | 0.653 | [0.575, 0.725] | 0.232 |
| `y_w4_fail` | combined | 135 | 0.842 | [0.781, 0.896] | 0.155 |
| `y_w4_fail` | oracle | 15 | 0.618 | [0.536, 0.697] | 0.249 |
| `y_rescuable` | attn | 90 | 0.603 | [0.451, 0.750] | 0.190 |
| `y_rescuable` | chunks | 45 | 0.395 | [0.218, 0.589] | 0.225 |
| `y_rescuable` | combined | 135 | 0.525 | [0.352, 0.700] | 0.187 |
| `y_rescuable` | oracle | 15 | 0.535 | [0.369, 0.691] | 0.229 |

## Baselines (no Stage-1 gating)

| Condition | n | SR |
|---|---:|---:|
| FP16 | 200 | 0.420 |
| W4-Floor | 200 | 0.460 |
| Random-W4 | 200 | 0.445 |
| AttnEntropy-W4 | 200 | 0.465 |
| S3-Tern-W4-l12h2 | 200 | 0.485 |

## Simulated gated SR (per detector)

| Stage-1 target | Features | α (FAR) | thr | n_fired | rescue cond | gated SR |
|---|---|---:|---:|---:|---|---:|
| `y_w4_fail` | attn | 0.05 | 0.621 | 83/200 | AttnEntropy-W4 | 0.465 |
| `y_w4_fail` | attn | 0.05 | 0.621 | 83/200 | S3-Tern-W4-l12h2 | 0.450 |
| `y_w4_fail` | attn | 0.10 | 0.513 | 93/200 | AttnEntropy-W4 | 0.465 |
| `y_w4_fail` | attn | 0.10 | 0.513 | 93/200 | S3-Tern-W4-l12h2 | 0.445 |
| `y_w4_fail` | attn | 0.20 | 0.457 | 105/200 | AttnEntropy-W4 | 0.455 |
| `y_w4_fail` | attn | 0.20 | 0.457 | 105/200 | S3-Tern-W4-l12h2 | 0.450 |
| `y_w4_fail` | chunks | 0.05 | 0.689 | 22/200 | AttnEntropy-W4 | 0.455 |
| `y_w4_fail` | chunks | 0.05 | 0.689 | 22/200 | S3-Tern-W4-l12h2 | 0.455 |
| `y_w4_fail` | chunks | 0.10 | 0.676 | 32/200 | AttnEntropy-W4 | 0.445 |
| `y_w4_fail` | chunks | 0.10 | 0.676 | 32/200 | S3-Tern-W4-l12h2 | 0.455 |
| `y_w4_fail` | chunks | 0.20 | 0.561 | 71/200 | AttnEntropy-W4 | 0.450 |
| `y_w4_fail` | chunks | 0.20 | 0.561 | 71/200 | S3-Tern-W4-l12h2 | 0.455 |
| `y_w4_fail` | combined | 0.05 | 0.685 | 75/200 | AttnEntropy-W4 | 0.455 |
| `y_w4_fail` | combined | 0.05 | 0.685 | 75/200 | S3-Tern-W4-l12h2 | 0.445 |
| `y_w4_fail` | combined | 0.10 | 0.553 | 91/200 | AttnEntropy-W4 | 0.460 |
| `y_w4_fail` | combined | 0.10 | 0.553 | 91/200 | S3-Tern-W4-l12h2 | 0.445 |
| `y_w4_fail` | combined | 0.20 | 0.447 | 105/200 | AttnEntropy-W4 | 0.470 |
| `y_w4_fail` | combined | 0.20 | 0.447 | 105/200 | S3-Tern-W4-l12h2 | 0.450 |
| `y_rescuable` | attn | 0.05 | 0.850 | 5/200 | AttnEntropy-W4 | 0.460 |
| `y_rescuable` | attn | 0.05 | 0.850 | 5/200 | S3-Tern-W4-l12h2 | 0.460 |
| `y_rescuable` | attn | 0.10 | 0.827 | 10/200 | AttnEntropy-W4 | 0.460 |
| `y_rescuable` | attn | 0.10 | 0.827 | 10/200 | S3-Tern-W4-l12h2 | 0.455 |
| `y_rescuable` | attn | 0.20 | 0.743 | 22/200 | AttnEntropy-W4 | 0.465 |
| `y_rescuable` | attn | 0.20 | 0.743 | 22/200 | S3-Tern-W4-l12h2 | 0.465 |
| `y_rescuable` | chunks | 0.05 | 0.863 | 6/200 | AttnEntropy-W4 | 0.460 |
| `y_rescuable` | chunks | 0.05 | 0.863 | 6/200 | S3-Tern-W4-l12h2 | 0.460 |
| `y_rescuable` | chunks | 0.10 | 0.758 | 14/200 | AttnEntropy-W4 | 0.465 |
| `y_rescuable` | chunks | 0.10 | 0.758 | 14/200 | S3-Tern-W4-l12h2 | 0.460 |
| `y_rescuable` | chunks | 0.20 | 0.645 | 28/200 | AttnEntropy-W4 | 0.465 |
| `y_rescuable` | chunks | 0.20 | 0.645 | 28/200 | S3-Tern-W4-l12h2 | 0.465 |
| `y_rescuable` | combined | 0.05 | 0.868 | 6/200 | AttnEntropy-W4 | 0.455 |
| `y_rescuable` | combined | 0.05 | 0.868 | 6/200 | S3-Tern-W4-l12h2 | 0.455 |
| `y_rescuable` | combined | 0.10 | 0.785 | 12/200 | AttnEntropy-W4 | 0.460 |
| `y_rescuable` | combined | 0.10 | 0.785 | 12/200 | S3-Tern-W4-l12h2 | 0.460 |
| `y_rescuable` | combined | 0.20 | 0.660 | 23/200 | AttnEntropy-W4 | 0.460 |
| `y_rescuable` | combined | 0.20 | 0.660 | 23/200 | S3-Tern-W4-l12h2 | 0.465 |

## Per-bucket breakdown of best gated detector (target=`y_w4_fail`, features=`combined`, α=0.20, thr=0.447)

| Bucket | n | n_fired | fire rate | gated SR |
|---|---:|---:|---:|---:|
| clean | 69 | 11 | 16% | 99% |
| rescuable | 15 | 3 | 20% | 20% |
| w4_better | 23 | 8 | 35% | 91% |
| unrescuable | 93 | 83 | 89% | 2% |

## Matched-pair McNemar — gated AttnEntropy-W4 vs others

| Comparison | a_only | b_only | Δ SR | McNemar p |
|---|---:|---:|---:|---:|
| Gated-AttnEnt vs AttnEnt-W4 | 12 | 11 | +0.005 | 1.000 |
| Gated-AttnEnt vs W4-Floor | 5 | 3 | +0.010 | 0.727 |
| Gated-AttnEnt vs Random-W4 | 20 | 15 | +0.025 | 0.500 |

## Read

**Trial-gate weak signal.** Best deployable AUC for `y_rescuable` = 0.60; best gated AttnEntropy-W4 SR = 0.47 vs ungated 0.47. Stage-1 not strong enough to rescue at this sample size. Consider broader feature space, larger n, or new signal sources.
