# ExpD Trial-Gate Analysis — Phase A (libero_pro_obj_x0.2)

_n trials = 50_

## Stage-1 detector AUC (LOOCV ridge LR, 95% bootstrap CI)

| Target | Features | n_feats | AUC | 95% CI | Brier |
|---|---|---:|---:|---|---:|
| `y_w4_fail` | attn | 90 | 0.740 | [0.593, 0.865] | 0.204 |
| `y_w4_fail` | chunks | 45 | 0.675 | [0.513, 0.826] | 0.237 |
| `y_w4_fail` | combined | 135 | 0.731 | [0.580, 0.862] | 0.207 |
| `y_w4_fail` | oracle | 15 | 0.569 | [0.409, 0.723] | 0.273 |
| `y_rescuable` | attn | 90 | 0.455 | [0.227, 0.697] | 0.254 |
| `y_rescuable` | chunks | 45 | 0.390 | [0.163, 0.621] | 0.261 |
| `y_rescuable` | combined | 135 | 0.405 | [0.203, 0.621] | 0.249 |
| `y_rescuable` | oracle | 15 | 0.390 | [0.182, 0.617] | 0.258 |

## Baselines (no Stage-1 gating)

| Condition | n | SR |
|---|---:|---:|
| FP16 | 50 | 0.540 |
| W4-Floor | 50 | 0.480 |
| Random-W4 | 50 | 0.580 |
| AttnEntropy-W4 | 50 | 0.520 |
| S3-Tern-W4-l12h2 | 50 | 0.560 |

## Simulated gated SR (per detector)

| Stage-1 target | Features | α (FAR) | thr | n_fired | rescue cond | gated SR |
|---|---|---:|---:|---:|---|---:|
| `y_w4_fail` | attn | 0.05 | 0.689 | 15/50 | AttnEntropy-W4 | 0.480 |
| `y_w4_fail` | attn | 0.05 | 0.689 | 15/50 | S3-Tern-W4-l12h2 | 0.500 |
| `y_w4_fail` | attn | 0.10 | 0.651 | 16/50 | AttnEntropy-W4 | 0.480 |
| `y_w4_fail` | attn | 0.10 | 0.651 | 16/50 | S3-Tern-W4-l12h2 | 0.500 |
| `y_w4_fail` | attn | 0.20 | 0.548 | 21/50 | AttnEntropy-W4 | 0.520 |
| `y_w4_fail` | attn | 0.20 | 0.548 | 21/50 | S3-Tern-W4-l12h2 | 0.540 |
| `y_w4_fail` | chunks | 0.05 | 0.735 | 5/50 | AttnEntropy-W4 | 0.480 |
| `y_w4_fail` | chunks | 0.05 | 0.735 | 5/50 | S3-Tern-W4-l12h2 | 0.480 |
| `y_w4_fail` | chunks | 0.10 | 0.650 | 17/50 | AttnEntropy-W4 | 0.520 |
| `y_w4_fail` | chunks | 0.10 | 0.650 | 17/50 | S3-Tern-W4-l12h2 | 0.540 |
| `y_w4_fail` | chunks | 0.20 | 0.606 | 20/50 | AttnEntropy-W4 | 0.500 |
| `y_w4_fail` | chunks | 0.20 | 0.606 | 20/50 | S3-Tern-W4-l12h2 | 0.520 |
| `y_w4_fail` | combined | 0.05 | 0.691 | 14/50 | AttnEntropy-W4 | 0.480 |
| `y_w4_fail` | combined | 0.05 | 0.691 | 14/50 | S3-Tern-W4-l12h2 | 0.500 |
| `y_w4_fail` | combined | 0.10 | 0.625 | 18/50 | AttnEntropy-W4 | 0.500 |
| `y_w4_fail` | combined | 0.10 | 0.625 | 18/50 | S3-Tern-W4-l12h2 | 0.520 |
| `y_w4_fail` | combined | 0.20 | 0.560 | 21/50 | AttnEntropy-W4 | 0.520 |
| `y_w4_fail` | combined | 0.20 | 0.560 | 21/50 | S3-Tern-W4-l12h2 | 0.540 |
| `y_rescuable` | attn | 0.05 | 0.678 | 2/50 | AttnEntropy-W4 | 0.480 |
| `y_rescuable` | attn | 0.05 | 0.678 | 2/50 | S3-Tern-W4-l12h2 | 0.480 |
| `y_rescuable` | attn | 0.10 | 0.646 | 4/50 | AttnEntropy-W4 | 0.500 |
| `y_rescuable` | attn | 0.10 | 0.646 | 4/50 | S3-Tern-W4-l12h2 | 0.480 |
| `y_rescuable` | attn | 0.20 | 0.588 | 6/50 | AttnEntropy-W4 | 0.480 |
| `y_rescuable` | attn | 0.20 | 0.588 | 6/50 | S3-Tern-W4-l12h2 | 0.460 |
| `y_rescuable` | chunks | 0.05 | 0.638 | 2/50 | AttnEntropy-W4 | 0.480 |
| `y_rescuable` | chunks | 0.05 | 0.638 | 2/50 | S3-Tern-W4-l12h2 | 0.460 |
| `y_rescuable` | chunks | 0.10 | 0.614 | 3/50 | AttnEntropy-W4 | 0.480 |
| `y_rescuable` | chunks | 0.10 | 0.614 | 3/50 | S3-Tern-W4-l12h2 | 0.440 |
| `y_rescuable` | chunks | 0.20 | 0.585 | 7/50 | AttnEntropy-W4 | 0.500 |
| `y_rescuable` | chunks | 0.20 | 0.585 | 7/50 | S3-Tern-W4-l12h2 | 0.460 |
| `y_rescuable` | combined | 0.05 | 0.665 | 2/50 | AttnEntropy-W4 | 0.480 |
| `y_rescuable` | combined | 0.05 | 0.665 | 2/50 | S3-Tern-W4-l12h2 | 0.480 |
| `y_rescuable` | combined | 0.10 | 0.645 | 4/50 | AttnEntropy-W4 | 0.500 |
| `y_rescuable` | combined | 0.10 | 0.645 | 4/50 | S3-Tern-W4-l12h2 | 0.480 |
| `y_rescuable` | combined | 0.20 | 0.587 | 7/50 | AttnEntropy-W4 | 0.520 |
| `y_rescuable` | combined | 0.20 | 0.587 | 7/50 | S3-Tern-W4-l12h2 | 0.460 |

## Per-bucket breakdown of best gated detector (target=`y_w4_fail`, features=`attn`, α=0.20, thr=0.548)

| Bucket | n | n_fired | fire rate | gated SR |
|---|---:|---:|---:|---:|
| clean | 17 | 2 | 12% | 100% |
| rescuable | 10 | 2 | 20% | 10% |
| w4_better | 7 | 3 | 43% | 86% |
| unrescuable | 16 | 14 | 88% | 12% |

## Matched-pair McNemar — gated AttnEntropy-W4 vs others

| Comparison | a_only | b_only | Δ SR | McNemar p |
|---|---:|---:|---:|---:|
| Gated-AttnEnt vs AttnEnt-W4 | 8 | 8 | +0.000 | 1.000 |
| Gated-AttnEnt vs W4-Floor | 3 | 1 | +0.040 | 0.625 |
| Gated-AttnEnt vs Random-W4 | 4 | 7 | -0.060 | 0.549 |

## Read

**Trial-gate weak signal.** Best deployable AUC for `y_rescuable` = 0.46; best gated AttnEntropy-W4 SR = 0.52 vs ungated 0.52. Stage-1 not strong enough to rescue at this sample size. Consider broader feature space, larger n, or new signal sources.
