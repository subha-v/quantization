# expJ_smoke results — 2026-05-10 09:47:06

**Phase A:** PASS (5/5)
**Phase B:** PASS (2/2)

## Phase A — synthetic kernel + bits-accounting checks
| Check | Result | Detail |
|---|---|---|
| custom_outlier_idx_lookup | PASS | default & custom outlier_idx_key both restore correct channels: default(gen=0.00e+00,cus=0.1289) custom(gen=0.9688,cus=0.00e+00) |
| int_n_sidecode_round_trip | PASS | BF16 sidecode delta=0.00e+00 (exact); INT8 sidecode delta=0.0625 (lossy); non-outlier match across sidecodes (delta=0.00e+00) |
| layer_adaptive_budget_resolve | PASS | resolved budget: 56/112 cells with budget=16, min(kept_risk)=59.40 >= max(dropped_risk)=55.20 |
| bits_accounting_J | PASS | all 14 conditions match avg_kv_bits spec to ±0.01 |
| seed2_split_supersets | PASS | seed=2: n=64 (64) ⊂ n=200 (200) |

## Phase B — live-model checks
| Check | Result | Detail |
|---|---|---|
| visual_span_seed2 | PASS | 8/8 v_end > v_start; first 4: [(15, 11535), (15, 11535), (15, 11535), (15, 11535)] |
| logits_differ | PASS | all 6 pairs differ: BF16-vs-J6=0.6883, BF16-vs-J9=0.4307, BF16-vs-J12=0.6532, J6-vs-J9=0.4287, J6-vs-J12=1.2744, J9-vs-J12=0.8458 |
