# expK_smoke results — 2026-05-11 19:19:42

**Phase A:** PASS (5/5)

| Check | Result | Detail |
|---|---|---|
| K_bits_accounting | PASS | all 11 K conditions match avg_kv_bits spec to ±0.01 |
| K10_random_block_partition | PASS | K10 partition: every cell's first 8 entries are 2 per block; seed=99 deterministic |
| K6_balanced_top2_per_block | PASS | K6 balanced top-2/block correctly picks 2 from each of TT/TV/VT/VV |
| K8_K9_budget_sizes | PASS | K8 (top-1/block) yields 4 unique per cell; K9 (top-3/block) yields ≥8 unique |
| seed_split_files_exist | PASS | 3/3 split file paths exist locally; rest will be generated on remote if missing |
