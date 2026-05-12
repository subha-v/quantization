# Exp K verdict matrix — seed=1

- K6 (balanced top-2/block BF16 sidecode): replicates J7 if acc ≥ K4 + 3 pp.
- K7 (balanced top-2/block INT8 sidecode): the best-shot Pareto win vs K2 F9.
- K10 (balanced-random by channel-position): if it ties K6, mechanism is balance not cross-modal.

| Condition | acc | rel_kv_mem | avg_kv_bits | verdict |
|---|---:|---:|---:|---|
| K0_BF16_128f | 0.615 | 2.000 | 16.000 | anchor |
| K1_F4_128f | 0.570 | 0.500 | 4.000 | anchor |
| K2_F9_BF16side_128f | 0.615 | 0.594 | 4.750 | anchor |
| K3_F9_INT8side_128f | 0.600 | 0.531 | 4.250 | borderline |
| K4_F8_BF16side_128f | 0.575 | 0.547 | 4.375 | anchor |
| K5_Random8_BF16side_128f | 0.585 | 0.547 | 4.375 | control_random |
| K6_Bal2pb_BF16side_128f | 0.575 | 0.547 | 4.375 | matches_baseline |
| K7_Bal2pb_INT8side_128f | 0.575 | 0.516 | 4.125 | borderline |
| K8_Bal1pb_BF16side_128f | 0.595 | 0.523 | 4.188 | matches_K6 |
| K9_Bal3pb_BF16side_128f | 0.630 | 0.570 | 4.562 | beats_K6 |
| K10_BalRandomPos_BF16side_128f | 0.565 | 0.547 | 4.375 | control_ties_K6 |
| K11_Pivot8_BF16side_128f | 0.610 | 0.547 | 4.375 | replicates_pivot_win |
