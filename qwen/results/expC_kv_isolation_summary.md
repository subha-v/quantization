# Experiment C — K/V isolation (paired analysis)

Universe: 100 item_ids present in all four C2 conditions.
All other rows are restricted to this same set for paired comparison.

Per-bucket: long=34, mid=16, short=16, very_long=34

| Condition | n | acc | 95% CI | avg KV bits | BF16-correct preserved | mean margin |
|---|---:|---:|---|---:|---:|---:|
| A1 BF16 (ceiling) | 100 | 0.550 | [0.460, 0.640] | 16.00 | 1.000 | +0.712 |
| A5 INT4-K/INT4-V | 100 | 0.210 | [0.130, 0.290] | 4.00 | 0.200 (n=55) | -0.871 |
| A6 INT4-K/INT8-V | 100 | 0.280 | [0.200, 0.370] | 6.00 | 0.345 (n=55) | -1.033 |
| A7 INT2-K/INT2-V | 100 | 0.210 | [0.130, 0.290] | 2.00 | 0.218 (n=55) | -1.092 |
| C2.1 K=BF16, V=INT4 | 100 | 0.530 | [0.430, 0.620] | 10.00 | 0.945 (n=55) | +0.674 |
| C2.2 K=INT4, V=BF16 | 100 | 0.290 | [0.200, 0.380] | 10.00 | 0.218 (n=55) | -0.850 |
| C2.3 K=BF16, V=INT2 | 100 | 0.210 | [0.130, 0.290] | 9.00 | 0.182 (n=55) | -1.277 |
| C2.4 K=INT2, V=BF16 | 100 | 0.330 | [0.240, 0.420] | 9.00 | 0.364 (n=55) | -0.820 |

## Diagnosis

- BF16 ceiling: 0.550;  A5 (INT4/INT4) floor: 0.210;  A7 (INT2/INT2) floor: 0.210
- Rescue midpoint (INT4): 0.380;  (INT2): 0.380

**Symmetric question** — at INT4:
- C2.1 (K=BF16, V=INT4) = 0.530.  Δ vs A5 = +0.320.
- C2.2 (K=INT4, V=BF16) = 0.290.  Δ vs A5 = +0.080.

**Symmetric question** — at INT2:
- C2.3 (K=BF16, V=INT2) = 0.210.  Δ vs A7 = +0.000.
- C2.4 (K=INT2, V=BF16) = 0.330.  Δ vs A7 = +0.120.

**Diagnosis @ INT4** — K-fragile: V is quantizable, K is the killer.
**Diagnosis @ INT2** — per-side fragility (both): neither side rescues even when the other is held at BF16; per-side fragility is the bottleneck.

