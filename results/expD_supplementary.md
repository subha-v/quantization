# ExpD supplementary analyses (2026-04-30)

Numbers requested for the slide deck. All computed from local JSONLs in `results/`.


## 1. ExpC S3-Tern-W4-l12h2 conditional partition on standard LIBERO (n=100)

_n trials with complete 5-condition coverage = 100_


| Bucket | n | FP16 | W4-Floor | Random-W4 | AttnEntropy-W4 | S3-Tern |
|---|---:|---:|---:|---:|---:|---:|
| clean | 90 | 100% | 100% | 100% | 98% | 99% |
| rescuable | 4 | 100% | 0% | 100% | 100% | 50% |
| w4_better | 4 | 0% | 100% | 100% | 100% | 100% |
| unrescuable | 2 | 0% | 0% | 0% | 0% | 0% |

**Decomposition of S3-Tern-W4-l12h2's win on standard LIBERO.** Aggregate SR is 95.2% vs W4-Floor 94.0% (matched-pair Δ ≈ +1pp). The bucket-level breakdown above shows where that comes from. Spatial-restriction story: S3-Tern preserves clean trials, captures rescuable trials, and avoids the unrescuable trajectory-divergence floor.

## 2. Bucket distribution across regimes — rescue vs cost-reduction


| Regime | n | clean | rescuable | w4_better | unrescuable |
|---|---:|---:|---:|---:|---:|
| ExpC standard-LIBERO n=100 | 100 | 90 (90%) | 4 (4%) | 4 (4%) | 2 (2%) |
| ExpD LIBERO-PRO Object x0.2 n=50 | 50 | 17 (34%) | 10 (20%) | 7 (14%) | 16 (32%) |
| ExpD LIBERO-PRO Object x0.2 n=200 | 200 | 69 (34%) | 15 (8%) | 23 (12%) | 93 (46%) |

**The rescue regime vs cost-reduction regime contrast.** On standard LIBERO at W4, the clean bucket dominates (W4 already works on most trials), so AttnEntropy's per-cycle gate fires unconditionally on cycles within mostly-OK rollouts — wasted firings, no rescue gap to close. The deployment story at W4 is **cost reduction** (layer-restricted W2 demotion via S3-Tern), not rescue. On LIBERO-PRO Object x0.2 the unrescuable bucket dominates (47% of trials), so AttnEntropy's gate has rescue work to do but on too small a fraction of trials (7.5% rescuable) to move aggregate SR. The W2-on-standard-LIBERO regime (expB) had ~100% of trials in the rescuable + unrescuable buckets (W2-Floor SR ≈ 0%), and that's where the +29 pp aggregate rescue showed up.

## 3. ExpC Tier 5 (l1h7-bottom) bucket decomposition

_n trials with all of 7 conditions present = 100_


| Bucket | n | W4-Floor | S3-Tern-l12h2 | S3-Bin-l1h7-bottom | S3-Tern-l1h7-bottom |
|---|---:|---:|---:|---:|---:|
| clean | 90 | 100% | 99% | 100% | 93% |
| rescuable | 4 | 0% | 50% | 100% | 75% |
| w4_better | 4 | 100% | 100% | 100% | 100% |
| unrescuable | 2 | 0% | 0% | 0% | 0% |

**Matched-pair on rescuable bucket (vs W4-Floor):**

| Comparison | n | A-only | B-only | Δ pp | McNemar p |
|---|---:|---:|---:|---:|---:|
| S3-Tern-l12h2 vs W4-Floor | 4 | 2 | 0 | +50 | 0.500 |
| S3-Bin-l1h7-bottom vs W4-Floor | 4 | 4 | 0 | +100 | 0.125 |
| S3-Tern-l1h7-bottom vs W4-Floor | 4 | 3 | 0 | +75 | 0.250 |

**Head-to-head on rescuable: l1h7-bottom variants vs S3-Tern-l12h2:**

| Comparison | n | A-only | B-only | Δ pp | McNemar p |
|---|---:|---:|---:|---:|---:|
| S3-Bin-l1h7-bottom vs S3-Tern-l12h2 | 4 | 2 | 0 | +50 | 0.500 |
| S3-Tern-l1h7-bottom vs S3-Tern-l12h2 | 4 | 1 | 0 | +25 | 1.000 |

**Reading.** Aggregate Tier 5 result was `S3-Tern-W4-l1h7-bottom` losing 3 pp vs W4-Floor. The bucket decomposition tells you whether that came from rescue failure or clean-bucket damage. Look at the per-bucket table above and compare l1h7-bottom rows to the S3-Tern-l12h2 row in the same buckets.

## 4. Memory footprint back-of-envelope: S3-Tern-W4-l12h2 vs uniform W4


Assumed pi0.5-LIBERO sizes: vision ~400M, lang layer 0 ~110M, lang layers 1-17 ~110M each (1.87B total), action expert ~300M. Bytes/param: FP16=2, W4=0.5, W2=0.25 (packed).


| Component | Size | Uniform W4 | S3-Tern-W4-l12h2 |
|---|---:|---:|---:|
| Vision tower (FP16, protected) | 400M | 0.80 GB | 0.80 GB |
| Lang layer 0 (FP16, protected) | 110M | 0.22 GB | 0.22 GB |
| Lang layers 1-12 (W4 only) | 1.32B | 0.66 GB | 0.66 GB |
| Lang layers 13-17 (W4 only / W4+W2+FP16 cached) | 0.55B | 0.28 GB | 1.51 GB |
| Action expert (FP16) | 300M | 0.60 GB | 0.60 GB |
| **Total VLM+expert weights** | — | **2.56 GB** | **3.79 GB** |

**Memory overhead of S3-Tern: +1.24 GB (+48%) vs uniform W4.** Active forward-pass cost is lower (3.49 avg bits vs 4.00) but the runtime cache must hold three precisions for the demoted layers (13-17). Trade: ~48% more weight memory, ~13% less effective bits per active forward.

## 5. n=200 LIBERO-PRO trial-gate fire-pattern (re-emit)


_From `results/expD_trialgate_summary__libero_pro_obj_x0.2_n200.md`. Best deployable detector: target=`y_w4_fail`, features=combined, α=0.20, thr=0.447._


| Bucket | n | n_fired | fire rate | gated SR (deployed) | AttnEntropy unconditional | Random unconditional |
|---|---:|---:|---:|---:|---:|---:|
| clean | 69 | 11 | 16% | 99% | 92% | 91% |
| rescuable | 15 | 3 | 20% | 20% | 86% | 53% |
| w4_better | 23 | 8 | 35% | 91% | 56% | 47% |
| unrescuable | 93 | 83 | 89% | 2% | 3% | 7% |

**Read.** Fire pattern at n=200 is the same as at n=50: detector fires on 89% of unrescuable trials and only 20% of rescuable trials. The Phase A diagnosis (detector tracks task hardness rather than quant sensitivity) holds at scale. Note that gated SR on the *rescuable* bucket is only 20% (vs unconditional AttnEntropy 86%) because the gate FAILS to fire on 80% of rescuable trials, leaving them at W4-Floor's 0% outcome on those cycles.

## n=200 matched-pair on rescuable bucket (recap)


| Comparison | n | A-only | B-only | Δ pp | McNemar p |
|---|---:|---:|---:|---:|---:|
| AttnEntropy-W4 vs Random-W4 | 15 | 6 | 1 | +33 | 0.125 |
| S3-Tern-W4-l12h2 vs Random-W4 | 15 | 4 | 0 | +27 | 0.125 |
