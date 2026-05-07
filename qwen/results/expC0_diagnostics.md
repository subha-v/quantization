# Experiment C0 — no-compute diagnostics

Inputs: expA=expA_rollouts_Qwen2.5-VL-7B-Instruct.jsonl, expB=expB_online_rollouts.jsonl, diag=diagnostic_signals.jsonl, static=static_entropy_risk.json
Routing budget: top-16 of 28×4 = 112 blocks → BF16.

## Block 1 — A5 (INT4) vs A7 (INT2) item-level complementarity

n eval items (both A5 & A7 present): **200**

| set | count | % of n |
|---|---:|---:|
| INT4 correct (A5) | 42 | 21.0% |
| INT2 correct (A7) | 54 | 27.0% |
| both correct (∩) | 13 | 6.5% |
| union correct (∪ = oracle of {INT4, INT2}) | 83 | 41.5% |
| **INT4-only correct** (INT4 ✓, INT2 ✗) | **29** | **14.5%** |
| **INT2-only correct** (INT2 ✓, INT4 ✗) | **41** | **20.5%** |
| both wrong | 117 | 58.5% |

**Derived metrics**

- Symmetric difference (INT4 △ INT2 on correct): **70** (35.0% of items)
- Jaccard(INT4-correct, INT2-correct): **0.157** (1.0 = perfectly aligned, 0.0 = disjoint)
- Observed both-correct: 13.  Expected under independence: 11.3.  Lift = +1.7 (positive → correlated)
- Oracle {INT4, INT2} accuracy ceiling: **41.5%** (vs A5=21.0%, A7=27.0%, BF16=56.5%)
- BF16-correct rescued by INT4 ∪ INT2: 46/113 (40.7%) of BF16-correct items

**Verdict.** INT4 and INT2 succeed on substantially different items — oracle union (41.5%) exceeds either alone by ≥29 items. A richer {INT2, INT4, BF16} tier set has real headroom *if* a router can pick the right tier per block.

**Per-duration-bucket breakdown**

| bucket | n | INT4-only | INT2-only | both ✓ | union ✓ | sym-diff |
|---|---:|---:|---:|---:|---:|---:|
| short | 33 | 9 | 8 | 1 | 18 | 17 |
| mid | 33 | 4 | 8 | 4 | 16 | 12 |
| long | 67 | 8 | 14 | 3 | 25 | 22 |
| very_long | 67 | 8 | 11 | 5 | 24 | 19 |


## Block 2 — Selected-block coverage for B6 / B8 / B9 / B10

Per item, each method scores 28 × 4 = 112 (layer, KV-head) blocks and picks the top **16** for BF16. We summarize coverage across the eval set (n=200).

Columns:
- **layer-cov** = mean # distinct layers receiving ≥1 BF16 head per item (max = 28; uniform-spread upper bound = min(28, 16) = 16)
- **heads-per-layer** = mean # BF16 heads per protected layer per item (max = 4)
- **layer-Hbits** = Shannon entropy (bits) of *summed* across-item layer-protection distribution; uniform = log2(28) ≈ 4.81
- **head-Hbits** = entropy of summed (L, h) protection distribution; uniform = log2(112) ≈ 6.81
- **stable-blocks** = # of (L, h) blocks selected by ≥80% of items (per-method 'always-on' core)
- **never-blocks** = # of (L, h) blocks *never* selected

| method | layer-cov / 28 | heads-per-layer / 4 | layer-Hbits | head-Hbits | stable-blocks | never-blocks |
|---|---:|---:|---:|---:|---:|---:|
| B6_StaticEntropy | 12.00 | 1.33 | 3.58 | 4.00 | 16 | 96 |
| B8_OnlineResidual | 12.73 | 1.26 | 3.89 | 4.34 | 13 | 79 |
| B9_OnlineNeed_Static | 12.87 | 1.25 | 3.83 | 4.38 | 12 | 85 |
| B10_OnlineNeed_AQ | 9.62 | 1.67 | 3.70 | 4.47 | 12 | 69 |

**Per-layer protection counts** (out of 200 eval items, a layer is 'protected' if any of its 4 KV-heads is selected):

| method | L00 | L01 | L02 | L03 | L04 | L05 | L06 | L07 | L08 | L09 | L10 | L11 | L12 | L13 | L14 | L15 | L16 | L17 | L18 | L19 | L20 | L21 | L22 | L23 | L24 | L25 | L26 | L27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B6_StaticEntropy | 0 | 0 | 0 | 0 | 200 | 200 | 200 | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 0 | 200 | 200 | 200 | 200 | 200 | 0 | 200 | 0 |
| B8_OnlineResidual | 0 | 0 | 179 | 200 | 200 | 200 | 200 | 101 | 194 | 68 | 183 | 200 | 75 | 12 | 130 | 200 | 0 | 5 | 2 | 0 | 0 | 0 | 0 | 196 | 2 | 200 | 0 | 0 |
| B9_OnlineNeed_Static | 0 | 0 | 0 | 0 | 200 | 200 | 200 | 190 | 200 | 0 | 174 | 200 | 0 | 0 | 200 | 0 | 0 | 13 | 181 | 0 | 79 | 11 | 198 | 196 | 132 | 200 | 0 | 0 |
| B10_OnlineNeed_AQ | 0 | 167 | 200 | 200 | 26 | 200 | 200 | 43 | 181 | 200 | 169 | 116 | 9 | 103 | 39 | 0 | 1 | 1 | 38 | 0 | 23 | 0 | 3 | 0 | 0 | 0 | 0 | 5 |

**Reading the table.**

- Lower **layer-cov** + lower **layer-Hbits** + higher **stable-blocks** = concentrated routing (few layers always-on, suspect over-concentration).
- High **layer-cov** + high entropies + low **stable-blocks** = spread routing (closer to random allocation).
- Compare layer protection rows side-by-side to see whether B9/B10's failures are because they pile BF16 onto a small set of layers (over-concentration) vs. spread it everywhere like random.


## Block 3 — Answer margin even when accuracy fails

**margin = log p(correct option) − max log p(other option)**.
Positive margin ⇒ argmax = correct (i.e., is_correct=True).
If a routing method is moving logits in the right direction even when it doesn't quite flip the prediction, the *wrong-only* margin should rise above the INT2/INT4 floors.

| condition | n | acc | mean margin (all) | mean margin (wrong only) | mean margin (BF16-correct subset) |
|---|---:|---:|---:|---:|---:|
| A1 BF16 (ceiling) | 200 | 0.565 | +0.921 | -1.371 (n=87) | +2.685 (n=113) |
| A5 INT4 (matched-avg) | 200 | 0.210 | -0.806 | -1.172 (n=158) | -0.789 (n=113) |
| A7 INT2 (floor) | 200 | 0.270 | -0.911 | -1.545 (n=146) | -0.886 (n=113) |
| B6 StaticEntropy (low→BF16) | 200 | 0.245 | -0.983 | -1.589 (n=151) | -0.959 (n=113) |
| B7 FlippedEntropy | 200 | 0.270 | -1.233 | -2.063 (n=146) | -1.221 (n=113) |
| B8 OnlineResidual | 200 | 0.265 | -0.896 | -1.529 (n=147) | -0.969 (n=113) |
| B9 OnlineNeed-Static | 200 | 0.195 | -1.032 | -1.504 (n=161) | -0.945 (n=113) |
| B10 OnlineNeed-AQ | 200 | 0.210 | -1.027 | -1.484 (n=158) | -1.058 (n=113) |

**Δ-margin vs floors** (mean over items present in both conditions):

| routed condition | Δ vs A5 (INT4) wrong-only | Δ vs A7 (INT2) wrong-only | Δ vs A5 BF16-subset | Δ vs A7 BF16-subset |
|---|---:|---:|---:|---:|
| B6 StaticEntropy | -0.823 (n=151) | -0.634 (n=151) | -0.170 (n=113) | -0.072 (n=113) |
| B7 FlippedEntropy | -1.173 (n=146) | -0.945 (n=146) | -0.432 (n=113) | -0.335 (n=113) |
| B8 OnlineResidual | -0.683 (n=147) | -0.531 (n=147) | -0.181 (n=113) | -0.083 (n=113) |
| B9 OnlineNeed-Static | -0.717 (n=161) | -0.561 (n=161) | -0.157 (n=113) | -0.059 (n=113) |
| B10 OnlineNeed-AQ | -0.698 (n=158) | -0.572 (n=158) | -0.269 (n=113) | -0.172 (n=113) |

**Reading the table.**

- Δ > 0 means the routed condition produced higher margin (more correct-leaning logits) than the uniform floor on the same items.
- A positive Δ even when accuracy didn't move means the routing signals *are* nudging logits toward the right answer — just not enough to flip the argmax. That would imply the floor isn't a complete loss of signal.
- A non-positive Δ means routing isn't even moving the needle: the destruction from 86% INT2 swamps any benefit from the BF16 16-block budget.
