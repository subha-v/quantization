# Qwen2.5-VL × LongVideoBench — KV-cache Quantization Experiments

**Status as of 2026-05-08:** **Four experiments — D0 just completed, D1 in flight.** Exp A (8/8 conditions × 200 eval items), Exp B Online Precision-Need Routing (8 routed conditions × 200 eval items at avg=4 KV bits), Exp C K/V isolation mini-sweep (4 conditions × 100 stratified eval items at avg=10 / avg=9 KV bits), and Exp D0 Evidence-window diagnostic (200 eval items × 8 BF16 conditions, 1h 26min wall on H100). Total: ~3600 rollouts of routed/baseline data + 33,600 (item × layer × KV-head) diagnostic signal rows + 200 D0 per-item rows with full per-(layer, head) answer-query attention windows.

## Headline

> **Exp A — Setup confirmed.** On Qwen2.5-VL-7B + LongVideoBench (200 eval items, 64 frames):
>
> 1. **Weight quantization to 4 bits is essentially free.** Fake-quant W4 and real AWQ both at **54.0%** (BF16 = 56.5%, Δ = −2.5 pp).
> 2. **Uniform KV cache quantization is catastrophic at every bit width tested.** FP8 / INT4 / INT4-K-INT8-V / INT2 / AWQ+INT4 all collapse to **21–27%** — at or near 4-way chance.
> 3. **30 pp "rescuable regime"** between BF16 (56.5%) and any uniform KV-quant (~25%). The largest rescue surface area in any of our experiments.
>
> **Exp B Online Precision-Need Routing — NEGATIVE RESULT.** At matched avg=4 KV bits with the {INT2, BF16} tier set (16/112 BF16 blocks, 96/112 INT2 blocks):
>
> 4. **No routing strategy at avg=4 with {INT2, BF16} recovers above the worst uniform baseline.** Random (3 seeds), MEDA-style layer entropy, per-(layer, KV-head) StaticEntropy (both directions), per-item OnlineResidual, and the multiplicative `OnlineNeed-Static` and `OnlineNeed-AQ` controllers all land in **[19.5%, 27.0%]** — the same band as Exp A's INT2/INT4 anchors.
> 5. **PNIG (the headline novelty metric) is NEGATIVE.** `acc(B9 OnlineNeed-Static) − max(acc(B6 StaticEntropy), acc(B8 OnlineResidual)) = 19.5 − 26.5 = −7.0 pp`. The multiplicative interaction *worsens* over its components rather than improving on them.
> 6. **Direction of static entropy is uninformative on Qwen long-video.** B6 (low-entropy → BF16) = 24.5% vs B7 (high-entropy → BF16) = 27.0%. Direction gap = −2.5 pp; flipped is *slightly* better. Opposite of the pi0.5 W2 finding (where low entropy was the right direction with a +27 pp gap), so the directionality claim does not transfer.
> 7. **Per-item OnlineResidual marginally beats StaticEntropy** (B8 26.5% > B6 24.5%) but the gap is ≪ bootstrap noise.
>
> **Diagnosis.** The 30 pp rescue gap is real, but at avg=4 with {INT2, BF16}, **86% of the cache must be at INT2**, and INT2 fundamentally breaks long-video VLM attention regardless of which 14% gets BF16 protection. The bottleneck is not which blocks to route to BF16 — it's that uniform INT2 alone is unrecoverably destructive in this model.
>
> **Implications for next steps.** The next step is *not* more routing. It's a richer tier set (e.g., {INT2, INT4, BF16} so 70%+ of the cache can sit at INT4 instead of INT2), or a stronger K/V quantizer baseline (KIVI-style per-channel K + per-token V; AKVQ-VL outlier reduction). Once uniform-INT4 is no longer at chance, routing to {INT4, BF16} or {INT2, INT4, BF16} can be revisited.
>
> **Exp C K/V isolation — STRONG ASYMMETRY at INT4.** On 100 stratified eval items per condition (frame budget unchanged at 64):
>
> 8. **K is the killer at INT4. V is essentially free at INT4.** C2.1 (K=BF16, V=INT4) lands at **53.0%** — within 2 pp of the BF16 ceiling (55.0% on the same 100 items) — and preserves **94.5%** of BF16-correct items. The mirror condition C2.2 (K=INT4, V=BF16) collapses to **29.0%**, only 8 pp above the A5 INT4-K/INT4-V floor. Δ vs A5 is +32.0 pp for C2.1 and +8.0 pp for C2.2. **The entire 30 pp rescue gap from Exp A is recoverable by leaving K alone.**
> 9. **At INT2, fragility flips and is per-side on both axes.** C2.3 (K=BF16, V=INT2) sits at **21.0%** (Δ vs A7 = +0.0 pp); C2.4 (K=INT2, V=BF16) at **33.0%** (Δ vs A7 = +12.0 pp). Neither crosses the rescue midpoint, but the *direction* of the asymmetry has reversed — V is the worse side at INT2. Mechanistically consistent with Exp A's surprise that A7 INT2 ternary slightly outperforms A5 INT4: ternary {−s, 0, +s} preserves K-row sign+scale exactly (which is what attention's "key match" needs), while INT2 V loses too much value-magnitude information for the attention×value matmul.
> 10. **Margin tracks accuracy.** C2.1 mean answer margin = +0.674 vs the 100-item BF16 ceiling +0.712. Logits are genuinely moving toward the right answer, not just flipping argmax. C2.2 margin = −0.850, basically tied with A5's −0.871.
>
> **Implication.** The next experiment is *not* a richer KV tier set, and *not* {INT2, INT4, BF16} routing. It's **K-side outlier reduction at INT4** (KIVI-style per-channel K, AKVQ-VL static outlier extraction, or post-RoPE channel-wise K calibration). The naive symmetric per-channel quantizer is broken specifically on K; once K is properly quantized, V can stay at INT4 with effectively no accuracy loss — that gives ~10 KV bits avg without the 30 pp Exp A collapse.
>
> **Exp D0 Evidence-window diagnostic — NEGATIVE for the all-pooled selector; one viable signal in the maxhead variant.** 200 eval items × 8 BF16 conditions, 64 frames, 8 windows × 8 frames each:
>
> 11. **Top-1-window-removed (0.560) is HIGHER accuracy than Full-64 (0.500).** Removing the supposedly most-attended visual window IMPROVES accuracy by 6 pp on the same 200 items. This is the opposite of the "attention identifies evidence" prediction.
> 12. **Top-1-window-removed (0.560) ≈ Random-window-removed (0.548 over 3 seeds).** No statistically meaningful difference. Among BF16-correct items (n=100), top-removal flips 6 to wrong; removing any one of 3 random windows flips 12. Random removal is *more* damaging.
> 13. **EvidenceCausalGap = TopCausalEffect − RandomCausalEffect** has median 0.000, mean 0.011 across 200 items. Sign of the gap is uncorrelated with whether BF16 gets the right answer.
> 14. **Mechanism: LM attention sink at the first visual token.** `top1_window_all` collapses to window 0 in 195/200 items (97.5%). Mean per-window mass = [0.245, 0.127, 0.111, 0.105, 0.100, 0.097, 0.098, 0.115]. Window 0 wins by a razor-thin margin because heads dump no-op attention on the first visual token after `<|vision_start|>`; raw-mass pooling cannot distinguish "this head spreads visual mass across windows" from "this head sinks all of its tiny visual mass on token 0". `evidence_width_90` = 7 of 8 windows in 195/200 items.
> 15. **`visual_mass_total` median = 0.061** — the answer-query position averages only ~6% of its attention on visual tokens (~94% on text/system/options/instruction). Below-the-fold visual usage is the regime where the sink artifact dominates.
> 16. **Maxhead diagnostic shows sharper localization.** Per-(L, h) normalize-pick gives `top1_window_maxhead` distribution {0: 135, 1: 9, 2: 1, 3: 4, 4: 4, 5: 6, 6: 4, 7: 37}: window-0 win rate drops to 67.5%, with a clear secondary mode at window 7 (recency, 18.5%). Median maxhead top-1-mass = 0.563 vs 0.245 for the all-pooled selector. Maxhead picks concentrate on layers 6, 9, 21, 24 (~56% of items).
> 17. **Mid-layer pooling does NOT help** — agreement with all-pooled = 96.5% (both pick window 0). Selector agreement: all == maxhead = 69.5%, mid == maxhead = 67.0%, all == mid = 96.5%.
> 18. **Per-bucket Full-64 BF16 accuracy is 6.5 pp below Exp A's 56.5%** — short 0.515 (vs 0.667), mid 0.818 (vs 0.848), long 0.493 (vs 0.537), very_long 0.343 (vs 0.403). Most likely cause: `Qwen2VLImageProcessor` defaulting to the "fast processor" path (transformers warning at load time). All D1 conditions inherit this shifted ceiling, so within-D1 comparisons remain valid.
>
> **Implications.** The all-pooled selector (the user's deliberate primary choice — raw-mass pooling, since per-head normalization would let text-focused heads dominate) is hijacked by the sink. The fix is *not* to switch the pooling rule (raw mass remains correct for distinguishing visual-vs-text heads); it's to **drop the first ~32 sink tokens before window-mass computation** OR to use the maxhead variant. D1 has been extended in flight to test top-1-maxhead and top-2-maxhead as additional conditions (D1.5a_mh, D1.5b_mh) alongside the original top-1/top-2-all conditions. The headline D1 prediction is now: **D1.5a-mh > D1.6a (random) > D1.5a-all on localized items**, while **D1.5a-all ≈ D1.7a (uniform window 4)** because both effectively protect window 0. The D1.4 (all visual K BF16) vs D1.3 (text K BF16, visual K INT4) comparison is independent of window selection and remains the foundational visual-K-matters test.

The methodology gates: BF16 vs INT2-KV first-token-logit perturbation ‖Δ‖∞ = 18.7 (smoke threshold 1e-3) — `FakeQuantKVCache.update()` feeds the SDPA matmul as designed. Diagnostic pass produces 0 NaN entropy/residual rows on 33,600 (item × L × H) signals; `bf16_pred ≠ uniform_int2_pred` on 90%+ of items, confirming the routing decisions are operating on real signal.

## Context

Pivot from the saturated pi0.5 / LIBERO line. There, `Static-W2-l13-17` already hits 100% on standard LIBERO Long (and ~92% on the n=64 partial Pareto sweep — see `EXPERIMENT_FINDINGS.md`), leaving no rescue gap for AttnEntropy gating to demonstrate value. Long-video VLMs are unsaturated (Qwen2.5-VL-7B reports 54.7% on LongVideoBench-val), and long videos produce thousands of visual tokens so KV-cache memory and precision are the active bottleneck. Recent literature (MEDA, AKVQ-VL, VidKV, MadaKV) frames KV-quant as the open problem.

**Research question.** Can attention entropy allocate per-(layer, head, token) KV-cache *precision* better than uniform / random / attention-mass / MEDA-style baselines on long-video VLM inference?

The thesis splits into two halves:
- **Experiment A** establishes that uniform KV quantization is non-trivially harmful — i.e., that there's a "rescuable regime" where BF16 is correct but uniform-quantized is wrong, with enough headroom for a controller to recover.
- **Experiment B** tests whether AttnEntropy V1/V2 controllers, with thresholds frozen on a 100-item calibration set, beat uniform / random / MEDA-style allocation at matched ~3 average KV bits on a held-out 200-item evaluation set.

## Setup

- **Model:** `Qwen/Qwen2.5-VL-7B-Instruct` — 28 decoder layers, 28 query heads, 4 KV heads (GQA: each KV head shared by 7 Q heads), `head_dim=128`. AWQ checkpoint also pulled (`Qwen/Qwen2.5-VL-7B-Instruct-AWQ`).
- **Benchmark:** LongVideoBench validation split (1337 items; mixed 4-way and 5-way MCQ).
- **Stratified split** (seed=0): 100 calibration / 200 evaluation, per-bucket counts:
  | Bucket | Duration | n cal | n eval |
  |---|---|---:|---:|
  | short | 0–15s | 17 | 33 |
  | mid | 15–60s | 17 | 33 |
  | long | 60–600s | 33 | 67 |
  | very_long | 600–3600s | 33 | 67 |
- **Frame budget:** 64 frames per video at `max_pixels=360×420` → ~3000–3500 visual tokens per item.
- **Scoring:** greedy decode, `max_new_tokens=1`, then 4- or 5-way logprob over `["A","B","C","D",("E")]` token ids; predict `argmax`.
- **Hardware:** tambe-server-1 H100 80 GB (GPU 0), shared with another user's ~50 GB co-tenant.

## Core primitive: `FakeQuantKVCache`

A subclass of `transformers.cache_utils.DynamicCache`. On every `update(K, V, layer_idx)` call, K and V are fake-quantized to an N-bit grid using **per-channel symmetric quantization along `head_dim`** (`group_size=128`). Tensors stay BF16 in storage; values are rounded to the integer grid.

Why this affects prefill logits: Qwen2.5-VL's SDPA forward feeds the **return value** of `cache.update(...)` into the attention matmul, so the rounded tensors are what hit the SDPA kernel. Validated end-to-end on 4 text prompts and 1 video item: ‖Δ_logits‖∞ = 18.7 between BF16 and INT2-KV first-token logits (well above the 1e-3 smoke threshold).

`BitController` modes:
- **V1** — scalar bits per layer (Exp B's primary AttnEntropy variant)
- **V2** — `[num_kv_heads]=[4]` bits per layer (per-KV-head allocation under GQA)
- **V3** — per-token protected mask + hi/lo bits *(deferred; needs eager attention runtime hook for live entropy)*

Quant math is lifted directly from `scripts/utils.py:fake_quantize_module`:
```
qmax = 2 ** (bits − 1) − 1
g    = x.reshape(..., -1, group_size)
s    = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
x_hat = ((g / s).round().clamp(-qmax, qmax) * s).reshape_as(x)
```
At `bits=2` this collapses to ternary {−s, 0, +s}. At `bits=8` the FP8 path uses an E4M3 round-trip cast (or falls through to INT8 grid if PyTorch's FP8 isn't available). At `bits>=16` it's a no-op.

## Experiment A — KV-quant sensitivity (final, n=200 per condition)

| # | Weights | KV cache | n | acc | 95% CI | avg KV bits | BF16-correct preserved | Δ vs BF16 |
|---|---|---|---:|---:|---|---:|---:|---:|
| A1 | BF16 | BF16 | 200 | **0.565** | [0.495, 0.635] | 16.00 | 1.000 | — |
| **A2** | **W4 fake-quant** | BF16 | 200 | **0.540** | [0.475, 0.610] | 16.00 | 0.867 | **−2.5 pp** |
| **A3** | **AWQ checkpoint** | BF16 | 200 | **0.540** | [0.470, 0.605] | 16.00 | 0.885 | **−2.5 pp** |
| A4 | BF16 | FP8 KV | 200 | 0.255 | [0.195, 0.315] | 8.00 | 0.257 | −31.0 pp |
| A5 | BF16 | INT4 KV | 200 | 0.210 | [0.155, 0.265] | 4.00 | 0.204 | −35.5 pp |
| A6 | BF16 | INT4-K / INT8-V | 200 | 0.250 | [0.190, 0.310] | 6.00 | 0.292 | −31.5 pp |
| A7 | BF16 | INT2 ternary | 200 | 0.270 | [0.210, 0.330] | 2.00 | 0.265 | −29.5 pp |
| A8 | AWQ | INT4 KV | 200 | 0.255 | [0.195, 0.315] | 4.00 | 0.292 | −31.0 pp |

### Per-duration-bucket breakdown (final)

| Condition | short (n=33) | mid (n=33) | long (n=67) | very_long (n=67) |
|---|---:|---:|---:|---:|
| A1 BF16 | 0.667 | 0.848 | 0.537 | 0.403 |
| A2 W4 fake | 0.636 | 0.848 | 0.478 | 0.403 |
| A3 AWQ | 0.576 | 0.848 | 0.537 | 0.373 |
| A4 FP8 KV | 0.394 | 0.212 | 0.269 | 0.194 |
| A5 INT4 KV | 0.303 | 0.242 | 0.164 | 0.194 |
| A6 INT4-K/INT8-V | 0.303 | 0.242 | 0.209 | 0.269 |
| A7 INT2 ternary | 0.273 | 0.364 | 0.254 | 0.239 |
| A8 AWQ + INT4 KV | 0.303 | 0.242 | 0.164 | 0.328 |

### Key findings

1. **Weight-only quantization is essentially lossless.** A2 (W4 fake-quant) and A3 (real AWQ) both land at 54.0% — identical to one decimal, only 2.5 pp below BF16. The two preserve different items (BF16-correct preservation: A2=86.7%, A3=88.5%) but converge on the same aggregate accuracy. **The action is not in weight quantization.**

2. **Uniform KV quantization is catastrophic at every bit width tested.** All four KV-only conditions (A4 FP8, A5 INT4, A6 INT4-K/INT8-V, A7 INT2) collapse to 21–27% — at or near 4-way chance. A bigger drop than published long-video KV literature (VidKV reports ~5 pp drop at sub-2-bit with their static method) — confirms uniform allocation is the wrong lever.

3. **Asymmetric K/V doesn't rescue.** A6 (INT4-K / INT8-V) sits at 25.0% — between A4 FP8 (25.5%) and A5 INT4 (21.0%) and well within bootstrap CI of both. Bumping V to 8-bit recovers ~4 pp vs full INT4, but K is the dominant bottleneck and full uniform precision can't be saved by per-tensor asymmetry.

4. **INT2 ternary slightly outperforms INT4 and FP8.** A7 (27.0%) > A4 (25.5%) > A5 (21.0%). Counterintuitive at first, but mechanistically plausible: ternary {−s, 0, +s} preserves *sign + scale* of each (head, token, channel) row exactly, which is precisely what attention's "which key matches" structure depends on. Coarser non-adaptive 8-bit (FP8 E4M3 with 3-bit mantissa) and even per-channel INT4 lose more relative-magnitude information than ternary.

5. **The rescuable regime is enormous.** With BF16 at 56.5% and any uniform KV-quant at ~25%, AttnEntropy V1/V2 in Exp B has ~30 pp of headroom to recover. This is the strongest "rescue regime" in any of our experiments to date — orders of magnitude wider than the LIBERO line's 4-trial rescuable bucket on standard LIBERO at W4.

6. **Combining weight-quant with KV-quant follows KV-quant.** A8 (AWQ + INT4 KV, final n=200) sits at 25.5% — within bootstrap CI of A5 (INT4 KV, 21.0%) and matching A4/A6 exactly. Weight-quant doesn't compound the KV penalty; KV-quant is the dominant cost. The combined condition lands cleanly in the KV-quant cluster, not below it.

7. **Per-bucket pattern**: BF16 shows the expected duration gradient (short > mid > long > very_long, modulo the mid-bucket anomaly noted below). All KV-quant conditions flatten this gradient — precision loss hurts every duration roughly equally, not selectively at long durations. This argues against the simple hypothesis that "long videos need more bits". *A counterargument to the duration-aware allocation strategy in MEDA, which we'll test directly via baseline B4.*

8. **A8's `very_long` bucket is anomalously high (32.8%)** vs the rest of A8 (long: 16.4%, short: 30.3%, mid: 24.2%). This is one of the few places where AWQ's activation-aware weight calibration interacts with KV-quant differently. Likely a small-n stratification effect (n=67 in `very_long`); the aggregate (25.5%) is firmly in the KV-quant cluster.

### Anomaly: mid bucket is harder *easier* than short on BF16

A1 BF16 has acc 0.667 on short (n=33), 0.848 on mid (n=33), 0.537 on long (n=67), 0.403 on very_long (n=67). Mid being 18 pp above short is unexpected — typical assumption is monotonic harder-with-duration. Could be a small-n stratification artifact (n=33 per bucket); could also reflect that short clips on LongVideoBench are more often "needle-in-haystack" temporal localization questions (harder per-frame) while mid clips have cleaner narrative structure. Worth sanity-checking on the full validation set in a future run.

## Methodological lessons learned

Three issues caught and fixed during this run; documenting them so they don't bite the next person:

1. **`torch>=2.10` ships CUDA-13 wheels that silently fall back to CPU on driver 12.6.** First calibration run completed in 7 seconds because the model was on CPU — `torch.cuda.is_available()` returned False with a non-fatal warning. Pinned to `torch==2.5.1+cu124` in `setup_qwen_env.sh`.

2. **Qwen2.5-VL's SDPA forward returns `attn_weights=None` regardless of `output_attentions=True`.** The kwarg is vestigial in the unified attention forward (it's just an unused parameter). The entropy hook for calibration silently produced NaN. Calibration now loads with `attn_implementation="eager"` (committed in `calibrate.py`).

3. **Memory hygiene matters under co-tenant pressure.** A1 finished cleanly at 200/200 then A2 OOM'd 7 seconds later because PyTorch's allocator didn't return the cumulative cache to the OS. Added `gc.collect() + torch.cuda.empty_cache()` every 5 items inside `run_inference.run_condition`, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the resume run. Also wrote an in-place `fake_quantize_weights_w4` (no `saved` clone) so the W4 path doesn't peak at 28 GB during the swap.

4. **Chunked entropy computation** to avoid OOM during calibration: `_entropy_from_attn` now chunks over the query dim and accumulates a `[H]` FP32 buffer instead of materializing the full `[B, H, Q, K]` FP32 tensor (~3 GB at H=28, Q=K=3000).

## Experiment B — Online Precision-Need Routing (final, 2026-05-07)

After Exp A established the rescue gap, Experiment B was rewritten from the original static V1/V2 plan to test the online-precision-need hypothesis directly:

> `precision_need = semantic/query importance × quantization difficulty`
>
> Given a specific input video, can we identify which KV blocks are both important for the current answer and hard to quantize, and spend BF16 precision only there?

### Setup

- **Tier set:** {INT2, BF16}. FP8 dropped (Exp A's A4 collapsed to chance, so it's not a reliable rescue tier in our fake-quant impl).
- **Granularity:** layer × KV-head (V2). Qwen2.5-VL-7B has 28 × 4 = 112 blocks.
- **Budget:** target avg = 4 KV bits → `p_BF16 = (4 − 2) / (16 − 2) = 14.3%` → **16 of 112 blocks at BF16, 96 at INT2**.
- **Architecture (two-pass per item):**
  1. **Diagnostic pass** (BF16 + eager attention) on 100 cal + 200 eval items at 64 frames. Captures per-(layer, KV-head): `entropy_mean` (over all queries), `entropy_answer_query` (last query position), `aq_topk_mass` (top-32 mass at last query), `kv_residual_int2` (Frobenius residual ‖K − Q2(K)‖_F / ‖K‖_F under simulated INT2). Also runs uniform INT4 / INT2 forwards to bake `bf16_pred`, `uniform_int4_pred`, `uniform_int2_pred` into the JSONL.
  2. **Routed eval pass** (SDPA, fast). For each (condition, item), compute per-(layer, KV-head) score → top-16 → BF16, rest → INT2 → V2 BitController → score the MCQ.
- **Split safety:** `StaticEntropyRisk` aggregated from cal-rows only (`split == "cal"` assert in `aggregate_static_risk`); eval signals used only as per-item online inputs.

### Conditions

All routed conditions allocate exactly 16 of 112 (L, h) blocks to BF16. Same 200 eval items as Exp A.

| ID | Method | Score per (L, h) | Source |
|---|---|---|---|
| A1 reused | BF16 KV (ceiling) | — | Exp A |
| A5 reused | Uniform INT4 KV (matched-avg anchor) | — | Exp A |
| A7 reused | Uniform INT2 KV (floor) | — | Exp A |
| B2 | Random-V2 (3 seeds) | random | per-seed |
| B4 | MEDA-style layer entropy | layer-mean entropy → top-4 layers BF16 | cal |
| B6 | StaticEntropy-V2 | percentile_rank(−mean entropy) (low → BF16) | cal |
| B7 | FlippedEntropy-V2 (symmetry control) | percentile_rank(+mean entropy) (high → BF16) | cal |
| B8 | OnlineResidual-V2 | kv_residual_int2 | per-eval-item |
| B9 | **OnlineNeed-Static-V2** | percentile_rank(static_low) × percentile_rank(residual) | cal × per-item |
| B10 | **OnlineNeed-AQ-V2** | percentile_rank(−aq_topk_mass) × percentile_rank(residual) | per-item |

### Results (n=200 per condition; A1/A5/A7 reused from Exp A)

| ID | Condition | n | acc | 95% CI | avg KV bits | BF16-pres | flip-rec INT4 | flip-rec INT2 | damage |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| A1 | BF16 (ceiling) | 200 | **0.565** | [0.495, 0.635] | 16.00 | 1.000 | — | — | — |
| A5 | Uniform INT4 | 200 | 0.210 | [0.155, 0.265] | 4.00 | 0.204 | — | — | — |
| A7 | Uniform INT2 | 200 | 0.270 | [0.210, 0.330] | 2.00 | 0.265 | — | — | — |
| B2 | Random seed=0 | 200 | 0.255 | [0.195, 0.320] | 4.00 | 0.260 | 0.253 (n=83) | 0.244 (n=78) | 0.706 (n=17) |
| B2 | Random seed=1 | 200 | 0.200 | [0.145, 0.255] | 4.00 | 0.260 | 0.265 (n=83) | 0.205 (n=78) | 0.765 (n=17) |
| B2 | Random seed=2 | 200 | 0.245 | [0.185, 0.305] | 4.00 | 0.210 | 0.181 (n=83) | 0.128 (n=78) | 0.647 (n=17) |
| B4 | MEDA-style layer entropy | 199 | 0.236 | [0.181, 0.291] | 4.00 | 0.182 | 0.220 (n=82) | 0.156 (n=77) | 1.000 (n=17) |
| B6 | StaticEntropy-V2 (low → BF16) | 200 | 0.245 | [0.185, 0.310] | 4.00 | 0.310 | 0.337 (n=83) | 0.333 (n=78) | 0.824 (n=17) |
| B7 | FlippedEntropy-V2 (high → BF16) | 200 | 0.270 | [0.210, 0.330] | 4.00 | 0.300 | 0.301 (n=83) | 0.269 (n=78) | 0.706 (n=17) |
| B8 | OnlineResidual-V2 | 200 | 0.265 | [0.205, 0.330] | 4.00 | 0.280 | 0.289 (n=83) | 0.282 (n=78) | 0.765 (n=17) |
| **B9** | **OnlineNeed-Static-V2** | 200 | **0.195** | [0.140, 0.250] | 4.00 | 0.210 | 0.229 (n=83) | 0.192 (n=78) | 0.882 (n=17) |
| **B10** | **OnlineNeed-AQ-V2** | 200 | **0.210** | [0.150, 0.270] | 4.00 | 0.200 | 0.193 (n=83) | 0.167 (n=78) | 0.765 (n=17) |

### Headline metrics

| Metric | Formula | Value |
|---|---|---:|
| BF16 ceiling | acc(A1) | 0.565 |
| Uniform INT4 anchor (matched avg=4) | acc(A5) | 0.210 |
| Random mean (B2 ×3 seeds) | mean over seeds | 0.233 |
| Static-vs-random gap | acc(B6) − mean(B2) | +1.2 pp |
| **Direction gap** | acc(B6) − acc(B7) | **−2.5 pp** *(flipped slightly better — opposite of pi0.5)* |
| OnlineResidual-vs-static | acc(B8) − acc(B6) | +2.0 pp |
| **OCG** (online conditioning gain) | acc(B9) − acc(B6) | **−5.0 pp** |
| **PNIG** (precision-need interaction gain) | acc(B9) − max(acc(B6), acc(B8)) | **−7.0 pp** |
| **AQ-vs-Static interaction** | acc(B10) − acc(B9) | **+1.5 pp** *(answer-query-driven routing slightly less destructive than static-entropy-driven, but both worst overall)* |
| Best routed acc | max over B2-B10 | 0.270 (B7 FlippedEntropy) |
| Worst routed acc | min over B2-B10 | 0.195 (B9 OnlineNeed-Static) |

### Diagnosis

The 30 pp rescue gap (BF16 56.5% → uniform INT4/INT2 ~25%) is real, but at avg=4 with the {INT2, BF16} tier set:
- **86% of the cache must be at INT2** (96/112 blocks), and INT2 is unrecoverable for long-video VLM attention regardless of which 14% gets BF16 protection.
- All 8 routed conditions land in [19.5%, 27.0%] — within the bootstrap CI of A5 (INT4) and A7 (INT2).
- The multiplicative interaction (B9, B10) doesn't help and slightly hurts — selecting blocks that are "important AND hard to quantize" concentrates BF16 in a tight subset that doesn't compensate for the 86% INT2 floor.
- StaticEntropy direction is uninformative (B6 ≈ B7 within noise), so the pi0.5 W2 directionality (low entropy → high sensitivity) does not transfer.
- BF16-correct preservation across routed conditions: 0.182–0.310. Even the best preservation (B6/B7 at 0.30+) means 70% of BF16-correct items still flip wrong under the routed quantization — confirming that 96/112 INT2 blocks corrupt attention enough to break most originally-correct answers.

### Implication for next steps

Not "smarter routing" — **a richer tier set or stronger baseline quantizer**.

1. **Richer tier set: {INT2, INT4, BF16}.** At avg=4 with this set, most blocks can sit at INT4 (which already collapses but is *less* destructive per-block than INT2), and a small fraction at BF16. Preliminary expectation: if INT4-anchored routing recovers some accuracy, the routing signal becomes meaningful — not because the signals improved, but because the floor is no longer at chance.
2. **KIVI-style asymmetric K/V layout.** Per-channel K + per-token V (currently we use per-channel along head_dim for both). The KIVI paper reports this matters for video-LLMs.
3. **AKVQ-VL-style outlier reduction.** The catastrophic INT2/INT4 collapse on Qwen2.5-VL+long-video may be driven by post-RoPE K-channel outliers; standard outlier-aware static methods are documented to recover most of that gap.

The Exp B framework, signals, and metrics are reusable: once a less-destructive baseline quantizer is in place, drop it into `fake_quantize_kv` and re-run `run_expB_online.sh` — same diagnostic JSONL, same routing logic, same summary table.

## Experiment C0 — no-compute diagnostics (2026-05-07)

Three diagnostics run on existing JSONLs, before committing to any further compute. Full output at `qwen/results/expC0_diagnostics.md`.

1. **A5 (INT4) ↔ A7 (INT2) item-level complementarity.** Of 200 eval items, INT4 was correct on 42, INT2 on 54, both on 13. Symmetric difference = 70 items (35%). Jaccard = 0.157 (near-disjoint). **Oracle union ceiling = 41.5%**, vs A5 alone 21.0% and A7 alone 27.0%. → A {INT2, INT4, BF16} tier set has real, non-trivial headroom (+14.5 pp over best single-tier anchor) *if* a router can pick the right tier per block.

2. **Selected-block coverage for B6/B8/B9/B10.** Every routed method protects ≤ 13 of 28 layers; B6 StaticEntropy is fully deterministic (16 stable / 96 never blocks across all 200 items, by construction). B10 OnlineNeed-AQ is the most spread (head-Hbits = 4.47 / 6.81 uniform) and shifts to earlier layers (L1-L11) — but still failed at 21.0%. **Concentration is descriptively true but mechanistically irrelevant**; spreading didn't rescue accuracy.

3. **Per-condition answer margin.** All routed methods have *negative* paired Δ-margin vs uniform INT4/INT2 floors on the BF16-correct subset: B6 −0.170, B8 −0.181, B9 −0.157, B10 −0.269 (vs A5). **Routing isn't even directionally helpful at avg=4 with {INT2, BF16}** — the routed logits sit *further* from the correct answer than uniform INT4/INT2 do on the same items.

**C0 conclusion:** sharpens the next-step claim from "richer tier set" to "richer tier set is *necessary* — without it routing has no signal headroom to express, regardless of method." Motivates Exp C K/V isolation as the cheapest test that could redirect the research path.

## Experiment C — K/V isolation mini-sweep (2026-05-07)

A6 in Exp A (INT4-K + INT8-V at 25.0%) tested asymmetric K/V but still quantized **both** sides, so it could not isolate which side is the actual fragility driver. Exp C runs four conditions that each leave one of K or V at full BF16 precision and quantize the other, on the same 100-item stratified eval subset (proportional bucket counts: short 16 / mid 16 / long 34 / very_long 34 = 100). Frame budget 64, model `Qwen2.5-VL-7B-Instruct`.

### Conditions and results (n=100 each, 64 frames)

| ID | K bits | V bits | avg KV bits | n | acc | 95% CI | BF16-correct preserved | mean margin |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| C2.1 `BF16K_INT4V` | 16 | 4 | 10.00 | 100 | **0.530** | [0.430, 0.630] | 0.945 (n=55) | +0.674 |
| C2.2 `INT4K_BF16V` | 4 | 16 | 10.00 | 100 | 0.290 | [0.210, 0.380] | 0.218 (n=55) | −0.850 |
| C2.3 `BF16K_INT2V` | 16 | 2 | 9.00 | 100 | 0.210 | [0.140, 0.290] | 0.182 (n=55) | −1.277 |
| C2.4 `INT2K_BF16V` | 2 | 16 | 9.00 | 100 | 0.330 | [0.240, 0.420] | 0.364 (n=55) | −0.820 |

### Paired comparison vs Exp A anchors (restricted to the same 100 item_ids)

| Condition | n | acc | 95% CI | mean margin |
|---|---:|---:|---|---:|
| A1 BF16 ceiling | 100 | 0.550 | [0.460, 0.640] | +0.712 |
| A5 INT4-K + INT4-V | 100 | 0.210 | [0.130, 0.290] | −0.871 |
| A6 INT4-K + INT8-V | 100 | 0.280 | [0.200, 0.370] | −1.033 |
| A7 INT2-K + INT2-V | 100 | 0.210 | [0.130, 0.290] | −1.092 |

### Diagnosis — two-regime asymmetry

**At INT4 (avg = 10 KV bits): K-fragile.** C2.1 (K=BF16, V=INT4) at 0.530 is within 2 pp of the BF16 ceiling on the same 100 items (0.550), with 94.5% BF16-correct preservation and answer margin +0.674 vs ceiling +0.712. The mirror C2.2 (K=INT4, V=BF16) at 0.290 sits only 8 pp above the A5 INT4/INT4 floor — keeping V at BF16 does almost nothing if K is at INT4. Δ vs A5: **+32.0 pp** for C2.1, **+8.0 pp** for C2.2. **The full Exp-A 30 pp rescue gap is recoverable simply by leaving K at full precision.**

**At INT2 (avg = 9 KV bits): asymmetry flips, but both sides break.** C2.3 (K=BF16, V=INT2) collapses to 0.210 (Δ vs A7 = +0.0 pp); C2.4 (K=INT2, V=BF16) sits at 0.330 (Δ vs A7 = +12.0 pp, BF16-correct preserved 0.364). Neither crosses the rescue midpoint of 0.380, but the *direction* of the asymmetry is reversed — at 2-bit, V is the *worse* side to quantize. This is mechanistically consistent with Exp A's surprise that A7 INT2 ternary (27.0%) slightly outperforms A5 INT4 (21.0%): ternary {−s, 0, +s} preserves K-row sign+scale per channel, which is exactly what attention's "key match" structure depends on. INT2 V loses too much value-magnitude information for the attention×value matmul.

### Implications

1. **The naive symmetric per-channel quantizer is broken specifically on K at INT4.** V at INT4 is essentially free in this setting. Until that is fixed, *any* routing scheme that tolerates symmetric INT4 K-quantization will lose 30 pp regardless of which subset of (layer, head) blocks it elevates.
2. **Next experiment is K-side outlier handling, not richer tiers and not routing.** Concrete options, in order of cost: (a) KIVI-style asymmetric K/V layout (per-channel K, per-token V); (b) AKVQ-VL static outlier extraction on K only; (c) post-RoPE K-channel calibration. All target the K-fragility specifically and should drop the symmetric INT4-K floor (currently 21.0%) into a regime where the C2.1 result (53.0%) is recoverable at avg ≈ 4 bits, not avg ≈ 10.
3. **Exp B's routing infrastructure is reusable.** Once a K-fragility-aware quantizer is plugged into `fake_quantize_kv`, the diagnostic JSONL + scoring + V2 BitController stack from Exp B can be re-run unchanged. The bottleneck has been the per-side fragility floor, not the routing signals.

### Files of record (Exp C)

- `qwen/scripts/expA_baseline.py` — extended with 4 C2 conditions and `--stratified_limit` flag.
- `qwen/scripts/run_expC_kv_isolation.sh` — single-step orchestrator (modeled on `run_resume.sh`).
- `qwen/scripts/expC_analyze.py` — paired-comparison analysis on the 100-item C2 universe.
- `qwen/scripts/expC0_diagnostics.py` — no-compute diagnostics from existing JSONLs.
- `qwen/results/expA_rollouts_Qwen2.5-VL-7B-Instruct.jsonl` — appended with 400 C2 rows; total now 8 × 200 + 4 × 100 = 2000 rows.
- `qwen/results/expA_summary_Qwen2.5-VL-7B-Instruct.md` — combined A1-A8 + C2.1-C2.4 table.
- `qwen/results/expC_kv_isolation_summary.md` — paired-on-100 comparison + diagnosis.
- `qwen/results/expC_kv_isolation.progress.log` — full server progress log.
- `qwen/results/expC0_diagnostics.md` — no-compute diagnostic output.

## Experiment D0 — Evidence-window diagnostic (2026-05-08)

After Exp C established that **K is the killer at INT4 and V is essentially free**, the natural next question shifts from "how do we route bits across layers" to a VLM-specific framing:

> **Does answer-query attention identify the visual evidence windows that matter for retrieval, so that we can selectively protect those K positions at higher precision?**

D0 is the BF16-only evidence-restriction half of the combined D0+D1 pipeline. It runs 8 BF16 forwards per item on the same 200 LongVideoBench eval items at 64 frames, captures per-(layer, KV-head) answer-query attention via an `EvidenceWindowAttentionHook`, and produces three pooled selectors over 8 temporal windows (8 frames each):

- `evidence_attn_all` — raw visual-window mass pooled across all 28 layers × 4 KV heads, normalized over windows. **Primary selector.**
- `evidence_attn_mid` — same but layers 8–20 only (sensitivity diagnostic for "do middle layers carry sharper cross-modal alignment").
- `evidence_attn_maxhead` — per-(L, h) normalize-over-windows, pick the (L, h) with the sharpest top-1 mass. Per-head upper-bound diagnostic.

Six conditions per item:

| ID | Condition | Frames the model sees |
|---|---|---|
| D0.1 | Full-64 BF16 (eager attention; captures attention) | 64 |
| D0.2 | Uniform-16 BF16 | 16 uniformly sampled |
| D0.3 | Top-1-window-only BF16 | 8 frames (the top-attended window via `evidence_attn_all`) |
| D0.4 | Top-2-windows-only BF16 | 16 frames (top-2 attended windows) |
| D0.5 | Top-1-window-removed BF16 | 56 frames (drop top window) |
| D0.6 | Random-window-removed BF16 (×3 seeds, exclude top) | 56 frames |

Frame manipulation v1 = frame removal (sequence length and temporal positions change; `mode="frame_removal_v1"` in JSONL). Blank-in-place v2 deferred behind a flag.

### Per-condition results (n=200 eval items)

| Condition | acc | 95% CI | mean answer margin |
|---|---:|---|---:|
| D0.1 Full-64 BF16 | **0.500** | [0.430, 0.570] | +0.646 |
| D0.2 Uniform-16 BF16 | 0.500 | [0.435, 0.570] | +0.326 |
| D0.3 Top-1-window-only | 0.425 | [0.360, 0.490] | −0.097 |
| D0.4 Top-2-windows-only | 0.520 | [0.450, 0.590] | +0.473 |
| **D0.5 Top-1-window-removed** | **0.560** | [0.495, 0.630] | **+0.675** |
| D0.6 Random-window-removed (3 seeds, n=600) | 0.548 | — | +0.686 |

### Per-bucket Full-64 BF16 accuracy

| Bucket | Exp A (slow processor) | Exp D0 (fast processor) | Δ |
|---|---:|---:|---:|
| short (n=33) | 0.667 | 0.515 | −15.2 pp |
| mid (n=33) | 0.848 | 0.818 | −3.0 pp |
| long (n=67) | 0.537 | 0.493 | −4.4 pp |
| very_long (n=67) | 0.403 | 0.343 | −6.0 pp |
| **all (n=200)** | **0.565** | **0.500** | **−6.5 pp** |

The drop from 56.5% → 50.0% on the same 200 items is unexpected and is probably the `Qwen2VLImageProcessor` "fast processor" change (transformers default since the last update; warning printed at model load: *"loaded as a fast processor by default, even if the checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs"*). The mid-bucket-easier-than-short anomaly is reproduced (0.818 > 0.515). All D1 conditions will see this same shifted ceiling, so within-D1 comparisons are still well-defined; the absolute gap-vs-Exp-C ceiling has just narrowed.

### Headline finding — answer-query attention does NOT identify causal evidence windows

Three independent signals converge on the same conclusion: **the all-layer raw-mass-pooled answer-query attention does not pick the visual windows that actually carry evidence.**

**1. Top-1-removed (0.560) is HIGHER accuracy than Full-64 (0.500).** Removing the supposedly most-attended visual window IMPROVES accuracy by 6 pp. If attention identified evidence, removal would *lower* accuracy.

**2. Top-1-removed (0.560) ≈ Random-removed (0.548).** No statistically meaningful difference. Random-window removal hurts about as much (or as little) as top-window removal.

**3. EvidenceCausalGap = TopWindowCausalEffect − RandomWindowCausalEffect** has median 0.000, mean 0.011 across 200 items, and the sign of the gap is uncorrelated with whether BF16 gets the right answer (positive-gap items: 50.0% bf16-correct; negative-gap items: 49.5% bf16-correct).

Among **BF16-correct items (n=100)** specifically: top-1-removal flips **6 items** to wrong, while removing **any one of 3 random windows** flips **12 items** to wrong. Removing the top-attended window is *less* damaging than removing a random window — directly opposite to the "attention identifies evidence" prediction.

### Mechanism — the LM attention-sink artifact at window 0

`top1_window_all` lands on **window 0 in 195/200 items (97.5%)**, with windows 1–7 as runners-up only 5 times total:

```
top1_window_all distribution: {0: 195, 1: 2, 2: 1, 7: 2}
```

But the win is a *razor-thin margin*, not a confident pick. Mean per-window mass across all 200 items:

```
evidence_attn_all   = [0.245, 0.127, 0.111, 0.105, 0.100, 0.097, 0.098, 0.115]
evidence_attn_mid   = [0.274, 0.123, 0.106, 0.100, 0.096, 0.093, 0.094, 0.113]
evidence_attn_maxhead = [0.414, 0.117, 0.065, 0.063, 0.061, 0.062, 0.070, 0.148]
```

`evidence_width_90` = 7 of 8 windows in 195/200 items. The "winner" (window 0) has only 24.5% mass on average, max 35.4%; window 7 has a consistent 11–15% recency bump. The remaining six windows are essentially uniform at 9–13%.

This is the well-documented **LM attention sink** ([Xiao et al., "Attention Sinks", 2023](https://arxiv.org/abs/2309.17453)): many transformer heads dump attention on the first token of any segment as a no-op when they have nothing semantic to attend to. With raw-mass pooling across all (L, h), the heads with sinks at the first visual token (immediately after `<|vision_start|>`) dominate the average and put a mild but consistent excess mass on window 0. Pooling raw mass (the user's deliberate choice — it correctly downweights heads that attend mostly to text) cannot distinguish *"this head spreads visual attention across windows"* from *"this head sinks all of its tiny visual attention on the first visual token."*

The mid-layer pooling makes the sink WORSE, not better (0.274 vs 0.245 on window 0): mid layers (8–20) have more attention sinks, not more semantic localization. Selector agreement: `top1_all == top1_mid` in 96.5% of items — they almost always pick the same window (window 0), just for different head-distribution reasons.

### Maxhead diagnostic — there ARE sharp heads, just not on average

The per-(L, h) max-head selector is the cleaner test of "does any single head localize evidence?":

```
top1_window_maxhead distribution: {0: 135, 1: 9, 2: 1, 3: 4, 4: 4, 5: 6, 6: 4, 7: 37}
maxhead_top_mass: median = 0.563, q75 = 0.629, max = 0.912
```

Window-0 win rate drops from 97.5% (all-pooled) to **67.5% (maxhead)** — still sink-dominated, but with a long tail and a clear secondary mode at window 7 (the last temporal window, 18.5% of items). The selected (L, h) is concentrated on a handful of layers:

| Layer | n items | % |
|---|---:|---:|
| 6 | 38 | 19.0% |
| 9 | 34 | 17.0% |
| 21 | 24 | 12.0% |
| 24 | 16 | 8.0% |
| 22 | 15 | 7.5% |
| 16 | 11 | 5.5% |
| 15 | 10 | 5.0% |
| 27 | 10 | 5.0% |

Layers 6, 9, 15, 16, 21, 22, 24, 27 cover **~79% of all maxhead picks**, and median top-1-mass at the maxhead is **56.3%** vs 24.5% for the all-pooled selector. So the mechanism is: the model *does* contain heads that localize sharply on a single visual window for any given item, but they're a minority that gets washed out when you average raw mass across all 28 × 4 = 112 heads. Selector agreement: `top1_all == top1_maxhead` in only 69.5% of items, so maxhead picks a *different* window from the all-pooled selector for nearly a third of items.

### Visual mass total — the answer query is mostly text-driven

The auxiliary `visual_mass_total = mean over (L, h) of the head's total visual-window mass`:

| Statistic | Value |
|---|---:|
| min | 0.0449 |
| q25 | 0.0566 |
| median | 0.0610 |
| q75 | 0.0673 |
| max | 0.0912 |

The answer-query position averages **6.1% of its attention on visual tokens** and ~94% on text (system prompt, question, options, instruction). This is the regime where the attention-sink artifact is dominant: 6% mass × small variations across 8 windows means the sink at window 0 (a constant ~24% of that 6%) easily wins by default. Items with `visual_mass_total > 0.07` represent the top quartile where the answer-query is genuinely doing visual retrieval — those are the items most likely to have meaningful evidence-window structure.

### Evidence-label distribution (post-hoc thresholds)

`expD_analyze.py` applies the threshold rules:

| Label | Rule | n | % |
|---|---|---:|---:|
| localized | full64 ✓ ∧ (top1_only or top2_only ✓) ∧ Δmargin > 0.5 ∧ EvidenceCausalGap > 0.2 | 11 | 5.5% |
| global | full64 ✓ ∧ uniform16 ✓ ∧ \|Δmargin\| < 0.3 | 19 | 9.5% |
| distributed | full64 ✓ ∧ top1_only ✗ ∧ top2_only ✗ | 2 | 1.0% |
| attention_not_causal | TopCausalEffect ≤ RandomCausalEffect | 49 | 24.5% |
| unlabeled | (full64 ✗, mostly) + boundary cases | 119 | 59.5% |

Among the 100 BF16-correct items, the largest category is **attention_not_causal (49 items)** — top-window removal hurts no more than random. Only 11/200 items qualify as **localized** under the strict thresholds, which is a small population for the planned D1.5a stratified test. Loosening the thresholds in `expD_analyze.py` is supported (labels are computed post-hoc, not baked into D0's JSONL) but will not change the headline diagnosis.

### Implications for D1

The D1 pipeline (currently in flight) tests the cross-modal K/V quantization hypothesis: V always at INT4, K varied between BF16/INT4 across text vs visual vs evidence-window subregions. The D0 finding reshapes what each D1 condition will tell us:

1. **D1.4 (all visual K = BF16, V INT4) vs D1.3 (text-K BF16 + visual K INT4, V INT4)** — independent of window selection. This pair tests "does visual K matter at all" and is unaffected by the attention-sink finding. **The most important D1 comparison.**

2. **D1.5a (top-1 visual K BF16, all-pooled selector) vs D1.6a (random-1 visual K BF16)** — under the all-pooled selector, D1.5a will protect window 0 in 195/200 items. This collapses to "does protecting K at the first visual token (the sink) recover anything specifically?" and will likely come out flat because the sink isn't carrying evidence. **Expected to be a null/negative result.**

3. **D1.5a-mh (top-1 visual K BF16, maxhead selector) vs D1.6a** — added to D1 mid-flight after D0's diagnosis. Uses `top1_window_maxhead` (window 0 in only 67.5% of items, with a meaningful tail). This is the cleaner test of "does attention pick causal evidence" and is predicted to outperform random *if* the maxhead variant is genuinely picking evidence. **The headline VLM-specific test.**

4. **D1.5b vs D1.6b** at top-2 budget — same logic at a more permissive budget.

### Remediation paths if D1.5a-mh also fails

If even the maxhead-derived window selection doesn't beat random, the next experiments are:

- **Sink correction.** Drop the first N visual tokens (the sink) before pooling and before window selection. Concretely: compute window mass over `[v_start + 32 : v_end]` instead of `[v_start : v_end]`. Quick post-hoc analysis on the saved `evidence_attn_*` data.
- **Sub-window K-mask.** Even within "window 0", mask only the first ~32 sink tokens at INT4 while protecting the rest of the window at BF16. Tests whether the issue is the entire window or just the sink prefix.
- **Sharp-head selection at calibration time.** Use the cal-100 split to identify (L, h) heads whose evidence-window distribution is sharpest and most predictive of BF16-correct outcomes, then use those heads' window picks for D1.

### Files of record (Exp D0)

- `qwen/scripts/expD0_evidence_diagnostic.py` — Phase 1 driver; 8 BF16 forwards/item, captures per-(L, h) answer-query attention.
- `qwen/scripts/visual_tokens.py` — `find_visual_token_span` (locates `<|vision_start|>`/`<|vision_end|>`) + `build_window_token_ranges`.
- `qwen/scripts/frame_manip.py` — `decode_uniform_frames` with decord/imageio/cv2 fallback; window-index helpers; blank-in-place v2 stub.
- `qwen/scripts/expD_smoke.py` — 5-check smoke (visual span, window mapping, V3K logits-differ, mask-cache alignment, frame-removal end-to-end).
- `qwen/scripts/run_expD.sh` — orchestrator (`smoke | d0 | d1 | analyze | full`).
- `qwen/results/expD0_evidence_diagnostic.jsonl` — 200 per-item rows; full schema in `expD0_evidence_diagnostic.py` docstring.
- `qwen/results/expD0_summary.md` — auto-generated by `expD_analyze.py`.
- `qwen/results/expD_pipeline.progress.log` + `expD0_evidence_diagnostic.progress.log` — server progress logs.

## Methodological lessons learned (cumulative)

1. **`torch>=2.10` ships CUDA-13 wheels that silently fall back to CPU on driver 12.6.** Pinned to `torch==2.5.1+cu124`. Calibration "completed" in 7 seconds the first time because the model was on CPU.
2. **Qwen2.5-VL's SDPA forward returns `attn_weights=None` regardless of `output_attentions=True`.** The kwarg is vestigial. Diagnostic pass loads with `attn_implementation="eager"` to capture attention weights.
3. **Memory hygiene under co-tenant pressure.** A1 finished cleanly then A2 OOM'd 7 seconds later (cumulative cache fragmentation). Added `gc.collect() + torch.cuda.empty_cache()` every 5 items inside `run_inference.run_condition`. Also wrote in-place `fake_quantize_weights_w4` (no `saved` clone) to avoid a 28 GB peak during the swap.
4. **Chunked entropy for long sequences.** `_entropy_from_attn` chunks over the query dim; full FP32 attention tensor at H=28, Q=K=3000 is ~3 GB and OOMs under tlandeg's 50 GB co-tenant. Chunked accumulator is ~86 MB.
5. **Eval-leakage discipline in routing experiments.** `aggregate_static_risk` hard-asserts `split == "cal"`. Easy to get wrong; should always be a code-level invariant.
6. **Auto-detect `(num_layers, num_kv_heads)` from the loaded model** rather than CLI defaults — Qwen2.5-VL-3B is 36×2, Qwen2.5-VL-7B is 28×4, and a hardcoded default silently corrupts the V2 BitController shape on the wrong model.

## Pipeline status

```
qwen-expB-online (tmux session, GPU 0) — PIPELINE COMPLETE (4h 8min total)
├── ✅ STEP 1  diagnostic pass on cal+eval (33 + 66 min, eager attention)
├── ✅ STEP 2  static_entropy_risk aggregated from cal-only (n_cal=100, 2 sec)
├── ✅ STEP 3  routed eval — 8 of 8 routed conditions complete (B2 ×3, B4, B6, B7, B8, B9; ~17 min each)
└── ✅ STEP 4  B10 OnlineNeed-AQ (diagnostic upper bound, 13:46)

qwen-expC (tmux session, GPU 0) — PIPELINE COMPLETE (~22 min total)
└── ✅ STEP 1  K/V isolation: 4 conditions × 100 stratified eval items × 64 frames
              C2.1 BF16K/INT4V → 0.530, C2.2 INT4K/BF16V → 0.290,
              C2.3 BF16K/INT2V → 0.210, C2.4 INT2K/BF16V → 0.330

qwen-expD-d0 (tmux session, GPU 0) — PIPELINE COMPLETE (1h 26min total)
└── ✅ STEP 1  Evidence-window diagnostic: 200 eval items × 8 BF16 conditions × 64 frames
              Full-64=0.500, Uniform-16=0.500, Top-1-only=0.425, Top-2-only=0.520,
              Top-1-removed=0.560, Random-removed (3 seeds, n=600)=0.548.
              EvidenceCausalGap median=0.000 — attention does NOT identify evidence
              under the all-pooled selector (sink artifact at window 0 dominates).
              Maxhead diagnostic: window-0 win rate drops to 67.5% (median top-mass 0.563).

qwen-expD-d1 (tmux session, GPU 0) — IN FLIGHT
└── 🟡 STEP 2  Cross-modal K/V quantization: 200 eval items × 14 conditions × 64 frames
              D1.3 (text-K BF16 / visual-K INT4 / V INT4) ... D1.7b
              + D1.5a_mh / D1.5b_mh (maxhead-derived windows; added in-flight)
              Pace ~30s per item; ETA ~1h40m total wall.
```

## Files of record

**Core primitives:**
- `qwen/scripts/fake_quant_kv_cache.py` — `BitController` (V1/V2 modes) + `FakeQuantKVCache` (DynamicCache subclass with per-(layer, KV-head) bit assignment).
- `qwen/scripts/run_inference.py` — MCQ scorer + memory hygiene every 5 items.
- `qwen/scripts/diagnostic_pass.py` — `DiagnosticCache` + `DiagnosticAttentionHook` + 3-forward-per-item diagnostic with reference predictions.
- `qwen/scripts/precision_need_scoring.py` — `aggregate_static_risk` (cal-only assertion), `compute_score` for 7 methods, `top_k_mask`, `bits_from_mask`.
- `qwen/scripts/expB_online.py` — Exp B online routing driver with auto-detection of model dims.
- `qwen/scripts/run_expB_online.sh` — orchestrator (diagnostic → static aggregate → routed eval).
- `qwen/scripts/smoke_expB_online.sh` — 3B + 5 cal/eval items + 32 frames smoke with hard assertions.

**Drivers (Exp A):**
- `qwen/scripts/expA_baseline.py` — 8-condition KV-quant sensitivity.
- `qwen/scripts/run_resume.sh` — orchestrator that cleanly recovered from the A1→A2 OOM.

**Results:**
- `qwen/results/expA_rollouts_Qwen2.5-VL-7B-Instruct.jsonl` — 1600 rollouts (8 × 200).
- `qwen/results/expA_summary_Qwen2.5-VL-7B-Instruct.md` — Exp A summary.
- `qwen/results/diagnostic_signals.jsonl` — 33,600 (item × layer × KV-head) signal rows (300 items × 28 × 4).
- `qwen/results/expB_online_rollouts.jsonl` — ~1600 routed rollouts (8 conditions × 200 items).
- `qwen/results/expB_online_summary.md` — Exp B summary table.
- `qwen/calibration/split_seed0.json` — frozen 100/200 stratified split.
- `qwen/calibration/static_entropy_risk.json` — frozen cal-only static entropy maps (low + high direction).

**Cross-experiment references:**
- `EXPERIMENT_FINDINGS.md` — pi0.5/LIBERO findings + Path 1 interim n=64 results.
- Plan: `/Users/subha/.claude/plans/you-can-make-a-parallel-crown.md`.
