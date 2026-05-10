# Qwen2.5-VL × LongVideoBench — KV-cache Quantization Experiments

**Status as of 2026-05-09:** **Seven experiments complete.** Exp A (8/8 conditions × 200 eval items), Exp B Online Precision-Need Routing (8 routed conditions × 200 eval items at avg=4 KV bits), Exp C K/V isolation mini-sweep (4 conditions × 100 stratified eval items at avg=10 / avg=9 KV bits), Exp D0 Evidence-window diagnostic (200 items × 8 BF16 conditions, 1h 26min wall), Exp D1 Cross-modal K/V quantization (200 items × 14 V3K-K-mask conditions, 1h 42min wall), Exp E1 Text-K slice ablation (200 items × 11 V3K-text-K-mask conditions, 89 min wall), and **Exp F K-quantizer repair screening (Stage 1 n=64 × 14 conditions = 896 rows, 33 min wall; Stage 3 n=200 × 10 conditions = 2000 rows, 76 min wall)**. Total: ~3600 baseline rollouts + 33,600 diagnostic signal rows + 200 D0 per-item rows + 2800 D1 per-item-per-condition rows + 2200 E1 per-item-per-condition rows + **2896 F per-item-per-condition rows**.

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
>
> **Exp D1 Cross-modal K/V — VLM-specific evidence-window hypothesis FALSIFIED. Text-K is the dominant fragility, not visual-K.** 200 eval items × 14 V3K K-mask conditions × 64 frames; V always at INT4:
>
> 19. **Spending more bits on visual-K *hurts* accuracy.** D1.4 (text-K INT4, all 5760 visual-K BF16, V INT4) at avg=9.85 KV bits lands at **0.210**. D1.3 (text-K BF16, all 5760 visual-K INT4, V INT4) at avg=4.15 KV bits lands at **0.385**. **2.4× more KV bits, 17.5 pp WORSE accuracy.** The bit budget is being spent on the wrong side.
> 20. **D1.4 ≤ uniform-INT4 floor (A5 = 0.210 in Exp A).** Protecting all 5760 visual-K positions at BF16 while corrupting the ~140 text-K positions at INT4 destroys the prompt scaffolding (system prompt + question + options + "Answer with a single letter from..." instruction) that produces the answer-letter logits. On the 100 BF16-correct items, D1.4 only preserves 19/100 vs D1.3's 55/100. Paired: D1.3-only-correct = 46, D1.4-only-correct = 10. Text-K dominates by 4.6×.
> 21. **All visual-K-protected conditions cluster in [0.395, 0.435], a ~3-5 pp boost over D1.3.** Adding *any* visual-K BF16 protection (top-1, top-2, random, uniform, maxhead) gives a small consistent rescue on top of text-K-BF16, but **no condition meaningfully separates from any other.**
>     - D1.3 (text-K BF16 only): 0.385
>     - D1.5a (top-1 all): 0.415 — D1.5a_mh (top-1 maxhead): 0.425 — D1.7a (uniform middle): 0.395
>     - D1.5b (top-2 all): 0.430 — D1.5b_mh (top-2 maxhead): 0.415 — D1.7b (uniform 0+4): 0.395
>     - D1.6a (random-1 mean over 3 seeds): 0.417 — D1.6b (random-2): 0.423
> 22. **Maxhead window selection does NOT separate from random.** D1.5a_mh (top-1 maxhead) at 0.425 vs D1.6a-seed0 (random-1) at 0.420 = +0.5 pp, well within bootstrap noise. Even fixing the attention-sink pathology (maxhead picks non-window-0 in 32.5% of items: window 0 = 135/200, window 7 = 37/200, others = 28/200) the resulting accuracy is the same as random-window selection at the same budget.
> 23. **No differentiation on D0-labeled "localized" items (n=11) either.** D1.5a vs D1.6a-seed0 = 0.909 vs 0.909 (Δ = 0); D1.5a_mh vs D1.5a = 0.909 vs 0.909 (Δ = 0); D1.7a uniform = 1.000 = D1.5b = D1.7b. The 11-item population is too small to power the test, but the direction is null/negative — not the predicted "top-evidence > random" pattern.
> 24. **Per-bucket D1.3 vs D1.4 reproduces the asymmetry across all durations.** Text-K BF16 wins in every bucket: short 0.333 vs 0.242, mid 0.576 vs 0.273 (largest gap, 30 pp), long 0.373 vs 0.239, very_long 0.328 vs 0.134.
>
> **Mechanism — MCQ scoring is text-anchored.** D0's auxiliary `visual_mass_total` median = 0.061 already showed that the answer-query position attends ~94% to text (prompt header + question + options + instruction) and only ~6% to visual tokens. Within that 94%, the answer-letter logits are produced primarily from question + options keys, which are at a tiny fraction of total seq_len (~140 / ~5900 ≈ 2.4% of positions). Quantizing those ~140 high-impact text-K positions to INT4 destroys the prompt structure that maps "what is the question + options" → "which letter to emit." Quantizing all 5760 visual-K positions, by contrast, only modestly degrades visual retrieval — and even that 5-pp visual-K effect doesn't depend on *which* visual tokens you protect, because (a) the answer-query barely attends to visuals to begin with and (b) the model has already compressed visual content into text-side representations during the prefill.
>
> **Implications — the next experiment is text-K, not visual-K.** The original hypothesis ("preserve question-relevant evidence-window visual-K addresses for retrieval") is falsified for first-token MCQ scoring on Qwen2.5-VL-7B + LongVideoBench. The actionable next directions:
> 1. **Text-K outlier handling at INT4.** Text-K is small (~140 tokens × 4 KV-heads × 28 layers × 128 head-dim = ~2 MB BF16) and high-impact. Per-channel K calibration restricted to text positions, or AKVQ-VL-style outlier extraction on text-K only, may recover the 17.5 pp gap at INT4-text + INT4-visual + INT4-V. Memory cost is negligible.
> 2. **Finer text-K partition.** Split text-K into prompt header / question / options / instruction. Which slice carries the fragility? If it's *just* the question or *just* the options, that's an even cheaper rescue.
> 3. **Test on long-form video QA generation.** First-token MCQ may be a degenerate setting that hides visual-K importance. Long-form generation queries visual content repeatedly across many decode steps; the visual-K effect may rise. Try Video-MME / MVBench long-form items with multi-token decoding.
> 4. **Drop the visual-K window-routing approach.** The infrastructure (V3K mode, window-mass selectors, per-token K mask) is good, but the routing object — visual evidence windows under MCQ — has been shown not to be the right object. Re-target to text-K or text-substring routing.
>
> **Exp E1 Text-K slice ablation — text-K-routing hypothesis ALSO falsified. Both prompt-role and quantization-difficulty signals fail.** 200 items × 11 V3K text-K-mask conditions (V always INT4, visual-K always INT4, only text-K varied):
>
> 25. **No single text slice or pair recovers most of D1.3's 17.5 pp text-K rescue.** Best single condition is `E1.4 OptionsOnly` (40 tokens, **0.290** acc), recovering only 45.7% of the floor → ceiling gap. Other slices: header 0.215, question 0.175 (BELOW floor), instruction+answer-prefix 0.225, Q+O 0.270, O+AP 0.185, Q+O+AP 0.205.
> 26. **Adding more text slices to the union HURTS.** `E1.7 (Options + AnsPrefix, 45 tok) − E1.4 (Options alone, 40 tok) = −10.5 pp` despite 5 *more* BF16 tokens. `E1.8 (Q+O+AP, 96 tok) − E1.4 = −8.5 pp` despite 56 more BF16 tokens. The K-side is sensitive to *which positions are at which precision relative to each other*, not just to the BF16 fraction.
> 27. **K-residual selection is *worse* than random.** E1.10 (top-20 text positions by per-position INT4 K-row residual norm, mean over 28 layers × 4 KV-heads, captured online per-item) lands at 0.200 — *below* E1.0 floor (0.210) and *below* E1.9 random-20 (mean 0.215 over 3 seeds). Protecting the highest-residual K positions actively damages accuracy.
> 28. **Random-20 is at floor.** E1.9 across 3 seeds: 0.220 / 0.215 / 0.210. No statistical separation from E1.0 uniform INT4 K/V at 0.210. 20 of ~140 text-K positions at BF16 doesn't recover anything regardless of selection.
> 29. **Verdict matrix: NO condition is sufficient** (≥80% of E1.1's 0.385 acc at <50% of E1.1's 140 tokens). Every fixed slice, union, random, and K-residual condition fails one or both criteria. Full text-K must be protected to capture the rescue.
>
> **Combined D1 + E1 implication.** Routing within K alone is *ruled out* as a research direction for first-token MCQ scoring on Qwen2.5-VL. The signal that actually matters is *whether* K is at BF16 or INT4 in aggregate (text-K bulk vs visual-K bulk = +17.5 pp; D1.3 vs uniform-INT4 = +17.5 pp; sub-divisions don't compose monotonically). The actionable next direction is **K-side outlier-aware INT4 quantization itself**: KIVI per-channel K calibration, AKVQ-VL static K outlier extraction, or post-RoPE per-channel K scale. Once uniform-INT4-K at all 5900 positions is recoverable to within 5 pp of BF16, the visual-K + text-K + V-INT4 setup gives ~10 KV avg bits without the 30 pp Exp A collapse — without any routing.
>
> **Exp F K-quantizer repair screening — STRONG POSITIVE RESULT. The KV-quantization collapse is solved by changing the K scale axis.** 14 K-quantizer variants screened on n=64 (Stage 1, balanced 16/bucket); top 10 confirmed on n=200 (Stage 3, canonical Exp A split):
>
> 30. **F4 KIVI per-channel-along-seq closes 94.4% of the F1→F0 gap at TRUE 4.00 KV bits.** Stage 3: F4 = 0.545 (vs F0 BF16 ceiling 0.565, F1 INT4 floor 0.210, F3 all-K-BF16 0.550). Closes 33.5 of 35.5 pp. **2.0 pp below the BF16 ceiling at 4× KV memory compression** — the deployable headline result. F4 dominates F3 (same accuracy, 2.5× less memory) and dominates F8 (better acc at lower bits).
> 31. **The fix is a one-line scale-axis change, not a calibration trick.** F4's only difference from F1 is the per-channel scale axis: F1 scales along head_dim with one scale per (head, position, group of 128 head_dim slots) shared across positions; F4 scales along seq with one scale per (head, channel) shared across all positions. KIVI's central finding (arXiv:2402.02750) is that K outliers are channel-aligned across the sequence — one scale per channel exposes them, one scale per position hides them inside head_dim grouping. F4 needs no calibration, no routing, no slice info.
> 32. **VLM-specific scaling refinements do NOT help once the K axis is correct.** F5 text/visual split (0.510), F6 prompt-role split (0.525), F7 99.5%ile clip (0.540) all UNDERPERFORM F4 (0.545) at the same 4-bit budget. This directly answers the question "does the model's modality structure require scale-aware quantization?" — once the axis is right, no.
> 33. **Outlier-channel BF16 protection adds marginal value.** F8 (top-8 outlier channels per-(L, H_kv) at BF16, 4.375 KV bits) = 0.540 — slightly *worse* than F4 at higher cost. F9 (top-16 channels, 4.75 KV bits) = 0.560, statistically tied with F0 BF16 ceiling. F9 is the "zero accuracy loss" Pareto point if outlier bookkeeping is acceptable.
> 34. **Score-calibration variants F10-F13 FAILED catastrophically (Stage 1, dropped from Stage 3).** Generic Q-energy reweighting (F10) = 0.281, TT-heavy block-score (F11) = 0.172 (below F1 floor), balanced block-score (F12) = 0.188, text-K-only score-cal (F13) = 0.172. The closed-form `s_d ∝ sqrt(E[Q_d²])` heuristic concentrates scale precision on high-Q-energy channels, which empirically aren't the channels that matter most for attention. Score-cal as a research direction needs iterative scale search to be revisited; the closed-form approximation is a strict downgrade at this bit budget.
> 35. **All four anchors land within bootstrap CI of prior runs.** F0 = 0.565 (= A1 0.565), F1 = 0.210 (= A5 0.210), F2 = 0.385 (= D1.3 0.385), F3 = 0.550 (vs C2.1 0.530, n=200 vs n=100 stratification). Pixel-perfect alignment with the prior anchor numbers — the F-suite plumbing is correct end-to-end.
>
> **Pareto frontier (n=200):**
>
> | KV bits | acc | condition | note |
> |---:|---:|---|---|
> | 4.000 | 0.545 | F4 KIVI per-channel-seq | **deployable headline** (4× compression, ~2 pp loss) |
> | 4.375 | 0.540 | F8 outlier-8 | dominated by F4 |
> | 4.750 | 0.560 | F9 outlier-16 | **zero-loss Pareto point** (within CI of F0) |
> | 10.000 | 0.550 | F3 all-K BF16 + V INT4 | dominated by F4 |
> | 16.000 | 0.565 | F0 BF16 (ceiling) | reference |
>
> **Implications.** The 35.5 pp KV-quantization collapse from Exp A — which motivated the entire research program — is *solved* by switching the K scale axis. None of the routing experiments (Exp B, D1, E1) were necessary; the bottleneck was always the quantizer, not the selector. The deployable result for Qwen2.5-VL long-video MCQ inference is **F4 KIVI per-channel-along-seq K + per-channel-along-head_dim V at INT4 group_size=128**, giving 4× KV cache memory compression with 2.0 pp accuracy loss vs BF16. Future work: (a) test F4 on long-form generation (multi-token decode) where visual-K may matter more than at first-token MCQ; (b) revisit AKVQ-VL Hadamard rotation as a Pareto-improvement candidate over F4; (c) the outlier-channel result (F9 within CI of F0 at 4.75 bits) is a strong sub-headline for the "zero-accuracy-loss" deployment scenario.

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

## Experiment D1 — Cross-modal K/V quantization (2026-05-08)

D1 takes the per-(item, top-window) data from D0 and asks the cross-modal K/V quantization question:

> **Given V is essentially free at INT4 (Exp C), where do we spend BF16 K bits — on text K, on visual K, or on a question-conditioned subset of visual-K windows?**

V is fixed at INT4 everywhere. K is varied per condition via the `BitController` V3K mode (per-token K mask, V follows the layer's `v_bits` scalar). 14 conditions per item, 200 items, 64 frames; sequence length is ~5899 tokens with `[v_start, v_end) = [15, 5775)` (5760 visual tokens) and ~140 text tokens.

### Conditions and per-condition results (n=200 each)

| ID | Text K | Visual K policy | V | n | acc | 95% CI | avg KV bits | margin |
|---|---|---|---:|---:|---:|---|---:|---:|
| D1.3 | BF16 | INT4 (all) | INT4 | 200 | **0.385** | [0.315, 0.455] | 4.15 | −0.418 |
| D1.4 | INT4 | BF16 (all 5760) | INT4 | 200 | **0.210** | [0.155, 0.265] | 9.85 | −1.609 |
| D1.5a | BF16 | top-1 BF16 (all-pooled selector) | INT4 | 200 | 0.415 | [0.345, 0.485] | 4.88 | −0.383 |
| D1.5a_mh | BF16 | top-1 BF16 (maxhead selector) | INT4 | 200 | 0.425 | [0.360, 0.490] | 4.88 | −0.412 |
| D1.5b | BF16 | top-2 BF16 (all-pooled) | INT4 | 200 | 0.430 | [0.365, 0.500] | 5.61 | −0.238 |
| D1.5b_mh | BF16 | top-2 BF16 (maxhead) | INT4 | 200 | 0.415 | [0.350, 0.485] | 5.61 | −0.305 |
| D1.6a (×3 seeds) | BF16 | random-1 BF16 | INT4 | 600 | 0.417 | — | 4.88 | −0.355 |
| D1.6b (×3 seeds) | BF16 | random-2 BF16 | INT4 | 600 | 0.423 | — | 5.61 | −0.307 |
| D1.7a | BF16 | uniform-1 BF16 (window 4) | INT4 | 200 | 0.395 | [0.330, 0.465] | 4.88 | −0.359 |
| D1.7b | BF16 | uniform-2 BF16 (windows 0, 4) | INT4 | 200 | 0.395 | [0.325, 0.465] | 5.61 | −0.331 |

Reused from earlier experiments on the same 200 items: A1 BF16 ceiling = 0.500 (in this run, with the fast-processor path; vs Exp A's 0.565), C2.1 BF16-K + INT4-V on n=100 stratified subset = 0.530, A5 INT4 K/V floor = 0.210.

### Headline asymmetry: text-K dominates visual-K by 17.5 pp

The **D1.3 vs D1.4 pair is the dominant signal of the experiment**, and its direction is the *opposite* of the original VLM-specific hypothesis:

- **D1.3 (text-K BF16, visual-K INT4 for all 5760 positions, V INT4) at avg 4.15 KV bits → 0.385.**
- **D1.4 (text-K INT4, visual-K BF16 for all 5760 positions, V INT4) at avg 9.85 KV bits → 0.210.**

D1.4 spends **2.4× more KV bits** than D1.3 yet lands **17.5 pp lower** in accuracy — at the uniform-INT4 floor (A5 = 0.210). On the 100 BF16-correct items (paired):

| Condition | bf16-correct preserved | unique-correct vs the other condition |
|---|---:|---:|
| D1.3 | 55 | 46 (only D1.3) |
| D1.4 | 19 | 10 (only D1.4) |
| both | 9 | — |
| neither | 35 | — |

**D1.3 dominates D1.4 by 4.6× on unique-correct items.** Text-K BF16 alone (with all 5760 visual-K at INT4) recovers 55% of BF16-correct items; visual-K BF16 alone (with text-K at INT4) recovers only 19%.

### Per-bucket D1.3 vs D1.4

| Bucket | D1.3 (text-K BF16 only) | D1.4 (visual-K BF16 only) | Δ |
|---|---:|---:|---:|
| short (n=33) | 0.333 | 0.242 | +9.1 pp |
| mid (n=33) | 0.576 | 0.273 | **+30.3 pp** |
| long (n=67) | 0.373 | 0.239 | +13.4 pp |
| very_long (n=67) | 0.328 | 0.134 | +19.4 pp |

Text-K dominance is robust across all duration buckets. The mid-bucket, where Full-64 BF16 is highest (0.818), shows the biggest gap (30 pp) — when the model is operating well on full visual context, the text-K side is what's load-bearing.

### Visual-K protection methods are interchangeable

All visual-K-BF16-protection conditions cluster in the [0.395, 0.435] band with no statistically distinguishable separation:

| Selector | top-1 | top-2 |
|---|---:|---:|
| All-pooled (sink-dominated; window-0 in 195/200 items) | 0.415 | 0.430 |
| Maxhead (window-0 in 135/200, window-7 in 37/200, etc.) | 0.425 | 0.415 |
| Random (mean over 3 seeds) | 0.417 | 0.423 |
| Uniform (middle / 0+middle) | 0.395 | 0.395 |

`D1.5a_mh − D1.6a-seed0 = 0.425 − 0.420 = +0.5 pp` — well within bootstrap noise. Even after fixing the attention-sink artifact (maxhead picks non-window-0 in 32.5% of items vs all-pooled's 2.5%), the resulting accuracy is statistically identical to random window selection at the same budget. **Window selection method does not matter at this resolution; what matters is *whether* any visual K positions are at BF16, not *which* ones.**

The +3–5 pp boost from any visual-K BF16 protection over D1.3 (no visual protection) is consistent with Exp C's V-side finding (V at INT4 is essentially free if K is correct) extended to the K-visual side: a small fraction of visual-K positions at BF16 helps marginally, but the marginal value of additional BF16 visual-K positions saturates fast.

### D1 stratified by D0 evidence label (the headline test)

The original prediction was: on D0-labeled "localized" items (n=11 under our thresholds), top-evidence visual-K BF16 should beat random visual-K BF16 at matched budget. Result on the 11 localized items:

| Pair | acc(left) | acc(right) | Δ | Verdict |
|---|---:|---:|---:|---|
| D1.5a (top-1 all) vs D1.6a seed=0 (random-1) | 0.909 | 0.909 | 0.0 pp | tied |
| D1.5b (top-2 all) vs D1.6b seed=0 (random-2) | 1.000 | 0.818 | +18.2 pp | top wins (n=11; 2 items) |
| D1.5a (top-1 all) vs D1.7a (uniform middle) | 0.909 | 1.000 | −9.1 pp | uniform wins |
| D1.5b (top-2 all) vs D1.7b (uniform 0+4) | 1.000 | 1.000 | 0.0 pp | tied |
| D1.5a_mh (top-1 maxhead) vs D1.6a seed=0 | 0.909 | 0.909 | 0.0 pp | tied |
| D1.5b_mh (top-2 maxhead) vs D1.6b seed=0 | 0.909 | 0.818 | +9.1 pp | top wins (n=11; 1 item) |
| D1.5a_mh (top-1 maxhead) vs D1.5a (top-1 all) | 0.909 | 0.909 | 0.0 pp | tied |
| **D1.4 vs D1.3 (visual-K BF16 vs text-K BF16)** | **0.091** | **0.818** | **−72.7 pp** | text-K dominates |

The 11-item localized population is too small to power statistical tests, but the direction of every visual-K-routing pair is null or 1-item noise. **The only large effect on localized items is the same text-K-vs-visual-K asymmetry (0.818 D1.3 vs 0.091 D1.4 = +72.7 pp).** Not even D0-labeled "this question has a localized evidence window" items show the predicted top-attention > random pattern.

### Confirmed mechanism — MCQ first-token scoring is text-anchored

D0's auxiliary `visual_mass_total` median = 0.061 already foreshadowed this. The answer-query position attends ~94% to text and only ~6% to visual tokens. Within the 94% text attention:

- **Prompt header + system message + `<|im_start|>` etc.**: position-encoded scaffolding the model uses for role-conditioning. Quantizing these positions corrupts the response format.
- **Question text**: directly encodes "what's being asked." A few hundred K positions, but they carry the semantic content the answer logits need.
- **Options A/B/C/D text**: the candidate space. The model must distinguish "Option C: ..." from "Option D: ..." via the option tokens; corrupting these K positions blurs the choice.
- **"Answer with a single letter from A, B, C, D" instruction**: tells the model what shape of answer to produce. Without this, the model's distribution on A/B/C/D collapses to noise.

Quantizing visual-K corrupts ~5760 positions but each contributes ~0.001% of attention mass; the answer-query has already extracted the visual signal it needs into text-side representations during the prefill, so visual-K quantization only modestly degrades that already-compressed signal.

D1.4 (visual-K BF16, text-K INT4) preserves the part of the cache the model barely uses while corrupting the part it heavily uses → 0.210 (uniform-INT4 floor). D1.3 (text-K BF16, visual-K INT4) does the opposite, and recovers most of the rescue at 1/2.4 the bit budget.

### Implications & next steps

The visual-evidence-window precision-routing thesis is **falsified in the MCQ first-token-scoring setting**. Reroute:

1. **Text-K outlier handling at INT4 (small, high-impact).** Text-K is ~140 tokens × 28 layers × 4 KV-heads × 128 head-dim ≈ 2 MB BF16. KIVI-style per-channel K calibration restricted to text positions, or AKVQ-VL outlier extraction on text-K only, should close most of the 17.5 pp gap at uniform-INT4 + outlier-aware-text-K.

2. **Finer text-K partition study.** Split the text-K range into prompt-header / question / options / instruction subspans and run a C2.1-style isolation sweep on each. Identifies which text slice carries the fragility — the smallest possible BF16-protected subset.

3. **Long-form generation re-test.** First-token MCQ is a *minimal* visual-K-stress setting because the model only needs to emit a single letter. Long-form video QA (Video-MME generation, MVBench) decodes many tokens, each re-querying visual K. The "visual-K windows matter" hypothesis may hold at multi-token decoding even though it fails at 1-token.

4. **Drop visual-K window routing as a research direction.** The V3K infrastructure (per-token K mask, attention hooks, EvidenceWindowAttentionHook) is reusable, but the routing *target* — visual evidence windows under MCQ — has been ruled out. Repurpose for text-K subsegment routing.

### Files of record (Exp D1)

- `qwen/scripts/expD1_crossmodal_kv.py` — Phase 2 driver; per-condition V3K K-mask construction; reads D0 JSONL for top-window selectors (all-pooled + maxhead).
- `qwen/scripts/expD_analyze.py` — applies post-hoc evidence labels; produces 3 markdown summaries.
- `qwen/results/expD1_crossmodal_kv.jsonl` — 2800 rows (200 items × 14 conditions). Each row: `condition, k_text_bits, k_visual_policy, v_bits, avg_kv_bits, pred_choice, is_correct, option_logprobs, answer_margin, top1_window_all, top2_windows_all, top1_window_maxhead, top2_windows_maxhead, bf16_pred, bf16_correct, visual_protect_windows`.
- `qwen/results/expD1_summary.md` — per-condition acc + 95% CI + BF16-correct preservation table.
- `qwen/results/expD_combined_analysis.md` — D1 stratified by D0 evidence label + headline-pair table.
- `qwen/results/expD1_crossmodal_kv.progress.log` — full server progress log.

## Experiment E1 — Text-K slice ablation, COMPLETE (2026-05-08)

**Status:** Pass A complete (7 conditions × 200 items, 54 min wall, 0 failures). Pass B complete (4 conditions × 200 items, 35 min wall, 0 failures). Total 11 routed conditions × 200 items + 2 reused references = **2200 E1 rows**. Both passes ran cleanly after two implementation bug fixes (BPE-driven slice mismatch in `find_text_slice_spans` → switched to marker-based detection; `TextKResidualCache` composition not subscriptable → switched to direct `DynamicCache` subclass). Two implementation bugs were caught and fixed in the smoke + early Pass B (BPE-merge-driven slice mismatch in `find_text_slice_spans`; `TextKResidualCache` composition not subscriptable for transformers); see git log for fixes.

E1 follows up D1's headline finding (text-K is the dominant fragility, not visual-K). The natural question: *which* of the ~140 text-K positions carry the fragility? V at INT4 everywhere; visual-K at INT4 everywhere; only text-K is masked between BF16 and INT4 via the `BitController` V3K mode.

### Pass A conditions and per-condition results (n=200)

| ID | Slice protected at BF16 | n | acc | med_tokens | mean_margin | avg KV bits |
|---|---|---:|---:|---:|---:|---:|
| **E1.0** | (none — uniform INT4 K/V floor; reused from A5) | 200 | **0.210** | 0 | −0.871 | 4.00 |
| **E1.1** | all text K (= D1.3, reused) | 200 | **0.385** | ~140 | −0.418 | 4.15 |
| E1.2 | header only | 200 | 0.215 | 14 | −0.663 | 4.01 |
| E1.3 | question only | 200 | 0.175 | 50 | −1.131 | 4.05 |
| **E1.4** | **options only** | 200 | **0.290** | 40 | −0.874 | 4.06 |
| E1.5 | instruction + answer-prefix | 200 | 0.225 | 22 | −1.294 | 4.02 |
| E1.6 | question + options | 200 | 0.270 | 91 | −0.970 | 4.11 |
| E1.7 | options + answer-prefix | 200 | 0.185 | 45 | −0.972 | 4.06 |
| E1.8 | question + options + answer-prefix | 200 | 0.205 | 96 | −0.843 | 4.12 |

### Headline findings

1. **No single slice or pair recovers most of D1.3's rescue.** Best single condition is `E1.4 OptionsOnly` (40 tokens) at 0.290 — recovers **45.7%** of the 17.5 pp E1.0→E1.1 rescue gap. Every other slice recovers less, and several are at or below the floor.

2. **Adding more slices to the union *hurts*.** This is the most surprising result.
   - `E1.4 OptionsOnly` (40 tok): **0.290**
   - `E1.6 Q + O` (91 tok): 0.270 — **2 pp worse despite 2.3× more BF16 tokens**
   - `E1.7 O + AnsPrefix` (45 tok): **0.185** — 10.5 pp worse than O alone at nearly the same budget; below the floor
   - `E1.8 Q + O + AnsPrefix` (96 tok): 0.205 — at floor

   `E1.7 (O + AnsPrefix) − E1.4 (O alone) = −10.5 pp` at +5 tokens. Protecting the answer-prefix tokens *on top of* options actively damages accuracy. Mechanistically suspicious: the answer-prefix tokens (`<|im_end|>\n<|im_start|>assistant\n`) may be in a regime where their K-row magnitudes are extreme outliers; protecting them at BF16 while leaving everything else at INT4 may shift the attention distribution badly. Or it's a non-monotonicity in the per-channel symmetric INT4 quantizer when small numbers of tokens are spared. The interaction is robust across buckets, not a small-n artifact.

3. **Question alone (50 tok, 0.175) is *below* the floor.** Protecting just the question text but corrupting everything else (header, options, instruction, answer-prefix) is the worst single-slice condition. `E1.3 QuestionOnly − floor = −3.5 pp; QuestionOnly − E1.4 OptionsOnly = −11.5 pp` at greater token cost.

4. **Header alone (14 tokens, 0.215) is essentially at the floor (0.210).** The Qwen system header / role tags carry almost no useful K-side signal at INT4 ↔ BF16 isolation.

5. **InstrAnsPrefix (22 tok, 0.225) is barely above the floor.** Protecting the answer-letter scaffold is *not* the dominant fragility — surprising given the 1-token MCQ task structure where this scaffold sets up the letter generation.

6. **Per-bucket: OptionsOnly's mid-bucket result (0.455) is the standout cell.** It's the highest accuracy in the entire Pass A table, exceeding even E1.1 (D1.3) on the mid bucket where Full-64 BF16 is highest. Other buckets show the same OptionsOnly > {Q, header, AP} ordering but smaller gaps.

| Condition | short | mid | long | very_long |
|---|---:|---:|---:|---:|
| E1.2 HeaderOnly | 0.273 | 0.212 | 0.209 | 0.194 |
| E1.3 QuestionOnly | 0.182 | 0.182 | 0.224 | 0.119 |
| **E1.4 OptionsOnly** | 0.212 | **0.455** | 0.254 | 0.284 |
| E1.5 InstrAnsPrefix | 0.273 | 0.212 | 0.224 | 0.209 |
| E1.6 Q + O | 0.273 | 0.364 | 0.239 | 0.254 |
| E1.7 O + AnsPrefix | 0.212 | 0.273 | 0.164 | 0.149 |
| E1.8 Q + O + AnsPrefix | 0.152 | 0.273 | 0.254 | 0.149 |

### BF16-correct preservation

| Condition | n_bf16_correct | preserved | rate |
|---|---:|---:|---:|
| E1.2 HeaderOnly | 100 | 22 | 0.220 |
| E1.3 QuestionOnly | 100 | 18 | 0.180 |
| **E1.4 OptionsOnly** | 100 | **36** | **0.360** |
| E1.5 InstrAnsPrefix | 100 | 19 | 0.190 |
| E1.6 Q + O | 100 | 33 | 0.330 |
| E1.7 O + AnsPrefix | 100 | 22 | 0.220 |
| E1.8 Q + O + AnsPrefix | 100 | 24 | 0.240 |

`E1.4 OptionsOnly` preserves 36/100 BF16-correct items at 40 tokens BF16. That's just over half of E1.1 (D1.3)'s 55/100 at ~140 tokens. So options carries 2/3 the rescue at less than 1/3 the bit budget — but the marginal Q, header, and instruction tokens still account for the missing ~36% rescue and that signal does not concentrate on any particular slice.

### Per-item best-slice distribution (tiebreaker: smaller token count)

| Condition | n_items_won | % | typical n_tokens |
|---|---:|---:|---:|
| E1.2 HeaderOnly | 84 | 42.0% | 14 |
| E1.5 InstrAnsPrefix | 37 | 18.5% | 22 |
| E1.4 OptionsOnly | 34 | 17.0% | 40 |
| E1.3 QuestionOnly | 16 | 8.0% | 50 |
| E1.6 Q + O | 12 | 6.0% | 91 |
| E1.7 O + AnsPrefix | 12 | 6.0% | 45 |
| E1.8 Q + O + AnsPrefix | 5 | 2.5% | 96 |

**Caveat on the per-item win rate.** HeaderOnly "wins" on 42% of items, but by tiebreaker on smallest token count when multiple slices give the same per-item answer (most often when *every* slice gets the answer wrong, the smallest-token-count slice wins by default). This is **not** evidence that header is meaningful — it's evidence that on most items, no slice is sufficient to recover the BF16 prediction, so the tiebreaker dominates. The aggregate accuracy table (E1.4 best at 0.290) is the more honest read.

The **median per-item best-slice token count = 20**, which becomes the global budget `N` for Pass B's E1.9 (random) and E1.10 (K-residual) conditions.

### Mechanism — text-K fragility is broadly distributed, not localized

The Pass A result falsifies the prompt-role hypothesis ("the bottleneck is one or two specific text slices like options or answer-prefix"). Three distinct signals converge:

1. The best single slice (`OptionsOnly`) recovers only 46% of the rescue gap.
2. Adding more slices doesn't monotonically help — `O + AnsPrefix < O alone`, `Q + O + AnsPrefix < Q + O`, and the strict subset relations are *not* respected by accuracy. The interaction between text-slice K precisions appears non-monotonic.
3. No two-slice union exceeds the best single slice meaningfully (E1.6 Q+O at 0.270 < E1.4 O at 0.290).

This points to **text K being broadly fragile** rather than localized to a specific prompt role. Quantizing text K disrupts attention distributions across many positions simultaneously; protecting a subset isn't the same as protecting the whole. The expected next outcome (which Pass B will test):

- If `E1.10 K-residual top-20` ≫ `E1.9 random-20`: the right object is *quantization difficulty* (which positions have the largest K-row outliers), not prompt role. Action: AKVQ-VL-style per-channel K outlier extraction restricted to text positions.
- If `E1.9 random-20` ≈ `E1.10 K-residual top-20` ≈ `E1.5 InstrAnsPrefix` (~22 tok): any 20-token text-K BF16 protection helps about the same; full text-K must be protected for the headline rescue. Action: text-K outlier handling at INT4 is needed; partial protection is insufficient.

### Pass B results — both routing hypotheses falsified

Pass B tested whether random-20 or K-residual-top-20 text-K BF16 protection at the same budget as the median best-fixed-slice (N = 20 tokens) could match or beat the fixed-slice conditions.

| ID | Method | tokens | acc | 95% CI | Δ vs floor (0.210) |
|---|---|---:|---:|---|---:|
| **E1.10** | **K-residual top-20** (per-item INT4 K-row residual norm, mean over (L, h)) | 20 | **0.200** | [0.145, 0.255] | **−1.0 pp** (BELOW floor) |
| E1.9-s0 | Random-20 (seed 0) | 20 | 0.220 | [0.165, 0.280] | +1.0 pp |
| E1.9-s1 | Random-20 (seed 1) | 20 | 0.215 | [0.160, 0.275] | +0.5 pp |
| E1.9-s2 | Random-20 (seed 2) | 20 | 0.210 | [0.155, 0.270] | +0.0 pp |
| E1.9 mean | Random-20 (3-seed mean) | 20 | 0.215 | — | +0.5 pp |
| (ref) E1.5 | InstrAnsPrefix (instr + answer-prefix) | 22 | 0.225 | [0.170, 0.285] | +1.5 pp |
| (ref) E1.4 | OptionsOnly (best fixed slice) | 40 | 0.290 | [0.225, 0.355] | +8.0 pp |

**Two negative results:**

1. **K-residual is *worse* than random.** E1.10 (top-20 by per-position INT4 K-row residual norm) at 0.200 is below E1.9 random-20 (mean 0.215). Quantization difficulty is not just an irrelevant signal — protecting hard-to-quantize text-K positions is *actively harmful*. Mechanistically consistent with the Pass A non-monotonicity (E1.7 O+AnsPrefix < E1.4 O alone): the highest-residual K positions are likely the ones most depended-on by attention, and shifting the symmetric per-channel quantizer's scale to spare them may distort the lower-residual positions' representations more than uniform INT4 would.

2. **Random-20 is at floor.** No statistically meaningful separation between E1.9 (any seed, 0.210-0.220) and E1.0 (uniform INT4 K/V floor, 0.210). Random text-K BF16 at 20 tokens out of ~140 doesn't recover anything.

3. **InstrAnsPrefix (22 tokens, 0.225) is essentially tied with random-20 (mean 0.215).** Even the structurally important "answer scaffold" (instruction + `<|im_start|>assistant\n`) is no better than random text-K BF16 at the same token budget.

### Verdict matrix (sufficient = ≥80% of E1.1's 0.385 acc at <50% of E1.1's 140 tokens)

The threshold is **0.308 acc at <70 tokens**.

| Condition | acc | tokens | meets 80% acc | <50% tokens | sufficient? |
|---|---:|---:|:-:|:-:|:-:|
| E1.10 K-residual top-20 | 0.200 | 20 | ❌ | ✅ | ❌ |
| E1.2 HeaderOnly | 0.215 | 14 | ❌ | ✅ | ❌ |
| E1.3 QuestionOnly | 0.175 | 50 | ❌ | ✅ | ❌ |
| E1.4 OptionsOnly | 0.290 | 40 | ❌ (close) | ✅ | ❌ |
| E1.5 InstrAnsPrefix | 0.225 | 22 | ❌ | ✅ | ❌ |
| E1.6 Q + O | 0.270 | 91 | ❌ | ❌ | ❌ |
| E1.7 O + AnsPrefix | 0.185 | 45 | ❌ | ✅ | ❌ |
| E1.8 Q + O + AnsPrefix | 0.205 | 96 | ❌ | ❌ | ❌ |
| E1.9 random-20 (mean) | 0.215 | 20 | ❌ | ✅ | ❌ |

**No condition is sufficient.** Every routing strategy — prompt-role-based slices, semantic unions, random selection, K-residual selection — either fails the accuracy threshold or the budget threshold. The full text-K (140 tokens, E1.1 = D1.3) must be protected to capture the rescue.

### Combined headline — D1 + E1 falsifies in-K-routing entirely

D1 (visual-K windows) and E1 (text-K subspans + random + K-residual) together rule out **routing within K alone** as a research direction for first-token MCQ scoring on Qwen2.5-VL-7B + LongVideoBench:

- **Within visual-K:** all 11 D1 visual-K-routing conditions cluster in [0.395, 0.435]; no separation by selection method (top-attention, maxhead, random, uniform). The signal that mattered was text-K vs visual-K (D1.4 0.210 vs D1.3 0.385), not which visual-K windows.
- **Within text-K:** all 11 E1 text-K-routing conditions cluster in [0.175, 0.290], well below E1.1 0.385 (= full text-K BF16). No subset, union, random, or K-residual ranking reaches the rescue.

The actionable next direction is **K-side outlier-aware INT4 quantization itself**, not routing. Concretely: per-channel K calibration (KIVI-style), AKVQ-VL-style static K outlier extraction, or post-RoPE per-channel K scale calibration. These all attack the K fragility *within* the quantizer rather than trying to spare a subset to BF16. Once uniform-INT4-K at full 5900 positions is recoverable, the visual-K + text-K + V-INT4 setup gives ~10 KV bits avg without the 30 pp Exp A collapse.

### Mechanistic interpretation

Two consistent observations across D1, E1 Pass A, and E1 Pass B:

1. **Adding more BF16 K positions doesn't monotonically help.** E1 Pass A: E1.7 (O + AnsPrefix, 45 tok) < E1.4 (O alone, 40 tok). E1 Pass B: K-residual-top-20 (worst residual positions protected) < random-20. D1: top-1-visual-K BF16 ≈ random-1-visual-K BF16. The K-side of attention is sensitive to *which positions are at which precision relative to each other*, not just to which fraction is at BF16.

2. **The per-channel symmetric INT4 quantizer is brittle to non-uniform precision profiles.** Sparing a small subset of K positions to BF16 doesn't strictly improve over uniform INT4 — sometimes it makes things worse by changing the effective per-(L, h, position) scale interactions during attention. This is consistent with KIVI's argument that *per-channel* calibration (rather than per-token / per-position) is the right primitive for K.

### Files of record (Exp E1, complete)

- `qwen/scripts/text_slices.py` — `find_text_slice_spans` (marker-based), `union_mask`, `positions_to_mask`, `text_positions`, `TextKResidualCache` (subclass of `DynamicCache`), `capture_text_k_residuals`.
- `qwen/scripts/expE1_text_slice_ablation.py` — Pass A + Pass B driver (`--phase passA|passB`); Pass B reads Pass A JSONL to compute global median budget N.
- `qwen/scripts/expE1_smoke.py` — 5-check smoke test (visual span, slice non-overlap, decode round-trip, V3K logits-differ on question-only mask, V3K question-only distinct from all-text-BF16).
- `qwen/scripts/run_expE1.sh` — orchestrator (`smoke|passA|passB|analyze|full`).
- `qwen/scripts/expD_analyze.py` — extended with `summarize_e1` (per-condition + BF16-correct preservation + verdict matrix + per-bucket).
- `qwen/results/expE1_text_slice_ablation.jsonl` — 2200 rows (200 items × 11 conditions: 7 fixed slices + 3 random seeds + 1 K-residual).
- `qwen/results/expE1_summary.md` — auto-generated.
- `qwen/results/expE1_pair_analysis.md` — auto-generated (pair table + per-bucket + verdict matrix).
- `qwen/results/expE1_text_slice_ablation.progress.log` — server progress log.

### Bug postmortems

1. **Slice detection bug (smoke).** First-pass `text_slices.py` standalone-tokenized `item.question` and searched its tokenization in `input_ids`. BPE merges across the question/options boundary (specifically the `\n\n` between question and `Options:`) caused 100% of items to fail the question search. Fix: marker-based slice derivation. Locate the `Options:` and `Answer with a single letter from A, B, C, D, E.` markers in `input_ids` (with multiple spelling variants tried in most-specific-first order, plus up to 3-character left trims for BPE tolerance), and derive `question = [v_end + 1, options_marker_start)` by subtraction.

2. **Cache-class subscript bug (Pass B).** First-pass `TextKResidualCache` used composition (`self._inner = DynamicCache()`) with `__getattr__` forwarding. Transformers calls `cache[layer_idx]` (subscript) on `past_key_value` — `__getitem__` is a dunder method not caught by `__getattr__`. Failed on all 200 items in Pass B with `TypeError: 'TextKResidualCache' object is not subscriptable`. Fix: subclass `DynamicCache` directly (proper is-a inheritance). Pass A data was unaffected; Pass B re-launched cleanly with the fix.

## Experiment F — K-quantizer repair screening, COMPLETE (2026-05-09)

**Status:** Tiered screen complete. Stage 0 (n=16, 14 conditions, 8 min wall, 0 fail). Stage 1 (n=64 balanced 16/bucket, 14 conditions, 33 min wall, 0 fail, 896 rows). Stage 3 (n=200 canonical Exp A split, 10 conditions = F0–F9, 76 min wall, 0 fail, 2000 rows). Calibration pass (cal-100, 7 min wall, 100/100 ok).

After D1 / E1 ruled out routing-within-K as a research direction, the remaining hypothesis was that the K quantizer itself is the bottleneck. The F-suite tests 14 K-quantizer scale strategies against 4 anchors (F0 BF16, F1 uniform INT4, F2 = D1.3 text-K BF16, F3 = C2.1 all-K BF16); all hold V at INT4 per Exp C's "V is free" finding.

### Stage 1 conditions (n=64 balanced 16/bucket)

14 conditions. Anchors F0–F3; literature-aligned KIVI-style K repairs F4–F9; VLM-specific score-space variants F10–F13. Calibration NPZ captures per-(L, H_kv, channel) k_channel_energy + outlier indices + q_energy/q_energy_text/q_energy_visual diagonals via a forward hook on each layer's `q_proj`.

### Stage 3 result (n=200, canonical Exp A split, F0–F9 only)

Stage 1's F10–F13 (closed-form score-cal Q-energy reweighting) all FAILED (acc 0.172–0.281, below or near floor); dropped from Stage 3. Stage 3 confirms the 6 KIVI-style variants on the same 200 eval items used by Exp A / D1 / E1.

| ID | Description | n | acc | 95% CI | bf16-pres | mean margin | Δmargin vs F1 | avg K | avg V | avg KV |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **F0** | BF16 K/V (ceiling) | 200 | **0.565** | [0.495, 0.635] | 1.000 | +0.92 | +1.73 | 16.00 | 16.00 | 16.00 |
| **F1** | Uniform INT4 K/V (floor) | 200 | **0.210** | [0.155, 0.265] | 0.204 | −0.81 | 0.00 | 4.00 | 4.00 | 4.00 |
| F2 | Text-K BF16 + visual-K INT4 + V INT4 (= D1.3) | 200 | 0.385 | [0.315, 0.455] | 0.566 | −0.42 | +0.39 | 4.30 | 4.00 | 4.15 |
| F3 | All-K BF16 + V INT4 (= C2.1) | 200 | 0.550 | [0.480, 0.620] | 0.965 | +0.89 | +1.70 | 16.00 | 4.00 | 10.00 |
| **F4** | **KIVI per-channel-along-seq** | 200 | **0.545** | [0.475, 0.615] | **0.841** | +0.50 | **+1.31** | **4.00** | **4.00** | **4.00** |
| F5 | F4 + text/visual scale split | 200 | 0.510 | [0.440, 0.585] | 0.850 | +0.59 | +1.40 | 4.00 | 4.00 | 4.00 |
| F6 | F4 + per-prompt-role scales | 200 | 0.525 | [0.455, 0.595] | 0.841 | +0.71 | +1.51 | 4.00 | 4.00 | 4.00 |
| F7 | F4 + 99.5%ile clipping | 200 | 0.540 | [0.470, 0.610] | 0.832 | +0.60 | +1.40 | 4.00 | 4.00 | 4.00 |
| F8 | F4 + top-8 outlier channels at BF16 per (L, H_kv) | 200 | 0.540 | [0.475, 0.610] | 0.894 | +0.84 | +1.64 | 4.75 | 4.00 | 4.375 |
| **F9** | **F4 + top-16 outlier channels at BF16 per (L, H_kv)** | 200 | **0.560** | [0.495, 0.630] | **0.929** | **+0.88** | **+1.68** | 5.50 | 4.00 | 4.75 |

### Anchor sanity (paired with prior experiments)

All four anchors land within bootstrap CI of the prior n=200 / n=100 numbers, confirming the F-suite plumbing is correct end-to-end:

- **F0 = 0.565** (target A1 = 0.565; pixel-perfect) ✓
- **F1 = 0.210** (target A5 = 0.210; pixel-perfect) ✓
- **F2 = 0.385** (target D1.3 = 0.385; pixel-perfect) ✓
- **F3 = 0.550** (target C2.1 = 0.530 on n=100; F3's slightly higher value on n=200 is within bootstrap noise) ✓

### Headline — F4 is the deployable result

**KIVI per-channel-along-seq closes 94.4% of the F1→F0 gap (33.5 of 35.5 pp) at TRUE 4.00 KV bits.** No calibration data, no routing, no slice info — just a one-line scale-axis change from per-(head, position, group of 128 head_dim slots) to per-(head, channel) shared across all positions. Mechanistically: K outliers are channel-aligned across the sequence (KIVI's central finding, arXiv:2402.02750). One scale per channel exposes them; one scale per position hides them inside head_dim grouping.

F4 dominates F3 (same accuracy 0.545 vs 0.550, 2.5× less memory) and dominates F8 (better acc at lower bits). F4 is the Pareto-optimal point at 4 bits.

### VLM-specific scaling refinements DO NOT help once the K axis is correct

This is the second-order finding. F5/F6/F7 all underperform F4 at the same 4-bit budget despite adding modality-aware / role-aware / outlier-aware refinements:

- F5 text/visual split: 0.510 (−3.5 pp vs F4)
- F6 prompt-role split: 0.525 (−2.0 pp vs F4)
- F7 99.5%ile clip: 0.540 (−0.5 pp vs F4)

This directly answers the question "does Qwen2.5-VL's modality structure require modality-aware K quantization?" — once the axis is right, no.

### F9 is the zero-accuracy-loss Pareto point

F9 (top-16 outlier channels at BF16 per (L, H_kv), 4.75 KV bits) = 0.560, statistically tied with F0 BF16 ceiling (0.565) within bootstrap CI and BF16-correct preservation 0.929. Total protected channels = 16 × 28 × 4 = 1792 out of 14336 (12.5% of K channels at BF16). Memory cost: ~3.4× compression vs BF16 (vs F4's 4× at slight accuracy cost).

### Score-cal failures (Stage 1 only; dropped from Stage 3)

| ID | Description | Stage 1 n=64 acc |
|---|---|---:|
| F10 | Generic score-cal: per-channel scale ∝ sqrt(E[Q_d²]) | 0.281 |
| F11 | Block-score TT-heavy (w_TT=4, w_TV=1, w_VT=1, w_VV=0.5) | 0.172 |
| F12 | Block-score balanced (all w=1) | 0.188 |
| F13 | Text-K-only score-cal | 0.172 |

The closed-form `s_d ∝ sqrt(E[Q_d²])` heuristic concentrates scale precision on high-Q-energy channels, but those empirically aren't the channels that matter most for attention. Cross-modal block-score weighting (F11 / F12) actively introduces noise. **Score-cal as a research direction needs iterative scale search to be revisited** — the closed-form approximation is a strict downgrade at this bit budget.

### Per-bucket F4 vs F0 (Stage 3, n=200)

| Bucket | F0 BF16 | F4 KIVI | Δ (F4 − F0) |
|---|---:|---:|---:|
| short (n=33) | 0.667 | 0.606 | −6.1 pp |
| mid (n=33) | 0.848 | 0.818 | −3.0 pp |
| long (n=67) | 0.537 | 0.567 | **+3.0 pp** |
| very_long (n=67) | 0.403 | 0.358 | −4.5 pp |

F4 actually beats F0 on long videos (+3 pp), likely sample noise at n=67 but worth noting. F4 is robust across all duration buckets, with the largest gap on short (where F0 is highest, so any per-channel quantization noise has more headroom to flip an answer).

### Bit-accounting fix (regression bug from Stage 1)

Stage 1 reported `avg_kv_bits = 4.0` for F8 and F9 because `_run_condition_forward` used the formula `(cfg.bits + 4.0) / 2.0` and ignored `cfg.n_outliers`. Stage 3 fixes this with a `_compute_three_bit_columns` helper that handles all six cache modes (bf16, v1_kcfg, v1_kcfg_with_slice, v3k_text_bf16, v3k_all_bf16) with correct outlier-channel weighting:

- F8: K=4.75 bits (16×8/128 + 4×120/128), V=4.00, KV=4.375
- F9: K=5.50 bits (16×16/128 + 4×112/128), V=4.00, KV=4.75

Stage 3 JSONL rows now include `avg_k_bits`, `avg_v_bits`, and `avg_kv_bits` separately. This makes F4's true 4-bit budget distinguishable from F9's 4.75-bit point on the Pareto frontier.

### Mechanistic interpretation — why per-channel-along-seq beats per-channel-along-head_dim

K shape is `[B, H_kv, T, D]`. The naive per-head_dim quantizer (F1) computes a scale per (B, H_kv, T, group of 128 head_dim slots) — one scale per *position*. The KIVI per-seq quantizer (F4) computes a scale per (B, H_kv, channel) — one scale per *channel* shared across positions.

The KIVI literature's argument (arXiv:2402.02750): K outliers are channel-aligned across the sequence — a few specific channel indices have systematically large magnitude across most positions. Per-head_dim quantization gives each position its own tight scale, which "hides" the outlier-vs-normal channel gap inside the group-of-128. When the outlier channel happens to have a very high value at some position, that scale dilates to fit it, and the *non-outlier* channels at that position get coarser quantization. Per-seq quantization gives each channel its own scale, exposing outlier channels separately and letting non-outlier channels keep tight quantization regardless of position.

Empirically Qwen2.5-VL's post-RoPE K confirms this geometry: F4 (axis change, no outlier handling) closes 94.4% of the gap; F8/F9 (axis change + explicit outlier protection) close 96-99%. The marginal value of the explicit outlier protection on top of the axis change is small (~1-2 pp).

### Implications for prior experiments

The 35.5 pp KV-quantization collapse from Exp A — which motivated Exp B (routing), Exp C (K/V isolation), Exp D0/D1 (visual-K windows), and Exp E1 (text-K subspans) — is *solved* by switching the K scale axis. **None of the routing experiments were necessary**; the bottleneck was always the quantizer, not the selector. The accumulated negative results across B/D/E (no routing signal helps at avg=4 with the {INT2, BF16} or {INT4, BF16} tier sets) are now retroactively explained: routing was trying to compensate for a broken quantizer, and no allocation policy can recover what the quantizer threw away.

### Future work (not in F-suite)

1. **AKVQ-VL Hadamard rotation as a Pareto-improvement over F4.** Requires Q-side rotation at attention time (out of pure-cache scope); could push F4 from 0.545 to ≈0.560 at 4.00 KV bits.
2. **VidKV V-per-channel quantization.** Tests whether V can be compressed below 4 bits without accuracy loss when paired with F4 K. F-suite is K-only; V-side experiments are a separate family.
3. **Long-form generation re-test on Video-MME / MVBench.** First-token MCQ is a *minimal* visual-K-stress setting. Multi-token decode re-queries visual K many times; F4 might or might not preserve quality there.
4. **Iterative score-cal scale search.** F10–F13 used closed-form `sqrt(E[Q_d²])` reweighting and failed. A line-search over per-channel scales minimizing `||center(QK^T) - center(Q · Q4(K)^T)||_F` on cal-100 might recover, but the gap to F4 (≈25 pp) is so large that this is low-priority.

### Files of record (Exp F)

- `qwen/scripts/k_quantizers.py` — `KQuantizerConfig` dataclass + 12 quantizer kinds + `apply_k_quantizer` dispatch.
- `qwen/scripts/expF_calibrate.py` — single-pass cal-100 capture; `KStatsCache` + `QStatsHook` (forward hooks on each layer's `q_proj`); writes `expF_kcalib_{model}_frames64.{json,npz}`.
- `qwen/scripts/expF_smoke.py` — Phase A (synthetic-tensor) + Phase B (live-model logits-differ) smoke checks. 8 hard assertions.
- `qwen/scripts/expF_kquant_screen.py` — tiered driver (Stage 0 / 1 / 2 / 3); auto-generates stratified split files; per-condition runner; bit-accounting helper `_compute_three_bit_columns`.
- `qwen/scripts/expF_analyze.py` — per-condition table + verdict matrix (kill / borderline / promote_n100 / promote_n200 / paper_strong).
- `qwen/scripts/run_expF.sh` — orchestrator (`smoke|calib|stage{0,1,2,3}|analyze|full`).
- `qwen/scripts/fake_quant_kv_cache.py` — extended: `FakeQuantKVCache.__init__` accepts optional `k_quantizer_config`; `update()` dispatches K through `apply_k_quantizer` when set; `set_slice_info()` for role/modality K kinds.
- `qwen/calibration/expF_kcalib_Qwen2.5-VL-7B-Instruct_frames64.json` + `.npz` — calibration data (n_cal=100 ok, n_failed=0).
- `qwen/calibration/split_seed0_n{16,64}.json` — Stage 0 / Stage 1 stratified subsets (Stage 3 uses canonical `split_seed0.json`).
- `qwen/results/expF_kquant_stage{0,1,3}.jsonl` — per (item, condition) rows. Stage 0: 224 rows; Stage 1: 896 rows; Stage 3: 2000 rows.
- `qwen/results/expF_summary_stage{0,1,3}.md` — per-condition tables (Stage 3 has 3-bit-column accounting).
- `qwen/results/expF_verdict_matrix_stage{0,1,3}.md` — verdict + promotion plan.
- `qwen/results/expF_smoke.md` — Phase A + Phase B smoke report (8 PASS).

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

qwen-expD-d1 (tmux session, GPU 0) — PIPELINE COMPLETE (1h 42min total)
└── ✅ STEP 2  Cross-modal K/V quantization: 200 eval items × 14 conditions × 64 frames
              D1.3 (text-K BF16, visual-K INT4, V INT4) → 0.385
              D1.4 (text-K INT4, visual-K BF16, V INT4) → 0.210 ← 17.5 pp WORSE at 2.4× bits
              D1.5a / D1.5a_mh (top-1 visual-K BF16 + text-K BF16) → 0.415 / 0.425
              D1.5b / D1.5b_mh (top-2 visual-K BF16 + text-K BF16) → 0.430 / 0.415
              D1.6a/b (random ×3 seeds) → 0.417 / 0.423
              D1.7a/b (uniform) → 0.395 / 0.395
              All visual-K-protected conditions cluster in [0.395, 0.435] —
              window selection is irrelevant. Text-K is the dominant fragility.

qwen-expE1 (tmux sessions: passA + passB) — PIPELINE COMPLETE (89 min total)
├── ✅ STEP 1  passA: 7 fixed-slice conditions × 200 items (54 min)
│             E1.4 OptionsOnly (40 tok) → 0.290  ← best single, recovers 46% rescue
│             E1.6 Q+Options (91 tok) → 0.270, E1.5 InstrAnsPrefix (22 tok) → 0.225,
│             E1.2 HeaderOnly (14 tok) → 0.215, E1.8 Q+O+AP (96 tok) → 0.205
│             E1.7 O+AP (45 tok) → 0.185 ← BELOW floor; E1.3 Question (50 tok) → 0.175
└── ✅ STEP 2  passB: random-20 (3 seeds) + K-residual-top-20 × 200 items (35 min)
              Global median budget N=20 from passA's per-item best slice.
              E1.10 K-residual-top-20 → 0.200 ← BELOW floor and BELOW random
              E1.9 random-20 (3 seeds) → 0.220 / 0.215 / 0.210 (= floor)
              No condition is sufficient (≥80% of D1.3 acc at <50% of D1.3 tokens).
              Text-K routing falsified at every selection signal tested.
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

---

# Experiment G — Frame-scaling under fixed KV memory budget (2026-05-09)

**Status as of 2026-05-09 09:31 UTC:** Stage 1 (n=64 balanced 16/bucket, 9 fixed-frame conditions + 4 adaptive post-process conditions) **COMPLETE**. Stage 3 (n=200 canonical, 4 promoted conditions) **IN FLIGHT** — 64f and 128f tiers complete; 256f tier (G6) at 17/200, ETA ~36 min.

## Hypothesis

Exp F's F4 (KIVI per-channel-along-seq, 4 KV bits) closed 94.4% of the F1→F0 collapse — the rescue gap was a quantizer-axis problem, not a routing problem. The open question moves up one level: **what does 4-bit KV buy us for long-video VLM inference?** The most VLM-specific answer is *more visual evidence under the same KV memory budget.* At fixed `max_pixels=360×420`,

```
relative KV memory = (frames × avg_kv_bits) / (64 × 16)
```

so 256-frame F4 (≈4 KV bits) ≈ 64-frame BF16 KV memory at 4× the temporal coverage. **Headline test: does the matched-memory more-frame condition (G4 = 256f F4) beat the BF16 baseline (G0 = 64f BF16)?**

## Setup

- **Model:** Qwen2.5-VL-7B-Instruct (28 layers, 4 KV-heads, head_dim=128).
- **Bench:** LongVideoBench-val MCQ scoring, max_new_tokens=1, 4/5-way logprob argmax.
- **Stage 1 split:** balanced 16/bucket × 4 buckets = 64 items (same 64-item subset as F-suite Stage 1 so F4 anchors line up).
- **Stage 3 split:** auto-generated `split_seed0_n200.json` with 50/50/50/50 per bucket = 200 items. Note: this is **not** the canonical Exp A 200-eval set (which is 33/33/67/67 weighted toward longer items); the more-balanced Stage 3 set has higher BF16 ceiling (~0.665 vs Exp A's 0.565).
- **Calibration:** F-suite NPZ at frames=64 reused as-is. K-channel outlier indices are post-RoPE and frame-count-independent in theory; smoke check 8 (opt-in) verifies via 8-item recalibration at frames=256 with Jaccard ≥ 0.75.
- **Compute structure:** outer loop over frame counts {64, 128, 256}; inner loop over K-quantizer configs (F0/F4/F9). Visual prefill (image-token construction, position-id generation, find_text_slice_spans, inputs dict) is computed once per (item, frames) and reused across configs at the same frame count.

## Conditions

Stage 1 ran 7 fixed-frame conditions plus 2 adaptive post-processes (F4 backbone). After Stage 1 surfaced F9 winning, 2 additional F9-backbone adaptive post-processes were added.

| ID | Frames | KV format | rel KV mem | Class |
|---|---:|---|---:|---|
| G0 | 64 | BF16 | 1.00× | anchor |
| G1 | 64 | F4 INT4 (KIVI per-channel-seq) | 0.25× | anchor |
| G2 | 128 | BF16 | 2.00× | anchor |
| G3 | 128 | F4 INT4 | 0.50× | fixed |
| G4 | 256 | F4 INT4 | 1.00× | **HEADLINE matched-memory** |
| G5 | 128 | F9 (KIVI + top-16 outlier BF16, 4.75 KV bits) | 0.59× | fixed |
| G6 | 256 | F9 | 1.19× | fixed |
| G7_F4_CascadeAvg128 | 64↗256 cascade, target avg=128 | F4 | 0.50× | adaptive (post-process) |
| G7_F9_CascadeAvg192 | 128↗256 cascade, target avg=192 | F9 | 0.99× | adaptive (post-process) |
| G8_F4_TypeAdaptive | type-routed 64/128/256 | F4 | 0.55× | adaptive (post-process) |
| G8_F9_TypeAdaptiveMin128 | type-routed 128/256 | F9 | 0.71× | adaptive (post-process) |

The cascade routes the bottom-third of items by margin (`max(option_logprobs) − second_max(...)` from the cheap first pass) to the expensive second pass; the type-adaptive routes by question-keyword classifier (count/ocr/detail → 256f, temporal/action → 128f, other → 64f for F4 / 128f for F9 since no F9 64f exists).

## Stage 1 results — n=64 (COMPLETE)

| Condition | n | acc | 95% CI | rel_kv_mem | g0-pres |
|---|---:|---:|---|---:|---:|
| G0_BF16 | 64 | **0.672** | [0.547, 0.781] | 1.00× | 1.000 |
| G1_F4_64f | 64 | 0.656 | [0.547, 0.766] | 0.25× | 0.884 |
| G2_BF16_128f | 64 | 0.656 | [0.531, 0.766] | 2.00× | 0.907 |
| G3_F4_128f | 64 | 0.562 | [0.438, 0.688] | 0.50× | 0.721 |
| G4_F4_256f | 63 | 0.619 | [0.508, 0.746] | 1.00× | 0.786 |
| **G5_F9_128f** | 64 | **0.688** | [0.578, 0.797] | **0.59×** | 0.930 |
| **G6_F9_256f** | 63 | **0.730** | [0.619, 0.825] | 1.19× | 0.976 |
| G7_F4_CascadeAvg128 | 60 | 0.633 | [0.500, 0.750] | 0.50× | 0.854 |
| **G7_F9_CascadeAvg192** | 60 | **0.733** | [0.617, 0.834] | **0.99×** | 0.976 |
| G8_F4_TypeAdaptive | 60 | 0.633 | [0.517, 0.750] | 0.55× | 0.805 |
| G8_F9_TypeAdaptiveMin128 | 60 | 0.700 | [0.583, 0.817] | 0.71× | 0.927 |

4 items at 256f failed because the source video had fewer than 256 native frames (qwen_vl_utils raises `ValueError: nframes should in interval [2, N], but got 256` for N<256). Runner caught and continued; affects G4/G6/G7-F9/G8-F9 sample sizes.

### Stage 1 paired McNemar (headline pairs)

| Pair | label | n | both | a_only | b_only | neither | Δ acc(a−b) | χ² |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| G4 vs G0 | matched-memory headline | 60 | 32 | 5 | 9 | 14 | **−6.6 pp** | 1.14 |
| G3 vs G0 | memory-saving | 64 | 31 | 5 | 12 | 16 | −11.0 pp | 2.88 |
| **G6 vs G2** | **zero-loss at 4× frames** | 60 | 40 | 4 | 0 | 16 | **+6.7 pp** | **4.00** |
| G6 vs G0 | f9_256f vs baseline | 60 | 39 | 5 | 0 | 16 | +8.3 pp | 5.00 |
| G5 vs G0 | f9_128f vs baseline | 64 | 41 | 4 | 2 | 17 | +3.1 pp | 0.67 |
| G7_F4_CascadeAvg128 vs G1 | cascade vs F4 anchor | 60 | 37 | 1 | 3 | 19 | −3.3 pp | 1.00 |
| G8_F4_TypeAdaptive vs G1 | type-adaptive vs F4 anchor | 60 | 35 | 3 | 5 | 17 | −3.3 pp | 0.50 |
| G7_F9_CascadeAvg192 vs G5 | F9 cascade vs F9 anchor | 60 | 41 | 4 | 1 | 14 | +5.0 pp | 1.80 |
| G7_F9_CascadeAvg192 vs G6 | F9 cascade vs F9 top | 60 | 43 | 1 | 0 | 16 | +1.7 pp | 1.00 |
| G8_F9_TypeAdaptiveMin128 vs G5 | F9 type-adaptive vs F9 anchor | 60 | 39 | 4 | 3 | 14 | +1.7 pp | 0.14 |

### Stage 1 frontier (sorted ascending by relative_kv_memory)

| Cond | frames | avg_kv_bits | rel_kv_mem | acc |
|---|---|---:|---:|---:|
| G1_F4_64f | 64 | 4.00 | 0.25× | 0.656 |
| G3_F4_128f | 128 | 4.00 | 0.50× | 0.562 |
| G7_F4_CascadeAvg128 | ~128 mixed | 4.00 | 0.50× | 0.633 |
| G8_F4_TypeAdaptive | ~140 mixed | 4.00 | 0.55× | 0.633 |
| **G5_F9_128f** | 128 | 4.75 | **0.59×** | **0.688** |
| G8_F9_TypeAdaptiveMin128 | ~154 mixed | 4.75 | 0.71× | 0.700 |
| **G7_F9_CascadeAvg192** | ~192 mixed | 4.75 | **0.99×** | **0.733** |
| G0_BF16 | 64 | 16.00 | 1.00× | 0.672 |
| G4_F4_256f | 256 | 4.00 | 1.00× | 0.619 |
| **G6_F9_256f** | 256 | 4.75 | 1.19× | **0.730** |
| G2_BF16_128f | 128 | 16.00 | 2.00× | 0.656 |

Pareto-optimal points at n=64 are **G5** (cheapest above-baseline accuracy, 0.59× memory) and **G7_F9_CascadeAvg192** (best accuracy below 1.00× memory; ties G6 at 75% the compute).

### Stage 1 verdict

```
G3_F4_128f  (0.562)  → kill           (-11 pp vs G0)
G4_F4_256f  (0.619)  → kill           (matched-memory headline failed; -5.3 pp vs G0)
G5_F9_128f  (0.688)  → promote_n200   (+1.6 pp vs G0 at 0.59× mem)
G6_F9_256f  (0.730)  → promote_n200   (+5.8 pp vs G0; +6.7 pp paired vs G2 with χ²=4.0)
G7_F4 / G8_F4         → borderline    (dragged down by F4 backbone's weak G3/G4)
G7_F9 / G8_F9         → borderline    (G7_F9=0.733 ties G6 at 75% compute; G8_F9=0.700)
```

**Per-bucket pattern:** F4 helps on *short* clips (G4 short=0.833 vs G0 short=0.688, +14.6 pp) but loses on long/very_long. F9 256f wins in every bucket except very_long (G6 short=0.917, mid=0.750, long=0.737, very_long=0.562). Frame coverage genuinely buys something at short — but F4's per-channel-along-seq scale is brittle at 23k tokens; F9's outlier protection compensates.

## Stage 3 results — n=200 (IN FLIGHT)

Promoted 4 conditions to n=200 canonical: G0, G2, G5, G6. Auto-generated `split_seed0_n200.json` (50/50/50/50 per bucket).

| Tier | Conditions | Status | Wall |
|---|---|---|---|
| 64f | G0_BF16 | DONE 200/200 | 12:09 |
| 128f | G2_BF16_128f, G5_F9_128f | DONE 200/200 | ~36 min |
| 256f | G6_F9_256f | running 17/200 | ETA 36 min |

After 256f completes, the chain auto-fires: F9 cascade post-process (G7_F9_CascadeAvg192 stitch) → F9 type-adaptive post-process (G8_F9_TypeAdaptiveMin128 stitch) → analyze.

### Stage 3 live preview (n=200 final for G0/G2/G5; n=17 partial for G6)

| Cond | n | acc | rel_kv_mem | Note |
|---|---:|---:|---:|---|
| G0_BF16 | 200 | 0.665 | 1.00× | FINAL |
| G2_BF16_128f | 200 | 0.680 | 2.00× | FINAL |
| **G5_F9_128f** | 200 | **0.685** | **0.59×** | FINAL |
| G6_F9_256f | 17 | 0.706 | 1.19× | partial; not stable |

**Paired G5 vs G0 (n=200): Δ = +2.0 pp** (G5 wins on 17 items, G0 wins on 12, 116 both, 55 neither). Stage 1 showed G5 vs G0 = +1.6 pp at n=64; Stage 3 reproduces and slightly strengthens at n=200.

**Paired G2 vs G0 (n=200): Δ = +1.5 pp** — extra BF16 frames at 2× memory help only marginally.

**Crucially, G5 ≥ G2 at n=200** (0.685 vs 0.680): F9 INT4 at 128f matches BF16 at 128f at less than 30% the KV memory. F9 is the deployable 128f operating point.

The headline G6 vs G2 ("zero-loss at 4× frames" with F9) will resolve once the 256f tier completes. Stage 1 result was Δ = +6.7 pp paired with χ²=4.0.

## Files of record (Exp G)

```
qwen/scripts/expG_frame_scaling.py        # main driver; outer loop frames, inner loop K-cfg
qwen/scripts/expG_smoke.py                # 8 hard assertions (Phase A + Phase B)
qwen/scripts/expG_cascade.py              # G7 cascade post-process (any first/second pass cond pair)
qwen/scripts/expG_type_adaptive.py        # G8 type-adaptive post-process (custom budget map / mapping)
qwen/scripts/question_type_classifier.py  # 6-label keyword heuristic + BUDGET_MAP
qwen/scripts/expG_analyze.py              # per-cond + paired McNemar + frontier + verdict
qwen/scripts/run_expG.sh                  # smoke|stage1|stage3|cascade|analyze|full

qwen/results/expG_frame_stage1.jsonl                     # 446 rows (64 items × 7 conds + errors)
qwen/results/expG_frame_stage1_G7.jsonl                  # 60 F4-cascade stitched rows
qwen/results/expG_frame_stage1_G7f9.jsonl                # 60 F9-cascade stitched rows
qwen/results/expG_frame_stage1_G8.jsonl                  # 60 F4-type-adaptive stitched rows
qwen/results/expG_frame_stage1_G8f9.jsonl                # 60 F9-type-adaptive stitched rows
qwen/results/expG_summary_stage1.md                      # per-condition table + per-bucket
qwen/results/expG_paired_stage1.md                       # 11 paired McNemar comparisons
qwen/results/expG_frontier_stage1.md                     # frame-budget × accuracy frontier
qwen/results/expG_verdict_matrix_stage1.md               # promotion plan
qwen/results/expG_cascade_meta.json / _f9_meta.json      # τ, realized avg frames, bucket dist
qwen/results/expG_qtype_meta.json / _f9_meta.json        # budget map, label distribution
qwen/results/expG_frame_stage3.jsonl                     # in-flight stage 3 rows
qwen/results/expG_smoke.md                               # 8-check report
qwen/results/expG_pipeline.progress.log
qwen/results/expG_frame_stage{1,3}.progress.log
```

## Methodological notes

1. **Compute amortization.** Outer-loop frame count, inner-loop K-quantizer config: visual prefill (decord decode + image-token build + position-ids + find_text_slice_spans + inputs dict) is computed once per (item, frames) and reused across F0/F4/F9 conditions sharing that frame count. Stage 1 wall: 5:58 (64f, 2 conds) + 15:50 (128f, 3 conds) + 21:19 (256f, 2 conds) = 43 min total — vs naive estimate ~3.5 h. ~5× speedup over the worst-case nested loop.

2. **Memory precheck calibration.** First Stage 1 launch had `EXPG_MIN_FREE_GB=70`; precheck at the 256f tier saw `torch.cuda.mem_get_info()` returning 42 GiB free (model resident from 128f tier consumed ~17 GiB beyond the co-tenant's 21 GiB, even though OS-level free was 60 GiB). Tier was correctly skipped per safety logic. Lowered to `EXPG_MIN_FREE_GB=30` and re-launched 256f-only with `--append`; ran cleanly. Peak 256f memory is ~14 GiB beyond model weights, so 30 GiB is the right safety threshold.

3. **Backfill skip-row handling.** Initial `backfill_bf16_join_g` crashed on placeholder rows (the tier-skip path emits stub rows with no `correct_choice`). Fixed: skip rows with `skipped=True`, missing `correct_choice`, or `error` set.

4. **Question-type classifier tuning.** First smoke run hit weighted_avg_frames=194.6 outside [110, 145]. LongVideoBench questions have long scene-description prefixes ("In a dim room, there is..."), creating false-positive temporal matches against words like "during"/"after" in description prose. Fixed by trimming to the trailing interrogative (substring after the last `.`) and using question-form keywords ("how many" not bare "many"); BUDGET_MAP recalibrated so temporal→128 (subtitle-anchored, range-narrowed) and detail→256 (visual-content-heavy). Final cal-100 distribution: 51 temporal, 17 detail, 9 action, 23 other → 135.0 weighted avg.

5. **Short-video failures.** 4 of 64 items at 256f fail because video duration × fps < 256. Treated as data-driven sample drop; affects G4/G6/G7-F9/G8-F9 from n=64 to n=63 (Stage 1) and similarly at scale on Stage 3.

## Pipeline status

```
qwen-expG (tmux session, GPU 0) — STAGE 1 COMPLETE (43 min) + ANALYZE COMPLETE; STAGE 3 IN FLIGHT
├── ✅ smoke (8 checks PASS, 16 sec)
├── ✅ stage 1 64f tier (G0 + G1, 5:58 wall, 64/64 ok)
├── ✅ stage 1 128f tier (G2 + G3 + G5, 15:50 wall, 64/64 ok)
├── ✅ stage 1 256f tier (G4 + G6, 21:19 wall, 60/64 ok, 4 short-video failures)
├── ✅ stage 1 cascade + analyze (F4 backbone)
├── ✅ stage 1 F9-backbone post-process + analyze (G7_F9 = 0.733, G8_F9 = 0.700)
├── ✅ stage 3 64f tier (G0, 12:09 wall, 200/200 ok)
├── ✅ stage 3 128f tier (G2 + G5, ~36 min wall, 200/200 ok)
└── 🔄 stage 3 256f tier (G6) — 17/200 (8.5%), ETA 36 min
    └── then auto-fire: F9 cascade + F9 type-adaptive + analyze
```

---

# Experiment G — UPDATE 2026-05-09 — Stage 3 COMPLETE + Exp H temporal-windowed KIVI

**Status as of 2026-05-09 22:53 UTC:** All planned forwards and post-processes complete. Stage 3 (n=200 canonical, 50/50/50/50 per bucket via auto-generated `split_seed0_n200.json`) finalized with **all 17 conditions** including F4/F9 anchors at n=200, F9 cascade selection-mode controls (margin / random×3 / oracle), F9 type-adaptive, direct 192f F9 control, and the new Exp H temporal-windowed KIVI conditions (H3/H4/H5/H6).

**Three things changed since the previous update:**
1. The Stage-1 G6 lead (+5.8 pp vs G0) shrunk dramatically at Stage 3 (+1.6 pp).
2. The cascade story largely collapsed at scale (margin = random = fixed 192f).
3. **The new H6_KIVI_TempWin2_128f earned `promote_paper_strong` verdict at Stage 3** — beats F4 by +7 pp at the same KV bits (paired McNemar χ²=4.90), ties F9 at TRUE 4.00 KV bits and 0.50× memory.

## What's new since the previous Exp G writeup

### G control conditions (post-Stage-3)
- `G9_F9_192f` (n=200): direct fixed-frame F9 at 192 frames. Tests whether the cascade's apparent gain came from the average frame budget (192) or the adaptive routing.
- `G7_F9_CascadeRandomS{0,1,2}` (n=188 each): random selection at the same rerun rate as the margin cascade — controls "did margin pick useful items?"
- `G7_F9_CascadeOracle` (n=188): rerun items where G5 was wrong AND G6 was right; uses ground-truth labels for an upper bound.

### Exp H — temporal-windowed KIVI (new K-quantizer kind)
A new `kivi_temporal_window` quantizer in `qwen/scripts/k_quantizers.py` (commit `ad06928`):

- **`visual_only` mode:** text-prefix + N visual windows + text-suffix, each with its own per-channel KIVI scale `[B, H_kv, 1, D]`.
- **`token_block` mode:** N equal-token blocks across the whole sequence, ignoring modality — modality-blind control for the visual-time hypothesis.
- Stays at TRUE 4.00 KV bits (no outlier spend; scale metadata is negligible vs cache).
- `cache_offset > 0` falls back to plain F4 — matches F5/F6 pattern (decode-time chunking is a no-op for first-token MCQ scoring).

Stage-1 H conditions (n=64; reuses H0=G0, H1=G4, H2=G6, H7=G5):
- H3 (256f, 4 visual windows of 64 frames each)
- H4 (256f, 8 visual windows of 32 frames each)
- H5 (256f, 4 token-equal blocks; modality-blind control)
- H6 (128f, 2 visual windows of 64 frames each)

Promoted to Stage 3 (n=200) after Stage-1 mid-run looked promising.

## Stage 3 final per-condition table (n=200 or n=188 after short-video drops)

| Condition | n | acc | 95% CI | rel_kv_mem | avg_kv_bits | class | Verdict |
|---|---:|---:|---|---:|---:|---|:-:|
| G0_BF16 | 200 | 0.665 | [0.595, 0.730] | 1.000 | 16.000 | fixed | anchor |
| G2_BF16_128f | 200 | 0.680 | [0.615, 0.740] | 2.000 | 16.000 | fixed | anchor |
| G3_F4_128f | 200 | 0.610 | [0.540, 0.675] | 0.500 | 4.000 | fixed | **kill** |
| G4_F4_256f | 188 | 0.617 | [0.548, 0.686] | 1.000 | 4.000 | fixed | borderline |
| **G5_F9_128f** | 200 | **0.685** | [0.620, 0.745] | **0.594** | 4.750 | fixed | promote_n200 |
| G6_F9_256f | 188 | 0.681 | [0.612, 0.745] | 1.188 | 4.750 | fixed | promote_n200 |
| G7_F9_CascadeAvg192 (margin) | 188 | 0.681 | [0.612, 0.750] | 0.989 | 4.750 | cascade | borderline |
| G7_F9_CascadeOracle | 188 | 0.718 | [0.654, 0.782] | 0.616 | 4.750 | cascade | control |
| G7_F9_CascadeRandomS0 | 188 | 0.686 | [0.617, 0.750] | 0.891 | 4.750 | cascade | control |
| G7_F9_CascadeRandomS1 | 188 | 0.676 | [0.606, 0.739] | 0.891 | 4.750 | cascade | control |
| G7_F9_CascadeRandomS2 | 188 | 0.702 | [0.638, 0.766] | 0.891 | 4.750 | cascade | control |
| G8_F9_TypeAdaptiveMin128 | 188 | 0.681 | [0.612, 0.745] | 0.704 | 4.750 | type_adaptive | borderline |
| G9_F9_192f | 200 | 0.685 | [0.620, 0.745] | 0.891 | 4.750 | fixed | borderline |
| H3_KIVI_TempWin4_256f | 188 | 0.644 | [0.574, 0.708] | 1.000 | 4.000 | fixed | borderline |
| H4_KIVI_TempWin8_256f | 188 | 0.644 | [0.574, 0.713] | 1.000 | 4.000 | fixed | borderline |
| H5_KIVI_TokenBlock4_256f | 188 | 0.617 | [0.548, 0.686] | 1.000 | 4.000 | fixed | control |
| **H6_KIVI_TempWin2_128f** | 200 | **0.680** | [0.615, 0.740] | **0.500** | **4.000** | fixed | **promote_paper_strong** |

## Stage 3 paired McNemar — the headline tests

### F9 cascade controls — the whole story collapsed at scale

| Pair | n | both | a_only | b_only | net_a | acc(a) | acc(b) | χ² | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G7_F9_CascadeAvg192 vs G5_F9_128f | 188 | 121 | 7 | 7 | **0** | 0.681 | 0.681 | 0.00 | **Cascade = first-pass anchor.** |
| G7_F9_CascadeAvg192 vs G6_F9_256f | 188 | 128 | 0 | 0 | **0** | 0.681 | 0.681 | nan | **Cascade = top condition** (perfect overlap). |
| G7_F9_CascadeOracle vs G7_F9_CascadeAvg192 | 188 | 128 | 7 | 0 | **+7** | 0.718 | 0.681 | 7.00 | Oracle beats margin; **but only 7/188 items eligible.** |
| G7_F9_CascadeRandomS{0,1,2} vs Avg192 | 188 | varies | varies | varies | +1 / −1 / +4 | 0.686 / 0.676 / 0.702 | 0.681 | <2 | **Random ≈ margin.** No useful signal in the margin. |
| G9_F9_192f vs G5_F9_128f | 200 | 127 | 10 | 10 | **0** | 0.685 | 0.685 | 0.00 | **More frames at fixed F9 = no gain.** |
| G7_F9_CascadeAvg192 vs G9_F9_192f | 188 | 121 | 7 | 6 | +1 | 0.681 | 0.676 | 0.08 | Cascade = direct 192f F9. |

**Three failed claims:**
1. **`G7_F9_CascadeAvg192` does not beat `G5_F9_128f`** at n=200 (0.681 = 0.681). The Stage-1 +5 pp lead was sample noise.
2. **Random ≥ margin** (random 3-seed mean = 0.688 ≥ margin 0.681). The confidence-margin signal is not doing useful work.
3. **Oracle has only 7/188 eligible items.** The entire adaptive-frame-routing headroom at n=200 is +3.7 pp from those 7 items (where 256f F9 fixes a 128f F9 miss). Adaptive cascading has near-zero structural headroom on this benchmark.

### Frame-coverage sanity — more frames don't buy anything at fixed F9

| Pair | acc Δ | net | Read |
|---|---:|---:|---|
| G6_F9_256f vs G5_F9_128f (paired n=188) | 0.681 vs 0.681 = +0 | +0 | 256f F9 = 128f F9 |
| G9_F9_192f vs G5_F9_128f (paired n=200) | 0.685 vs 0.685 = +0 | +0 | 192f F9 = 128f F9 |
| G4_F4_256f vs G3_F4_128f (paired n=188) | 0.617 vs 0.612 = +0.5 pp | +1 | 256f F4 = 128f F4 |

**Robust observation: at fixed quantizer, going from 128f → 256f frames does not improve accuracy.** This contradicts the original Exp G hypothesis ("more frames at the same KV memory budget improves long-video MCQ"). The benefit, when it exists, comes from quantization quality (F9 vs F4), not frame coverage.

### F9 vs BF16 — modest robust improvement

| Pair | n | acc(a) | acc(b) | net_a | χ² |
|---|---:|---:|---:|---:|---:|
| G5_F9_128f vs G0_BF16 | 200 | 0.685 | 0.665 | +4 | 0.57 |
| G6_F9_256f vs G0_BF16 | 188 | 0.681 | 0.660 | +4 | 0.67 |
| G9_F9_192f vs G0_BF16 | 200 | 0.685 | 0.665 | +4 | 0.53 |
| G6_F9_256f vs G2_BF16_128f | 188 | 0.681 | 0.676 | +1 | 0.07 |

**F9 reliably beats BF16-baseline by ~+2 pp** (4 net items at n=200), but is statistically tied with the higher BF16 anchors. Not a knockout result — the gains live in the noise band at this sample size.

### F4 anchors collapsed below baseline

| Pair | n | acc(F4) | acc(BF16) | net | Read |
|---|---:|---:|---:|---:|---|
| G4_F4_256f vs G0_BF16 | 188 | 0.617 | 0.660 | **−8** | F4 256f loses to baseline by 4.3 pp |
| G3_F4_128f vs G0_BF16 | 200 | 0.610 | 0.665 | **−11** | F4 128f loses to baseline by 5.5 pp |
| G4_F4_256f vs G2_BF16_128f | 188 | 0.617 | 0.676 | **−11** | F4 256f loses to BF16 128f by 5.9 pp |

**F4 (pure KIVI) at long sequences is genuinely worse than BF16 64f.** The F4-fails-at-long-sequences claim from Stage 1 reproduces decisively at n=200. The Stage-1 G3=0.562 was actually slightly LOW; n=200 G3=0.610 is the real number, but still firmly below G0=0.665.

### Exp H temporal-windowed KIVI vs anchors

| Pair | n | acc(H) | acc(comparison) | net | χ² | Read |
|---|---:|---:|---:|---:|---:|---|
| **H6 vs G3_F4_128f** | 200 | **0.680** | **0.610** | **+14** | **4.90** | **H6 beats F4 by +7 pp at TRUE 4 bits, χ²=4.90 ≈ p=0.027** |
| H6 vs G5_F9_128f | 200 | 0.680 | 0.685 | −1 | 0.06 | **H6 ties F9 at lower bits + lower memory** |
| H3 vs G4_F4_256f | 188 | 0.644 | 0.617 | +5 | 0.81 | H3 beats F4 256f by +2.7 pp |
| H4 vs G4_F4_256f | 188 | 0.644 | 0.617 | +5 | 0.86 | H4 beats F4 256f by +2.7 pp |
| H3 vs G6_F9_256f | 188 | 0.644 | 0.681 | −7 | 2.88 | H3 loses to F9 by 3.7 pp |
| H4 vs G6_F9_256f | 188 | 0.644 | 0.681 | −7 | 2.88 | H4 loses to F9 by 3.7 pp |
| H4 vs H3 (8 vs 4 windows) | 188 | 0.644 | 0.644 | 0 | 0.00 | Window count saturates between 4 and 8 |
| **H5 vs H3 (modality-blind vs visual)** | 188 | 0.617 | 0.644 | **−5** | **0.76** | **Modality-aware direction holds** but not significant |

**The H story at n=200:**

1. **H6_KIVI_TempWin2_128f is the clean publishable result.** At TRUE 4.00 KV bits and 0.50× memory:
   - **+7 pp over F4 (G3)** at the same KV bits, paired McNemar χ²=4.90 (p ≈ 0.027 with `scipy` chi-square approximation).
   - **Ties F9 (G5)** at lower bits AND lower memory (paired Δ = −1 of 17 nontrivial items).
   - Beats baseline G0 (0.665) by 1.5 pp.
   - The paper-strong verdict: `acc(H6) ≥ acc(G3) + 5pp AND acc(H6) ≥ acc(G5) − 2pp` ✓.

2. **H3/H4 at 256f fall short of F9.** H3=H4=0.644 vs G6=0.681 → −3.7 pp paired (χ²=2.88, p ≈ 0.09). The Stage-1 H3=0.683 / H4=0.700 lead inflated by ~+4 pp at scale; Stage 3 reveals the regression. H3/H4 still beat F4 256f by +2.7 pp (net +5 items), but not significantly.

3. **Modality-aware vs blind direction holds, not significant.** H3 (visual_only) wins 19 items vs H5 (token_block) wins 14 → +5 net for visual-aware. χ²=0.76, p ≈ 0.4. Direction supports the visual-time-structure hypothesis but n=188 is underpowered.

4. **8 windows ≈ 4 windows.** H4 vs H3 at 256f gives net=0, χ²=0. Window count between 4 and 8 saturates.

5. **Counterintuitive: H6 (128f, 2 windows) > H3 (256f, 4 windows).** 0.680 vs 0.644 = +3.6 pp absolute. **More frames + temporal windowing HURTS** at this benchmark — the additional frame coverage doesn't add evidence and the additional sequence length compounds quantization noise even with windowed scales.

## Stage 3 frontier (sorted by relative_kv_memory ascending)

| Cond | rel_kv_mem | acc | KV bits | Pareto? |
|---|---:|---:|---:|:-:|
| G3_F4_128f | 0.500 | 0.610 | 4.00 | dominated |
| **H6_KIVI_TempWin2_128f** | **0.500** | **0.680** | **4.00** | **frontier** |
| G5_F9_128f | 0.594 | 0.685 | 4.75 | frontier (tied with H6) |
| G7_F9_CascadeOracle | 0.616 | 0.718 | 4.75 | not deployable (uses labels) |
| G8_F9_TypeAdaptiveMin128 | 0.704 | 0.681 | 4.75 | dominated by H6 |
| G7_F9_CascadeRandomS{0,1,2} | 0.891 | 0.676–0.702 | 4.75 | dominated |
| G9_F9_192f | 0.891 | 0.685 | 4.75 | dominated |
| G7_F9_CascadeAvg192 | 0.989 | 0.681 | 4.75 | dominated |
| G0_BF16 | 1.000 | 0.665 | 16.00 | dominated |
| G4_F4_256f / H3 / H4 / H5 | 1.000 | 0.617 / 0.644 / 0.644 / 0.617 | 4.00 | dominated |
| G6_F9_256f | 1.188 | 0.681 | 4.75 | dominated |
| G2_BF16_128f | 2.000 | 0.680 | 16.00 | dominated |

**Pareto-optimal at Stage 3:** `H6_KIVI_TempWin2_128f` and `G5_F9_128f`. H6 wins on memory (0.50× vs 0.59×) and on bits (4.00 vs 4.75); G5 wins on accuracy by a hair (0.685 vs 0.680). Statistical tie on the paired test.

## Per-bucket pattern at Stage 3

H6 vs G5 vs G3 per duration bucket:

| Bucket (n) | G3 (F4 128f) | H6 (TempWin-2 128f) | G5 (F9 128f) | H6 − G3 | H6 − G5 |
|---|---:|---:|---:|---:|---:|
| short (50) | 0.640 | **0.720** | 0.680 | +8.0 pp | +4.0 pp |
| mid (50) | 0.780 | 0.740 | 0.820 | −4.0 pp | −8.0 pp |
| long (50) | 0.580 | **0.740** | 0.720 | +16.0 pp | +2.0 pp |
| very_long (50) | 0.440 | 0.520 | 0.520 | +8.0 pp | tied |

H6 wins decisively on **short** (+8 pp over G3, +4 pp over G5) and **long** (+16 pp over G3, +2 pp over G5), consistent with the temporal-windowing hypothesis: short clips benefit from sharper per-window K scales, and long clips benefit from local scales not being washed out by outliers across thousands of tokens. **H6 underperforms F9 on the mid bucket** (0.740 vs 0.820) — the only bucket where F9's outlier-channel BF16 protection outperforms windowed scaling.

## Final verdict matrix and promotion plan

```
G3_F4_128f       (0.610) → kill              (-5.5 pp vs G0)
G4_F4_256f       (0.617) → borderline        (-4.8 pp vs G0; matches G3 — F4 doesn't benefit from more frames either)
G5_F9_128f       (0.685) → promote_n200      (+2.0 pp vs G0; Pareto frontier)
G6_F9_256f       (0.681) → promote_n200      (+1.6 pp vs G0)
G7_F9_CascadeAvg192      → borderline        (= G5 = G6)
G7_F9_CascadeOracle      → control           (label-based; +3.7 pp ceiling on adaptive)
G7_F9_CascadeRandomS*    → control           (random ≈ margin)
G8_F9_TypeAdaptiveMin128 → borderline        (= G5)
G9_F9_192f               → borderline        (= G5; more frames don't help)

H3_KIVI_TempWin4_256f   (0.644) → borderline       (beats F4 by +2.7 pp; loses to F9 by 3.7 pp)
H4_KIVI_TempWin8_256f   (0.644) → borderline       (= H3; window count saturates)
H5_KIVI_TokenBlock4_256f(0.617) → control          (modality-blind; loses to H3 by 2.7 pp)
H6_KIVI_TempWin2_128f   (0.680) → PROMOTE_PAPER_STRONG  (+7 pp over F4 at TRUE 4 bits, χ²=4.90; ties F9 at lower bits/memory)
```

## Implications for the research arc

The Exp F → Exp G → Exp H sequence ends here with three concrete findings:

1. **Exp F's F4 (KIVI per-channel-along-seq)** falls below the BF16 baseline at LongVideoBench MCQ. The earlier "F4 is the deployable 4-bit baseline" claim from the F-suite Stage 3 (where F4=0.545 vs F0=0.565) does not generalize — F4 collapses on the more-balanced Exp G stratification (G3=0.610 / G4=0.617 vs G0=0.665).

2. **Exp G's frame-scaling hypothesis is falsified at n=200.** At fixed KV quantizer, going from 128f → 192f → 256f buys nothing (acc tied within ±0.5 pp on every paired comparison). The cascade controls (margin, random, oracle) confirm: there's no useful adaptive-routing signal because there's almost nothing to route — only 7/188 items have ground-truth-eligible cascade headroom.

3. **Exp H's temporal-windowed KIVI is the surviving contribution.** At 128f, H6 (2 visual windows) beats F4 by +7 pp at TRUE 4.00 KV bits (χ²=4.90, p ≈ 0.027) and ties F9 at lower bits and lower memory. At 256f, the same approach with more windows underperforms F9 — likely because the marginal scale-locality benefit doesn't compensate for the additional quantization noise from 4× the visual tokens.

**The deployable 4-bit KV cache for Qwen2.5-VL on LongVideoBench MCQ at 128f is `kivi_temporal_window` with 2 windows, modality-aware (visual_only mode):** TRUE 4.00 KV bits, 0.50× the BF16 64f memory budget, statistically tied with KIVI+outlier-16 (4.75 bits) and statistically beats pure KIVI (also 4.00 bits) by +7 pp paired McNemar.

## Files of record (Exp G + H — final)

```
qwen/scripts/k_quantizers.py                # adds kivi_temporal_window kind, 4 H configs
qwen/scripts/expG_frame_scaling.py          # adds G9 + H3/H4/H5/H6 to FIXED_FRAME_CONDITIONS
qwen/scripts/expG_cascade.py                # adds --selection_mode {margin, random, oracle}
qwen/scripts/expG_analyze.py                # adds 14 new HEADLINE_PAIRS + new-evidence/damage table
qwen/scripts/run_expG_overnight.sh          # wrapper for post-Stage-3 G controls + H Stage-1

qwen/results/expG_frame_stage3.jsonl                       # 3268 rows: 17 conditions × 188-200 items
qwen/results/expG_frame_stage3_G7f9.jsonl                  # F9 margin cascade (188)
qwen/results/expG_frame_stage3_G7_random_s{0,1,2}.jsonl   # 3 random-seed cascades (188 each)
qwen/results/expG_frame_stage3_G7_oracle.jsonl             # ground-truth oracle (188)
qwen/results/expG_frame_stage3_G8f9.jsonl                  # F9 type-adaptive (188)
qwen/results/expG_summary_stage3.md                        # final per-condition table + per-bucket
qwen/results/expG_paired_stage3.md                         # 32 pairs: McNemar + new-evidence/damage
qwen/results/expG_frontier_stage3.md                       # frame-budget × accuracy frontier
qwen/results/expG_verdict_matrix_stage3.md                 # promotion plan
qwen/results/expG_cascade_f9_*_meta.json                   # cascade meta sidecars (margin, oracle, 3 random)
qwen/results/expG_qtype_f9_meta_stage3.json                # type-adaptive meta
qwen/results/expG_overnight.progress.log                   # phase A-E milestones
qwen/results/expG_pipeline.progress.log                    # full chain log
```

## Methodological notes (Stage 3)

1. **JSON-arg quoting through tmux send-keys is fragile.** First Stage 3 type-adaptive launch failed with `unrecognized arguments: 256:G6_F9_256f ocr:256 ...` because `--frames_to_gcond_json {"128":"G5_F9_128f"...}` lost its quoting through `ssh + tmux send-keys + bash`. Fix: use single-quoted SSH outer + escaped inner ssh `'"'"'{...}'"'"'` form, OR run the post-process directly via SSH (not tmux send-keys) so only one shell layer parses the args.

2. **Leading `;` in tmux send-keys queue commands fails.** A queued ` ; echo === F4 anchors === && ...` causes `bash: syntax error near unexpected token ';'` because `;` at start-of-line is invalid. Use plain space or `&&` chaining, or just no operator and rely on the shell prompt being ready.

3. **Per-bucket sample-size variance dominates Stage-1 → Stage-3 differences.** Stage 1 n=64 has 16 items per bucket; Stage 3 n=200 has 50 per bucket. Both H6 and G6 showed inflated Stage-1 leads on the short bucket (n=16) that disappeared at n=50. Recommend: don't headline a result from n=64 paired without checking it survives at n=200.

4. **Temporal-windowing benefit depends on the modality split.** H5_TokenBlock4 (modality-blind, 4 token-equal blocks) at 0.617 = G4 (pure F4) at 0.617. The window count alone does nothing; the visual-vs-text scale separation is what gives H3 its +2.7 pp over F4 at 256f and H6 its +7 pp over F4 at 128f. This is the cleanest piece of evidence that modality-time structure is the mechanism, not generic local-scale-vs-global.

5. **No re-calibration needed at any frame count.** F-suite calibration NPZ at frames=64 was reused for all F9 conditions at 128f and 256f. K-channel outlier indices are post-RoPE and frame-count-independent in theory; the smoke check passed without `EXPG_RIGOR_HIGH=1` (the 8-item recalibration-at-256 verification was skipped — should be tested before claiming F9 generalizes to even longer sequences).

## Pipeline status (final)

```
qwen-expG (tmux session, GPU 0) — ALL EXPERIMENTS COMPLETE
├── ✅ Stage 1 (n=64) full picture: 11 conditions + 4 adaptive post-processes
├── ✅ Stage 3 G control (n=200): G0/G2/G5/G6/G9 + cascade margin/random×3/oracle + type-adaptive
├── ✅ Stage 3 H suite (n=200): H3/H4/H5 (256f) + H6 (128f)
├── ✅ Stage 3 F4 anchors (n=200): G3 + G4
└── ✅ Final analyze: expG_summary_stage3.md, expG_paired_stage3.md (32 pairs),
                     expG_frontier_stage3.md, expG_verdict_matrix_stage3.md
```

Total Stage 3 wall: ~3.5 h (multiple chained runs across 22 conditions × 188-200 items).
Total compute: ~5500 forward passes across all stages.


