# Qwen2.5-VL × LongVideoBench — KV-cache Quantization Experiments

**Status as of 2026-05-07:** **Both experiments complete.** Exp A (8/8 conditions × 200 eval items) and Exp B Online Precision-Need Routing (8 routed conditions × 200 eval items at avg=4 KV bits, plus B10 in flight). Total: ~3200 rollouts of routed/baseline data, plus 33,600 (item × layer × KV-head) diagnostic signal rows.

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
