# Research Plan: Quantization of Flow-Matching VLAs

## Research Question

**How does weight quantization of the VLM backbone in flow-matching VLAs affect downstream trajectory quality, and does the indirect error propagation pathway create different sensitivity patterns than the direct pathway studied in autoregressive VLAs?**

In autoregressive VLAs (OpenVLA, RT-2), the LLM backbone directly produces action tokens. QVLA (ICLR 2026) showed that standard LLM quantization methods fail here — you need action-centric, channel-wise sensitivity analysis because uniform quantization causes compounding errors in closed-loop control.

But flow-matching VLAs (pi0, Alpamayo-R1, GR00T-N1) have a fundamentally different architecture. The VLM backbone produces hidden states and a KV cache, which a **separate action expert** consumes across **multiple denoising iterations** via cross-attention. This creates an indirect error propagation pathway:

```
Autoregressive VLA (QVLA's setting):
  VLM weights → action tokens (direct output)

Flow-matching VLA (our setting):
  VLM weights → hidden states / KV cache → cross-attention → action expert → N denoising steps → trajectory
```

This architectural difference raises two questions that nobody has answered:

1. **Does the action expert act as an error buffer or an error amplifier?** The attention interface between the VLM and action expert might filter out small quantization errors (making flow-matching VLAs more robust than AR VLAs). Or the iterative denoising might amplify errors because the action expert reads the same degraded KV cache N times.

2. **Do standard LLM sensitivity metrics (perplexity) predict trajectory quality?** QVLA showed that in AR VLAs, perplexity-based metrics are insufficient — you need action-space sensitivity. But in flow-matching VLAs, the decoupled architecture might change this relationship entirely.

### Why This Matters

Flow-matching VLAs are becoming the dominant architecture for robotic control (pi0, GR00T-N1, Alpamayo-R1). Deploying them on resource-constrained robots (Jetson, edge GPUs) requires quantization. But the only systematic VLA quantization study (QVLA) was done on autoregressive architectures. If flow-matching VLAs have different quantization dynamics — which the architectural differences strongly suggest — then the field needs a dedicated analysis.

---

## Connection to BlockDialect

This research directly extends Wonsuk's BlockDialect work in several ways:

### 1. Action-aware formatbook design

BlockDialect assigns an optimal FP4 dialect to each fine-grained block based on the block's data distribution, using a formatbook of 16 dialects with different maximum magnitudes and representable value configurations. Currently, the dialect is selected to minimize **reconstruction error** (MSE between original and quantized values).

In a flow-matching VLA, the right objective for dialect selection may not be reconstruction error but **action-space error** — how much a block's quantization affects the final trajectory. This mirrors QVLA's core finding that action-centric sensitivity analysis outperforms standard LLM metrics, but now applied to BlockDialect's fine-grained format selection.

**Concrete research question:** If we profile BlockDialect's per-block dialect assignments using action-space sensitivity (trajectory ADE) instead of MSE, do we get different optimal assignments? If yes, we can build an "action-aware formatbook" that preserves trajectory quality better than the standard formatbook at the same effective bitwidth.

### 2. Module-aware formatbook specialization

BlockDialect currently uses a single global formatbook across all layers. But QVLA's Figure 1(a) shows that VLA modules have dramatically different quantization sensitivity — the vision encoder is robust, the LLM backbone is moderate, and the projector/action head are extremely sensitive.

This suggests that the formatbook itself should be **module-specific**: the VLM backbone might work well with the standard 16-dialect DialectFP4, while the projector and action expert layers might need a specialized formatbook with higher-precision dialects. This is a natural extension of BlockDialect's "if a group of numbers deserves its own scaling factor, why not a number format?" philosophy — if different modules have fundamentally different distributions, they deserve different formatbooks.

### 3. KV cache format selection for denoising

In flow-matching VLAs, the action expert reads the VLM's KV cache across multiple denoising steps via cross-attention. This is a unique setting where:

- The KV cache is written once (by the VLM) but read many times (by the action expert at each denoising iteration)
- Small format-level errors in the cached K/V vectors get amplified through repeated reads
- Different KV cache blocks may have very different impact on the final trajectory

BlockDialect's online two-stage dialect selection (Section 3.2 of the paper) could be applied here: as the VLM produces K/V vectors, assign each block a dialect based on a fast estimate of its action-space importance. Blocks that the action expert attends to heavily across denoising steps get higher-precision dialects; blocks that are barely attended to get lower-precision dialects or even pruned.

This is the bridge between BlockDialect (hardware-efficient fine-grained format selection) and VLA quantization (action-aware sensitivity) — and it's something that only makes sense for flow-matching architectures where the KV cache is an explicit interface between modules.

### 4. Online vs. offline dialect selection

A key insight from DP-LLM is that sensitivity changes dynamically across decoding steps. In a VLA setting, this is even more pronounced: the sensitivity of a KV cache block may depend on what the action expert is doing at each denoising step. Early denoising steps (coarse trajectory shape) might be robust to KV cache quantization, while late denoising steps (fine trajectory details) might be very sensitive.

BlockDialect's efficient online dialect selection could potentially be extended to support **denoising-step-aware** format assignment — using a higher-precision dialect for KV cache reads during fine denoising steps and a lower-precision dialect during coarse steps. This is speculative but would be a novel contribution if the sensitivity analysis in our experiments supports it.

---

## Prior Work and Positioning

| Paper | What it does | Gap relative to our work |
|-------|-------------|------------------------|
| **QVLA** (ICLR 2026) | Action-centric channel-wise weight quantization for AR VLAs (OpenVLA). Shows standard LLM quantization fails for VLAs. | Only AR architectures. No flow-matching VLAs. |
| **EAQVLA** (2025) | Encoding-aligned quantization for VLAs. | Also AR-focused. |
| **BlockDialect** (ICML 2025) | Block-wise fine-grained mixed format quantization with formatbook of FP4 dialects. | Applied to general LLM weights/activations, not VLA-specific. |
| **MicroMix** (2025) | Mixed-precision MXFP4/6/8 with adaptive channel allocation. | LLM-focused, not action-aware. |
| **KVQuant** (NeurIPS 2024) | KV cache quantization for LLMs with per-layer NUQ. | LLM text generation only, no trajectory/action quality. |
| **pi0** (2024) | Flow-matching VLA architecture. | No quantization study. |
| **TinyVLA** (2024) | Architectural compression for VLAs. | Focuses on distillation/pruning, not quantization. |
| **EfficientVLA** (2025) | Training-free VLA acceleration. | Focuses on token pruning, not weight/activation quantization. |

**Our positioning:** The first systematic study of quantization in flow-matching VLAs, revealing how the indirect error propagation pathway (VLM → KV cache → action expert → iterative denoising) creates different sensitivity patterns than the direct pathway studied in prior work. Combined with BlockDialect's fine-grained format selection, this enables action-aware, hardware-efficient quantization for the dominant VLA architecture class.

---

## Experimental Plan

### Setup Requirements

**Model:** pi0 (or a comparable open-source flow-matching VLA)
**Benchmark:** LIBERO (4 task suites: Spatial, Object, Goal, Long)
**Hardware:** 1x H100 GPU (inference + rollouts)
**Quantization tools:** AWQ, GPTQ, SmoothQuant (off-the-shelf baselines)
**Estimated timeline:** 1-2 weeks total

### Experiment 1: Does Standard LLM Quantization Break Flow-Matching VLAs? (1-2 days)

**Goal:** Establish whether off-the-shelf LLM quantization methods work for flow-matching VLAs, or whether they break as QVLA showed for AR VLAs.

**What to do:**
1. Set up pi0 with full-precision (BF16) VLM backbone on LIBERO. Run baseline evaluation across all 4 suites. Record task success rates.
2. Apply AWQ (W4A16) to the VLM backbone **only** — leave the action expert in full precision. Re-run LIBERO evaluation.
3. Apply SmoothQuant (W8A8) to the VLM backbone. Re-run evaluation.
4. Apply more aggressive quantization: W4A8 and W3A16 (if supported). Re-run.

**What you're looking for:**

| Outcome | What it means | Next step |
|---------|--------------|-----------|
| Success rates drop >5% with W4A16 | Standard LLM quant breaks flow-matching VLAs too → action-aware methods needed | Proceed to Experiment 3 |
| Success rates drop <2% with W4A16 | Action expert buffers errors → flow-matching VLAs are more robust than AR VLAs | Still interesting — push to W3A16 or W2A16 to find the breaking point |
| Spatial/Object tasks survive but Long tasks break | Error accumulation over long horizons is the bottleneck | Focus Experiment 2 on temporal error analysis |

**Deliverable:** Table comparing FP16 vs. quantized success rates across LIBERO suites + comparison to QVLA's OpenVLA results at the same bitwidths.

### Experiment 2: Where Does Quantization Error Concentrate? (2-3 days)

**Goal:** Trace where quantization error accumulates in the flow-matching VLA pipeline. This reveals whether the action expert amplifies or filters VLM quantization errors.

**What to do:**

For a set of ~50 test episodes, run inference with both FP16 and W4A16 models and record at each timestep:

1. **Reasoning trace text** — Compare FP16 vs. quantized reasoning via text similarity (ROUGE/BERTScore). This measures how much the VLM's language output is affected.

2. **KV cache hidden states at the last VLM layer** — Compute L2 distance between FP16 and quantized KV cache. This is what the action expert actually sees.

3. **Denoising intermediate states** — At each denoising step (e.g., steps 1, 5, 10, 20), record the intermediate trajectory. Compute L2 distance from the FP16 trajectory at the same denoising step.

4. **Final trajectory** — Compute Average Displacement Error (ADE) between FP16 and quantized final trajectories.

**What you're looking for:**

Plot error magnitude at each stage of the pipeline:

```
VLM text output error → KV cache error → denoising step 1 error → ... → final trajectory error
```

- **Error amplification pattern:** KV cache error is small but trajectory error is large → the action expert amplifies errors through iterative denoising. This means you need action-aware quantization (supports QVLA-like approach for flow-matching VLAs).

- **Error filtering pattern:** KV cache error is significant but trajectory error is small → the action expert is robust and acts as a low-pass filter. Standard LLM quantization may be sufficient, but you can push to more aggressive compression.

- **Error accumulation over time:** Plot trajectory ADE vs. timestep within an episode. If error grows superlinearly, long-horizon tasks are disproportionately affected by quantization (matches QVLA's temporal accumulation finding in their Figure 3).

**Deliverable:** Error propagation diagram showing how quantization error flows through each stage of the pipeline. This is the key figure for the paper.

### Experiment 3: Per-Layer Sensitivity — Perplexity vs. Trajectory (3-4 days)

**Goal:** The core experiment. Quantize each VLM layer independently and measure both perplexity and trajectory quality. If the two sensitivity curves diverge, that's the main finding.

**What to do:**

For each layer $l$ of the VLM backbone (one at a time, everything else in FP16):
1. Quantize layer $l$ to W4A16
2. Measure perplexity on reasoning trace text (standard LLM metric)
3. Measure trajectory ADE across 50 LIBERO episodes (action-space metric)
4. Record both metrics

Repeat for all layers. This produces two sensitivity curves across layers.

**Analysis:**

Plot both curves on the same graph (dual y-axis: perplexity on left, trajectory ADE on right). Compute Spearman rank correlation between the two.

| Correlation | What it means |
|-------------|--------------|
| r > 0.8 | Perplexity predicts trajectory quality well → standard LLM quantization metrics are sufficient for flow-matching VLAs |
| 0.4 < r < 0.8 | Partial correlation → some layers matter differently for text vs. actions, mixed-precision should account for this |
| r < 0.4 | **Strong divergence** → perplexity is a poor proxy for trajectory quality → action-aware sensitivity is essential, even for flow-matching VLAs |

**Why divergence would occur:** Some VLM layers might primarily encode spatial/geometric information that the action expert relies on heavily but that barely affects text perplexity. Conversely, some layers might be critical for coherent reasoning text but the action expert doesn't attend to their KV cache entries.

**Deliverable:** Dual-axis sensitivity plot across layers. This is the primary evidence for or against the need for action-aware quantization in flow-matching VLAs.

### Experiment 4 (Stretch): BlockDialect Format Selection Comparison (3-5 days)

**Goal:** Directly test whether BlockDialect's dialect selection changes when optimized for action-space error vs. reconstruction error.

**What to do:**
1. Apply BlockDialect with standard MSE-based dialect selection to the VLM backbone. Evaluate on LIBERO.
2. Profile per-block action-space sensitivity: for each block, measure how the dialect choice affects trajectory ADE.
3. Re-assign dialects using action-space sensitivity as the objective instead of MSE.
4. Compare LIBERO success rates between MSE-optimized and action-optimized dialect assignments at the same effective bitwidth.

**If action-optimized dialects outperform MSE-optimized dialects**, that's the direct evidence that BlockDialect can be extended with action-aware formatbook selection for VLAs.

---

## Decision Points

### After Experiment 1:
- If standard quantization works fine (< 2% drop at W4A16): The story shifts to "flow-matching VLAs are surprisingly robust to backbone quantization — here's why and how far you can push it." Still publishable but different framing.
- If it breaks significantly (> 5% drop): Proceed with full study. The narrative is "action-aware quantization is needed for flow-matching VLAs, and here's how the sensitivity patterns differ from AR VLAs."

### After Experiment 3:
- If perplexity and trajectory sensitivity correlate strongly: Standard LLM quantization metrics are sufficient for flow-matching VLAs (unlike AR VLAs). The action expert's cross-attention interface provides enough decoupling. Contribution: characterizing this robustness and pushing to more aggressive compression.
- If they diverge: The main finding. Different layers matter for text vs. actions, and you need action-aware mixed-precision. This motivates extending BlockDialect with action-aware format selection (Experiment 4).

---

## What to Bring to Wonsuk

1. **Experiment 1 results:** Table of success rates at various quantization levels. Direct comparison to QVLA's OpenVLA results. This immediately shows whether flow-matching VLAs behave differently.

2. **Experiment 2 error propagation diagram:** Where does quantization error concentrate — does the action expert amplify or filter it? This is visually compelling and tells the mechanistic story.

3. **Experiment 3 dual sensitivity plot:** The key figure. If perplexity and trajectory sensitivity diverge, pitch: "BlockDialect's dialect selection should be optimized for action-space error, not reconstruction error, when applied to VLA quantization."

4. **Concrete proposal for BlockDialect extension:** Based on the sensitivity analysis, propose how the formatbook or dialect selection criteria should be modified for VLAs. This connects your empirical findings directly to Wonsuk's ongoing work.

---

## Timeline

| Day | Activity |
|-----|----------|
| 1 | Set up pi0 + LIBERO environment, verify FP16 baseline |
| 2 | Experiment 1: AWQ/SmoothQuant on VLM backbone, evaluate |
| 3-4 | Experiment 2: Error propagation analysis |
| 5-7 | Experiment 3: Per-layer sensitivity sweep (most compute-intensive) |
| 8-9 | Analysis, plotting, write-up |
| 10+ | Experiment 4 (stretch): BlockDialect comparison |

**Total estimated compute:** ~50-80 H100 hours (mostly Experiment 3 rollouts).
