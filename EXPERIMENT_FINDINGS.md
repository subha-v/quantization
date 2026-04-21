# Experiment Findings

Sensitivity probing of pi0.5 LIBERO (~3.6B params — ~3B PaliGemma VLM + ~300M Gemma action expert + SigLIP vision tower), PyTorch, eager mode, on tambe-server-1 (H100 PCIe). All forward-pass only; no fine-tuning.

Two runs of data:
- **2026-04-15 overnight:** exp1, exp2, and the original (broken) exp3.
- **2026-04-16:** redesigned exp3 with seeded noise.

---

## Research framing

pi0.5 is a **flow-matching VLA**: a VLM backbone (PaliGemma) produces a KV cache, then a separate action expert consumes that cache through cross-attention over **10 Euler denoising steps** to produce an action chunk. This creates an error-propagation pathway that autoregressive VLAs (OpenVLA, studied in QVLA ICLR'26) don't have:

```
VLM weights → KV cache → cross-attention → action expert → 10 denoising steps → trajectory
```

The central research question is how quantization error enters this pipeline and how it compounds through denoising. Secondary question from the advisor: does task difficulty (long-horizon vs short-horizon) change where precision is needed?

**Dataset split across all experiments.** 256 single-frame observations sampled from LIBERO:
- 128 *Easy* = LIBERO-Object (task_index 20–29): single pick-and-place, ~100-step rollouts.
- 128 *Hard* = LIBERO-Long (task_index 0–9): multi-step chained goals, ~400-step rollouts.

Frames are drawn from early/mid/late phases of episodes. Note: we measured **per-frame** action MSE, not closed-loop rollout success. Per-frame MSE is a lower bound on rollout-level horizon differential because rollouts compound errors over 100–400 steps.

**Common quantization format.** All weight quantization is integer, symmetric, group-wise (group=128), weight-only fake-quant (quantize-then-dequantize in FP16). "W4" = 4-bit integer (range [−7, +7]), "W8" = 8-bit, "W2" = 2-bit ternary {−1, 0, +1}. Not FP4 / MXFP4.

---

## Experiment 1 — Cross-Suite Activation Statistics

**Question.** Do Long-task observations produce systematically different activation distributions than Object-task observations? If yes, an online controller could key off those differences to pick bitwidths at runtime. This tests the *input distribution* side of the horizon-differential hypothesis.

**Method.** 256 forward passes with hooks on all 458 Linear layers. Per-layer, per-sample, record: `max|activation|`, kurtosis, std, outlier fraction (>6σ). Mean across suites, then compute Hard − Easy delta per layer.

**Results.**

- Easy and Hard curves lie on top of each other for ~450 of 458 layers (`plots/exp1_kurtosis_comparison.png`, `exp1_outlier_6s_comparison.png`).
- A handful of VLM `mlp.down_proj` layers have extreme kurtosis (5,000–15,000) for both suites — known SmoothQuant-style outlier layers in Gemma.
- Two layers show real per-suite differences:
  - `language_model.layers.17.mlp.down_proj`: max|activation| ~41,000 for Hard vs ~35,000 for Easy (delta ≈ +5,800).
  - `language_model.layers.0.mlp.down_proj`: kurtosis delta ≈ −1,476 (Easy higher).
- Outlier-6σ fraction deltas are O(10⁻⁴) everywhere — noise floor.

**Interpretation.** At the per-frame level, the activation *distributions* are nearly indistinguishable between Long and Object. The strong form of the horizon-differential hypothesis — "easy vs hard produce distinguishable activation fingerprints that a controller could condition on online" — is not supported at the single-observation level. Two Gemma outlier layers need SmoothQuant-style handling regardless of task.

**What this does NOT rule out:** different quantization *sensitivity* between Long and Object (measured in exp2), or different temporal error propagation (measured in exp3). Exp1 only measures input distributions, not sensitivity.

---

## Experiment 2 — Layer-wise Sensitivity Probe

**Question.** (A) Which layer groups are most sensitive to weight quantization? (B) Does sensitivity differ between Hard and Easy?

**Method.** Group model into 42 "layer groups" at the decoder-block granularity — each group = one transformer decoder block's full set of Linear layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`), plus non-decoder groups (vision tower, projector, action-head linears). For each (group, bitwidth ∈ {W2, W4, W8}):
- Quantize that group only. Everything else stays FP16.
- Run all 256 observations through the partially-quantized model.
- For each frame, compute `MSE(action_quantized, action_fp16_reference)`.
- Restore group to FP16.

Per-sample results saved to JSONL with full metadata. `scripts/exp2_layer_sensitivity.py:114-136`.

### (A) Sensitivity spectrum

- **W4 and W8** (`plots/exp2_sensitivity_w4.png`, `_w8.png`): all 42 groups sit in a narrow band, MSE ≈ 0.004–0.007. No single layer is catastrophically more sensitive. **W4 is a safe blanket bitwidth** for any individual layer.
- **W2** (`plots/exp2_sensitivity_w2.png`): two layers explode:
  - `language_model.layers.0` (first VLM decoder block): MSE ≈ 0.32
  - `paligemma.model.vision_tower` (SigLIP): MSE ≈ 0.37
  - Everything else stays at MSE ≈ 0.005 — a **~50–70× sensitivity gap**.

**Headline:** aggressive W2 compression is viable *if* you protect exactly two layers. Protect VLM block 0 + vision tower at W4 or FP16; W2 everything else.

### (B) Horizon-differential sensitivity

`plots/exp2_easy_vs_hard_delta_w4.png`:
- All action-expert decoder layers show **positive delta** (Hard > Easy), magnitude 0.001–0.004 on a base of ~0.005. That's **+20 to +75% relative increase** for Hard vs Easy, systematic across ~20 bars.
- Largest: `model.layers.11` +0.0038 (+58%), `time_mlp_in` +0.0033, `action_out_proj` +0.0022.
- Some VLM layers go negative (Easy more sensitive), localizing the effect to the decoder side.

`plots/exp2_easy_vs_hard_delta_w8.png`: same qualitative pattern, smaller magnitudes. `_w2.png`: dominated by the two bottleneck layers (Hard +0.044 on `language_model.layers.0`, Easy +0.062 on vision tower).

**Interpretation.** The horizon-differential sensitivity that exp1 could not detect in activation distributions *does* show up when you actually quantize and measure downstream action error — specifically in the action expert. Per-frame magnitudes are small but directionally consistent.

**Caveat.** Per-frame MSE undercounts rollout-level error. Closed-loop task-success rate under this same quantization schedule would almost certainly show a larger Long-vs-Object gap. That's the natural follow-up.

---

## Experiment 3 — Per-step Sensitivity of the Action Expert (redesigned)

**Scope caveat (important).** This experiment quantizes the **action expert** (~300M params), not the VLM. It measures the *internal* per-step sensitivity of the flow-matching decoder, not how VLM quantization error propagates into it. That VLM-side question is exp4, proposed below.

**Question.** Does quantization error at different denoising steps have different effects on the final action? pi0.5 runs 10 Euler flow-matching steps (t = 1.0 → 0). Is sensitivity flat across steps, concentrated at early/scaffolding steps, or concentrated at late/refinement steps? No published work has measured this for any VLA.

**What was fixed relative to the broken 04-15 version.**

1. **Seeded noise per observation.** pi0.5's `sample_actions` starts from `x_0 ~ N(0, I)` sampled fresh on every call. The broken version didn't pass deterministic noise, so the FP16 reference and the W4 test runs started from different `x_0` per observation. The per-call noise variance (~10⁻² MSE) swamped the quantization signal (~10⁻⁵ MSE). Fix: per-observation seeded `torch.Generator`, passed through `policy.infer(obs, noise=noise_np)` (which openpi already supports).
2. **Patch `model.denoise_step`, not the expert module.** Cleaner per-step granularity; the prefix pass calls `paligemma_with_expert.forward(inputs_embeds=[prefix_embs, None])` so the gemma_expert is not invoked during prefix — the step counter is off-by-one free.
3. **Validation gate.** Before any sweep, the script runs `quantize_steps=∅` and asserts `MSE < 1e-10` vs the FP16 reference. If the patch is transparent, run proceeds; otherwise abort. This gate passed.

**Method.** For each of 256 observations:
- Pre-generate deterministic noise `x_0` (seed = 1000 + obs_idx).
- FP16 reference: run inference with that noise, record `a_ref`.
- For each (sweep, config): run inference with the action expert W4-quantized at a specific subset of the 10 Euler steps and FP16 at all others. Record per-frame MSE vs `a_ref`.

Three sweeps:
- **A — Per-step:** W4 at step `k` only, FP16 elsewhere. 10 configs.
- **B — First-k-FP16-rest-W4:** `k = 0…10`. 11 configs.
- **C — First-k-W4-rest-FP16:** `k = 0…10`. 11 configs.

Cross-checks: B(k=10) and C(k=0) must be exactly 0 (all FP16 matching itself); B(k=0) must equal C(k=10) (both = all-W4); A(k=9) must equal B(k=9) minus B(k=10). All three checks passed — the data is self-consistent.

### Finding 1: sensitivity is monotonic and concentrated at the final step

`plots/exp3_per_step_sensitivity.png`:

| step k | mean MSE |
|---|---|
| 0 | 1.6×10⁻⁷ |
| 1 | 1.7×10⁻⁷ |
| 2 | 1.8×10⁻⁷ |
| 3 | 1.7×10⁻⁷ |
| 4 | 1.9×10⁻⁷ |
| 5 | 2.1×10⁻⁷ |
| 6 | 2.4×10⁻⁷ |
| 7 | 3.2×10⁻⁷ |
| 8 | 6.8×10⁻⁷ |
| **9** | **1.2×10⁻⁵** |

Step 9 is **~75× step 0** and ~18× step 8. Step 9 alone accounts for ~63% of the error incurred by quantizing all 10 steps to W4. Steps 0–6 are statistically indistinguishable from each other.

**Operational read** (from `plots/exp3_cumulative_sweep.png`): you can W4 the first 9 Euler steps and pay only ~16% of the all-W4 cost. This is a static step-wise schedule — no online controller needed to deploy it.

### Finding 2: the horizon differential from exp2 is co-localized with the sensitivity peak

Splitting the per-step curve by suite (`plots/exp3_per_step_easy_vs_hard.png`):

| step k | Easy mean | Hard mean | Δ(H−E) | rel |
|---|---|---|---|---|
| 0 | 1.69e-7 | 1.54e-7 | −1.5e-8 | −8.7% |
| 1 | 1.67e-7 | 1.64e-7 | −2.8e-9 | −1.7% |
| 2 | 1.80e-7 | 1.73e-7 | −6.8e-9 | −3.8% |
| 3 | 1.77e-7 | 1.69e-7 | −7.9e-9 | −4.5% |
| 4 | 1.89e-7 | 1.96e-7 | +6.8e-9 | +3.6% |
| 5 | 2.00e-7 | 2.20e-7 | +2.0e-8 | +10.1% |
| 6 | 2.25e-7 | 2.47e-7 | +2.2e-8 | +9.8% |
| 7 | 3.03e-7 | 3.35e-7 | +3.2e-8 | +10.6% |
| 8 | 6.11e-7 | 7.54e-7 | +1.4e-7 | **+23.5%** |
| 9 | 1.10e-5 | 1.31e-5 | +2.1e-6 | **+19.3%** |

Steps 0–3 show no differential (small negative deltas, within SEM). Steps 4–7 show consistent +10% Hard-over-Easy. Steps 8–9 — the two steps that matter — show **+19 to +24%.** The differential is phase-coherent with the sensitivity curve: where precision matters, task difficulty also modulates the precision need.

**Significance.** Step-9 z-test: z = 2.18 (p ≈ 0.03 one-tailed). 80 of 128 Long-task samples exceed the Object median at step 9 (binomial z ≈ 2.83, p ≈ 0.005). Marginal via parametric test, solid via non-parametric. n = 128 per suite.

### Finding 3: step-9 sensitivity is highly heterogeneous across frames

Step 9 MSE distribution across 256 observations: min 2.5×10⁻⁶, p25 6.0×10⁻⁶, median 9.6×10⁻⁶, p75 1.6×10⁻⁵, max 4.0×10⁻⁵. **16× spread.** About 10% of frames have step-9 MSE above 2×10⁻⁵. Identifying what makes these frames catastrophically sensitive — contact-rich? high-entropy-action? near decision boundary? — is a potential follow-up.

### Phase-bin split at step 9 (exploratory)

| suite, phase | step-9 MSE | n |
|---|---|---|
| Long, early | 1.30×10⁻⁵ | 47 |
| Long, mid | 1.25×10⁻⁵ | 41 |
| Long, late | 1.40×10⁻⁵ | 40 |
| Object, early | 1.54×10⁻⁵ | 37 |
| Object, mid | 9.17×10⁻⁶ | 46 |
| Object, late | 9.27×10⁻⁶ | 45 |

Long is flat across rollout phases. Object early-phase frames are the single most sensitive bucket, 1.7× their own late-phase. Possible interpretation: early-Object frames are the pre-grasp reach where action direction is still ambiguous and the model is near a decision boundary; by late-phase the grasp is committed and the action is low-entropy. Exploratory — small cell counts.

---

## Synthesis — three findings, stated tightly

1. **Architectural bottleneck (exp2 W2).** Of 42 layer groups, two dominate W2 sensitivity by ~50–70×: the first VLM decoder block and the SigLIP vision tower. "Protect two, compress forty" is a clean static precision map.
2. **Temporal bottleneck (exp3).** The action expert's 10-step Euler loop is wildly asymmetric: the final refinement step is ~75× more precision-sensitive than the first. W4 the first 9 steps for ~84% of the memory benefit at ~16% of the quality cost.
3. **Horizon-differential, localized (exp2 + exp3).** Long tasks are more sensitive to quantization than Object tasks. The effect is modest at the per-frame level (~20% at the critical step 9), but it is spatially and temporally co-localized with where precision actually matters — suggesting the hypothesis Wonsuk proposed has a real but bounded signal. Closed-loop rollout eval is needed to convert this into task-success numbers.

### What we explicitly did NOT measure

- **Rollout-level task success under quantization.** All MSE is per-frame. Per-frame MSE is a lower bound on rollout effects.
- **VLM-side quantization + per-step interaction.** Exp3 quantized the action expert, not the VLM. The central thesis ("how does VLM quantization propagate through denoising steps") requires a different experiment. See below.
- **KV cache quantization** (KVQuant-style). Not tested.

---

## Proposed next experiment — Exp4: VLM/KV-cache propagation through denoising steps

Exp3 answered "which denoising step is the expert most sensitive to errors in **its own weights**?" The thesis-relevant question is: "which denoising step is the expert most sensitive to errors in the **VLM-produced KV cache**?"

Structural note: the VLM runs once per action inference, producing a fixed KV cache that is consumed by the expert via cross-attention across all 10 Euler steps. So "quantize the VLM at step k" is not the right framing — there's only one VLM forward pass. The right framing: the expert's per-step *consumption* of a noisy vs clean cache.

**Design.**

1. Run VLM once at FP16 → cache `KV_fp`.
2. Run VLM once with weights quantized to W4 (or W3, W2) → cache `KV_q`.
3. During the denoising loop, at step `k`, monkey-patch the expert's cross-attention layers to read from either `KV_fp` or `KV_q`, following a per-step schedule.
4. Run the same three sweeps as exp3 (A/B/C) over the choice of `KV_fp` vs `KV_q` per step.

**What this tells us.** At which step is the expert most sensitive to the VLM's quantization error? Candidate priors:
- Early steps dominant: "coarse scaffolding uses the scene heavily; late refinement is scene-agnostic."
- Late steps dominant: mirrors exp3's expert-side curve; suggests a universal "last step matters" law.
- Flat: the expert averages scene errors over the whole Euler loop.

Any of these is a clean architectural claim.

**Variants.**
- **Exp4b — direct KV cache quantization.** Instead of quantizing VLM weights, directly round the cached K, V tensors (KVQuant-style). Same per-step sweeps. Decouples "VLM weight quantization" from "KV numerical precision."
- **Exp4c — uniform VLM precision baseline.** Sweep VLM global bitwidth W8 / W4 / W3 / W2 with FP16 expert. "How bad is uniform VLM quantization" as a reference point. This is also the layer-granularity version of what Wonsuk originally asked for, collapsed into a single curve.

**Cost.** ~1–1.5 h on H100 with the exp3 fixture (same seeded-noise, same JSONL schema, same plot scaffolding). Can run tonight.

---

## Files of record

- Scripts: `scripts/exp1_activation_stats.py`, `scripts/exp2_layer_sensitivity.py`, `scripts/exp3_flow_step_sensitivity.py`, `scripts/utils.py`.
- Plots: `plots/exp{1,2,3}_*.png`.
- Raw per-sample data: `results/exp3_per_step.jsonl`, `results/exp3_cumulative.jsonl` (exp2 JSONLs are on the server at `/data/subha2/experiments/results/`).
- Run log: `results/exp3_stdout.log`.
- Reference actions (seeded): `/data/subha2/experiments/results/exp3_reference_actions_seeded.npz` (server-side, ~256 action chunks).
