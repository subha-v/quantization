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

## Bridge to Phase 1 — pivoting from Exp4 to trajectory attention analysis

We originally planned **Exp4** (VLM/KV-cache propagation through denoising steps) as the natural follow-up to exp3. After a closer code audit, that design was reframed: openpi's `denoise_step` is called with `use_cache=False`, so the VLM-produced `past_key_values` is **never mutated across the 10 Euler steps** — it's a read-only input. "Which denoising step is most sensitive to VLM-side error" is ill-posed because the KV is temporally constant within one inference. Exp4 was deferred in favor of a more mechanistically interesting question: *do attention dynamics during a closed-loop rollout predict where quantization precision needs to be spent?*

This section documents exp0 (rollout infrastructure), exp5 (trajectory attention capture + suite classifier), exp6 (attention vs per-rollout quantization sensitivity), and exp7 (attention vs per-frame W4 action MSE). The narrative matters because exp6 produced an initial "signal dead" verdict that was walked back after a methodological critique and exp7's per-frame reanalysis.

---

## Experiment 0 — LIBERO closed-loop rollout infrastructure (2026-04-20)

**Goal.** Build a reusable in-process rollout harness for pi0.5 on LIBERO and validate it reproduces published FP16 success rates on a subset. All prior experiments (exp1-3) used static parquet observations; attention-dynamics analysis required actual closed-loop rollouts with the MuJoCo simulator.

**Method.** Single-process integration of openpi's `Policy.infer()` with LIBERO's `OffScreenRenderEnv` via a new `scripts/rollout.py`. Observation adapter matches openpi's `examples/libero/main.py` preprocessing (180° image rotation, resize_with_pad to 224, eef_pos + axisangle(quat) + gripper_qpos → state vector). Seeded per-episode via LIBERO's `task_suite.get_task_init_states(task_id)[episode_idx]`. Callback seams (`obs_callback`, `action_callback`) for downstream hook integration.

**Setup friction.** Documented in `COMMON_ERRORS.md`. Two non-obvious fixes required:
1. LIBERO's `libero/libero/__init__.py` calls `input()` at import time if `~/.libero/config.yaml` is missing. Under `uv run python -c "import libero.libero"` (stdin redirected) this blocks forever. Fix: pre-create `$LIBERO_CONFIG_PATH/config.yaml` in `run_phase0.sh`.
2. PyTorch 2.6 flipped `torch.load` default `weights_only` to True. LIBERO's legacy numpy-pickled `.pruned_init` task-state files fail the new safe loader. Fix: monkey-patch `torch.load` in `rollout.py` before LIBERO is imported.

Also: `robosuite` is unpinned in LIBERO's requirements → uv resolves to 1.5.x which broke LIBERO's API. Downgraded to `robosuite==1.4.1`.

**Result.** 3 tasks × 3 seeds × {Object, Long} = 18 rollouts at FP16 on one H100 (tambe-server-1 GPU 0).

| suite | our FP16 | published (QuantVLA Table 2) | Δ | n |
|---|---|---|---|---|
| Object | 9/9 = 100.0% | 99.0% | +1.0pp | 9 |
| Long | 9/9 = 100.0% | 93.5% | +6.5pp | 9 |

Total runtime 5.5 min on H100. Object rollouts ~8–15s each, Long ~13–18s each.

**Caveats.** n=9 per suite is too small for sharp significance claims; only first 3 tasks per suite tested. But infrastructure is validated and reusable for all subsequent experiments.

---

## Experiment 5 — Trajectory attention dynamics + Easy-vs-Hard classifier

**Framing.** Exp1 found that per-frame activation distributions don't separate Easy from Hard observations. Working hypothesis (ThinKV-inspired): difficulty signal lives in attention *dynamics* across a rollout — how attention shifts (or doesn't) as the task unfolds — not per-frame snapshots. An Object-task rollout might lock onto one object; a Long-task rollout might shift across sub-goals.

**Method.**
- 50 rollouts: 5 tasks × 5 seeds × {Object, Long} using exp0's harness.
- Monkey-patched all 45 VLM attention modules (27 SigLIP vision tower + 18 PaliGemma decoder, excluding `gemma_expert`) to force `output_attentions=True` and capture softmax probs at every VLM forward pass.
- Per (layer, head, infer-call): sparsity (ThinKV-style <1% of row max), entropy, top-1 mass, top-5 mass, attention-sink mass at position 0.
- Per rollout: aggregated across time into static features (mean) AND dynamic features (std, max, range, lag-1 autocorrelation). 45 layers × 5 metrics × 6 stats = **1350 features per rollout**.
- Classifier: logistic regression, 5-fold stratified CV, then leave-one-task-pair-out (LOTP) CV to control for task-identity leakage.

**Result: AUC = 1.000 under both CV protocols** (25 LOTP folds, per-fold range [1.00, 1.00]). Object and Long are perfectly linearly separable from attention features. Static features alone give AUC=1.000, dynamic features alone give AUC=1.000. Even 5 features per-fold selected by univariate t-test give AUC=1.000. Top-30 differentiating features all live in the **vision tower**, not the language model.

**Interpretation — the confound.** Perfect separation is suspicious. Three systematic differences between suites confound the result:
1. **Prompt grammar.** Object: "pick up X and place it in Y" (~10 tokens). Long: "put both X and Y in Z" (~15 tokens, conjunction structure). Language-model attention fingerprints grammatical shape.
2. **Scene visual complexity.** Long tasks show more objects. Vision tower attention on cluttered scenes differs from single-object scenes. Top features in vision_tower layer 0 are consistent with this.
3. **Rollout length.** Object averages 30 VLM calls, Long averages 53. Dynamic features computed over longer time series have different statistical properties by construction.

All three are correlated with "Long is harder" but none is the **quantization-relevant** signal we care about. AUC=1.0 proved attention classifies tasks cleanly — it did **not** prove attention identifies a difficulty-for-quantization signal.

**Runtime.** ~13 min on H100 for all 50 rollouts.

---

## Experiment 6 — Attention features vs per-rollout quantization sensitivity (the payoff test, part 1)

**Question.** Do the same 1350 FP16 attention features that gave AUC=1.0 actually predict which rollouts will be *quantization-sensitive*? This is the test that matters for the adaptive-precision-controller hypothesis. Exp5 answered "does attention fingerprint suite"; exp6 asks "does attention fingerprint quant-sensitivity."

**Method.**
- Reuse exp5's 50 FP16 rollouts (matched by task × seed × episode_idx for determinism).
- For each rollout, re-run under a quantization config and record outcome (success, step count, `steps_delta = quant_steps - fp16_steps`, `broke_by_quant`).
- Two configs:
  - `w4_both` — W4 on both VLM and expert (QuantVLA-mild setup)
  - `w2_vlm_protect` — W2 on VLM with layer 0 + vision_tower kept at FP16 per exp2's protection recommendation (aggressive)
- Regress FP16 attention features from exp5 against `steps_delta` with Ridge + LOTP CV. Also: binary `broke_by_quant` classifier.

**Outcome variance:**

| config | FP16 succ | quant succ | broken | Δsteps mean ± std | range |
|---|---|---|---|---|---|
| w4_both | 49/50 | 50/50 | 0 (0.0%) | −4.1 ± 55.9 | [−310, +170] |
| w2_vlm_protect | 49/50 | **0/50** | 49 (98.0%) | +199.9 ± 77.6 | [0, +313] |

w4_both was too mild (nobody broke). w2_vlm_protect was too aggressive (everybody maxed out at `max_steps` — outcomes became deterministic per suite). Neither gave the ideal "some rollouts break, others don't" heterogeneity for binary classification.

**Initial verdict (later walked back):**

| config | R² on steps_delta (LOTP) | AUC on broke_by_quant (LOTP) |
|---|---|---|
| w4_both | −1.632 ± 1.873 | n/a (0 broken) |
| w2_vlm_protect | +0.324 ± 0.439 | n/a (1 unbroken) |

Suite-only baseline under w2_vlm_protect: R² = +0.665. Attention's best: +0.450. Read naively, the 1-feature suite classifier beats the 1350-feature attention regression, and within-suite R² was strongly negative (−0.92 to −6.89). I reported "hypothesis falsified."

### Exp6 diagnostics — the walked-back verdict

After methodological critique (n=50 vs 1350 features is a regime where bootstrap CIs on R² are ~±0.3 wide; point estimate gaps may not be significant), I reran with proper diagnostics:

**Bootstrap 95% CIs on R² (500 resamples):**

| config | model | point R² | 95% CI |
|---|---|---|---|
| w2_vlm_protect | suite (1 feat) | +0.663 | [+0.128, +0.855] |
| w2_vlm_protect | ridge α=1000 (1350 feat) | +0.450 | [**−0.150, +0.717**] |
| w2_vlm_protect | random forest (d=4) | +0.616 | point only |
| w2_vlm_protect | gradient boost (d=3) | +0.610 | point only |
| w4_both | suite | −0.359 | [−9.04, −0.28] |
| w4_both | ridge α=1000 | −0.478 | [−13.98, −0.56] |

**The CIs overlap heavily.** The "suite beats attention" conclusion is not statistically robust at n=50. RF and GB also match suite-baseline roughly, so nonlinearity doesn't rescue the ridge regression either.

**Spearman + Bonferroni (more sensitive to weak monotonic signal):**
- `w2_vlm_protect`: **169 of 1315** features survive Bonferroni correction at p<0.05. Top feature |ρ|=0.79, Bonferroni-p ≈ 1e-8. Strong nonlinear/monotonic signal that ridge couldn't convert to incremental R² over a 1-feature suite classifier.
- `w4_both`: 0 of 1315 survive Bonferroni. Under mild quantization, there's barely any per-rollout variance to predict.

**Revised interpretation.** At n=50 with a coarse rollout-level target, we can neither confirm nor reject the hypothesis with the precision needed. Point estimates favor "suite beats attention" but CIs don't. The 169 Bonferroni-significant Spearman features say signal exists at the per-feature level but doesn't translate to ridge R² at this scale. The right next move: test at per-frame granularity, where n is ~30× larger and the target is a direct quantity from exp3 (action MSE, not rollout success).

---

## Experiment 7 — Per-frame attention features vs per-frame W4 action MSE (the payoff test, part 2)

**Setup (mentor-motivated correction to exp6's aggregation).** Instead of aggregating attention per rollout and regressing against rollout outcome, regress at the granularity the hypothesis is actually about: a single VLM call.

**Data collection.** For each of the 50 FP16 rollouts:
1. Roll out under FP16, capturing at every VLM call: `(observation_dict, fp16_action_chunk)`.
2. Install W4 weights (both VLM and expert).
3. For each captured observation, call `policy.infer(obs)` to get the W4 action chunk. **This is a single 1-chunk inference per captured obs — not a re-rollout** — so no trajectory divergence confounds the comparison.
4. Per-call target: `MSE(FP16_chunk, W4_chunk)` for the same input observation. This is exactly the quantity exp3 measured at a single-frame level, now across ~1900 frames drawn from real LIBERO rollouts.

**Data joined with attention features.** Per-call attention records already in `exp5_per_call.jsonl` (~200K records, 45 layers × ~4200 VLM calls across 50 rollouts × 5 metrics per head). Joined with exp7's W4-MSE output on `(rollout_idx, call_idx)`, head-averaged per layer-metric → **n = 1879 per-frame samples, 225 features per sample**.

**Target distribution:**

| suite | n | mean MSE | std | median | min | max |
|---|---|---|---|---|---|---|
| all | 1879 | 9.7e-3 | 3.0e-2 | 5.0e-4 | 3.2e-5 | 4.0e-1 |
| Object | 664 | 8.4e-3 | 2.4e-2 | 4.8e-4 | 4.3e-5 | 1.7e-1 |
| Long | 1215 | 1.0e-2 | 3.4e-2 | 5.2e-4 | 3.2e-5 | 4.0e-1 |

Important: Object and Long mean MSEs differ by only ~25%, well within std. **Suite confound is almost absent at per-frame granularity**, unlike exp6 where it dominated.

**LOTP CV R² (25 folds):**

| model | R² (mean ± std) | 95% CI |
|---|---|---|
| suite label (1 feat) | **+0.000 ± 0.001** | — |
| ridge α=100 (225 feat) | +0.006 ± 0.039 | — |
| ridge α=1000 (225 feat) | **+0.032 ± 0.025** | [−0.017, +0.052] |
| random forest (d=6) | −0.286 ± 0.630 | overfits catastrophically |

**Within-suite R²:**

| suite | n | y std | ridge α=1000 | RF |
|---|---|---|---|---|
| Object | 664 | 2.4e-2 | **+0.125 ± 0.074** | −0.042 |
| Long | 1215 | 3.4e-2 | +0.007 ± 0.019 | −0.227 |

**Spearman per feature (Bonferroni-corrected over 90 effective features):**
- **32 of 90 features survive Bonferroni correction at p<0.05.**
- Strongest: `language_model.layers.8.self_attn` with |ρ|=0.165, Bonferroni-p ≈ 5.6e-11.
- All top-15 features are in **`language_model` decoder layers** (layers 2, 3, 8, 9, 11, 13, 15, 17). Not vision_tower — a different locus than exp5's suite-classification signal.

### Interpretation

**What's real.** There IS a statistically robust per-frame correlation between attention features and W4 action MSE. 32 features survive Bonferroni at p<0.05; the strongest Bonferroni-p ≈ 5e-11. The suite-label baseline gives R²=0.000 at per-frame granularity, so this signal is NOT the exp5 suite-identity confound reappearing.

**What's also real.** The effect is **weak**. Best aggregate R² = 0.032 across suites, 0.125 within Object, 0.007 within Long. Linear ridge with strong regularization wins; random forest overfits catastrophically. The within-Long R² is near zero — the signal shows up under Object but not Long.

**Two distinct attention signals in two parts of the VLM.** This was not obvious going in:
- `vision_tower` attention → **suite-level scene fingerprinting** (exp5's AUC=1.0 classification confound).
- `language_model` attention → **per-frame W4 sensitivity** (exp7's weak-but-real regression signal).

These are not the same thing and carrying them through to the paper requires care.

---

## Revised synthesis

1. **Architectural bottleneck (exp2 W2).** Of 42 layer groups, VLM block 0 + SigLIP vision tower dominate W2 sensitivity by ~50–70×. "Protect two, compress forty" — clean static spatial precision map. **Unchanged.**

2. **Temporal bottleneck (exp3).** Expert's 10-step Euler loop is wildly asymmetric: step 9 is ~75× more sensitive than step 0. W4 first 9 steps → ~84% of memory benefit at ~16% of quality cost. Static step-wise schedule; no runtime controller needed. **Unchanged, and still the strongest novel finding vs QuantVLA's uniform-bitwidth baseline.**

3. **Horizon-differential, localized (exp2 + exp3).** Long tasks are ~20% more sensitive to quantization than Object tasks at step 9. Spatially and temporally co-localized with where precision matters. **Unchanged, modest effect.**

4. **Attention-as-controller hypothesis — nuanced result (exp5 + exp6 + exp7).**
   - Exp5: AUC=1.0 classifying Object vs Long from FP16 attention features — but confounded by prompt grammar + scene complexity + rollout length, all mediated through vision_tower attention.
   - Exp6: Regression from per-rollout attention to per-rollout quant-sensitivity is ambiguous at n=50 — point estimates favor suite-baseline but bootstrap CIs overlap; Spearman finds 169 Bonferroni-significant features under w2_vlm_protect.
   - Exp7: At per-frame granularity (n=1879, direct action-MSE target), 32 language_model attention features survive Bonferroni at p<0.05 with R²=0.125 within-Object. **Signal is real but quantitatively too weak to motivate an adaptive-precision controller as a headline contribution.**
   - Distinct signals localize to distinct VLM components: vision_tower → task-identity, language_model → per-frame quant-sensitivity.

### What we do NOT have

- **Rollout-level task success under a deployable quantization schedule.** We have FP16 baselines (18/18 at 100% from exp0) and destructive-aggressive baselines (w2_vlm_protect: 0/50 at exp6). The clean Phase-3 experiment — does the exp3 schedule ("W4 expert steps 0-8, FP16 step 9" + exp2-protected VLM) maintain rollout success? — is not yet run.
- **KV cache direct quantization (KVQuant-style).** Not tested. Exp4 was deferred and remains deferred; exp7 shows that the VLM's per-frame attention signal for sensitivity is weak, which indirectly suggests KV-cache precision effects may also be weak for this model, but this wasn't directly measured.

### What the paper should say

Headline contributions remain exp2 (layer-sensitivity map) and exp3 (step-asymmetric expert sensitivity). These are strong, clean, and novel against QuantVLA. The paper should:
- Lead with "Flow-matching VLAs have step-asymmetric precision requirements 75× concentrated at the final Euler step." Static schedule from exp3 is the deployable artifact.
- Include exp2's "protect VLM layer 0 + vision tower" as the architectural precision map.
- Include exp7 as a supplementary section: per-frame attention in language_model decoder layers weakly predicts W4 sensitivity (Bonferroni-significant 32/90 features; R²=0.125 within Object). Effect size is too small to motivate an adaptive controller as the paper's primary claim, but validates the hypothesis space isn't empty and provides a direction for future work with more aggressive quantization.
- Explicitly distinguish the two attention signals: vision_tower (suite/scene fingerprinting, NOT precision-relevant) vs language_model (weak precision-relevant signal).

### Methodological lessons (for the writeup)

- Per-rollout aggregation can hide per-frame signals. When the hypothesis is frame-level, test at frame-level even if rollout-level metrics are the eventual target.
- At n=50 with 1350 features, bootstrap CIs on R² are ±0.3 wide. Point-estimate gaps of 0.2 are not significant. Spearman with Bonferroni is the right diagnostic for weak monotonic signals.
- Random forest and gradient boosting overfit catastrophically at n/p ratios worse than ~10:1 even with regularization. Linear ridge with strong α is the correct baseline.

---

## Files of record

**Scripts**
- `scripts/utils.py` — shared model loading, quantization (`fake_quantize_module`, `precompute_quantized_weights`, `swap_weights`), JSONL/NPZ I/O, logging. Added `MUJOCO_GL=egl` env default for headless rendering.
- `scripts/rollout.py` — LIBERO closed-loop rollout harness with obs/action callbacks. Monkey-patches `torch.load` for LIBERO's legacy pickles.
- `scripts/setup_libero.sh`, `scripts/run_phase0.sh` — idempotent server setup + Phase 0 orchestrator (env → LIBERO install → smoke tests → full run).
- `scripts/exp0_rollout_reproduce.py` — 18-rollout FP16 reproduction.
- `scripts/exp1_activation_stats.py`, `scripts/exp2_layer_sensitivity.py`, `scripts/exp3_flow_step_sensitivity.py` — original static-obs experiments.
- `scripts/exp5_trajectory_attention.py` — 50-rollout attention capture, aggregation, classifier.
- `scripts/exp5_reanalyze.py` — LOTP CV re-analysis to control for task-identity leakage.
- `scripts/exp6_attention_predicts_quant.py` — re-run 50 rollouts under quantization configs, record outcome deltas, regress.
- `scripts/exp6_diagnostics.py` — bootstrap CIs, Spearman + Bonferroni, RF/GB point estimates.
- `scripts/exp7_per_frame_sensitivity.py` — capture per-call (obs, FP16_chunk) during rollout, replay each obs under W4, compute per-call MSE.
- `scripts/exp7_analyze.py` — per-frame regression: suite-baseline vs ridge vs RF, bootstrap CI, within-suite, Spearman + Bonferroni.

**Results on the server (`/data/subha2/experiments/results/`)**
- `exp0_rollouts.jsonl` — FP16 reproduction baseline.
- `exp3_per_step.jsonl`, `exp3_cumulative.jsonl` — step-wise expert sensitivity.
- `exp5_per_call.jsonl` (200K records), `exp5_rollout_summary.jsonl` (50 rollouts).
- `exp6_per_rollout.jsonl` (100 rows = 50 rollouts × 2 configs), `exp6_tables.md`, `exp6_reanalysis_tables.md`, `exp6_diagnostics.md`.
- `exp7_per_frame__w4_both.jsonl` (1984 rows), `exp7_analysis.md`.

**Local**
- `plots/exp{1,2,3}_*.png` — generated from exp1-3 analysis.

## Timeline

- **2026-04-15 (overnight):** exp1, exp2, broken exp3 (fresh noise bug).
- **2026-04-16:** redesigned exp3 with seeded noise; findings documented above.
- **2026-04-20:** Phase 0 (exp0 rollout infra) + Phase 1 (exp5 trajectory attention) + Phase 2 (exp6 attention-vs-quant + reanalysis) + Phase 2b (exp7 per-frame regression). Research direction evolved from "attention-as-online-controller" to "static temporal + spatial schedule with attention as supplementary signal."
- **2026-04-21:** Three follow-ups (D1/D2/D3) after mentor critique that the exp7 null verdict was overconfident at n=50 with coarse target. D1 hit big (W2 raises within-Object R² to 0.333), D2 localized signal to language_model layer 12 head 2, D3 was falsified (decoupling hurt not helped). Attention story strengthens from "weak supplementary" to "quant-aggressiveness-specific predictor of per-frame sensitivity."

---

## Experiments 8a + 8b — D1/D2/D3 follow-ups to exp7 (2026-04-21)

Three independent follow-ups to exp7 after a methodological critique that "attention dead" was an overconfident read of a coarse per-rollout regression at n=50. Each tested a different alternative explanation for exp7's weak R²=0.032.

### Context

Exp7 measured per-frame W4 action MSE across 1984 frames drawn from 50 LIBERO rollouts, regressed against per-call FP16 attention features head-averaged by layer. Got within-Object R²=0.125, with 32 of 90 features Bonferroni-significant at p<0.05, localized to `language_model` decoder layers. Honest read was "real but weak signal." Three follow-ups:

- **D1 — Stronger quantization.** Exp7 used W4 on VLM+expert (mild). Under W2 with exp2-protection the target variance should spread out and any genuine signal should rise.
- **D2 — Per-head deep dive.** Exp7 head-averaged, collapsing ~8 heads into 1 per layer. If the signal lives in specific heads, averaging diluted it. Compute Spearman at per-(layer, head, metric) granularity.
- **D3 — Decoupled target.** Exp7's W4 MSE target mixed VLM-side error with expert-side error. VLM attention can only plausibly predict VLM-side sensitivity. Isolate via `w4_vlm` (VLM-only) vs `w4_expert` (expert-only) configs.

All three use the exp7 fixture unchanged: same 50 rollouts, same attention-feature extraction from `exp5_per_call.jsonl`, just different per-frame MSE targets and finer feature granularity. New scripts: `exp8_compare_configs.py` (cross-config R² table + per-frame MSE correlations), `exp8_per_head_analysis.py` (per-(layer, head, metric) Spearman with Bonferroni over ~2800 tests).

### Cross-config comparison (exp8a)

| config | n | y_std | suite R² | ridge R² | within-Obj R² | within-Long R² | n_sig / 90 |
|---|---|---|---|---|---|---|---|
| w4_both (exp7 baseline) | 1879 | 3.05e-2 | +0.000 | +0.032 | **+0.125** | +0.007 | 32 |
| w4_vlm (D3 run 1) | 1896 | 3.46e-2 | −0.026 | −0.017 | **−0.207** | −0.087 | 37 |
| w4_expert (D3 run 2) | 1910 | 3.52e-2 | −0.013 | −0.026 | **−0.183** | −0.089 | 35 |
| **w2_vlm_protect (D1)** | 1901 | **2.01e-1** | −0.020 | **+0.111** | **+0.333** | −0.062 | 37 |

**Per-frame MSE correlations (Spearman) between configs:**

| A × B | Spearman ρ | interpretation |
|---|---|---|
| w4_both × w4_vlm | +0.331 | moderate: joint target partly driven by VLM error |
| w4_both × w4_expert | +0.253 | moderate: joint target partly driven by expert error |
| w4_vlm × w4_expert | +0.247 | moderate: some frames hard under both (shared difficulty) |
| w4_both × w2_vlm_protect | **+0.046** | **near-zero: W4 and W2 sensitivity patterns are essentially unrelated** |
| w4_vlm × w2_vlm_protect | +0.043 | near-zero |
| w4_expert × w2_vlm_protect | +0.059 | near-zero |

### D1 verdict — HIT, large effect

Under W2-with-protection, within-Object R² jumps from 0.125 to **0.333** — a 2.7× gain. Target variance increased 6.6× (y_std 3.0e-2 → 2.0e-1), and attention features converted the extra spread into genuine predictive power. Random forest (0.158 overall) now beats ridge (0.111) — nonlinear structure is relevant at this scale. The w4_both signal was real but noise-limited; the W2 signal clears the threshold for meaningful prediction.

**Critical detail:** w2_vlm_protect sensitivity is **uncorrelated** (Spearman ~0.05) with any W4 config's sensitivity. It's not measuring "generic frame difficulty" — it's specifically measuring which frames break under aggressive quantization. The top Spearman features also shifted to earlier layers (1, 3, 4, 5, 12) versus W4's (8, 11, 15, 17). Different signal, not just a louder version of the same signal.

### D2 verdict — HIT, mechanistically localized

Per-head analysis on w2_vlm_protect data: **225 of 2754** (layer, head, metric) triples survive Bonferroni correction at p<0.05. The strongest single feature is `language_model.layers.12.self_attn` head 2, entropy — Spearman ρ = −0.294, Bonferroni-p = 6.9e-36. Same (layer, head) appears in top 3 features across three different metrics (entropy, top5, sparsity), confirming a specific computational unit rather than a metric-wide pattern.

**Top-5 per-head features:**

| rank | layer | head | metric | ρ | mean(top-MSE-decile) | mean(bot-MSE-decile) |
|---|---|---|---|---|---|---|
| 1 | language_model.layers.12 | 2 | entropy | −0.294 | 4.58 | 4.76 |
| 2 | language_model.layers.12 | 2 | top5 | +0.294 | 0.363 | 0.322 |
| 3 | language_model.layers.12 | 2 | sparsity | +0.284 | 0.522 | 0.516 |
| 4 | language_model.layers.9 | 2 | top5 | +0.277 | 0.416 | 0.402 |
| 5 | language_model.layers.9 | 2 | entropy | −0.268 | 4.32 | 4.39 |

**Concrete mechanistic claim:** when `language_model.layers.12.self_attn` head 2 shows lower entropy (more concentrated attention distribution on fewer tokens), the frame is more sensitive to W2 quantization. Higher top-5 mass (attention concentrated on few positions) correlates positively with sensitivity. This is the first mechanistically specific claim linking attention pattern to quant sensitivity in a VLA.

**Pattern breakdown across all 225 Bonferroni-significant features:**
- **By component:** 225/225 (100%) in `language_model`, 0/225 in `vision_tower`. Reconfirms the two-signals-in-two-parts finding from exp7.
- **By metric:** entropy (55, 24%), top5 (51, 23%), top1 (47, 21%), sparsity (42, 19%), sink (30, 13%). Well-distributed across all five metrics.
- **By head:** head 7 (38), head 4 (33), head 2 (30), head 6 (28), head 3 (27), head 5 (26). Distributed across all 8 heads with slight concentration in head 7.
- **By layer:** layer 15 (23), layers 1/3/4 (18 each), layer 5 (17), layer 12 (16). **Spans the full decoder depth (layers 1 to 17)**, not just late layers.

**Per-suite split:** the signal is stronger in Object than Long. E.g., top feature (layer 12 head 2 entropy) has Object-ρ = −0.38 vs Long-ρ = −0.23. Layer 17 head 4 top1: Object −0.42 vs Long −0.14. The Within-Long regression R² stays negative across all configs — Long rollouts have their per-frame sensitivity poorly predicted from FP16 attention even under W2. Plausible reason: Long rollouts visit a wider range of sim states as sub-goals change, so attention-pattern-vs-sensitivity gets weaker.

### D3 verdict — FALSIFIED

Target decoupling HURT, not helped. Within-Object R² for VLM-only target was −0.207 (vs w4_both +0.125 — a 0.33 R² loss). Expert-only target gave similar −0.183. Top features under both decoupled configs were the SAME `language_model` layers (8, 11, 15, 17) that appeared under w4_both, with nearly identical rank order and ρ magnitudes.

**Implication:** the w4_both signal wasn't "attention predicting VLM-side error that got masked by expert noise." Attention features correlate with expert-side and VLM-side MSE approximately equally — so they're tracking a shared frame-level property ("which frames are more sensitive to any W4 perturbation"), not a mechanism-specific signal. The joint (w4_both) target was *easier* to predict than either single-source target because combining two noisy sources smooths per-frame stochasticity, making the shared difficulty signal more visible.

This rules out a particular kind of adaptive controller: you cannot use VLM attention to decide "spend precision on VLM here vs expert here" — the same features flag both kinds of sensitivity. You can only use it as a frame-level gate for overall precision allocation.

### Revised synthesis

Updated the exp5 + exp6 + exp7 section of this document: the "attention as controller" hypothesis is no longer "too weak to matter" but **quant-aggressiveness-specific**. Restated:

1. **Under mild quantization (W4)**, attention features carry a shared per-frame difficulty signal with modest multivariate R² (0.125 within-Object). Individual features are weakly correlated with target (|ρ| ≤ 0.17). Not enough for an adaptive controller.

2. **Under aggressive quantization (W2 with exp2-protection)**, attention predicts per-frame sensitivity with within-Object R² = 0.333. Signal is mechanistically localized to specific `language_model` (layer, head) combinations — notably layer 12 head 2. Nonlinear interactions are real (RF > Ridge). Bonferroni-significant correlations jump to 225/2754 per-head features from 32/90 head-averaged features.

3. **The W4 and W2 sensitivity patterns are uncorrelated** (cross-config Spearman ~0.05). Attention isn't tracking a universal "difficulty" property; it's tracking config-specific sensitivity. Different heads predict different precisions.

4. **D3 is falsified.** Attention features predict VLM-side and expert-side MSE equally (|ρ| ~0.18-0.20 for both), so they're not mechanism-specific. They flag frames that are broadly sensitive.

**Paper implications:**
- Headline stays on exp2 + exp3 — the static temporal schedule is still the deployable, clean contribution against QuantVLA's uniform bitwidth.
- Exp7 + D1/D2/D3 become a stronger supplementary section: "aggressive-quantization sensitivity is per-frame predictable from specific language-model attention heads; signal requires ≥W2 to clear noise." This is a mechanistic finding with a concrete predictor, not just a statistical note.
- Open direction: build an adaptive precision controller that uses attention features to drop selected frames to higher precision when running the W2 schedule. Unclear if the R²=0.33 is high enough for deployable gating — would require rollout-level validation (Phase 3).

### Files of record

**New scripts (2026-04-21):**
- `scripts/exp8_compare_configs.py` — cross-config R² table + per-frame MSE correlation matrix
- `scripts/exp8_per_head_analysis.py` — per-(layer, head, metric) Spearman with Bonferroni + high/low-decile distributional comparison
- `scripts/exp7_analyze.py` — patched so output includes config suffix (`exp7_analysis__{config}.md`)

**New data (server):**
- `results/exp7_per_frame__w4_vlm.jsonl` (1896 per-frame records, VLM-only target)
- `results/exp7_per_frame__w4_expert.jsonl` (1910 records, expert-only target)
- `results/exp7_per_frame__w2_vlm_protect.jsonl` (1901 records, W2 target)
- `results/exp7_analysis__{w4_both,w4_vlm,w4_expert,w2_vlm_protect}.md`
- `results/exp8_compare_configs.md` — unified cross-config comparison
- `results/exp8_per_head__w2_vlm_protect.md` — per-head deep dive on winning config

---

## Experiment B — SIS-gated PTQ validation (partial pilot, 2026-04-21)

**Question.** Does the SQIL paper's State Importance Score (SIS) — perturbation-based action sensitivity, designed for QAT loss weighting — work as a frame-level precision gate for our PTQ setting? Concretely: run W2-with-protection as the cheap default and override with FP16 inference at top-SIS frames. Does that recover the rollout-level success gap?

**Method.** Seven matched conditions per (task, seed, episode_idx): pure W2, pure FP16, SIS-top-20% override, Random-20% (null), Oracle-20% (top-20% by ground-truth ‖a_FP-a_W2‖² on the W2 trajectory), Bottom-SIS-20%, AttnEntropy-top-20% (bottom-20% by `language_model.layers.12.self_attn` head 2 entropy from D2). All conditions share noise schedule per (seed, cycle); only the precision schedule differs. SIS computed with 4×4 Gaussian-blur grid on the base camera, stride k=4, σ=8 — paper-faithful but amortized for inference-time cost. Code in `scripts/sis_utils.py` and `scripts/expB_sis_validation.py`.

**Pilot setup.** 20 seeds on libero_10 task 0, conditions {W2, FP16, SIS-top-20, Random-20, Oracle-20}. Decision rule from the plan: kill if SR(SIS) ≤ SR(Random); proceed otherwise.

### Pilot result — partial (9 of 20 trials before stopping)

| Condition | success | n |
|---|---:|---:|
| W2-with-protection | 0/9 | 9 |
| FP16 | 8/9 | 9 |
| SIS-top-20 | 0/9 | 9 |
| Random-20 | 0/9 | 9 |
| **Oracle-20** | **0/9** | **9** |

Stopped early (9 of 20 planned) because the pattern was already informative for a different reason: **the Oracle baseline failed on every trial too.** That rules out the SIS-vs-Random comparison cleanly — the override-fraction itself (20%) is too sparse to rescue a failed W2-with-protection trajectory on Long task 0. We can't tell whether SIS "works" because no method was given enough overrides to demonstrate signal.

### Per-frame Spearman from the smoke trial (n=104 cycles, libero_10 task 0 ep 0)

| pair | ρ |
|---|---:|
| SIS vs ‖a_FP − a_W2‖² (Oracle target) | **−0.031** |
| SIS vs −entropy at l12h2 (D2 target) | +0.274 |
| ‖a_FP − a_W2‖² vs −entropy at l12h2 | +0.014 |
| Top-5 SIS frames ∩ Top-5 MSE frames | 0/5 |

SIS *does* detect something — it tracks layer-12 head-2 attention concentration as expected. But that something is essentially uncorrelated with per-frame quantization MSE in this flow-matching pi0.5 setup. Possibly a real adaptation gap: SQIL's logic assumes input-perturbation sensitivity ≈ quantization sensitivity, which holds for autoregressive OpenVLA where errors propagate through next-token generation; in flow matching, quantization affects the action expert through the cached prefix, decoupled from the image-perturbation pathway.

### Diagnosis

Two independent issues conflated in this pilot:

1. **Override sparsity issue (method-agnostic).** Even Oracle at 20% can't rescue Long task 0. The W2 trajectory diverges fast enough that one-in-five FP16 chunks can't put it back on path. The hypothesis matrix needs Oracle ≥ 50% to give SIS any chance of demonstrating value below it.

2. **SIS-vs-MSE alignment issue (method-specific).** Even before the rollout-level test, the per-frame Spearman is near zero. SIS picks frames that are sensitive to image perturbation; MSE-top-20 picks frames where W2's action diverges from FP16's. These appear to be different sets in pi0.5. Whether higher override frac changes this is empirical — possible that SIS would still beat random at 50% even with weak per-frame correlation, because rollout-level rescue depends on chunk-level dynamics not pure per-frame MSE.

### Next step — frac=0.5 retry

Re-running the pilot at frac=0.5 (50% override). This:
- gives Oracle a real chance to rescue (4× the override budget)
- preserves a meaningful precision-savings story (50% W2)
- keeps the SIS-vs-Random kill switch test intact
- if Oracle ≈ 80% and SIS only ≈ 30%, that's still a real signal about SIS quality

If frac=0.5 still leaves Oracle < 50%, the next retry will switch to Object suite (easier task; W2 should have non-zero baseline success there per exp6).

### Files of record

**New scripts (2026-04-21):**
- `scripts/sis_utils.py` — perturbation-based SIS, FP16↔W2 PrecisionController, l12h2 attention-entropy hook
- `scripts/expB_sis_validation.py` — diagnostic + override rollouts, per-rollout 80th-percentile threshold, bootstrap-CI summary

**Partial pilot data (server, frac=0.20):**
- `results/expB_diagnostic.jsonl` (936 per-cycle records across 9 trials)
- `results/expB_rollouts.jsonl` (45 rollouts: 9 trials × 5 conditions)
- `results/expB_pilot_stdout.log`

### Pilot retry — frac=0.5 (20 trials, libero_10 task 0, 2026-04-22)

| Condition | success | rate | 95% bootstrap CI |
|---|---:|---:|---|
| W2-with-protection | 0/20 | 0.000 | [0.00, 0.00] |
| FP16 | 19/20 | 0.950 | [0.85, 1.00] |
| **Oracle-20 (top 50% by ‖a_FP-a_W2‖²)** | **5/20** | **0.250** | [0.05, 0.45] |
| **SIS-top-20** | **5/20** | **0.250** | [0.10, 0.45] |
| Random-20 | 2/20 | 0.100 | [0.00, 0.25] |

**SIS exactly matches Oracle's success rate (5/20 each), and both beat Random by +15 percentage points (2.5× rescue rate).** Per the plan's hypothesis matrix this is the **STRONG verdict**: SIS recovers the full per-frame-MSE-oracle gap over random selection.

Bootstrap CIs overlap (n=20 is small), so this isn't yet statistically significant — but the rank ordering SIS = Oracle ≫ Random is clean across the matched-seed pairs. FP16 baseline at 19/20 = 95% matches QuantVLA's published Long success rate (93.5%), confirming the rollout infrastructure is calibrated correctly.

**Per-trial pattern.** SIS succeeded on trials {0, 7, 12, 17, 18}; Oracle on {3, 7, 10, 17, 18}. Overlap = {7, 17, 18} (3 of 5). Same hit rate, partially different sets. Three possible interpretations:

1. **Coincidence at n=20** — the full experiment will resolve this.
2. **Per-frame-MSE Oracle isn't a true upper bound** for rollout success. Per-frame MSE is computed at the diagnostic's W2 trajectory states; the actual override rollouts diverge from those states. Combinatorial mask search would give a true upper bound but is intractable. Our "Oracle" is the best fixed-mask achievable from observable W2-trajectory metrics.
3. **SIS captures complementary signal.** Image-perturbation sensitivity may flag frames whose action-distribution shape (not just per-frame MSE magnitude) is critical for trajectory rescue — chunks where the FP16 action vector points in a substantively different *direction* than W2, even if the per-frame MSE is mid-range.

(2) and (3) both predict that adding more conditions (the Bottom-SIS-20 symmetry control and AttnEntropy-top-20 cheap proxy from the original plan) will be informative on the full experiment.

### Notes for the full experiment

- Override frac should be **0.5**, not the original plan's 0.2 (frac=0.2 produced 0% even for Oracle on Long; budget-bound, not method-bound).
- Run all 7 conditions including Bottom-SIS-20 (symmetry check: ρ-flipped SIS should help less than top-SIS, otherwise SIS is a frame-difficulty detector not a quantization-sensitivity detector) and AttnEntropy-top-20 (D2 cheap proxy).
- 100 trials (50 Long + 50 Object). Object likely has more rescue headroom — exp6 saw W2-with-protection didn't catastrophically break Object, so Oracle should clear 50% there.
- Cost at current per-trial speed (~4.4 min): 100 × 4.4 × 7/5 ≈ 10 hours. Will need batched-SIS optimization (16 perturb passes → 1 batched call, ~10× speedup on diagnostic) before launching.

**Updated pilot data (server, frac=0.5):**
- `results/expB_diagnostic.jsonl` (2080 per-cycle records, 20 trials × ~104 cycles)
- `results/expB_rollouts.jsonl` (100 rollouts: 20 trials × 5 conditions)
- `results/expB_pilot_frac50_stdout.log`
- `results/expB_summary.md`
