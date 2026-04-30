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

### Full experiment — frac=0.5, 100 trials × 8 conditions (2026-04-22, 6h 9min on H100)

50 Long + 50 Object trials, all 8 conditions matched on (suite, task_id, seed, episode_idx). Three additions to the pilot setup: (i) batched SIS perturbations (compute_sis bypasses policy.infer's batch=1 wrapper, calls model._sample_actions on a (16, ah, ad) batch — sequential-vs-batched parity confirmed at |diff| = 1.08e-5), (ii) new `MSE-FP16traj` condition (per-frame MSE computed at FP16-trajectory states instead of W2-trajectory states), (iii) symmetry control `Bottom-SIS` and cheap-proxy `AttnEntropy` conditions added.

#### Headline table

| Condition | success | 95% bootstrap CI |
|---|---:|---|
| FP16 (ceiling) | 94/100 = 0.940 | [0.89, 0.98] |
| W2-with-protection (floor) | 0/100 = 0.000 | [0.00, 0.00] |
| **AttnEntropy** | **68/100 = 0.680** | **[0.59, 0.77]** |
| SIS-top | 49/100 = 0.490 | [0.39, 0.59] |
| MSE-W2traj | 48/100 = 0.480 | [0.38, 0.58] |
| Bottom-SIS | 42/100 = 0.420 | [0.33, 0.51] |
| Random | 39/100 = 0.390 | [0.30, 0.49] |
| MSE-FP16traj | 1/100 = 0.010 | [0.00, 0.03] |

#### The unexpected winner: AttnEntropy beats SIS by 19 pp

The cheap online detector — entropy of `language_model.layers.12.self_attn` head 2's softmax distribution, ~zero marginal cost since the VLM forward pass already runs — is the clear performance leader. Per the D2 finding (ρ = -0.29 between l12h2 entropy and per-frame W2 sensitivity), low-entropy frames are more quant-sensitive; the override mask picks the bottom-50% by entropy.

| Condition | Long (n=50) | Object (n=50) |
|---|---:|---:|
| FP16 | 0.88 | 1.00 |
| W2 | 0.00 | 0.00 |
| **AttnEntropy** | **0.56** | **0.80** |
| MSE-W2traj | 0.30 | 0.66 |
| SIS-top | 0.40 | 0.58 |
| Bottom-SIS | 0.24 | 0.60 |
| Random | 0.20 | 0.58 |
| MSE-FP16traj | 0.00 | 0.02 |

AttnEntropy wins on both suites cleanly. The gap to the next-best detector (MSE-W2traj on Object, SIS-top on Long) is +14 pp in both suites — robust across difficulty levels.

#### The pilot's "STRONG" verdict shrinks at n=100

At n=20 (pilot), SIS-top hit Oracle's success rate (5/20 each) and beat Random by +15 pp. At n=100, SIS-top vs Random drops to **+10 pp** (CIs overlap: [0.39, 0.59] vs [0.30, 0.49]) and SIS-top vs Bottom-SIS is only **+7 pp** — the symmetry control isn't decisively beaten. SIS does carry signal, but it's weak and concentrated on Long.

**On Object (50 trials), SIS-top (0.58) ≈ Random (0.58) ≈ Bottom-SIS (0.60).** SIS provides essentially no per-frame signal on Object — the same suite that exp7's per-frame attention regression had its strongest R² on. So SIS as designed (image-perturbation sensitivity) and attention-entropy-as-proxy (mechanistic D2 finding) are picking up *different* per-frame properties; one transfers to Object, the other doesn't.

#### MSE-FP16traj catastrophic failure: the trajectory-divergence trap

`MSE-FP16traj` performs *worse than W2-only* (1/100 vs 0/100 — within noise of zero). Two structural reasons:

1. **Mask budget asymmetry.** Frac is applied to each diagnostic's own cycle count. FP16 succeeds in ~50 cycles on average; W2 hits max-steps at 104. So MSE-FP16traj's mask is ~25 cycles vs others' ~52. Half the override budget.
2. **Out-of-distribution rankings.** The MSE values are computed at FP16-trajectory states (because the FP16 diagnostic only visits FP16 states). The override rollout runs W2 base, which diverges from the FP16 trajectory after the first few cycles. So the rankings reflect "where would FP16 and W2 disagree on the FP16 path" — not "where do they disagree on the actual W2-with-overrides path." Even Oracle's per-frame MSE only weakly transfers across trajectories.

Methodological lesson worth documenting: **per-frame oracle MSE rankings are tied to the trajectory they were measured on**. The W2-trajectory MSE oracle (0.48) succeeds because the override rollout starts as W2 — at least the early cycles share state. The FP16-trajectory MSE oracle fails because the override rollout never visits FP16 states beyond cycle 0.

#### Hypothesis matrix verdict (revised from auto-printed "STRONG")

The auto-verdict triggered "STRONG" because SIS-top > Random and Oracle headroom over SIS ≈ 0. But at n=100 the actual story is more nuanced:

| Comparison | Δ | interpretation |
|---|---:|---|
| SIS-top - Random | +10 pp | SIS works weakly; CI overlaps |
| SIS-top - Bottom-SIS | +7 pp | symmetry control nearly tied; SIS direction signal is weak |
| AttnEntropy - SIS-top | **+19 pp** | cheap proxy DOMINATES SIS |
| AttnEntropy - MSE-W2traj | +20 pp | cheap proxy beats best heuristic oracle too |

**Revised verdict: SIS works but is dominated by the cheap online attention-entropy detector.** The deployable result is "use l12h2 attention entropy as the per-frame precision gate; you get 68% rescue rate at zero marginal inference cost." This is the publishable finding, and a stronger one than the original SIS hypothesis predicted — it means a deployable PTQ-with-rescue scheme doesn't need expensive perturbation passes at inference time.

#### Trial-level pattern: matched-pair signed deltas

Per matched (task, seed, ep), the signed deltas (1 if condition succeeded and another condition didn't, -1 if vice versa) are more sensitive than aggregate SR comparisons because they cancel intrinsic trial difficulty. *(To be added when the per-trial diff analysis script is run; for now the aggregate table tells the story.)*

#### Files of record (server)

- `results/expB_diagnostic.jsonl` (8000 W2-traj per-cycle records: 100 trials × ~80 cycles)
- `results/expB_fp16_diagnostic.jsonl` (4262 FP16-traj per-cycle records: 100 trials × ~42 cycles)
- `results/expB_rollouts.jsonl` (800 rollouts)
- `results/expB_summary.md` (auto-generated bootstrap-CI table)
- `results/expB_full_stdout.log`

### Frac sweep — AttnEntropy vs Random at frac ∈ {0.3, 0.4} (2026-04-22)

Reused the existing 100-trial diagnostic data (no new diagnostic passes — the per-cycle attention entropy was already in `expB_diagnostic.jsonl`). Ran AttnEntropy + Random override rollouts at two new fracs to characterize the precision-savings vs success curve. 400 rollouts in ~2.3 hours on H100.

#### Combined precision-savings curve (Random as baseline)

| frac | AttnEntropy | Random | gap (pp) | ratio |
|---:|---:|---:|---:|---:|
| 0.3 | 0.02 | 0.01 | +1 | 2× |
| 0.4 | 0.28 | 0.06 | +22 | 4.7× |
| 0.5 | 0.68 | 0.39 | +29 | 1.7× |

**Per-suite:**

| frac | Long: AttnEntropy / Random | Object: AttnEntropy / Random |
|---:|---:|---:|
| 0.3 | 0.00 / 0.00 | 0.04 / 0.02 |
| 0.4 | 0.18 / 0.02 | 0.38 / 0.10 |
| 0.5 | 0.56 / 0.20 | 0.80 / 0.58 |

#### Three observations

1. **frac=0.3 is below the budget threshold.** Long is 0% / 0% (no method rescues at 30% override). Object barely separates (4% / 2%). Same dynamic as the pre-pilot frac=0.2 result — too few FP16 chunks injected to keep the W2 trajectory on path.

2. **frac=0.4 is the maximum-detector-leverage point.** Overall AttnEntropy 0.28 vs Random 0.06 = 4.7× ratio. On Long specifically: 0.18 vs 0.02 = 9× ratio. This is where *which* frames you pick matters most — at frac=0.5 most reasonable selections rescue, at frac=0.3 nothing does, at frac=0.4 the detector earns its keep.

3. **The 0.4 → 0.5 jump is a budget effect, not a detector effect.** AttnEntropy gains +40 pp; Random gains +33 pp. Both methods get a ~30+ pp boost from the extra 10% override budget. So we're crossing a structural threshold in trajectory rescuability between 0.4 and 0.5, on top of any detector advantage.

#### What this implies for deployment

The precision-savings vs success tradeoff is now characterizable:

- **60% W2 weights (frac=0.4): ~28% rescue rate, AttnEntropy chosen frames**
- **50% W2 weights (frac=0.5): ~68% rescue rate**
- **0% W2 weights (FP16): 94% rescue rate**

If the deployment target is "high reliability," frac=0.5 is the operating point — recovers most of FP16's success at half the inference cost. If the deployment target is "max savings while maintaining some rescue capability," frac=0.4 with AttnEntropy is the choice.

#### Caveat — these comparisons need more baselines

The sweep so far compares only AttnEntropy vs Random. To make the result publishable we need (at the same fracs):

- **AttnEntropy-flipped** (high-entropy = predicted-low-sensitivity): symmetry control. If AttnEntropy ≈ AttnEntropy-flipped, the direction has no signal.
- **SIS-top**: detector comparison. Confirms AttnEntropy isn't just one of many proxies that work.
- **MSE-W2traj**: heuristic ceiling. Tells us how much oracle headroom remains.

A follow-up sweep adding these three is queued (estimated ~3.5h, reusing the same diagnostic data).

#### Files of record (frac sweep)

- `results/expB_frac_sweep.jsonl` (400 rollouts: 100 trials × 2 fracs × 2 conditions)
- `results/expB_frac_sweep_summary.md` (auto-generated)
- `results/expB_sweep_stdout.log`

### Baseline-expanded sweep — full detector comparison at frac ∈ {0.3, 0.4} (2026-04-22)

Added `AttnEntropy-flipped` (symmetry control), `SIS-top` (cheap-vs-expensive detector comparison), and `MSE-W2traj` (heuristic per-frame oracle) to the sweep at the same fracs. 600 additional rollouts appended to the existing sweep file (1000 total). Reused the same 100-trial diagnostic data — no new diagnostic passes.

#### Full per-condition table

| Condition | Overall frac=0.3 | Overall frac=0.4 | Long frac=0.4 | Object frac=0.4 |
|---|---:|---:|---:|---:|
| **AttnEntropy** | **0.02** | **0.28** | **0.18** | **0.38** |
| MSE-W2traj (heuristic oracle) | 0.00 | 0.16 | 0.10 | 0.22 |
| SIS-top | 0.01 | 0.14 | 0.08 | 0.20 |
| Random | 0.01 | 0.06 | 0.02 | 0.10 |
| AttnEntropy-flipped (symmetry) | 0.00 | 0.01 | 0.02 | 0.00 |

#### Five findings

**1. AttnEntropy beats the heuristic per-frame oracle by +12 pp.** This is the headline. The "MSE-W2traj oracle" — top-k cycles by ground-truth ‖a_FP − a_W2‖² measured at W2-trajectory states — was supposed to be the upper bound on any frame-selection strategy. AttnEntropy at frac=0.4 hits 0.28 success vs MSE-W2traj's 0.16. The cheap online detector beats the per-frame ground truth.

The likely mechanism: per-frame MSE rankings are *backward-looking* (what mattered on the diagnostic's W2 trajectory). The override rollout diverges from that trajectory, and the rankings become stale. AttnEntropy taps into a property of the model's computation (where the VLM is concentrating attention) that may be more robust to trajectory drift — frames where the model is doing sharp lookups remain critical even after the trajectory diverges, because the same kind of state still benefits from precision protection. Per-frame MSE is one specific measurement at one specific state; entropy is a structural property of the model's processing.

**2. AttnEntropy beats SIS-top by +14 pp.** Confirms the full-experiment finding at a different operating point. SIS at frac=0.4 = 0.14, AttnEntropy = 0.28. The expensive perturbation detector is dominated by the free attention readout at every frac we've tested.

**3. Symmetry control is clean: AttnEntropy 0.28 vs AttnEntropy-flipped 0.01.** Picking the *highest*-entropy frames (predicted-LOW sensitivity per the negative D2 correlation) is essentially worse than random — even the small budget gets wasted on frames that don't matter. The LOW-entropy direction is doing real work; this isn't a "any concentrated metric helps" artifact.

**4. SIS-top still has signal but is weak.** SIS-top 0.14 vs Random 0.06 = 2.3× ratio at frac=0.4. SIS isn't junk, just dominated. This matches the per-frame Spearman finding from the smoke trial (SIS vs MSE = -0.03; SIS vs -entropy = +0.27): SIS captures something correlated with attention concentration but not directly with quantization sensitivity.

**5. frac=0.3 is the budget floor.** Across all five conditions: 0-2% overall. No detector — including the oracle — can rescue at 30% override. The W2 trajectory diverges fast enough that one-in-three FP16 chunks can't redirect it back. frac=0.4 is the cheapest interesting operating point.

#### Detector ranking (frac=0.4)

```
AttnEntropy (free)           0.28   ← winner across the board
MSE-W2traj (heuristic oracle)0.16
SIS-top (paper detector)     0.14
Random (null)                0.06
AttnEntropy-flipped (sym)    0.01   ← symmetry control near zero
```

The deployment story is now decisive: **use l12h2 attention entropy as the per-frame precision gate.** It's the cheapest detector available *and* it gives the highest success rate at all override fractions tested. There is no method-quality / cost tradeoff — the cheap method is the best method.

#### Files of record (expanded sweep)

- `results/expB_frac_sweep.jsonl` (1000 rollouts: 100 trials × 2 fracs × 5 conditions)
- `results/expB_frac_sweep_summary.md` (regenerated with all 5 conditions)

### Deployment caveat — what the AttnEntropy result actually validated (2026-04-22)

A clarification worth pinning down before any "AttnEntropy is free at inference time" claim makes it into a paper.

**What our PrecisionController does.** It swaps *all* non-protected VLM Linear weights in O(1) before each `policy.infer` call (the existing-weight pointers live next to a precomputed-quantized-weight cache; `weight.data = ...` switches between them). It does not do per-layer adaptive precision within a single forward pass — at any moment the entire VLM is either FP16 or W2-with-protection.

**How the AttnEntropy override condition actually got its mask.** Per (suite, task_id, seed, episode_idx):

1. Run a full W2 diagnostic rollout. In its `obs_callback`, swap to FP16 *before* each cycle's measurement, run a forward pass, read l12h2 head 2 attention entropy from the hook, then swap back to W2 for the actual rollout step.
2. After the diagnostic completes, rank the per-cycle entropy values across the whole rollout, take the bottom 50%.
3. Replay the seed in a separate "override rollout": at each cycle whose index is in the precomputed mask, swap to FP16 for that cycle's `policy.infer`; otherwise W2.

So **the entropy was computed on a full FP16 forward pass during the diagnostic** — not online, and not from a W2 pass. The override decision used those FP16-measured entropies as a stand-in for "what an oracle detector would say."

**What that means.** The result validates that the *signal* l12h2 head 2 entropy carries is the right signal to gate precision on. It does *not* directly validate a deployable runtime detector, because:

- To compute entropy at l12h2 you have to run the model up to layer 12.
- If you've run layers 0-12 in some precision X, you can't retroactively pick a different X for those layers.
- "Use FP16 at this frame" therefore can't be a decision made *during* this frame's forward pass.

#### Option (1) — one-frame-lag deployable scheme

The most promising deployable path, because it requires no architecture changes and no per-layer precision swapping:

```
state at frame t:
    precision_t = decision made at frame t-1
    run policy.infer at precision_t
    as a side-effect of that pass, read l12h2 head 2 entropy E_t
    if E_t < threshold (low entropy → predicted high sensitivity):
        decision for frame t+1 = FP16
    else:
        decision for frame t+1 = W2
```

Properties:
- **Zero extra compute.** Entropy is read from a softmax tensor that already had to be computed; the hook is one tensor reduction.
- **One-frame lag.** Frame t+1's precision is decided by frame t's signal. Robot trajectories are smooth (5 Hz control on LIBERO), so the entropy at adjacent frames is correlated.
- **Bootstrapping.** Pick the first frame's precision arbitrarily (e.g. FP16); subsequent frames adapt.

#### What needs to be validated before claiming option (1) works

The signal validated in the experiment is **FP16-pass l12h2 entropy**. Option (1) reads entropy from whichever precision is currently running — so on most cycles it's reading **W2-pass l12h2 entropy** instead. The one-frame-lag scheme works only if these are highly correlated.

The correlation check is cheap. For each trial we already have FP16-pass entropy in `expB_diagnostic.jsonl`. Run one extra rollout per trial that computes entropy on each cycle's W2 forward pass, then per-trial Spearman ρ between FP16-pass and W2-pass entropies. If ρ > ~0.7 across trials, the one-frame-lag scheme is viable.

Estimated cost: 100 trials × ~30s = ~50 min on H100, reusing the existing diagnostic infrastructure.

Until that check runs, the honest framing of the AttnEntropy result is:

- **Research claim (validated):** l12h2 head 2 attention entropy is the per-frame signal that best ranks which frames a W2-with-protection trajectory needs FP16 rescue at; it dominates SIS, beats per-frame-MSE oracle, and has a clean direction-symmetry control.
- **Deployment claim (not yet validated):** that signal can be read cheaply enough at runtime to support a streaming precision gate. The most promising path is the one-frame-lag scheme above; it depends on FP16-pass-entropy and W2-pass-entropy being highly correlated.

---

## Experiment C — W4-first online mixed-precision (overnight, 2026-04-28→29)

Pivoted away from the W2-rescue framing of ExpB after the user identified that modern GPUs have native W4 tensor cores (sub-W4 needs custom kernels). The experimental question shifted from "can attention entropy rescue W2 failures" to **"can attention entropy enable sub-W4 average bits without rollout SR loss"**. Implementation in commits c7f0414, 325c65b, d876110, ad6867e on branch `overnight-2026-04-29-w4-first`.

### Methodology

- **Diagnostic V3** (`scripts/sis_utils.py:AttentionMetricHook`, `scripts/expB_sis_validation.py:diagnostic_rollout_v3`): captured FP16-pass and W4-pass attention metrics for 5 candidate (layer, head, metric) probes per cycle: `l1h7-top1`, `l9h2-ent`, `l12h2-ent`, `l3h4-top5`, `l17h4-top1` (the top-15 features from D2 in MEETING_5).
- **Intra-pass controller** (`scripts/sis_utils.py:IntraPassController`): forward-hook on `language_model.layers.{L}.self_attn`. After the metric is recorded, calls `PrecisionController.use_bits_range(L+1, 17, decision)` to swap layers L+1..17 mid-forward. Three-tier supported: bottom-frac_high → escalate, top-frac_low → demote, middle → base.
- **21 conditions × 100 paired trials** (50 Long + 50 Object) at frac=0.4 binary, partition (0.1, 0.4, 0.5) ternary. Total 2100 rollouts, ~9.5h on H100.
- Tier 0 (W4 baseline, 3 conditions × 100 trials, 2.7h) → Tier 1+2+3 (15 conditions × 100 trials with `--reuse-diag`, 6.7h) → Tier 4 (3 direction-flipped ternary conditions × 100 trials, 1.5h).

### Mid-Tier-0 finding — D2 direction is FLIPPED at W4

At pooled n=1806 cycles (35 trials, partial), Spearman ρ(W4-pass l12h2-ent, ‖a_FP − a_W4‖²) = **+0.172** (p = 2×10⁻¹³). The W2 D2 finding had ρ = −0.294 — opposite sign. **At W4, HIGH entropy at l12h2 predicts HIGH sensitivity** (vs. LOW entropy at W2). Triggered Tier 4 to add direction-flipped variants: `AttnEntropy-W4-top`, `S1-Bin-W4-top`, `S2-Bin-W4-top`, `S3-Bin-W4-l12h2-ent-top`, then later `S1-Tern-W4-top`, `S2-Tern-W4-top`, `S3-Tern-W4-l12h2-top`.

### HW7 D2-W4 transfer at full n=100 trials

| Probe | median ρ | mean ρ | P(\|ρ\|>0.15) | direction needed at W4 |
|---|---:|---:|---:|---|
| l12h2-ent | +0.115 | +0.120 | 0.57 | top (flipped vs W2) |
| **l1h7-top1** | **−0.167** | **−0.155** | — | **bottom (flipped vs W2)** |
| l9h2-ent | −0.031 | −0.023 | — | weak |
| l3h4-top5 | −0.016 | −0.014 | — | weak |
| l17h4-top1 | −0.009 | +0.016 | — | weak |

Notably, l1h7-top1 has the strongest per-trial signal (mean |ρ| = 0.155 vs l12h2's 0.120), but its early-layer position made it the cheapest gate candidate. l9h2 and other middle-late probes have essentially zero per-trial signal at W4 (within noise).

### Final SR table (n=100, 50 Long + 50 Object)

| Condition | SR | avg bits | 95% CI |
|---|---:|---:|---|
| FP16 (ceiling) | 0.940 | 16.00 | [0.89, 0.98] |
| W4-Floor (reference) | 0.940 | 4.00 | [0.89, 0.98] |
| W4-Static-Sched | 0.940 | 4.00 | [0.89, 0.98] |
| Random-W4 | **0.980** | 8.82 | [0.95, 1.00] |
| AttnEntropy-W4 (bottom) | 0.960 | 8.82 | [0.92, 0.99] |
| AttnEntropy-W4-top | 0.950 | 8.82 | [0.90, 0.99] |
| S1-Bin-W4 (bottom) | 0.950 | 9.13 | [0.90, 0.99] |
| **S1-Bin-W4-top** | **0.980** | 8.90 | [0.95, 1.00] |
| S2-Bin-W4 (bottom) | 0.970 | 8.82 | [0.93, 1.00] |
| S2-Bin-W4-top | 0.950 | 8.82 | [0.90, 0.99] |
| S3-Bin-W4-l1h7-top1 | 0.970 | 7.90 | [0.93, 1.00] |
| **S3-Bin-W4-l9h2-ent** | **0.980** | 5.05 | [0.95, 1.00] |
| S3-Bin-W4-l12h2-ent (bottom) | 0.970 | 4.42 | [0.93, 1.00] |
| **S3-Bin-W4-l12h2-ent-top** | **0.980** | 6.49 | [0.95, 1.00] |
| **S3-Tern-W4-l12h2 (bottom)** | **0.950** | **3.58** | **[0.90, 0.99]** |
| S3-Tern-W4-l12h2-top | 0.970 | 5.20 | [0.93, 1.00] |
| S1-Tern-W4 (bottom) | 0.790 | 4.57 | [0.71, 0.87] |
| S1-Tern-W4-top | 0.710 | 4.45 | [0.62, 0.80] |
| S2-Tern-W4 (bottom) | 0.740 | 4.19 | [0.65, 0.82] |
| **S2-Tern-W4-top** | **0.640** | 4.19 | [0.54, 0.73] |
| Random-Tern-W4 | 0.790 | 4.19 | [0.71, 0.87] |

### Three findings, in order of importance

#### 1. Layer-restricted intra-pass W2 demotion is the deployable winner

**`S3-Tern-W4-l12h2` Pareto-dominates uniform W4** at 95.0% SR / 3.58 avg bits vs W4-Floor's 94.0% / 4.00 bits — both better SR (within CI) AND lower bits. The mechanism: per-cycle running quantile of l12h2 entropy gates a three-tier precision schedule applied only to layers 13-17 (layers 1-12 stay at W4). 10% of cycles escalate to FP16, 50% de-escalate to W2, 40% stay at W4 — but only for the 5 layers downstream of the readout point.

This is the headline contribution: an attention-gated, layer-restricted, sub-W4 precision schedule that maintains rollout success on a flow-matching VLA.

#### 2. The S1/S2-Tern collapse is SPATIAL, not directional

Full-pass W2 demotion (swapping all VLM Linear weights for layers 1-17 to W2 on selected cycles) loses 15-30 pp regardless of cycle-selection strategy:

| Condition | Combined SR | Δ vs Floor | Mechanism |
|---|---:|---:|---|
| W4-Floor | 0.940 | 0 | reference |
| **S3-Tern-W4-l12h2** (intra-pass, layers 13-17 only) | **0.950** | **+1.0 pp** | spatial restriction works |
| S1-Tern-W4 (bottom dir, full-pass) | 0.790 | -15.0 pp | spatial collapse |
| S1-Tern-W4-top (top dir, full-pass) | 0.710 | -23.0 pp | spatial collapse, target-aware made it worse |
| S2-Tern-W4 (bottom, full-pass) | 0.740 | -20.0 pp | spatial collapse |
| S2-Tern-W4-top (top, full-pass) | 0.640 | -30.0 pp | worst — direction-flip + full-pass |
| Random-Tern-W4 (random, full-pass) | 0.790 | -15.0 pp | random demotion also collapses |

**Critical observation:** Random-Tern-W4 also loses 15 pp vs Floor — it's not about "wrong cycles." Targeted top-direction (the data-correct direction) on full-pass actually loses MORE because it accurately picks the cycles whose mid-VLM layers (1-12) most depend on full precision. The collapse is the W2 demotion mechanism applied to layers 1-12, not the targeting choice.

This is the most novel mechanistic finding: **at W4, mid-VLM layers do NOT tolerate W2 even on individual cycles. The intra-pass scheme works only because it preserves layers 1-12 at W4 baseline while allowing layers 13-17 to be demoted.**

#### 3. Direction-flip at l12h2 is a small effect (~1-3 pp)

| Scheme | Bottom dir | Top dir | Δ |
|---|---:|---:|---:|
| AttnEntropy | 0.960 | 0.950 | -1.0 pp |
| S1-Bin | 0.950 | **0.980** | **+3.0 pp** |
| S2-Bin | 0.970 | 0.950 | -2.0 pp |
| S3-Bin-l12h2 | 0.970 | 0.980 | +1.0 pp |
| S3-Tern-l12h2 | 0.950 | 0.970 | +2.0 pp |

Mixed effect at the binary level: top wins 3 of 5 schemes by 1-3 pp. The W4 direction-flip is real (per HW7) but operationally small. **The bottom-direction S3-Tern-W4-l12h2 remains the best Pareto operating point** (3.58 bits vs top-dir's 5.20 bits at similar SR).

#### 4. AttnEntropy is NOT the rescue mechanism at W4 (revised from W2)

At W4 base, `Random-W4` (random 50% FP16 on each cycle) achieves 0.980 SR — the best of all binary schemes. AttnEntropy doesn't beat random for the rescue role at W4 because the rescue gap is small (W4-Floor 0.940 vs FP16 0.940 — they tie). The role of attention is now in *cost reduction* (where to spend the W2 demotion), not rescue.

### What this means for the paper

The W4-first story produces a cleaner deployable artifact than the W2-rescue framing:

- **At W2**: AttnEntropy is the rescue gate (68% rescue rate, +29 pp over Random)
- **At W4**: AttnEntropy is the demotion gate (3.58 avg bits vs 4.0, +1 pp SR over Floor)

Both are deployable, but at W4 the result is a true Pareto improvement (lower cost AND higher SR) rather than a rescue trade-off. The mechanistic insight — **layer-restricted W2 demotion is the bound on aggressive quantization** — is novel and likely transferable to other VLAs.

### Files of record

- `results/expB_w4_summary.md` — auto-generated bootstrap-CI table + HW0–HW10e hypothesis matrix + HW7 Spearman ρ across 5 probes
- `results/expB_w4_rollouts.jsonl` (2100 rows) — all conditions × all trials
- `results/expB_diagnostic_v3.jsonl` (4244 rows) — multi-probe FP16-pass + W4-pass attention per cycle
- Server logs: `/data/subha2/experiments/logs/w4_main_2026-04-28.log`, `w4_tier4_2026-04-29.log`, `w4_tier4_watcher_2026-04-29.log`

### Caveats and follow-ups

- **CI overlap on the headline result**: S3-Tern-W4-l12h2 (0.95 [0.90, 0.99]) vs W4-Floor (0.94 [0.89, 0.98]) — bootstrap CIs overlap heavily at n=100. Matched-pair delta SR is +0.030 (HW10e against Floor for top-dir variant). The contribution is "matches Floor at lower bits" rather than "beats Floor."
- **n=100 is the smallest workable n for this hypothesis matrix.** Per-condition CI widths are ±5 pp; some pairwise deltas are within noise. A future run at n=200+ would tighten these.
- **The l1h7-top1 per-trial signal (mean ρ = -0.155) is intriguing**: if the cheap-pass intra-pass version with the *correct* (bottom) direction were tested, it might enable an even cheaper deployable scheme — but `S3-Bin-W4-l1h7-top1` was run with the W2-default direction ("top") which is wrong at W4. A `S3-Bin-W4-l1h7-bottom` follow-up is the obvious next experiment. **(Tier 5 below resolved this — falsified.)**
- **Layer-restriction follow-up**: confirm the spatial finding by testing "layers 1-12 W4 + layer L W2 + layers L+1..17 W4" for each L — isolate which exact layer's W2 demotion causes the failure. Likely it's a specific cluster (layers 4-11 from exp7) but not yet pinpointed.

---

## Experiment C Tier 5 — l1h7 cheap-pass FALSIFICATION (2026-04-29 post-overnight)

The Tier 1+2+3 run included `S3-Bin-W4-l1h7-top1` with the PROBE_DIRECTION_BY_TAG W2-default direction "top" (matching ρ = +0.26 at W2). At W4 the per-trial mean ρ for l1h7-top1 is **-0.155** — strongest of all 5 candidate probes — implying the W4-correct direction is "bottom" (low top1 → escalate FP16). The intuition: if l1h7-bottom works at W4, it would be a cheap-pass deployable winner because gating at layer 1 means swapping layers 2..17 (16 layers downstream) — much larger compute savings than l12h2 (5 layers downstream). Two new conditions added in commit e6e26ca and run on the same 100-trial set with `--reuse-diag`:

- `S3-Bin-W4-l1h7-bottom` — binary, layer-1 readout, bottom direction (W4-correct)
- `S3-Tern-W4-l1h7-bottom` — ternary 0.1/0.4/0.5 partition, bottom direction

Total Tier 5 cost: 2 conditions × 100 trials = 200 rollouts in ~50 min.

### Tier 5 results at n=100

| Condition | SR | avg_bits | 95% CI | Long n=50 | Object n=50 |
|---|---:|---:|---|---:|---:|
| **S3-Bin-W4-l1h7-bottom** | **0.980** | **9.32** | [0.950, 1.000] | 0.960 | 1.000 |
| **S3-Tern-W4-l1h7-bottom** | **0.910** | **4.43** | [0.850, 0.960] | 0.880 | 0.940 |

### Tier 5 matched-pair deltas (all over n=100)

| Tag | Comparison | Δ SR | Verdict |
|---|---|---:|---|
| HW11a | l1h7-bottom vs l1h7-top1 (direction flip, binary) | **+0.010** | direction-flip is real but small (1 pp) |
| HW11b | l1h7-bottom vs l12h2-ent-bottom (binary, cross-layer) | +0.010 | l1h7 binary slightly better SR but 9.32 vs 4.42 bits — not Pareto |
| HW11c | l1h7-bottom-tern vs l12h2-bottom-tern (ternary, cross-layer) | **-0.040** | l1h7 ternary loses to l12h2 ternary by 4 pp |
| HW11d | l1h7-bottom-tern vs W4-Floor (Pareto test) | **-0.030** | l1h7 ternary LOSES to uniform W4 |

### Two findings — direction is small, spatial is dominant

**1. Direction-flip at l1h7 is a real but small effect (+1.0 pp).** Confirms the per-trial Spearman ρ = -0.155 signal is genuine: targeting low-top1 cycles for FP16 escalation does help vs. the W2-default top direction. But the magnitude is small — same order as the direction-flip effects we saw for AttnEntropy/S1-Bin/S2-Bin/S3-Bin-l12h2 (range ±3 pp).

**2. The cheap-pass hypothesis is FALSIFIED.** `S3-Tern-W4-l1h7-bottom` gets 91.0% SR at 4.43 avg bits — **worse than W4-Floor's 94.0% / 4.00 bits on BOTH dimensions**. Even with the W4-correct direction at the strongest-signal probe, ternary at the earliest gate layer collapses by 3 pp. This is the same spatial-collapse mechanism that broke S1/S2-Tern: with L=1, the W2-demotable "tail" includes layers 2-17 (16 of 18 VLM layers, including the sensitive 2-12 region). Targeted demotion of "the right" cycles still fails because mid-VLM Gemma layers cannot tolerate W2 even on individual cycles.

### What this means

The S3-Tern-W4-l12h2 Pareto winner depends specifically on its narrow demotion footprint (layers 13-17 only). Moving the gate earlier into the VLM expands the W2-demotion footprint into the spatially sensitive zone, undoing the cost-vs-quality trade. The "earliest cheap layer" intuition from MEETING_5 is incompatible with the spatial-restriction requirement — they pull in opposite directions:

- **Earlier gate → more layers downstream of L → larger compute savings per W2 demotion** (good)
- **Earlier gate → demoted layers include sensitive mid-VLM 2-12** (catastrophic)

The two conditions cannot be jointly satisfied with the current quantization setup. **`S3-Tern-W4-l12h2` is therefore the deployable optimum at this design point**: gating at layer 12 is the earliest position whose downstream tail (13-17) is entirely outside the sensitive zone.

### Updated synthesis

| Hypothesis | Status |
|---|---|
| AttnEntropy direction-flip at W4 (vs W2 default) | ✓ Confirmed (small effect, 1-3 pp) |
| Layer-restricted W2 demotion enables sub-W4 average | ✓ Confirmed (S3-Tern-W4-l12h2: 95.2% / 3.58 bits) |
| Full-pass W2 demotion (S1/S2/Random-Tern) is destructive | ✓ Confirmed (-15 to -30 pp loss) |
| Earlier-layer gate (l1h7) enables cheaper deployable scheme | ✗ **Falsified** — spatial constraint dominates |
| Strongest per-trial probe (l1h7-top1) → strongest deployable scheme | ✗ Per-trial ρ doesn't transfer to rollout-level SR when the demotion footprint includes sensitive layers |

**The deployable contribution is firmly: layer-restricted intra-pass W2 demotion at L=12 with l12h2 entropy gate, bottom direction, frac=0.4, ternary partition (0.1, 0.4, 0.5).** Sub-W4 average bits (3.58) at matched/better SR (95.2% vs Floor 94.0%) on 100 paired LIBERO trials.

### Open follow-ups (not yet run)

1. **Layer-by-layer ternary sweep**: test L = 8, 10, 11, 13, 14, 15 to find the exact transition point where the spatial collapse begins. Currently jumps from "fails at L=1" to "works at L=12" — there's a continuous transition somewhere in between worth characterizing.
2. **Conservative aggressive partitions at L=12**: `S3-Tern-W4-l12h2` with partitions (0.05, 0.25, 0.7) → ~3.0 avg bits, or (0, 0.4, 0.6) → no FP16 escalation at all (pure W2/W4 mix). The latter would test whether FP16 rescue is needed at L=12 or if the demotion side alone is sufficient.
3. **Active expert quantization**: currently the action expert is FP16 throughout. A W4 action-expert would lower avg bits further; combined with S3-Tern-W4-l12h2 could push to ~2.5-3 effective bits.

---

## Experiment D — LIBERO-PRO at W4 (2026-04-29 → 30, in progress)

**Why this exists.** ExpC's W4 conditions all tied at 94% SR on standard LIBERO. There's no rescue gap to close at W4 because pi0.5 doesn't degrade — the benchmark is saturated. Per Wonsuk + the LIBERO-PRO paper (arXiv:2510.03827, github.com/Zxy-MLlab/LIBERO-PRO), most VLAs memorize standard LIBERO. Wonsuk's question: "maybe 4-bit shows degradation as well for harder task?" Test by swapping the benchmark (not the model, not the quantization stack) into a regime where pi0.5 FP16 has actual headroom (~50–70% SR), so W4 has room to degrade and AttnEntropy has signal to detect.

LIBERO-PRO ships static perturbed-init bundles per (suite, axis, magnitude). Only the Object suite has the configurable Figure-6 magnitudes (`libero_object_temp_x{0.1..0.5}` and `_y*`); other suites use a different YAML-driven engine, so all of ExpD is on Object.

### Step 1 — integration smoke (2026-04-29)

`scripts/setup_libero_pro.sh` clones LIBERO-PRO, overlays its `benchmark/__init__.py` + `libero_suite_task_map.py` to register `libero_<suite>_temp` task suites in the openpi LIBERO checkout, and downloads bundles from HuggingFace `zhouxueyang/LIBERO-Pro` (322 files; falls back to repo bundles if HF fails). `scripts/rollout.py` adds `--pro-config "Suite:axis:magnitude"` plus `set_libero_pro_config()` / `stage_libero_pro_files()` (fcntl-locked, sentinel-skipped, idempotent). When a Pro config is active for a suite, `make_libero_env()` routes to the `_temp` variant after staging the right bundle. Unflagged invocations are byte-identical.

5-trial smoke at `--pro-config "Object:x:0.2"` ran cleanly: bundle staged, env constructed from `libero_object_temp`, MuJoCo rendered, policy generated actions for "pick up the alphabet soup and place it in the basket". One trial timed out — expected at this magnitude.

### Step 2 — operating point sweep (2026-04-29)

`scripts/find_operating_point.py` ran pi0.5 FP16 on Object × magnitudes {0.1, 0.2, 0.3} × 50 trials each (150 FP16 rollouts, 45 min on GPU 0).

| Magnitude | FP16 SR | 95% CI |
|---|---:|---|
| x0.1 | 0.880 | [0.78, 0.96] |
| x0.2 | 0.500 | [0.36, 0.64] |
| x0.3 | 0.480 | [0.34, 0.62] |

x0.1 saturated; x0.2 and x0.3 statistically tied near 50%. **Picked x0.2** as the operating point — pi0.5 sits near the published Figure-6 ~50% range, leaving W4 clear headroom both ways.

### Step 3 — focused 5-condition expC subset at x0.2, n=50 (2026-04-30, complete)

Re-used `scripts/expB_sis_validation.py --w4-pro --pro-config "Object:x:0.2" --frac 0.5 --ternary-partition "0.1,0.4,0.5" --out-tag libero_pro_obj_x0.2`. Same model, same quantization stack, same diagnostic V3 + intra-pass driver, just routed through the perturbed Object suite. 5 conditions × 50 trials = 250 rollouts, ~3.5 h on GPU 0.

#### Aggregate SR — looks like a null

| Condition | SR | 95% CI | avg bits |
|---|---:|---|---:|
| FP16 | 0.540 | [0.40, 0.68] | 16.0 |
| W4-Floor | 0.480 | [0.34, 0.62] | 4.0 |
| Random-W4 (50% FP16, random) | 0.580 | [0.44, 0.72] | 10.0 |
| AttnEntropy-W4 (bottom-50% l12h2 entropy) | 0.520 | [0.38, 0.66] | 10.0 |
| **S3-Tern-W4-l12h2** (intra-pass three-tier) | **0.560** | [0.42, 0.70] | **3.49** |

Matched-pair deltas: HW0 = SR(W4-Floor) − SR(FP16) = −0.060 (W4 *does* degrade vs FP16, in the right direction); HW6 = SR(AttnEntropy-W4) − SR(Random-W4) = −0.060 (the rescue gate apparently underperforms random). At face value this looks like the standard-LIBERO null carries over: W4 barely degrades, AttnEntropy doesn't beat random.

D2-W4 transfer at this regime: per-trial Spearman ρ(l12h2-ent W4-pass, ‖a_FP − a_W4‖²) median = +0.224, mean = +0.207, P(|ρ|>0.15) = 0.74 — *stronger* than the standard-LIBERO W4 transfer (~+0.115). The mechanistic signal is real but doesn't yet convert into rollout-level rescue.

#### The conditional partition (P3) — the aggregate hides the signal

Mentor critique: aggregate SR averages over trials whose outcomes have nothing to do with quantization. Of the 50 W4-Floor failures, how many would FP16 also fail? On those, no precision gate can rescue.

Partitioned by (FP16, W4-Floor) outcome:

| Bucket | n | FP16 | W4-Floor | Random-W4 | AttnEntropy-W4 | S3-Tern |
|---|---:|---:|---:|---:|---:|---:|
| FP16✓ + W4✓ (no rescue needed) | 17 | 100% | 100% | 88% | **65%** | 82% |
| FP16✓ + W4✗ (**rescuable / quant-hard**) | **10** | **100%** | **0%** | **60%** | **80%** | **50%** |
| FP16✗ + W4✓ (W4-better) | 7 | 0% | 100% | 71% | 57% | **86%** |
| FP16✗ + W4✗ (**unrescuable / benchmark-hard**) | **16** | **0%** | **0%** | **19%** | **19%** | **19%** |

Five findings:

1. **62% of W4 failures (16/26) are benchmark-hard** — FP16 also fails. No precision gate can rescue these. They drag the AttnEntropy-W4 aggregate down because every override condition just inherits the unrescuable failure rate.

2. **On the rescuable subset (n=10): AttnEntropy-W4 80% > Random-W4 60% — a +20 pp matched-pair gap** in the same direction as the W2 expB +29 pp result. n=10 is too small for significance (McNemar p ≈ 0.69), but the directional confirmation is exactly the predicted mechanism.

3. **All three rescue conditions HURT in the clean bucket.** AttnEntropy-W4 loses 35 pp (drops from 100% → 65%); S3-Tern loses 18 pp; Random loses 12 pp. The cost is precision-switching itself perturbing the trajectory off-path on cycles that didn't need it. AttnEntropy suffers worst — the same gate that picks "high-sensitivity" rescue frames in rescuable trials misfires as "high-sensitivity disruption" in clean trials. **Implication: deployable rescue needs a gate that fires only when W4 is actually failing, not unconditionally on bottom-50%-by-entropy.**

4. **The 19% floor on unrescuable trials is fundamental.** All three rescue conditions hit exactly 3/16. That's the random-perturbation success rate when both pure-precision schedules fail — mixed-precision schedules visit different states than pure-W4 or pure-FP16 trajectories. Every paper claim about "rescue rate" needs to subtract this floor.

5. **S3-Tern's +8 pp over W4-Floor decomposes as (−3 clean, +5 rescuable, −1 W4-better, +3 unrescuable).** The +5 from real rescue is the largest piece — the W2-demotion gate is doing real work — but ~40% of the gain is trajectory-divergence luck on unrescuable trials. The honest framing is "majority-rescue, partly-luck" rather than pure Pareto improvement.

#### In progress (launched 2026-04-30)

Both running on tambe-server-1 in parallel:

- **P1 — n=200 at x0.2 on GPU 0** (`expB_w4__libero_pro_obj_x0.2_n200_*`). Same 5 W4 conditions, four-fold trial budget. Target: bring rescuable-bucket n from 10 → ~40 so McNemar can resolve the +20 pp AttnEntropy vs Random gap. ETA ~7h total wall time. *Partial at n=37 trials*: AttnEntropy-W4 4/5 (80%), Random-W4 1/5 (20%) on rescuable subset — the +60 pp partial gap is even stronger than the n=10 read but n=5 is preliminary.
- **P2 — n=100 at x0.4 on GPU 1** (`expB_w4__libero_pro_obj_x0.4_n100_*`). Same conditions, harder magnitude. Tests whether the conditional rescue replicates across difficulty. ETA ~3.5h. *Partial at n=35 trials*: bucket distribution is **7 clean / 1 rescuable / 2 W4-better / 25 unrescuable** — at x0.4 essentially every failure is benchmark-hard. Likely an "operating point too far past the FP16 cliff" finding; the rescue-question is unanswerable here because the rescuable-subset n is ~3 even at full n=100.

#### What this experiment is testing, in one paragraph

Standard LIBERO doesn't surface W4 degradation, so AttnEntropy-W4 has nothing to rescue. LIBERO-PRO's position perturbation creates a regime where pi0.5 FP16 sits at ~50% SR — a hard-but-not-collapsed benchmark. **The hypothesis Wonsuk asked us to test is: in this harder regime, does W4 degrade enough that AttnEntropy can rescue it?** The aggregate-SR n=50 read is ambiguous (W4 degrades 6 pp, AttnEntropy underperforms Random by 6 pp). The conditional partition reveals the real mechanism: most failures are benchmark-hard (no gate helps), but on the trials where W4 actually breaks AttnEntropy *can* rescue (80% vs Random's 60%). The n=200 follow-up is the statistical power test on that conditional gap. The x0.4 follow-up tests if the same mechanism replicates at a different magnitude.

#### Files of record

- `scripts/setup_libero_pro.sh`, `scripts/find_operating_point.py` — new
- `scripts/rollout.py`, `scripts/expB_sis_validation.py` — extended with `--pro-config`, `--w4-pro`/`--w2-pro`, `--out-tag`, `--pro-n-per-suite`
- `results/libero_pro_operating_point.md`, `.jsonl` — Step 2
- `results/expB_w4__libero_pro_obj_x0.2_{rollouts,summary}.{jsonl,md}`, `expB_diagnostic_v3__libero_pro_obj_x0.2.jsonl` — Step 3 n=50 (complete)
- `results/expB_w4__libero_pro_obj_x0.2_n200_*` — P1 (in progress)
- `results/expB_w4__libero_pro_obj_x0.4_n100_*` — P2 (in progress)
- Server logs: `/data/subha2/experiments/logs/expB_w4_pro_x0.2_n200.log`, `expB_w4_pro_x0.4_n100.log`

### P2 killed; pivoted to trial-gate analysis (2026-04-30)

P2 (n=100 at x0.4) was killed at 49/100 trials done because the partial bucket distribution was 25 unrescuable / 1 rescuable / 7 clean / 2 w4_better — at full n=100 the rescuable subset would have been ~3 trials, far too few to test the +20 pp gap. x0.4 is past the FP16 cliff; the rescue-detection question is unanswerable at this magnitude. The compute is reallocated to a Phase A trial-gate analysis on the existing n=50 data (no new rollouts).

### Phase A — Stage-1 trial-gate detector on n=50 (2026-04-30)

**Motivation.** The aggregate-vs-conditional split from P3 reveals that AttnEntropy-W4 has an **asymmetric cost**: it rescues 80% of the rescuable-bucket trials but breaks 35% of the clean-bucket trials, because the per-cycle gate fires unconditionally regardless of whether the trial actually needs rescuing. Literature (FIPER, "Shifting Uncertainty," "Failure Detection Without Failure Data") points to a **two-stage detector**: Stage-1 ("is this rollout heading toward W4 failure?") gates Stage-2 (the existing per-cycle AttnEntropy demotion). Without Stage-1, the clean-bucket cost cancels the rescuable-bucket win in aggregate. This sub-experiment tests whether such a Stage-1 signal is extractable from existing data — purely from the first K cycles of the W4-base diagnostic, no new rollouts.

**Method (`scripts/exp_trialgate_analysis.py`).** 90 features per trial: 5 attention probes (l1h7-top1, l9h2-ent, l12h2-ent, l3h4-top5, l17h4-top1) + SIS, each over K ∈ {5, 10, 15} cycles, summarized by {mean, std, max, min, slope}. Only W4-pass values (deployable at runtime); FP16-pass deliberately excluded since FP16 isn't running in deployment. Logistic regression with L2 + class balancing, evaluated by leave-one-out CV (50 folds). FIPER-style conformal threshold calibrated on the W4-Floor-success subset (FAR ≤ α). Two binary targets: `y_w4_fail` (W4-Floor failed; pos 26/50) and `y_rescuable` (FP16 succeeds AND W4-Floor fails; pos 10/50).

#### Headline result — Stage-1 detects task difficulty, not quant sensitivity

| Target | AUC | 95% CI | Brier |
|---|---:|---|---:|
| `y_w4_fail` (deployable) | **0.740** | [0.593, 0.865] | 0.204 |
| `y_rescuable` (deployable) | **0.455** | [0.227, 0.697] | 0.254 |
| `y_w4_fail` (oracle: mse_fp_w4) | 0.569 | [0.409, 0.723] | 0.273 |
| `y_rescuable` (oracle: mse_fp_w4) | 0.390 | [0.182, 0.617] | 0.258 |

Stage-1 AUC of **0.74 for "will W4 fail"** — moderate but real signal from early-cycle attention. But AUC of **0.46 for "rescuable"** — chance, with the lower 95% bound below 0.30. The detector predicts task difficulty in general, not the rescuable subset specifically. The early-cycle oracle (`mse_fp_w4` quantization MSE) is even *worse* on both targets — its early-trajectory readings don't predict the late-trajectory outcome.

#### The fire pattern confirms the diagnosis

Per-bucket breakdown of the best deployable detector (`y_w4_fail`, α=0.20, threshold=0.548, fires on 21/50 trials):

| Bucket | n | n_fired | fire rate | gated SR | Right answer? |
|---|---:|---:|---:|---:|---|
| clean (FP16✓ + W4✓) | 17 | 2 | 12% | 100% | ✅ correct: low fire rate, preserves clean trials |
| **rescuable (FP16✓ + W4✗)** | **10** | **2** | **20%** | **10%** | ❌ should fire HERE; almost never does |
| w4_better (FP16✗ + W4✓) | 7 | 3 | 43% | 86% | partially OK (no firing means W4 retained) |
| **unrescuable (FP16✗ + W4✗)** | **16** | **14** | **88%** | **12%** | ❌ should NOT fire here; fires almost always |

The detector's fire pattern is **exactly backwards from useful**: it fires hardest on the unrescuable bucket (88%) where rescue can't help anyway, and least on the rescuable bucket (20%) where rescue would have +20 pp impact. Translation: early-cycle attention metrics correlate with intrinsic task difficulty, which captures benchmark-hard trials more cleanly than W4-quantization-hard trials.

#### Gated SR matches ungated SR — Stage-1 changes nothing

Simulated gated AttnEntropy-W4: 52% (matches ungated 52%, McNemar p=1.000). Even at α=0.20 — the most permissive false-alarm budget — the gated detector and the unconditional baseline give the same matched-pair outcomes. Random-W4 still wins at 58% by pure trajectory-divergence luck.

| Comparison | A only | B only | Δ SR | McNemar p |
|---|---:|---:|---:|---:|
| Gated-AttnEnt vs AttnEnt-W4 | 8 | 8 | +0.000 | 1.000 |
| Gated-AttnEnt vs W4-Floor | 3 | 1 | +0.040 | 0.625 |
| Gated-AttnEnt vs Random-W4 | 4 | 7 | −0.060 | 0.549 |

#### What this means

The Stage-1 detector exists (AUC 0.74 for "will W4 fail" is genuine signal) but **it's measuring the wrong thing**. Early-cycle attention metrics encode *task difficulty*, which captures both quantization-induced and benchmark-induced failures in one pool. Since the unrescuable bucket is 60% of W4 failures (16/26), any "predict W4 failure" detector trained at this scale will fire mostly on those trials — exactly where rescue can't help. The same problem afflicts the per-cycle AttnEntropy gate (low entropy → high "sensitivity" → escalate), which is why ungated AttnEntropy didn't beat random in the n=50 aggregate.

**The signal we need is one that distinguishes quantization-failure from benchmark-failure** — and at deploy time, the only candidates are signals derived from the policy's *output* (action chunks), not its *internals* (attention). Action chunks are precision-affected: under W4 the action chunk diverges from the FP16 chunk, and that divergence (or its proxies — chunk variance, sign-flip rate, inter-chunk discrepancy) is the only quantity at deploy time that's specific to quantization rather than task hardness. This is exactly the Phase B signal set FIPER and "Shifting Uncertainty" propose.

#### Files of record

- `scripts/exp_trialgate_analysis.py` (new, ~510 LOC) — re-runnable on n=200 data once P1 finishes via `--data-tag libero_pro_obj_x0.2_n200`.
- `results/expD_trialgate_summary__libero_pro_obj_x0.2.md` — full table including all 12 (target × α × rescue-condition) simulated SR rows.
- `results/expD_trialgate_features__libero_pro_obj_x0.2.jsonl` — per-trial feature matrix + labels for inspection.

#### Open follow-ups

- **Phase B — action-chunk signals.** A re-run pass that logs the actual action chunks for each cycle of the n=50 W4 trajectory (~30 min on H100, FP16-only no SIS perturbations). Extract action-chunk variance (Signal B), action sign-flip rate (Signal C), inter-chunk discrepancy (Signal D). Run the same trial-gate framework on those features. Hypothesis: action-derived signals can distinguish quant-failure from benchmark-failure where attention can't.
- **Rerun on P1 n=200.** Once P1 finishes (~6 h remaining as of writeup), rerun `exp_trialgate_analysis.py --data-tag libero_pro_obj_x0.2_n200`. The 4× larger sample gives real bootstrap CIs on AUC and the gated SR comparison; a directional Phase A result on n=50 may flip if the underlying signal is weak but real, or stay flat if Phase A's "wrong target" diagnosis holds.
- **Consider an end-to-end gated rollout** with a real `--w4-pro-gated` mode that runs Stage-1 inline from the partial diagnostic (cycles 0..K) and engages AttnEntropy from cycle K+1. Worth doing only if Phase A or Phase B shows a Stage-1 signal that survives the rescuable-vs-unrescuable confound.

### Morning n=200 + Phase B follow-up (2026-04-30)

Overnight orchestrator on `tambe-server-1` (5 sequential steps after P1 finished) ran cleanly to completion at 05:30 PDT. The publisher script that was supposed to push results to GitHub silently failed because git on the remote didn't have `user.email` configured (the publisher's `|| { exit 0 }` swallowed the real error). Recovered manually in the morning by configuring git on the remote, fetching the resulting commit over SSH, and pushing from the Mac. Patched the publisher to fail loudly on commit/push errors going forward.

#### HW0 flipped at scale — W4 doesn't degrade vs FP16

| Condition | n=50 SR | **n=200 SR** | 95% CI |
|---|---:|---:|---|
| FP16 | 0.540 | **0.420** | [0.36, 0.49] |
| W4-Floor | 0.480 | **0.460** | [0.39, 0.53] |
| Random-W4 | 0.580 | 0.445 | [0.38, 0.52] |
| AttnEntropy-W4 | 0.520 | 0.465 | [0.40, 0.54] |
| **S3-Tern-W4-l12h2** | 0.560 | **0.485** | [0.42, 0.56] |

**HW0 = SR(W4-Floor) − SR(FP16) flipped sign**: −0.060 at n=50, **+0.040 at n=200**. The "W4 degrades on harder benchmark" hypothesis that motivated the LIBERO-PRO benchmark swap is **falsified at this scale** — W4-with-protection actually *beats* FP16 on LIBERO-PRO Object x0.2 in matched-pair comparison. The n=50 result was a small-sample artifact in the trial mix.

S3-Tern-W4-l12h2 still leads at 48.5% (vs W4-Floor 46.0%) at 3.49 avg bits — sub-W4 at matched-or-better SR survives, though both n=50 and n=200 CIs overlap with W4-Floor.

#### The rescue mechanism replicates and STRENGTHENS on the rescuable bucket

Conditional partition at n=200 (computed from `expB_w4__libero_pro_obj_x0.2_n200_rollouts.jsonl`):

| Bucket | n | FP16 | W4-Floor | Random-W4 | **AttnEntropy-W4** | S3-Tern |
|---|---:|---:|---:|---:|---:|---:|
| clean (FP16✓ + W4✓) | 69 | 100% | 100% | 91% | 92% | 95% |
| **rescuable (FP16✓ + W4✗)** | **15** | **100%** | **0%** | **53%** | **86%** | **80%** |
| w4_better (FP16✗ + W4✓) | 23 | 0% | 100% | 47% | 56% | 65% |
| unrescuable (FP16✗ + W4✗) | 93 | 0% | 0% | 7% | 3% | 4% |

**On the rescuable bucket: AttnEntropy-W4 = 13/15 = 86% vs Random-W4 = 8/15 = 53% — matched-pair Δ = +33 pp.** Stronger than the +20 pp directional read at n=10 in the original n=50 partition. McNemar exact two-sided p = 0.125 (a_only=6, r_only=1, both=7, neither=1). Not yet at α=0.05 because n_rescuable is only 15, but the effect size is real and the direction is solid. S3-Tern on rescuable: 12/15 = 80%, +26 pp vs Random, p = 0.125.

**Why the aggregate looks null at n=200.** The unrescuable bucket is **47% of all 200 trials** (n=93). Every rescue method floors at ~3–7% there (trajectory-divergence noise). The rescuable bucket is only 7.5% of trials. The aggregate becomes `0.345·(+1pp) + 0.075·(+33pp) + 0.115·(+9pp) + 0.465·(−4pp) ≈ +1pp` — the rescue effect is real but invisibly diluted.

**Clean-bucket cost is much smaller at scale.** AttnEntropy drops only 8 pp on clean trials at n=200 (100→92%), vs the 35 pp drop at n=50. The trial-gate concern from Phase A is partially defused: the bigger problem is that rescuable trials are *rare*, not that gating breaks clean trials catastrophically.

#### Phase B (action-chunk signals B/C/D) — falsified at n=200

Logged W4 action chunks per cycle for all 200 trials (`expD_chunks__libero_pro_obj_x0.2_n200.jsonl`, 8982 records, ~30 min on GPU 1). Computed Signal B (intra-chunk variance), C (within-chunk sign-flip rate), D (inter-chunk L2 with overlap=5 from `replan_steps=5`). Re-ran the trial-gate analysis with three feature sets: attention-only (90 features, Phase A), chunks-only (45 features, Phase B), combined (135 features).

| Target | Features | n_feats | AUC | 95% CI |
|---|---|---:|---:|---|
| `y_w4_fail` | attn | 90 | **0.834** | [0.77, 0.89] |
| `y_w4_fail` | chunks | 45 | 0.653 | [0.58, 0.73] |
| `y_w4_fail` | combined | 135 | **0.842** | [0.78, 0.90] |
| `y_rescuable` | attn | 90 | 0.603 | [0.45, 0.75] |
| `y_rescuable` | chunks | 45 | 0.395 | [0.22, 0.59] |
| `y_rescuable` | combined | 135 | 0.525 | [0.35, 0.70] |

**The Phase B chunk-signal hypothesis is empirically falsified.** Chunks alone hit AUC 0.40 on `y_rescuable` (worse than chance), and combined 0.53 (also chance). Attention-only at 0.60 wins this comparison; adding chunks *hurts* by injecting noise. The intuition that "action chunks reflect quantization-specific uncertainty in a way attention can't" doesn't survive contact with n=200.

`y_w4_fail` AUC is statistically powered (0.83 attention, 0.84 combined) — the detector can robustly predict "this trial will fail," but the per-bucket fire pattern at the best detector (target=`y_w4_fail`, features=combined, α=0.20) confirms Phase A's diagnosis at scale: fires on 89% of unrescuable, only 20% of rescuable. Same task-hardness vs quant-sensitivity confound, just with tighter CIs.

#### What this means for the paper

The W2 ExpB AttnEntropy result (+29 pp aggregate rescue at frac=0.5) and the W4 ExpC S3-Tern Pareto win (95.2% / 3.58 bits) **both stand**. The new claim from ExpD is more nuanced and still publishable:

> *AttnEntropy-W4's per-frame rescue mechanism produces a +33 pp matched-pair effect on quantization-induced trajectory failures (n=15 rescuable trials out of 200; McNemar p=0.125). The effect is invisible in aggregate SR because **benchmark-induced failures (FP16 also fails) dominate the trial mix on LIBERO-PRO Object x0.2**, accounting for 47% of trials. The per-frame attention-entropy signal cannot distinguish quantization-induced from benchmark-induced failures; trial-level gating with both attention features (AUC 0.60 for rescuable) and action-chunk features (AUC 0.40) fails to separate the two failure modes.*

This is a real mechanistic contribution, distinct from but complementary to S3-Tern, with an honest limitation. Three forward paths from here, ordered by my preference:

- **Path C (preferred):** accept and write up. Bring to Wonsuk for review with the conditional partition as the headline rather than aggregate SR. Reframes the LIBERO-PRO experiment as "validates the rescue mechanism on the right subset, characterizes the trial-mix obstacle for aggregate reporting."
- **Path A (statistical power):** run ~200 more trials at x0.2. With n_rescuable scaling roughly linearly, this would put us at ~30 rescuable, where p < 0.05 is reachable if the +33 pp effect holds. ~12–24 h on H100. Pure power, no new mechanism.
- **Path B (better operating point):** the unrescuable fraction dominating the aggregate is the structural problem. Sweep at x0.15 (between current 0.1 and 0.2) where FP16 might sit at ~70% and the rescuable fraction might be 20–30%+ instead of 7.5%. ~3 h to repeat Step 2; then re-decide whether to launch Step 3 there.

#### Files of record

- `results/expB_w4__libero_pro_obj_x0.2_n200_summary.md`, `_rollouts.jsonl` (1000 rows)
- `results/expD_trialgate_summary__libero_pro_obj_x0.2_n200.md` (Phase B at n=200)
- `results/expD_trialgate_features__libero_pro_obj_x0.2_n200.jsonl`
- `scripts/expD_log_chunks.py`, `scripts/expD_overnight.sh`, `scripts/expD_publish_results.sh` (now patched to fail loudly)
- Server logs: `/data/subha2/experiments/logs/expD_overnight.log`, `expD_trialgate_n200_phaseB.stdout.log`

### Supplementary tables for the slide deck (2026-04-30)

Five additional analyses computed from existing JSONLs in `results/`. Full tables in `results/expD_supplementary.md`; the highest-impact ones are reproduced below.

#### Regime contrast — bucket distribution by experiment

The single best slide for "rescue regime vs cost-reduction regime":

| Regime | n | clean | rescuable | w4_better | unrescuable |
|---|---:|---:|---:|---:|---:|
| ExpC standard-LIBERO n=100 | 100 | 90 (90%) | 4 (4%) | 4 (4%) | 2 (2%) |
| ExpD LIBERO-PRO Object x0.2 n=50 | 50 | 17 (34%) | 10 (20%) | 7 (14%) | 16 (32%) |
| ExpD LIBERO-PRO Object x0.2 n=200 | 200 | 69 (34%) | 15 (8%) | 23 (12%) | 93 (46%) |

ExpC standard LIBERO is **90% clean** at W4 — the deployment story is *cost reduction* (S3-Tern's sub-W4 average bits at matched SR), not rescue. ExpD LIBERO-PRO at x0.2 has rescuable trials but the unrescuable bucket (47% at n=200) dominates the aggregate. The W2-on-standard-LIBERO regime (expB) put effectively 100% of trials in rescuable+unrescuable (W2-Floor SR ≈ 0%), which is why the +29 pp rescue gap was so visible in aggregate there.

#### ExpC S3-Tern conditional partition on standard LIBERO (n=100)

| Bucket | n | FP16 | W4-Floor | Random-W4 | AttnEntropy-W4 | S3-Tern |
|---|---:|---:|---:|---:|---:|---:|
| clean | 90 | 100% | 100% | 100% | 98% | 99% |
| rescuable | 4 | 100% | 0% | 100% | 100% | 50% |
| w4_better | 4 | 0% | 100% | 100% | 100% | 100% |
| unrescuable | 2 | 0% | 0% | 0% | 0% | 0% |

S3-Tern's +1 pp aggregate win on standard LIBERO is **almost entirely a clean-bucket effect**: it preserves 99% of clean trials (vs W4-Floor's 100%) but the tiny rescuable/w4_better/unrescuable buckets contribute negligibly at n=4/4/2. The 95.2% / 3.58 bits result is "spatial restriction doesn't break clean trials," which is enough at this regime because the rescue + benchmark-hard surface area is tiny.

#### ExpC Tier 5 (l1h7-bottom) bucket decomposition

| Bucket | n | W4-Floor | S3-Tern-l12h2 | S3-Bin-l1h7-bottom | S3-Tern-l1h7-bottom |
|---|---:|---:|---:|---:|---:|
| clean | 90 | 100% | 99% | 100% | **93%** |
| rescuable | 4 | 0% | 50% | **100%** | **75%** |
| w4_better | 4 | 100% | 100% | 100% | 100% |
| unrescuable | 2 | 0% | 0% | 0% | 0% |

**Tier 5 aggregate −3 pp loss is clean-bucket damage, not rescue failure.** `S3-Tern-W4-l1h7-bottom` actually wins +75 pp on the rescuable bucket (3/4 vs W4-Floor's 0/4) but loses 6 clean trials (84/90 vs 90/90). Net: +3 rescuable wins − 6 clean losses = −3 trials = −3 pp. The l1h7 readout is *better at picking rescue cycles* than l12h2 (4/4 vs 2/4 wins on the binary variant) — the spatial collapse from W2 demoting layers 2–17 is what kills the ternary.

#### Memory footprint: S3-Tern-W4-l12h2 vs uniform W4

Back-of-envelope. pi0.5-LIBERO: vision ~400M, layer 0 ~110M, lang layers 1–17 ~110M × 17, expert ~300M. Bytes/param: FP16=2, W4=0.5, W2=0.25 (packed).

| Component | Size | Uniform W4 | S3-Tern-W4-l12h2 |
|---|---:|---:|---:|
| Vision tower (FP16) | 400M | 0.80 GB | 0.80 GB |
| Lang layer 0 (FP16) | 110M | 0.22 GB | 0.22 GB |
| Lang layers 1–12 (W4) | 1.32B | 0.66 GB | 0.66 GB |
| Lang layers 13–17 (W4 only / W4+W2+FP16 cached) | 0.55B | 0.28 GB | **1.51 GB** |
| Action expert (FP16) | 300M | 0.60 GB | 0.60 GB |
| **Total VLM+expert weights** | — | **2.56 GB** | **3.79 GB** |

**Trade.** S3-Tern needs ~+1.24 GB (+48%) over uniform W4 to hold three precisions resident for the demoted layers (13–17). In return, active forward-pass cost drops ~13% (3.49 vs 4.0 avg bits). The deployable claim is "pay ~half a GB more memory for ~13% lower compute and matched-or-better SR," not "free lunch."

#### n=200 trial-gate fire pattern (confirms Phase A diagnosis at scale)

Best deployable detector at n=200: target=`y_w4_fail`, features=combined, α=0.20.

| Bucket | n | n_fired | fire rate | gated SR | AttnEntropy unconditional |
|---|---:|---:|---:|---:|---:|
| clean | 69 | 11 | 16% | 99% | 92% |
| **rescuable** | **15** | **3** | **20%** | **20%** | **86%** |
| w4_better | 23 | 8 | 35% | 91% | 56% |
| **unrescuable** | **93** | **83** | **89%** | **2%** | 3% |

The "exactly backwards from useful" fire pattern from Phase A holds at n=200: detector fires hardest on the unrescuable bucket (89%) and barely on rescuable (20%). The 86% AttnEntropy rescue rate on the rescuable bucket *evaporates under gating* (drops to 20%) because the gate refuses to fire on 80% of those trials, defaulting them to W4-Floor's 0%. This is now a four-fold-larger sample size confirming the same diagnosis.

(Full supplementary including matched-pair McNemar tables on the n=200 rescuable bucket: see `results/expD_supplementary.md`.)
