# Experiment Findings — 2026-04-16

Overnight sensitivity probing of pi0.5 LIBERO (3.6B params, PyTorch, eager mode) on tambe-server-1 (H100 PCIe). 256 observations: 128 Long (hard, multi-step tasks) + 128 Object (easy, single pick-and-place).

---

## Experiment 1: Cross-Suite Activation Statistics

**Question:** Do long-horizon (hard) observations produce systematically different activation distributions than short-horizon (easy) observations? If yes, an online precision controller has a signal to condition on.

**Method:** 256 forward passes with hooks on all 458 linear layers. Per-layer per-sample: max|activation|, kurtosis, outlier fraction (>6 sigma), std.

**Results:**

- The easy and hard activation curves overlap almost everywhere. For the vast majority of layers (~450/458), there is no meaningful distributional difference between suites.
- A handful of VLM `mlp.down_proj` layers have extreme kurtosis (5,000-15,000 for both suites) with moderate deltas between suites:
  - `language_model.layers.0.mlp.down_proj`: kurtosis delta = -1,476 (easy higher)
  - `language_model.layers.17.mlp.down_proj`: kurtosis delta = +1,475 (hard higher), max activation delta = +5,848 (hard produces larger outliers, up to 41,000 in magnitude)
- The outlier fraction (6-sigma) deltas are tiny (~0.0005) across all layers. Easy and hard suites produce nearly identical outlier structure.
- Action expert layers and vision tower layers show no suite-dependent differences.

**Interpretation:** The strong form of the horizon-differential hypothesis — "hard-task observations produce systematically different activations that demand different quantization treatment" — is **not well-supported**. At the single-observation level, the model doesn't distinguish between task difficulties in a way that changes the quantization-relevant activation statistics (outlier structure, dynamic range). The few layers with large kurtosis deltas are VLM `mlp.down_proj` layers that have extreme kurtosis for *both* suites; the deltas are a small fraction of the total.

This aligns with the pre-experiment prediction: single observations from Long vs Object tasks look similar to the model because both involve similar visual scenes (kitchen environments, similar objects). The real difficulty difference manifests over rollout length (error compounding), not at the per-observation level.

**What is useful:** The kurtosis/max-activation profiling identifies `language_model.layers.0` and `language_model.layers.17` as extreme-outlier layers that would break naive quantization regardless of task type. These need SmoothQuant-style outlier handling or higher precision.

---

## Experiment 2: Layer-wise Sensitivity Probe

**Question:** (A) Which layers are most sensitive to weight quantization? (B) Does sensitivity differ between hard and easy tasks?

**Method:** For each of 42 layer groups, quantize only that group to W4/W8/W2, run all 256 observations, measure action MSE vs FP16 reference. Per-sample results saved with metadata.

**Results:**

### (A) Sensitivity spectrum

**At W4:** Sensitivity is surprisingly uniform. All 42 layer groups produce MSE in the range 0.004-0.007 when individually quantized. No single layer is catastrophically more sensitive. The action expert layers, VLM layers, vision tower, projector, and action head all sit in the same narrow band. This means W4 is a safe blanket bitwidth — you could quantize any individual layer to W4 with minimal action quality impact.

**At W8:** Nearly identical pattern to W4 with slightly lower absolute MSE. The relative layer ordering is preserved.

**At W2:** Two layers explode while everything else stays near zero:
- `language_model.layers.0` (first VLM decoder layer): MSE ~0.32
- `paligemma.model.vision_tower` (SigLip vision encoder): MSE ~0.37

These are 50-100x more sensitive than all other layers at W2. Everything else remains at MSE ~0.001-0.01.

**This is the strongest finding.** It means:
- W4 is uniformly safe — no layer-level mixed-precision needed at W4
- The sensitivity gap only emerges at W2-W3, where two specific layers become bottlenecks
- A simple static mixed-precision scheme — keep VLM layer 0 and vision tower at W4, quantize everything else to W2 — would capture most of the compression benefit with minimal quality loss
- The action expert's 18 layers are all safe to quantize aggressively (W2 MSE stays low for all of them)

### (B) Horizon-differential sensitivity (delta: hard - easy)

At W4, the delta plot shows a pattern:
- Most action expert layers have positive deltas (0.001-0.004): hard tasks are slightly more sensitive to quantization of these layers
- Some VLM layers show negative deltas
- `time_mlp_in` has the largest positive delta

However, the deltas are small relative to the base MSE (~0.005). A delta of 0.002 on a base of 0.005 is a ~40% relative difference, which is not negligible, but the absolute magnitude is low enough that the practical impact on action quality is questionable.

**Interpretation:** There is a *trend* toward task-dependent sensitivity in the action expert, but it is not strong enough to justify the complexity of an online dynamic precision controller. A static mixed-precision scheme based on the W2 sensitivity spectrum is likely sufficient and much simpler to implement.

---

## Experiment 3: Flow-Step Sensitivity

**Question:** Do different denoising steps (pi0.5 runs 10 Euler steps from t=1.0 to t~0) have different quantization sensitivity?

**Method:** Attempted per-step weight swapping via forward hooks on the action expert. Fallback: all-FP16 vs all-W4 brute-force comparison.

**Results:** The per-step hook approach failed. The hook-based step control did not produce distinguishable outputs between all-FP16 and all-W4 configurations (both had MSE ~0.057 vs reference). The brute-force fallback produced two nearly identical bars with overlapping error bars.

**Root cause:** The pi0.5 denoising loop starts from random noise `x_t ~ N(0, I)`. Each call to `policy.infer()` samples new noise, so the FP16 reference and the test condition use different starting points. The noise-induced variance (~0.05 MSE) completely swamps the quantization-induced error (~0.005 MSE from exp2). The experiment is measuring noise, not quantization sensitivity.

**This experiment needs to be redesigned:**
1. Seed the random noise so FP16 and W4 runs start from identical `x_t` for each observation
2. This requires modifying the `sample_actions` method in PI0Pytorch to accept a noise tensor or RNG seed, rather than using the hook-based approach
3. The per-step weight swapping also needs a different mechanism — instead of hooks (which intercept the module-level forward, not the denoising-step level), modify the denoising loop directly to swap weights between steps

**The per-step question remains genuinely novel and unanswered.** No published work has measured this for any VLA. Getting it right is worth the implementation effort.

---

## Summary

| Finding | Confidence | Implication |
|---|---|---|
| Easy/hard observation activations are similar | High | Per-observation horizon-differential framing is weak. Error compounding over rollout length is more likely the explanation for LIBERO-Long's sensitivity. |
| W4 sensitivity is uniform across all 42 layer groups | High | W4 is safe for any individual layer. No mixed-precision needed at W4. |
| W2 reveals two critical bottleneck layers (VLM layer 0, vision tower) | Very high | Static mixed-precision at aggressive bitwidths is viable: protect 2 layers, compress 40. |
| Action expert layers slightly more sensitive for hard tasks | Moderate | Small effect. May not justify online dynamic precision. |
| Per-step flow-matching sensitivity | Not measured | Experiment needs redesign with seeded noise. This is the remaining novel axis. |

## Research Direction Implications

The original pitch — "online, scene-conditional dynamic precision for VLAs" — has weak empirical grounding from these results. The per-observation signal is too small for an online estimator to exploit meaningfully.

Two stronger stories emerge from the data:

**Story A (simpler, well-supported):** "Aggressive mixed-precision quantization for flow-matching VLAs: W2 everywhere except two critical layers." The W2 sensitivity spectrum is dramatic and publishable. Combined with the finding that the action expert tolerates W2 gracefully (unlike the VLM first layer and vision tower), this tells a clean architectural story about where precision matters in flow-matching VLAs.

**Story B (novel, needs exp3 fixed):** "Flow-matching step-wise precision scheduling." If the redesigned exp3 shows non-uniform per-step sensitivity, you get a genuinely novel result that no one else has. Even if the dynamic range is modest, the measurement itself is new and the architectural insight (e.g., "late denoising steps need precision, early ones don't") would be valuable.

Both stories are independent of the horizon-differential framing and don't require an online estimator. They're also more practically useful — a static mixed-precision map or a step-based scheduling rule is something an engineer could actually deploy.
