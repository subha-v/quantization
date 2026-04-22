# Quantization Research

Research project on quantization for flow-matching Vision-Language-Action (VLA) models.

## Research Direction

Studying how weight quantization of the VLM backbone in flow-matching VLAs (pi0, pi0.5) affects downstream trajectory quality. The core question: does the indirect error propagation pathway (VLM → KV cache → action expert → iterative denoising → trajectory) create different sensitivity patterns than the direct pathway in autoregressive VLAs studied by QVLA (ICLR 2026)?

**Working title:** *DP-VLA: Scene-Conditional Dynamic Precision for Flow-Matching Vision-Language-Action Models.*

### Key Novelty Axes
1. **Input-conditional (online) bit allocation for VLAs** — no published work does this
2. **Flow-matching-step-wise precision scheduling** — unique to pi0/pi0.5 architecture
3. **Layer-wise sensitivity characterization** comparing long-horizon vs short-horizon tasks

## Project Structure

```
├── RESEARCH_IDEAS.md         — Original brainstorming with advisor notes
├── RESEARCH_PLAN.md          — Detailed experiment plan (original)
├── INITIAL_EXPERIMENT.md     — Quick validation experiment template
├── UPDATED_IDEAS.md          — Refined ideas after mentor meeting
├── papers/                   — Reference papers (QVLA, BlockDialect, KVQuant, etc.)
└── scripts/                  — Experiment scripts for H100
    ├── setup_env.sh          — Environment setup (clone openpi, install deps, convert checkpoint)
    ├── setup_libero.sh       — LIBERO install helper (third_party/libero → openpi venv)
    ├── run_phase0.sh         — One-shot Phase 0 orchestrator (sync + install + smokes + full run)
    ├── utils.py              — Shared utilities (model loading, data, quantization, hooks, I/O, MUJOCO_GL)
    ├── rollout.py            — LIBERO closed-loop rollout harness (in-process, callback seams)
    ├── setup_and_verify.py   — Go/no-go verification gate
    ├── exp0_rollout_reproduce.py — FP16 pi0.5 LIBERO reproduction check (18 rollouts)
    ├── exp1_activation_stats.py  — Cross-suite activation statistics
    ├── exp2_layer_sensitivity.py — Per-layer sensitivity probe with per-sample metadata
    ├── exp3_flow_step_sensitivity.py — Per-denoising-step sensitivity (novel)
    ├── exp5_trajectory_attention.py — 50-rollout VLM attention capture + Easy/Hard classifier
    ├── exp5_reanalyze.py     — Leave-one-task-pair CV against task-identity leakage
    ├── exp6_attention_predicts_quant.py — Re-run rollouts under W4/W2 quant; regress attention → outcome
    ├── exp6_diagnostics.py   — Bootstrap CIs, Spearman + Bonferroni, RF/GB nonlinear comparison
    ├── exp7_per_frame_sensitivity.py — Per-call (obs, FP16, W4) capture for per-frame MSE
    ├── exp7_analyze.py       — Per-frame regression (n=1879, LOTP, ridge/RF, Spearman)
    ├── sis_utils.py          — SIS computation, FP16/W2 PrecisionController, layer-12-h2 attn hook
    ├── expB_sis_validation.py — ExpB: counterfactual SIS frame-selection test (7 conditions)
    └── run_all.py            — Orchestrator for overnight runs
```

## Phase 0 — LIBERO Rollout Infrastructure (2026-04-20)

Pivoted from single-frame attention analysis to **trajectory-level** attention dynamics, which requires closed-loop rollouts. Phase 0 builds that infrastructure and validates it reproduces pi0.5's published LIBERO success rates on a subset.

New pieces:
- `scripts/rollout.py` — in-process LIBERO rollout harness with callback seams for Phase 1's attention hooks. Lifted from openpi's `examples/libero/main.py` but restructured so quantization/attention experiments can monkey-patch the model in the same process.
- `scripts/exp0_rollout_reproduce.py` — 18-rollout FP16 reproduction check (3 tasks × 3 seeds × {Object, Long}). Emits a 4-table markdown summary (`results/exp0_rollout_tables.md`) with per-rollout detail, per-task success, per-suite vs QuantVLA-published, and error diagnostics.
- `scripts/setup_libero.sh` — idempotent LIBERO installer for the openpi venv.
- `scripts/run_phase0.sh` — single orchestrator: syncs scripts from the repo to `$EXPERIMENT_DIR`, installs LIBERO, runs both smoke tests (headless render + 1 rollout end-to-end), then the full 18-rollout sweep. EGL → osmesa → glx fallback for MuJoCo rendering.
- `scripts/utils.py` — added `MUJOCO_GL=egl` env default alongside the existing `TORCHDYNAMO_DISABLE` block.

### Running Phase 0
```bash
# On tambe-server-1, after git pull:
cd /data/subha2/quantization
tmux new -s phase0
bash scripts/run_phase0.sh
```

Result (2026-04-20): 18/18 FP16 rollouts succeeded, matching/exceeding QuantVLA published numbers on Object (99.0%→100%) and Long (93.5%→100%).

## Phase 1 — Trajectory Attention Analysis (2026-04-20)

`exp5_trajectory_attention.py`: 50 rollouts (5 tasks × 5 seeds × {Object, Long}) with forward-hook capture of all 45 VLM attention modules at every VLM call. Per (layer, head, call): sparsity, entropy, top-1/top-5 mass, attention-sink mass. Aggregated per rollout into static + dynamic features (1350 per rollout). Trained binary classifier Object vs Long.

**Result: AUC = 1.000** under leave-one-task-pair-out CV. Three confounds make this result tell us less than it looks: prompt grammar differs between suites, scenes have different visual complexity, and rollout lengths differ systematically. All three are mediated through vision_tower attention, not a difficulty-specific signal.

## Phase 2 — Attention vs Quantization Sensitivity (2026-04-20)

Two experiments testing whether the exp5 signal actually predicts which rollouts are quantization-sensitive:

`exp6_attention_predicts_quant.py`: re-run the 50 rollouts under quantization (W4 both, W2 VLM-protect) and regress FP16 attention features against steps_delta. Initial verdict "signal dead" was walked back after bootstrap CI analysis showed the 1350-feature, n=50 regime is too noisy to distinguish "attention beats suite" from "ties suite."

`exp7_per_frame_sensitivity.py`: capture (obs, FP16_chunk, W4_chunk) at each of ~1900 VLM calls across all 50 rollouts. Regress per-call attention features against per-call action MSE. **Finding: 32 of 90 attention features survive Bonferroni correction at p<0.05; signal localizes to language_model decoder layers; best R² = 0.125 within Object rollouts. Real but weak.**

Two distinct attention signals in two parts of the VLM:
- `vision_tower` → suite/scene fingerprinting (exp5 AUC=1.0 confound)
- `language_model` → per-frame W4 sensitivity (weak but statistically robust)

## Phase 2 follow-ups D1/D2/D3 (2026-04-21)

After a methodological critique that exp7's "null" verdict was overconfident, ran three follow-ups:

- **D1 — Stronger quantization (w2_vlm_protect): HIT.** Within-Object R² jumped from 0.125 (W4) to **0.333** (W2 with layer 0 + vision tower protected). Peak Spearman |ρ| rose from 0.17 to 0.29 (Bonferroni-p = 6.9e-36). Random forest now beats ridge — nonlinear structure matters at W2.
- **D2 — Per-head deep dive: HIT.** 225 of 2754 per-(layer, head, metric) features survive Bonferroni. Strongest signal: `language_model.layers.12.self_attn` head 2, where lower entropy (more concentrated attention) predicts higher W2 sensitivity. Same head appears in top 3 features across three metrics (entropy, top5, sparsity).
- **D3 — Decouple VLM vs expert target: FALSIFIED.** Decoupling hurt, not helped. VLM-only R² and expert-only R² are both strongly negative; attention features correlate with VLM-side and expert-side error equally (|ρ| ~0.18–0.20 for both), meaning they track a shared frame-level property, not a mechanism-specific signal.

**Revised story:** attention features **do** predict per-frame quantization sensitivity, but only clearly under aggressive quantization (W2+). Under mild W4 the signal is noise-limited. The W4 and W2 sensitivity patterns are uncorrelated (cross-config Spearman ~0.05), so attention isn't tracking universal difficulty — it's tracking config-specific sensitivity.

Paper headline remains exp2 (layer-sensitivity) + exp3 (step-asymmetric expert); exp7+D1/D2/D3 becomes a stronger supplementary section with a concrete mechanistic predictor (layer 12 head 2) rather than just a statistical note.

Full write-up with tables in `EXPERIMENT_FINDINGS.md`.

## Overnight Experiments (2026-04-15)

Three forward-pass-only experiments on a single H100, targeting pi0.5 LIBERO:

1. **Cross-Suite Activation Statistics** (~30 min) — Do hard-task observations produce different activation distributions? Tests the horizon-differential premise.
2. **Layer-wise Sensitivity Probe** (~2-4 hours) — Quantize each layer group individually to W4/W8, measure action MSE. Per-sample metadata enables post-hoc temporal analysis.
3. **Flow-Step Sensitivity** (~1-2 hours) — First-ever measurement of per-denoising-step quantization sensitivity for a VLA.

Findings from the first run are in `EXPERIMENT_FINDINGS.md`; plots in `plots/`.

### Running on GCP
```bash
export WORKSPACE=/path/to/local/ssd
bash scripts/setup_env.sh
cd $WORKSPACE/openpi
uv run python $EXPERIMENT_DIR/run_all.py
```

## Exp3 Redesign (2026-04-16)

The first exp3 run failed because `policy.infer()` samples fresh noise per call,
so FP16 reference and W4 test conditions started from different `x_t`. The
noise-induced variance (~0.05 MSE) swamped the quantization-induced error
(~0.005 MSE).

The redesigned `scripts/exp3_flow_step_sensitivity.py`:
- Seeds per-observation noise with `torch.Generator(seed=1000+i)` and passes
  it through `policy.infer(obs, noise=noise_np)` so reference and test runs
  start from identical `x_t` per observation.
- Monkey-patches `model.denoise_step` (not the expert module) to control
  weight swapping at the correct per-step granularity. The prefix pass does
  not call the gemma_expert, so the step counter is clean.
- Validates before running sweeps: with `quantize_steps=∅` the patched run
  must reproduce the FP16 reference within 1e-10 MSE; otherwise the script
  aborts instead of wasting compute.
- Streams per-config progress, running means, and a suite-split (easy/hard)
  summary at each k, with a final tabular summary of all three sweeps.

### Running exp3 on a shared server
```bash
# pick an idle GPU (check `nvidia-smi` first) and run unbuffered:
CUDA_VISIBLE_DEVICES=1 python -u scripts/exp3_flow_step_sensitivity.py \
  2>&1 | tee /data/<user>/experiments/results/exp3_stdout.log
```

## ExpB — Saliency-Aware PTQ Validation (2026-04-21)

After mixed evidence from D1/D2/D3 that attention features predict per-frame W2 sensitivity (within-Object R²=0.333), set up a decisive counterfactual test based on the SQIL paper's perturbation-based State Importance Score: SIS(s_t) = E_k ‖π(s_t) − π(φ(s_t,k))‖² where φ Gaussian-blurs an N×N grid cell. SQIL itself is QAT; we adapt the *detector* for PTQ-time precision selection — run W2-with-protection as the cheap default, override with FP16 at top-SIS frames.

Seven matched conditions on (suite, task_id, seed, episode_idx): pure W2, pure FP16, SIS-top-20 override, Random-20 (null), Oracle-20 (top-20% by ground-truth ‖a_FP−a_W2‖²), Bottom-SIS-20 (symmetry control), AttnEntropy-top-20 (cheap-proxy via D2 finding's layer-12 head-2 entropy).

Decision rule from a 20-seed Long-task pilot: if SR(SIS-top-20) ≤ SR(Random-20), kill the project; if SR(SIS-top-20) ≈ SR(Oracle-20) ≫ SR(Random-20), publish.

New code:
- `scripts/sis_utils.py` — `compute_sis()` with seeded flow-matching noise (otherwise noise variance dominates SIS), `PrecisionController` for O(1) FP16↔W2 weight-pointer swaps, `L12H2EntropyHook` for the attention-entropy probe.
- `scripts/expB_sis_validation.py` — diagnostic rollout (one pass collects SIS + attn entropy + oracle MSE per cycle), override rollout (replays seed with W2 base + FP16 at masked cycles), per-rollout 80th-percentile threshold, bootstrap-CI markdown summary.

### Running ExpB
```bash
# On tambe-server-1 after git pull:
cd /data/subha2/quantization
python -u scripts/expB_sis_validation.py --smoke    # 1 trial, all 7 conditions — verifies plumbing
python -u scripts/expB_sis_validation.py --pilot    # 20 trials × 5 conditions (kill switch)
python -u scripts/expB_sis_validation.py --full     # 100 trials × 7 conditions
python -u scripts/expB_sis_validation.py --analyze  # markdown summary from JSONLs
```

### ExpB results (50 Long + 50 Object × 8 conditions @ frac=0.5, 2026-04-22)

Full experiment took 6h 9min on a single H100 (batched SIS perturbations gave ~1.7× speedup on the diagnostic phase; sequential-vs-batched parity verified at |diff| = 1.08e-5).

**Headline: AttnEntropy beats every other detector by 14-29 pp.** The cheap online detector — entropy of `language_model.layers.12.self_attn` head 2 (D2 finding) — is essentially free at inference time and is the clear performance leader.

| Condition | success | 95% CI |
|---|---:|---|
| FP16 (ceiling) | 0.940 | [0.89, 0.98] |
| W2 (floor) | 0.000 | [0.00, 0.00] |
| **AttnEntropy** | **0.680** | **[0.59, 0.77]** |
| SIS-top | 0.490 | [0.39, 0.59] |
| MSE-W2traj | 0.480 | [0.38, 0.58] |
| Bottom-SIS | 0.420 | [0.33, 0.51] |
| Random | 0.390 | [0.30, 0.49] |
| MSE-FP16traj | 0.010 | [0.00, 0.03] |

**Three takeaways:**

1. **AttnEntropy is the deployable result.** Beats SIS by +19 pp at zero marginal cost. A precision-savings PTQ scheme can use l12h2 attention entropy as the per-frame gate without paying for perturbation passes.
2. **SIS works but only weakly, and not on Object.** SIS-top (0.49) > Random (0.39) overall, but per-suite: on Object SIS-top = Random = Bottom-SIS ≈ 0.58 (no signal), only on Long does SIS beat Random by +20 pp.
3. **MSE-FP16traj catastrophically failed (0.01).** Per-frame MSE rankings only transfer across trajectories that visit similar states — the W2-base override rollout doesn't visit the FP16 states the FP16-traj diagnostic ranked. Methodological caution.

Full writeup with per-suite tables, verdict revision (the auto-printed "STRONG" verdict was based on weak SIS vs Random, missed the AttnEntropy headline), and trajectory-divergence analysis in `EXPERIMENT_FINDINGS.md` → "Experiment B → Full experiment (2026-04-22)".

## Key References

- **QVLA** (ICLR 2026) — Action-centric channel-wise quantization for AR VLAs
- **QuantVLA** (arXiv 2602.20309) — Training-free PTQ for DiT-head VLAs (pi0.5 on LIBERO)
- **BlockDialect** (ICML 2025) — Block-wise fine-grained mixed format quantization with FP4 formatbook
- **DP-LLM** (NeurIPS 2025) — Dynamic layer-wise precision assignment at runtime
- **KVQuant** (NeurIPS 2024) — KV cache quantization with per-layer sensitivity-weighted NUQ
- **MicroMix** (2025) — Mixed-precision quantization with MXFP4/6/8 formats

Advisor: Wonsuk Jang (Stanford EE, BlockDialect author)
