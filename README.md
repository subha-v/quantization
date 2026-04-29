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
    ├── sis_utils.py          — SIS computation, FP16/W2/W4 PrecisionController, layer-12-h2 attn hook
    ├── expB_sis_validation.py — ExpB: 8 legacy conditions + 5 deployable Scheme 1/2 conditions (W2/W4/FP16 ternary)
    ├── expB_schemes_analyze.py — Bootstrap-CI analysis + matched-trial deltas for the deployable-schemes run
    ├── exp2_suite_split_table.py — Per-suite (Object vs Long) per-layer-group ExpB split into LaTeX
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

### ExpB headline (post-sweep, 2026-04-22)

Across all override fractions tested ({0.3, 0.4, 0.5}), **AttnEntropy beats every other detector — including the per-frame ground-truth-MSE oracle.** At frac=0.4 (60% W2 weights, 40% FP16 rescue): AttnEntropy 0.28 success vs MSE-W2traj 0.16, SIS-top 0.14, Random 0.06, AttnEntropy-flipped 0.01. The cheap online detector (entropy of `language_model.layers.12.self_attn` head 2, ~zero marginal cost since the VLM forward pass already runs) is the deployable answer — there is no method-quality / cost tradeoff.

Full per-frac × per-condition tables in `EXPERIMENT_FINDINGS.md` → "Baseline-expanded sweep". Below is the original frac=0.5 result that motivated the sweep.

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

## ExpB — Deployable Schemes (staged 2026-04-22, awaiting GPU)

Follow-up to ExpB's AttnEntropy headline. The original 68% rescue rate used l12h2 entropy captured during a forced FP16 forward pass inside the diagnostic — not a deployable signal in production. Two deployable variants now staged in `scripts/expB_sis_validation.py`:

- **Scheme 1 (one-frame-lag, `S1-Bin`/`S1-Tern`):** read W2-pass entropy from frame *t*, gate frame *t+1*'s precision; cycle 0 bootstraps to FP16. Zero extra compute.
- **Scheme 2 (speculative, `S2-Bin`/`S2-Tern`):** read W2-pass entropy from frame *t*'s cheap pass, optionally re-run *t* at higher precision. No lag; pays a bandwidth tax on escalations (reported via `condition_avg_bits` in the rollout JSONL).

Both schemes available at **binary {W2, FP16}** (frac=0.5) and **ternary {W2, W4, FP16}** (default 50/30/20 → ~5.4 avg bits, configurable via `--ternary-partition`). `Random-Tern` is the matched-fraction control. `PrecisionController` extended to cache multiple bit tiers simultaneously via `bits_list=(2, 4)`.

```bash
# When a GPU on tambe-server-1 frees up (always run nvidia-smi --query-compute-apps first):
cd /data/subha2/experiments
CUDA_VISIBLE_DEVICES=<idx> nohup /data/subha2/openpi/.venv/bin/python \
  /data/subha2/quantization/scripts/expB_sis_validation.py --schemes \
  --frac 0.5 --ternary-partition "0.2,0.3,0.5" \
  > results/expB_schemes_stdout.log 2>&1 &
# Post-hoc analysis:
/data/subha2/openpi/.venv/bin/python /data/subha2/quantization/scripts/expB_schemes_analyze.py
```

The analysis script evaluates H1–H5 from the plan: Scheme-1 viability, Scheme-2 vs Scheme-1 lag tax, ternary-vs-binary granularity gain, ternary direction symmetry control, and W2-pass-vs-FP16-pass entropy correlation.

## ExpB — W4-first Online Mixed-Precision (staged 2026-04-29)

Pivoting the deployment story from "W2 with FP16 rescue" to **"W4 default with intelligent {W2/W4/FP16} allocation"** — modern GPUs have native W4 tensor cores, while sub-4-bit needs custom kernels. The new plan is in `MEETING_5.md`; the implementation lives across `sis_utils.py` + `expB_sis_validation.py`.

### What's new in `sis_utils.py`
- `AttentionMetricHook(model, layer_idx, head_idx, metric)` — generalized attention probe at any (layer, head, metric ∈ {entropy, top1, top5, sparsity, sink}). Captures all five metrics per forward; `metric` selects the scalar `get_last()` returns. Replaces ad-hoc per-layer probes; the legacy `L12H2EntropyHook` is kept for back-compat.
- `PrecisionController.use_bits_range(start_layer, end_layer, target)` — per-layer-range pointer swap. Lets a forward hook on layer L swap layers L+1..17 to a different precision *mid-forward*, without touching layer 0 / vision tower / projector. Idempotent via per-layer state cache.
- `IntraPassController(model, ctrl, layer_L, head, metric, base_prec, decision_high_prec, decision_low_prec, direction, frac_high, frac_low)` — wraps `language_model.layers.{layer_L}.self_attn` so the metric is read on each forward, ranked against a per-rollout running quantile, and used to call `use_bits_range(L+1, 17, ...)` *during* the same forward pass. Three-tier supported (e.g., `decision_high="fp16"`, `decision_low="w2"` over a `base="w4"`).
- `parse_probe_tag` / `format_probe_tag` / `PROBE_DIRECTION_BY_TAG` — helpers for the 5 candidate readouts from MEETING_5's top-15 (l1h7-top1, l9h2-ent, l12h2-ent, l3h4-top5, l17h4-top1).

### What's new in `expB_sis_validation.py`
- 17 W4-base conditions added to `ALL_CONDITIONS`: `W4-Floor`, `W4-Static-Sched`, `Random-W4`, `AttnEntropy-W4`, `S1-Bin-W4`, `S2-Bin-W4`, `S1-Tern-W4`, `S2-Tern-W4`, `Random-Tern-W4`, three `S3-Bin-W4-*` (intra-pass at l1h7/l9h2/l12h2), `S3-Tern-W4-l12h2`, four `Probe-W4-*` (lag-1 candidate-readout sweep).
- `diagnostic_rollout_v3` — captures W4-pass attention metrics for ALL 5 candidate probes per cycle (writes nested `attn_probes_w4` and `attn_probes_fp16` dicts to `expB_diagnostic_v3.jsonl`).
- `intrapass_rollout` — replays a (suite, task_id, seed, episode_idx) trial with intra-pass online precision control via `IntraPassController`.
- `build_masks_w4` — reads V3 diagnostic, builds W4-base mask for each non-S3 condition.
- `analyze_w4` — bootstrap-CI summary table + HW0–HW7 hypothesis matrix (matched-pair signed deltas) + HW7 D2-W4 transfer Spearman ρ across all 5 probes.
- New CLI modes: `--w4-tier0` (W4 baseline only, 3 conditions × 100 trials, ~1h), `--w4-schemes` (full W4 plan, 17 conditions × 100 trials, ~7h), `--analyze-w4` (post-hoc summary).

### Running the W4-first overnight
```bash
# On tambe-server-1 (after git pull + GPU pin):
cd /data/subha2/experiments
# Smoke (~30 min):
CUDA_VISIBLE_DEVICES=<idx> python /data/subha2/quantization/scripts/expB_sis_validation.py \
    --smoke --conditions W4-Floor S1-Bin-W4 S3-Bin-W4-l1h7-top1 S3-Bin-W4-l9h2-ent \
                          S3-Bin-W4-l12h2-ent Probe-W4-l1h7-top1 \
    --candidate-readouts l1h7-top1 l9h2-ent l12h2-ent l3h4-top5 l17h4-top1 --verbose
# Tier 0 — W4 baseline (~1 h):
CUDA_VISIBLE_DEVICES=<idx> nohup python /data/subha2/quantization/scripts/expB_sis_validation.py \
    --w4-tier0 --verbose > logs/tier0_w4baseline.log 2>&1 &
# Tier 1+2+3 — full W4 plan (~7 h):
CUDA_VISIBLE_DEVICES=<idx> nohup python /data/subha2/quantization/scripts/expB_sis_validation.py \
    --w4-schemes --frac 0.4 --ternary-partition "0.1,0.4,0.5" --verbose \
    > logs/w4_overnight.log 2>&1 &
# Analysis:
python /data/subha2/quantization/scripts/expB_sis_validation.py --analyze-w4
```

The hypothesis matrix tests:
- HW0: SR(W4-Floor) − SR(FP16) — does the FP16 rescue framing apply at W4?
- HW2: SR(S3-Bin-W4-l12h2) − SR(S1-Bin-W4) — does intra-pass beat lag-1?
- HW3a: SR(S3-Bin-W4-l1h7-top1) − SR(S3-Bin-W4-l12h2-ent) — viable cheap-pass at the earliest layer?
- HW4: SR(S1-Tern-W4) − SR(W4-Floor) at avg_bits < 4 — sub-W4 average beats uniform W4?
- HW7: per-trial Spearman ρ(l12h2-ent W4-pass, ‖a_FP − a_W4‖²) — does D2 transfer to W4?

### W4-first results — completed 2026-04-29 (n=100, 50 Long + 50 Object × 21 conditions = 2100 rollouts)

Headline: **`S3-Tern-W4-l12h2` Pareto-dominates uniform W4** at 95.0% SR / 3.58 avg bits vs W4-Floor's 94.0% / 4.00 bits. The mechanism is **layer-restricted intra-pass W2 demotion** (only layers 13-17 are demoted; layers 1-12 stay at W4 base) gated per-cycle by an l12h2 attention entropy running quantile.

| Condition | SR | avg bits | Note |
|---|---:|---:|---|
| FP16 (ceiling) | 0.940 | 16.00 | reference |
| W4-Floor | 0.940 | 4.00 | reference |
| **S3-Tern-W4-l12h2** | **0.950** | **3.58** | **Pareto winner** |
| S3-Bin-W4-l12h2-ent | 0.970 | 4.42 | best binary at <5 bits |
| S3-Bin-W4-l9h2-ent | 0.980 | 5.05 | strongest binary signal |
| S1-Tern-W4 (full-pass W2 demotion) | 0.790 | 4.57 | spatial collapse: -15 pp |
| S2-Tern-W4 (full-pass W2 demotion) | 0.740 | 4.19 | spatial collapse: -20 pp |
| Random-Tern-W4 | 0.790 | 4.19 | random demotion also collapses |

**Three findings**:
1. **Layer-restricted intra-pass demotion** (S3-Tern only swaps layers 13-17 to W2) achieves sub-W4 avg bits at matched/better SR.
2. **Full-pass W2 demotion (layers 1-17) is fundamentally broken at W4** — random, bottom-direction, and top-direction targeting all fail (-15 to -30 pp). Mid-VLM layers (1-12) do not tolerate W2 even on individual cycles.
3. **The D2 direction is FLIPPED at W4 vs W2**: l12h2-ent ρ = +0.115 at W4 (vs -0.294 at W2). But the directional effect is small (+1-3 pp) compared to the spatial effect.

Full writeup with HW0–HW10e hypothesis matrix and per-suite splits in `EXPERIMENT_FINDINGS.md` → "Experiment C — W4-first online mixed-precision".

Implementation: branch `overnight-2026-04-29-w4-first` (commits c7f0414, 325c65b, d876110, ad6867e). New code in `scripts/sis_utils.py` (`AttentionMetricHook`, `PrecisionController.use_bits_range`, `IntraPassController`) and `scripts/expB_sis_validation.py` (V3 diagnostic, intra-pass driver, 21 W4-base conditions). Results in `results/expB_w4_summary.md`, `results/expB_w4_rollouts.jsonl`, `results/expB_diagnostic_v3.jsonl`.

## LIBERO-PRO at W4 (staged 2026-04-29)

ExpC's W4 conditions tied FP16 at 94% combined SR on standard LIBERO — there's no rescue gap for AttnEntropy to close at W4. Per the LIBERO-PRO paper (arXiv:2510.03827, github.com/Zxy-MLlab/LIBERO-PRO), most VLAs memorize standard LIBERO; the benchmark needs perturbation to surface real generalization. Plan: swap the benchmark, keep the model and quantization stack identical, and re-run a focused 7-condition expC subset at the LIBERO-PRO position-perturbation operating point closest to FP16 ≈ 70% — the regime where W4 has headroom to degrade and AttnEntropy has signal to detect.

### What's new

- `scripts/setup_libero_pro.sh` — idempotent remote setup. Clones Zxy-MLlab/LIBERO-PRO, overlays its updated `benchmark/__init__.py` + `libero_suite_task_map.py` into the openpi LIBERO checkout (registers `libero_<suite>_temp` task suites), and downloads the per-(suite, axis, magnitude) bddl + init bundles from the HuggingFace dataset `zhouxueyang/LIBERO-Pro` into `bddl_files/` and `init_files/`. Falls back to the cloned repo's bundled files if HuggingFace is unavailable.
- `scripts/rollout.py` — added `set_libero_pro_config()` / `parse_pro_config_str()` / `stage_libero_pro_files()` (process-safe, idempotent, fcntl-locked). When a Pro config is active for a suite, `make_libero_env()` routes to that suite's `_temp` variant after staging the right bundle. Unflagged invocations are byte-identical to the prior harness. New `--pro-config "Suite:axis:mag ..."` CLI flag.
- `scripts/find_operating_point.py` — Step 2 driver. Sweeps configurable suites × magnitudes × N FP16 trials per cell. Writes `results/libero_pro_operating_point.md` with bootstrap CIs and a `D*` recommendation per suite.
- `scripts/expB_sis_validation.py` — added `--pro-config`, `--out-tag`, `--w4-pro`, `--w2-pro`. The W4 mode runs the 5 W4-base conditions (`FP16`, `W4-Floor`, `AttnEntropy-W4`, `Random-W4`, `S3-Tern-W4-l12h2`); the W2 mode runs `W2` + `AttnEntropy` for sanity. Output paths get the `--out-tag` suffix so PRO results don't clobber the existing `expB_w4_*` files.

### Running the LIBERO-PRO experiment
```bash
# Remote tambe-server-1 setup (one-time):
bash /data/subha2/quantization/scripts/setup_libero_pro.sh

# Step 1 — integration smoke (15 trials, ~5 min):
CUDA_VISIBLE_DEVICES=<idx> MUJOCO_GL=egl uv run python \
    /data/subha2/quantization/scripts/rollout.py \
    --single-rollout --suite Object --task-id 20 --seed 0 \
    --pro-config "Object:x:0.2"

# Step 2 — operating point sweep (Object + Goal × {0.1, 0.2, 0.3}, 50 trials/cell, ~6h):
CUDA_VISIBLE_DEVICES=<idx> uv run python \
    /data/subha2/quantization/scripts/find_operating_point.py \
    --suites Object Goal --magnitudes 0.1 0.2 0.3 --axis x --n-per-cell 50

# Step 3 — focused expC at chosen D* per suite (5 W4 conditions × 100 trials, ~6-8h):
CUDA_VISIBLE_DEVICES=<idx> uv run python \
    /data/subha2/quantization/scripts/expB_sis_validation.py \
    --w4-pro --pro-config "Object:x:<D*_Object> Goal:x:<D*_Goal>" \
    --out-tag libero_pro --frac 0.5 --ternary-partition "0.1,0.4,0.5"

# Optional W2 sanity (W2-Floor + AttnEntropy at W2; 2 conditions × 100 trials):
CUDA_VISIBLE_DEVICES=<idx> uv run python \
    /data/subha2/quantization/scripts/expB_sis_validation.py \
    --w2-pro --pro-config "Object:x:<D*_Object> Goal:x:<D*_Goal>" \
    --out-tag libero_pro --frac 0.5
```

The decision points are documented in the plan: W4-Floor SR < FP16 SR by ≥ 10 pp confirms W4 degrades on harder tasks (the headline hypothesis); AttnEntropy-W4 SR > Random-W4 SR with non-overlapping bootstrap CIs confirms the rescue gate works at W4 in this regime.

## Key References

- **QVLA** (ICLR 2026) — Action-centric channel-wise quantization for AR VLAs
- **QuantVLA** (arXiv 2602.20309) — Training-free PTQ for DiT-head VLAs (pi0.5 on LIBERO)
- **BlockDialect** (ICML 2025) — Block-wise fine-grained mixed format quantization with FP4 formatbook
- **DP-LLM** (NeurIPS 2025) — Dynamic layer-wise precision assignment at runtime
- **KVQuant** (NeurIPS 2024) — KV cache quantization with per-layer sensitivity-weighted NUQ
- **MicroMix** (2025) — Mixed-precision quantization with MXFP4/6/8 formats

Advisor: Wonsuk Jang (Stanford EE, BlockDialect author)
