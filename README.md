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

## Path 1 — Static vs Dynamic Late-Layer W2 Comparison (staged 2026-05-06)

The S3-Tern-W4-l12h2 Pareto claim (95% SR / 3.58 avg bits vs W4-Floor's 94% / 4.0) has a methodological hole: it was never compared to a **static late-layer W2 baseline** (layers 13-17 always W2, no gating). If a static schedule matches S3-Tern at the same parameter-weighted bits, the AttnEntropy gating contributes nothing — the gain is purely the spatial restriction already established by exp2. This experiment closes that gap.

### Eight new conditions in `scripts/expB_sis_validation.py`

All operate on the late layer range = PaliGemma decoder layers 13-17 (5 layers). Layers 1-12 stay W4; layer 0 + vision tower stay FP16-protected (PrecisionController default).

| Condition | Mechanism |
|---|---|
| `Static-W2-l13-17` | Layers 13-17 W2 every cycle, every trial. Trial-level install via `PrecisionController.use_bits_range(13, 17, 2)`. No per-cycle swap. |
| `Random-Bin-W2-l13-17-{f25,f50,f75}` | On a random fraction of cycles (seeded), layers 13-17 → W2; rest → W4. Three operating points. |
| `AttnEnt-Bin-W2-l13-17-{f25,f50,f75}` | On bottom-frac% by W4-pass `l12h2` entropy, layers 13-17 → W2. Same direction as S3-Tern (W2 D2 default). |
| `Hybrid-Static-W2-AttnEnt-FP16-f50` | Layers 13-17 W2 base + bottom-50% l12h2-ent cycles escalated to FP16 on layers 13-17. Tests whether dynamic adds value on top of static. |

### What's new in the script

- `L_LATE_START = 13`, `L_LATE_END = 17`, `LATE_LAYER_FRACS = (0.25, 0.5, 0.75)`, `LATE_LAYER_W2_CONDITIONS` constants.
- `_layer_param_counts(model)` and `_avg_bits_param_weighted(...)` — parameter-weighted average-bits accounting (formula: per-cycle weighted mean of P_protected·16 + P_early·bits(early) + P_late·bits(late) / P_total). Reported alongside the existing cycle-weighted `condition_avg_bits` so all conditions sit on a comparable Pareto axis.
- `override_rollout(...)` extended with `base_layer_range_install`, `late_range_per_cycle`, `late_range`, `early_pin` kwargs. Backward-compatible defaults (existing per-cycle whole-model schedules behave identically).
- `build_masks_w4(...)` builds Random-Bin and AttnEnt-Bin schedules using a new `{"_kind": "late_range", ...}` schedule shape.
- `run_seed_w4(...)` dispatches Static, Hybrid, and the late_range schedule shape to the new `override_rollout` path.
- `analyze_w4(...)` summary table now reports `avg bits (param-weighted)` alongside cycle-weighted bits.
- 8 new HW tags (HW12a-h) in the matched-pair hypothesis matrix covering the static-vs-dynamic comparisons.

### Run order
1. **n=20 pilot** (Static-W2-l13-17 only, ~1 h) — confirms Static doesn't catastrophically collapse before committing the full compute budget.
2. **Standard LIBERO n=100, 8 conditions** (~3.5 h, ~800 rollouts) — the Pareto-sweep comparison.
3. **LIBERO-PRO Object x0.2 n=200** (conditional on Phase 2 results, ~7 h).

```bash
# n=20 pilot:
CUDA_VISIBLE_DEVICES=<idx> uv run python scripts/expB_sis_validation.py \
    --w4-schemes --conditions Static-W2-l13-17 \
    --trial-range 0:20 --reuse-diag --out-tag static_pilot_n20

# Full Pareto sweep (8 conditions × 100 trials = 800 rollouts):
CUDA_VISIBLE_DEVICES=<idx> uv run python scripts/expB_sis_validation.py \
    --w4-schemes --conditions \
        Static-W2-l13-17 \
        Random-Bin-W2-l13-17-f25 Random-Bin-W2-l13-17-f50 Random-Bin-W2-l13-17-f75 \
        AttnEnt-Bin-W2-l13-17-f25 AttnEnt-Bin-W2-l13-17-f50 AttnEnt-Bin-W2-l13-17-f75 \
        Hybrid-Static-W2-AttnEnt-FP16-f50 \
    --reuse-diag --out-tag static_dynamic_n100
```

Plan in `/Users/subha/.claude/plans/let-s-go-ahead-and-snoopy-lighthouse.md`. Implementation in commits on branch `overnight-2026-04-29-w4-first`.

### Pilot result (2026-05-06, n=20 Long task 0-3)

| Condition | n | SR | 95% CI | avg bits (cycle) | avg bits (param) |
|---|---:|---:|---|---:|---:|
| Static-W2-l13-17 | 20 | **1.000** | [1.000, 1.000] | 9.48 | 9.48 |

Static late-layer W2 demotion does not collapse on standard LIBERO Long at n=20 — confirms the pilot acceptance criterion (SR ≥ 80%) and clears the way for the full n=100 Pareto sweep. The param-weighted bits (~9.48) are dominated by FP16-protected components (vision tower + action expert), not the weight-quantized PaliGemma decoder, which is why the cycle-weighted "4.0 → 3.4" headline differs from the param-weighted axis.

Pilot output: `results/expB_w4__static_pilot_n20_{rollouts.jsonl,summary.md}`. Phase 2 (full 8-condition sweep) pending GPU availability on tambe-server-1.

## Qwen2.5-VL × MM-NIAH — Experiment P: Query-Adaptive Page-Format Routing (VLM-FormatBook diagnostic, 2026-05-12)

After the static-KV story for Qwen2.5-VL × LongVideoBench MCQ was effectively exhausted (F4 deployable, F9 zero-loss, J12 engineering Pareto; J7 retracted, K9 trending), Exp P pivots from *"how many bits per element?"* to *"which pages does this query need to read at all?"* This is the Quest/PRISM-style query-adaptive page-routing direction, combined with per-page format selection (FormatBook). Structurally orthogonal to F4/F9/J12 — those decide encoding; Quest/FormatBook decides access and format-per-page.

**Benchmark switch.** LongVideoBench MCQ is text-anchored on the answer query (Exps D1/E1 falsified visual-K routing), so it's the wrong stress test. P0 uses **MM-NIAH** (OpenGVLab/MM-NIAH, NeurIPS 2024) retrieval-image, which has a well-defined visual needle page that varies by query.

### Primary conditions (n=200 stratified by context-length bucket)

| Code | Description | Active K | Cold K | Attention | Page budget |
|---|---|---|---|---|---|
| P0 | BF16 baseline | BF16 | BF16 | dense | — |
| P1 | Full F4 KIVI | F4 | F4 | dense | — (MM-NIAH F4 anchor) |
| P2 | Full F9 (F4 + top-16 BF16 outliers) | F9 | F9 | dense | — (dense anchor, clean: BF16 sidecode) |
| P3 | **Quest sparse** | F9 | masked (-inf) | sparse, last-Q row only | **top-25% visual + all text** |
| P4 | **Random sparse** (seeded per item) | F9 | masked (-inf) | sparse, last-Q row only | top-25% visual + all text |
| P5 | **Oracle sparse** (budget-matched: needle + top-(K-1) Quest) | F9 | masked (-inf) | sparse, last-Q row only | top-25% (= P3 budget, but needle forced into active) |
| P6 | **FormatBook** (Quest-routed F9, cold F4) | F9 | F4 (per-page) | dense | top-50% visual = F9 |

Stretch: P3b/P4b at top-50% budget; **P2b** = F9+INT8 sidecode (J12) dense, characterizes the INT8-sidecode confound separately so primary P2-P6 stay clean on F9/BF16-sidecode.

**Note on absolute accuracy.** F9 calibration is reused from the LongVideoBench seed=0 NPZ (out-of-distribution on MM-NIAH); a follow-up should recalibrate F9 on MM-NIAH before claiming absolute accuracy numbers. Paired routing comparisons (P3 vs P4, P5 vs P3, P6 vs P2) remain valid since they share the same calibration and quantizer family. `logical_page_read_fraction` is the *implied* sparsity per the routing decision, not measured bandwidth — both sparse and FormatBook routes still run dense SDPA underneath, with cold pages masked/downgraded only at the last query row.

### Key implementation details

- **Prefill-not-decode masking.** With `max_new_tokens=1`, first-token logits come from the prefill forward at the last prompt position — there is *no* length-1 decode step before the scored token. The SDPA wrapper therefore patches `torch.nn.functional.scaled_dot_product_attention` *during prefill* and writes -inf only into the last query row's cold-page columns. All other query rows attend normally, preserving correct K/V values for upstream layers.
- **Per-page envelopes captured during prefill.** Inside `PageAwareFakeQuantKVCache.update()`, after K is quantized, we compute (k_min, k_max) per (KV head, channel, page) and stash them as `most_recent_envelope` for the SDPA wrapper to read within the same layer's attention forward (no global counter).
- **GQA aggregation.** 28 Q heads share 4 KV heads (group of 7). Quest scores are aggregated across the 7 Q-heads sharing each KV-head via sum (default; smoke test compares vs max).
- **FormatBook downgrade.** P6 doesn't mask attention — it re-quantizes cold-page K rows in place from J12 to F4 inside the SDPA wrapper, then runs dense SDPA. Tests "noisier K for low-attention pages" vs P3's "skip the page entirely."

### What's new in `qwen/scripts/`

- `mm_niah_loader.py` — MM-NIAH retrieval-image loader: dataclass, stratified split by `context_length` bucket (short/mid/long), interleaved-image Qwen2.5-VL chat formatting, needle-page identification via `find-image`/`abnormal_pic` path heuristic (validated on 520/520 val records).
- `page_layout.py` — `find_all_visual_spans` (multi-image extension of `find_visual_token_span`) + `build_page_layout` that builds the per-item page table (text + in-context-image + choice-image roles).
- `quest_scorer.py` — per-page (k_min, k_max) envelope computation + Quest upper-bound score `sum_d max(q*k_min, q*k_max)` against the last query row + routing decision builder.
- `page_envelope_cache.py` — `PageAwareFakeQuantKVCache(FakeQuantKVCache)`: overrides `update()` to capture envelopes and write `most_recent_envelope` / `most_recent_layer_idx` for the SDPA wrapper.
- `attention_router.py` — `page_routing_sdpa_context` context manager: monkey-patches `F.scaled_dot_product_attention` with a wrapper that (a) writes -inf into the last query row's cold-page columns for sparse routes, or (b) re-quantizes cold-page K in place for FormatBook. Last-row trick preserves non-last K contributions for upstream layers.
- `expP_smoke.py` — n=5 short-bucket smoke. Asserts: page coverage, envelope well-formedness, layer-index sync, **prefill-mask changes logits** (load-bearing), oracle needle-hit, cooperative pass-through.
- `expP_driver.py` — sweep 7 primary + 2 stretch conditions on stratified n=200 MM-NIAH items. Periodic summary callback; per-item routing diagnostics.
- `expP_analyze.py` — accuracy ± bootstrap CI, McNemar paired pairs (P1/P0, P3/P4, P5/P3, P6/P2, P6/P1), verdict matrix.

### Pass/fail question

> Can query-aware page selection find the visual needle page better than random, AND can FormatBook match full J12 accuracy while reading fewer J12-format pages?

A pass: any of (1) Quest needle-hit > random at top-25%, (2) P3 acc > P4 acc (McNemar p<0.05), (3) P6 acc ≥ P2 acc CI at ≤50% sidecode-read, (4) P5 >> P3 ≈ P4 (benchmark has signal, scorer needs work). A clear fail: Quest needle-hit ≈ random AND P3 ≈ P4 AND P5 ≈ P3.

### Running on tambe-server-1

```bash
ssh subha2@tambe-server-1
cd /data/subha2/quantization && git pull
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv
# MM-NIAH val is at /data/subha2/mm_niah/mm_niah_val/{annotations/retrieval-image.jsonl, images/}
CUDA_VISIBLE_DEVICES=<idx> python3 qwen/scripts/expP_smoke.py --bucket short --n-items 5
CUDA_VISIBLE_DEVICES=<idx> python3 qwen/scripts/expP_driver.py --n-items 200
python3 qwen/scripts/expP_analyze.py  # no GPU
```

Plan: `/Users/subha/.claude/plans/look-through-the-qwen-crispy-curry.md`.

## Qwen2.5-VL × LongVideoBench — Experiment G: Frame-scaling under fixed KV memory budget (staged 2026-05-09)

After Exp F (KIVI per-channel-along-seq) closed 94.4% of the KV-quantization collapse at TRUE 4.00 KV bits, the open research question moves up one level: **what does 4-bit KV buy us for long-video VLM inference?** The most VLM-specific answer is *more visual evidence under the same KV memory budget*. The theoretical math at fixed `max_pixels=360×420`:

```
relative KV memory = (frames × avg_kv_bits) / (64 × 16)
```

so 256-frame F4 (≈4 KV bits) ≈ 64-frame BF16 KV memory at 4× temporal coverage. **The headline test is whether `G4` (256f F4) beats `G0` (64f BF16) on LongVideoBench.** If it does, the contribution shifts from "we applied KIVI" to "correct KV quantization changes the long-video operating point."

### Stage-1 conditions (n=64 balanced 16/bucket, 9 conditions)

Reuses F-suite F0 (BF16), F4 (KIVI per-channel-seq INT4), F9 (KIVI + top-16 outlier BF16). Same n=64 stratified subset as F-suite Stage 1 so the F4 anchor is paired exactly.

| ID | Frames | KV | rel KV mem | Purpose |
|---|---:|---|---:|---|
| G0 | 64 | BF16 | 1.00× | Baseline ceiling (re-anchor to A1=0.565) |
| G1 | 64 | F4 INT4 | 0.25× | Re-anchor to F4=0.545; cascade & type-adaptive substrate |
| G2 | 128 | BF16 | 2.00× | Upper-bound for what extra frames buy |
| G3 | 128 | F4 INT4 | 0.50× | Memory-saving point: 2× frames at half KV mem |
| **G4** | **256** | **F4 INT4** | **1.00×** | **Matched-memory headline: 4× frames at G0 budget** |
| G5 | 128 | F9 (4.75 bits) | 0.59× | Zero-loss point at 2× frames |
| G6 | 256 | F9 (4.75 bits) | 1.19× | Zero-loss point at 4× frames |
| G7 | 64↗256 cascade | F4 INT4 | ≈1.00× (target avg=128) | Spend frames only on low-margin items |
| G8 | type-adaptive 64/128/256 | F4 INT4 | ≈0.50× (target avg=128) | Detail/temporal/count/OCR → 256f; detail/action → 128f; other → 64f |

### Compute structure (drives ~3.5 h Stage-1 wall vs naive ~8.4 h)

The visual prefill (image-token construction, position-id generation, find_text_slice_spans, inputs dict) is the heavy work and is **identical across K-quantizer configs at the same frame count**. The driver iterates outer over `frames ∈ {64, 128, 256}`, inner over the K-quantizer configs at that frame count, sharing the prefill across F0/F4/F9 conditions. G7/G8 are pure JSONL post-processes (no new forwards).

**Calibration reuse.** F-suite calibration NPZ at frames=64 is reused as-is. K-channel outlier indices are post-RoPE and frame-count-independent in theory; smoke check 8 (opt-in via `EXPG_RIGOR_HIGH=1`) verifies on 8 cal items at frames=256 with Jaccard ≥ 0.75 per (L, H_kv).

### What's new in `qwen/scripts/`

- `expG_frame_scaling.py` — main driver; outer loop over frame counts {64,128,256}, inner loop over K-quantizer configs (F0/F4/F9). Reuses F-suite `_run_condition_forward`, `_option_logprobs_and_pred`, `_compute_three_bit_columns`, `ensure_stage_split`, `backfill_bf16_join` verbatim. Adds `relative_kv_memory` to each row.
- `expG_smoke.py` — 8 hard assertions: F4 dispatch at T∈{5760, 11520, 23040}; F9 calibration loadable; n=64 split present; cascade margin definition (max−second_max ≡ answer_margin@argmax); qtype classifier coverage on cal-100 (weighted_avg_frames ∈ [110, 145]); `nvidia-smi` memory feasibility for the 256f tier; G0 anchor sanity (acc bounds 0.45–0.60 to catch the Exp-D fast-processor regression); optional 8-item recalibration at frames=256 to verify outlier-index Jaccard.
- `expG_cascade.py` — G7 confidence cascade post-process. Loads G1 (64f F4) + G4 (256f F4) rows, computes `cascade_margin = max(option_logprobs) − second_max(...)` per G1 row, picks bottom-third by margin to substitute G4 rows for. For target_avg=128, rerun fraction = (128−64)/(256−64) = 1/3 exactly. Pure JSONL stitch; zero new forwards.
- `expG_type_adaptive.py` — G8 question-type adaptive post-process. Routes each item to G1/G3/G4 by `classify_question_type(question)` and `BUDGET_MAP`. Pure JSONL stitch; zero new forwards.
- `question_type_classifier.py` — keyword heuristic with 6 labels (count/temporal/ocr/detail/action/other) → BUDGET_MAP {64, 128, 256}. Tunable on cal-100 split (disjoint from any stage-N eval set per `make_split` semantics).
- `expG_analyze.py` — per-condition table with 95% bootstrap CI; **paired McNemar** for headline pairs (G4↔G0, G3↔G0, G6↔G2, G7↔G1, G8↔G1, G4↔G2); frame-budget vs accuracy frontier (sorted by `relative_kv_memory`); verdict matrix with G-specific promotion rules.
- `run_expG.sh` — orchestrator with subcommands `smoke|stage1|stage3|cascade|analyze|full`. Env vars: `PIPELINE_MODEL`, `EXPG_N_LIMIT`, `EXPG_MIN_FREE_GB=70`, `EXPG_RIGOR_HIGH`, `CUDA_VISIBLE_DEVICES`.

### Promotion rules (Stage 1 → Stage 3)

```
G4 (256f F4) ≥ G0 + 3 pp                        → promote_paper_strong  [HEADLINE]
G4 ∈ [G0 − 3 pp, G0 + 3 pp]                     → promote_n200          (matched-memory)
G3 (128f F4) ≥ G0                               → promote_n200          (memory-saving)
G6 (256f F9) ≥ G2 − 1 pp                        → promote_n200          (zero-loss at 4×)
adaptive (G7 or G8) ≥ best fixed F4 + 2 pp      → promote_n200
G4 < G0 − 5 pp                                  → kill                  (frame coverage doesn't help)
```

### Running on tambe-server-1

```bash
ssh subha2@tambe-server-1
cd /data/subha2/quantization && git pull
source /data/subha2/experiments/qwen_venv/bin/activate
nvidia-smi  # confirm GPU 0 is free; co-tenant typically holds GPU 1
tmux new -s qwen-expG
CUDA_VISIBLE_DEVICES=<idx> bash qwen/scripts/run_expG.sh smoke    # ~30s (no model fwd) or ~5min (with --model)
CUDA_VISIBLE_DEVICES=<idx> bash qwen/scripts/run_expG.sh stage1   # ~3.5h: G0/G1 (64f) → G2/G3/G5 (128f) → G4/G6 (256f)
CUDA_VISIBLE_DEVICES=<idx> bash qwen/scripts/run_expG.sh cascade  # ~1 min post-process: G7 + G8
bash qwen/scripts/run_expG.sh analyze                              # no GPU
```

After Stage 1: promote 4–6 conditions to Stage 3 (n=200 canonical) per the verdict matrix; expect ~11 h Stage-3 wall.

Plan in `/Users/subha/.claude/plans/read-through-the-qwen-experiments-md-ancient-shell.md`.

## Qwen2.5-VL × LongVideoBench — Experiment F: K-quantizer repair screening (2026-05-09, COMPLETE)

**Headline: the 35.5 pp KV-quantization collapse from Exp A is *solved* by switching the K scale axis.** F4 KIVI per-channel-along-seq closes **94.4% of the F1→F0 gap (33.5 of 35.5 pp) at TRUE 4.00 KV bits**: F4 = 0.545 vs F0 BF16 ceiling 0.565, F1 INT4 floor 0.210. The fix is a one-line scale-axis change from per-(head, position, group of 128 head_dim slots) to per-(head, channel) shared across all positions — no calibration, no routing, no slice info. **None of the routing experiments (Exp B, D1, E1) were necessary**; the bottleneck was always the quantizer, not the selector.

### Stage 3 Pareto frontier (n=200 canonical eval split)

| KV bits | acc | condition | note |
|---:|---:|---|---|
| 4.000 | **0.545** | F4 KIVI per-channel-seq | **deployable headline** (4× compression, ~2 pp loss) |
| 4.375 | 0.540 | F8 outlier-8 | dominated by F4 |
| 4.750 | **0.560** | F9 outlier-16 | **zero-loss Pareto point** (within CI of F0) |
| 10.000 | 0.550 | F3 all-K BF16 + V INT4 | dominated by F4 |
| 16.000 | 0.565 | F0 BF16 (ceiling) | reference |

VLM-specific scaling refinements (F5 text/visual split=0.510, F6 prompt-role split=0.525, F7 99.5%ile clip=0.540) all UNDERPERFORM F4 at the same 4-bit budget. Once the K axis is right, modality/role-aware quantization adds nothing. Score-cal closed-form variants (F10–F13) failed catastrophically (0.172–0.281); iterative scale search would be needed to revisit. Full writeup in `QWEN_EXPERIMENTS.md` → "Experiment F".

### Files of record

- `qwen/scripts/k_quantizers.py` — `KQuantizerConfig` dataclass + 12 quantizer kinds + `apply_k_quantizer` dispatch.
- `qwen/scripts/expF_calibrate.py` — single-pass cal-100 capture; per-(L, H_kv, channel) `k_channel_energy` + outlier indices + `q_energy{,_text,_visual}` via forward hooks on each layer's `q_proj`.
- `qwen/scripts/expF_smoke.py` — Phase A (synthetic-tensor) + Phase B (live-model logits-differ) smoke checks.
- `qwen/scripts/expF_kquant_screen.py` — tiered driver (Stages 0/1/2/3); `_compute_three_bit_columns` for K/V/KV avg-bit accounting.
- `qwen/scripts/expF_analyze.py` — per-condition table + verdict matrix.
- `qwen/scripts/run_expF.sh smoke|calib|stage{0,1,2,3}|analyze|full`.

## Qwen2.5-VL × LongVideoBench — Experiment E1: Text-K Slice Ablation (2026-05-08, COMPLETE)

**Headline: text-K-routing hypothesis ALSO falsified.** After D1 showed text-K is the dominant fragility (vs visual-K), E1 asked *which* of the ~140 text-K positions carry the fragility. 200 items × 11 V3K text-K-mask conditions (V always INT4, visual-K always INT4, only text-K varied):

- **Best single slice** is `E1.4 OptionsOnly` (40 tokens, **0.290** acc), recovering only **46% of D1.3's 17.5 pp text-K rescue**. All other slices land at or below the floor (0.210).
- **Adding more slices to the union HURTS.** `E1.7 (Options + AnsPrefix, 45 tok) − E1.4 (Options alone, 40 tok) = −10.5 pp` despite 5 *more* BF16 tokens. The K-side is sensitive to *which positions are at which precision relative to each other*, not just the BF16 fraction.
- **K-residual selection is *worse* than random.** E1.10 (top-20 text positions by per-position INT4 K-row residual norm) at 0.200 < E1.9 random-20 (mean 0.215 over 3 seeds) < E1.0 floor 0.210. Quantization difficulty is the *opposite* of the right signal.
- **Verdict matrix: NO condition is sufficient** (≥80% of E1.1 = D1.3's 0.385 acc at <50% of E1.1's 140 tokens). Every fixed slice, union, random, and K-residual condition fails.

**Combined D1 + E1 implication.** Routing within K alone — visual-K windows OR text-K subsets — is *ruled out* as a research direction for first-token MCQ scoring on Qwen2.5-VL. The signal that matters is whether K is at BF16 or INT4 *in aggregate*; sub-divisions don't compose monotonically. The actionable next direction is **K-side outlier-aware INT4 quantization itself** (KIVI per-channel K calibration, AKVQ-VL static K outlier extraction, post-RoPE per-channel K scale) — fix the quantizer, don't route around it. Full writeup in `QWEN_EXPERIMENTS.md` → "Experiment E1".

### E1 implementation notes

- `qwen/scripts/text_slices.py` — `find_text_slice_spans` (marker-based: locates `Options:` and `Answer with...` markers in input_ids and derives the question span by subtraction; robust against BPE merges that broke standalone tokenize-and-search). `TextKResidualCache` (subclass of `DynamicCache`) captures per-position INT4 K-row residual during a BF16 forward.
- `qwen/scripts/expE1_text_slice_ablation.py` — Pass A (7 fixed-slice conditions) + Pass B (random ×3 seeds + K-residual at global median budget computed from Pass A).
- `qwen/scripts/expE1_smoke.py` — 5-check correctness gate (visual span, slice non-overlap, decode round-trip, V3K logits-differ on question-only mask, V3K question-only distinct from all-text-BF16).
- `qwen/scripts/run_expE1.sh smoke|passA|passB|analyze|full`.

Wall on H100: smoke ~30s, passA 54 min, passB 35 min, analyze <1s. Two implementation bugs caught and fixed: BPE-driven slice mismatch (smoke), and `TextKResidualCache` composition not subscriptable for transformers (Pass B). Both fixed; pipeline ran cleanly thereafter.

## Qwen2.5-VL × LongVideoBench — Experiment D: Evidence-Window + Cross-Modal Visual-Key Diagnostic (2026-05-08, COMPLETE)

**Headline: the visual-evidence-window precision-routing hypothesis is FALSIFIED for first-token MCQ scoring. Text-K is the dominant fragility.** D1.3 (text-K BF16, all 5760 visual-K INT4, V INT4) at avg 4.15 KV bits → 0.385. D1.4 (text-K INT4, all 5760 visual-K BF16, V INT4) at avg 9.85 KV bits → 0.210. D1.4 has 2.4× more KV bits and lands 17.5 pp WORSE — at the uniform-INT4 floor. All visual-K-routing conditions (top-1/2 attention, random, uniform, maxhead) cluster in [0.395, 0.435] with no separation by selection method. D0 also revealed the LM attention sink at the first visual token: `top1_window_all` collapses to window 0 in 195/200 items because heads dump no-op attention there; even fixing this via the maxhead diagnostic doesn't change the D1 outcome. Full writeup in `QWEN_EXPERIMENTS.md`.

After Exp C nailed K-fragility at INT4, **Exp D** asked the VLM-specific question: does low-bit visual-key quantization break long-video VLMs by corrupting *question-specific evidence retrieval*? Combined two-phase pipeline on the same 200 LongVideoBench eval items at 64 frames:

**Phase D0 — Evidence-window diagnostic (BF16 only).** For each item, run 8 BF16 forwards: Full-64 (eager + answer-query attention capture), Uniform-16, Top-1-window-only, Top-2-windows-only, Top-1-window-removed, Random-window-removed × 3 seeds. Classify by what visual evidence the question needs:
- *localized* — full64 correct ∧ top-window-only correct ∧ top-window-removed flips ∧ removal hurts more than random removal
- *global* — full64 correct ∧ uniform-16 correct ∧ top-window-removal does not hurt
- *distributed* — full64 correct ∧ top-1-only fails
- *attention_not_causal* — top-attention removal hurts no more than random

The top-window selector pools **raw visual-window attention mass** across all 28 layers × 4 KV-heads, then normalizes over 8 windows (8 frames each). Raw-mass pooling first naturally downweights text-focused heads. Mid-layer (8–20) and per-head-max variants saved as sensitivity diagnostics.

**Phase D1 — Cross-modal K/V quantization.** V is fixed at INT4 everywhere (Exp C: V at INT4 is essentially free if K is BF16). K is varied per condition via the new `BitController` V3K mode (per-token K mask, V uses layer's v_bits):

| Condition | Text-K | Visual-K policy | V |
|---|---|---|---:|
| D1.3 | BF16 | INT4 (all) | INT4 |
| D1.4 | INT4 | BF16 (all) | INT4 |
| **D1.5a** | BF16 | top-1 BF16 / rest INT4 | INT4 |
| D1.5b | BF16 | top-2 BF16 / rest INT4 | INT4 |
| D1.6a/b | BF16 | random-1/2 BF16 (×3 seeds) | INT4 |
| D1.7a/b | BF16 | uniform-spaced 1/2 BF16 | INT4 |

Headline test: on **localized** items, does `D1.5a` (top-1 visual-K BF16) beat `D1.6a` (random-1 visual-K BF16) at matched bit budget? If yes → answer-query attention picks visual evidence windows that matter for retrieval under K quantization.

### Implementation

- `qwen/scripts/fake_quant_kv_cache.py` — added `BitController` mode `V3K` (K masked per-token, V uses layer's v_bits). Backward-compatible with existing V1/V2/V3.
- `qwen/scripts/data_longvideobench.py` — added `format_mcq_messages_with_frames(item, frames)` for precomputed-frame inputs.
- `qwen/scripts/visual_tokens.py` — `find_visual_token_span` (locates `<|vision_start|>`/`<|vision_end|>`) + `build_window_token_ranges`.
- `qwen/scripts/frame_manip.py` — `decode_uniform_frames` (decord→imageio→cv2 fallback), window-index helpers, optional blank-in-place v2.
- `qwen/scripts/expD0_evidence_diagnostic.py` — Phase D0 driver (8 BF16 forwards/item with `EvidenceWindowAttentionHook`).
- `qwen/scripts/expD1_crossmodal_kv.py` — Phase D1 driver (per-condition V3K K-mask construction, V always INT4).
- `qwen/scripts/expD_smoke.py` — 5 correctness checks (visual span, window mapping, V3K logits-differ, mask-cache alignment, frame-removal end-to-end).
- `qwen/scripts/expD_analyze.py` — applies evidence labels post-hoc; produces `expD0_summary.md`, `expD1_summary.md`, `expD_combined_analysis.md` (D1 stratified by D0 label).
- `qwen/scripts/run_expD.sh smoke|d0|d1|analyze|full` — orchestrator.

### Running on tambe-server-1

```bash
ssh subha2@tambe-server-1
cd /data/subha2/quantization && git pull
source /data/subha2/experiments/qwen_venv/bin/activate
nvidia-smi
tmux new -s qwen-expD
CUDA_VISIBLE_DEVICES=<idx> bash qwen/scripts/run_expD.sh smoke      # ~30s, halts on fail
CUDA_VISIBLE_DEVICES=<idx> bash qwen/scripts/run_expD.sh d0         # ~1.5h (faster than 4h estimate)
CUDA_VISIBLE_DEVICES=<idx> bash qwen/scripts/run_expD.sh d1         # ~1.7h (faster than 5h estimate)
bash qwen/scripts/run_expD.sh analyze                                # no GPU
```

Actual wall on H100 with 25 GB co-tenant on GPU 0: smoke = 25s, D0 = 1h 26min, D1 = 1h 42min, analyze = <1s. Total pipeline = ~3h 10min.

Frame-manipulation v1 = frame removal (sequence length and temporal positions change for top-only / window-removed conditions). Marked in JSONL as `mode="frame_removal_v1"`. Stretch-goal v2 = blank-in-place via decord pre-decode + black-frame substitution; not pursued because v1 results already falsify the visual-K-window-routing hypothesis at the bigger-effect text-vs-visual-K level.

### Next steps (post-D)

The original visual-K-window precision-routing thesis is ruled out. Pivots:
1. **Text-K outlier handling at INT4.** Text K is small (~140 tokens × 28 layers × 4 KV-heads × 128 head-dim ≈ 2 MB) and high-impact; per-channel K calibration restricted to text positions or AKVQ-VL outlier extraction on text-K should close the 17.5 pp gap.
2. **Finer text-K partition** (header / question / options / instruction) via a C2.1-style isolation sweep on the text subspans.
3. **Long-form generation re-test.** Multi-token decoding (Video-MME generation, MVBench) may stress visual-K differently; first-token MCQ is the minimal-visual-K-stress setting.

## Qwen2.5-VL × LongVideoBench — Experiment C: K/V isolation mini-sweep (2026-05-07)

**Headline: K-quantization is the killer at INT4; V is essentially free.** Four conditions on 100 stratified eval items each, leaving one of K or V at BF16 and quantizing the other:

| Condition | K bits | V bits | acc | BF16-correct preserved |
|---|---:|---:|---:|---:|
| **C2.1** BF16-K + INT4-V | 16 | 4 | **0.530** | **0.945** |
| C2.2 INT4-K + BF16-V | 4 | 16 | 0.290 | 0.218 |
| C2.3 BF16-K + INT2-V | 16 | 2 | 0.210 | 0.182 |
| C2.4 INT2-K + BF16-V | 2 | 16 | 0.330 | 0.364 |

For paired comparison on the same 100 items: BF16 ceiling = 0.550, A5 (INT4/INT4) = 0.210, A7 (INT2/INT2) = 0.210.

**Two-regime asymmetry:**
- **At INT4**: K-fragile. C2.1 (BF16-K) lands within 2 pp of the BF16 ceiling and preserves 94.5% of BF16-correct items. **The full Exp-A 30 pp rescue gap is recoverable simply by leaving K at full precision.** Mirror C2.2 (BF16-V) lifts only 8 pp off the floor.
- **At INT2**: asymmetry flips — V is the worse side. Neither C2.3 nor C2.4 crosses the rescue midpoint, but C2.4 (V kept BF16) is +12 pp over C2.3. Consistent with Exp A's surprise that A7 INT2-ternary slightly outperforms A5 INT4: ternary {−s, 0, +s} preserves K-row sign+scale per channel (what attention's "key match" needs); INT2 V loses too much value-magnitude information for the attention×value matmul.

**Implication:** the next experiment is *not* a richer KV tier set, and *not* {INT2, INT4, BF16} routing — it's **K-side outlier handling at INT4** (KIVI per-channel K + per-token V; AKVQ-VL static K outlier extraction; post-RoPE K-channel calibration). Once K is fixed, V can stay at INT4 with effectively no accuracy loss.

### Implementation

- `qwen/scripts/expA_baseline.py` — extended `CONDITIONS` with 4 `C2_*` rows and added `--stratified_limit N` for proportional bucket sub-sampling.
- `qwen/scripts/run_expC_kv_isolation.sh` — single-step orchestrator (modeled on `run_resume.sh`).
- `qwen/scripts/expC_analyze.py` — paired-comparison analysis on the 100-item C2 universe with auto-classified diagnosis (K-fragile / V-fragile / joint).
- `qwen/scripts/expC0_diagnostics.py` — no-compute diagnostics from existing JSONLs (A5↔A7 complementarity, B6/B8/B9/B10 selected-block coverage, paired Δ-margin vs uniform floors).

Full writeup, paired tables, and diagnosis text in `QWEN_EXPERIMENTS.md` → "Experiment C — K/V isolation mini-sweep". Companion no-compute diagnostics in "Experiment C0".

### Running on tambe-server-1

```bash
# After git pull and venv activation (qwen_venv):
ssh subha2@tambe-server-1
cd /data/subha2/quantization && git pull
source /data/subha2/experiments/qwen_venv/bin/activate
nvidia-smi  # confirm an unused GPU
tmux new -s qwen-expC
CUDA_VISIBLE_DEVICES=<idx> EXPC_N_ITEMS=100 \
    bash /data/subha2/quantization/qwen/scripts/run_expC_kv_isolation.sh
# ~22 min for 4 conditions × 100 items × 64 frames.
# Analysis (after sync back locally):
python3 qwen/scripts/expC_analyze.py \
    --jsonl qwen/results/expA_rollouts_Qwen2.5-VL-7B-Instruct.jsonl \
    --out   qwen/results/expC_kv_isolation_summary.md
```

## Qwen2.5-VL × LongVideoBench KV-cache experiments (staged 2026-05-06)

Pivot from the saturated pi0.5/LIBERO line (where `Static-W2-l13-17` already hits 100% on standard LIBERO Long, leaving no rescue gap for AttnEntropy) to **Qwen2.5-VL-7B-Instruct on LongVideoBench-val**, where the model is unsaturated (54.7%) and long-video produces thousands of visual tokens, so KV-cache precision matters. The novelty thesis shifts from "AttnEntropy rescues weight quant" to: **attention entropy can allocate per-(layer, head, token) KV-cache *precision* better than uniform / random / attention-mass / MEDA-style baselines.**

New top-level folder `qwen/` with parallel infrastructure (separate from the openpi/LIBERO line):

```
qwen/
├── README.md
├── scripts/
│   ├── setup_qwen_env.sh         # uv venv + transformers + qwen-vl-utils + auto-awq
│   ├── data_longvideobench.py    # LongVideoBench-val loader, 100-cal/200-eval stratified split
│   ├── fake_quant_kv_cache.py    # FakeQuantKVCache(DynamicCache) + BitController (V1/V2/V3)
│   ├── attn_entropy_hook.py      # multi-layer entropy hook (port of L12H2EntropyHook)
│   ├── run_inference.py          # 4-way MCQ logprob scorer
│   ├── calibrate.py              # frozen layer/head/token thresholds from cal split
│   ├── expA_baseline.py          # 8 conditions × {64,128} frame budgets
│   ├── expB_attnentropy.py       # 9 conditions @ matched ~3 avg KV bits
│   ├── expB_pareto_plot.py       # avg KV bits vs accuracy Pareto curve
│   ├── run_smoke.sh              # 3B + 10 items + hard logits-differ assertion
│   └── run_main.sh               # 7B + 200-eval + nvidia-smi gate
├── calibration/                  # frozen thresholds JSON, per (model, target_avg_bits, frames)
├── results/                      # per-sample JSONL + summary markdown
└── plots/
```

### Correctness invariant (critical)

For MCQ scoring with `max_new_tokens=1`, the scored logit is produced by the prefill forward pass. Quantizing K/V *only* in `cache.update()` is not always sufficient — it depends on whether the attention backend uses the value returned by `update()`. For Qwen2.5-VL's SDPA forward (transformers ≥ 4.49), the returned tensors *do* feed the SDPA matmul, so our `FakeQuantKVCache.update()` quantizing on write affects first-token logits. `run_smoke.sh` asserts this by computing `||logits_BF16 − logits_INT2||_∞` across items; the test fails loud if max-diff ≤ 1e-3. `install_explicit_kv_quant_patch()` provides a fallback for backends where the cache-return is not consumed.

### Conditions

**Experiment A** (8): BF16+BF16 (ceiling), W4-fakequant + BF16-KV, AWQ + BF16-KV, BF16 + FP8-KV, BF16 + INT4-KV, BF16 + INT4-K/INT8-V (asymmetric), BF16 + INT2-KV (sub-2-bit stress), AWQ + INT4-KV (combined realistic).

**Experiment B** (9, matched ~3 avg KV bits): Uniform INT4 / INT2 (Pareto anchors), Random mixed (3 seeds), Attention-mass token protection, MEDA-style entropy bit-budget, **AttnEntropy V1 (layer-level)**, **AttnEntropy V2 ((layer, KV-head)-level under GQA, 4 KV-heads)**, **AttnEntropy V3 ((layer, token)-level via online attention-mass mask)**, optional Oracle (logit-KL).

### Running

```bash
# On tambe-server-1 (after git pull):
cd /data/subha2/quantization/qwen/scripts
bash setup_qwen_env.sh
source /data/subha2/experiments/qwen_venv/bin/activate

nvidia-smi
export CUDA_VISIBLE_DEVICES=<unused-gpu>
bash run_smoke.sh   # MUST pass the BF16-vs-INT2 logits-differ assertion before scaling up
bash run_main.sh
```

Plan in `/Users/subha/.claude/plans/you-can-make-a-parallel-crown.md`.

## Key References

- **QVLA** (ICLR 2026) — Action-centric channel-wise quantization for AR VLAs
- **QuantVLA** (arXiv 2602.20309) — Training-free PTQ for DiT-head VLAs (pi0.5 on LIBERO)
- **BlockDialect** (ICML 2025) — Block-wise fine-grained mixed format quantization with FP4 formatbook
- **DP-LLM** (NeurIPS 2025) — Dynamic layer-wise precision assignment at runtime
- **KVQuant** (NeurIPS 2024) — KV cache quantization with per-layer sensitivity-weighted NUQ
- **MicroMix** (2025) — Mixed-precision quantization with MXFP4/6/8 formats

Advisor: Wonsuk Jang (Stanford EE, BlockDialect author)
