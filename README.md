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
└── scripts/                  — Overnight experiment scripts for H100
    ├── setup_env.sh          — GCP environment setup (clone openpi, install deps, convert checkpoint)
    ├── utils.py              — Shared utilities (model loading, data, quantization, hooks, I/O)
    ├── setup_and_verify.py   — Go/no-go verification gate
    ├── exp1_activation_stats.py  — Cross-suite activation statistics
    ├── exp2_layer_sensitivity.py — Per-layer sensitivity probe with per-sample metadata
    ├── exp3_flow_step_sensitivity.py — Per-denoising-step sensitivity (novel)
    └── run_all.py            — Orchestrator for overnight runs
```

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

## Key References

- **QVLA** (ICLR 2026) — Action-centric channel-wise quantization for AR VLAs
- **QuantVLA** (arXiv 2602.20309) — Training-free PTQ for DiT-head VLAs (pi0.5 on LIBERO)
- **BlockDialect** (ICML 2025) — Block-wise fine-grained mixed format quantization with FP4 formatbook
- **DP-LLM** (NeurIPS 2025) — Dynamic layer-wise precision assignment at runtime
- **KVQuant** (NeurIPS 2024) — KV cache quantization with per-layer sensitivity-weighted NUQ
- **MicroMix** (2025) — Mixed-precision quantization with MXFP4/6/8 formats

Advisor: Wonsuk Jang (Stanford EE, BlockDialect author)
