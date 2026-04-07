# Quantization Research

Research project on quantization for flow-matching Vision-Language-Action (VLA) models.

## Research Direction

Studying how weight quantization of the VLM backbone in flow-matching VLAs (pi0, Alpamayo-R1, GR00T-N1) affects downstream trajectory quality. The core question: does the indirect error propagation pathway (VLM → KV cache → action expert → iterative denoising → trajectory) create different sensitivity patterns than the direct pathway in autoregressive VLAs studied by QVLA (ICLR 2026)?

Connection to BlockDialect (ICML 2025): extending block-wise fine-grained mixed format quantization with action-aware formatbook design for VLA modules.

## Project Structure

- `RESEARCH_IDEAS.md` — Original brainstorming and advisor notes
- `RESEARCH_PLAN.md` — Detailed experiment plan for the VLA quantization study
- `INITIAL_EXPERIMENT.md` — Quick 1-hour activation profiling experiment for initial validation
- `papers/` — Reference papers (KVQuant, BlockDialect, MicroMix, DP-LLM, QVLA, Alpamayo-R1)

## Key References

- **QVLA** (ICLR 2026) — Action-centric channel-wise quantization for AR VLAs
- **BlockDialect** (ICML 2025) — Block-wise fine-grained mixed format quantization with FP4 formatbook
- **KVQuant** (NeurIPS 2024) — KV cache quantization with per-layer sensitivity-weighted NUQ
- **MicroMix** (2025) — Mixed-precision quantization with MXFP4/6/8 formats
- **DP-LLM** (NeurIPS 2025) — Dynamic layer-wise precision assignment at runtime
