#!/usr/bin/env python3
"""
Experiment 9 — FLOPs + wall-time benchmark: VLM prefix vs action-expert denoise loop.

Question: on an H100, what fraction of pi0.5's per-observation inference budget is
spent in the VLM prefix pass (PaliGemma + SigLIP, called once) vs. the 10-step
action-expert denoising loop (Gemma-style expert, called 10 times)? And is each
phase compute-bound, memory-bound, or latency-bound?

Two measurement passes (kept separate so profiler overhead doesn't distort timing):

  1. Wall-time pass — CUDA events around
       - policy.infer(obs)                               [end-to-end]
       - paligemma_with_expert.forward(inputs_embeds=[prefix, None], ...)   [prefix]
       - model.denoise_step(...)                         [each of 10 Euler steps]
     Warmup + measured iterations; report mean/std/p50/p95.

  2. FLOPs pass — torch.profiler(with_flops=True, record_shapes=True) over ONE
     policy.infer call. Attribute FLOPs to prefix vs expert via a phase flag the
     wrappers set. Cross-check with an analytical count built from each
     nn.Linear's shape and observed seq-len.

Phase discrimination (see openpi/models_pytorch/pi0_pytorch.py):
  - prefix  = paligemma_with_expert.forward(inputs_embeds=[X,    None], ...)
  - denoise = paligemma_with_expert.forward(inputs_embeds=[None,    X], ...)

Outputs:
  $EXPERIMENT_DIR/results/exp9_flops_benchmark.jsonl   — per-iter raw rows
  $EXPERIMENT_DIR/results/exp9_flops_benchmark.md      — summary tables + bound read

Usage:
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/exp9_flops_benchmark.py --smoke
  CUDA_VISIBLE_DEVICES=0 uv run python scripts/exp9_flops_benchmark.py \
      --n-obs 20 --n-warmup 5 --n-measure 50 --batch-size 1
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import utils


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------
H100_PCIE_PEAK_FP16_TFLOPS = 756.0   # dense FP16 TFLOPs/s (no sparsity)
H100_SXM_PEAK_FP16_TFLOPS = 989.0
H100_PCIE_HBM_TBps = 2.0             # HBM2e / HBM3 on PCIe
H100_SXM_HBM_TBps = 3.35


def gpu_info():
    info = {"name": None, "variant": None, "peak_tflops_fp16": None,
            "hbm_bandwidth_TBps": None, "cuda_idx": None}
    if torch.cuda.is_available():
        info["cuda_idx"] = int(torch.cuda.current_device())
        info["name"] = torch.cuda.get_device_name(info["cuda_idx"])
        lname = info["name"].lower()
        if "h100" in lname:
            if "pcie" in lname:
                info["variant"] = "H100 PCIe"
                info["peak_tflops_fp16"] = H100_PCIE_PEAK_FP16_TFLOPS
                info["hbm_bandwidth_TBps"] = H100_PCIE_HBM_TBps
            else:
                info["variant"] = "H100 SXM"
                info["peak_tflops_fp16"] = H100_SXM_PEAK_FP16_TFLOPS
                info["hbm_bandwidth_TBps"] = H100_SXM_HBM_TBps
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,utilization.gpu",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True).strip()
        info["nvidia_smi"] = out
    except Exception:
        info["nvidia_smi"] = "nvidia-smi unavailable"
    return info


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------
def find_pw_expert(model):
    """paligemma_with_expert is an attribute on the model — reach it directly."""
    pw = getattr(model, "paligemma_with_expert", None)
    if pw is None:
        for name, m in model.named_modules():
            if name.endswith("paligemma_with_expert"):
                return name, m
        raise RuntimeError("paligemma_with_expert not found on model")
    return "paligemma_with_expert", pw


def module_param_counts(model):
    """Return a flat dict of {role: n_params} for the major components."""
    buckets = {
        "vision_tower": 0,
        "paligemma_language_model": 0,
        "paligemma_projector_other": 0,
        "gemma_expert": 0,
        "action_in_out_proj": 0,
        "other": 0,
    }
    for name, p in model.named_parameters():
        n = p.numel()
        if "vision_tower" in name:
            buckets["vision_tower"] += n
        elif "gemma_expert" in name:
            buckets["gemma_expert"] += n
        elif "paligemma_with_expert.paligemma.language_model" in name:
            buckets["paligemma_language_model"] += n
        elif "paligemma_with_expert.paligemma" in name:
            buckets["paligemma_projector_other"] += n
        elif "action_in_proj" in name or "action_out_proj" in name or "action_time_mlp" in name \
             or "state_proj" in name:
            buckets["action_in_out_proj"] += n
        else:
            buckets["other"] += n
    buckets["total"] = sum(v for k, v in buckets.items() if k != "total")
    return buckets


# ---------------------------------------------------------------------------
# Timing + phase-flagging wrapper
#
# Strategy: wrap paligemma_with_expert.forward once. Inside, discriminate prefix
# vs denoise from inputs_embeds=[x, y] (prefix: x set, y None; denoise: x None,
# y set). Also wrap model.denoise_step for per-step totals (which include
# embed_suffix + mask prep + paligemma_with_expert.forward + action_out_proj).
# ---------------------------------------------------------------------------
class PhaseTimer:
    def __init__(self, model):
        self.model = model
        self._pw_name, self._pw = find_pw_expert(model)
        self._orig_pw_fwd = self._pw.forward
        self._orig_denoise = model.denoise_step

        # current-infer buffers
        self.reset()

        # current active phase flag (for FLOPs-pass attribution in the outer scope)
        self.active_phase = None   # "prefix" | "expert_forward" | None
        self.active_step_idx = None

        # install wrappers
        self._install()

    # ----- lifecycle -----
    def reset(self):
        self.prefix_ms = None
        self.denoise_step_ms = []          # len == 10 after a full infer
        self.pw_forward_ms = []            # 11 entries (1 prefix + 10 expert)
        self._current_infer_t0 = None

    def close(self):
        # Restore originals.
        self._pw.forward = self._orig_pw_fwd
        self.model.denoise_step = self._orig_denoise

    # ----- helpers -----
    @staticmethod
    def _classify_call(kwargs):
        emb = kwargs.get("inputs_embeds", None)
        if not (isinstance(emb, (list, tuple)) and len(emb) == 2):
            return "unknown"
        a, b = emb
        if a is not None and b is None:
            return "prefix"
        if a is None and b is not None:
            return "expert_forward"
        return "joint"

    def _cuda_event(self):
        return torch.cuda.Event(enable_timing=True)

    # ----- install / wrappers -----
    def _install(self):
        timer = self

        orig_pw_fwd = self._orig_pw_fwd
        def pw_forward_wrapper(*args, **kwargs):
            phase = timer._classify_call(kwargs)
            timer.active_phase = phase
            if torch.cuda.is_available():
                start = timer._cuda_event()
                end = timer._cuda_event()
                start.record()
                try:
                    out = orig_pw_fwd(*args, **kwargs)
                finally:
                    end.record()
                    # Don't sync here — sync outside the whole infer, then read ms.
                    timer.pw_forward_ms.append({
                        "phase": phase, "start": start, "end": end,
                        "step_idx": timer.active_step_idx,
                    })
                    timer.active_phase = None
            else:
                t0 = time.perf_counter()
                try:
                    out = orig_pw_fwd(*args, **kwargs)
                finally:
                    timer.pw_forward_ms.append({
                        "phase": phase, "ms_cpu": (time.perf_counter() - t0) * 1000.0,
                        "step_idx": timer.active_step_idx,
                    })
                    timer.active_phase = None
            return out
        self._pw.forward = pw_forward_wrapper

        orig_denoise = self._orig_denoise
        def denoise_wrapper(*args, **kwargs):
            step_idx = len(timer.denoise_step_ms)
            timer.active_step_idx = step_idx
            if torch.cuda.is_available():
                start = timer._cuda_event()
                end = timer._cuda_event()
                start.record()
                try:
                    out = orig_denoise(*args, **kwargs)
                finally:
                    end.record()
                    timer.denoise_step_ms.append({
                        "step_idx": step_idx, "start": start, "end": end,
                    })
                    timer.active_step_idx = None
            else:
                t0 = time.perf_counter()
                try:
                    out = orig_denoise(*args, **kwargs)
                finally:
                    timer.denoise_step_ms.append({
                        "step_idx": step_idx, "ms_cpu": (time.perf_counter() - t0) * 1000.0,
                    })
                    timer.active_step_idx = None
            return out
        self.model.denoise_step = denoise_wrapper

    # ----- readout (call AFTER sync) -----
    def read_ms(self):
        out = {"denoise_steps_ms": [], "pw_forward_ms": []}
        for rec in self.denoise_step_ms:
            if "start" in rec:
                out["denoise_steps_ms"].append(rec["start"].elapsed_time(rec["end"]))
            else:
                out["denoise_steps_ms"].append(rec["ms_cpu"])
        for rec in self.pw_forward_ms:
            if "start" in rec:
                ms = rec["start"].elapsed_time(rec["end"])
            else:
                ms = rec["ms_cpu"]
            out["pw_forward_ms"].append({"phase": rec["phase"], "step_idx": rec["step_idx"], "ms": ms})
        # prefix = first pw_forward whose phase == "prefix"
        prefixes = [r for r in out["pw_forward_ms"] if r["phase"] == "prefix"]
        out["prefix_ms"] = prefixes[0]["ms"] if prefixes else None
        return out


# ---------------------------------------------------------------------------
# Analytical FLOPs from Linear shapes + observed seq lengths
# ---------------------------------------------------------------------------
class SeqLenRecorder:
    """Records the last observed input seq_len on each Linear. Uses forward hook."""
    def __init__(self, model):
        self.model = model
        self._handles = []
        self._seq_lens = {}   # name → last observed seq_len
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                h = m.register_forward_hook(self._mk(name))
                self._handles.append(h)

    def _mk(self, name):
        def hook(mod, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            if isinstance(x, torch.Tensor) and x.dim() >= 2:
                # last-dim is features; second-to-last (or product of leading
                # dims excluding last) is effective sequence.
                self._seq_lens[name] = int(np.prod(list(x.shape[:-1])))
        return hook

    def seq_lens(self):
        return dict(self._seq_lens)

    def close(self):
        for h in self._handles:
            h.remove()


def analytical_flops(model, seq_lens):
    """For each Linear m with observed effective seq_len S:
        FLOPs per call = 2 * S * in * out
    Attribute to a phase bucket based on module-name substring.

    We treat a "prefix call" as the observation when VLM tokens dominate seq_len
    (most language_model + vision_tower Linears fire during prefix). We treat
    "expert call" as when gemma_expert Linears fire. This is bounded by what
    SeqLenRecorder sees; for a faithful analytical count, capture seq_lens over
    BOTH a prefix pass and a denoise pass (see main).
    """
    by_bucket = {
        "vision_tower": 0.0,
        "paligemma_language_model": 0.0,
        "paligemma_other": 0.0,
        "gemma_expert": 0.0,
        "action_proj": 0.0,
        "other": 0.0,
    }
    unmapped = []
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        seq = seq_lens.get(name, None)
        if seq is None:
            continue
        out_f, in_f = m.weight.shape
        flops = 2.0 * seq * in_f * out_f
        if "vision_tower" in name:
            by_bucket["vision_tower"] += flops
        elif "gemma_expert" in name:
            by_bucket["gemma_expert"] += flops
        elif "paligemma_with_expert.paligemma.language_model" in name:
            by_bucket["paligemma_language_model"] += flops
        elif "paligemma_with_expert.paligemma" in name:
            by_bucket["paligemma_other"] += flops
        elif "action_in_proj" in name or "action_out_proj" in name \
             or "action_time_mlp" in name or "state_proj" in name:
            by_bucket["action_proj"] += flops
        else:
            by_bucket["other"] += flops
            unmapped.append(name)
    return by_bucket, unmapped


# ---------------------------------------------------------------------------
# Wall-time benchmark loop
# ---------------------------------------------------------------------------
def bench_wall_time(policy, model, observations, n_warmup, n_measure, jsonl_path):
    timer = PhaseTimer(model)

    def _infer_once(obs):
        timer.reset()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                _ = policy.infer(obs)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = policy.infer(obs)
            total_ms = (time.perf_counter() - t0) * 1000.0
        readout = timer.read_ms()
        return total_ms, readout

    utils.log(f"[exp9] wall-time: warmup={n_warmup} measure={n_measure} n_obs={len(observations)}")

    # WARMUP (discard timings)
    for i in range(n_warmup):
        obs = observations[i % len(observations)]
        total_ms, _ = _infer_once(obs)
        if i < 3 or i == n_warmup - 1:
            utils.log(f"[exp9]  warmup {i+1}/{n_warmup}  total={total_ms:.2f} ms")

    # MEASURE
    rows = []
    for i in range(n_measure):
        obs = observations[i % len(observations)]
        total_ms, readout = _infer_once(obs)
        row = {
            "iter": i,
            "total_ms": total_ms,
            "prefix_ms": readout["prefix_ms"],
            "denoise_steps_ms": readout["denoise_steps_ms"],   # 10 entries
            "pw_forward_ms": readout["pw_forward_ms"],         # 11 entries
        }
        rows.append(row)
        utils.append_jsonl(row, jsonl_path)
        if i == 0 or (i + 1) % 10 == 0:
            utils.log(f"[exp9]  iter {i+1}/{n_measure}  "
                      f"total={total_ms:.2f}  prefix={readout['prefix_ms']:.2f}  "
                      f"denoise_mean={float(np.mean(readout['denoise_steps_ms'])):.2f} ms")

    timer.close()
    return rows


def summarize_timing(rows):
    totals = np.array([r["total_ms"] for r in rows])
    prefixes = np.array([r["prefix_ms"] for r in rows if r["prefix_ms"] is not None])
    # steps[iter][step_idx] — assume exactly 10 steps per infer
    steps = np.array([r["denoise_steps_ms"] for r in rows])
    assert steps.shape[1] == 10, f"Expected 10 denoise steps; got {steps.shape[1]}"
    denoise_sums = steps.sum(axis=1)

    def stat(a):
        return {
            "mean_ms": float(a.mean()),
            "std_ms": float(a.std()),
            "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)),
            "min_ms": float(a.min()),
            "max_ms": float(a.max()),
        }

    return {
        "total": stat(totals),
        "prefix": stat(prefixes),
        "denoise_sum_10_steps": stat(denoise_sums),
        "denoise_per_step_mean": [float(steps[:, k].mean()) for k in range(10)],
        "denoise_per_step_std": [float(steps[:, k].std()) for k in range(10)],
        "prefix_frac_of_total_mean": float(prefixes.mean() / totals.mean()),
        "denoise_frac_of_total_mean": float(denoise_sums.mean() / totals.mean()),
        "residual_frac_of_total_mean": float(1.0 - (prefixes.mean() + denoise_sums.mean()) / totals.mean()),
        "n_iters": len(rows),
    }


# ---------------------------------------------------------------------------
# FLOPs pass — torch.profiler + per-phase attribution
# ---------------------------------------------------------------------------
def bench_flops(policy, model, observation):
    """Run one policy.infer under torch.profiler with with_flops=True.
    Attribute FLOPs to (prefix, expert) by tagging the active phase via wrappers.
    Returns: dict with prefix_gflops, expert_per_step_gflops (list, 10),
    expert_total_gflops, analytical_gflops_by_bucket."""
    from torch.profiler import profile, ProfilerActivity, record_function

    # --- First, capture per-Linear seq-lens during prefix + during denoise ---
    rec_prefix = SeqLenRecorder(model)
    timer = PhaseTimer(model)
    # Use a recorded_function context to label phases in the trace.
    # We need to attribute FLOPs per-phase from the profiler, so we push labels
    # via record_function inside wrappers.

    # Re-install wrappers that additionally emit record_function labels.
    # Simpler: run one infer to capture seq_lens first; then run another under
    # profiler.
    with torch.no_grad():
        _ = policy.infer(observation)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    seq_lens_mixed = rec_prefix.seq_lens()
    rec_prefix.close()
    timer.close()

    # Analytical count from mixed seq_lens (last-observed per Linear).
    # We additionally run a prefix-only pass and a denoise-only pass to get
    # phase-pure seq_lens, by re-hooking around those specific forward calls.
    prefix_seq_lens = {}
    expert_seq_lens = {}

    # Re-install wrappers + fresh recorders that isolate per-phase seq_lens.
    timer = PhaseTimer(model)

    class PhaseAwareSeqLenRecorder:
        def __init__(self, model, timer):
            self.timer = timer
            self._handles = []
            for name, m in model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    self._handles.append(m.register_forward_hook(self._mk(name)))

        def _mk(self, name):
            def hook(mod, inp, out):
                phase = self.timer.active_phase
                x = inp[0] if isinstance(inp, tuple) else inp
                if isinstance(x, torch.Tensor) and x.dim() >= 2:
                    s = int(np.prod(list(x.shape[:-1])))
                    if phase == "prefix":
                        prefix_seq_lens[name] = s
                    elif phase == "expert_forward":
                        expert_seq_lens[name] = s
            return hook

        def close(self):
            for h in self._handles:
                h.remove()

    rec2 = PhaseAwareSeqLenRecorder(model, timer)
    with torch.no_grad():
        _ = policy.infer(observation)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rec2.close()

    # ----- Analytical FLOPs -----
    # For prefix: sum over Linears that fired during prefix.
    prefix_by_bucket, _ = analytical_flops(model, prefix_seq_lens)
    # For one denoise step: sum over Linears that fired during a single expert_forward.
    # expert_seq_lens was overwritten 10 times — it's the LAST step's seq_lens.
    # Since seq_len within expert is constant across steps, using the last is fine.
    expert_by_bucket, _ = analytical_flops(model, expert_seq_lens)

    analytical = {
        "prefix_gflops": sum(prefix_by_bucket.values()) / 1e9,
        "prefix_by_bucket_gflops": {k: v / 1e9 for k, v in prefix_by_bucket.items()},
        "expert_per_step_gflops": sum(expert_by_bucket.values()) / 1e9,
        "expert_per_step_by_bucket_gflops": {k: v / 1e9 for k, v in expert_by_bucket.items()},
        "expert_total_10_steps_gflops": 10 * sum(expert_by_bucket.values()) / 1e9,
    }

    # ----- torch.profiler pass -----
    # Uses the same wrappers (still installed). The wrappers don't push
    # record_function labels; instead we tag with record_function in a
    # closure around policy.infer. Simpler attribution: profiler totals
    # across the whole call, then use the ratio from analytical to split.
    profiler_total_flops = 0.0
    try:
        acts = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            acts.append(ProfilerActivity.CUDA)
        with profile(activities=acts, with_flops=True, record_shapes=True) as prof:
            with torch.no_grad():
                _ = policy.infer(observation)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        for evt in prof.key_averages():
            f = getattr(evt, "flops", 0) or 0
            profiler_total_flops += f
    except Exception as e:
        utils.log(f"[exp9] profiler pass failed: {e}")

    timer.close()

    return {
        "analytical": analytical,
        "profiler_total_gflops": profiler_total_flops / 1e9,
        "prefix_seq_lens_sample": dict(list(prefix_seq_lens.items())[:10]),
        "expert_seq_lens_sample": dict(list(expert_seq_lens.items())[:10]),
    }


# ---------------------------------------------------------------------------
# Bound classification
# ---------------------------------------------------------------------------
def classify_bound(phase_ms, phase_gflops, peak_tflops):
    if phase_ms is None or phase_ms <= 0 or phase_gflops is None or peak_tflops is None:
        return "unknown", 0.0
    achieved_tflops = (phase_gflops / 1e3) / (phase_ms / 1e3)  # GFLOPs/ms → TFLOPs/s (equiv)
    util = achieved_tflops / peak_tflops
    if util > 0.55:
        label = "compute-bound"
    elif util < 0.08:
        label = "latency-or-memory-bound"
    else:
        label = "mixed"
    return label, util


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------
def fmt_ms(x):
    return f"{x:.2f}" if x is not None and not np.isnan(x) else "—"


def write_markdown(md_path, cfg, gpu, params, timing_summary, flops_summary, batch_size):
    lines = []
    lines.append("# Exp9 — pi0.5 FLOPs + wall-time benchmark\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Batch size: **{batch_size}**  |  n_warmup={cfg['n_warmup']}  n_measure={cfg['n_measure']}  n_obs={cfg['n_obs']}")
    lines.append("")

    # --- Hardware ---
    lines.append("## Hardware")
    lines.append("")
    lines.append(f"- GPU: **{gpu.get('name')}** (variant: {gpu.get('variant')})")
    lines.append(f"- CUDA_VISIBLE_DEVICES index inside process: {gpu.get('cuda_idx')}")
    lines.append(f"- Peak FP16 TFLOPs/s (dense): {gpu.get('peak_tflops_fp16')}")
    lines.append(f"- HBM bandwidth: {gpu.get('hbm_bandwidth_TBps')} TB/s")
    lines.append("```")
    lines.append(gpu.get("nvidia_smi", ""))
    lines.append("```")
    lines.append("")

    # --- Parameters ---
    lines.append("## Parameter counts (numel)")
    lines.append("")
    lines.append("| component | params | FP16 MiB |")
    lines.append("|---|---:|---:|")
    for k, v in params.items():
        if k == "total":
            continue
        lines.append(f"| {k} | {v:,} | {v * 2 / 1024 / 1024:.1f} |")
    lines.append(f"| **total** | **{params['total']:,}** | **{params['total'] * 2 / 1024 / 1024:.1f}** |")
    lines.append("")

    # --- Wall time ---
    t = timing_summary
    lines.append("## Wall-time summary (CUDA events)")
    lines.append("")
    lines.append("| phase | mean ms | std | p50 | p95 | frac of total |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| end-to-end policy.infer | {fmt_ms(t['total']['mean_ms'])} | {fmt_ms(t['total']['std_ms'])} | {fmt_ms(t['total']['p50_ms'])} | {fmt_ms(t['total']['p95_ms'])} | 1.000 |")
    lines.append(f"| VLM prefix (1 call) | {fmt_ms(t['prefix']['mean_ms'])} | {fmt_ms(t['prefix']['std_ms'])} | {fmt_ms(t['prefix']['p50_ms'])} | {fmt_ms(t['prefix']['p95_ms'])} | {t['prefix_frac_of_total_mean']:.3f} |")
    lines.append(f"| expert 10-step denoise (sum) | {fmt_ms(t['denoise_sum_10_steps']['mean_ms'])} | {fmt_ms(t['denoise_sum_10_steps']['std_ms'])} | {fmt_ms(t['denoise_sum_10_steps']['p50_ms'])} | {fmt_ms(t['denoise_sum_10_steps']['p95_ms'])} | {t['denoise_frac_of_total_mean']:.3f} |")
    lines.append(f"| residual (preproc + postproc + framework) | — | — | — | — | {t['residual_frac_of_total_mean']:.3f} |")
    lines.append("")
    lines.append("Per-step expert timing (mean ± std, ms):")
    lines.append("")
    lines.append("| step k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |")
    lines.append("|---|" + "|".join(["---:"] * 10) + "|")
    means = t["denoise_per_step_mean"]
    stds  = t["denoise_per_step_std"]
    lines.append("| ms | " + " | ".join([f"{m:.2f}±{s:.2f}" for m, s in zip(means, stds)]) + " |")
    lines.append("")

    # --- FLOPs ---
    a = flops_summary["analytical"]
    lines.append("## FLOPs (analytical, from nn.Linear shapes × observed seq-lens)")
    lines.append("")
    lines.append(f"- VLM prefix (1 call): **{a['prefix_gflops']:.1f} GFLOPs**")
    lines.append(f"- Expert denoise (1 step): **{a['expert_per_step_gflops']:.1f} GFLOPs**")
    lines.append(f"- Expert denoise total (10 steps): **{a['expert_total_10_steps_gflops']:.1f} GFLOPs**")
    lines.append(f"- End-to-end (prefix + 10 expert steps): **{a['prefix_gflops'] + a['expert_total_10_steps_gflops']:.1f} GFLOPs**")
    lines.append("")
    lines.append("Breakdown by submodule (GFLOPs):")
    lines.append("")
    lines.append("| bucket | prefix | per-expert-step | 10-step expert |")
    lines.append("|---|---:|---:|---:|")
    for k in ["vision_tower", "paligemma_language_model", "paligemma_other",
              "gemma_expert", "action_proj", "other"]:
        p = a["prefix_by_bucket_gflops"].get(k, 0.0)
        e = a["expert_per_step_by_bucket_gflops"].get(k, 0.0)
        lines.append(f"| {k} | {p:.2f} | {e:.2f} | {10*e:.2f} |")
    lines.append("")
    lines.append(f"- torch.profiler total (sanity cross-check, one infer): **{flops_summary['profiler_total_gflops']:.1f} GFLOPs**")
    lines.append("  (profiler only counts ops that emit aten-level FLOPs; should be within ~2× of analytical)")
    lines.append("")

    # --- Bound read ---
    peak = gpu.get("peak_tflops_fp16")
    prefix_bound, prefix_util = classify_bound(t["prefix"]["mean_ms"], a["prefix_gflops"], peak)
    per_step_bound, per_step_util = classify_bound(t["denoise_per_step_mean"][0],
                                                   a["expert_per_step_gflops"], peak)
    expert_total_bound, expert_total_util = classify_bound(
        t["denoise_sum_10_steps"]["mean_ms"], a["expert_total_10_steps_gflops"], peak)
    e2e_bound, e2e_util = classify_bound(t["total"]["mean_ms"],
                                          a["prefix_gflops"] + a["expert_total_10_steps_gflops"], peak)

    lines.append("## Bound classification")
    lines.append("")
    lines.append("| phase | achieved TFLOPs/s | peak TFLOPs/s | util | read |")
    lines.append("|---|---:|---:|---:|---|")
    def _ach(ms, gf):
        if ms and ms > 0 and gf:
            return f"{(gf/1e3)/(ms/1e3):.1f}"
        return "—"
    lines.append(f"| prefix (1 call) | {_ach(t['prefix']['mean_ms'], a['prefix_gflops'])} | {peak} | {prefix_util*100:.1f}% | {prefix_bound} |")
    lines.append(f"| expert (1 step) | {_ach(t['denoise_per_step_mean'][0], a['expert_per_step_gflops'])} | {peak} | {per_step_util*100:.1f}% | {per_step_bound} |")
    lines.append(f"| expert 10 steps | {_ach(t['denoise_sum_10_steps']['mean_ms'], a['expert_total_10_steps_gflops'])} | {peak} | {expert_total_util*100:.1f}% | {expert_total_bound} |")
    lines.append(f"| end-to-end | {_ach(t['total']['mean_ms'], a['prefix_gflops'] + a['expert_total_10_steps_gflops'])} | {peak} | {e2e_util*100:.1f}% | {e2e_bound} |")
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("- util > 55% → compute-bound (matmuls saturate the SMs; quantization → INT4 matmul speedups help)")
    lines.append("- util < 8% → latency- or memory-bound (small-batch overhead; quantization helps mainly via memory traffic)")
    lines.append("- 8–55% → mixed; typical for single-batch transformer inference on H100")
    lines.append("")

    # --- Headline ---
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- Per observation: VLM prefix is **{t['prefix_frac_of_total_mean']*100:.1f}%** of wall time "
        f"and {100*a['prefix_gflops']/(a['prefix_gflops']+a['expert_total_10_steps_gflops']):.1f}% of FLOPs. "
        f"The 10-step expert loop is **{t['denoise_frac_of_total_mean']*100:.1f}%** of wall time "
        f"and {100*a['expert_total_10_steps_gflops']/(a['prefix_gflops']+a['expert_total_10_steps_gflops']):.1f}% of FLOPs. "
        f"Residual (preprocessing, framework) is {t['residual_frac_of_total_mean']*100:.1f}%.")
    lines.append("")

    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    utils.log(f"[exp9] markdown → {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-obs", type=int, default=20,
                   help="How many distinct LIBERO observations to rotate through.")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-measure", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Ignored for now; pi0.5 policy.infer uses batch=1 natively.")
    p.add_argument("--smoke", action="store_true",
                   help="1 obs, 1 warmup, 3 measured iters, 1 profiler pass.")
    p.add_argument("--jsonl", default=None)
    p.add_argument("--md", default=None)
    args = p.parse_args()

    if args.smoke:
        args.n_obs = 1
        args.n_warmup = 1
        args.n_measure = 3

    utils.setup_logging()
    utils.log("=" * 60)
    utils.log("EXP9: pi0.5 FLOPs + wall-time benchmark (VLM prefix vs expert loop)")
    utils.log("=" * 60)
    gpu = gpu_info()
    utils.log(f"[exp9] GPU: {gpu.get('name')} ({gpu.get('variant')})")
    utils.log(f"[exp9] nvidia-smi:\n{gpu.get('nvidia_smi','')}")

    # Load model
    with utils.Timer("load_policy"):
        policy, model = utils.load_policy("pi05_libero")
    model.eval()
    utils.log(f"[exp9] {utils.gpu_mem_str()}")

    # Parameter counts
    params = module_param_counts(model)
    utils.log(f"[exp9] params: total={params['total']:,}  vlm_lang={params['paligemma_language_model']:,}  "
              f"vision={params['vision_tower']:,}  expert={params['gemma_expert']:,}")

    # Load observations
    n_easy = args.n_obs // 2
    n_hard = args.n_obs - n_easy
    try:
        observations, _meta = utils.load_libero_observations(n_easy=n_easy, n_hard=n_hard)
    except Exception as e:
        utils.log(f"[exp9] FATAL: load_libero_observations failed: {e}")
        return 1
    utils.log(f"[exp9] loaded {len(observations)} observations")

    jsonl_path = args.jsonl or os.path.join(utils.RESULTS_DIR, "exp9_flops_benchmark.jsonl")
    md_path    = args.md    or os.path.join(utils.RESULTS_DIR, "exp9_flops_benchmark.md")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    # Wall-time pass
    rows = bench_wall_time(policy, model, observations,
                           n_warmup=args.n_warmup, n_measure=args.n_measure,
                           jsonl_path=jsonl_path)
    timing_summary = summarize_timing(rows)

    # Sanity gate
    for r in rows:
        n_steps = len(r["denoise_steps_ms"])
        if n_steps != 10:
            utils.log(f"[exp9] WARNING: iter {r['iter']} had {n_steps} denoise steps (expected 10)")

    # FLOPs pass (one observation is enough; we want the analytical shape anyway)
    utils.log("[exp9] running FLOPs pass (torch.profiler + analytical)...")
    flops_summary = bench_flops(policy, model, observations[0])

    # Markdown
    cfg = {"n_obs": len(observations), "n_warmup": args.n_warmup, "n_measure": args.n_measure}
    write_markdown(md_path, cfg, gpu, params, timing_summary, flops_summary, args.batch_size)

    # Dump a summary JSON alongside
    summary_json = os.path.join(utils.RESULTS_DIR, "exp9_flops_benchmark_summary.json")
    with open(summary_json, "w") as f:
        json.dump({
            "cfg": cfg, "gpu": gpu, "params": params,
            "timing": timing_summary, "flops": flops_summary,
            "batch_size": args.batch_size,
        }, f, indent=2, default=str)
    utils.log(f"[exp9] summary → {summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
