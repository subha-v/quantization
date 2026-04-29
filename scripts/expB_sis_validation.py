#!/usr/bin/env python3
"""
ExpB — Saliency-Aware PTQ Validation (SIS-gated FP16 rescue).

Counterfactual frame-selection test for SQIL's State Importance Score (SIS),
adapted for our PTQ setup: instead of using SIS to weight a QAT loss, we use
it as a per-frame precision gate that overrides W2-with-protection inference
with FP16 inference at high-SIS frames.

Eight matched conditions per (suite, task_id, seed, episode_idx) at frac=0.5:
  1. FP16          — pure FP16 (ceiling, = exp0)
  2. W2            — pure w2_vlm_protect (floor)
  3. SIS-top       — W2 base + FP16 override at top-50% by SIS
  4. Random        — W2 base + FP16 override at random 50% (per-rollout seeded)
  5. Bottom-SIS    — W2 base + FP16 override at bottom-50% by SIS (symmetry control)
  6. MSE-W2traj    — W2 base + FP16 override at top-50% by ‖a_FP-a_W2‖² on W2 trajectory states
  7. MSE-FP16traj  — W2 base + FP16 override at top-50% by ‖a_FP-a_W2‖² on FP16 trajectory states (NEW)
  8. AttnEntropy   — W2 base + FP16 override at bottom-50% by l12h2 entropy (low entropy → high sensitivity per D2 ρ=-0.29)

Phase D: pilot (20 seeds × {1,2,3,4,5} on libero_10 task 0) — kill switch on SR(3) ≤ SR(4).
Phase E: full (100 seeds × all 7) — 50 Long + 50 Object.
Phase F: analysis via --analyze.

Usage:
  python expB_sis_validation.py --smoke                              # 1 seed, all 7
  python expB_sis_validation.py --pilot                              # 20 seeds × 5 conds
  python expB_sis_validation.py --full                               # 100 seeds × 7 conds
  python expB_sis_validation.py --analyze                            # markdown summary
  python expB_sis_validation.py --suite Long --task-id 0 --seeds 0 1 --conditions all
"""

import argparse
import os
import random as _random
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import utils
import rollout as rollout_mod
from exp3_flow_step_sensitivity import infer_with_noise, make_noise, get_action_shape
from sis_utils import (
    PrecisionController,
    L12H2EntropyHook,
    AttentionMetricHook,
    IntraPassController,
    PROBE_DIRECTION_BY_TAG,
    compute_sis,
    cycle_noise,
    parse_probe_tag,
    format_probe_tag,
)


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(utils.RESULTS_DIR)
DIAG_PATH = RESULTS_DIR / "expB_diagnostic.jsonl"
DIAG_V2_PATH = RESULTS_DIR / "expB_diagnostic_v2.jsonl"   # augmented with W2-pass entropy
DIAG_V3_PATH = RESULTS_DIR / "expB_diagnostic_v3.jsonl"   # multi-probe + W4-pass diagnostic (2026-04-29)
FP16_DIAG_PATH = RESULTS_DIR / "expB_fp16_diagnostic.jsonl"
ROLLOUT_PATH = RESULTS_DIR / "expB_rollouts.jsonl"
SCHEMES_ROLLOUT_PATH = RESULTS_DIR / "expB_schemes_rollouts.jsonl"  # S1/S2 (W2 base) conditions
W4_ROLLOUT_PATH = RESULTS_DIR / "expB_w4_rollouts.jsonl"            # W4-base conditions (2026-04-29)
SUMMARY_PATH = RESULTS_DIR / "expB_summary.md"
SCHEMES_SUMMARY_PATH = RESULTS_DIR / "expB_schemes_summary.md"
W4_SUMMARY_PATH = RESULTS_DIR / "expB_w4_summary.md"

# ---------------------------------------------------------------------------
# Conditions
# Original 8 (binary, FP16-pass entropy or oracle):
# Plus 5 deployable Scheme 1 / Scheme 2 conditions added 2026-04-22.
#
# Scheme 1 (S1): one-frame-lag — at cycle t, decision uses W2-pass entropy from
#   cycle t-1; cycle 0 defaults to FP16. Zero extra forward passes vs W2 baseline.
# Scheme 2 (S2): speculative — at cycle t, decision uses W2-pass entropy from
#   cycle t itself; if non-W2 chosen, deployed system pays a re-run on that cycle
#   (we measure rescue rate; the bandwidth tax is reported separately).
#
# Granularities:
#   binary  — {W2, FP16} at frac=0.5 (50% FP16, avg ≈ 9 bits)
#   ternary — {W2, W4, FP16} at 50/30/20 (50% W2, 30% W4, 20% FP16, avg ≈ 5.4 bits)
# ---------------------------------------------------------------------------
ALL_CONDITIONS = [
    "FP16",          # 1 — full mask, ceiling
    "W2",            # 2 — empty mask, floor
    "SIS-top",       # 3 — top-frac% by SIS
    "Random",        # 4 — random frac% (per-rollout seeded), null
    "Bottom-SIS",    # 5 — bottom-frac% by SIS, symmetry control
    "MSE-W2traj",    # 6 — top-frac% by ‖a_FP-a_W2‖² on W2-traj states
    "MSE-FP16traj",  # 7 — top-frac% by ‖a_FP-a_W2‖² on FP16-traj states
    "AttnEntropy",   # 8 — bottom-frac% by FP16-pass l12h2 entropy (legacy oracle-deployable)
    # ---- 2026-04-22: deployable schemes (W2-pass entropy, three-tier optional) ----
    "S1-Bin",        # 9 — Scheme 1 lag-1 binary {W2, FP16} from W2-pass entropy
    "S2-Bin",        # 10 — Scheme 2 speculative binary {W2, FP16} from W2-pass entropy
    "S1-Tern",       # 11 — Scheme 1 lag-1 ternary {W2, W4, FP16} from W2-pass entropy
    "S2-Tern",       # 12 — Scheme 2 speculative ternary {W2, W4, FP16} from W2-pass entropy
    "Random-Tern",   # 13 — random ternary partition (matches S1/S2-Tern fractions), null
    # ---- 2026-04-29: W4-first deployable conditions ----
    # Tier 0: baseline characterization
    "W4-Floor",          # 14 — pure W4-with-protection (no schedule, no overrides)
    "W4-Static-Sched",   # 15 — exp3-style static expert step schedule on W4-protect VLM (placeholder; not yet wired)
    # Tier 1: W4 base + FP16 rescue (binary)
    "Random-W4",                    # 16 — random binary mask on W4 base
    "AttnEntropy-W4",               # 17 — bottom-frac% by FP16-pass l12h2 entropy on W4 base (oracle)
    "S1-Bin-W4",                    # 18 — lag-1 binary on W4 base, W4-pass l12h2 entropy
    "S2-Bin-W4",                    # 19 — speculative binary on W4 base, W4-pass l12h2 entropy
    "S3-Bin-W4-l1h7-top1",          # 20 — intra-pass at l1h7 top1, W4 base → FP16 escalation
    "S3-Bin-W4-l9h2-ent",           # 21 — intra-pass at l9h2 entropy, W4 base → FP16 escalation
    "S3-Bin-W4-l12h2-ent",          # 22 — intra-pass at l12h2 entropy, W4 base → FP16 escalation
    # Tier 2: W4 base + ternary {W2, W4, FP16}
    "S1-Tern-W4",                   # 23 — lag-1 ternary on W4 base
    "S2-Tern-W4",                   # 24 — speculative ternary on W4 base
    "S3-Tern-W4-l12h2",             # 25 — intra-pass three-tier at l12h2 (W2 ↓ W4 base ↑ FP16)
    "Random-Tern-W4",               # 26 — random ternary partition on W4 base, null
    # Tier 3: candidate-readout sweep (lag-1 binary at different probes, W4 base)
    "Probe-W4-l1h7-top1",           # 27
    "Probe-W4-l9h2-ent",            # 28
    "Probe-W4-l3h4-top5",           # 29
    "Probe-W4-l17h4-top1",          # 30
    # ---- 2026-04-29 mid-Tier-0 finding: at W4 base, l12h2-ent ρ flipped vs W2.
    # Pooled n=1806 cycles (35 Long trials) showed ρ(W4-pass l12h2-ent, mse_fp_w4) = +0.172
    # (W2 D2 finding had ρ = -0.294, opposite direction). Add -top variants to test
    # whether high-entropy = high-W4-sensitivity is the right direction at W4.
    "AttnEntropy-W4-top",           # 31 — FP16-pass oracle, top-frac% by entropy
    "S1-Bin-W4-top",                # 32 — lag-1 binary, W4-pass entropy, top direction
    "S2-Bin-W4-top",                # 33 — speculative binary, W4-pass entropy, top direction
    "S3-Bin-W4-l12h2-ent-top",      # 34 — intra-pass at l12h2 ent, top direction
    # ---- Tier 4 direction-flipped ternary (added 2026-04-29 mid-run; queue for follow-up) ----
    # Tests whether direction-flip rescues the broken S1/S2-Tern.
    # If S1-Tern-W4-top SR matches W4-Floor (96.7%) while bottom dir collapsed to 78%,
    # the direction-flip story is confirmed for ternary too.
    "S1-Tern-W4-top",               # 35 — lag-1 ternary, top direction
    "S2-Tern-W4-top",               # 36 — speculative ternary, top direction
    "S3-Tern-W4-l12h2-top",         # 37 — intra-pass three-tier at l12h2, top direction
    # ---- Tier 5 (added 2026-04-29 post-Tier-4): l1h7 with W4-correct direction (bottom) ----
    # Tier 1+2+3 ran S3-Bin-W4-l1h7-top1 with the W2-default direction "top" (ρ=+0.26 at W2).
    # At W4 the per-trial mean ρ is -0.155 — strongest of all 5 candidate probes — but the
    # CORRECT direction is "bottom". The earlier-layer position means swapping layers 2-17
    # (vs l12h2's 13-17) → much larger compute savings. May Pareto-dominate S3-Tern-W4-l12h2
    # at lower avg_bits / matching SR.
    "S3-Bin-W4-l1h7-bottom",        # 38 — intra-pass at l1h7 top1, BOTTOM direction (W4-correct)
    "S3-Tern-W4-l1h7-bottom",       # 39 — intra-pass three-tier at l1h7 top1, BOTTOM direction
]

# Legacy pilot conditions (frac=0.5 results live under old labels in the previous JSONLs)
PILOT_CONDITIONS = ["FP16", "W2", "SIS-top", "Random", "MSE-W2traj"]
# W2-base deployable schemes (the 2026-04-22 plan); needs attn_entropy_l12h2_w2.
SCHEMES_CONDITIONS = ["S1-Bin", "S2-Bin", "S1-Tern", "S2-Tern", "Random-Tern"]
# Conditions that require attn_entropy_l12h2_w2 in the diagnostic (the augmented field).
NEEDS_W2_PASS_ENTROPY = {"S1-Bin", "S2-Bin", "S1-Tern", "S2-Tern", "Random-Tern"}
# W4-base conditions added 2026-04-29.
W4_BASELINE_CONDITIONS = {"W4-Floor", "W4-Static-Sched", "FP16"}
W4_BIN_CONDITIONS = {
    "Random-W4", "AttnEntropy-W4", "S1-Bin-W4", "S2-Bin-W4",
    # 2026-04-29 mid-Tier-0: direction-flipped variants (top instead of bottom)
    "AttnEntropy-W4-top", "S1-Bin-W4-top", "S2-Bin-W4-top",
}
W4_INTRAPASS_CONDITIONS = {
    "S3-Bin-W4-l1h7-top1", "S3-Bin-W4-l9h2-ent", "S3-Bin-W4-l12h2-ent",
    "S3-Tern-W4-l12h2",
    # 2026-04-29 mid-Tier-0: direction-flipped intra-pass at l12h2
    "S3-Bin-W4-l12h2-ent-top",
    # 2026-04-29 Tier 4: direction-flipped intra-pass ternary at l12h2
    "S3-Tern-W4-l12h2-top",
    # 2026-04-29 Tier 5: l1h7 with W4-correct (bottom) direction
    "S3-Bin-W4-l1h7-bottom", "S3-Tern-W4-l1h7-bottom",
}
W4_TERN_CONDITIONS = {
    "S1-Tern-W4", "S2-Tern-W4", "Random-Tern-W4",
    # 2026-04-29 Tier 4: direction-flipped ternary variants
    "S1-Tern-W4-top", "S2-Tern-W4-top",
}
W4_PROBE_CONDITIONS = {
    "Probe-W4-l1h7-top1", "Probe-W4-l9h2-ent",
    "Probe-W4-l3h4-top5", "Probe-W4-l17h4-top1",
}
# All W4 conditions that need the V3 diagnostic (W4-pass entropy + multi-probe).
W4_ALL_CONDITIONS = (
    {"W4-Floor", "W4-Static-Sched"}
    | W4_BIN_CONDITIONS
    | W4_INTRAPASS_CONDITIONS
    | W4_TERN_CONDITIONS
    | W4_PROBE_CONDITIONS
)
# Mapping from intra-pass condition name to (layer, head, metric) — used to
# instantiate IntraPassController with the right probe.
INTRAPASS_PROBE_BY_CONDITION = {
    "S3-Bin-W4-l1h7-top1":      (1, 7, "top1"),
    "S3-Bin-W4-l9h2-ent":       (9, 2, "entropy"),
    "S3-Bin-W4-l12h2-ent":      (12, 2, "entropy"),
    "S3-Tern-W4-l12h2":         (12, 2, "entropy"),
    "S3-Bin-W4-l12h2-ent-top":  (12, 2, "entropy"),
    "S3-Tern-W4-l12h2-top":     (12, 2, "entropy"),
    # 2026-04-29 Tier 5: l1h7 with W4-correct (bottom) direction
    "S3-Bin-W4-l1h7-bottom":    (1, 7, "top1"),
    "S3-Tern-W4-l1h7-bottom":   (1, 7, "top1"),
}
# Per-condition direction override. None → use PROBE_DIRECTION_BY_TAG default
# (typically "bottom" for entropy probes per the W2 D2 finding).
# "top" → flip direction (high metric → escalate). Used for the 2026-04-29
# mid-Tier-0 finding that l12h2-ent ρ flipped sign at W4 base.
CONDITION_DIRECTION_OVERRIDE = {
    "AttnEntropy-W4-top":       "top",
    "S1-Bin-W4-top":            "top",
    "S2-Bin-W4-top":            "top",
    "S3-Bin-W4-l12h2-ent-top":  "top",
    # 2026-04-29 Tier 4: direction-flipped ternary
    "S1-Tern-W4-top":           "top",
    "S2-Tern-W4-top":           "top",
    "S3-Tern-W4-l12h2-top":     "top",
    # 2026-04-29 Tier 5: l1h7 W4-correct direction (bottom; opposite of W2 default "top")
    "S3-Bin-W4-l1h7-bottom":    "bottom",
    "S3-Tern-W4-l1h7-bottom":   "bottom",
}
# Mapping from probe condition to (layer, head, metric).
PROBE_BY_CONDITION = {
    "Probe-W4-l1h7-top1":   (1, 7, "top1"),
    "Probe-W4-l9h2-ent":    (9, 2, "entropy"),
    "Probe-W4-l3h4-top5":   (3, 4, "top5"),
    "Probe-W4-l17h4-top1":  (17, 4, "top1"),
}
# Default candidate readouts captured in the V3 diagnostic per cycle.
DEFAULT_CANDIDATE_READOUTS = ["l1h7-top1", "l9h2-ent", "l12h2-ent", "l3h4-top5", "l17h4-top1"]

# Bits per param for cost reporting; FP16 = 16, W4 = 4, W2 = 2.
BITS_BY_PRECISION = {"fp16": 16, "w4": 4, "w2": 2}


# ---------------------------------------------------------------------------
# Per-rollout deterministic-noise context manager
# ---------------------------------------------------------------------------
class SeededInferContext:
    """Monkey-patches policy.infer for the duration of a rollout so that calls
    with `noise=None` (i.e., the rollout harness's own calls) get deterministic
    cycle-indexed noise. Calls that already pass `noise=` (our diagnostic
    forwards) are passed through unchanged and do NOT advance the counter.

    This makes condition-vs-condition comparisons noise-matched per (base_seed,
    cycle_idx); only the precision schedule differs across conditions.
    """

    def __init__(self, policy, model, base_seed: int):
        self.policy = policy
        self.model = model
        self.base_seed = int(base_seed)
        self._original = None
        self.cycle_counter = -1

    def peek_next_seed(self) -> int:
        return self.base_seed * 100_000 + (self.cycle_counter + 1)

    def __enter__(self):
        ah, ad = get_action_shape(self.model)
        device = next(self.model.parameters()).device
        ctx = self
        original = self.policy.infer
        self._original = original

        def patched(obs, noise=None, **kwargs):
            if noise is None:
                ctx.cycle_counter += 1
                seed = ctx.base_seed * 100_000 + ctx.cycle_counter
                noise_t = make_noise(ah, ad, seed=seed, device=device)
                noise = noise_t.cpu().numpy().astype(np.float32)
            return original(obs, noise=noise, **kwargs)

        self.policy.infer = patched
        return self

    def __exit__(self, *_):
        if self._original is not None:
            self.policy.infer = self._original
            self._original = None


# ---------------------------------------------------------------------------
# Diagnostic rollout — produces condition-1 outcome + per-cycle scores
# ---------------------------------------------------------------------------
def diagnostic_rollout(
    policy,
    model,
    ctrl: PrecisionController,
    attn_hook: L12H2EntropyHook,
    suite: str,
    task_id: int,
    seed: int,
    episode_idx: int,
    n_grid: int = 4,
    sigma: float = 8.0,
    sis_stride: int = 4,
    verbose: bool = False,
):
    """Run a W2-base rollout that also collects per-cycle SIS, l12h2 attention
    entropy, and ||a_FP - a_W2||² (oracle MSE) for every inference cycle.

    SIS perturbation passes are the dominant cost (n_grid² FP16 forwards).
    `sis_stride` amortizes them: SIS is fully recomputed every `sis_stride`
    cycles and reused (with the most recent value) for the in-between cycles
    — same trick as the SQIL paper's k=4 in supplementary §3, table 12.
    Per-cycle a_FP / attn entropy / a_W2 / oracle MSE are still computed every
    cycle (cheap: 2 forward passes).

    Returns (rollout_record, per_cycle_records). The rollout's executed
    trajectory is condition-1 (pure W2-with-protection).
    """
    per_cycle = []
    cycle_idx = [-1]
    last_sis = [float("nan")]

    def obs_callback(t, libero_obs, openpi_obs):
        cycle_idx[0] += 1
        i = cycle_idx[0]
        sis_noise_seed = seeded.peek_next_seed()
        ah, ad = get_action_shape(model)
        device = next(model.parameters()).device
        noise_t = make_noise(ah, ad, seed=sis_noise_seed, device=device)
        noise_np = noise_t.cpu().numpy().astype(np.float32)

        ctrl.use_fp16()
        attn_hook.reset()
        a_FP = infer_with_noise(policy, openpi_obs, noise_np)
        attn_e = attn_hook.get_last_entropy_h2()

        # SIS only on stride-aligned cycles; reuse last value otherwise.
        is_stride_cycle = (i % sis_stride == 0)
        if is_stride_cycle:
            sis, _ = compute_sis(
                policy, openpi_obs, noise_np,
                n_grid=n_grid, sigma=sigma, a_clean=a_FP,
            )
            last_sis[0] = float(sis)
        sis_value = last_sis[0]

        ctrl.use_quant()
        attn_hook.reset()  # clear FP16-pass value so the next read is purely W2
        a_W2 = infer_with_noise(policy, openpi_obs, noise_np)
        attn_e_w2 = attn_hook.get_last_entropy_h2()  # W2-pass entropy (deployable signal)
        mse = float(np.mean((a_FP - a_W2) ** 2))

        per_cycle.append({
            "cycle_idx": i,
            "env_step": int(t),
            "sis": sis_value,
            "sis_recomputed": bool(is_stride_cycle),
            "attn_entropy_l12h2": float(attn_e) if attn_e == attn_e else None,
            "attn_entropy_l12h2_w2": float(attn_e_w2) if attn_e_w2 == attn_e_w2 else None,
            "mse_fp_w2": mse,
        })
        if verbose:
            tag = "S" if is_stride_cycle else "."
            utils.log(
                f"[diag {tag}] seed={seed} ep={episode_idx} cyc={i:3d} t={t:4d} "
                f"sis={sis_value:.5f} attn_e_fp={attn_e:.4f} attn_e_w2={attn_e_w2:.4f} mse={mse:.5f}"
            )

    def pre_infer_callback(t):
        # Make sure the executed rollout step uses W2-with-protection.
        ctrl.use_quant()

    # Outer SeededInferContext patches policy.infer to inject cycle-seeded noise.
    seeded = SeededInferContext(policy, model, base_seed=seed)
    with seeded:
        ctrl.use_quant()
        rec = rollout_mod.run_rollout(
            policy,
            task_id=task_id,
            suite=suite,
            seed=seed,
            episode_idx=episode_idx,
            obs_callback=obs_callback,
            pre_infer_callback=pre_infer_callback,
            verbose=False,
        )

    return rec, per_cycle


# ---------------------------------------------------------------------------
# FP16 diagnostic — pure FP16 rollout that ALSO records ‖a_FP - a_W2‖²
# at FP16-trajectory states (the mask source for MSE-FP16traj condition)
# ---------------------------------------------------------------------------
def fp16_diagnostic_rollout(
    policy,
    model,
    ctrl: PrecisionController,
    suite: str,
    task_id: int,
    seed: int,
    episode_idx: int,
    verbose: bool = False,
):
    """Run a pure FP16 rollout AND, at each cycle, also run a W2 forward pass
    at the same state with the same noise to record ‖a_FP - a_W2‖². The
    rollout's executed trajectory is condition `FP16` (= condition 1); the
    per-cycle MSE records are the mask source for `MSE-FP16traj` (= condition 7).

    No SIS or attention-entropy is computed here — those come from the W2
    diagnostic (which has the W2-trajectory states the SIS/attn metrics were
    designed for in the per-rollout 80th-percentile calibration).
    """
    per_cycle = []
    cycle_idx = [-1]

    def obs_callback(t, libero_obs, openpi_obs):
        cycle_idx[0] += 1
        sis_noise_seed = seeded.peek_next_seed()
        ah, ad = get_action_shape(model)
        device = next(model.parameters()).device
        noise_t = make_noise(ah, ad, seed=sis_noise_seed, device=device)
        noise_np = noise_t.cpu().numpy().astype(np.float32)

        # FP16 forward (matches the rollout's actual call below — we just need
        # a_FP for the MSE; the rollout will also call FP16 with same noise).
        ctrl.use_fp16()
        a_FP = infer_with_noise(policy, openpi_obs, noise_np)

        # W2 forward at the SAME FP16-trajectory state and SAME noise → MSE.
        ctrl.use_quant()
        a_W2 = infer_with_noise(policy, openpi_obs, noise_np)
        mse = float(np.mean((a_FP - a_W2) ** 2))

        per_cycle.append({
            "cycle_idx": cycle_idx[0],
            "env_step": int(t),
            "mse_fp_w2_fp16traj": mse,
        })
        if verbose:
            utils.log(
                f"[fp16-diag] seed={seed} ep={episode_idx} cyc={cycle_idx[0]:3d} "
                f"t={t:4d} mse={mse:.5f}"
            )

    def pre_infer_callback(t):
        # Actual rollout step uses FP16 (this is the FP16 condition).
        ctrl.use_fp16()

    seeded = SeededInferContext(policy, model, base_seed=seed)
    with seeded:
        ctrl.use_fp16()
        rec = rollout_mod.run_rollout(
            policy,
            task_id=task_id,
            suite=suite,
            seed=seed,
            episode_idx=episode_idx,
            obs_callback=obs_callback,
            pre_infer_callback=pre_infer_callback,
            verbose=False,
        )
    return rec, per_cycle


# ---------------------------------------------------------------------------
# Override rollout — replays seed with per-cycle precision schedule
# ---------------------------------------------------------------------------
def override_rollout(
    policy,
    model,
    ctrl: PrecisionController,
    suite: str,
    task_id: int,
    seed: int,
    episode_idx: int,
    override_set=None,
    precision_per_cycle=None,
    default_precision: str = "w2",
    verbose: bool = False,
):
    """Replay the same (suite, task_id, seed, episode_idx) starting state with a
    per-cycle weight precision schedule.

    Two equivalent ways to specify the schedule (legacy `override_set` kept for
    backwards compat with the original 8 conditions):

      override_set: set[int]          — binary {default, FP16}; cycles in the
                                        set go to FP16, others use `default_precision`.
      precision_per_cycle: dict[int,str] — full per-cycle map; values ∈
                                        {"fp16", "w4", "w2"}. Cycles not present
                                        in the map fall back to `default_precision`.

    Both default_precision values must be installable on `ctrl` (FP16 is always
    available; "w4" requires the controller was built with bits_list=(2, 4)).

    Returns the rollout_record.
    """
    if override_set is not None and precision_per_cycle is not None:
        raise ValueError("pass either override_set or precision_per_cycle, not both")
    if override_set is not None:
        # Convert binary mask to per-cycle dict for unified dispatch.
        precision_per_cycle = {int(c): "fp16" for c in override_set}
    if precision_per_cycle is None:
        precision_per_cycle = {}

    default_precision = default_precision.lower()
    if default_precision not in BITS_BY_PRECISION:
        raise ValueError(f"default_precision must be one of {list(BITS_BY_PRECISION)}; got {default_precision}")

    cycle_idx = [-1]

    def _set_precision(prec: str) -> None:
        prec = prec.lower()
        if prec == "fp16":
            ctrl.use_fp16()
        elif prec == "w2":
            ctrl.use_bits(2)
        elif prec == "w4":
            ctrl.use_bits(4)
        else:
            raise ValueError(f"unknown precision tag: {prec}")

    def pre_infer_callback(t):
        cycle_idx[0] += 1
        prec = precision_per_cycle.get(cycle_idx[0], default_precision)
        _set_precision(prec)

    seeded = SeededInferContext(policy, model, base_seed=seed)
    with seeded:
        rec = rollout_mod.run_rollout(
            policy,
            task_id=task_id,
            suite=suite,
            seed=seed,
            episode_idx=episode_idx,
            pre_infer_callback=pre_infer_callback,
            verbose=False,
        )
    return rec


def _avg_bits(precision_per_cycle: dict, n_cycles: int, default_precision: str) -> float:
    """Average bits per parameter across `n_cycles` cycles given the schedule."""
    if n_cycles <= 0:
        return float("nan")
    bits_default = BITS_BY_PRECISION[default_precision.lower()]
    total = 0.0
    for i in range(n_cycles):
        prec = precision_per_cycle.get(i)
        if prec is None:
            total += bits_default
        else:
            total += BITS_BY_PRECISION[prec.lower()]
    return total / n_cycles


# ---------------------------------------------------------------------------
# Mask construction from diagnostic per-cycle scores
# ---------------------------------------------------------------------------
def _topk_indices(scores, k: int, largest: bool = True) -> set:
    """Return the indices of the top-k (largest=True) or bottom-k scores.
    NaN scores are excluded — they cannot be ranked."""
    valid = [(i, s) for i, s in enumerate(scores) if s is not None and s == s]
    if not valid:
        return set()
    valid.sort(key=lambda x: -x[1] if largest else x[1])
    return {i for i, _ in valid[:k]}


def _rank_indices(scores, ascending: bool):
    """Return cycle indices sorted by score (ascending=True puts smallest first).
    NaN scores are dropped and ranked-out — they cannot be ordered."""
    valid = [(i, s) for i, s in enumerate(scores) if s is not None and s == s]
    valid.sort(key=lambda x: x[1] if ascending else -x[1])
    return [i for i, _ in valid]


def _ternary_assignment(
    ranked_low_to_high,
    n_total: int,
    frac_fp16: float,
    frac_w4: float,
    frac_w2: float,
):
    """Given indices ranked low-entropy-first (most-sensitive-first per D2 ρ<0),
    assign the bottom frac_fp16 → 'fp16', next frac_w4 → 'w4', remainder → 'w2'.
    Indices not appearing in `ranked_low_to_high` (e.g. NaN entropies) get 'w2'.

    Returns {cycle_idx: precision_str}. Cycles not in the dict default to 'w2'
    at override_rollout time (matches default_precision).
    """
    n_fp16 = max(0, int(round(frac_fp16 * n_total)))
    n_w4 = max(0, int(round(frac_w4 * n_total)))
    out = {}
    for k, idx in enumerate(ranked_low_to_high):
        if k < n_fp16:
            out[idx] = "fp16"
        elif k < n_fp16 + n_w4:
            out[idx] = "w4"
        else:
            out[idx] = "w2"
    return out


def _binary_assignment(ranked_low_to_high, n_total: int, frac_fp16: float):
    """Bottom frac_fp16 of the rank (most-sensitive-first) → 'fp16'; others → 'w2'."""
    n_fp16 = max(1, int(round(frac_fp16 * n_total)))
    out = {idx: "fp16" for idx in ranked_low_to_high[:n_fp16]}
    return out


def _lag_one(precision_per_cycle: dict, n_cycles: int) -> dict:
    """Apply one-frame lag to a per-cycle precision dict: cycle t inherits the
    decision that would have been made for cycle t-1's signal. Cycle 0 defaults
    to 'fp16' (bootstrap — no prior frame to read).
    """
    lagged = {0: "fp16"}
    for t in range(1, n_cycles):
        if (t - 1) in precision_per_cycle:
            lagged[t] = precision_per_cycle[t - 1]
    return lagged


def build_masks(
    per_cycle_w2,
    per_cycle_fp16,
    frac: float,
    seed: int,
    ternary_partition=(0.2, 0.3, 0.5),
) -> dict:
    """Build per-condition precision schedules from the diagnostic data.

    Returns dict[condition_name → schedule]. The schedule type depends on the
    condition's granularity:
      - Binary conditions return dict[cycle, "fp16"] for cycles in the FP16 set
        (`override_rollout` will call cycles missing from the dict at the default
        precision — typically W2). Legacy callers that expected `set` are still
        supported via `override_set` in `override_rollout`.
      - Ternary conditions return dict[cycle, "fp16"|"w4"|"w2"].

    `ternary_partition` = (frac_fp16, frac_w4, frac_w2). Must sum to ≤ 1; any
    remaining cycles default to W2.

    `per_cycle_w2` provides SIS, FP16-pass attn entropy, W2-pass attn entropy,
    and MSE-W2traj scores (cycles indexed in the W2 trajectory). `per_cycle_fp16`
    provides MSE-FP16traj scores (cycles indexed in the FP16 trajectory).
    """
    n_w2 = len(per_cycle_w2)
    if n_w2 == 0:
        return {}
    k_w2 = max(1, int(round(frac * n_w2)))

    sis      = [c["sis"] for c in per_cycle_w2]
    mse      = [c["mse_fp_w2"] for c in per_cycle_w2]
    attn_fp  = [c["attn_entropy_l12h2"] for c in per_cycle_w2]
    # The augmented field is None on legacy diagnostic rows; downstream code
    # gracefully skips conditions that need it when all values are None.
    attn_w2  = [c.get("attn_entropy_l12h2_w2") for c in per_cycle_w2]

    rng = _random.Random(seed)

    # ---- legacy binary masks (sets, kept for backwards compatibility) ----
    masks = {
        "FP16":                 set(range(n_w2)),
        "SIS-top":              _topk_indices(sis, k_w2, largest=True),
        "Random":               set(rng.sample(range(n_w2), k_w2)),
        "Bottom-SIS":           _topk_indices(sis, k_w2, largest=False),
        "MSE-W2traj":           _topk_indices(mse, k_w2, largest=True),
        # Low entropy → predicted high sensitivity (D2 finding ρ=-0.294); pick smallest entropy.
        "AttnEntropy":          _topk_indices(attn_fp, k_w2, largest=False),
        # Symmetry control: high entropy → predicted LOW sensitivity. If
        # AttnEntropy ≈ AttnEntropy-flipped, the direction has no signal.
        "AttnEntropy-flipped":  _topk_indices(attn_fp, k_w2, largest=True),
    }

    if per_cycle_fp16 is not None and len(per_cycle_fp16) > 0:
        n_fp = len(per_cycle_fp16)
        k_fp = max(1, int(round(frac * n_fp)))
        mse_fp = [c["mse_fp_w2_fp16traj"] for c in per_cycle_fp16]
        masks["MSE-FP16traj"] = _topk_indices(mse_fp, k_fp, largest=True)

    # ---- 2026-04-22: deployable Scheme 1 / Scheme 2 conditions ----
    # Only build these if the diagnostic has W2-pass entropy (augmented run).
    if any(v is not None and v == v for v in attn_w2):
        # Rank cycles low-entropy-first (= predicted-most-sensitive-first per D2 ρ<0).
        ranked_low = _rank_indices(attn_w2, ascending=True)

        frac_fp16, frac_w4, frac_w2 = ternary_partition
        if frac_fp16 + frac_w4 + frac_w2 > 1.0 + 1e-6:
            raise ValueError(f"ternary_partition fractions must sum to ≤ 1; got {ternary_partition}")

        # Scheme 2 (no-lag): apply assignment directly to W2-pass-entropy ranking.
        s2_bin = _binary_assignment(ranked_low, n_w2, frac_fp16=frac)
        s2_tern = _ternary_assignment(ranked_low, n_w2, frac_fp16, frac_w4, frac_w2)

        # Scheme 1 (one-frame-lag): same assignment but shift decisions by +1 cycle.
        s1_bin = _lag_one(s2_bin, n_w2)
        s1_tern = _lag_one(s2_tern, n_w2)

        # Random ternary control: independent random partition matching the same fractions.
        rng_t = _random.Random(seed * 7919)  # distinct stream from binary Random
        all_idx = list(range(n_w2))
        rng_t.shuffle(all_idx)
        rand_ternary = _ternary_assignment(all_idx, n_w2, frac_fp16, frac_w4, frac_w2)

        masks["S1-Bin"]      = s1_bin
        masks["S2-Bin"]      = s2_bin
        masks["S1-Tern"]     = s1_tern
        masks["S2-Tern"]     = s2_tern
        masks["Random-Tern"] = rand_ternary

    return masks


# ===========================================================================
# W4-first additions (2026-04-29)
# ===========================================================================
# All W4-base experiment plumbing lives below. The legacy W2-base functions
# above are unchanged.
# ---------------------------------------------------------------------------

def diagnostic_rollout_v3(
    policy,
    model,
    ctrl: PrecisionController,
    legacy_attn_hook: L12H2EntropyHook,
    probe_hooks: dict,             # {tag: AttentionMetricHook}
    suite: str,
    task_id: int,
    seed: int,
    episode_idx: int,
    base_precision: str = "w4",
    n_grid: int = 4,
    sigma: float = 8.0,
    sis_stride: int = 4,
    verbose: bool = False,
):
    """V3 diagnostic — supports W4 base AND multi-probe per-cycle attention metrics.

    For each cycle:
      1. FP16 forward at the cycle's state with seeded noise → a_FP, all probes
         populate `_per_pass_fp16` snapshot.
      2. Base-precision forward at the same state, same noise → a_base, all probes
         populate `_per_pass_base` snapshot. (base_precision = "w4" or "w2")
      3. Per-cycle MSE = ‖a_FP − a_base‖² (saved as `mse_fp_base`).

    Per-cycle JSONL row format:
        {
          cycle_idx, env_step, sis, sis_recomputed, base_precision,
          mse_fp_base,
          attn_entropy_l12h2,        # FP16-pass legacy hook (back-compat with v2)
          attn_entropy_l12h2_base,   # base-pass legacy hook
          attn_probes_fp16: {tag: float, ...},
          attn_probes_base: {tag: float, ...},
        }

    The rollout's executed trajectory is base_precision (W4-with-protection or
    W2-with-protection per `base_precision`).
    """
    base = base_precision.lower()
    if base not in ("w2", "w4"):
        raise ValueError(f"base_precision must be 'w2' or 'w4'; got {base!r}")
    base_bits = 2 if base == "w2" else 4

    per_cycle = []
    cycle_idx = [-1]
    last_sis = [float("nan")]

    def _probe_snapshot() -> dict:
        out = {}
        for tag, h in probe_hooks.items():
            out[tag] = float(h.get_last())
        return out

    def _reset_all_probes():
        if legacy_attn_hook is not None:
            legacy_attn_hook.reset()
        for h in probe_hooks.values():
            h.reset()

    def obs_callback(t, libero_obs, openpi_obs):
        cycle_idx[0] += 1
        i = cycle_idx[0]
        sis_noise_seed = seeded.peek_next_seed()
        ah, ad = get_action_shape(model)
        device = next(model.parameters()).device
        noise_t = make_noise(ah, ad, seed=sis_noise_seed, device=device)
        noise_np = noise_t.cpu().numpy().astype(np.float32)

        # ---- FP16 pass ----
        ctrl.use_fp16()
        _reset_all_probes()
        a_FP = infer_with_noise(policy, openpi_obs, noise_np)
        attn_fp16_legacy = legacy_attn_hook.get_last_entropy_h2() if legacy_attn_hook else float("nan")
        probes_fp16 = _probe_snapshot()

        # SIS only on stride-aligned cycles; reuse last value otherwise.
        is_stride_cycle = (i % sis_stride == 0)
        if is_stride_cycle:
            sis_val, _ = compute_sis(
                policy, openpi_obs, noise_np,
                n_grid=n_grid, sigma=sigma, a_clean=a_FP,
            )
            last_sis[0] = float(sis_val)
        sis_value = last_sis[0]

        # ---- Base-precision pass ----
        ctrl.use_bits(base_bits)
        _reset_all_probes()
        a_base = infer_with_noise(policy, openpi_obs, noise_np)
        attn_base_legacy = legacy_attn_hook.get_last_entropy_h2() if legacy_attn_hook else float("nan")
        probes_base = _probe_snapshot()
        mse = float(np.mean((a_FP - a_base) ** 2))

        per_cycle.append({
            "cycle_idx": i,
            "env_step": int(t),
            "base_precision": base,
            "sis": sis_value,
            "sis_recomputed": bool(is_stride_cycle),
            "attn_entropy_l12h2": float(attn_fp16_legacy) if attn_fp16_legacy == attn_fp16_legacy else None,
            f"attn_entropy_l12h2_{base}": float(attn_base_legacy) if attn_base_legacy == attn_base_legacy else None,
            "attn_probes_fp16": probes_fp16,
            f"attn_probes_{base}": probes_base,
            f"mse_fp_{base}": mse,
        })
        if verbose:
            tag = "S" if is_stride_cycle else "."
            utils.log(
                f"[diag-v3-{base} {tag}] seed={seed} ep={episode_idx} cyc={i:3d} t={t:4d} "
                f"sis={sis_value:.5f} attn_fp={attn_fp16_legacy:.4f} attn_{base}={attn_base_legacy:.4f} mse={mse:.5f}"
            )

    def pre_infer_callback(t):
        # The executed rollout step uses base precision.
        ctrl.use_bits(base_bits)

    seeded = SeededInferContext(policy, model, base_seed=seed)
    with seeded:
        ctrl.use_bits(base_bits)
        rec = rollout_mod.run_rollout(
            policy,
            task_id=task_id,
            suite=suite,
            seed=seed,
            episode_idx=episode_idx,
            obs_callback=obs_callback,
            pre_infer_callback=pre_infer_callback,
            verbose=False,
        )

    return rec, per_cycle


def intrapass_rollout(
    policy,
    model,
    ctrl: PrecisionController,
    intrapass_ctrl: IntraPassController,
    suite: str,
    task_id: int,
    seed: int,
    episode_idx: int,
    verbose: bool = False,
):
    """Replay (suite, task_id, seed, episode_idx) with intra-pass online
    precision control. Each cycle's `pre_infer` resets layers 1..MAX to the
    base precision; the IntraPassController's forward hook on layer L reads the
    attention metric and swaps layers L+1..MAX mid-forward when the metric
    crosses the per-rollout running quantile.

    Returns (rec, per_cycle_decisions) where per_cycle_decisions is a list
    of (metric_value, decision_str) tuples.
    """
    intrapass_ctrl.reset_per_rollout()

    def pre_infer_callback(t):
        intrapass_ctrl.pre_infer(t)

    seeded = SeededInferContext(policy, model, base_seed=seed)
    with seeded:
        # Initialize at base precision.
        intrapass_ctrl.pre_infer(0)
        rec = rollout_mod.run_rollout(
            policy,
            task_id=task_id,
            suite=suite,
            seed=seed,
            episode_idx=episode_idx,
            pre_infer_callback=pre_infer_callback,
            verbose=False,
        )
    return rec, list(intrapass_ctrl.cycle_decisions)


def _per_candidate_assignment(
    scores,
    n_total: int,
    frac: float,
    direction: str,
    granularity: str = "binary",
    ternary_partition=(0.2, 0.3, 0.5),
):
    """Build a per-cycle precision dict from a per-cycle score series.

    `direction='bottom'` means LOW scores predict HIGH sensitivity (escalate);
    'top' flips this.

    `granularity='binary'` returns dict[cycle, "fp16"] for the selected fraction.
    `granularity='ternary'` partitions: bottom-frac_fp16 → "fp16",
        next frac_w4 → "w4", remainder → "w2" (if direction='bottom'); flipped
        for direction='top'.
    """
    if direction not in ("bottom", "top"):
        raise ValueError(f"direction must be 'bottom' or 'top'; got {direction}")
    ranked = _rank_indices(scores, ascending=(direction == "bottom"))
    if granularity == "binary":
        k = max(1, int(round(frac * n_total)))
        return {idx: "fp16" for idx in ranked[:k]}
    elif granularity == "ternary":
        frac_fp16, frac_w4, frac_w2 = ternary_partition
        return _ternary_assignment(ranked, n_total, frac_fp16, frac_w4, frac_w2)
    else:
        raise ValueError(f"granularity must be 'binary' or 'ternary'; got {granularity}")


def build_masks_w4(
    per_cycle_v3,
    frac: float,
    seed: int,
    ternary_partition=(0.1, 0.4, 0.5),
) -> dict:
    """Build per-condition precision schedules for W4-base conditions from a
    V3 diagnostic.

    Reads each cycle's:
      - `attn_probes_w4` (dict tag → float)  for deployable per-cycle decision signal
      - `attn_probes_fp16` (dict tag → float)  for oracle baseline (AttnEntropy-W4)
      - `mse_fp_w4`                          for oracle/MSE-style fallback masks

    Returns dict[condition_name → schedule], where schedule is dict[int → "fp16"|"w4"|"w2"].
    Cycles not in the dict default to "w4" at override_rollout time.

    Note: S3-* conditions are NOT built here — they're handled by intrapass_rollout
    which doesn't use a precomputed mask.
    """
    n = len(per_cycle_v3)
    if n == 0:
        return {}

    rng = _random.Random(seed)
    masks: dict = {}

    # ---- Random binary on W4 base ----
    rand_set = set(rng.sample(range(n), max(1, int(round(frac * n)))))
    masks["Random-W4"] = {idx: "fp16" for idx in rand_set}

    # ---- AttnEntropy-W4: oracle deployable using FP16-pass entropy ----
    # bottom direction = W2 D2 default; top direction = 2026-04-29 W4 finding
    attn_fp16 = []
    for c in per_cycle_v3:
        probes = c.get("attn_probes_fp16", {})
        attn_fp16.append(probes.get("l12h2-ent"))
    masks["AttnEntropy-W4"] = _per_candidate_assignment(
        attn_fp16, n, frac, direction="bottom", granularity="binary",
    )
    masks["AttnEntropy-W4-top"] = _per_candidate_assignment(
        attn_fp16, n, frac, direction="top", granularity="binary",
    )

    # ---- S1-Bin-W4 / S2-Bin-W4: lag-1 / speculative on W4-pass l12h2 entropy ----
    attn_w4 = []
    for c in per_cycle_v3:
        probes = c.get("attn_probes_w4", {})
        attn_w4.append(probes.get("l12h2-ent"))
    # Bottom direction (W2 D2 default)
    s2_bin_w4 = _per_candidate_assignment(
        attn_w4, n, frac, direction="bottom", granularity="binary",
    )
    s1_bin_w4 = _lag_one(s2_bin_w4, n)
    masks["S1-Bin-W4"] = s1_bin_w4
    masks["S2-Bin-W4"] = s2_bin_w4
    # Top direction (W4 mid-Tier-0 finding: ρ flipped sign at W4 base)
    s2_bin_w4_top = _per_candidate_assignment(
        attn_w4, n, frac, direction="top", granularity="binary",
    )
    s1_bin_w4_top = _lag_one(s2_bin_w4_top, n)
    masks["S1-Bin-W4-top"] = s1_bin_w4_top
    masks["S2-Bin-W4-top"] = s2_bin_w4_top

    # ---- S1-Tern-W4 / S2-Tern-W4: lag-1 / speculative ternary on W4 base ----
    # bottom direction (W2 default)
    s2_tern_w4 = _per_candidate_assignment(
        attn_w4, n, frac, direction="bottom", granularity="ternary",
        ternary_partition=ternary_partition,
    )
    s1_tern_w4 = _lag_one(s2_tern_w4, n)
    masks["S1-Tern-W4"] = s1_tern_w4
    masks["S2-Tern-W4"] = s2_tern_w4
    # 2026-04-29 Tier 4: top direction (testing whether direction-flip rescues
    # the broken S1/S2-Tern bottom-dir result observed mid-Tier-1+2+3).
    s2_tern_w4_top = _per_candidate_assignment(
        attn_w4, n, frac, direction="top", granularity="ternary",
        ternary_partition=ternary_partition,
    )
    s1_tern_w4_top = _lag_one(s2_tern_w4_top, n)
    masks["S1-Tern-W4-top"] = s1_tern_w4_top
    masks["S2-Tern-W4-top"] = s2_tern_w4_top

    # ---- Random-Tern-W4: random ternary partition matching same fractions ----
    rng_t = _random.Random(seed * 7919)
    all_idx = list(range(n))
    rng_t.shuffle(all_idx)
    masks["Random-Tern-W4"] = _ternary_assignment(
        all_idx, n, *ternary_partition,
    )

    # ---- Probe-W4-* sweep: lag-1 binary at alternative readouts (W4-pass) ----
    for cond_name, (layer, head, metric) in PROBE_BY_CONDITION.items():
        tag = format_probe_tag(layer, head, metric)
        scores = []
        for c in per_cycle_v3:
            probes = c.get("attn_probes_w4", {})
            scores.append(probes.get(tag))
        direction = PROBE_DIRECTION_BY_TAG.get(tag, "bottom")
        s2_probe = _per_candidate_assignment(
            scores, n, frac, direction=direction, granularity="binary",
        )
        masks[cond_name] = _lag_one(s2_probe, n)  # lag-1 (Scheme 1) for cheap deployability

    return masks


def _avg_bits_w4(precision_per_cycle: dict, n_cycles: int) -> float:
    """Avg bits given a W4-base override schedule (default = W4)."""
    return _avg_bits(precision_per_cycle, n_cycles, default_precision="w4")


def _load_v3_diag_for_trial(diag_path: Path, suite: str, task_id: int, seed: int, episode_idx: int):
    """Load per-cycle V3 diagnostic rows for a single trial from JSONL.
    Returns sorted list (by cycle_idx) or empty list if no rows match.

    Used by run_seed_w4 to skip re-running the diagnostic when prior records
    exist (e.g., Tier 0 wrote them; Tier 1+2+3 reuses them)."""
    if not diag_path.exists():
        return []
    rows = utils.load_jsonl(diag_path)
    matching = [
        r for r in rows
        if r.get("suite") == suite and int(r.get("task_id", -1)) == int(task_id)
        and int(r.get("seed", -1)) == int(seed) and int(r.get("episode_idx", -1)) == int(episode_idx)
    ]
    matching.sort(key=lambda r: int(r.get("cycle_idx", -1)))
    return matching


def run_seed_w4(
    policy, model, ctrl,
    legacy_attn_hook,
    probe_hooks: dict,
    suite: str, task_id: int, seed: int, episode_idx: int,
    conditions,
    n_grid: int = 4, sigma: float = 8.0,
    sis_stride: int = 4,
    frac: float = 0.4,
    ternary_partition=(0.1, 0.4, 0.5),
    diag_path: Path = None,
    rollout_path: Path = None,
    reuse_cached_diag: bool = False,
    verbose: bool = False,
) -> list:
    """W4-base per-seed driver. Mirrors run_seed but routes through the V3
    diagnostic + W4 mask builder, and dispatches S3-* conditions to
    intrapass_rollout.

    `reuse_cached_diag=True` skips the diagnostic_rollout_v3 call when prior
    per-cycle records exist for this trial in `diag_path`. Used to chain
    Tier 0 → Tier 1+2+3 without redoing the diagnostic. The skipped diagnostic
    means W4-Floor / W4-Static-Sched conditions can't be re-emitted as rollout
    rows (we don't have a fresh rollout_record); those should already be in
    rollout_path from the prior run.
    """
    if diag_path is None:
        diag_path = DIAG_V3_PATH
    if rollout_path is None:
        rollout_path = W4_ROLLOUT_PATH

    out = []
    rollout_key = {
        "suite": suite, "task_id": int(task_id),
        "seed": int(seed), "episode_idx": int(episode_idx),
    }

    # Conditions that require the V3 W4 diagnostic (any condition that needs
    # per-cycle attention probes or per-cycle ‖a_FP - a_W4‖² targets).
    NEEDS_W4_DIAG = (
        W4_BIN_CONDITIONS | W4_TERN_CONDITIONS | W4_PROBE_CONDITIONS | {"AttnEntropy-W4"}
    )
    # Conditions that need an FP16 baseline rollout.
    NEEDS_FP16_BASELINE = {"FP16"}
    # Floor conditions that just need a base rollout (no overrides).
    BASELINE_CONDS = {"FP16", "W4-Floor", "W4-Static-Sched"}

    w4_rec, w4_per_cycle = None, None
    cached_per_cycle = []
    if reuse_cached_diag:
        cached_per_cycle = _load_v3_diag_for_trial(
            diag_path, suite, task_id, seed, episode_idx
        )

    if any(c in conditions for c in NEEDS_W4_DIAG | {"W4-Floor"}):
        if cached_per_cycle:
            utils.log(
                f"[expB-W4] W4-DIAG CACHED seed={seed} task={task_id} ep={episode_idx} "
                f"({len(cached_per_cycle)} cycles loaded from {diag_path.name})"
            )
            w4_per_cycle = cached_per_cycle
            # w4_rec stays None — we don't have the rollout_record. Code below
            # gracefully skips emitting W4-Floor/W4-Static-Sched rows when w4_rec is None.
        else:
            utils.log(f"[expB-W4] W4-DIAG seed={seed} task={task_id} ep={episode_idx}")
            t0 = time.time()
            w4_rec, w4_per_cycle = diagnostic_rollout_v3(
                policy, model, ctrl, legacy_attn_hook, probe_hooks,
                suite, task_id, seed, episode_idx,
                base_precision="w4",
                n_grid=n_grid, sigma=sigma, sis_stride=sis_stride,
                verbose=verbose,
            )
            utils.log(
                f"[expB-W4] W4-DIAG done seed={seed} success={w4_rec.success} "
                f"steps={w4_rec.steps} cycles={len(w4_per_cycle)} wall={time.time()-t0:.1f}s"
            )
            for c in w4_per_cycle:
                utils.append_jsonl({**rollout_key, **c}, diag_path)

    fp16_rec = None
    if "FP16" in conditions:
        utils.log(f"[expB-W4] FP16-BASELINE seed={seed} task={task_id} ep={episode_idx}")
        t0 = time.time()
        ctrl.use_fp16()
        seeded = SeededInferContext(policy, model, base_seed=seed)
        with seeded:
            fp16_rec = rollout_mod.run_rollout(
                policy,
                task_id=task_id, suite=suite,
                seed=seed, episode_idx=episode_idx,
                pre_infer_callback=lambda t: ctrl.use_fp16(),
                verbose=False,
            )
        utils.log(
            f"[expB-W4] FP16-BASELINE done seed={seed} success={fp16_rec.success} "
            f"steps={fp16_rec.steps} wall={time.time()-t0:.1f}s"
        )

    # Build W4-base masks (for non-S3 conditions).
    if w4_per_cycle:
        masks = build_masks_w4(
            w4_per_cycle, frac=frac, seed=seed,
            ternary_partition=ternary_partition,
        )
    else:
        masks = {}
    n_cycles = len(w4_per_cycle) if w4_per_cycle else 0

    for cond in conditions:
        rec = None
        precision_per_cycle = {}
        n_overrides = 0
        avg_bits = float("nan")
        intrapass_decisions = None

        if cond == "FP16":
            rec = fp16_rec
            avg_bits = float(BITS_BY_PRECISION["fp16"])

        elif cond == "W4-Floor":
            # Re-use the W4 diagnostic's executed trajectory (which IS W4-Floor).
            if w4_rec is None:
                # Cached-diag path: rollout already in rollout_path from prior run.
                utils.log(f"[expB-W4] skip W4-Floor — diagnostic cached (row should exist from prior tier)")
                continue
            rec = w4_rec
            avg_bits = float(BITS_BY_PRECISION["w4"])

        elif cond == "W4-Static-Sched":
            # The exp3 static expert step schedule on W4-protect VLM. The expert-
            # side step controller lives in expA; not yet ported here. For this
            # overnight we treat W4-Static-Sched the same as W4-Floor and flag it
            # so the analysis can highlight that step 9 wasn't separately handled.
            # TODO(expA-port): wire StepController for the expert.
            if w4_rec is None:
                utils.log(f"[expB-W4] skip W4-Static-Sched — diagnostic cached")
                continue
            utils.log(f"[expB-W4] WARN: W4-Static-Sched not yet wired (uses W4-Floor for now)")
            rec = w4_rec
            avg_bits = float(BITS_BY_PRECISION["w4"])

        elif cond in W4_INTRAPASS_CONDITIONS:
            layer, head, metric = INTRAPASS_PROBE_BY_CONDITION[cond]
            granularity = "ternary" if cond.startswith("S3-Tern") else "binary"
            tag = format_probe_tag(layer, head, metric)
            # Use per-condition override if set (e.g. for `-top` flipped variants);
            # otherwise fall back to PROBE_DIRECTION_BY_TAG default.
            direction = CONDITION_DIRECTION_OVERRIDE.get(
                cond, PROBE_DIRECTION_BY_TAG.get(tag, "bottom")
            )
            if granularity == "binary":
                intrapass_ctrl_obj = IntraPassController(
                    model, ctrl, layer_L=layer, head=head, metric=metric,
                    base_prec="w4", decision_high_prec="fp16",
                    decision_low_prec=None,
                    direction=direction, frac_high=frac, frac_low=0.0,
                )
            else:  # ternary
                # Three-tier: bottom-frac_high → fp16, top-frac_low → w2, middle → w4
                frac_fp16 = ternary_partition[0]
                frac_w2 = ternary_partition[2]
                intrapass_ctrl_obj = IntraPassController(
                    model, ctrl, layer_L=layer, head=head, metric=metric,
                    base_prec="w4", decision_high_prec="fp16",
                    decision_low_prec="w2",
                    direction=direction, frac_high=frac_fp16, frac_low=frac_w2,
                )
            try:
                t0 = time.time()
                utils.log(
                    f"[expB-W4] {cond} seed={seed} task={task_id} ep={episode_idx} "
                    f"L={layer} head={head} metric={metric} dir={direction} frac_high={frac_high if False else (frac if granularity == 'binary' else ternary_partition[0])}"
                )
                rec, intrapass_decisions = intrapass_rollout(
                    policy, model, ctrl, intrapass_ctrl_obj,
                    suite, task_id, seed, episode_idx,
                    verbose=verbose,
                )
                avg_bits = float(intrapass_ctrl_obj.avg_bits())
                n_overrides = intrapass_ctrl_obj.n_escalations() + intrapass_ctrl_obj.n_de_escalations()
                utils.log(
                    f"[expB-W4] {cond} done success={rec.success} steps={rec.steps} "
                    f"avg_bits={avg_bits:.2f} escalations={intrapass_ctrl_obj.n_escalations()} "
                    f"de-escalations={intrapass_ctrl_obj.n_de_escalations()} "
                    f"wall={time.time()-t0:.1f}s"
                )
            finally:
                intrapass_ctrl_obj.uninstall()

        else:
            # Override-style W4-base condition (Random-W4, AttnEntropy-W4, S1/S2-Bin/Tern-W4, Probe-W4-*)
            if cond not in masks:
                utils.log(f"[expB-W4] WARN: skipping {cond} — mask not available")
                continue
            schedule = masks[cond]
            if isinstance(schedule, set):
                precision_per_cycle = {int(c): "fp16" for c in schedule}
            else:
                precision_per_cycle = dict(schedule)
            n_overrides = sum(1 for p in precision_per_cycle.values() if p != "w4")
            avg_bits = _avg_bits_w4(precision_per_cycle, n_cycles)

            t0 = time.time()
            utils.log(
                f"[expB-W4] {cond} seed={seed} task={task_id} ep={episode_idx} "
                f"|overrides|={n_overrides}/{n_cycles} avg_bits={avg_bits:.2f}"
            )
            rec = override_rollout(
                policy, model, ctrl,
                suite, task_id, seed, episode_idx,
                precision_per_cycle=precision_per_cycle,
                default_precision="w4",
                verbose=verbose,
            )
            utils.log(
                f"[expB-W4] {cond} done success={rec.success} steps={rec.steps} "
                f"wall={time.time()-t0:.1f}s"
            )

        if rec is None:
            continue

        entry = {
            **rollout_key,
            "condition": cond,
            "base_precision": "w4",
            "success": bool(rec.success),
            "steps": int(rec.steps),
            "wall_time_s": float(rec.wall_time_s),
            "termination_reason": rec.termination_reason,
            "n_overrides": int(n_overrides),
            "n_cycles": int(n_cycles),
            "condition_avg_bits": float(avg_bits) if avg_bits == avg_bits else None,
            "ternary_partition": list(ternary_partition) if cond.endswith("-Tern-W4") or cond == "S3-Tern-W4-l12h2" else None,
        }
        if intrapass_decisions is not None:
            # Record the per-cycle decision string sequence so analysis can
            # recover realized avg_bits histograms and decision frequency.
            entry["intrapass_decisions"] = [d for _, d in intrapass_decisions]
            entry["intrapass_metric_values"] = [v for v, _ in intrapass_decisions]
        utils.append_jsonl(entry, rollout_path)
        out.append(entry)

    return out


def run_trials_w4(
    trials, conditions,
    candidate_readouts=None,
    n_grid=4, sigma=8.0, sis_stride=4,
    frac=0.4,
    ternary_partition=(0.1, 0.4, 0.5),
    diag_path: Path = None,
    rollout_path: Path = None,
    reuse_cached_diag: bool = False,
    verbose=False,
):
    """W4-base top-level driver. Loads policy + builds PrecisionController with
    bits_list=(2, 4) (need both for ternary), installs the multi-probe hook
    set, then runs each trial through run_seed_w4.

    `reuse_cached_diag=True` skips the V3 diagnostic per trial when prior records
    are present in `diag_path` (Tier 0 → Tier 1+2+3 chaining).
    """
    if candidate_readouts is None:
        candidate_readouts = list(DEFAULT_CANDIDATE_READOUTS)

    utils.log("[expB-W4] loading policy + model...")
    policy, model = utils.load_policy()

    utils.log("[expB-W4] building PrecisionController bits_list=(2, 4)...")
    ctrl = PrecisionController(model, bits_list=(2, 4), group_size=128)
    ctrl.use_fp16()

    utils.log("[expB-W4] installing legacy L12H2 hook + multi-probe hooks...")
    legacy_hook = L12H2EntropyHook(model)
    probe_hooks: dict = {}
    for tag in candidate_readouts:
        try:
            layer, head, metric = parse_probe_tag(tag)
        except ValueError as e:
            utils.log(f"[expB-W4] WARN: skipping bad probe tag {tag!r}: {e}")
            continue
        try:
            probe_hooks[tag] = AttentionMetricHook(model, layer, head, metric)
        except Exception as e:
            utils.log(f"[expB-W4] WARN: failed to install probe {tag!r}: {e}")

    utils.log(f"[expB-W4] {len(probe_hooks)} probe hooks installed: {list(probe_hooks)}")

    try:
        for (suite, task_id, seed, episode_idx) in trials:
            try:
                run_seed_w4(
                    policy, model, ctrl, legacy_hook, probe_hooks,
                    suite, task_id, seed, episode_idx,
                    conditions=conditions,
                    n_grid=n_grid, sigma=sigma, sis_stride=sis_stride,
                    frac=frac,
                    ternary_partition=ternary_partition,
                    diag_path=diag_path,
                    rollout_path=rollout_path,
                    reuse_cached_diag=reuse_cached_diag,
                    verbose=verbose,
                )
            except Exception as e:
                utils.log(f"[expB-W4] FAILED trial {(suite, task_id, seed, episode_idx)}: {e}")
                import traceback
                traceback.print_exc()
    finally:
        for h in probe_hooks.values():
            h.uninstall()
        legacy_hook.uninstall()
        ctrl.use_fp16()


def _matched_pair_delta(rows_a, rows_b):
    """SR(A) − SR(B) over trials present in BOTH conditions, plus n_matched."""
    def _key(r):
        return (r["suite"], int(r["task_id"]), int(r["seed"]), int(r["episode_idx"]))
    by_a = {_key(r): bool(r["success"]) for r in rows_a}
    by_b = {_key(r): bool(r["success"]) for r in rows_b}
    common = set(by_a) & set(by_b)
    if not common:
        return float("nan"), 0
    deltas = [int(by_a[k]) - int(by_b[k]) for k in common]
    return float(np.mean(deltas)), len(deltas)


def _spearman_per_trial_w4(diag_v3_rows, probe_tag: str = "l12h2-ent"):
    """Per-trial Spearman ρ between W4-pass `attn_probes_w4[probe_tag]` and
    per-cycle `mse_fp_w4`. Tests whether the D2 finding (W2 sensitivity ↔ l12h2
    entropy) transfers to W4 sensitivity.

    Returns list of (trial_key, n_cycles, rho).
    """
    from collections import defaultdict
    by_trial = defaultdict(list)
    for r in diag_v3_rows:
        key = (r["suite"], int(r["task_id"]), int(r["seed"]), int(r["episode_idx"]))
        by_trial[key].append(r)
    out = []
    for k, rs in by_trial.items():
        rs.sort(key=lambda r: r["cycle_idx"])
        # Pull (probe_tag, mse) pairs.
        probes = [r.get("attn_probes_w4", {}).get(probe_tag) for r in rs]
        mses = [r.get("mse_fp_w4") for r in rs]
        valid = [
            (p, m) for p, m in zip(probes, mses)
            if p is not None and m is not None and p == p and m == m
        ]
        if len(valid) < 4:
            continue
        try:
            from scipy.stats import rankdata  # type: ignore
        except ImportError:
            return out
        p_arr = np.array([x[0] for x in valid])
        m_arr = np.array([x[1] for x in valid])
        rho = float(np.corrcoef(rankdata(p_arr), rankdata(m_arr))[0, 1])
        out.append((k, len(valid), rho))
    return out


def analyze_w4():
    """Bootstrap-CI summary table + HW0-HW7 hypothesis tests (writes expB_w4_summary.md)."""
    if not W4_ROLLOUT_PATH.exists():
        utils.log(f"[expB-W4] no rollouts at {W4_ROLLOUT_PATH}; nothing to analyze")
        return
    rows = utils.load_jsonl(W4_ROLLOUT_PATH)
    if not rows:
        utils.log("[expB-W4] empty rollouts file")
        return

    from collections import defaultdict
    by_cond_full = defaultdict(list)
    by_cond = defaultdict(list)
    by_suite_cond = defaultdict(list)
    for r in rows:
        by_cond_full[r["condition"]].append(r)
        by_cond[r["condition"]].append(r["success"])
        by_suite_cond[(r["suite"], r["condition"])].append(r["success"])

    lines = []
    lines.append("# ExpB W4-First — Online Mixed-Precision Quantization Summary\n")
    lines.append(f"_n rollouts = {len(rows)}_\n")

    # ---- Overall SR table ----
    lines.append("## Overall success rate (95% bootstrap CI, n_boot=10k)\n")
    lines.append("| Condition | n | success rate | 95% CI | avg bits |")
    lines.append("|---|---:|---:|---|---:|")
    avg_bits_by_cond = defaultdict(list)
    for r in rows:
        b = r.get("condition_avg_bits")
        if b is not None:
            avg_bits_by_cond[r["condition"]].append(b)
    for cond in ALL_CONDITIONS:
        if cond not in by_cond:
            continue
        vals = by_cond[cond]
        m, lo, hi = _bootstrap_ci(vals)
        bits = avg_bits_by_cond.get(cond, [])
        bits_str = f"{np.mean(bits):.2f}" if bits else "—"
        lines.append(f"| {cond} | {len(vals)} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | {bits_str} |")

    # ---- Per-suite SR table ----
    lines.append("\n## Per-suite success rate\n")
    suites = sorted({r["suite"] for r in rows})
    lines.append("| Condition | " + " | ".join(suites) + " |")
    lines.append("|---|" + "---:|" * len(suites))
    for cond in ALL_CONDITIONS:
        cells = [cond]
        any_data = False
        for s in suites:
            vals = by_suite_cond.get((s, cond), [])
            if vals:
                any_data = True
                m, lo, hi = _bootstrap_ci(vals, n_boot=2000)
                cells.append(f"{m:.3f} [{lo:.2f},{hi:.2f}] (n={len(vals)})")
            else:
                cells.append("—")
        if any_data:
            lines.append("| " + " | ".join(cells) + " |")

    # ---- HW0-HW7 matched-pair hypothesis matrix ----
    lines.append("\n## Hypothesis matrix (matched-pair signed deltas)\n")
    lines.append("Each row computes SR(A) − SR(B) over trials present in BOTH conditions.")
    lines.append("Positive = A wins. Matched seeds cancel intrinsic trial difficulty.\n")
    lines.append("| Tag | A | B | n_matched | SR(A) − SR(B) | Question |")
    lines.append("|---|---|---|---:|---:|---|")
    pairs = [
        ("HW0", "W4-Floor", "FP16", "Is W4 alone good enough? (defines whether FP16-rescue is meaningful)"),
        ("HW1", "S1-Bin-W4", "Random-W4", "Does the lag-1 mechanism work at W4? (bottom dir)"),
        ("HW2", "S3-Bin-W4-l12h2-ent", "S1-Bin-W4", "Does intra-pass beat lag-1 at W4? (bottom dir)"),
        ("HW3a", "S3-Bin-W4-l1h7-top1", "S3-Bin-W4-l12h2-ent", "Earlier-layer cheap-pass viable at W4?"),
        ("HW3b", "S3-Bin-W4-l9h2-ent", "S3-Bin-W4-l12h2-ent", "Mid-layer alt viable?"),
        ("HW4", "S1-Tern-W4", "W4-Floor", "Sub-W4 average preserves SR vs uniform W4?"),
        ("HW5", "S2-Bin-W4", "S1-Bin-W4", "No-lag advantage at W4?"),
        ("HW6", "AttnEntropy-W4", "Random-W4", "Oracle direction validation (bottom dir, W2 default)"),
        ("HW8a", "Probe-W4-l1h7-top1", "AttnEntropy-W4", "Lag-1 probe l1h7 vs lag-0 oracle l12h2"),
        ("HW8b", "Probe-W4-l17h4-top1", "AttnEntropy-W4", "Late-layer signal-strength upper bound"),
        ("HW8c", "Probe-W4-l9h2-ent", "AttnEntropy-W4", "Mid-layer probe vs oracle"),
        # 2026-04-29: direction-flip tests (mid-Tier-0 finding flagged ρ sign-flip at W4)
        ("HW9a", "AttnEntropy-W4-top", "Random-W4", "Top-direction oracle vs random — does flipped direction work?"),
        ("HW9b", "AttnEntropy-W4-top", "AttnEntropy-W4", "Top vs bottom direction oracle — which is right at W4?"),
        ("HW9c", "S1-Bin-W4-top", "S1-Bin-W4", "Top vs bottom lag-1 — which is right at W4?"),
        ("HW9d", "S3-Bin-W4-l12h2-ent-top", "S3-Bin-W4-l12h2-ent", "Top vs bottom intra-pass at l12h2"),
        ("HW9e", "S2-Bin-W4-top", "S2-Bin-W4", "Top vs bottom speculative"),
        # 2026-04-29 Tier 4: direction-flipped TERNARY (the broken S1/S2-Tern story)
        ("HW10a", "S1-Tern-W4-top", "S1-Tern-W4", "Top vs bottom lag-1 ternary — does flip rescue S1-Tern?"),
        ("HW10b", "S2-Tern-W4-top", "S2-Tern-W4", "Top vs bottom speculative ternary — does flip rescue S2-Tern?"),
        ("HW10c", "S3-Tern-W4-l12h2-top", "S3-Tern-W4-l12h2", "Top vs bottom intra-pass ternary at l12h2"),
        ("HW10d", "S1-Tern-W4-top", "W4-Floor", "Direction-flipped ternary: does it match Floor at sub-W4 bits?"),
        ("HW10e", "S3-Tern-W4-l12h2-top", "W4-Floor", "Intra-pass top-dir ternary: even better than bottom?"),
        # 2026-04-29 Tier 5: l1h7 with W4-correct direction (bottom)
        ("HW11a", "S3-Bin-W4-l1h7-bottom", "S3-Bin-W4-l1h7-top1", "l1h7 bottom (W4-correct) vs top (W2-default)"),
        ("HW11b", "S3-Bin-W4-l1h7-bottom", "S3-Bin-W4-l12h2-ent", "l1h7 bottom vs l12h2 bottom — earlier layer better?"),
        ("HW11c", "S3-Tern-W4-l1h7-bottom", "S3-Tern-W4-l12h2", "l1h7 bottom ternary vs l12h2 bottom ternary"),
        ("HW11d", "S3-Tern-W4-l1h7-bottom", "W4-Floor", "l1h7 ternary vs W4-Floor (cheap-pass Pareto test)"),
    ]
    for tag, a, b, q in pairs:
        if a not in by_cond_full or b not in by_cond_full:
            continue
        d, n = _matched_pair_delta(by_cond_full[a], by_cond_full[b])
        lines.append(f"| {tag} | {a} | {b} | {n} | {d:+.3f} | {q} |")

    # ---- HW7 — D2-W4 transfer Spearman ----
    if DIAG_V3_PATH.exists():
        diag_rows = utils.load_jsonl(DIAG_V3_PATH)
        lines.append("\n## HW7 — D2-W4 transfer (per-trial Spearman ρ)\n")
        lines.append("Per-trial ρ between l12h2-entropy on W4-pass and ‖a_FP − a_W4‖² per cycle.")
        lines.append("If |median ρ| > 0.15, the D2 mechanism transfers cleanly from W2 to W4.\n")
        rhos = _spearman_per_trial_w4(diag_rows, probe_tag="l12h2-ent")
        if rhos:
            rho_vals = np.array([r for (_, _, r) in rhos])
            lines.append(
                f"_n={len(rho_vals)} trials. median ρ = {float(np.median(rho_vals)):.3f}, "
                f"mean ρ = {float(np.mean(rho_vals)):.3f}, "
                f"P(|ρ| > 0.15) = {float(np.mean(np.abs(rho_vals) > 0.15)):.2f}, "
                f"min/max = {float(np.min(rho_vals)):.3f}/{float(np.max(rho_vals)):.3f}._"
            )
            lines.append("")
            lines.append("| quantile | ρ |")
            lines.append("|---|---:|")
            for q in (0.10, 0.25, 0.50, 0.75, 0.90):
                lines.append(f"| p{int(q*100)} | {float(np.quantile(rho_vals, q)):.3f} |")
        else:
            lines.append("_No trials with sufficient valid (probe, mse) pairs._")

        # Also report ρ for alternative probes — early-layer transfer test.
        lines.append("\n### HW7-extension — alternative probes\n")
        lines.append("| probe tag | n trials | median ρ | mean ρ | p25 | p75 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for tag in DEFAULT_CANDIDATE_READOUTS:
            tag_rhos = _spearman_per_trial_w4(diag_rows, probe_tag=tag)
            if not tag_rhos:
                continue
            tv = np.array([r for (_, _, r) in tag_rhos])
            lines.append(
                f"| {tag} | {len(tv)} | {float(np.median(tv)):+.3f} | "
                f"{float(np.mean(tv)):+.3f} | {float(np.quantile(tv, 0.25)):+.3f} | "
                f"{float(np.quantile(tv, 0.75)):+.3f} |"
            )

    W4_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    W4_SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    utils.log(f"[expB-W4] wrote {W4_SUMMARY_PATH}")


# ---------------------------------------------------------------------------
# Per-seed driver: diagnostic + 6 override conditions
# ---------------------------------------------------------------------------
def run_seed(
    policy, model, ctrl, attn_hook,
    suite: str, task_id: int, seed: int, episode_idx: int,
    conditions,
    n_grid: int = 4, sigma: float = 8.0,
    sis_stride: int = 4,
    frac: float = 0.20,
    ternary_partition=(0.2, 0.3, 0.5),
    diag_path: Path = None,
    verbose: bool = False,
) -> list:
    """Run all requested conditions for one (suite, task_id, seed, episode_idx).
    Appends to JSONL files incrementally and returns the list of rollout records."""
    if diag_path is None:
        diag_path = DIAG_PATH
    out = []
    rollout_key = {
        "suite": suite, "task_id": int(task_id),
        "seed": int(seed), "episode_idx": int(episode_idx),
    }

    # Conditions that need the W2 diagnostic (SIS, attn entropy, MSE-W2traj scores)
    NEEDS_W2_DIAG = (
        {"W2", "SIS-top", "Random", "Bottom-SIS", "MSE-W2traj", "AttnEntropy", "AttnEntropy-flipped"}
        | set(SCHEMES_CONDITIONS)
    )
    # Conditions that need the FP16 diagnostic (MSE-FP16traj scores)
    NEEDS_FP16_DIAG = {"FP16", "MSE-FP16traj"}

    w2_rec, w2_per_cycle = None, None
    if any(c in conditions for c in NEEDS_W2_DIAG):
        utils.log(f"[expB] W2-DIAG seed={seed} task={task_id} ep={episode_idx}")
        t0 = time.time()
        w2_rec, w2_per_cycle = diagnostic_rollout(
            policy, model, ctrl, attn_hook,
            suite, task_id, seed, episode_idx,
            n_grid=n_grid, sigma=sigma, sis_stride=sis_stride,
            verbose=verbose,
        )
        utils.log(
            f"[expB] W2-DIAG done seed={seed} success={w2_rec.success} "
            f"steps={w2_rec.steps} cycles={len(w2_per_cycle)} wall={time.time()-t0:.1f}s"
        )
        for c in w2_per_cycle:
            utils.append_jsonl({**rollout_key, **c}, diag_path)

    fp16_rec, fp16_per_cycle = None, None
    if any(c in conditions for c in NEEDS_FP16_DIAG):
        utils.log(f"[expB] FP16-DIAG seed={seed} task={task_id} ep={episode_idx}")
        t0 = time.time()
        fp16_rec, fp16_per_cycle = fp16_diagnostic_rollout(
            policy, model, ctrl,
            suite, task_id, seed, episode_idx,
            verbose=verbose,
        )
        utils.log(
            f"[expB] FP16-DIAG done seed={seed} success={fp16_rec.success} "
            f"steps={fp16_rec.steps} cycles={len(fp16_per_cycle)} wall={time.time()-t0:.1f}s"
        )
        for c in fp16_per_cycle:
            utils.append_jsonl({**rollout_key, **c}, FP16_DIAG_PATH)

    masks = build_masks(
        w2_per_cycle, fp16_per_cycle, frac=frac, seed=seed,
        ternary_partition=ternary_partition,
    ) if w2_per_cycle else {}
    n_cycles_w2 = len(w2_per_cycle) if w2_per_cycle else 0

    for cond in conditions:
        if cond == "W2":
            rec = w2_rec
            schedule = {}
            n_overrides = 0
            override_indices = []
            avg_bits = float(BITS_BY_PRECISION["w2"])
        elif cond == "FP16":
            rec = fp16_rec
            schedule = masks.get("FP16", set())
            n_overrides = len(schedule)
            override_indices = sorted(schedule) if isinstance(schedule, set) else sorted(schedule.keys())
            avg_bits = float(BITS_BY_PRECISION["fp16"])
        else:
            if cond not in masks:
                utils.log(f"[expB] WARN: skipping {cond} — mask not available")
                continue
            schedule = masks[cond]
            # Normalize set → dict (binary FP16-override) for unified dispatch.
            if isinstance(schedule, set):
                precision_per_cycle = {int(c): "fp16" for c in schedule}
            else:
                precision_per_cycle = dict(schedule)
            n_overrides = sum(1 for p in precision_per_cycle.values() if p != "w2")
            override_indices = sorted(precision_per_cycle.keys())
            avg_bits = _avg_bits(precision_per_cycle, n_cycles_w2, default_precision="w2")

            t0 = time.time()
            utils.log(
                f"[expB] {cond} seed={seed} task={task_id} ep={episode_idx} "
                f"|overrides|={n_overrides}/{n_cycles_w2} avg_bits={avg_bits:.2f}"
            )
            rec = override_rollout(
                policy, model, ctrl,
                suite, task_id, seed, episode_idx,
                precision_per_cycle=precision_per_cycle,
                default_precision="w2",
                verbose=verbose,
            )
            utils.log(
                f"[expB] {cond} done success={rec.success} steps={rec.steps} "
                f"wall={time.time()-t0:.1f}s"
            )

        # Route schemes conditions to the schemes JSONL; legacy to ROLLOUT_PATH.
        target_path = SCHEMES_ROLLOUT_PATH if cond in SCHEMES_CONDITIONS else ROLLOUT_PATH
        entry = {
            **rollout_key,
            "condition": cond,
            "success": bool(rec.success),
            "steps": int(rec.steps),
            "wall_time_s": float(rec.wall_time_s),
            "termination_reason": rec.termination_reason,
            "n_overrides": n_overrides,
            "override_indices": override_indices,
            "n_cycles_w2": n_cycles_w2,
            "n_cycles_fp16": len(fp16_per_cycle) if fp16_per_cycle else None,
            "condition_avg_bits": avg_bits,
            "ternary_partition": list(ternary_partition) if cond in ("S1-Tern", "S2-Tern", "Random-Tern") else None,
        }
        utils.append_jsonl(entry, target_path)
        out.append(entry)

    return out


# ---------------------------------------------------------------------------
# Frac sweep — reuse existing diagnostic JSONL, run only override rollouts at
# the requested fracs. Output to a separate JSONL so the main results aren't
# polluted.
# ---------------------------------------------------------------------------
SWEEP_PATH = RESULTS_DIR / "expB_frac_sweep.jsonl"


def _load_diagnostic_by_trial(diag_path: Path = None):
    """Group diagnostic JSONL rows by (suite, task_id, seed, episode_idx).
    Returns dict[trial_key] = sorted list of per-cycle records.

    Defaults to the legacy DIAG_PATH; pass DIAG_V2_PATH to read the augmented
    diagnostic that contains attn_entropy_l12h2_w2.
    """
    if diag_path is None:
        diag_path = DIAG_PATH
    if not diag_path.exists():
        return {}
    rows = utils.load_jsonl(diag_path)
    from collections import defaultdict
    by_trial = defaultdict(list)
    for r in rows:
        key = (r["suite"], int(r["task_id"]), int(r["seed"]), int(r["episode_idx"]))
        by_trial[key].append(r)
    for k in by_trial:
        by_trial[k].sort(key=lambda r: r["cycle_idx"])
    return dict(by_trial)


def run_frac_sweep(
    fracs, conditions,
    bits_list=(2,),
    ternary_partition=(0.2, 0.3, 0.5),
    diag_path: Path = None,
    verbose=False,
):
    """For each (frac, condition, trial) combo, build mask from existing W2
    diagnostic data and run an override rollout. Cheap because we skip both
    diagnostic passes entirely.

    `diag_path` defaults to DIAG_PATH; for the deployable schemes use DIAG_V2_PATH
    which has attn_entropy_l12h2_w2.
    """
    diag_by_trial = _load_diagnostic_by_trial(diag_path=diag_path)
    if not diag_by_trial:
        utils.log(f"[sweep] no diagnostic data at {diag_path or DIAG_PATH}; cannot sweep")
        return

    needs_w4 = any(c in conditions for c in ("S1-Tern", "S2-Tern", "Random-Tern"))
    if needs_w4 and 4 not in bits_list:
        bits_list = tuple(sorted(set(bits_list) | {4}))
        utils.log(f"[sweep] auto-extending bits_list to {bits_list} for ternary conditions")

    utils.log(f"[sweep] {len(diag_by_trial)} trials × {len(fracs)} fracs × "
              f"{len(conditions)} conditions = {len(diag_by_trial)*len(fracs)*len(conditions)} rollouts")
    utils.log(f"[sweep] fracs: {fracs}")
    utils.log(f"[sweep] conditions: {conditions}")

    utils.log("[sweep] loading policy + model...")
    policy, model = utils.load_policy()
    utils.log(f"[sweep] building PrecisionController (bits_list={bits_list} + protect)...")
    ctrl = PrecisionController(model, bits_list=bits_list, group_size=128)
    ctrl.use_fp16()

    try:
        for trial_key in sorted(diag_by_trial.keys()):
            suite, task_id, seed, episode_idx = trial_key
            per_cycle_w2 = diag_by_trial[trial_key]
            n_cycles = len(per_cycle_w2)

            for frac in fracs:
                masks = build_masks(
                    per_cycle_w2, None, frac=frac, seed=seed,
                    ternary_partition=ternary_partition,
                )

                for cond in conditions:
                    if cond not in masks:
                        utils.log(f"[sweep] skip {cond}: not in masks {list(masks.keys())}")
                        continue
                    schedule = masks[cond]
                    if isinstance(schedule, set):
                        precision_per_cycle = {int(c): "fp16" for c in schedule}
                    else:
                        precision_per_cycle = dict(schedule)
                    n_overrides = sum(1 for p in precision_per_cycle.values() if p != "w2")
                    avg_bits = _avg_bits(precision_per_cycle, n_cycles, default_precision="w2")

                    t0 = time.time()
                    utils.log(
                        f"[sweep] frac={frac} {cond} seed={seed} task={task_id} "
                        f"ep={episode_idx} |overrides|={n_overrides}/{n_cycles} avg_bits={avg_bits:.2f}"
                    )
                    try:
                        rec = override_rollout(
                            policy, model, ctrl,
                            suite, task_id, seed, episode_idx,
                            precision_per_cycle=precision_per_cycle,
                            default_precision="w2",
                            verbose=verbose,
                        )
                    except Exception as e:
                        utils.log(f"[sweep] FAILED frac={frac} {cond} trial={trial_key}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    utils.log(
                        f"[sweep] frac={frac} {cond} done success={rec.success} "
                        f"steps={rec.steps} wall={time.time()-t0:.1f}s"
                    )
                    entry = {
                        "suite": suite, "task_id": int(task_id),
                        "seed": int(seed), "episode_idx": int(episode_idx),
                        "condition": cond, "frac": float(frac),
                        "success": bool(rec.success), "steps": int(rec.steps),
                        "wall_time_s": float(rec.wall_time_s),
                        "termination_reason": rec.termination_reason,
                        "n_overrides": n_overrides,
                        "n_cycles_w2": n_cycles,
                        "condition_avg_bits": avg_bits,
                        "ternary_partition": list(ternary_partition) if cond in ("S1-Tern", "S2-Tern", "Random-Tern") else None,
                    }
                    utils.append_jsonl(entry, SWEEP_PATH)
    finally:
        ctrl.use_fp16()


def analyze_sweep():
    """Per-frac × per-condition success-rate table (overall + per-suite)."""
    if not SWEEP_PATH.exists():
        utils.log(f"[sweep] no sweep data at {SWEEP_PATH}")
        return
    rows = utils.load_jsonl(SWEEP_PATH)
    if not rows:
        return
    from collections import defaultdict
    by_frac_cond = defaultdict(list)
    by_suite_frac_cond = defaultdict(list)
    for r in rows:
        by_frac_cond[(r["frac"], r["condition"])].append(r["success"])
        by_suite_frac_cond[(r["suite"], r["frac"], r["condition"])].append(r["success"])

    fracs = sorted({f for (f, _) in by_frac_cond})
    conds = sorted({c for (_, c) in by_frac_cond})

    lines = ["# ExpB Frac Sweep Summary\n", f"_n rollouts = {len(rows)}_\n"]

    lines.append("## Overall success rate by (condition, frac)\n")
    lines.append("| Condition | " + " | ".join(f"frac={f}" for f in fracs) + " |")
    lines.append("|---|" + "---:|" * len(fracs))
    for c in conds:
        cells = [c]
        for f in fracs:
            vals = by_frac_cond.get((f, c), [])
            if vals:
                m, lo, hi = _bootstrap_ci(vals, n_boot=2000)
                cells.append(f"{m:.3f} [{lo:.2f},{hi:.2f}] (n={len(vals)})")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    suites = sorted({r["suite"] for r in rows})
    for s in suites:
        lines.append(f"\n## {s} success rate by (condition, frac)\n")
        lines.append("| Condition | " + " | ".join(f"frac={f}" for f in fracs) + " |")
        lines.append("|---|" + "---:|" * len(fracs))
        for c in conds:
            cells = [c]
            for f in fracs:
                vals = by_suite_frac_cond.get((s, f, c), [])
                if vals:
                    m, lo, hi = _bootstrap_ci(vals, n_boot=2000)
                    cells.append(f"{m:.3f} [{lo:.2f},{hi:.2f}] (n={len(vals)})")
                else:
                    cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")

    sweep_summary = RESULTS_DIR / "expB_frac_sweep_summary.md"
    sweep_summary.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    utils.log(f"[sweep] wrote {sweep_summary}")


# ---------------------------------------------------------------------------
# Multi-seed driver
# ---------------------------------------------------------------------------
def run_trials(
    trials, conditions,
    n_grid=4, sigma=8.0, sis_stride=4, frac=0.20,
    bits_list=(2,),
    ternary_partition=(0.2, 0.3, 0.5),
    diag_path: Path = None,
    verbose=False,
):
    """trials: list of (suite, task_id, seed, episode_idx) tuples.

    `bits_list` controls which quantized weight tiers PrecisionController caches.
    Use (2, 4) when running ternary conditions (S1-Tern / S2-Tern / Random-Tern);
    legacy default (2,) preserves the original ExpB memory footprint.
    `diag_path` overrides DIAG_PATH (e.g. write augmented W2-pass entropy to
    DIAG_V2_PATH instead of clobbering the legacy expB_diagnostic.jsonl).
    """
    needs_w4 = any(c in conditions for c in ("S1-Tern", "S2-Tern", "Random-Tern"))
    if needs_w4 and 4 not in bits_list:
        bits_list = tuple(sorted(set(bits_list) | {4}))
        utils.log(f"[expB] auto-extending bits_list to {bits_list} for ternary conditions")

    utils.log(f"[expB] loading policy + model...")
    policy, model = utils.load_policy()

    utils.log(f"[expB] building PrecisionController (bits_list={bits_list} + protect)...")
    ctrl = PrecisionController(model, bits_list=bits_list, group_size=128)
    ctrl.use_fp16()  # start clean

    utils.log(f"[expB] installing L12H2 attention hook...")
    attn_hook = L12H2EntropyHook(model)

    try:
        for (suite, task_id, seed, episode_idx) in trials:
            try:
                run_seed(
                    policy, model, ctrl, attn_hook,
                    suite, task_id, seed, episode_idx,
                    conditions=conditions,
                    n_grid=n_grid, sigma=sigma, sis_stride=sis_stride,
                    frac=frac,
                    ternary_partition=ternary_partition,
                    diag_path=diag_path,
                    verbose=verbose,
                )
            except Exception as e:
                utils.log(f"[expB] FAILED trial {(suite, task_id, seed, episode_idx)}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to the next trial — partial results are useful
    finally:
        attn_hook.uninstall()
        ctrl.use_fp16()


# ---------------------------------------------------------------------------
# Analysis (Phase F)
# ---------------------------------------------------------------------------
def _bootstrap_ci(values, n_boot=10_000, alpha=0.05, seed=0):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    n = arr.size
    for _ in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        means.append(sample.mean())
    means = np.array(means)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def analyze():
    if not ROLLOUT_PATH.exists():
        utils.log(f"[expB] no rollouts at {ROLLOUT_PATH}; nothing to analyze")
        return
    rows = utils.load_jsonl(ROLLOUT_PATH)
    if not rows:
        utils.log("[expB] empty rollouts file")
        return

    # group by (suite, condition)
    from collections import defaultdict
    by_cond = defaultdict(list)
    by_suite_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r["success"])
        by_suite_cond[(r["suite"], r["condition"])].append(r["success"])

    lines = []
    lines.append("# ExpB — Saliency-Aware PTQ Validation Summary\n")
    lines.append(f"_n rollouts = {len(rows)}_\n")

    lines.append("## Overall success rate (95% bootstrap CI, n_boot=10k)\n")
    lines.append("| Condition | n | success rate | 95% CI |")
    lines.append("|---|---:|---:|---|")
    for cond in ALL_CONDITIONS:
        if cond not in by_cond:
            continue
        vals = by_cond[cond]
        m, lo, hi = _bootstrap_ci(vals)
        lines.append(f"| {cond} | {len(vals)} | {m:.3f} | [{lo:.3f}, {hi:.3f}] |")

    lines.append("\n## Per-suite success rate\n")
    suites = sorted({r["suite"] for r in rows})
    lines.append("| Condition | " + " | ".join(suites) + " |")
    lines.append("|---|" + "---:|" * len(suites))
    for cond in ALL_CONDITIONS:
        cells = [cond]
        any_data = False
        for s in suites:
            vals = by_suite_cond.get((s, cond), [])
            if vals:
                any_data = True
                m, lo, hi = _bootstrap_ci(vals, n_boot=2000)
                cells.append(f"{m:.3f} [{lo:.2f},{hi:.2f}] (n={len(vals)})")
            else:
                cells.append("—")
        if any_data:
            lines.append("| " + " | ".join(cells) + " |")

    # Hypothesis matrix interpretation (only if all key conditions present)
    needed = {"W2", "FP16", "SIS-top", "Random", "MSE-W2traj"}
    if needed.issubset(by_cond.keys()):
        sr = {c: float(np.mean(by_cond[c])) for c in ALL_CONDITIONS if c in by_cond}
        lines.append("\n## Hypothesis matrix\n")
        for c in ALL_CONDITIONS:
            if c in sr:
                lines.append(f"- SR({c}) = {sr[c]:.3f}")

        # Pick the "oracle" — prefer max(MSE-W2traj, MSE-FP16traj) since both are heuristic ceilings
        oracle_w2 = sr.get("MSE-W2traj", float("nan"))
        oracle_fp = sr.get("MSE-FP16traj", float("nan"))
        oracle = max(o for o in (oracle_w2, oracle_fp) if o == o) if (oracle_w2 == oracle_w2 or oracle_fp == oracle_fp) else float("nan")

        delta_sis_rand = sr["SIS-top"] - sr["Random"]
        delta_sis_oracle = oracle - sr["SIS-top"]
        lines.append("")
        lines.append(f"- SIS over Random = {delta_sis_rand:+.3f}")
        lines.append(f"- Oracle headroom over SIS (best of MSE-W2traj, MSE-FP16traj) = {delta_sis_oracle:+.3f}")

        if "Bottom-SIS" in sr:
            delta_top_bottom = sr["SIS-top"] - sr["Bottom-SIS"]
            lines.append(f"- SIS-top over Bottom-SIS (symmetry) = {delta_top_bottom:+.3f}")
        if "AttnEntropy" in sr:
            delta_attn_sis = sr["AttnEntropy"] - sr["SIS-top"]
            lines.append(f"- AttnEntropy vs SIS (cheap proxy gap) = {delta_attn_sis:+.3f}")

        if delta_sis_rand <= 0:
            verdict = "**KILL**: SR(SIS-top) ≤ SR(Random). SIS does not carry quant-specific signal at this quantization level."
        elif "Bottom-SIS" in sr and sr["SIS-top"] <= sr["Bottom-SIS"]:
            verdict = "**KILL**: SR(SIS-top) ≤ SR(Bottom-SIS). SIS rank direction has no signal — failed symmetry control."
        elif delta_sis_oracle > 0.10:
            verdict = "**PARTIAL**: SIS beats random but leaves significant oracle headroom. Detector has signal but misses sensitive frames."
        else:
            verdict = "**STRONG**: SIS recovers most of the oracle gap and beats random. PTQ-SQIL works."
        lines.append("\n" + verdict)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    utils.log(f"[expB] wrote {SUMMARY_PATH}")


# ---------------------------------------------------------------------------
# Trial-set builders
# ---------------------------------------------------------------------------
def pilot_trials():
    # Single Long task, 20 distinct (seed, episode_idx) pairs.
    return [("Long", 0, s, s % 10) for s in range(20)]


def full_trials():
    # 50 Long + 50 Object. For each, 10 tasks × 5 episodes = 50 trials per suite.
    out = []
    for suite, base in [("Long", 0), ("Object", 20)]:
        for task_off in range(10):
            for ep in range(5):
                global_task_id = base + task_off
                seed = task_off * 10 + ep   # deterministic, distinct per trial
                out.append((suite, global_task_id, seed, ep))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_ternary_partition(s: str) -> tuple:
    """Parse '0.2,0.3,0.5' into (frac_fp16, frac_w4, frac_w2)."""
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"ternary-partition must be three comma-separated floats (FP16,W4,W2); got {s!r}"
        )
    if any(x < 0 for x in parts) or sum(parts) > 1.0 + 1e-6:
        raise argparse.ArgumentTypeError(
            f"ternary-partition fractions must be ≥0 and sum ≤1; got {parts}"
        )
    return tuple(parts)


def _parse_trial_range(s: str) -> slice:
    """Parse 'A:B' into a slice(A, B). Either bound may be omitted."""
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"trial-range must be 'A:B' (slice); got {s!r}")
    a, b = s.split(":", 1)
    return slice(int(a) if a else None, int(b) if b else None)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true",
                   help="1 trial, all 8 conditions — verifies plumbing")
    g.add_argument("--pilot", action="store_true",
                   help="20 trials × {FP16,W2,SIS-top,Random,MSE-W2traj} on libero_10 task 0")
    g.add_argument("--full", action="store_true",
                   help="100 trials (50 Long + 50 Object) × all legacy conditions @ frac=0.5")
    g.add_argument("--schemes", action="store_true",
                   help="100 trials × {S1-Bin, S2-Bin, S1-Tern, S2-Tern, Random-Tern} @ frac=0.5; "
                        "implies --recompute-diagnostic and writes to expB_diagnostic_v2.jsonl + "
                        "expB_schemes_rollouts.jsonl. Default for the 2026-04-22 deployable plan.")
    g.add_argument("--analyze", action="store_true",
                   help="produce summary markdown from existing JSONL")
    g.add_argument("--frac-sweep", nargs="+", type=float, default=None,
                   help="sweep over given fracs, reusing existing W2 diagnostic; "
                        "writes to expB_frac_sweep.jsonl")
    g.add_argument("--analyze-sweep", action="store_true",
                   help="produce per-frac × per-condition table from sweep JSONL")
    g.add_argument("--w4-schemes", action="store_true",
                   help="2026-04-29 W4-first plan: 100 trials × W4-base conditions "
                        "(W4-Floor, S1/S2-Bin/Tern-W4, S3-Bin-W4-*, Probe-W4-*). "
                        "Writes diag to expB_diagnostic_v3.jsonl + rollouts to "
                        "expB_w4_rollouts.jsonl. Implies bits_list=(2,4).")
    g.add_argument("--w4-tier0", action="store_true",
                   help="W4 baseline only: FP16 + W4-Floor + W4-Static-Sched "
                        "on 100 trials. Sets the gate decision for whether the "
                        "FP16-rescue framing is meaningful.")
    g.add_argument("--analyze-w4", action="store_true",
                   help="produce W4 summary markdown from expB_w4_rollouts.jsonl")

    p.add_argument("--suite", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--conditions", nargs="+", default=None,
                   help='subset of conditions, or "all" / "pilot" / "schemes"')

    p.add_argument("--n-grid", type=int, default=4,
                   help="SIS perturbation grid (NxN patches); default 4 for ~10x speedup over paper's 8")
    p.add_argument("--sigma", type=float, default=8.0)
    p.add_argument("--sis-stride", type=int, default=4,
                   help="recompute SIS every k cycles, reuse last value in between (paper §3 table 12)")
    p.add_argument("--frac", type=float, default=0.5,
                   help="override fraction (binary granularity); default 0.5 (pilot showed 0.2 was budget-bound)")
    p.add_argument("--ternary-partition", type=_parse_ternary_partition, default=(0.2, 0.3, 0.5),
                   help="comma-separated FP16,W4,W2 fractions for ternary conditions; default 0.2,0.3,0.5")
    p.add_argument("--trial-range", type=_parse_trial_range, default=None,
                   help="slice into the full/schemes trial list as 'A:B' (e.g. 0:50 / 50:100). "
                        "Use with --schemes to split work across two GPUs.")
    p.add_argument("--recompute-diagnostic", action="store_true",
                   help="write the augmented diagnostic (with attn_entropy_l12h2_w2) to "
                        "expB_diagnostic_v2.jsonl, leaving the legacy expB_diagnostic.jsonl intact. "
                        "Implied by --schemes.")
    p.add_argument("--sweep-conditions", nargs="+",
                   default=["AttnEntropy", "Random"],
                   help="conditions to run in --frac-sweep mode (default: AttnEntropy Random)")
    p.add_argument("--sweep-diag", choices=("legacy", "v2"), default="legacy",
                   help="which diagnostic JSONL to read in sweep mode: 'legacy' = expB_diagnostic.jsonl "
                        "(default; works with all original conditions), 'v2' = expB_diagnostic_v2.jsonl "
                        "(required for S1-* / S2-* / Random-Tern conditions in sweep mode)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--reset", action="store_true",
                   help="delete existing JSONLs before running")
    p.add_argument("--candidate-readouts", nargs="+", default=None,
                   help="probe tags to capture in the V3 diagnostic, e.g. "
                        "'l1h7-top1 l9h2-ent l12h2-ent l3h4-top5 l17h4-top1'. "
                        "Default: all 5 from MEETING_5 top-15.")
    p.add_argument("--reuse-diag", action="store_true",
                   help="W4 modes only: skip the V3 diagnostic per trial when "
                        "prior records exist for that trial in DIAG_V3_PATH. "
                        "Used to chain Tier 0 → Tier 1+2+3 without redoing the "
                        "diagnostic (~half the per-trial wall time).")

    args = p.parse_args()

    if args.reset:
        # Sweep mode: only reset the sweep file, never the main diagnostic JSONLs
        # (we depend on them as input).
        if args.frac_sweep is not None:
            if SWEEP_PATH.exists():
                utils.log(f"[expB] removing {SWEEP_PATH}")
                SWEEP_PATH.unlink()
        elif args.schemes:
            # Schemes mode: reset only the augmented diag + schemes rollouts (preserve legacy).
            for f in (DIAG_V2_PATH, SCHEMES_ROLLOUT_PATH):
                if f.exists():
                    utils.log(f"[expB] removing {f}")
                    f.unlink()
        elif args.w4_schemes or args.w4_tier0:
            # W4 mode: reset V3 diagnostic + W4 rollouts (preserve W2 legacy).
            for f in (DIAG_V3_PATH, W4_ROLLOUT_PATH):
                if f.exists():
                    utils.log(f"[expB-W4] removing {f}")
                    f.unlink()
        else:
            for f in (DIAG_PATH, FP16_DIAG_PATH, ROLLOUT_PATH):
                if f.exists():
                    utils.log(f"[expB] removing {f}")
                    f.unlink()

    if args.analyze:
        analyze()
        return

    if args.analyze_sweep:
        analyze_sweep()
        return

    if args.analyze_w4:
        analyze_w4()
        return

    if args.frac_sweep is not None:
        sweep_diag_path = DIAG_V2_PATH if args.sweep_diag == "v2" else DIAG_PATH
        run_frac_sweep(
            args.frac_sweep, args.sweep_conditions,
            bits_list=(2, 4),
            ternary_partition=args.ternary_partition,
            diag_path=sweep_diag_path,
            verbose=args.verbose,
        )
        analyze_sweep()
        return

    # ---- W4-first modes (2026-04-29) ----
    if args.w4_tier0 or args.w4_schemes or (args.smoke and any(
        c in W4_ALL_CONDITIONS for c in (args.conditions or [])
    )):
        if args.smoke:
            trials = [("Long", 0, 0, 0)]
            if args.conditions and args.conditions != ["all"]:
                conditions = args.conditions
            else:
                conditions = ["W4-Floor", "S1-Bin-W4", "S3-Bin-W4-l1h7-top1",
                              "S3-Bin-W4-l12h2-ent", "Probe-W4-l1h7-top1"]
        elif args.w4_tier0:
            trials = full_trials()
            conditions = ["FP16", "W4-Floor", "W4-Static-Sched"]
        elif args.w4_schemes:
            trials = full_trials()
            if args.conditions and args.conditions != ["all"]:
                # Allow user to subset
                unknown = [c for c in args.conditions if c not in ALL_CONDITIONS]
                if unknown:
                    p.error(f"unknown conditions: {unknown}")
                conditions = args.conditions
            else:
                # Default: ALL W4 conditions (Tier 1 + Tier 2 + Tier 3 + W4-Floor)
                conditions = (
                    ["FP16", "W4-Floor"]
                    + sorted(W4_BIN_CONDITIONS)
                    + sorted(W4_INTRAPASS_CONDITIONS)
                    + sorted(W4_TERN_CONDITIONS)
                    + sorted(W4_PROBE_CONDITIONS)
                )

        if args.trial_range is not None:
            trials = trials[args.trial_range]
            utils.log(f"[expB-W4] applied --trial-range {args.trial_range}: {len(trials)} trials remain")

        readouts = args.candidate_readouts or list(DEFAULT_CANDIDATE_READOUTS)
        utils.log(f"[expB-W4] {len(trials)} trials × {len(conditions)} conditions; "
                  f"readouts={readouts}; frac={args.frac}; ternary={args.ternary_partition}")
        run_trials_w4(
            trials, conditions,
            candidate_readouts=readouts,
            n_grid=args.n_grid, sigma=args.sigma, sis_stride=args.sis_stride,
            frac=args.frac,
            ternary_partition=args.ternary_partition,
            reuse_cached_diag=args.reuse_diag,
            verbose=args.verbose,
        )
        analyze_w4()
        return

    # Build trial set
    diag_path_for_trials = DIAG_PATH
    if args.smoke:
        trials = [("Long", 0, 0, 0)]
        conditions = ALL_CONDITIONS
    elif args.pilot:
        trials = pilot_trials()
        conditions = PILOT_CONDITIONS
    elif args.full:
        trials = full_trials()
        conditions = [c for c in ALL_CONDITIONS if c not in SCHEMES_CONDITIONS]
    elif args.schemes:
        trials = full_trials()
        conditions = SCHEMES_CONDITIONS
        # Schemes mode requires the augmented diagnostic. Implies --recompute-diagnostic.
        diag_path_for_trials = DIAG_V2_PATH
    else:
        if args.suite is None or args.task_id is None or not args.seeds:
            p.error("must specify --smoke / --pilot / --full / --schemes / --analyze, "
                    "or (--suite + --task-id + --seeds)")
        trials = [(args.suite, args.task_id, s, args.episode_idx) for s in args.seeds]
        if args.conditions is None or args.conditions == ["all"]:
            conditions = ALL_CONDITIONS
        elif args.conditions == ["pilot"]:
            conditions = PILOT_CONDITIONS
        elif args.conditions == ["schemes"]:
            conditions = SCHEMES_CONDITIONS
        else:
            unknown = [c for c in args.conditions if c not in ALL_CONDITIONS]
            if unknown:
                p.error(f"unknown conditions: {unknown}; known: {ALL_CONDITIONS}")
            conditions = args.conditions
        if any(c in SCHEMES_CONDITIONS for c in conditions) or args.recompute_diagnostic:
            diag_path_for_trials = DIAG_V2_PATH

    if args.trial_range is not None:
        trials = trials[args.trial_range]
        utils.log(f"[expB] applied --trial-range {args.trial_range}: {len(trials)} trials remain")

    # Decide bits_list: ternary conditions need W4 cached.
    bits_list = (2, 4) if any(c in conditions for c in ("S1-Tern", "S2-Tern", "Random-Tern")) else (2,)

    utils.log(f"[expB] {len(trials)} trials × {len(conditions)} conditions = "
              f"{len(trials) * len(conditions)} rollouts (+ {len(trials)} diagnostic passes)")
    utils.log(f"[expB] conditions: {conditions}")
    utils.log(f"[expB] diag path: {diag_path_for_trials}")
    utils.log(f"[expB] bits_list: {bits_list}; ternary_partition: {args.ternary_partition}")

    run_trials(
        trials, conditions,
        n_grid=args.n_grid, sigma=args.sigma, sis_stride=args.sis_stride,
        frac=args.frac,
        bits_list=bits_list,
        ternary_partition=args.ternary_partition,
        diag_path=diag_path_for_trials,
        verbose=args.verbose,
    )
    analyze()


if __name__ == "__main__":
    main()
