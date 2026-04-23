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
    compute_sis,
    cycle_noise,
)


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(utils.RESULTS_DIR)
DIAG_PATH = RESULTS_DIR / "expB_diagnostic.jsonl"
DIAG_V2_PATH = RESULTS_DIR / "expB_diagnostic_v2.jsonl"   # NEW: augmented with W2-pass entropy
FP16_DIAG_PATH = RESULTS_DIR / "expB_fp16_diagnostic.jsonl"
ROLLOUT_PATH = RESULTS_DIR / "expB_rollouts.jsonl"
SCHEMES_ROLLOUT_PATH = RESULTS_DIR / "expB_schemes_rollouts.jsonl"  # NEW: S1/S2 conditions
SUMMARY_PATH = RESULTS_DIR / "expB_summary.md"
SCHEMES_SUMMARY_PATH = RESULTS_DIR / "expB_schemes_summary.md"

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
]
# Legacy pilot conditions (frac=0.5 results live under old labels in the previous JSONLs)
PILOT_CONDITIONS = ["FP16", "W2", "SIS-top", "Random", "MSE-W2traj"]
# Deployable-schemes conditions (the 2026-04-22 plan); diagnostic must capture W2-pass entropy.
SCHEMES_CONDITIONS = ["S1-Bin", "S2-Bin", "S1-Tern", "S2-Tern", "Random-Tern"]
# Conditions that require attn_entropy_l12h2_w2 in the diagnostic (the augmented field).
NEEDS_W2_PASS_ENTROPY = {"S1-Bin", "S2-Bin", "S1-Tern", "S2-Tern", "Random-Tern"}

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
