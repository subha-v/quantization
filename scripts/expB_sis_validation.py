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
FP16_DIAG_PATH = RESULTS_DIR / "expB_fp16_diagnostic.jsonl"
ROLLOUT_PATH = RESULTS_DIR / "expB_rollouts.jsonl"
SUMMARY_PATH = RESULTS_DIR / "expB_summary.md"

# ---------------------------------------------------------------------------
# Conditions (post-pilot: 8 conditions; renamed Oracle-20 → MSE-W2traj for
# symmetry with new MSE-FP16traj; dropped "-20" suffix since frac is configurable)
# ---------------------------------------------------------------------------
ALL_CONDITIONS = [
    "FP16",          # 1 — full mask, ceiling
    "W2",            # 2 — empty mask, floor
    "SIS-top",       # 3 — top-frac% by SIS
    "Random",        # 4 — random frac% (per-rollout seeded), null
    "Bottom-SIS",    # 5 — bottom-frac% by SIS, symmetry control
    "MSE-W2traj",    # 6 — top-frac% by ‖a_FP-a_W2‖² on W2-traj states
    "MSE-FP16traj",  # 7 — top-frac% by ‖a_FP-a_W2‖² on FP16-traj states (NEW)
    "AttnEntropy",   # 8 — bottom-frac% by l12h2 entropy (low entropy = high sensitivity per D2 ρ=-0.29)
]
# Legacy pilot conditions (frac=0.5 results live under old labels in the previous JSONLs)
PILOT_CONDITIONS = ["FP16", "W2", "SIS-top", "Random", "MSE-W2traj"]


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
        a_W2 = infer_with_noise(policy, openpi_obs, noise_np)
        mse = float(np.mean((a_FP - a_W2) ** 2))

        per_cycle.append({
            "cycle_idx": i,
            "env_step": int(t),
            "sis": sis_value,
            "sis_recomputed": bool(is_stride_cycle),
            "attn_entropy_l12h2": float(attn_e) if attn_e == attn_e else None,
            "mse_fp_w2": mse,
        })
        if verbose:
            tag = "S" if is_stride_cycle else "."
            utils.log(
                f"[diag {tag}] seed={seed} ep={episode_idx} cyc={i:3d} t={t:4d} "
                f"sis={sis_value:.5f} attn_e={attn_e:.4f} mse={mse:.5f}"
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
# Override rollout — replays seed with W2 base + FP16 override mask
# ---------------------------------------------------------------------------
def override_rollout(
    policy,
    model,
    ctrl: PrecisionController,
    suite: str,
    task_id: int,
    seed: int,
    episode_idx: int,
    override_set: set,
    verbose: bool = False,
):
    """Replay the same (suite, task_id, seed, episode_idx) starting state, with
    W2-with-protection as the base; at inference cycles whose 0-indexed
    `cycle_idx` is in `override_set`, run that cycle's `policy.infer` under
    FP16 weights instead.

    Returns the rollout_record.
    """
    cycle_idx = [-1]

    def pre_infer_callback(t):
        cycle_idx[0] += 1
        if cycle_idx[0] in override_set:
            ctrl.use_fp16()
        else:
            ctrl.use_quant()

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


def build_masks(per_cycle_w2, per_cycle_fp16, frac: float, seed: int) -> dict:
    """Build the override-frame-index masks from the W2 + FP16 diagnostic data.

    `per_cycle_w2` provides SIS, attn entropy, and MSE-W2traj scores (cycles
    indexed in the W2 trajectory). `per_cycle_fp16` provides MSE-FP16traj
    scores (cycles indexed in the FP16 trajectory). The two trajectories
    diverge so cycle counts may differ; each mask is sized as `frac` of its
    own diagnostic's cycle count, and reflects indices within THAT diagnostic
    — which align with override-rollout cycle indices since all rollouts share
    the same starting state.

    Cycle indices in the override mask that aren't reached during a particular
    rollout simply have no effect.
    """
    n_w2 = len(per_cycle_w2)
    k_w2 = max(1, int(round(frac * n_w2)))

    sis  = [c["sis"] for c in per_cycle_w2]
    mse  = [c["mse_fp_w2"] for c in per_cycle_w2]
    attn = [c["attn_entropy_l12h2"] for c in per_cycle_w2]

    rng = _random.Random(seed)

    masks = {
        "FP16":         set(range(n_w2)),
        "SIS-top":      _topk_indices(sis, k_w2, largest=True),
        "Random":       set(rng.sample(range(n_w2), k_w2)),
        "Bottom-SIS":   _topk_indices(sis, k_w2, largest=False),
        "MSE-W2traj":   _topk_indices(mse, k_w2, largest=True),
        # Low entropy → high sensitivity (D2 finding ρ=-0.294); pick smallest entropy.
        "AttnEntropy":  _topk_indices(attn, k_w2, largest=False),
    }

    if per_cycle_fp16 is not None and len(per_cycle_fp16) > 0:
        n_fp = len(per_cycle_fp16)
        k_fp = max(1, int(round(frac * n_fp)))
        mse_fp = [c["mse_fp_w2_fp16traj"] for c in per_cycle_fp16]
        masks["MSE-FP16traj"] = _topk_indices(mse_fp, k_fp, largest=True)

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
    verbose: bool = False,
) -> list:
    """Run all requested conditions for one (suite, task_id, seed, episode_idx).
    Appends to JSONL files incrementally and returns the list of rollout records."""
    out = []
    rollout_key = {
        "suite": suite, "task_id": int(task_id),
        "seed": int(seed), "episode_idx": int(episode_idx),
    }

    # Conditions that need the W2 diagnostic (SIS, attn entropy, MSE-W2traj scores)
    NEEDS_W2_DIAG = {"W2", "SIS-top", "Random", "Bottom-SIS", "MSE-W2traj", "AttnEntropy"}
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
            utils.append_jsonl({**rollout_key, **c}, DIAG_PATH)

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

    masks = build_masks(w2_per_cycle, fp16_per_cycle, frac=frac, seed=seed) \
        if w2_per_cycle else {}
    n_cycles_w2 = len(w2_per_cycle) if w2_per_cycle else 0

    for cond in conditions:
        if cond == "W2":
            rec = w2_rec
            mask = set()
        elif cond == "FP16":
            rec = fp16_rec
            mask = masks.get("FP16", set())  # informational only; FP16 diag IS the FP16 rollout
        else:
            if cond not in masks:
                utils.log(f"[expB] WARN: skipping {cond} — mask not available")
                continue
            mask = masks[cond]
            t0 = time.time()
            utils.log(
                f"[expB] {cond} seed={seed} task={task_id} ep={episode_idx} "
                f"|mask|={len(mask)}/{n_cycles_w2}"
            )
            rec = override_rollout(
                policy, model, ctrl,
                suite, task_id, seed, episode_idx,
                override_set=mask,
                verbose=verbose,
            )
            utils.log(
                f"[expB] {cond} done success={rec.success} steps={rec.steps} "
                f"wall={time.time()-t0:.1f}s"
            )

        entry = {
            **rollout_key,
            "condition": cond,
            "success": bool(rec.success),
            "steps": int(rec.steps),
            "wall_time_s": float(rec.wall_time_s),
            "termination_reason": rec.termination_reason,
            "n_overrides": len(mask),
            "override_indices": sorted(mask),
            "n_cycles_w2": n_cycles_w2,
            "n_cycles_fp16": len(fp16_per_cycle) if fp16_per_cycle else None,
        }
        utils.append_jsonl(entry, ROLLOUT_PATH)
        out.append(entry)

    return out


# ---------------------------------------------------------------------------
# Multi-seed driver
# ---------------------------------------------------------------------------
def run_trials(trials, conditions, n_grid=4, sigma=8.0, sis_stride=4, frac=0.20, verbose=False):
    """trials: list of (suite, task_id, seed, episode_idx) tuples."""
    utils.log(f"[expB] loading policy + model...")
    policy, model = utils.load_policy()

    utils.log(f"[expB] building PrecisionController (W2 + protect)...")
    ctrl = PrecisionController(model, bits=2, group_size=128)
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
                    frac=frac, verbose=verbose,
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
def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group()
    g.add_argument("--smoke", action="store_true",
                   help="1 trial, all 8 conditions — verifies plumbing")
    g.add_argument("--pilot", action="store_true",
                   help="20 trials × {FP16,W2,SIS-top,Random,MSE-W2traj} on libero_10 task 0")
    g.add_argument("--full", action="store_true",
                   help="100 trials (50 Long + 50 Object) × all 8 conditions @ frac=0.5")
    g.add_argument("--analyze", action="store_true",
                   help="produce summary markdown from existing JSONL")

    p.add_argument("--suite", default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--episode-idx", type=int, default=0)
    p.add_argument("--conditions", nargs="+", default=None,
                   help='subset of conditions, or "all" / "pilot"')

    p.add_argument("--n-grid", type=int, default=4,
                   help="SIS perturbation grid (NxN patches); default 4 for ~10x speedup over paper's 8")
    p.add_argument("--sigma", type=float, default=8.0)
    p.add_argument("--sis-stride", type=int, default=4,
                   help="recompute SIS every k cycles, reuse last value in between (paper §3 table 12)")
    p.add_argument("--frac", type=float, default=0.5,
                   help="override fraction; default 0.5 (pilot showed 0.2 was budget-bound)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--reset", action="store_true",
                   help="delete existing JSONLs before running")

    args = p.parse_args()

    if args.reset:
        for f in (DIAG_PATH, FP16_DIAG_PATH, ROLLOUT_PATH):
            if f.exists():
                utils.log(f"[expB] removing {f}")
                f.unlink()

    if args.analyze:
        analyze()
        return

    # Build trial set
    if args.smoke:
        trials = [("Long", 0, 0, 0)]
        conditions = ALL_CONDITIONS
    elif args.pilot:
        trials = pilot_trials()
        conditions = PILOT_CONDITIONS
    elif args.full:
        trials = full_trials()
        conditions = ALL_CONDITIONS
    else:
        if args.suite is None or args.task_id is None or not args.seeds:
            p.error("must specify --smoke / --pilot / --full / --analyze, "
                    "or (--suite + --task-id + --seeds)")
        trials = [(args.suite, args.task_id, s, args.episode_idx) for s in args.seeds]
        if args.conditions is None or args.conditions == ["all"]:
            conditions = ALL_CONDITIONS
        elif args.conditions == ["pilot"]:
            conditions = PILOT_CONDITIONS
        else:
            unknown = [c for c in args.conditions if c not in ALL_CONDITIONS]
            if unknown:
                p.error(f"unknown conditions: {unknown}; known: {ALL_CONDITIONS}")
            conditions = args.conditions

    utils.log(f"[expB] {len(trials)} trials × {len(conditions)} conditions = "
              f"{len(trials) * len(conditions)} rollouts (+ {len(trials)} diagnostic passes)")
    utils.log(f"[expB] conditions: {conditions}")

    run_trials(trials, conditions,
               n_grid=args.n_grid, sigma=args.sigma, sis_stride=args.sis_stride,
               frac=args.frac, verbose=args.verbose)
    analyze()


if __name__ == "__main__":
    main()
