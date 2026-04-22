#!/usr/bin/env python3
"""
ExpB — Saliency-Aware PTQ Validation (SIS-gated FP16 rescue).

Counterfactual frame-selection test for SQIL's State Importance Score (SIS),
adapted for our PTQ setup: instead of using SIS to weight a QAT loss, we use
it as a per-frame precision gate that overrides W2-with-protection inference
with FP16 inference at high-SIS frames.

Seven matched conditions per (suite, task_id, seed, episode_idx):
  1. W2-only         — pure w2_vlm_protect (lower bound)
  2. FP16-only       — pure FP16 (upper bound, = exp0)
  3. SIS-top-20      — W2 base + FP16 override at top-20% inference cycles by SIS
  4. Random-20       — W2 base + FP16 override at random 20%
  5. Oracle-20       — W2 base + FP16 override at top-20% by ground-truth ||a_FP-a_W2||²
  6. Bottom-SIS-20   — W2 base + FP16 override at bottom-20% by SIS (symmetry control)
  7. AttnEntropy-top-20 — W2 base + FP16 override at top-20% by 1/entropy at l12h2

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
ROLLOUT_PATH = RESULTS_DIR / "expB_rollouts.jsonl"
SUMMARY_PATH = RESULTS_DIR / "expB_summary.md"

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
ALL_CONDITIONS = [
    "W2",                  # 1
    "FP16",                # 2
    "SIS-top-20",          # 3
    "Random-20",           # 4
    "Oracle-20",           # 5
    "Bottom-SIS-20",       # 6
    "AttnEntropy-top-20",  # 7
]
PILOT_CONDITIONS = ["W2", "FP16", "SIS-top-20", "Random-20", "Oracle-20"]


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


def build_masks(per_cycle, frac: float, seed: int) -> dict:
    """Build the override frame-index masks for the 5 score-based conditions
    plus the trivial FP16-all mask."""
    n = len(per_cycle)
    k = max(1, int(round(frac * n)))

    sis = [c["sis"] for c in per_cycle]
    mse = [c["mse_fp_w2"] for c in per_cycle]
    attn = [c["attn_entropy_l12h2"] for c in per_cycle]

    rng = _random.Random(seed)
    rand_indices = set(rng.sample(range(n), k))

    return {
        "FP16":                set(range(n)),
        "SIS-top-20":          _topk_indices(sis, k, largest=True),
        "Random-20":           rand_indices,
        "Oracle-20":           _topk_indices(mse, k, largest=True),
        "Bottom-SIS-20":       _topk_indices(sis, k, largest=False),
        # Low entropy → high sensitivity (D2 finding ρ=-0.294); pick smallest entropy.
        "AttnEntropy-top-20":  _topk_indices(attn, k, largest=False),
    }


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

    diag_rec, per_cycle = None, None
    if "W2" in conditions or any(c in conditions for c in
                                 ["SIS-top-20", "Random-20", "Oracle-20",
                                  "Bottom-SIS-20", "AttnEntropy-top-20"]):
        utils.log(f"[expB] DIAG seed={seed} task={task_id} ep={episode_idx}")
        t0 = time.time()
        diag_rec, per_cycle = diagnostic_rollout(
            policy, model, ctrl, attn_hook,
            suite, task_id, seed, episode_idx,
            n_grid=n_grid, sigma=sigma, sis_stride=sis_stride,
            verbose=verbose,
        )
        utils.log(
            f"[expB] DIAG done seed={seed} success={diag_rec.success} "
            f"steps={diag_rec.steps} cycles={len(per_cycle)} wall={time.time()-t0:.1f}s"
        )
        # Persist diagnostic
        for c in per_cycle:
            utils.append_jsonl({**rollout_key, **c}, DIAG_PATH)

    masks = build_masks(per_cycle, frac=frac, seed=seed) if per_cycle else {}

    for cond in conditions:
        if cond == "W2":
            rec = diag_rec
            mask = set()
        else:
            mask = masks[cond]
            t0 = time.time()
            utils.log(
                f"[expB] {cond} seed={seed} task={task_id} ep={episode_idx} "
                f"|mask|={len(mask)}/{len(per_cycle) if per_cycle else 0}"
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
            "n_cycles": len(per_cycle) if per_cycle else None,
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
    needed = {"W2", "FP16", "SIS-top-20", "Random-20", "Oracle-20"}
    if needed.issubset(by_cond.keys()):
        sr = {c: float(np.mean(by_cond[c])) for c in ALL_CONDITIONS if c in by_cond}
        lines.append("\n## Hypothesis matrix\n")
        lines.append(f"- SR(W2) = {sr['W2']:.3f}")
        lines.append(f"- SR(FP16) = {sr['FP16']:.3f}")
        lines.append(f"- SR(Oracle-20) = {sr['Oracle-20']:.3f}")
        lines.append(f"- SR(SIS-top-20) = {sr['SIS-top-20']:.3f}")
        lines.append(f"- SR(Random-20) = {sr['Random-20']:.3f}")
        if "Bottom-SIS-20" in sr:
            lines.append(f"- SR(Bottom-SIS-20) = {sr['Bottom-SIS-20']:.3f}")
        if "AttnEntropy-top-20" in sr:
            lines.append(f"- SR(AttnEntropy-top-20) = {sr['AttnEntropy-top-20']:.3f}")

        delta_sis_rand = sr["SIS-top-20"] - sr["Random-20"]
        delta_sis_oracle = sr["Oracle-20"] - sr["SIS-top-20"]
        lines.append("")
        lines.append(f"- SIS over Random = {delta_sis_rand:+.3f}")
        lines.append(f"- Oracle headroom over SIS = {delta_sis_oracle:+.3f}")

        if delta_sis_rand <= 0:
            verdict = "**KILL**: SR(SIS-top-20) ≤ SR(Random-20). SIS does not carry quant-specific signal at this quantization level."
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
                   help="1 trial, all 7 conditions — verifies plumbing")
    g.add_argument("--pilot", action="store_true",
                   help="20 trials × {W2,FP16,SIS,Random,Oracle} on libero_10 task 0")
    g.add_argument("--full", action="store_true",
                   help="100 trials (50 Long + 50 Object) × all 7 conditions")
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
    p.add_argument("--frac", type=float, default=0.20)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--reset", action="store_true",
                   help="delete existing JSONLs before running")

    args = p.parse_args()

    if args.reset:
        for f in (DIAG_PATH, ROLLOUT_PATH):
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
