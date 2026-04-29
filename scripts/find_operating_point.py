#!/usr/bin/env python3
"""
Step 2 of the LIBERO-PRO at W4 plan: find the per-suite operating point where
pi0.5 FP16 sits near ~70% success rate, so W4 has headroom to degrade and
attention-entropy-based rescue has signal to detect.

Sweeps `--magnitudes` (default 0.1, 0.2, 0.3) on `--suites` (default Object Goal),
runs `--n-per-cell` FP16 trials at each (suite, magnitude) cell using the same
matched-pair trial generator that Step 3 will use, and writes a markdown summary
with bootstrap CIs + a recommended D* per suite (closest to 70% FP16 SR).

Usage (on remote tambe-server-1, after setup_libero_pro.sh has run):
  CUDA_VISIBLE_DEVICES=<free-gpu> python scripts/find_operating_point.py \
      --suites Object Goal --magnitudes 0.1 0.2 0.3 --axis x --n-per-cell 50

Output: results/libero_pro_operating_point.md (overwritten on rerun)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import utils
import rollout as rollout_mod
from expB_sis_validation import pro_full_trials, _SUITE_TASK_BASE


RESULTS_DIR = Path(utils.RESULTS_DIR)
OUTPUT_PATH = RESULTS_DIR / "libero_pro_operating_point.md"


def _bootstrap_ci(values, n_boot=10_000, alpha=0.05, seed=0):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def run_cell(policy, suite: str, axis: str, magnitude: str, n_trials: int, verbose=False):
    """Stage the LIBERO-PRO bundle for (suite, axis, magnitude), then run
    `n_trials` FP16 rollouts at the matched-pair trial set for this suite.
    Returns list of dicts (one per trial) with keys success, suite, task_id,
    seed, episode_idx, steps, wall_time_s.
    """
    rollout_mod.set_libero_pro_config({suite: (axis, magnitude)})

    # Build the trial set for just this suite (n_per_suite=50 default → 10 tasks × 5 eps).
    eps_per_task = max(1, n_trials // 10)
    base = _SUITE_TASK_BASE[suite]
    trials = []
    for task_off in range(10):
        for ep in range(eps_per_task):
            global_task_id = base + task_off
            seed = task_off * 10 + ep
            trials.append((suite, global_task_id, seed, ep))
    trials = trials[:n_trials]

    out = []
    t_cell_start = time.time()
    for i, (s, task_id, seed, ep) in enumerate(trials):
        t0 = time.time()
        rec = rollout_mod.run_rollout(
            policy, task_id=task_id, suite=s, seed=seed, episode_idx=ep, verbose=False,
        )
        wall = time.time() - t0
        out.append({
            "suite": s, "axis": axis, "magnitude": magnitude,
            "task_id": int(task_id), "seed": int(seed), "episode_idx": int(ep),
            "success": bool(rec.success), "steps": int(rec.steps),
            "wall_time_s": float(rec.wall_time_s),
        })
        if verbose or (i + 1) % 10 == 0:
            so_far = sum(1 for r in out if r["success"])
            utils.log(
                f"[op-point] {s} {axis}{magnitude} trial {i+1}/{len(trials)} "
                f"succ={so_far}/{i+1} ({so_far/(i+1):.0%}) wall={wall:.1f}s"
            )
    cell_wall = time.time() - t_cell_start
    successes = [r["success"] for r in out]
    sr, lo, hi = _bootstrap_ci(successes)
    utils.log(
        f"[op-point] CELL DONE {suite} {axis}{magnitude}: "
        f"SR={sr:.2f} [{lo:.2f}, {hi:.2f}] over {len(out)} trials, "
        f"cell wall={cell_wall/60:.1f} min"
    )
    return out


def write_markdown(rows: list, out_path: Path, target_sr: float = 0.70):
    """Group rows by (suite, magnitude), compute SR + CI, pick D* per suite."""
    by_cell = {}
    for r in rows:
        key = (r["suite"], r["axis"], r["magnitude"])
        by_cell.setdefault(key, []).append(r["success"])

    lines = []
    lines.append("# LIBERO-PRO operating point sweep\n")
    lines.append("Per-cell FP16 success rate (50 trials each) on LIBERO-PRO position perturbation.\n")
    lines.append("Goal: pick `D*` per suite closest to FP16 ≈ 70% SR — the regime where W4 has\n"
                 "headroom to degrade. Step 3 uses these `D*` values for the focused expC subset.\n")

    # Per-cell table
    lines.append("\n## Per-cell results\n")
    lines.append("| Suite | Axis | Magnitude | n | SR | 95% CI | gap to 70% |")
    lines.append("|-------|------|-----------|---|------:|-------|-----------:|")
    cell_summary = {}
    for key in sorted(by_cell):
        suite, axis, mag = key
        succs = by_cell[key]
        sr, lo, hi = _bootstrap_ci(succs)
        gap = abs(sr - target_sr)
        cell_summary[key] = (sr, lo, hi, gap, len(succs))
        lines.append(
            f"| {suite} | {axis} | {mag} | {len(succs)} | {sr:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {gap:.3f} |"
        )

    # Per-suite D* recommendation
    lines.append(f"\n## Recommended D* per suite (target FP16 ≈ {target_sr:.0%})\n")
    lines.append("| Suite | D* | SR at D* | rule |")
    lines.append("|-------|----|----------:|------|")
    suites_seen = sorted({k[0] for k in cell_summary})
    for suite in suites_seen:
        # Pick the cell for this suite with smallest gap to target_sr.
        cands = [(k, v) for k, v in cell_summary.items() if k[0] == suite]
        cands.sort(key=lambda kv: kv[1][3])  # sort by gap
        (suite_, axis, mag), (sr, lo, hi, gap, n) = cands[0]
        # Edge-case rules from the plan: avoid saturated/collapsed regimes.
        if sr >= 0.85:
            note = "saturated; consider extending sweep to higher magnitudes"
        elif sr < 0.30:
            note = "collapsed; consider dropping suite or going to lower magnitudes"
        else:
            note = "healthy operating point"
        lines.append(
            f"| {suite} | {axis}{mag} | {sr:.3f} | {note} |"
        )

    # Footer
    lines.append("\n## Files\n")
    lines.append(f"- Per-trial rows written to: `results/libero_pro_operating_point.jsonl`")

    out_path.write_text("\n".join(lines) + "\n")
    utils.log(f"[op-point] wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suites", nargs="+", default=["Object", "Goal"],
                   choices=list(rollout_mod.SUITE_TO_OPENPI_NAME))
    p.add_argument("--magnitudes", nargs="+", default=["0.1", "0.2", "0.3"],
                   help="LIBERO-PRO magnitude levels (one of 0.1/0.2/0.3/0.4/0.5)")
    p.add_argument("--axis", default="x", choices=["x", "y"])
    p.add_argument("--n-per-cell", type=int, default=50)
    p.add_argument("--target-sr", type=float, default=0.70)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    utils.setup_logging()

    # Validate magnitudes
    for m in args.magnitudes:
        try:
            mf = float(m)
        except ValueError:
            p.error(f"magnitude must be numeric, got {m!r}")
        if mf not in (0.1, 0.2, 0.3, 0.4, 0.5):
            p.error(f"magnitude {m!r} not in {{0.1, 0.2, 0.3, 0.4, 0.5}}")

    utils.log(f"[op-point] suites={args.suites} mags={args.magnitudes} axis={args.axis} "
              f"n_per_cell={args.n_per_cell} -> {len(args.suites) * len(args.magnitudes)} cells, "
              f"{len(args.suites) * len(args.magnitudes) * args.n_per_cell} total trials")

    utils.log("[op-point] loading policy...")
    policy, _ = utils.load_policy()

    out_jsonl = RESULTS_DIR / "libero_pro_operating_point.jsonl"
    if out_jsonl.exists():
        utils.log(f"[op-point] WARN: {out_jsonl} already exists; appending")

    all_rows = []
    for suite in args.suites:
        for mag in args.magnitudes:
            mag_f = float(mag)
            mag_str = f"{mag_f:.1f}"
            cell_rows = run_cell(
                policy, suite, args.axis, mag_str,
                n_trials=args.n_per_cell, verbose=args.verbose,
            )
            for r in cell_rows:
                utils.append_jsonl(r, out_jsonl)
            all_rows.extend(cell_rows)

    write_markdown(all_rows, OUTPUT_PATH, target_sr=args.target_sr)
    utils.log("[op-point] DONE")


if __name__ == "__main__":
    main()
