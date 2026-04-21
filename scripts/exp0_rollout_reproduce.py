#!/usr/bin/env python3
"""
Experiment 0 — FP16 pi0.5 LIBERO reproduction check.

Validates that our in-process rollout harness (scripts/rollout.py) can
reproduce pi0.5's published success rates on a small subset. Must pass
before we layer quantization or attention capture on top.

Matrix:
  3 tasks × 3 seeds × {Object, Long} = 18 rollouts at FP16.

Compares against QuantVLA Table 2 FP16 reference:
  pi0.5 Object 99.0%, Long 93.5%, avg 97.1%.

Runtime estimate: ~20 minutes on H100.

Usage:
  python scripts/exp0_rollout_reproduce.py                 # full 18-rollout run
  python scripts/exp0_rollout_reproduce.py --smoke         # single quick rollout
  python scripts/exp0_rollout_reproduce.py --suites Object # only one suite
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import utils
import rollout


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_SUITES     = ("Object", "Long")      # task_index 20-29, 0-9
TASKS_PER_SUITE    = 3                        # first N task_ids within the suite
SEEDS              = (0, 1, 2)
SEEDS_SMOKE        = (0,)

# QuantVLA Table 2 published FP16 numbers (pi0.5 on LIBERO)
PUBLISHED_FP16 = {
    "Object":  0.990,
    "Long":    0.935,
    "Goal":    0.975,
    "Spatial": 0.985,
}

# Global task_id ranges per suite (matches utils.suite_of())
SUITE_GLOBAL_TASK_BASE = {
    "Long":    0,
    "Goal":    10,
    "Object":  20,
    "Spatial": 30,
}


# ---------------------------------------------------------------------------
# Table formatting (fixed-width ASCII; primary output per user preference)
# ---------------------------------------------------------------------------
def fmt_table(header, rows, aligns=None):
    """Build a fixed-width ASCII table. aligns: '<' left (default) or '>' right."""
    n = len(header)
    if aligns is None:
        aligns = ["<"] * n
    widths = [len(str(h)) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    def _fmt_row(row):
        return " | ".join(f"{str(c):{aligns[i]}{widths[i]}}" for i, c in enumerate(row))
    sep = "-+-".join("-" * w for w in widths)
    lines = [_fmt_row(header), sep]
    lines.extend(_fmt_row(r) for r in rows)
    return "\n".join(lines)


def write_tables(records, out_path):
    """Write all 4 tables to `out_path` as markdown + print to stdout."""
    lines = ["# Exp0 — FP16 pi0.5 LIBERO Reproduction\n",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"n_rollouts: {len(records)}\n"]

    # Table 1: per-rollout detail
    rows = []
    for r in records:
        rows.append([
            r["suite"],
            r["task_id"],
            r["seed"],
            ("✓" if r["success"] else "✗"),
            r["steps"],
            r["termination_reason"],
            f"{r['wall_time_s']:.1f}",
        ])
    t1 = fmt_table(
        ["suite", "task_id", "seed", "success", "steps", "termination", "wall_s"],
        rows,
        aligns=["<", ">", ">", "^", ">", "<", ">"],
    )
    lines += ["\n## Table 1 — Per-rollout detail\n", "```", t1, "```\n"]

    # Table 2: per-task success rate
    by_task = defaultdict(list)
    for r in records:
        by_task[(r["suite"], r["task_id"], r.get("task_description", ""))].append(r["success"])
    rows = []
    for (suite, tid, desc), succs in sorted(by_task.items()):
        rows.append([
            suite,
            tid,
            (desc[:60] + "...") if len(desc) > 60 else desc,
            f"{sum(succs)}/{len(succs)}",
        ])
    t2 = fmt_table(
        ["suite", "task_id", "task_prompt", "success"],
        rows,
        aligns=["<", ">", "<", ">"],
    )
    lines += ["\n## Table 2 — Per-task success rate\n", "```", t2, "```\n"]

    # Table 3: per-suite summary vs published
    by_suite = defaultdict(list)
    for r in records:
        by_suite[r["suite"]].append(r["success"])
    rows = []
    for suite, succs in sorted(by_suite.items()):
        n = len(succs)
        k = sum(succs)
        pub = PUBLISHED_FP16.get(suite)
        our = k / max(n, 1)
        delta = f"{(our - pub)*100:+.1f}pp" if pub is not None else "n/a"
        rows.append([
            suite,
            f"{k}/{n} = {our*100:.1f}%",
            f"{pub*100:.1f}%" if pub is not None else "n/a",
            delta,
            n,
        ])
    t3 = fmt_table(
        ["suite", "our FP16 success", "published (QuantVLA T2)", "Δ", "n"],
        rows,
        aligns=["<", ">", ">", ">", ">"],
    )
    lines += ["\n## Table 3 — Per-suite summary vs published\n", "```", t3, "```\n"]

    # Table 4: error diagnostic (only if any)
    errors = [r for r in records if r["termination_reason"] == "error"]
    if errors:
        rows = [[r["task_id"], r["seed"], r.get("exception", "")] for r in errors]
        t4 = fmt_table(["task_id", "seed", "exception"], rows,
                       aligns=[">", ">", "<"])
        lines += ["\n## Table 4 — Errors\n", "```", t4, "```\n"]
    else:
        lines += ["\n## Table 4 — Errors\n", "_none_\n"]

    # Overall verdict
    total_n = len(records)
    total_k = sum(r["success"] for r in records)
    overall = total_k / max(total_n, 1)
    lines.append("\n## Verdict\n")
    lines.append(f"Overall success: {total_k}/{total_n} = {overall*100:.1f}%")
    if overall < 0.60:
        lines.append("\n**LIKELY BUG — investigate before trusting numbers.**")
        lines.append("Common causes: obs-format mismatch (check adapter), "
                     "image rotation (must be ::-1,::-1), "
                     "action space mismatch, "
                     "headless rendering artifacts (try MUJOCO_GL=osmesa).")
    elif overall >= 0.80:
        lines.append("\n**Infrastructure OK** — proceed to Phase 1 (trajectory attention analysis).")
    else:
        lines.append("\n**Marginal** — acceptable but check per-suite breakdown for outliers.")

    content = "\n".join(lines) + "\n"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(content)
    utils.log(f"[exp0] tables → {out_path}")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_exp0(suites, tasks_per_suite, seeds, jsonl_path, smoke=False):
    """Load policy once, run rollout grid, append per-rollout JSONL, emit tables."""
    with utils.Timer("Model loading"):
        policy, _ = utils.load_policy("pi05_libero")

    # Pre-flight: headless render + import sanity
    utils.log("[exp0] Pre-flight headless render check...")
    try:
        rollout.smoke_render()
        utils.log("[exp0]   PASS")
    except Exception as e:
        utils.log(f"[exp0]   FAIL: {e}")
        utils.log("[exp0] Aborting — fix MUJOCO_GL and retry.")
        return 1

    # Clear previous results
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    records = []
    total = len(suites) * tasks_per_suite * len(seeds)
    done = 0
    t_all = time.time()

    for suite in suites:
        base = SUITE_GLOBAL_TASK_BASE[suite]
        for local_tid in range(tasks_per_suite):
            global_task_id = base + local_tid

            # Build env ONCE per task (reuse across seeds for that task)
            try:
                env, task_desc, init_states = rollout.make_libero_env(
                    suite=suite, task_id=global_task_id, seed=seeds[0],
                )
            except Exception as e:
                utils.log(f"[exp0] make_libero_env FAILED for {suite}#{global_task_id}: {e}")
                for seed in seeds:
                    rec = rollout.RolloutRecord(
                        success=False, steps=0, task_id=global_task_id, suite=suite,
                        seed=seed, episode_idx=0,
                        task_description="(env-build-failed)",
                        termination_reason="error", wall_time_s=0.0,
                        exception=f"{type(e).__name__}: {e}",
                    )
                    records.append(rec.to_dict())
                    utils.append_jsonl(rec.to_dict(), jsonl_path)
                    done += 1
                continue

            try:
                for ep_idx, seed in enumerate(seeds):
                    t0 = time.time()
                    utils.log(
                        f"\n[exp0] ({done+1}/{total}) "
                        f"suite={suite} task_id={global_task_id} seed={seed} "
                        f"desc={task_desc[:50]!r}"
                    )
                    rec = rollout.run_rollout(
                        policy,
                        task_id=global_task_id,
                        suite=suite,
                        seed=seed,
                        episode_idx=ep_idx,
                        env=env,
                        initial_states=init_states,
                        task_description=task_desc,
                        verbose=False,
                    )
                    records.append(rec.to_dict())
                    utils.append_jsonl(rec.to_dict(), jsonl_path)
                    done += 1
                    dt = time.time() - t0
                    utils.log(
                        f"[exp0]   result: success={rec.success} "
                        f"steps={rec.steps} termination={rec.termination_reason} "
                        f"wall={dt:.1f}s  | total so far: "
                        f"{sum(r['success'] for r in records)}/{len(records)} "
                        f"({100*sum(r['success'] for r in records)/max(len(records),1):.1f}%)"
                    )
            finally:
                try:
                    env.close()
                except Exception:
                    pass

    utils.log(f"\n[exp0] Completed {len(records)} rollouts in {(time.time()-t_all)/60:.1f} min")
    return records


def main():
    utils.setup_logging()
    utils.log("=" * 60)
    utils.log("EXPERIMENT 0: FP16 pi0.5 LIBERO Reproduction Check")
    utils.log("=" * 60)

    p = argparse.ArgumentParser()
    p.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES),
                   choices=list(SUITE_GLOBAL_TASK_BASE))
    p.add_argument("--tasks-per-suite", type=int, default=TASKS_PER_SUITE)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--smoke", action="store_true",
                   help="Small run: 1 task × 1 seed × Object only.")
    args = p.parse_args()

    if args.smoke:
        args.suites = ["Object"]
        args.tasks_per_suite = 1
        args.seeds = list(SEEDS_SMOKE)
        utils.log("[exp0] SMOKE mode: Object × 1 task × 1 seed")

    utils.log(f"[exp0] suites={args.suites}  tasks/suite={args.tasks_per_suite}  "
              f"seeds={args.seeds}  → {len(args.suites) * args.tasks_per_suite * len(args.seeds)} rollouts")

    jsonl_path = os.path.join(utils.RESULTS_DIR, "exp0_rollouts.jsonl")
    records = run_exp0(args.suites, args.tasks_per_suite, args.seeds, jsonl_path,
                       smoke=args.smoke)
    if isinstance(records, int):
        return records  # error code

    tables_path = os.path.join(utils.RESULTS_DIR, "exp0_rollout_tables.md")
    write_tables(records, tables_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
