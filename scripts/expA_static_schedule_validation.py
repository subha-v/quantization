#!/usr/bin/env python3
"""
Experiment A — Rollout-level validation of the exp2+exp3 static precision schedule.

Exp2 found that at W2, two VLM layers (layer 0 + SigLIP vision tower) are
catastrophic bottlenecks and must be protected. Exp3 found that the action
expert's final denoising step (step 9) is ~75× more sensitive than step 0 at
W4, so a W4-first-9-steps / FP16-step-9 schedule preserves per-frame accuracy.

This experiment runs full closed-loop LIBERO rollouts under 5 quantization
configs to test whether the static schedule actually preserves rollout success:

  C0 fp16             — FP16 ceiling baseline
  C1 w4_uniform       — uniform W4 everywhere (naive baseline, no schedule)
  C2 static_schedule  — VLM W4-with-protection + expert W4-steps-0-8-FP16-step-9  ← headline
  C3 reverse_step     — VLM W4-with-protection + expert FP16-steps-0-8-W4-step-9  ← control
  C4 w2_protect       — VLM W2-with-protection + expert W2 (expected to fail)

C2 vs C3 is the critical comparison: if C2 preserves success while C3 collapses,
exp3's "step 9 dominates" finding is validated at rollout level.

50 rollouts per config = 5 tasks × 5 seeds × 2 suites (Object, Long).
Expected runtime ~90 min on H100 for all 5 configs.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import utils
import rollout
from exp6_attention_predicts_quant import install_quant, uninstall_quant, find_expert


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASKS_PER_SUITE = 5
SEEDS = (0, 1, 2, 3, 4)
SUITES = ("Object", "Long")
SUITE_BASE = {"Long": 0, "Goal": 10, "Object": 20, "Spatial": 30}
NUM_STEPS = 10   # pi0.5 Euler step count


# ---------------------------------------------------------------------------
# StepController — per-step expert weight swap, adapted from exp3
# ---------------------------------------------------------------------------
class StepController:
    """Monkey-patches model.denoise_step so the action expert can be swapped
    between FP16 and a pre-quantized weight set at each Euler step boundary.

    Adapted from exp3. Unlike exp3's usage pattern (one obs at a time), we use
    this during closed-loop rollouts where policy.infer() is called many times
    per rollout. The caller must invoke controller.set(quantize_steps) via the
    pre_infer_callback to reset the step counter before each policy.infer()."""

    def __init__(self, model, expert_module, orig_weights, quant_weights):
        self.model = model
        self.expert_module = expert_module
        self.orig_weights = orig_weights
        self.quant_weights = quant_weights
        self.quantize_steps = set()
        self.step = 0
        self._original_denoise_step = model.denoise_step
        self._installed = False

    def set(self, quantize_steps):
        """Reset counter and set which steps quantize. Call before each infer."""
        self.quantize_steps = set(quantize_steps)
        self.step = 0

    def install(self):
        if self._installed:
            return
        controller = self
        original = self._original_denoise_step

        def patched(state, prefix_pad_masks, past_key_values, x_t, timestep):
            # Auto-wrap if counter exceeded (safety net — shouldn't happen if
            # caller always calls set() before each infer)
            if controller.step >= NUM_STEPS:
                controller.step = 0
            if controller.step in controller.quantize_steps:
                utils.swap_weights(controller.expert_module, controller.quant_weights)
            else:
                utils.swap_weights(controller.expert_module, controller.orig_weights)
            try:
                return original(state, prefix_pad_masks, past_key_values, x_t, timestep)
            finally:
                controller.step += 1

        self.model.denoise_step = patched
        self._installed = True

    def uninstall(self):
        if not self._installed:
            return
        try:
            del self.model.denoise_step
        except AttributeError:
            pass
        utils.swap_weights(self.expert_module, self.orig_weights)
        self._installed = False


# ---------------------------------------------------------------------------
# Per-config install / uninstall
# ---------------------------------------------------------------------------
def install_config(model, cfg_name):
    """Install the quantization setup for a given config.
    Returns (vlm_saved, expert_handle) where expert_handle is either
    a StepController (for C2/C3) or None (for C0/C1/C4 — expert handled by
    install_quant together with VLM).
    """
    utils.log(f"[expA] Installing config: {cfg_name}")

    if cfg_name == "fp16":
        return None, None

    if cfg_name == "w4_uniform":
        # Uniform W4 on VLM + expert, no protection, no step schedule
        saved = install_quant(model, "w4_both")
        return saved, None

    if cfg_name == "w2_protect":
        # VLM W2 with layer-0/vision-tower protection + expert W2 uniform
        saved = install_quant(model, "w2_both")
        return saved, None

    if cfg_name in ("static_schedule", "reverse_step"):
        # VLM: W4 with protection (install_quant handles this)
        vlm_saved = install_quant(model, "w4_vlm_protect")
        # Expert: per-step switching via StepController
        expert_name, expert_module = find_expert(model)
        orig_ptrs, quant_tensors = utils.precompute_quantized_weights(
            expert_module, bits=4)
        controller = StepController(model, expert_module, orig_ptrs, quant_tensors)
        controller.install()
        utils.log(f"[expA]   expert: {expert_name}  ({len(orig_ptrs)} Linears at W4 for schedule)")
        return vlm_saved, controller

    raise ValueError(f"unknown config {cfg_name}")


def uninstall_config(model, vlm_saved, expert_handle):
    if expert_handle is not None:
        # StepController case (C2/C3)
        expert_handle.uninstall()
    if vlm_saved is not None:
        uninstall_quant(model, vlm_saved)


def make_pre_infer_callback(cfg_name, expert_handle):
    """Returns a pre_infer_callback for run_rollout, or None.

    For static_schedule (C2): W4 at steps 0-8, FP16 at step 9 → quantize_steps = {0..8}
    For reverse_step (C3):  FP16 at steps 0-8, W4 at step 9 → quantize_steps = {9}
    For C0/C1/C4: no callback needed (no per-step switching)
    """
    if cfg_name == "static_schedule":
        q_steps = set(range(9))   # steps 0, 1, ..., 8
    elif cfg_name == "reverse_step":
        q_steps = {9}
    else:
        return None

    def cb(step_idx):
        expert_handle.set(q_steps)
    return cb


# ---------------------------------------------------------------------------
# Smoke rollout — per config, before full sweep
# ---------------------------------------------------------------------------
def smoke_rollout(policy, cfg_name, pre_infer_cb):
    """1 rollout on Object task 20 seed 0 to verify config doesn't crash.
    Returns the RolloutRecord."""
    utils.log(f"[expA] Smoke rollout for {cfg_name}...")
    t0 = time.time()
    rec = rollout.run_rollout(
        policy, task_id=20, suite="Object", seed=0, episode_idx=0,
        pre_infer_callback=pre_infer_cb, verbose=False,
    )
    wall = time.time() - t0
    utils.log(f"[expA]   smoke: success={rec.success} steps={rec.steps} "
              f"termination={rec.termination_reason} wall={wall:.1f}s")
    return rec


# ---------------------------------------------------------------------------
# Per-config full sweep
# ---------------------------------------------------------------------------
def run_config_sweep(policy, cfg_name, vlm_saved, expert_handle, out_path):
    """Run 5 tasks × 5 seeds × 2 suites under the currently-installed config."""
    pre_infer_cb = make_pre_infer_callback(cfg_name, expert_handle)

    # Smoke check first — skip full sweep if config is broken (unless it's w2_protect
    # which we expect to fail rollouts)
    smoke = smoke_rollout(policy, cfg_name, pre_infer_cb)
    if cfg_name != "w2_protect" and smoke.termination_reason == "error":
        utils.log(f"[expA] SMOKE ERROR for {cfg_name}: {smoke.exception}; SKIPPING full sweep")
        return []

    records = []
    t_cfg_start = time.time()
    done = 0
    total = TASKS_PER_SUITE * len(SEEDS) * len(SUITES)

    for suite in SUITES:
        base = SUITE_BASE[suite]
        for local_tid in range(TASKS_PER_SUITE):
            global_task_id = base + local_tid
            try:
                env, desc, init_states = rollout.make_libero_env(
                    suite=suite, task_id=global_task_id, seed=SEEDS[0])
            except Exception as e:
                utils.log(f"[expA] env-build FAILED {suite}#{global_task_id}: {e}")
                # Emit error records for all seeds of this task
                for seed in SEEDS:
                    rec_d = {
                        "quant_config": cfg_name,
                        "suite": suite, "task_id": global_task_id, "seed": seed,
                        "episode_idx": seed,
                        "task_description": "(env-build-failed)",
                        "success": False, "steps": 0,
                        "termination_reason": "error", "wall_time_s": 0.0,
                        "final_reward": 0.0,
                        "exception": f"{type(e).__name__}: {e}",
                    }
                    records.append(rec_d)
                    utils.append_jsonl(rec_d, out_path)
                    done += 1
                continue

            try:
                for seed in SEEDS:
                    done += 1
                    t0 = time.time()
                    rec = rollout.run_rollout(
                        policy, task_id=global_task_id, suite=suite,
                        seed=seed, episode_idx=seed,
                        env=env, initial_states=init_states, task_description=desc,
                        pre_infer_callback=pre_infer_cb,
                    )
                    wall = time.time() - t0
                    rec_d = {"quant_config": cfg_name, **rec.to_dict()}
                    records.append(rec_d)
                    utils.append_jsonl(rec_d, out_path)

                    # Cumulative success
                    ok_so_far = sum(1 for r in records if r["success"])
                    utils.log(
                        f"[expA] ({done}/{total}) cfg={cfg_name} {suite}#{global_task_id} s{seed}: "
                        f"success={'✓' if rec.success else '✗'} steps={rec.steps} "
                        f"term={rec.termination_reason} wall={wall:.1f}s | "
                        f"cumul: {ok_so_far}/{len(records)}"
                    )
            finally:
                try: env.close()
                except Exception: pass

    dt = time.time() - t_cfg_start
    utils.log(f"[expA] {cfg_name} sweep done: {dt/60:.1f} min  "
              f"{sum(1 for r in records if r['success'])}/{len(records)} succeeded")
    return records


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def fmt_table(header, rows, aligns=None):
    n = len(header)
    if aligns is None: aligns = ["<"] * n
    widths = [len(str(h)) for h in header]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(str(c)))
    def _r(row):
        return " | ".join(f"{str(c):{aligns[i]}{widths[i]}}" for i, c in enumerate(row))
    return "\n".join([_r(header), "-+-".join("-" * w for w in widths)] + [_r(r) for r in rows])


def write_tables(all_records, out_path):
    lines = ["# Experiment A — Rollout-level static schedule validation\n",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"Total rollouts: {len(all_records)}\n"]

    by_cfg = defaultdict(list)
    for r in all_records:
        by_cfg[r["quant_config"]].append(r)

    config_order = ["fp16", "w4_uniform", "static_schedule", "reverse_step", "w2_protect"]

    # --- Table 1: per-config × per-suite success rate ---
    rows = []
    for cfg in config_order:
        if cfg not in by_cfg: continue
        recs = by_cfg[cfg]
        by_suite = defaultdict(list)
        for r in recs:
            by_suite[r["suite"]].append(r)
        row = [cfg]
        for suite in ("Object", "Long"):
            s = by_suite.get(suite, [])
            n = len(s); k = sum(1 for r in s if r["success"])
            row.append(f"{k}/{n} = {k/max(n,1)*100:.0f}%")
        all_k = sum(1 for r in recs if r["success"])
        row.append(f"{all_k}/{len(recs)} = {all_k/max(len(recs),1)*100:.0f}%")
        rows.append(row)
    lines += ["\n## Table 1 — Success rate per config × suite\n", "```",
              fmt_table(["config", "Object", "Long", "overall"], rows,
                        ["<", ">", ">", ">"]),
              "```\n"]

    # --- Table 2: per-task × per-config success matrix ---
    rows = []
    for suite in ("Object", "Long"):
        base = SUITE_BASE[suite]
        for local_tid in range(TASKS_PER_SUITE):
            tid = base + local_tid
            row = [suite, tid]
            for cfg in config_order:
                if cfg not in by_cfg:
                    row.append("-")
                    continue
                subset = [r for r in by_cfg[cfg]
                          if r["suite"] == suite and r["task_id"] == tid]
                k = sum(1 for r in subset if r["success"])
                row.append(f"{k}/{len(subset)}")
            rows.append(row)
    lines += ["\n## Table 2 — Per-task success matrix\n", "```",
              fmt_table(["suite", "task"] + config_order, rows,
                        ["<", ">"] + [">"] * len(config_order)),
              "```\n"]

    # --- Table 3: C2 vs C3 direct comparison (paired by task × seed) ---
    if "static_schedule" in by_cfg and "reverse_step" in by_cfg:
        lines += ["\n## Table 3 — C2 vs C3 direct (headline comparison)\n",
                  "If C2 >> C3 in success rate, exp3's step-9 finding is validated at rollout level.\n"]
        rows = []
        for suite in ("Object", "Long"):
            c2 = [r for r in by_cfg["static_schedule"] if r["suite"] == suite]
            c3 = [r for r in by_cfg["reverse_step"] if r["suite"] == suite]
            c2k = sum(1 for r in c2 if r["success"])
            c3k = sum(1 for r in c3 if r["success"])
            gap_pp = (c2k/max(len(c2),1) - c3k/max(len(c3),1)) * 100
            # Paired disagreements: same (task, seed) where one succeeds and other fails
            c2_map = {(r["task_id"], r["seed"]): r["success"] for r in c2}
            c3_map = {(r["task_id"], r["seed"]): r["success"] for r in c3}
            common = set(c2_map) & set(c3_map)
            c2_only = sum(1 for k in common if c2_map[k] and not c3_map[k])
            c3_only = sum(1 for k in common if c3_map[k] and not c2_map[k])
            both = sum(1 for k in common if c2_map[k] and c3_map[k])
            neither = sum(1 for k in common if not c2_map[k] and not c3_map[k])
            rows.append([
                suite,
                f"{c2k}/{len(c2)}", f"{c3k}/{len(c3)}",
                f"{gap_pp:+.0f}pp",
                both, c2_only, c3_only, neither,
            ])
        lines += ["```",
                  fmt_table(
                      ["suite", "C2 succ", "C3 succ", "Δ (pp)",
                       "both ✓", "only C2 ✓", "only C3 ✓", "both ✗"],
                      rows,
                      ["<", ">", ">", ">", ">", ">", ">", ">"]),
                  "```\n"]

    # --- Table 4: per-config runtime + step stats ---
    rows = []
    for cfg in config_order:
        if cfg not in by_cfg: continue
        recs = by_cfg[cfg]
        steps = [r["steps"] for r in recs if r["steps"] > 0]
        walls = [r["wall_time_s"] for r in recs]
        rows.append([
            cfg,
            len(recs),
            f"{np.mean(steps):.0f}" if steps else "—",
            f"{np.median(steps):.0f}" if steps else "—",
            f"{np.mean(walls):.1f}",
            f"{sum(walls)/60:.1f} min",
        ])
    lines += ["\n## Table 4 — Runtime / step-count\n", "```",
              fmt_table(
                  ["config", "n", "mean steps", "median steps",
                   "mean wall_s", "total wall"],
                  rows, ["<", ">", ">", ">", ">", ">"]),
              "```\n"]

    # --- Table 5: errors (only if any) ---
    errs = [r for r in all_records if r["termination_reason"] == "error"]
    if errs:
        rows = [[r["quant_config"], r["suite"], r["task_id"], r["seed"],
                 (r.get("exception", "")[:70])]
                for r in errs]
        lines += ["\n## Table 5 — Errors\n", "```",
                  fmt_table(["cfg", "suite", "task", "seed", "exception"], rows,
                            ["<", "<", ">", ">", "<"]),
                  "```\n"]

    # Verdict
    lines.append("\n## Verdict\n")
    if "static_schedule" in by_cfg and "reverse_step" in by_cfg:
        c2 = by_cfg["static_schedule"]
        c3 = by_cfg["reverse_step"]
        c2_long = [r for r in c2 if r["suite"] == "Long"]
        c3_long = [r for r in c3 if r["suite"] == "Long"]
        c2_k = sum(1 for r in c2_long if r["success"])
        c3_k = sum(1 for r in c3_long if r["success"])
        gap = c2_k - c3_k
        lines.append(f"- Long success: static_schedule = {c2_k}/{len(c2_long)}, "
                     f"reverse_step = {c3_k}/{len(c3_long)}, gap = {gap}")
        if gap >= 15:
            lines.append("- **Exp3 step-9 finding CONFIRMED at rollout level.** Schedule matters; order is non-symmetric.")
        elif gap >= 5:
            lines.append("- **Partial confirmation.** C2 > C3 but gap is smaller than the per-frame result predicted.")
        else:
            lines.append("- **Exp3 step-9 finding NOT confirmed at rollout level.** Per-frame asymmetry may not compound to rollout-level outcome differences.")

    if "fp16" in by_cfg and "static_schedule" in by_cfg:
        fp = [r for r in by_cfg["fp16"] if r["suite"] == "Long"]
        c2 = [r for r in by_cfg["static_schedule"] if r["suite"] == "Long"]
        fp_k = sum(1 for r in fp if r["success"])
        c2_k = sum(1 for r in c2 if r["success"])
        gap = fp_k - c2_k
        lines.append(f"- FP16 − static_schedule Long success gap = {gap}. "
                     f"(Small gap → schedule preserves quality; large gap → schedule still hurts.)")

    content = "\n".join(lines) + "\n"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(content)
    utils.log(f"[expA] tables → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_CONFIGS = ["fp16", "w4_uniform", "static_schedule", "reverse_step", "w2_protect"]


def main():
    utils.setup_logging()
    utils.log("=" * 70)
    utils.log("EXPERIMENT A: Rollout-level validation of static precision schedule")
    utils.log("=" * 70)

    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=ALL_CONFIGS,
                    choices=ALL_CONFIGS)
    ap.add_argument("--smoke", action="store_true",
                    help="1 rollout per config — validates install/uninstall + pre_infer_cb")
    args = ap.parse_args()

    out_path = os.path.join(utils.RESULTS_DIR, "expA_per_rollout.jsonl")
    tables_path = os.path.join(utils.RESULTS_DIR, "expA_tables.md")

    # Clear previous output if full run (smoke appends)
    if not args.smoke and os.path.exists(out_path):
        os.remove(out_path)

    # Pre-flight
    utils.log(f"[expA] configs: {args.configs}")
    utils.log(f"[expA] matrix: {TASKS_PER_SUITE} tasks × {len(SEEDS)} seeds × "
              f"{len(SUITES)} suites = {TASKS_PER_SUITE * len(SEEDS) * len(SUITES)} rollouts/config")
    utils.log(f"[expA] total: {TASKS_PER_SUITE * len(SEEDS) * len(SUITES) * len(args.configs)} rollouts")
    rollout.smoke_render()

    with utils.Timer("Model loading"):
        policy, model = utils.load_policy("pi05_libero")

    all_records = []
    t_global = time.time()

    for cfg_name in args.configs:
        utils.log(f"\n[expA] ========================================")
        utils.log(f"[expA]   CONFIG: {cfg_name}")
        utils.log(f"[expA] ========================================")
        vlm_saved, expert_handle = install_config(model, cfg_name)
        try:
            if args.smoke:
                pre_infer_cb = make_pre_infer_callback(cfg_name, expert_handle)
                rec = smoke_rollout(policy, cfg_name, pre_infer_cb)
                rec_d = {"quant_config": cfg_name, **rec.to_dict()}
                utils.append_jsonl(rec_d, out_path)
                all_records.append(rec_d)
            else:
                cfg_records = run_config_sweep(
                    policy, cfg_name, vlm_saved, expert_handle, out_path)
                all_records.extend(cfg_records)
        finally:
            uninstall_config(model, vlm_saved, expert_handle)

    utils.log(f"\n[expA] ALL CONFIGS DONE. Total wall: {(time.time()-t_global)/60:.1f} min")
    utils.log(f"[expA] {len(all_records)} rollouts total, "
              f"{sum(1 for r in all_records if r['success'])} succeeded")

    # Write tables
    if all_records:
        write_tables(all_records, tables_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
