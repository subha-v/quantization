#!/usr/bin/env python3
"""
Experiment 2 — Layer-wise Sensitivity Probe  (~2-4 hours on H100)

Quantize each decoder-layer group individually to W4 (then W8, optionally W2),
measure action MSE vs FP16 reference.  Save per-sample results with metadata.

Includes:
  - Wall-time smoke test before committing to the full sweep
  - Incremental JSONL writes (survives crashes)
  - Per-sample metadata for post-hoc temporal analysis
"""

import sys
import os
import json
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import utils

import torch


def main():
    print("=" * 60)
    print("EXPERIMENT 2: Layer-wise Sensitivity Probe")
    print("=" * 60)

    # ---- suite map ----
    suite_map_path = os.path.join(utils.RESULTS_DIR, "task_suite_map.json")
    suite_map = None
    if os.path.exists(suite_map_path):
        with open(suite_map_path) as f:
            suite_map = {int(k): v for k, v in json.load(f).get("suite_map", {}).items()}

    # ---- model ----
    with utils.Timer("Model loading"):
        policy, model = utils.load_policy("pi05_libero")
        if model is None:
            print("FATAL: no model")
            return 1

    # ---- data ----
    with utils.Timer("Data loading"):
        observations, metadata = utils.load_libero_observations(
            n_easy=128, n_hard=128, seed=42, suite_map=suite_map,
        )
    n_obs = len(observations)

    # ---- layer groups ----
    groups = utils.get_layer_groups(model)
    print(f"\n{len(groups)} layer groups to sweep")

    # ---- FP16 reference actions ----
    ref_path = os.path.join(utils.RESULTS_DIR, "exp2_reference_actions.npz")
    if os.path.exists(ref_path):
        print(f"[exp2] Loading cached reference actions from {ref_path}")
        ref_data = np.load(ref_path, allow_pickle=True)
        ref_actions = [ref_data[f"action_{i}"] for i in range(n_obs)]
    else:
        print(f"[exp2] Computing FP16 reference actions for {n_obs} observations...")
        with utils.Timer("Reference actions"):
            ref_actions = utils.compute_reference_actions(policy, observations)
        arrays = {f"action_{i}": a for i, a in enumerate(ref_actions)}
        utils.save_npz(ref_path, **arrays)
        print(f"  Saved to {ref_path}")

    # ---- bit-widths to sweep ----
    bit_widths = [4, 8]  # W2 added only if smoke test shows we have time

    # ---- smoke test: time one group ----
    print(f"\n[exp2] Smoke test: timing 1 group x {n_obs} observations at W4...")
    g0 = groups[0]
    t0 = time.time()
    saved = utils.fake_quantize_module(g0["module"], bits=4)
    for obs in observations:
        utils.run_inference(policy, obs)
    utils.restore_weights(g0["module"], saved)
    smoke_time = time.time() - t0
    est_total = smoke_time * len(groups) * len(bit_widths)
    print(f"  1 group: {smoke_time:.1f}s")
    print(f"  Estimated total ({len(groups)} groups x {len(bit_widths)} bitwidths): {est_total/3600:.1f}h")

    if smoke_time > 240:  # >4 min per group
        print("  WARNING: >4 min/group. Dropping W2 and reducing to 128 observations.")
        bit_widths = [4, 8]
        # Reduce observations
        if n_obs > 128:
            observations = observations[:128]
            metadata = metadata[:128]
            ref_actions = ref_actions[:128]
            n_obs = 128
            est_total = smoke_time * 0.5 * len(groups) * len(bit_widths)
            print(f"  Revised estimate: {est_total/3600:.1f}h")
    elif est_total < 10800:  # < 3 hours with headroom
        print("  Time budget OK — adding W2 sweep.")
        bit_widths = [4, 8, 2]

    # ---- main sweep ----
    for bits in bit_widths:
        jsonl_path = os.path.join(utils.RESULTS_DIR, f"exp2_sensitivity_w{bits}.jsonl")
        # Check if partially completed
        done_groups = set()
        if os.path.exists(jsonl_path):
            for rec in utils.load_jsonl(jsonl_path):
                done_groups.add(rec.get("layer_group"))
            print(f"\n[exp2] W{bits}: resuming, {len(done_groups)} groups already done")
        else:
            print(f"\n[exp2] W{bits}: starting sweep over {len(groups)} groups")

        for gi, group in enumerate(groups):
            if group["name"] in done_groups:
                continue

            t0 = time.time()
            saved = utils.fake_quantize_module(group["module"], bits=bits)

            for si in range(n_obs):
                q_action = utils.run_inference(policy, observations[si])
                mse = utils.action_mse(q_action, ref_actions[si])
                entry = {
                    "layer_group": group["name"],
                    "group_type": group["group_type"],
                    "bits": bits,
                    "sample_idx": si,
                    "mse": mse,
                    **metadata[si],
                }
                utils.append_jsonl(entry, jsonl_path)

            utils.restore_weights(group["module"], saved)
            dt = time.time() - t0
            print(f"  [{gi+1}/{len(groups)}] W{bits} {group['name']} ({group['group_type']}): {dt:.1f}s")

    # ---- aggregate + plot ----
    print("\n[exp2] Generating summary and plots...")
    for bits in bit_widths:
        jsonl_path = os.path.join(utils.RESULTS_DIR, f"exp2_sensitivity_w{bits}.jsonl")
        if os.path.exists(jsonl_path):
            _summarize_and_plot(jsonl_path, bits)

    print("\nExperiment 2 complete.")
    return 0


def _summarize_and_plot(jsonl_path, bits):
    """Compute per-layer aggregates and generate plots from per-sample JSONL."""
    records = utils.load_jsonl(jsonl_path)
    if not records:
        return

    from collections import defaultdict
    by_group = defaultdict(lambda: {"all": [], "easy": [], "hard": []})

    for r in records:
        name = r["layer_group"]
        mse = r["mse"]
        by_group[name]["all"].append(mse)
        suite = r.get("suite", "unknown")
        if suite in ("easy", "spatial", "Object", "Spatial"):
            by_group[name]["easy"].append(mse)
        elif suite in ("hard", "long", "Long"):
            by_group[name]["hard"].append(mse)

    # Summary
    summary = {}
    for name, buckets in by_group.items():
        summary[name] = {
            "mean_mse": float(np.mean(buckets["all"])),
            "std_mse": float(np.std(buckets["all"])),
            "easy_mse": float(np.mean(buckets["easy"])) if buckets["easy"] else 0,
            "hard_mse": float(np.mean(buckets["hard"])) if buckets["hard"] else 0,
            "delta": (float(np.mean(buckets["hard"])) - float(np.mean(buckets["easy"])))
                     if buckets["easy"] and buckets["hard"] else 0,
        }
    utils.save_json(summary, os.path.join(utils.RESULTS_DIR, f"exp2_summary_w{bits}.json"))

    # Plots
    plt = utils.setup_plotting()
    names = sorted(summary.keys())
    short = [".".join(n.split(".")[-3:]) for n in names]
    x = np.arange(len(names))

    # ---- sensitivity spectrum ----
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.35), 6))
    vals = [summary[n]["mean_mse"] for n in names]
    colors = []
    for n in names:
        for r in records:
            if r["layer_group"] == n:
                gtype = r.get("group_type", "")
                break
        else:
            gtype = ""
        colors.append("#d32f2f" if "expert" in gtype else "#1565c0" if "vlm" in gtype else "#388e3c")

    ax.bar(x, vals, color=colors, alpha=0.8)
    ax.set_ylabel("Action MSE (FP16 ref)")
    ax.set_title(f"Layer-wise Sensitivity at W{bits}")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#d32f2f", label="Action Expert"),
        Patch(color="#1565c0", label="VLM"),
        Patch(color="#388e3c", label="Other"),
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(utils.PLOTS_DIR, f"exp2_sensitivity_w{bits}.png"))
    plt.close()

    # ---- delta: easy vs hard ----
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.35), 5))
    deltas = [summary[n]["delta"] for n in names]
    dcolors = ["#d32f2f" if d > 0 else "#1565c0" for d in deltas]
    ax.bar(x, deltas, color=dcolors, alpha=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Delta MSE: Hard − Easy")
    ax.set_title(f"Horizon-Differential Sensitivity at W{bits}")
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(utils.PLOTS_DIR, f"exp2_easy_vs_hard_delta_w{bits}.png"))
    plt.close()

    print(f"  Plots saved for W{bits}")


if __name__ == "__main__":
    sys.exit(main())
