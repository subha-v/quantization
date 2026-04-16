#!/usr/bin/env python3
"""
Generate plots from already-collected experiment data.
Run standalone — doesn't need the model or GPU.

Usage:
    cd /data/subha2/openpi
    uv run python /data/subha2/experiments/generate_plots.py
"""

import sys
import os
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Redirect matplotlib config off NFS home
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("WORKSPACE", "/data/subha2") + "/.matplotlib")

RESULTS_DIR = os.environ.get("EXPERIMENT_DIR", os.path.join(os.environ.get("WORKSPACE", "/data/subha2"), "experiments")) + "/results"
PLOTS_DIR = os.environ.get("EXPERIMENT_DIR", os.path.join(os.environ.get("WORKSPACE", "/data/subha2"), "experiments")) + "/plots"
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 150,
})


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


# ===================================================================
# Exp 1 plots
# ===================================================================

def plot_exp1():
    jsonl_path = os.path.join(RESULTS_DIR, "exp1_activation_stats.jsonl")
    summary_path = os.path.join(RESULTS_DIR, "exp1_summary.json")

    if not os.path.exists(summary_path):
        if not os.path.exists(jsonl_path):
            print("[exp1] No data found, skipping")
            return
        # Rebuild summary from JSONL
        print("[exp1] Rebuilding summary from JSONL...")
        records = load_jsonl(jsonl_path)
        summary = _aggregate_exp1(records)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    else:
        with open(summary_path) as f:
            summary = json.load(f)

    # Remove metadata keys
    summary = {k: v for k, v in summary.items() if not k.startswith("_")}

    if not summary:
        print("[exp1] Empty summary, skipping")
        return

    layers = sorted(summary.keys())
    short = [".".join(n.split(".")[-3:]) for n in layers]

    for metric in ["kurtosis", "outlier_6s", "max_abs", "std"]:
        if metric not in summary[layers[0]]:
            continue

        easy = [summary[n][metric]["easy_mean"] for n in layers]
        hard = [summary[n][metric]["hard_mean"] for n in layers]
        delta = [summary[n][metric]["delta"] for n in layers]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(16, len(layers) * 0.12), 10))
        x = np.arange(len(layers))

        ax1.plot(x, easy, "b-o", ms=1.5, label="Easy (Object)", alpha=0.7)
        ax1.plot(x, hard, "r-o", ms=1.5, label="Hard (Long)", alpha=0.7)
        ax1.set_ylabel(metric)
        ax1.set_title(f"Per-Layer {metric}: Easy (Object) vs Hard (Long)")
        ax1.legend()

        colors = ["#d32f2f" if d > 0 else "#1565c0" for d in delta]
        ax2.bar(x, delta, color=colors, alpha=0.7, width=0.8)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_ylabel(f"Delta: Hard - Easy")
        ax2.set_xlabel("Layer")

        step = max(1, len(layers) // 40)
        ax2.set_xticks(x[::step])
        ax2.set_xticklabels([short[i] for i in range(0, len(short), step)], rotation=90, fontsize=5)

        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"exp1_{metric}_comparison.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  [exp1] {path}")


def _aggregate_exp1(records):
    by_layer = defaultdict(lambda: {"easy": [], "hard": []})
    metrics = ["max_abs", "mean_abs", "l2_norm", "std", "kurtosis", "outlier_6s"]

    for r in records:
        suite = r.get("suite", "unknown")
        bucket = "easy" if suite in ("easy", "spatial", "Object", "Spatial") else "hard" if suite in ("hard", "long", "Long") else None
        if bucket is None:
            continue
        by_layer[r["layer"]][bucket].append({k: r[k] for k in metrics if k in r})

    result = {}
    for layer, buckets in by_layer.items():
        lr = {}
        for metric in metrics:
            e_vals = [s[metric] for s in buckets["easy"] if metric in s]
            h_vals = [s[metric] for s in buckets["hard"] if metric in s]
            lr[metric] = {
                "easy_mean": float(np.mean(e_vals)) if e_vals else 0,
                "hard_mean": float(np.mean(h_vals)) if h_vals else 0,
                "delta": float(np.mean(h_vals) - np.mean(e_vals)) if e_vals and h_vals else 0,
            }
        result[layer] = lr
    return result


# ===================================================================
# Exp 2 plots
# ===================================================================

def plot_exp2():
    for bits in [4, 8, 2]:
        jsonl_path = os.path.join(RESULTS_DIR, f"exp2_sensitivity_w{bits}.jsonl")
        if not os.path.exists(jsonl_path):
            continue

        print(f"  [exp2] Processing W{bits}...")
        records = load_jsonl(jsonl_path)
        if not records:
            continue

        by_group = defaultdict(lambda: {"all": [], "easy": [], "hard": []})
        group_types = {}

        for r in records:
            name = r["layer_group"]
            mse = r["mse"]
            by_group[name]["all"].append(mse)
            group_types[name] = r.get("group_type", "")
            suite = r.get("suite", "unknown")
            if suite in ("easy", "spatial", "Object", "Spatial"):
                by_group[name]["easy"].append(mse)
            elif suite in ("hard", "long", "Long"):
                by_group[name]["hard"].append(mse)

        names = sorted(by_group.keys())
        short = [".".join(n.split(".")[-3:]) for n in names]
        x = np.arange(len(names))

        # Sensitivity spectrum
        fig, ax = plt.subplots(figsize=(max(16, len(names) * 0.35), 6))
        vals = [np.mean(by_group[n]["all"]) for n in names]
        colors = []
        for n in names:
            gt = group_types.get(n, "")
            if "expert" in gt:
                colors.append("#d32f2f")
            elif "vlm" in gt:
                colors.append("#1565c0")
            else:
                colors.append("#388e3c")

        ax.bar(x, vals, color=colors, alpha=0.8)
        ax.set_ylabel("Action MSE vs FP16")
        ax.set_title(f"Layer-wise Sensitivity at W{bits}")
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=90, fontsize=7)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#d32f2f", label="Action Expert"),
            Patch(color="#1565c0", label="VLM"),
            Patch(color="#388e3c", label="Other"),
        ])
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"exp2_sensitivity_w{bits}.png"), bbox_inches="tight")
        plt.close()
        print(f"  [exp2] exp2_sensitivity_w{bits}.png")

        # Delta: easy vs hard
        fig, ax = plt.subplots(figsize=(max(16, len(names) * 0.35), 5))
        deltas = []
        for n in names:
            e = by_group[n]["easy"]
            h = by_group[n]["hard"]
            d = (np.mean(h) - np.mean(e)) if e and h else 0
            deltas.append(d)
        dcolors = ["#d32f2f" if d > 0 else "#1565c0" for d in deltas]
        ax.bar(x, deltas, color=dcolors, alpha=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Delta MSE: Hard - Easy")
        ax.set_title(f"Horizon-Differential Sensitivity at W{bits}")
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"exp2_easy_vs_hard_delta_w{bits}.png"), bbox_inches="tight")
        plt.close()
        print(f"  [exp2] exp2_easy_vs_hard_delta_w{bits}.png")

    # Multi-precision comparison (if multiple bitwidths available)
    all_data = {}
    for bits in [2, 4, 8]:
        jsonl_path = os.path.join(RESULTS_DIR, f"exp2_sensitivity_w{bits}.jsonl")
        if os.path.exists(jsonl_path):
            records = load_jsonl(jsonl_path)
            by_group = defaultdict(list)
            for r in records:
                by_group[r["layer_group"]].append(r["mse"])
            all_data[bits] = {n: np.mean(v) for n, v in by_group.items()}

    if len(all_data) > 1:
        names = sorted(set().union(*[d.keys() for d in all_data.values()]))
        short = [".".join(n.split(".")[-3:]) for n in names]
        x = np.arange(len(names))

        fig, ax = plt.subplots(figsize=(max(16, len(names) * 0.35), 6))
        width = 0.8 / len(all_data)
        colors = {2: "#d32f2f", 4: "#ff9800", 8: "#1565c0"}
        for i, (bits, data) in enumerate(sorted(all_data.items())):
            vals = [data.get(n, 0) for n in names]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=f"W{bits}", color=colors.get(bits, "#888"), alpha=0.8)

        ax.set_ylabel("Action MSE vs FP16")
        ax.set_title("Layer-wise Sensitivity: Multi-Precision Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=90, fontsize=6)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "exp2_multi_precision.png"), bbox_inches="tight")
        plt.close()
        print(f"  [exp2] exp2_multi_precision.png")


# ===================================================================
# Exp 3 plots
# ===================================================================

def plot_exp3():
    per_step_path = os.path.join(RESULTS_DIR, "exp3_per_step.jsonl")
    if not os.path.exists(per_step_path):
        print("[exp3] No data found, skipping")
        return

    records = load_jsonl(per_step_path)
    if not records:
        return

    # Check if we have per-step data or only brute-force
    sweep_types = set(r.get("sweep") for r in records)
    print(f"  [exp3] Sweep types: {sweep_types}")

    if "brute_force" in sweep_types:
        bf_records = [r for r in records if r.get("sweep") == "brute_force"]
        by_config = defaultdict(list)
        for r in bf_records:
            by_config[r["config"]].append(r["mse"])

        fig, ax = plt.subplots(figsize=(8, 5))
        configs = sorted(by_config.keys())
        means = [np.mean(by_config[c]) for c in configs]
        stds = [np.std(by_config[c]) for c in configs]
        ax.bar(configs, means, yerr=stds, capsize=5, color=["#1565c0", "#d32f2f"], alpha=0.8)
        ax.set_ylabel("Action MSE vs FP16 Reference")
        ax.set_title("Exp 3: All-FP16 vs All-W4 (brute force — per-step hooks failed)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "exp3_brute_force.png"), bbox_inches="tight")
        plt.close()
        print(f"  [exp3] exp3_brute_force.png")

    if "per_step" in sweep_types:
        by_step = defaultdict(list)
        for r in records:
            if r.get("sweep") == "per_step":
                by_step[r["quant_step"]].append(r["mse"])

        steps = sorted(by_step.keys())
        means = [np.mean(by_step[s]) for s in steps]
        stds = [np.std(by_step[s]) for s in steps]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(steps, means, yerr=stds, capsize=4, color="#1565c0", alpha=0.8)
        ax.set_xlabel("Denoising Step (quantized to W4)")
        ax.set_ylabel("Action MSE vs FP16 Reference")
        ax.set_title("Per-Step Quantization Sensitivity")
        ax.set_xticks(steps)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "exp3_per_step_sensitivity.png"), bbox_inches="tight")
        plt.close()
        print(f"  [exp3] exp3_per_step_sensitivity.png")


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("Generating plots from collected data")
    print(f"Results: {RESULTS_DIR}")
    print(f"Plots:   {PLOTS_DIR}")
    print("=" * 60)

    print("\n--- Exp 1: Activation Statistics ---")
    try:
        plot_exp1()
    except Exception as e:
        print(f"  [exp1] FAILED: {e}")

    print("\n--- Exp 2: Layer Sensitivity ---")
    try:
        plot_exp2()
    except Exception as e:
        print(f"  [exp2] FAILED: {e}")

    print("\n--- Exp 3: Flow-Step Sensitivity ---")
    try:
        plot_exp3()
    except Exception as e:
        print(f"  [exp3] FAILED: {e}")

    print(f"\nDone. Plots saved to {PLOTS_DIR}/")
    print("To view: scp the plots/ directory to your laptop")


if __name__ == "__main__":
    main()
