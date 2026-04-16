#!/usr/bin/env python3
"""
Experiment 1 — Cross-Suite Activation Statistics  (~30 min on H100)

Tests whether LIBERO-Long (hard) observations produce systematically different
activation distributions than LIBERO-Spatial (easy) observations.

Saves per-sample stats with full metadata to JSONL for post-hoc temporal analysis.
"""

import sys, os, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import utils

import torch


def main():
    print("=" * 60)
    print("EXPERIMENT 1: Cross-Suite Activation Statistics")
    print("=" * 60)

    # ---- load suite map (from setup_and_verify) ----
    suite_map_path = os.path.join(utils.RESULTS_DIR, "task_suite_map.json")
    suite_map = None
    if os.path.exists(suite_map_path):
        with open(suite_map_path) as f:
            data = json.load(f)
        suite_map = {int(k): v for k, v in data.get("suite_map", {}).items()}
        print(f"[exp1] Loaded suite map: {len(suite_map)} tasks")

    # ---- model ----
    with utils.Timer("Model loading"):
        policy, model = utils.load_policy("pi05_libero")
        if model is None:
            print("FATAL: no model. Run setup_and_verify.py first.")
            return 1

    # ---- data ----
    with utils.Timer("Data loading"):
        observations, metadata = utils.load_libero_observations(
            n_easy=128, n_hard=128, seed=42, suite_map=suite_map,
        )

    # ---- smoke test: 2 observations through full pipeline ----
    def _smoke():
        h, s = utils.register_activation_hooks(model)
        for obs in observations[:2]:
            utils.run_inference(policy, obs)
        utils.remove_hooks(h)
        assert len(s) > 0, "No activation stats captured"
        utils.log(f"  Smoke: {len(s)} layers captured, {utils.gpu_mem_str()}")

    if not utils.run_smoke_test("Exp1 forward+hooks", _smoke):
        print("FATAL: smoke test failed")
        return 1

    # ---- hooks ----
    hooks, raw_stats = utils.register_activation_hooks(model)

    # ---- forward passes ----
    jsonl_path = os.path.join(utils.RESULTS_DIR, "exp1_activation_stats.jsonl")
    # Clear previous run
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    print(f"\n[exp1] Running {len(observations)} forward passes...")
    with utils.Timer("Forward passes"):
        for i, obs in enumerate(observations):
            # Clear accumulated stats before each observation
            for v in raw_stats.values():
                v.clear()

            with torch.no_grad():
                utils.run_inference(policy, obs)

            # Write per-layer stats for this observation
            meta = metadata[i]
            for layer_name, stat_list in raw_stats.items():
                if not stat_list:
                    continue
                # Average across fires (prefix + denoising steps)
                avg = {}
                for key in stat_list[0]:
                    avg[key] = float(np.mean([s[key] for s in stat_list]))
                entry = {
                    "layer": layer_name,
                    **meta,
                    **avg,
                }
                utils.append_jsonl(entry, jsonl_path)

            if i % 25 == 0:
                print(f"  {i}/{len(observations)} done")

    utils.remove_hooks(hooks)
    print(f"[exp1] Per-sample stats saved to {jsonl_path}")

    # ---- aggregate + summary ----
    print("[exp1] Aggregating...")
    records = utils.load_jsonl(jsonl_path)
    summary = _aggregate(records)
    utils.save_json(summary, os.path.join(utils.RESULTS_DIR, "exp1_summary.json"))

    # ---- print top deltas ----
    _print_top_deltas(summary)

    # ---- plots ----
    print("[exp1] Generating plots...")
    _plot(summary)

    print("\nExperiment 1 complete.")
    return 0


def _aggregate(records):
    """Aggregate per-sample records into per-layer easy/hard means."""
    from collections import defaultdict

    by_layer = defaultdict(lambda: {"easy": [], "hard": []})
    metrics = ["max_abs", "mean_abs", "l2_norm", "std", "kurtosis", "outlier_6s"]

    for r in records:
        suite = r.get("suite", "unknown")
        bucket = "easy" if suite in ("easy", "spatial") else "hard" if suite in ("hard", "long") else None
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
                "easy_std": float(np.std(e_vals)) if e_vals else 0,
                "hard_mean": float(np.mean(h_vals)) if h_vals else 0,
                "hard_std": float(np.std(h_vals)) if h_vals else 0,
                "delta": float(np.mean(h_vals) - np.mean(e_vals)) if e_vals and h_vals else 0,
            }
        result[layer] = lr
    return result


def _print_top_deltas(summary):
    print("\n--- Largest |Delta| (Hard − Easy) ---")
    for metric in ["kurtosis", "outlier_6s", "max_abs"]:
        deltas = [(n, summary[n][metric]["delta"]) for n in summary if metric in summary[n]]
        deltas.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Top-5 by |delta({metric})|:")
        for name, d in deltas[:5]:
            e = summary[name][metric]["easy_mean"]
            h = summary[name][metric]["hard_mean"]
            print(f"    {name}: easy={e:.4f}  hard={h:.4f}  delta={d:+.4f}")


def _plot(summary):
    plt = utils.setup_plotting()
    layers = sorted(summary.keys())
    if not layers:
        return

    short = [n.split(".")[-3:] for n in layers]
    short = [".".join(s) for s in short]

    for metric in ["kurtosis", "outlier_6s", "max_abs", "std"]:
        easy = [summary[n][metric]["easy_mean"] for n in layers]
        hard = [summary[n][metric]["hard_mean"] for n in layers]
        delta = [summary[n][metric]["delta"] for n in layers]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(14, len(layers) * 0.12), 10))
        x = np.arange(len(layers))

        ax1.plot(x, easy, "b-o", ms=1.5, label="Easy (Spatial)", alpha=0.7)
        ax1.plot(x, hard, "r-o", ms=1.5, label="Hard (Long)", alpha=0.7)
        ax1.set_ylabel(metric)
        ax1.set_title(f"Per-Layer {metric}: Easy vs Hard")
        ax1.legend()

        colors = ["#d32f2f" if d > 0 else "#1565c0" for d in delta]
        ax2.bar(x, delta, color=colors, alpha=0.7, width=0.8)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_ylabel(f"Delta: Hard − Easy")
        ax2.set_xlabel("Layer")

        step = max(1, len(layers) // 40)
        ax2.set_xticks(x[::step])
        ax2.set_xticklabels([short[i] for i in range(0, len(short), step)], rotation=90, fontsize=5)

        plt.tight_layout()
        path = os.path.join(utils.PLOTS_DIR, f"exp1_{metric}_comparison.png")
        plt.savefig(path)
        plt.close()
        print(f"  {path}")


if __name__ == "__main__":
    sys.exit(main())
