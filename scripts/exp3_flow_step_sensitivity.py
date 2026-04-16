#!/usr/bin/env python3
"""
Experiment 3 — Flow-Step Sensitivity  (~1-2 hours on H100)

First-ever measurement of per-denoising-step quantization sensitivity for a VLA.
pi0.5 runs 10 Euler flow-matching steps (t=1.0 → ~0).  We measure which steps
need full precision.

Three sweep types:
  A) Per-step:     W4 at step k only, FP16 elsewhere          (10 configs)
  B) Cumulative-A: first k steps FP16, rest W4                (11 configs)
  C) Cumulative-B: first k steps W4, rest FP16                (11 configs)

Implementation:
  We pre-compute W4 weights for the action expert once.
  A custom wrapper around the model's denoising loop pointer-swaps
  between FP16 and W4 weights at each step boundary.  Swap is O(1).
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


# ===================================================================
# Find the action expert module and the denoising interface
# ===================================================================

def find_action_expert(model):
    """Locate the action expert (Gemma expert) sub-module."""
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if "expert" in name.lower() or "Expert" in cls:
            n_lin = sum(1 for _, m in mod.named_modules() if isinstance(m, torch.nn.Linear))
            if n_lin > 5:
                print(f"[exp3] Action expert: {name} ({cls}, {n_lin} linears)")
                return name, mod
    # Fallback: look for known openpi attribute names
    for attr in ["gemma_expert", "action_expert"]:
        for name, mod in model.named_modules():
            if name.endswith(attr):
                print(f"[exp3] Action expert (fallback): {name}")
                return name, mod
    raise RuntimeError("Could not find action expert in model. Dump model structure and check.")


def find_sample_actions_method(model):
    """Find the method that runs the denoising loop.

    Returns (method, method_name) or (None, None).
    """
    for attr in ["sample_actions", "_sample_actions", "generate_actions"]:
        if hasattr(model, attr):
            return getattr(model, attr), attr
    return None, None


# ===================================================================
# Custom inference with per-step weight control
# ===================================================================

def run_inference_with_step_control(policy, model, observation, expert_module,
                                     orig_weights, quant_weights,
                                     quantize_steps, num_steps=10):
    """Run policy inference, quantizing the action expert at specific denoising steps.

    This works by registering a pre-forward hook on the expert that swaps weights
    based on an external step counter, then calling the standard policy.infer().

    Args:
        quantize_steps: set of step indices where the expert should be W4.
                       Empty set = all FP16 (reference).  {0..9} = all W4.
    """
    # Track which denoising step we're on by counting expert forward calls.
    # The expert is called once per denoising step (plus possibly once for
    # the prefix pass).  We identify denoising steps by counting calls
    # after the first one (prefix).
    state = {"call_count": 0, "prefix_done": False}

    def _pre_hook(mod, inputs):
        # First call is the prefix pass (no denoising yet).
        # Subsequent calls correspond to denoising steps 0..N-1.
        if not state["prefix_done"]:
            # The prefix pass runs the VLM + expert jointly.
            # We want the prefix in FP16.
            state["prefix_done"] = True
            state["call_count"] = 0
            utils.swap_weights(expert_module, orig_weights)
            return

        step = state["call_count"]
        if step in quantize_steps:
            utils.swap_weights(expert_module, quant_weights)
        else:
            utils.swap_weights(expert_module, orig_weights)
        state["call_count"] += 1

    hook = expert_module.register_forward_pre_hook(_pre_hook)
    try:
        action = utils.run_inference(policy, observation)
    finally:
        hook.remove()
        # Always restore to FP16
        utils.swap_weights(expert_module, orig_weights)

    return action


# ===================================================================
# Alternative: step control via direct weight swap around infer()
# ===================================================================

def run_inference_brute_force(policy, observation, expert_module,
                               orig_weights, quant_weights, use_quant=False):
    """Simpler fallback: run entire inference with expert either FP16 or W4.

    Use this if the hook-based approach doesn't work (e.g., the expert module
    isn't called separately per step).
    """
    if use_quant:
        utils.swap_weights(expert_module, quant_weights)
    else:
        utils.swap_weights(expert_module, orig_weights)
    try:
        action = utils.run_inference(policy, observation)
    finally:
        utils.swap_weights(expert_module, orig_weights)
    return action


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 60)
    print("EXPERIMENT 3: Flow-Step Sensitivity")
    print("=" * 60)

    NUM_STEPS = 10  # pi0.5 default

    # ---- suite map ----
    suite_map = None
    suite_map_path = os.path.join(utils.RESULTS_DIR, "task_suite_map.json")
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

    # ---- find expert ----
    expert_name, expert_module = find_action_expert(model)

    # ---- pre-compute W4 weights ----
    print("[exp3] Pre-computing W4 weights for action expert...")
    orig_ptrs, quant_tensors = utils.precompute_quantized_weights(expert_module, bits=4)
    print(f"  {len(orig_ptrs)} linear layers")

    # ---- FP16 reference actions (reuse from exp2 if available) ----
    ref_path = os.path.join(utils.RESULTS_DIR, "exp2_reference_actions.npz")
    if os.path.exists(ref_path):
        print(f"[exp3] Loading cached reference actions")
        ref_data = np.load(ref_path, allow_pickle=True)
        ref_actions = [ref_data[f"action_{i}"] for i in range(n_obs)]
    else:
        print(f"[exp3] Computing FP16 reference actions...")
        with utils.Timer("Reference actions"):
            ref_actions = utils.compute_reference_actions(policy, observations)
        arrays = {f"action_{i}": a for i, a in enumerate(ref_actions)}
        utils.save_npz(ref_path, **arrays)

    # ---- detect whether hook-based step control works ----
    print("\n[exp3] Testing hook-based step control...")
    hook_works = _test_hook_approach(
        policy, model, observations[0], expert_module,
        orig_ptrs, quant_tensors, ref_actions[0], NUM_STEPS,
    )

    if hook_works:
        print("  Hook-based step control: OK")
        run_fn = lambda obs, q_steps: run_inference_with_step_control(
            policy, model, obs, expert_module, orig_ptrs, quant_tensors,
            q_steps, NUM_STEPS,
        )
    else:
        print("  Hook-based step control: FAILED — falling back to brute-force (all-or-nothing per step)")
        print("  NOTE: this means per-step probing is NOT possible; only all-FP16 vs all-W4.")
        run_fn = None

    # ---- smoke test: time one config ----
    print(f"\n[exp3] Smoke test: timing 1 config x {n_obs} observations...")
    t0 = time.time()
    for obs in observations:
        if run_fn:
            run_fn(obs, {0})  # W4 at step 0 only
        else:
            run_inference_brute_force(policy, obs, expert_module, orig_ptrs, quant_tensors, True)
    smoke_time = time.time() - t0
    n_configs = 10 + 11 + 11 if run_fn else 2
    est = smoke_time * n_configs
    print(f"  1 config: {smoke_time:.1f}s")
    print(f"  Estimated total ({n_configs} configs): {est/3600:.1f}h")

    # ---- per-step sweep ----
    per_step_path = os.path.join(utils.RESULTS_DIR, "exp3_per_step.jsonl")
    cumul_path = os.path.join(utils.RESULTS_DIR, "exp3_cumulative.jsonl")

    if run_fn:
        # Clear previous
        for p in [per_step_path, cumul_path]:
            if os.path.exists(p):
                os.remove(p)

        # A) Per-step: W4 at step k only
        print(f"\n[exp3] Sweep A: per-step probing (10 configs)...")
        for k in range(NUM_STEPS):
            t0 = time.time()
            for si in range(n_obs):
                action = run_fn(observations[si], {k})
                mse = utils.action_mse(action, ref_actions[si])
                utils.append_jsonl({
                    "sweep": "per_step", "quant_step": k,
                    "sample_idx": si, "mse": mse, **metadata[si],
                }, per_step_path)
            dt = time.time() - t0
            print(f"  step {k}: {dt:.1f}s")

        # B) Cumulative: first k steps FP16, rest W4
        print(f"\n[exp3] Sweep B: first-k-FP16 (11 configs)...")
        for k in range(NUM_STEPS + 1):
            quant_steps = set(range(k, NUM_STEPS))
            t0 = time.time()
            for si in range(n_obs):
                action = run_fn(observations[si], quant_steps)
                mse = utils.action_mse(action, ref_actions[si])
                utils.append_jsonl({
                    "sweep": "first_k_fp16", "k": k,
                    "sample_idx": si, "mse": mse, **metadata[si],
                }, cumul_path)
            dt = time.time() - t0
            label = "all-FP16" if k == NUM_STEPS else f"FP16[0:{k}]+W4[{k}:{NUM_STEPS}]"
            print(f"  k={k:2d} ({label}): {dt:.1f}s")

        # C) Cumulative: first k steps W4, rest FP16
        print(f"\n[exp3] Sweep C: first-k-W4 (11 configs)...")
        for k in range(NUM_STEPS + 1):
            quant_steps = set(range(0, k))
            t0 = time.time()
            for si in range(n_obs):
                action = run_fn(observations[si], quant_steps)
                mse = utils.action_mse(action, ref_actions[si])
                utils.append_jsonl({
                    "sweep": "first_k_w4", "k": k,
                    "sample_idx": si, "mse": mse, **metadata[si],
                }, cumul_path)
            dt = time.time() - t0
            label = "all-FP16" if k == 0 else f"W4[0:{k}]+FP16[{k}:{NUM_STEPS}]"
            print(f"  k={k:2d} ({label}): {dt:.1f}s")

    else:
        # Brute-force fallback: just all-FP16 vs all-W4
        print("\n[exp3] Brute-force: all-FP16 vs all-W4 only")
        if os.path.exists(per_step_path):
            os.remove(per_step_path)

        for use_q, label in [(False, "all_fp16"), (True, "all_w4")]:
            t0 = time.time()
            for si in range(n_obs):
                action = run_inference_brute_force(
                    policy, observations[si], expert_module,
                    orig_ptrs, quant_tensors, use_q,
                )
                mse = utils.action_mse(action, ref_actions[si])
                utils.append_jsonl({
                    "sweep": "brute_force", "config": label,
                    "sample_idx": si, "mse": mse, **metadata[si],
                }, per_step_path)
            dt = time.time() - t0
            print(f"  {label}: {dt:.1f}s")

    # ---- plots ----
    print("\n[exp3] Generating plots...")
    _plot(per_step_path, cumul_path, NUM_STEPS)

    print("\nExperiment 3 complete.")
    return 0


def _test_hook_approach(policy, model, obs, expert_module,
                         orig_ptrs, quant_tensors, ref_action, num_steps):
    """Test whether the hook-based step control produces different results
    for different step configurations (proving it actually works)."""
    try:
        # All FP16 — should match reference
        a_fp16 = run_inference_with_step_control(
            policy, model, obs, expert_module, orig_ptrs, quant_tensors,
            set(), num_steps,
        )
        mse_fp16 = utils.action_mse(a_fp16, ref_action)

        # All W4 — should differ from reference
        a_w4 = run_inference_with_step_control(
            policy, model, obs, expert_module, orig_ptrs, quant_tensors,
            set(range(num_steps)), num_steps,
        )
        mse_w4 = utils.action_mse(a_w4, ref_action)

        print(f"    all-FP16 vs ref MSE: {mse_fp16:.8f} (should be ~0)")
        print(f"    all-W4 vs ref MSE:   {mse_w4:.6f} (should be >0)")

        # FP16 should be close to reference; W4 should differ
        return mse_fp16 < 1e-6 and mse_w4 > 1e-8
    except Exception as e:
        print(f"    Hook test failed: {e}")
        return False


def _plot(per_step_path, cumul_path, num_steps):
    plt = utils.setup_plotting()

    # Per-step sensitivity
    if os.path.exists(per_step_path):
        recs = utils.load_jsonl(per_step_path)
        per_step_recs = [r for r in recs if r.get("sweep") == "per_step"]

        if per_step_recs:
            from collections import defaultdict
            by_step = defaultdict(list)
            for r in per_step_recs:
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
            plt.savefig(os.path.join(utils.PLOTS_DIR, "exp3_per_step_sensitivity.png"))
            plt.close()
            print(f"  exp3_per_step_sensitivity.png")

    # Cumulative sweep
    if os.path.exists(cumul_path):
        recs = utils.load_jsonl(cumul_path)

        from collections import defaultdict

        for sweep_name, label, xlabel in [
            ("first_k_fp16", "First k steps FP16, rest W4", "k (FP16 steps)"),
            ("first_k_w4", "First k steps W4, rest FP16", "k (W4 steps)"),
        ]:
            sweep_recs = [r for r in recs if r.get("sweep") == sweep_name]
            if not sweep_recs:
                continue
            by_k = defaultdict(list)
            for r in sweep_recs:
                by_k[r["k"]].append(r["mse"])

            ks = sorted(by_k.keys())
            means = [np.mean(by_k[k]) for k in ks]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ks, means, "o-", color="#d32f2f", linewidth=2)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Action MSE vs FP16 Reference")
            ax.set_title(label)
            ax.set_xticks(ks)
            plt.tight_layout()
            fname = f"exp3_cumulative_{sweep_name}.png"
            plt.savefig(os.path.join(utils.PLOTS_DIR, fname))
            plt.close()
            print(f"  {fname}")

        # Combined cumulative plot
        fig, ax = plt.subplots(figsize=(10, 5))
        for sweep_name, color, label in [
            ("first_k_fp16", "#1565c0", "First k FP16, rest W4"),
            ("first_k_w4", "#d32f2f", "First k W4, rest FP16"),
        ]:
            sweep_recs = [r for r in recs if r.get("sweep") == sweep_name]
            if not sweep_recs:
                continue
            by_k = defaultdict(list)
            for r in sweep_recs:
                by_k[r["k"]].append(r["mse"])
            ks = sorted(by_k.keys())
            means = [np.mean(by_k[k]) for k in ks]
            ax.plot(ks, means, "o-", color=color, linewidth=2, label=label)

        ax.set_xlabel("k (switchover step)")
        ax.set_ylabel("Action MSE vs FP16 Reference")
        ax.set_title("Cumulative Precision Scheduling")
        ax.legend()
        ax.set_xticks(range(num_steps + 1))
        plt.tight_layout()
        plt.savefig(os.path.join(utils.PLOTS_DIR, "exp3_cumulative_sweep.png"))
        plt.close()
        print(f"  exp3_cumulative_sweep.png")


if __name__ == "__main__":
    sys.exit(main())
