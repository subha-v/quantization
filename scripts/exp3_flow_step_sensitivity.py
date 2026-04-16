#!/usr/bin/env python3
"""
Experiment 3 — Flow-Step Sensitivity  (redesigned, deterministic)  (~1-2 h on H100)

First-ever measurement of per-denoising-step quantization sensitivity for a VLA.
pi0.5 runs 10 Euler flow-matching steps (t=1.0 → ~0).

Why the previous version failed:
  pi0.5's denoise loop starts from random x_t ~ N(0, I).  policy.infer() samples
  fresh noise per call, so FP16 and W4 runs started from different x_t.  The
  noise-induced variance (~0.05 MSE) swamped the quantization-induced error
  (~0.005 MSE), giving indistinguishable bars.

Fix:
  1. Seed per-observation noise (torch.Generator, seed = 1000+sample_idx) and
     pass it through policy.infer(obs, noise=noise_np).  Both the reference and
     the test conditions start from identical x_t for a given obs.
  2. Monkey-patch model.denoise_step with a wrapper that tracks step index
     and swaps expert weights between steps.  denoise_step is the correct
     granularity: one call per Euler step; the prefix pass does NOT invoke
     the gemma_expert (inputs_embeds=[prefix_embs, None]), so the counter is
     clean.

Three sweep types:
  A) Per-step:     W4 at step k only, FP16 elsewhere          (10 configs)
  B) Cumulative-A: first k steps FP16, rest W4                (11 configs)
  C) Cumulative-B: first k steps W4, rest FP16                (11 configs)

Validation: with quantize_steps=∅ we must reproduce the FP16 reference exactly
(the sanity test aborts the run if that fails).
"""

import os
import sys
import time
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import utils

import torch


NUM_STEPS = 10       # pi0.5 default
NOISE_SEED_OFFSET = 1000
LIVE_EVERY = 32      # print a running summary every N observations within a config


# ===================================================================
# Locate the action expert
# ===================================================================

def find_action_expert(model):
    """Locate the gemma_expert sub-module (flow-matching action expert)."""
    # Direct attribute path in openpi PaliGemmaWithExpert
    candidates = []
    for name, mod in model.named_modules():
        if name.endswith("gemma_expert"):
            n_lin = sum(1 for _, m in mod.named_modules() if isinstance(m, torch.nn.Linear))
            if n_lin > 5:
                candidates.append((name, mod, n_lin))

    if not candidates:
        # Fallback: any "expert" submodule with many linears
        for name, mod in model.named_modules():
            cls = type(mod).__name__
            if "expert" in name.lower() or "Expert" in cls:
                n_lin = sum(1 for _, m in mod.named_modules() if isinstance(m, torch.nn.Linear))
                if n_lin > 5:
                    candidates.append((name, mod, n_lin))

    if not candidates:
        raise RuntimeError("Could not locate action expert.")

    # Prefer the shortest matching name (outermost expert module).
    candidates.sort(key=lambda c: len(c[0]))
    name, mod, n_lin = candidates[0]
    utils.log(f"[exp3] Action expert: {name}  ({type(mod).__name__}, {n_lin} linears)")
    return name, mod


# ===================================================================
# Seeded noise
# ===================================================================

def make_noise(action_horizon, action_dim, seed, device):
    """Deterministic noise tensor of shape (action_horizon, action_dim)."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return torch.normal(
        mean=0.0, std=1.0,
        size=(action_horizon, action_dim),
        generator=g, dtype=torch.float32, device=device,
    )


def get_action_shape(model):
    """Extract (action_horizon, action_dim) from the model config."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("Model has no .config")
    ah = getattr(cfg, "action_horizon", None)
    ad = getattr(cfg, "action_dim", None)
    if ah is None or ad is None:
        raise RuntimeError(f"Model config missing action_horizon/action_dim: {cfg}")
    return int(ah), int(ad)


def infer_with_noise(policy, obs, noise_np):
    """Run policy.infer(obs, noise=...) and return action ndarray."""
    with torch.no_grad():
        result = policy.infer(obs, noise=noise_np)
    if isinstance(result, dict):
        for key in ("actions", "action", "raw_actions"):
            if key in result:
                v = result[key]
                return v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    if isinstance(result, torch.Tensor):
        return result.cpu().numpy()
    return np.asarray(result)


# ===================================================================
# Per-step weight-swap controller — patches model.denoise_step
# ===================================================================

class StepController:
    """Monkey-patches model.denoise_step so the action expert can be swapped
    between FP16 and a pre-quantized weight set at each Euler step boundary."""

    def __init__(self, model, expert_module, orig_weights, quant_weights):
        self.model = model
        self.expert_module = expert_module
        self.orig_weights = orig_weights
        self.quant_weights = quant_weights
        self.quantize_steps = set()
        self.step = 0
        # Keep the bound original so we can restore it cleanly.
        self._original_denoise_step = model.denoise_step
        self._installed = False

    def set(self, quantize_steps):
        """Prepare for the next inference call: reset counter, set schedule."""
        self.quantize_steps = set(quantize_steps)
        self.step = 0

    def install(self):
        if self._installed:
            return

        controller = self
        original = self._original_denoise_step

        def patched(state, prefix_pad_masks, past_key_values, x_t, timestep):
            if controller.step in controller.quantize_steps:
                utils.swap_weights(controller.expert_module, controller.quant_weights)
            else:
                utils.swap_weights(controller.expert_module, controller.orig_weights)
            try:
                return original(state, prefix_pad_masks, past_key_values, x_t, timestep)
            finally:
                controller.step += 1

        # Setting an instance attribute overrides the class method via normal
        # attribute lookup; sample_actions uses self.denoise_step(...) which
        # will resolve to our patched function.
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


# ===================================================================
# Validation
# ===================================================================

def _validate_controller(policy, controller, observations, noises, n_check=3):
    """Ensure that the patched denoise_step is transparent when quantize_steps=∅
    (must reproduce FP16 within floating-point noise) and that quantizing all
    10 steps produces a measurably different output."""
    utils.log("[exp3] Validating deterministic step control...")

    # Baseline: FP16, seeded.  This is our reference.
    ref = [infer_with_noise(policy, observations[i], noises[i]) for i in range(n_check)]

    # Install controller with empty quantize set → should be identical to FP16.
    controller.install()
    try:
        for i in range(n_check):
            controller.set(set())          # no steps quantized
            a = infer_with_noise(policy, observations[i], noises[i])
            mse_fp16 = utils.action_mse(a, ref[i])
            if mse_fp16 > 1e-10:
                utils.log(
                    f"  FAIL: controller with quantize_steps=∅ does not match FP16 "
                    f"(obs {i}, MSE={mse_fp16:.3e})"
                )
                return False

        # All W4.  Should produce a measurable difference.
        all_q = set(range(NUM_STEPS))
        total_q_mse = 0.0
        for i in range(n_check):
            controller.set(all_q)
            a = infer_with_noise(policy, observations[i], noises[i])
            total_q_mse += utils.action_mse(a, ref[i])
        mean_q_mse = total_q_mse / n_check
    finally:
        controller.uninstall()

    utils.log(f"  all-FP16 vs ref:        MSE ~ 0 (passed)")
    utils.log(f"  all-W4  vs ref (mean):  MSE = {mean_q_mse:.6e}")
    if mean_q_mse < 1e-8:
        utils.log("  FAIL: W4 produces no distinguishable output — quantization ineffective.")
        return False
    return True


# ===================================================================
# Live logging helpers
# ===================================================================

def _suite_of(meta):
    s = meta.get("suite", "")
    if s in ("Long", "hard"):
        return "hard"
    if s in ("Object", "easy"):
        return "easy"
    return "other"


def _summarize(mses, metas):
    """Return a compact one-line summary of per-sample MSEs, split by suite."""
    if not mses:
        return "no samples"
    arr = np.asarray(mses, dtype=np.float64)
    easy = [m for m, md in zip(mses, metas) if _suite_of(md) == "easy"]
    hard = [m for m, md in zip(mses, metas) if _suite_of(md) == "hard"]
    parts = [f"mean={arr.mean():.4e} std={arr.std():.3e} "
             f"min={arr.min():.3e} max={arr.max():.3e}"]
    if easy and hard:
        delta = float(np.mean(hard)) - float(np.mean(easy))
        parts.append(
            f"easy={float(np.mean(easy)):.4e} (n={len(easy)}) "
            f"hard={float(np.mean(hard)):.4e} (n={len(hard)}) "
            f"Δ(h-e)={delta:+.3e}"
        )
    return "  ".join(parts)


def _progress(done, total, start):
    pct = 100.0 * done / max(total, 1)
    elapsed = time.time() - start
    rate = done / max(elapsed, 1e-6)
    eta = (total - done) / max(rate, 1e-6)
    return f"{done:3d}/{total} ({pct:5.1f}%)  elapsed={elapsed:5.1f}s  eta={eta:5.1f}s"


# ===================================================================
# Main
# ===================================================================

def main():
    utils.setup_logging()
    utils.log("=" * 60)
    utils.log("EXPERIMENT 3: Flow-Step Sensitivity (deterministic, redesigned)")
    utils.log("=" * 60)

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
            utils.log("FATAL: no model")
            return 1

    device = next(model.parameters()).device
    utils.log(f"[exp3] Device: {device}")

    # ---- data ----
    with utils.Timer("Data loading"):
        observations, metadata = utils.load_libero_observations(
            n_easy=128, n_hard=128, seed=42, suite_map=suite_map,
        )
    n_obs = len(observations)

    # ---- seeded noise ----
    action_horizon, action_dim = get_action_shape(model)
    utils.log(f"[exp3] Action shape: ({action_horizon}, {action_dim})")
    utils.log(f"[exp3] Pre-generating {n_obs} seeded noise tensors...")
    noises_np = []
    for i in range(n_obs):
        n = make_noise(action_horizon, action_dim, NOISE_SEED_OFFSET + i, device)
        noises_np.append(n.cpu().numpy())
    utils.log(f"  seed range: [{NOISE_SEED_OFFSET}, {NOISE_SEED_OFFSET + n_obs - 1}]")

    # ---- FP16 reference with SEEDED noise (cannot reuse exp2's cache) ----
    ref_path = os.path.join(utils.RESULTS_DIR, "exp3_reference_actions_seeded.npz")
    if os.path.exists(ref_path):
        utils.log(f"[exp3] Loading cached seeded reference: {ref_path}")
        ref_data = np.load(ref_path, allow_pickle=True)
        ref_actions = [ref_data[f"action_{i}"] for i in range(n_obs)]
    else:
        utils.log(f"[exp3] Computing FP16 reference with SEEDED noise ({n_obs} obs)...")
        ref_actions = []
        t0 = time.time()
        for i in range(n_obs):
            ref_actions.append(infer_with_noise(policy, observations[i], noises_np[i]))
            if i % 50 == 0:
                utils.log(f"  ref {i}/{n_obs}")
        utils.log(f"  Reference took {time.time() - t0:.1f}s")
        arrays = {f"action_{i}": a for i, a in enumerate(ref_actions)}
        utils.save_npz(ref_path, **arrays)

    # ---- find expert + pre-compute W4 weights ----
    expert_name, expert_module = find_action_expert(model)
    utils.log("[exp3] Pre-computing W4 weights for action expert...")
    orig_ptrs, quant_tensors = utils.precompute_quantized_weights(expert_module, bits=4)
    utils.log(f"  {len(orig_ptrs)} linear layers in expert")

    controller = StepController(model, expert_module, orig_ptrs, quant_tensors)

    # ---- validate ----
    if not _validate_controller(policy, controller, observations, noises_np, n_check=3):
        utils.log("FATAL: step controller validation failed.  Not running sweeps.")
        return 1

    # ---- smoke test: time one config ----
    utils.log(f"\n[exp3] Smoke test: timing 1 config x {n_obs} observations...")
    controller.install()
    try:
        t0 = time.time()
        for si in range(n_obs):
            controller.set({0})
            infer_with_noise(policy, observations[si], noises_np[si])
        smoke_time = time.time() - t0
    finally:
        controller.uninstall()
    n_configs = 10 + 11 + 11
    est = smoke_time * n_configs
    utils.log(f"  1 config: {smoke_time:.1f}s  |  estimated total ({n_configs} configs): {est/3600:.2f}h")

    # ---- sweeps ----
    per_step_path = os.path.join(utils.RESULTS_DIR, "exp3_per_step.jsonl")
    cumul_path = os.path.join(utils.RESULTS_DIR, "exp3_cumulative.jsonl")
    for p in (per_step_path, cumul_path):
        if os.path.exists(p):
            os.remove(p)

    controller.install()
    try:
        # A) Per-step: W4 at step k only
        utils.log(f"\n[exp3] Sweep A: per-step probing ({NUM_STEPS} configs)...")
        a_means = []  # per-step means for running summary
        for k in range(NUM_STEPS):
            q_steps = {k}
            t0 = time.time()
            mses_k, metas_k = [], []
            for si in range(n_obs):
                controller.set(q_steps)
                a = infer_with_noise(policy, observations[si], noises_np[si])
                mse = utils.action_mse(a, ref_actions[si])
                utils.append_jsonl({
                    "sweep": "per_step", "quant_step": k,
                    "sample_idx": si, "mse": mse, **metadata[si],
                }, per_step_path)
                mses_k.append(mse)
                metas_k.append(metadata[si])
                if (si + 1) % LIVE_EVERY == 0 or si == n_obs - 1:
                    utils.log(
                        f"    step={k}  {_progress(si + 1, n_obs, t0)}  "
                        f"running-mean={float(np.mean(mses_k)):.4e}"
                    )
            dt = time.time() - t0
            m = float(np.mean(mses_k))
            a_means.append(m)
            utils.log(f"  [A] step {k:2d}: {dt:.1f}s  |  {_summarize(mses_k, metas_k)}")
            utils.log(f"       running per-step means so far: "
                      + ", ".join(f"k={i}:{v:.3e}" for i, v in enumerate(a_means)))

        # B) first k steps FP16, rest W4
        utils.log(f"\n[exp3] Sweep B: first-k-FP16 ({NUM_STEPS + 1} configs)...")
        b_means = []
        for k in range(NUM_STEPS + 1):
            q_steps = set(range(k, NUM_STEPS))
            t0 = time.time()
            mses_k, metas_k = [], []
            for si in range(n_obs):
                controller.set(q_steps)
                a = infer_with_noise(policy, observations[si], noises_np[si])
                mse = utils.action_mse(a, ref_actions[si])
                utils.append_jsonl({
                    "sweep": "first_k_fp16", "k": k,
                    "sample_idx": si, "mse": mse, **metadata[si],
                }, cumul_path)
                mses_k.append(mse)
                metas_k.append(metadata[si])
                if (si + 1) % LIVE_EVERY == 0 or si == n_obs - 1:
                    utils.log(
                        f"    B k={k}  {_progress(si + 1, n_obs, t0)}  "
                        f"running-mean={float(np.mean(mses_k)):.4e}"
                    )
            label = "all-FP16" if k == NUM_STEPS else f"FP16[0:{k}]+W4[{k}:{NUM_STEPS}]"
            m = float(np.mean(mses_k))
            b_means.append(m)
            utils.log(f"  [B] k={k:2d} ({label}): {time.time()-t0:.1f}s  |  {_summarize(mses_k, metas_k)}")
            utils.log(f"       B curve so far: " + ", ".join(f"k={i}:{v:.3e}" for i, v in enumerate(b_means)))

        # C) first k steps W4, rest FP16
        utils.log(f"\n[exp3] Sweep C: first-k-W4 ({NUM_STEPS + 1} configs)...")
        c_means = []
        for k in range(NUM_STEPS + 1):
            q_steps = set(range(0, k))
            t0 = time.time()
            mses_k, metas_k = [], []
            for si in range(n_obs):
                controller.set(q_steps)
                a = infer_with_noise(policy, observations[si], noises_np[si])
                mse = utils.action_mse(a, ref_actions[si])
                utils.append_jsonl({
                    "sweep": "first_k_w4", "k": k,
                    "sample_idx": si, "mse": mse, **metadata[si],
                }, cumul_path)
                mses_k.append(mse)
                metas_k.append(metadata[si])
                if (si + 1) % LIVE_EVERY == 0 or si == n_obs - 1:
                    utils.log(
                        f"    C k={k}  {_progress(si + 1, n_obs, t0)}  "
                        f"running-mean={float(np.mean(mses_k)):.4e}"
                    )
            label = "all-FP16" if k == 0 else f"W4[0:{k}]+FP16[{k}:{NUM_STEPS}]"
            m = float(np.mean(mses_k))
            c_means.append(m)
            utils.log(f"  [C] k={k:2d} ({label}): {time.time()-t0:.1f}s  |  {_summarize(mses_k, metas_k)}")
            utils.log(f"       C curve so far: " + ", ".join(f"k={i}:{v:.3e}" for i, v in enumerate(c_means)))

        # ---- final end-of-run summary, tabular ----
        utils.log("\n" + "=" * 60)
        utils.log("EXP3 FINAL SUMMARY (mean action MSE vs FP16 reference)")
        utils.log("=" * 60)
        utils.log("[A] Per-step (W4 at step k only, FP16 elsewhere):")
        for i, v in enumerate(a_means):
            utils.log(f"    step {i:2d}:  {v:.6e}")
        utils.log("\n[B] First k steps FP16, rest W4:")
        for i, v in enumerate(b_means):
            utils.log(f"    k={i:2d}:    {v:.6e}")
        utils.log("\n[C] First k steps W4, rest FP16:")
        for i, v in enumerate(c_means):
            utils.log(f"    k={i:2d}:    {v:.6e}")
        utils.log("=" * 60)
    finally:
        controller.uninstall()

    # ---- plots ----
    utils.log("\n[exp3] Generating plots...")
    _plot(per_step_path, cumul_path, NUM_STEPS)
    utils.log("\nExperiment 3 complete.")
    return 0


# ===================================================================
# Plots
# ===================================================================

def _plot(per_step_path, cumul_path, num_steps):
    from collections import defaultdict
    plt = utils.setup_plotting()

    # ---- per-step sensitivity ----
    if os.path.exists(per_step_path):
        recs = [r for r in utils.load_jsonl(per_step_path) if r.get("sweep") == "per_step"]
        if recs:
            by_step_all = defaultdict(list)
            by_step_easy = defaultdict(list)
            by_step_hard = defaultdict(list)
            for r in recs:
                s = r["quant_step"]
                by_step_all[s].append(r["mse"])
                suite = r.get("suite", "")
                if suite in ("Object", "easy"):
                    by_step_easy[s].append(r["mse"])
                elif suite in ("Long", "hard"):
                    by_step_hard[s].append(r["mse"])

            steps = sorted(by_step_all.keys())
            means = [float(np.mean(by_step_all[s])) for s in steps]
            stds = [float(np.std(by_step_all[s])) for s in steps]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(steps, means, yerr=stds, capsize=4, color="#1565c0", alpha=0.85)
            ax.set_xlabel("Denoising step k (quantized to W4, all others FP16)")
            ax.set_ylabel("Action MSE vs FP16 reference")
            ax.set_title(f"Per-Step Sensitivity of Action Expert (seeded noise, n={len(recs)//num_steps})")
            ax.set_xticks(steps)
            plt.tight_layout()
            plt.savefig(os.path.join(utils.PLOTS_DIR, "exp3_per_step_sensitivity.png"))
            plt.close()
            utils.log("  exp3_per_step_sensitivity.png")

            # easy vs hard split
            if by_step_easy and by_step_hard:
                means_e = [float(np.mean(by_step_easy[s])) if by_step_easy[s] else 0 for s in steps]
                means_h = [float(np.mean(by_step_hard[s])) if by_step_hard[s] else 0 for s in steps]
                fig, ax = plt.subplots(figsize=(10, 5))
                width = 0.4
                x = np.array(steps)
                ax.bar(x - width/2, means_e, width, label="Easy (Object)", color="#1565c0", alpha=0.85)
                ax.bar(x + width/2, means_h, width, label="Hard (Long)", color="#d32f2f", alpha=0.85)
                ax.set_xlabel("Denoising step k (quantized to W4)")
                ax.set_ylabel("Action MSE vs FP16 reference")
                ax.set_title("Per-Step Sensitivity: Easy vs Hard")
                ax.set_xticks(steps)
                ax.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(utils.PLOTS_DIR, "exp3_per_step_easy_vs_hard.png"))
                plt.close()
                utils.log("  exp3_per_step_easy_vs_hard.png")

    # ---- cumulative sweeps ----
    if os.path.exists(cumul_path):
        recs = utils.load_jsonl(cumul_path)
        for sweep_name, label, xlabel, color in [
            ("first_k_fp16", "First k steps FP16, rest W4", "k (FP16 prefix length)", "#1565c0"),
            ("first_k_w4",   "First k steps W4, rest FP16", "k (W4 prefix length)",   "#d32f2f"),
        ]:
            sub = [r for r in recs if r.get("sweep") == sweep_name]
            if not sub:
                continue
            by_k = defaultdict(list)
            for r in sub:
                by_k[r["k"]].append(r["mse"])
            ks = sorted(by_k.keys())
            means = [float(np.mean(by_k[k])) for k in ks]
            stds = [float(np.std(by_k[k])) for k in ks]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.errorbar(ks, means, yerr=stds, fmt="o-", color=color, linewidth=2, capsize=3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Action MSE vs FP16 reference")
            ax.set_title(label)
            ax.set_xticks(ks)
            plt.tight_layout()
            plt.savefig(os.path.join(utils.PLOTS_DIR, f"exp3_cumulative_{sweep_name}.png"))
            plt.close()
            utils.log(f"  exp3_cumulative_{sweep_name}.png")

        # combined
        fig, ax = plt.subplots(figsize=(10, 5))
        for sweep_name, color, label in [
            ("first_k_fp16", "#1565c0", "First k FP16, rest W4"),
            ("first_k_w4",   "#d32f2f", "First k W4, rest FP16"),
        ]:
            sub = [r for r in recs if r.get("sweep") == sweep_name]
            if not sub:
                continue
            by_k = defaultdict(list)
            for r in sub:
                by_k[r["k"]].append(r["mse"])
            ks = sorted(by_k.keys())
            means = [float(np.mean(by_k[k])) for k in ks]
            ax.plot(ks, means, "o-", color=color, linewidth=2, label=label)
        ax.set_xlabel("k (switchover step)")
        ax.set_ylabel("Action MSE vs FP16 reference")
        ax.set_title("Cumulative Precision Scheduling")
        ax.legend()
        ax.set_xticks(range(num_steps + 1))
        plt.tight_layout()
        plt.savefig(os.path.join(utils.PLOTS_DIR, "exp3_cumulative_sweep.png"))
        plt.close()
        utils.log("  exp3_cumulative_sweep.png")


if __name__ == "__main__":
    sys.exit(main())
