#!/usr/bin/env python3
"""
Phase 0 — GO / NO-GO gate.

Verifies: imports, model loading, data loading, forward-pass sanity
(actions not NaN/constant → confirms fine-tuned checkpoint), hooks,
quantization round-trip, and task→suite mapping.

Run from the openpi directory:
    cd $OPENPI_DIR && uv run python $EXPERIMENT_DIR/setup_and_verify.py
"""

import sys, os, json, time, traceback
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import utils  # our shared module

import torch


def main():
    report = {"status": "RUNNING", "checks": {}, "errors": []}
    ok = True

    def check(name, fn):
        nonlocal ok
        print(f"\n{'='*60}\nCHECK: {name}\n{'='*60}")
        t0 = time.time()
        try:
            detail = fn()
            dt = time.time() - t0
            report["checks"][name] = {"status": "PASS", "seconds": round(dt, 1), "detail": detail}
            print(f"  -> PASS ({dt:.1f}s)")
            return detail
        except Exception as e:
            dt = time.time() - t0
            report["checks"][name] = {"status": "FAIL", "seconds": round(dt, 1), "error": str(e)}
            report["errors"].append({"check": name, "error": str(e), "tb": traceback.format_exc()})
            print(f"  -> FAIL: {e}")
            traceback.print_exc()
            ok = False
            return None

    # ---------------------------------------------------------------- GPU
    def _gpu():
        assert torch.cuda.is_available(), "CUDA not available"
        n = torch.cuda.get_device_name(0)
        m = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  {n}, {m:.1f} GB")
        return {"gpu": n, "mem_gb": round(m, 1)}
    check("GPU", _gpu)

    # ---------------------------------------------------------------- imports
    def _imports():
        mods = {}
        for m in ["openpi.training.config", "openpi.policies.policy_config", "openpi.shared.download"]:
            __import__(m)
            mods[m] = "OK"
            print(f"  {m}: OK")
        return mods
    check("openpi imports", _imports)

    # ---------------------------------------------------------------- model
    policy = [None]
    model = [None]

    def _model():
        p, m = utils.load_policy("pi05_libero")
        policy[0] = p
        model[0] = m
        if m is None:
            raise RuntimeError("Could not extract nn.Module from policy")
        summary = utils.print_model_summary(m)
        groups = utils.get_layer_groups(m)
        return {
            "model_type": type(m).__name__,
            "params": summary["total_params"],
            "linears": summary["linear_count"],
            "groups": len(groups),
            "group_names": [g["name"] for g in groups],
        }
    info = check("Load pi0.5 LIBERO model", _model)

    # fallback to pi0
    if policy[0] is None:
        def _fallback():
            p, m = utils.load_policy("pi0_libero")
            policy[0] = p
            model[0] = m
            if m is None:
                raise RuntimeError("pi0 fallback also failed")
            return {"model_type": type(m).__name__, "fallback": True}
        check("Load pi0 LIBERO (fallback)", _fallback)

    # ---------------------------------------------------------------- memory
    def _mem():
        torch.cuda.synchronize()
        a = torch.cuda.memory_allocated() / 1e9
        t = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Allocated {a:.2f} GB / {t:.1f} GB")
        return {"allocated_gb": round(a, 2), "total_gb": round(t, 1)}
    check("GPU memory", _mem)

    # ---------------------------------------------------------------- data (small)
    observations = [None]
    metadata_list = [None]

    def _data():
        obs, meta = utils.load_libero_observations(n_easy=2, n_hard=2, seed=42)
        observations[0] = obs
        metadata_list[0] = meta
        print(f"  Loaded {len(obs)} observations")
        print(f"  Keys: {list(obs[0].keys())}")
        print(f"  State shape: {obs[0].get('observation/state', np.array([])).shape}")
        for k in ["observation/image", "observation/wrist_image"]:
            if k in obs[0]:
                print(f"  {k} shape: {np.array(obs[0][k]).shape}")
        print(f"  Metadata[0]: {meta[0]}")
        return {"n": len(obs), "keys": list(obs[0].keys()), "meta_sample": meta[0]}
    check("Load LIBERO data (4 samples)", _data)

    # ---------------------------------------------------------------- forward pass sanity
    def _forward():
        assert policy[0] is not None and len(observations[0]) > 0
        obs = observations[0]
        a1 = utils.run_inference(policy[0], obs[0])
        print(f"  Action shape: {a1.shape}, dtype: {a1.dtype}")
        print(f"  Range: [{a1.min():.4f}, {a1.max():.4f}]")
        print(f"  Mean: {a1.mean():.4f}, Std: {a1.std():.4f}")
        assert not np.isnan(a1).any(), "NaN in actions!"
        assert not np.isinf(a1).any(), "Inf in actions!"
        assert a1.std() > 1e-6, (
            f"Near-zero action variance ({a1.std():.8f}). "
            "This looks like an untrained base model, not the LIBERO fine-tuned checkpoint!"
        )
        # two-observation diff
        a2 = utils.run_inference(policy[0], obs[1])
        d = float(np.mean((a1 - a2) ** 2))
        print(f"  MSE(obs0, obs1): {d:.6f}")
        if d < 1e-8:
            print("  WARNING: identical actions for different inputs — model may not be processing observations")
        return {"shape": list(a1.shape), "std": float(a1.std()), "two_obs_mse": d}
    check("Forward pass + action sanity", _forward)

    # ---------------------------------------------------------------- hooks
    def _hooks():
        assert model[0] is not None and policy[0] is not None and len(observations[0]) > 0
        hooks, stats = utils.register_activation_hooks(model[0])
        utils.run_inference(policy[0], observations[0][0])
        n_triggered = sum(1 for v in stats.values() if v)
        print(f"  Triggered: {n_triggered}/{len(hooks)}")
        for name, slist in list(stats.items())[:3]:
            if slist:
                s = slist[0]
                print(f"  {name}: max_abs={s['max_abs']:.3f}, kurtosis={s['kurtosis']:.1f}")
        utils.remove_hooks(hooks)
        return {"registered": len(hooks), "triggered": n_triggered}
    check("Activation hooks", _hooks)

    # ---------------------------------------------------------------- quantization round-trip
    def _quant():
        assert model[0] is not None and policy[0] is not None and len(observations[0]) > 0
        groups = utils.get_layer_groups(model[0])
        g = groups[0]
        print(f"  Testing on: {g['name']} ({g['group_type']})")

        # Get a reference weight tensor to verify restore directly
        first_linear = None
        for _, m in g["module"].named_modules():
            if isinstance(m, torch.nn.Linear):
                first_linear = m
                break
        orig_weight_snap = first_linear.weight.data.clone()

        # Quantize
        saved = utils.fake_quantize_module(g["module"], bits=4)

        # Verify weights actually changed
        w_diff = (first_linear.weight.data - orig_weight_snap).abs().max().item()
        print(f"  Weight diff after quantize: {w_diff:.6f} (should be >0)")
        assert w_diff > 1e-6, "Quantization didn't change weights!"

        # Restore
        utils.restore_weights(g["module"], saved)

        # Verify weights are restored by direct tensor comparison (not inference output,
        # since the denoising loop uses random noise making action comparison unreliable)
        w_restore_diff = (first_linear.weight.data - orig_weight_snap).abs().max().item()
        print(f"  Weight diff after restore: {w_restore_diff:.10f} (should be 0)")
        assert w_restore_diff == 0.0, f"Weight restore failed! max diff={w_restore_diff}"

        # Also run a quick inference to check it doesn't crash
        utils.run_inference(policy[0], observations[0][0])
        print(f"  Post-restore inference: OK")

        return {"group": g["name"], "w_diff_quant": w_diff, "w_diff_restore": w_restore_diff}
    check("Quantization smoke test", _quant)

    # ---------------------------------------------------------------- task→suite mapping
    def _tasks():
        prompts = utils.load_task_prompts()
        print(f"\n  task_index → suite → instruction:")
        suite_map = {}
        for tid in sorted(prompts.keys()):
            suite = utils.suite_of(tid)
            suite_map[tid] = suite
            print(f"    {tid:3d}  [{suite:8s}]  {prompts[tid][:80]}")
        suites = {}
        for tid, s in suite_map.items():
            suites.setdefault(s, []).append(tid)
        print(f"\n  Suite summary:")
        for s, tids in sorted(suites.items()):
            print(f"    {s}: task indices {tids}")
        return {"descriptions": {str(k): v for k, v in prompts.items()},
                "suite_map": {str(k): v for k, v in suite_map.items()},
                "suites": {s: tids for s, tids in suites.items()}}
    task_info = check("Task→suite mapping", _tasks)

    # ---- save suite map for other scripts to reuse ----
    if task_info and "suite_map" in task_info:
        utils.save_json(
            {"suite_map": task_info["suite_map"], "descriptions": task_info["descriptions"]},
            os.path.join(utils.RESULTS_DIR, "task_suite_map.json"),
        )
        print(f"\n  Saved task_suite_map.json for downstream experiments")

    # ---------------------------------------------------------------- verdict
    print(f"\n{'='*60}")
    if ok:
        report["status"] = "ALL_PASSED"
        print("ALL CHECKS PASSED — safe to start overnight experiments.")
    else:
        report["status"] = "SOME_FAILED"
        print("SOME CHECKS FAILED:")
        for e in report["errors"]:
            print(f"  {e['check']}: {e['error']}")
    print(f"{'='*60}\n")

    utils.save_json(report, os.path.join(utils.RESULTS_DIR, "verification.json"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
