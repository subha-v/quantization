#!/usr/bin/env python3
"""
Experiment 7 — Per-FRAME attention features vs per-frame W4 action MSE.

Responding to the "per-rollout regression throws away the mechanistic signal"
critique of exp6. This rebuilds the regression at the granularity the hypothesis
actually cares about: a single VLM call.

Procedure:
  1. Re-run each of the 50 exp5 rollouts in FP16. At every VLM call, capture
     the openpi observation dict AND the FP16 action chunk. Rollout continues
     as usual (proceeds through the trajectory).
  2. Install W4 quantization on VLM + expert. For each captured observation,
     call policy.infer(obs) to get the W4 action chunk (this is a single
     1-chunk inference, NOT a re-rollout, so there's no trajectory divergence).
  3. Per-call MSE_w4 = mean((FP16_chunk - W4_chunk)^2). Continuous, directly
     measurable per-frame sensitivity signal — same quantity exp3 used but
     now across ~1500 frames drawn from real rollouts (mix of easy/hard tasks
     and early/mid/late within-rollout phases).
  4. Join by (rollout_idx, call_idx) with per-call attention features already
     in exp5_per_call.jsonl (~2871 records / 50 rollouts).
  5. Output: exp7_per_frame.jsonl — one row per (rollout_idx, call_idx) with
     the attention features + W4 MSE target.

Downstream analysis is in exp7_analyze.py. This script only collects data.

Usage:
  python exp7_per_frame_sensitivity.py --config w4_both
  python exp7_per_frame_sensitivity.py --config w2_both
"""

import argparse
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
# Reuse exp6's quantization install/uninstall helpers
from exp6_attention_predicts_quant import install_quant, uninstall_quant


# ---------------------------------------------------------------------------
# Per-rollout capture: stash (openpi_obs, fp16_action_chunk) at every VLM call
# ---------------------------------------------------------------------------
class RolloutCapture:
    """Accumulates per-call data during a single FP16 rollout."""
    def __init__(self):
        self.calls = []    # list of {call_idx, t, openpi_obs, fp16_chunk}
        self._pending_obs = None
        self._pending_t   = None
        self.call_idx = 0

    def obs_cb(self, t, libero_obs, openpi_obs):
        # Deep-copy the openpi obs so later mutations don't clobber it
        self._pending_obs = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                             for k, v in openpi_obs.items()}
        self._pending_t = int(t)

    def action_cb(self, t, action_chunk):
        # Pair the pending obs with this chunk
        if self._pending_obs is None:
            return
        self.calls.append({
            "call_idx": self.call_idx,
            "t": int(self._pending_t),
            "obs": self._pending_obs,
            "fp16_chunk": np.asarray(action_chunk, dtype=np.float32),
        })
        self.call_idx += 1
        self._pending_obs = None
        self._pending_t = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    utils.setup_logging()
    utils.log("=" * 60)
    utils.log("EXP7: Per-frame attention vs per-frame W4 action MSE")
    utils.log("=" * 60)

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="w4_both",
                   choices=["w4_vlm", "w4_expert", "w4_both",
                            "w2_vlm_protect", "w2_both"])
    p.add_argument("--smoke", action="store_true",
                   help="1 rollout end-to-end — validates capture + quant")
    args = p.parse_args()

    # Load FP16 rollouts from exp5 so we know which (task, seed) triples to replay
    fp16_path = os.path.join(utils.RESULTS_DIR, "exp5_rollout_summary.jsonl")
    exp5 = [json.loads(l) for l in open(fp16_path) if l.strip()]
    utils.log(f"[exp7] exp5 rollouts to replay: {len(exp5)}")

    if args.smoke:
        exp5 = exp5[:1]
        utils.log("[exp7] SMOKE mode: 1 rollout")

    # Pre-flight
    rollout.smoke_render()

    with utils.Timer("Model loading"):
        policy, model = utils.load_policy("pi05_libero")

    out_path = os.path.join(utils.RESULTS_DIR, f"exp7_per_frame__{args.config}.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    total_calls = 0
    t_all = time.time()

    # Group rollouts by task for env reuse
    by_task = defaultdict(list)
    for r in exp5:
        by_task[(r["suite"], r["task_id"])].append(r)

    for (suite, task_id), rs in by_task.items():
        rs = sorted(rs, key=lambda r: r["seed"])
        try:
            env, desc, init_states = rollout.make_libero_env(
                suite=suite, task_id=task_id, seed=rs[0]["seed"])
        except Exception as e:
            utils.log(f"[exp7] env-build FAILED {suite}#{task_id}: {e}")
            continue

        try:
            for r in rs:
                rollout_idx = r["rollout_idx"]
                utils.log(
                    f"\n[exp7] rollout {rollout_idx} suite={suite} "
                    f"task={task_id} seed={r['seed']}"
                )

                # Phase 1: FP16 rollout with capture
                cap = RolloutCapture()
                ep_idx = int(r["seed"])
                t0 = time.time()
                fp_rec = rollout.run_rollout(
                    policy, task_id=task_id, suite=suite,
                    seed=r["seed"], episode_idx=ep_idx,
                    env=env, initial_states=init_states, task_description=desc,
                    obs_callback=cap.obs_cb,
                    action_callback=cap.action_cb,
                )
                fp_wall = time.time() - t0
                utils.log(f"[exp7]   FP16: success={fp_rec.success} "
                          f"steps={fp_rec.steps} calls={len(cap.calls)} wall={fp_wall:.1f}s")

                # Phase 2: install quantization, replay each captured obs
                saved = install_quant(model, args.config)
                try:
                    t0 = time.time()
                    for c in cap.calls:
                        with torch.no_grad():
                            result = policy.infer(c["obs"])
                        w4_chunk = np.asarray(
                            result["actions"] if isinstance(result, dict) else result,
                            dtype=np.float32,
                        )
                        # Per-call MSE (mean over action dims × horizon)
                        fp = c["fp16_chunk"]
                        if w4_chunk.shape != fp.shape:
                            # Unlikely but guard against — policy.infer may pad differently
                            k = min(w4_chunk.shape[0], fp.shape[0])
                            mse = float(np.mean((fp[:k] - w4_chunk[:k]) ** 2))
                        else:
                            mse = float(np.mean((fp - w4_chunk) ** 2))

                        # Emit record (no observation payload — just metadata + MSE)
                        utils.append_jsonl({
                            "rollout_idx": rollout_idx,
                            "call_idx":    c["call_idx"],
                            "t":           c["t"],
                            "suite":       suite,
                            "task_id":     task_id,
                            "seed":        r["seed"],
                            "quant_config": args.config,
                            "fp16_action_norm": float(np.linalg.norm(fp)),
                            "w4_action_norm":   float(np.linalg.norm(w4_chunk)),
                            "w4_mse":           mse,
                            "fp16_succeeded":   bool(fp_rec.success),
                            "rollout_steps":    int(fp_rec.steps),
                            "rollout_n_calls":  len(cap.calls),
                        }, out_path)
                        total_calls += 1
                    q_wall = time.time() - t0
                    utils.log(f"[exp7]   Quant {args.config}: {len(cap.calls)} infers "
                              f"wall={q_wall:.1f}s  (cumul records: {total_calls})")
                finally:
                    uninstall_quant(model, saved)
        finally:
            try: env.close()
            except Exception: pass

    utils.log(f"\n[exp7] Done. {total_calls} per-frame records in {out_path}")
    utils.log(f"[exp7] Total wall: {(time.time()-t_all)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
