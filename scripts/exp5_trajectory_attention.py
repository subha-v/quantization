#!/usr/bin/env python3
"""
Experiment 5 — Trajectory-level attention dynamics during LIBERO rollouts.

Research question: do attention patterns over a rollout (not just at a single frame)
distinguish short-horizon (Object) from long-horizon (Long) tasks? If yes, we have
a runtime-cheap difficulty signal usable for adaptive-precision quantization.

Inspired by ThinKV (2510.01290), which decomposes CoT into thought types via
attention sparsity. We translate: does a similar sparsity signal separate Long
from Object rollouts?

Method:
  • 5 tasks × 5 seeds × {Object, Long} = 50 rollouts using the rollout.py harness.
  • During every `policy.infer()` call, monkey-patched VLM attention layers capture
    the softmax attention probabilities and compute per-(layer, head, call) stats:
    sparsity, entropy, top1_mass, top5_mass, attention_sink_mass.
  • Per-rollout aggregate features: static (mean) AND dynamic (std over time,
    autocorrelation) per (layer, head).
  • Logistic regression classifier on features → Long/Object. Ablation: static-only
    vs static+dynamic, to test whether dynamics add real signal.

Output: markdown tables + JSONL per-call features + NPZ per-rollout aggregates.

Usage:
  python exp5_trajectory_attention.py --smoke           # 1 rollout to validate hooks
  python exp5_trajectory_attention.py                    # full 50-rollout sweep
  python exp5_trajectory_attention.py --tasks-per-suite 3 --seeds 0 1  # quick run
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
import rollout  # also monkey-patches torch.load for LIBERO compat

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_SUITES = ("Object", "Long")
DEFAULT_TASKS_PER_SUITE = 5
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

SUITE_GLOBAL_TASK_BASE = {"Long": 0, "Goal": 10, "Object": 20, "Spatial": 30}


# ---------------------------------------------------------------------------
# Attention recorder — monkey-patches VLM attention modules' forward() to
# capture softmax probabilities and compute per-call stats.
# ---------------------------------------------------------------------------
class AttentionRecorder:
    """Installs wrappers on VLM attention modules that capture attn_weights
    via output_attentions=True. Computes derived scalar stats per (layer, head)
    and stores them keyed by (call_idx, layer_name)."""

    def __init__(self, model):
        self.model = model
        self._patched = []   # (module, original_forward)
        self.records = []    # list of dicts
        self.call_idx = 0
        self.step_in_rollout = 0

        vlm_attn_modules = self._find_vlm_attention(model)
        utils.log(f"[exp5] Found {len(vlm_attn_modules)} VLM attention modules")
        for name, mod in vlm_attn_modules:
            self._install(name, mod)

    # -----------------------------------------------------------------------
    def _find_vlm_attention(self, model):
        """Locate attention modules inside the VLM (paligemma + vision tower),
        excluding the action expert."""
        found = []
        for name, mod in model.named_modules():
            if "gemma_expert" in name:
                continue
            # Attention modules in PaliGemma: have .q_proj, .k_proj, .v_proj children
            children = {n for n, _ in mod.named_children()}
            if {"q_proj", "k_proj", "v_proj"}.issubset(children) or \
               {"q_proj", "k_proj", "v_proj", "out_proj"}.issubset(children) or \
               {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(children):
                found.append((name, mod))
        return found

    # -----------------------------------------------------------------------
    def _install(self, name, module):
        original_forward = module.forward
        recorder = self

        def wrapped_forward(*args, **kwargs):
            # Force output_attentions=True so we get attn_weights in the return.
            prev = kwargs.get("output_attentions", None)
            kwargs["output_attentions"] = True
            try:
                result = original_forward(*args, **kwargs)
            except TypeError:
                # Some modules may not accept output_attentions; fall back.
                kwargs.pop("output_attentions", None)
                if prev is not None:
                    kwargs["output_attentions"] = prev
                return original_forward(*args, **kwargs)

            # Extract attn_weights. Gemma convention: (attn_output, attn_weights, past_kv)
            attn_weights = None
            if isinstance(result, tuple):
                for item in result:
                    if isinstance(item, torch.Tensor) and item.dim() == 4:
                        # (batch, num_heads, q_len, k_len)
                        attn_weights = item
                        break

            if attn_weights is not None:
                recorder._record(name, attn_weights)

            return result

        module.forward = wrapped_forward
        self._patched.append((module, original_forward))

    # -----------------------------------------------------------------------
    def _record(self, layer_name, attn):
        with torch.no_grad():
            a = attn.detach().float()
            # Shape (B=1, H, Q, K). Take first row of batch.
            if a.size(0) > 1:
                a = a[:1]

            # Row-max threshold for ThinKV-style sparsity (<1% of row max)
            row_max = a.amax(dim=-1, keepdim=True).clamp(min=1e-12)
            below = (a < 0.01 * row_max).float()
            sparsity_h = below.mean(dim=(0, 2, 3))    # (H,)

            # Entropy per row, averaged across Q positions, per head
            eps = 1e-12
            entropy_h = -(a * (a + eps).log()).sum(dim=-1).mean(dim=(0, 2))  # (H,)

            # Top-k row masses per head
            top1_h = a.amax(dim=-1).mean(dim=(0, 2))  # (H,)
            top5_h = a.topk(min(5, a.size(-1)), dim=-1).values.sum(dim=-1).mean(dim=(0, 2))

            # Attention sink mass on position 0
            sink_h = a[..., 0].mean(dim=(0, 2))

        self.records.append({
            "call_idx": int(self.call_idx),
            "step_in_rollout": int(self.step_in_rollout),
            "layer": layer_name,
            "sparsity_per_head": sparsity_h.cpu().numpy().tolist(),
            "entropy_per_head": entropy_h.cpu().numpy().tolist(),
            "top1_per_head": top1_h.cpu().numpy().tolist(),
            "top5_per_head": top5_h.cpu().numpy().tolist(),
            "sink_per_head": sink_h.cpu().numpy().tolist(),
            "seq_len": int(a.size(-1)),
            "num_heads": int(a.size(1)),
        })

    # -----------------------------------------------------------------------
    def mark_new_call(self, step_in_rollout):
        self.call_idx += 1
        self.step_in_rollout = int(step_in_rollout)

    def reset(self):
        self.records.clear()
        self.call_idx = 0
        self.step_in_rollout = 0

    def uninstall(self):
        for module, original_forward in self._patched:
            module.forward = original_forward
        self._patched.clear()


# ---------------------------------------------------------------------------
# Per-rollout feature aggregation
# ---------------------------------------------------------------------------
def _autocorr_lag1(x):
    """Lag-1 autocorrelation of a 1D array. NaN-safe (returns 0 if degenerate)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return 0.0
    x_centered = x - x.mean()
    denom = (x_centered ** 2).sum()
    if denom < 1e-12:
        return 0.0
    num = (x_centered[:-1] * x_centered[1:]).sum()
    return float(num / denom)


def aggregate_rollout_features(records):
    """Turn per-(call, layer) records into per-rollout summary features.

    Output: dict of feature_name -> scalar.
      Static features:  mean_sparsity[layer], mean_entropy[layer], ...
      Dynamic features: std_sparsity[layer], autocorr_sparsity[layer],
                        max_sparsity[layer], range_sparsity[layer], ...
    All values are averaged across heads (since head count is constant per layer).
    """
    by_layer = defaultdict(list)
    for r in records:
        by_layer[r["layer"]].append(r)

    out = {"n_calls": len(set(r["call_idx"] for r in records))}
    for layer_name, rs in by_layer.items():
        # Sort by call_idx for time-series analysis
        rs = sorted(rs, key=lambda r: r["call_idx"])
        # Per-call, head-averaged values (time series)
        sparsity_ts  = np.array([np.mean(r["sparsity_per_head"]) for r in rs])
        entropy_ts   = np.array([np.mean(r["entropy_per_head"])  for r in rs])
        top1_ts      = np.array([np.mean(r["top1_per_head"])     for r in rs])
        top5_ts      = np.array([np.mean(r["top5_per_head"])     for r in rs])
        sink_ts      = np.array([np.mean(r["sink_per_head"])     for r in rs])

        # Static (snapshot-equivalent) features — averaged over time
        safe = lambda arr: {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr)),
            "max":    float(np.max(arr)) if arr.size else 0.0,
            "min":    float(np.min(arr)) if arr.size else 0.0,
            "range":  float(np.max(arr) - np.min(arr)) if arr.size else 0.0,
            "acorr1": _autocorr_lag1(arr),
        }
        for metric_name, ts in [("sparsity", sparsity_ts), ("entropy", entropy_ts),
                                ("top1", top1_ts), ("top5", top5_ts),
                                ("sink", sink_ts)]:
            stats = safe(ts)
            for stat_name, v in stats.items():
                out[f"{layer_name}||{metric_name}||{stat_name}"] = v

    return out


# ---------------------------------------------------------------------------
# Classifier: static vs dynamic ablation
# ---------------------------------------------------------------------------
def run_classifier(X, y, feature_names, label="all"):
    """Logistic regression with 5-fold CV + permutation test.
    Returns (mean_auc, std_auc, perm_p, top_features)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    # Drop constant columns to avoid instability
    keep = np.std(X, axis=0) > 1e-9
    X = X[:, keep]
    feature_names_kept = [f for f, k in zip(feature_names, keep) if k]

    # 5-fold stratified
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        if len(set(y[te])) > 1:
            fold_aucs.append(roc_auc_score(y[te], prob))
    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
    std_auc = float(np.std(fold_aucs)) if fold_aucs else 0.0

    # Permutation test (100 shuffles)
    rng = np.random.default_rng(123)
    null_aucs = []
    for _ in range(100):
        y_perm = rng.permutation(y)
        fold_aucs_p = []
        for tr, te in skf.split(X, y_perm):
            scaler = StandardScaler().fit(X[tr])
            Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
            clf.fit(Xtr, y_perm[tr])
            prob = clf.predict_proba(Xte)[:, 1]
            if len(set(y_perm[te])) > 1:
                fold_aucs_p.append(roc_auc_score(y_perm[te], prob))
        null_aucs.append(np.mean(fold_aucs_p) if fold_aucs_p else 0.5)
    null_aucs = np.array(null_aucs)
    perm_p = float((null_aucs >= mean_auc).mean())

    # Top features (full-data fit)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
    clf.fit(Xs, y)
    coefs = clf.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    top = [(feature_names_kept[i], float(coefs[i])) for i in order[:10]]

    return {
        "label": label,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "perm_p": perm_p,
        "top_features": top,
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def fmt_table(header, rows, aligns=None):
    n = len(header)
    if aligns is None:
        aligns = ["<"] * n
    widths = [len(str(h)) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    def _r(row):
        return " | ".join(f"{str(c):{aligns[i]}{widths[i]}}" for i, c in enumerate(row))
    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([_r(header), sep] + [_r(r) for r in rows])


def write_tables(rollouts, clf_results, per_layer_stats, out_path):
    lines = ["# Exp5 — Trajectory Attention Dynamics\n",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"n_rollouts: {len(rollouts)}\n"]

    # Table 1 — rollout success by suite
    by_suite = defaultdict(lambda: {"success": 0, "total": 0, "steps": [], "calls": []})
    for r in rollouts:
        s = by_suite[r["suite"]]
        s["total"] += 1
        if r["success"]:
            s["success"] += 1
        s["steps"].append(r["steps"])
        s["calls"].append(r.get("n_calls", 0))
    rows = []
    for suite in sorted(by_suite):
        s = by_suite[suite]
        rows.append([
            suite, f"{s['success']}/{s['total']}",
            f"{np.mean(s['steps']):.0f}", f"{np.mean(s['calls']):.1f}",
        ])
    lines += ["\n## Table 1 — Rollout success + step budget\n", "```",
              fmt_table(["suite", "success", "mean_steps", "mean_vlm_calls"],
                        rows, ["<", ">", ">", ">"]), "```\n"]

    # Table 2 — per-layer attention dynamics Easy vs Hard
    # per_layer_stats[layer][metric] = {'easy_mean': ..., 'hard_mean': ..., 'delta': ..., 'p': ...}
    rows = []
    for layer, metrics in sorted(per_layer_stats.items()):
        for metric, stats in metrics.items():
            rows.append([
                (layer[:50] + "...") if len(layer) > 50 else layer,
                metric,
                f"{stats['easy_mean']:.4f}",
                f"{stats['hard_mean']:.4f}",
                f"{stats['delta']:+.4f}",
                f"{stats['rel']:+.1%}",
                f"{stats['t']:+.2f}",
                f"{stats['p']:.3f}",
            ])
    # Only show top 30 by |t| to keep table readable
    rows = sorted(rows, key=lambda r: -abs(float(r[6])))[:30]
    lines += ["\n## Table 2 — Top 30 (layer × metric) by |t-stat|, Easy vs Hard\n", "```",
              fmt_table(["layer", "metric", "easy_mean", "hard_mean",
                         "delta(H-E)", "rel%", "t", "p"],
                        rows, ["<", "<", ">", ">", ">", ">", ">", ">"]),
              "```\n"]

    # Table 3 — classifier ablation
    rows = []
    for r in clf_results:
        rows.append([
            r["label"], r["n_samples"], r["n_features"],
            f"{r['mean_auc']:.3f} ± {r['std_auc']:.3f}",
            f"{r['perm_p']:.3f}",
        ])
    lines += ["\n## Table 3 — Classifier ablation (Easy vs Hard, 5-fold CV)\n", "```",
              fmt_table(["feature set", "n", "p(features)", "mean AUC ± std", "perm p"],
                        rows, ["<", ">", ">", ">", ">"]), "```\n"]

    # Table 4 — top features per ablation
    for r in clf_results:
        lines.append(f"\n### Top features — {r['label']}\n")
        lines.append("```")
        rows = [[name, f"{coef:+.3f}"] for name, coef in r["top_features"]]
        lines.append(fmt_table(["feature", "coef"], rows, ["<", ">"]))
        lines.append("```")

    # Verdict
    best_auc = max((r["mean_auc"] for r in clf_results), default=0.5)
    lines.append("\n## Verdict\n")
    if best_auc >= 0.75:
        lines.append(f"**Strong signal — best AUC = {best_auc:.3f}.** Attention dynamics separate Easy from Hard. Proceed to Phase 2 (adaptive controller).")
    elif best_auc >= 0.60:
        lines.append(f"**Weak signal — best AUC = {best_auc:.3f}.** Suggestive but not conclusive; consider enlarging the sweep or enriching features.")
    else:
        lines.append(f"**No signal — best AUC = {best_auc:.3f}.** Attention dynamics do not separate Easy from Hard at this scale. Reconsider the hypothesis.")

    content = "\n".join(lines) + "\n"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(content)
    utils.log(f"[exp5] tables → {out_path}")


# ---------------------------------------------------------------------------
# Per-layer Easy-vs-Hard stats (for Table 2)
# ---------------------------------------------------------------------------
def per_layer_easy_vs_hard(rollouts_summary):
    """For each (layer, metric, stat), compute Easy vs Hard mean + Welch t-test."""
    from scipy import stats as sps
    # Separate feature dicts by suite
    easy = [r for r in rollouts_summary if r["suite"] == "Object"]
    hard = [r for r in rollouts_summary if r["suite"] == "Long"]
    if not easy or not hard:
        return {}

    # Gather all feature keys
    keys = set()
    for r in rollouts_summary:
        keys.update(k for k in r["features"] if "||" in k)
    result = defaultdict(dict)
    for k in keys:
        layer, metric, stat = k.split("||")
        e_vals = [r["features"].get(k, 0.0) for r in easy]
        h_vals = [r["features"].get(k, 0.0) for r in hard]
        if np.std(e_vals) < 1e-12 and np.std(h_vals) < 1e-12:
            continue
        e_mean = float(np.mean(e_vals))
        h_mean = float(np.mean(h_vals))
        delta = h_mean - e_mean
        rel = delta / (abs(e_mean) + 1e-12)
        try:
            t, p = sps.ttest_ind(h_vals, e_vals, equal_var=False)
            t, p = float(t), float(p)
        except Exception:
            t, p = 0.0, 1.0
        result[layer][f"{metric}.{stat}"] = {
            "easy_mean": e_mean, "hard_mean": h_mean,
            "delta": delta, "rel": rel, "t": t, "p": p,
        }
    return dict(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    utils.setup_logging()
    utils.log("=" * 60)
    utils.log("EXP5: Trajectory Attention Dynamics during LIBERO Rollouts")
    utils.log("=" * 60)

    p = argparse.ArgumentParser()
    p.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES),
                   choices=list(SUITE_GLOBAL_TASK_BASE))
    p.add_argument("--tasks-per-suite", type=int, default=DEFAULT_TASKS_PER_SUITE)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--smoke", action="store_true",
                   help="1 rollout, 1 task, 1 seed — validate hook pipeline.")
    args = p.parse_args()

    if args.smoke:
        args.suites = ["Object"]
        args.tasks_per_suite = 1
        args.seeds = [0]
        utils.log("[exp5] SMOKE mode")

    utils.log(f"[exp5] suites={args.suites} tasks/suite={args.tasks_per_suite} seeds={args.seeds}")
    n_total = len(args.suites) * args.tasks_per_suite * len(args.seeds)
    utils.log(f"[exp5] total rollouts: {n_total}")

    # Pre-flight
    utils.log("[exp5] Pre-flight headless render check...")
    rollout.smoke_render()

    with utils.Timer("Model loading"):
        policy, model = utils.load_policy("pi05_libero")

    # Install attention recorder
    recorder = AttentionRecorder(model)
    if len(recorder._patched) == 0:
        utils.log("[exp5] FATAL: no VLM attention modules hooked")
        return 1

    per_call_path = os.path.join(utils.RESULTS_DIR, "exp5_per_call.jsonl")
    summary_path = os.path.join(utils.RESULTS_DIR, "exp5_rollout_summary.jsonl")
    for p_ in (per_call_path, summary_path):
        if os.path.exists(p_):
            os.remove(p_)

    rollouts_summary = []
    t_all = time.time()
    done = 0

    try:
        for suite in args.suites:
            base = SUITE_GLOBAL_TASK_BASE[suite]
            for local_tid in range(args.tasks_per_suite):
                global_task_id = base + local_tid
                try:
                    env, task_desc, init_states = rollout.make_libero_env(
                        suite=suite, task_id=global_task_id, seed=args.seeds[0])
                except Exception as e:
                    utils.log(f"[exp5] env-build FAILED {suite}#{global_task_id}: {e}")
                    continue
                try:
                    for ep_idx, seed in enumerate(args.seeds):
                        done += 1
                        recorder.reset()

                        def action_cb(t_step, action_chunk):
                            recorder.mark_new_call(t_step)

                        t0 = time.time()
                        utils.log(f"\n[exp5] ({done}/{n_total}) "
                                  f"suite={suite} task={global_task_id} seed={seed} "
                                  f"desc={task_desc[:50]!r}")
                        rec = rollout.run_rollout(
                            policy, task_id=global_task_id, suite=suite,
                            seed=seed, episode_idx=ep_idx,
                            env=env, initial_states=init_states,
                            task_description=task_desc,
                            action_callback=action_cb,
                        )
                        wall = time.time() - t0
                        n_calls = recorder.call_idx + 1  # 0-indexed → +1
                        utils.log(f"[exp5]   success={rec.success} steps={rec.steps} "
                                  f"vlm_calls={n_calls} records={len(recorder.records)} "
                                  f"wall={wall:.1f}s")

                        # Persist per-call records (prepend rollout metadata)
                        for rr in recorder.records:
                            rr = {**rr,
                                  "rollout_idx": done - 1,
                                  "suite": suite, "task_id": global_task_id,
                                  "seed": seed}
                            utils.append_jsonl(rr, per_call_path)

                        # Aggregate into rollout features
                        feats = aggregate_rollout_features(recorder.records)
                        summary = {
                            "rollout_idx": done - 1,
                            "suite": suite,
                            "task_id": global_task_id,
                            "seed": seed,
                            "success": rec.success,
                            "steps": rec.steps,
                            "n_calls": n_calls,
                            "wall_s": wall,
                            "task_description": task_desc,
                            "features": feats,
                        }
                        rollouts_summary.append(summary)
                        utils.append_jsonl(summary, summary_path)
                finally:
                    try: env.close()
                    except Exception: pass
    finally:
        recorder.uninstall()
        utils.log(f"\n[exp5] Total wall time: {(time.time()-t_all)/60:.1f} min")

    if not rollouts_summary:
        utils.log("[exp5] No rollouts completed — aborting.")
        return 1

    # ---- Feature matrix + classifier ablation ----
    utils.log("\n[exp5] Building feature matrix...")
    suite_labels = {"Object": 0, "Long": 1}
    feat_keys = sorted({k for r in rollouts_summary for k in r["features"] if "||" in k})
    X = np.array([[r["features"].get(k, 0.0) for k in feat_keys]
                  for r in rollouts_summary], dtype=np.float64)
    y = np.array([suite_labels.get(r["suite"], -1) for r in rollouts_summary])
    valid = y >= 0
    X, y = X[valid], y[valid]
    utils.log(f"[exp5]   X={X.shape} y={y.shape} "
              f"Easy={int((y==0).sum())} Hard={int((y==1).sum())}")

    # Ablation: static-only (mean) vs dynamics-only (std/max/range/acorr) vs all
    static_mask  = np.array(["||mean" in k for k in feat_keys])
    dynamic_mask = np.array([any(x in k for x in ("||std", "||max", "||range", "||acorr1")) for k in feat_keys])
    all_mask     = static_mask | dynamic_mask

    ablations = [
        ("all",     all_mask),
        ("static",  static_mask),
        ("dynamic", dynamic_mask),
    ]
    clf_results = []
    for label, mask in ablations:
        if int(mask.sum()) == 0:
            continue
        names = [k for k, m in zip(feat_keys, mask) if m]
        utils.log(f"[exp5] Classifier: label={label} features={mask.sum()}")
        res = run_classifier(X[:, mask], y, names, label=label)
        utils.log(f"[exp5]   AUC={res['mean_auc']:.3f}±{res['std_auc']:.3f} perm_p={res['perm_p']:.3f}")
        clf_results.append(res)

    # Per-layer Easy-vs-Hard stats
    per_layer_stats = per_layer_easy_vs_hard(rollouts_summary)

    # Write tables + save classifier JSON
    tables_path = os.path.join(utils.RESULTS_DIR, "exp5_tables.md")
    write_tables(rollouts_summary, clf_results, per_layer_stats, tables_path)
    with open(os.path.join(utils.RESULTS_DIR, "exp5_classifier.json"), "w") as f:
        json.dump(clf_results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
