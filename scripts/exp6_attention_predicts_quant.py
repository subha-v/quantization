#!/usr/bin/env python3
"""
Experiment 6 — Do FP16 attention features predict where quantization breaks?

This is the payoff experiment for the attention-as-controller hypothesis. Exp5
showed FP16 attention features perfectly classify Object vs Long rollouts. That
correlates with suite-level difficulty but could be driven by prompt-grammar or
scene-complexity confounds rather than an actionable "precision-need" signal.

Here we directly test: given the same FP16 attention feature vector, can we
predict per-rollout **quantization sensitivity** — i.e., which rollouts collapse
(or take more steps) when we W4-quantize the model?

Procedure:
  1. Load the 50 FP16 rollouts from exp5_rollout_summary.jsonl (each has its
     1350-dim attention feature vector + FP16 success/steps).
  2. For each quantization config in CONFIGS, install quantized weights on the
     model and re-run the same 50 (task × seed) rollouts via rollout.py. Record
     outcomes; do NOT capture attention (saves ~10% per rollout).
  3. Compute per-rollout outcome deltas: steps_delta, broke_by_quant.
  4. Regress FP16 attention features → outcome deltas. Use Ridge (p >> n) and
     LOTP CV (leave-one-task-pair out, as in exp5 reanalysis).
  5. Baseline comparison: suite-only classifier (is attention info better than
     just knowing Object-vs-Long?) and constant predictor.

Quantization configs (all weight-only, symmetric, group=128, fake-quant):
  fp16            — no quantization (sanity: should reproduce exp5 outcomes)
  w4_vlm          — VLM only at W4, expert FP16
  w4_expert       — expert only at W4 (all 10 denoise steps), VLM FP16
  w4_both         — VLM + expert at W4
  w2_vlm_protect  — VLM at W2 except the two exp2-bottleneck layers (layer 0 +
                    vision tower) kept at FP16; expert FP16
  w2_both         — most aggressive; will likely break many rollouts

Usage:
  python exp6_attention_predicts_quant.py --configs fp16 w4_both
  python exp6_attention_predicts_quant.py --configs w4_vlm w4_expert w4_both
  python exp6_attention_predicts_quant.py --all
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


# ---------------------------------------------------------------------------
# VLM vs expert module discovery
# ---------------------------------------------------------------------------
def find_expert(model):
    """Locate gemma_expert submodule."""
    for name, mod in model.named_modules():
        if name.endswith("gemma_expert"):
            n_lin = sum(1 for _, m in mod.named_modules() if isinstance(m, torch.nn.Linear))
            if n_lin > 5:
                return name, mod
    raise RuntimeError("Could not find action expert")


def find_vlm_root(model):
    """Locate the paligemma_with_expert.paligemma submodule (full VLM including
    vision tower, decoder, projector — everything except gemma_expert)."""
    for name, mod in model.named_modules():
        if name.endswith("paligemma_with_expert.paligemma"):
            return name, mod
    # Fallback: root-level submodule whose children include both vision and decoder
    raise RuntimeError("Could not find paligemma VLM root")


def _get_bottleneck_protect_modules(model):
    """Locate VLM layer 0 + vision tower — the two W2-critical groups from exp2.
    Returns list of (name, module) to exclude from aggressive quantization."""
    protect = []
    for name, mod in model.named_modules():
        # First paligemma decoder layer
        if name.endswith("paligemma.model.language_model.layers.0"):
            protect.append((name, mod))
        # Entire vision tower
        if name.endswith("paligemma.model.vision_tower"):
            protect.append((name, mod))
    return protect


# ---------------------------------------------------------------------------
# Quantization installer (reuses utils.fake_quantize_module)
# ---------------------------------------------------------------------------
def install_quant(model, config_name):
    """Install quantized weights on model in-place. Returns (saved_state,
    linear_module_list) for clean restoration. saved_state is a dict
    {global_name: original_weight_tensor} keyed by the FULL module-path name."""
    saved = {}
    vlm_name, vlm = find_vlm_root(model)
    expert_name, expert = find_expert(model)

    def _quantize_linears_in(module, prefix, bits, exclude_prefixes=()):
        """Quantize each Linear in `module`, unless its global name starts with
        any exclude prefix. Save originals into `saved`."""
        for n, m in module.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            global_name = f"{prefix}.{n}" if n else prefix
            if any(global_name.startswith(p) for p in exclude_prefixes):
                continue
            saved[global_name] = m.weight.data.clone()
            # Inline: use fake_quantize logic on this single Linear
            qmax = 2 ** (bits - 1) - 1
            w = m.weight.data.float()
            group_size = 128
            if w.shape[1] >= group_size and w.shape[1] % group_size == 0:
                g = w.reshape(w.shape[0], -1, group_size)
                s = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
                q = ((g / s).round().clamp(-qmax, qmax) * s).reshape_as(w).to(m.weight.dtype)
            else:
                s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
                q = ((w / s).round().clamp(-qmax, qmax) * s).to(m.weight.dtype)
            m.weight.data.copy_(q)

    cfg = config_name
    if cfg == "fp16":
        pass  # no changes
    elif cfg == "w4_vlm":
        _quantize_linears_in(vlm, vlm_name, bits=4)
    elif cfg == "w4_expert":
        _quantize_linears_in(expert, expert_name, bits=4)
    elif cfg == "w4_both":
        _quantize_linears_in(vlm, vlm_name, bits=4)
        _quantize_linears_in(expert, expert_name, bits=4)
    elif cfg == "w4_vlm_protect":
        # Protect layer 0 + vision tower per exp2; W4 the rest of VLM
        protect = _get_bottleneck_protect_modules(model)
        protect_names = [n for n, _ in protect]
        _quantize_linears_in(vlm, vlm_name, bits=4, exclude_prefixes=protect_names)
    elif cfg == "w2_vlm_protect":
        # Protect layer 0 + vision tower per exp2; W2 the rest of VLM
        protect = _get_bottleneck_protect_modules(model)
        protect_names = [n for n, _ in protect]
        _quantize_linears_in(vlm, vlm_name, bits=2, exclude_prefixes=protect_names)
    elif cfg == "w2_both":
        protect = _get_bottleneck_protect_modules(model)
        protect_names = [n for n, _ in protect]
        _quantize_linears_in(vlm, vlm_name, bits=2, exclude_prefixes=protect_names)
        _quantize_linears_in(expert, expert_name, bits=2)
    else:
        raise ValueError(f"Unknown config: {cfg}")

    utils.log(f"[exp6]   installed config={cfg}  quantized {len(saved)} Linear layers")
    return saved


def uninstall_quant(model, saved):
    """Restore original weights."""
    # Find all Linears and match by name
    name_to_mod = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    restored = 0
    for name, orig in saved.items():
        if name in name_to_mod:
            name_to_mod[name].weight.data.copy_(orig)
            restored += 1
    utils.log(f"[exp6]   restored {restored}/{len(saved)} weights")


# ---------------------------------------------------------------------------
# Rerun the 50 rollouts under a config
# ---------------------------------------------------------------------------
def rerun_under_config(policy, model, config_name, fp16_rollouts, per_rollout_path):
    """Run each FP16 rollout again under the quantization config; record outcomes.
    Uses the same (task_id, suite, seed, episode_idx) triple for determinism."""
    saved = install_quant(model, config_name)
    results = []
    t_all = time.time()
    try:
        # Group fp16 rollouts by task for env reuse
        by_task = defaultdict(list)
        for r in fp16_rollouts:
            by_task[(r["suite"], r["task_id"])].append(r)

        done = 0
        total = len(fp16_rollouts)
        for (suite, task_id), rs in by_task.items():
            rs = sorted(rs, key=lambda r: r["seed"])
            try:
                env, desc, init_states = rollout.make_libero_env(
                    suite=suite, task_id=task_id, seed=rs[0]["seed"])
            except Exception as e:
                utils.log(f"[exp6] env-build FAILED {suite}#{task_id}: {e}")
                continue
            try:
                for r in rs:
                    done += 1
                    # In exp5, episode_idx was enumerate(seeds); seeds were [0..4];
                    # so episode_idx == seed here.
                    ep_idx = int(r["seed"])
                    t0 = time.time()
                    new_rec = rollout.run_rollout(
                        policy, task_id=task_id, suite=suite,
                        seed=r["seed"], episode_idx=ep_idx,
                        env=env, initial_states=init_states, task_description=desc,
                    )
                    wall = time.time() - t0
                    rec = {
                        "quant_config":     config_name,
                        "fp16_rollout_idx": r["rollout_idx"],
                        "suite":            suite,
                        "task_id":          task_id,
                        "seed":             r["seed"],
                        "fp16_success":     bool(r["success"]),
                        "fp16_steps":       int(r["steps"]),
                        "quant_success":    bool(new_rec.success),
                        "quant_steps":      int(new_rec.steps),
                        "quant_wall_s":     float(wall),
                        "steps_delta":      int(new_rec.steps) - int(r["steps"]),
                        "broke_by_quant":   bool(r["success"] and not new_rec.success),
                        "saved_by_quant":   bool((not r["success"]) and new_rec.success),
                    }
                    results.append(rec)
                    utils.append_jsonl(rec, per_rollout_path)
                    utils.log(
                        f"[exp6] ({done}/{total}) cfg={config_name} "
                        f"{suite}#{task_id} s{r['seed']}: "
                        f"fp16={'✓' if r['success'] else '✗'}({r['steps']}) "
                        f"quant={'✓' if new_rec.success else '✗'}({new_rec.steps}) "
                        f"Δ={new_rec.steps - r['steps']:+d} "
                        f"wall={wall:.1f}s"
                    )
            finally:
                try: env.close()
                except Exception: pass
    finally:
        uninstall_quant(model, saved)
        utils.log(f"[exp6] config={config_name} total wall: {(time.time()-t_all)/60:.1f} min")

    return results


# ---------------------------------------------------------------------------
# Regression: attention features → outcome
# ---------------------------------------------------------------------------
def regress(X, y, groups, task_is_binary, suite_baseline=None):
    """LOTP CV regression/classification. Returns dict with cv scores."""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, r2_score

    uniq = sorted(set(groups))
    obj_groups = [g for g in uniq if g.startswith("Object__")]
    long_groups = [g for g in uniq if g.startswith("Long__")]
    folds = []
    for og in obj_groups:
        for lg in long_groups:
            te = np.where((groups == og) | (groups == lg))[0]
            tr = np.where((groups != og) & (groups != lg))[0]
            folds.append((tr, te))

    scores = []
    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        # Drop constant cols
        keep = np.std(Xtr, axis=0) > 1e-9
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        if Xtr.shape[1] == 0:
            continue
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        if task_is_binary:
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=0.5, solver="liblinear")
            clf.fit(Xtr_s, y[tr])
            prob = clf.predict_proba(Xte_s)[:, 1]
            scores.append(roc_auc_score(y[te], prob))
        else:
            if np.std(y[tr]) < 1e-9 or np.std(y[te]) < 1e-9:
                continue
            reg = Ridge(alpha=10.0)
            reg.fit(Xtr_s, y[tr])
            pred = reg.predict(Xte_s)
            scores.append(r2_score(y[te], pred))

    if not scores:
        return {"mean": float("nan"), "std": 0.0, "n_folds": 0, "scores": []}
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "n_folds": len(scores),
        "scores": [float(s) for s in scores],
    }


# ---------------------------------------------------------------------------
# Analysis + tables
# ---------------------------------------------------------------------------
def analyze_config(config_name, quant_records, fp16_rollouts):
    """For one quant config, compute outcome stats and regression."""
    # Index FP16 rollouts for feature lookup
    fp16_by_idx = {r["rollout_idx"]: r for r in fp16_rollouts}

    # Outcome summary
    broke = [r for r in quant_records if r["broke_by_quant"]]
    n_total = len(quant_records)
    broke_n = len(broke)
    deltas = [r["steps_delta"] for r in quant_records]
    fp16_succ = sum(1 for r in quant_records if r["fp16_success"])
    quant_succ = sum(1 for r in quant_records if r["quant_success"])

    # Build (X, y, groups) for regression
    feat_keys = sorted({k for r in fp16_rollouts for k in r["features"] if "||" in k})
    rows, y_delta, y_broke, suites, groups = [], [], [], [], []
    for r in quant_records:
        fp = fp16_by_idx.get(r["fp16_rollout_idx"])
        if fp is None: continue
        rows.append([fp["features"].get(k, 0.0) for k in feat_keys])
        y_delta.append(r["steps_delta"])
        y_broke.append(int(r["broke_by_quant"]))
        suites.append(r["suite"])
        groups.append(f"{r['suite']}__{r['task_id']}")

    X = np.asarray(rows, dtype=np.float64)
    y_d = np.asarray(y_delta, dtype=np.float64)
    y_b = np.asarray(y_broke, dtype=np.int64)
    grp = np.asarray(groups)

    # Only regress if there's variance to predict
    result = {"config": config_name, "n": n_total,
              "broke_n": broke_n, "broke_frac": broke_n / max(n_total, 1),
              "fp16_success_n": fp16_succ, "quant_success_n": quant_succ,
              "steps_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
              "steps_delta_std": float(np.std(deltas)) if deltas else 0.0,
              "steps_delta_max": int(np.max(deltas)) if deltas else 0,
              "steps_delta_min": int(np.min(deltas)) if deltas else 0,
              }

    if np.std(y_d) > 1e-9:
        r2_res = regress(X, y_d, grp, task_is_binary=False)
        result["r2_steps_delta"] = r2_res
    else:
        result["r2_steps_delta"] = {"mean": float("nan"), "note": "no variance in y"}

    if (y_b == 1).sum() >= 3 and (y_b == 0).sum() >= 3:
        auc_res = regress(X, y_b.astype(np.float64), grp, task_is_binary=True)
        result["auc_broke_by_quant"] = auc_res
    else:
        result["auc_broke_by_quant"] = {
            "mean": float("nan"),
            "note": f"too few broken ({(y_b==1).sum()}) or unbroken ({(y_b==0).sum()})"
        }

    return result


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


def write_tables(all_results, out_path):
    lines = ["# Exp6 — Attention Predicts Quantization Sensitivity\n",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]

    # Table 1: outcome summary per config
    rows = []
    for r in all_results:
        rows.append([
            r["config"],
            r["n"],
            f"{r['fp16_success_n']}/{r['n']}",
            f"{r['quant_success_n']}/{r['n']}",
            f"{r['broke_n']} ({r['broke_frac']*100:.1f}%)",
            f"{r['steps_delta_mean']:+.1f} ± {r['steps_delta_std']:.1f}",
            f"[{r['steps_delta_min']}, {r['steps_delta_max']}]",
        ])
    lines += ["\n## Table 1 — Rollout outcomes by quantization config\n", "```",
              fmt_table(["config", "n", "FP16 succ", "quant succ",
                         "broken", "Δsteps mean±std", "Δsteps range"],
                        rows, ["<", ">", ">", ">", ">", ">", ">"]),
              "```\n"]

    # Table 2: regression R² and AUC
    rows = []
    for r in all_results:
        r2 = r["r2_steps_delta"]
        auc = r["auc_broke_by_quant"]
        r2_s = (f"{r2['mean']:+.3f} ± {r2.get('std', 0):.3f} (n={r2.get('n_folds', 0)})"
                if "n_folds" in r2 else f"n/a — {r2.get('note', '')}")
        auc_s = (f"{auc['mean']:.3f} ± {auc.get('std', 0):.3f} (n={auc.get('n_folds', 0)})"
                 if "n_folds" in auc else f"n/a — {auc.get('note', '')}")
        rows.append([r["config"], r2_s, auc_s])
    lines += ["\n## Table 2 — Can FP16 attention features predict quantization outcome?\n",
              "```",
              fmt_table(["config", "R² on steps_delta (LOTP)",
                         "AUC on broke_by_quant (LOTP)"],
                        rows, ["<", ">", ">"]),
              "```\n"]
    lines.append("R² > 0.3 (continuous target) or AUC > 0.70 (binary) under LOTP CV = real controller signal.\n")

    # Verdict
    best_r2 = max((r["r2_steps_delta"].get("mean", float("-inf"))
                   for r in all_results if "n_folds" in r["r2_steps_delta"]),
                  default=float("-inf"))
    best_auc = max((r["auc_broke_by_quant"].get("mean", 0.5)
                    for r in all_results if "n_folds" in r["auc_broke_by_quant"]),
                   default=0.5)
    lines.append("\n## Verdict\n")
    lines.append(f"Best R² (cross-task) on steps_delta: {best_r2:.3f}")
    lines.append(f"Best AUC (cross-task) on broke_by_quant: {best_auc:.3f}")
    if best_r2 >= 0.30 or best_auc >= 0.70:
        lines.append("\n**Attention features carry a quantization-sensitivity signal beyond suite-identity.**")
        lines.append("Justifies building the adaptive-precision controller on top of attention features.")
    elif best_r2 > 0 or best_auc > 0.55:
        lines.append("\n**Weak signal.** Attention has *some* predictive info for quant outcomes but the signal")
        lines.append("is modest. The Object-vs-Long AUC=1.0 from exp5 was likely driven by prompt/scene confounds")
        lines.append("that don't translate to precision-need prediction.")
    else:
        lines.append("\n**No signal.** Attention features do not predict quant outcomes. The exp5 perfect")
        lines.append("separation was entirely task-structure fingerprinting with no quantization-relevant signal.")
        lines.append("Attention-based adaptive controller is not justified by this evidence.")

    content = "\n".join(lines) + "\n"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(content)
    utils.log(f"[exp6] tables → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ALL_CONFIGS = ["fp16", "w4_vlm", "w4_expert", "w4_both", "w2_vlm_protect", "w2_both"]


def main():
    utils.setup_logging()
    utils.log("=" * 60)
    utils.log("EXP6: Does FP16 attention predict quantization sensitivity?")
    utils.log("=" * 60)

    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["w4_vlm", "w4_both"])
    p.add_argument("--all", action="store_true", help="Run all configs in ALL_CONFIGS")
    args = p.parse_args()

    configs = ALL_CONFIGS if args.all else args.configs
    for c in configs:
        assert c in ALL_CONFIGS, f"unknown config: {c}"

    # Load FP16 rollouts from exp5
    fp16_path = os.path.join(utils.RESULTS_DIR, "exp5_rollout_summary.jsonl")
    fp16_rollouts = []
    with open(fp16_path) as f:
        for line in f:
            if line.strip():
                fp16_rollouts.append(json.loads(line))
    utils.log(f"[exp6] Loaded {len(fp16_rollouts)} FP16 rollouts with attention features")

    # Pre-flight
    rollout.smoke_render()

    # Load policy
    with utils.Timer("Model loading"):
        policy, model = utils.load_policy("pi05_libero")

    # Clear per-rollout output
    per_rollout_path = os.path.join(utils.RESULTS_DIR, "exp6_per_rollout.jsonl")
    if os.path.exists(per_rollout_path):
        os.remove(per_rollout_path)

    # Run each config
    all_results = []
    for cfg in configs:
        utils.log(f"\n[exp6] ==== Config: {cfg} ====")
        records = rerun_under_config(policy, model, cfg, fp16_rollouts, per_rollout_path)
        utils.log(f"[exp6] Analyzing config={cfg}...")
        result = analyze_config(cfg, records, fp16_rollouts)
        all_results.append(result)

    # Write tables + JSON dump of results
    tables_path = os.path.join(utils.RESULTS_DIR, "exp6_tables.md")
    write_tables(all_results, tables_path)
    with open(os.path.join(utils.RESULTS_DIR, "exp6_regression.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
