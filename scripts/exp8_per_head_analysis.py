#!/usr/bin/env python3
"""
Exp8b — Per-head deep-dive (D2) for the strongest-signal config.

exp7_analyze aggregated features by averaging across heads within each layer.
That may have diluted a real signal concentrated in specific heads. Here we
compute Spearman correlations at per-(layer, head, metric) granularity, then
for the top surviving features look at how the metric distribution differs
between high-sensitivity and low-sensitivity frames.

Usage:
  python exp8_per_head_analysis.py --config w4_vlm
  python exp8_per_head_analysis.py --config w2_vlm_protect

Input: exp5_per_call.jsonl (per-head arrays) + exp7_per_frame__{config}.jsonl.
Output: results/exp8_per_head__{config}.md.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import utils

METRICS = ("sparsity", "entropy", "top1", "top5", "sink")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def short_layer(name):
    """Shorten long dotted paths for display."""
    parts = name.split(".")
    # keep last 4 components
    return ".".join(parts[-4:]) if len(parts) > 4 else name


def build_per_head_matrix(exp7_records, exp5_per_call):
    """Build (X, y, suites, groups, feat_keys) where X columns are
    (layer, head_idx, metric) triples joined on (rollout_idx, call_idx).

    Because different layers have different head counts (vision tower often
    16, language model often 8), we identify features by (layer, h, metric)
    and fill missing entries with 0.
    """
    # Index exp5 by (rollout_idx, call_idx) → list of per-layer records
    attn_by_key = defaultdict(list)
    for r in exp5_per_call:
        attn_by_key[(r["rollout_idx"], r["call_idx"])].append(r)

    # Determine num_heads per layer (assume constant across calls)
    heads_per_layer = {}
    for r in exp5_per_call[:5000]:  # sample a few thousand
        layer = r["layer"]
        if layer not in heads_per_layer:
            vals = r.get("sparsity_per_head", [])
            heads_per_layer[layer] = len(vals)

    # Build feature schema: sorted (layer, head, metric) triples
    triples = []
    for layer, nh in heads_per_layer.items():
        for h in range(nh):
            for m in METRICS:
                triples.append((layer, h, m))
    triples.sort()
    feat_keys = [f"{l}||h{h}||{m}" for (l, h, m) in triples]
    triple_to_col = {(l, h, m): i for i, (l, h, m) in enumerate(triples)}

    # Populate X, y
    n_records = len(exp7_records)
    p = len(feat_keys)
    X = np.zeros((n_records, p), dtype=np.float32)
    y = np.zeros(n_records, dtype=np.float64)
    suites = []
    groups = []
    keep_mask = np.zeros(n_records, dtype=bool)

    for i, r in enumerate(exp7_records):
        key = (r["rollout_idx"], r["call_idx"])
        if key not in attn_by_key:
            continue
        layer_records = attn_by_key[key]
        for lr in layer_records:
            layer = lr["layer"]
            nh = heads_per_layer.get(layer, 0)
            for m in METRICS:
                arr = lr.get(f"{m}_per_head", [])
                for h, v in enumerate(arr[:nh]):
                    col = triple_to_col.get((layer, h, m))
                    if col is not None:
                        X[i, col] = float(v)
        y[i] = r["w4_mse"]
        suites.append(r["suite"])
        groups.append(f"{r['suite']}__{r['task_id']}")
        keep_mask[i] = True

    X = X[keep_mask]
    y = y[keep_mask]
    suites = np.array(suites)
    groups = np.array(groups)
    return X, y, suites, groups, feat_keys, triples


def spearman_all(X, y):
    """Compute Spearman ρ for every column. Returns array of (ρ, raw_p)."""
    from scipy import stats as sps
    n, p = X.shape
    rhos = np.zeros(p)
    raws = np.ones(p)
    stds = np.std(X, axis=0)
    for j in range(p):
        if stds[j] < 1e-12:
            continue
        try:
            rho, pv = sps.spearmanr(X[:, j], y)
            if np.isfinite(rho):
                rhos[j] = rho
                raws[j] = pv
        except Exception:
            pass
    return rhos, raws


def decile_split(X, y, triples, top_features, feat_keys):
    """For each top feature, mean value on top-decile-MSE vs bottom-decile-MSE."""
    p90 = np.quantile(y, 0.9)
    p10 = np.quantile(y, 0.1)
    hi = y >= p90
    lo = y <= p10
    out = []
    for idx, rho, raw, adj in top_features:
        vals_hi = X[hi, idx]
        vals_lo = X[lo, idx]
        out.append({
            "feature": feat_keys[idx],
            "rho": rho,
            "bonferroni_p": adj,
            "mean_hi": float(np.mean(vals_hi)),
            "mean_lo": float(np.mean(vals_lo)),
            "delta_hi_lo": float(np.mean(vals_hi) - np.mean(vals_lo)),
            "n_hi": int(hi.sum()),
            "n_lo": int(lo.sum()),
        })
    return out


def suite_stratified(X, y, suites, groups, triples, top_idxs, feat_keys):
    """Per-suite |ρ| for each top feature to check the effect isn't one-sided."""
    from scipy import stats as sps
    out = []
    for idx in top_idxs:
        entry = {"feature": feat_keys[idx], "col": idx}
        for s in ("Object", "Long"):
            mask = suites == s
            if mask.sum() < 10:
                entry[s] = None
                continue
            col = X[mask, idx]
            if np.std(col) < 1e-12:
                entry[s] = None
                continue
            try:
                rho, p = sps.spearmanr(col, y[mask])
                entry[s] = {"rho": float(rho), "p": float(p), "n": int(mask.sum())}
            except Exception:
                entry[s] = None
        out.append(entry)
    return out


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


def main():
    utils.setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Quant config matching exp7_per_frame__{config}.jsonl")
    p.add_argument("--topk", type=int, default=15)
    args = p.parse_args()

    utils.log("=" * 60)
    utils.log(f"EXP8b — per-head analysis, config={args.config}")
    utils.log("=" * 60)

    exp7_path = os.path.join(utils.RESULTS_DIR, f"exp7_per_frame__{args.config}.jsonl")
    exp5_path = os.path.join(utils.RESULTS_DIR, "exp5_per_call.jsonl")
    utils.log(f"[exp8b] loading {exp7_path}  +  {exp5_path}")
    exp7_records = load_jsonl(exp7_path)
    exp5_per_call = load_jsonl(exp5_path)
    utils.log(f"[exp8b] exp7={len(exp7_records)}  exp5_per_call={len(exp5_per_call)}")

    utils.log("[exp8b] building per-head feature matrix...")
    X, y, suites, groups, feat_keys, triples = build_per_head_matrix(
        exp7_records, exp5_per_call)
    utils.log(f"[exp8b] X={X.shape}  y mean={np.mean(y):.3e}  std={np.std(y):.3e}")

    utils.log("[exp8b] computing per-feature Spearman...")
    rhos, raws = spearman_all(X, y)

    # Bonferroni over the number of non-constant features
    valid = np.std(X, axis=0) > 1e-12
    n_tests = int(valid.sum())
    adj = np.minimum(raws * n_tests, 1.0)
    sig_mask = (adj < 0.05) & valid
    n_sig = int(sig_mask.sum())
    utils.log(f"[exp8b] n_tests={n_tests}  n_bonferroni_sig={n_sig}")

    # Top-k by |ρ| (only Bonferroni-significant ones)
    sig_idx = np.where(sig_mask)[0]
    sig_idx = sig_idx[np.argsort(-np.abs(rhos[sig_idx]))]
    top_idx = sig_idx[:args.topk]

    top_list = [(int(i), float(rhos[i]), float(raws[i]), float(adj[i])) for i in top_idx]
    decile_stats = decile_split(X, y, triples, top_list, feat_keys)
    suite_stats = suite_stratified(X, y, suites, groups, triples,
                                     [i for (i, _, _, _) in top_list], feat_keys)

    # Pattern breakdown across ALL significant features (not just top-k)
    sig_triples = [triples[i] for i in np.where(sig_mask)[0]]
    by_metric = defaultdict(int)
    by_layer = defaultdict(int)
    by_head = defaultdict(int)
    by_layer_group = defaultdict(int)
    for layer, h, m in sig_triples:
        by_metric[m] += 1
        by_head[h] += 1
        by_layer[layer] += 1
        if "vision_tower" in layer:
            by_layer_group["vision_tower"] += 1
        elif "language_model" in layer:
            by_layer_group["language_model"] += 1
        else:
            by_layer_group["other"] += 1

    # --- Tables ---
    lines = [f"# Exp8b — Per-head deep-dive  (config={args.config})\n",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"n_frames: {X.shape[0]}  n_features_tested: {n_tests}  "
             f"**n_Bonferroni_sig: {n_sig}**\n"]

    # Top features
    lines += ["\n## Top Bonferroni-significant (layer, head, metric) features\n", "```",
              fmt_table(
                  ["# ", "layer", "head", "metric", "ρ", "Bonferroni p",
                   "mean@top-decile", "mean@bot-decile", "Δ(hi−lo)"],
                  [[i + 1,
                    short_layer(triples[top_idx[i]][0]),
                    triples[top_idx[i]][1],
                    triples[top_idx[i]][2],
                    f"{rhos[top_idx[i]]:+.3f}",
                    f"{adj[top_idx[i]]:.2e}",
                    f"{d['mean_hi']:.4f}",
                    f"{d['mean_lo']:.4f}",
                    f"{d['delta_hi_lo']:+.4f}"]
                   for i, d in enumerate(decile_stats)],
                  [">", "<", ">", "<", ">", ">", ">", ">", ">"]),
              "```\n"]

    # Suite stratification
    lines += ["\n## Top features — per-suite Spearman ρ (checks not Long-only or Object-only)\n", "```",
              fmt_table(
                  ["# ", "layer", "head", "metric",
                   "all ρ",
                   "Object ρ (n)",
                   "Long ρ (n)"],
                  [[i + 1,
                    short_layer(triples[top_idx[i]][0]),
                    triples[top_idx[i]][1],
                    triples[top_idx[i]][2],
                    f"{rhos[top_idx[i]]:+.3f}",
                    (f"{suite_stats[i]['Object']['rho']:+.3f} ({suite_stats[i]['Object']['n']})"
                     if suite_stats[i].get("Object") else "n/a"),
                    (f"{suite_stats[i]['Long']['rho']:+.3f} ({suite_stats[i]['Long']['n']})"
                     if suite_stats[i].get("Long") else "n/a")]
                   for i in range(len(top_idx))],
                  [">", "<", ">", "<", ">", ">", ">"]),
              "```\n"]

    # Pattern breakdown
    lines += ["\n## Pattern breakdown over ALL Bonferroni-significant features\n"]
    lines.append(f"\n### By metric type")
    rows = [[m, by_metric[m], f"{by_metric[m] / max(n_sig,1) * 100:.1f}%"] for m in METRICS]
    lines += ["```",
              fmt_table(["metric", "n_sig", "fraction"], rows, ["<", ">", ">"]),
              "```"]

    lines.append(f"\n### By VLM component")
    rows = [[comp, by_layer_group[comp],
             f"{by_layer_group[comp] / max(n_sig,1) * 100:.1f}%"]
            for comp in sorted(by_layer_group)]
    lines += ["```",
              fmt_table(["component", "n_sig", "fraction"], rows, ["<", ">", ">"]),
              "```"]

    lines.append(f"\n### By head index (which heads within their layers)")
    rows = [[h, by_head[h]] for h in sorted(by_head)]
    lines += ["```",
              fmt_table(["head_idx", "n_sig"], rows, [">", ">"]),
              "```"]

    lines.append(f"\n### Top 10 layers by Bonferroni-sig feature count")
    top_layers = sorted(by_layer.items(), key=lambda x: -x[1])[:10]
    rows = [[short_layer(l), n] for l, n in top_layers]
    lines += ["```",
              fmt_table(["layer", "n_sig"], rows, ["<", ">"]),
              "```"]

    # Interpretation hints
    lines.append("\n## Interpretation hints\n")
    lines.append("- If top features are dominated by one (layer, head) pair, the signal is mechanistically concentrated; that specific computation is doing quant-relevant work.")
    lines.append("- If top features span many layers but share a metric (all `sparsity` or all `sink`), the signal is about a general attention-distribution property (spread vs concentration), not a specific layer's function.")
    lines.append("- If Object-only ρ is strong but Long-only ρ is near zero, the signal is conditional on task-type and wouldn't be load-bearing as a general controller signal.")
    lines.append("- Δ(hi-lo) gives the effect size in the feature's native units: how far apart are the metric means on high-MSE vs low-MSE frames. Look for |Δ| > 1 std of the feature to say 'meaningful'.")

    out = "\n".join(lines) + "\n"
    out_path = os.path.join(utils.RESULTS_DIR, f"exp8_per_head__{args.config}.md")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out)
    print(out)
    utils.log(f"[exp8b] → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
