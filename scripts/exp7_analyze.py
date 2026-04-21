#!/usr/bin/env python3
"""
Exp7 analysis — per-frame attention features vs per-frame W4 action MSE.

Joins per-call attention features from exp5_per_call.jsonl with per-call W4 MSE
from exp7_per_frame__w4_both.jsonl on (rollout_idx, call_idx). Builds a (n, p)
matrix where n ≈ 1500 per-frame samples and p ≈ 225 features (after aggregating
the 45 layers × 5 metrics per-call features).

Runs:
  1. Target variance diagnostic: per-suite distribution of w4_mse
  2. Spearman rank correlation per feature with Bonferroni correction
  3. Ridge regression R² under LOTP CV + bootstrap 95% CI
  4. Random-forest R² under LOTP CV (point estimate)
  5. Within-suite ridge R² (removes suite-level confound)

Writes results/exp7_analysis.md.
"""

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


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Build per-frame feature matrix
# ---------------------------------------------------------------------------
def build_per_frame_matrix(exp7_records, exp5_per_call_records):
    """Join (rollout_idx, call_idx) between attention and MSE."""
    # Index exp5 per-call records by (rollout_idx, call_idx) → list of layer records
    attn_by_key = defaultdict(list)
    for r in exp5_per_call_records:
        key = (r["rollout_idx"], r["call_idx"])
        attn_by_key[key].append(r)

    # For each (rollout_idx, call_idx), aggregate per-layer, per-head → per-layer head-avg
    rows = []
    y = []
    suites = []
    groups = []
    rollout_idxs = []
    for r in exp7_records:
        key = (r["rollout_idx"], r["call_idx"])
        if key not in attn_by_key:
            continue
        layer_records = attn_by_key[key]
        # Aggregate per-layer: average across heads
        feats = {}
        for lr in layer_records:
            layer = lr["layer"]
            for metric in ("sparsity", "entropy", "top1", "top5", "sink"):
                key_name = f"{layer}||{metric}"
                vals = lr.get(f"{metric}_per_head", [])
                if vals:
                    feats[key_name] = float(np.mean(vals))
        rows.append(feats)
        y.append(r["w4_mse"])
        suites.append(r["suite"])
        groups.append(f"{r['suite']}__{r['task_id']}")
        rollout_idxs.append(r["rollout_idx"])

    if not rows:
        return None, None, None, None, None, None

    # Union of feature keys across all rows
    all_keys = sorted({k for row in rows for k in row})
    X = np.array([[row.get(k, 0.0) for k in all_keys] for row in rows],
                 dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    suites = np.array(suites)
    groups = np.array(groups)
    rollout_idxs = np.array(rollout_idxs)
    return X, y, suites, groups, rollout_idxs, all_keys


# ---------------------------------------------------------------------------
# LOTP folds (hold out one Object task + one Long task)
# ---------------------------------------------------------------------------
def lotp_folds(groups):
    uniq = sorted(set(groups))
    obj = [g for g in uniq if g.startswith("Object__")]
    lng = [g for g in uniq if g.startswith("Long__")]
    return [(np.where((groups != og) & (groups != lg))[0],
             np.where((groups == og) | (groups == lg))[0])
            for og in obj for lg in lng]


def within_suite_lotp_folds(groups, suite_prefix):
    uniq = sorted(set(groups))
    suite_groups = [g for g in uniq if g.startswith(suite_prefix)]
    folds = []
    for held in suite_groups:
        te = np.where(groups == held)[0]
        # Only samples in this suite
        suite_mask = np.array([g.startswith(suite_prefix) for g in groups])
        tr = np.where((groups != held) & suite_mask)[0]
        if len(tr) > 0 and len(te) > 0:
            folds.append((tr, te))
    return folds


# ---------------------------------------------------------------------------
# CV functions
# ---------------------------------------------------------------------------
def cv_r2(X, y, folds, model_fn):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    r2s = []
    for tr, te in folds:
        if len(tr) < 10 or len(te) < 5:
            continue
        if np.std(y[tr]) < 1e-12 or np.std(y[te]) < 1e-12:
            continue
        Xt, Xe = X[tr], X[te]
        keep = np.std(Xt, axis=0) > 1e-9
        Xt, Xe = Xt[:, keep], Xe[:, keep]
        if Xt.shape[1] == 0:
            continue
        scaler = StandardScaler().fit(Xt)
        pred = model_fn(scaler.transform(Xt), y[tr], scaler.transform(Xe))
        r2s.append(r2_score(y[te], pred))
    return {
        "mean": float(np.mean(r2s)) if r2s else float("nan"),
        "std": float(np.std(r2s)) if r2s else 0.0,
        "n": len(r2s),
        "scores": [float(s) for s in r2s],
    }


def ridge_fn(alpha):
    from sklearn.linear_model import Ridge
    def fit(Xt, yt, Xe):
        m = Ridge(alpha=alpha); m.fit(Xt, yt); return m.predict(Xe)
    return fit


def rf_fn(n_estimators=100, max_depth=6, random_state=0):
    from sklearn.ensemble import RandomForestRegressor
    def fit(Xt, yt, Xe):
        m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state, n_jobs=2)
        m.fit(Xt, yt); return m.predict(Xe)
    return fit


def suite_baseline_r2(suites, y, folds):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    s = np.array([1.0 if x == "Long" else 0.0 for x in suites]).reshape(-1, 1)
    r2s = []
    for tr, te in folds:
        if len(tr) < 10 or len(te) < 5:
            continue
        if np.std(y[tr]) < 1e-12 or np.std(y[te]) < 1e-12:
            continue
        m = Ridge(alpha=0.01); m.fit(s[tr], y[tr])
        r2s.append(r2_score(y[te], m.predict(s[te])))
    return {"mean": float(np.mean(r2s)) if r2s else float("nan"),
            "std": float(np.std(r2s)) if r2s else 0.0,
            "n": len(r2s)}


def bootstrap_ci(X, y, groups, model_fn, n_boot=100, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gb = groups[idx]
        if len(set(g for g in gb if g.startswith("Object__"))) < 2:
            continue
        if len(set(g for g in gb if g.startswith("Long__"))) < 2:
            continue
        folds = lotp_folds(gb)
        res = cv_r2(X[idx], y[idx], folds, model_fn)
        if not np.isnan(res["mean"]):
            boots.append(res["mean"])
    boots = np.array(boots)
    return {
        "ci_lower": float(np.percentile(boots, 2.5)) if len(boots) else float("nan"),
        "ci_upper": float(np.percentile(boots, 97.5)) if len(boots) else float("nan"),
        "n_boot": len(boots),
    }


def spearman_features(X, y, feat_keys):
    from scipy import stats as sps
    n, p = X.shape
    results = []
    for j in range(p):
        col = X[:, j]
        if np.std(col) < 1e-12:
            continue
        try:
            rho, p_raw = sps.spearmanr(col, y)
            if not np.isfinite(rho):
                continue
            results.append((feat_keys[j], float(rho), float(p_raw)))
        except Exception:
            continue
    n_tests = len(results)
    return sorted([(name, rho, p_raw, min(p_raw * n_tests, 1.0))
                   for name, rho, p_raw in results],
                  key=lambda t: -abs(t[1]))


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


# ---------------------------------------------------------------------------
def main():
    import argparse
    utils.setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="w4_both")
    args = p.parse_args()

    exp7_path = os.path.join(utils.RESULTS_DIR, f"exp7_per_frame__{args.config}.jsonl")
    exp5_path = os.path.join(utils.RESULTS_DIR, "exp5_per_call.jsonl")
    utils.log(f"[exp7-analyze] loading {exp7_path} + {exp5_path}")
    exp7_records = load_jsonl(exp7_path)
    exp5_records = load_jsonl(exp5_path)
    utils.log(f"[exp7-analyze] exp7={len(exp7_records)}  exp5_per_call={len(exp5_records)}")

    X, y, suites, groups, idxs, feat_keys = build_per_frame_matrix(exp7_records, exp5_records)
    if X is None:
        utils.log("[exp7-analyze] No overlap between exp7 and exp5 per-call records")
        return 1
    utils.log(f"[exp7-analyze] X={X.shape}  y std={np.std(y):.4e}  Object={int(sum(suites=='Object'))}  Long={int(sum(suites=='Long'))}")

    lines = [f"# Exp7 analysis — per-frame attention → per-frame W4 MSE (config={args.config})\n",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"n_frames: {len(y)}, n_features: {X.shape[1]}\n"]

    # --- (A) Target variance ---
    lines += ["\n## (A) Per-frame W4 MSE distribution\n"]
    rows = []
    rows.append(["all", len(y), f"{np.mean(y):.3e}", f"{np.std(y):.3e}",
                 f"{np.median(y):.3e}", f"{np.min(y):.3e}", f"{np.max(y):.3e}"])
    for s in ("Object", "Long"):
        mask = suites == s
        if mask.sum() == 0: continue
        ys = y[mask]
        rows.append([s, int(mask.sum()),
                     f"{np.mean(ys):.3e}", f"{np.std(ys):.3e}",
                     f"{np.median(ys):.3e}", f"{np.min(ys):.3e}", f"{np.max(ys):.3e}"])
    lines += ["```",
              fmt_table(["suite", "n", "mean", "std", "median", "min", "max"], rows,
                        ["<", ">", ">", ">", ">", ">", ">"]),
              "```\n"]

    # --- (B) LOTP CV comparison ---
    folds = lotp_folds(groups)
    lines += [f"\n## (B) LOTP CV R² ({len(folds)} folds)\n"]

    utils.log("[exp7-analyze] suite baseline...")
    sb = suite_baseline_r2(suites, y, folds)

    utils.log("[exp7-analyze] ridge α=100...")
    ridge_100 = cv_r2(X, y, folds, ridge_fn(alpha=100.0))

    utils.log("[exp7-analyze] ridge α=1000...")
    ridge_1000 = cv_r2(X, y, folds, ridge_fn(alpha=1000.0))

    utils.log("[exp7-analyze] random forest...")
    rf = cv_r2(X, y, folds, rf_fn(max_depth=6))

    utils.log("[exp7-analyze] bootstrap CI on ridge α=1000 (100 reps)...")
    ridge_ci = bootstrap_ci(X, y, groups, ridge_fn(alpha=1000.0), n_boot=100)

    rows = [
        ["suite-label (1 feat)",
         f"{sb['mean']:+.3f} ± {sb['std']:.3f}", "—", sb["n"]],
        ["ridge α=100 (all feat)",
         f"{ridge_100['mean']:+.3f} ± {ridge_100['std']:.3f}", "—", ridge_100["n"]],
        ["ridge α=1000 (all feat)",
         f"{ridge_1000['mean']:+.3f} ± {ridge_1000['std']:.3f}",
         f"[{ridge_ci['ci_lower']:+.3f}, {ridge_ci['ci_upper']:+.3f}]",
         ridge_1000["n"]],
        ["random forest (d=6)",
         f"{rf['mean']:+.3f} ± {rf['std']:.3f}", "—", rf["n"]],
    ]
    lines += ["```",
              fmt_table(["model", "R² (mean ± std)", "95% CI", "n_folds"], rows,
                        ["<", ">", ">", ">"]),
              "```\n"]

    # --- (C) Within-suite R² ---
    lines += ["\n## (C) Within-suite R² (removes suite confound)\n"]
    rows = []
    for suite_name, prefix in [("Object", "Object__"), ("Long", "Long__")]:
        mask = np.array([g.startswith(prefix) for g in groups])
        if mask.sum() == 0: continue
        sub_folds = within_suite_lotp_folds(groups, prefix)
        # Remap indices to sub-matrix
        local_idx = np.where(mask)[0]
        local_map = {int(i): j for j, i in enumerate(local_idx)}
        X_s = X[mask]; y_s = y[mask]
        local_folds = []
        for tr, te in sub_folds:
            tr_l = [local_map[int(i)] for i in tr if int(i) in local_map]
            te_l = [local_map[int(i)] for i in te if int(i) in local_map]
            if tr_l and te_l:
                local_folds.append((np.array(tr_l), np.array(te_l)))
        utils.log(f"[exp7-analyze] within {suite_name} ridge α=1000...")
        rd = cv_r2(X_s, y_s, local_folds, ridge_fn(alpha=1000.0))
        utils.log(f"[exp7-analyze] within {suite_name} random forest...")
        rf_s = cv_r2(X_s, y_s, local_folds, rf_fn(max_depth=6))
        rows.append([suite_name, int(mask.sum()), f"{np.std(y_s):.2e}",
                     f"{rd['mean']:+.3f} ± {rd['std']:.3f}",
                     f"{rf_s['mean']:+.3f} ± {rf_s['std']:.3f}",
                     rd["n"]])
    lines += ["```",
              fmt_table(["suite", "n_frames", "y std",
                         "ridge α=1000", "random forest",
                         "n_folds"],
                        rows, ["<", ">", ">", ">", ">", ">"]),
              "```\n"]

    # --- (D) Spearman per feature with Bonferroni ---
    utils.log("[exp7-analyze] Spearman ranks...")
    sp = spearman_features(X, y, feat_keys)
    sig = [t for t in sp if t[3] < 0.05]
    lines += ["\n## (D) Top Spearman-|ρ| features (Bonferroni-corrected)\n",
              f"- Features with Bonferroni-p < 0.05: **{len(sig)} / {len(sp)}**\n"]
    rows = [[(name[:70] + "…") if len(name) > 70 else name,
             f"{rho:+.3f}", f"{p_raw:.2e}", f"{p_adj:.2e}"]
            for name, rho, p_raw, p_adj in sp[:15]]
    lines += ["```",
              fmt_table(["feature", "ρ", "raw p", "Bonferroni p"], rows,
                        ["<", ">", ">", ">"]),
              "```\n"]

    # --- Verdict ---
    best_attn = max(ridge_100["mean"], ridge_1000["mean"], rf["mean"])
    lines.append("\n## Verdict\n")
    lines.append(f"Best attention-based R² (LOTP): **{best_attn:+.3f}**")
    lines.append(f"Suite-baseline R² (LOTP): **{sb['mean']:+.3f}**")
    lines.append(f"Best within-suite R² across configs: check table (C)\n")
    if best_attn > sb["mean"] + 0.1 and best_attn > 0.2:
        lines.append("**Attention features carry per-frame quantization-sensitivity signal.**")
        lines.append("The per-rollout null (exp6) was aggregation/sample-size artifact.")
        lines.append("Adaptive-controller direction revived at per-frame granularity.")
    elif best_attn > sb["mean"] + 0.05 and len(sig) > 0:
        lines.append("**Weak but present signal.** Some features survive Bonferroni at per-frame scale.")
        lines.append("Real effect but modest; not a load-bearing paper claim.")
    else:
        lines.append("**Null robustly confirmed at per-frame scale.** With n≈1500 and direct target,")
        lines.append("attention features do not predict quantization sensitivity beyond suite identity.")
        lines.append("Adaptive-controller direction definitively dead; static-schedule paper is the story.")

    out = "\n".join(lines) + "\n"
    out_path = os.path.join(utils.RESULTS_DIR, f"exp7_analysis__{args.config}.md")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out)
    print(out)
    utils.log(f"[exp7-analyze] → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
