#!/usr/bin/env python3
"""
Exp6 reanalysis — clean regression tests to disambiguate overfitting from no-signal.

The initial exp6 run showed R² = -1.6 on w4_both steps_delta under LOTP CV with
Ridge(alpha=10) on 1350 features from 40 training samples per fold. That's
almost guaranteed to be overfitting, not true no-signal.

This reanalysis tries:
  1. Heavy regularization: Ridge alpha ∈ {10, 100, 1000, 10_000}.
  2. Per-fold univariate feature selection (top-K on training fold only).
  3. Baseline: suite-only regression (1 feature: Long=1, Object=0). Anything
     attention-based must beat this to claim signal.
  4. Within-suite R²: does attention predict steps_delta AMONG Object rollouts
     only, AMONG Long rollouts only? This removes the suite-level confound.

Reads /data/subha2/experiments/results/exp6_per_rollout.jsonl and
/data/subha2/experiments/results/exp5_rollout_summary.jsonl.
Writes results/exp6_reanalysis_tables.md.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import utils


# ---------------------------------------------------------------------------
def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
def build_matrix(exp6_records, exp5_rollouts, config_name):
    """Build (X, y, suites, groups, rollout_idxs) for the given quant config.
    X: (n, 1350) attention features from FP16 rollouts.
    y: steps_delta (continuous).
    suites: "Object" or "Long".
    groups: suite__task_id (for LOTP CV).
    """
    fp_by_idx = {r["rollout_idx"]: r for r in exp5_rollouts}
    feat_keys = sorted({k for r in exp5_rollouts for k in r["features"] if "||" in k})
    rows, y, suites, groups, idxs = [], [], [], [], []
    for r in exp6_records:
        if r["quant_config"] != config_name:
            continue
        fp = fp_by_idx.get(r["fp16_rollout_idx"])
        if fp is None:
            continue
        rows.append([fp["features"].get(k, 0.0) for k in feat_keys])
        y.append(float(r["steps_delta"]))
        suites.append(r["suite"])
        groups.append(f"{r['suite']}__{r['task_id']}")
        idxs.append(r["fp16_rollout_idx"])
    return (np.asarray(rows, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(suites), np.asarray(groups), np.asarray(idxs),
            feat_keys)


def lotp_folds(groups):
    uniq = sorted(set(groups))
    obj = [g for g in uniq if g.startswith("Object__")]
    lng = [g for g in uniq if g.startswith("Long__")]
    folds = []
    for og in obj:
        for lg in lng:
            te = np.where((groups == og) | (groups == lg))[0]
            tr = np.where((groups != og) & (groups != lg))[0]
            folds.append((tr, te))
    return folds


def lotp_single_suite_folds(groups, suite_prefix):
    """Leave-one-task-out CV within a single suite."""
    suite_groups = [g for g in sorted(set(groups)) if g.startswith(suite_prefix)]
    folds = []
    for held in suite_groups:
        te = np.where(groups == held)[0]
        tr = np.where((groups != held) & np.array([g.startswith(suite_prefix) for g in groups]))[0]
        folds.append((tr, te))
    return folds


def cv_ridge(X, y, folds, alpha, top_k=None):
    """Ridge regression over CV folds with optional per-fold univariate top-K."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from scipy import stats as sps

    r2s = []
    for tr, te in folds:
        if len(tr) < 4 or len(te) < 2:
            continue
        if np.std(y[tr]) < 1e-9 or np.std(y[te]) < 1e-9:
            continue
        Xtr, Xte = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # Per-fold feature selection via univariate |corr(x, y)| on train
        if top_k is not None and top_k < Xtr.shape[1]:
            corrs = np.zeros(Xtr.shape[1])
            y_c = y_tr - y_tr.mean()
            for j in range(Xtr.shape[1]):
                col = Xtr[:, j]
                if np.std(col) < 1e-12:
                    corrs[j] = 0.0
                    continue
                col_c = col - col.mean()
                denom = np.sqrt((col_c ** 2).sum() * (y_c ** 2).sum())
                corrs[j] = abs((col_c * y_c).sum() / (denom + 1e-12))
            keep = np.argsort(corrs)[::-1][:top_k]
            Xtr, Xte = Xtr[:, keep], Xte[:, keep]

        # Drop constants
        std = np.std(Xtr, axis=0)
        ok = std > 1e-9
        Xtr, Xte = Xtr[:, ok], Xte[:, ok]
        if Xtr.shape[1] == 0:
            continue

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        reg = Ridge(alpha=alpha)
        reg.fit(Xtr_s, y_tr)
        pred = reg.predict(Xte_s)
        r2s.append(r2_score(y_te, pred))

    if not r2s:
        return {"mean": float("nan"), "std": 0.0, "n": 0}
    return {"mean": float(np.mean(r2s)), "std": float(np.std(r2s)), "n": len(r2s)}


def cv_suite_baseline(suites, y, folds):
    """Baseline predictor: predict y based on suite label alone."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    s_num = np.array([1.0 if s == "Long" else 0.0 for s in suites]).reshape(-1, 1)
    r2s = []
    for tr, te in folds:
        if len(tr) < 4 or len(te) < 2:
            continue
        if np.std(y[tr]) < 1e-9 or np.std(y[te]) < 1e-9:
            continue
        reg = Ridge(alpha=1.0)
        reg.fit(s_num[tr], y[tr])
        r2s.append(r2_score(y[te], reg.predict(s_num[te])))
    return ({"mean": float(np.mean(r2s)), "std": float(np.std(r2s)), "n": len(r2s)}
            if r2s else {"mean": float("nan"), "n": 0})


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


def fmt_r2(r):
    if not r or r.get("n", 0) == 0 or np.isnan(r.get("mean", float("nan"))):
        return "n/a"
    return f"{r['mean']:+.3f} ± {r['std']:.3f} (n={r['n']})"


# ---------------------------------------------------------------------------
def analyze_config(exp6, exp5, cfg):
    X, y, suites, groups, idxs, _ = build_matrix(exp6, exp5, cfg)
    folds = lotp_folds(groups)

    out = {"config": cfg, "n": len(y),
           "y_mean": float(np.mean(y)) if len(y) else 0.0,
           "y_std":  float(np.std(y))  if len(y) else 0.0,
           "folds": len(folds)}

    # Suite baseline (essentially asks: does just knowing Object/Long predict Δsteps?)
    out["suite_baseline"] = cv_suite_baseline(suites, y, folds)

    # Attention regression with varying regularization + top-K
    for alpha in (10.0, 100.0, 1000.0, 10000.0):
        for top_k in (None, 20, 5):
            key = f"alpha={alpha}_topk={top_k}"
            out[f"attn_{key}"] = cv_ridge(X, y, folds, alpha=alpha, top_k=top_k)

    # Within-suite analyses (removes suite-level confound)
    within = {}
    for suite_name, suite_prefix in [("Object", "Object__"), ("Long", "Long__")]:
        suite_folds = lotp_single_suite_folds(groups, suite_prefix)
        mask = np.array([g.startswith(suite_prefix) for g in groups])
        if mask.sum() == 0 or len(suite_folds) == 0:
            continue
        X_s, y_s = X[mask], y[mask]
        # Remap fold indices to the sub-matrix
        local_folds = []
        orig_idx = np.where(mask)[0]
        local_map = {g: i for i, g in enumerate(orig_idx)}
        for tr, te in suite_folds:
            tr_local = [local_map[i] for i in tr if i in local_map]
            te_local = [local_map[i] for i in te if i in local_map]
            if tr_local and te_local:
                local_folds.append((np.array(tr_local), np.array(te_local)))
        within[suite_name] = {
            "n": int(mask.sum()),
            "y_std": float(np.std(y_s)),
            "attn_alpha100_topk20": cv_ridge(X_s, y_s, local_folds,
                                              alpha=100.0, top_k=20),
            "attn_alpha1000_topk5":  cv_ridge(X_s, y_s, local_folds,
                                              alpha=1000.0, top_k=5),
        }
    out["within_suite"] = within

    return out


# ---------------------------------------------------------------------------
def main():
    utils.setup_logging()
    exp6 = load_jsonl(os.path.join(utils.RESULTS_DIR, "exp6_per_rollout.jsonl"))
    exp5 = load_jsonl(os.path.join(utils.RESULTS_DIR, "exp5_rollout_summary.jsonl"))
    utils.log(f"[exp6-reanalysis] exp6={len(exp6)}  exp5={len(exp5)}")

    configs = sorted({r["quant_config"] for r in exp6})
    utils.log(f"[exp6-reanalysis] configs: {configs}")

    results = [analyze_config(exp6, exp5, c) for c in configs]

    # ---- Tables ----
    lines = ["# Exp6 — Reanalysis (overfitting vs no-signal disambiguation)\n"]

    # Table 1: baseline comparisons per config
    lines += ["\n## Table 1 — Suite-only baseline vs attention-based R²\n",
              "If attention-based R² is **not higher than** the suite baseline, attention adds",
              "no information beyond knowing which suite the rollout came from.\n",
              "If both are near 0 or negative, the signal is absent or drowned in noise.\n"]
    rows = []
    for r in results:
        rows.append([
            r["config"],
            r["n"],
            f"{r['y_std']:.1f}",
            fmt_r2(r["suite_baseline"]),
            fmt_r2(r["attn_alpha=10.0_topk=None"]),
            fmt_r2(r["attn_alpha=100.0_topk=None"]),
            fmt_r2(r["attn_alpha=1000.0_topk=None"]),
            fmt_r2(r["attn_alpha=10000.0_topk=None"]),
        ])
    lines += ["```",
              fmt_table(
                  ["config", "n", "y_std", "suite baseline R²",
                   "α=10", "α=100", "α=1000", "α=10k"],
                  rows, ["<", ">", ">", ">", ">", ">", ">", ">"]),
              "```\n"]

    # Table 2: feature-selected
    lines += ["\n## Table 2 — Attention R² with per-fold univariate feature selection\n",
              "Drops from 1350 features to top-20 and top-5 (selected on training fold only).\n",
              "Fewer features = less overfitting. If *still* negative, the univariate signal is noise.\n"]
    rows = []
    for r in results:
        rows.append([
            r["config"],
            fmt_r2(r["suite_baseline"]),
            fmt_r2(r["attn_alpha=100.0_topk=20"]),
            fmt_r2(r["attn_alpha=1000.0_topk=5"]),
        ])
    lines += ["```",
              fmt_table(["config", "suite baseline", "α=100, top-20", "α=1000, top-5"],
                        rows, ["<", ">", ">", ">"]),
              "```\n"]

    # Table 3: within-suite
    lines += ["\n## Table 3 — Within-suite R² (suite-confound removed)\n",
              "LOTP over tasks within a single suite. If attention captured something other than",
              "suite identity, it should predict within-suite step-count variation.\n"]
    rows = []
    for r in results:
        w = r.get("within_suite", {})
        for suite in ("Object", "Long"):
            if suite not in w: continue
            rows.append([
                r["config"], suite, w[suite]["n"],
                f"{w[suite]['y_std']:.1f}",
                fmt_r2(w[suite]["attn_alpha100_topk20"]),
                fmt_r2(w[suite]["attn_alpha1000_topk5"]),
            ])
    lines += ["```",
              fmt_table(["config", "suite", "n", "y_std",
                         "α=100, top-20", "α=1000, top-5"],
                        rows, ["<", "<", ">", ">", ">", ">"]),
              "```\n"]

    # ---- Verdict ----
    # Pick the best attention R² across all settings, compare to suite baseline.
    best_attn = -999
    best_suite = -999
    for r in results:
        bs = r["suite_baseline"].get("mean", -999) if r["suite_baseline"] else -999
        if bs is not None and not np.isnan(bs):
            best_suite = max(best_suite, bs)
        for k, v in r.items():
            if not k.startswith("attn_"): continue
            if not isinstance(v, dict): continue
            m = v.get("mean", float("nan"))
            if m is not None and not np.isnan(m):
                best_attn = max(best_attn, m)

    lines.append("\n## Verdict\n")
    lines.append(f"Best suite-baseline R² (across configs): **{best_suite:.3f}**")
    lines.append(f"Best attention-based R² (across configs, regularizers, top-K): **{best_attn:.3f}**\n")
    if best_attn - best_suite > 0.15:
        lines.append("**Attention carries information beyond suite identity.** Build the controller.")
    elif best_attn > best_suite + 0.05:
        lines.append("**Weak advantage.** Attention modestly improves over suite-only baseline.")
    elif abs(best_attn - best_suite) <= 0.05:
        lines.append("**No additional signal.** Attention-based R² ≈ suite-baseline R². "
                     "Knowing Object-vs-Long is ≈ all the information the attention features carry.")
    else:
        lines.append("**Attention hurts.** Even with heavy regularization, attention-based R² is lower")
        lines.append("than a 1-feature suite classifier. Features are noise w.r.t. quant sensitivity.")

    out = "\n".join(lines) + "\n"
    out_path = os.path.join(utils.RESULTS_DIR, "exp6_reanalysis_tables.md")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out)
    print(out)
    utils.log(f"[exp6-reanalysis] tables → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
