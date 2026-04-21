#!/usr/bin/env python3
"""
Exp5 reanalysis — proper generalization tests that control for task-identity leakage.

The original exp5 run used random 5-fold StratifiedKFold, which allowed the
same (task_id × suite) to appear in both train and test folds — the classifier
probably memorized task-identity. This script re-runs the classifier with:

  1. **Leave-One-Task-Out (LOTO) CV**: for each task_id, hold out all 5 of its
     seeds as test, train on the remaining 4 tasks × 5 seeds. Uses sklearn's
     GroupKFold with group = (suite, task_id).
  2. **Random-seed CV (sanity)**: like original — expected to remain AUC≈1.0.
  3. **Top-K feature reduction** then LOTO: mitigate the 22:1 features-to-samples
     overfitting risk by keeping top-K features per univariate t-stat computed
     ONLY on the training fold (to avoid leakage through feature selection).

Also splits vision-tower features from language-model features to see whether
the signal is carried by image-side attention (potential scene-complexity
artifact) or language-side attention (potential task-semantic signal).

Reads /data/subha2/experiments/results/exp5_rollout_summary.jsonl and writes
results/exp5_reanalysis_tables.md.
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
def load_summary(path):
    rollouts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))
    return rollouts


# ---------------------------------------------------------------------------
def classify(X, y, groups, splitter_name, feature_names, top_k=None):
    """Run CV with the given splitter. Optionally do per-fold top-K feature
    selection via univariate Welch-t (computed only on train).

    Splitters:
      stratified_5fold    — random 5-fold (leaky baseline; same task in train+test)
      leave_one_task_pair — for each (Object task, Long task) pair, hold out ALL
                            5 seeds of each (10 rollouts test, 40 train). 25 folds.
                            This is the correct generalization test for binary
                            suite classification with 5 tasks × 5 seeds.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from scipy import stats as sps

    if splitter_name == "stratified_5fold":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_iter = splitter.split(X, y)
    elif splitter_name == "leave_one_task_pair":
        # Build fold indices manually: for each pair (obj_task, long_task) held out.
        uniq = sorted(set(groups))
        obj_groups = [g for g in uniq if g.startswith("Object__")]
        long_groups = [g for g in uniq if g.startswith("Long__")]
        folds = []
        for og in obj_groups:
            for lg in long_groups:
                te = np.where((groups == og) | (groups == lg))[0]
                tr = np.where((groups != og) & (groups != lg))[0]
                folds.append((tr, te))
        fold_iter = iter(folds)
    else:
        raise ValueError(splitter_name)

    aucs = []
    for tr, te in fold_iter:
        # Optional: per-fold top-K feature selection via univariate t-stat on train only
        if top_k is not None and top_k < X.shape[1]:
            t_stats = np.zeros(X.shape[1])
            y_tr = y[tr]
            for j in range(X.shape[1]):
                col = X[tr, j]
                if np.std(col) < 1e-12:
                    t_stats[j] = 0.0
                    continue
                try:
                    t, _ = sps.ttest_ind(col[y_tr == 1], col[y_tr == 0], equal_var=False)
                    t_stats[j] = abs(float(t)) if np.isfinite(t) else 0.0
                except Exception:
                    t_stats[j] = 0.0
            keep = np.argsort(t_stats)[::-1][:top_k]
            Xtr, Xte = X[tr][:, keep], X[te][:, keep]
        else:
            Xtr, Xte = X[tr], X[te]

        # Drop constant columns
        std = np.std(Xtr, axis=0)
        keep_ok = std > 1e-9
        Xtr = Xtr[:, keep_ok]
        Xte = Xte[:, keep_ok]
        if Xtr.shape[1] == 0:
            continue

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)

        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(Xtr_s, y[tr])
        if len(set(y[te])) > 1:
            prob = clf.predict_proba(Xte_s)[:, 1]
            aucs.append(roc_auc_score(y[te], prob))
    if not aucs:
        return {"mean_auc": 0.5, "std_auc": 0.0, "n_folds": 0, "aucs": []}
    return {
        "mean_auc": float(np.mean(aucs)),
        "std_auc":  float(np.std(aucs)),
        "n_folds":  len(aucs),
        "aucs":     [float(a) for a in aucs],
    }


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


# ---------------------------------------------------------------------------
def main():
    utils.setup_logging()
    summary_path = os.path.join(utils.RESULTS_DIR, "exp5_rollout_summary.jsonl")
    rollouts = load_summary(summary_path)
    utils.log(f"[exp5-reanalysis] loaded {len(rollouts)} rollouts")

    feat_keys = sorted({k for r in rollouts for k in r["features"] if "||" in k})
    X_all = np.array([[r["features"].get(k, 0.0) for k in feat_keys] for r in rollouts],
                     dtype=np.float64)
    y_all = np.array([0 if r["suite"] == "Object" else 1 for r in rollouts])
    # Group by (suite, task_id) so leave-one-task-out holds out ALL seeds of a task.
    groups = np.array([f"{r['suite']}__{r['task_id']}" for r in rollouts])

    utils.log(f"[exp5-reanalysis]   X={X_all.shape} y Easy={int((y_all==0).sum())} "
              f"Hard={int((y_all==1).sum())} unique groups={len(set(groups))}")

    # Feature subsets
    vision_mask = np.array(["vision_tower" in k for k in feat_keys])
    lm_mask     = np.array(["language_model" in k for k in feat_keys])
    static_mask = np.array(["||mean" in k for k in feat_keys])
    dynamic_mask = np.array([any(x in k for x in ("||std", "||max", "||range", "||acorr1"))
                             for k in feat_keys])

    experiments = [
        # (label, feature-subset mask, CV splitter, top_k)
        ("all_features | stratified_5fold",            None,                "stratified_5fold",    None),
        ("all_features | LOTP",                        None,                "leave_one_task_pair",  None),
        ("all_features | LOTP + top-20",               None,                "leave_one_task_pair",  20),
        ("all_features | LOTP + top-5",                None,                "leave_one_task_pair",  5),
        ("vision_tower only | LOTP",                   vision_mask,         "leave_one_task_pair",  None),
        ("language_model only | LOTP",                 lm_mask,             "leave_one_task_pair",  None),
        ("static only | LOTP",                         static_mask,         "leave_one_task_pair",  None),
        ("dynamic only | LOTP",                        dynamic_mask,        "leave_one_task_pair",  None),
        ("vision_tower + static | LOTP",               vision_mask & static_mask, "leave_one_task_pair",  None),
        ("language_model + static | LOTP",             lm_mask & static_mask,     "leave_one_task_pair",  None),
        ("language_model + dynamic | LOTP",            lm_mask & dynamic_mask,    "leave_one_task_pair",  None),
    ]

    rows = []
    for label, mask, splitter, top_k in experiments:
        X = X_all if mask is None else X_all[:, mask]
        if X.shape[1] == 0:
            rows.append([label, "0", "skipped (no features)", "-", "-"])
            continue
        res = classify(X, y_all, groups, splitter, [], top_k=top_k)
        rows.append([
            label,
            X.shape[1] if top_k is None else f"{X.shape[1]}→{top_k}",
            f"{res['mean_auc']:.3f} ± {res['std_auc']:.3f}",
            res['n_folds'],
            f"[{min(res['aucs']):.2f},{max(res['aucs']):.2f}]" if res['aucs'] else "-",
        ])
        utils.log(f"[exp5-reanalysis] {label}: "
                  f"AUC={res['mean_auc']:.3f}±{res['std_auc']:.3f} over {res['n_folds']} folds")

    # Also: sanity — what's the rollout-length confound?
    # Compute mean_steps correlation with suite.
    steps_easy = [r["steps"] for r in rollouts if r["suite"] == "Object"]
    steps_hard = [r["steps"] for r in rollouts if r["suite"] == "Long"]
    calls_easy = [r["n_calls"] for r in rollouts if r["suite"] == "Object"]
    calls_hard = [r["n_calls"] for r in rollouts if r["suite"] == "Long"]

    # Write tables
    lines = ["# Exp5 — Reanalysis (generalization + confound controls)\n",
             f"n_rollouts: {len(rollouts)}\n"]

    lines += ["## Rollout-level confound check\n",
              "The suites differ systematically in rollout LENGTH and VLM-CALL COUNT — any",
              "feature that leaks these quantities will produce perfect classification",
              "regardless of attention content.\n",
              "```",
              fmt_table(["metric", "Object mean", "Long mean", "delta"],
                        [
                          ["rollout steps",   f"{np.mean(steps_easy):.1f}", f"{np.mean(steps_hard):.1f}",
                           f"+{np.mean(steps_hard)-np.mean(steps_easy):.1f}"],
                          ["vlm calls",       f"{np.mean(calls_easy):.1f}", f"{np.mean(calls_hard):.1f}",
                           f"+{np.mean(calls_hard)-np.mean(calls_easy):.1f}"],
                        ], ["<", ">", ">", ">"]),
              "```\n"]

    lines += ["## Classifier AUC by split & feature subset\n",
              "`stratified_5fold` is the *leaky* baseline (same tasks in train & test).",
              "`LOTO` (leave-one-task-out) is the real test: train on 4 tasks per suite,",
              "test on the 5 seeds of the held-out task. Any AUC substantially above 0.5",
              "under LOTO reflects genuine cross-task generalization.\n",
              "```",
              fmt_table(["experiment", "features", "AUC ± std", "folds", "per-fold range"],
                        rows, ["<", ">", ">", ">", "<"]),
              "```\n"]

    lines.append("\n## Interpretation guide\n")
    lines.append("- `all_features | stratified_5fold` ≈ `1.000` → task-identity leakage confirmed.")
    lines.append("- `all_features | LOTP` = the generalization ceiling. If this remains high,")
    lines.append("  attention features carry difficulty signal beyond task-identity.")
    lines.append("- `vision_tower only | LOTP` vs `language_model only | LOTP` → where the")
    lines.append("  signal lives. Vision carries scene-level info (potentially a proxy for")
    lines.append("  'number of objects') while language carries task-semantic info.")

    out = "\n".join(lines) + "\n"
    out_path = os.path.join(utils.RESULTS_DIR, "exp5_reanalysis_tables.md")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out)
    print(out)
    utils.log(f"[exp5-reanalysis] tables → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
