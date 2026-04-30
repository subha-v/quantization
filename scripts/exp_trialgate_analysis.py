#!/usr/bin/env python3
"""
Phase A trial-gate analysis (ExpD): Stage-1 detector that, given the first K
cycles of a W4-base diagnostic rollout, predicts whether the trial is heading
toward W4 failure. Without Stage-1, AttnEntropy-W4 fires unconditionally and
pays a -35 pp clean-bucket cost that washes out its +20 pp rescuable-bucket
win in aggregate. With Stage-1 working, AttnEntropy fires only on rollouts
that need it.

Inputs (read from /data/subha2/experiments/results/ on remote, or locally):
  - expB_diagnostic_v3__<tag>.jsonl    (per-cycle metrics, full W4 trajectory)
  - expB_w4__<tag>_rollouts.jsonl      (per-condition outcomes per trial)

Outputs:
  - results/expD_trialgate_features__<tag>.jsonl  (feature matrix + labels)
  - results/expD_trialgate_summary__<tag>.md      (auto-generated writeup)

Re-runnable on n=50 (default) and n=200 once P1 finishes.
"""

import argparse
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

RESULTS_DIR = Path(utils.RESULTS_DIR)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
PROBE_KEYS = ("l1h7-top1", "l9h2-ent", "l12h2-ent", "l3h4-top5", "l17h4-top1")
WINDOWS = (5, 10, 15)
STATS = ("mean", "std", "max", "min", "slope")


def _stats_over_window(values: list, K: int) -> dict:
    """Compute summary stats over the first K elements of `values`."""
    arr = np.asarray(values[:K], dtype=np.float64)
    if arr.size == 0:
        return {s: 0.0 for s in STATS}
    out = {
        "mean": float(arr.mean()),
        "std": float(arr.std()) if arr.size > 1 else 0.0,
        "max": float(arr.max()),
        "min": float(arr.min()),
    }
    if arr.size > 1:
        # slope = linear fit of cycle_idx (0..K-1) vs value
        xs = np.arange(arr.size, dtype=np.float64)
        out["slope"] = float(np.polyfit(xs, arr, 1)[0])
    else:
        out["slope"] = 0.0
    return out


def build_features_for_trial(diag_rows: list) -> dict:
    """Build a feature dict for one trial from its per-cycle V3 diagnostic
    records. Only deployable signals (W4-pass attention + sis); FP16-pass
    probes are NOT used as features (not available at runtime).

    Also computes oracle features from `mse_fp_w4` for ceiling comparison.
    """
    diag_rows = sorted(diag_rows, key=lambda r: r.get("cycle_idx", 0))

    # Pull per-cycle scalar series. Some trials may have shorter rollouts; we
    # take whatever cycles we have up to K. Window stats handle short sequences.
    series = {p: [] for p in PROBE_KEYS}
    series["sis"] = []
    series["mse_fp_w4"] = []
    for r in diag_rows:
        probes_w4 = r.get("attn_probes_w4") or {}
        for p in PROBE_KEYS:
            v = probes_w4.get(p)
            if v is not None:
                series[p].append(v)
        s = r.get("sis")
        if s is not None and not (isinstance(s, float) and np.isnan(s)):
            series["sis"].append(s)
        m = r.get("mse_fp_w4")
        if m is not None and not (isinstance(m, float) and np.isnan(m)):
            series["mse_fp_w4"].append(m)

    feats = {}
    deployable_metrics = list(PROBE_KEYS) + ["sis"]
    oracle_metrics = ["mse_fp_w4"]

    for K in WINDOWS:
        for metric in deployable_metrics:
            stats = _stats_over_window(series[metric], K)
            for s in STATS:
                feats[f"K{K}__{metric}__{s}"] = stats[s]
        for metric in oracle_metrics:
            stats = _stats_over_window(series[metric], K)
            for s in STATS:
                feats[f"oracle_K{K}__{metric}__{s}"] = stats[s]

    feats["n_cycles_logged"] = len(diag_rows)
    return feats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(diag_path: Path, rollouts_path: Path):
    """Load and join V3 diagnostic + per-condition rollouts.

    Returns:
      trials: list of dicts, one per (suite, task_id, seed, episode_idx) trial,
              each with: trial_key, features, outcomes (dict of condition -> bool)
    """
    if not diag_path.exists():
        raise FileNotFoundError(f"diagnostic not found: {diag_path}")
    if not rollouts_path.exists():
        raise FileNotFoundError(f"rollouts not found: {rollouts_path}")

    diag_by_trial = defaultdict(list)
    for line in diag_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["suite"], r["task_id"], r["seed"], r["episode_idx"])
        diag_by_trial[key].append(r)

    outcomes_by_trial = defaultdict(dict)
    for line in rollouts_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["suite"], r["task_id"], r["seed"], r["episode_idx"])
        outcomes_by_trial[key][r["condition"]] = bool(r["success"])

    REQUIRED_CONDS = ("FP16", "W4-Floor", "Random-W4", "AttnEntropy-W4", "S3-Tern-W4-l12h2")
    trials = []
    for key in sorted(diag_by_trial):
        outs = outcomes_by_trial.get(key, {})
        if not all(c in outs for c in REQUIRED_CONDS):
            continue
        feats = build_features_for_trial(diag_by_trial[key])
        trials.append({
            "trial_key": list(key),
            "features": feats,
            "outcomes": outs,
            "y_w4_fail": int(not outs["W4-Floor"]),
            "y_rescuable": int(outs["FP16"] and not outs["W4-Floor"]),
        })

    utils.log(f"[trialgate] loaded {len(trials)} trials with all 5 conditions present")
    if not trials:
        raise RuntimeError("no trials with complete condition set; check data tag")
    return trials


# ---------------------------------------------------------------------------
# LOOCV ridge classifier
# ---------------------------------------------------------------------------
def loocv_predict(X: np.ndarray, y: np.ndarray, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """Return (y_proba, alpha_used) via LOOCV. Inner CV picks alpha by mean
    log-loss on a 5-fold split of the 49-trial training set; test fold is the
    held-out trial."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold

    n = len(y)
    y_proba = np.zeros(n)
    alphas_used = []

    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        test_idx = np.array([i])

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]

        # Inner CV to pick C = 1/alpha. Cap K based on positive class count.
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        kK = min(5, n_pos, n_neg)
        if kK < 2:
            best_C = 1.0
        else:
            best_C = 1.0
            best_loss = np.inf
            skf = StratifiedKFold(n_splits=kK, shuffle=True, random_state=0)
            for alpha in alphas:
                C = 1.0 / max(alpha, 1e-12)
                losses = []
                for tr2, va2 in skf.split(X_tr, y_tr):
                    sc = StandardScaler().fit(X_tr[tr2])
                    Xa = sc.transform(X_tr[tr2])
                    Xb = sc.transform(X_tr[va2])
                    clf = LogisticRegression(
                        C=C, penalty="l2", solver="lbfgs",
                        max_iter=2000, class_weight="balanced",
                    )
                    clf.fit(Xa, y_tr[tr2])
                    p = clf.predict_proba(Xb)[:, 1]
                    losses.append(log_loss(y_tr[va2], p, labels=[0, 1]))
                m = np.mean(losses)
                if m < best_loss:
                    best_loss = m
                    best_C = C
        alphas_used.append(1.0 / best_C)

        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(
            C=best_C, penalty="l2", solver="lbfgs",
            max_iter=2000, class_weight="balanced",
        )
        clf.fit(sc.transform(X_tr), y_tr)
        y_proba[i] = clf.predict_proba(sc.transform(X_te))[0, 1]

    return y_proba, alphas_used


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auc_with_bootstrap_ci(y_true, y_proba, n_boot=2000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            aucs.append(roc_auc_score(y_true[idx], y_proba[idx]))
        except ValueError:
            continue
    if not aucs:
        return float("nan"), float("nan"), float("nan")
    point = roc_auc_score(y_true, y_proba)
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def conformal_threshold(y_true, y_proba, alpha: float):
    """FIPER-style: calibrate threshold on success-only subset (y_true == 0).
    Pick threshold so the false-alarm rate (firing on a truly-OK trial) <= alpha.
    Returns the threshold; predictions firing iff p > threshold.
    """
    success_proba = y_proba[y_true == 0]
    if len(success_proba) == 0:
        return 1.0  # no data to calibrate; never fire
    # quantile such that fraction above is alpha
    return float(np.quantile(success_proba, 1.0 - alpha))


def mcnemar_p(b, c):
    """Two-sided exact McNemar p-value. b = A_only, c = B_only."""
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2.0
    return min(1.0, p)


# ---------------------------------------------------------------------------
# Simulated downstream evaluation
# ---------------------------------------------------------------------------
def simulate_gated_sr(trials, fire_mask, base_cond: str, rescue_cond: str):
    """Simulate the gated detector: if fire_mask[i] is True, use rescue_cond's
    outcome on trial i; else use base_cond's outcome.
    Returns (gated_sr, n_fired, gated_outcomes_list)."""
    sr_total = 0
    n_fired = 0
    out = []
    for i, t in enumerate(trials):
        fire = bool(fire_mask[i])
        cond_used = rescue_cond if fire else base_cond
        succ = bool(t["outcomes"][cond_used])
        if fire:
            n_fired += 1
        out.append((fire, cond_used, succ))
        sr_total += int(succ)
    return sr_total / len(trials), n_fired, out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-tag", default="libero_pro_obj_x0.2",
                   help="suffix matching expB_diagnostic_v3__<tag>.jsonl + "
                        "expB_w4__<tag>_rollouts.jsonl")
    p.add_argument("--results-dir", default=str(RESULTS_DIR),
                   help="overrides results dir if reading/writing elsewhere")
    args = p.parse_args()

    utils.setup_logging()
    rdir = Path(args.results_dir)
    diag_path = rdir / f"expB_diagnostic_v3__{args.data_tag}.jsonl"
    rollouts_path = rdir / f"expB_w4__{args.data_tag}_rollouts.jsonl"
    out_features = rdir / f"expD_trialgate_features__{args.data_tag}.jsonl"
    out_summary = rdir / f"expD_trialgate_summary__{args.data_tag}.md"

    trials = load_data(diag_path, rollouts_path)

    # Build feature matrix. Two flavors: deployable (W4-pass + sis only) and
    # oracle (also includes mse_fp_w4 features as a ceiling reference).
    feature_names = sorted(trials[0]["features"].keys())
    deployable_feats = [f for f in feature_names if not f.startswith("oracle_")
                        and f != "n_cycles_logged"]
    oracle_feats = [f for f in feature_names if f.startswith("oracle_")]

    X_dep = np.array([[t["features"][f] for f in deployable_feats] for t in trials])
    X_oracle = np.array([[t["features"][f] for f in oracle_feats] for t in trials])
    y_w4_fail = np.array([t["y_w4_fail"] for t in trials])
    y_rescuable = np.array([t["y_rescuable"] for t in trials])

    utils.log(f"[trialgate] X_deployable shape: {X_dep.shape}, "
              f"y_w4_fail pos: {int(y_w4_fail.sum())}/{len(y_w4_fail)}, "
              f"y_rescuable pos: {int(y_rescuable.sum())}/{len(y_rescuable)}")

    # Persist features for inspection
    with open(out_features, "w") as fh:
        for t in trials:
            fh.write(json.dumps({
                "trial_key": t["trial_key"],
                "y_w4_fail": t["y_w4_fail"],
                "y_rescuable": t["y_rescuable"],
                "outcomes": t["outcomes"],
                "features": t["features"],
            }) + "\n")
    utils.log(f"[trialgate] wrote {out_features}")

    # ---- LOOCV detectors ----
    detectors = {}
    for label_name, y in [("y_w4_fail", y_w4_fail), ("y_rescuable", y_rescuable)]:
        for X_name, X in [("deployable", X_dep), ("oracle", X_oracle)]:
            utils.log(f"[trialgate] LOOCV {label_name} | {X_name} | X.shape={X.shape}")
            y_proba, _ = loocv_predict(X, y)
            auc, lo, hi = auc_with_bootstrap_ci(y, y_proba)
            from sklearn.metrics import brier_score_loss
            brier = brier_score_loss(y, y_proba)
            detectors[(label_name, X_name)] = {
                "y_proba": y_proba, "auc": auc, "auc_lo": lo, "auc_hi": hi,
                "brier": brier,
            }

    # ---- Simulated downstream SR for each detector + threshold ----
    ALPHAS = (0.05, 0.10, 0.20)
    sim_rows = []
    headline_dep = detectors[("y_w4_fail", "deployable")]
    headline_resc = detectors[("y_rescuable", "deployable")]

    base_outcomes = {
        "FP16":            np.array([t["outcomes"]["FP16"] for t in trials], dtype=int),
        "W4-Floor":        np.array([t["outcomes"]["W4-Floor"] for t in trials], dtype=int),
        "Random-W4":       np.array([t["outcomes"]["Random-W4"] for t in trials], dtype=int),
        "AttnEntropy-W4":  np.array([t["outcomes"]["AttnEntropy-W4"] for t in trials], dtype=int),
        "S3-Tern-W4-l12h2":np.array([t["outcomes"]["S3-Tern-W4-l12h2"] for t in trials], dtype=int),
    }

    for label_name in ("y_w4_fail", "y_rescuable"):
        det = detectors[(label_name, "deployable")]
        y_proba = det["y_proba"]
        for alpha in ALPHAS:
            thr = conformal_threshold(y_w4_fail, y_proba, alpha)
            fire_mask = y_proba > thr
            for rescue_cond in ("AttnEntropy-W4", "S3-Tern-W4-l12h2"):
                sr, nfire, _ = simulate_gated_sr(trials, fire_mask, "W4-Floor", rescue_cond)
                sim_rows.append({
                    "label": label_name, "alpha": alpha, "threshold": thr,
                    "n_fired": int(nfire),
                    "rescue_cond": rescue_cond, "gated_sr": sr,
                })

    # ---- Per-bucket breakdown of best gated detector ----
    # "Best" = the (label, alpha) that maximizes gated AttnEntropy-W4 SR.
    best = max([r for r in sim_rows if r["rescue_cond"] == "AttnEntropy-W4"],
               key=lambda r: r["gated_sr"])
    label = best["label"]; alpha = best["alpha"]
    det = detectors[(label, "deployable")]
    fire_mask = det["y_proba"] > best["threshold"]

    buckets = {"clean": [], "rescuable": [], "w4_better": [], "unrescuable": []}
    for i, t in enumerate(trials):
        fp = t["outcomes"]["FP16"]; w4 = t["outcomes"]["W4-Floor"]
        if fp and w4: buckets["clean"].append(i)
        elif fp:      buckets["rescuable"].append(i)
        elif w4:      buckets["w4_better"].append(i)
        else:         buckets["unrescuable"].append(i)

    bucket_table = []
    for bname, idxs in buckets.items():
        n = len(idxs)
        if n == 0: continue
        n_fire = int(fire_mask[idxs].sum())
        # Gated SR on this bucket = AttnEntropy when fired, W4-Floor when not
        gated = sum(
            int(t["outcomes"]["AttnEntropy-W4"] if fire_mask[i] else t["outcomes"]["W4-Floor"])
            for i, t in [(i, trials[i]) for i in idxs]
        )
        bucket_table.append({
            "bucket": bname, "n": n, "n_fired": n_fire,
            "fire_rate": n_fire / n,
            "gated_sr": gated / n,
        })

    # ---- McNemar: gated AttnEntropy-W4 vs ungated AttnEntropy-W4 ----
    a_attn = base_outcomes["AttnEntropy-W4"]
    a_gated = np.array([
        int(t["outcomes"]["AttnEntropy-W4"] if fire_mask[i] else t["outcomes"]["W4-Floor"])
        for i, t in enumerate(trials)
    ])
    a_w4 = base_outcomes["W4-Floor"]
    a_random = base_outcomes["Random-W4"]

    def _mc(a, b, name_a, name_b):
        a_only = int(((a == 1) & (b == 0)).sum())
        b_only = int(((a == 0) & (b == 1)).sum())
        return {
            "comparison": f"{name_a} vs {name_b}",
            "a_only": a_only, "b_only": b_only,
            "delta": (int(a.sum()) - int(b.sum())) / len(a),
            "p_mcnemar": mcnemar_p(a_only, b_only),
        }

    mc_rows = [
        _mc(a_gated, a_attn, "Gated-AttnEnt", "AttnEnt-W4"),
        _mc(a_gated, a_w4, "Gated-AttnEnt", "W4-Floor"),
        _mc(a_gated, a_random, "Gated-AttnEnt", "Random-W4"),
    ]

    # ---- Write summary markdown ----
    lines = []
    lines.append(f"# ExpD Trial-Gate Analysis — Phase A ({args.data_tag})\n")
    lines.append(f"_n trials = {len(trials)}_\n")

    lines.append("## Stage-1 detector AUC (LOOCV ridge LR, 95% bootstrap CI)\n")
    lines.append("| Target | Features | AUC | 95% CI | Brier |")
    lines.append("|---|---|---:|---|---:|")
    for label in ("y_w4_fail", "y_rescuable"):
        for fname in ("deployable", "oracle"):
            d = detectors[(label, fname)]
            lines.append(
                f"| `{label}` | {fname} | {d['auc']:.3f} | "
                f"[{d['auc_lo']:.3f}, {d['auc_hi']:.3f}] | {d['brier']:.3f} |"
            )

    lines.append("\n## Baselines (no Stage-1 gating)\n")
    lines.append("| Condition | n | SR |")
    lines.append("|---|---:|---:|")
    for cond in ("FP16", "W4-Floor", "Random-W4", "AttnEntropy-W4", "S3-Tern-W4-l12h2"):
        v = base_outcomes[cond]
        lines.append(f"| {cond} | {len(v)} | {v.mean():.3f} |")

    lines.append("\n## Simulated gated SR (deployable detector)\n")
    lines.append("| Stage-1 target | α (FAR) | thr | n_fired | rescue cond | gated SR |")
    lines.append("|---|---:|---:|---:|---|---:|")
    for r in sim_rows:
        lines.append(
            f"| `{r['label']}` | {r['alpha']:.2f} | {r['threshold']:.3f} | "
            f"{r['n_fired']}/{len(trials)} | {r['rescue_cond']} | {r['gated_sr']:.3f} |"
        )

    lines.append(
        f"\n## Per-bucket breakdown of best gated detector "
        f"(target=`{best['label']}`, α={best['alpha']:.2f}, "
        f"thr={best['threshold']:.3f})\n"
    )
    lines.append("| Bucket | n | n_fired | fire rate | gated SR |")
    lines.append("|---|---:|---:|---:|---:|")
    for b in bucket_table:
        lines.append(
            f"| {b['bucket']} | {b['n']} | {b['n_fired']} | "
            f"{b['fire_rate']:.0%} | {b['gated_sr']:.0%} |"
        )

    lines.append("\n## Matched-pair McNemar — gated AttnEntropy-W4 vs others\n")
    lines.append("| Comparison | a_only | b_only | Δ SR | McNemar p |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in mc_rows:
        lines.append(
            f"| {r['comparison']} | {r['a_only']} | {r['b_only']} | "
            f"{r['delta']:+.3f} | {r['p_mcnemar']:.3f} |"
        )

    lines.append("\n## Read\n")
    if headline_resc["auc"] > 0.65 and best["gated_sr"] > base_outcomes["AttnEntropy-W4"].mean():
        lines.append("**Phase A directional pass.** Stage-1 carries detectable signal "
                     "(AUC > 0.65 for at least one target) AND gated AttnEntropy-W4 "
                     "beats ungated AttnEntropy-W4 in aggregate SR. "
                     "Ready to rerun on the n=200 P1 data once that finishes.")
    else:
        lines.append("**Phase A weak signal.** Stage-1 AUC is too low or the simulated "
                     "gated SR fails to beat ungated AttnEntropy-W4. Consider Phase B "
                     "(action-chunk signals B/C/D) before scaling to n=200.")

    out_summary.write_text("\n".join(lines) + "\n")
    utils.log(f"[trialgate] wrote {out_summary}")


if __name__ == "__main__":
    main()
