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
CHUNK_OVERLAP = 5   # pi0.5 LIBERO replan_steps=5 → 10-action chunk, 5 executed,
                    # consecutive chunks overlap on the last 5 / first 5 actions.


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


def chunk_variance(chunk: np.ndarray) -> float:
    """Signal B: mean of per-DoF variance within a single 10x7 action chunk.
    High variance = the model is unsure how the next 10 actions evolve."""
    if chunk.ndim != 2 or chunk.shape[0] < 2:
        return 0.0
    return float(chunk.var(axis=0).mean())


def chunk_sign_flips(chunk: np.ndarray) -> float:
    """Signal C: fraction of consecutive (timestep, DoF) pairs where sign flips.
    High rate = direction instability. Normalized to [0, 1]."""
    if chunk.ndim != 2 or chunk.shape[0] < 2:
        return 0.0
    s = np.sign(chunk)
    flips = int((s[1:] != s[:-1]).sum())
    total = (chunk.shape[0] - 1) * chunk.shape[1]
    return flips / max(1, total)


def inter_chunk_discrepancy(chunk_t: np.ndarray, chunk_t1: np.ndarray,
                            overlap: int = CHUNK_OVERLAP) -> float:
    """Signal D: L2 distance between the overlap region of two consecutive
    chunks. chunk_t[-overlap:] should match chunk_{t+1}[:overlap] if the model
    is consistent across cycles. Normalized by sqrt(overlap * DoF) so it's a
    per-element RMS magnitude comparable across configs."""
    if chunk_t.ndim != 2 or chunk_t1.ndim != 2:
        return 0.0
    if chunk_t.shape[0] < overlap or chunk_t1.shape[0] < overlap:
        return 0.0
    diff = chunk_t[-overlap:] - chunk_t1[:overlap]
    return float(np.sqrt((diff ** 2).mean()))


def per_cycle_chunk_signals(chunks_for_trial: list) -> dict:
    """For one trial's per-cycle chunk records (sorted by cycle_idx), compute
    Signal B/C per cycle and Signal D per consecutive-cycle pair.
    Returns three lists of per-cycle / per-pair scalars."""
    chunks_for_trial = sorted(chunks_for_trial, key=lambda r: r.get("cycle_idx", 0))
    arrs = []
    for r in chunks_for_trial:
        c = np.asarray(r.get("chunk", []), dtype=np.float32)
        arrs.append(c)
    sig_b = [chunk_variance(c) for c in arrs]
    sig_c = [chunk_sign_flips(c) for c in arrs]
    sig_d = [
        inter_chunk_discrepancy(arrs[i], arrs[i + 1])
        for i in range(len(arrs) - 1)
    ]
    return {"B_var": sig_b, "C_signflip": sig_c, "D_interchunk": sig_d}


def build_features_for_trial(diag_rows: list, chunk_rows: list = None) -> dict:
    """Build a feature dict for one trial from its per-cycle V3 diagnostic
    records. Only deployable signals (W4-pass attention + sis); FP16-pass
    probes are NOT used as features (not available at runtime).

    Also computes oracle features from `mse_fp_w4` for ceiling comparison.

    If `chunk_rows` is provided, additionally computes Phase B chunk-derived
    signals (B = chunk variance, C = within-chunk sign-flip rate, D = inter-
    chunk discrepancy). These are PURE-W4 signals (no FP16 dependency at
    runtime) so they're fair-game deployable features.
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

    # Phase B chunk signals (B=variance, C=sign-flip, D=inter-chunk discrepancy)
    if chunk_rows:
        chunk_signals = per_cycle_chunk_signals(chunk_rows)
        for K in WINDOWS:
            for sig_name, sig_series in chunk_signals.items():
                stats = _stats_over_window(sig_series, K)
                for s in STATS:
                    feats[f"chunk_K{K}__{sig_name}__{s}"] = stats[s]

    return feats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(diag_path: Path, rollouts_path: Path, chunks_path: Path = None):
    """Load and join V3 diagnostic + per-condition rollouts (and chunks if path given).

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

    chunks_by_trial = defaultdict(list)
    if chunks_path is not None and chunks_path.exists():
        for line in chunks_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["suite"], r["task_id"], r["seed"], r["episode_idx"])
            chunks_by_trial[key].append(r)
        utils.log(f"[trialgate] loaded chunk data for {len(chunks_by_trial)} trials "
                  f"from {chunks_path.name}")
    elif chunks_path is not None:
        utils.log(f"[trialgate] WARNING: chunks path {chunks_path} not found; "
                  f"chunk features will be absent")

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
        chunk_rows = chunks_by_trial.get(key) if chunks_by_trial else None
        feats = build_features_for_trial(diag_by_trial[key], chunk_rows=chunk_rows)
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
    p.add_argument("--chunks-tag", default=None,
                   help="suffix for chunk JSONL (expD_chunks__<tag>.jsonl). "
                        "Defaults to --data-tag if a matching file exists. "
                        "When set, Phase B chunk-derived signals (B/C/D) are "
                        "added as a separate feature set in the comparison.")
    args = p.parse_args()

    utils.setup_logging()
    rdir = Path(args.results_dir)
    diag_path = rdir / f"expB_diagnostic_v3__{args.data_tag}.jsonl"
    rollouts_path = rdir / f"expB_w4__{args.data_tag}_rollouts.jsonl"
    out_features = rdir / f"expD_trialgate_features__{args.data_tag}.jsonl"
    out_summary = rdir / f"expD_trialgate_summary__{args.data_tag}.md"

    chunks_tag = args.chunks_tag or args.data_tag
    chunks_path = rdir / f"expD_chunks__{chunks_tag}.jsonl"
    if not chunks_path.exists():
        chunks_path = None

    trials = load_data(diag_path, rollouts_path, chunks_path=chunks_path)

    # Build feature matrix. Four flavors:
    #   - attn_only      : W4-pass attention probes + SIS (Phase A baseline)
    #   - chunks_only    : Phase B B/C/D signals only (Phase B core)
    #   - combined       : attn_only + chunks_only (Phase A+B union)
    #   - oracle         : mse_fp_w4 features (ceiling reference, not deployable)
    feature_names = sorted(trials[0]["features"].keys())
    attn_feats = [f for f in feature_names
                  if not f.startswith("oracle_")
                  and not f.startswith("chunk_")
                  and f != "n_cycles_logged"]
    chunk_feats = [f for f in feature_names if f.startswith("chunk_")]
    oracle_feats = [f for f in feature_names if f.startswith("oracle_")]
    combined_feats = attn_feats + chunk_feats

    X_attn = np.array([[t["features"][f] for f in attn_feats] for t in trials])
    X_chunks = (np.array([[t["features"][f] for f in chunk_feats] for t in trials])
                if chunk_feats else np.empty((len(trials), 0)))
    X_combined = (np.array([[t["features"][f] for f in combined_feats] for t in trials])
                  if chunk_feats else X_attn)
    X_oracle = np.array([[t["features"][f] for f in oracle_feats] for t in trials])
    y_w4_fail = np.array([t["y_w4_fail"] for t in trials])
    y_rescuable = np.array([t["y_rescuable"] for t in trials])

    have_chunks = X_chunks.shape[1] > 0
    utils.log(
        f"[trialgate] feature matrices: attn={X_attn.shape} "
        f"chunks={X_chunks.shape} combined={X_combined.shape} oracle={X_oracle.shape} | "
        f"y_w4_fail pos: {int(y_w4_fail.sum())}/{len(y_w4_fail)}, "
        f"y_rescuable pos: {int(y_rescuable.sum())}/{len(y_rescuable)}"
    )

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
    feature_sets = [("attn", X_attn), ("oracle", X_oracle)]
    if have_chunks:
        feature_sets.extend([("chunks", X_chunks), ("combined", X_combined)])
    for label_name, y in [("y_w4_fail", y_w4_fail), ("y_rescuable", y_rescuable)]:
        for X_name, X in feature_sets:
            if X.shape[1] == 0:
                continue
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

    base_outcomes = {
        "FP16":            np.array([t["outcomes"]["FP16"] for t in trials], dtype=int),
        "W4-Floor":        np.array([t["outcomes"]["W4-Floor"] for t in trials], dtype=int),
        "Random-W4":       np.array([t["outcomes"]["Random-W4"] for t in trials], dtype=int),
        "AttnEntropy-W4":  np.array([t["outcomes"]["AttnEntropy-W4"] for t in trials], dtype=int),
        "S3-Tern-W4-l12h2":np.array([t["outcomes"]["S3-Tern-W4-l12h2"] for t in trials], dtype=int),
    }

    # Iterate over every (label, deployable feature set) pair we've trained.
    sim_feature_sets = ["attn"] + (["chunks", "combined"] if have_chunks else [])
    for label_name in ("y_w4_fail", "y_rescuable"):
        for X_name in sim_feature_sets:
            if (label_name, X_name) not in detectors:
                continue
            det = detectors[(label_name, X_name)]
            y_proba = det["y_proba"]
            for alpha in ALPHAS:
                thr = conformal_threshold(y_w4_fail, y_proba, alpha)
                fire_mask = y_proba > thr
                for rescue_cond in ("AttnEntropy-W4", "S3-Tern-W4-l12h2"):
                    sr, nfire, _ = simulate_gated_sr(trials, fire_mask, "W4-Floor", rescue_cond)
                    sim_rows.append({
                        "label": label_name, "features": X_name, "alpha": alpha,
                        "threshold": thr, "n_fired": int(nfire),
                        "rescue_cond": rescue_cond, "gated_sr": sr,
                    })

    # ---- Per-bucket breakdown of best gated detector ----
    # "Best" = the (label, features, alpha) that maximizes gated AttnEntropy-W4 SR.
    best = max([r for r in sim_rows if r["rescue_cond"] == "AttnEntropy-W4"],
               key=lambda r: r["gated_sr"])
    det = detectors[(best["label"], best["features"])]
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
    lines.append("| Target | Features | n_feats | AUC | 95% CI | Brier |")
    lines.append("|---|---|---:|---:|---|---:|")
    feat_order = ["attn"] + (["chunks", "combined"] if have_chunks else []) + ["oracle"]
    n_feats_by = {
        "attn": X_attn.shape[1],
        "chunks": X_chunks.shape[1],
        "combined": X_combined.shape[1],
        "oracle": X_oracle.shape[1],
    }
    for lbl in ("y_w4_fail", "y_rescuable"):
        for fname in feat_order:
            if (lbl, fname) not in detectors:
                continue
            d = detectors[(lbl, fname)]
            lines.append(
                f"| `{lbl}` | {fname} | {n_feats_by[fname]} | {d['auc']:.3f} | "
                f"[{d['auc_lo']:.3f}, {d['auc_hi']:.3f}] | {d['brier']:.3f} |"
            )

    lines.append("\n## Baselines (no Stage-1 gating)\n")
    lines.append("| Condition | n | SR |")
    lines.append("|---|---:|---:|")
    for cond in ("FP16", "W4-Floor", "Random-W4", "AttnEntropy-W4", "S3-Tern-W4-l12h2"):
        v = base_outcomes[cond]
        lines.append(f"| {cond} | {len(v)} | {v.mean():.3f} |")

    lines.append("\n## Simulated gated SR (per detector)\n")
    lines.append("| Stage-1 target | Features | α (FAR) | thr | n_fired | rescue cond | gated SR |")
    lines.append("|---|---|---:|---:|---:|---|---:|")
    for r in sim_rows:
        lines.append(
            f"| `{r['label']}` | {r['features']} | {r['alpha']:.2f} | {r['threshold']:.3f} | "
            f"{r['n_fired']}/{len(trials)} | {r['rescue_cond']} | {r['gated_sr']:.3f} |"
        )

    lines.append(
        f"\n## Per-bucket breakdown of best gated detector "
        f"(target=`{best['label']}`, features=`{best['features']}`, "
        f"α={best['alpha']:.2f}, thr={best['threshold']:.3f})\n"
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
    # Use the best deployable detector for the headline read.
    best_resc_auc = max(
        [detectors[(t, f)]["auc"]
         for t in ("y_rescuable",) for f in feat_order
         if (t, f) in detectors and f != "oracle"],
        default=0.0,
    )
    aggregate_attn_sr = float(base_outcomes["AttnEntropy-W4"].mean())
    if best_resc_auc > 0.65 and best["gated_sr"] > aggregate_attn_sr:
        lines.append(
            f"**Trial-gate directional pass.** Best deployable detector hits AUC "
            f"{best_resc_auc:.2f} for `y_rescuable` AND simulated gated AttnEntropy-W4 "
            f"({best['gated_sr']:.2f}) beats ungated AttnEntropy-W4 ({aggregate_attn_sr:.2f}). "
            f"Ready to rerun on the n=200 P1 data once that finishes."
        )
    else:
        lines.append(
            f"**Trial-gate weak signal.** Best deployable AUC for `y_rescuable` = "
            f"{best_resc_auc:.2f}; best gated AttnEntropy-W4 SR = {best['gated_sr']:.2f} vs "
            f"ungated {aggregate_attn_sr:.2f}. Stage-1 not strong enough to rescue at this "
            f"sample size. Consider broader feature space, larger n, or new signal sources."
        )

    out_summary.write_text("\n".join(lines) + "\n")
    utils.log(f"[trialgate] wrote {out_summary}")


if __name__ == "__main__":
    main()
