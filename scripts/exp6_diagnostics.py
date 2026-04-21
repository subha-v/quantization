#!/usr/bin/env python3
"""
Exp6 diagnostics — proper statistics on existing data before accepting null.

Responding to methodological critique of exp6's "signal dead" verdict:
  1. Bootstrap 95% CIs on R² for suite-baseline vs attention-based regressions.
     The 0.665 vs 0.450 point-estimate gap may not survive CI overlap.
  2. Spearman rank correlation between each of 1350 features and steps_delta,
     with Bonferroni-corrected p-values. Tests for monotonic nonlinear signal.
  3. Random-forest regression compared to Ridge — tests whether nonlinear
     interactions recover signal that ridge misses.
  4. Within-suite target variance diagnostic. If W4-sensitivity has tiny
     within-suite variance, within-suite R² is undefined regardless of features.

Uses existing data — no new rollouts required.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import utils


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_matrix(exp6, exp5, config_name):
    fp_by_idx = {r["rollout_idx"]: r for r in exp5}
    feat_keys = sorted({k for r in exp5 for k in r["features"] if "||" in k})
    rows, y, suites, groups = [], [], [], []
    for r in exp6:
        if r["quant_config"] != config_name: continue
        fp = fp_by_idx.get(r["fp16_rollout_idx"])
        if fp is None: continue
        rows.append([fp["features"].get(k, 0.0) for k in feat_keys])
        y.append(float(r["steps_delta"]))
        suites.append(r["suite"])
        groups.append(f"{r['suite']}__{r['task_id']}")
    return (np.asarray(rows), np.asarray(y), np.asarray(suites),
            np.asarray(groups), feat_keys)


def lotp_folds(groups):
    uniq = sorted(set(groups))
    obj = [g for g in uniq if g.startswith("Object__")]
    lng = [g for g in uniq if g.startswith("Long__")]
    return [(np.where((groups != og) & (groups != lg))[0],
             np.where((groups == og) | (groups == lg))[0])
            for og in obj for lg in lng]


def bootstrap_cv_r2(X, y, groups, model_fn, n_boot=1000, seed=0):
    """Bootstrap 95% CI of cross-validated R² by resampling rollouts.
    model_fn: (X_train_std, y_train, X_test_std) -> predictions."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(seed)
    n = len(y)

    # Get one fold iteration's R² for this exact (X, y, groups) — base estimate
    def one_r2(X_, y_, groups_):
        r2s = []
        for tr, te in lotp_folds(groups_):
            if len(tr) < 4 or len(te) < 2: continue
            if np.std(y_[tr]) < 1e-9 or np.std(y_[te]) < 1e-9: continue
            Xt, Xe = X_[tr], X_[te]
            keep = np.std(Xt, axis=0) > 1e-9
            Xt, Xe = Xt[:, keep], Xe[:, keep]
            if Xt.shape[1] == 0: continue
            scaler = StandardScaler().fit(Xt)
            pred = model_fn(scaler.transform(Xt), y_[tr], scaler.transform(Xe))
            r2s.append(r2_score(y_[te], pred))
        return float(np.mean(r2s)) if r2s else float("nan")

    point_est = one_r2(X, y, groups)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Only keep bootstrap samples that retain >= 2 groups per suite
        gb = groups[idx]
        if len(set(g for g in gb if g.startswith("Object__"))) < 2: continue
        if len(set(g for g in gb if g.startswith("Long__"))) < 2: continue
        r2 = one_r2(X[idx], y[idx], gb)
        if not np.isnan(r2): boots.append(r2)

    boots = np.array(boots)
    return {
        "point": point_est,
        "ci_lower": float(np.percentile(boots, 2.5)) if len(boots) else float("nan"),
        "ci_upper": float(np.percentile(boots, 97.5)) if len(boots) else float("nan"),
        "n_boot": len(boots),
    }


def ridge_fn(alpha):
    from sklearn.linear_model import Ridge
    def fit(Xt, yt, Xe):
        m = Ridge(alpha=alpha); m.fit(Xt, yt); return m.predict(Xe)
    return fit


def rf_fn(n_estimators=200, max_depth=5, random_state=0):
    from sklearn.ensemble import RandomForestRegressor
    def fit(Xt, yt, Xe):
        m = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state, n_jobs=-1)
        m.fit(Xt, yt); return m.predict(Xe)
    return fit


def gb_fn(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0):
    from sklearn.ensemble import GradientBoostingRegressor
    def fit(Xt, yt, Xe):
        m = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                       learning_rate=learning_rate, random_state=random_state)
        m.fit(Xt, yt); return m.predict(Xe)
    return fit


def suite_baseline_fn(suites):
    from sklearn.linear_model import Ridge
    s = np.array([1.0 if x == "Long" else 0.0 for x in suites])
    def fit(Xt, yt, Xe):
        # Ignore Xt; use suite encoding built from pre-scaled X
        # Need to map from fold indices — this is tricky; we'll do it separately
        return np.full(Xe.shape[0], yt.mean())  # constant predictor (fallback)
    return fit


def spearman_features(X, y, feat_keys):
    """Per-feature Spearman ρ with Bonferroni correction."""
    from scipy import stats as sps
    n, p = X.shape
    results = []
    for j in range(p):
        col = X[:, j]
        if np.std(col) < 1e-12: continue
        try:
            rho, p_raw = sps.spearmanr(col, y)
            if not np.isfinite(rho): continue
            results.append((feat_keys[j], float(rho), float(p_raw)))
        except Exception:
            continue
    # Bonferroni: multiply p by # of tests
    n_tests = len(results)
    return sorted([(name, rho, p_raw, min(p_raw * n_tests, 1.0))
                   for name, rho, p_raw in results],
                  key=lambda t: -abs(t[1]))


def suite_baseline_cv(suites, y, groups):
    """Pure suite-label regression with bootstrap CI."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    s = np.array([1.0 if x == "Long" else 0.0 for x in suites]).reshape(-1, 1)
    def _r2(s_, y_, g_):
        r2s = []
        for tr, te in lotp_folds(g_):
            if len(tr) < 4 or len(te) < 2: continue
            if np.std(y_[tr]) < 1e-9 or np.std(y_[te]) < 1e-9: continue
            m = Ridge(alpha=0.01); m.fit(s_[tr], y_[tr])
            r2s.append(r2_score(y_[te], m.predict(s_[te])))
        return float(np.mean(r2s)) if r2s else float("nan")

    rng = np.random.default_rng(0)
    n = len(y)
    point = _r2(s, y, groups)
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, n, size=n)
        gb = groups[idx]
        if len(set(g for g in gb if g.startswith("Object__"))) < 2: continue
        if len(set(g for g in gb if g.startswith("Long__"))) < 2: continue
        r2 = _r2(s[idx], y[idx], gb)
        if not np.isnan(r2): boots.append(r2)
    boots = np.array(boots)
    return {
        "point": point,
        "ci_lower": float(np.percentile(boots, 2.5)) if len(boots) else float("nan"),
        "ci_upper": float(np.percentile(boots, 97.5)) if len(boots) else float("nan"),
        "n_boot": len(boots),
    }


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
    utils.log("=" * 60)
    utils.log("EXP6 DIAGNOSTICS — bootstrap + nonlinear + within-suite variance")
    utils.log("=" * 60)

    exp6 = load_jsonl(os.path.join(utils.RESULTS_DIR, "exp6_per_rollout.jsonl"))
    exp5 = load_jsonl(os.path.join(utils.RESULTS_DIR, "exp5_rollout_summary.jsonl"))
    configs = sorted({r["quant_config"] for r in exp6})

    lines = ["# Exp6 diagnostics — bootstrap, nonlinear, and within-suite variance\n"]

    # --- (A) Within-suite variance of steps_delta ---
    lines += ["\n## (A) Target variance per suite × config\n",
              "If within-suite std of steps_delta is near zero, within-suite R² is **undefined**",
              "regardless of attention content. This diagnoses whether there's anything to predict.\n"]
    rows = []
    for cfg in configs:
        for suite in ("Object", "Long"):
            ys = [r["steps_delta"] for r in exp6 if r["quant_config"] == cfg and r["suite"] == suite]
            if not ys: continue
            ys = np.array(ys)
            rows.append([cfg, suite, len(ys), f"{ys.mean():+.1f}",
                         f"{ys.std():.1f}", f"[{ys.min()}, {ys.max()}]"])
    lines += ["```",
              fmt_table(["config", "suite", "n", "mean", "std", "range"], rows,
                        ["<", "<", ">", ">", ">", ">"]),
              "```\n"]

    # --- (B) Bootstrap CIs per config ---
    lines += ["\n## (B) Bootstrap 95% CIs on R² (1000 resamples, LOTP CV)\n"]
    for cfg in configs:
        X, y, suites, groups, feat_keys = build_matrix(exp6, exp5, cfg)
        if len(y) == 0: continue

        utils.log(f"\n[{cfg}] bootstrap suite baseline...")
        sb = suite_baseline_cv(suites, y, groups)

        utils.log(f"[{cfg}] bootstrap ridge α=1000 top-features...")
        # Need per-fold top-K inside bootstrap — too expensive with 1000 reps.
        # Instead use ridge with strong regularization on full feature set.
        ridge_boot = bootstrap_cv_r2(X, y, groups, ridge_fn(alpha=1000.0), n_boot=500)

        utils.log(f"[{cfg}] bootstrap random forest...")
        rf_boot = bootstrap_cv_r2(X, y, groups, rf_fn(max_depth=4), n_boot=200)

        utils.log(f"[{cfg}] bootstrap gradient boosting...")
        gb_boot = bootstrap_cv_r2(X, y, groups, gb_fn(max_depth=3), n_boot=200)

        lines += [f"\n### Config: {cfg}  (n={len(y)}, y std = {np.std(y):.1f})\n",
                  "```",
                  fmt_table(
                    ["model", "point R²", "95% CI",  "CI width", "n_boot"],
                    [
                      ["suite label (1 feat)",
                       f"{sb['point']:+.3f}",
                       f"[{sb['ci_lower']:+.3f}, {sb['ci_upper']:+.3f}]",
                       f"{sb['ci_upper']-sb['ci_lower']:.3f}",
                       sb["n_boot"]],
                      ["ridge α=1000 (1350 feat)",
                       f"{ridge_boot['point']:+.3f}",
                       f"[{ridge_boot['ci_lower']:+.3f}, {ridge_boot['ci_upper']:+.3f}]",
                       f"{ridge_boot['ci_upper']-ridge_boot['ci_lower']:.3f}",
                       ridge_boot["n_boot"]],
                      ["random forest (d=4)",
                       f"{rf_boot['point']:+.3f}",
                       f"[{rf_boot['ci_lower']:+.3f}, {rf_boot['ci_upper']:+.3f}]",
                       f"{rf_boot['ci_upper']-rf_boot['ci_lower']:.3f}",
                       rf_boot["n_boot"]],
                      ["gradient boost",
                       f"{gb_boot['point']:+.3f}",
                       f"[{gb_boot['ci_lower']:+.3f}, {gb_boot['ci_upper']:+.3f}]",
                       f"{gb_boot['ci_upper']-gb_boot['ci_lower']:.3f}",
                       gb_boot["n_boot"]],
                    ], ["<", ">", ">", ">", ">"]),
                  "```\n"]

        # Does attention CI upper-bound exceed suite CI upper-bound?
        best_attn_upper = max(ridge_boot["ci_upper"], rf_boot["ci_upper"], gb_boot["ci_upper"])
        gap = best_attn_upper - sb["ci_upper"]
        lines.append(f"- Best-attention upper CI: {best_attn_upper:+.3f}  vs  suite upper CI: {sb['ci_upper']:+.3f}  (gap: {gap:+.3f})")

    # --- (C) Spearman per-feature with Bonferroni ---
    lines += ["\n\n## (C) Top Spearman-|ρ| features per config (Bonferroni-corrected)\n",
              "If any feature has Bonferroni-adjusted p < 0.05 after correcting for 1350 tests,",
              "that's evidence of monotonic nonlinear signal that ridge might have missed.\n"]
    for cfg in configs:
        X, y, suites, groups, feat_keys = build_matrix(exp6, exp5, cfg)
        if len(y) == 0: continue
        utils.log(f"[{cfg}] spearman ranks...")
        sp = spearman_features(X, y, feat_keys)
        sig = [t for t in sp if t[3] < 0.05]
        lines += [f"\n### {cfg}\n",
                  f"- Features with Bonferroni-p < 0.05: **{len(sig)} / {len(sp)}**"]
        if sig:
            rows = [[name[:70] + ("…" if len(name) > 70 else ""),
                     f"{rho:+.3f}", f"{p_raw:.2e}", f"{p_adj:.2e}"]
                    for name, rho, p_raw, p_adj in sig[:10]]
            lines += ["```",
                      fmt_table(["feature", "ρ", "raw p", "Bonferroni p"], rows,
                                ["<", ">", ">", ">"]),
                      "```"]
        else:
            lines.append("- _no features survive Bonferroni correction_")
            # Show raw top-5 anyway
            rows = [[name[:70] + ("…" if len(name) > 70 else ""),
                     f"{rho:+.3f}", f"{p_raw:.2e}", f"{p_adj:.2e}"]
                    for name, rho, p_raw, p_adj in sp[:5]]
            lines += ["\nTop-5 by |ρ| (uncorrected, for context):\n```",
                      fmt_table(["feature", "ρ", "raw p", "Bonferroni p"], rows,
                                ["<", ">", ">", ">"]),
                      "```"]

    # --- Verdict ---
    lines.append("\n\n## Revised interpretation\n")
    lines.append("Look at Table (A) first: if within-suite std is small (say, <30 steps) then within-suite")
    lines.append("R² is trying to predict noise. Table (B) shows whether the suite-vs-attention gap survives")
    lines.append("confidence intervals. Table (C) looks for any monotonic signal in individual features.\n")
    lines.append("If all three give null signals → the no-signal verdict is robust at the rollout-summary level.")
    lines.append("If any of them points to signal → the per-frame experiment (exp7) is the right next step anyway.\n")

    out = "\n".join(lines) + "\n"
    out_path = os.path.join(utils.RESULTS_DIR, "exp6_diagnostics.md")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out)
    print(out)
    utils.log(f"[exp6-diag] → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
