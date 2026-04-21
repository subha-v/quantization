#!/usr/bin/env python3
"""
Exp8a — Cross-config comparison of per-frame attention → W4 MSE predictability.

Runs the same regression pipeline as exp7_analyze.py against multiple
per-frame MSE datasets produced under different quantization configs:
  - w4_both         (baseline — mixed VLM+expert error, exp7 original)
  - w4_vlm          (VLM-side only — isolates attention-visible error)
  - w4_expert       (expert-side only — attention can't see this a priori)
  - w2_vlm_protect  (stronger VLM quantization with exp2-protected layers)

Emits a comparative table + cross-config MSE correlations.

Key diagnostic: if w4_vlm R² > w4_both R² by a meaningful margin, the exp7
ceiling was driven by mixing attention-invisible expert error into the target.
If w2_vlm_protect R² > w4_vlm R², stronger quantization increases the signal.
If none beats ~0.15 within-suite, the 0.03 ceiling is real.
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

CONFIGS = ("w4_both", "w4_vlm", "w4_expert", "w2_vlm_protect")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_matrix(exp7_records, exp5_per_call):
    attn_by_key = defaultdict(list)
    for r in exp5_per_call:
        attn_by_key[(r["rollout_idx"], r["call_idx"])].append(r)

    rows, y, suites, groups = [], [], [], []
    for r in exp7_records:
        key = (r["rollout_idx"], r["call_idx"])
        if key not in attn_by_key:
            continue
        layer_records = attn_by_key[key]
        feats = {}
        for lr in layer_records:
            layer = lr["layer"]
            for metric in ("sparsity", "entropy", "top1", "top5", "sink"):
                vals = lr.get(f"{metric}_per_head", [])
                if vals:
                    feats[f"{layer}||{metric}"] = float(np.mean(vals))
        rows.append(feats)
        y.append(r["w4_mse"])
        suites.append(r["suite"])
        groups.append(f"{r['suite']}__{r['task_id']}")

    if not rows:
        return None, None, None, None, None
    all_keys = sorted({k for row in rows for k in row})
    X = np.array([[row.get(k, 0.0) for k in all_keys] for row in rows],
                 dtype=np.float64)
    return X, np.array(y), np.array(suites), np.array(groups), all_keys


def lotp_folds(groups):
    uniq = sorted(set(groups))
    obj = [g for g in uniq if g.startswith("Object__")]
    lng = [g for g in uniq if g.startswith("Long__")]
    return [(np.where((groups != og) & (groups != lg))[0],
             np.where((groups == og) | (groups == lg))[0])
            for og in obj for lg in lng]


def within_suite_folds(groups, suite_prefix):
    uniq = sorted(set(groups))
    suite_grps = [g for g in uniq if g.startswith(suite_prefix)]
    folds = []
    for held in suite_grps:
        te = np.where(groups == held)[0]
        mask = np.array([g.startswith(suite_prefix) for g in groups])
        tr = np.where((groups != held) & mask)[0]
        if len(tr) > 0 and len(te) > 0:
            folds.append((tr, te))
    return folds


def cv_r2(X, y, folds, model_fn):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    r2s = []
    for tr, te in folds:
        if len(tr) < 10 or len(te) < 5: continue
        if np.std(y[tr]) < 1e-12 or np.std(y[te]) < 1e-12: continue
        Xt, Xe = X[tr], X[te]
        keep = np.std(Xt, axis=0) > 1e-9
        Xt, Xe = Xt[:, keep], Xe[:, keep]
        if Xt.shape[1] == 0: continue
        sc = StandardScaler().fit(Xt)
        pred = model_fn(sc.transform(Xt), y[tr], sc.transform(Xe))
        r2s.append(r2_score(y[te], pred))
    return {"mean": float(np.mean(r2s)) if r2s else float("nan"),
            "std": float(np.std(r2s)) if r2s else 0.0,
            "n": len(r2s)}


def ridge_fn(alpha):
    from sklearn.linear_model import Ridge
    def fit(Xt, yt, Xe):
        m = Ridge(alpha=alpha); m.fit(Xt, yt); return m.predict(Xe)
    return fit


def suite_baseline_r2(suites, y, folds):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    s = np.array([1.0 if x == "Long" else 0.0 for x in suites]).reshape(-1, 1)
    r2s = []
    for tr, te in folds:
        if len(tr) < 10 or len(te) < 5: continue
        if np.std(y[tr]) < 1e-12 or np.std(y[te]) < 1e-12: continue
        m = Ridge(alpha=0.01); m.fit(s[tr], y[tr])
        r2s.append(r2_score(y[te], m.predict(s[te])))
    return {"mean": float(np.mean(r2s)) if r2s else float("nan"),
            "std": float(np.std(r2s)) if r2s else 0.0,
            "n": len(r2s)}


def spearman_n_bonferroni_sig(X, y, feat_keys, alpha=0.05):
    from scipy import stats as sps
    hits = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.std(col) < 1e-12: continue
        try:
            rho, p_raw = sps.spearmanr(col, y)
            if not np.isfinite(rho): continue
            hits.append((feat_keys[j], float(rho), float(p_raw)))
        except Exception:
            continue
    n_tests = len(hits)
    adj = [(n, r, p, min(p * n_tests, 1.0)) for n, r, p in hits]
    sig = [t for t in adj if t[3] < alpha]
    return len(sig), sorted(sig, key=lambda t: -abs(t[1]))[:10], n_tests


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


def fmt_r2(r):
    if r is None or r.get("n", 0) == 0 or np.isnan(r.get("mean", float("nan"))):
        return "n/a"
    return f"{r['mean']:+.3f} ± {r['std']:.3f}"


def main():
    utils.setup_logging()
    utils.log("=" * 60)
    utils.log("EXP8a — Cross-config comparison")
    utils.log("=" * 60)

    exp5 = load_jsonl(os.path.join(utils.RESULTS_DIR, "exp5_per_call.jsonl"))
    utils.log(f"[exp8a] exp5 per-call records: {len(exp5)}")

    results = {}
    per_frame_mse_by_config = {}  # for cross-config correlations

    for cfg in CONFIGS:
        path = os.path.join(utils.RESULTS_DIR, f"exp7_per_frame__{cfg}.jsonl")
        if not os.path.exists(path):
            utils.log(f"[exp8a] SKIP {cfg}: {path} missing")
            continue
        exp7 = load_jsonl(path)
        utils.log(f"\n[exp8a] {cfg}: {len(exp7)} per-frame records")

        X, y, suites, groups, feat_keys = build_matrix(exp7, exp5)
        if X is None:
            utils.log(f"[exp8a]   no overlap; skip")
            continue
        utils.log(f"[exp8a]   X={X.shape}  y mean={np.mean(y):.3e}  std={np.std(y):.3e}")

        # Stash per-frame y keyed by (rollout_idx, call_idx) for cross-config correlation
        per_frame_mse_by_config[cfg] = {(r["rollout_idx"], r["call_idx"]): r["w4_mse"]
                                         for r in exp7}

        folds = lotp_folds(groups)
        utils.log(f"[exp8a]   LOTP folds: {len(folds)}  suite baseline...")
        sb = suite_baseline_r2(suites, y, folds)
        utils.log(f"[exp8a]   ridge α=1000...")
        rd = cv_r2(X, y, folds, ridge_fn(alpha=1000.0))

        # Within-Object and within-Long
        within = {}
        for sname, sprefix in [("Object", "Object__"), ("Long", "Long__")]:
            mask = np.array([g.startswith(sprefix) for g in groups])
            if mask.sum() == 0: continue
            sub_folds = within_suite_folds(groups, sprefix)
            local_idx = np.where(mask)[0]
            local_map = {int(i): j for j, i in enumerate(local_idx)}
            X_s, y_s = X[mask], y[mask]
            local_folds = []
            for tr, te in sub_folds:
                tr_l = [local_map[int(i)] for i in tr if int(i) in local_map]
                te_l = [local_map[int(i)] for i in te if int(i) in local_map]
                if tr_l and te_l:
                    local_folds.append((np.array(tr_l), np.array(te_l)))
            utils.log(f"[exp8a]   within-{sname} ridge...")
            within[sname] = cv_r2(X_s, y_s, local_folds, ridge_fn(alpha=1000.0))

        utils.log(f"[exp8a]   Spearman + Bonferroni...")
        n_sig, top_feats, n_tests = spearman_n_bonferroni_sig(X, y, feat_keys)

        results[cfg] = {
            "n_frames": len(y), "n_features": X.shape[1],
            "y_mean": float(np.mean(y)), "y_std": float(np.std(y)),
            "suite_r2": sb, "ridge_r2": rd,
            "within_object_r2": within.get("Object", {"mean": float("nan"), "std": 0, "n": 0}),
            "within_long_r2": within.get("Long", {"mean": float("nan"), "std": 0, "n": 0}),
            "n_bonferroni_sig": n_sig, "n_tests": n_tests,
            "top_features": top_feats,
        }

    # ---- Cross-config per-frame MSE correlations ----
    corr_matrix = {}
    common_cfgs = list(per_frame_mse_by_config.keys())
    for i, a in enumerate(common_cfgs):
        for b in common_cfgs[i+1:]:
            keys_a = set(per_frame_mse_by_config[a])
            keys_b = set(per_frame_mse_by_config[b])
            common = keys_a & keys_b
            if len(common) < 100:
                continue
            common = sorted(common)
            ya = np.array([per_frame_mse_by_config[a][k] for k in common])
            yb = np.array([per_frame_mse_by_config[b][k] for k in common])
            from scipy import stats as sps
            pearson, pp = sps.pearsonr(ya, yb)
            spearman, sp_p = sps.spearmanr(ya, yb)
            corr_matrix[(a, b)] = {
                "n_common": len(common),
                "pearson": float(pearson), "pearson_p": float(pp),
                "spearman": float(spearman), "spearman_p": float(sp_p),
            }

    # ---- Tables ----
    lines = ["# Exp8a — Cross-config comparison of per-frame attention → W4 MSE\n",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"]

    lines += ["\n## Table 1 — Per-config R² and Bonferroni-sig feature counts\n",
              "```",
              fmt_table(
                  ["config", "n", "y_std", "suite R²", "ridge R²",
                   "within-Obj R²", "within-Long R²", "n_sig / n_tests"],
                  [[cfg, r["n_frames"], f"{r['y_std']:.3e}",
                    fmt_r2(r["suite_r2"]), fmt_r2(r["ridge_r2"]),
                    fmt_r2(r["within_object_r2"]),
                    fmt_r2(r["within_long_r2"]),
                    f"{r['n_bonferroni_sig']} / {r['n_tests']}"]
                   for cfg, r in results.items()],
                  ["<", ">", ">", ">", ">", ">", ">", ">"]),
              "```\n"]

    if corr_matrix:
        lines += ["\n## Table 2 — Per-frame MSE correlation between configs\n",
                  "If w4_vlm and w4_expert MSEs are highly correlated, the same frames are",
                  "sensitive to both kinds of error and decoupling is less informative.\n",
                  "```",
                  fmt_table(
                      ["config A", "config B", "n_common", "Pearson r", "Spearman ρ"],
                      [[a, b, d["n_common"],
                        f"{d['pearson']:+.3f}", f"{d['spearman']:+.3f}"]
                       for (a, b), d in corr_matrix.items()],
                      ["<", "<", ">", ">", ">"]),
                  "```\n"]

    # Top features per config
    for cfg, r in results.items():
        if not r["top_features"]:
            continue
        lines += [f"\n## Top Bonferroni-significant features — {cfg}\n", "```",
                  fmt_table(
                      ["feature", "ρ", "raw p", "Bonferroni p"],
                      [[(n[:70] + "…") if len(n) > 70 else n,
                        f"{rho:+.3f}", f"{p:.2e}", f"{adj:.2e}"]
                       for n, rho, p, adj in r["top_features"]],
                      ["<", ">", ">", ">"]),
                  "```\n"]

    # Verdict
    lines.append("\n## Verdict\n")
    if "w4_vlm" in results and "w4_both" in results:
        rv = results["w4_vlm"].get("within_object_r2", {}).get("mean", float("nan"))
        rb = results["w4_both"].get("within_object_r2", {}).get("mean", float("nan"))
        lines.append(f"- within-Object R²: w4_vlm={rv:+.3f}  w4_both={rb:+.3f}  gap={rv-rb:+.3f}")
        if rv - rb > 0.10:
            lines.append("- **D3 HIT: decoupling helped.** exp7 ceiling was an unfair-target artifact.")
        elif rv - rb > 0.03:
            lines.append("- **D3 MILD: decoupling helped a little.** Most of the ceiling is real.")
        else:
            lines.append("- **D3 NULL: decoupling didn't help.** The 0.03 ceiling is a real property of the attention features, not an artifact of mixing expert error into the target.")

    if "w2_vlm_protect" in results and "w4_vlm" in results:
        r2 = results["w2_vlm_protect"].get("within_object_r2", {}).get("mean", float("nan"))
        r4 = results["w4_vlm"].get("within_object_r2", {}).get("mean", float("nan"))
        lines.append(f"- within-Object R²: w2_vlm_protect={r2:+.3f}  w4_vlm={r4:+.3f}  gap={r2-r4:+.3f}")
        if r2 - r4 > 0.10:
            lines.append("- **D1 HIT: stronger quant amplifies signal.** Target variance was the limiter.")
        else:
            lines.append("- **D1 NULL: stronger quant doesn't help.** Signal ceiling holds regardless of quant strength.")

    out = "\n".join(lines) + "\n"
    out_path = os.path.join(utils.RESULTS_DIR, "exp8_compare_configs.md")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(out)
    print(out)
    utils.log(f"[exp8a] → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
