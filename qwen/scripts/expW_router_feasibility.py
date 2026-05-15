"""Exp W — Router feasibility postprocess on Exp V1 rollouts.

NO GPU. Reads the existing V1 JSONL rows and asks:

  Step 1: Can a simple S4 escalation detector predict per-item whether the
          S4 anchor (V3) suffices or we should route to a more expensive
          residual-extra policy?

  Step 2: For escalated items, can a policy classifier pick the cheapest-
          correct policy among a portfolio (V15 BAL4, V4 GEN8, V9/V10
          TV/VT, V11 BAL8, V12 MMNIAH8, V5/V6/V7 random-diverse)?

  Step 3: Evaluate held-out (NOT train-set oracle):
            - 70/30 within retrieval
            - train retrieval → test reasoning
            - train retrieval+reasoning → test LVB

Two variants of the escalation detector:
  - **one-pass (deployable)**: features available BEFORE running S4 —
    context_length, num_images, seq_len, task type.
  - **two-pass (diagnostic)**: features from a cheap first S4 pass —
    S4 answer logits, top-1/top-2 margin, entropy, predicted choice.

The router must clear the bar set by:
  - S4 alone (the cheapest tier)
  - V15 BAL4 (the static Pareto winner)
  - best static policy on the test slice
  - oracle upper bound (cheapest-correct policy per item)

Primary metric: **fraction of oracle lift recovered**.
"""
from __future__ import annotations
import json
import math
import collections
import statistics
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

RESULTS = Path("/data/subha2/quantization/qwen/results")

KV_BITS = {
    "V0": 16.0, "V1": 4.0, "V2": 4.75, "V3": 4.1875,
    "V4": 4.28125, "V5": 4.28125, "V6": 4.28125, "V7": 4.28125,
    "V8": 4.28125, "V9": 4.28125, "V10": 4.28125,
    "V11": 4.28125, "V12": 4.28125, "V13": 4.28125,
    "V14": 4.375, "V15": 4.234375, "V16": 4.328125, "V17": 4.375,
}
S4 = "V3"
F9 = "V2"

DATASETS = [
    ("retrieval", RESULTS / "expV_rollouts_sliceV_retrieval.jsonl"),
    ("reasoning", RESULTS / "expV_rollouts_sliceV_reasoning.jsonl"),
    ("lvb",       RESULTS / "expV_lvb_stage3_seed2_normalized.jsonl"),
]

# Tier-2 portfolio for the escalation policy classifier. Picked per the user's
# spec — includes random-diverse codebooks because the oracle showed random
# has more error diversity than structured TT/TV/VT.
TIER2_POLICIES_FULL = ["V15", "V4", "V9", "V10", "V11", "V12", "V5", "V6", "V7"]


def short_cond(c):
    if not c:
        return c
    return c.split("_", 1)[0]


def load_rows(path: Path):
    """Return: matrix[item_id][cond] = row dict, meta[item_id] = item metadata."""
    matrix = collections.defaultdict(dict)
    meta = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        item_id = r.get("item_id")
        if item_id is None:
            continue
        cond = short_cond(r.get("condition") or r.get("cond_name"))
        if cond is None:
            continue
        matrix[item_id][cond] = r
        if item_id not in meta:
            meta[item_id] = {
                "context_length_bucket": r.get("context_length_bucket"),
                "num_images": r.get("num_images"),
                "context_length": r.get("context_length"),
                "duration_bucket": r.get("duration_bucket"),
                "seq_len": r.get("seq_len"),
            }
    return matrix, meta


def num_img_bucket(n):
    if n is None: return -1
    n = int(n)
    if n <= 4:  return 0
    if n <= 7:  return 1
    if n <= 11: return 2
    if n <= 19: return 3
    return 4


def ctx_len_bucket(b):
    return {"short": 0, "mid": 1, "long": 2}.get(b, -1)


def extract_features(matrix, item_id, dataset_tag, include_s4_features=True):
    """Per-item feature vector. dataset_tag is one of {retrieval, reasoning, lvb}."""
    d = matrix[item_id]
    meta = {
        "context_length_bucket": d.get(S4, {}).get("context_length_bucket"),
        "num_images": d.get(S4, {}).get("num_images"),
        "seq_len": d.get(S4, {}).get("seq_len"),
        "context_length": d.get(S4, {}).get("context_length"),
    }
    feats = []
    feats.append(ctx_len_bucket(meta["context_length_bucket"]))
    feats.append(int(meta["num_images"] or 0))
    feats.append(num_img_bucket(meta["num_images"]))
    feats.append(int(meta["seq_len"] or 0))
    feats.append(int(meta["context_length"] or 0))
    # Task one-hot.
    feats.append(1 if dataset_tag == "retrieval" else 0)
    feats.append(1 if dataset_tag == "reasoning" else 0)
    feats.append(1 if dataset_tag == "lvb" else 0)
    if include_s4_features:
        # Two-pass diagnostic: features from a cheap first S4 pass.
        s4_row = d.get(S4, {})
        logits = s4_row.get("first_logits_AD") or s4_row.get("first_logits") or [0.0, 0.0, 0.0, 0.0]
        if isinstance(logits, list) and len(logits) >= 4:
            arr = np.array(logits[:4], dtype=np.float64)
            top1 = float(arr.max())
            top2 = float(np.sort(arr)[-2])
            margin = top1 - top2
            p = np.exp(arr - arr.max())
            p = p / max(p.sum(), 1e-9)
            entropy = -float((p * np.log(np.clip(p, 1e-9, 1.0))).sum())
            argmax = int(arr.argmax())
        else:
            top1 = top2 = margin = 0.0
            entropy = 0.0
            argmax = 0
        feats += [top1, top2, margin, entropy]
        feats += [1 if argmax == k else 0 for k in range(4)]
    return np.array(feats, dtype=np.float64)


def build_dataset(matrix, meta, dataset_tag, tier2_policies, include_s4=True):
    """Per-item: (feature_vec, s4_correct, extras_correct_dict, cheapest_correct_pol)."""
    out = []
    for it in matrix:
        d = matrix[it]
        if S4 not in d:
            continue
        s4_corr = bool(d[S4].get("is_correct"))
        extras_corr = {}
        for p in tier2_policies:
            if p in d:
                extras_corr[p] = bool(d[p].get("is_correct"))
        any_extras = any(extras_corr.values())
        # Cheapest-correct policy overall (S4 + extras).
        candidates = []
        if s4_corr:
            candidates.append((S4, KV_BITS[S4]))
        for p, c in extras_corr.items():
            if c:
                candidates.append((p, KV_BITS[p]))
        if candidates:
            cheap_p = min(candidates, key=lambda t: t[1])[0]
        else:
            cheap_p = None
        feats = extract_features(matrix, it, dataset_tag, include_s4_features=include_s4)
        out.append({
            "item_id": it,
            "feats": feats,
            "s4_correct": s4_corr,
            "extras_correct": extras_corr,
            "any_correct": s4_corr or any_extras,
            "cheap_p": cheap_p,
            # Stage-A label: 1 iff S4 wrong AND ≥1 extras correct (routing helps)
            "stage_a_label": (not s4_corr) and any_extras,
            # Stage-B label (only meaningful when stage_a_label==1): the
            # cheapest-correct extras policy.
            "stage_b_label": (
                min([(p, KV_BITS[p]) for p, c in extras_corr.items() if c],
                    key=lambda t: t[1])[0]
                if (not s4_corr) and any_extras else None
            ),
        })
    return out


def acc_of_router(rows, route_fn):
    """Apply route_fn(row) -> policy_name, return (acc, mean_kv_bits, picks)."""
    n_correct = 0
    bits = []
    picks = collections.Counter()
    for r in rows:
        p = route_fn(r)
        picks[p] += 1
        if p == S4:
            correct = r["s4_correct"]
        else:
            correct = r["extras_correct"].get(p, False)
        n_correct += int(correct)
        bits.append(KV_BITS.get(p, 4.75))
    return n_correct / max(1, len(rows)), statistics.mean(bits), picks


def static_router(p):
    return lambda r: p


def oracle_router(r):
    return r["cheap_p"] if r["cheap_p"] is not None else S4


def train_eval_router(train_rows, test_rows, tier2_policies,
                       stage_b_strategy="default_v15",
                       classifier="gbm",
                       stage_a_threshold=0.5):
    """Train Stage-A escalation detector + optional Stage-B portfolio classifier.

    stage_b_strategy:
      - "default_v15": for escalated items, always pick V15 BAL4
      - "default_best": pick the best static policy in the tier-2 pool on the train set
      - "classifier":  multi-class classifier over tier2_policies (predicting cheapest_extras)

    classifier: "gbm" (GradientBoosting) or "lr" (LogisticRegression).
    """
    # Stage A training data.
    Xa = np.array([r["feats"] for r in train_rows])
    ya = np.array([int(r["stage_a_label"]) for r in train_rows])
    # Skip training if class imbalance is degenerate.
    if ya.sum() == 0 or ya.sum() == len(ya):
        clf_a = None
    else:
        scaler_a = StandardScaler()
        Xa_s = scaler_a.fit_transform(Xa)
        if classifier == "gbm":
            clf_a = GradientBoostingClassifier(n_estimators=80, max_depth=3,
                                                random_state=0)
        else:
            clf_a = LogisticRegression(max_iter=500, random_state=0)
        clf_a.fit(Xa_s, ya)
    # Stage B training (only on items where stage_a_label==1).
    clf_b = None
    if stage_b_strategy == "classifier":
        train_pos = [r for r in train_rows if r["stage_a_label"]]
        if len(train_pos) >= 10:
            Xb = np.array([r["feats"] for r in train_pos])
            yb_raw = [r["stage_b_label"] for r in train_pos]
            label_set = sorted(set(yb_raw))
            label_to_idx = {l: i for i, l in enumerate(label_set)}
            yb = np.array([label_to_idx[l] for l in yb_raw])
            if len(label_set) >= 2:
                scaler_b = StandardScaler()
                Xb_s = scaler_b.fit_transform(Xb)
                clf_b = GradientBoostingClassifier(n_estimators=60, max_depth=3,
                                                    random_state=0)
                clf_b.fit(Xb_s, yb)
                idx_to_label = {i: l for l, i in label_to_idx.items()}
            else:
                clf_b = None
    # Fallback policy for Stage B.
    if stage_b_strategy == "default_v15":
        fallback = "V15"
    elif stage_b_strategy == "default_best":
        # Best static policy in the tier-2 pool on the train set.
        best_p = None; best_acc = -1.0
        for p in tier2_policies:
            n_c = sum(1 for r in train_rows if r["extras_correct"].get(p, False))
            acc = n_c / max(1, len(train_rows))
            if acc > best_acc:
                best_acc = acc; best_p = p
        fallback = best_p
    else:
        fallback = "V15"  # safety net
    # Make sure fallback is available on test set; else find the best fallback
    # available in test rows.
    avail_pol = set()
    for r in test_rows:
        avail_pol |= set(r["extras_correct"].keys())
    if fallback not in avail_pol:
        fallback_candidates = [p for p in tier2_policies if p in avail_pol]
        fallback = fallback_candidates[0] if fallback_candidates else None
    # Build router function.
    def route(r):
        feats = r["feats"]
        if clf_a is None:
            # Degenerate: default to S4 (one class).
            return S4
        feats_s = scaler_a.transform(feats.reshape(1, -1))
        p_escalate = clf_a.predict_proba(feats_s)[0, 1]
        if p_escalate < stage_a_threshold:
            return S4
        # Escalate. Choose tier-2 policy.
        if clf_b is not None:
            feats_b = scaler_b.transform(feats.reshape(1, -1))
            pred_idx = int(clf_b.predict(feats_b)[0])
            pred_p = idx_to_label[pred_idx]
            if pred_p in r["extras_correct"]:
                return pred_p
            # Fallback if predicted policy isn't in test set's available set.
        return fallback if fallback is not None else S4
    return route, {
        "classifier": classifier,
        "stage_b_strategy": stage_b_strategy,
        "stage_a_threshold": stage_a_threshold,
        "stage_a_label_pos_rate": float(ya.sum() / max(1, len(ya))),
        "stage_b_fallback": fallback,
    }


def evaluate_router(test_rows, route_fn, baselines, oracle_acc):
    """Return acc, mean_kv, picks, fraction_oracle_lift_recovered."""
    acc, mkv, picks = acc_of_router(test_rows, route_fn)
    # Fraction of oracle lift recovered relative to "best static" baseline.
    best_static_acc = max(baselines.values())
    lift_oracle = oracle_acc - best_static_acc
    lift_router = acc - best_static_acc
    frac = (lift_router / lift_oracle) if lift_oracle > 1e-9 else 0.0
    return {
        "acc": acc,
        "mean_kv_bits": mkv,
        "picks": dict(picks),
        "lift_vs_best_static": lift_router,
        "lift_oracle_max": lift_oracle,
        "fraction_oracle_lift_recovered": frac,
        "oracle_acc": oracle_acc,
        "best_static_acc": best_static_acc,
    }


def fmt_pct(x):
    if x is None or x != x:
        return "—"
    return f"{x*100:.1f}%"


def split_70_30(rows, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    cut = int(0.7 * len(rows))
    train = [rows[i] for i in idx[:cut]]
    test = [rows[i] for i in idx[cut:]]
    return train, test


def baselines_on(rows, tier2_policies):
    """Static baselines for a set of test rows. Returns {name: acc}."""
    out = {}
    for p in [S4] + tier2_policies + [F9]:
        avail = sum(1 for r in rows if p in r["extras_correct"] or p == S4)
        if avail == 0 and p != S4:
            continue
        if p == S4:
            n_c = sum(1 for r in rows if r["s4_correct"])
        else:
            n_c = sum(1 for r in rows if r["extras_correct"].get(p, False))
        out[p] = n_c / max(1, len(rows))
    return out


def oracle_acc_on(rows):
    n_c = sum(1 for r in rows if r["any_correct"])
    return n_c / max(1, len(rows))


def report_split(name, train_rows, test_rows, tier2_policies):
    """Train + evaluate multiple router variants on a train/test split."""
    print(f"\n## Split: {name}  (n_train={len(train_rows)}, n_test={len(test_rows)})")
    bases = baselines_on(test_rows, tier2_policies)
    oracle = oracle_acc_on(test_rows)
    print(f"  baselines (test): " +
          ", ".join(f"{p}={a:.3f}" for p, a in sorted(bases.items())))
    print(f"  oracle (test): {oracle:.3f}")
    print(f"  oracle lift over best static: "
          f"+{(oracle - max(bases.values()))*100:.1f} pp")
    print()
    print(f"  | router | classifier | stage_b | thresh | test_acc | mean_KV | "
          f"lift_vs_best_static | frac_oracle_lift_recovered |")
    print(f"  |---|---|---|---:|---:|---:|---:|---:|")

    # Pure baselines as router functions.
    for static_p, label in [(S4, "always-S4"), ("V15", "always-V15"),
                            ("V11", "always-V11"), ("V2", "always-F9 (V2)"),
                            (max(bases, key=bases.get), "always-best-static")]:
        if static_p not in bases and static_p != S4:
            continue
        acc, mkv, picks = acc_of_router(test_rows, static_router(static_p))
        ev = evaluate_router(test_rows, static_router(static_p), bases, oracle)
        print(f"  | {label} | — | — | — | {acc:.3f} | {mkv:.3f} | "
              f"{ev['lift_vs_best_static']*100:+.1f}pp | "
              f"{fmt_pct(ev['fraction_oracle_lift_recovered'])} |")

    # Random uniform router.
    rng = np.random.default_rng(0)
    def random_router(r):
        return rng.choice([S4] + tier2_policies)
    acc, mkv, picks = acc_of_router(test_rows, random_router)
    ev = evaluate_router(test_rows, random_router, bases, oracle)
    print(f"  | uniform-random | — | — | — | {acc:.3f} | {mkv:.3f} | "
          f"{ev['lift_vs_best_static']*100:+.1f}pp | "
          f"{fmt_pct(ev['fraction_oracle_lift_recovered'])} |")

    # Routers we train.
    variants = []
    # One-pass (no S4 features) — deployable.
    # Two-pass (with S4 features) — diagnostic.
    for tag, include_s4 in (("one-pass", False), ("two-pass", True)):
        # Re-extract features for this variant.
        tag_str = tag
        for clf in ("gbm", "lr"):
            for stage_b in ("default_v15", "default_best", "classifier"):
                for tau in (0.3, 0.5, 0.7):
                    # Build feature vectors for train + test (matching include_s4).
                    train_feats = [r.copy() for r in train_rows]
                    test_feats = [r.copy() for r in test_rows]
                    # Re-extract features per include_s4 setting using the row's stored matrix.
                    # We saved features at row build time; rerun extraction.
                    pass
    # Simplified — use already-extracted features. The row already has the
    # default feature vector (with S4 features). For one-pass we slice off
    # the S4 features (indices >= 8 in extract_features layout).
    # Layout: [0]=ctx_bucket, [1]=num_images, [2]=num_img_bucket, [3]=seq_len,
    #         [4]=context_length, [5..7]=task one-hot,
    #         [8]=top1, [9]=top2, [10]=margin, [11]=entropy, [12..15]=argmax one-hot.
    N_ONE_PASS = 8  # features before S4-features
    for tag, slice_idx in (("one-pass", slice(0, N_ONE_PASS)),
                            ("two-pass", slice(0, None))):
        tr = [{**r, "feats": r["feats"][slice_idx]} for r in train_rows]
        te = [{**r, "feats": r["feats"][slice_idx]} for r in test_rows]
        for clf in ("gbm", "lr"):
            for stage_b in ("default_v15", "default_best", "classifier"):
                for tau in (0.3, 0.5, 0.7):
                    try:
                        route_fn, info = train_eval_router(
                            tr, te, tier2_policies,
                            stage_b_strategy=stage_b,
                            classifier=clf,
                            stage_a_threshold=tau,
                        )
                    except Exception as e:
                        continue
                    acc, mkv, picks = acc_of_router(te, route_fn)
                    ev = evaluate_router(te, route_fn, bases, oracle)
                    print(f"  | {tag} | {clf} | {stage_b} | {tau:.1f} | "
                          f"{acc:.3f} | {mkv:.3f} | "
                          f"{ev['lift_vs_best_static']*100:+.1f}pp | "
                          f"{fmt_pct(ev['fraction_oracle_lift_recovered'])} |")
    print()


def main():
    print("# Exp W — Router feasibility postprocess on Exp V1 rollouts\n")
    # Load all 3 datasets.
    data = {}
    for ds_name, p in DATASETS:
        if not p.exists():
            print(f"SKIP {ds_name} (missing {p.name})")
            continue
        matrix, meta = load_rows(p)
        if ds_name == "lvb":
            tier2 = [pol for pol in TIER2_POLICIES_FULL
                     if any(pol in d for d in matrix.values())]
        else:
            tier2 = [pol for pol in TIER2_POLICIES_FULL
                     if any(pol in d for d in matrix.values())]
        rows = build_dataset(matrix, meta, ds_name, tier2, include_s4=True)
        data[ds_name] = (rows, tier2)
        print(f"Loaded {ds_name}: {len(rows)} items, tier-2 portfolio "
              f"= {tier2}")
        # Quick label distribution.
        labels = collections.Counter(r["stage_a_label"] for r in rows)
        print(f"  Stage-A label distribution: "
              f"escalate=1: {labels[True]}, escalate=0: {labels[False]}")
        oracle = oracle_acc_on(rows)
        s4 = sum(1 for r in rows if r["s4_correct"]) / max(1, len(rows))
        print(f"  S4 alone acc: {s4:.3f}; oracle (S4 + extras): {oracle:.3f}; "
              f"oracle lift over S4: +{(oracle-s4)*100:.1f} pp\n")

    # ----- Split 1: 70/30 within retrieval -----
    if "retrieval" in data:
        rows, tier2 = data["retrieval"]
        train, test = split_70_30(rows, seed=0)
        report_split("retrieval 70/30 (seed=0)", train, test, tier2)
        train, test = split_70_30(rows, seed=1)
        report_split("retrieval 70/30 (seed=1)", train, test, tier2)

    # ----- Split 2: train retrieval, test reasoning -----
    if "retrieval" in data and "reasoning" in data:
        train, _ = data["retrieval"]
        _, test_tier2 = data["reasoning"]
        test, _ = data["reasoning"]
        # Use intersection of tier-2 policies; require feature vectors match.
        # Features include task one-hot, so cross-task works.
        common_tier2 = sorted(set(data["retrieval"][1]) & set(test_tier2))
        # Rebuild rows with the common tier-2 portfolio.
        train_common = [
            {**r, "extras_correct": {p: r["extras_correct"][p] for p in common_tier2 if p in r["extras_correct"]}}
            for r in train
        ]
        test_common = [
            {**r, "extras_correct": {p: r["extras_correct"][p] for p in common_tier2 if p in r["extras_correct"]}}
            for r in test
        ]
        report_split("train retrieval → test reasoning  (cross-task)",
                     train_common, test_common, common_tier2)

    # ----- Split 3: train retrieval+reasoning, test LVB -----
    if "retrieval" in data and "reasoning" in data and "lvb" in data:
        rows_r, _ = data["retrieval"]
        rows_n, _ = data["reasoning"]
        rows_l, tier2_l = data["lvb"]
        # Intersection across all three datasets.
        common = sorted(set(data["retrieval"][1]) & set(data["reasoning"][1]) & set(tier2_l))
        def restrict(rows, common):
            return [
                {**r, "extras_correct": {p: r["extras_correct"][p] for p in common if p in r["extras_correct"]}}
                for r in rows
            ]
        train = restrict(rows_r + rows_n, common)
        test = restrict(rows_l, common)
        report_split("train retrieval+reasoning → test LVB  (cross-dataset)",
                     train, test, common)


if __name__ == "__main__":
    main()
