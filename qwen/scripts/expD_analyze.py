"""Analysis for Exp D0 + D1.

Reads:
  qwen/results/expD0_evidence_diagnostic.jsonl  (one row per item)
  qwen/results/expD1_crossmodal_kv.jsonl        (one row per (item, condition))

Writes three markdown summaries:
  qwen/results/expD0_summary.md
  qwen/results/expD1_summary.md
  qwen/results/expD_combined_analysis.md   (D1 stratified by D0 evidence label)

Evidence labels are computed here (post-hoc thresholds, revisable without re-running GPU):
  localized            : full64 correct AND (top1_only or top2_only correct)
                         AND (margin_full - margin_top1_removed) > 0.5
                         AND evidence_causal_gap > 0.2
  global               : full64 correct AND uniform16 correct
                         AND |margin_full - margin_top1_removed| < 0.3
  distributed          : full64 correct AND top1_only wrong AND top2_only wrong
  attention_not_causal : top_window_causal_effect <= random_window_causal_effect_mean
  unlabeled            : everything else (e.g. full64 wrong)
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# ---------------- Evidence label rules ----------------


def evidence_label_for(d0: dict,
                       margin_top_threshold: float = 0.5,
                       causal_gap_threshold: float = 0.2,
                       global_margin_threshold: float = 0.3) -> str:
    """Apply thresholds to a D0 row and return the evidence label."""
    if d0.get("partial"):
        return "unlabeled"
    correct = int(d0["correct_choice"])
    full_correct = int(d0.get("pred_full64", -1)) == correct
    if not full_correct:
        return "unlabeled"
    twce = float(d0.get("top_window_causal_effect", float("nan")))
    rwce = float(d0.get("random_window_causal_effect_mean", float("nan")))
    egap = float(d0.get("evidence_causal_gap", float("nan")))

    top1_correct = int(d0.get("pred_top1_only", -1)) == correct
    top2_correct = int(d0.get("pred_top2_only", -1)) == correct
    u16_correct = int(d0.get("pred_uniform16", -1)) == correct

    margin_full = float(d0.get("margin_full64", float("nan")))
    margin_t1r = float(d0.get("margin_top1_removed", float("nan")))

    # localized
    if (top1_correct or top2_correct) and \
       not math.isnan(margin_full - margin_t1r) and \
       (margin_full - margin_t1r) > margin_top_threshold and \
       not math.isnan(egap) and egap > causal_gap_threshold:
        return "localized"
    # attention_not_causal
    if not math.isnan(twce) and not math.isnan(rwce) and twce <= rwce:
        return "attention_not_causal"
    # global
    if u16_correct and not math.isnan(margin_full - margin_t1r) and \
       abs(margin_full - margin_t1r) < global_margin_threshold:
        return "global"
    # distributed
    if (not top1_correct) and (not top2_correct):
        return "distributed"
    return "unlabeled"


# ---------------- Bootstrap CI ----------------


def bootstrap_ci(arr, n_boot: int = 2000, seed: int = 0):
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, a.size, a.size)
        means.append(a[idx].mean())
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------- D0 summary ----------------


def summarize_d0(d0_jsonl: Path, out_md: Path) -> dict[str, dict]:
    """Returns dict: item_id -> labeled D0 row (with `evidence_label` injected)."""
    rows = []
    if not d0_jsonl.exists():
        out_md.write_text(f"# D0 Summary\n\n(no D0 JSONL at {d0_jsonl})\n")
        return {}
    for line in d0_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))

    labeled = {}
    label_counts = defaultdict(int)
    for r in rows:
        if r.get("phase") != "D0":
            continue
        lab = evidence_label_for(r)
        r["evidence_label"] = lab
        labeled[r["item_id"]] = r
        label_counts[lab] += 1

    # Per-condition accuracy
    def acc(field):
        ok = sum(1 for r in labeled.values()
                 if (not r.get("partial")) and r.get(field) == r["correct_choice"])
        n = sum(1 for r in labeled.values() if (not r.get("partial")) and field in r)
        return ok, n

    rand_pred_field = "pred_random_removed"
    rand_correct = 0
    rand_total = 0
    for r in labeled.values():
        if r.get("partial"):
            continue
        for p in r.get(rand_pred_field, []):
            rand_total += 1
            if p == r["correct_choice"]:
                rand_correct += 1

    lines = ["# Experiment D0 — Evidence-window diagnostic\n"]
    lines.append(f"n_items: {len(labeled)}\n")
    lines.append("## Evidence label distribution\n")
    lines.append("| Label | n | % |")
    lines.append("|---|---:|---:|")
    n = sum(label_counts.values())
    for k in ("localized", "global", "distributed", "attention_not_causal", "unlabeled"):
        v = label_counts.get(k, 0)
        pct = v / n * 100 if n else 0
        lines.append(f"| {k} | {v} | {pct:.1f}% |")
    lines.append("")
    lines.append("## Per-condition accuracy (frame-restriction conditions)\n")
    lines.append("| Condition | n | acc | 95% CI |")
    lines.append("|---|---:|---:|---|")
    for cond_label, field in [
        ("D0.1 Full-64 BF16", "pred_full64"),
        ("D0.2 Uniform-16 BF16", "pred_uniform16"),
        ("D0.3 Top-1-window-only", "pred_top1_only"),
        ("D0.4 Top-2-windows-only", "pred_top2_only"),
        ("D0.5 Top-1-window-removed", "pred_top1_removed"),
    ]:
        bools = [int(r.get(field, -1)) == r["correct_choice"]
                 for r in labeled.values() if (not r.get("partial")) and field in r]
        m, lo, hi = bootstrap_ci(bools)
        lines.append(f"| {cond_label} | {len(bools)} | {m:.3f} | [{lo:.3f}, {hi:.3f}] |")
    if rand_total > 0:
        rand_acc = rand_correct / rand_total
        lines.append(
            f"| D0.6 Random-window-removed (3 seeds pooled) | {rand_total} | {rand_acc:.3f} | — |"
        )

    lines.append("\n## Mean answer margin\n")
    lines.append("| Condition | mean | std | n |")
    lines.append("|---|---:|---:|---:|")
    for cond_label, field in [
        ("Full-64", "margin_full64"),
        ("Uniform-16", "margin_uniform16"),
        ("Top-1-only", "margin_top1_only"),
        ("Top-2-only", "margin_top2_only"),
        ("Top-1-removed", "margin_top1_removed"),
    ]:
        vals = [float(r[field]) for r in labeled.values()
                if field in r and not (math.isnan(r[field]) if isinstance(r[field], float) else False)]
        m = float(np.mean(vals)) if vals else float("nan")
        s = float(np.std(vals)) if vals else float("nan")
        lines.append(f"| {cond_label} | {m:.3f} | {s:.3f} | {len(vals)} |")

    lines.append("\n## EvidenceCausalGap by duration bucket\n")
    lines.append("| Bucket | n | median EvidenceCausalGap | IQR |")
    lines.append("|---|---:|---:|---|")
    by_bucket = defaultdict(list)
    for r in labeled.values():
        gap = r.get("evidence_causal_gap")
        if isinstance(gap, (int, float)) and not math.isnan(gap):
            by_bucket[r["duration_bucket"]].append(float(gap))
    for bk in ("short", "mid", "long", "very_long"):
        v = by_bucket.get(bk, [])
        if not v:
            lines.append(f"| {bk} | 0 | — | — |")
            continue
        med = float(np.median(v))
        q1 = float(np.percentile(v, 25))
        q3 = float(np.percentile(v, 75))
        lines.append(f"| {bk} | {len(v)} | {med:.3f} | [{q1:.3f}, {q3:.3f}] |")

    lines.append("\n## Visual mass total (mean over items)\n")
    lines.append("| Pool | mean | std | min | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for label_, field in [("all-layer", "visual_mass_total_all"),
                          ("mid-layer", "visual_mass_total_mid")]:
        vals = [float(r[field]) for r in labeled.values() if field in r]
        if not vals:
            continue
        lines.append(f"| {label_} | {np.mean(vals):.4f} | {np.std(vals):.4f} | "
                     f"{np.min(vals):.4f} | {np.max(vals):.4f} |")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"[expD] wrote {out_md}")
    return labeled


# ---------------- D1 summary ----------------


def summarize_d1(d1_jsonl: Path, out_md: Path) -> dict[tuple[str, str], dict]:
    """Returns dict: (item_id, condition) -> D1 row."""
    if not d1_jsonl.exists():
        out_md.write_text(f"# D1 Summary\n\n(no D1 JSONL at {d1_jsonl})\n")
        return {}

    rows = []
    for line in d1_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    rows = [r for r in rows if r.get("phase") == "D1"]
    by_key = {(r["item_id"], r["condition"]): r for r in rows}
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    lines = ["# Experiment D1 — Cross-modal K/V quantization\n"]
    lines.append(f"n_rows: {len(rows)}, n_conditions: {len(by_cond)}\n")
    lines.append("| Condition | n | acc | 95% CI | avg KV bits | mean margin |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for cond, rs in sorted(by_cond.items()):
        accs = [int(r["is_correct"]) for r in rs]
        m, lo, hi = bootstrap_ci(accs)
        avg = float(np.mean([float(r["avg_kv_bits"]) for r in rs])) if rs else float("nan")
        margin = float(np.mean([float(r["answer_margin"]) for r in rs
                                if not math.isnan(float(r.get("answer_margin", float("nan"))))]))
        lines.append(f"| {cond} | {len(rs)} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | {avg:.2f} | {margin:.3f} |")

    # BF16-correct preservation
    lines.append("\n## BF16-correct preservation\n")
    lines.append("| Condition | n_bf16_correct | preserved (bf16_correct AND pred==correct) | rate |")
    lines.append("|---|---:|---:|---:|")
    for cond, rs in sorted(by_cond.items()):
        bf16_subset = [r for r in rs if r.get("bf16_correct")]
        preserved = sum(1 for r in bf16_subset if r["is_correct"])
        rate = preserved / len(bf16_subset) if bf16_subset else float("nan")
        lines.append(f"| {cond} | {len(bf16_subset)} | {preserved} | {rate:.3f} |")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"[expD] wrote {out_md}")
    return by_key


# ---------------- Combined: D1 x D0 evidence label ----------------


def combined_analysis(labeled_d0: dict[str, dict],
                      d1_by_key: dict[tuple[str, str], dict],
                      out_md: Path) -> None:
    rows_with_label: list[dict] = []
    missing_d0 = 0
    for (iid, cond), r in d1_by_key.items():
        d0 = labeled_d0.get(iid)
        if d0 is None:
            missing_d0 += 1
            continue
        merged = dict(r)
        merged["evidence_label"] = d0.get("evidence_label", "unlabeled")
        rows_with_label.append(merged)

    by_lc = defaultdict(list)
    for r in rows_with_label:
        by_lc[(r["evidence_label"], r["condition"])].append(r)

    conditions = sorted({r["condition"] for r in rows_with_label})
    labels = ["localized", "global", "distributed", "attention_not_causal", "unlabeled"]

    lines = ["# Experiment D1 stratified by D0 evidence label\n",
             f"n_d1_rows: {len(d1_by_key)}, with_label: {len(rows_with_label)}, missing_d0: {missing_d0}\n",
             "Cell format: `acc (n)`. Pattern to expect on **localized** items: D1.5a > D1.6a (top-1 vs random-1) at matched budget.\n"]

    lines.append("| Condition | " + " | ".join(labels) + " |")
    lines.append("|---" * (len(labels) + 1) + "|")
    for cond in conditions:
        cells = [f"`{cond}`"]
        for lab in labels:
            rs = by_lc.get((lab, cond), [])
            if not rs:
                cells.append("— (0)")
                continue
            acc = sum(1 for r in rs if r["is_correct"]) / len(rs)
            cells.append(f"{acc:.3f} ({len(rs)})")
        lines.append("| " + " | ".join(cells) + " |")

    # Headline pairs
    lines.append("\n## Headline pairs on **localized** items\n")
    lines.append("| Pair | acc(left) | acc(right) | Δ |")
    lines.append("|---|---:|---:|---:|")
    pairs = [
        ("D1_5a_TextBF16_Top1VisBF16_VInt4", "D1_6a_TextBF16_Rand1VisBF16_VInt4_seed0"),
        ("D1_5b_TextBF16_Top2VisBF16_VInt4", "D1_6b_TextBF16_Rand2VisBF16_VInt4_seed0"),
        ("D1_5a_TextBF16_Top1VisBF16_VInt4", "D1_7a_TextBF16_UniformMidVisBF16_VInt4"),
        ("D1_5b_TextBF16_Top2VisBF16_VInt4", "D1_7b_TextBF16_Uniform2VisBF16_VInt4"),
        ("D1_5a_mh_TextBF16_Top1MaxheadVisBF16_VInt4",
            "D1_6a_TextBF16_Rand1VisBF16_VInt4_seed0"),
        ("D1_5b_mh_TextBF16_Top2MaxheadVisBF16_VInt4",
            "D1_6b_TextBF16_Rand2VisBF16_VInt4_seed0"),
        ("D1_5a_mh_TextBF16_Top1MaxheadVisBF16_VInt4",
            "D1_5a_TextBF16_Top1VisBF16_VInt4"),
        ("D1_4_TextInt4_VisBF16_VInt4",     "D1_3_TextBF16_VisInt4_VInt4"),
    ]
    for a, b in pairs:
        rsa = by_lc.get(("localized", a), [])
        rsb = by_lc.get(("localized", b), [])
        accA = (sum(1 for r in rsa if r["is_correct"]) / len(rsa)) if rsa else float("nan")
        accB = (sum(1 for r in rsb if r["is_correct"]) / len(rsb)) if rsb else float("nan")
        delta = (accA - accB) * 100 if (rsa and rsb) else float("nan")
        lines.append(f"| `{a}` vs `{b}` | {accA:.3f} | {accB:.3f} | {delta:+.1f} pp |")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"[expD] wrote {out_md}")


# ---------------- E1: text-K slice ablation ----------------


def summarize_e1(e1_jsonl: Path, d1_jsonl: Path, out_summary: Path,
                 out_pair: Path) -> None:
    """Produce expE1_summary.md (per-condition) + expE1_pair_analysis.md
    (pair table + per-bucket + verdict matrix).
    """
    if not e1_jsonl.exists():
        out_summary.write_text(f"# E1 Summary\n\n(no E1 JSONL at {e1_jsonl})\n")
        out_pair.write_text(f"# E1 Pair Analysis\n\n(no E1 JSONL at {e1_jsonl})\n")
        return

    rows = [json.loads(l) for l in e1_jsonl.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("phase") in ("E1_A", "E1_B")]
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    # Pull D1 reference rows for E1.0 (= D1.1) and E1.1 (= D1.3)
    d1_rows = []
    if d1_jsonl.exists():
        for line in d1_jsonl.read_text().splitlines():
            if line.strip():
                d1_rows.append(json.loads(line))
    d1_by_cond = defaultdict(list)
    for r in d1_rows:
        if r.get("phase") == "D1":
            d1_by_cond[r["condition"]].append(r)
    e1_0_rows = d1_by_cond.get("D1_3_TextBF16_VisInt4_VInt4", [])  # = E1.1
    e1_uniform_rows = d1_by_cond.get("D1_4_TextInt4_VisBF16_VInt4", [])  # not exactly E1.0; A5 floor below
    # E1.0 floor proxy: any uniform-INT4 condition. Use D1.4 as approximate ceiling baseline only;
    # actual E1.0 floor is uniform INT4 K/V which equals A5 in expA. We'll just use the empty mask
    # case from E1 data if present, otherwise fall back to estimates.

    # Build per-condition summary
    lines = ["# Experiment E1 — Text-K slice ablation\n",
             f"n_rows: {len(rows)}, n_conditions: {len(by_cond)}\n"]
    lines.append("| Condition | n | acc | 95% CI | mean margin | median n_text_protected | avg KV bits |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    # Add reference E1.1 row from D1.3
    if e1_0_rows:
        accs = [int(r["is_correct"]) for r in e1_0_rows]
        m, lo, hi = bootstrap_ci(accs)
        margin = float(np.mean([float(r["answer_margin"]) for r in e1_0_rows
                                if r.get("answer_margin") == r.get("answer_margin")]))
        avg = float(np.mean([float(r["avg_kv_bits"]) for r in e1_0_rows]))
        lines.append(f"| **E1.1 = D1.3 (all text-K BF16)** [reused] | {len(e1_0_rows)} | "
                     f"{m:.3f} | [{lo:.3f}, {hi:.3f}] | {margin:.3f} | ~140 | {avg:.2f} |")
    for cond, rs in sorted(by_cond.items()):
        accs = [int(r["is_correct"]) for r in rs]
        m, lo, hi = bootstrap_ci(accs)
        margin_vals = [float(r["answer_margin"]) for r in rs
                       if r.get("answer_margin") == r.get("answer_margin")]
        margin = float(np.mean(margin_vals)) if margin_vals else float("nan")
        n_med = int(np.median([int(r["n_text_protected_tokens"]) for r in rs])) if rs else 0
        avg = float(np.mean([float(r["avg_kv_bits"]) for r in rs])) if rs else float("nan")
        lines.append(f"| {cond} | {len(rs)} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | {margin:.3f} | {n_med} | {avg:.2f} |")

    # BF16-correct preservation
    lines.append("\n## BF16-correct preservation (paired on the BF16-correct subset)\n")
    lines.append("| Condition | n_bf16_correct | preserved | rate |")
    lines.append("|---|---:|---:|---:|")
    for cond, rs in sorted(by_cond.items()):
        bf16_subset = [r for r in rs if r.get("bf16_correct")]
        preserved = sum(1 for r in bf16_subset if r["is_correct"])
        rate = preserved / len(bf16_subset) if bf16_subset else float("nan")
        lines.append(f"| {cond} | {len(bf16_subset)} | {preserved} | {rate:.3f} |")

    out_summary.write_text("\n".join(lines) + "\n")
    print(f"[expE1] wrote {out_summary}")

    # ---- Pair analysis ----
    pair_lines = ["# Experiment E1 — Pair analysis & verdict matrix\n"]

    # Build accuracy + token-count for each E1 condition
    cond_acc = {}
    for cond, rs in by_cond.items():
        if not rs:
            continue
        cond_acc[cond] = {
            "acc": sum(1 for r in rs if r["is_correct"]) / len(rs),
            "n_med_tokens": int(np.median([int(r["n_text_protected_tokens"]) for r in rs])),
            "n": len(rs),
        }
    # Include E1.1 reference
    if e1_0_rows:
        cond_acc["E1_1_AllTextBF16_D1_3"] = {
            "acc": sum(1 for r in e1_0_rows if r["is_correct"]) / len(e1_0_rows),
            "n_med_tokens": 140,  # approximate — full text-K
            "n": len(e1_0_rows),
        }

    # E1.x vs E1.1 (rescue fraction)
    pair_lines.append("\n## E1.x vs E1.1 (all text-K BF16) — what fraction of D1.3's rescue does this slice deliver?\n")
    pair_lines.append("| Condition | acc | n_text_protected | acc - E1.1 | tokens / E1.1 tokens |")
    pair_lines.append("|---|---:|---:|---:|---:|")
    base_acc = cond_acc.get("E1_1_AllTextBF16_D1_3", {}).get("acc", float("nan"))
    base_tokens = cond_acc.get("E1_1_AllTextBF16_D1_3", {}).get("n_med_tokens", 140)
    for cond, info in sorted(cond_acc.items()):
        if cond == "E1_1_AllTextBF16_D1_3":
            continue
        delta = info["acc"] - base_acc
        ratio = info["n_med_tokens"] / max(1, base_tokens)
        pair_lines.append(f"| {cond} | {info['acc']:.3f} | {info['n_med_tokens']} | {delta*100:+.1f} pp | {ratio:.2f} |")

    # Per-bucket accuracy
    pair_lines.append("\n## Per-bucket accuracy\n")
    pair_lines.append("| Condition | short | mid | long | very_long |")
    pair_lines.append("|---|---:|---:|---:|---:|")
    by_cb = defaultdict(list)
    for r in rows:
        by_cb[(r["condition"], r["duration_bucket"])].append(int(r["is_correct"]))
    for cond in sorted(by_cond.keys()):
        cells = [f"`{cond}`"]
        for bk in ("short", "mid", "long", "very_long"):
            v = by_cb.get((cond, bk), [])
            cells.append(f"{(sum(v)/len(v) if v else 0):.3f} (n={len(v)})" if v else "—")
        pair_lines.append("| " + " | ".join(cells) + " |")

    # Verdict: smallest sufficient subset (>=80% of E1.1 acc at <50% tokens)
    pair_lines.append("\n## Verdict: smallest sufficient text-K subset\n")
    pair_lines.append(
        f"Reference: E1.1 (all text-K BF16) acc = {base_acc:.3f}, n_text_protected ≈ {base_tokens}.\n"
        f"\n80% of E1.1 acc threshold = {0.8 * base_acc:.3f}; <50% tokens = {0.5 * base_tokens:.0f}.\n"
    )
    pair_lines.append("\n| Condition | acc | n_tokens | meets 80% acc | <50% tokens | sufficient? |")
    pair_lines.append("|---|---:|---:|:-:|:-:|:-:|")
    for cond, info in sorted(cond_acc.items()):
        if cond == "E1_1_AllTextBF16_D1_3":
            continue
        meets_acc = info["acc"] >= 0.8 * base_acc
        small_enough = info["n_med_tokens"] < 0.5 * base_tokens
        sufficient = "YES" if (meets_acc and small_enough) else "no"
        pair_lines.append(f"| {cond} | {info['acc']:.3f} | {info['n_med_tokens']} | "
                          f"{'yes' if meets_acc else 'no'} | "
                          f"{'yes' if small_enough else 'no'} | {sufficient} |")

    out_pair.write_text("\n".join(pair_lines) + "\n")
    print(f"[expE1] wrote {out_pair}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d0_jsonl", type=Path,
                    default=RESULTS_DIR / "expD0_evidence_diagnostic.jsonl")
    ap.add_argument("--d1_jsonl", type=Path,
                    default=RESULTS_DIR / "expD1_crossmodal_kv.jsonl")
    ap.add_argument("--e1_jsonl", type=Path,
                    default=RESULTS_DIR / "expE1_text_slice_ablation.jsonl")
    ap.add_argument("--d0_summary", type=Path, default=RESULTS_DIR / "expD0_summary.md")
    ap.add_argument("--d1_summary", type=Path, default=RESULTS_DIR / "expD1_summary.md")
    ap.add_argument("--combined", type=Path, default=RESULTS_DIR / "expD_combined_analysis.md")
    ap.add_argument("--e1_summary", type=Path, default=RESULTS_DIR / "expE1_summary.md")
    ap.add_argument("--e1_pair", type=Path, default=RESULTS_DIR / "expE1_pair_analysis.md")
    args = ap.parse_args()

    labeled_d0 = summarize_d0(args.d0_jsonl, args.d0_summary)
    d1 = summarize_d1(args.d1_jsonl, args.d1_summary)
    combined_analysis(labeled_d0, d1, args.combined)
    summarize_e1(args.e1_jsonl, args.d1_jsonl, args.e1_summary, args.e1_pair)


if __name__ == "__main__":
    main()
