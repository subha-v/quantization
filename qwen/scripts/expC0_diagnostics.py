"""
Experiment C0 — no-compute diagnostics from existing JSONLs.

Three blocks:
  1. Item-level complementarity between A5 (INT4) and A7 (INT2) from expA rollouts.
  2. Selected-block coverage for B6 / B8 / B9 / B10 — recompute per-item top-16 masks
     from diagnostic_signals.jsonl using qwen.scripts.precision_need_scoring.
  3. Per-condition answer margin (correct-vs-best-other logprob) from option_logprobs
     across A1, A5, A7, B6, B8, B9, B10. Pull from expA + expB rollout JSONLs.

Run:
  python qwen/scripts/expC0_diagnostics.py \
      --expA   qwen/results/expA_rollouts_Qwen2.5-VL-7B-Instruct.jsonl \
      --expB   qwen/results/expB_online_rollouts.jsonl \
      --diag   qwen/results/diagnostic_signals.jsonl \
      --static qwen/calibration/static_entropy_risk.json \
      --num_layers 28 --num_kv_heads 4 --k_bf16 16
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from precision_need_scoring import (
    compute_score,
    item_signals,
    load_diagnostic_jsonl,
    load_static_risk,
    top_k_mask,
)


# ===================================================================
# Helpers
# ===================================================================

def read_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def margin(option_logprobs, correct_choice):
    """log p(correct) - max log p(other). Higher = more confident in correct answer."""
    if option_logprobs is None or correct_choice is None:
        return None
    if correct_choice >= len(option_logprobs):
        return None
    correct_lp = option_logprobs[correct_choice]
    others = [lp for i, lp in enumerate(option_logprobs) if i != correct_choice]
    if not others:
        return None
    return correct_lp - max(others)


def fmt_pct(x, n):
    return f"{x}/{n} ({100.0 * x / max(1, n):.1f}%)"


def shannon_entropy_bits(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    H = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            H -= p * math.log2(p)
    return H


# ===================================================================
# (1) A5 vs A7 complementarity
# ===================================================================

def block1_complementarity(expA_path: Path) -> str:
    by_item = defaultdict(dict)  # item_id -> {condition: is_correct}
    bf16_correct = {}
    int4_correct = {}
    int2_correct = {}
    n_options_by_item = {}
    duration_bucket = {}

    for r in read_jsonl(expA_path):
        cond = r["condition"]
        iid = r["item_id"]
        by_item[iid][cond] = bool(r["is_correct"])
        n_options_by_item[iid] = r.get("n_options", 4)
        duration_bucket[iid] = r.get("duration_bucket", "unknown")

    iids = sorted(by_item.keys())
    A1, A5, A7 = "A1_BF16_frames64", "A5_BF16_INT4KV_frames64", "A7_BF16_INT2KV_frames64"

    # Filter to items present in both A5 and A7
    iids = [i for i in iids if A5 in by_item[i] and A7 in by_item[i]]

    int4_only_correct = 0   # INT4 correct, INT2 wrong
    int2_only_correct = 0   # INT2 correct, INT4 wrong
    both_correct = 0
    both_wrong = 0

    bf16_correct_int4_or_int2 = 0      # BF16-correct items rescued by at least one of INT4/INT2
    bf16_correct_neither = 0           # BF16-correct items missed by both
    bf16_correct_total = 0

    for iid in iids:
        c4 = by_item[iid][A5]
        c2 = by_item[iid][A7]
        if c4 and c2:
            both_correct += 1
        elif c4 and not c2:
            int4_only_correct += 1
        elif c2 and not c4:
            int2_only_correct += 1
        else:
            both_wrong += 1

        if A1 in by_item[iid] and by_item[iid][A1]:
            bf16_correct_total += 1
            if c4 or c2:
                bf16_correct_int4_or_int2 += 1
            else:
                bf16_correct_neither += 1

    n = len(iids)
    int4_total = sum(by_item[i][A5] for i in iids)
    int2_total = sum(by_item[i][A7] for i in iids)
    union_correct = both_correct + int4_only_correct + int2_only_correct
    intersect_correct = both_correct

    chance = 1.0 / 4.0  # mostly 4-way (very few 5-way; treat 4 as anchor)

    lines = []
    lines.append("## Block 1 — A5 (INT4) vs A7 (INT2) item-level complementarity")
    lines.append("")
    lines.append(f"n eval items (both A5 & A7 present): **{n}**")
    lines.append("")
    lines.append("| set | count | % of n |")
    lines.append("|---|---:|---:|")
    lines.append(f"| INT4 correct (A5) | {int4_total} | {100*int4_total/n:.1f}% |")
    lines.append(f"| INT2 correct (A7) | {int2_total} | {100*int2_total/n:.1f}% |")
    lines.append(f"| both correct (∩) | {intersect_correct} | {100*intersect_correct/n:.1f}% |")
    lines.append(f"| union correct (∪ = oracle of {{INT4, INT2}}) | {union_correct} | {100*union_correct/n:.1f}% |")
    lines.append(f"| **INT4-only correct** (INT4 ✓, INT2 ✗) | **{int4_only_correct}** | **{100*int4_only_correct/n:.1f}%** |")
    lines.append(f"| **INT2-only correct** (INT2 ✓, INT4 ✗) | **{int2_only_correct}** | **{100*int2_only_correct/n:.1f}%** |")
    lines.append(f"| both wrong | {both_wrong} | {100*both_wrong/n:.1f}% |")
    lines.append("")

    # Useful derived quantities
    sym_diff = int4_only_correct + int2_only_correct
    overlap_J = intersect_correct / max(1, union_correct)  # Jaccard on correct sets
    expected_overlap_under_independence = (int4_total / n) * (int2_total / n) * n
    lift = intersect_correct - expected_overlap_under_independence

    lines.append("**Derived metrics**")
    lines.append("")
    lines.append(f"- Symmetric difference (INT4 △ INT2 on correct): **{sym_diff}** "
                 f"({100*sym_diff/n:.1f}% of items)")
    lines.append(f"- Jaccard(INT4-correct, INT2-correct): **{overlap_J:.3f}** "
                 f"(1.0 = perfectly aligned, 0.0 = disjoint)")
    lines.append(f"- Observed both-correct: {intersect_correct}.  Expected under independence: "
                 f"{expected_overlap_under_independence:.1f}.  Lift = {lift:+.1f} "
                 f"({'positive → correlated' if lift > 0 else 'negative → complementary'})")
    lines.append(f"- Oracle {{INT4, INT2}} accuracy ceiling: **{100*union_correct/n:.1f}%** "
                 f"(vs A5={100*int4_total/n:.1f}%, A7={100*int2_total/n:.1f}%, BF16=56.5%)")
    lines.append(f"- BF16-correct rescued by INT4 ∪ INT2: "
                 f"{fmt_pct(bf16_correct_int4_or_int2, bf16_correct_total)} of BF16-correct items")
    lines.append("")

    # Verdict
    if union_correct > max(int4_total, int2_total) + 5:
        verdict = (f"**Verdict.** INT4 and INT2 succeed on substantially different items — "
                   f"oracle union ({100*union_correct/n:.1f}%) exceeds either alone "
                   f"by ≥{union_correct - max(int4_total, int2_total)} items. A richer "
                   f"{{INT2, INT4, BF16}} tier set has real headroom *if* a router can pick the "
                   f"right tier per block.")
    elif union_correct >= max(int4_total, int2_total) + 2:
        verdict = (f"**Verdict.** Modest complementarity — oracle union "
                   f"({100*union_correct/n:.1f}%) beats best-of-A5/A7 by "
                   f"{union_correct - max(int4_total, int2_total)} items. A richer tier set "
                   f"could help marginally, but the corrected-set overlap is still high.")
    else:
        verdict = (f"**Verdict.** Correct sets mostly overlap — oracle union "
                   f"({100*union_correct/n:.1f}%) ≈ best-of-A5/A7 "
                   f"({100*max(int4_total, int2_total)/n:.1f}%). A {{INT2, INT4, BF16}} "
                   f"tier set is unlikely to unlock much: both quantizers fail on largely "
                   f"the same items, suggesting a shared underlying failure mode.")
    lines.append(verdict)
    lines.append("")

    # Per-bucket complementarity
    bucket_count = defaultdict(lambda: [0, 0, 0, 0])  # n, int4_only, int2_only, both_correct
    for iid in iids:
        b = duration_bucket[iid]
        c4 = by_item[iid][A5]
        c2 = by_item[iid][A7]
        bucket_count[b][0] += 1
        if c4 and c2: bucket_count[b][3] += 1
        elif c4 and not c2: bucket_count[b][1] += 1
        elif c2 and not c4: bucket_count[b][2] += 1

    lines.append("**Per-duration-bucket breakdown**")
    lines.append("")
    lines.append("| bucket | n | INT4-only | INT2-only | both ✓ | union ✓ | sym-diff |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for b in ["short", "mid", "long", "very_long"]:
        if b not in bucket_count:
            continue
        nb, i4, i2, bc = bucket_count[b]
        lines.append(f"| {b} | {nb} | {i4} | {i2} | {bc} | {bc + i4 + i2} | {i4 + i2} |")
    lines.append("")

    return "\n".join(lines)


# ===================================================================
# (2) Selected-block coverage for B6, B8, B9, B10
# ===================================================================

def block2_block_coverage(diag_path: Path, static_path: Path,
                          num_layers: int, num_kv_heads: int, k_bf16: int) -> str:
    by_item = load_diagnostic_jsonl(diag_path)
    static = load_static_risk(static_path)

    # Restrict to eval split — these are the rows the routing actually saw
    eval_iids = sorted([iid for iid, rec in by_item.items() if rec["split"] == "eval"])

    methods = {
        "B6_StaticEntropy":   "static_low",
        "B8_OnlineResidual":  "online_residual",
        "B9_OnlineNeed_Static": "online_need_static",
        "B10_OnlineNeed_AQ":  "online_need_aq",
    }

    lines = []
    lines.append("## Block 2 — Selected-block coverage for B6 / B8 / B9 / B10")
    lines.append("")
    lines.append(f"Per item, each method scores {num_layers} × {num_kv_heads} = "
                 f"{num_layers * num_kv_heads} (layer, KV-head) blocks and picks the top "
                 f"**{k_bf16}** for BF16. We summarize coverage across the eval set "
                 f"(n={len(eval_iids)}).")
    lines.append("")
    lines.append("Columns:")
    lines.append("- **layer-cov** = mean # distinct layers receiving ≥1 BF16 head per item "
                 f"(max = {num_layers}; uniform-spread upper bound = "
                 f"min({num_layers}, {k_bf16}) = {min(num_layers, k_bf16)})")
    lines.append("- **heads-per-layer** = mean # BF16 heads per protected layer per item "
                 f"(max = {num_kv_heads})")
    lines.append("- **layer-Hbits** = Shannon entropy (bits) of *summed* across-item layer-protection "
                 f"distribution; uniform = log2({num_layers}) ≈ {math.log2(num_layers):.2f}")
    lines.append("- **head-Hbits** = entropy of summed (L, h) protection distribution; "
                 f"uniform = log2({num_layers * num_kv_heads}) ≈ {math.log2(num_layers * num_kv_heads):.2f}")
    lines.append("- **stable-blocks** = # of (L, h) blocks selected by ≥80% of items "
                 f"(per-method 'always-on' core)")
    lines.append("- **never-blocks** = # of (L, h) blocks *never* selected")
    lines.append("")
    lines.append(f"| method | layer-cov / {num_layers} | heads-per-layer / {num_kv_heads} | "
                 f"layer-Hbits | head-Hbits | stable-blocks | never-blocks |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    # Also store layer-protection vector per method for printing
    layer_dist_by_method: dict[str, list[float]] = {}
    head_dist_by_method: dict[str, list[float]] = {}

    for label, method in methods.items():
        per_item_layer_cov = []
        per_item_heads_per_layer = []
        layer_count = torch.zeros(num_layers, dtype=torch.long)
        block_count = torch.zeros(num_layers, num_kv_heads, dtype=torch.long)

        for iid in eval_iids:
            rec = by_item[iid]
            sig = item_signals(rec, num_layers, num_kv_heads)
            scores = compute_score(
                method,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                static=static,
                item_sig=sig,
                seed=0,
            )
            mask = top_k_mask(scores, k_bf16)            # [L, H] bool
            block_count += mask.long()
            # Per-item layer coverage: layers with at least one head selected
            layer_has_any = mask.any(dim=-1)             # [L] bool
            per_item_layer_cov.append(int(layer_has_any.sum().item()))
            n_protected_layers = max(1, int(layer_has_any.sum().item()))
            heads_in_protected = int(mask[layer_has_any].sum().item()) if layer_has_any.any() else 0
            per_item_heads_per_layer.append(heads_in_protected / n_protected_layers)
            # accumulate layer counts
            layer_count += mask.any(dim=-1).long()

        n = len(eval_iids)
        mean_layer_cov = sum(per_item_layer_cov) / n
        mean_hpl = sum(per_item_heads_per_layer) / n

        layer_dist = layer_count.tolist()
        head_dist = block_count.flatten().tolist()
        layer_H = shannon_entropy_bits(layer_dist)
        head_H = shannon_entropy_bits(head_dist)

        # Stable blocks = selected ≥ 80% of items
        stable = int((block_count >= int(0.8 * n)).sum().item())
        # Never selected = 0
        never = int((block_count == 0).sum().item())

        lines.append(f"| {label} | {mean_layer_cov:.2f} | {mean_hpl:.2f} | "
                     f"{layer_H:.2f} | {head_H:.2f} | {stable} | {never} |")

        layer_dist_by_method[label] = layer_dist
        head_dist_by_method[label] = head_dist

    lines.append("")

    # Per-layer protection counts (out of n_eval items)
    n = len(eval_iids)
    lines.append(f"**Per-layer protection counts** (out of {n} eval items, "
                 f"a layer is 'protected' if any of its {num_kv_heads} KV-heads is selected):")
    lines.append("")
    header = "| method | " + " | ".join(f"L{L:02d}" for L in range(num_layers)) + " |"
    sep = "|---|" + "|".join(["---:"] * num_layers) + "|"
    lines.append(header)
    lines.append(sep)
    for label, vals in layer_dist_by_method.items():
        row = f"| {label} | " + " | ".join(str(v) for v in vals) + " |"
        lines.append(row)
    lines.append("")

    # Verdict text
    lines.append("**Reading the table.**")
    lines.append("")
    lines.append("- Lower **layer-cov** + lower **layer-Hbits** + higher **stable-blocks** = "
                 "concentrated routing (few layers always-on, suspect over-concentration).")
    lines.append("- High **layer-cov** + high entropies + low **stable-blocks** = spread routing "
                 "(closer to random allocation).")
    lines.append("- Compare layer protection rows side-by-side to see whether B9/B10's failures "
                 "are because they pile BF16 onto a small set of layers (over-concentration) "
                 "vs. spread it everywhere like random.")
    lines.append("")

    return "\n".join(lines)


# ===================================================================
# (3) Per-condition answer margin
# ===================================================================

def block3_margins(expA_path: Path, expB_path: Path) -> str:
    """For each condition, report mean margin = lp[correct] - max(lp[other]).

    Margin > 0 ⇒ predicted correct (definitionally).
    Margin ↑ when wrong ⇒ logits moving toward the correct answer.

    We report:
      * mean margin (all items)
      * mean margin restricted to items where the model was *wrong*
        (since accuracy was already reported in QWEN_EXPERIMENTS.md)
      * mean margin restricted to items where BF16 was correct
        (the rescuable subset — the relevant universe)
    """
    # Pull everything keyed by (condition, item_id)
    rows = defaultdict(dict)  # condition -> {item_id: row}
    bf16_correct_set = set()

    for r in read_jsonl(expA_path):
        rows[r["condition"]][r["item_id"]] = r
        if r["condition"] == "A1_BF16_frames64" and r["is_correct"]:
            bf16_correct_set.add(r["item_id"])

    for r in read_jsonl(expB_path):
        rows[r["condition"]][r["item_id"]] = r

    targets = [
        ("A1 BF16 (ceiling)",          "A1_BF16_frames64"),
        ("A5 INT4 (matched-avg)",      "A5_BF16_INT4KV_frames64"),
        ("A7 INT2 (floor)",            "A7_BF16_INT2KV_frames64"),
        ("B6 StaticEntropy (low→BF16)","B6_StaticEntropy"),
        ("B7 FlippedEntropy",          "B7_FlippedEntropy"),
        ("B8 OnlineResidual",          "B8_OnlineResidual"),
        ("B9 OnlineNeed-Static",       "B9_OnlineNeed_Static"),
        ("B10 OnlineNeed-AQ",          "B10_OnlineNeed_AQ"),
    ]

    lines = []
    lines.append("## Block 3 — Answer margin even when accuracy fails")
    lines.append("")
    lines.append("**margin = log p(correct option) − max log p(other option)**.")
    lines.append("Positive margin ⇒ argmax = correct (i.e., is_correct=True).")
    lines.append("If a routing method is moving logits in the right direction even when it "
                 "doesn't quite flip the prediction, the *wrong-only* margin should rise above "
                 "the INT2/INT4 floors.")
    lines.append("")
    lines.append("| condition | n | acc | mean margin (all) | mean margin (wrong only) | "
                 "mean margin (BF16-correct subset) |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for label, cond in targets:
        rs = list(rows.get(cond, {}).values())
        if not rs:
            lines.append(f"| {label} | 0 | — | — | — | — |")
            continue
        n = len(rs)
        n_correct = sum(int(r["is_correct"]) for r in rs)
        margins_all = []
        margins_wrong = []
        margins_bf16_subset = []
        for r in rs:
            m = margin(r.get("option_logprobs"), r.get("correct_choice"))
            if m is None:
                continue
            margins_all.append(m)
            if not r["is_correct"]:
                margins_wrong.append(m)
            if r["item_id"] in bf16_correct_set:
                margins_bf16_subset.append(m)
        def mean(xs):
            return sum(xs) / len(xs) if xs else float("nan")
        lines.append(f"| {label} | {n} | {n_correct/n:.3f} | {mean(margins_all):+.3f} | "
                     f"{mean(margins_wrong):+.3f} (n={len(margins_wrong)}) | "
                     f"{mean(margins_bf16_subset):+.3f} (n={len(margins_bf16_subset)}) |")

    lines.append("")

    # Pairwise: routing improves margin over uniform INT2/INT4 ?
    lines.append("**Δ-margin vs floors** (mean over items present in both conditions):")
    lines.append("")
    lines.append("| routed condition | Δ vs A5 (INT4) wrong-only | Δ vs A7 (INT2) wrong-only | "
                 "Δ vs A5 BF16-subset | Δ vs A7 BF16-subset |")
    lines.append("|---|---:|---:|---:|---:|")

    a5 = rows.get("A5_BF16_INT4KV_frames64", {})
    a7 = rows.get("A7_BF16_INT2KV_frames64", {})

    def paired_delta(rsA: dict, rsB: dict, predicate):
        """For items in A∩B passing predicate (on row from A or B), return (mean(A.m - B.m), n)."""
        deltas = []
        for iid, rb in rsB.items():
            if iid not in rsA:
                continue
            ra = rsA[iid]
            if not predicate(ra, rb, iid):
                continue
            ma = margin(ra.get("option_logprobs"), ra.get("correct_choice"))
            mb = margin(rb.get("option_logprobs"), rb.get("correct_choice"))
            if ma is None or mb is None:
                continue
            deltas.append(ma - mb)
        return (sum(deltas) / len(deltas) if deltas else float("nan"), len(deltas))

    routed_targets = [
        ("B6 StaticEntropy",   "B6_StaticEntropy"),
        ("B7 FlippedEntropy",  "B7_FlippedEntropy"),
        ("B8 OnlineResidual",  "B8_OnlineResidual"),
        ("B9 OnlineNeed-Static","B9_OnlineNeed_Static"),
        ("B10 OnlineNeed-AQ",  "B10_OnlineNeed_AQ"),
    ]

    for label, cond in routed_targets:
        rs = rows.get(cond, {})
        d_a5_wrong, n1 = paired_delta(rs, a5, lambda ra, rb, iid: not ra["is_correct"])
        d_a7_wrong, n2 = paired_delta(rs, a7, lambda ra, rb, iid: not ra["is_correct"])
        d_a5_bf, n3 = paired_delta(rs, a5, lambda ra, rb, iid: iid in bf16_correct_set)
        d_a7_bf, n4 = paired_delta(rs, a7, lambda ra, rb, iid: iid in bf16_correct_set)
        lines.append(f"| {label} | {d_a5_wrong:+.3f} (n={n1}) | {d_a7_wrong:+.3f} (n={n2}) | "
                     f"{d_a5_bf:+.3f} (n={n3}) | {d_a7_bf:+.3f} (n={n4}) |")

    lines.append("")
    lines.append("**Reading the table.**")
    lines.append("")
    lines.append("- Δ > 0 means the routed condition produced higher margin "
                 "(more correct-leaning logits) than the uniform floor on the same items.")
    lines.append("- A positive Δ even when accuracy didn't move means the routing signals "
                 "*are* nudging logits toward the right answer — just not enough to flip "
                 "the argmax. That would imply the floor isn't a complete loss of signal.")
    lines.append("- A non-positive Δ means routing isn't even moving the needle: "
                 "the destruction from 86% INT2 swamps any benefit from the BF16 16-block budget.")
    lines.append("")

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expA", type=Path, required=True)
    ap.add_argument("--expB", type=Path, required=True)
    ap.add_argument("--diag", type=Path, required=True)
    ap.add_argument("--static", type=Path, required=True)
    ap.add_argument("--num_layers", type=int, default=28)
    ap.add_argument("--num_kv_heads", type=int, default=4)
    ap.add_argument("--k_bf16", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = []
    out.append("# Experiment C0 — no-compute diagnostics")
    out.append("")
    out.append(f"Inputs: expA={args.expA.name}, expB={args.expB.name}, "
               f"diag={args.diag.name}, static={args.static.name}")
    out.append(f"Routing budget: top-{args.k_bf16} of "
               f"{args.num_layers}×{args.num_kv_heads} = {args.num_layers*args.num_kv_heads} blocks → BF16.")
    out.append("")
    out.append(block1_complementarity(args.expA))
    out.append("")
    out.append(block2_block_coverage(args.diag, args.static,
                                      args.num_layers, args.num_kv_heads, args.k_bf16))
    out.append("")
    out.append(block3_margins(args.expA, args.expB))
    text = "\n".join(out)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\n[wrote] {args.out}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
