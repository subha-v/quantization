"""
Experiment C — K/V isolation analysis.

Reads the appended expA rollouts JSONL (now containing A1-A8 at n=200 plus
C2.1-C2.4 at n=100) and emits a paired-comparison table:

  * For each C2.x: acc, 95% CI, BF16-correct preserved, mean margin
  * For A1, A5, A6, A7 *restricted to the same 100 item_ids* used by C2.x
    (so the comparison is paired, not 100 vs 200)
  * One-line diagnosis (K-fragile / V-fragile / joint)

Run:
  python qwen/scripts/expC_analyze.py \
      --jsonl qwen/results/expA_rollouts_Qwen2.5-VL-7B-Instruct.jsonl \
      --out   qwen/results/expC_kv_isolation_summary.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


C2_CONDS = [
    ("C2_1_BF16K_INT4V_frames64", "C2.1 K=BF16, V=INT4", 16, 4),
    ("C2_2_INT4K_BF16V_frames64", "C2.2 K=INT4, V=BF16",  4, 16),
    ("C2_3_BF16K_INT2V_frames64", "C2.3 K=BF16, V=INT2", 16, 2),
    ("C2_4_INT2K_BF16V_frames64", "C2.4 K=INT2, V=BF16",  2, 16),
]
PAIR_CONDS = [
    ("A1_BF16_frames64",            "A1 BF16 (ceiling)"),
    ("A5_BF16_INT4KV_frames64",     "A5 INT4-K/INT4-V"),
    ("A6_BF16_INT4K_INT8V_frames64","A6 INT4-K/INT8-V"),
    ("A7_BF16_INT2KV_frames64",     "A7 INT2-K/INT2-V"),
]


def bootstrap_ci(arr, n_boot=2000, seed=0):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [arr[rng.integers(0, arr.size, arr.size)].mean() for _ in range(n_boot)]
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def margin(option_logprobs, correct_choice):
    if option_logprobs is None or correct_choice is None or correct_choice >= len(option_logprobs):
        return None
    others = [lp for i, lp in enumerate(option_logprobs) if i != correct_choice]
    return option_logprobs[correct_choice] - max(others) if others else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    by_cond: dict[str, dict[str, dict]] = defaultdict(dict)  # condition -> {item_id: row}
    for line in args.jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_cond[r["condition"]][r["item_id"]] = r

    bf16_correct = {iid for iid, r in by_cond["A1_BF16_frames64"].items() if r["is_correct"]}

    # Item universe = items present in *all* four C2 conditions
    c2_iids = None
    for cond, _, _, _ in C2_CONDS:
        rs = by_cond.get(cond, {})
        if c2_iids is None:
            c2_iids = set(rs.keys())
        else:
            c2_iids &= set(rs.keys())
    c2_iids = sorted(c2_iids or [])

    out = []
    out.append("# Experiment C — K/V isolation (paired analysis)")
    out.append("")
    out.append(f"Universe: {len(c2_iids)} item_ids present in all four C2 conditions.")
    out.append("All other rows are restricted to this same set for paired comparison.")
    out.append("")

    # Per-bucket counts on the C2 universe
    bucket_counts: dict[str, int] = defaultdict(int)
    for iid in c2_iids:
        r = by_cond["A1_BF16_frames64"].get(iid) or next(
            (rr for cond, _, _, _ in C2_CONDS for rr in [by_cond[cond].get(iid)] if rr), None
        )
        if r:
            bucket_counts[r.get("duration_bucket", "?")] += 1
    out.append("Per-bucket: " + ", ".join(f"{b}={n}" for b, n in sorted(bucket_counts.items())))
    out.append("")

    out.append("| Condition | n | acc | 95% CI | avg KV bits | BF16-correct preserved | mean margin |")
    out.append("|---|---:|---:|---|---:|---:|---:|")

    def row_for(cond, label, kb=None, vb=None):
        rs = by_cond.get(cond, {})
        if not rs:
            return None
        # Restrict to C2 universe
        items = [rs[iid] for iid in c2_iids if iid in rs]
        if not items:
            return None
        n = len(items)
        accs = [int(r["is_correct"]) for r in items]
        mean, lo, hi = bootstrap_ci(accs)
        avg_bits = items[0].get("avg_kv_bits")
        if avg_bits is None and kb is not None:
            avg_bits = (kb + vb) / 2.0
        # BF16-correct preserved
        if cond == "A1_BF16_frames64":
            preserved_str = "1.000"
        else:
            shared = [r for r in items if r["item_id"] in bf16_correct]
            preserved = (sum(1 for r in shared if r["is_correct"]) / max(1, len(shared)))
            preserved_str = f"{preserved:.3f} (n={len(shared)})"
        margins = [margin(r.get("option_logprobs"), r.get("correct_choice")) for r in items]
        margins = [m for m in margins if m is not None]
        m_mean = sum(margins) / len(margins) if margins else float("nan")
        return (f"| {label} | {n} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] | "
                f"{avg_bits:.2f} | {preserved_str} | {m_mean:+.3f} |")

    # Order: A1, A5, A6, A7 (Exp A anchors), then C2.1-C2.4
    for cond, label in PAIR_CONDS:
        r = row_for(cond, label)
        if r:
            out.append(r)
    for cond, label, kb, vb in C2_CONDS:
        r = row_for(cond, label, kb=kb, vb=vb)
        if r:
            out.append(r)

    out.append("")

    # Diagnosis
    out.append("## Diagnosis")
    out.append("")
    accs = {}
    for cond, _ in PAIR_CONDS:
        rs = by_cond.get(cond, {})
        items = [rs[iid] for iid in c2_iids if iid in rs]
        if items:
            accs[cond] = sum(r["is_correct"] for r in items) / len(items)
    for cond, _, _, _ in C2_CONDS:
        rs = by_cond.get(cond, {})
        items = [rs[iid] for iid in c2_iids if iid in rs]
        if items:
            accs[cond] = sum(r["is_correct"] for r in items) / len(items)

    bf16 = accs.get("A1_BF16_frames64", float("nan"))
    a5 = accs.get("A5_BF16_INT4KV_frames64", float("nan"))
    a7 = accs.get("A7_BF16_INT2KV_frames64", float("nan"))
    c1 = accs.get("C2_1_BF16K_INT4V_frames64", float("nan"))   # K=BF16, V=INT4 → if rescues, V is fine alone, K is OK; quantizing K killed it
    c2 = accs.get("C2_2_INT4K_BF16V_frames64", float("nan"))   # K=INT4, V=BF16 → if rescues, K is fine alone; quantizing V killed it
    c3 = accs.get("C2_3_BF16K_INT2V_frames64", float("nan"))
    c4 = accs.get("C2_4_INT2K_BF16V_frames64", float("nan"))

    # Use a "rescue threshold" of (BF16 + uniform-floor) / 2 — anything above the midpoint is meaningful recovery.
    midpoint_int4 = (bf16 + a5) / 2 if not (np.isnan(bf16) or np.isnan(a5)) else float("nan")
    midpoint_int2 = (bf16 + a7) / 2 if not (np.isnan(bf16) or np.isnan(a7)) else float("nan")

    out.append(f"- BF16 ceiling: {bf16:.3f};  A5 (INT4/INT4) floor: {a5:.3f};  "
               f"A7 (INT2/INT2) floor: {a7:.3f}")
    out.append(f"- Rescue midpoint (INT4): {midpoint_int4:.3f};  (INT2): {midpoint_int2:.3f}")
    out.append("")
    out.append("**Symmetric question** — at INT4:")
    out.append(f"- C2.1 (K=BF16, V=INT4) = {c1:.3f}.  Δ vs A5 = {c1 - a5:+.3f}.")
    out.append(f"- C2.2 (K=INT4, V=BF16) = {c2:.3f}.  Δ vs A5 = {c2 - a5:+.3f}.")
    out.append("")
    out.append("**Symmetric question** — at INT2:")
    out.append(f"- C2.3 (K=BF16, V=INT2) = {c3:.3f}.  Δ vs A7 = {c3 - a7:+.3f}.")
    out.append(f"- C2.4 (K=INT2, V=BF16) = {c4:.3f}.  Δ vs A7 = {c4 - a7:+.3f}.")
    out.append("")

    # Auto-classify
    def diagnose(c_kv_bf16: float, c_kbf16_v: float, floor: float, ceiling: float, midpoint: float):
        # c_kv_bf16: K-quantized side; c_kbf16_v: V-quantized side
        # Wait — variable names confusing. Re-read:
        #   C2.1 has K=BF16, V=INT4: keeps K intact, quantizes V → if this rescues, V quant alone is benign
        #   C2.2 has K=INT4, V=BF16: keeps V intact, quantizes K → if this rescues, K quant alone is benign
        # So:
        #   c_kept_K = C2.1 (K stays BF16, only V is quantized)
        #   c_kept_V = C2.2 (V stays BF16, only K is quantized)
        c_kept_K, c_kept_V = c_kv_bf16, c_kbf16_v
        kK_rescues = c_kept_K > midpoint
        kV_rescues = c_kept_V > midpoint
        if kK_rescues and not kV_rescues:
            return ("V is quantizable, K is the killer", "K-fragile")
        if kV_rescues and not kK_rescues:
            return ("K is quantizable, V is the killer", "V-fragile")
        if kK_rescues and kV_rescues:
            return ("both sides recoverable when isolated; uniform quantization compounds them", "joint-interaction")
        return ("neither side rescues even when the other is held at BF16; per-side fragility is the bottleneck", "per-side fragility (both)")

    if not any(np.isnan(x) for x in [c1, c2, bf16, a5, midpoint_int4]):
        verdict_int4, label_int4 = diagnose(c1, c2, a5, bf16, midpoint_int4)
        out.append(f"**Diagnosis @ INT4** — {label_int4}: {verdict_int4}.")
    if not any(np.isnan(x) for x in [c3, c4, bf16, a7, midpoint_int2]):
        verdict_int2, label_int2 = diagnose(c3, c4, a7, bf16, midpoint_int2)
        out.append(f"**Diagnosis @ INT2** — {label_int2}: {verdict_int2}.")
    out.append("")

    text = "\n".join(out)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
