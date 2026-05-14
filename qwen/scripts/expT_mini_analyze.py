"""Exp T-mini analyzer: bucketed headline + paired McNemar matrices.

Reads a per-condition JSONL (produced by expQ_driver in --exp-t-mini /
--exp-t-mini-counting modes) and emits:

  * Headline accuracy / KV-bits table — primary headline is the long +
    multi-image + image-needle slice; full-overall is also reported.
  * Per-bucket breakdown by context-length, num_images, needle depth.
  * Paired McNemar matrices for the load-bearing comparisons (PageLocal,
    PageSentinel, anchors).
  * Counting-image extra metrics: valid-format rate, length-match rate,
    soft accuracy, sum-match rate.

Usage:
    python3 expT_mini_analyze.py \
        --in-jsonl results/expT_mini_rollouts_retrieval-image.jsonl \
        --out-md results/expT_mini_summary_retrieval-image.md \
        --task retrieval-image
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ---------------- IO ----------------

def load_rollouts(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


# ---------------- bucketing ----------------

CTX_BUCKETS = [
    ("0-8K",   0,     8_000),
    ("8-16K",  8_000, 16_000),
    ("16-24K", 16_000, 24_000),
    ("24-32K", 24_000, 32_001),
]
NIMG_BUCKETS = [("1-4", 1, 5), ("5-8", 5, 9), ("9+", 9, 10**9)]
DEPTH_BUCKETS = [("early", 0.0, 0.34), ("mid", 0.34, 0.67), ("late", 0.67, 1.01)]


def _bucket(value, ranges):
    for name, lo, hi in ranges:
        if lo <= value < hi:
            return name
    return "other"


def ctx_bucket(r: dict) -> str:
    return _bucket(int(r.get("context_length", 0)), CTX_BUCKETS)


def nimg_bucket(r: dict) -> str:
    return _bucket(int(r.get("num_images", 0)), NIMG_BUCKETS)


def depth_bucket(r: dict) -> str:
    return _bucket(float(r.get("placed_depth", 0.0)), DEPTH_BUCKETS)


# ---------------- accuracy / stats ----------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cond_accuracy(rows: list[dict]) -> tuple[int, int, float]:
    correct = sum(1 for r in rows if r.get("is_correct"))
    return correct, len(rows), (correct / max(1, len(rows)))


def mean(values: list[float]) -> Optional[float]:
    vs = [v for v in values if v is not None]
    return sum(vs) / len(vs) if vs else None


# ---------------- paired McNemar ----------------

def paired_mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Paired by item_id; report n10 (a-only), n01 (b-only), and McNemar chi^2."""
    by_id_a = {r["item_id"]: r for r in rows_a}
    by_id_b = {r["item_id"]: r for r in rows_b}
    common = set(by_id_a) & set(by_id_b)
    n10 = n01 = n11 = n00 = 0
    for iid in common:
        a = bool(by_id_a[iid].get("is_correct"))
        b = bool(by_id_b[iid].get("is_correct"))
        if a and not b:
            n10 += 1
        elif b and not a:
            n01 += 1
        elif a and b:
            n11 += 1
        else:
            n00 += 1
    discordant = n10 + n01
    chi2 = (((n10 - n01) ** 2) / discordant) if discordant > 0 else 0.0
    p_approx = math.exp(-chi2 / 2) if chi2 > 0 else 1.0  # crude bound
    return {
        "n_paired": len(common), "n10": n10, "n01": n01, "n11": n11, "n00": n00,
        "discordant": discordant, "chi2_continuity": chi2,
        "p_approx": p_approx,
        "net": n10 - n01,
    }


# ---------------- counting-image extras ----------------

def counting_summary(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n": n,
        "exact_match_rate": sum(1 for r in rows if r.get("exact_match")) / n,
        "valid_format_rate": sum(1 for r in rows if r.get("valid_format")) / n,
        "length_match_rate": sum(1 for r in rows if r.get("length_match")) / n,
        "sum_match_rate": sum(1 for r in rows if r.get("sum_match")) / n,
        "missing_format_rate": sum(1 for r in rows if r.get("missing_format")) / n,
        "mean_soft_accuracy": mean([r.get("soft_accuracy", 0.0) for r in rows]) or 0.0,
    }


# ---------------- pair definitions ----------------

def pair_definitions(task: str) -> list[tuple[str, str, str]]:
    """(label, A, B) triples for which we compute paired McNemar."""
    if task == "counting-image":
        # C-* names. Use C8 (PageSentinel-4) and C6 (PageLocal-F4) as anchors.
        return [
            ("PageLocal vs F4",                 "C6", "C1"),
            ("PageLocal vs TokenBlock",         "C6", "C5"),
            ("PageLocal vs F9",                 "C6", "C2"),
            ("PageSentinel-4 vs F4",            "C8", "C1"),
            ("PageSentinel-4 vs RandomSentinel-4", "C8", "C9"),
            ("PageSentinel-4 vs LastSentinel-4",   "C8", "C10"),
            ("PageSentinel-4 vs TextSentinel-4",   "C8", "C11"),
            ("PageSentinel-4 vs PageLocal",        "C8", "C6"),
            ("Combined vs PageLocal",              "C12", "C6"),
            ("Combined vs PageSentinel-4",         "C12", "C8"),
            ("Combined vs F9",                     "C12", "C2"),
            ("F4 vs BF16",                         "C1", "C0"),
            ("F9 vs BF16",                         "C2", "C0"),
        ]
    return [
        ("PageLocal vs F4",                    "T8", "T1"),
        ("PageLocal vs TokenBlock",            "T8", "T6"),
        ("PageLocal vs RandomPageLocal",       "T8", "T7"),
        ("PageLocal vs TextVisualLocal",       "T8", "T5"),
        ("ImageOnlyLocal vs TextOnlyLocal",    "T9", "T10"),
        ("Combined (T16) vs PageLocal (T8)",   "T16", "T8"),
        ("PageSentinel-4 vs RandomSentinel-4", "T12", "T13"),
        ("PageSentinel-4 vs LastSentinel-4",   "T12", "T14"),
        ("PageSentinel-4 vs TextSentinel-4",   "T12", "T15"),
        ("PageSentinel-4 vs F4",               "T12", "T1"),
        ("Combined (T16) vs PageSentinel-4",   "T16", "T12"),
        ("F4 vs BF16",                         "T1", "T0"),
        ("F9 vs BF16",                         "T2", "T0"),
        ("SJ vs F9",                           "T3", "T2"),
        ("S4 vs F9",                           "T4", "T2"),
        ("PageLocal vs F9",                    "T8", "T2"),
        ("Combined vs F9",                     "T16", "T2"),
    ]


# ---------------- formatter ----------------

def fmt_acc_row(name: str, rows: list[dict]) -> str:
    n_corr, n, acc = cond_accuracy(rows)
    lo, hi = wilson_ci(n_corr, n)
    ekvb = mean([r.get("effective_kv_bits") for r in rows])
    ekb = mean([r.get("effective_k_bits") for r in rows])
    ekvb_s = f"{ekvb:.3f}" if ekvb is not None else "—"
    ekb_s = f"{ekb:.3f}" if ekb is not None else "—"
    lat = mean([r.get("latency_ms", 0.0) for r in rows])
    lat_s = f"{lat:.0f}" if lat is not None else "—"
    return (f"| {name} | {n} | {acc:.3f} | [{lo:.3f}, {hi:.3f}] | "
            f"{ekb_s} | {ekvb_s} | {lat_s} |")


def write_summary_md(rows: list[dict], out_md: Path, task: str) -> None:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    cond_order = sorted(by_cond.keys())
    is_counting = (task == "counting-image")

    lines = [f"# Exp T-mini summary — {task}",
             f"_Generated: {Path(out_md).name}_", "",
             f"Total rows: {len(rows)} across {len(by_cond)} conditions", ""]

    # Headline 1: long + multi-image + image-needle slice (primary).
    if not is_counting:
        # Long context, num_images >= 5, image-needle (all rows here are image-needle by task).
        long_multi = [r for r in rows
                      if r.get("context_length", 0) >= 16_000 and r.get("num_images", 0) >= 5]
        lines.append("## Headline 1 — long + multi-image (context ≥ 16K, num_images ≥ 5)")
        lines.append("")
        if long_multi:
            lines.append("| condition | n | acc | 95% CI | K bits | KV bits | latency ms |")
            lines.append("|---|---|---|---|---|---|---|")
            grp: dict[str, list[dict]] = defaultdict(list)
            for r in long_multi:
                grp[r["condition"]].append(r)
            for c in cond_order:
                if c in grp:
                    lines.append(fmt_acc_row(c, grp[c]))
        else:
            lines.append("_no rows match long+multi filter_")
        lines.append("")

    # Headline 2: overall.
    lines.append("## Headline 2 — overall accuracy and KV bits")
    lines.append("")
    lines.append("| condition | n | acc | 95% CI | K bits | KV bits | latency ms |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in cond_order:
        lines.append(fmt_acc_row(c, by_cond[c]))
    lines.append("")

    # Per-bucket breakdown.
    lines.append("## Per-bucket accuracy")
    for bucket_name, bucket_fn, bucket_def in [
        ("context-length", ctx_bucket, CTX_BUCKETS),
        ("num_images",     nimg_bucket, NIMG_BUCKETS),
        ("needle-depth",   depth_bucket, DEPTH_BUCKETS),
    ]:
        lines.append(f"### by {bucket_name}")
        lines.append("")
        bk_names = [b[0] for b in bucket_def]
        lines.append("| condition | " + " | ".join(f"{b} (n / acc)" for b in bk_names) + " |")
        lines.append("|" + "---|" * (1 + len(bk_names)))
        for c in cond_order:
            cells = []
            for bk in bk_names:
                rs = [r for r in by_cond[c] if bucket_fn(r) == bk]
                if rs:
                    corr, n, _ = cond_accuracy(rs)
                    cells.append(f"{n} / {corr/max(1,n):.3f}")
                else:
                    cells.append("—")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")

    # Counting-image extras.
    if is_counting:
        lines.append("## Counting-image extra metrics")
        lines.append("")
        lines.append("| condition | n | exact | valid_format | length_match | sum_match | "
                     "missing_format | mean soft_acc |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for c in cond_order:
            cs = counting_summary(by_cond[c])
            if cs:
                lines.append(
                    f"| {c} | {cs['n']} | {cs['exact_match_rate']:.3f} | "
                    f"{cs['valid_format_rate']:.3f} | {cs['length_match_rate']:.3f} | "
                    f"{cs['sum_match_rate']:.3f} | {cs['missing_format_rate']:.3f} | "
                    f"{cs['mean_soft_accuracy']:.3f} |"
                )
        lines.append("")

        # Counting BF16-correct preservation.
        if "C0" in by_cond:
            bf16_correct_ids = {r["item_id"] for r in by_cond["C0"] if r.get("is_correct")}
            lines.append("## BF16-correct preservation (counting-image)")
            lines.append("")
            lines.append(f"BF16 correct on {len(bf16_correct_ids)} items.")
            lines.append("")
            lines.append("| condition | n_preserved / n_bf16_correct | preservation_rate |")
            lines.append("|---|---|---|")
            for c in cond_order:
                if c == "C0":
                    continue
                rows_c = [r for r in by_cond[c] if r["item_id"] in bf16_correct_ids]
                if not rows_c:
                    continue
                preserved = sum(1 for r in rows_c if r.get("is_correct"))
                lines.append(f"| {c} | {preserved} / {len(rows_c)} | "
                             f"{preserved/max(1,len(rows_c)):.3f} |")
            lines.append("")

    # Paired McNemar matrix.
    lines.append("## Paired McNemar matrix")
    lines.append("")
    lines.append("| comparison | n_paired | n10 (A only) | n01 (B only) | discordant | net | chi^2 |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, a, b in pair_definitions(task):
        if a not in by_cond or b not in by_cond:
            continue
        m = paired_mcnemar(by_cond[a], by_cond[b])
        lines.append(f"| **{label}** (A={a}, B={b}) | {m['n_paired']} | {m['n10']} | "
                     f"{m['n01']} | {m['discordant']} | {m['net']:+d} | {m['chi2_continuity']:.3f} |")
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--task",
                    choices=("retrieval-image", "reasoning-image", "counting-image"),
                    required=True)
    args = ap.parse_args()
    rows = load_rollouts(args.in_jsonl)
    if not rows:
        print(f"[warn] no rows loaded from {args.in_jsonl}")
        return
    write_summary_md(rows, args.out_md, args.task)
    print(f"wrote {args.out_md} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
