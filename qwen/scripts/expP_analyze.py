"""Exp P analyzer — generates summary / paired McNemar / verdict matrix.

Reads `expP_rollouts.jsonl` and emits three Markdown files:
  - expP_summary.md         : per-condition accuracy ± 95% bootstrap CI,
                              needle-hit, page-read fraction, latency.
  - expP_paired.md          : McNemar χ² for the load-bearing pairs.
  - expP_verdict_matrix.md  : pass/fail tags per condition, headline call on direction.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------- IO ----------------

def load_rollouts(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not jsonl_path.exists():
        return rows
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def group_by_condition(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["condition"]].append(r)
    return out


def map_by_item(rs: list[dict]) -> dict[str, dict]:
    """Map item_id -> last seen row for this condition."""
    return {r["item_id"]: r for r in rs}


# ---------------- statistics ----------------

def bootstrap_ci(values: list[float], n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) for the supplied list of 0/1 (or float) values."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = arr[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """McNemar χ² with continuity correction. Returns (chi2, p)."""
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # p from χ²₁ via complementary error function:
    # p = 1 - CDF_chi2_1(chi2)  ;  for χ²₁ this equals erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return float(chi2), float(p)


def paired_mcnemar(a_rows: dict[str, dict], b_rows: dict[str, dict]) -> dict:
    """Compute McNemar over paired (a, b) item_id correctness."""
    keys = sorted(set(a_rows) & set(b_rows))
    b_a_correct_only = 0  # A correct, B wrong
    b_b_correct_only = 0  # A wrong, B correct
    n_paired = 0
    for k in keys:
        a_c = bool(a_rows[k].get("is_correct"))
        b_c = bool(b_rows[k].get("is_correct"))
        if a_c == b_c:
            continue
        if a_c and not b_c:
            b_a_correct_only += 1
        else:
            b_b_correct_only += 1
        n_paired += 1
    chi2, p = mcnemar(b_a_correct_only, b_b_correct_only)
    return {
        "n_paired": n_paired,
        "a_only_correct": b_a_correct_only,
        "b_only_correct": b_b_correct_only,
        "chi2": chi2,
        "p": p,
        "favored": "A" if b_a_correct_only > b_b_correct_only else (
            "B" if b_b_correct_only > b_a_correct_only else "tie"
        ),
    }


# ---------------- per-condition summary ----------------

def cond_metrics(rs: list[dict]) -> dict:
    n = len(rs)
    accs = [int(bool(r.get("is_correct"))) for r in rs]
    acc_mean, lo, hi = bootstrap_ci(accs) if accs else (float("nan"),) * 3
    nhit = [r.get("needle_in_active_layer_mean") for r in rs
            if r.get("needle_in_active_layer_mean") is not None]
    prf = [r.get("page_read_fraction") for r in rs
           if r.get("page_read_fraction") is not None]
    lat = [r.get("latency_ms") for r in rs if r.get("latency_ms") is not None]
    seq_lens = [r.get("seq_len") for r in rs if r.get("seq_len") is not None]
    nrank = [r.get("needle_rank_median") for r in rs
             if r.get("needle_rank_median") is not None]
    return {
        "n": n,
        "acc_mean": acc_mean,
        "acc_lo": lo,
        "acc_hi": hi,
        "needle_hit_mean": float(np.mean(nhit)) if nhit else None,
        "page_read_frac_mean": float(np.mean(prf)) if prf else None,
        "needle_rank_median_overall": float(np.median(nrank)) if nrank else None,
        "latency_ms_mean": float(np.mean(lat)) if lat else None,
        "seq_len_mean": float(np.mean(seq_lens)) if seq_lens else None,
    }


# ---------------- writers ----------------

def write_summary(rows: list[dict], out_md: Path) -> None:
    by_cond = group_by_condition(rows)
    lines = [f"# Exp P summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append(f"Total rows: {len(rows)} across {len(by_cond)} conditions.")
    lines.append("")
    lines.append("> `logical_page_read` = fraction of routable visual pages kept active per the routing decision. This is the *implied* sparsity, not measured bandwidth or runtime — both sparse and FormatBook routes still run dense SDPA underneath, with the cold pages masked or downgraded only at the last query row.")
    lines.append("")
    lines.append("| condition | n | acc | 95% CI | needle_hit (layer-avg) | logical_page_read | needle_rank median | latency_ms | seq_len mean |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    cond_order = [
        "P0", "P1", "P2", "P2b",
        "P3", "P3b",
        "P4", "P4_s1", "P4_s2", "P4b",
        "P5", "P5_only",
        "P6", "P6R", "P6O",
    ]
    seen = set()
    for c in cond_order + sorted(set(by_cond) - set(cond_order)):
        if c not in by_cond or c in seen:
            continue
        seen.add(c)
        m = cond_metrics(by_cond[c])
        nhit = f"{m['needle_hit_mean']:.3f}" if m["needle_hit_mean"] is not None else "—"
        prf = f"{m['page_read_frac_mean']:.3f}" if m["page_read_frac_mean"] is not None else "—"
        nr = f"{m['needle_rank_median_overall']:.1f}" if m["needle_rank_median_overall"] is not None else "—"
        lat = f"{m['latency_ms_mean']:.0f}" if m["latency_ms_mean"] is not None else "—"
        sl = f"{m['seq_len_mean']:.0f}" if m["seq_len_mean"] is not None else "—"
        lines.append(
            f"| {c} | {m['n']} | {m['acc_mean']:.3f} | "
            f"[{m['acc_lo']:.3f}, {m['acc_hi']:.3f}] | {nhit} | {prf} | {nr} | {lat} | {sl} |"
        )
    # Per-bucket breakdown
    lines.append("")
    lines.append("## Per-bucket accuracy")
    lines.append("")
    lines.append("| condition | short | mid | long |")
    lines.append("|---|---|---|---|")
    for c in cond_order + sorted(set(by_cond) - set(cond_order)):
        if c not in by_cond:
            continue
        by_bucket: dict[str, list[int]] = defaultdict(list)
        for r in by_cond[c]:
            by_bucket[r.get("context_length_bucket", "—")].append(int(bool(r.get("is_correct"))))
        s, m, l = (
            f"{np.mean(by_bucket['short']):.3f} (n={len(by_bucket['short'])})" if by_bucket["short"] else "—",
            f"{np.mean(by_bucket['mid']):.3f} (n={len(by_bucket['mid'])})" if by_bucket["mid"] else "—",
            f"{np.mean(by_bucket['long']):.3f} (n={len(by_bucket['long'])})" if by_bucket["long"] else "—",
        )
        lines.append(f"| {c} | {s} | {m} | {l} |")

    out_md.write_text("\n".join(lines) + "\n")


PAIRS = [
    ("P1", "P0", "F4 same-benchmark anchor: F4 vs BF16 dense"),
    ("P2", "P0", "F9 same-benchmark anchor: F9 vs BF16 dense"),
    ("P3", "P4", "Quest vs Random at top-25% sparse"),
    ("P5", "P3", "Oracle (budget-matched) headroom over Quest"),
    ("P5_only", "P3", "Oracle (needle-only) vs Quest at top-25% — does needle alone suffice?"),
    ("P6", "P2", "FormatBook (Quest top-50%) vs dense F9"),
    ("P6", "P6R", "FormatBook Quest vs Random — does Quest selection matter?"),
    ("P6O", "P6", "FormatBook Oracle headroom over Quest"),
    ("P6", "P1", "FormatBook vs dense F4"),
    ("P2b", "P2", "J12 (INT8 sidecode) vs F9 dense — INT8 confound check"),
]


def write_paired(rows: list[dict], out_md: Path) -> None:
    by_cond = group_by_condition(rows)
    rows_by_item = {c: map_by_item(rs) for c, rs in by_cond.items()}
    lines = [f"# Exp P paired McNemar — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append("| pair | description | n_paired | A_only | B_only | χ² | p | favored |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for a, b, desc in PAIRS:
        if a not in rows_by_item or b not in rows_by_item:
            lines.append(f"| {a} vs {b} | {desc} | — | — | — | — | — | (missing) |")
            continue
        m = paired_mcnemar(rows_by_item[a], rows_by_item[b])
        lines.append(
            f"| {a} vs {b} | {desc} | {m['n_paired']} | {m['a_only_correct']} | "
            f"{m['b_only_correct']} | {m['chi2']:.2f} | {m['p']:.4f} | {m['favored']} |"
        )
    # Add stretch pairs if present
    if "P3b" in by_cond and "P4b" in by_cond:
        m = paired_mcnemar(rows_by_item["P3b"], rows_by_item["P4b"])
        lines.append(
            f"| P3b vs P4b | Quest vs Random at top-50% (stretch) | {m['n_paired']} | "
            f"{m['a_only_correct']} | {m['b_only_correct']} | {m['chi2']:.2f} | "
            f"{m['p']:.4f} | {m['favored']} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def paired_needle_hit_p3_vs_p4(p3_rows: dict[str, dict],
                                p4_rows: dict[str, dict]) -> dict:
    """Compare per-item needle-hit (Quest top-K vs Random top-K) on items that
    appear in BOTH conditions. Reports:
      - mean per-item delta in layer-averaged needle hit (P3 - P4)
      - paired sign test on majority-vote per-item needle hits (binary)
      - n routable-pages > 1 (the only items where routing is non-trivial)
    """
    keys = sorted(set(p3_rows) & set(p4_rows))
    deltas: list[float] = []
    p3_majority_wins = 0
    p4_majority_wins = 0
    ties = 0
    n_nontrivial = 0  # items where the routing choice mattered (>1 routable)
    for k in keys:
        a = p3_rows[k]
        b = p4_rows[k]
        # Layer-averaged needle hit fraction
        ah = a.get("needle_in_active_layer_mean")
        bh = b.get("needle_in_active_layer_mean")
        if ah is None or bh is None:
            continue
        # Skip items where there was no real routing choice (≤1 routable page →
        # active set is forced to contain the needle trivially).
        n_routable = (
            (a.get("active_routable_pages_layer_mean") or 0) +
            (a.get("cold_routable_pages_layer_mean") or 0)
        )
        if n_routable <= 1:
            continue
        n_nontrivial += 1
        deltas.append(ah - bh)
        # Majority vote per item: was the needle in active for >50% of layers?
        a_hit = ah > 0.5
        b_hit = bh > 0.5
        if a_hit and not b_hit:
            p3_majority_wins += 1
        elif b_hit and not a_hit:
            p4_majority_wins += 1
        else:
            ties += 1
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    chi2, p = mcnemar(p3_majority_wins, p4_majority_wins)
    return {
        "n_nontrivial": n_nontrivial,
        "mean_delta": mean_delta,
        "p3_majority_wins": p3_majority_wins,
        "p4_majority_wins": p4_majority_wins,
        "ties": ties,
        "chi2": chi2,
        "p": p,
    }


def write_verdict(rows: list[dict], out_md: Path) -> None:
    by_cond = group_by_condition(rows)
    rows_by_item = {c: map_by_item(rs) for c, rs in by_cond.items()}
    # Compute key signals
    verdicts: list[str] = []
    if "P3" in rows_by_item and "P4" in rows_by_item:
        m = paired_mcnemar(rows_by_item["P3"], rows_by_item["P4"])
        if m["chi2"] >= 6.63 and m["favored"] == "A":  # p < 0.01 threshold
            verdicts.append(f"PASS: Quest > Random at top-25% accuracy (χ²={m['chi2']:.2f}, p={m['p']:.4f})")
        elif m["chi2"] >= 3.84 and m["favored"] == "A":  # p < 0.05
            verdicts.append(f"TREND: Quest > Random at top-25% accuracy (χ²={m['chi2']:.2f}, p={m['p']:.4f})")
        else:
            verdicts.append(f"NEUTRAL: Quest ≈ Random at top-25% accuracy (χ²={m['chi2']:.2f}, p={m['p']:.4f})")

        # Needle-hit head-to-head (the primary signal the plan calls for).
        nh = paired_needle_hit_p3_vs_p4(rows_by_item["P3"], rows_by_item["P4"])
        verdicts.append(
            f"Quest vs Random needle-hit (n_nontrivial={nh['n_nontrivial']}): "
            f"mean ΔP3−P4 = {nh['mean_delta']:+.3f}; majority-vote wins "
            f"P3={nh['p3_majority_wins']} P4={nh['p4_majority_wins']} ties={nh['ties']} "
            f"(McNemar χ²={nh['chi2']:.2f}, p={nh['p']:.4f})"
        )

    if "P6" in rows_by_item and "P2" in rows_by_item:
        m = paired_mcnemar(rows_by_item["P6"], rows_by_item["P2"])
        if m["chi2"] < 3.84:
            verdicts.append(f"PASS: FormatBook matches J12 (χ²={m['chi2']:.2f}, p={m['p']:.4f}, favored={m['favored']})")
        else:
            verdicts.append(f"GAP: FormatBook ≠ J12 (χ²={m['chi2']:.2f}, p={m['p']:.4f}, favored={m['favored']})")

    if "P1" in by_cond:
        p1_acc = np.mean([int(bool(r.get("is_correct"))) for r in by_cond["P1"]])
        p0_acc = np.mean([int(bool(r.get("is_correct"))) for r in by_cond.get("P0", [])])
        if not np.isnan(p0_acc):
            delta = p0_acc - p1_acc
            verdicts.append(
                f"MM-NIAH quant sensitivity: P0 acc={p0_acc:.3f}, P1 acc={p1_acc:.3f}, "
                f"Δ={delta:+.3f} {'(F4 generalizes from LVB)' if abs(delta) < 0.05 else '(F4 hits MM-NIAH harder than LVB)'}"
            )

    lines = [f"# Exp P verdict matrix — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append("## Headline signals")
    lines.extend(f"- {v}" for v in verdicts)
    lines.append("")
    lines.append("## Per-condition status")
    lines.append("| condition | n | acc | status |")
    lines.append("|---|---|---|---|")
    cond_order = [
        "P0", "P1", "P2", "P2b",
        "P3", "P3b",
        "P4", "P4_s1", "P4_s2", "P4b",
        "P5", "P5_only",
        "P6", "P6R", "P6O",
    ]
    for c in cond_order:
        if c not in by_cond:
            continue
        rs = by_cond[c]
        n = len(rs)
        acc = np.mean([int(bool(r.get("is_correct"))) for r in rs])
        status = "ANCHOR" if c in ("P0", "P1", "P2") else (
            "SPARSE" if c in ("P3", "P3b", "P4", "P4b") else (
                "ORACLE" if c == "P5" else "PROPOSED"
            )
        )
        lines.append(f"| {c} | {n} | {acc:.3f} | {status} |")
    out_md.write_text("\n".join(lines) + "\n")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", type=Path, default=RESULTS_DIR / "expP_rollouts.jsonl")
    ap.add_argument("--out-summary", type=Path, default=RESULTS_DIR / "expP_summary.md")
    ap.add_argument("--out-paired", type=Path, default=RESULTS_DIR / "expP_paired.md")
    ap.add_argument("--out-verdict", type=Path, default=RESULTS_DIR / "expP_verdict_matrix.md")
    args = ap.parse_args()

    rows = load_rollouts(args.in_jsonl)
    if not rows:
        print(f"no rows in {args.in_jsonl}", flush=True)
        return
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    write_summary(rows, args.out_summary)
    write_paired(rows, args.out_paired)
    write_verdict(rows, args.out_verdict)
    print(f"wrote {args.out_summary}\nwrote {args.out_paired}\nwrote {args.out_verdict}",
          flush=True)


if __name__ == "__main__":
    main()
