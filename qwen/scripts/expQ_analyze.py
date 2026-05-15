"""Exp Q analyzer — summary, paired McNemar, verdict matrix, branch JSON.

Reads `expQ_rollouts_slice{A|B}.jsonl` and emits four Markdown files:

  - expQ_summary_slice{A|B}.md       Per-condition acc + 95% CI + effective_kv_bits +
                                     effective_k_bits + f9_sidecode_token_fraction +
                                     needle_hit + logical_page_read + latency (diagnostic),
                                     plus per-bucket + per-num_images breakdowns
                                     (8-11 / 12-19 / 20+).
  - expQ_paired_slice{A|B}.md        Paired McNemar χ² for load-bearing pairs.
  - expQ_verdict_matrix_slice{A|B}.md   Pass/fail tags per condition + headline call
                                       per the Exp Q interpretation tree, plus a
                                       Slice B recommendation (for Slice A only).
  - expQ_branch_slice{A|B}.json      Machine-readable for run_expQ_overnight.sh
                                     branching rule. Fields:
                                       need_q5_seeds, need_q8_seeds, need_q11_seeds
                                       slice_b_recommendation: "RUN" | "DEFER"
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
    return {r["item_id"]: r for r in rs}


# ---------------- statistics ----------------

def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
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
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return float(chi2), float(p)


def paired_mcnemar(a_rows: dict[str, dict], b_rows: dict[str, dict]) -> dict:
    keys = sorted(set(a_rows) & set(b_rows))
    b_a_correct_only = 0
    b_b_correct_only = 0
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

    def collect(key):
        return [r.get(key) for r in rs if r.get(key) is not None]

    nhit = collect("needle_in_active_layer_mean")
    prf = collect("page_read_fraction")
    lat = collect("latency_ms")
    seq_lens = collect("seq_len")
    nrank = collect("needle_rank_median")
    ekvb = collect("effective_kv_bits")
    ekb = collect("effective_k_bits")
    sidetok = collect("f9_sidecode_token_fraction")
    sidepg = collect("f9_sidecode_page_fraction")
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
        "effective_kv_bits_mean": float(np.mean(ekvb)) if ekvb else None,
        "effective_k_bits_mean": float(np.mean(ekb)) if ekb else None,
        "f9_sidecode_token_fraction_mean": float(np.mean(sidetok)) if sidetok else None,
        "f9_sidecode_page_fraction_mean": float(np.mean(sidepg)) if sidepg else None,
    }


def _fmt(x, prec=3, dash="—"):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return dash
    return f"{x:.{prec}f}"


# ---------------- num_images buckets ----------------

NUM_IMG_BUCKETS = [
    ("8-11", 8, 12),
    ("12-19", 12, 20),
    ("20+", 20, 10_000_000),
]


def num_img_bucket(n: int) -> str:
    for name, lo, hi in NUM_IMG_BUCKETS:
        if lo <= n < hi:
            return name
    return "<8"


# ---------------- writers ----------------

def slice_cond_order(slice_tag: str) -> list[str]:
    if slice_tag == "A":
        return [
            "Q0", "Q1", "Q2", "Q3",
            "Q4", "Q5", "Q5_s1", "Q5_s2", "Q6",
            "Q7", "Q8", "Q8_s1", "Q8_s2", "Q9",
            "Q10", "Q11", "Q11_s1", "Q11_s2",
            "Q_allhot",
        ]
    if slice_tag == "B":
        return ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
    if slice_tag == "C":
        # Exp R Sub-experiment C ordering.
        return [
            "C0", "C1", "C2", "C3", "C3b",
            "C4", "C5", "C6",
            "C7", "C8",
            "S4", "S8", "S12", "SJ",
        ]
    if slice_tag == "S":
        # Exp S Phase 1 sidecode ladder.
        return ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
    if slice_tag == "U":
        # Exp U1 residual channel oracle/policy screen.
        return ["U0", "U1", "U2", "U3", "U4", "U5",
                "U6", "U7", "U8", "U9", "U10",
                "U11", "U12", "U13"]
    return []


def write_summary(rows: list[dict], out_md: Path, slice_tag: str) -> None:
    by_cond = group_by_condition(rows)
    lines = [f"# Exp Q summary slice {slice_tag} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append(f"Total rows: {len(rows)} across {len(by_cond)} conditions.")
    lines.append("")
    lines.append("> Metric semantics — `effective_k_bits` and `effective_kv_bits` capture the *storage* "
                 "format. `f9_sidecode_token_fraction` is what FormatBook trades against; lower is "
                 "cheaper sidecode. `logical_page_read` captures *read* reduction for sparse routes "
                 "(should be 1.0 for FormatBook since dense attention still runs). Latency is "
                 "diagnostic-only — dense attention runs throughout this fake implementation.")
    lines.append("")
    lines.append("| condition | n | acc | 95% CI | eff_kv_bits | eff_k_bits | f9_sidecode_tok | "
                 "f9_sidecode_pg | logical_page_read | needle_hit | needle_rank med | latency_ms |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    cond_order = slice_cond_order(slice_tag)
    seen = set()
    for c in cond_order + sorted(set(by_cond) - set(cond_order)):
        if c not in by_cond or c in seen:
            continue
        seen.add(c)
        m = cond_metrics(by_cond[c])
        lines.append(
            f"| {c} | {m['n']} | {_fmt(m['acc_mean'])} | "
            f"[{_fmt(m['acc_lo'])}, {_fmt(m['acc_hi'])}] | "
            f"{_fmt(m['effective_kv_bits_mean'])} | {_fmt(m['effective_k_bits_mean'])} | "
            f"{_fmt(m['f9_sidecode_token_fraction_mean'])} | "
            f"{_fmt(m['f9_sidecode_page_fraction_mean'])} | "
            f"{_fmt(m['page_read_frac_mean'])} | "
            f"{_fmt(m['needle_hit_mean'])} | "
            f"{_fmt(m['needle_rank_median_overall'], prec=1)} | "
            f"{_fmt(m['latency_ms_mean'], prec=0)} |"
        )

    # Per-bucket (context length: short/mid/long)
    lines.append("")
    lines.append("## Per-context-length-bucket accuracy")
    lines.append("")
    lines.append("| condition | short | mid | long |")
    lines.append("|---|---|---|---|")
    for c in cond_order + sorted(set(by_cond) - set(cond_order)):
        if c not in by_cond:
            continue
        by_bucket: dict[str, list[int]] = defaultdict(list)
        for r in by_cond[c]:
            by_bucket[r.get("context_length_bucket", "—")].append(int(bool(r.get("is_correct"))))
        s_str = f"{np.mean(by_bucket['short']):.3f} (n={len(by_bucket['short'])})" if by_bucket["short"] else "—"
        m_str = f"{np.mean(by_bucket['mid']):.3f} (n={len(by_bucket['mid'])})" if by_bucket["mid"] else "—"
        l_str = f"{np.mean(by_bucket['long']):.3f} (n={len(by_bucket['long'])})" if by_bucket["long"] else "—"
        lines.append(f"| {c} | {s_str} | {m_str} | {l_str} |")

    # Per-num_images bucket (new)
    lines.append("")
    lines.append("## Per-num_images breakdown (8-11 / 12-19 / 20+)")
    lines.append("")
    lines.append("Tests whether routing only helps in the long-context regime where the budget actually binds.")
    lines.append("")
    lines.append("| condition | 8-11 imgs | 12-19 imgs | 20+ imgs |")
    lines.append("|---|---|---|---|")
    for c in cond_order + sorted(set(by_cond) - set(cond_order)):
        if c not in by_cond:
            continue
        by_nimg: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for r in by_cond[c]:
            n_img = r.get("num_images") or 0
            bucket = num_img_bucket(int(n_img))
            ekvb = r.get("effective_kv_bits")
            by_nimg[bucket].append((int(bool(r.get("is_correct"))),
                                    float(ekvb) if ekvb is not None else float("nan")))

        def cell(rows):
            if not rows:
                return "—"
            acc = np.mean([r[0] for r in rows])
            ekvb_vals = [r[1] for r in rows if not math.isnan(r[1])]
            ekvb_avg = np.mean(ekvb_vals) if ekvb_vals else float("nan")
            return f"acc={acc:.3f} kv_bits={_fmt(ekvb_avg)} (n={len(rows)})"

        lines.append(f"| {c} | {cell(by_nimg['8-11'])} | {cell(by_nimg['12-19'])} | {cell(by_nimg['20+'])} |")

    # ---------------- Pareto frontier (Exp R section) ----------------
    lines.append("")
    lines.append("## Pareto frontier (accuracy vs effective_kv_bits)")
    lines.append("")
    lines.append("Pareto-optimal: no other condition has BOTH higher accuracy AND lower KV-bits.")
    lines.append("")
    lines.append("| condition | n | acc | eff_kv_bits | eff_k_bits | on Pareto front? |")
    lines.append("|---|---|---|---|---|---|")
    # Build the (cond, acc, kv_bits) list across all conditions for which we have valid metrics.
    cond_points: list[tuple[str, int, float, float, float]] = []
    for c in cond_order + sorted(set(by_cond) - set(cond_order)):
        if c not in by_cond:
            continue
        m = cond_metrics(by_cond[c])
        if m["acc_mean"] is None or m["effective_kv_bits_mean"] is None:
            continue
        if math.isnan(m["acc_mean"]) or math.isnan(m["effective_kv_bits_mean"]):
            continue
        cond_points.append((
            c, m["n"], float(m["acc_mean"]),
            float(m["effective_kv_bits_mean"]),
            float(m["effective_k_bits_mean"]) if m["effective_k_bits_mean"] is not None else float("nan"),
        ))
    # Pareto: a point (acc, kvb) is on the frontier if no other point has
    # acc' >= acc AND kvb' <= kvb (with at least one strict inequality).
    pareto_set = set()
    for i, (ci, _, acci, kvbi, _) in enumerate(cond_points):
        dominated = False
        for j, (cj, _, accj, kvbj, _) in enumerate(cond_points):
            if i == j:
                continue
            if accj >= acci and kvbj <= kvbi and (accj > acci or kvbj < kvbi):
                dominated = True
                break
        if not dominated:
            pareto_set.add(ci)
    for c, n, acc, kvb, kb in cond_points:
        on_front = "**YES**" if c in pareto_set else "no"
        lines.append(f"| {c} | {n} | {acc:.3f} | {kvb:.3f} | {kb:.3f} | {on_front} |")

    out_md.write_text("\n".join(lines) + "\n")


# ---------------- paired McNemar ----------------

def pairs_slice_s() -> list[tuple[str, str, str]]:
    """Exp S Phase 1 sidecode bit-ladder load-bearing pairs.

    S2 = F9 dense (4.75 KV bits anchor); S3 = SJ = J12 INT8 sidecode (4.25);
    S4 = top-16 INT7 (4.1875); S5 = top-16 INT6 (4.125); S6 = top-16 INT5 (4.0625);
    S7 = top-24 INT6 (4.1875); S8 = top-32 INT6 (4.250); S9 = TextOnly-SJ.
    """
    return [
        ("S3", "S2", "SJ INT8 sidecode vs F9 dense — PARETO TIE TEST (Exp R replication)"),
        ("S3", "S0", "SJ vs BF16 ceiling — headroom"),
        ("S4", "S3", "INT7 sidecode vs INT8 (one step lower precision)"),
        ("S5", "S3", "INT6 sidecode vs INT8 (two steps; -0.125 KV bits)"),
        ("S5", "S4", "INT6 vs INT7 sidecode (one step lower)"),
        ("S6", "S5", "INT5 vs INT6 sidecode — does precision collapse?"),
        ("S6", "S2", "INT5 sidecode (lowest ladder point) vs F9 dense"),
        ("S5", "S2", "INT6 sidecode vs F9 dense — the Pareto candidate"),
        ("S7", "S4", "top-24 INT6 vs top-16 INT7 — SAME bits, WIDER channels"),
        ("S8", "S3", "top-32 INT6 vs top-16 INT8 — SAME bits, WIDER channels"),
        ("S9", "S2", "TextOnly-SJ vs F9 dense — visual=F4 + text=SJ"),
        ("S9", "S3", "TextOnly-SJ vs SJ dense — does visual-F4 hurt vs all-SJ?"),
        ("S2", "S0", "F9 vs BF16 anchor (sanity)"),
        ("S1", "S0", "F4 vs BF16 anchor (sanity)"),
    ]


def pairs_slice_c() -> list[tuple[str, str, str]]:
    """Exp R Sub-experiment C load-bearing paired-McNemar tests.
    Includes Exp S Phase 0 SJ-anchored pairs (added 2026-05-13)."""
    return [
        ("C4", "C2", "AllVisual-Quest vs F9 dense — PARETO TIE TEST"),
        ("C4", "C3", "AllVisual-Quest vs TextOnly — does visual routing matter?"),
        ("C4", "C3b", "AllVisual-Quest vs ChoiceOnly — CRITICAL: not choice-mediated"),
        ("C4", "C5", "AllVisual-Quest vs AllVisual-Random — Quest selection matters?"),
        ("C6", "C4", "AllVisual-Oracle headroom over Quest"),
        ("C7", "C4", "Split-Quest vs global Quest"),
        ("C7", "C8", "Split-Quest vs Split-Random"),
        ("C7", "C3b", "Split-Quest vs ChoiceOnly (parallel diagnostic)"),
        ("C4", "S8", "AllVisual-Quest vs static F8 (matched budget)"),
        ("C4", "S4", "AllVisual-Quest vs static S4 (4.19 KV bits)"),
        ("C4", "SJ", "AllVisual-Quest vs J12 (F9 INT8 sidecode)"),
        # Exp S Phase 0 — SJ-anchored sidecode-format tests on existing Exp R data
        ("SJ", "C2", "Exp S Phase 0: SJ (J12 INT8 sidecode) vs F9 dense — IS INT8 SIDECODE A REAL WIN?"),
        ("SJ", "C0", "Exp S Phase 0: SJ vs BF16 ceiling — how much accuracy is left on the table?"),
        ("SJ", "C3", "Exp S Phase 0: SJ vs C3 TextOnly — competing static recipes"),
        ("SJ", "S12", "Exp S Phase 0: SJ INT8 sidecode vs static top-12 BF16 — same bit budget, different format"),
        ("C3", "C2", "Exp S Phase 0: TextOnly vs F9 dense — is text-only F9 protection enough?"),
    ]


def pairs_slice_u() -> list[tuple[str, str, str]]:
    """Exp U1 — residual channel oracle/policy screen load-bearing pairs.

    Anchors: U2 = F9 (4.75 KV bits), U3 = S4 (4.1875 KV bits, anchor).
    Extra-8 conditions: U4 GEN, U5 RND, U6 TT, U7 TV, U8 VT, U9 VV, U10 BAL,
                        U11 MMNIAH-prior, U12 LVB-prior. All at 4.28125 KV bits.
    U13 = ALL-16 extra (4.375 KV bits).
    """
    return [
        # Anchors
        ("U2", "U0", "F9 vs BF16 ceiling (anchor sanity)"),
        ("U3", "U0", "S4 vs BF16 ceiling"),
        ("U3", "U2", "S4 (INT7, 4.1875) vs F9 (4.75) — Pareto-tie test from Exp S"),
        # Does ANY extra help over the S4 anchor?
        ("U4", "U3", "GEN extra-8 vs S4 anchor — does generic-energy extra help?"),
        ("U6", "U3", "TT extra-8 vs S4 anchor"),
        ("U7", "U3", "TV extra-8 vs S4 anchor"),
        ("U8", "U3", "VT extra-8 vs S4 anchor"),
        ("U9", "U3", "VV extra-8 vs S4 anchor"),
        ("U10", "U3", "BAL extra-8 vs S4 anchor"),
        ("U11", "U3", "MM-NIAH-prior extra-8 vs S4 anchor"),
        ("U12", "U3", "LVB-prior extra-8 vs S4 anchor"),
        ("U13", "U3", "ALL-16 extra vs S4 anchor (extra-16 at 4.375 KV bits)"),
        # Structured vs random (central question)
        ("U6", "U5", "TT vs RND extra-8 — structured selection >= random?"),
        ("U7", "U5", "TV vs RND extra-8"),
        ("U8", "U5", "VT vs RND extra-8"),
        ("U9", "U5", "VV vs RND extra-8"),
        ("U10", "U5", "BAL vs RND extra-8"),
        ("U11", "U5", "MM-NIAH-prior vs RND extra-8"),
        ("U12", "U5", "LVB-prior vs RND extra-8"),
        # Structured vs generic-energy
        ("U6", "U4", "TT vs GEN extra-8"),
        ("U7", "U4", "TV vs GEN extra-8"),
        ("U8", "U4", "VT vs GEN extra-8"),
        ("U9", "U4", "VV vs GEN extra-8"),
        ("U10", "U4", "BAL vs GEN extra-8"),
        ("U11", "U4", "MM-NIAH-prior vs GEN extra-8"),
        ("U12", "U4", "LVB-prior vs GEN extra-8"),
        # Balanced vs single-block
        ("U10", "U6", "BAL vs TT extra-8"),
        ("U10", "U7", "BAL vs TV extra-8"),
        ("U10", "U8", "BAL vs VT extra-8"),
        ("U10", "U9", "BAL vs VV extra-8"),
        # Same-domain prior vs foreign-domain prior — KEY transfer test
        ("U11", "U12", "MM-NIAH-prior vs LVB-prior — cross-domain transfer"),
        # Extra-16 vs extra-8 (diminishing returns?)
        ("U13", "U4", "ALL-16 vs GEN-8 (4.375 vs 4.28125 KV bits)"),
        # Does best-U match F9 below 4.75 KV bits?
        ("U6", "U2", "TT extra-8 vs F9 dense"),
        ("U10", "U2", "BAL extra-8 vs F9 dense"),
        ("U11", "U2", "MM-NIAH-prior extra-8 vs F9 dense"),
        ("U13", "U2", "ALL-16 extra vs F9 dense (closest to F9 bits)"),
    ]


def pairs_slice_a() -> list[tuple[str, str, str]]:
    return [
        ("Q2", "Q0", "F9 vs BF16 anchor"),
        ("Q1", "Q0", "F4 vs BF16 anchor"),
        ("Q3", "Q2", "RoleOnly vs F9 — does in-context routing matter at all?"),
        ("Q4", "Q2", "Quest top-50 FB vs F9 — does FormatBook match F9?"),
        ("Q4", "Q3", "Quest top-50 FB vs RoleOnly — does promoting hot pages help?"),
        ("Q4", "Q5", "Quest vs Random top-50 FB — does Quest selection matter?"),
        ("Q6", "Q4", "Oracle headroom over Quest (top-50)"),
        ("Q7", "Q2", "Quest top-25 FB vs F9 — tighter Pareto candidate"),
        ("Q7", "Q3", "Quest top-25 FB vs RoleOnly"),
        ("Q7", "Q8", "Quest vs Random top-25"),
        ("Q9", "Q7", "Oracle headroom over Quest (top-25)"),
        ("Q10", "Q2", "INT2-cold Quest top-25 vs F9 — cheap-cold viability"),
        ("Q10", "Q11", "INT2-cold Quest vs Random"),
    ]


def pairs_slice_b() -> list[tuple[str, str, str]]:
    return [
        ("R2", "R0", "F9 vs BF16 anchor (reasoning-image)"),
        ("R1", "R0", "F4 vs BF16 anchor (reasoning-image)"),
        ("R3", "R2", "RoleOnly vs F9 — in-context content load-bearing?"),
        ("R4", "R2", "Quest top-50 FB vs F9 on reasoning-image"),
        ("R4", "R3", "Quest top-50 FB vs RoleOnly on reasoning-image"),
        ("R4", "R5", "Quest vs Random top-50 (reasoning-image)"),
        ("R6", "R4", "Oracle headroom over Quest (top-50, reasoning-image)"),
        ("R7", "R2", "Quest top-25 FB vs F9 on reasoning-image"),
        ("R7", "R8", "Quest vs Random top-25 (reasoning-image)"),
    ]


def write_paired(rows: list[dict], out_md: Path, slice_tag: str) -> None:
    by_cond = group_by_condition(rows)
    rows_by_item = {c: map_by_item(rs) for c, rs in by_cond.items()}
    if slice_tag == "A":
        pairs = pairs_slice_a()
    elif slice_tag == "B":
        pairs = pairs_slice_b()
    elif slice_tag == "C":
        pairs = pairs_slice_c()
    elif slice_tag == "S":
        pairs = pairs_slice_s()
    elif slice_tag == "U":
        pairs = pairs_slice_u()
    else:
        pairs = []
    lines = [f"# Exp Q/R paired McNemar slice {slice_tag} — "
             f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append("| pair | description | n_paired | A_only | B_only | χ² | p | favored |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for a, b, desc in pairs:
        if a not in rows_by_item or b not in rows_by_item:
            lines.append(f"| {a} vs {b} | {desc} | — | — | — | — | — | (missing) |")
            continue
        m = paired_mcnemar(rows_by_item[a], rows_by_item[b])
        lines.append(
            f"| {a} vs {b} | {desc} | {m['n_paired']} | {m['a_only_correct']} | "
            f"{m['b_only_correct']} | {m['chi2']:.2f} | {m['p']:.4f} | {m['favored']} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


# ---------------- branch JSON ----------------

def _acc(rs: list[dict]) -> float:
    if not rs:
        return float("nan")
    return float(np.mean([int(bool(r.get("is_correct"))) for r in rs]))


def _paired_net(a_rows: dict[str, dict], b_rows: dict[str, dict]) -> int:
    """A_only - B_only (positive = A wins more discordant items)."""
    m = paired_mcnemar(a_rows, b_rows)
    return m["a_only_correct"] - m["b_only_correct"]


def write_branch_json(rows: list[dict], out_json: Path, slice_tag: str) -> dict:
    by_cond = group_by_condition(rows)
    rows_by_item = {c: map_by_item(rs) for c, rs in by_cond.items()}

    def trigger(a, b):
        if a not in rows_by_item or b not in rows_by_item:
            return False
        gap = _acc(by_cond.get(a, [])) - _acc(by_cond.get(b, []))
        net = _paired_net(rows_by_item[a], rows_by_item[b])
        return (not math.isnan(gap) and gap >= 0.02) or (net >= 5)

    out: dict = {}
    if slice_tag == "A":
        out["need_q5_seeds"] = trigger("Q4", "Q5")
        out["need_q8_seeds"] = trigger("Q7", "Q8")
        out["need_q11_seeds"] = trigger("Q10", "Q11")

        # Slice B recommendation: RUN if Slice A looks choice-dominated.
        q2 = _acc(by_cond.get("Q2", []))
        q3 = _acc(by_cond.get("Q3", []))
        q4 = _acc(by_cond.get("Q4", []))
        q5 = _acc(by_cond.get("Q5", []))
        if all(not math.isnan(x) for x in (q2, q3, q4, q5)):
            choice_dominated = (q3 >= q2 - 0.01) and (q4 <= q5 + 0.01)
            out["slice_b_recommendation"] = "RUN" if choice_dominated else "DEFER"
        else:
            out["slice_b_recommendation"] = "INSUFFICIENT_DATA"

        # Diagnostic numbers for the orchestrator.
        out["accuracy"] = {c: _acc(rs) for c, rs in by_cond.items()}
        out["paired_net"] = {}
        for a, b in (("Q4", "Q5"), ("Q7", "Q8"), ("Q10", "Q11"), ("Q4", "Q3"), ("Q7", "Q3")):
            if a in rows_by_item and b in rows_by_item:
                out["paired_net"][f"{a}_vs_{b}"] = _paired_net(rows_by_item[a], rows_by_item[b])
    elif slice_tag == "C":
        # Exp R Sub-experiment C gate.
        c2 = _acc(by_cond.get("C2", []))
        c3b = _acc(by_cond.get("C3b", []))
        s8 = _acc(by_cond.get("S8", []))

        winners: list[dict] = []
        for cand in ("C4", "C7"):
            if cand not in rows_by_item or "C2" not in rows_by_item:
                continue
            tie_with_f9 = paired_mcnemar(rows_by_item[cand], rows_by_item["C2"])
            ekvb_vals = [r.get("effective_kv_bits") for r in by_cond.get(cand, [])
                         if r.get("effective_kv_bits") is not None]
            ekvb_mean = float(np.mean(ekvb_vals)) if ekvb_vals else float("nan")
            paired_tie = tie_with_f9["chi2"] < 3.84
            beats_choice_only = (
                "C3b" in rows_by_item
                and _paired_net(rows_by_item[cand], rows_by_item["C3b"]) >= 5
            )
            beats_static_s8 = (
                "S8" in rows_by_item
                and _paired_net(rows_by_item[cand], rows_by_item["S8"]) >= 3
            )
            under_4_35 = (not math.isnan(ekvb_mean)) and ekvb_mean <= 4.35
            passes = paired_tie and beats_choice_only and beats_static_s8 and under_4_35
            winners.append({
                "cond": cand,
                "acc": _acc(by_cond.get(cand, [])),
                "effective_kv_bits": ekvb_mean,
                "paired_tie_with_F9": paired_tie,
                "beats_choice_only": beats_choice_only,
                "beats_static_S8": beats_static_s8,
                "under_4_35_kv_bits": under_4_35,
                "passes_c_gate": passes,
            })

        # Pick the winner: any condition that passes; prefer C7 over C4 if both pass.
        passing = [w for w in winners if w["passes_c_gate"]]
        if passing:
            # Prefer SplitQuest (C7) over global Quest (C4) — it explicitly
            # balances across in-context and choice, which the brief flags as
            # the "right routing object" if it wins.
            chosen = next((w for w in passing if w["cond"] == "C7"), passing[0])
            out["c_gate_passed"] = True
            out["winning_allvisual_cond"] = chosen["cond"]
            out["winner_route_name"] = (
                "formatbook_split_quest" if chosen["cond"] == "C7"
                else "formatbook_quest_allvisual"
            )
        else:
            out["c_gate_passed"] = False
            out["winning_allvisual_cond"] = None
            out["winner_route_name"] = None

        out["candidates"] = winners
        out["accuracy"] = {c: _acc(rs) for c, rs in by_cond.items()}
        out["paired_net"] = {}
        for a, b in (("C4", "C5"), ("C7", "C8"), ("C4", "C3b"), ("C7", "C3b"),
                     ("C4", "S8"), ("C7", "S8"), ("C4", "C2"), ("C7", "C2")):
            if a in rows_by_item and b in rows_by_item:
                out["paired_net"][f"{a}_vs_{b}"] = _paired_net(rows_by_item[a], rows_by_item[b])
    elif slice_tag == "U":
        # Exp U1 — residual channel oracle/policy screen.
        out["accuracy"] = {c: _acc(rs) for c, rs in by_cond.items()}
        out["paired_net"] = {}

        u_extras = ("U4", "U5", "U6", "U7", "U8", "U9",
                    "U10", "U11", "U12", "U13")

        # 1. Does ANY extra-N policy paired-significantly beat the S4 anchor?
        any_beats_s4 = False
        for u in u_extras:
            if u in rows_by_item and "U3" in rows_by_item:
                m = paired_mcnemar(rows_by_item[u], rows_by_item["U3"])
                out["paired_net"][f"{u}_vs_U3"] = (
                    m["a_only_correct"] - m["b_only_correct"]
                )
                if m["chi2"] >= 3.84 and m["favored"] == "A":
                    any_beats_s4 = True
        out["pass_any_extra_beats_s4"] = any_beats_s4

        # 2. Does any structured extra paired-significantly beat random (U5)?
        structured = ("U6", "U7", "U8", "U9", "U10",
                      "U11", "U12", "U13")
        structured_beats_random = False
        for u in structured:
            if u in rows_by_item and "U5" in rows_by_item:
                m = paired_mcnemar(rows_by_item[u], rows_by_item["U5"])
                out["paired_net"][f"{u}_vs_U5"] = (
                    m["a_only_correct"] - m["b_only_correct"]
                )
                if m["chi2"] >= 3.84 and m["favored"] == "A":
                    structured_beats_random = True
        out["pass_structured_beats_random"] = structured_beats_random

        # 3. Does any U paired-tie or beat F9 (U2) at < 4.75 KV bits?
        def _mean_ekvb(cond_name: str) -> float:
            if cond_name not in by_cond:
                return float("nan")
            vals = [r.get("effective_kv_bits") for r in by_cond[cond_name]
                    if r.get("effective_kv_bits") is not None]
            return float(np.mean(vals)) if vals else float("nan")

        match_or_beat_f9 = False
        for u in u_extras:
            if u not in rows_by_item or "U2" not in rows_by_item:
                continue
            m = paired_mcnemar(rows_by_item[u], rows_by_item["U2"])
            kvb = _mean_ekvb(u)
            tie_or_better = m["chi2"] < 3.84 or m["favored"] == "A"
            if tie_or_better and not math.isnan(kvb) and kvb < 4.75 - 1e-3:
                match_or_beat_f9 = True
                break
        out["pass_match_or_beat_f9"] = match_or_beat_f9

        # 4. Winning policy — argmax accuracy among U4..U13 (tie-break = lower KV bits).
        candidates = []
        for u in u_extras:
            if u in by_cond:
                candidates.append((u, _acc(by_cond[u]), _mean_ekvb(u)))
        candidates_sorted = sorted(candidates, key=lambda t: (-t[1], t[2]))
        out["winning_policy"] = candidates_sorted[0][0] if candidates_sorted else None
        out["candidate_ranking"] = [
            {"cond": c, "acc": a, "eff_kv_bits": k} for c, a, k in candidates_sorted
        ]

        # 5. Same-prior beats foreign-prior?
        if "U11" in rows_by_item and "U12" in rows_by_item:
            m = paired_mcnemar(rows_by_item["U11"], rows_by_item["U12"])
            out["pass_same_prior_beats_foreign"] = (
                m["chi2"] >= 3.84 and m["favored"] == "A"
            )
            out["paired_net"]["U11_vs_U12"] = (
                m["a_only_correct"] - m["b_only_correct"]
            )
        else:
            out["pass_same_prior_beats_foreign"] = False

        # Anchor sanity: U3 ~ Exp S S4 (retrieval = 0.571 ± 0.04 expected);
        # U2 ~ Exp S F9 (0.548 ± 0.04). Just record actuals; the orchestrator
        # decides whether to fail on drift.
        out["anchor_actual"] = {
            "U3_acc": _acc(by_cond.get("U3", [])),
            "U2_acc": _acc(by_cond.get("U2", [])),
        }
    else:
        # Slice B does not branch into further reseeds in this plan.
        out["accuracy"] = {c: _acc(rs) for c, rs in by_cond.items()}

    out_json.write_text(json.dumps(out, indent=2) + "\n")
    return out


# ---------------- verdict ----------------

def write_verdict(rows: list[dict], out_md: Path, slice_tag: str,
                  branch: dict) -> None:
    by_cond = group_by_condition(rows)
    rows_by_item = {c: map_by_item(rs) for c, rs in by_cond.items()}

    def acc(c):
        return _acc(by_cond.get(c, []))

    lines = [f"# Exp Q verdict matrix slice {slice_tag} — "
             f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append("## Headline signals")

    verdicts: list[str] = []
    if slice_tag == "A":
        # Anchors
        q0, q1, q2 = acc("Q0"), acc("Q1"), acc("Q2")
        if not any(math.isnan(x) for x in (q0, q1, q2)):
            verdicts.append(f"Anchor sanity: Q0 BF16={q0:.3f}, Q1 F4={q1:.3f}, Q2 F9={q2:.3f}")

        # Interpretation tree from the plan
        q3, q4, q5, q6 = acc("Q3"), acc("Q4"), acc("Q5"), acc("Q6")
        q7, q8, q9 = acc("Q7"), acc("Q8"), acc("Q9")
        q10, q11 = acc("Q10"), acc("Q11")

        def paired_tie_with_q2(c):
            if c not in rows_by_item or "Q2" not in rows_by_item:
                return False
            m = paired_mcnemar(rows_by_item[c], rows_by_item["Q2"])
            return m["chi2"] < 3.84  # p > 0.05

        # Quest top-50 (Q4) or top-25 (Q7) ≈ Q2 AND > Q3, Q5/Q8 AND eff_kv_bits < 4.75
        def fb_wins(qc, qrand):
            if qc not in rows_by_item or qrand not in rows_by_item or "Q3" not in rows_by_item:
                return None
            ekvb_vals = [r.get("effective_kv_bits") for r in by_cond.get(qc, [])
                         if r.get("effective_kv_bits") is not None]
            ekvb_mean = float(np.mean(ekvb_vals)) if ekvb_vals else float("nan")
            tie_with_q2 = paired_tie_with_q2(qc)
            beats_role_only = _paired_net(rows_by_item[qc], rows_by_item["Q3"]) > 0
            beats_random = _paired_net(rows_by_item[qc], rows_by_item[qrand]) > 0
            below_475 = ekvb_mean < 4.75 - 1e-3
            return {
                "name": qc, "tie_with_F9": tie_with_q2, "beats_RoleOnly": beats_role_only,
                "beats_Random": beats_random, "eff_kv_bits_below_F9": below_475,
                "all_four": tie_with_q2 and beats_role_only and beats_random and below_475,
                "acc": acc(qc), "eff_kv_bits": ekvb_mean,
            }

        for qc, qrand in (("Q4", "Q5"), ("Q7", "Q8")):
            w = fb_wins(qc, qrand)
            if w is None:
                continue
            tag = "HEADLINE" if w["all_four"] else "PARTIAL"
            verdicts.append(
                f"{tag} ({qc}): tie_F9={w['tie_with_F9']} beats_RoleOnly={w['beats_RoleOnly']} "
                f"beats_Random={w['beats_Random']} eff_kv_bits<4.75={w['eff_kv_bits_below_F9']} "
                f"(acc={w['acc']:.3f}, eff_kv_bits={_fmt(w['eff_kv_bits'])})"
            )

        # RoleOnly ≈ F9 → choice-dominated
        if "Q3" in rows_by_item and "Q2" in rows_by_item:
            m = paired_mcnemar(rows_by_item["Q3"], rows_by_item["Q2"])
            if m["chi2"] < 3.84:
                verdicts.append(
                    f"Q3 ≈ Q2 (χ²={m['chi2']:.2f}, p={m['p']:.3f}) — "
                    f"retrieval-image is choice/text dominated; in-context precision is not load-bearing."
                )

        # Oracle headroom
        if "Q6" in rows_by_item and "Q4" in rows_by_item:
            m = paired_mcnemar(rows_by_item["Q6"], rows_by_item["Q4"])
            if m["chi2"] >= 3.84 and m["favored"] == "A":
                verdicts.append(
                    f"ORACLE HEADROOM (Q6 vs Q4): χ²={m['chi2']:.2f}, p={m['p']:.3f} — "
                    f"Quest envelope scorer leaves room; consider a better page scorer."
                )

        # INT2 cold viability
        if "Q10" in rows_by_item and "Q2" in rows_by_item:
            m = paired_mcnemar(rows_by_item["Q10"], rows_by_item["Q2"])
            n_q11 = _paired_net(rows_by_item["Q10"], rows_by_item.get("Q11", {})) if "Q11" in rows_by_item else None
            if m["chi2"] < 3.84 and n_q11 is not None and n_q11 > 0:
                verdicts.append(
                    f"STRONG (Q10 vs Q2 tie + Q10 > Q11): hot F9 / cold-K INT2 is a real low-bit "
                    f"FormatBook policy. Query routing enables a cheaper cold format."
                )

        # Branch flags
        verdicts.append(f"Branch flags: need_q5_seeds={branch.get('need_q5_seeds')} "
                        f"need_q8_seeds={branch.get('need_q8_seeds')} "
                        f"need_q11_seeds={branch.get('need_q11_seeds')}")
        verdicts.append(f"Slice B recommendation: {branch.get('slice_b_recommendation')}")
    elif slice_tag == "U":
        u0, u1, u2, u3 = acc("U0"), acc("U1"), acc("U2"), acc("U3")
        if not any(math.isnan(x) for x in (u0, u1, u2, u3)):
            verdicts.append(
                f"Anchors: U0 BF16={u0:.3f} U1 F4={u1:.3f} "
                f"U2 F9={u2:.3f} U3 S4={u3:.3f}"
            )

        # Headline rules per the plan
        verdicts.append(
            f"pass_any_extra_beats_s4 = {branch.get('pass_any_extra_beats_s4')}"
        )
        verdicts.append(
            f"pass_structured_beats_random = {branch.get('pass_structured_beats_random')}"
        )
        verdicts.append(
            f"pass_match_or_beat_f9 = {branch.get('pass_match_or_beat_f9')}"
        )
        verdicts.append(
            f"pass_same_prior_beats_foreign (U11 > U12) = "
            f"{branch.get('pass_same_prior_beats_foreign')}"
        )
        verdicts.append(
            f"winning_policy = {branch.get('winning_policy')}"
        )

        # Top-3 candidates with their KV bits.
        ranking = branch.get("candidate_ranking") or []
        for r in ranking[:3]:
            verdicts.append(
                f"  {r['cond']}: acc={r['acc']:.3f} eff_kv_bits={r['eff_kv_bits']:.3f}"
            )

        # Strong-pass / clean-fail summary.
        strong_pass = (
            branch.get("pass_any_extra_beats_s4")
            and branch.get("pass_structured_beats_random")
            and branch.get("pass_match_or_beat_f9")
        )
        if strong_pass:
            verdicts.append(
                f"STRONG PASS on this slice — extra-channel structure is "
                f"load-bearing; cross-slice differential (vs other datasets) "
                f"determines the Wonsuk-central headline."
            )
        else:
            verdicts.append(
                f"NOT STRONG PASS — at least one of the three pass conditions failed."
            )
    else:
        # Slice B
        r0, r2 = acc("R0"), acc("R2")
        r3, r4, r5 = acc("R3"), acc("R4"), acc("R5")
        r7, r8 = acc("R7"), acc("R8")
        if not any(math.isnan(x) for x in (r0, r2, r3, r4)):
            verdicts.append(f"Anchors: R0 BF16={r0:.3f}, R2 F9={r2:.3f}")
            verdicts.append(f"R3 RoleOnly={r3:.3f}, R4 Quest top-50={r4:.3f}")
        for qc, qrand in (("R4", "R5"), ("R7", "R8")):
            if qc in rows_by_item and qrand in rows_by_item:
                m = paired_mcnemar(rows_by_item[qc], rows_by_item[qrand])
                verdicts.append(
                    f"{qc} vs {qrand}: χ²={m['chi2']:.2f}, p={m['p']:.3f}, favored={m['favored']}"
                )

    lines.extend(f"- {v}" for v in verdicts)
    lines.append("")
    lines.append("## Per-condition status")
    lines.append("| condition | n | acc | status |")
    lines.append("|---|---|---|---|")
    cond_order = slice_cond_order(slice_tag)
    for c in cond_order:
        if c not in by_cond:
            continue
        rs = by_cond[c]
        a = _acc(rs)
        if slice_tag == "A":
            status = {
                "Q0": "ANCHOR", "Q1": "ANCHOR", "Q2": "ANCHOR",
                "Q3": "ROLE_ONLY",
                "Q6": "ORACLE", "Q9": "ORACLE",
            }.get(c, "PROPOSED")
        elif slice_tag == "U":
            status = {
                "U0": "ANCHOR", "U1": "ANCHOR", "U2": "ANCHOR", "U3": "ANCHOR",
                "U4": "GENERIC_EXTRA", "U5": "RANDOM_CONTROL",
                "U6": "TT", "U7": "TV", "U8": "VT", "U9": "VV",
                "U10": "BALANCED", "U11": "MMNIAH_PRIOR",
                "U12": "LVB_PRIOR", "U13": "ALL16_EXTRA",
            }.get(c, "PROPOSED")
        else:
            status = {"R0": "ANCHOR", "R1": "ANCHOR", "R2": "ANCHOR",
                      "R3": "ROLE_ONLY", "R6": "ORACLE"}.get(c, "PROPOSED")
        lines.append(f"| {c} | {len(rs)} | {_fmt(a)} | {status} |")
    out_md.write_text("\n".join(lines) + "\n")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", choices=("A", "B", "C", "S", "U"), default="A",
                    help="A=Slice A retrieval-image (Exp Q), B=Slice B reasoning-image (Exp Q), "
                         "C=Exp R Sub-experiment C (AllVisual + static baselines), "
                         "S=Exp S Phase 1 sidecode bit-ladder (S0..S9), "
                         "U=Exp U1 residual channel oracle/policy screen (U0..U13).")
    ap.add_argument("--out-prefix", default=None,
                    help="Override the output file prefix. Default: expQ for "
                         "A/B/C/S; expU for U. Useful when running multiple Exp U "
                         "slices that need separate output paths.")
    ap.add_argument("--in-jsonl", type=Path, default=None,
                    help="Default: results/expQ_rollouts_slice{A|B}.jsonl")
    ap.add_argument("--out-summary", type=Path, default=None)
    ap.add_argument("--out-paired", type=Path, default=None)
    ap.add_argument("--out-verdict", type=Path, default=None)
    ap.add_argument("--out-branch", type=Path, default=None)
    ap.add_argument("--branch-check", action="store_true",
                    help="Emit branch JSON only (no markdown writes).")
    args = ap.parse_args()

    if args.out_prefix is not None:
        prefix = args.out_prefix
    elif args.slice == "U":
        prefix = "expU"
    else:
        prefix = "expQ"

    if args.in_jsonl is None:
        args.in_jsonl = RESULTS_DIR / f"{prefix}_rollouts_slice{args.slice}.jsonl"
    if args.out_summary is None:
        args.out_summary = RESULTS_DIR / f"{prefix}_summary_slice{args.slice}.md"
    if args.out_paired is None:
        args.out_paired = RESULTS_DIR / f"{prefix}_paired_slice{args.slice}.md"
    if args.out_verdict is None:
        args.out_verdict = RESULTS_DIR / f"{prefix}_verdict_matrix_slice{args.slice}.md"
    if args.out_branch is None:
        args.out_branch = RESULTS_DIR / f"{prefix}_branch_slice{args.slice}.json"

    rows = load_rollouts(args.in_jsonl)
    if not rows:
        print(f"no rows in {args.in_jsonl}", flush=True)
        return

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    branch = write_branch_json(rows, args.out_branch, args.slice)
    if args.branch_check:
        print(json.dumps(branch, indent=2))
        return

    write_summary(rows, args.out_summary, args.slice)
    write_paired(rows, args.out_paired, args.slice)
    write_verdict(rows, args.out_verdict, args.slice, branch)
    print(f"wrote {args.out_summary}\nwrote {args.out_paired}\n"
          f"wrote {args.out_verdict}\nwrote {args.out_branch}", flush=True)


if __name__ == "__main__":
    main()
