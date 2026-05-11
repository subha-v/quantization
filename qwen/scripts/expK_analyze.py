"""Experiment K — Balanced Cross-Modal Replication analyzer.

Reads expK_balanced_stage3_seed{S}.jsonl per seed and emits per-seed:
  expK_summary_seed{S}.md
  expK_paired_seed{S}.md
  expK_verdict_matrix_seed{S}.md

Headline pairs (focus on the four replication questions):
  Q1. balanced_vs_generic  K6 vs K4   does balanced top-2/block still beat generic top-8?
  Q2. balanced_beats_random K6 vs K5  does balanced beat random top-8?
  Q3. balanced_decouple    K6 vs K10  does cross-modal scoring matter, or only balance structure?
  Q4a. sidecode_int8       K7 vs K6   can balanced top-2/block use INT8 sidecode for free?
  Q4b. sidecode_int8_f9    K3 vs K2   reproduce J12 INT8-sidecode result on F9
  Q5. top1_vs_top2         K8 vs K6   smaller budget (4 channels) — does it suffice?
  Q6. top3_vs_top2         K9 vs K6   larger budget (12 channels) — does it improve?
  Q7. pareto_K7_vs_F9      K7 vs K2   the best-shot Pareto win: J7+INT8 vs F9
  Q8. f9_reproduces        K2 vs K1   F9 lift over F4 on this seed

Reuses bootstrap_ci, mcnemar_pair from expG_analyze.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from expG_analyze import bootstrap_ci, mcnemar_pair


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


ANCHORS_K = {"K0_BF16_128f", "K1_F4_128f"}


HEADLINE_PAIRS_K = [
    # Q1: replication of balanced > generic
    ("K6_Bal2pb_BF16side_128f", "K4_F8_BF16side_128f", "balanced_vs_generic"),
    # Q2: replication of balanced > random
    ("K6_Bal2pb_BF16side_128f", "K5_Random8_BF16side_128f", "balanced_vs_random"),
    # Q3: balance structure vs cross-modal scoring
    ("K6_Bal2pb_BF16side_128f", "K10_BalRandomPos_BF16side_128f", "crossmodal_vs_balanced_random"),
    # Q4a: INT8 sidecode for balanced
    ("K7_Bal2pb_INT8side_128f", "K6_Bal2pb_BF16side_128f", "balanced_int8_vs_bf16"),
    # Q4b: INT8 sidecode for F9 (J12 reproduction)
    ("K3_F9_INT8side_128f", "K2_F9_BF16side_128f", "f9_int8_vs_bf16"),
    # Q5: budget exploration — top-1
    ("K8_Bal1pb_BF16side_128f", "K6_Bal2pb_BF16side_128f", "top1pb_vs_top2pb"),
    # Q6: budget exploration — top-3
    ("K9_Bal3pb_BF16side_128f", "K6_Bal2pb_BF16side_128f", "top3pb_vs_top2pb"),
    # Q7: the best-shot Pareto win
    ("K7_Bal2pb_INT8side_128f", "K2_F9_BF16side_128f", "K7_vs_F9_pareto"),
    ("K6_Bal2pb_BF16side_128f", "K2_F9_BF16side_128f", "K6_vs_F9"),
    # Q8: F9 reproduction
    ("K2_F9_BF16side_128f", "K1_F4_128f", "f9_reproduces"),
    # Plus K11 pivot reproduction
    ("K11_Pivot8_BF16side_128f", "K4_F8_BF16side_128f", "pivot_vs_generic"),
    ("K11_Pivot8_BF16side_128f", "K5_Random8_BF16side_128f", "pivot_vs_random"),
]


def _cond_sort_key(name: str) -> tuple[int, str]:
    try:
        n = int(name.split("_")[0].lstrip("K"))
    except Exception:
        n = 9999
    return (n, name)


def load_k_rows(in_jsonl: Path) -> list[dict]:
    rows: list[dict] = []
    if in_jsonl.exists():
        rows.extend(json.loads(l) for l in in_jsonl.read_text().splitlines() if l.strip())
    rows = [r for r in rows if r.get("error") is None and not r.get("skipped", False)]
    return rows


def verdict_k(cond: str, by_cond_acc: dict[str, float],
              by_cond_kv: dict[str, float]) -> str:
    if cond in ANCHORS_K:
        return "anchor"
    acc = by_cond_acc.get(cond, float("nan"))
    if not np.isfinite(acc):
        return "missing"
    k2 = by_cond_acc.get("K2_F9_BF16side_128f", float("nan"))
    k4 = by_cond_acc.get("K4_F8_BF16side_128f", float("nan"))
    k6 = by_cond_acc.get("K6_Bal2pb_BF16side_128f", float("nan"))
    kv_self = by_cond_kv.get(cond, float("nan"))
    kv_k2 = by_cond_kv.get("K2_F9_BF16side_128f", float("nan"))

    # K6 (the J7 replication): judged vs K4 (F8) and K5 (random).
    if cond == "K6_Bal2pb_BF16side_128f":
        if np.isfinite(k4) and acc >= k4 + 0.03:
            return "replicates_J7"
        if np.isfinite(k4) and acc >= k4 - 0.01:
            return "matches_baseline"
        return "fails_to_replicate"

    # K7 (J7 + INT8 sidecode): the best shot. Compare vs K2 F9.
    if cond == "K7_Bal2pb_INT8side_128f":
        if np.isfinite(k2) and acc >= k2 - 0.01 and kv_self < kv_k2 - 0.001:
            return "pareto_winner"
        if np.isfinite(k2) and acc >= k2 + 0.03:
            return "paper_strong"
        if np.isfinite(k2) and acc >= k2 - 0.01:
            return "promote"
        return "borderline"

    # K10 (balanced-random control): a control, but if it ties K6 it undermines the mechanism.
    if cond == "K10_BalRandomPos_BF16side_128f":
        if np.isfinite(k6) and abs(acc - k6) < 0.01:
            return "control_ties_K6"
        return "control"

    # K5 fully-random control.
    if cond == "K5_Random8_BF16side_128f":
        return "control_random"

    # K3 (F9 INT8 sidecode): J12 reproduction. Compare vs K2 BF16.
    if cond == "K3_F9_INT8side_128f":
        if np.isfinite(k2) and acc >= k2 - 0.01 and kv_self < kv_k2 - 0.001:
            return "pareto_winner"
        if np.isfinite(k2) and acc < k2 - 0.05:
            return "kill"
        return "borderline"

    # K8/K9 budget exploration.
    if cond in {"K8_Bal1pb_BF16side_128f", "K9_Bal3pb_BF16side_128f"}:
        if np.isfinite(k6) and acc >= k6 + 0.03:
            return "beats_K6"
        if np.isfinite(k6) and acc >= k6 - 0.01:
            return "matches_K6"
        return "borderline"

    # K11 pivot reproduction.
    if cond == "K11_Pivot8_BF16side_128f":
        if np.isfinite(k4) and acc >= k4 + 0.03:
            return "replicates_pivot_win"
        if np.isfinite(k4) and acc >= k4 - 0.01:
            return "matches_baseline"
        return "borderline"

    # K2, K4: anchors, but mark whether F9 lift reproduces.
    if cond == "K2_F9_BF16side_128f":
        return "anchor"
    if cond == "K4_F8_BF16side_128f":
        return "anchor"

    return "borderline"


def write_summary(rows: list[dict], seed: int, out_summary: Path) -> dict:
    by_cond: dict[str, list[dict]] = {}
    for r in rows:
        c = r.get("condition")
        if not c:
            continue
        by_cond.setdefault(c, []).append(r)
    cond_acc: dict[str, dict] = {}
    for cond in sorted(by_cond, key=_cond_sort_key):
        crows = by_cond[cond]
        correct = np.array([1.0 if r.get("is_correct") else 0.0 for r in crows],
                           dtype=np.float32)
        m, lo, hi = bootstrap_ci(correct)
        rel = float(crows[0].get("relative_kv_memory", float("nan")))
        avg_kv = float(crows[0].get("avg_kv_bits", float("nan")))
        cond_acc[cond] = {"acc": float(m), "n": int(correct.size),
                          "ci_lo": float(lo), "ci_hi": float(hi),
                          "rel_kv_memory": rel, "avg_kv_bits": avg_kv}

    by_cond_acc = {c: d["acc"] for c, d in cond_acc.items()}
    by_cond_kv = {c: d["avg_kv_bits"] for c, d in cond_acc.items()}
    n_per_cond = max((d["n"] for d in cond_acc.values()), default=0)
    lines = [f"# Exp K summary — seed={seed} (n={n_per_cond})", ""]
    lines.append("| Condition | acc | 95% CI | n | rel_kv_mem | avg_kv_bits | verdict |")
    lines.append("|---|---:|---|---:|---:|---:|---|")
    for cond in sorted(cond_acc, key=_cond_sort_key):
        d = cond_acc[cond]
        v = verdict_k(cond, by_cond_acc, by_cond_kv)
        lines.append(
            f"| {cond} | {d['acc']:.3f} | [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}] | "
            f"{d['n']} | {d['rel_kv_memory']:.3f} | {d['avg_kv_bits']:.3f} | {v} |"
        )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text("\n".join(lines) + "\n")
    print(f"[expK] wrote {out_summary}")
    return cond_acc


def write_paired(rows: list[dict], seed: int, out_paired: Path) -> None:
    by_cond: dict[str, list[dict]] = {}
    for r in rows:
        c = r.get("condition")
        if not c:
            continue
        by_cond.setdefault(c, []).append(r)
    lines = [f"# Exp K paired McNemar — seed={seed}", ""]
    lines.append("`(a, b)` = (alternative, baseline). χ² = `(b_only - a_only)² / (a_only + b_only)`.")
    lines.append("")
    lines.append("| label | a | b | n | acc(a) | acc(b) | a_only | b_only | both | neither | χ² | p |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a, b, label in HEADLINE_PAIRS_K:
        ra = by_cond.get(a, [])
        rb = by_cond.get(b, [])
        if not ra or not rb:
            lines.append(f"| {label} | {a} | {b} | — | — | — | — | — | — | — | — | — |")
            continue
        n, both, a_only, b_only, neither, chi2, p = mcnemar_pair(ra, rb)
        acc_a = (both + a_only) / max(1, n)
        acc_b = (both + b_only) / max(1, n)
        lines.append(
            f"| {label} | {a} | {b} | {n} | {acc_a:.3f} | {acc_b:.3f} | "
            f"{a_only} | {b_only} | {both} | {neither} | {chi2:.3f} | {p:.4f} |"
        )
    out_paired.parent.mkdir(parents=True, exist_ok=True)
    out_paired.write_text("\n".join(lines) + "\n")
    print(f"[expK] wrote {out_paired}")


def write_verdict(cond_acc: dict[str, dict], seed: int, out_verdict: Path) -> None:
    by_cond_acc = {c: d["acc"] for c, d in cond_acc.items()}
    by_cond_kv = {c: d["avg_kv_bits"] for c, d in cond_acc.items()}
    lines = [f"# Exp K verdict matrix — seed={seed}", ""]
    lines.append("- K6 (balanced top-2/block BF16 sidecode): replicates J7 if acc ≥ K4 + 3 pp.")
    lines.append("- K7 (balanced top-2/block INT8 sidecode): the best-shot Pareto win vs K2 F9.")
    lines.append("- K10 (balanced-random by channel-position): if it ties K6, mechanism is balance not cross-modal.")
    lines.append("")
    lines.append("| Condition | acc | rel_kv_mem | avg_kv_bits | verdict |")
    lines.append("|---|---:|---:|---:|---|")
    for cond in sorted(cond_acc, key=_cond_sort_key):
        d = cond_acc[cond]
        v = verdict_k(cond, by_cond_acc, by_cond_kv)
        lines.append(
            f"| {cond} | {d['acc']:.3f} | {d['rel_kv_memory']:.3f} | "
            f"{d['avg_kv_bits']:.3f} | {v} |"
        )
    out_verdict.parent.mkdir(parents=True, exist_ok=True)
    out_verdict.write_text("\n".join(lines) + "\n")
    print(f"[expK] wrote {out_verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--in_jsonl", type=Path, default=None)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--paired", type=Path, default=None)
    ap.add_argument("--verdict", type=Path, default=None)
    args = ap.parse_args()
    if args.in_jsonl is None:
        args.in_jsonl = RESULTS_DIR / f"expK_balanced_stage3_seed{args.seed}.jsonl"
    if args.summary is None:
        args.summary = RESULTS_DIR / f"expK_summary_seed{args.seed}.md"
    if args.paired is None:
        args.paired = RESULTS_DIR / f"expK_paired_seed{args.seed}.md"
    if args.verdict is None:
        args.verdict = RESULTS_DIR / f"expK_verdict_matrix_seed{args.seed}.md"
    if not args.in_jsonl.exists():
        raise SystemExit(f"[expK] input JSONL not found: {args.in_jsonl}")
    rows = load_k_rows(args.in_jsonl)
    print(f"[expK] loaded {len(rows)} rows from {args.in_jsonl}", flush=True)
    cond_acc = write_summary(rows, args.seed, args.summary)
    write_paired(rows, args.seed, args.paired)
    write_verdict(cond_acc, args.seed, args.verdict)


if __name__ == "__main__":
    main()
