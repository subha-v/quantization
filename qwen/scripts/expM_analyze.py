"""Experiment M — Matched-budget controls analyzer.

The decisive paired comparisons:
  M9 vs M5: balanced top-3/block vs generic top-12 (matched 12 ch / 4.56 bits)
  M9 vs M6: balanced top-3/block vs random top-12
  M9 vs M7: cross-modal scoring vs balance-without-scoring (3/block matched)
  M9 vs M2: matches/beats F9 (16 ch / 4.75 bits) at fewer bits?
  M10 vs M9: INT8 sidecode for balanced top-3?
  M12 vs M5: pivot at 12 channels vs generic at 12 channels
  M12 vs M9: pivot top-12 vs balanced top-3/block
  M12 vs M2: pivot top-12 vs F9 16 BF16

Plus reproducibility:
  M2 vs M1: F9 lift over F4 on this seed
  M11 vs M4: pivot top-8 vs generic top-8 (matched at 4.375 bits, 8 ch)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from expG_analyze import bootstrap_ci, mcnemar_pair


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


ANCHORS_M = {"M0_BF16_128f", "M1_F4_128f"}


HEADLINE_PAIRS_M = [
    # Decisive matched-budget comparisons (all at 12 channels, 4.56 KV bits):
    ("M9_Bal3pb_BF16side_128f", "M5_Generic12_BF16side_128f", "bal3pb_vs_generic12"),
    ("M9_Bal3pb_BF16side_128f", "M6_Random12_BF16side_128f", "bal3pb_vs_random12"),
    ("M9_Bal3pb_BF16side_128f", "M7_BalRandomPos3pb_BF16side_128f", "balanced_vs_balrandom_3pb"),
    ("M12_Pivot12_BF16side_128f", "M5_Generic12_BF16side_128f", "pivot12_vs_generic12"),
    ("M12_Pivot12_BF16side_128f", "M9_Bal3pb_BF16side_128f", "pivot12_vs_bal3pb"),

    # Crossing into F9 territory (4.56 vs 4.75 bits):
    ("M9_Bal3pb_BF16side_128f", "M2_F9_BF16side_128f", "bal3pb_vs_f9"),
    ("M12_Pivot12_BF16side_128f", "M2_F9_BF16side_128f", "pivot12_vs_f9"),
    ("M5_Generic12_BF16side_128f", "M2_F9_BF16side_128f", "generic12_vs_f9"),

    # Sidecode compression on the top-3/block winner:
    ("M10_Bal3pb_INT8side_128f", "M9_Bal3pb_BF16side_128f", "bal3_int8_vs_bf16"),
    ("M10_Bal3pb_INT8side_128f", "M2_F9_BF16side_128f", "bal3_int8_vs_f9"),

    # J7 (8-channel) replication checks:
    ("M8_Bal2pb_BF16side_128f", "M4_Generic8_BF16side_128f", "bal2pb_vs_generic8"),
    ("M11_Pivot8_BF16side_128f", "M4_Generic8_BF16side_128f", "pivot8_vs_generic8"),

    # F9 INT8 sidecode replication on seed=0:
    ("M3_F9_INT8side_128f", "M2_F9_BF16side_128f", "f9_int8_vs_bf16"),

    # F9 reproduces:
    ("M2_F9_BF16side_128f", "M1_F4_128f", "f9_reproduces"),
]


def _cond_sort_key(name: str) -> tuple[int, str]:
    try:
        n = int(name.split("_")[0].lstrip("M"))
    except Exception:
        n = 9999
    return (n, name)


def load_m_rows(in_jsonl: Path) -> list[dict]:
    rows: list[dict] = []
    if in_jsonl.exists():
        rows.extend(json.loads(l) for l in in_jsonl.read_text().splitlines() if l.strip())
    rows = [r for r in rows if r.get("error") is None and not r.get("skipped", False)]
    return rows


def verdict_m(cond: str, by_cond_acc: dict[str, float],
              by_cond_kv: dict[str, float]) -> str:
    if cond in ANCHORS_M:
        return "anchor"
    acc = by_cond_acc.get(cond, float("nan"))
    if not np.isfinite(acc):
        return "missing"
    m2 = by_cond_acc.get("M2_F9_BF16side_128f", float("nan"))
    m5 = by_cond_acc.get("M5_Generic12_BF16side_128f", float("nan"))
    m6 = by_cond_acc.get("M6_Random12_BF16side_128f", float("nan"))
    m7 = by_cond_acc.get("M7_BalRandomPos3pb_BF16side_128f", float("nan"))
    m9 = by_cond_acc.get("M9_Bal3pb_BF16side_128f", float("nan"))
    kv_self = by_cond_kv.get(cond, float("nan"))
    kv_m2 = by_cond_kv.get("M2_F9_BF16side_128f", float("nan"))

    # M9: the candidate. Judged against matched-budget controls (M5/M6/M7).
    if cond == "M9_Bal3pb_BF16side_128f":
        ge_all = all(np.isfinite(v) and m9 >= v + 0.03 for v in (m5, m6, m7))
        if ge_all:
            return "beats_matched_controls"
        any_close = any(np.isfinite(v) and m9 >= v - 0.01 for v in (m5, m6, m7))
        any_loses = any(np.isfinite(v) and m9 < v - 0.03 for v in (m5, m6, m7))
        if any_loses:
            return "loses_to_matched_control"
        return "ties_matched_controls"

    # M5/M6/M7 controls: passive labels.
    if cond == "M5_Generic12_BF16side_128f":
        return "matched_budget_anchor"
    if cond == "M6_Random12_BF16side_128f":
        return "control_random_matched"
    if cond == "M7_BalRandomPos3pb_BF16side_128f":
        return "control_balanced_random_matched"

    # M10: balanced top-3 INT8 sidecode. Pareto winner vs M9 if lower bits + tied acc.
    if cond == "M10_Bal3pb_INT8side_128f":
        if np.isfinite(m9) and acc >= m9 - 0.01 and kv_self < by_cond_kv.get("M9_Bal3pb_BF16side_128f", 100.0) - 0.001:
            return "pareto_int8"
        return "borderline"

    # M12: pivot top-12. Judged against matched controls.
    if cond == "M12_Pivot12_BF16side_128f":
        if np.isfinite(m5) and acc >= m5 + 0.03:
            return "pivot_beats_matched_generic"
        if np.isfinite(m5) and acc >= m5 - 0.01:
            return "pivot_matches_matched_generic"
        return "borderline"

    # M3 F9 INT8 sidecode: J12 replication.
    if cond == "M3_F9_INT8side_128f":
        if np.isfinite(m2) and acc >= m2 - 0.01 and kv_self < kv_m2 - 0.001:
            return "f9_int8_pareto"
        if np.isfinite(m2) and acc < m2 - 0.05:
            return "kill"
        return "borderline"

    # M8 / M11: J7 / J8 replication at 8 channels.
    if cond == "M8_Bal2pb_BF16side_128f":
        m4 = by_cond_acc.get("M4_Generic8_BF16side_128f", float("nan"))
        if np.isfinite(m4) and acc >= m4 + 0.03:
            return "j7_replicates"
        return "j7_fails_to_replicate"
    if cond == "M11_Pivot8_BF16side_128f":
        m4 = by_cond_acc.get("M4_Generic8_BF16side_128f", float("nan"))
        if np.isfinite(m4) and acc >= m4 + 0.03:
            return "pivot_replicates"
        return "borderline"

    return "anchor" if cond in {"M2_F9_BF16side_128f", "M4_Generic8_BF16side_128f"} else "borderline"


def write_summary(rows, seed, out):
    by_cond = {}
    for r in rows:
        c = r.get("condition")
        if c:
            by_cond.setdefault(c, []).append(r)
    cond_acc = {}
    for cond in sorted(by_cond, key=_cond_sort_key):
        crows = by_cond[cond]
        correct = np.array([1.0 if r.get("is_correct") else 0.0 for r in crows], dtype=np.float32)
        m, lo, hi = bootstrap_ci(correct)
        cond_acc[cond] = {
            "acc": float(m), "n": int(correct.size),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "rel_kv_memory": float(crows[0].get("relative_kv_memory", float("nan"))),
            "avg_kv_bits": float(crows[0].get("avg_kv_bits", float("nan"))),
        }
    by_acc = {c: d["acc"] for c, d in cond_acc.items()}
    by_kv = {c: d["avg_kv_bits"] for c, d in cond_acc.items()}
    n_per = max((d["n"] for d in cond_acc.values()), default=0)
    lines = [f"# Exp M summary — seed={seed} (n={n_per})", ""]
    lines.append("| Condition | acc | 95% CI | n | rel_kv_mem | avg_kv_bits | verdict |")
    lines.append("|---|---:|---|---:|---:|---:|---|")
    for cond in sorted(cond_acc, key=_cond_sort_key):
        d = cond_acc[cond]
        v = verdict_m(cond, by_acc, by_kv)
        lines.append(
            f"| {cond} | {d['acc']:.3f} | [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}] | "
            f"{d['n']} | {d['rel_kv_memory']:.3f} | {d['avg_kv_bits']:.3f} | {v} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[expM] wrote {out}")
    return cond_acc


def write_paired(rows, seed, out):
    by_cond = {}
    for r in rows:
        c = r.get("condition")
        if c:
            by_cond.setdefault(c, []).append(r)
    lines = [f"# Exp M paired McNemar — seed={seed}", ""]
    lines.append("| label | a | b | n | acc(a) | acc(b) | a_only | b_only | both | neither | χ² | p |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a, b, label in HEADLINE_PAIRS_M:
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[expM] wrote {out}")


def write_verdict(cond_acc, seed, out):
    by_acc = {c: d["acc"] for c, d in cond_acc.items()}
    by_kv = {c: d["avg_kv_bits"] for c, d in cond_acc.items()}
    lines = [f"# Exp M verdict matrix — seed={seed}", ""]
    lines.append("- M9 (Balanced 3/block, K9 replication): the candidate to validate.")
    lines.append("- Decisive controls: M5 (generic 12), M6 (random 12), M7 (balanced-random 12).")
    lines.append("- M9 verdict states:")
    lines.append("  - `beats_matched_controls`: M9 >= each of M5/M6/M7 by >= 3 pp.")
    lines.append("  - `ties_matched_controls`: M9 within 1 pp of all.")
    lines.append("  - `loses_to_matched_control`: M9 < any control by >= 3 pp.")
    lines.append("")
    lines.append("| Condition | acc | rel_kv_mem | avg_kv_bits | verdict |")
    lines.append("|---|---:|---:|---:|---|")
    for cond in sorted(cond_acc, key=_cond_sort_key):
        d = cond_acc[cond]
        v = verdict_m(cond, by_acc, by_kv)
        lines.append(f"| {cond} | {d['acc']:.3f} | {d['rel_kv_memory']:.3f} | "
                     f"{d['avg_kv_bits']:.3f} | {v} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[expM] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--in_jsonl", type=Path, default=None)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--paired", type=Path, default=None)
    ap.add_argument("--verdict", type=Path, default=None)
    args = ap.parse_args()
    if args.in_jsonl is None:
        args.in_jsonl = RESULTS_DIR / f"expM_matched_stage3_seed{args.seed}.jsonl"
    if args.summary is None:
        args.summary = RESULTS_DIR / f"expM_summary_seed{args.seed}.md"
    if args.paired is None:
        args.paired = RESULTS_DIR / f"expM_paired_seed{args.seed}.md"
    if args.verdict is None:
        args.verdict = RESULTS_DIR / f"expM_verdict_matrix_seed{args.seed}.md"
    if not args.in_jsonl.exists():
        raise SystemExit(f"[expM] input JSONL not found: {args.in_jsonl}")
    rows = load_m_rows(args.in_jsonl)
    print(f"[expM] loaded {len(rows)} rows from {args.in_jsonl}", flush=True)
    cond_acc = write_summary(rows, args.seed, args.summary)
    write_paired(rows, args.seed, args.paired)
    write_verdict(cond_acc, args.seed, args.verdict)


if __name__ == "__main__":
    main()
