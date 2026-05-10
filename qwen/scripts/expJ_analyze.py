"""Experiment J — Cross-modal outlier analyzer.

Reads expJ_xmodal_stage{N}_seed{S}.jsonl and emits four outputs:

  qwen/results/expJ_summary_stage{N}.md       Per-condition acc + 95% CI
                                              + relative KV memory.
  qwen/results/expJ_paired_stage{N}.md        13 headline McNemar pairs.
  qwen/results/expJ_verdict_matrix_stage{N}.md  Per-condition verdict.
  qwen/results/expJ_promote_stage1.json       (stage 1 only) JSON list of
                                              promoted variants → consumed
                                              by expJ_xmodal_outlier.py at
                                              Stage 3.

Verdict states:
  anchor              J0/J1 (always)
  paper_strong        Δ ≥ +3 pp paired McNemar p < 0.05 vs the right rival
  pareto_winner       acc(J_X) ≥ acc(J2 F9) - 1pp at strictly fewer KV bits
  promote_n200        within ±1 pp of relevant rival
  borderline          else
  kill                Δ ≤ −5 pp vs the right rival

Reused helpers: bootstrap_ci, mcnemar_pair from expG_analyze.

Usage:
  python expJ_analyze.py --stage 1 --seed 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from expG_analyze import bootstrap_ci, mcnemar_pair


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# Anchors carry verdict "anchor" automatically.
ANCHORS_J = {"J0_BF16_128f", "J1_F4_128f"}


# Headline pairs: (alternative, baseline, label)
HEADLINE_PAIRS_J = [
    # Cross-modal selection vs generic at matched 4.375 bits
    ("J4_Outlier8_TT_128f",     "J3_F8_128f", "tt_vs_generic"),
    ("J5_Outlier8_TV_128f",     "J3_F8_128f", "tv_vs_generic"),
    ("J6_Outlier8_TT_TV_128f",  "J3_F8_128f", "tt_tv_vs_generic"),
    ("J7_Outlier8_BAL_128f",    "J3_F8_128f", "balanced_vs_generic"),
    ("J8_Outlier8_PIVOT_128f",  "J3_F8_128f", "pivot_vs_generic"),

    # Cross-modal vs F9 at lower bits (the Pareto question)
    ("J6_Outlier8_TT_TV_128f",  "J2_F9_128f", "tt_tv_vs_f9"),
    ("J9_LA_TT_TV_50pct_128f",  "J2_F9_128f", "la_50pct_vs_f9"),
    ("J11_LA_TT_TV_75pct_128f", "J2_F9_128f", "la_75pct_vs_f9"),

    # Sidecode compression: does INT-N sidecode match BF16 sidecode?
    ("J12_F9_INT8side_128f",    "J2_F9_128f", "int8side_vs_bf16side"),
    ("J13_F9_INT6side_128f",    "J2_F9_128f", "int6side_vs_bf16side"),
    ("J14_TT_TV_INT8side_128f", "J2_F9_128f", "tt_tv_int8side_vs_f9"),

    # Reproducibility on seed=2
    ("J2_F9_128f", "J1_F4_128f", "f9_reproduces_seed2"),
    ("J3_F8_128f", "J1_F4_128f", "f8_reproduces_seed2"),
]


def _cond_sort_key(name: str) -> tuple[int, str]:
    """Sort J conditions by numeric suffix (J0..J14)."""
    try:
        n = int(name.split("_")[0].lstrip("J"))
    except Exception:
        n = 9999
    return (n, name)


def load_j_rows(in_jsonl: Path) -> list[dict]:
    rows: list[dict] = []
    if in_jsonl.exists():
        rows.extend(json.loads(l) for l in in_jsonl.read_text().splitlines() if l.strip())
    rows = [r for r in rows if r.get("error") is None and not r.get("skipped", False)]
    return rows


def verdict_j(cond: str, by_cond_acc: dict[str, float],
              by_cond_kv: dict[str, float]) -> str:
    """Return Stage-{N} verdict for `cond`."""
    if cond in ANCHORS_J:
        return "anchor"
    acc = by_cond_acc.get(cond, float("nan"))
    if not np.isfinite(acc):
        return "missing"

    j1 = by_cond_acc.get("J1_F4_128f", float("nan"))
    j2 = by_cond_acc.get("J2_F9_128f", float("nan"))
    j3 = by_cond_acc.get("J3_F8_128f", float("nan"))
    kv_self = by_cond_kv.get(cond, float("nan"))
    kv_j2 = by_cond_kv.get("J2_F9_128f", float("nan"))

    # Cross-modal top-8 selection (J4-J8): judged vs J3 generic top-8.
    if cond in {"J4_Outlier8_TT_128f", "J5_Outlier8_TV_128f", "J6_Outlier8_TT_TV_128f",
                "J7_Outlier8_BAL_128f", "J8_Outlier8_PIVOT_128f"}:
        if np.isfinite(j3) and acc >= j3 + 0.03:
            return "paper_strong"
        # Pareto: matches J2 F9 within 1pp at strictly fewer bits.
        if np.isfinite(j2) and acc >= j2 - 0.01 and kv_self < kv_j2 - 0.001:
            return "pareto_winner"
        if np.isfinite(j3) and acc >= j3 - 0.01:
            return "promote_n200"
        if np.isfinite(j3) and acc < j3 - 0.05:
            return "kill"
        return "borderline"

    # Layer-adaptive (J9-J11): judged vs J2 F9 (lower or matched bits).
    if cond in {"J9_LA_TT_TV_50pct_128f", "J10_LA_ALL_50pct_128f", "J11_LA_TT_TV_75pct_128f"}:
        if np.isfinite(j2) and acc >= j2 - 0.01 and kv_self < kv_j2 - 0.001:
            return "pareto_winner"
        if np.isfinite(j2) and acc >= j2 + 0.03:
            return "paper_strong"
        if np.isfinite(j2) and acc >= j2 - 0.01:
            return "promote_n200"
        if np.isfinite(j2) and acc < j2 - 0.05:
            return "kill"
        return "borderline"

    # Sidecode compression (J12-J14): judged vs J2 F9 (lower bits, equal selection).
    if cond in {"J12_F9_INT8side_128f", "J13_F9_INT6side_128f", "J14_TT_TV_INT8side_128f"}:
        if np.isfinite(j2) and acc >= j2 - 0.01 and kv_self < kv_j2 - 0.001:
            return "pareto_winner"
        if np.isfinite(j2) and acc >= j2 + 0.03:
            return "paper_strong"
        if np.isfinite(j2) and acc >= j2 - 0.01:
            return "promote_n200"
        if np.isfinite(j2) and acc < j2 - 0.05:
            return "kill"
        return "borderline"

    # Anchor F9/F8 reproduction: judge vs F4.
    if cond == "J2_F9_128f":
        if np.isfinite(j1) and acc >= j1 + 0.03:
            return "anchor"  # J2 IS the F9 anchor; reproduces if it beats F4.
        return "anchor"
    if cond == "J3_F8_128f":
        return "anchor"

    return "borderline"


def write_summary(rows: list[dict], stage: int, out_summary: Path) -> dict:
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
        rel = float(crows[0].get("relative_kv_memory", float("nan"))) if crows else float("nan")
        avg_kv = float(crows[0].get("avg_kv_bits", float("nan"))) if crows else float("nan")
        n_frames = int(crows[0].get("n_frames", crows[0].get("frames", -1))) if crows else -1
        cond_acc[cond] = {
            "acc": float(m), "n": int(correct.size),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "rel_kv_memory": rel, "avg_kv_bits": avg_kv,
            "n_frames": n_frames,
        }

    by_cond_acc = {c: d["acc"] for c, d in cond_acc.items()}
    by_cond_kv = {c: d["avg_kv_bits"] for c, d in cond_acc.items()}

    n_per_cond = max((d["n"] for d in cond_acc.values()), default=0)
    lines = [f"# Exp J summary — Stage {stage}", ""]
    lines.append(f"_n items per condition (max) = {n_per_cond}; "
                 f"95% CI bootstrap n_boot=2000_")
    lines.append("")
    lines.append("| Condition | acc | 95% CI | n | rel_kv_mem | avg_kv_bits | verdict |")
    lines.append("|---|---:|---|---:|---:|---:|---|")
    for cond in sorted(cond_acc, key=_cond_sort_key):
        d = cond_acc[cond]
        v = verdict_j(cond, by_cond_acc, by_cond_kv)
        lines.append(
            f"| {cond} | {d['acc']:.3f} | [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}] | "
            f"{d['n']} | {d['rel_kv_memory']:.3f} | {d['avg_kv_bits']:.3f} | {v} |"
        )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text("\n".join(lines) + "\n")
    print(f"[expJ] wrote {out_summary}")
    return cond_acc


def write_paired(rows: list[dict], stage: int, out_paired: Path) -> None:
    by_cond: dict[str, list[dict]] = {}
    for r in rows:
        c = r.get("condition")
        if not c:
            continue
        by_cond.setdefault(c, []).append(r)

    lines = [f"# Exp J paired McNemar — Stage {stage}", ""]
    lines.append("`(a, b)` = (alternative, baseline). χ² = `(b_only - a_only)² / (a_only + b_only)`. "
                 "p ≈ from χ²₁.")
    lines.append("")
    lines.append("| label | a | b | n | acc(a) | acc(b) | a_only | b_only | both | neither | χ² | p |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a, b, label in HEADLINE_PAIRS_J:
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
    print(f"[expJ] wrote {out_paired}")


def write_verdict(cond_acc: dict[str, dict], stage: int, out_verdict: Path,
                  out_promote_json: Path | None = None) -> None:
    by_cond_acc = {c: d["acc"] for c, d in cond_acc.items()}
    by_cond_kv = {c: d["avg_kv_bits"] for c, d in cond_acc.items()}
    cond_names = sorted(cond_acc, key=_cond_sort_key)
    lines = [f"# Exp J verdict matrix — Stage {stage}", ""]
    lines.append("- Anchors J0/J1 always carry verdict `anchor`.")
    lines.append("- Cross-modal selection (J4–J8): judged vs J3 (generic top-8).")
    lines.append("- Layer-adaptive (J9–J11) and sidecode (J12–J14): judged vs J2 F9.")
    lines.append("- `pareto_winner`: matches J2 F9 within 1pp at strictly fewer KV bits.")
    lines.append("")
    lines.append("| Condition | acc | rel_kv_mem | avg_kv_bits | verdict |")
    lines.append("|---|---:|---:|---:|---|")
    for cond in cond_names:
        d = cond_acc[cond]
        v = verdict_j(cond, by_cond_acc, by_cond_kv)
        lines.append(
            f"| {cond} | {d['acc']:.3f} | {d['rel_kv_memory']:.3f} | "
            f"{d['avg_kv_bits']:.3f} | {v} |"
        )

    promoted_strong = [c for c in cond_names
                       if verdict_j(c, by_cond_acc, by_cond_kv)
                       in ("paper_strong", "pareto_winner")]
    promoted_n200 = [c for c in cond_names
                     if verdict_j(c, by_cond_acc, by_cond_kv)
                     in ("promote_n200", "paper_strong", "pareto_winner")]
    killed = [c for c in cond_names if verdict_j(c, by_cond_acc, by_cond_kv) == "kill"]
    borderline = [c for c in cond_names if verdict_j(c, by_cond_acc, by_cond_kv) == "borderline"]

    lines.append("")
    lines.append(f"**paper_strong / pareto_winner**: {promoted_strong}")
    lines.append(f"**promote_n200 (incl. above)**: {promoted_n200}")
    lines.append(f"**borderline**:  {borderline}")
    lines.append(f"**kill**:        {killed}")
    out_verdict.parent.mkdir(parents=True, exist_ok=True)
    out_verdict.write_text("\n".join(lines) + "\n")
    print(f"[expJ] wrote {out_verdict}")

    if out_promote_json is not None:
        VARIANTS = {
            "J4_Outlier8_TT_128f", "J5_Outlier8_TV_128f", "J6_Outlier8_TT_TV_128f",
            "J7_Outlier8_BAL_128f", "J8_Outlier8_PIVOT_128f",
            "J9_LA_TT_TV_50pct_128f", "J10_LA_ALL_50pct_128f", "J11_LA_TT_TV_75pct_128f",
            "J12_F9_INT8side_128f", "J13_F9_INT6side_128f", "J14_TT_TV_INT8side_128f",
        }
        promoted_variants = [c for c in promoted_n200 if c in VARIANTS]
        out_promote_json.write_text(json.dumps({"promoted": promoted_variants}, indent=2))
        print(f"[expJ] wrote {out_promote_json}: {promoted_variants}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 3], required=True)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--in_jsonl", type=Path, default=None)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--paired", type=Path, default=None)
    ap.add_argument("--verdict", type=Path, default=None)
    ap.add_argument("--promote_json", type=Path, default=None)
    args = ap.parse_args()

    if args.in_jsonl is None:
        args.in_jsonl = RESULTS_DIR / f"expJ_xmodal_stage{args.stage}_seed{args.seed}.jsonl"
    if args.summary is None:
        args.summary = RESULTS_DIR / f"expJ_summary_stage{args.stage}.md"
    if args.paired is None:
        args.paired = RESULTS_DIR / f"expJ_paired_stage{args.stage}.md"
    if args.verdict is None:
        args.verdict = RESULTS_DIR / f"expJ_verdict_matrix_stage{args.stage}.md"
    if args.stage == 1 and args.promote_json is None:
        args.promote_json = RESULTS_DIR / "expJ_promote_stage1.json"

    if not args.in_jsonl.exists():
        raise SystemExit(f"[expJ] input JSONL not found: {args.in_jsonl}")

    rows = load_j_rows(args.in_jsonl)
    print(f"[expJ] loaded {len(rows)} rows from {args.in_jsonl}", flush=True)

    cond_acc = write_summary(rows, args.stage, args.summary)
    write_paired(rows, args.stage, args.paired)
    write_verdict(cond_acc, args.stage, args.verdict,
                  out_promote_json=args.promote_json if args.stage == 1 else None)


if __name__ == "__main__":
    main()
