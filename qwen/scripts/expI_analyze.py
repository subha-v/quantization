"""Experiment I — Mechanism-screen analyzer.

Reads expI_tempkivi_stage{N}_seed1.jsonl (+ I15/I16 stitched JSONL) and
emits four outputs:

  qwen/results/expI_summary_stage{N}.md       Per-condition accuracy table
                                              with bootstrap 95% CI and
                                              relative KV memory.
  qwen/results/expI_paired_stage{N}.md        Headline McNemar pairs +
                                              new-evidence-wins vs damage
                                              decomposition.
  qwen/results/expI_verdict_matrix_stage{N}.md  Verdict per variant
                                              (paper_strong / promote_n200 /
                                              borderline / kill / anchor).
  qwen/results/expI_promote_stage1.json       (stage 1 only) JSON list of
                                              variant condition names that
                                              passed promotion. Read by
                                              expI_temporal_kivi.py at
                                              Stage 3.

The Stage-3 promotion JSON is the data-driven gate for the I-driver:
{ "promoted": [<variant cond names>] }. Variants are from
VARIANTS_DATA_DRIVEN ∪ {I15, I16}; anchors are always promoted by the driver
itself (ANCHORS_ALWAYS_PROMOTED in expI_temporal_kivi).

Reuses helpers from expG_analyze:
  bootstrap_ci, mcnemar_pair

Headline pairs (HEADLINE_PAIRS_I) cover:
  - Mechanism: I3 vs I4 (modality split alone), I3 vs I5 (scale-group count),
               I3 vs I6 (window granularity), I11 vs I12 (256f mod-blind ctrl)
  - Add-ons:   I3 vs I7 (VidKV V), I3 vs I8 (outlier-8 on TempWin),
               I11 vs I13 (TempWin4 + outlier-8), I11 vs I14 (256f VidKV V)
  - Hybrid:    I15 vs I16 (duration vs random matched-rate),
               I15 vs I3, I15 vs I2

Usage:
  python expI_analyze.py --stage 1
  python expI_analyze.py --stage 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from expG_analyze import bootstrap_ci, mcnemar_pair


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# Anchors carry verdict "anchor" automatically; never promoted/killed.
ANCHORS_I = {"I0_BF16_128f", "I1_F4_128f", "I9_F4_256f"}


# Pairs to surface in the paired-test table.
# Convention: (a, b, label) where a is the alternative and b is the baseline.
HEADLINE_PAIRS_I = [
    # --- Mechanism @128f ---
    ("I3_TempWin2_128f",        "I4_TextVisualSplit_128f",     "tempwin_vs_modality_split_only_128f"),
    ("I3_TempWin2_128f",        "I5_TokenBlock4_128f",         "modality_aware_vs_blind_128f"),
    ("I3_TempWin2_128f",        "I6_TempWin4_128f",            "windowcount_2_vs_4_128f"),
    # --- Add-ons @128f ---
    ("I3_TempWin2_128f",        "I7_TempWin2_VidKVV_128f",     "vidkv_v_addition_128f"),
    ("I3_TempWin2_128f",        "I8_TempWin2_Outlier8_128f",   "outlier8_addition_128f"),
    # --- Reproducibility of H6 win on seed=1 ---
    ("I3_TempWin2_128f",        "I1_F4_128f",                  "tempwin2_vs_f4_128f"),
    ("I3_TempWin2_128f",        "I2_F9_128f",                  "tempwin2_vs_f9_128f"),
    # --- Mechanism @256f ---
    ("I11_TempWin4_256f",       "I12_TokenBlock6_256f",        "modality_aware_vs_blind_256f"),
    # --- Add-ons @256f ---
    ("I11_TempWin4_256f",       "I13_TempWin4_Outlier8_256f",  "outlier8_addition_256f"),
    ("I11_TempWin4_256f",       "I14_TempWin4_VidKVV_256f",    "vidkv_v_addition_256f"),
    # --- Reproducibility @256f ---
    ("I11_TempWin4_256f",       "I9_F4_256f",                  "tempwin4_vs_f4_256f"),
    ("I11_TempWin4_256f",       "I10_F9_256f",                 "tempwin4_vs_f9_256f"),
    # --- Duration-hybrid policy ---
    ("I15_F9MidElseTempWin",    "I16_F9RandomMatched",         "duration_vs_random_hybrid"),
    ("I15_F9MidElseTempWin",    "I3_TempWin2_128f",            "hybrid_vs_tempwin_only"),
    ("I15_F9MidElseTempWin",    "I2_F9_128f",                  "hybrid_vs_f9_only"),
]


def _cond_sort_key(name: str) -> tuple[int, str]:
    """Sort I conditions by their numeric suffix (I0, I1, ..., I16)."""
    try:
        n = int(name.split("_")[0].lstrip("I"))
    except Exception:
        n = 9999
    return (n, name)


def load_i_rows(in_jsonl: Path,
                hybrid_jsonls: list[Path] | None = None) -> list[dict]:
    """Load I-stage rows from base JSONL plus any hybrid stitched JSONLs."""
    rows: list[dict] = []
    paths = [in_jsonl]
    if hybrid_jsonls:
        paths.extend(hybrid_jsonls)
    for p in paths:
        if p is None or not p.exists():
            continue
        rows.extend(
            json.loads(l) for l in p.read_text().splitlines() if l.strip()
        )
    rows = [r for r in rows if r.get("error") is None and not r.get("skipped", False)]
    return rows


def verdict_i(cond: str, by_cond_acc: dict[str, float]) -> str:
    """Return the Stage-{N} verdict for `cond`.

    paper_strong: Δ ≥ +3pp vs the most relevant rival (mechanism or anchor)
                  AND McNemar χ² will need to be checked separately in the
                  paired table; this verdict is necessary-not-sufficient.
    promote_n200: matched within ±1pp of the relevant rival.
    kill:         Δ ≤ −5pp vs the relevant rival.
    borderline:   else.
    """
    if cond in ANCHORS_I:
        return "anchor"
    acc = by_cond_acc.get(cond, float("nan"))
    if not np.isfinite(acc):
        return "missing"

    # Mechanism conditions: compare vs I3 (128f anchor for the proposed method).
    i3 = by_cond_acc.get("I3_TempWin2_128f", float("nan"))
    i11 = by_cond_acc.get("I11_TempWin4_256f", float("nan"))
    i2 = by_cond_acc.get("I2_F9_128f", float("nan"))

    # I3 itself is judged vs F4 / F9 baselines (does H6 reproduce?).
    if cond == "I3_TempWin2_128f":
        i1 = by_cond_acc.get("I1_F4_128f", float("nan"))
        if np.isfinite(i1) and acc >= i1 + 0.03:
            return "promote_paper_strong"
        if np.isfinite(i1) and acc >= i1 - 0.01:
            return "promote_n200"
        if np.isfinite(i1) and acc < i1 - 0.05:
            return "kill"
        return "borderline"

    # I11 (TempWin4 256f) judged vs F4/F9 256f.
    if cond == "I11_TempWin4_256f":
        i9 = by_cond_acc.get("I9_F4_256f", float("nan"))
        if np.isfinite(i9) and acc >= i9 + 0.03:
            return "promote_paper_strong"
        if np.isfinite(i9) and acc >= i9 - 0.01:
            return "promote_n200"
        if np.isfinite(i9) and acc < i9 - 0.05:
            return "kill"
        return "borderline"

    # Mechanism controls: I4, I5, I12 — compare vs the proposed method on the
    # same frame tier. These are "controls" — they should LOSE to the proposal.
    # We promote them to Stage 3 as anchors anyway (always run); verdict here
    # is for reporting only.
    if cond in ("I4_TextVisualSplit_128f", "I5_TokenBlock4_128f"):
        if np.isfinite(i3) and acc >= i3 + 0.03:
            return "promote_paper_strong"  # surprising: control beats proposal
        if np.isfinite(i3) and acc >= i3 - 0.01:
            return "promote_n200"  # control matches proposal — bad sign for mechanism
        return "borderline"  # control loses — expected if mechanism holds
    if cond == "I12_TokenBlock6_256f":
        if np.isfinite(i11) and acc >= i11 + 0.03:
            return "promote_paper_strong"
        if np.isfinite(i11) and acc >= i11 - 0.01:
            return "promote_n200"
        return "borderline"

    # Add-on variants — promote if they beat their respective TempWin anchor.
    if cond in ("I6_TempWin4_128f", "I7_TempWin2_VidKVV_128f", "I8_TempWin2_Outlier8_128f"):
        if np.isfinite(i3) and acc >= i3 + 0.03:
            return "promote_paper_strong"
        if np.isfinite(i3) and acc >= i3 - 0.01:
            return "promote_n200"
        if np.isfinite(i3) and acc < i3 - 0.05:
            return "kill"
        return "borderline"
    if cond in ("I13_TempWin4_Outlier8_256f", "I14_TempWin4_VidKVV_256f"):
        if np.isfinite(i11) and acc >= i11 + 0.03:
            return "promote_paper_strong"
        if np.isfinite(i11) and acc >= i11 - 0.01:
            return "promote_n200"
        if np.isfinite(i11) and acc < i11 - 0.05:
            return "kill"
        return "borderline"

    # Duration-hybrid (I15) — promote if it beats both source conditions.
    if cond == "I15_F9MidElseTempWin":
        rivals = [v for v in (i3, i2) if np.isfinite(v)]
        if not rivals:
            return "borderline"
        best_rival = max(rivals)
        if acc >= best_rival + 0.03:
            return "promote_paper_strong"
        if acc >= best_rival - 0.01:
            return "promote_n200"
        return "borderline"
    # Random-hybrid (I16) — control for I15; never auto-promoted unless it
    # surprisingly beats both source conditions (would invalidate I15).
    if cond == "I16_F9RandomMatched":
        return "control"

    return "borderline"


def write_summary(rows: list[dict], stage: int, out_summary: Path) -> dict:
    """Per-condition accuracy + bootstrap CI + relative KV memory.

    Returns a dict {cond: {"acc": float, "n": int, "ci_lo": float, "ci_hi":
    float, "rel_kv_memory": float}} for downstream use.
    """
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

    lines = [f"# Exp I summary — Stage {stage}", ""]
    lines.append(f"_n items = {len(rows) // max(1, len(cond_acc))}; "
                 f"95% CI bootstrap n_boot=2000_")
    lines.append("")
    lines.append("| Condition | n_frames | acc | 95% CI | n | rel_kv_mem | avg_kv_bits | verdict |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---|")
    for cond in sorted(cond_acc, key=_cond_sort_key):
        d = cond_acc[cond]
        v = verdict_i(cond, by_cond_acc)
        lines.append(
            f"| {cond} | {d['n_frames']} | {d['acc']:.3f} | "
            f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}] | {d['n']} | "
            f"{d['rel_kv_memory']:.3f} | {d['avg_kv_bits']:.3f} | {v} |"
        )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text("\n".join(lines) + "\n")
    print(f"[expI] wrote {out_summary}")
    return cond_acc


def write_paired(rows: list[dict], stage: int, out_paired: Path) -> None:
    by_cond: dict[str, list[dict]] = {}
    for r in rows:
        c = r.get("condition")
        if not c:
            continue
        by_cond.setdefault(c, []).append(r)

    lines = [f"# Exp I paired McNemar — Stage {stage}", ""]
    lines.append("`(a, b)` = (alternative, baseline). χ² = "
                 "`(b_only - a_only)² / (a_only + b_only)`. p ≈ from χ²₁.")
    lines.append("")
    lines.append("| label | a | b | n_paired | acc(a) | acc(b) | a_only | b_only | both | neither | χ² | p |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a, b, label in HEADLINE_PAIRS_I:
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
    print(f"[expI] wrote {out_paired}")


def write_verdict(cond_acc: dict[str, dict], stage: int, out_verdict: Path,
                  out_promote_json: Path | None = None) -> None:
    by_cond_acc = {c: d["acc"] for c, d in cond_acc.items()}
    lines = [f"# Exp I verdict matrix — Stage {stage}", ""]
    lines.append("- Anchors I0, I1, I9 always carry verdict `anchor`.")
    lines.append("- Mechanism controls I4 / I5 / I12 are expected to LOSE "
                 "to their TempWin counterpart — `borderline` is the success "
                 "case for the temporal-locality hypothesis.")
    lines.append("- I3 / I11 are judged vs F4 baseline at the same frame tier.")
    lines.append("- Add-on variants (I6 / I7 / I8 / I13 / I14) and the hybrid "
                 "I15 are judged vs their respective TempWin anchors / source.")
    lines.append("")
    lines.append("| Condition | acc | rel_kv_mem | avg_kv_bits | verdict |")
    lines.append("|---|---:|---:|---:|---|")
    cond_names = sorted(cond_acc, key=_cond_sort_key)
    for cond in cond_names:
        d = cond_acc[cond]
        v = verdict_i(cond, by_cond_acc)
        lines.append(
            f"| {cond} | {d['acc']:.3f} | {d['rel_kv_memory']:.3f} | "
            f"{d['avg_kv_bits']:.3f} | {v} |"
        )

    promoted_strong = [c for c in cond_names
                       if verdict_i(c, by_cond_acc) == "promote_paper_strong"]
    promoted_n200 = [c for c in cond_names
                     if verdict_i(c, by_cond_acc) in ("promote_n200", "promote_paper_strong")]
    killed = [c for c in cond_names if verdict_i(c, by_cond_acc) == "kill"]
    borderline = [c for c in cond_names if verdict_i(c, by_cond_acc) == "borderline"]

    lines.append("")
    lines.append(f"**paper_strong**: {promoted_strong}")
    lines.append(f"**promote_n200**: {promoted_n200}")
    lines.append(f"**borderline**:  {borderline}")
    lines.append(f"**kill**:        {killed}")
    out_verdict.parent.mkdir(parents=True, exist_ok=True)
    out_verdict.write_text("\n".join(lines) + "\n")
    print(f"[expI] wrote {out_verdict}")

    # Stage-1 promotion JSON: drive Stage-3 condition gating.
    # Variants from VARIANTS_DATA_DRIVEN that get any promote_* verdict;
    # plus I15/I16 if I15 gets promote_*.
    if out_promote_json is not None:
        VARIANTS = {
            "I6_TempWin4_128f", "I7_TempWin2_VidKVV_128f", "I8_TempWin2_Outlier8_128f",
            "I12_TokenBlock6_256f", "I13_TempWin4_Outlier8_256f", "I14_TempWin4_VidKVV_256f",
        }
        promoted_variants = [c for c in promoted_n200 if c in VARIANTS]
        if "I15_F9MidElseTempWin" in promoted_n200:
            promoted_variants.append("I15_F9MidElseTempWin")
            promoted_variants.append("I16_F9RandomMatched")  # always carry the matched control
        out_promote_json.write_text(json.dumps({"promoted": promoted_variants}, indent=2))
        print(f"[expI] wrote {out_promote_json}: {promoted_variants}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[1, 3], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--in_jsonl", type=Path, default=None,
                    help="Default: results/expI_tempkivi_stage{stage}_seed{seed}.jsonl")
    ap.add_argument("--hybrid_jsonl", type=Path, nargs="*", default=None,
                    help="I15/I16 stitched JSONLs from expI_duration_hybrid.py.")
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--paired", type=Path, default=None)
    ap.add_argument("--verdict", type=Path, default=None)
    ap.add_argument("--promote_json", type=Path, default=None,
                    help="Stage-1 only. Default: results/expI_promote_stage1.json")
    args = ap.parse_args()

    if args.in_jsonl is None:
        args.in_jsonl = RESULTS_DIR / f"expI_tempkivi_stage{args.stage}_seed{args.seed}.jsonl"
    if args.hybrid_jsonl is None:
        cand_i15 = args.in_jsonl.with_name(args.in_jsonl.stem + "_I15.jsonl")
        cand_i16 = args.in_jsonl.with_name(args.in_jsonl.stem + "_I16.jsonl")
        args.hybrid_jsonl = [p for p in (cand_i15, cand_i16) if p.exists()]
    if args.summary is None:
        args.summary = RESULTS_DIR / f"expI_summary_stage{args.stage}.md"
    if args.paired is None:
        args.paired = RESULTS_DIR / f"expI_paired_stage{args.stage}.md"
    if args.verdict is None:
        args.verdict = RESULTS_DIR / f"expI_verdict_matrix_stage{args.stage}.md"
    if args.stage == 1 and args.promote_json is None:
        args.promote_json = RESULTS_DIR / "expI_promote_stage1.json"

    if not args.in_jsonl.exists():
        raise SystemExit(f"[expI] input JSONL not found: {args.in_jsonl}")

    rows = load_i_rows(args.in_jsonl, args.hybrid_jsonl)
    print(f"[expI] loaded {len(rows)} rows from {args.in_jsonl}"
          + (f" + {len(args.hybrid_jsonl)} hybrid files" if args.hybrid_jsonl else ""),
          flush=True)

    cond_acc = write_summary(rows, args.stage, args.summary)
    write_paired(rows, args.stage, args.paired)
    write_verdict(cond_acc, args.stage, args.verdict,
                  out_promote_json=args.promote_json if args.stage == 1 else None)


if __name__ == "__main__":
    main()
