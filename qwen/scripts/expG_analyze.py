"""Analysis for Exp G frame-scaling.

Reads:
  qwen/results/expG_frame_stage{N}.jsonl       (G0..G6 fixed-frame rows)
  qwen/results/expG_frame_stage{N}_G7.jsonl    (cascade post-process; optional)
  qwen/results/expG_frame_stage{N}_G8.jsonl    (type-adaptive post-process; optional)

Writes:
  qwen/results/expG_summary_stage{N}.md         per-condition table + per-bucket
  qwen/results/expG_paired_stage{N}.md          McNemar contingency tables for
                                                the headline pairs:
                                                    G4 vs G0  (matched-memory headline)
                                                    G3 vs G0  (memory-saving)
                                                    G6 vs G2  (zero-loss at 4x frames)
                                                    G7 vs G1  (cascade vs first-pass anchor)
                                                    G8 vs G1  (type-adaptive vs first-pass anchor)
                                                    G4 vs G2  (256f F4 vs 128f BF16)
  qwen/results/expG_frontier_stage{N}.md        Per-condition frame-budget vs
                                                accuracy frontier, sorted by
                                                relative_kv_memory ascending.
  qwen/results/expG_verdict_matrix_stage{N}.md  Verdict + promotion plan to Stage 3.

Verdict rules (G-suite):
  G4 (256f F4) >= G0 + 3 pp                              -> promote_paper_strong  [HEADLINE]
  G4 in [G0 - 3 pp, G0 + 3 pp]                           -> promote_n200          (matched-memory)
  G3 (128f F4) >= G0                                     -> promote_n200          (memory-saving)
  G6 (256f F9) >= G2 - 1 pp                              -> promote_n200          (zero-loss at 4x)
  adaptive (G7 or G8) >= max(G1, G3, G4) + 2 pp          -> promote_n200
  G4 < G0 - 5 pp                                         -> kill (frame coverage doesn't help)
  Anchors G0, G1, G2 always carry verdict "anchor".
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------- Bootstrap CI (copied from expF_analyze.py) ----------------


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


# ---------------- Loading ----------------


ANCHORS = {"G0_BF16", "G1_F4_64f", "G2_BF16_128f"}


HEADLINE_PAIRS = [
    # (a, b) -- a is the "alternative", b is the "baseline".
    ("G4_F4_256f",   "G0_BF16",         "matched_memory_headline"),
    ("G3_F4_128f",   "G0_BF16",         "memory_saving"),
    ("G6_F9_256f",   "G2_BF16_128f",    "zero_loss_at_4x_frames"),
    ("G6_F9_256f",   "G0_BF16",         "f9_256f_vs_baseline"),
    ("G5_F9_128f",   "G0_BF16",         "f9_128f_vs_baseline"),
    ("G7_F4_CascadeAvg128",  "G1_F4_64f",   "cascade_vs_anchor"),
    ("G8_F4_TypeAdaptive",   "G1_F4_64f",   "type_adaptive_vs_anchor"),
    ("G4_F4_256f",   "G2_BF16_128f",    "256f_quant_vs_128f_bf16"),
    # F9-backbone adaptive variants
    ("G7_F9_CascadeAvg192",       "G5_F9_128f",  "f9_cascade_vs_f9_anchor"),
    ("G7_F9_CascadeAvg192",       "G6_F9_256f",  "f9_cascade_vs_f9_top"),
    ("G8_F9_TypeAdaptiveMin128",  "G5_F9_128f",  "f9_type_adaptive_vs_f9_anchor"),
    ("G8_F9_TypeAdaptiveMin128",  "G6_F9_256f",  "f9_type_adaptive_vs_f9_top"),
]


# Sort order for tables (stable across stages).
def _cond_sort_key(name: str) -> tuple[int, str]:
    try:
        n = int(name.split("_")[0].lstrip("G"))
    except Exception:
        n = 9999
    return (n, name)


def load_g_rows(stage: int, in_jsonl: Path,
                cascade_jsonl: Path | None,
                qtype_jsonl: Path | None,
                extra_jsonl: list[Path] | None = None) -> list[dict]:
    """Load G stage rows from base JSONL + cascade + qtype + any extras."""
    rows: list[dict] = []
    paths = [in_jsonl]
    if cascade_jsonl is not None:
        paths.append(cascade_jsonl)
    if qtype_jsonl is not None:
        paths.append(qtype_jsonl)
    if extra_jsonl:
        paths.extend(extra_jsonl)
    for p in paths:
        if p is None or not p.exists():
            continue
        rows.extend(
            json.loads(l) for l in p.read_text().splitlines() if l.strip()
        )
    rows = [r for r in rows if r.get("error") is None and not r.get("skipped", False)]
    return rows


# ---------------- McNemar ----------------


def mcnemar_pair(rows_a: list[dict], rows_b: list[dict]):
    """Return (n_paired, both_correct, a_only, b_only, neither, mcnemar_chi2,
    p_approx) where the chi-square uses the standard McNemar formula
    `(b_only - a_only)**2 / (a_only + b_only)` if (a_only + b_only) > 0,
    else nan. Returns p_approx via 1 - chi2.cdf if scipy is available, else
    nan.

    NB: this is a paired test on item_ids -- both lists are reduced to the
    intersection of their item_ids before counting.
    """
    by_a = {r["item_id"]: r for r in rows_a if "item_id" in r}
    by_b = {r["item_id"]: r for r in rows_b if "item_id" in r}
    ids = sorted(set(by_a) & set(by_b))
    both = a_only = b_only = neither = 0
    for iid in ids:
        ac = bool(by_a[iid].get("is_correct"))
        bc = bool(by_b[iid].get("is_correct"))
        if ac and bc:
            both += 1
        elif ac and not bc:
            a_only += 1
        elif bc and not ac:
            b_only += 1
        else:
            neither += 1
    n = len(ids)
    if (a_only + b_only) == 0:
        chi2 = float("nan")
        p = float("nan")
    else:
        chi2 = ((b_only - a_only) ** 2) / (a_only + b_only)
        try:
            from scipy.stats import chi2 as _chi2dist
            p = float(1.0 - _chi2dist.cdf(chi2, df=1))
        except ImportError:
            p = float("nan")
    return n, both, a_only, b_only, neither, float(chi2), p


# ---------------- Verdict ----------------


def verdict_g(cond: str, by_cond_acc: dict[str, float]) -> str:
    if cond in ANCHORS:
        return "anchor"
    g0 = by_cond_acc.get("G0_BF16", float("nan"))
    g2 = by_cond_acc.get("G2_BF16_128f", float("nan"))
    best_fixed_f4 = max(
        (by_cond_acc.get(c, float("-inf"))
         for c in ("G1_F4_64f", "G3_F4_128f", "G4_F4_256f")),
        default=float("-inf"),
    )
    best_fixed_f9 = max(
        (by_cond_acc.get(c, float("-inf"))
         for c in ("G5_F9_128f", "G6_F9_256f")),
        default=float("-inf"),
    )
    acc = by_cond_acc.get(cond, float("nan"))
    if cond == "G4_F4_256f":
        if acc >= g0 + 0.03:
            return "promote_paper_strong"
        if acc >= g0 - 0.03:
            return "promote_n200"
        if acc < g0 - 0.05:
            return "kill"
        return "borderline"
    if cond == "G3_F4_128f":
        if acc >= g0:
            return "promote_n200"
        if acc < g0 - 0.05:
            return "kill"
        return "borderline"
    if cond == "G6_F9_256f":
        if acc >= g2 - 0.01:
            return "promote_n200"
        return "borderline"
    if cond == "G5_F9_128f":
        # Smaller-frame F9 is informational; promote if it hits G0.
        if acc >= g0 - 0.01:
            return "promote_n200"
        return "borderline"
    if cond in ("G7_F4_CascadeAvg128", "G8_F4_TypeAdaptive"):
        if acc >= best_fixed_f4 + 0.02:
            return "promote_n200"
        return "borderline"
    # F9-backbone adaptive variants: compare against best fixed-F9 condition.
    if cond in ("G7_F9_CascadeAvg192", "G8_F9_TypeAdaptiveMin128"):
        if acc >= best_fixed_f9 + 0.02:
            return "promote_n200"
        if acc >= best_fixed_f9 - 0.01:
            return "borderline"
        return "borderline"
    return "unranked"


# ---------------- Summary writer ----------------


def write_summary(rows: list[dict], stage: int, out_summary: Path) -> dict:
    """Returns a per-condition acc dict for downstream use."""
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    cond_names = sorted(by_cond.keys(), key=_cond_sort_key)

    g0_correct_items = {r["item_id"] for r in rows
                        if r.get("condition") == "G0_BF16" and r.get("is_correct")}

    lines = [
        f"# Experiment G -- frame-scaling summary (stage {stage})\n",
        f"n_rows: {len(rows)}, n_conditions: {len(cond_names)}\n",
        "Reference: BF16 ceiling ~0.50-0.57; F4 INT4 anchor ~0.545; F9 4.75-bit ~0.560.\n",
        "## Per-condition table\n",
        "| Condition | n | acc | 95% CI | mean margin | g0-pres | "
        "frames | rel_kv_mem | avg_kv_bits | class |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]

    cond_acc: dict[str, dict] = {}
    for cond in cond_names:
        rs = by_cond[cond]
        accs = [int(r.get("is_correct", False)) for r in rs]
        m, lo, hi = bootstrap_ci(accs)
        margin_vals = [float(r["answer_margin"]) for r in rs
                       if r.get("answer_margin") == r.get("answer_margin")]
        margin_mean = float(np.mean(margin_vals)) if margin_vals else float("nan")
        # G0-correct preservation: of the items G0 got right, how many does this cond also get right?
        g0_subset = [r for r in rs if r["item_id"] in g0_correct_items]
        g0_n = len(g0_subset)
        g0_pres = (sum(1 for r in g0_subset if r.get("is_correct")) / g0_n) if g0_n else float("nan")
        # frame budget reporting: for type_adaptive / cascade, give the realized average.
        frame_vals = [int(r.get("assigned_frames", r.get("frames", 0))) for r in rs]
        frames_repr = (
            f"{int(np.mean(frame_vals))}" if len(set(frame_vals)) == 1 else
            f"~{np.mean(frame_vals):.0f} (mixed)"
        )
        rel_mem_vals = [float(r["relative_kv_memory"]) for r in rs
                        if "relative_kv_memory" in r and r["relative_kv_memory"] is not None]
        rel_mem = float(np.mean(rel_mem_vals)) if rel_mem_vals else float("nan")
        kv_bits_vals = [float(r["avg_kv_bits"]) for r in rs if "avg_kv_bits" in r]
        avg_kv_bits = float(np.mean(kv_bits_vals)) if kv_bits_vals else float("nan")
        cond_class = (rs[0].get("condition_class", "fixed_frame") if rs else "fixed_frame")
        lines.append(
            f"| `{cond}` | {len(rs)} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | "
            f"{margin_mean:+.3f} | {g0_pres:.3f} (n={g0_n}) | "
            f"{frames_repr} | {rel_mem:.3f} | {avg_kv_bits:.3f} | {cond_class} |"
        )
        cond_acc[cond] = {
            "acc": m, "lo": lo, "hi": hi, "n": len(rs),
            "margin": margin_mean, "g0_pres": g0_pres,
            "frames_repr": frames_repr, "rel_mem": rel_mem,
            "avg_kv_bits": avg_kv_bits, "cond_class": cond_class,
        }

    # Per-bucket
    lines += [
        "\n## Per-bucket accuracy\n",
        "| Condition | short | mid | long | very_long |",
        "|---|---:|---:|---:|---:|",
    ]
    by_cb: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        if "duration_bucket" in r:
            by_cb[(r["condition"], r["duration_bucket"])].append(int(r.get("is_correct", False)))
    for cond in cond_names:
        cells = [f"`{cond}`"]
        for bk in ("short", "mid", "long", "very_long"):
            v = by_cb.get((cond, bk), [])
            cells.append(f"{(sum(v)/len(v) if v else 0):.3f} (n={len(v)})" if v else "—")
        lines.append("| " + " | ".join(cells) + " |")

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text("\n".join(lines) + "\n")
    print(f"[expG] wrote {out_summary}")
    return cond_acc


def write_paired(rows: list[dict], stage: int, out_paired: Path) -> None:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    lines = [
        f"# Exp G paired comparisons (stage {stage})\n",
        "Each pair is restricted to item_ids present in both conditions. The "
        "'new-evidence wins' metric is `b_only - a_only` (signed): a positive "
        "value means the alternative recovered items the baseline got wrong.\n",
        "| Pair | label | n_paired | both_correct | a_only_correct | b_only_correct | "
        "neither | new_evidence_wins | acc(a) | acc(b) | mcnemar_chi2 | p |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b, label in HEADLINE_PAIRS:
        ra = by_cond.get(a, [])
        rb = by_cond.get(b, [])
        if not ra or not rb:
            lines.append(f"| `{a}` vs `{b}` | {label} | — | — | — | — | — | — | — | — | — | — |")
            continue
        n, both, a_only, b_only, neither, chi2, p = mcnemar_pair(ra, rb)
        acc_a = (both + a_only) / max(1, n)
        acc_b = (both + b_only) / max(1, n)
        new_ev = a_only - b_only  # rows_a substitutes for rows_b -- a "wins" means a_only > b_only
        # Phrase consistently: "new_evidence_wins" is signed against the *alternative* (a).
        lines.append(
            f"| `{a}` vs `{b}` | {label} | {n} | {both} | {a_only} | {b_only} | "
            f"{neither} | {new_ev:+d} | {acc_a:.3f} | {acc_b:.3f} | "
            f"{chi2:.3f} | {p:.3f} |"
        )

    out_paired.parent.mkdir(parents=True, exist_ok=True)
    out_paired.write_text("\n".join(lines) + "\n")
    print(f"[expG] wrote {out_paired}")


def write_frontier(cond_acc: dict[str, dict], stage: int, out_frontier: Path) -> None:
    lines = [
        f"# Exp G frame-budget vs accuracy frontier (stage {stage})\n",
        "Sorted ascending by relative_kv_memory. Conditions on the frontier "
        "are those that maximize accuracy at their memory budget.\n",
        "| Condition | frames | avg_kv_bits | rel_kv_mem | acc | 95% CI | class |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    rows_to_sort = []
    for cond, info in cond_acc.items():
        rel = info.get("rel_mem", float("nan"))
        rows_to_sort.append((rel, cond, info))
    rows_to_sort.sort(key=lambda x: (x[0] if not (x[0] != x[0]) else float("inf"), x[1]))
    for rel, cond, info in rows_to_sort:
        lines.append(
            f"| `{cond}` | {info['frames_repr']} | {info['avg_kv_bits']:.3f} | "
            f"{info['rel_mem']:.3f} | {info['acc']:.3f} | "
            f"[{info['lo']:.3f}, {info['hi']:.3f}] | {info['cond_class']} |"
        )

    out_frontier.parent.mkdir(parents=True, exist_ok=True)
    out_frontier.write_text("\n".join(lines) + "\n")
    print(f"[expG] wrote {out_frontier}")


def write_verdict(cond_acc: dict[str, dict], stage: int, out_verdict: Path) -> None:
    by_cond_acc = {c: info["acc"] for c, info in cond_acc.items()}
    g0 = by_cond_acc.get("G0_BF16", float("nan"))
    g2 = by_cond_acc.get("G2_BF16_128f", float("nan"))
    cond_names = sorted(cond_acc.keys(), key=_cond_sort_key)

    lines = [
        f"# Exp G Verdict Matrix (stage {stage})\n",
        "Decision rules:\n",
        "- `G4 (256f F4) >= G0 + 3 pp`                                                -> `promote_paper_strong`",
        "- `G4 in [G0 - 3 pp, G0 + 3 pp]`                                            -> `promote_n200` (matched-memory)",
        "- `G3 (128f F4) >= G0`                                                      -> `promote_n200` (memory-saving)",
        "- `G6 (256f F9) >= G2 - 1 pp`                                               -> `promote_n200` (zero-loss at 4x)",
        "- `adaptive (G7 or G8) >= max(G1, G3, G4) + 2 pp`                           -> `promote_n200`",
        "- `G4 < G0 - 5 pp`                                                          -> `kill`",
        "- Anchors G0, G1, G2 always carry verdict `anchor`.\n",
        f"## Anchors\n",
        f"- G0 (64f BF16) = {g0:.3f}",
        f"- G2 (128f BF16) = {g2:.3f}\n",
        "## Verdict by condition\n",
        "| Condition | acc | CI | rel_kv_mem | g0-pres | Verdict |",
        "|---|---:|---|---:|---:|:-:|",
    ]
    for cond in cond_names:
        info = cond_acc[cond]
        v = verdict_g(cond, by_cond_acc)
        lines.append(
            f"| `{cond}` | {info['acc']:.3f} | "
            f"[{info['lo']:.3f}, {info['hi']:.3f}] | {info['rel_mem']:.3f} | "
            f"{info['g0_pres']:.3f} | **{v}** |"
        )
    promote_200 = [c for c in cond_names
                   if verdict_g(c, by_cond_acc) in ("promote_n200", "promote_paper_strong")]
    paper_strong = [c for c in cond_names
                    if verdict_g(c, by_cond_acc) == "promote_paper_strong"]
    killed = [c for c in cond_names if verdict_g(c, by_cond_acc) == "kill"]
    borderline = [c for c in cond_names if verdict_g(c, by_cond_acc) == "borderline"]

    lines += [
        "\n## Promotion plan to Stage 3 (n=200)\n",
        f"**Paper-strong:** {paper_strong if paper_strong else 'none'}",
        f"**Promote to n=200:** {promote_200 if promote_200 else 'none'}",
        f"**Borderline (manual review):** {borderline if borderline else 'none'}",
        f"**Killed:** {killed if killed else 'none'}",
    ]
    out_verdict.parent.mkdir(parents=True, exist_ok=True)
    out_verdict.write_text("\n".join(lines) + "\n")
    print(f"[expG] wrote {out_verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--in_jsonl", type=Path, default=None)
    ap.add_argument("--cascade_jsonl", type=Path, default=None)
    ap.add_argument("--qtype_jsonl", type=Path, default=None)
    ap.add_argument("--extra_jsonl", type=Path, nargs="*", default=None,
                    help="Additional JSONL files to merge in (e.g. F9-backbone "
                         "cascade/type-adaptive stitched outputs).")
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--paired", type=Path, default=None)
    ap.add_argument("--frontier", type=Path, default=None)
    ap.add_argument("--verdict", type=Path, default=None)
    args = ap.parse_args()

    if args.in_jsonl is None:
        args.in_jsonl = RESULTS_DIR / f"expG_frame_stage{args.stage}.jsonl"
    if args.cascade_jsonl is None:
        args.cascade_jsonl = RESULTS_DIR / f"expG_frame_stage{args.stage}_G7.jsonl"
    if args.qtype_jsonl is None:
        args.qtype_jsonl = RESULTS_DIR / f"expG_frame_stage{args.stage}_G8.jsonl"
    if args.summary is None:
        args.summary = RESULTS_DIR / f"expG_summary_stage{args.stage}.md"
    if args.paired is None:
        args.paired = RESULTS_DIR / f"expG_paired_stage{args.stage}.md"
    if args.frontier is None:
        args.frontier = RESULTS_DIR / f"expG_frontier_stage{args.stage}.md"
    if args.verdict is None:
        args.verdict = RESULTS_DIR / f"expG_verdict_matrix_stage{args.stage}.md"

    rows = load_g_rows(args.stage, args.in_jsonl, args.cascade_jsonl,
                       args.qtype_jsonl, extra_jsonl=args.extra_jsonl)
    if not rows:
        msg = f"# Exp G stage {args.stage}\n\n(no data: ran with empty JSONLs)\n"
        for path in (args.summary, args.paired, args.frontier, args.verdict):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(msg)
        return

    cond_acc = write_summary(rows, args.stage, args.summary)
    write_paired(rows, args.stage, args.paired)
    write_frontier(cond_acc, args.stage, args.frontier)
    write_verdict(cond_acc, args.stage, args.verdict)


if __name__ == "__main__":
    main()
