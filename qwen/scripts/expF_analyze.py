"""Analysis for Exp F K-quantizer screening.

Reads:
  qwen/results/expF_kquant_stage{N}.jsonl
Writes:
  qwen/results/expF_summary_stage{N}.md         per-condition table
  qwen/results/expF_verdict_matrix_stage{N}.md  verdict + promotion plan

Verdict rules (Stage 1; reused at later stages with the same thresholds):

  acc_upper_ci <= 0.27        -> kill
  0.27 < acc_mean <= 0.34     -> borderline (inspect Δmargin + bf16-pres)
  0.34 < acc_mean < 0.40      -> promote_n100
  0.40 <= acc_mean < 0.45     -> promote_n200
  0.45 <= acc_mean            -> paper_strong

Anchors F0..F3 always carry verdict "anchor".
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------- Bootstrap CI (copied from expD_analyze.py) ----------------


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


# ---------------- Verdict logic ----------------


ANCHORS = {"F0_BF16", "F1_UniformInt4", "F2_TextBF16_VisInt4", "F3_AllKBF16_VInt4"}


def _verdict(cond: str, acc_mean: float, acc_lo: float, acc_hi: float) -> str:
    if cond in ANCHORS:
        return "anchor"
    if acc_hi <= 0.27:
        return "kill"
    if acc_mean <= 0.34:
        return "borderline"
    if acc_mean < 0.40:
        return "promote_n100"
    if acc_mean < 0.45:
        return "promote_n200"
    return "paper_strong"


# ---------------- Summary ----------------


def summarize_f(f_jsonl: Path, out_summary: Path, out_verdict: Path,
                stage: int) -> None:
    if not f_jsonl.exists():
        out_summary.write_text(f"# Exp F Summary stage {stage}\n\n(no F JSONL at {f_jsonl})\n")
        out_verdict.write_text(f"# Exp F Verdict Matrix stage {stage}\n\n(no data)\n")
        return

    rows = [json.loads(l) for l in f_jsonl.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "condition" in r and r.get("error") is None]
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    # Determine ordering by condition name (F0, F1, F2, ...).
    def _key(name: str) -> tuple[int, str]:
        try:
            n = int(name.split("_")[0].lstrip("F"))
        except Exception:
            n = 9999
        return (n, name)
    cond_names = sorted(by_cond.keys(), key=_key)

    # Anchor: F1 (uniform_int4) is the floor for Δmargin baseline.
    f1_rows = by_cond.get("F1_UniformInt4", [])
    f1_margin = (np.mean([float(r["answer_margin"]) for r in f1_rows
                          if r.get("answer_margin") == r.get("answer_margin")])
                 if f1_rows else float("nan"))

    # F0_BF16-correct subset (paired): items where bf16_correct==True.
    bf16_correct_items = {r["item_id"] for r in rows
                          if r.get("condition") == "F0_BF16" and r.get("is_correct")}

    lines = [
        f"# Experiment F — K-quantizer screening summary (stage {stage})\n",
        f"n_rows: {len(rows)}, n_conditions: {len(cond_names)}\n",
        "Reference numbers from prior runs: BF16 ceiling ~0.50-0.57; uniform-INT4 floor ~0.21; "
        "text-K BF16 rescue ~0.385; all-K BF16 + V INT4 (C2.1) ~0.530.\n",
        "## Per-condition table\n",
        "| Condition | n | acc | 95% CI | mean margin | bf16-pres | margin_on_bf16_correct | Δmargin vs F1 | avg KV bits |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    cond_acc: dict[str, dict] = {}
    for cond in cond_names:
        rs = by_cond[cond]
        accs = [int(r["is_correct"]) for r in rs]
        m, lo, hi = bootstrap_ci(accs)
        margin_vals = [float(r["answer_margin"]) for r in rs
                       if r.get("answer_margin") == r.get("answer_margin")]
        margin_mean = float(np.mean(margin_vals)) if margin_vals else float("nan")
        # bf16 preservation (paired): proportion of BF16-correct items that this cond also gets right.
        bf16_subset = [r for r in rs if r["item_id"] in bf16_correct_items]
        bf16_n = len(bf16_subset)
        bf16_pres = (sum(1 for r in bf16_subset if r["is_correct"]) / bf16_n) if bf16_n else float("nan")
        margin_on_bf16 = ([float(r["answer_margin"]) for r in bf16_subset
                           if r.get("answer_margin") == r.get("answer_margin")])
        margin_on_bf16_mean = (float(np.mean(margin_on_bf16)) if margin_on_bf16 else float("nan"))
        delta_margin = (float(margin_mean - f1_margin)
                        if (margin_mean == margin_mean and f1_margin == f1_margin)
                        else float("nan"))
        avg_bits = (float(np.mean([float(r.get("avg_kv_bits", 0)) for r in rs]))
                    if rs else float("nan"))
        lines.append(
            f"| `{cond}` | {len(rs)} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | "
            f"{margin_mean:+.3f} | {bf16_pres:.3f} (n={bf16_n}) | "
            f"{margin_on_bf16_mean:+.3f} | {delta_margin:+.3f} | {avg_bits:.2f} |"
        )
        cond_acc[cond] = {"acc": m, "lo": lo, "hi": hi,
                          "margin": margin_mean, "bf16_pres": bf16_pres,
                          "margin_on_bf16": margin_on_bf16_mean,
                          "delta_margin": delta_margin, "n": len(rs),
                          "avg_bits": avg_bits}

    # Per-bucket
    lines += ["\n## Per-bucket accuracy\n",
              "| Condition | short | mid | long | very_long |",
              "|---|---:|---:|---:|---:|"]
    by_cb: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        by_cb[(r["condition"], r["duration_bucket"])].append(int(r["is_correct"]))
    for cond in cond_names:
        cells = [f"`{cond}`"]
        for bk in ("short", "mid", "long", "very_long"):
            v = by_cb.get((cond, bk), [])
            cells.append(f"{(sum(v)/len(v) if v else 0):.3f} (n={len(v)})" if v else "—")
        lines.append("| " + " | ".join(cells) + " |")

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text("\n".join(lines) + "\n")
    print(f"[expF] wrote {out_summary}")

    # ----------------- Verdict matrix -----------------
    vlines = [
        f"# Exp F Verdict Matrix (stage {stage})\n",
        "Decision rules:\n",
        "- `kill`         : acc_upper_CI <= 0.27",
        "- `borderline`   : 0.27 < acc_mean <= 0.34",
        "- `promote_n100` : 0.34 < acc_mean < 0.40",
        "- `promote_n200` : 0.40 <= acc_mean < 0.45",
        "- `paper_strong` : 0.45 <= acc_mean",
        "- `anchor`       : F0..F3 (reference, no decision)\n",
        "## Verdict by condition\n",
        "| Condition | acc | CI | bf16-pres | Δmargin vs F1 | Verdict |",
        "|---|---:|---|---:|---:|:-:|",
    ]
    for cond in cond_names:
        v = _verdict(cond, cond_acc[cond]["acc"], cond_acc[cond]["lo"],
                     cond_acc[cond]["hi"])
        vlines.append(
            f"| `{cond}` | {cond_acc[cond]['acc']:.3f} | "
            f"[{cond_acc[cond]['lo']:.3f}, {cond_acc[cond]['hi']:.3f}] | "
            f"{cond_acc[cond]['bf16_pres']:.3f} | "
            f"{cond_acc[cond]['delta_margin']:+.3f} | **{v}** |"
        )

    # Promotion plan
    promote_100 = [c for c in cond_names if c not in ANCHORS and
                   _verdict(c, cond_acc[c]["acc"], cond_acc[c]["lo"], cond_acc[c]["hi"])
                   == "promote_n100"]
    promote_200 = [c for c in cond_names if c not in ANCHORS and
                   _verdict(c, cond_acc[c]["acc"], cond_acc[c]["lo"], cond_acc[c]["hi"])
                   in ("promote_n200", "paper_strong")]
    borderline = [c for c in cond_names if c not in ANCHORS and
                  _verdict(c, cond_acc[c]["acc"], cond_acc[c]["lo"], cond_acc[c]["hi"])
                  == "borderline"]

    vlines += [
        "\n## Promotion plan\n",
        f"**Promote to n=100 (Stage 2):** {promote_100 if promote_100 else 'none'}",
        f"**Promote to n=200 (Stage 3):** {promote_200 if promote_200 else 'none'}",
        f"**Borderline (manual review of Δmargin + bf16-pres):** {borderline if borderline else 'none'}",
    ]
    out_verdict.parent.mkdir(parents=True, exist_ok=True)
    out_verdict.write_text("\n".join(vlines) + "\n")
    print(f"[expF] wrote {out_verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--f_jsonl", type=Path, default=None,
                    help="Default: results/expF_kquant_stage{stage}.jsonl")
    ap.add_argument("--summary", type=Path, default=None,
                    help="Default: results/expF_summary_stage{stage}.md")
    ap.add_argument("--verdict", type=Path, default=None,
                    help="Default: results/expF_verdict_matrix_stage{stage}.md")
    args = ap.parse_args()
    if args.f_jsonl is None:
        args.f_jsonl = RESULTS_DIR / f"expF_kquant_stage{args.stage}.jsonl"
    if args.summary is None:
        args.summary = RESULTS_DIR / f"expF_summary_stage{args.stage}.md"
    if args.verdict is None:
        args.verdict = RESULTS_DIR / f"expF_verdict_matrix_stage{args.stage}.md"
    summarize_f(args.f_jsonl, args.summary, args.verdict, args.stage)


if __name__ == "__main__":
    main()
