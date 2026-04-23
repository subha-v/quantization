#!/usr/bin/env python3
"""Analyze the deployable-schemes ExpB run (S1-Bin, S2-Bin, S1-Tern, S2-Tern, Random-Tern).

Reads:
  - results/expB_schemes_rollouts.jsonl   — new conditions (this experiment)
  - results/expB_rollouts.jsonl           — legacy baselines (FP16, W2, AttnEntropy)
  - results/expB_diagnostic_v2.jsonl      — augmented per-cycle records with W2-pass entropy

Writes:
  - results/expB_schemes_summary.md       — bootstrap CI table + matched-seed deltas
                                             + Spearman ρ(FP16-pass, W2-pass) entropy

The hypotheses (H1–H5) from the plan are evaluated explicitly at the bottom.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import utils  # noqa: E402

RESULTS_DIR = Path(utils.RESULTS_DIR)
SCHEMES_ROLLOUTS = RESULTS_DIR / "expB_schemes_rollouts.jsonl"
LEGACY_ROLLOUTS = RESULTS_DIR / "expB_rollouts.jsonl"
DIAG_V2 = RESULTS_DIR / "expB_diagnostic_v2.jsonl"
SUMMARY_PATH = RESULTS_DIR / "expB_schemes_summary.md"

NEW_CONDITIONS = ["S1-Bin", "S2-Bin", "S1-Tern", "S2-Tern", "Random-Tern"]
BASELINE_CONDITIONS = ["FP16", "W2", "AttnEntropy"]
ALL_DISPLAY_ORDER = [
    "FP16", "W2", "AttnEntropy",
    "S1-Bin", "S2-Bin", "S1-Tern", "S2-Tern", "Random-Tern",
]


def load_rows(path: Path) -> list:
    if not path.exists():
        return []
    return utils.load_jsonl(path)


def trial_key(r: dict) -> tuple:
    return (r["suite"], int(r["task_id"]), int(r["seed"]), int(r["episode_idx"]))


def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=0):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def matched_pair_delta(rows_a, rows_b):
    """Per-trial signed delta SR(A) − SR(B), only over matched (suite,task,seed,ep)."""
    by_key_a = {trial_key(r): bool(r["success"]) for r in rows_a}
    by_key_b = {trial_key(r): bool(r["success"]) for r in rows_b}
    common = set(by_key_a) & set(by_key_b)
    if not common:
        return float("nan"), 0
    deltas = [int(by_key_a[k]) - int(by_key_b[k]) for k in common]
    return float(np.mean(deltas)), len(deltas)


def spearman_per_trial(diag_rows):
    """Per-trial Spearman ρ(attn_entropy_l12h2, attn_entropy_l12h2_w2). Returns
    list of (trial_key, n_cycles, rho) tuples."""
    by_trial = defaultdict(list)
    for r in diag_rows:
        by_trial[trial_key(r)].append(r)

    out = []
    for k, rs in by_trial.items():
        rs.sort(key=lambda r: r["cycle_idx"])
        fp = np.array([r.get("attn_entropy_l12h2") for r in rs], dtype=float)
        w2 = np.array([r.get("attn_entropy_l12h2_w2") for r in rs], dtype=float)
        valid = np.isfinite(fp) & np.isfinite(w2)
        if valid.sum() < 4:
            continue
        # Spearman = Pearson on ranks (avoid scipy dep).
        from scipy.stats import rankdata  # type: ignore
        rho = float(np.corrcoef(rankdata(fp[valid]), rankdata(w2[valid]))[0, 1])
        out.append((k, int(valid.sum()), rho))
    return out


def section_per_suite_table(rows_by_cond, suites, conditions):
    lines = []
    lines.append("| Condition | " + " | ".join(suites) + " |")
    lines.append("|---|" + "---:|" * len(suites))
    for cond in conditions:
        if cond not in rows_by_cond:
            continue
        rows = rows_by_cond[cond]
        by_suite = defaultdict(list)
        for r in rows:
            by_suite[r["suite"]].append(int(r["success"]))
        cells = [cond]
        any_cell = False
        for s in suites:
            vals = by_suite.get(s, [])
            if vals:
                any_cell = True
                m, lo, hi = bootstrap_ci(vals)
                cells.append(f"{m:.3f} [{lo:.2f},{hi:.2f}] (n={len(vals)})")
            else:
                cells.append("—")
        if any_cell:
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def section_overall(rows_by_cond, conditions):
    lines = []
    lines.append("| Condition | n | success rate | 95% CI | avg bits |")
    lines.append("|---|---:|---:|---|---:|")
    for cond in conditions:
        if cond not in rows_by_cond:
            continue
        rows = rows_by_cond[cond]
        succ = [int(r["success"]) for r in rows]
        bits = [r.get("condition_avg_bits") for r in rows]
        bits = [b for b in bits if b is not None]
        m, lo, hi = bootstrap_ci(succ)
        bits_str = f"{np.mean(bits):.2f}" if bits else "—"
        lines.append(f"| {cond} | {len(rows)} | {m:.3f} | [{lo:.3f}, {hi:.3f}] | {bits_str} |")
    return lines


def section_matched_deltas(rows_by_cond):
    """Pairwise matched-trial SR deltas for the comparisons that map to H1–H5."""
    pairs = [
        ("H1", "S1-Bin",  "AttnEntropy",  "Scheme 1 binary vs current FP16-pass-derived AttnEntropy (lag tax)"),
        ("H2", "S2-Bin",  "S1-Bin",       "Scheme 2 vs Scheme 1 (no-lag advantage)"),
        ("H3a", "S1-Tern", "S1-Bin",      "ternary vs binary, Scheme 1 (granularity gain at lower bit cost)"),
        ("H3b", "S2-Tern", "S2-Bin",      "ternary vs binary, Scheme 2 (granularity gain at lower bit cost)"),
        ("H4", "S1-Tern", "Random-Tern",  "S1-Tern signal direction vs random ternary"),
        ("H4", "S2-Tern", "Random-Tern",  "S2-Tern signal direction vs random ternary"),
        ("—",  "AttnEntropy", "Random",   "(reference) current AttnEntropy vs random binary"),
    ]
    lines = []
    lines.append("| H | A | B | n_matched | SR(A) − SR(B) | comment |")
    lines.append("|---|---|---|---:|---:|---|")
    for tag, a, b, comment in pairs:
        if a not in rows_by_cond or b not in rows_by_cond:
            continue
        d, n = matched_pair_delta(rows_by_cond[a], rows_by_cond[b])
        sign = "+" if d >= 0 else ""
        lines.append(f"| {tag} | {a} | {b} | {n} | {sign}{d:+.3f} | {comment} |")
    return lines


def section_h5(diag_v2_rows):
    """H5: Spearman ρ(FP16-pass entropy, W2-pass entropy) per trial."""
    lines = []
    if not diag_v2_rows:
        lines.append("_No augmented diagnostic rows found at expB_diagnostic_v2.jsonl — H5 not evaluated._")
        return lines
    rhos = spearman_per_trial(diag_v2_rows)
    if not rhos:
        lines.append("_No trials had ≥4 valid (FP16-pass, W2-pass) pairs — H5 not evaluated._")
        return lines
    rho_vals = np.array([r for (_, _, r) in rhos])
    lines.append(
        f"_n={len(rho_vals)} trials with both entropies. "
        f"median ρ = {float(np.median(rho_vals)):.3f}, "
        f"mean ρ = {float(np.mean(rho_vals)):.3f}, "
        f"P(ρ > 0.6) = {float(np.mean(rho_vals > 0.6)):.2f}, "
        f"min/max = {float(np.min(rho_vals)):.3f}/{float(np.max(rho_vals)):.3f}._"
    )
    # Distribution table
    lines.append("")
    lines.append("| quantile | ρ |")
    lines.append("|---|---:|")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        lines.append(f"| p{int(q*100)} | {float(np.quantile(rho_vals, q)):.3f} |")
    return lines


def main():
    new_rows = load_rows(SCHEMES_ROLLOUTS)
    legacy_rows = load_rows(LEGACY_ROLLOUTS)
    diag_v2_rows = load_rows(DIAG_V2)

    if not new_rows and not legacy_rows:
        print(f"No rollout data found at {SCHEMES_ROLLOUTS} or {LEGACY_ROLLOUTS}; nothing to analyze")
        return 1

    by_cond = defaultdict(list)
    for r in new_rows + legacy_rows:
        by_cond[r["condition"]].append(r)

    suites = sorted({r["suite"] for r in new_rows + legacy_rows})

    lines = []
    lines.append("# ExpB Schemes — Deployable Adaptive Precision (Scheme 1 vs Scheme 2 × Binary vs Ternary)")
    lines.append("")
    lines.append(f"_Generated from {len(new_rows)} schemes rollouts + {len(legacy_rows)} legacy baselines._")
    lines.append("")

    lines.append("## Overall success rate (95% bootstrap CI, 1k boot)")
    lines.append("")
    lines.extend(section_overall(by_cond, ALL_DISPLAY_ORDER))
    lines.append("")

    lines.append("## Per-suite success rate")
    lines.append("")
    lines.extend(section_per_suite_table(by_cond, suites, ALL_DISPLAY_ORDER))
    lines.append("")

    lines.append("## Matched-trial signed deltas (hypothesis tests)")
    lines.append("")
    lines.append("Each row computes SR(A) − SR(B) over the trials present in BOTH conditions.")
    lines.append("Positive = A wins. Matched seeds cancel intrinsic trial difficulty.")
    lines.append("")
    lines.extend(section_matched_deltas(by_cond))
    lines.append("")

    lines.append("## H5 — W2-pass vs FP16-pass entropy correlation per trial")
    lines.append("")
    lines.extend(section_h5(diag_v2_rows))
    lines.append("")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[wrote] {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
