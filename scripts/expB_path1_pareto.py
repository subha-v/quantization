#!/usr/bin/env python3
"""
Path 1 analysis: SR-vs-bits Pareto plot for the static-vs-dynamic late-layer
W2 comparison (Static, Random-Bin, AttnEnt-Bin, Hybrid). Combines the new
tagged rollouts (--out-tag static_dynamic_n100) with existing baselines from
the production W4 rollouts file (FP16, W4-Floor, AttnEntropy-W4, S3-Tern-W4-l12h2).

Outputs:
  results/expB_path1_pareto.png   — SR vs param-weighted avg bits
  results/expB_path1_pareto.md    — focused comparison table + matched-pair McNemar

Usage:
    cd /data/subha2/quantization && \
      PYTHONPATH=/data/subha2/openpi/third_party/libero \
      /data/subha2/openpi/.venv/bin/python scripts/expB_path1_pareto.py

No GPU needed.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.environ.get("WORKSPACE", "/data/subha2") + "/.matplotlib",
)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_DIR = Path(
    os.environ.get(
        "EXPERIMENT_DIR",
        os.path.join(os.environ.get("WORKSPACE", "/data/subha2"), "experiments"),
    )
) / "results"

PATH1_TAG = "static_dynamic_n100"
NEW_ROLLOUTS = RESULTS_DIR / f"expB_w4__{PATH1_TAG}_rollouts.jsonl"
BASELINE_ROLLOUTS = RESULTS_DIR / "expB_w4_rollouts.jsonl"
OUT_PLOT = RESULTS_DIR / "expB_path1_pareto.png"
OUT_MD = RESULTS_DIR / "expB_path1_pareto.md"

BASELINES_TO_OVERLAY = {
    "FP16",
    "W4-Floor",
    "AttnEntropy-W4",
    "S3-Tern-W4-l12h2",
}

PATH1_CONDITIONS = [
    "Static-W2-l13-17",
    "Random-Bin-W2-l13-17-f25",
    "Random-Bin-W2-l13-17-f50",
    "Random-Bin-W2-l13-17-f75",
    "AttnEnt-Bin-W2-l13-17-f25",
    "AttnEnt-Bin-W2-l13-17-f50",
    "AttnEnt-Bin-W2-l13-17-f75",
    "Hybrid-Static-W2-AttnEnt-FP16-f50",
]

# Plot styling per condition family.
COND_STYLE = {
    "FP16":            dict(color="#0a0a0a", marker="*",  s=260, label="FP16 (ceiling)"),
    "W4-Floor":        dict(color="#777777", marker="s",  s=180, label="W4-Floor"),
    "AttnEntropy-W4":  dict(color="#1f77b4", marker="d",  s=160, label="AttnEntropy-W4 (legacy)"),
    "S3-Tern-W4-l12h2":dict(color="#2ca02c", marker="P",  s=200, label="S3-Tern-W4-l12h2 (existing dynamic)"),
    "Static-W2-l13-17":dict(color="#d62728", marker="X",  s=220, label="Static-W2-l13-17 (NEW)"),
}
RANDOM_STYLE = dict(color="#9467bd", marker="o", s=140)
ATTNENT_STYLE = dict(color="#ff7f0e", marker="^", s=160)
HYBRID_STYLE = dict(color="#8c564b", marker="D", s=180, label="Hybrid-Static-W2 + AttnEnt-FP16-f50 (NEW)")


def load_jsonl(path):
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def bootstrap_ci(vals, n_boot=10000, alpha=0.05, seed=0):
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def collect(rows, conditions):
    """Return {cond: {"sr": float, "lo": float, "hi": float, "bits_param": float, "n": int}}."""
    by_cond = defaultdict(list)
    bits_by_cond = defaultdict(list)
    for r in rows:
        c = r.get("condition")
        if conditions is None or c in conditions:
            by_cond[c].append(bool(r["success"]))
            bp = r.get("condition_avg_bits_param")
            if bp is not None:
                bits_by_cond[c].append(float(bp))
    out = {}
    for c, succ in by_cond.items():
        m, lo, hi = bootstrap_ci(succ)
        bp_vals = bits_by_cond.get(c, [])
        bp_mean = float(np.mean(bp_vals)) if bp_vals else float("nan")
        out[c] = {"sr": m, "lo": lo, "hi": hi, "bits_param": bp_mean, "n": len(succ), "successes": succ}
    return out


def matched_pair_delta(rows, A, B):
    """SR(A) − SR(B) over trials present in both. Returns (delta, n_matched, mcnemar_p)."""
    keyA = {(r["suite"], r["task_id"], r["seed"], r["episode_idx"]): bool(r["success"]) for r in rows if r["condition"] == A}
    keyB = {(r["suite"], r["task_id"], r["seed"], r["episode_idx"]): bool(r["success"]) for r in rows if r["condition"] == B}
    common = sorted(set(keyA) & set(keyB))
    if not common:
        return float("nan"), 0, float("nan")
    a_only = sum(1 for k in common if keyA[k] and not keyB[k])
    b_only = sum(1 for k in common if keyB[k] and not keyA[k])
    delta = (sum(keyA[k] for k in common) - sum(keyB[k] for k in common)) / len(common)
    # McNemar exact two-sided binomial: discordant pairs only
    n_disc = a_only + b_only
    if n_disc == 0:
        p = 1.0
    else:
        from math import comb
        # P(X >= max(a,b) | X ~ Binomial(n_disc, 0.5))
        k = max(a_only, b_only)
        p_one = sum(comb(n_disc, i) for i in range(k, n_disc + 1)) / (2 ** n_disc)
        p = min(1.0, 2 * p_one)
    return delta, len(common), p


def main():
    new_rows = load_jsonl(NEW_ROLLOUTS)
    base_rows = load_jsonl(BASELINE_ROLLOUTS)
    if not new_rows:
        print(f"[path1-pareto] no rows at {NEW_ROLLOUTS}; nothing to plot")
        sys.exit(0)

    new_data = collect(new_rows, set(PATH1_CONDITIONS))
    base_data = collect(base_rows, BASELINES_TO_OVERLAY)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(11, 7))
    plotted_handles = {}

    def _plot(cond, info, **kwargs):
        x = info["bits_param"]
        y = info["sr"]
        lo = info["lo"]
        hi = info["hi"]
        if not np.isfinite(x) or not np.isfinite(y):
            return
        h = ax.scatter([x], [y], **kwargs, edgecolor="black", linewidth=1, zorder=5)
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], fmt="none",
                    ecolor=kwargs.get("color", "gray"), elinewidth=1.5,
                    capsize=4, alpha=0.7, zorder=4)
        label = kwargs.get("label", cond)
        plotted_handles[label] = h
        # short text annotation
        ax.annotate(cond, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8.5)

    # Existing baselines first (so new conds overlay on top)
    for cond in ["FP16", "W4-Floor", "AttnEntropy-W4", "S3-Tern-W4-l12h2"]:
        if cond in base_data:
            style = COND_STYLE[cond].copy()
            _plot(cond, base_data[cond], **style)

    # New: Static (single point)
    if "Static-W2-l13-17" in new_data:
        style = COND_STYLE["Static-W2-l13-17"].copy()
        _plot("Static-W2-l13-17", new_data["Static-W2-l13-17"], **style)

    # New: Random-Bin sweep — connect the three fracs
    rb = []
    for frac_label in ("f25", "f50", "f75"):
        cond = f"Random-Bin-W2-l13-17-{frac_label}"
        if cond in new_data:
            rb.append((new_data[cond]["bits_param"], new_data[cond]["sr"]))
            style = RANDOM_STYLE.copy()
            style["label"] = "Random-Bin-W2 sweep" if frac_label == "f25" else None
            _plot(cond, new_data[cond], **style)
    if len(rb) >= 2:
        xs, ys = zip(*sorted(rb))
        ax.plot(xs, ys, color=RANDOM_STYLE["color"], alpha=0.5, lw=1.5, ls="--", zorder=2)

    # New: AttnEnt-Bin sweep
    ab = []
    for frac_label in ("f25", "f50", "f75"):
        cond = f"AttnEnt-Bin-W2-l13-17-{frac_label}"
        if cond in new_data:
            ab.append((new_data[cond]["bits_param"], new_data[cond]["sr"]))
            style = ATTNENT_STYLE.copy()
            style["label"] = "AttnEnt-Bin-W2 sweep (NEW)" if frac_label == "f25" else None
            _plot(cond, new_data[cond], **style)
    if len(ab) >= 2:
        xs, ys = zip(*sorted(ab))
        ax.plot(xs, ys, color=ATTNENT_STYLE["color"], alpha=0.6, lw=1.8, zorder=3)

    # New: Hybrid (single point)
    if "Hybrid-Static-W2-AttnEnt-FP16-f50" in new_data:
        style = HYBRID_STYLE.copy()
        _plot("Hybrid-Static-W2-AttnEnt-FP16-f50",
              new_data["Hybrid-Static-W2-AttnEnt-FP16-f50"], **style)

    ax.set_xlabel("Average bits per parameter (parameter-weighted)")
    ax.set_ylabel("Rollout success rate (95% bootstrap CI)")
    ax.set_title("Path 1: Static vs Dynamic Late-Layer W2 — SR vs Bits Pareto")
    ax.set_ylim(0, 1.05)
    ax.legend(plotted_handles.values(), plotted_handles.keys(), loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PLOT)
    print(f"[path1-pareto] wrote {OUT_PLOT}")

    # ---- Markdown table + matched-pair McNemar ----
    lines = []
    lines.append("# Path 1 — Static vs Dynamic Late-Layer W2: SR-vs-Bits Pareto\n")
    lines.append("Standard LIBERO (50 Long + 50 Object), all conditions matched on (suite, task_id, seed, episode_idx).\n")

    lines.append("## Bootstrap-CI table\n")
    lines.append("| Condition | n | SR | 95% CI | bits (param-weighted) |")
    lines.append("|---|---:|---:|---|---:|")

    def _row(cond, d):
        if cond not in d:
            return None
        info = d[cond]
        return f"| {cond} | {info['n']} | {info['sr']:.3f} | [{info['lo']:.3f}, {info['hi']:.3f}] | {info['bits_param']:.2f} |"

    for cond in ["FP16", "W4-Floor", "AttnEntropy-W4", "S3-Tern-W4-l12h2"]:
        r = _row(cond, base_data)
        if r:
            lines.append(r)
    for cond in PATH1_CONDITIONS:
        r = _row(cond, new_data)
        if r:
            lines.append(r)

    # Matched-pair tests need rollouts from the same file (since join key is trial id).
    # Combine both files into one corpus for the McNemar pair tests.
    combined = base_rows + new_rows

    lines.append("\n## Matched-pair McNemar (Path 1 hypothesis tags)\n")
    lines.append("| Tag | A | B | n_matched | Δ SR (A − B) | McNemar p |")
    lines.append("|---|---|---|---:|---:|---:|")
    pairs = [
        ("HW12a", "Static-W2-l13-17", "W4-Floor"),
        ("HW12b", "AttnEnt-Bin-W2-l13-17-f50", "Random-Bin-W2-l13-17-f50"),
        ("HW12c", "AttnEnt-Bin-W2-l13-17-f25", "Random-Bin-W2-l13-17-f25"),
        ("HW12d", "AttnEnt-Bin-W2-l13-17-f75", "Random-Bin-W2-l13-17-f75"),
        ("HW12e", "AttnEnt-Bin-W2-l13-17-f50", "Static-W2-l13-17"),
        ("HW12f", "Hybrid-Static-W2-AttnEnt-FP16-f50", "Static-W2-l13-17"),
        ("HW12g", "Hybrid-Static-W2-AttnEnt-FP16-f50", "S3-Tern-W4-l12h2"),
        ("HW12h", "Static-W2-l13-17", "S3-Tern-W4-l12h2"),
    ]
    for tag, A, B in pairs:
        delta, n, p = matched_pair_delta(combined, A, B)
        if n == 0:
            continue
        lines.append(f"| {tag} | {A} | {B} | {n} | {delta:+.3f} | {p:.3g} |")

    lines.append("\n_Plot: `expB_path1_pareto.png` — Pareto frontier of SR vs param-weighted bits._\n")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"[path1-pareto] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
