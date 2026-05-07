"""
Pareto plot: avg KV bits (x) vs accuracy (y) across Experiment A + B conditions.

Pulls per-condition mean accuracy and avg KV bits from the JSONL output files,
plots one point per condition with bootstrap-CI error bars, color-codes by
method family (uniform / random / attention-mass / MEDA / AttnEntropy V1-V3 /
oracle / BF16-ceiling). Ports the layout from scripts/expB_path1_pareto.py.

Usage:
    python expB_pareto_plot.py \
        --jsonl results/expA_rollouts_*.jsonl results/expB_rollouts_*.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"


FAMILY_COLORS = {
    "BF16": "black",
    "Weight-only": "#888888",
    "Uniform": "#1f77b4",
    "Random": "#9467bd",
    "AttnMass": "#ff7f0e",
    "MEDA": "#8c564b",
    "Oracle": "#2ca02c",
    "AttnEnt-V1": "#d62728",
    "AttnEnt-V2": "#e377c2",
    "AttnEnt-V3": "#bcbd22",
}


def family_for(cond: str) -> str:
    if "BF16_BF16" in cond or cond.startswith("A1"):
        return "BF16"
    if cond.startswith(("A2", "A3")):
        return "Weight-only"
    if "Uniform" in cond or cond.startswith(("A4", "A5", "A6", "A7", "A8")):
        return "Uniform"
    if "Random" in cond:
        return "Random"
    if "AttnMass" in cond:
        return "AttnMass"
    if "MEDA" in cond:
        return "MEDA"
    if "Oracle" in cond:
        return "Oracle"
    if "V1" in cond:
        return "AttnEnt-V1"
    if "V2" in cond:
        return "AttnEnt-V2"
    if "V3" in cond:
        return "AttnEnt-V3"
    return "Other"


def _bootstrap_ci(arr, n_boot=2000, seed=0):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [arr[rng.integers(0, arr.size, arr.size)].mean() for _ in range(n_boot)]
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--out", type=Path, default=PLOTS_DIR / "expB_pareto_avgbits_vs_acc.png")
    args = ap.parse_args()

    rows = []
    for p in args.jsonl:
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("n_frames") == args.frames]

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, rs in sorted(by_cond.items()):
        accs = [r["is_correct"] for r in rs]
        mean, lo, hi = _bootstrap_ci(accs)
        bits = next((r["avg_kv_bits"] for r in rs if r.get("avg_kv_bits") is not None), None)
        if bits is None:
            bits = 16.0
        fam = family_for(cond)
        color = FAMILY_COLORS.get(fam, "#444444")
        ax.errorbar(bits, mean, yerr=[[mean - lo], [hi - mean]],
                    fmt="o", color=color, capsize=3, label=cond)
        ax.annotate(cond.split("_frames")[0], (bits, mean),
                    textcoords="offset points", xytext=(5, 3), fontsize=7, color=color)

    ax.set_xlabel("Average KV bits (per K/V tensor element)")
    ax.set_ylabel(f"Accuracy (frames={args.frames})")
    ax.set_title("LongVideoBench-val: KV-quant Pareto curve")
    ax.set_xlim(left=1.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"[pareto] -> {args.out}")


if __name__ == "__main__":
    main()
