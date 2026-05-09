"""Experiment G7 — confidence-cascade post-process.

After expG_frame_scaling.py has produced G1 (64f F4) and G4 (256f F4) rows,
this script stitches a virtual G7_F4_CascadeAvg128 condition by:

  1. Loading G1 rows (64f F4 forward).
  2. Computing `cascade_margin = max(option_logprobs) - second_max(...)` per
     G1 row. NB: this is the "confidence" margin computed without knowing the
     correct answer (max - second_max), distinct from `_answer_margin` which
     uses the correct index.
  3. Picking the bottom-third by cascade_margin to "rerun at 256f". The
     target average frames is 128, so rerun fraction = (128-64)/(256-64) =
     1/3.
  4. For low-margin items: substitute the corresponding G4 row (precomputed,
     no new forward).
  5. For high-margin items: keep the G1 row.
  6. Emit stitched rows with `condition="G7_F4_CascadeAvg128"`,
     `condition_class="cascade"`, `cascade_target_avg=128`,
     `cascade_threshold_tau=<float>`, `first_pass_margin=<float>`,
     `second_pass_used=<bool>`, `assigned_frames in {64, 256}`.

Sidecar `expG_cascade_meta.json` records the threshold, realized average
frames, bucket distribution of low-margin items, and the seed for tiebreaks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Optional


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _confidence_margin(option_logprobs: list[float]) -> float:
    """max - second_max over option_logprobs. Ignores correct_choice on
    purpose -- this is the model's self-reported confidence at inference time,
    which is what an adaptive controller would have access to."""
    if not option_logprobs or len(option_logprobs) < 2:
        return float("-inf")
    sorted_lp = sorted(option_logprobs, reverse=True)
    return float(sorted_lp[0] - sorted_lp[1])


def _hash_break(item_id: str, seed: int = 0) -> int:
    """Deterministic tie-break for ranking when margins collide."""
    h = hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()
    return int(h[:8], 16)


def stitch_cascade(in_jsonl: Path,
                   first_pass_cond: str = "G1_F4_64f",
                   second_pass_cond: str = "G4_F4_256f",
                   target_avg_frames: int = 128,
                   first_pass_frames: int = 64,
                   second_pass_frames: int = 256,
                   stitched_cond_name: str = "G7_F4_CascadeAvg128",
                   tie_seed: int = 0) -> tuple[list[dict], dict]:
    """Return (stitched_rows, meta_dict)."""
    if not in_jsonl.exists():
        raise FileNotFoundError(f"input JSONL does not exist: {in_jsonl}")
    rows = [json.loads(l) for l in in_jsonl.read_text().splitlines() if l.strip()]
    g1_rows = {r["item_id"]: r for r in rows
               if r.get("condition") == first_pass_cond
               and r.get("error") is None
               and "option_logprobs" in r}
    g4_rows = {r["item_id"]: r for r in rows
               if r.get("condition") == second_pass_cond
               and r.get("error") is None
               and "option_logprobs" in r}

    common_items = sorted(set(g1_rows) & set(g4_rows))
    if not common_items:
        raise RuntimeError(
            f"no items have both {first_pass_cond} and {second_pass_cond} rows in "
            f"{in_jsonl}. Run expG_frame_scaling.py for both conditions first."
        )

    # Compute margins for G1 (the first-pass forward).
    by_item_margin: list[tuple[str, float, int]] = []  # (id, margin, hashbreak)
    for iid in common_items:
        m = _confidence_margin(g1_rows[iid]["option_logprobs"])
        by_item_margin.append((iid, m, _hash_break(iid, seed=tie_seed)))

    # Sort ascending by (margin, hashbreak): low-margin items rerun first.
    by_item_margin.sort(key=lambda x: (x[1], x[2]))

    # Pick the bottom-third (rounded). target_avg = 128 -> rerun_fraction = 1/3.
    span = max(1, second_pass_frames - first_pass_frames)
    rerun_fraction = (target_avg_frames - first_pass_frames) / span
    rerun_fraction = max(0.0, min(1.0, rerun_fraction))
    n_rerun = int(round(len(common_items) * rerun_fraction))
    low_margin_ids = {iid for iid, _, _ in by_item_margin[:n_rerun]}
    tau = (by_item_margin[n_rerun - 1][1] if n_rerun > 0
           else float("-inf"))

    realized_avg = (
        (n_rerun * second_pass_frames + (len(common_items) - n_rerun) * first_pass_frames)
        / max(1, len(common_items))
    )

    # Build stitched rows: substitute G4 for low-margin, keep G1 for the rest.
    stitched: list[dict] = []
    bucket_low_margin = Counter()
    for iid in common_items:
        g1_row = g1_rows[iid]
        g4_row = g4_rows[iid]
        margin = _confidence_margin(g1_row["option_logprobs"])
        is_low = iid in low_margin_ids
        src = g4_row if is_low else g1_row
        new = dict(src)  # shallow copy
        new["experiment"] = "G"
        new["condition"] = stitched_cond_name
        new["condition_class"] = "cascade"
        new["cascade_target_avg"] = target_avg_frames
        new["cascade_threshold_tau"] = float(tau)
        new["first_pass_margin"] = float(margin)
        new["second_pass_used"] = bool(is_low)
        new["assigned_frames"] = (second_pass_frames if is_low else first_pass_frames)
        # `frames` reflects the actual forward used (matches `src`).
        new["frames"] = int(src.get("frames", src.get("n_frames", -1)))
        # Carry source provenance for analyzer auditing.
        new["cascade_first_pass_cond"] = first_pass_cond
        new["cascade_second_pass_cond"] = second_pass_cond
        if is_low:
            bucket_low_margin[src.get("duration_bucket", "unknown")] += 1
        stitched.append(new)

    meta = {
        "stitched_cond_name": stitched_cond_name,
        "first_pass_cond": first_pass_cond,
        "second_pass_cond": second_pass_cond,
        "target_avg_frames": target_avg_frames,
        "first_pass_frames": first_pass_frames,
        "second_pass_frames": second_pass_frames,
        "rerun_fraction": rerun_fraction,
        "n_items": len(common_items),
        "n_rerun": n_rerun,
        "cascade_threshold_tau": float(tau),
        "realized_avg_frames": float(realized_avg),
        "low_margin_bucket_distribution": dict(bucket_low_margin),
        "tie_seed": tie_seed,
    }
    return stitched, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=Path,
                    default=RESULTS_DIR / "expG_frame_stage1.jsonl",
                    help="Source JSONL containing G1 + G4 rows.")
    ap.add_argument("--out_jsonl", type=Path,
                    default=RESULTS_DIR / "expG_frame_stage1_G7.jsonl")
    ap.add_argument("--meta", type=Path,
                    default=RESULTS_DIR / "expG_cascade_meta.json")
    ap.add_argument("--first_pass", default="G1_F4_64f")
    ap.add_argument("--second_pass", default="G4_F4_256f")
    ap.add_argument("--target_avg_frames", type=int, default=128)
    ap.add_argument("--stitched_name", default="G7_F4_CascadeAvg128")
    ap.add_argument("--tie_seed", type=int, default=0)
    args = ap.parse_args()

    stitched, meta = stitch_cascade(
        args.in_jsonl,
        first_pass_cond=args.first_pass,
        second_pass_cond=args.second_pass,
        target_avg_frames=args.target_avg_frames,
        stitched_cond_name=args.stitched_name,
        tie_seed=args.tie_seed,
    )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in stitched:
            f.write(json.dumps(r) + "\n")
    args.meta.write_text(json.dumps(meta, indent=2))
    n_correct = sum(1 for r in stitched if r.get("is_correct"))
    print(f"[expG_cascade] wrote {len(stitched)} stitched rows -> {args.out_jsonl}")
    print(f"[expG_cascade] tau={meta['cascade_threshold_tau']:.4f} "
          f"realized_avg_frames={meta['realized_avg_frames']:.1f} "
          f"acc={n_correct}/{len(stitched)} = {n_correct/max(1,len(stitched)):.3f}")
    print(f"[expG_cascade] meta -> {args.meta}")


if __name__ == "__main__":
    main()
