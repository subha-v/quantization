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
                   tie_seed: int = 0,
                   selection_mode: str = "margin",
                   random_seed: int = 0) -> tuple[list[dict], dict]:
    """Return (stitched_rows, meta_dict).

    selection_mode:
      "margin" -- bottom-third by confidence margin (max - second_max) of the
                  first-pass option_logprobs. The original cascade.
      "random" -- bottom-third by deterministic hash with random_seed. Tests
                  whether margin-based selection beats arbitrary selection at
                  the same rerun rate.
      "oracle" -- items where first_pass.is_correct=False AND
                  second_pass.is_correct=True. Uses ground-truth labels;
                  upper-bound on cascade gain. Realized rerun rate may differ
                  from the target.
    """
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

    # Target rerun fraction (used by margin/random modes; informational for oracle).
    span = max(1, second_pass_frames - first_pass_frames)
    rerun_fraction = (target_avg_frames - first_pass_frames) / span
    rerun_fraction = max(0.0, min(1.0, rerun_fraction))
    n_rerun_target = int(round(len(common_items) * rerun_fraction))

    # Compute margins for the first pass (used by margin mode + always recorded
    # in the per-row first_pass_margin field).
    margins: dict[str, float] = {
        iid: _confidence_margin(g1_rows[iid]["option_logprobs"])
        for iid in common_items
    }

    if selection_mode == "margin":
        # Sort ascending by (margin, hashbreak); pick bottom-third.
        ranked = sorted(
            ((iid, margins[iid], _hash_break(iid, seed=tie_seed)) for iid in common_items),
            key=lambda x: (x[1], x[2]),
        )
        low_margin_ids = {iid for iid, _, _ in ranked[:n_rerun_target]}
        n_rerun = n_rerun_target
        tau = ranked[n_rerun - 1][1] if n_rerun > 0 else float("-inf")
    elif selection_mode == "random":
        # Deterministic random selection by hash seeded with random_seed.
        ranked = sorted(common_items, key=lambda iid: _hash_break(iid, seed=random_seed))
        low_margin_ids = set(ranked[:n_rerun_target])
        n_rerun = n_rerun_target
        tau = float("nan")  # not meaningful for random selection
    elif selection_mode == "oracle":
        # Pick items where first-pass is wrong but second-pass is right; rank
        # by margin (lowest first) and cap at n_rerun_target if there are
        # more eligible items than the budget.
        eligible = [
            iid for iid in common_items
            if not g1_rows[iid].get("is_correct", False)
            and g4_rows[iid].get("is_correct", False)
        ]
        if len(eligible) > n_rerun_target:
            eligible.sort(key=lambda iid: (margins[iid], _hash_break(iid, seed=tie_seed)))
            eligible = eligible[:n_rerun_target]
        low_margin_ids = set(eligible)
        n_rerun = len(low_margin_ids)
        tau = float("nan")  # selection was label-based, not margin-thresholded
    else:
        raise ValueError(f"unknown selection_mode={selection_mode!r}")

    realized_avg = (
        (n_rerun * second_pass_frames + (len(common_items) - n_rerun) * first_pass_frames)
        / max(1, len(common_items))
    )

    realized_avg = (
        (n_rerun * second_pass_frames + (len(common_items) - n_rerun) * first_pass_frames)
        / max(1, len(common_items))
    )

    # Build stitched rows: substitute second-pass for selected items, keep
    # first-pass for the rest.
    stitched: list[dict] = []
    bucket_low_margin = Counter()
    for iid in common_items:
        g1_row = g1_rows[iid]
        g4_row = g4_rows[iid]
        margin = margins[iid]
        is_low = iid in low_margin_ids
        src = g4_row if is_low else g1_row
        new = dict(src)  # shallow copy
        new["experiment"] = "G"
        new["condition"] = stitched_cond_name
        new["condition_class"] = "cascade"
        new["cascade_selection_mode"] = selection_mode
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
        if selection_mode == "random":
            new["cascade_random_seed"] = random_seed
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
        "selection_mode": selection_mode,
        "rerun_fraction_target": rerun_fraction,
        "n_items": len(common_items),
        "n_rerun_target": n_rerun_target,
        "n_rerun_realized": n_rerun,
        "cascade_threshold_tau": float(tau),
        "realized_avg_frames": float(realized_avg),
        "low_margin_bucket_distribution": dict(bucket_low_margin),
        "tie_seed": tie_seed,
    }
    if selection_mode == "random":
        meta["random_seed"] = random_seed
    if selection_mode == "oracle":
        # Diagnostic: how many items in total satisfy the oracle condition?
        n_oracle_eligible = sum(
            1 for iid in common_items
            if not g1_rows[iid].get("is_correct", False)
            and g4_rows[iid].get("is_correct", False)
        )
        meta["n_oracle_eligible"] = n_oracle_eligible
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
    ap.add_argument("--first_pass_frames", type=int, default=64)
    ap.add_argument("--second_pass_frames", type=int, default=256)
    ap.add_argument("--stitched_name", default="G7_F4_CascadeAvg128")
    ap.add_argument("--tie_seed", type=int, default=0)
    ap.add_argument("--selection_mode", choices=["margin", "random", "oracle"],
                    default="margin",
                    help="margin = bottom-third by confidence margin (default); "
                         "random = bottom-third by deterministic hash with --random_seed; "
                         "oracle = items where first-pass wrong AND second-pass right "
                         "(uses ground-truth labels; upper-bound).")
    ap.add_argument("--random_seed", type=int, default=0,
                    help="Seed for random selection mode.")
    args = ap.parse_args()

    stitched, meta = stitch_cascade(
        args.in_jsonl,
        first_pass_cond=args.first_pass,
        second_pass_cond=args.second_pass,
        target_avg_frames=args.target_avg_frames,
        first_pass_frames=args.first_pass_frames,
        second_pass_frames=args.second_pass_frames,
        stitched_cond_name=args.stitched_name,
        tie_seed=args.tie_seed,
        selection_mode=args.selection_mode,
        random_seed=args.random_seed,
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
