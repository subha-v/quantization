"""Experiment I — Duration-aware hybrid post-process (I15 / I16).

After expI_temporal_kivi.py has produced I2 (F9 128f) and I3 (H6 TempWin2
128f) rows, this script stitches two virtual conditions:

  I15 Duration-Hybrid: per-item routing rule based on duration_bucket.
    if duration_bucket in {"mid"}: use I2 (F9 outlier-16) row
    else:                          use I3 (H6 TempWin2) row

    Motivation: H6's per-bucket profile at n=200 shows mid-bucket as the
    only place F9 strictly beats H6 (mid: 0.820 vs 0.740). Routing F9 only
    at mid keeps H6's wins at short/long/very_long while picking up F9's
    mid advantage. Avg KV bits ≈ weighted mean of 4.00 (H6) and 4.75 (F9)
    by bucket frequency.

  I16 Random-Hybrid: matched-rerun-rate control.
    Pick the same number of items to assign to F9, but selected by
    deterministic hash (random) rather than bucket. Tests whether the
    bucket-routing rule beats arbitrary same-rate routing.

Provenance fields per stitched row:
  experiment            "I"
  condition             "I15_F9MidElseTempWin" | "I16_F9RandomMatched"
  condition_class       "duration_hybrid"
  hybrid_selection_mode "duration" | "random"
  hybrid_used_f9        bool
  f9_source_cond        "I2_F9_128f"
  tempwin_source_cond   "I3_TempWin2_128f"
  duration_bucket       carried from source row

Sidecar `expI_duration_hybrid_meta.json` records swap counts per bucket and
the seed used for random selection.

Usage (Stage 1):
  python expI_duration_hybrid.py \\
      --in_jsonl qwen/results/expI_tempkivi_stage1_seed1.jsonl \\
      --selection_mode duration
  python expI_duration_hybrid.py \\
      --in_jsonl qwen/results/expI_tempkivi_stage1_seed1.jsonl \\
      --selection_mode random
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _hash_break(item_id: str, seed: int = 1) -> int:
    """Deterministic hash for random-selection mode."""
    h = hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()
    return int(h[:8], 16)


def stitch_duration_hybrid(in_jsonl: Path,
                           f9_cond: str = "I2_F9_128f",
                           tempwin_cond: str = "I3_TempWin2_128f",
                           f9_buckets: tuple[str, ...] = ("mid",),
                           stitched_cond_name: str = "I15_F9MidElseTempWin",
                           selection_mode: str = "duration",
                           random_seed: int = 1) -> tuple[list[dict], dict]:
    """Stitch a virtual hybrid condition by joining two source conditions
    on item_id.

    selection_mode:
      "duration" (I15) -- assign F9 to items in f9_buckets, H6 to others.
      "random"   (I16) -- assign F9 to the same number of items selected by
                          deterministic hash with random_seed (matched-rate
                          control for I15).
    """
    if not in_jsonl.exists():
        raise FileNotFoundError(f"input JSONL does not exist: {in_jsonl}")
    rows = [json.loads(l) for l in in_jsonl.read_text().splitlines() if l.strip()]
    f9_rows = {r["item_id"]: r for r in rows
               if r.get("condition") == f9_cond
               and r.get("error") is None
               and "pred_choice" in r}
    tw_rows = {r["item_id"]: r for r in rows
               if r.get("condition") == tempwin_cond
               and r.get("error") is None
               and "pred_choice" in r}

    common_items = sorted(set(f9_rows) & set(tw_rows))
    if not common_items:
        raise RuntimeError(
            f"no items have both {f9_cond} and {tempwin_cond} rows in "
            f"{in_jsonl}. Run expI_temporal_kivi.py for both conditions first."
        )

    # First pass: figure out which items get F9 (duration mode).
    duration_f9_ids: set[str] = {
        iid for iid in common_items
        if tw_rows[iid].get("duration_bucket") in f9_buckets
    }

    if selection_mode == "duration":
        f9_assigned = duration_f9_ids
    elif selection_mode == "random":
        n_target = len(duration_f9_ids)
        ranked = sorted(common_items, key=lambda iid: _hash_break(iid, seed=random_seed))
        f9_assigned = set(ranked[:n_target])
    else:
        raise ValueError(f"unknown selection_mode={selection_mode!r}")

    # Build stitched rows.
    stitched: list[dict] = []
    bucket_used_f9 = Counter()
    bucket_total = Counter()
    for iid in common_items:
        used_f9 = iid in f9_assigned
        src = f9_rows[iid] if used_f9 else tw_rows[iid]
        new = dict(src)  # shallow copy
        new["experiment"] = "I"
        new["condition"] = stitched_cond_name
        new["condition_class"] = "duration_hybrid"
        new["hybrid_selection_mode"] = selection_mode
        new["hybrid_used_f9"] = bool(used_f9)
        new["f9_source_cond"] = f9_cond
        new["tempwin_source_cond"] = tempwin_cond
        new["hybrid_random_seed"] = (random_seed if selection_mode == "random" else None)
        new["hybrid_f9_buckets"] = list(f9_buckets) if selection_mode == "duration" else None
        bucket = src.get("duration_bucket", "unknown")
        bucket_total[bucket] += 1
        if used_f9:
            bucket_used_f9[bucket] += 1
        stitched.append(new)

    n_correct = sum(1 for r in stitched if r.get("is_correct"))
    realized_f9_rate = len(f9_assigned) / max(1, len(common_items))

    meta = {
        "stitched_cond_name": stitched_cond_name,
        "f9_source_cond": f9_cond,
        "tempwin_source_cond": tempwin_cond,
        "selection_mode": selection_mode,
        "f9_buckets": list(f9_buckets),
        "n_items": len(common_items),
        "n_used_f9": len(f9_assigned),
        "realized_f9_rate": realized_f9_rate,
        "bucket_used_f9": dict(bucket_used_f9),
        "bucket_total": dict(bucket_total),
        "acc": n_correct / max(1, len(stitched)),
    }
    if selection_mode == "random":
        meta["random_seed"] = random_seed
    return stitched, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=Path, required=True,
                    help="Source JSONL containing I2 (F9_128f) + I3 (TempWin2_128f) rows.")
    ap.add_argument("--out_jsonl", type=Path, default=None,
                    help="Default: same dir as in_jsonl, suffix _I15.jsonl or _I16.jsonl.")
    ap.add_argument("--meta", type=Path, default=None,
                    help="Default: results/expI_duration_hybrid_{mode}_meta.json")
    ap.add_argument("--f9_cond", default="I2_F9_128f")
    ap.add_argument("--tempwin_cond", default="I3_TempWin2_128f")
    ap.add_argument("--f9_buckets", nargs="+", default=["mid"])
    ap.add_argument("--selection_mode", choices=["duration", "random"], default="duration")
    ap.add_argument("--random_seed", type=int, default=1)
    ap.add_argument("--stitched_name", default=None,
                    help="Default: I15_F9MidElseTempWin (duration) or "
                         "I16_F9RandomMatched (random).")
    args = ap.parse_args()

    if args.stitched_name is None:
        args.stitched_name = ("I15_F9MidElseTempWin" if args.selection_mode == "duration"
                              else "I16_F9RandomMatched")
    if args.out_jsonl is None:
        suffix = "I15" if args.selection_mode == "duration" else "I16"
        args.out_jsonl = args.in_jsonl.with_name(args.in_jsonl.stem + f"_{suffix}.jsonl")
    if args.meta is None:
        args.meta = RESULTS_DIR / f"expI_duration_hybrid_{args.selection_mode}_meta.json"

    stitched, meta = stitch_duration_hybrid(
        args.in_jsonl,
        f9_cond=args.f9_cond,
        tempwin_cond=args.tempwin_cond,
        f9_buckets=tuple(args.f9_buckets),
        stitched_cond_name=args.stitched_name,
        selection_mode=args.selection_mode,
        random_seed=args.random_seed,
    )
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in stitched:
            f.write(json.dumps(r) + "\n")
    args.meta.write_text(json.dumps(meta, indent=2))
    print(f"[expI_hybrid] wrote {len(stitched)} stitched rows -> {args.out_jsonl}")
    print(f"[expI_hybrid] selection={args.selection_mode} n_used_f9={meta['n_used_f9']}/"
          f"{meta['n_items']} acc={meta['acc']:.3f} bucket_f9={meta['bucket_used_f9']}")
    print(f"[expI_hybrid] meta -> {args.meta}")


if __name__ == "__main__":
    main()
