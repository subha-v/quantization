"""Build a cal+eval split for Exp L (seed=1 recalibration sanity check).

The existing seed=1 eval set (split_seed1_n200.json) has 200 stratified items
(50/bucket) and was generated with cal_fraction=0.0 — so no cal items.

For Exp L we need:
  eval = the SAME 200 items as Exp K seed=1 (preserves paired McNemar)
  cal  = 100 fresh items, stratified, DISJOINT from eval

This script samples 25 items/bucket from items NOT in the existing eval,
using a different random seed (1009 — far from any other seed in use) so
the cal selection is reproducible. Writes:
  qwen/calibration/split_seed1_cal100_for_existing_eval.json

Usage:
  python make_seed1_cal_split.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from data_longvideobench import load_all_items, load_split


CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"
EXISTING_EVAL_FILE = CALIBRATION_DIR / "split_seed1_n200.json"
OUT_FILE = CALIBRATION_DIR / "split_seed1_cal100_for_existing_eval.json"
CAL_SEED = 1009
N_CAL_PER_BUCKET = 25  # 4 buckets × 25 = 100 cal items


def main():
    if not EXISTING_EVAL_FILE.exists():
        raise SystemExit(f"existing seed=1 eval file not found: {EXISTING_EVAL_FILE}")
    existing = load_split(EXISTING_EVAL_FILE)
    eval_ids = set(existing["eval"])
    print(f"[mk_seed1_cal] existing seed=1 eval items: {len(eval_ids)}")

    items = load_all_items()
    print(f"[mk_seed1_cal] LongVideoBench items loaded: {len(items)}")

    # Stratify by bucket, drop items in existing eval.
    by_bucket: dict[str, list] = {}
    for it in items:
        if it.id in eval_ids:
            continue
        by_bucket.setdefault(it.duration_bucket, []).append(it)
    for b in ("short", "mid", "long", "very_long"):
        n = len(by_bucket.get(b, []))
        print(f"[mk_seed1_cal]   bucket {b!r}: {n} items available (excluding eval)")

    rng = random.Random(CAL_SEED)
    cal_ids: list[str] = []
    for bucket in ("short", "mid", "long", "very_long"):
        pool = by_bucket.get(bucket, [])
        if len(pool) < N_CAL_PER_BUCKET:
            raise SystemExit(
                f"[mk_seed1_cal] bucket {bucket!r}: only {len(pool)} items "
                f"available; need {N_CAL_PER_BUCKET}"
            )
        rng.shuffle(pool)
        chosen = pool[:N_CAL_PER_BUCKET]
        cal_ids.extend(it.id for it in chosen)
        print(f"[mk_seed1_cal]   bucket {bucket!r}: picked {len(chosen)} cal items")

    # Final disjointness check.
    overlap = set(cal_ids) & eval_ids
    if overlap:
        raise SystemExit(f"[mk_seed1_cal] FATAL: {len(overlap)} cal/eval overlap")

    split = {"cal": cal_ids, "eval": sorted(eval_ids)}
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(split, indent=2))
    print(f"[mk_seed1_cal] wrote {OUT_FILE} (cal={len(cal_ids)}, eval={len(eval_ids)})")


if __name__ == "__main__":
    main()
