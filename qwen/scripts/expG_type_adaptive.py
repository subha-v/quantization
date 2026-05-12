"""Experiment G8 — question-type adaptive frame budget post-process.

After expG_frame_scaling.py has produced G1 (64f), G3 (128f), and G4 (256f)
rows -- all using F4 K quantizer -- this script stitches a virtual
G8_F4_TypeAdaptive condition by:

  1. Loading G1 / G3 / G4 rows.
  2. For each item, classifying the question text into one of
     {count, temporal, ocr, detail, action, other} via
     question_type_classifier.classify_question_type.
  3. Looking up the assigned frame budget from BUDGET_MAP, then substituting
     the matching precomputed row (G1 if 64f, G3 if 128f, G4 if 256f).
  4. Emitting a stitched row with `condition="G8_F4_TypeAdaptive"`,
     `condition_class="type_adaptive"`, `question_type=<label>`,
     `assigned_frames=<int>`.

Sidecar `expG_qtype_meta.json` records the budget map, label distribution on
the eval set, and the resulting weighted-average frames.

The classifier itself is tunable on the cal-100 split (disjoint from any
stage-N eval set per make_split semantics) -- see expG_smoke.py check 6 for
the budget-map sanity assertion.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data_longvideobench import filter_items, load_all_items, load_split
from question_type_classifier import (
    BUDGET_MAP,
    classify_question_type,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"


# Map from frame budget back to the canonical G fixed-frame condition name.
FRAMES_TO_GCOND = {
    64: "G1_F4_64f",
    128: "G3_F4_128f",
    256: "G4_F4_256f",
}


def stitch_type_adaptive(in_jsonl: Path,
                         items_by_id: dict[str, object],
                         stitched_cond_name: str = "G8_F4_TypeAdaptive",
                         budget_map: dict[str, int] = BUDGET_MAP,
                         frames_to_gcond: dict[int, str] = FRAMES_TO_GCOND
                         ) -> tuple[list[dict], dict]:
    """Return (stitched_rows, meta_dict)."""
    if not in_jsonl.exists():
        raise FileNotFoundError(f"input JSONL does not exist: {in_jsonl}")
    rows = [json.loads(l) for l in in_jsonl.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("error") is None]

    # Index rows by (item_id, condition).
    by_ic: dict[tuple[str, str], dict] = {}
    for r in rows:
        cond = r.get("condition")
        iid = r.get("item_id")
        if cond is None or iid is None:
            continue
        if cond in frames_to_gcond.values():
            by_ic[(iid, cond)] = r

    needed_conds = set(frames_to_gcond.values())
    missing_per_cond: dict[str, int] = {c: 0 for c in needed_conds}

    # Pull common item_ids that have all three conditions present.
    cands = sorted({iid for (iid, _) in by_ic.keys()})
    items_with_all = []
    for iid in cands:
        if all((iid, c) in by_ic for c in needed_conds):
            items_with_all.append(iid)
        else:
            for c in needed_conds:
                if (iid, c) not in by_ic:
                    missing_per_cond[c] += 1

    if not items_with_all:
        raise RuntimeError(
            f"no items have all three of {sorted(needed_conds)} in {in_jsonl}. "
            f"Run expG_frame_scaling.py for G1, G3, and G4 first."
        )

    label_counts: Counter[str] = Counter()
    frame_counts: Counter[int] = Counter()
    stitched: list[dict] = []
    for iid in items_with_all:
        item = items_by_id.get(iid)
        if item is None:
            # Skip items not in the loaded set (rare; logged in meta).
            continue
        label = classify_question_type(item.question)
        assigned_frames = budget_map[label]
        src_cond = frames_to_gcond[assigned_frames]
        src = by_ic[(iid, src_cond)]
        new = dict(src)
        new["experiment"] = "G"
        new["condition"] = stitched_cond_name
        new["condition_class"] = "type_adaptive"
        new["question_type"] = label
        new["assigned_frames"] = int(assigned_frames)
        new["frames"] = int(src.get("frames", src.get("n_frames", assigned_frames)))
        new["type_adaptive_source_cond"] = src_cond
        stitched.append(new)
        label_counts[label] += 1
        frame_counts[int(assigned_frames)] += 1

    n = len(stitched)
    weighted_avg_frames = (
        sum(f * c for f, c in frame_counts.items()) / max(1, n)
    )
    meta = {
        "stitched_cond_name": stitched_cond_name,
        "budget_map": dict(budget_map),
        "frames_to_gcond": {str(k): v for k, v in frames_to_gcond.items()},
        "n_items": n,
        "label_distribution": dict(label_counts),
        "frame_distribution": {str(k): v for k, v in frame_counts.items()},
        "weighted_avg_frames": float(weighted_avg_frames),
        "missing_per_cond": dict(missing_per_cond),
    }
    return stitched, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=Path,
                    default=RESULTS_DIR / "expG_frame_stage1.jsonl")
    ap.add_argument("--out_jsonl", type=Path,
                    default=RESULTS_DIR / "expG_frame_stage1_G8.jsonl")
    ap.add_argument("--meta", type=Path,
                    default=RESULTS_DIR / "expG_qtype_meta.json")
    ap.add_argument("--split_file", type=Path,
                    default=CALIBRATION_DIR / "split_seed0_n64.json",
                    help="Stage-1 split. Items not in this split are dropped.")
    ap.add_argument("--stitched_name", default="G8_F4_TypeAdaptive")
    ap.add_argument("--frames_to_gcond_json", type=str, default=None,
                    help="JSON dict mapping frame budget (str) to source "
                         "condition name. Default: {64:G1_F4_64f, 128:G3_F4_128f, "
                         "256:G4_F4_256f} -- override for an F9-backbone variant "
                         "with e.g. '{\"128\":\"G5_F9_128f\",\"256\":\"G6_F9_256f\"}'.")
    ap.add_argument("--budget_map_json", type=str, default=None,
                    help="JSON dict mapping qtype label to frame count. Default: "
                         "BUDGET_MAP from question_type_classifier. Override for "
                         "F9-backbone variants where 64f isn't available, e.g. "
                         "'{\"count\":256,\"ocr\":256,\"detail\":256,\"temporal\":128,"
                         "\"action\":128,\"other\":128}'.")
    args = ap.parse_args()

    items_all = load_all_items()
    if args.split_file.exists():
        split = load_split(args.split_file)
        eval_ids = set(split["eval"])
        items_all = [it for it in items_all if it.id in eval_ids]
    items_by_id = {it.id: it for it in items_all}

    frames_to_gcond = FRAMES_TO_GCOND
    if args.frames_to_gcond_json:
        raw = json.loads(args.frames_to_gcond_json)
        frames_to_gcond = {int(k): v for k, v in raw.items()}
    budget_map = BUDGET_MAP
    if args.budget_map_json:
        budget_map = json.loads(args.budget_map_json)

    stitched, meta = stitch_type_adaptive(
        args.in_jsonl,
        items_by_id=items_by_id,
        stitched_cond_name=args.stitched_name,
        budget_map=budget_map,
        frames_to_gcond=frames_to_gcond,
    )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in stitched:
            f.write(json.dumps(r) + "\n")
    args.meta.write_text(json.dumps(meta, indent=2))

    n = len(stitched)
    n_correct = sum(1 for r in stitched if r.get("is_correct"))
    print(f"[expG_qtype] wrote {n} stitched rows -> {args.out_jsonl}")
    print(f"[expG_qtype] weighted_avg_frames={meta['weighted_avg_frames']:.1f}")
    print(f"[expG_qtype] label_distribution={meta['label_distribution']}")
    print(f"[expG_qtype] acc={n_correct}/{n} = {n_correct/max(1,n):.3f}")
    print(f"[expG_qtype] meta -> {args.meta}")


if __name__ == "__main__":
    main()
