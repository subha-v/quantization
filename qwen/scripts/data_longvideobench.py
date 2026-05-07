"""
LongVideoBench-val loader + stratified calibration/evaluation split.

Assumes the dataset is downloaded under $LONGVIDEOBENCH_ROOT (default
/data/subha2/longvideobench/) with the standard HF layout:
  $ROOT/lvb_val.json           # one record per example
  $ROOT/videos/<video_id>.mp4  # raw videos
The records have fields: id, video_id, video_path, question, candidates,
correct_choice (int 0..3), duration_group, duration (seconds), subtitle_path
(optional). Unknown fields are tolerated.

If $LONGVIDEOBENCH_ROOT is unset or missing, the script falls back to
HuggingFace datasets `longvideobench/LongVideoBench` (validation split).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_ROOT = Path(os.environ.get("LONGVIDEOBENCH_ROOT", "/data/subha2/longvideobench"))
DEFAULT_SPLIT_FILE = Path(__file__).resolve().parents[1] / "calibration" / "split_seed0.json"


@dataclass
class LVBItem:
    id: str
    video_path: str
    duration_seconds: float
    duration_bucket: str
    question: str
    candidates: list[str]
    correct_choice: int
    subtitle_path: Optional[str] = None
    raw: Optional[dict] = None


# ---------------- duration bucketing ----------------

# Duration buckets aligned with LongVideoBench's official duration_group field
# (verified against lvb_val.json: groups 15/60/600/3600 with 189/172/412/564 items).
DURATION_BUCKETS = [
    ("short", 0, 15),
    ("mid", 15, 60),
    ("long", 60, 600),
    ("very_long", 600, 3600),
]
STRATA_TARGETS = {"short": 50, "mid": 50, "long": 100, "very_long": 100}  # 300 total


def bucket_for(seconds: float) -> Optional[str]:
    for name, lo, hi in DURATION_BUCKETS:
        if lo <= seconds < hi:
            return name
    return None


# ---------------- raw dataset reading ----------------

def _from_local_json(root: Path) -> list[dict]:
    candidate = root / "lvb_val.json"
    if not candidate.exists():
        return []
    with open(candidate) as f:
        records = json.load(f)
    return records


def _from_hf_datasets() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Neither LONGVIDEOBENCH_ROOT/lvb_val.json nor `datasets` library is available. "
            "Install via: uv pip install datasets, or download LongVideoBench-val manually."
        ) from e
    ds = load_dataset("longvideobench/LongVideoBench", split="validation")
    return [dict(r) for r in ds]


def load_raw_records(root: Path = DEFAULT_ROOT) -> list[dict]:
    records = _from_local_json(root)
    if records:
        return records
    return _from_hf_datasets()


# ---------------- normalization ----------------

def _normalize(rec: dict, root: Path) -> Optional[LVBItem]:
    rid = str(rec.get("id") or rec.get("question_id") or rec.get("uid") or rec.get("video_id"))
    vid = rec.get("video_path") or rec.get("video") or rec.get("video_id")
    if vid is None:
        return None
    vid_path = vid if os.path.isabs(str(vid)) else str(root / "videos" / str(vid))
    duration = float(
        rec.get("duration") or rec.get("duration_seconds") or rec.get("video_duration") or 0.0
    )
    bucket = bucket_for(duration)
    if bucket is None:
        return None
    cand = rec.get("candidates") or rec.get("options") or rec.get("choices")
    # LongVideoBench has 3-, 4-, and 5-way MCQ items (verified against lvb_val.json:
    # 4 cands -> 353, 5 cands -> 983, 3 cands -> 1). Accept any in [2, 5].
    if cand is None or not (2 <= len(cand) <= 5):
        return None
    correct = rec.get("correct_choice", rec.get("answer", rec.get("label")))
    if correct is None:
        return None
    if isinstance(correct, str):
        correct = "ABCDE".index(correct.strip().upper())
    correct = int(correct)
    if not (0 <= correct < len(cand)):
        return None
    return LVBItem(
        id=rid,
        video_path=str(vid_path),
        duration_seconds=duration,
        duration_bucket=bucket,
        question=str(rec.get("question") or ""),
        candidates=[str(c) for c in cand],
        correct_choice=correct,
        subtitle_path=rec.get("subtitle_path"),
        raw=rec,
    )


def load_all_items(root: Path = DEFAULT_ROOT) -> list[LVBItem]:
    raw = load_raw_records(root)
    items = [it for it in (_normalize(r, root) for r in raw) if it is not None]
    return items


# ---------------- stratified split ----------------

def make_split(items: list[LVBItem], seed: int = 0,
               cal_per_bucket: Optional[dict[str, int]] = None,
               targets: Optional[dict[str, int]] = None,
               cal_fraction: float = 1.0 / 3.0,
               ) -> dict[str, list[str]]:
    """Return {'cal': [ids], 'eval': [ids]} with stratification.

    By default: 100 cal + 200 eval, stratified per bucket using STRATA_TARGETS
    and `cal_fraction = 1/3`.
    """
    targets = targets or STRATA_TARGETS
    rng = random.Random(seed)
    by_bucket: dict[str, list[LVBItem]] = {}
    for it in items:
        by_bucket.setdefault(it.duration_bucket, []).append(it)

    cal_ids: list[str] = []
    eval_ids: list[str] = []
    for bucket, total in targets.items():
        pool = by_bucket.get(bucket, [])
        if len(pool) < total:
            print(f"[warn] bucket={bucket} has only {len(pool)} items (need {total})")
            total = len(pool)
        rng.shuffle(pool)
        chosen = pool[:total]
        n_cal = int(round(total * cal_fraction)) if cal_per_bucket is None else cal_per_bucket[bucket]
        cal_ids.extend(it.id for it in chosen[:n_cal])
        eval_ids.extend(it.id for it in chosen[n_cal:])
    return {"cal": cal_ids, "eval": eval_ids}


def save_split(split: dict[str, list[str]], path: Path = DEFAULT_SPLIT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: Path = DEFAULT_SPLIT_FILE) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


def filter_items(items: list[LVBItem], ids: list[str]) -> list[LVBItem]:
    idset = set(ids)
    return [it for it in items if it.id in idset]


# ---------------- prompt formatting ----------------

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def format_mcq_messages(item: LVBItem, n_frames: int, max_pixels: int = 360 * 420) -> list[dict]:
    """Build the Qwen2.5-VL chat-format messages for a LongVideoBench MCQ item.

    Returns a single-turn user message with one video + the question prompt.
    The processor + qwen_vl_utils.process_vision_info handle the rest.
    Supports any 2-5-way MCQ (LongVideoBench has both 4- and 5-way items).
    """
    n = len(item.candidates)
    letters = OPTION_LETTERS[:n]
    options_text = "\n".join(f"{letter}. {cand}" for letter, cand in zip(letters, item.candidates))
    letter_list = ", ".join(letters)
    user_text = (
        f"{item.question}\n\nOptions:\n{options_text}\n\n"
        f"Answer with a single letter from {letter_list}."
    )
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{item.video_path}",
                    "max_pixels": max_pixels,
                    "nframes": n_frames,
                },
                {"type": "text", "text": user_text},
            ],
        }
    ]


def answer_token_ids(processor, n: int = 5) -> list[int]:
    """Resolve single-token IDs for the first n option letters (default 5: A..E).

    Falls back to encoding ' <letter>' if the bare letter is multi-token (Qwen
    tokenizers usually map 'A' to a single token, but we don't trust it).
    """
    tok = processor.tokenizer
    ids: list[int] = []
    for letter in OPTION_LETTERS[:n]:
        for cand in (letter, " " + letter):
            tids = tok.encode(cand, add_special_tokens=False)
            if len(tids) == 1:
                ids.append(tids[0])
                break
        else:
            raise RuntimeError(f"Could not resolve single-token id for option letter {letter!r}.")
    return ids


# ---------------- CLI helper ----------------

def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_SPLIT_FILE)
    args = ap.parse_args()
    items = load_all_items(args.root)
    print(f"loaded {len(items)} items from {args.root}")
    split = make_split(items, seed=args.seed)
    save_split(split, args.out)
    print(f"wrote split: cal={len(split['cal'])} eval={len(split['eval'])} -> {args.out}")


if __name__ == "__main__":
    _cli()
