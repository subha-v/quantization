"""MM-NIAH (Needle In A Multimodal Haystack) loader for Exp P.

Dataset: OpenGVLab/MM-NIAH on HuggingFace.
Mirrors data_longvideobench.py structure: dataclass + stratified split + Qwen2.5-VL
chat formatting + answer-token resolution. Uses the retrieval-image, val split.

Each MM-NIAH item is an interleaved (text, image) sequence with one image needle
inserted at a known `placed_depth`. The MCQ presents 4 candidate images (A-D) and
asks the model which one matches the in-context needle.

Record schema (one JSONL line per item):
  id: int
  context: str               # text with <image> placeholders, one per in-context image
  question: str              # "Which of the following images appears in..."
  answer: int                # 0-indexed correct choice (matches choices_image_path)
  images_list: list[str]     # all in-context image paths, in <image>-occurrence order
  meta:
    placed_depth: list[float]    # where needle was placed (0..1)
    context_length: int          # context token length (rough)
    num_images: int              # len(images_list)
    needles: list[str]           # needle filename(s) — different naming from images_list
    choices_image_path: list[str] # 4 candidate image paths (A,B,C,D)
    category: str                # "find-image"

Needle identification: in retrieval-image, exactly one entry of images_list lives in
`obelics_paste/find-image/` or `abnormal_pic*/`. That's the needle. All other entries
are distractor images from the surrounding Obelics article.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_ROOT = Path(os.environ.get("MM_NIAH_ROOT", "/data/subha2/mm_niah"))
DEFAULT_SUBSET = "mm_niah_val"
DEFAULT_TASK = "retrieval-image"
SUPPORTED_TASKS = ("retrieval-image", "reasoning-image", "counting-image")
DEFAULT_SPLIT_FILE = (
    Path(__file__).resolve().parents[1] / "calibration" / "mm_niah_split_seed0.json"
)


def split_file_for_task(task: str) -> Path:
    """Per-task split file. retrieval-image uses the legacy DEFAULT_SPLIT_FILE
    for back-compat with Exp P; reasoning-image gets its own seed-0 split.
    """
    if task == "retrieval-image":
        return DEFAULT_SPLIT_FILE
    return DEFAULT_SPLIT_FILE.parent / f"mm_niah_{task}_split_seed0.json"


@dataclass
class MMNiahItem:
    id: str
    context: str
    question: str
    correct_choice: int             # 0..3 for MCQ tasks; -1 for counting-image
    images_list: list[str]          # absolute paths after `_resolve_image_paths`
    choices_image_paths: list[str]  # 4 absolute paths, in choice order A..D
                                    # (empty for counting-image)
    needle_idx_in_images: int       # which position of images_list is the needle
    placed_depth: float             # 0..1
    context_length: int             # from dataset metadata (text+image rough tokens)
    context_length_bucket: str      # short / mid / long (binned by context_length)
    num_images: int
    # T-mini extensions:
    # - For counting-image, gold_counts holds the per-image-page integer counts
    #   (length = num_images - 1; the first image in images_list is the needle
    #   pattern, the rest are the haystack pages where counts apply).
    # - task records the originating task so downstream code can branch
    #   (MCQ scoring vs counting-image generation+list-parse).
    gold_counts: Optional[list[int]] = None
    task: str = DEFAULT_TASK
    raw: Optional[dict] = field(default=None, repr=False)


# ---------------- bucketing ----------------

# Brackets chosen empirically from the val split distribution (median context_length
# is ~15K; quartiles ~5K/15K/38K). We filter very_long > 32K because Qwen2.5-VL's
# default context window is 32K and items >32K would either OOM or need extension.
CONTEXT_BUCKETS = [
    ("short", 596, 5_000),
    ("mid", 5_000, 12_000),
    ("long", 12_000, 32_000),
]
STRATA_TARGETS = {"short": 67, "mid": 67, "long": 66}  # 200 total


def context_bucket(ctx_len: int) -> Optional[str]:
    for name, lo, hi in CONTEXT_BUCKETS:
        if lo <= ctx_len < hi:
            return name
    return None


# ---------------- needle identification ----------------

_NEEDLE_PATH_TOKENS = ("find-image", "abnormal_pic")


def _find_needle_idx_retrieval(images_list: list[str]) -> int:
    """Return the index of the needle image within images_list (retrieval-image).

    Heuristic: needles are inserted from `obelics_paste/find-image/` or
    `abnormal_pic*/`; all other images come from the surrounding Obelics article
    and contain neither token. Verified on 520/520 retrieval-image records.
    """
    for i, p in enumerate(images_list):
        if any(tok in p for tok in _NEEDLE_PATH_TOKENS):
            return i
    return -1


def _find_needle_idx_reasoning(images_list: list[str], meta: dict) -> int:
    """Return the index of the needle image for reasoning-image.

    Reasoning-image's annotation puts the needle filename(s) in `meta["needles"]`.
    Match by basename against each image in images_list.
    """
    needles = meta.get("needles") or []
    if not needles:
        return -1
    needle_basenames = {os.path.basename(str(n)) for n in needles}
    for i, p in enumerate(images_list):
        if os.path.basename(p) in needle_basenames:
            return i
    # Fallback: substring match on full path (some annotations use sub-paths).
    for i, p in enumerate(images_list):
        for n in needles:
            if str(n) and str(n) in p:
                return i
    return -1


def _find_needle_idx_by_task(images_list: list[str], meta: dict, task: str) -> int:
    if task == "retrieval-image":
        return _find_needle_idx_retrieval(images_list)
    if task == "reasoning-image":
        return _find_needle_idx_reasoning(images_list, meta)
    if task == "counting-image":
        # In counting-image the first image is the needle pattern (e.g.
        # "abnormal_pic_val/sun.jpg") and the remaining images are the
        # haystack pages whose counts the model must report.
        return 0
    raise ValueError(f"unsupported task={task!r}")


# Back-compat alias.
def _find_needle_idx(images_list: list[str]) -> int:
    return _find_needle_idx_retrieval(images_list)


# ---------------- raw dataset reading ----------------

def _annotations_path(root: Path, subset: str, task: str) -> Path:
    return root / subset / "annotations" / f"{task}.jsonl"


def _images_root(root: Path, subset: str) -> Path:
    # images.tar.gz extracts to mm_niah_val/images/ (or mm_niah_test/images/)
    return root / subset / "images"


def load_raw_records(root: Path = DEFAULT_ROOT,
                     subset: str = DEFAULT_SUBSET,
                     task: str = DEFAULT_TASK) -> list[dict]:
    path = _annotations_path(root, subset, task)
    if not path.exists():
        raise FileNotFoundError(
            f"MM-NIAH annotations not found at {path}. "
            f"Run: huggingface-cli download OpenGVLab/MM-NIAH --repo-type dataset "
            f"--local-dir {root} --include 'mm_niah_val/*'"
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------- normalization ----------------

def _resolve_image_paths(paths: list[str], images_root: Path) -> list[str]:
    """Prepend the images root to relative image paths."""
    return [str(images_root / p) for p in paths]


def _normalize(rec: dict, images_root: Path, task: str = DEFAULT_TASK) -> Optional[MMNiahItem]:
    rid = str(rec.get("id"))
    context = rec.get("context", "")
    question = rec.get("question", "")
    answer = rec.get("answer")
    images_list_raw = rec.get("images_list", [])
    meta = rec.get("meta", {})
    choices_raw = meta.get("choices_image_path") or []
    placed = meta.get("placed_depth", [0.0])
    ctx_len = int(meta.get("context_length", 0))
    n_imgs = int(meta.get("num_images", len(images_list_raw)))

    if task == "counting-image":
        # Counting-image: answer is a list[int] of per-image needle counts.
        # The first image in images_list is the needle pattern; the answer
        # has length num_images - 1.
        if not isinstance(answer, list) or not all(isinstance(x, int) for x in answer):
            return None
        if len(answer) != max(0, n_imgs - 1):
            return None
        if len(images_list_raw) != n_imgs:
            return None
        if context.count("<image>") != n_imgs - 1:
            # Counting-image context only has placeholders for haystack pages;
            # the needle pattern is referenced inline in the question.
            return None
        bucket = context_bucket(ctx_len)
        if bucket is None:
            return None
        needle_idx = _find_needle_idx_by_task(images_list_raw, meta, task)
        if needle_idx < 0:
            return None
        return MMNiahItem(
            id=rid,
            context=context,
            question=question,
            correct_choice=-1,
            images_list=_resolve_image_paths(images_list_raw, images_root),
            choices_image_paths=[],  # no MCQ choices
            needle_idx_in_images=needle_idx,
            placed_depth=float(placed[0]) if placed else 0.0,
            context_length=ctx_len,
            context_length_bucket=bucket,
            num_images=n_imgs,
            gold_counts=list(answer),
            task=task,
            raw=rec,
        )

    # retrieval-image and reasoning-image are both 4-way single-choice MCQ over
    # candidate images.
    if answer is None or not isinstance(answer, int):
        return None
    if not (0 <= answer < 4):
        return None
    if len(choices_raw) != 4:
        return None
    if len(images_list_raw) != n_imgs:
        return None
    if context.count("<image>") != n_imgs:
        return None
    bucket = context_bucket(ctx_len)
    if bucket is None:
        return None
    needle_idx = _find_needle_idx_by_task(images_list_raw, meta, task)
    if needle_idx < 0:
        return None
    return MMNiahItem(
        id=rid,
        context=context,
        question=question,
        correct_choice=int(answer),
        images_list=_resolve_image_paths(images_list_raw, images_root),
        choices_image_paths=_resolve_image_paths(choices_raw, images_root),
        needle_idx_in_images=needle_idx,
        placed_depth=float(placed[0]) if placed else 0.0,
        context_length=ctx_len,
        context_length_bucket=bucket,
        num_images=n_imgs,
        task=task,
        raw=rec,
    )


def load_all_items(root: Path = DEFAULT_ROOT,
                   subset: str = DEFAULT_SUBSET,
                   task: str = DEFAULT_TASK) -> list[MMNiahItem]:
    raw = load_raw_records(root, subset, task)
    images_root = _images_root(root, subset)
    items = [it for it in (_normalize(r, images_root, task=task) for r in raw) if it is not None]
    return items


# ---------------- stratified split ----------------

CAL_TARGETS = {"short": 34, "mid": 33, "long": 33}  # 100 total


def make_split(items: list[MMNiahItem], seed: int = 0,
               eval_targets: Optional[dict[str, int]] = None,
               cal_targets: Optional[dict[str, int]] = None) -> dict[str, list[str]]:
    """Return {'cal': [ids], 'eval': [ids]} stratified by context-length bucket.

    Default: 100 cal (34/33/33) + 200 eval (67/67/66). Cal and eval are
    disjoint within each bucket. Items not picked go in neither set.
    """
    eval_targets = eval_targets or STRATA_TARGETS
    cal_targets = cal_targets or CAL_TARGETS
    rng = random.Random(seed)
    by_bucket: dict[str, list[MMNiahItem]] = {}
    for it in items:
        by_bucket.setdefault(it.context_length_bucket, []).append(it)
    cal_ids: list[str] = []
    eval_ids: list[str] = []
    for bucket in sorted(set(eval_targets) | set(cal_targets)):
        pool = by_bucket.get(bucket, [])
        n_eval = eval_targets.get(bucket, 0)
        n_cal = cal_targets.get(bucket, 0)
        need = n_eval + n_cal
        if len(pool) < need:
            print(f"[warn] mm_niah bucket={bucket} has only {len(pool)} items "
                  f"(need {need} = {n_eval} eval + {n_cal} cal)")
        rng.shuffle(pool)
        chosen = pool[:min(need, len(pool))]
        # First n_cal go to cal, next n_eval go to eval (disjoint within bucket).
        cal_ids.extend(it.id for it in chosen[:n_cal])
        eval_ids.extend(it.id for it in chosen[n_cal:n_cal + n_eval])
    return {"cal": cal_ids, "eval": eval_ids}


def save_split(split: dict[str, list[str]], path: Path = DEFAULT_SPLIT_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: Path = DEFAULT_SPLIT_FILE) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


def filter_items(items: list[MMNiahItem], ids: list[str]) -> list[MMNiahItem]:
    idset = set(ids)
    return [it for it in items if it.id in idset]


# ---------------- prompt formatting ----------------

OPTION_LETTERS = ["A", "B", "C", "D"]


def format_mcq_messages(item: MMNiahItem,
                        max_pixels_context: int = 144 * 144,
                        max_pixels_choices: int = 224 * 224) -> list[dict]:
    """Build Qwen2.5-VL chat messages for an MM-NIAH retrieval-image item.

    Builds a single user-turn with content alternating between text fragments
    (split on '<image>') and the corresponding in-context images, then appends the
    4 labeled candidate images and the answer instruction.

    Token-budget knobs:
      - max_pixels_context: caps in-context images small (~30-80 tokens each) to
        keep total prefill under Qwen2.5-VL's 32K window.
      - max_pixels_choices: larger budget for the 4 choices so the model can
        distinguish them in the comparison.
    """
    text_parts = item.context.split("<image>")
    if len(text_parts) - 1 != len(item.images_list):
        raise RuntimeError(
            f"MM-NIAH item {item.id}: <image> count ({len(text_parts) - 1}) does "
            f"not match images_list len ({len(item.images_list)})"
        )
    content: list[dict] = []
    for i, txt in enumerate(text_parts):
        if txt:
            content.append({"type": "text", "text": txt})
        if i < len(item.images_list):
            content.append({
                "type": "image",
                "image": "file://" + item.images_list[i],
                "max_pixels": max_pixels_context,
            })
    # MCQ tail
    content.append({"type": "text", "text": f"\n\n{item.question}\n\nChoices:\n"})
    for letter, img_path in zip(OPTION_LETTERS, item.choices_image_paths):
        content.append({"type": "text", "text": f"{letter}. "})
        content.append({
            "type": "image",
            "image": "file://" + img_path,
            "max_pixels": max_pixels_choices,
        })
        content.append({"type": "text", "text": "\n"})
    content.append({"type": "text", "text": "\nAnswer with a single letter from A, B, C, D."})
    return [{"role": "user", "content": content}]


def format_counting_messages(item: MMNiahItem,
                             max_pixels_context: int = 144 * 144,
                             max_pixels_needle: int = 144 * 144) -> list[dict]:
    """Build Qwen2.5-VL chat messages for an MM-NIAH counting-image item.

    Counting-image format (different from retrieval/reasoning-image MCQ):
      images_list[0]   - the needle pattern (referenced inline in the question)
      images_list[1:]  - the haystack pages, one <image> placeholder per page
      question         - explicit instruction: "Please help me collect the number
                         of this sun: <image> in each image in the above document,
                         for example: [x, x, x...]". The <image> in the question
                         refers to images_list[0] (the needle pattern).

    The model is expected to output a JSON list of length num_images - 1.
    """
    # The context contains num_images - 1 placeholders (haystack pages).
    text_parts = item.context.split("<image>")
    haystack_imgs = item.images_list[1:]
    if len(text_parts) - 1 != len(haystack_imgs):
        raise RuntimeError(
            f"MM-NIAH counting-image item {item.id}: <image> count "
            f"({len(text_parts) - 1}) does not match haystack image count "
            f"({len(haystack_imgs)})"
        )
    content: list[dict] = []
    for i, txt in enumerate(text_parts):
        if txt:
            content.append({"type": "text", "text": txt})
        if i < len(haystack_imgs):
            content.append({
                "type": "image",
                "image": "file://" + haystack_imgs[i],
                "max_pixels": max_pixels_context,
            })
    # Counting-image question has its OWN <image> referring to the needle pattern.
    q_parts = item.question.split("<image>")
    if len(q_parts) >= 2:
        content.append({"type": "text", "text": "\n\n" + q_parts[0]})
        content.append({
            "type": "image",
            "image": "file://" + item.images_list[0],
            "max_pixels": max_pixels_needle,
        })
        content.append({"type": "text", "text": "".join(q_parts[1:])})
    else:
        # Defensive fallback: needle inserted at end of question.
        content.append({"type": "text", "text": "\n\n" + item.question})
        content.append({
            "type": "image",
            "image": "file://" + item.images_list[0],
            "max_pixels": max_pixels_needle,
        })
    return [{"role": "user", "content": content}]


def answer_token_ids(processor, n: int = 4) -> list[int]:
    """Resolve single-token IDs for option letters A..D.

    Reuses the same fallback (' <letter>' if bare letter is multi-token) as the
    LongVideoBench loader. Qwen tokenizers usually map a bare 'A' to one token.
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
    ap.add_argument("--subset", default=DEFAULT_SUBSET)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output split JSON path. Defaults to per-(task, seed) "
                         "via split_file_for_task() with seed suffix.")
    args = ap.parse_args()
    if args.out is None:
        base = split_file_for_task(args.task)
        if args.seed == 0 and args.task == "retrieval-image":
            # Preserve legacy DEFAULT_SPLIT_FILE name for backward compat.
            args.out = base
        else:
            args.out = base.with_name(
                f"mm_niah_{args.task}_split_seed{args.seed}.json"
            )
    items = load_all_items(args.root, args.subset, args.task)
    print(f"loaded {len(items)} {args.task} items from {args.root}")
    bcounts: dict[str, int] = {}
    for it in items:
        bcounts[it.context_length_bucket] = bcounts.get(it.context_length_bucket, 0) + 1
    print(f"by bucket: {bcounts}")
    split = make_split(items, seed=args.seed)
    save_split(split, args.out)
    print(f"wrote split: cal={len(split['cal'])} eval={len(split['eval'])} -> {args.out}")


if __name__ == "__main__":
    _cli()
