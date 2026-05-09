"""Question-type heuristic classifier for Exp G G8 (type-adaptive frame budget).

Tags each LongVideoBench MCQ question into one of six labels by keyword
matching, then maps the label to a frame budget. The mapping is intentionally
simple — the goal is to test whether *any* text-side signal predicts which
items deserve more frames, not to ship a great classifier.

Labels are checked in order of specificity: a question that contains both
"how many" and "color" is classified as `count`, not `detail`, because count
questions are more frame-hungry on LongVideoBench.

Used by:
  - `expG_type_adaptive.py` (post-process: routes each item's row to the
    matching G1 / G3 / G4 precomputed forward by frame budget)
  - `expG_smoke.py` (coverage check + budget-distribution sanity on cal-100)
"""
from __future__ import annotations

from typing import Iterable, Literal


QuestionType = Literal["count", "temporal", "ocr", "detail", "action", "other"]

# Frame budget per type. Tuned to land at average ~128 frames on a typical
# LongVideoBench question mix (verified on cal-100 in expG_smoke.py).
BUDGET_MAP: dict[QuestionType, int] = {
    "count": 256,
    "temporal": 256,
    "ocr": 256,
    "detail": 128,
    "action": 128,
    "other": 64,
}


# Keyword tables. Order of `classify_question_type` matters: count > temporal
# > ocr > detail > action > other.
_COUNT_KEYWORDS = ("how many", "count ", "number of", "total of", "amount of")
_TEMPORAL_KEYWORDS = (
    "when ", "after ", "before ", "begin", "end ", "ends", "ending",
    "first time", "last time", "during", "while ", "earlier", "later",
    "happens next", "happens first", "in the beginning", "at the end",
    "right before", "right after",
)
_OCR_KEYWORDS = (
    "text ", "sign ", "label ", "labeled ", "written", "letters", "word ",
    "words ", "logo", "title ", "subtitle", "caption", "name on ",
)
_DETAIL_KEYWORDS = (
    "color", "shape", "small", "large", "size of", "what kind",
    "what type of", "describe", "specific", "specifically", "exact",
    "object on", "item on",
)
_ACTION_KEYWORDS = (
    "who ", "what does", "what is the man", "what is the woman",
    "what is the person", "doing", "action", "performs", "is performing",
    "takes", "takes the", "puts", "picks up", "places", "moves the",
)


def classify_question_type(q: str) -> QuestionType:
    """Return one of {count, temporal, ocr, detail, action, other}.

    Pure keyword heuristic, case-insensitive, lowest-cost dispatch. Returns
    `other` for anything that doesn't match the more specific labels.
    """
    q_low = q.lower()
    if any(w in q_low for w in _COUNT_KEYWORDS):
        return "count"
    if any(w in q_low for w in _TEMPORAL_KEYWORDS):
        return "temporal"
    if any(w in q_low for w in _OCR_KEYWORDS):
        return "ocr"
    if any(w in q_low for w in _DETAIL_KEYWORDS):
        return "detail"
    if any(w in q_low for w in _ACTION_KEYWORDS):
        return "action"
    return "other"


def assigned_frames(q: str) -> int:
    """Convenience: classify q then look up frame budget."""
    return BUDGET_MAP[classify_question_type(q)]


def label_distribution(questions: Iterable[str]) -> dict[QuestionType, int]:
    """Count items per label. Used for budget-map sanity in smoke tests."""
    counts: dict[QuestionType, int] = {k: 0 for k in BUDGET_MAP}
    for q in questions:
        counts[classify_question_type(q)] += 1
    return counts


def weighted_avg_frames(questions: Iterable[str]) -> float:
    """Mean of `assigned_frames(q)` over the supplied questions."""
    qs = list(questions)
    if not qs:
        return float("nan")
    total = sum(assigned_frames(q) for q in qs)
    return total / len(qs)
