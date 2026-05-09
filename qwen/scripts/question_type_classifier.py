"""Question-type heuristic classifier for Exp G G8 (type-adaptive frame budget).

Tags each LongVideoBench MCQ question into one of six labels by keyword
matching, then maps the label to a frame budget. The mapping is intentionally
simple — the goal is to test whether *any* text-side signal predicts which
items deserve more frames, not to ship a great classifier.

LongVideoBench items often include a long scene description prefix before
the actual interrogative ("On a grass field, a woman dressed in a sleeveless
white shirt is crouching on the grass. In front of her lies ... what does
she do next?"). To avoid false-positive matches against scene-description
prose ("during", "before", "after" occur naturally in description text), we
classify only on the **trailing interrogative clause**: the substring after
the last `.` (or start of string), with the trailing `?` stripped. Keywords
are also question-form-specific ("how many", "what color is", "what does",
not bare "many" / "color" / "doing").

Used by:
  - `expG_type_adaptive.py` (post-process: routes each item's row to the
    matching G1 / G3 / G4 precomputed forward by frame budget)
  - `expG_smoke.py` (coverage check + budget-distribution sanity on cal-100)
"""
from __future__ import annotations

from typing import Iterable, Literal


QuestionType = Literal["count", "temporal", "ocr", "detail", "action", "other"]

# Frame budget per type. Tuned to land at weighted-avg ~128-135 frames on
# the LongVideoBench cal-100 question mix (verified by expG_smoke.py check 6).
#
# Design intent: items whose questions ask for high-detail content (count,
# OCR, visual change/detail) get the full 256-frame budget; temporally-
# anchored items (subtitle / phrase anchor) and action items get 128 because
# the anchor narrows the relevant temporal range; everything else falls back
# to 64. LongVideoBench is temporal-anchored-question-heavy, so temporal at
# 128 keeps the weighted avg in target range.
BUDGET_MAP: dict[QuestionType, int] = {
    "count": 256,
    "ocr": 256,
    "detail": 256,
    "temporal": 128,
    "action": 128,
    "other": 64,
}


# Question-form keyword tables. Order of `classify_question_type` matters:
# count > temporal > ocr > detail > action > other.
#
# We match against the *trailing interrogative* only (see _trim_to_question).
# Keywords are question-form-specific to keep precision high; they assume the
# trimmed string starts with "what/when/how/who" or similar interrogative.
_COUNT_KEYWORDS = (
    "how many", "the number of", "total number", "the count of",
)
# Temporal includes subtitle-anchored prompts ("when the subtitle ...",
# "after mentioning ...") because those force the model to LOCATE a moment
# in the video before answering, which benefits from denser frame coverage.
_TEMPORAL_KEYWORDS = (
    "when does", "when is", "when did", "when do",
    "when the subtitle", "when the phrase", "when the text",
    "when the image", "when the figure", "when '", 'when "',
    "after the subtitle", "after the phrase", "after the text",
    "after mentioning", "after the image",
    "before the subtitle", "before the phrase", "before mentioning",
    "what happens after", "what happens before",
    "what happens next", "what happens first",
    "what is the first", "what is the last",
    "the first time", "the last time",
    "in what order", "in chronological order", "sequences of scenes",
    "right before", "right after",
    "do next", "do afterwards", "do after",
    "happen next", "happen first", "happen last",
    "appeared first", "appeared before", "appears first",
    "appeared at", "appears at",
    "scene appears", "scenes appear",
    "appeared in", "where else has",
    "at the same time", "simultaneously",
    "first appearance", "first appeared", "first time",
)
_OCR_KEYWORDS = (
    "what text", "what is written", "what does the sign",
    "what does the label", "what does the title",
    "what word", "what words", "what letters", "what does the caption",
    "what is the name on", "what is the title",
    "what does the subtitle",
)
_DETAIL_KEYWORDS = (
    "what color", "what colour", "what shape",
    "what is the color", "what is the colour",
    "what is the shape", "what kind of", "what type of",
    "what changes", "what change", "what other changes",
    "what objects", "what object",
    "describe", "look like",
)
_ACTION_KEYWORDS = (
    "what does the man", "what does the woman", "what does the person",
    "what is the man", "what is the woman", "what is the person",
    "what did the man", "what did the woman", "what did the person",
    "what did the", "what does the",
    "what is he doing", "what is she doing", "what are they doing",
    "what is this man", "what is this woman", "what is this person",
    "what did this", "what is this",
    "what action", "who is",
    "do on her", "do on his", "do on their",
)


def _trim_to_question(q: str) -> str:
    """Return the trailing interrogative clause: substring after the last `.`
    (or the whole string if no `.`), lowercased, with trailing whitespace and
    `?` stripped.
    """
    s = q.strip()
    # Take everything after the last sentence-ending period. Keep the result
    # if non-empty after stripping whitespace; otherwise fall back to the full
    # question (no `.` at all).
    if "." in s:
        candidate = s.rsplit(".", 1)[-1].strip()
        if candidate:
            s = candidate
    s = s.rstrip("?").strip().lower()
    return s


def classify_question_type(q: str) -> QuestionType:
    """Return one of {count, temporal, ocr, detail, action, other}.

    Pure keyword heuristic on the *trailing interrogative clause*, in order:
    count > temporal > ocr > detail > action > other. Case-insensitive.
    """
    q_low = _trim_to_question(q)
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
