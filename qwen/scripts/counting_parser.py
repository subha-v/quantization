"""Counting-image output parser and scorer (Exp T-mini Phase 3).

The MM-NIAH counting-image task asks the model to return a JSON list of
integers giving the count of the needle pattern in each haystack image,
e.g. "[0, 1, 0, 2, 0]" for num_images=6 (5 haystack pages + 1 needle pattern).

The model output is freeform text and may include extra commentary, code
fences, or invalid formats. This parser:
  - Locates the first plausible "[...]" list-of-ints in the output.
  - Reports whether the output is well-formed JSON (or close enough).
  - Compares the parsed list against the gold counts and reports several
    metrics: exact match, length match, soft accuracy (1 - normalized L1).
"""
from __future__ import annotations

import json
import re
from typing import Optional


_LIST_INT_RE = re.compile(r"\[\s*(?:-?\d+\s*(?:,\s*-?\d+\s*)*)?\]")


def parse_counting_output(text: str) -> dict:
    """Parse the model's freeform text into a list of ints.

    Returns:
      {
        "parsed": list[int] | None,
        "valid_format": bool,
        "predicted_length": int | None,
        "raw_match": str | None,
      }

    valid_format is True if a well-formed "[i, i, ...]" pattern was found
    and successfully decoded. Note: an empty list "[]" is considered valid
    (length 0).
    """
    if text is None:
        return {"parsed": None, "valid_format": False,
                "predicted_length": None, "raw_match": None}
    # Strip code fences if present.
    s = text.strip()
    # Try direct JSON decode first (cheapest path for clean outputs).
    parsed: Optional[list[int]] = None
    raw_match: Optional[str] = None
    try:
        candidate = json.loads(s)
        if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
            parsed = list(candidate)
            raw_match = s
    except (json.JSONDecodeError, ValueError):
        pass

    if parsed is None:
        # Find the first list-of-ints substring.
        m = _LIST_INT_RE.search(s)
        if m is not None:
            raw_match = m.group(0)
            try:
                candidate = json.loads(raw_match)
                if isinstance(candidate, list) and all(isinstance(x, int) for x in candidate):
                    parsed = list(candidate)
            except (json.JSONDecodeError, ValueError):
                parsed = None

    valid_format = parsed is not None
    return {
        "parsed": parsed,
        "valid_format": valid_format,
        "predicted_length": (len(parsed) if parsed is not None else None),
        "raw_match": raw_match,
    }


def score_counting(parsed: Optional[list[int]], gold: list[int]) -> dict:
    """Compare parsed prediction against gold counts.

    Returns metrics:
      exact_match:    bool — parsed == gold exactly
      length_match:   bool — len(parsed) == len(gold)
      soft_accuracy:  float in [0, 1] — 1 - (L1 distance / max(L1 of gold, 1)).
                      Robust to partial errors. If lengths differ, the shorter
                      list is right-padded with zeros for the L1 calc.
      pred_sum:       int | None — sum of predicted counts
      gold_sum:       int — sum of gold counts
      sum_match:      bool — pred_sum == gold_sum
      missing_format: bool — True if parsed is None (parser couldn't decode)
    """
    gold_list = list(gold)
    gold_sum = int(sum(gold_list))
    if parsed is None:
        return {
            "exact_match": False,
            "length_match": False,
            "soft_accuracy": 0.0,
            "pred_sum": None,
            "gold_sum": gold_sum,
            "sum_match": False,
            "missing_format": True,
        }
    pred = list(parsed)
    exact = (pred == gold_list)
    length_match = (len(pred) == len(gold_list))
    # L1 distance with right-padding to max length.
    nmax = max(len(pred), len(gold_list), 1)
    pred_padded = pred + [0] * (nmax - len(pred))
    gold_padded = gold_list + [0] * (nmax - len(gold_list))
    l1 = sum(abs(a - b) for a, b in zip(pred_padded, gold_padded))
    # Normalize: divide by max(L1 of gold, 1). Soft accuracy = 1 - normalized L1.
    denom = max(sum(abs(x) for x in gold_padded), 1)
    soft = max(0.0, 1.0 - (l1 / denom))
    pred_sum = int(sum(pred))
    return {
        "exact_match": bool(exact),
        "length_match": bool(length_match),
        "soft_accuracy": float(soft),
        "pred_sum": pred_sum,
        "gold_sum": gold_sum,
        "sum_match": (pred_sum == gold_sum),
        "missing_format": False,
    }
