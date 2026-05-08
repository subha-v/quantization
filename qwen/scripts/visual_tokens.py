"""Visual-token range and per-window mapping utilities for Qwen2.5-VL.

Qwen2.5-VL surrounds each video region in the tokenized prompt with a
`<|vision_start|>` ... `<|vision_end|>` pair. The tokens between are the
flattened video patch IDs (one per visual KV-cache position). Visual tokens
are temporally ordered (T then H then W after the patch flatten), so windows
of consecutive frames map to contiguous slices of the visual token range.
"""
from __future__ import annotations

from typing import Optional

import torch


def _resolve_special_id(processor, name: str) -> Optional[int]:
    tok = processor.tokenizer
    fn = getattr(tok, "convert_tokens_to_ids", None)
    if fn is None:
        return None
    tid = fn(name)
    # convert_tokens_to_ids returns the unk_id when the token is not in the vocab;
    # treat unk_id (or None) as "not found".
    if tid is None or tid == getattr(tok, "unk_token_id", -1):
        return None
    return int(tid)


def find_visual_token_span(input_ids: torch.Tensor, processor) -> tuple[int, int]:
    """Return (v_start, v_end) — the half-open range of visual token positions
    in the (single-batch) input_ids sequence.

    v_start is the position immediately AFTER `<|vision_start|>`.
    v_end is the position of `<|vision_end|>` (exclusive).

    Raises RuntimeError if the markers can't be located.
    """
    if input_ids.dim() == 2:
        ids = input_ids[0].tolist()
    else:
        ids = input_ids.tolist()

    vstart_id = _resolve_special_id(processor, "<|vision_start|>")
    vend_id = _resolve_special_id(processor, "<|vision_end|>")
    if vstart_id is None or vend_id is None:
        raise RuntimeError(
            "Could not resolve <|vision_start|>/<|vision_end|> token ids on this processor."
        )

    try:
        s = ids.index(vstart_id) + 1
    except ValueError as e:
        raise RuntimeError("No <|vision_start|> token found in input_ids") from e
    try:
        e = ids.index(vend_id, s)
    except ValueError as exc:
        raise RuntimeError("No <|vision_end|> token found after <|vision_start|>") from exc
    if e <= s:
        raise RuntimeError(f"Empty visual span: v_start={s}, v_end={e}")
    return s, e


def build_window_token_ranges(
    v_start: int, v_end: int, n_windows: int = 8
) -> list[tuple[int, int]]:
    """Partition [v_start, v_end) into n_windows contiguous (a, b) slices.

    Last window absorbs any remainder so the union covers the full visual span.
    """
    n_visual = v_end - v_start
    if n_visual <= 0:
        raise ValueError(f"Empty visual span: v_start={v_start}, v_end={v_end}")
    if n_windows <= 0:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    stride = n_visual // n_windows
    if stride == 0:
        raise ValueError(
            f"visual span ({n_visual} tokens) shorter than n_windows ({n_windows})"
        )
    ranges: list[tuple[int, int]] = []
    for k in range(n_windows):
        a = v_start + k * stride
        b = v_start + (k + 1) * stride if k < n_windows - 1 else v_end
        ranges.append((a, b))
    return ranges


def build_text_protect_visual_mask(
    seq_len: int,
    v_start: int,
    v_end: int,
    visual_protect_ranges: list[tuple[int, int]],
    *,
    text_protect: bool = True,
) -> torch.Tensor:
    """Build a [seq_len] bool mask: True = protected (BF16), False = INT4.

    Shape of the policy:
      - text positions ([:v_start] and [v_end:]) -> True if text_protect else False
      - visual positions ([v_start:v_end]) -> False by default
      - then for each (a, b) in visual_protect_ranges: mask[a:b] = True

    visual_protect_ranges may be empty (=> all visual at INT4).
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    if text_protect:
        if v_start > 0:
            mask[:v_start] = True
        if v_end < seq_len:
            mask[v_end:] = True
    for a, b in visual_protect_ranges:
        mask[a:b] = True
    return mask
