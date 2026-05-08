"""Text-K slice boundary detection for Exp E1 (text-K slice ablation).

The Qwen2.5-VL prompt for an LVB MCQ item layouts as roughly:

  [0, last_im_end_before_vision):   header
      <|im_start|>system\\n<system message>\\n<|im_end|>\\n
      <|im_start|>user\\n
  [last_im_end_before_vision, v_start):  the <|vision_start|> tag (1 token)
  [v_start, v_end):                  visual tokens (~5760 for 64 frames)
  [v_end, v_end + 1):                <|vision_end|> tag (1 token)
  [v_end + 1, last_im_end):          question + "\\n\\nOptions:\\n" + options
                                     + "\\n\\nAnswer with..." + <|im_end|>
  [last_im_end + 1, seq_len):        answer-prefix
                                     <|im_start|>assistant\\n

The five logical text slices E1 wants to ablate (separately or in unions):
  header           — everything before the vision tag (Qwen's auto-injected scaffolding)
  question         — item.question
  options          — "Options:\\nA. ...\\nB. ..." block
  instruction      — "Answer with a single letter from A, B, C, D."
  answer_prefix    — <|im_end|>\\n<|im_start|>assistant\\n at the end

This module finds those spans by:
  1. Locating special-token IDs (<|im_start|>, <|im_end|>, <|vision_start|>, <|vision_end|>)
  2. Re-tokenizing each substring (item.question, options block, instruction line)
     with `add_special_tokens=False` and finding the resulting token sequence
     in input_ids via subsequence match (with up to 3 leading-character
     trims to absorb BPE merges across boundaries).
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch

from data_longvideobench import LVBItem, OPTION_LETTERS
from visual_tokens import _resolve_special_id, find_visual_token_span


# ---------------- subsequence matching ----------------

def _find_subsequence(haystack: list[int], needle: list[int], start: int = 0,
                      end: Optional[int] = None) -> int:
    """Find the leftmost index >= start where needle appears in haystack.
    Returns -1 if not found. O((haystack-end - start) * len(needle))."""
    if not needle:
        return start
    H = len(haystack) if end is None else end
    n = len(needle)
    if start + n > H:
        return -1
    first = needle[0]
    i = start
    while i + n <= H:
        if haystack[i] == first and haystack[i:i + n] == needle:
            return i
        i += 1
    return -1


def _tokenize(processor, text: str) -> list[int]:
    return list(processor.tokenizer.encode(text, add_special_tokens=False))


def _find_substring_span(haystack: list[int], processor, text: str,
                         start: int = 0, end: Optional[int] = None) -> tuple[int, int, str]:
    """Try to locate `text` as a contiguous span in haystack[start:end].

    Returns (a, b, status):
      - (a, b, "exact")    if exact tokenization matches
      - (a, b, "trim_N")   if matched after trimming N leading characters from `text`
      - (-1, -1, "fail")   if no match within 3 leading-character trims

    Used for both the user_text-side substrings (question / options / instruction)
    and any other substring search.
    """
    for trim in range(0, 4):
        s = text[trim:] if trim else text
        if not s:
            continue
        ids = _tokenize(processor, s)
        if not ids:
            continue
        i = _find_subsequence(haystack, ids, start=start, end=end)
        if i >= 0:
            tag = "exact" if trim == 0 else f"trim_{trim}"
            return i, i + len(ids), tag
    return -1, -1, "fail"


# ---------------- slice detection ----------------

def find_text_slice_spans(input_ids: torch.Tensor, processor, item: LVBItem
                          ) -> dict[str, object]:
    """Return half-open [a, b) ranges for each text slice + diagnostic info.

    Returned dict keys (always present):
      header           : (a, b) inclusive at start, ending at the <|vision_start|> tag (exclusive)
      visual_wrapper   : (v_start - 1, v_end + 1)  — the <|vision_start|>...<|vision_end|> tags
      question         : (a, b)
      options          : (a, b) — covers "\\n\\nOptions:\\nA. ...\\nB. ..."
      instruction      : (a, b) — covers "\\n\\nAnswer with a single letter from ..."
      answer_prefix    : (a, b) — covers <|im_end|>\\n<|im_start|>assistant\\n at the end
      _seq_len         : int
      _v_start, _v_end : ints
      _warnings        : list[str] (slice names that fell back to fuzzy match or failed)
    """
    if input_ids.dim() == 2:
        ids = input_ids[0].tolist()
    else:
        ids = input_ids.tolist()
    seq_len = len(ids)

    # 1. vision span
    v_start, v_end = find_visual_token_span(input_ids, processor)

    # 2. special tokens
    im_start_id = _resolve_special_id(processor, "<|im_start|>")
    im_end_id = _resolve_special_id(processor, "<|im_end|>")
    if im_start_id is None or im_end_id is None:
        raise RuntimeError("Could not resolve <|im_start|> / <|im_end|> token ids.")

    # 3. last <|im_end|> before assistant prefix — locate by walking from end
    #    The chat template structure is: ... user message ... <|im_end|>\n<|im_start|>assistant\n
    last_im_start = -1
    for i in range(seq_len - 1, -1, -1):
        if ids[i] == im_start_id:
            last_im_start = i
            break
    if last_im_start < 0:
        raise RuntimeError("No <|im_start|> token found anywhere in input_ids.")
    # The user message's closing <|im_end|> is the last im_end BEFORE last_im_start
    last_im_end = -1
    for i in range(last_im_start - 1, -1, -1):
        if ids[i] == im_end_id:
            last_im_end = i
            break
    if last_im_end < 0:
        raise RuntimeError("No <|im_end|> token found before assistant prefix.")

    # 4. header = [0, v_start - 1)
    header_span = (0, max(0, v_start - 1))
    visual_wrapper_span = (max(0, v_start - 1), min(seq_len, v_end + 1))

    # 5. user-message body = [v_end + 1, last_im_end)
    body_lo = v_end + 1
    body_hi = last_im_end

    warnings: list[str] = []

    # 6. Marker-based slice detection. We do NOT tokenize item.question standalone
    # because BPE merges across boundaries cause that to mismatch in 100% of items.
    # Instead, locate the structural markers ('Options:' and 'Answer with...') in
    # the user-message body, then derive the question span as the range between
    # them. This is robust across tokenizer behavior.
    n_options = len(item.candidates)
    letter_list = ", ".join(OPTION_LETTERS[:n_options])

    # Try several spelling variants in order of specificity (most-specific first).
    options_marker_candidates = [
        "\n\nOptions:\n", "\n\nOptions:", "\nOptions:\n", "\nOptions:", "Options:",
    ]
    answer_marker_candidates = [
        f"\n\nAnswer with a single letter from {letter_list}.",
        f"\n\nAnswer with a single letter from {letter_list}",
        f"\nAnswer with a single letter from {letter_list}",
        f"Answer with a single letter from {letter_list}",
        "Answer with a single letter",
    ]

    o_a, o_b, o_status = -1, -1, "fail"
    for s in options_marker_candidates:
        a, b, st = _find_substring_span(ids, processor, s, start=body_lo, end=body_hi)
        if a >= 0:
            o_a, o_b, o_status = a, b, f"{s!r}/{st}"
            break

    i_a, i_b, i_status = -1, -1, "fail"
    instr_search_start = (o_b if o_b >= 0 else body_lo)
    for s in answer_marker_candidates:
        a, b, st = _find_substring_span(ids, processor, s, start=instr_search_start, end=body_hi)
        if a >= 0:
            i_a, i_b, i_status = a, b, f"{s!r}/{st}"
            break

    if o_a < 0:
        warnings.append("options_marker:fail")
    elif "/exact" not in o_status:
        warnings.append(f"options_marker:{o_status}")
    if i_a < 0:
        warnings.append("instruction_marker:fail")
    elif "/exact" not in i_status:
        warnings.append(f"instruction_marker:{i_status}")

    # Derive spans from markers
    if o_a >= 0:
        question_span = (body_lo, o_a)
        if i_a >= 0:
            options_span = (o_a, i_a)
            instruction_span = (i_a, body_hi)
        else:
            options_span = (o_a, body_hi)
            instruction_span = (body_hi, body_hi)
    else:
        # Last-resort fallback: dump entire body into question slice
        warnings.append("fallback:body_as_question")
        question_span = (body_lo, body_hi)
        options_span = (body_hi, body_hi)
        instruction_span = (body_hi, body_hi)

    # 7. answer_prefix = [last_im_end, seq_len)
    answer_prefix_span = (last_im_end, seq_len)

    return {
        "header": header_span,
        "visual_wrapper": visual_wrapper_span,
        "question": question_span,
        "options": options_span,
        "instruction": instruction_span,
        "answer_prefix": answer_prefix_span,
        "_seq_len": seq_len,
        "_v_start": v_start,
        "_v_end": v_end,
        "_body_lo": body_lo,
        "_body_hi": body_hi,
        "_warnings": warnings,
    }


# ---------------- mask construction ----------------

def union_mask(seq_len: int, slice_ranges: Iterable[tuple[int, int]]) -> torch.Tensor:
    """Build a [seq_len] bool mask True for any position in any provided (a, b) range."""
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for a, b in slice_ranges:
        if a < 0 or b > seq_len or a >= b:
            continue
        mask[a:b] = True
    return mask


def positions_to_mask(seq_len: int, positions: Iterable[int]) -> torch.Tensor:
    """Build a [seq_len] bool mask True for each scalar position in `positions`."""
    mask = torch.zeros(seq_len, dtype=torch.bool)
    for p in positions:
        if 0 <= p < seq_len:
            mask[int(p)] = True
    return mask


def text_positions(seq_len: int, v_start: int, v_end: int) -> list[int]:
    """All non-visual position indices [0, v_start) U [v_end, seq_len)."""
    return list(range(0, max(0, v_start))) + list(range(min(seq_len, v_end), seq_len))


# ---------------- per-text-token K residual capture (for E1.10) ----------------


from transformers.cache_utils import DynamicCache as _DynamicCache


class TextKResidualCache(_DynamicCache):
    """`DynamicCache` subclass that records per-position INT4 K residual.

    Storage stays BF16 (DynamicCache behavior). On each `update()` we
    additionally compute (per query position t)
        rel[t] = ||K[t, L, h, :] - Q_int4(K[t, L, h, :])||_2 / ||K[t, L, h, :]||_2
    summed across heads h and accumulated across layers L. After one prefill
    the running sum has been incremented num_layers * num_kv_heads times per
    position; `finalize()` divides by that count.

    NOTE: subclasses DynamicCache (proper `is-a` inheritance) so transformers
    can do `cache[layer_idx]`, `len(cache)`, etc. via the parent class.
    """

    def __init__(self, num_layers: int, seq_len: int):
        super().__init__()
        self.num_layers = num_layers
        self.seq_len = seq_len
        # accumulator shape [seq_len], summed across (L, h)
        self._sum = torch.zeros(seq_len, dtype=torch.float32)
        # number of (L, h) increments accumulated; divisor for finalize().
        self._count = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        from fake_quant_kv_cache import fake_quantize_kv
        with torch.no_grad():
            B, H, T, D = key_states.shape
            if T > 0:
                Kq = fake_quantize_kv(key_states, 4)
                num = (key_states - Kq).pow(2).sum(dim=-1).sqrt()  # [B, H, T]
                den = key_states.pow(2).sum(dim=-1).sqrt().clamp_min(1e-8)  # [B, H, T]
                rel = (num / den)[0]  # [H, T]
                per_pos = rel.sum(dim=0).float().cpu()  # [T]
                # Position offset in the running cache (0 during prefill).
                cache_offset = super().get_seq_length(layer_idx)
                end = cache_offset + T
                if end <= self.seq_len:
                    self._sum[cache_offset:end] += per_pos
                    self._count += int(H)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def finalize(self) -> torch.Tensor:
        """Return [seq_len] tensor of mean per-position residual norm.

        Mean is over (L * num_kv_heads) increments — so for a 28L x 4-head
        Qwen2.5-VL model with one prefill, this divides by 28 * 4 = 112.
        """
        if self._count == 0:
            return torch.zeros(self.seq_len, dtype=torch.float32)
        return self._sum / float(self._count)


@torch.no_grad()
def capture_text_k_residuals(model, processor, inputs, num_kv_heads: int, num_layers: int
                             ) -> torch.Tensor:
    """Run a single forward pass with a TextKResidualCache; return [seq_len] residual.

    Caller can then slice by `text_positions(...)` to get residuals at text positions.
    """
    seq_len = int(inputs["input_ids"].shape[1])
    cache = TextKResidualCache(num_layers=num_layers, seq_len=seq_len)
    _ = model.generate(
        **inputs, past_key_values=cache, max_new_tokens=1, do_sample=False,
        return_dict_in_generate=True, output_scores=True, use_cache=True,
    )
    return cache.finalize()
