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

    # 6. Locate question / options / instruction within body
    n_options = len(item.candidates)
    letters = OPTION_LETTERS[:n_options]
    options_text = "\n".join(f"{letter}. {cand}" for letter, cand in zip(letters, item.candidates))
    letter_list = ", ".join(letters)
    options_block = f"\n\nOptions:\n{options_text}"
    instruction_block = f"\n\nAnswer with a single letter from {letter_list}."

    q_a, q_b, q_status = _find_substring_span(ids, processor, item.question,
                                              start=body_lo, end=body_hi)
    if q_status == "fail":
        warnings.append("question:fail")
        q_a, q_b = body_lo, body_lo  # empty span — treated as 0 tokens
    elif q_status != "exact":
        warnings.append(f"question:{q_status}")

    o_a, o_b, o_status = _find_substring_span(ids, processor, options_block,
                                              start=q_b, end=body_hi)
    if o_status == "fail":
        warnings.append("options:fail")
        o_a, o_b = q_b, q_b
    elif o_status != "exact":
        warnings.append(f"options:{o_status}")

    i_a, i_b, i_status = _find_substring_span(ids, processor, instruction_block,
                                              start=o_b, end=body_hi)
    if i_status == "fail":
        warnings.append("instruction:fail")
        i_a, i_b = o_b, o_b
    elif i_status != "exact":
        warnings.append(f"instruction:{i_status}")

    # 7. answer_prefix = [last_im_end, seq_len)  -- includes <|im_end|> + <|im_start|>assistant\n
    answer_prefix_span = (last_im_end, seq_len)

    return {
        "header": header_span,
        "visual_wrapper": visual_wrapper_span,
        "question": (q_a, q_b),
        "options": (o_a, o_b),
        "instruction": (i_a, i_b),
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


class TextKResidualCache:
    """`DynamicCache` subclass that records per-text-position INT4 K residual.

    Storage stays BF16; on each `update()` we additionally compute
        ||K[t, L, h, :] - Q_int4(K[t, L, h, :])||_2 / ||K[t, L, h, :]||_2
    averaged across all KV-heads (h) and accumulated across all layers (L).

    Final result: tensor of shape [seq_len] with mean residual norm per position.
    Only the positions in `text_position_set` have non-zero accumulated values.
    Positions outside text_position_set are returned as 0 (or NaN if you query
    `relative=True` and the position has zero K-norm).
    """

    def __init__(self, num_layers: int, seq_len: int):
        from transformers.cache_utils import DynamicCache
        self._inner = DynamicCache()
        self.num_layers = num_layers
        self.seq_len = seq_len
        # accumulator shape [seq_len], summed across (L, h) then divided by L*h at the end
        self._sum = torch.zeros(seq_len, dtype=torch.float32)
        self._count = 0  # number of (L, h) increments per position; equals num_layers * num_kv_heads after one prefill

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # key_states: [B, H_kv, T_new, D] -- in prefill T_new == seq_len; on later
        # decode steps T_new == 1.
        from fake_quant_kv_cache import fake_quantize_kv
        with torch.no_grad():
            B, H, T, D = key_states.shape
            # We expect prefill (T == seq_len). If not, just no-op (decode steps).
            if T <= 0:
                return self._inner.update(key_states, value_states, layer_idx, cache_kwargs)
            Kq = fake_quantize_kv(key_states, 4)
            # Per-position residual norm: ||K[b=0, h, t, :] - Kq[b=0, h, t, :]||_2 / ||K[b=0, h, t, :]||_2
            num = (key_states - Kq).pow(2).sum(dim=-1).sqrt()  # [B, H, T]
            den = key_states.pow(2).sum(dim=-1).sqrt().clamp_min(1e-8)  # [B, H, T]
            rel = (num / den)[0]  # [H, T]
            # Sum over heads and accumulate by position. Append to self._sum.
            per_pos = rel.sum(dim=0).float().cpu()  # [T]
            # In prefill T may equal seq_len (or smaller if cache_kwargs split). Append.
            cache_offset = self._inner.get_seq_length(layer_idx)
            end = cache_offset + T
            if end <= self.seq_len:
                self._sum[cache_offset:end] += per_pos
                self._count += int(H)
        return self._inner.update(key_states, value_states, layer_idx, cache_kwargs)

    def __getattr__(self, name):
        # Forward all attribute access to the inner cache (DynamicCache)
        return getattr(self._inner, name)

    def get_seq_length(self, *args, **kwargs):
        return self._inner.get_seq_length(*args, **kwargs)

    def finalize(self) -> torch.Tensor:
        """Return [seq_len] tensor of mean per-position residual norm.

        Mean is taken over (L * h) increments (so num_layers * num_kv_heads).
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
