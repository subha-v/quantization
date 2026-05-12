"""Page-layout builder for Exp P (query-adaptive page routing on MM-NIAH).

A *page* is a contiguous chunk of input-token positions with a single role and
modality. For Qwen2.5-VL with MM-NIAH retrieval-image inputs the sequence layout is:

  [0, header_end)                   # system prompt + user turn opening
  text page
  <|vision_start|> visual tokens (in-context image 1) <|vision_end|>
  text page (between img 1 and img 2)
  <|vision_start|> visual tokens (in-context image 2) <|vision_end|>
  ... (repeated for each in-context image)
  text page (question + "Choices:")
  "A. " <|vision_start|> visual tokens (choice A) <|vision_end|> "\n"
  "B. " <|vision_start|> visual tokens (choice B) <|vision_end|> "\n"
  ... C, D
  text page (instruction + answer prefix)

Each `<|vision_start|>...<|vision_end|>` pair becomes one *visual page*. Visual
pages are tagged as either `in_context_image` (routable by Quest; can contain
the needle) or `choice_image` (always-on; the model needs to see all 4 choices
to compare). Text pages are always-on too.

The page layout is computed once per item, from the tokenized `input_ids`,
before the prefill forward. It is then passed to `PageAwareFakeQuantKVCache`
and to the SDPA wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from visual_tokens import _resolve_special_id


PageKind = str  # "text" | "in_context_image" | "choice_image"


@dataclass
class Page:
    """One page covering [start, end) in absolute token coordinates."""
    start: int
    end: int
    kind: PageKind
    page_idx: int                    # global page index (0..n_pages-1)
    image_idx: Optional[int] = None  # for image pages: 0..n-1 within its image group
    is_needle: bool = False          # True for the in-context image that is the needle
    is_routable: bool = False        # True only for in_context_image pages

    @property
    def n_tokens(self) -> int:
        return self.end - self.start


@dataclass
class PageLayout:
    pages: list[Page]
    seq_len: int
    needle_page_idx: Optional[int] = None   # global page index of the needle page
    n_in_context_images: int = 0
    n_choice_images: int = 0
    _warnings: list[str] = field(default_factory=list)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    def routable_pages(self) -> list[Page]:
        return [p for p in self.pages if p.is_routable]

    def visual_pages(self) -> list[Page]:
        return [p for p in self.pages if p.kind in ("in_context_image", "choice_image")]

    def assignment_array(self) -> torch.Tensor:
        """Return a [seq_len] int tensor mapping each position -> page_idx.
        Positions outside any page (should not happen for well-formed sequences)
        map to -1.
        """
        arr = torch.full((self.seq_len,), -1, dtype=torch.long)
        for p in self.pages:
            arr[p.start:p.end] = p.page_idx
        return arr


# ---------------- vision span enumeration ----------------

def find_all_visual_spans(input_ids: torch.Tensor, processor) -> list[tuple[int, int]]:
    """Return [(v_start, v_end), ...] — one half-open range per <|vision_start|>...<|vision_end|>.

    v_start is the position immediately AFTER <|vision_start|>; v_end is the position
    of <|vision_end|> (exclusive of the marker itself). Matches the single-span
    `find_visual_token_span` semantics from visual_tokens.py but enumerates every
    span instead of just the first.
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
    spans: list[tuple[int, int]] = []
    i = 0
    N = len(ids)
    while i < N:
        if ids[i] == vstart_id:
            try:
                j = ids.index(vend_id, i + 1)
            except ValueError as e:
                raise RuntimeError(f"<|vision_start|> at {i} has no matching <|vision_end|>") from e
            spans.append((i + 1, j))  # exclude markers themselves
            i = j + 1
        else:
            i += 1
    return spans


# ---------------- layout builder ----------------

def build_page_layout(input_ids: torch.Tensor,
                      processor,
                      n_in_context_images: int,
                      n_choice_images: int,
                      needle_idx_in_images: int) -> PageLayout:
    """Build a PageLayout for an MM-NIAH retrieval-image item.

    n_in_context_images: number of in-context (haystack) images
    n_choice_images: typically 4 (A..D)
    needle_idx_in_images: which of the n_in_context_images is the needle

    The first n_in_context_images visual spans are tagged in_context_image;
    the last n_choice_images visual spans are tagged choice_image. The text
    regions in between become text pages.
    """
    if input_ids.dim() == 2:
        ids = input_ids[0]
    else:
        ids = input_ids
    seq_len = int(ids.shape[0])

    spans = find_all_visual_spans(input_ids, processor)
    expected = n_in_context_images + n_choice_images
    warnings: list[str] = []
    if len(spans) != expected:
        warnings.append(
            f"visual span count mismatch: got {len(spans)} expected {expected} "
            f"(n_in_context_images={n_in_context_images}, n_choice_images={n_choice_images}). "
            f"Treating first {n_in_context_images} as in_context, rest as choice."
        )

    n_visual = len(spans)
    # First k spans are in-context images, remaining are choices.
    k_in_context = min(n_in_context_images, n_visual)

    pages: list[Page] = []
    cursor = 0
    page_idx = 0
    needle_page_idx: Optional[int] = None

    for span_i, (v_start, v_end) in enumerate(spans):
        # Text region from cursor up to <|vision_start|> (which is at v_start - 1).
        # We include the <|vision_start|> marker in the text page (cheap, harmless).
        text_lo = cursor
        text_hi = v_start  # exclude the actual visual tokens
        # Actually include the <|vision_start|> marker token (position v_start - 1)
        # inside the text page; everything in [text_lo, text_hi) is text.
        if text_hi > text_lo:
            pages.append(Page(
                start=text_lo, end=text_hi, kind="text",
                page_idx=page_idx, image_idx=None, is_needle=False, is_routable=False,
            ))
            page_idx += 1

        if span_i < k_in_context:
            kind = "in_context_image"
            image_idx = span_i
            is_needle = (span_i == needle_idx_in_images)
            is_routable = True
        else:
            kind = "choice_image"
            image_idx = span_i - k_in_context
            is_needle = False
            is_routable = False  # choice images are always-on
        pages.append(Page(
            start=v_start, end=v_end, kind=kind,
            page_idx=page_idx, image_idx=image_idx,
            is_needle=is_needle, is_routable=is_routable,
        ))
        if is_needle:
            needle_page_idx = page_idx
        page_idx += 1
        cursor = v_end  # exclude <|vision_end|> from text — include it in next text page

    # Trailing text after the last visual span (question instruction + answer prefix)
    if cursor < seq_len:
        pages.append(Page(
            start=cursor, end=seq_len, kind="text",
            page_idx=page_idx, image_idx=None, is_needle=False, is_routable=False,
        ))
        page_idx += 1

    if needle_page_idx is None and needle_idx_in_images >= 0:
        warnings.append(f"needle_idx_in_images={needle_idx_in_images} could not be located")

    return PageLayout(
        pages=pages, seq_len=seq_len,
        needle_page_idx=needle_page_idx,
        n_in_context_images=k_in_context,
        n_choice_images=n_visual - k_in_context,
        _warnings=warnings,
    )


# ---------------- sanity helpers ----------------

def coverage_ok(layout: PageLayout) -> bool:
    """Return True if pages exactly cover [0, seq_len) with no overlap or gap."""
    cursor = 0
    for p in sorted(layout.pages, key=lambda x: x.start):
        if p.start != cursor:
            return False
        cursor = p.end
    return cursor == layout.seq_len


def page_summary(layout: PageLayout) -> str:
    lines = [f"PageLayout n_pages={layout.n_pages} seq_len={layout.seq_len} "
             f"in_context_imgs={layout.n_in_context_images} choice_imgs={layout.n_choice_images} "
             f"needle_page_idx={layout.needle_page_idx}"]
    for p in layout.pages:
        tag = ""
        if p.is_needle:
            tag = " [NEEDLE]"
        elif p.is_routable:
            tag = " [routable]"
        lines.append(f"  p{p.page_idx:3d} [{p.start:5d},{p.end:5d}) "
                     f"n={p.n_tokens:5d} {p.kind:18s} image_idx={p.image_idx}{tag}")
    if layout._warnings:
        lines.append("  warnings: " + "; ".join(layout._warnings))
    return "\n".join(lines)
