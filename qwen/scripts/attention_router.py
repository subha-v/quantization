"""SDPA-level page-routing monkey-patch for Exp P.

`page_routing_sdpa_context(cache, ...)` is a context manager that replaces
`torch.nn.functional.scaled_dot_product_attention` with a wrapper which:

  1. Reads `cache.most_recent_envelope` and `cache.page_layout` (written by
     `PageAwareFakeQuantKVCache.update()` immediately before SDPA fires within
     the same layer's attention forward).
  2. Computes Quest upper-bound scores per page against the LAST query row of
     the prefill (= the answer-prediction row for first-token MCQ with
     `max_new_tokens=1`).
  3. Selects active vs cold visual pages per the routing policy.
  4. Modifies attention so the cold pages either:
       (a) "sparse" routes: get -inf in the last query row's attn_mask (P3/P4/P5)
       (b) "formatbook" routes: their K is re-quantized to F4 in place (P6)
     For routes (a), the LAST ROW of attention is recomputed manually with the
     page mask, then patched into the full SDPA output. For route (b), K is
     mutated before SDPA is called normally.

Why the last-row trick: in a decoder forward, attn output for rows < T_q - 1
feeds K/V at those positions to the NEXT layer. Modifying those rows would
corrupt downstream layers' K, which would corrupt the answer-prediction at the
final layer. Restricting the page mask to row T_q-1 only affects the
answer-prediction logits at the *current* layer, leaving all upstream
inter-position interactions intact across layers.

The wrapper is a no-op for route_policy="none" — used by P0 (BF16) and P2 (dense
J12) so envelopes are still captured but no masking happens.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from page_envelope_cache import PageAwareFakeQuantKVCache
from page_layout import PageLayout
from quest_scorer import (
    RoutingDecision, needle_rank, quest_scores_for_layer, select_active_pages,
)


# ---------------- F4 re-quantization (for FormatBook cold downgrade) ----------------

def _f4_per_channel_seq(K: torch.Tensor) -> torch.Tensor:
    """KIVI per-channel-seq INT4: per-(B, H, channel) scale from seq-dim max-abs.

    Applied per-page during FormatBook to "downgrade" already-J12-quantized K
    for cold pages. The result is BF16 values on F4's coarser grid; this
    approximates the effect of having stored that page at F4 from the start.
    """
    qmax = 7.0  # 2^(4-1) - 1
    Kf = K.float()
    amax = Kf.abs().amax(dim=-2, keepdim=True)  # over seq dim
    s = amax.clamp_min(1e-8) / qmax
    return ((Kf / s).round().clamp(-qmax, qmax) * s).to(K.dtype)


# ---------------- routing policy parsing ----------------

@dataclass
class RoutePolicy:
    """Parsed routing policy.

    name (sparse: mask cold pages with -inf on last query row):
      "none"               -> no-op (P0/P1/P2)
      "quest_sparse"       -> Quest top-K (P3)
      "random_sparse"      -> random top-K (P4; per-(item, condition) seeded)
      "oracle_sparse"      -> needle + Quest top-(K-1) fill at matched budget (P5)
      "oracle_needle_only" -> needle ONLY, no fill (P5_only; strict)

    name (formatbook: cold pages stay but downgrade to F4):
      "formatbook_quest"   -> Quest top-K hot pages get J12, others F4 (P6)
      "formatbook_random"  -> Random top-K hot pages get J12, others F4 (P6R)
      "formatbook_oracle"  -> Oracle (needle + Quest fill) hot, others F4 (P6O)

    budget_fraction: required for everything except "none" and
                     "oracle_needle_only".
    gqa_aggregate:   "sum" or "max" — Q-head aggregation for Quest scoring.
    """
    name: str
    budget_fraction: Optional[float] = None
    gqa_aggregate: str = "sum"

    def is_sparse(self) -> bool:
        return self.name in (
            "quest_sparse", "random_sparse", "oracle_sparse", "oracle_needle_only",
        )

    def is_formatbook(self) -> bool:
        return self.name in (
            "formatbook_quest", "formatbook_random", "formatbook_oracle",
        )

    def selection_policy(self) -> str:
        return {
            "quest_sparse": "quest_top",
            "random_sparse": "random_top",
            "oracle_sparse": "oracle_needle",
            "oracle_needle_only": "oracle_needle_only",
            "formatbook_quest": "quest_top",
            "formatbook_random": "random_top",
            "formatbook_oracle": "oracle_needle",
        }[self.name]


# ---------------- core last-row computation ----------------

def _build_last_row_mask(cold_token_mask: torch.Tensor,
                          attn_mask: Optional[torch.Tensor],
                          T_k: int, device, dtype) -> torch.Tensor:
    """Build a [1, 1, 1, T_k] additive mask for the last query row.

    Combines:
      - `cold_token_mask`: bool [T_k], True at cold-page tokens → -inf
      - `attn_mask`: optional original mask (2D/3D/4D), last-row slice added in
    """
    out = torch.zeros(1, 1, 1, T_k, dtype=dtype, device=device)
    if cold_token_mask is not None:
        m = cold_token_mask.to(device, non_blocking=True)
        out = out.masked_fill(m.view(1, 1, 1, T_k), float("-inf"))
    if attn_mask is not None:
        if attn_mask.dim() == 4:
            last_attn = attn_mask[..., -1:, :]                  # [B, H, 1, T_k]
        elif attn_mask.dim() == 3:
            last_attn = attn_mask[:, -1:, :].unsqueeze(1)       # [B, 1, 1, T_k]
        elif attn_mask.dim() == 2:
            last_attn = attn_mask[-1:, :].unsqueeze(0).unsqueeze(0)  # [1, 1, 1, T_k]
        else:
            last_attn = None
        if last_attn is not None:
            if last_attn.dtype == torch.bool:
                last_attn = torch.where(last_attn,
                                        torch.zeros_like(out),
                                        torch.full_like(out, float("-inf")))
            out = out + last_attn.to(out.dtype).to(out.device)
    return out


# ---------------- token-mask builder ----------------

def build_cold_token_mask(layout: PageLayout, cold_pages: list[int],
                          seq_len_keys: int) -> torch.Tensor:
    """[T_k] bool mask, True at every token position inside any cold page.

    seq_len_keys is K's length; for prefill it equals seq_len. We clamp page
    ranges to seq_len_keys defensively.
    """
    mask = torch.zeros(seq_len_keys, dtype=torch.bool)
    cold_set = set(cold_pages)
    for p in layout.pages:
        if p.page_idx in cold_set:
            s = max(0, p.start)
            e = min(seq_len_keys, p.end)
            if s < e:
                mask[s:e] = True
    return mask


# ---------------- the context manager ----------------

@contextmanager
def page_routing_sdpa_context(cache: PageAwareFakeQuantKVCache,
                              policy: RoutePolicy,
                              record_routing: bool = True):
    """Patch F.scaled_dot_product_attention with a page-aware wrapper.

    Usage:
        with page_routing_sdpa_context(cache, RoutePolicy("quest_sparse", 0.25)):
            out = model.generate(..., past_key_values=cache, max_new_tokens=1, ...)

    For policy.name == "none" the patch is still installed but is a pure pass-through
    (so we can verify zero perturbation on P0 with the wrapper enabled).
    """
    original_sdpa = F.scaled_dot_product_attention

    def patched(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, **kwargs):
        # Use getattr defaults so a non-page-aware cache (e.g. plain
        # FakeQuantKVCache for P1) or pre-LLM-layer SDPA calls (e.g. vision
        # encoder, which fires before any language-model decoder layer has set
        # `most_recent_envelope`) cleanly pass through.
        layout = getattr(cache, "page_layout", None)
        env = getattr(cache, "most_recent_envelope", None)
        layer_idx = getattr(cache, "most_recent_layer_idx", None)
        if layout is None or env is None or policy.name == "none":
            return original_sdpa(query, key, value, attn_mask=attn_mask,
                                 dropout_p=dropout_p, is_causal=is_causal,
                                 scale=scale, **kwargs)

        T_q = query.shape[-2]
        T_k = key.shape[-2]
        if T_q <= 0:
            return original_sdpa(query, key, value, attn_mask=attn_mask,
                                 dropout_p=dropout_p, is_causal=is_causal,
                                 scale=scale, **kwargs)

        H_q, D = query.shape[1], query.shape[-1]
        H_kv = env.shape[0]
        gqa_group = H_q // H_kv

        # 1) Always compute Quest scores when we have an envelope and a non-"none"
        # policy. Oracle needs them for top-(K-1) fill; random doesn't strictly
        # need them but we log them for free diagnostics anyway.
        q_last = query[:, :, -1:, :].detach()
        scores = quest_scores_for_layer(q_last, env,
                                        gqa_group=gqa_group,
                                        aggregate=policy.gqa_aggregate)

        # 2) Selection (with reproducible RNG for random_top).
        selection_policy = policy.selection_policy()
        rng = getattr(cache, "rng", None)
        decision = select_active_pages(
            layout, scores, selection_policy,
            budget_fraction=policy.budget_fraction,
            rng=rng,
        )

        if record_routing and layer_idx is not None:
            n_rank = needle_rank(scores, layout) if scores is not None else None
            cache.routing_log[layer_idx] = {
                "policy": policy.name,
                "active_routable_pages": decision.active_routable_pages,
                "cold_routable_pages": decision.cold_routable_pages,
                "needle_in_active": decision.needle_in_active,
                "needle_rank": n_rank,
                "scores_top5": (
                    sorted(scores.tolist(), reverse=True)[:5] if scores is not None else None
                ),
            }

        cold_token_mask = build_cold_token_mask(layout, decision.cold_routable_pages, T_k)

        if policy.is_formatbook():
            # Downgrade cold pages' K in place (F4 re-quantization). All pages
            # still participate in attention; cold pages just have noisier K.
            if cold_token_mask.any():
                key = key.clone()
                # Apply F4 per page slice (per-page-seq scale).
                cold_set = set(decision.cold_routable_pages)
                for p in layout.pages:
                    if p.page_idx in cold_set:
                        s = max(0, p.start)
                        e = min(T_k, p.end)
                        if s < e:
                            key[:, :, s:e, :] = _f4_per_channel_seq(key[:, :, s:e, :])
            return original_sdpa(query, key, value, attn_mask=attn_mask,
                                 dropout_p=dropout_p, is_causal=is_causal,
                                 scale=scale, **kwargs)

        # Sparse routes: compute full SDPA causally, then overwrite the last row
        # with a separately-masked SDPA call on q_last only. We use the original
        # SDPA (not our patched copy) so GQA / scaling / kernel dispatch are
        # handled identically to the non-routed call — preserving the K/V
        # contributions of non-last rows to all upstream layers.
        full = original_sdpa(query, key, value, attn_mask=attn_mask,
                             dropout_p=dropout_p, is_causal=is_causal,
                             scale=scale, **kwargs)
        if not cold_token_mask.any():
            return full
        last_mask = _build_last_row_mask(cold_token_mask, attn_mask,
                                          T_k=T_k, device=query.device,
                                          dtype=query.dtype)
        q_last = query[:, :, -1:, :]
        last_row = original_sdpa(q_last, key, value, attn_mask=last_mask,
                                  dropout_p=dropout_p, is_causal=False,
                                  scale=scale, **kwargs)
        full = full.clone()
        full[:, :, -1:, :] = last_row
        return full

    # Install patch — also patch torch.functional and torch.scaled_dot_product_attention
    # if they're re-exported. PyTorch's transformers SDPA backend calls F.scaled_dot_product_attention
    # from torch.nn.functional, so we patch that one symbol.
    F.scaled_dot_product_attention = patched
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa
