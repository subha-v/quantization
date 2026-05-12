"""Quest-style per-page envelope + upper-bound scoring for Exp P.

Given quantized K of shape [B, H_kv, T, D] for one decoder layer's chunk and a
PageLayout, computes per-page (k_min, k_max) envelopes per (KV head, channel).

At decode time (= the answer-prediction row in prefill, for first-token MCQ),
the Quest upper-bound score for one page against the last query row is:

    score(q, page) = sum_d max(q[d] * k_min_p[d], q[d] * k_max_p[d])

This is the canonical Quest upper bound (Tang et al. 2024, eq. 1): for any
single token inside `page`, q · k ≤ score(q, page). Pages whose score is in
the top-K are the ones the query "must read"; the rest can be skipped or
quantized cheaper.

GQA: Qwen2.5-VL has 28 Q heads sharing 4 KV heads (group of 7). Envelopes are
per-KV-head. To aggregate Q-head scores onto pages, we sum (default) across the
group; max is an option the smoke test compares against.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from page_layout import PageLayout


# ---------------- envelope computation ----------------

def compute_page_envelope(K: torch.Tensor, layout: PageLayout) -> torch.Tensor:
    """Compute per-page K min/max envelope across the chunk's tokens.

    K: [B, H_kv, T, D]  — usually B=1 during prefill of a single MCQ item.
    Returns: envelope[H_kv, n_pages, D, 2] where [..., 0] = min, [..., 1] = max,
    in float32 on CPU (small footprint per layer).

    Pages that fall partly inside the chunk are handled by clamping the page
    range to [cache_offset, cache_offset + T). During prefill the chunk *is*
    the full sequence so this reduces to a simple slice per page.
    """
    assert K.dim() == 4 and K.shape[0] == 1, f"unexpected K shape {K.shape}"
    H_kv, T, D = K.shape[1], K.shape[2], K.shape[3]
    if T == 0:
        return torch.empty(H_kv, layout.n_pages, D, 2, dtype=torch.float32)

    # Detached float view to keep envelopes in fp32 regardless of K dtype.
    Kf = K[0].detach().float()  # [H_kv, T, D]

    env = torch.empty(H_kv, layout.n_pages, D, 2, dtype=torch.float32)
    # Default fill for pages that don't intersect this chunk: large positive min,
    # large negative max so they won't dominate top-K scoring against any q.
    env[..., 0] = float("inf")
    env[..., 1] = float("-inf")

    for p in layout.pages:
        s = max(0, p.start)
        e = min(T, p.end)
        if s >= e:
            continue
        slc = Kf[:, s:e, :]  # [H_kv, n_tok_in_page, D]
        env[:, p.page_idx, :, 0] = slc.amin(dim=1)
        env[:, p.page_idx, :, 1] = slc.amax(dim=1)
    return env


# ---------------- quest score ----------------

def quest_scores_for_layer(q_last: torch.Tensor,
                           envelope: torch.Tensor,
                           gqa_group: int,
                           aggregate: str = "sum") -> torch.Tensor:
    """Compute per-page Quest upper-bound score against the last query row.

    q_last:    [B, H_q, 1, D]  — typically B=1
    envelope:  [H_kv, n_pages, D, 2]
    gqa_group: H_q // H_kv (=7 for Qwen2.5-VL-7B)
    aggregate: "sum" or "max" — how to combine per-Q-head scores onto pages

    Returns: [n_pages] float32 tensor.
    """
    assert q_last.dim() == 4 and q_last.shape[0] == 1 and q_last.shape[2] == 1
    H_q, D = q_last.shape[1], q_last.shape[3]
    H_kv, n_pages, _, _ = envelope.shape
    assert H_q == H_kv * gqa_group, (
        f"q_heads={H_q} != H_kv*group={H_kv * gqa_group}"
    )

    q = q_last[0, :, 0, :].detach().float()  # [H_q, D]

    # Group each Q-head with its corresponding KV-head envelope.
    # env_per_q: [H_q, n_pages, D, 2] via repeat_interleave along H_kv axis.
    env_per_q = envelope.to(q.device).repeat_interleave(gqa_group, dim=0)  # [H_q, n_pages, D, 2]

    # Score: sum_d max(q[h, d] * env_per_q[h, p, d, 0], q[h, d] * env_per_q[h, p, d, 1])
    q_b = q.view(H_q, 1, D, 1)                            # [H_q, 1, D, 1]
    prods = q_b * env_per_q                               # [H_q, n_pages, D, 2]
    per_d_max = prods.amax(dim=-1)                        # [H_q, n_pages, D]
    per_page_per_qhead = per_d_max.sum(dim=-1)            # [H_q, n_pages]

    if aggregate == "sum":
        return per_page_per_qhead.sum(dim=0)              # [n_pages]
    elif aggregate == "max":
        return per_page_per_qhead.amax(dim=0)             # [n_pages]
    else:
        raise ValueError(f"unknown aggregate={aggregate!r}")


# ---------------- routable selection ----------------

@dataclass
class RoutingDecision:
    """Page-level routing outcome for one layer."""
    active_routable_pages: list[int]   # global page indices kept active
    cold_routable_pages: list[int]     # global page indices masked / cold
    scores: Optional[torch.Tensor]     # [n_pages] Quest scores, or None for random/oracle
    policy: str
    needle_in_active: bool             # True if the routing decision keeps the needle


def select_active_pages(layout: PageLayout,
                        scores: Optional[torch.Tensor],
                        policy: str,
                        budget_fraction: Optional[float],
                        rng: Optional[torch.Generator] = None) -> RoutingDecision:
    """Return which routable (in-context-image) pages to keep active.

    layout, scores: from compute_page_envelope + quest_scores_for_layer.
    policy:
      - "quest_top": top-(budget_fraction * n_routable) by Quest score, ceil
      - "random_top": random sample of the same budget (rng MUST be supplied
                      for reproducibility)
      - "oracle_needle": needle forced into active set, remaining (K-1) slots
                        filled by top-Quest at the same budget_fraction. This
                        gives a budget-matched oracle upper bound for Quest.
      - "none": keep all routable pages active (no masking)
    budget_fraction: e.g. 0.25 (top-25%) or 0.5 (top-50%). Required for all
                     policies except "none".
    """
    routable = layout.routable_pages()
    routable_idx = [p.page_idx for p in routable]
    n = len(routable)

    if policy in ("none", "all_hot") or n == 0:
        return RoutingDecision(
            active_routable_pages=routable_idx[:],
            cold_routable_pages=[],
            scores=scores, policy=policy,
            needle_in_active=(layout.needle_page_idx in routable_idx),
        )

    if policy == "role_only":
        # FormatBook RoleOnly: zero in-context pages hot, all cold.
        # (Text + choice pages are always-on and stay at hot precision via
        # the cache; only in-context routable pages downgrade.)
        return RoutingDecision(
            active_routable_pages=[],
            cold_routable_pages=routable_idx[:],
            scores=scores, policy=policy,
            needle_in_active=False,
        )

    # oracle_needle_only ignores budget_fraction (always returns just the needle);
    # other selection policies require a budget.
    if policy != "oracle_needle_only" and budget_fraction is None:
        raise ValueError(f"policy={policy} requires budget_fraction")
    import math
    if budget_fraction is not None:
        k = max(1, math.ceil(budget_fraction * n))
        k = min(k, n)
    else:
        k = 1  # oracle_needle_only never reads k; satisfy the type

    if policy == "quest_top":
        if scores is None:
            raise ValueError("quest_top requires scores")
        routable_scores = scores[torch.tensor(routable_idx, dtype=torch.long, device=scores.device)]
        topk = torch.topk(routable_scores, k=k, largest=True).indices.tolist()
        active = sorted(routable_idx[i] for i in topk)
    elif policy == "random_top":
        # Reproducibility requires an explicit generator; default-RNG sampling
        # has bitten paired comparisons before.
        if rng is None:
            raise ValueError(
                "random_top requires an explicit torch.Generator for reproducibility"
            )
        perm = torch.randperm(n, generator=rng).tolist()
        active = sorted(routable_idx[i] for i in perm[:k])
    elif policy == "oracle_needle":
        # Budget-matched oracle: force the needle into the active set, then
        # fill the remaining (k - 1) slots with the top Quest scorers among
        # the OTHER routable pages.
        needle_idx = layout.needle_page_idx
        other_idx = [i for i in routable_idx if i != needle_idx]
        n_fill = max(0, k - 1) if needle_idx is not None else k
        if n_fill > 0 and other_idx:
            if scores is None:
                # No Quest scores → fall back to first-K-1 in declared order
                # (deterministic; we never hit this with the wrapper plumbing).
                fill = other_idx[:n_fill]
            else:
                other_scores = scores[
                    torch.tensor(other_idx, dtype=torch.long, device=scores.device)
                ]
                topk = torch.topk(
                    other_scores, k=min(n_fill, len(other_idx)), largest=True
                ).indices.tolist()
                fill = [other_idx[i] for i in topk]
        else:
            fill = []
        active = sorted(([needle_idx] if needle_idx is not None else []) + fill)
    elif policy == "oracle_needle_only":
        # Strict oracle: needle ONLY in active set, regardless of budget.
        # Tests whether the needle page alone (+ text + choice images, which
        # are always-on) is enough to answer.
        needle_idx = layout.needle_page_idx
        active = [needle_idx] if needle_idx is not None else []
    else:
        raise ValueError(f"unknown policy={policy!r}")

    active_set = set(active)
    cold = [i for i in routable_idx if i not in active_set]
    return RoutingDecision(
        active_routable_pages=active, cold_routable_pages=cold,
        scores=scores, policy=policy,
        needle_in_active=(layout.needle_page_idx in active_set),
    )


def needle_rank(scores: torch.Tensor, layout: PageLayout) -> Optional[int]:
    """Return the rank (0 = top) of the needle page among routable pages
    according to `scores`. Returns None if no needle is set.
    """
    if layout.needle_page_idx is None:
        return None
    routable = layout.routable_pages()
    if not routable:
        return None
    routable_idx = [p.page_idx for p in routable]
    routable_scores = scores[torch.tensor(routable_idx, dtype=torch.long, device=scores.device)]
    # Sort descending; find needle's position
    order = torch.argsort(routable_scores, descending=True).tolist()
    for rank, i in enumerate(order):
        if routable_idx[i] == layout.needle_page_idx:
            return rank
    return None
