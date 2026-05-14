"""Exp Q driver — FormatBook v2 on MM-NIAH.

Two slices supported via --task:
  retrieval-image (Slice A; multi-image filtered; Q0..Q11)
  reasoning-image (Slice B; R0..R8)

Hot format = F9 (BF16 sidecode); cold quantizer = F4 (default) or INT2 (Q10/Q11).
Equal-resolution context vs choices (default 336² each = 112,896 px).

Per-item JSONL row contains accuracy + routing diagnostics + per-page format
accounting: effective_k_bits, effective_kv_bits, f9_sidecode_token_fraction,
f9_sidecode_page_fraction, tokens_per_format, pages_per_format.

Conditions:
  Slice A (retrieval-image, --task retrieval-image):
    Q0   BF16 dense                                  (anchor)
    Q1   F4 dense                                    (anchor)
    Q2   F9 dense (BF16 sidecode)                    (anchor)
    Q3   RoleOnly FormatBook (no in-context hot)     (new)
    Q4   Quest  top-50% FormatBook
    Q5   Random top-50% FormatBook        seed 0
    Q6   Oracle top-50% FormatBook
    Q7   Quest  top-25% FormatBook
    Q8   Random top-25% FormatBook        seed 0
    Q9   Oracle top-25% FormatBook
    Q10  Quest  top-25% FormatBook, cold-K INT2 / V INT4   (stretch)
    Q11  Random top-25% FormatBook, cold-K INT2 / V INT4   (stretch, seed 0)

  Branch-conditional reseeds (orchestrator launches if rule fires):
    Q5_s1/Q5_s2, Q8_s1/Q8_s2, Q11_s1/Q11_s2

  Smoke-only:
    Q_allhot  FormatBook with budget=1.0 (every routable page hot;
              must match Q2 logits within 1e-4)

  Slice B (reasoning-image, --task reasoning-image):
    R0   BF16 dense
    R1   F4 dense
    R2   F9 dense
    R3   RoleOnly FormatBook
    R4   Quest  top-50% FormatBook
    R5   Random top-50% FormatBook        seed 0
    R6   Oracle top-50% FormatBook
    R7   Quest  top-25% FormatBook
    R8   Random top-25% FormatBook        seed 0
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from attention_router import RoutePolicy, page_routing_sdpa_context
from fake_quant_kv_cache import BitController, FakeQuantKVCache
from k_quantizers import build_f_conditions
from mm_niah_loader import (
    MMNiahItem, SUPPORTED_TASKS, answer_token_ids, filter_items, format_mcq_messages,
    load_all_items, load_split, make_split, save_split, split_file_for_task,
)
from page_envelope_cache import PageAwareFakeQuantKVCache
from page_layout import build_page_layout
from quest_scorer import needle_rank
from run_inference import load_model


SCRIPTS_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPTS_DIR.parent / "calibration"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"


# ---------------- condition spec ----------------

@dataclass
class CondSpec:
    name: str
    k_cfg_name: Optional[str]
    route: RoutePolicy
    use_page_cache: bool


HOT = "F9_KIVI_Outlier16"        # F9 with BF16 outlier sidecode — clean MM-NIAH hot anchor.
F4 = "F4_KIVI_PerChannelSeq"      # F4 dense anchor.

# Per-token K bits per page format (Exp R extends Exp Q with INT3 and FP8).
PAGE_K_BITS = {"F9": 5.50, "F4": 4.00, "INT3": 3.00, "INT2": 2.00, "FP8": 8.00}
# Per-token V bits. Default V is uniformly INT4 from FakeQuantKVCache.
# Cold-V INT2 is per-page (only when policy.cold_v_quantizer="int2").
V_BITS_DEFAULT = 4.00
V_BITS_COLD_INT2 = 2.00
V_BITS_BF16_DENSE = 16.00   # Q0/A0/B0/C0/D0: V is BF16 (cache bypassed entirely)

# Static matched-budget baselines for Exp R. Bits are computed in
# _page_k_bits_dense via S-name lookup since these are dense conditions.
S4_K_BITS = (16 * 4 + 124 * 4) / 128.0   # = 4.375
S8_K_BITS = (16 * 8 + 120 * 4) / 128.0   # = 4.750 (= F8)
S12_K_BITS = (16 * 12 + 116 * 4) / 128.0 # = 5.125
SJ_K_BITS = (8 * 16 + 4 * 112) / 128.0   # = 4.500 (J12 — INT8 sidecode)


def q_conditions_slice_a(include_int2: bool = True) -> list[CondSpec]:
    """Slice A primary conditions Q0..Q9 (+ Q10/Q11 stretch)."""
    conds = [
        CondSpec("Q0",  None, RoutePolicy("none"),                        False),
        CondSpec("Q1",  F4,   RoutePolicy("none"),                        False),
        CondSpec("Q2",  HOT,  RoutePolicy("none"),                        True),
        CondSpec("Q3",  HOT,  RoutePolicy("formatbook_role_only"),        True),
        CondSpec("Q4",  HOT,  RoutePolicy("formatbook_quest", 0.5),       True),
        CondSpec("Q5",  HOT,  RoutePolicy("formatbook_random", 0.5),      True),
        CondSpec("Q6",  HOT,  RoutePolicy("formatbook_oracle", 0.5),      True),
        CondSpec("Q7",  HOT,  RoutePolicy("formatbook_quest", 0.25),      True),
        CondSpec("Q8",  HOT,  RoutePolicy("formatbook_random", 0.25),     True),
        CondSpec("Q9",  HOT,  RoutePolicy("formatbook_oracle", 0.25),     True),
    ]
    if include_int2:
        conds.extend([
            CondSpec("Q10", HOT, RoutePolicy("formatbook_quest", 0.25, cold_quantizer="int2"), True),
            CondSpec("Q11", HOT, RoutePolicy("formatbook_random", 0.25, cold_quantizer="int2"), True),
        ])
    return conds


def q_conditions_reseed() -> list[CondSpec]:
    """Branch-conditional reseed conditions. Each one re-runs the seed-0
    random-FormatBook condition with the (item, cond.name) hash deriving a
    different sample (per attention_router.py:rng_seed plumbing)."""
    return [
        CondSpec("Q5_s1",  HOT, RoutePolicy("formatbook_random", 0.5),  True),
        CondSpec("Q5_s2",  HOT, RoutePolicy("formatbook_random", 0.5),  True),
        CondSpec("Q8_s1",  HOT, RoutePolicy("formatbook_random", 0.25), True),
        CondSpec("Q8_s2",  HOT, RoutePolicy("formatbook_random", 0.25), True),
        CondSpec("Q11_s1", HOT, RoutePolicy("formatbook_random", 0.25, cold_quantizer="int2"), True),
        CondSpec("Q11_s2", HOT, RoutePolicy("formatbook_random", 0.25, cold_quantizer="int2"), True),
    ]


def q_conditions_smoke_only() -> list[CondSpec]:
    """Smoke-only sanity conditions, never run on the full pool."""
    return [
        # FormatBook with all pages hot — must match Q2 F9 dense logits within 1e-4.
        CondSpec("Q_allhot", HOT, RoutePolicy("formatbook_all_hot"), True),
    ]


def r_conditions_slice_b() -> list[CondSpec]:
    """Slice B reasoning-image conditions R0..R8."""
    return [
        CondSpec("R0", None, RoutePolicy("none"),                     False),
        CondSpec("R1", F4,   RoutePolicy("none"),                     False),
        CondSpec("R2", HOT,  RoutePolicy("none"),                     True),
        CondSpec("R3", HOT,  RoutePolicy("formatbook_role_only"),     True),
        CondSpec("R4", HOT,  RoutePolicy("formatbook_quest", 0.5),    True),
        CondSpec("R5", HOT,  RoutePolicy("formatbook_random", 0.5),   True),
        CondSpec("R6", HOT,  RoutePolicy("formatbook_oracle", 0.5),   True),
        CondSpec("R7", HOT,  RoutePolicy("formatbook_quest", 0.25),   True),
        CondSpec("R8", HOT,  RoutePolicy("formatbook_random", 0.25),  True),
    ]


# ============================================================
# Exp R condition builders
# ============================================================

def c_conditions_allvisual() -> list[CondSpec]:
    """Exp R Sub-experiment C — AllVisual routing.

    All FormatBook routes here use token_budgeted=True so the 25% budget
    applies to *tokens*, not page-count. Requires
    --include-choice-routing in the driver so build_page_layout marks
    choice pages as routable too.
    """
    return [
        CondSpec("C0",  None, RoutePolicy("none"),                        False),
        CondSpec("C1",  F4,   RoutePolicy("none"),                        False),
        CondSpec("C2",  HOT,  RoutePolicy("none"),                        True),
        CondSpec("C3",  HOT,  RoutePolicy("formatbook_role_only"),        True),
        CondSpec("C3b", HOT,  RoutePolicy("formatbook_choice_only"),      True),
        CondSpec("C4",  HOT,  RoutePolicy("formatbook_quest_allvisual",  0.25, token_budgeted=True), True),
        CondSpec("C5",  HOT,  RoutePolicy("formatbook_random_allvisual", 0.25, token_budgeted=True), True),
        CondSpec("C6",  HOT,  RoutePolicy("formatbook_oracle_allvisual", 0.25, token_budgeted=True), True),
        CondSpec("C7",  HOT,  RoutePolicy("formatbook_split_quest",      0.25, token_budgeted=True), True),
        CondSpec("C8",  HOT,  RoutePolicy("formatbook_split_random",     0.25, token_budgeted=True), True),
    ]


def s_conditions_static_baselines() -> list[CondSpec]:
    """Exp R static matched-budget baselines for C.
    SJ alias = J12_F9_INT8side.
    """
    return [
        CondSpec("S4",  "S4_Outlier4_BF16side",  RoutePolicy("none"), True),
        CondSpec("S8",  "F8_KIVI_Outlier8",      RoutePolicy("none"), True),
        CondSpec("S12", "S12_Outlier12_BF16side", RoutePolicy("none"), True),
        CondSpec("SJ",  "J12_F9_INT8side",        RoutePolicy("none"), True),
    ]


def t_conditions_main() -> list[CondSpec]:
    """Exp T-mini main conditions T0..T16.

    Identical set runs on retrieval-image (Phase 1) and reasoning-image (Phase 2).
    All conditions are dense (no FormatBook routing). T0 is BF16 (no cache),
    T1-T16 use PageAwareFakeQuantKVCache for per-page bit accounting.

      T0  BF16 dense                                      (anchor ceiling)
      T1  Global-F4                                       (broken 4-bit floor)
      T2  F9 top-16 BF16 sidecode                         (strong baseline 4.75 KV bits)
      T3  SJ top-16 INT8 sidecode                         (sidecode anchor 4.25)
      T4  S4 top-16 INT7 sidecode                         (lowest useful 4.1875)
      T5  TextVisualLocal-F4                              (coarse modality-local control)
      T6  TokenBlockLocal-F4 (16 segments)                (no-page-structure control)
      T7  RandomPageLocal-F4                              (same n_pages, shuffled boundaries)
      T8  PageLocal-F4                                    (MAIN hypothesis)
      T9  ImageOnlyLocal-F4                               (image-side ablation)
      T10 TextOnlyLocal-F4                                (text-side ablation)
      T11 PageSentinel-1 (Global-F4 base)                 (minimal image anchor)
      T12 PageSentinel-4 (Global-F4 base)                 (stronger image anchor)
      T13 RandomSentinel-4 (Global-F4 base)               (position-control)
      T14 LastSentinel-4 (Global-F4 base)                 (image-end vs start)
      T15 TextSentinel-4 (Global-F4 base)                 (text-boundary control)
      T16 PageLocal-F4 + PageSentinel-4 (combined)        (combined method)
    """
    return [
        CondSpec("T0",  None,                              RoutePolicy("none"), False),
        CondSpec("T1",  F4,                                RoutePolicy("none"), True),
        CondSpec("T2",  HOT,                               RoutePolicy("none"), True),
        CondSpec("T3",  "J12_F9_INT8side",                 RoutePolicy("none"), True),
        CondSpec("T4",  "SL_Outlier16_INT7side",           RoutePolicy("none"), True),
        CondSpec("T5",  "F5_KIVI_TextVisualSplit",         RoutePolicy("none"), True),
        CondSpec("T6",  "T6_TokenBlock16_F4",              RoutePolicy("none"), True),
        CondSpec("T7",  "T7_RandomPageLocal_F4",           RoutePolicy("none"), True),
        CondSpec("T8",  "T8_PageLocal_F4",                 RoutePolicy("none"), True),
        CondSpec("T9",  "T9_ImageOnlyLocal_F4",            RoutePolicy("none"), True),
        CondSpec("T10", "T10_TextOnlyLocal_F4",            RoutePolicy("none"), True),
        CondSpec("T11", "T11_PageSentinel1_F4base",        RoutePolicy("none"), True),
        CondSpec("T12", "T12_PageSentinel4_F4base",        RoutePolicy("none"), True),
        CondSpec("T13", "T13_RandomSentinel4_F4base",      RoutePolicy("none"), True),
        CondSpec("T14", "T14_LastSentinel4_F4base",        RoutePolicy("none"), True),
        CondSpec("T15", "T15_TextSentinel4_F4base",        RoutePolicy("none"), True),
        CondSpec("T16", "T16_PageLocal_PageSentinel4",     RoutePolicy("none"), True),
    ]


def c_conditions_counting() -> list[CondSpec]:
    """Exp T-mini Phase 3 counting-image conditions C0..C12.

    Counting-image needs all-retained (no sparse routing). Drops T5/T9/T10 and
    re-orders to match the user spec.
    """
    return [
        CondSpec("C0",  None,                              RoutePolicy("none"), False),
        CondSpec("C1",  F4,                                RoutePolicy("none"), True),
        CondSpec("C2",  HOT,                               RoutePolicy("none"), True),
        CondSpec("C3",  "J12_F9_INT8side",                 RoutePolicy("none"), True),
        CondSpec("C4",  "SL_Outlier16_INT7side",           RoutePolicy("none"), True),
        CondSpec("C5",  "T6_TokenBlock16_F4",              RoutePolicy("none"), True),
        CondSpec("C6",  "T8_PageLocal_F4",                 RoutePolicy("none"), True),
        CondSpec("C7",  "T11_PageSentinel1_F4base",        RoutePolicy("none"), True),
        CondSpec("C8",  "T12_PageSentinel4_F4base",        RoutePolicy("none"), True),
        CondSpec("C9",  "T13_RandomSentinel4_F4base",      RoutePolicy("none"), True),
        CondSpec("C10", "T14_LastSentinel4_F4base",        RoutePolicy("none"), True),
        CondSpec("C11", "T15_TextSentinel4_F4base",        RoutePolicy("none"), True),
        CondSpec("C12", "T16_PageLocal_PageSentinel4",     RoutePolicy("none"), True),
    ]


def s_conditions_sidecode_ladder() -> list[CondSpec]:
    """Exp S Phase 1 — sidecode bit-ladder on the same multi-image slice.

    All conditions are dense (no FormatBook routing). They sweep the
    protected-channel count and sidecode storage width while keeping the
    INT4 base. The user-defined ladder, in order of increasing aggressiveness:

      S0  BF16 dense                                  16.000 KV bits
      S1  F4 dense                                     4.000 KV bits
      S2  F9 top-16 BF16 sidecode (anchor)             4.750 KV bits
      S3  SJ top-16 INT8 sidecode (Exp R winner)       4.250 KV bits
      S4  top-16 INT7 sidecode                         4.1875 KV bits
      S5  top-16 INT6 sidecode                         4.125 KV bits
      S6  top-16 INT5 sidecode                         4.0625 KV bits
      S7  top-24 INT6 sidecode                         4.1875 KV bits  (= S4 budget)
      S8  top-32 INT6 sidecode                         4.250  KV bits  (= S3 budget)
      S9  TextOnly-SJ: text pages SJ, visual pages F4

    The S4/S7 and S3/S8 pairs are matched-budget controls: at the same
    KV-bit budget, do "fewer channels at higher precision" or "more channels
    at lower precision" win?

    S9 uses formatbook_role_only with include_choice_routing=True so ALL
    visual pages (in-context + choice) get F4 cold, and text pages stay
    at the cache's hot format (SJ = J12).
    """
    return [
        CondSpec("S0", None,                       RoutePolicy("none"),                    False),
        CondSpec("S1", F4,                         RoutePolicy("none"),                    False),
        CondSpec("S2", "F9_KIVI_Outlier16",        RoutePolicy("none"),                    True),
        CondSpec("S3", "J12_F9_INT8side",          RoutePolicy("none"),                    True),
        CondSpec("S4", "SL_Outlier16_INT7side",    RoutePolicy("none"),                    True),
        CondSpec("S5", "SL_Outlier16_INT6side",    RoutePolicy("none"),                    True),
        CondSpec("S6", "SL_Outlier16_INT5side",    RoutePolicy("none"),                    True),
        CondSpec("S7", "SL_Outlier24_INT6side",    RoutePolicy("none"),                    True),
        CondSpec("S8", "SL_Outlier32_INT6side",    RoutePolicy("none"),                    True),
        CondSpec("S9", "J12_F9_INT8side",          RoutePolicy("formatbook_role_only"),    True),
    ]


def e_conditions_cold_ladder(best_route_name: str,
                             budget_fraction: float = 0.25) -> list[CondSpec]:
    """Exp R Sub-experiment E — cold-format ladder on the best AllVisual policy.

    best_route_name should be one of the AllVisual policy names, e.g.
    "formatbook_quest_allvisual" or "formatbook_split_quest" — whichever
    won Overnight 1's Sub-experiment C.

    For each cold-format variant, runs both Quest (best_route_name) and
    a Random control (random_allvisual / split_random variant).
    """
    is_split = "split" in best_route_name
    random_name = "formatbook_split_random" if is_split else "formatbook_random_allvisual"
    variants = [
        # (suffix, cold_quantizer, cold_v_quantizer)
        ("F4",     "f4",   "none"),   # E0 baseline
        ("K3_V4",  "int3", "none"),   # E1
        ("K4_V2",  "f4",   "int2"),   # E2
        ("K3_V2",  "int3", "int2"),   # E3
        ("FP8_V4", "fp8",  "none"),   # E4 diagnostic
    ]
    conds = []
    for i, (suffix, cq, cv) in enumerate(variants):
        for tag, route in (("Q", best_route_name), ("R", random_name)):
            conds.append(
                CondSpec(
                    f"E{i}_{suffix}_{tag}", HOT,
                    RoutePolicy(route, budget_fraction,
                                cold_quantizer=cq, cold_v_quantizer=cv,
                                token_budgeted=True),
                    True,
                )
            )
    return conds


def resolve_k_cfg(name: str, calib: Optional[dict]):
    for cfg in build_f_conditions(calib=calib):
        if cfg.name == name:
            return cfg
    raise KeyError(f"K-quantizer config {name!r} not in build_f_conditions")


def num_layers_and_kv_heads(model) -> tuple[int, int]:
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None)
    n_kv = getattr(cfg, "num_key_value_heads", None)
    if n_layers is None or n_kv is None:
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            n_layers = n_layers or text_cfg.num_hidden_layers
            n_kv = n_kv or text_cfg.num_key_value_heads
    return int(n_layers), int(n_kv)


# ---------------- T-mini slice_info plumbing ----------------

def _build_t_mini_slice_info(layout, seq_len: int, item) -> dict:
    """Construct slice_info dict for PageLocal / PageSentinel / Random* K kinds.

    Reads the PageLayout to populate:
      - v_start, v_end:   span of the FIRST visual page (kept for F5/F6 back-compat).
      - text_positions, visual_positions: lists of absolute prefill positions
        partitioned by modality.
      - role_spans:       dict {role_name -> (start, end)}; only "visual" is
        meaningful here (best-effort role inference; full role-spans would need
        the prompt template parser used by Exp F).
      - page_boundaries:  list of (start, end, kind) tuples per Page.
      - visual_token_positions_per_image: list[list[int]] — one list per visual
        page in order (in-context first, then choice). Used by sentinel kinds.
      - text_chunk_positions: list[list[int]] — one list per text page. Used by
        the "first_text" sentinel.
      - item_id: stringified item identifier; seeds per-item RNG for random
        boundaries / random sentinel positions.

    Falls back gracefully when layout is None (returns minimal slice_info with
    seq_len only).
    """
    if layout is None or not layout.pages:
        return {"seq_len": int(seq_len), "item_id": str(getattr(item, "id", ""))}
    page_boundaries: list[tuple[int, int, str]] = []
    visual_token_positions_per_image: list[list[int]] = []
    text_chunk_positions: list[list[int]] = []
    text_positions: list[int] = []
    visual_positions: list[int] = []
    first_visual_span: Optional[tuple[int, int]] = None
    for p in layout.pages:
        page_boundaries.append((int(p.start), int(p.end), str(p.kind)))
        positions = list(range(int(p.start), int(p.end)))
        if p.kind in ("in_context_image", "choice_image"):
            visual_token_positions_per_image.append(positions)
            visual_positions.extend(positions)
            if first_visual_span is None:
                first_visual_span = (int(p.start), int(p.end))
        else:
            text_chunk_positions.append(positions)
            text_positions.extend(positions)
    v_start, v_end = (first_visual_span if first_visual_span is not None
                      else (-1, -1))
    role_spans: dict[str, tuple[int, int]] = {}
    if v_start >= 0 and v_end > v_start:
        role_spans["visual"] = (v_start, v_end)
    return {
        "v_start": int(v_start),
        "v_end": int(v_end),
        "seq_len": int(seq_len),
        "text_positions": text_positions,
        "visual_positions": visual_positions,
        "role_spans": role_spans,
        "page_boundaries": page_boundaries,
        "visual_token_positions_per_image": visual_token_positions_per_image,
        "text_chunk_positions": text_chunk_positions,
        "item_id": str(getattr(item, "id", "")),
    }


# ---------------- per-page format / bit accounting ----------------

def _page_format_dense(k_cfg_name: Optional[str]) -> str:
    """Format label for a dense condition (no FormatBook). BF16 returns 'BF16'."""
    if k_cfg_name is None:
        return "BF16"
    if k_cfg_name == HOT:
        return "F9"
    if k_cfg_name == F4:
        return "F4"
    if k_cfg_name.startswith("S4_") or k_cfg_name.startswith("S8_") \
       or k_cfg_name.startswith("S12_") or k_cfg_name == "F8_KIVI_Outlier8":
        return k_cfg_name  # static baseline labels carry their bit budget
    if k_cfg_name.startswith("J12_") or k_cfg_name.startswith("SJ"):
        return "SJ"
    return k_cfg_name


def _k_bits_top_n_int_m(n: int, m: int) -> float:
    """K-bits/token for top-N outlier channels stored at INT-M (or BF16 if m=16),
    with the remaining (128-N) channels at INT4 base.
        K-bits = (m·N + 4·(128 − N)) / 128
    """
    return (m * n + 4 * (128 - n)) / 128.0


_STATIC_K_BITS = {
    "S4_Outlier4_BF16side":   _k_bits_top_n_int_m(4, 16),   # 4.375
    "S8_Outlier8_BF16side":   _k_bits_top_n_int_m(8, 16),   # 4.750
    "F8_KIVI_Outlier8":       _k_bits_top_n_int_m(8, 16),   # 4.750 (alias S8 = F8)
    "S12_Outlier12_BF16side": _k_bits_top_n_int_m(12, 16),  # 5.125
    # Exp R: J12 = F9 + INT8 sidecode
    "J12_F9_INT8side":        _k_bits_top_n_int_m(16, 8),   # 4.500
    # Exp S Phase 1 sidecode ladder
    "SL_Outlier16_INT7side":  _k_bits_top_n_int_m(16, 7),   # 4.375
    "SL_Outlier16_INT6side":  _k_bits_top_n_int_m(16, 6),   # 4.250
    "SL_Outlier16_INT5side":  _k_bits_top_n_int_m(16, 5),   # 4.125
    "SL_Outlier24_INT6side":  _k_bits_top_n_int_m(24, 6),   # 4.375
    "SL_Outlier32_INT6side":  _k_bits_top_n_int_m(32, 6),   # 4.500
    # F5 TextVisualLocal-F4 is also a true 4.00 K-bits format (text/visual
    # scales add only metadata overhead).
    "F5_KIVI_TextVisualSplit": 4.0,
    # T-mini: page-aware K formats. All page-local kinds keep TRUE 4.00 K-bits;
    # scale metadata is negligible vs cache size.
    "T6_TokenBlock16_F4":     4.0,
    "T7_RandomPageLocal_F4":  4.0,
    "T8_PageLocal_F4":        4.0,
    "T9_ImageOnlyLocal_F4":   4.0,
    "T10_TextOnlyLocal_F4":   4.0,
    # PageSentinel base K-bits (before adding sentinel BF16 overhead). The
    # sentinel adjustment happens per-item in _compute_bit_metrics via
    # _t_mini_sentinel_k_bits_adjustment().
    "T11_PageSentinel1_F4base": 4.0,
    "T12_PageSentinel4_F4base": 4.0,
    "T13_RandomSentinel4_F4base": 4.0,
    "T14_LastSentinel4_F4base": 4.0,
    "T15_TextSentinel4_F4base": 4.0,
    "T16_PageLocal_PageSentinel4": 4.0,
}


# T-mini PageSentinel cfg-name → (sentinel_kind, n_per_page). Used to compute
# the per-item sentinel-token count for accurate bit accounting.
_T_MINI_SENTINEL_PROPS: dict[str, tuple[str, int]] = {
    "T11_PageSentinel1_F4base":    ("first_visual", 1),
    "T12_PageSentinel4_F4base":    ("first_visual", 4),
    "T13_RandomSentinel4_F4base":  ("random_visual", 4),
    "T14_LastSentinel4_F4base":    ("last_visual", 4),
    "T15_TextSentinel4_F4base":    ("first_text", 4),
    "T16_PageLocal_PageSentinel4": ("first_visual", 4),
}


def _t_mini_sentinel_token_count(layout, cfg_name: str) -> int:
    """Compute the number of sentinel-protected tokens for a given T-mini
    PageSentinel cfg, using the PageLayout. Matches the resolution rules in
    k_quantizers._resolve_sentinel_positions: ``per_image`` lists for visual
    sentinels (clamped by min(n_per_page, page.n_tokens)) and ``per_chunk``
    lists for first_text.
    """
    if layout is None or cfg_name not in _T_MINI_SENTINEL_PROPS:
        return 0
    kind, n_per_page = _T_MINI_SENTINEL_PROPS[cfg_name]
    count = 0
    if kind in ("first_visual", "last_visual", "random_visual"):
        for p in layout.pages:
            if p.kind in ("in_context_image", "choice_image"):
                count += min(n_per_page, p.n_tokens)
    elif kind == "first_text":
        for p in layout.pages:
            if p.kind == "text":
                count += min(n_per_page, p.n_tokens)
    return count


def _page_k_bits_dense(k_cfg_name: Optional[str]) -> float:
    """K-bits/token for a dense (non-FormatBook) condition. Used for the
    storage metric only; the cache-side path is what actually quantizes."""
    if k_cfg_name is None:
        return 16.0
    if k_cfg_name == HOT:
        return PAGE_K_BITS["F9"]
    if k_cfg_name == F4:
        return PAGE_K_BITS["F4"]
    if k_cfg_name in _STATIC_K_BITS:
        return _STATIC_K_BITS[k_cfg_name]
    return PAGE_K_BITS["F9"]


def _v_bits_dense(k_cfg_name: Optional[str]) -> float:
    """V-bits/token for a dense condition. BF16 dense (k_cfg_name=None) leaves
    V at BF16 because FakeQuantKVCache is bypassed entirely. All other dense
    K-cfgs use FakeQuantKVCache which quantizes V to INT4.
    Fixes the Q0 V_BITS=4 bug from Exp Q.
    """
    if k_cfg_name is None:
        return V_BITS_BF16_DENSE   # BF16 dense → V is BF16, not INT4
    return V_BITS_DEFAULT


def _accumulate_format_counts(layout, layer_log, policy: RoutePolicy,
                              cold_format_label: str) -> tuple[dict, dict, dict]:
    """Per-page format assignment for one layer's routing decision.

    Returns (format_per_page: {page_idx: label}, page_counts: {label: n_pages},
             token_counts: {label: n_tokens}).

    Always-on pages (text, choice) get the hot format. Routable in-context pages
    get hot if active in this layer's routing decision, else cold_format_label.
    For sparse policies, cold pages are masked at attention time but stored at
    hot precision — so we still label them "F9" for the storage-bits metric.
    """
    active_set = set(layer_log.get("active_routable_pages", [])) if layer_log else set()
    is_fb = policy.is_formatbook()
    fmt_per_page: dict[int, str] = {}
    page_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    for p in layout.pages:
        if not p.is_routable:
            label = "F9"
        elif not is_fb:
            label = "F9"           # sparse routes don't downgrade storage
        elif p.page_idx in active_set:
            label = "F9"
        else:
            label = cold_format_label
        fmt_per_page[p.page_idx] = label
        page_counts[label] = page_counts.get(label, 0) + 1
        token_counts[label] = token_counts.get(label, 0) + p.n_tokens
    return fmt_per_page, page_counts, token_counts


def _compute_bit_metrics(layout, cache, cond: CondSpec) -> dict:
    """Compute effective_k_bits / kv_bits / f9_sidecode_*_fraction.

    For dense conditions (use_page_cache=False or RoutePolicy("none")):
      every page uses the same K format derived from cond.k_cfg_name; no
      per-layer variation.

    For FormatBook conditions:
      uses cache.routing_log[layer] to determine the per-layer hot/cold
      partition; averages effective_k_bits over layers.

    For RolyOnly / role_only:
      active set is empty; every routable page is cold; metric is identical
      across layers.
    """
    total_tokens = sum(p.n_tokens for p in layout.pages)
    total_pages = layout.n_pages
    if total_tokens == 0:
        return {}

    # Dense path (no routing → no FormatBook): one constant K-bit value.
    is_dense = (not cond.use_page_cache) or cond.route.name == "none"
    if is_dense:
        k_bits = _page_k_bits_dense(cond.k_cfg_name)
        v_bits = _v_bits_dense(cond.k_cfg_name)
        label = _page_format_dense(cond.k_cfg_name)
        # T-mini sentinel overhead: PageSentinel kinds restore N positions per
        # page to BF16. Adjust k_bits to reflect the per-item sentinel-token
        # fraction; record diagnostics.
        sentinel_count = _t_mini_sentinel_token_count(layout, cond.k_cfg_name or "")
        if sentinel_count > 0 and total_tokens > 0:
            k_bits = (
                (total_tokens - sentinel_count) * k_bits + sentinel_count * 16.0
            ) / total_tokens
        row = {
            "effective_k_bits": float(k_bits),
            "effective_v_bits": float(v_bits),
            "effective_kv_bits": float((k_bits + v_bits) / 2.0),
            "f9_sidecode_token_fraction": (1.0 if label == "F9" else 0.0),
            "f9_sidecode_page_fraction": (1.0 if label == "F9" else 0.0),
            "tokens_per_format": {label: int(total_tokens)},
            "pages_per_format": {label: int(total_pages)},
        }
        if sentinel_count > 0:
            row["n_sentinel_tokens"] = int(sentinel_count)
            row["sentinel_token_fraction"] = float(sentinel_count) / float(total_tokens)
        return row

    # Routed path: read cache.routing_log per layer.
    rl = getattr(cache, "routing_log", {}) or {}
    # Cold-K label
    cq = cond.route.cold_quantizer
    cold_label = {"f4": "F4", "int3": "INT3", "int2": "INT2", "fp8": "FP8"}.get(cq, "F4")
    # Cold-V handling (Exp R cold-format ladder).
    cold_v_int2 = (cond.route.cold_v_quantizer == "int2")
    if not rl:
        k_bits = PAGE_K_BITS["F9"]
        v_bits = V_BITS_DEFAULT
        return {
            "effective_k_bits": float(k_bits),
            "effective_v_bits": float(v_bits),
            "effective_kv_bits": float((k_bits + v_bits) / 2.0),
            "f9_sidecode_token_fraction": 1.0,
            "f9_sidecode_page_fraction": 1.0,
            "tokens_per_format": {"F9": int(total_tokens)},
            "pages_per_format": {"F9": int(total_pages)},
        }

    per_layer_k_bits: list[float] = []
    per_layer_v_bits: list[float] = []
    f9_token_frac_layers: list[float] = []
    f9_page_frac_layers: list[float] = []
    tot_token_counts: dict[str, float] = {}
    tot_page_counts: dict[str, float] = {}
    n_layers = 0
    for L, log in rl.items():
        _, page_counts, token_counts = _accumulate_format_counts(
            layout, log, cond.route, cold_label
        )
        k_bits_per_token = sum(
            PAGE_K_BITS[fmt] * cnt for fmt, cnt in token_counts.items()
        ) / max(1, total_tokens)
        per_layer_k_bits.append(k_bits_per_token)
        # V-bits: default V_BITS_DEFAULT everywhere; if cold-V INT2 is on,
        # cold-page tokens get V_BITS_COLD_INT2.
        if cold_v_int2:
            cold_tokens = sum(cnt for fmt, cnt in token_counts.items() if fmt == cold_label)
            v_bits_per_token = (cold_tokens * V_BITS_COLD_INT2
                                + (total_tokens - cold_tokens) * V_BITS_DEFAULT
                                ) / max(1, total_tokens)
        else:
            v_bits_per_token = V_BITS_DEFAULT
        per_layer_v_bits.append(v_bits_per_token)
        f9_token_frac_layers.append(token_counts.get("F9", 0) / max(1, total_tokens))
        f9_page_frac_layers.append(page_counts.get("F9", 0) / max(1, total_pages))
        for k, v in token_counts.items():
            tot_token_counts[k] = tot_token_counts.get(k, 0.0) + v
        for k, v in page_counts.items():
            tot_page_counts[k] = tot_page_counts.get(k, 0.0) + v
        n_layers += 1

    k_bits = float(np.mean(per_layer_k_bits)) if per_layer_k_bits else PAGE_K_BITS["F9"]
    v_bits = float(np.mean(per_layer_v_bits)) if per_layer_v_bits else V_BITS_DEFAULT
    f9_tok_frac = float(np.mean(f9_token_frac_layers)) if f9_token_frac_layers else 1.0
    f9_page_frac = float(np.mean(f9_page_frac_layers)) if f9_page_frac_layers else 1.0
    return {
        "effective_k_bits": k_bits,
        "effective_v_bits": float(v_bits),
        "effective_kv_bits": float((k_bits + v_bits) / 2.0),
        "f9_sidecode_token_fraction": f9_tok_frac,
        "f9_sidecode_page_fraction": f9_page_frac,
        "tokens_per_format": {k: float(v / max(1, n_layers)) for k, v in tot_token_counts.items()
                              if v > 0},
        "pages_per_format": {k: float(v / max(1, n_layers)) for k, v in tot_page_counts.items()
                             if v > 0},
    }


# ---------------- counting-image scoring (multi-token generation) ----------------

@torch.no_grad()
def score_item_counting(model, processor, item: MMNiahItem, cond: CondSpec,
                        k_cfg_obj, num_layers: int, num_kv_heads: int,
                        max_pixels_context: int,
                        max_pixels_needle: int,
                        include_choice_routing: bool = False,
                        max_new_tokens: int = 96) -> dict:
    """One forward + multi-token generation for an MM-NIAH counting-image item.

    Counting-image has no MCQ choices — the model emits a JSON list of integers
    that the parser decodes. Compared to MCQ items the page layout uses
    n_in_context_images = num_images - 1 (haystack pages) and n_choice_images = 1
    (the needle pattern which appears in the question, always-on).
    """
    from qwen_vl_utils import process_vision_info  # type: ignore
    from mm_niah_loader import format_counting_messages
    from counting_parser import parse_counting_output, score_counting

    messages = format_counting_messages(item, max_pixels_context=max_pixels_context,
                                        max_pixels_needle=max_pixels_needle)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    seq_len = int(input_ids.shape[1])

    layout = None
    cache = None
    if k_cfg_obj is not None or cond.use_page_cache:
        controller = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                                   mode="V1", default_k_bits=4, default_v_bits=4)
        if cond.use_page_cache:
            layout = build_page_layout(
                input_ids, processor,
                n_in_context_images=max(0, item.num_images - 1),  # haystack pages
                n_choice_images=1,                                # needle pattern
                needle_idx_in_images=-1,                          # no in-context needle
                include_choice_routing=include_choice_routing,
            )
            cache = PageAwareFakeQuantKVCache(controller, k_quantizer_config=k_cfg_obj)
            rng_seed = (abs(hash(f"{item.id}:{cond.name}")) % (2**31)) ^ 0xCAFEBABE
            cache.set_page_layout(layout, rng_seed=rng_seed)
            cache.correct_choice_idx = 0
        else:
            cache = FakeQuantKVCache(controller, k_quantizer_config=k_cfg_obj)

        slice_info = _build_t_mini_slice_info(layout, seq_len, item)
        if hasattr(cache, "set_slice_info"):
            cache.set_slice_info(slice_info)

    t0 = time.perf_counter()
    if cache is not None and cond.route.name != "none":
        with page_routing_sdpa_context(cache, cond.route):
            out = model.generate(**inputs, past_key_values=cache,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False, return_dict_in_generate=True,
                                 use_cache=True)
    elif cache is not None:
        with page_routing_sdpa_context(cache, RoutePolicy("none")):
            out = model.generate(**inputs, past_key_values=cache,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=False, return_dict_in_generate=True,
                                 use_cache=True)
    else:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, return_dict_in_generate=True,
                             use_cache=True)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    gen_tokens = out.sequences[0, input_ids.shape[1]:]
    output_text = processor.tokenizer.decode(gen_tokens, skip_special_tokens=True)
    parse_res = parse_counting_output(output_text)
    score_res = score_counting(parse_res["parsed"], item.gold_counts or [])

    row: dict = {
        "condition": cond.name,
        "item_id": item.id,
        "task": item.task,
        "num_images": item.num_images,
        "context_length_bucket": item.context_length_bucket,
        "context_length": item.context_length,
        "placed_depth": item.placed_depth,
        "gold_counts": list(item.gold_counts or []),
        "output_text": output_text,
        "parsed": parse_res["parsed"],
        "valid_format": parse_res["valid_format"],
        "predicted_length": parse_res["predicted_length"],
        "exact_match": score_res["exact_match"],
        "is_correct": score_res["exact_match"],
        "length_match": score_res["length_match"],
        "soft_accuracy": score_res["soft_accuracy"],
        "pred_sum": score_res["pred_sum"],
        "gold_sum": score_res["gold_sum"],
        "sum_match": score_res["sum_match"],
        "missing_format": score_res["missing_format"],
        "seq_len": seq_len,
        "latency_ms": latency_ms,
        "route_policy": cond.route.name,
        "route_budget": cond.route.budget_fraction,
        "cold_quantizer": cond.route.cold_quantizer,
        "cold_v_quantizer": cond.route.cold_v_quantizer,
        "token_budgeted": cond.route.token_budgeted,
        "include_choice_routing": include_choice_routing,
        "k_cfg": cond.k_cfg_name,
    }

    if layout is not None:
        row["n_pages"] = layout.n_pages
        row["n_in_context_images"] = layout.n_in_context_images
        row["n_choice_images"] = layout.n_choice_images
        bits_row = _compute_bit_metrics(layout, cache, cond)
        row.update(bits_row)
    else:
        # Dense BF16 (C0) — single static bit value.
        k_bits = _page_k_bits_dense(cond.k_cfg_name)
        v_bits = _v_bits_dense(cond.k_cfg_name)
        label = _page_format_dense(cond.k_cfg_name)
        row.update({
            "effective_k_bits": float(k_bits),
            "effective_v_bits": float(v_bits),
            "effective_kv_bits": float((k_bits + v_bits) / 2.0),
            "f9_sidecode_token_fraction": (1.0 if label == "F9" else 0.0),
            "f9_sidecode_page_fraction": (1.0 if label == "F9" else 0.0),
        })

    return row


# ---------------- per-item scoring ----------------

@torch.no_grad()
def score_item(model, processor, item: MMNiahItem, cond: CondSpec,
               k_cfg_obj, num_layers: int, num_kv_heads: int,
               answer_ids: list[int],
               max_pixels_context: int,
               max_pixels_choices: int,
               include_choice_routing: bool = False,
               max_new_tokens_counting: int = 96) -> dict:
    """One forward, one condition, one item. Returns a JSONL-ready dict.

    Dispatches on item.task:
      - retrieval-image / reasoning-image: MCQ first-token logit scoring (existing path)
      - counting-image: multi-token generation + list-output parsing (T-mini Phase 3)

    include_choice_routing (Exp R AllVisual): when True, choice-image pages
    are flagged is_routable=True in the layout so Quest/Random/Oracle
    FormatBook can route them. Default False preserves Exp Q behavior.
    """
    if item.task == "counting-image":
        return score_item_counting(model, processor, item, cond, k_cfg_obj,
                                   num_layers, num_kv_heads,
                                   max_pixels_context=max_pixels_context,
                                   max_pixels_needle=max_pixels_choices,
                                   include_choice_routing=include_choice_routing,
                                   max_new_tokens=max_new_tokens_counting)

    from qwen_vl_utils import process_vision_info  # type: ignore

    messages = format_mcq_messages(item, max_pixels_context=max_pixels_context,
                                   max_pixels_choices=max_pixels_choices)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    seq_len = int(input_ids.shape[1])

    layout = None
    cache = None
    if k_cfg_obj is not None or cond.use_page_cache:
        controller = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                                   mode="V1", default_k_bits=4, default_v_bits=4)
        if cond.use_page_cache:
            layout = build_page_layout(
                input_ids, processor,
                n_in_context_images=item.num_images,
                n_choice_images=4,
                needle_idx_in_images=item.needle_idx_in_images,
                include_choice_routing=include_choice_routing,
            )
            cache = PageAwareFakeQuantKVCache(controller, k_quantizer_config=k_cfg_obj)
            rng_seed = (abs(hash(f"{item.id}:{cond.name}")) % (2**31)) ^ 0xCAFEBABE
            cache.set_page_layout(layout, rng_seed=rng_seed)
            # Plumb item.correct_choice so AllVisual-Oracle can find the
            # correct-choice page from the routing wrapper.
            cache.correct_choice_idx = int(item.correct_choice)
        else:
            cache = FakeQuantKVCache(controller, k_quantizer_config=k_cfg_obj)

        # T-mini: plumb page-aware slice_info into the cache for PageLocal /
        # PageSentinel / Random* kinds. Reuses existing F5/F6 plumbing pattern
        # (cache.set_slice_info) — backward-compatible because non-T-mini kinds
        # simply ignore the new fields.
        slice_info = _build_t_mini_slice_info(layout, seq_len, item)
        if hasattr(cache, "set_slice_info"):
            cache.set_slice_info(slice_info)

    t0 = time.perf_counter()
    if cache is not None and cond.route.name != "none":
        with page_routing_sdpa_context(cache, cond.route):
            out = model.generate(**inputs, past_key_values=cache, max_new_tokens=1,
                                 do_sample=False, return_dict_in_generate=True,
                                 output_scores=True, use_cache=True)
    elif cache is not None:
        with page_routing_sdpa_context(cache, RoutePolicy("none")):
            out = model.generate(**inputs, past_key_values=cache, max_new_tokens=1,
                                 do_sample=False, return_dict_in_generate=True,
                                 output_scores=True, use_cache=True)
    else:
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                             return_dict_in_generate=True, output_scores=True,
                             use_cache=True)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    first_logits = out.scores[0]
    logprobs = torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist()
    pred = int(max(range(len(answer_ids)), key=lambda i: logprobs[i]))

    row: dict = {
        "condition": cond.name,
        "item_id": item.id,
        "context_length_bucket": item.context_length_bucket,
        "context_length": item.context_length,
        "num_images": item.num_images,
        "needle_idx_in_images": item.needle_idx_in_images,
        "placed_depth": item.placed_depth,
        "correct_choice": item.correct_choice,
        "pred_choice": pred,
        "is_correct": (pred == item.correct_choice),
        "option_logprobs": logprobs,
        "seq_len": seq_len,
        "latency_ms": latency_ms,
        "route_policy": cond.route.name,
        "route_budget": cond.route.budget_fraction,
        "cold_quantizer": cond.route.cold_quantizer,
        "cold_v_quantizer": cond.route.cold_v_quantizer,
        "token_budgeted": cond.route.token_budgeted,
        "include_choice_routing": include_choice_routing,
        "k_cfg": cond.k_cfg_name,
    }

    if layout is not None:
        row["n_pages"] = layout.n_pages
        row["n_in_context_images"] = layout.n_in_context_images
        row["n_choice_images"] = layout.n_choice_images
        row["needle_page_idx"] = layout.needle_page_idx

    # Routing log diagnostics + per-page bits accounting.
    if cache is not None and isinstance(cache, PageAwareFakeQuantKVCache):
        rl = cache.routing_log
        if rl:
            needle_hits = [v.get("needle_in_active") for v in rl.values()]
            row["needle_in_active_per_layer"] = needle_hits
            row["needle_in_active_layer_mean"] = (
                sum(1 for h in needle_hits if h) / max(1, len(needle_hits))
            )
            ranks = [v.get("needle_rank") for v in rl.values()]
            row["needle_rank_per_layer"] = ranks
            valid_ranks = [r for r in ranks if r is not None]
            if valid_ranks:
                row["needle_rank_median"] = float(np.median(valid_ranks))
                row["needle_rank_mean"] = float(np.mean(valid_ranks))
            n_active = [len(v.get("active_routable_pages", [])) for v in rl.values()]
            n_cold = [len(v.get("cold_routable_pages", [])) for v in rl.values()]
            row["active_routable_pages_layer_mean"] = float(np.mean(n_active)) if n_active else 0
            row["cold_routable_pages_layer_mean"] = float(np.mean(n_cold)) if n_cold else 0
            total_routable = (n_active[0] + n_cold[0]) if n_active else 0
            row["page_read_fraction"] = (
                float(np.mean(n_active)) / total_routable if total_routable else 1.0
            )

    # Per-page bit accounting for both routed and dense conditions.
    if layout is not None:
        bits_row = _compute_bit_metrics(layout, cache, cond)
        row.update(bits_row)
    else:
        # Dense BF16 (Q0) or dense F4 (Q1) with no PageLayout — single static bit value.
        k_bits = _page_k_bits_dense(cond.k_cfg_name)
        v_bits = _v_bits_dense(cond.k_cfg_name)
        label = _page_format_dense(cond.k_cfg_name)
        row.update({
            "effective_k_bits": float(k_bits),
            "effective_v_bits": float(v_bits),
            "effective_kv_bits": float((k_bits + v_bits) / 2.0),
            "f9_sidecode_token_fraction": (1.0 if label == "F9" else 0.0),
            "f9_sidecode_page_fraction": (1.0 if label == "F9" else 0.0),
        })

    return row


# ---------------- driver ----------------

def _append_progress(progress_log: Path, line: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_log, "a") as f:
        f.write(f"[{ts}] {line}\n")
        f.flush()


def write_summary_md(jsonl_path: Path, out_md: Path) -> None:
    """Aggregate JSONL into a current-state summary table (periodic callback)."""
    rows: list[dict] = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    by_cond: dict[str, list[dict]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Exp Q current — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append(f"total rows: {len(rows)} across {len(by_cond)} conditions\n")
    lines.append("| condition | n | acc | eff_kv_bits | f9_sidecode_tok | needle_hit | page_read_frac | latency_ms |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for cond_name in sorted(by_cond.keys()):
        rs = by_cond[cond_name]
        n = len(rs)
        acc = sum(1 for r in rs if r.get("is_correct")) / max(1, n)
        ekvb = [r.get("effective_kv_bits") for r in rs
                if r.get("effective_kv_bits") is not None]
        ekvb_str = f"{np.mean(ekvb):.3f}" if ekvb else "—"
        sidecode = [r.get("f9_sidecode_token_fraction") for r in rs
                    if r.get("f9_sidecode_token_fraction") is not None]
        sidecode_str = f"{np.mean(sidecode):.3f}" if sidecode else "—"
        nhit = [r.get("needle_in_active_layer_mean") for r in rs
                if r.get("needle_in_active_layer_mean") is not None]
        nhit_str = f"{np.mean(nhit):.3f}" if nhit else "—"
        prf = [r.get("page_read_fraction") for r in rs
               if r.get("page_read_fraction") is not None]
        prf_str = f"{np.mean(prf):.3f}" if prf else "—"
        lat = [r.get("latency_ms", 0) for r in rs]
        lat_str = f"{np.mean(lat):.0f}" if lat else "—"
        lines.append(
            f"| {cond_name} | {n} | {acc:.3f} | {ekvb_str} | {sidecode_str} | "
            f"{nhit_str} | {prf_str} | {lat_str} |"
        )
    out_md.write_text("\n".join(lines) + "\n")


def run_condition_on_items(model, processor, items: list[MMNiahItem],
                           cond: CondSpec, k_cfg_obj,
                           num_layers: int, num_kv_heads: int,
                           answer_ids: list[int],
                           out_jsonl: Path, progress_log: Path,
                           max_pixels_context: int,
                           max_pixels_choices: int,
                           progress_every: int = 10, summary_every: int = 25,
                           summary_callback=None,
                           skip_ids: Optional[set] = None,
                           include_choice_routing: bool = False,
                           max_new_tokens_counting: int = 96) -> None:
    skip_ids = skip_ids or set()
    n = len(items)
    n_done = 0
    n_correct = 0
    t_start = time.perf_counter()
    _append_progress(progress_log, f"START {cond.name} n_items={n} (skip={len(skip_ids)})")
    with open(out_jsonl, "a") as f:
        for i, it in enumerate(items):
            if it.id in skip_ids:
                continue
            try:
                row = score_item(model, processor, it, cond, k_cfg_obj,
                                 num_layers=num_layers, num_kv_heads=num_kv_heads,
                                 answer_ids=answer_ids,
                                 max_pixels_context=max_pixels_context,
                                 max_pixels_choices=max_pixels_choices,
                                 include_choice_routing=include_choice_routing,
                                 max_new_tokens_counting=max_new_tokens_counting)
            except torch.cuda.OutOfMemoryError:
                _append_progress(progress_log,
                                 f"WARN [{cond.name}] OOM on item {it.id} "
                                 f"seq_len_est={it.context_length} -- skipping")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                _append_progress(progress_log,
                                 f"ERR [{cond.name}] item {it.id} {type(e).__name__}: {e}")
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_done += 1
            if row["is_correct"]:
                n_correct += 1
            if n_done % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if n_done % progress_every == 0 or i == n - 1:
                elapsed = time.perf_counter() - t_start
                rate = elapsed / max(1, n_done)
                eta = rate * (n - i - 1)
                ekvb = row.get("effective_kv_bits")
                ekvb_str = f"{ekvb:.2f}" if ekvb is not None else "—"
                line = (f"[{cond.name}] {n_done}/{n - len(skip_ids)} "
                        f"acc={n_correct/n_done:.3f} kv_bits={ekvb_str} "
                        f"seq_len={row['seq_len']} latency={row['latency_ms']:.0f}ms "
                        f"elapsed={timedelta(seconds=int(elapsed))} "
                        f"ETA={timedelta(seconds=int(eta))}")
                print(line, flush=True)
                _append_progress(progress_log, line)
            if summary_callback is not None and n_done % summary_every == 0:
                try:
                    summary_callback(out_jsonl)
                except Exception as e:
                    _append_progress(progress_log, f"WARN summary cb failed: {e!r}")
    if summary_callback is not None:
        try:
            summary_callback(out_jsonl)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _append_progress(progress_log, f"DONE {cond.name} acc={n_correct}/{n_done}")


def existing_completion_ids(jsonl_path: Path, cond_name: str) -> set:
    if not jsonl_path.exists():
        return set()
    done = set()
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("condition") == cond_name:
                done.add(r["item_id"])
    return done


# ---------------- main ----------------

def _default_calib_npz(task: str, seed: int = 0) -> Path:
    if task == "retrieval-image":
        if seed == 0:
            return CALIBRATION_DIR / "expP_mmniah_kcalib_Qwen2.5-VL-7B-Instruct_seed0.npz"
        # Exp R seed=1 calibration on the new seed=1 cal split.
        return CALIBRATION_DIR / f"expP_mmniah_kcalib_Qwen2.5-VL-7B-Instruct_seed{seed}.npz"
    # T-mini Phase 2/3: per-task NPZ named with the task slug.
    return CALIBRATION_DIR / (
        f"expP_mmniah_kcalib_Qwen2.5-VL-7B-Instruct_{task}_seed{seed}.npz"
    )


def _default_split_path(task: str, seed: int = 0) -> Path:
    base = split_file_for_task(task)
    if seed == 0 and task == "retrieval-image":
        return base
    return base.with_name(f"mm_niah_{task}_split_seed{seed}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=10_000,
                    help="Cap on per-condition items; large default lets full pool run.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--task", choices=SUPPORTED_TASKS, default="retrieval-image")
    ap.add_argument("--calib-npz", type=Path, default=None,
                    help="K-quant calibration NPZ. Defaults to per-task path.")
    ap.add_argument("--out-jsonl", type=Path, default=None,
                    help="Default: results/expQ_rollouts_slice{A|B}.jsonl")
    ap.add_argument("--out-summary", type=Path, default=None,
                    help="Default: results/expQ_summary_slice{A|B}.md")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Subset of condition names to run.")
    ap.add_argument("--include-int2-stretch", action="store_true",
                    help="Add Q10/Q11 INT2-cold conditions (Slice A only).")
    ap.add_argument("--include-reseed", action="store_true",
                    help="Add Q5_s1/s2, Q8_s1/s2, Q11_s1/s2 reseed conditions.")
    ap.add_argument("--include-smoke-only", action="store_true",
                    help="Add Q_allhot smoke-only sanity condition. Not for main pool.")
    # Exp R flags
    ap.add_argument("--exp-r-c", action="store_true",
                    help="Exp R Sub-experiment C: AllVisual routing + static "
                         "matched-budget baselines (C0..C8 + S4/S8/S12/SJ).")
    ap.add_argument("--exp-r-e-best-route", default=None,
                    help="Exp R Sub-experiment E cold-format ladder. Pass the "
                         "name of the AllVisual policy that won C (e.g. "
                         "'formatbook_quest_allvisual' or 'formatbook_split_quest').")
    ap.add_argument("--include-choice-routing", action="store_true",
                    help="Exp R: build_page_layout flags choice-image pages as "
                         "routable so AllVisual policies can route them. "
                         "Required for C and E conditions.")
    # Exp S flags
    ap.add_argument("--exp-s-ladder", action="store_true",
                    help="Exp S Phase 1: sidecode bit-ladder S0..S9 (no AllVisual).")
    # Exp T-mini flags
    ap.add_argument("--exp-t-mini", action="store_true",
                    help="Exp T-mini Phase 1/2: T0..T16 page-aware K formats on "
                         "retrieval-image / reasoning-image.")
    ap.add_argument("--exp-t-mini-counting", action="store_true",
                    help="Exp T-mini Phase 3: C0..C12 page-aware K formats on "
                         "counting-image (multi-token generation + list-output parsing).")
    ap.add_argument("--max-new-tokens-counting", type=int, default=96,
                    help="max_new_tokens for counting-image generation (default 96).")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-run items already in the JSONL.")
    ap.add_argument("--split-path", type=Path, default=None,
                    help="Per-task split JSON path. Defaults via split_file_for_task(task).")
    ap.add_argument("--min-num-images", type=int, default=0,
                    help="Filter eval items to those with num_images >= N.")
    ap.add_argument("--use-full-pool", action="store_true",
                    help="Ignore eval split; use all items (minus cal) matching --min-num-images.")
    ap.add_argument("--max-pixels-context", type=int, default=336 * 336,
                    help="max_pixels cap for in-context images (default 336²=112896).")
    ap.add_argument("--max-pixels-choices", type=int, default=336 * 336,
                    help="max_pixels cap for the 4 choice images (default 336²=112896 — "
                         "equal to context, fixes Exp P bias).")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    if args.task == "retrieval-image":
        slice_tag = "A"
    elif args.task == "reasoning-image":
        slice_tag = "B"
    elif args.task == "counting-image":
        slice_tag = "Count"
    else:
        slice_tag = args.task
    if args.out_jsonl is None:
        args.out_jsonl = RESULTS_DIR / f"expQ_rollouts_slice{slice_tag}.jsonl"
    if args.out_summary is None:
        args.out_summary = RESULTS_DIR / f"expQ_summary_slice{slice_tag}.md"
    if args.calib_npz is None:
        args.calib_npz = _default_calib_npz(args.task, seed=args.seed)
    if args.split_path is None:
        args.split_path = _default_split_path(args.task, seed=args.seed)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_log = args.out_jsonl.with_name(args.out_jsonl.stem + ".progress.log")
    _append_progress(progress_log, f"=== launch args={vars(args)} ===")

    # Items + split
    items = load_all_items(task=args.task)
    if args.split_path.exists():
        split = load_split(args.split_path)
    else:
        split = make_split(items, seed=args.seed)
        save_split(split, args.split_path)
    if args.use_full_pool:
        cal_ids = set(split.get("cal", []))
        eval_items = [it for it in items if it.id not in cal_ids]
        print(f"use_full_pool: {len(eval_items)} items (full pool minus {len(cal_ids)} cal)",
              flush=True)
    else:
        eval_items = filter_items(items, split["eval"])
    if args.min_num_images > 0:
        before = len(eval_items)
        eval_items = [it for it in eval_items if it.num_images >= args.min_num_images]
        print(f"min_num_images filter: {before} -> {len(eval_items)} "
              f"(num_images >= {args.min_num_images})", flush=True)
    eval_items = eval_items[:args.n_items]
    print(f"task={args.task} loaded {len(items)} items; eval {len(eval_items)} of {len(split['eval'])}",
          flush=True)
    _append_progress(progress_log,
                     f"items: total={len(items)} eval={len(eval_items)} "
                     f"min_num_images={args.min_num_images} "
                     f"max_pixels=(ctx={args.max_pixels_context}, choices={args.max_pixels_choices}) "
                     f"buckets="
                     f"{ {b: sum(1 for it in eval_items if it.context_length_bucket == b) for b in ['short','mid','long']} }")

    # Calibration
    calib = None
    if args.calib_npz.exists():
        arr = np.load(args.calib_npz)
        calib = {k: arr[k] for k in arr.files}
        # Exp S Phase 1: derive outlier_channel_idx_top32 from k_channel_energy
        # if the NPZ only has top-16. Lets S7 (top-24) and S8 (top-32) use the
        # existing NPZ without recalibration. _outlier_channel_indices also
        # falls back to energy argsort, but precomputing here is cheap and
        # makes the path explicit.
        if "k_channel_energy" in calib:
            energy = np.asarray(calib["k_channel_energy"])  # [L, H_kv, D]
            top32 = np.argsort(energy, axis=-1)[..., -32:][..., ::-1].copy().astype(np.int32)
            calib["outlier_channel_idx_top32"] = top32
            print(f"  derived outlier_channel_idx_top32 from k_channel_energy "
                  f"(shape {top32.shape})", flush=True)
        print(f"loaded calibration {args.calib_npz} ({len(calib)} keys)", flush=True)
    else:
        print(f"[warn] calib NPZ not found at {args.calib_npz} — F9 conditions will fail",
              flush=True)

    # Build conditions
    if args.exp_t_mini_counting:
        # Exp T-mini Phase 3: counting-image C0..C12.
        primary = c_conditions_counting()
    elif args.exp_t_mini:
        # Exp T-mini Phase 1/2: T0..T16 on retrieval-image or reasoning-image.
        primary = t_conditions_main()
    elif args.exp_s_ladder:
        # Exp S Phase 1: sidecode bit-ladder S0..S9.
        primary = s_conditions_sidecode_ladder()
    elif args.exp_r_c:
        # Exp R Sub-exp C: AllVisual + static matched-budget baselines.
        primary = c_conditions_allvisual() + s_conditions_static_baselines()
    elif args.exp_r_e_best_route:
        # Exp R Sub-exp E: cold-format ladder on the chosen AllVisual policy.
        primary = e_conditions_cold_ladder(args.exp_r_e_best_route)
    elif args.task == "retrieval-image":
        primary = q_conditions_slice_a(include_int2=args.include_int2_stretch)
    elif args.task == "reasoning-image":
        primary = r_conditions_slice_b()
    else:
        primary = q_conditions_slice_a(include_int2=False)
    all_conds = primary
    if args.include_reseed:
        all_conds = all_conds + q_conditions_reseed()
    if args.include_smoke_only:
        all_conds = all_conds + q_conditions_smoke_only()
    if args.conditions:
        keep = set(args.conditions)
        all_conds = [c for c in all_conds if c.name in keep]
    print(f"running conditions: {[c.name for c in all_conds]}", flush=True)

    # Resolve k_cfgs once
    cond_to_kcfg = {}
    for c in all_conds:
        cond_to_kcfg[c.name] = resolve_k_cfg(c.k_cfg_name, calib) if c.k_cfg_name else None

    # Load model
    print(f"loading model {args.model} ...", flush=True)
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    num_layers, num_kv_heads = num_layers_and_kv_heads(model)
    answer_ids = answer_token_ids(processor, n=4)
    print(f"model loaded num_layers={num_layers} num_kv_heads={num_kv_heads}", flush=True)

    def summary_cb(jsonl_path: Path):
        write_summary_md(jsonl_path, args.out_summary)

    # Exp R: auto-enable include_choice_routing for AllVisual conditions
    # (C0..C8 + E*) unless the user explicitly disabled it.
    auto_choice_routing = args.include_choice_routing or args.exp_r_c \
        or args.exp_r_e_best_route is not None
    if auto_choice_routing and not args.include_choice_routing:
        print("[exp R] auto-enabling --include-choice-routing for AllVisual / E conditions", flush=True)
        args.include_choice_routing = True

    t_overall = time.perf_counter()
    for cond in all_conds:
        skip = existing_completion_ids(args.out_jsonl, cond.name) if not args.no_resume else set()
        kcfg = cond_to_kcfg[cond.name]
        run_condition_on_items(
            model, processor, eval_items, cond, kcfg,
            num_layers=num_layers, num_kv_heads=num_kv_heads,
            answer_ids=answer_ids,
            out_jsonl=args.out_jsonl, progress_log=progress_log,
            max_pixels_context=args.max_pixels_context,
            max_pixels_choices=args.max_pixels_choices,
            summary_callback=summary_cb, skip_ids=skip,
            include_choice_routing=args.include_choice_routing,
            max_new_tokens_counting=args.max_new_tokens_counting,
        )
    total_wall = time.perf_counter() - t_overall
    _append_progress(progress_log, f"=== overall wall={timedelta(seconds=int(total_wall))} ===")
    summary_cb(args.out_jsonl)
    print(f"\ndone. summary -> {args.out_summary}", flush=True)


if __name__ == "__main__":
    main()
