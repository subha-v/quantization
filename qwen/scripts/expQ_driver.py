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

# Per-token K bits per page format. V is uniformly INT4 in this setup (= 4.0).
PAGE_K_BITS = {"F9": 5.50, "F4": 4.00, "INT2": 2.00}
V_BITS = 4.00


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


# ---------------- per-page format / bit accounting ----------------

def _page_format_dense(k_cfg_name: Optional[str]) -> str:
    """Format label for a dense condition (no FormatBook). BF16 returns 'BF16'."""
    if k_cfg_name is None:
        return "BF16"
    if k_cfg_name == HOT:
        return "F9"
    if k_cfg_name == F4:
        return "F4"
    # Unknown / new K-cfg — best-effort.
    return k_cfg_name


def _page_k_bits_dense(k_cfg_name: Optional[str]) -> float:
    if k_cfg_name is None:
        return 16.0
    if k_cfg_name == HOT:
        return PAGE_K_BITS["F9"]
    if k_cfg_name == F4:
        return PAGE_K_BITS["F4"]
    return PAGE_K_BITS["F9"]  # default to F9 for unrecognized hot-style configs


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
        label = _page_format_dense(cond.k_cfg_name)
        return {
            "effective_k_bits": float(k_bits),
            "effective_v_bits": float(V_BITS),
            "effective_kv_bits": float((k_bits + V_BITS) / 2.0),
            "f9_sidecode_token_fraction": (1.0 if label == "F9" else 0.0),
            "f9_sidecode_page_fraction": (1.0 if label == "F9" else 0.0),
            "tokens_per_format": {label: int(total_tokens)},
            "pages_per_format": {label: int(total_pages)},
        }

    # Routed path: read cache.routing_log per layer.
    rl = getattr(cache, "routing_log", {}) or {}
    cold_label = "INT2" if cond.route.cold_quantizer == "int2" else "F4"
    if not rl:
        # No layers fired (shouldn't happen for use_page_cache=True). Fall back
        # to assuming everything is F9.
        k_bits = PAGE_K_BITS["F9"]
        return {
            "effective_k_bits": float(k_bits),
            "effective_v_bits": float(V_BITS),
            "effective_kv_bits": float((k_bits + V_BITS) / 2.0),
            "f9_sidecode_token_fraction": 1.0,
            "f9_sidecode_page_fraction": 1.0,
            "tokens_per_format": {"F9": int(total_tokens)},
            "pages_per_format": {"F9": int(total_pages)},
        }

    per_layer_k_bits: list[float] = []
    f9_token_frac_layers: list[float] = []
    f9_page_frac_layers: list[float] = []
    # Average page and token counts across layers (policy is layer-agnostic in
    # this setup; all layers see the same routing decision because the SDPA
    # wrapper picks per-Q query). But sum-per-layer-mean is robust to future
    # per-layer routing.
    tot_token_counts: dict[str, float] = {"F9": 0.0, "F4": 0.0, "INT2": 0.0}
    tot_page_counts: dict[str, float] = {"F9": 0.0, "F4": 0.0, "INT2": 0.0}
    n_layers = 0
    for L, log in rl.items():
        _, page_counts, token_counts = _accumulate_format_counts(
            layout, log, cond.route, cold_label
        )
        k_bits_per_token = sum(
            PAGE_K_BITS[fmt] * cnt for fmt, cnt in token_counts.items()
        ) / max(1, total_tokens)
        per_layer_k_bits.append(k_bits_per_token)
        f9_token_frac_layers.append(token_counts.get("F9", 0) / max(1, total_tokens))
        f9_page_frac_layers.append(page_counts.get("F9", 0) / max(1, total_pages))
        for k, v in token_counts.items():
            tot_token_counts[k] = tot_token_counts.get(k, 0.0) + v
        for k, v in page_counts.items():
            tot_page_counts[k] = tot_page_counts.get(k, 0.0) + v
        n_layers += 1

    k_bits = float(np.mean(per_layer_k_bits)) if per_layer_k_bits else PAGE_K_BITS["F9"]
    f9_tok_frac = float(np.mean(f9_token_frac_layers)) if f9_token_frac_layers else 1.0
    f9_page_frac = float(np.mean(f9_page_frac_layers)) if f9_page_frac_layers else 1.0
    # Average per-layer (token/page) format counts so a row's bookkeeping
    # matches per-layer means.
    return {
        "effective_k_bits": k_bits,
        "effective_v_bits": float(V_BITS),
        "effective_kv_bits": float((k_bits + V_BITS) / 2.0),
        "f9_sidecode_token_fraction": f9_tok_frac,
        "f9_sidecode_page_fraction": f9_page_frac,
        "tokens_per_format": {k: float(v / max(1, n_layers)) for k, v in tot_token_counts.items()
                              if v > 0},
        "pages_per_format": {k: float(v / max(1, n_layers)) for k, v in tot_page_counts.items()
                             if v > 0},
    }


# ---------------- per-item scoring ----------------

@torch.no_grad()
def score_item(model, processor, item: MMNiahItem, cond: CondSpec,
               k_cfg_obj, num_layers: int, num_kv_heads: int,
               answer_ids: list[int],
               max_pixels_context: int,
               max_pixels_choices: int) -> dict:
    """One forward, one condition, one item. Returns a JSONL-ready dict."""
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
            )
            cache = PageAwareFakeQuantKVCache(controller, k_quantizer_config=k_cfg_obj)
            rng_seed = (abs(hash(f"{item.id}:{cond.name}")) % (2**31)) ^ 0xCAFEBABE
            cache.set_page_layout(layout, rng_seed=rng_seed)
        else:
            cache = FakeQuantKVCache(controller, k_quantizer_config=k_cfg_obj)

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
        label = _page_format_dense(cond.k_cfg_name)
        row.update({
            "effective_k_bits": float(k_bits),
            "effective_v_bits": float(V_BITS),
            "effective_kv_bits": float((k_bits + V_BITS) / 2.0),
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
                           skip_ids: Optional[set] = None) -> None:
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
                                 max_pixels_choices=max_pixels_choices)
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

def _default_calib_npz(task: str) -> Path:
    if task == "retrieval-image":
        return CALIBRATION_DIR / "expP_mmniah_kcalib_Qwen2.5-VL-7B-Instruct_seed0.npz"
    return CALIBRATION_DIR / f"expQ_mmniah_{task}_kcalib_Qwen2.5-VL-7B-Instruct_seed0.npz"


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

    slice_tag = "A" if args.task == "retrieval-image" else "B"
    if args.out_jsonl is None:
        args.out_jsonl = RESULTS_DIR / f"expQ_rollouts_slice{slice_tag}.jsonl"
    if args.out_summary is None:
        args.out_summary = RESULTS_DIR / f"expQ_summary_slice{slice_tag}.md"
    if args.calib_npz is None:
        args.calib_npz = _default_calib_npz(args.task)
    if args.split_path is None:
        args.split_path = split_file_for_task(args.task)

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
        print(f"loaded calibration {args.calib_npz} ({len(calib)} keys)", flush=True)
    else:
        print(f"[warn] calib NPZ not found at {args.calib_npz} — F9 conditions will fail",
              flush=True)

    # Build conditions
    if args.task == "retrieval-image":
        primary = q_conditions_slice_a(include_int2=args.include_int2_stretch)
    else:
        primary = r_conditions_slice_b()
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
        )
    total_wall = time.perf_counter() - t_overall
    _append_progress(progress_log, f"=== overall wall={timedelta(seconds=int(total_wall))} ===")
    summary_cb(args.out_jsonl)
    print(f"\ndone. summary -> {args.out_summary}", flush=True)


if __name__ == "__main__":
    main()
