"""Exp P main driver — sweep page-routing conditions on MM-NIAH retrieval-image.

Conditions (primary, 7):
  P0  BF16 dense baseline
  P1  Full F4 (KIVI per-channel-seq INT4)        — same-benchmark F4 anchor
  P2  Full J12 (F9 + INT8 sidecode)              — dense near-lossless anchor
  P3  Quest sparse, top-25% visual + all text    — sparse attention, J12 K
  P4  Random sparse, top-25% visual + all text   — random baseline
  P5  Oracle sparse, needle + text only          — headroom upper bound
  P6  FormatBook top-50% (Quest J12 / cold F4)   — query-adaptive per-page format

Stretch (run if primary completes with budget remaining):
  P3b Quest sparse top-50%
  P4b Random sparse top-50%

Per-item JSONL row contains accuracy + rich routing diagnostics. Periodic
summary callback every 25 items keeps `expP_summary.md` current during the run.
"""
from __future__ import annotations

import argparse
import gc
import json
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
    MMNiahItem, answer_token_ids, filter_items, format_mcq_messages, load_all_items,
    load_split, make_split, save_split, DEFAULT_SPLIT_FILE,
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


F9 = "F9_KIVI_Outlier16"
F4 = "F4_KIVI_PerChannelSeq"
J12 = "J12_F9_INT8side"


def primary_conditions() -> list[CondSpec]:
    """Primary diagnostic conditions for the FormatBook + Quest routing test.

    P5 oracle is budget-matched to P3 (top-25%): needle forced into active,
    remaining (K-1) slots filled by top-Quest. P6/P6R/P6O are the three
    FormatBook variants — without P6R/P6O a positive P6 result is ambiguous
    (could just mean "any 50% of pages at J12 works").
    """
    return [
        CondSpec("P0",  None, RoutePolicy("none"),                       False),
        CondSpec("P1",  F4,   RoutePolicy("none"),                       False),
        CondSpec("P2",  F9,   RoutePolicy("none"),                       True),
        CondSpec("P3",  F9,   RoutePolicy("quest_sparse", 0.25),         True),
        CondSpec("P4",  F9,   RoutePolicy("random_sparse", 0.25),        True),
        CondSpec("P5",  F9,   RoutePolicy("oracle_sparse", 0.25),        True),
        # FormatBook trio at top-50% — Quest / Random / Oracle hot-page selection.
        CondSpec("P6",  F9,   RoutePolicy("formatbook_quest", 0.5),      True),
        CondSpec("P6R", F9,   RoutePolicy("formatbook_random", 0.5),     True),
        CondSpec("P6O", F9,   RoutePolicy("formatbook_oracle", 0.5),     True),
    ]


def stretch_conditions() -> list[CondSpec]:
    """Stretch conditions — same JSONL, run after primary if budget allows."""
    return [
        # Multi-seed random sparse (noise reduction for the Quest > Random claim)
        CondSpec("P4_s1",   F9, RoutePolicy("random_sparse", 0.25),  True),
        CondSpec("P4_s2",   F9, RoutePolicy("random_sparse", 0.25),  True),
        # Top-50% sparse stretch (Pareto curve)
        CondSpec("P3b",     F9, RoutePolicy("quest_sparse", 0.5),    True),
        CondSpec("P4b",     F9, RoutePolicy("random_sparse", 0.5),   True),
        # Strict needle-only oracle (no Quest fill) — tests whether the model
        # needs ANY surrounding pages or just the needle.
        CondSpec("P5_only", F9, RoutePolicy("oracle_needle_only"),   True),
        # J12 (F9 + INT8 sidecode) dense — characterizes the INT8-sidecode
        # confound separately so primary P2-P6 stay clean on F9/BF16-sidecode.
        CondSpec("P2b",     J12, RoutePolicy("none"),                True),
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


# ---------------- per-item scoring ----------------

@torch.no_grad()
def score_item(model, processor, item: MMNiahItem, cond: CondSpec,
               k_cfg_obj, num_layers: int, num_kv_heads: int,
               answer_ids: list[int],
               max_pixels_context: int = 144 * 144,
               max_pixels_choices: int = 224 * 224,
               record_layout_warnings: bool = True) -> dict:
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
    if k_cfg_obj is not None:
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
            # Deterministic per-(item, condition) RNG so:
            #  - reruns and resumes reproduce the exact random samples
            #  - P4 / P4_s1 / P4_s2 produce DIFFERENT random samples on the
            #    same item (multi-seed noise reduction for the Quest > Random
            #    claim)
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
        # PageAware cache without routing — install no-op wrapper so envelopes
        # are captured (we record Quest needle ranks from P2 for free this way).
        with page_routing_sdpa_context(cache, RoutePolicy("none")):
            out = model.generate(**inputs, past_key_values=cache, max_new_tokens=1,
                                 do_sample=False, return_dict_in_generate=True,
                                 output_scores=True, use_cache=True)
    else:
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                             return_dict_in_generate=True, output_scores=True,
                             use_cache=True)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    first_logits = out.scores[0]  # [1, vocab]
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
        "k_cfg": cond.k_cfg_name,
    }

    if layout is not None:
        row["n_pages"] = layout.n_pages
        row["n_in_context_images"] = layout.n_in_context_images
        row["n_choice_images"] = layout.n_choice_images
        row["needle_page_idx"] = layout.needle_page_idx

    # For PageAware cache + sparse/formatbook routes, capture routing log.
    if cache is not None and isinstance(cache, PageAwareFakeQuantKVCache):
        rl = cache.routing_log
        if rl:
            # Per-layer needle-in-active list
            needle_hits = [v.get("needle_in_active") for v in rl.values()]
            row["needle_in_active_per_layer"] = needle_hits
            row["needle_in_active_layer_mean"] = (
                sum(1 for h in needle_hits if h) / max(1, len(needle_hits))
            )
            # Per-layer needle rank (when scores were computed, i.e. Quest / FormatBook)
            ranks = [v.get("needle_rank") for v in rl.values()]
            row["needle_rank_per_layer"] = ranks
            valid_ranks = [r for r in ranks if r is not None]
            if valid_ranks:
                row["needle_rank_median"] = float(np.median(valid_ranks))
                row["needle_rank_mean"] = float(np.mean(valid_ranks))
            # Active count
            n_active = [len(v.get("active_routable_pages", [])) for v in rl.values()]
            n_cold = [len(v.get("cold_routable_pages", [])) for v in rl.values()]
            row["active_routable_pages_layer_mean"] = float(np.mean(n_active)) if n_active else 0
            row["cold_routable_pages_layer_mean"] = float(np.mean(n_cold)) if n_cold else 0
            total_routable = (n_active[0] + n_cold[0]) if n_active else 0
            row["page_read_fraction"] = (
                float(np.mean(n_active)) / total_routable if total_routable else 1.0
            )

        # ALWAYS also compute Quest needle rank from envelopes (independent of route).
        # This gives us "needle hit rate from P2 envelopes" for free.
        from quest_scorer import quest_scores_for_layer
        if layout is not None and layout.needle_page_idx is not None and cache.envelopes:
            # We can't reconstruct Q here post-hoc; instead, if route was already Quest,
            # the routing_log already has it. Otherwise we leave the rank fields null.
            pass

    return row


# ---------------- driver ----------------

def _append_progress(progress_log: Path, line: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_log, "a") as f:
        f.write(f"[{ts}] {line}\n")
        f.flush()


def write_summary_md(jsonl_path: Path, out_md: Path) -> None:
    """Aggregate JSONL into a current-state summary table."""
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
    lines = [f"# Exp P summary — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append(f"total rows: {len(rows)} across {len(by_cond)} conditions\n")
    lines.append("| condition | n | acc | needle_hit_layer_mean | page_read_frac | latency_ms |")
    lines.append("|---|---|---|---|---|---|")
    for cond_name in sorted(by_cond.keys()):
        rs = by_cond[cond_name]
        n = len(rs)
        acc = sum(1 for r in rs if r.get("is_correct")) / max(1, n)
        nhit = [r.get("needle_in_active_layer_mean") for r in rs
                if r.get("needle_in_active_layer_mean") is not None]
        nhit_str = f"{np.mean(nhit):.3f}" if nhit else "—"
        prf = [r.get("page_read_fraction") for r in rs
               if r.get("page_read_fraction") is not None]
        prf_str = f"{np.mean(prf):.3f}" if prf else "—"
        lat = [r.get("latency_ms", 0) for r in rs]
        lat_str = f"{np.mean(lat):.0f}" if lat else "—"
        lines.append(f"| {cond_name} | {n} | {acc:.3f} | {nhit_str} | {prf_str} | {lat_str} |")
    out_md.write_text("\n".join(lines) + "\n")


def run_condition_on_items(model, processor, items: list[MMNiahItem],
                           cond: CondSpec, k_cfg_obj,
                           num_layers: int, num_kv_heads: int,
                           answer_ids: list[int],
                           out_jsonl: Path, progress_log: Path,
                           progress_every: int = 10, summary_every: int = 25,
                           summary_callback=None,
                           skip_ids: Optional[set] = None,
                           max_pixels_context: int = 144 * 144,
                           max_pixels_choices: int = 224 * 224) -> None:
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
            except torch.cuda.OutOfMemoryError as e:
                _append_progress(progress_log,
                                 f"WARN [{cond.name}] OOM on item {it.id} seq_len_est={it.context_length} -- skipping")
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
            # Memory hygiene every 5 items
            if n_done % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if n_done % progress_every == 0 or i == n - 1:
                elapsed = time.perf_counter() - t_start
                rate = elapsed / max(1, n_done)
                eta = rate * (n - i - 1)
                line = (f"[{cond.name}] {n_done}/{n - len(skip_ids)} acc={n_correct/n_done:.3f} "
                        f"seq_len={row['seq_len']} latency={row['latency_ms']:.0f}ms "
                        f"elapsed={timedelta(seconds=int(elapsed))} ETA={timedelta(seconds=int(eta))}")
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
    """IDs already scored for this condition (for resume)."""
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--calib-npz", type=Path,
                    default=CALIBRATION_DIR / "expP_mmniah_kcalib_Qwen2.5-VL-7B-Instruct_seed0.npz",
                    help="MM-NIAH-calibrated F9 outlier indices. Recomputed by "
                         "expP_calibrate.py on cal-100, not reused from LVB.")
    ap.add_argument("--out-jsonl", type=Path, default=RESULTS_DIR / "expP_rollouts.jsonl")
    ap.add_argument("--out-summary", type=Path, default=RESULTS_DIR / "expP_summary.md")
    ap.add_argument("--include-stretch", action="store_true",
                    help="Also run P3b/P4b stretch conditions after primary")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Subset of condition names to run (default: all primary)")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-run items already in the JSONL")
    ap.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--min-num-images", type=int, default=0,
                    help="Filter eval items to those with num_images >= N "
                         "(sharpens routing budget; 0 = no filter)")
    ap.add_argument("--use-full-pool", action="store_true",
                    help="Ignore eval split; use all items (minus cal) matching "
                         "--min-num-images. Used for the multi-image follow-up where "
                         "the original n=190 eval has only ~47 items with ≥8 images.")
    ap.add_argument("--max-pixels-context", type=int, default=144 * 144,
                    help="max_pixels cap for in-context (haystack) images")
    ap.add_argument("--max-pixels-choices", type=int, default=224 * 224,
                    help="max_pixels cap for the 4 A/B/C/D choice images")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_log = args.out_jsonl.with_name(args.out_jsonl.stem + ".progress.log")
    _append_progress(progress_log, f"=== launch args={vars(args)} ===")

    # Items + split
    items = load_all_items()
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
        print(f"min_num_images filter: {before} -> {len(eval_items)} items "
              f"(num_images >= {args.min_num_images})", flush=True)
    eval_items = eval_items[:args.n_items]
    print(f"loaded {len(items)} items; eval split {len(eval_items)} of {len(split['eval'])}", flush=True)
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

    # Build conditions
    primary = primary_conditions()
    if args.include_stretch:
        all_conds = primary + stretch_conditions()
    else:
        all_conds = primary
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

    # Periodic summary callback
    def summary_cb(jsonl_path: Path):
        write_summary_md(jsonl_path, args.out_summary)

    # Run
    t_overall = time.perf_counter()
    for cond in all_conds:
        skip = existing_completion_ids(args.out_jsonl, cond.name) if not args.no_resume else set()
        kcfg = cond_to_kcfg[cond.name]
        run_condition_on_items(
            model, processor, eval_items, cond, kcfg,
            num_layers=num_layers, num_kv_heads=num_kv_heads,
            answer_ids=answer_ids,
            out_jsonl=args.out_jsonl, progress_log=progress_log,
            summary_callback=summary_cb, skip_ids=skip,
            max_pixels_context=args.max_pixels_context,
            max_pixels_choices=args.max_pixels_choices,
        )
    total_wall = time.perf_counter() - t_overall
    _append_progress(progress_log, f"=== overall wall={timedelta(seconds=int(total_wall))} ===")
    summary_cb(args.out_jsonl)
    print(f"\ndone. summary -> {args.out_summary}", flush=True)


if __name__ == "__main__":
    main()
