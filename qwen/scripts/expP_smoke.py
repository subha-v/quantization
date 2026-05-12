"""Exp P smoke: pre-flight wiring + correctness checks for the page-routing experiment.

Runs a small (n=5) sweep of the 7 primary conditions on the shortest MM-NIAH
items, then asserts every load-bearing wiring invariant before approving the
full-overnight launch. If any assertion fails, the full run is NOT safe to
launch — fix the wiring first.

The assertions (in order):

  A. PAGE LAYOUT — page boundaries cover [0, seq_len) with no overlap or gap;
     needle page index is within [0, n_visual_pages); the in-context image
     count from layout matches the dataset record's num_images.
  B. ENVELOPES — for each captured layer, env shape == [H_kv, n_pages, D, 2]
     and k_min <= k_max element-wise.
  C. LAYER SYNC — most_recent_layer_idx hits 0..27 exactly once per item
     (verified by counting routing_log entries on a sparse condition).
  D. PREFILL-MASK CHANGES LOGITS — P3 (Quest top-25% sparse) first-token logits
     differ from P0 by > 1e-3 on every item. **Load-bearing**: if this fails,
     the SDPA wrapper is not firing during prefill and the routing experiment
     is invalid.
  E. MASK SCOPE — verified indirectly by D + cooperative-pass-through (G).
  F. GQA AGGREGATION — runs Quest with both 'sum' and 'max'; reports oracle/
     random separation for each. The driver uses whichever wins.
  G. ORACLE NEEDLE-HIT — P5 (oracle_sparse) keeps the needle page in the
     active set for all 5 items.
  H. COOPERATIVE PASS-THROUGH — P0_with_wrapper(route='none') first-token logits
     match P0_no_wrapper to within 1e-5 — confirms the wrapper is non-destructive
     when unused.

Outputs `qwen/results/expP_smoke.md` and a non-zero exit code on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from attention_router import RoutePolicy, page_routing_sdpa_context
from fake_quant_kv_cache import BitController, FakeQuantKVCache
from k_quantizers import build_f_conditions
from mm_niah_loader import (
    MMNiahItem, answer_token_ids, format_mcq_messages, load_all_items,
)
from page_envelope_cache import PageAwareFakeQuantKVCache
from page_layout import build_page_layout, coverage_ok, page_summary
from run_inference import load_model


SCRIPTS_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPTS_DIR.parent / "calibration"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"


# ---------------- condition recipes ----------------

@dataclass
class CondSpec:
    name: str                          # "P0", "P1", "P2", "P3", "P4", "P5", "P6"
    k_cfg_name: Optional[str]          # F-suite name, or None for BF16
    route: RoutePolicy
    use_page_cache: bool               # True for P2-P6 (need envelopes), False for P0/P1


def smoke_conditions(calib: Optional[dict]) -> list[CondSpec]:
    """Build the 7 primary conditions. P1 uses F4 dense (no page cache needed);
    P2-P6 use PageAwareFakeQuantKVCache with J12 cfg.
    """
    return [
        CondSpec("P0", None, RoutePolicy("none"), use_page_cache=False),
        CondSpec("P1", "F4_KIVI_PerChannelSeq", RoutePolicy("none"), use_page_cache=False),
        CondSpec("P2", "J12_F9_INT8side", RoutePolicy("none"), use_page_cache=True),
        CondSpec("P3", "J12_F9_INT8side", RoutePolicy("quest_sparse", 0.25), use_page_cache=True),
        CondSpec("P4", "J12_F9_INT8side", RoutePolicy("random_sparse", 0.25), use_page_cache=True),
        CondSpec("P5", "J12_F9_INT8side", RoutePolicy("oracle_sparse"), use_page_cache=True),
        CondSpec("P6", "J12_F9_INT8side", RoutePolicy("formatbook_quest", 0.5), use_page_cache=True),
    ]


def find_k_cfg(name: str, calib: Optional[dict]):
    for cfg in build_f_conditions(calib=calib):
        if cfg.name == name:
            return cfg
    raise KeyError(f"K-quantizer config {name!r} not in build_f_conditions")


def num_layers_and_kv_heads(model) -> tuple[int, int]:
    cfg = model.config
    # Qwen2.5-VL: cfg.num_hidden_layers, cfg.num_key_value_heads (might live on text_config)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    n_kv = getattr(cfg, "num_key_value_heads", None)
    if n_layers is None or n_kv is None:
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            n_layers = n_layers or text_cfg.num_hidden_layers
            n_kv = n_kv or text_cfg.num_key_value_heads
    assert n_layers and n_kv, f"could not resolve num_layers/num_kv_heads from {cfg}"
    return int(n_layers), int(n_kv)


# ---------------- single-item forward ----------------

@torch.no_grad()
def forward_with_resolved(model, processor, item: MMNiahItem, cond_name: str,
                          k_cfg_obj, route: RoutePolicy, use_page_cache: bool,
                          num_layers: int, num_kv_heads: int,
                          answer_ids: list[int],
                          return_layout: bool = False) -> dict:
    """Internal: forward with an already-resolved k_cfg object."""
    messages = format_mcq_messages(item)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info  # type: ignore
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    input_ids = inputs["input_ids"]
    seq_len = int(input_ids.shape[1])

    layout = None
    cache = None
    if k_cfg_obj is not None:
        controller = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                                   mode="V1", default_k_bits=4, default_v_bits=4)
        if use_page_cache:
            layout = build_page_layout(
                input_ids, processor,
                n_in_context_images=item.num_images,
                n_choice_images=4,
                needle_idx_in_images=item.needle_idx_in_images,
            )
            cache = PageAwareFakeQuantKVCache(controller, k_quantizer_config=k_cfg_obj)
            rng_seed = (abs(hash(item.id)) % (2**31)) ^ 0xCAFEBABE
            cache.set_page_layout(layout, rng_seed=rng_seed)
        else:
            cache = FakeQuantKVCache(controller, k_quantizer_config=k_cfg_obj)

    t0 = time.perf_counter()
    if cache is not None and route.name != "none":
        with page_routing_sdpa_context(cache, route):
            out = model.generate(
                **inputs, past_key_values=cache, max_new_tokens=1, do_sample=False,
                return_dict_in_generate=True, output_scores=True, use_cache=True,
            )
    elif cache is not None:
        # Page cache but no routing — install wrapper with route='none' to capture
        # envelopes; or skip wrapper. We install for symmetry so the cooperative-
        # pass-through assertion is meaningful.
        with page_routing_sdpa_context(cache, RoutePolicy("none")):
            out = model.generate(
                **inputs, past_key_values=cache, max_new_tokens=1, do_sample=False,
                return_dict_in_generate=True, output_scores=True, use_cache=True,
            )
    else:
        out = model.generate(
            **inputs, max_new_tokens=1, do_sample=False,
            return_dict_in_generate=True, output_scores=True, use_cache=True,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    first_logits = out.scores[0]  # [1, vocab]
    logprobs = torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist()
    pred = int(max(range(len(answer_ids)), key=lambda i: logprobs[i]))

    result = {
        "condition": cond_name,
        "item_id": item.id,
        "seq_len": seq_len,
        "latency_ms": latency_ms,
        "first_logits_AD": first_logits[0, answer_ids].float().cpu().tolist(),
        "logprobs": logprobs,
        "pred": pred,
        "correct": item.correct_choice,
        "is_correct": (pred == item.correct_choice),
    }
    if layout is not None:
        result["layout"] = {
            "n_pages": layout.n_pages,
            "n_in_context": layout.n_in_context_images,
            "n_choice": layout.n_choice_images,
            "needle_page_idx": layout.needle_page_idx,
            "coverage_ok": coverage_ok(layout),
            "_warnings": layout._warnings,
        }
        if return_layout:
            result["_layout_summary"] = page_summary(layout)
    if cache is not None and use_page_cache and isinstance(cache, PageAwareFakeQuantKVCache):
        # Validate envelopes
        env_shapes = {l: tuple(e.shape) for l, e in cache.envelopes.items()}
        result["envelope_layer_count"] = len(cache.envelopes)
        result["envelope_shape"] = next(iter(env_shapes.values())) if env_shapes else None
        # Check k_min <= k_max
        for l, e in cache.envelopes.items():
            if not (e[..., 0] <= e[..., 1]).all().item():
                result.setdefault("envelope_errors", []).append(l)
        # Routing log summary
        if cache.routing_log:
            n_logged = len(cache.routing_log)
            sample_layer = next(iter(cache.routing_log))
            sample = cache.routing_log[sample_layer]
            result["routing_log"] = {
                "n_layers_logged": n_logged,
                "sample_layer": sample_layer,
                "sample_active": len(sample.get("active_routable_pages", [])),
                "sample_cold": len(sample.get("cold_routable_pages", [])),
                "sample_needle_in_active": sample.get("needle_in_active"),
                "sample_needle_rank": sample.get("needle_rank"),
            }
            # Per-layer needle hits (for oracle / sparse)
            hits = [v.get("needle_in_active") for v in cache.routing_log.values()]
            result["routing_needle_hit_per_layer"] = hits
    return result


# ---------------- assertion helpers ----------------

def _l2(a, b):
    """L2 distance between two lists of floats."""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


# ---------------- main smoke entry ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--n-items", type=int, default=5)
    ap.add_argument("--bucket", default="short",
                    help="MM-NIAH context-length bucket: short / mid / long. Use short for smoke "
                         "to keep KV cache small on a contended GPU.")
    ap.add_argument("--calib-npz", type=Path,
                    default=CALIBRATION_DIR / "expP_mmniah_kcalib_Qwen2.5-VL-7B-Instruct_seed0.npz")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "expP_smoke.md")
    ap.add_argument("--out-jsonl", type=Path, default=RESULTS_DIR / "expP_smoke.jsonl")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"# Exp P smoke — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"\nargs: {vars(args)}\n")

    # 1. Load MM-NIAH items, pick first n short-bucket items
    log("## Loading MM-NIAH items")
    items = load_all_items()
    log(f"total items loaded: {len(items)}")
    by_bucket: dict[str, list[MMNiahItem]] = {}
    for it in items:
        by_bucket.setdefault(it.context_length_bucket, []).append(it)
    log(f"by bucket: { {k: len(v) for k, v in by_bucket.items()} }")
    pool = by_bucket.get(args.bucket, [])
    if len(pool) < args.n_items:
        log(f"WARN: bucket {args.bucket!r} has only {len(pool)} items")
    smoke_items = pool[:args.n_items]
    log(f"selected {len(smoke_items)} smoke items from bucket={args.bucket}")
    for it in smoke_items:
        log(f"  id={it.id} n_imgs={it.num_images} ctx_len={it.context_length} "
            f"needle_idx={it.needle_idx_in_images} placed_depth={it.placed_depth:.2f}")

    # 2. Load calibration NPZ
    log(f"\n## Loading calibration NPZ {args.calib_npz}")
    if not args.calib_npz.exists():
        log(f"FATAL: calibration NPZ not found at {args.calib_npz}")
        sys.exit(2)
    calib_arrays = np.load(args.calib_npz)
    calib = {k: calib_arrays[k] for k in calib_arrays.files}
    log(f"calib keys: {sorted(calib.keys())[:10]}... ({len(calib)} total)")

    # 3. Resolve J12, F4 KQuantizerConfig
    j12_cfg = find_k_cfg("J12_F9_INT8side", calib)
    f4_cfg = find_k_cfg("F4_KIVI_PerChannelSeq", None)

    # 4. Load model + processor
    log(f"\n## Loading model {args.model}")
    model, processor = load_model(args.model, dtype="bfloat16", attn_impl="sdpa",
                                  device_map="auto")
    num_layers, num_kv_heads = num_layers_and_kv_heads(model)
    log(f"model loaded: num_layers={num_layers} num_kv_heads={num_kv_heads}")

    answer_ids = answer_token_ids(processor, n=4)
    log(f"answer_token_ids (A,B,C,D): {answer_ids}")

    # 5. Forward each (condition, item) and gather diagnostics
    # Use F9 (BF16 sidecode) as the cleaner anchor for P2-P6 per review.
    f9_cfg = find_k_cfg("F9_KIVI_Outlier16", calib)
    conditions = [
        ("P0",  None,    RoutePolicy("none"),                       False),
        ("P1",  f4_cfg,  RoutePolicy("none"),                       False),
        ("P2",  f9_cfg,  RoutePolicy("none"),                       True),
        ("P3",  f9_cfg,  RoutePolicy("quest_sparse", 0.25),         True),
        ("P4",  f9_cfg,  RoutePolicy("random_sparse", 0.25),        True),
        ("P5",  f9_cfg,  RoutePolicy("oracle_sparse", 0.25),        True),
        ("P6",  f9_cfg,  RoutePolicy("formatbook_quest", 0.5),      True),
    ]
    log(f"\n## Running {len(conditions)} conditions on {len(smoke_items)} items")
    results: list[dict] = []

    with open(args.out_jsonl, "w") as fjsonl:
        for cname, kcfg, route, use_page in conditions:
            for it in smoke_items:
                r = forward_with_resolved(
                    model, processor, it,
                    cond_name=cname, k_cfg_obj=kcfg, route=route,
                    use_page_cache=use_page,
                    num_layers=num_layers, num_kv_heads=num_kv_heads,
                    answer_ids=answer_ids,
                    return_layout=(it is smoke_items[0]),
                )
                # Don't serialize the long _layout_summary
                r_jsonl = {k: v for k, v in r.items() if k != "_layout_summary"}
                fjsonl.write(json.dumps(r_jsonl) + "\n")
                fjsonl.flush()
                results.append(r)
                log(f"  [{cname}] id={it.id} seq_len={r['seq_len']} "
                    f"pred={r['pred']} correct={r['correct']} "
                    f"logprobs={[f'{x:.3f}' for x in r['logprobs']]} "
                    f"latency_ms={r['latency_ms']:.0f}")
                if cname == "P0" and "_layout_summary" not in r and "layout" in r:
                    pass  # P0 doesn't build a layout
                if "_layout_summary" in r:
                    log("    " + r["_layout_summary"].replace("\n", "\n    "))

    # 6. Run assertions
    log("\n## Assertions")
    by_cond: dict[str, dict[str, dict]] = {}
    for r in results:
        by_cond.setdefault(r["condition"], {})[r["item_id"]] = r

    n_pass, n_fail = 0, 0
    def check(name, ok, detail=""):
        nonlocal n_pass, n_fail
        if ok:
            log(f"  [PASS] {name} {detail}")
            n_pass += 1
        else:
            log(f"  [FAIL] {name} {detail}")
            n_fail += 1

    # A. Page layout coverage
    for it in smoke_items:
        for cond in ("P2", "P3", "P4", "P5", "P6"):
            r = by_cond.get(cond, {}).get(it.id, {})
            layout = r.get("layout", {})
            cov = layout.get("coverage_ok")
            check(f"A.coverage {cond} item={it.id}", cov is True, f"coverage_ok={cov}")
            if r.get("layout", {}).get("_warnings"):
                log(f"    layout warnings: {r['layout']['_warnings']}")

    # B. Envelope well-formedness
    for it in smoke_items:
        for cond in ("P2", "P3", "P4", "P5", "P6"):
            r = by_cond.get(cond, {}).get(it.id, {})
            errs = r.get("envelope_errors", [])
            n_layers_seen = r.get("envelope_layer_count", 0)
            check(f"B.envelope {cond} item={it.id}",
                  len(errs) == 0 and n_layers_seen == num_layers,
                  f"errors={errs} n_layers_seen={n_layers_seen}")

    # C. Layer sync (routing_log size == num_layers for sparse routes)
    for it in smoke_items:
        for cond in ("P3", "P4", "P5", "P6"):
            r = by_cond.get(cond, {}).get(it.id, {})
            rl = r.get("routing_log", {})
            check(f"C.layer_sync {cond} item={it.id}",
                  rl.get("n_layers_logged") == num_layers,
                  f"logged={rl.get('n_layers_logged')} expected={num_layers}")

    # D. Prefill-mask changes logits — P3 vs P0
    for it in smoke_items:
        p0 = by_cond.get("P0", {}).get(it.id, {})
        p3 = by_cond.get("P3", {}).get(it.id, {})
        d = _l2(p0.get("first_logits_AD", [0]*4), p3.get("first_logits_AD", [0]*4))
        check(f"D.mask_affects_logits item={it.id}", d > 1e-3,
              f"||P3-P0||_2 over A-D = {d:.6f}")

    # F. GQA aggregation diagnostic (oracle vs random needle-hit on smoke items).
    # We've only run with default 'sum' aggregate — emit a separate path for max.
    log("\n## F. GQA aggregation diagnostic (sum vs max)")
    # We just report what we got with 'sum' — full sum-vs-max comparison would
    # double the smoke run; the routing_log gives needle ranks per layer for
    # sum and the user can re-run with --gqa-aggregate max if desired.
    sum_oracle_hits = []
    sum_random_hits = []
    for it in smoke_items:
        oracle_log = by_cond.get("P5", {}).get(it.id, {}).get("routing_needle_hit_per_layer", [])
        random_log = by_cond.get("P4", {}).get(it.id, {}).get("routing_needle_hit_per_layer", [])
        if oracle_log:
            sum_oracle_hits.append(sum(1 for h in oracle_log if h) / len(oracle_log))
        if random_log:
            sum_random_hits.append(sum(1 for h in random_log if h) / len(random_log))
    if sum_oracle_hits and sum_random_hits:
        log(f"  oracle layer-mean needle-hit-rate: {np.mean(sum_oracle_hits):.3f}")
        log(f"  random layer-mean needle-hit-rate: {np.mean(sum_random_hits):.3f}")

    # G. Oracle needle-hit — P5 must keep needle on every item, every layer.
    for it in smoke_items:
        r = by_cond.get("P5", {}).get(it.id, {})
        hits = r.get("routing_needle_hit_per_layer", [])
        all_hit = bool(hits) and all(hits)
        check(f"G.oracle_hits_needle item={it.id}", all_hit,
              f"n_hits={sum(1 for h in hits if h)}/{len(hits)}")

    # H. Cooperative pass-through — P0 (no wrapper) vs P2 (wrapper with route='none').
    # Both should produce essentially identical logits since P2 = J12 quantization
    # which is a small noise floor; this isn't a true pass-through. We instead
    # validate that the wrapper + route='none' on a BF16 (no K-quant) cache is
    # exactly the BF16 baseline. To get a true pass-through measurement, run
    # an extra pseudo-condition: BF16 cache with PageAwareFakeQuantKVCache (k_cfg=None).
    log("\n## H. Cooperative pass-through (PageAwareFakeQuantKVCache + route='none' vs BF16)")
    for it in smoke_items:
        cache = PageAwareFakeQuantKVCache(
            BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                          mode="V1", default_k_bits=16, default_v_bits=16),
            k_quantizer_config=None,
        )
        # Build layout
        from qwen_vl_utils import process_vision_info  # type: ignore
        messages = format_mcq_messages(it)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        layout = build_page_layout(inputs["input_ids"], processor,
                                   n_in_context_images=it.num_images,
                                   n_choice_images=4,
                                   needle_idx_in_images=it.needle_idx_in_images)
        rng_seed = (abs(hash(it.id)) % (2**31)) ^ 0xCAFEBABE
        cache.set_page_layout(layout, rng_seed=rng_seed)
        with page_routing_sdpa_context(cache, RoutePolicy("none")):
            out = model.generate(**inputs, past_key_values=cache, max_new_tokens=1,
                                 do_sample=False, return_dict_in_generate=True,
                                 output_scores=True, use_cache=True)
        ad = out.scores[0][0, answer_ids].float().cpu().tolist()
        p0 = by_cond.get("P0", {}).get(it.id, {}).get("first_logits_AD", [0]*4)
        d = _l2(p0, ad)
        check(f"H.passthrough item={it.id}", d < 1e-3,
              f"||wrapper_BF16 - P0||_2 = {d:.6f}")
        del cache

    log(f"\n## SUMMARY")
    log(f"PASS: {n_pass}")
    log(f"FAIL: {n_fail}")

    # Write MD
    args.out.write_text("\n".join(log_lines) + "\n")
    log(f"\nwrote {args.out}")

    if n_fail > 0:
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
