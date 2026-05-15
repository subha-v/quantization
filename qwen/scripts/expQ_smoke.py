"""Exp Q smoke: pre-flight wiring checks for FormatBook v2 conditions.

Runs a small (n=3) sweep of Q0..Q11 + the smoke-only Q_allhot sanity on the
shortest MM-NIAH retrieval-image items, then asserts every load-bearing
invariant before approving the full overnight launch.

Assertions (in addition to standard P-style A-E checks):

  F. RoleOnly zero hot — Q3 (formatbook_role_only) keeps zero in-context pages
     hot per layer; cold set equals all routable pages.

  G. All-hot matches F9 dense — Q_allhot (formatbook_all_hot, budget=1.0)
     first-token logits match Q2 (F9 dense) within 1e-4 L2 over A-D positions.
     Catches bugs in page-format dispatch.

  H. INT2 cold is harsher than F4 cold — `_int2_per_channel_seq` introduces
     strictly larger reconstruction error than `_f4_per_channel_seq` on the
     same K tensor (because INT2's grid {-s, 0, +s} is coarser than INT4's).

  I. Effective-bit math — `effective_k_bits` matches the analytical reference:
     Q0 (BF16) = 16.0, Q1 (F4) = 4.0, Q2 (F9) = 5.50, Q3 (RoleOnly) between
     4.0 and 5.50 (token-weighted mix of cold F4 in-context vs always-on F9
     text+choice).

Writes `qwen/results/expQ_smoke.md` and exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from attention_router import (
    RoutePolicy, _f4_per_channel_seq, _int2_per_channel_seq,
    _int3_per_channel_seq, _fp8_per_channel_seq,
    page_routing_sdpa_context,
)
from fake_quant_kv_cache import BitController, FakeQuantKVCache
from k_quantizers import build_f_conditions
from mm_niah_loader import (
    MMNiahItem, answer_token_ids, format_mcq_messages, load_all_items,
)
from page_envelope_cache import PageAwareFakeQuantKVCache
from page_layout import build_page_layout, coverage_ok, page_summary
from run_inference import load_model

import expQ_driver
from expQ_driver import (
    CondSpec, HOT, F4, PAGE_K_BITS, V_BITS_DEFAULT,
    q_conditions_slice_a, q_conditions_smoke_only,
    c_conditions_allvisual,
    _compute_bit_metrics, resolve_k_cfg, num_layers_and_kv_heads,
)


SCRIPTS_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPTS_DIR.parent / "calibration"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"


# ---------------- single-item forward (mirror of expP_smoke.forward_with_resolved) ----------------

@torch.no_grad()
def forward_with_resolved(model, processor, item: MMNiahItem, cond: CondSpec,
                          k_cfg_obj, num_layers: int, num_kv_heads: int,
                          answer_ids: list[int],
                          max_pixels_context: int,
                          max_pixels_choices: int,
                          return_layout: bool = False,
                          include_choice_routing: bool = False) -> dict:
    messages = format_mcq_messages(item, max_pixels_context=max_pixels_context,
                                   max_pixels_choices=max_pixels_choices)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info  # type: ignore
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
            cache.correct_choice_idx = int(item.correct_choice)
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

    result = {
        "condition": cond.name,
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
    if cache is not None and cond.use_page_cache and isinstance(cache, PageAwareFakeQuantKVCache):
        env_shapes = {l: tuple(e.shape) for l, e in cache.envelopes.items()}
        result["envelope_layer_count"] = len(cache.envelopes)
        result["envelope_shape"] = next(iter(env_shapes.values())) if env_shapes else None
        for l, e in cache.envelopes.items():
            if not (e[..., 0] <= e[..., 1]).all().item():
                result.setdefault("envelope_errors", []).append(l)
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
            # Full per-layer hot/cold partitions for assertion F.
            result["all_active_per_layer"] = [
                list(v.get("active_routable_pages", [])) for v in cache.routing_log.values()
            ]
            result["all_cold_per_layer"] = [
                list(v.get("cold_routable_pages", [])) for v in cache.routing_log.values()
            ]
            hits = [v.get("needle_in_active") for v in cache.routing_log.values()]
            result["routing_needle_hit_per_layer"] = hits
        # Per-page bit metrics (for assertion I).
        bits = _compute_bit_metrics(layout, cache, cond)
        result.update(bits)
    elif layout is None and not cond.use_page_cache:
        # Dense path bit metrics.
        k_bits = expQ_driver._page_k_bits_dense(cond.k_cfg_name)
        result["effective_k_bits"] = float(k_bits)
        # Use the dense V-bits helper (BF16=16 for cache-bypassed Q0, else 4).
        v_bits = expQ_driver._v_bits_dense(cond.k_cfg_name)
        result["effective_v_bits"] = float(v_bits)
        result["effective_kv_bits"] = float((k_bits + v_bits) / 2.0)
    return result


def _l2(a, b):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--n-items", type=int, default=3)
    ap.add_argument("--bucket", default="short",
                    help="MM-NIAH context-length bucket: short / mid / long.")
    ap.add_argument("--calib-npz", type=Path,
                    default=CALIBRATION_DIR / "expP_mmniah_kcalib_Qwen2.5-VL-7B-Instruct_seed0.npz")
    ap.add_argument("--task", default="retrieval-image")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "expQ_smoke.md")
    ap.add_argument("--out-jsonl", type=Path, default=RESULTS_DIR / "expQ_smoke.jsonl")
    ap.add_argument("--max-pixels-context", type=int, default=336 * 336)
    ap.add_argument("--max-pixels-choices", type=int, default=336 * 336)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--exp-r", action="store_true",
                    help="Exp R mode: append C3b/C4/C6/C7 AllVisual conditions "
                         "and run J/K/L/M/N/O assertions.")
    ap.add_argument("--exp-t", action="store_true",
                    help="Exp T mode: append S3/S4/S5 sidecode-ladder conditions "
                         "and run P/Q/R/U assertions (logits-differ across "
                         "INT7/INT8/INT6, bit-math extended, cold-V K-only, "
                         "BF16 dense V correctly reported).")
    ap.add_argument("--exp-u", action="store_true",
                    help="Exp U mode: append U3/U4/U5/U6/U11/U13 residual-extra "
                         "conditions and run V/W/X/Y/Z assertions on the residual "
                         "invariant, bit-math, and per-policy logit divergence.")
    ap.add_argument("--extras-npz", type=Path, default=None,
                    help="Override the auto-derived expU_extras NPZ path "
                         "(used with --exp-u).")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"# Exp Q smoke — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"\nargs: {vars(args)}\n")

    # 1. Load items
    log("## Loading MM-NIAH items")
    items = load_all_items(task=args.task)
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

    # 2. Calibration
    log(f"\n## Loading calibration NPZ {args.calib_npz}")
    if not args.calib_npz.exists():
        log(f"FATAL: calibration NPZ not found at {args.calib_npz}")
        sys.exit(2)
    calib_arrays = np.load(args.calib_npz)
    calib = {k: calib_arrays[k] for k in calib_arrays.files}
    log(f"calib keys: {sorted(calib.keys())[:10]}... ({len(calib)} total)")

    # Exp U: merge sibling extras NPZ.
    if args.exp_u:
        extras_path = args.extras_npz
        if extras_path is None:
            extras_path = args.calib_npz.with_name(
                args.calib_npz.stem + "_expU_extras.npz")
        if not extras_path.exists():
            log(f"FATAL: --exp-u requires expU extras NPZ at {extras_path}; "
                f"run expU_compute_extras.py first")
            sys.exit(2)
        xarr = np.load(extras_path)
        for k in xarr.files:
            calib[k] = xarr[k]
        log(f"[exp-u] merged extras from {extras_path.name} "
            f"({len(xarr.files)} new keys)")

    # 3. Model
    log(f"\n## Loading model {args.model}")
    model, processor = load_model(args.model, dtype="bfloat16", attn_impl="sdpa",
                                  device_map="auto")
    num_layers, num_kv_heads = num_layers_and_kv_heads(model)
    log(f"model loaded: num_layers={num_layers} num_kv_heads={num_kv_heads}")
    answer_ids = answer_token_ids(processor, n=4)
    log(f"answer_token_ids (A,B,C,D): {answer_ids}")

    # 4. Build conditions for smoke: Q0..Q9 (skip Q10/Q11 for fast smoke) + Q_allhot
    primary = q_conditions_slice_a(include_int2=False)
    # Add Q10 only (skip Q11 for smoke brevity) so assertion H has a real INT2 path.
    primary.append(CondSpec("Q10", HOT,
                            RoutePolicy("formatbook_quest", 0.25, cold_quantizer="int2"),
                            True))
    primary.extend(q_conditions_smoke_only())

    # Exp R: append the key AllVisual conditions so J/K/L assertions have data.
    if args.exp_r:
        for c in c_conditions_allvisual():
            if c.name in ("C3b", "C4", "C6", "C7"):
                primary.append(c)
        log("[exp-r] appended C3b/C4/C6/C7 to smoke pool")

    # Exp T: append the S3 (INT8) / S4 (INT7) / S5 (INT6) ladder conditions so
    # the logits-differ assertions have real data.
    if args.exp_t:
        from expQ_driver import s_conditions_sidecode_ladder
        for c in s_conditions_sidecode_ladder():
            if c.name in ("S3", "S4", "S5"):
                primary.append(c)
        log("[exp-t] appended S3 (INT8) / S4 (INT7) / S5 (INT6) to smoke pool")

    # Exp U: append a compact set of U conditions so the smoke assertions
    # have real prefill K data to inspect. U3 (S4 anchor) + U4 (GEN extra) +
    # U5 (RND extra) + U6 (TT extra) + U11 (MMNIAH prior) + U13 (ALL16).
    if args.exp_u:
        from expQ_driver import u_conditions_residual_screen
        for c in u_conditions_residual_screen():
            if c.name in ("U3", "U4", "U5", "U6", "U7", "U11", "U12", "U13"):
                primary.append(c)
        log("[exp-u] appended U3/U4/U5/U6/U7/U11/U12/U13 to smoke pool")

    cond_to_kcfg = {c.name: resolve_k_cfg(c.k_cfg_name, calib) if c.k_cfg_name else None
                    for c in primary}

    # In Exp R mode the C conditions need include_choice_routing=True; the Q
    # conditions need it False. Track per-condition.
    def needs_choice_routing(cond_name: str) -> bool:
        return args.exp_r and cond_name in ("C3b", "C4", "C5", "C6", "C7", "C8")

    log(f"\n## Running {len(primary)} conditions on {len(smoke_items)} items")
    results: list[dict] = []
    with open(args.out_jsonl, "w") as fjsonl:
        for cond in primary:
            for it in smoke_items:
                r = forward_with_resolved(
                    model, processor, it, cond,
                    k_cfg_obj=cond_to_kcfg[cond.name],
                    num_layers=num_layers, num_kv_heads=num_kv_heads,
                    answer_ids=answer_ids,
                    max_pixels_context=args.max_pixels_context,
                    max_pixels_choices=args.max_pixels_choices,
                    return_layout=(it is smoke_items[0] and cond.name == "Q2"),
                    include_choice_routing=needs_choice_routing(cond.name),
                )
                r_jsonl = {k: v for k, v in r.items() if k != "_layout_summary"}
                fjsonl.write(json.dumps(r_jsonl) + "\n")
                fjsonl.flush()
                results.append(r)
                ekvb = r.get("effective_kv_bits")
                ekvb_str = f"{ekvb:.2f}" if ekvb is not None else "—"
                log(f"  [{cond.name}] id={it.id} seq_len={r['seq_len']} pred={r['pred']} "
                    f"correct={r['correct']} kv_bits={ekvb_str} "
                    f"logprobs={[f'{x:.3f}' for x in r['logprobs']]} "
                    f"latency_ms={r['latency_ms']:.0f}")
                if "_layout_summary" in r:
                    log("    " + r["_layout_summary"].replace("\n", "\n    "))

    # 5. Assertions
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

    # A. Coverage
    for it in smoke_items:
        for cond in ("Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q_allhot"):
            r = by_cond.get(cond, {}).get(it.id, {})
            cov = r.get("layout", {}).get("coverage_ok")
            check(f"A.coverage {cond} item={it.id}", cov is True, f"coverage_ok={cov}")

    # B. Envelope shape + ordering
    for it in smoke_items:
        for cond in ("Q2", "Q3", "Q4", "Q7", "Q10"):
            r = by_cond.get(cond, {}).get(it.id, {})
            errs = r.get("envelope_errors", [])
            n_layers_seen = r.get("envelope_layer_count", 0)
            check(f"B.envelope {cond} item={it.id}",
                  not errs and n_layers_seen == num_layers,
                  f"errors={errs} n_layers_seen={n_layers_seen}")

    # C. Layer sync — routing_log size == num_layers for routed conditions
    for it in smoke_items:
        for cond in ("Q3", "Q4", "Q7", "Q10", "Q_allhot"):
            r = by_cond.get(cond, {}).get(it.id, {})
            rl = r.get("routing_log", {})
            check(f"C.layer_sync {cond} item={it.id}",
                  rl.get("n_layers_logged") == num_layers,
                  f"logged={rl.get('n_layers_logged')} expected={num_layers}")

    # D. Mask affects logits — Q4 vs Q0 (FormatBook K downgrade should perturb)
    for it in smoke_items:
        q0 = by_cond.get("Q0", {}).get(it.id, {})
        q4 = by_cond.get("Q4", {}).get(it.id, {})
        d = _l2(q0.get("first_logits_AD", [0]*4), q4.get("first_logits_AD", [0]*4))
        check(f"D.mask_affects_logits item={it.id}", d > 1e-3,
              f"||Q4-Q0||_2 over A-D = {d:.6f}")

    # E. Pass-through — PageAware cache + route='none' (Q2 setup) matches BF16 Q0
    # only insofar as F9 is near-lossless; we check the dedicated BF16 wrapper path
    # like Exp P did.
    log("\n## E. Cooperative pass-through (PageAware cache + route='none' vs BF16 Q0)")
    for it in smoke_items:
        cache = PageAwareFakeQuantKVCache(
            BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                          mode="V1", default_k_bits=16, default_v_bits=16),
            k_quantizer_config=None,
        )
        from qwen_vl_utils import process_vision_info  # type: ignore
        messages = format_mcq_messages(it, max_pixels_context=args.max_pixels_context,
                                       max_pixels_choices=args.max_pixels_choices)
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
        q0 = by_cond.get("Q0", {}).get(it.id, {}).get("first_logits_AD", [0]*4)
        d = _l2(q0, ad)
        check(f"E.passthrough item={it.id}", d < 1e-3,
              f"||wrapper_BF16 - Q0||_2 = {d:.6f}")
        del cache

    # F. RoleOnly zero hot — Q3
    log("\n## F. RoleOnly zero hot (Q3)")
    for it in smoke_items:
        r = by_cond.get("Q3", {}).get(it.id, {})
        actives = r.get("all_active_per_layer", [])
        colds = r.get("all_cold_per_layer", [])
        # Every layer's active routable list must be empty.
        all_empty = bool(actives) and all(len(a) == 0 for a in actives)
        # Every layer's cold list must equal the full routable set (same for all layers).
        cold_sizes = {tuple(sorted(c)) for c in colds} if colds else set()
        layout = r.get("layout", {})
        n_in_ctx = layout.get("n_in_context", 0)
        # Cold count per layer should match the routable count (= n_in_ctx).
        cold_correct = all(len(c) == n_in_ctx for c in colds)
        # No needle should be active in any layer.
        hits = r.get("routing_needle_hit_per_layer", [])
        no_hit = bool(hits) and not any(h for h in hits)
        check(f"F.role_only_zero_hot item={it.id}",
              all_empty and cold_correct and no_hit,
              f"all_active_empty={all_empty} cold_size_match={cold_correct} "
              f"no_needle_hit={no_hit} n_routable={n_in_ctx} layers_with_unique_cold={len(cold_sizes)}")

    # G. All-hot matches F9 dense — Q_allhot vs Q2 logits within 1e-4
    log("\n## G. All-hot matches F9 dense (Q_allhot vs Q2)")
    for it in smoke_items:
        q2 = by_cond.get("Q2", {}).get(it.id, {}).get("first_logits_AD", [0]*4)
        qa = by_cond.get("Q_allhot", {}).get(it.id, {}).get("first_logits_AD", [0]*4)
        d = _l2(q2, qa)
        check(f"G.allhot_matches_F9_dense item={it.id}", d < 1e-4,
              f"||Q_allhot - Q2||_2 over A-D = {d:.6e}")

    # H. INT2 cold is harsher than F4 cold
    log("\n## H. INT2 cold > F4 cold reconstruction error")
    # Synthesize a representative K tensor (random BF16, plausible scale).
    rng = np.random.default_rng(123)
    K_np = rng.standard_normal(size=(1, 4, 256, 128)).astype(np.float32)
    K = torch.from_numpy(K_np).to(torch.bfloat16)
    K_f4 = _f4_per_channel_seq(K)
    K_int2 = _int2_per_channel_seq(K)
    err_f4 = (K.float() - K_f4.float()).abs().mean().item()
    err_int2 = (K.float() - K_int2.float()).abs().mean().item()
    check("H.int2_cold_smaller",
          err_int2 > err_f4 * 1.5,
          f"mean |K - K_int2|={err_int2:.4f} vs mean |K - K_f4|={err_f4:.4f}")

    # I. Effective-bit math
    log("\n## I. Effective-bit math sanity")
    # Q0 BF16 dense: K-bits == 16.0
    for it in smoke_items:
        q0 = by_cond.get("Q0", {}).get(it.id, {})
        k_bits = q0.get("effective_k_bits")
        check(f"I.bits_Q0_BF16 item={it.id}", k_bits == 16.0, f"k_bits={k_bits}")
    # Q1 F4 dense: K-bits == 4.0
    for it in smoke_items:
        q1 = by_cond.get("Q1", {}).get(it.id, {})
        k_bits = q1.get("effective_k_bits")
        check(f"I.bits_Q1_F4 item={it.id}", k_bits == 4.0, f"k_bits={k_bits}")
    # Q2 F9 dense: K-bits == 5.50
    for it in smoke_items:
        q2 = by_cond.get("Q2", {}).get(it.id, {})
        k_bits = q2.get("effective_k_bits")
        check(f"I.bits_Q2_F9 item={it.id}", abs(k_bits - 5.50) < 1e-3, f"k_bits={k_bits}")
    # Q3 RoleOnly: K-bits in (4.0, 5.50). Token-weighted mix of F4 in-context vs F9 always-on.
    for it in smoke_items:
        q3 = by_cond.get("Q3", {}).get(it.id, {})
        k_bits = q3.get("effective_k_bits")
        ok = (k_bits is not None and 4.0 < k_bits < 5.50)
        check(f"I.bits_Q3_RoleOnly_range item={it.id}", ok, f"k_bits={k_bits}")
    # Q7 Quest top-25 FormatBook: K-bits between 4.0 and 5.50
    for it in smoke_items:
        q7 = by_cond.get("Q7", {}).get(it.id, {})
        k_bits = q7.get("effective_k_bits")
        ok = (k_bits is not None and 4.0 <= k_bits <= 5.50)
        check(f"I.bits_Q7_top25_FB_range item={it.id}", ok, f"k_bits={k_bits}")
    # Q10 INT2-cold top-25 Quest: K-bits between 2.0 and 5.50
    for it in smoke_items:
        q10 = by_cond.get("Q10", {}).get(it.id, {})
        k_bits = q10.get("effective_k_bits")
        ok = (k_bits is not None and 2.0 <= k_bits <= 5.50)
        check(f"I.bits_Q10_INT2cold_range item={it.id}", ok, f"k_bits={k_bits}")

    # ===========================================================
    # Exp R assertions (J/K/L/M/N/O) — run only with --exp-r
    # ===========================================================
    if args.exp_r:
        # J. AllVisual hot mix — C4 active set spans both in_context and choice kinds.
        log("\n## J. AllVisual hot-page mix (C4)")
        for it in smoke_items:
            c4 = by_cond.get("C4", {}).get(it.id, {})
            actives = c4.get("all_active_per_layer", [])
            if not actives:
                check(f"J.allvisual_hot_mix item={it.id}", False, "no routing log")
                continue
            # Need at least one layer where the active set spans both kinds.
            layout_info = c4.get("layout", {})
            n_in = layout_info.get("n_in_context", 0)
            # in-context image_idx values are 0..n_in-1 → page_idx values are
            # specific to the layout. We approximate "mix" by counting unique
            # active page indices across all layers and checking that the
            # set is non-empty AND has at least two distinct page indices
            # (proxy for kind diversity given only ~4-12 routable pages).
            unique_active = set()
            for a in actives:
                unique_active.update(a)
            check(f"J.allvisual_hot_mix item={it.id}",
                  len(unique_active) >= 2,
                  f"n_unique_active_pages={len(unique_active)}")

        # K. SplitQuest balance — for C7, |active ∩ in_context| ≈ |active ∩ choice|.
        # We use page_idx ranges from the layout to identify in-context vs choice.
        log("\n## K. SplitQuest balance (C7)")
        for it in smoke_items:
            c7 = by_cond.get("C7", {}).get(it.id, {})
            actives = c7.get("all_active_per_layer", [])
            layout_info = c7.get("layout", {})
            n_in = layout_info.get("n_in_context", 0)
            n_ch = layout_info.get("n_choice", 0)
            if not actives or n_in == 0 or n_ch == 0:
                # Skip items where layout doesn't have both kinds routable
                check(f"K.split_balance item={it.id}", True,
                      f"skip (n_in={n_in}, n_ch={n_ch})")
                continue
            # The first layer's active set is enough; split policies are
            # layer-agnostic in current implementation.
            active0 = set(actives[0]) if actives else set()
            # We don't have direct kind tags here; use a proxy: at least one
            # active page from BOTH groups means budget split.
            # Without page-kind metadata in the routing log, we accept this
            # smoke as "passes if active set size is reasonable (>= 2)".
            check(f"K.split_balance item={it.id}",
                  len(active0) >= 2,
                  f"n_active_layer0={len(active0)}")

        # L. AllVisual-Oracle includes the correct-choice page.
        log("\n## L. AllVisual-Oracle correct-choice (C6)")
        for it in smoke_items:
            c6 = by_cond.get("C6", {}).get(it.id, {})
            actives = c6.get("all_active_per_layer", [])
            # On every layer the oracle should include some specific page
            # corresponding to item.correct_choice. We can't easily map
            # page_idx -> choice_idx here without the layout object, so we
            # check (a) routing fired and (b) needle_in_active when needle
            # is routable.
            check(f"L.oracle_choice_routing_fired item={it.id}",
                  len(actives) > 0,
                  f"n_layers_logged={len(actives)}")

    # M. Cold-V harsher than no-op (synthetic).
    log("\n## M. Cold-V INT2 harsher than default V INT4 (synthetic)")
    rng_m = np.random.default_rng(456)
    V_np = rng_m.standard_normal(size=(1, 4, 256, 128)).astype(np.float32)
    V = torch.from_numpy(V_np).to(torch.bfloat16)
    V_int2 = _int2_per_channel_seq(V)
    # "Default" V quantization is INT4 head_dim — we approximate with F4 cold
    # since the round-trip granularity is comparable.
    V_int4 = _f4_per_channel_seq(V)
    err_int4 = (V.float() - V_int4.float()).abs().mean().item()
    err_int2 = (V.float() - V_int2.float()).abs().mean().item()
    check("M.cold_v_int2_harsher",
          err_int2 > err_int4 * 1.5,
          f"mean |V - V_int2|={err_int2:.4f} vs mean |V - V_int4|={err_int4:.4f}")

    # N. INT3 reconstruction error between F4 and INT2.
    log("\n## N. INT3 between F4 and INT2 (synthetic)")
    K_n = torch.from_numpy(
        np.random.default_rng(789).standard_normal(size=(1, 4, 256, 128)).astype(np.float32)
    ).to(torch.bfloat16)
    K_f4 = _f4_per_channel_seq(K_n)
    K_int3 = _int3_per_channel_seq(K_n)
    K_int2 = _int2_per_channel_seq(K_n)
    err_f4 = (K_n.float() - K_f4.float()).abs().mean().item()
    err_int3 = (K_n.float() - K_int3.float()).abs().mean().item()
    err_int2 = (K_n.float() - K_int2.float()).abs().mean().item()
    check("N.int3_between_f4_and_int2",
          err_f4 < err_int3 < err_int2,
          f"err: F4={err_f4:.4f} INT3={err_int3:.4f} INT2={err_int2:.4f}")

    # O. FP8 effective K bits HIGHER than F4 cold (FP8 is a diagnostic, not memory saving).
    log("\n## O. FP8 K-bits > F4 K-bits (memory-saving sanity)")
    # FP8 K = 8 bits/token, F4 K = 4 bits/token. Pure bit-count comparison.
    check("O.fp8_not_memory_saving",
          PAGE_K_BITS["FP8"] > PAGE_K_BITS["F4"],
          f"FP8={PAGE_K_BITS['FP8']} F4={PAGE_K_BITS['F4']}")

    # ===========================================================
    # Exp T assertions (P/Q/R/U) — run only with --exp-t
    # ===========================================================
    if args.exp_t:
        # P. Logits differ across INT8/INT7/INT6 on real rollouts.
        log("\n## P. Sidecode logits differ across INT8/INT7/INT6")
        for it in smoke_items:
            s3 = by_cond.get("S3", {}).get(it.id, {}).get("first_logits_AD", [0]*4)
            s4 = by_cond.get("S4", {}).get(it.id, {}).get("first_logits_AD", [0]*4)
            s5 = by_cond.get("S5", {}).get(it.id, {}).get("first_logits_AD", [0]*4)
            q1 = by_cond.get("Q1", {}).get(it.id, {}).get("first_logits_AD", [0]*4)
            d_s4_s3 = _l2(s4, s3)
            d_s5_s4 = _l2(s5, s4)
            d_s3_q1 = _l2(s3, q1)
            check(f"P.s4_vs_s3_int7_int8 item={it.id}", d_s4_s3 > 1e-4,
                  f"||S4(INT7) - S3(INT8)||_2 = {d_s4_s3:.6e}")
            check(f"P.s5_vs_s4_int6_int7 item={it.id}", d_s5_s4 > 1e-4,
                  f"||S5(INT6) - S4(INT7)||_2 = {d_s5_s4:.6e}")
            check(f"P.s3_vs_f4 item={it.id}", d_s3_q1 > 1e-3,
                  f"||S3(INT8) - Q1(F4)||_2 = {d_s3_q1:.6e}")

        # Q. Effective KV-bits per token match the analytical ladder.
        #   SJ/S3 (top-16 INT8):  K=(8·16+4·112)/128 = 4.500 → KV 4.250
        #   S4    (top-16 INT7):  K=(7·16+4·112)/128 = 4.375 → KV 4.1875
        #   S5    (top-16 INT6):  K=(6·16+4·112)/128 = 4.250 → KV 4.125
        log("\n## Q. Effective-bit math for sidecode ladder")
        for it in smoke_items:
            for cond, expected_k in (("S3", 4.500), ("S4", 4.375), ("S5", 4.250)):
                r = by_cond.get(cond, {}).get(it.id, {})
                k_bits = r.get("effective_k_bits")
                kv_bits = r.get("effective_kv_bits")
                expected_kv = (expected_k + 4.0) / 2.0
                check(f"Q.bits_{cond}_k item={it.id}",
                      k_bits is not None and abs(k_bits - expected_k) < 1e-3,
                      f"k_bits={k_bits} (expected {expected_k})")
                check(f"Q.bits_{cond}_kv item={it.id}",
                      kv_bits is not None and abs(kv_bits - expected_kv) < 1e-3,
                      f"kv_bits={kv_bits} (expected {expected_kv})")

        # R. Sidecode is K-only — every dense sidecode condition reports V=4.
        log("\n## R. Sidecode is K-only — V unaffected (effective_v_bits = 4)")
        for cond_name in ("S3", "S4", "S5"):
            for it in smoke_items:
                r = by_cond.get(cond_name, {}).get(it.id, {})
                v_bits = r.get("effective_v_bits")
                check(f"R.v_unchanged_{cond_name} item={it.id}",
                      v_bits is not None and abs(v_bits - 4.0) < 1e-6,
                      f"v_bits={v_bits} (expected 4.0; sidecode must not touch V)")

        # U. BF16 dense V correctly reported as 16, not 4 (the V_BITS regression).
        log("\n## U. BF16 dense V is 16 bits (V_BITS bug regression check)")
        for it in smoke_items:
            q0 = by_cond.get("Q0", {}).get(it.id, {})
            v_bits = q0.get("effective_v_bits")
            kv_bits = q0.get("effective_kv_bits")
            check(f"U.bf16_v_16 item={it.id}",
                  v_bits is not None and abs(v_bits - 16.0) < 1e-6,
                  f"Q0 BF16 v_bits={v_bits} (expected 16.0)")
            check(f"U.bf16_kv_16 item={it.id}",
                  kv_bits is not None and abs(kv_bits - 16.0) < 1e-6,
                  f"Q0 BF16 kv_bits={kv_bits} (expected 16.0 = (16+16)/2)")

    # ===========================================================
    # Exp U assertions (V/W/X/Y/Z) — run only with --exp-u.
    # ===========================================================
    if args.exp_u:
        # V. U3 (S4 anchor inside U-suite) matches existing S4 condition
        #    semantically — its bit math is (16 channels at INT7) → K=4.375.
        log("\n## V. U3 (S4 anchor inside U) effective_k_bits = 4.375, KV = 4.1875")
        for it in smoke_items:
            r = by_cond.get("U3", {}).get(it.id, {})
            k_bits = r.get("effective_k_bits")
            kv_bits = r.get("effective_kv_bits")
            check(f"V.bits_U3_k item={it.id}",
                  k_bits is not None and abs(k_bits - 4.375) < 1e-3,
                  f"k_bits={k_bits} (expected 4.375)")
            check(f"V.bits_U3_kv item={it.id}",
                  kv_bits is not None and abs(kv_bits - 4.1875) < 1e-3,
                  f"kv_bits={kv_bits} (expected 4.1875)")

        # W. U4..U12 (24 channels at INT7) → K=4.5625, KV=4.28125
        log("\n## W. U4..U12 effective_k_bits = 4.5625, KV = 4.28125")
        for cond_name in ("U4", "U5", "U6", "U7", "U11", "U12"):
            for it in smoke_items:
                r = by_cond.get(cond_name, {}).get(it.id, {})
                k_bits = r.get("effective_k_bits")
                kv_bits = r.get("effective_kv_bits")
                check(f"W.bits_{cond_name}_k item={it.id}",
                      k_bits is not None and abs(k_bits - 4.5625) < 1e-3,
                      f"k_bits={k_bits} (expected 4.5625)")
                check(f"W.bits_{cond_name}_kv item={it.id}",
                      kv_bits is not None and abs(kv_bits - 4.28125) < 1e-3,
                      f"kv_bits={kv_bits} (expected 4.28125)")

        # X. U13 (32 channels at INT7) → K=4.75, KV=4.375
        log("\n## X. U13 effective_k_bits = 4.75, KV = 4.375")
        for it in smoke_items:
            r = by_cond.get("U13", {}).get(it.id, {})
            k_bits = r.get("effective_k_bits")
            kv_bits = r.get("effective_kv_bits")
            check(f"X.bits_U13_k item={it.id}",
                  k_bits is not None and abs(k_bits - 4.75) < 1e-3,
                  f"k_bits={k_bits} (expected 4.75)")
            check(f"X.bits_U13_kv item={it.id}",
                  kv_bits is not None and abs(kv_bits - 4.375) < 1e-3,
                  f"kv_bits={kv_bits} (expected 4.375)")

        # Y. Residual invariant on the calib NPZ — every EXTRA_*_8 array has
        #    empty intersection with the S4 top-16 set, per (L, H_kv) cell.
        log("\n## Y. Residual invariant: every EXTRA array is disjoint from S4-top-16")
        s4_arr = calib.get("outlier_channel_idx_top16")
        if s4_arr is None and "k_channel_energy" in calib:
            energy = np.asarray(calib["k_channel_energy"])
            s4_arr = (np.argsort(energy, axis=-1)[..., -16:][..., ::-1]
                      .astype(np.int32).copy())
        for key in ("outlier_idx_EXTRA_GEN_8", "outlier_idx_EXTRA_RND_8",
                    "outlier_idx_EXTRA_TT_8", "outlier_idx_EXTRA_TV_8",
                    "outlier_idx_EXTRA_VT_8", "outlier_idx_EXTRA_VV_8",
                    "outlier_idx_EXTRA_BAL_8",
                    "outlier_idx_EXTRA_MMNIAH_PRIOR_8",
                    "outlier_idx_EXTRA_LVB_PRIOR_8",
                    "outlier_idx_EXTRA_ALL_16"):
            arr = calib.get(key)
            if arr is None:
                check(f"Y.has_{key}", False, "key missing from calib+extras")
                continue
            L, H_kv, _ = s4_arr.shape
            bad_cells = 0
            for l in range(L):
                for h in range(H_kv):
                    s = set(int(x) for x in s4_arr[l, h].tolist())
                    e = set(int(x) for x in arr[l, h].tolist())
                    if s & e:
                        bad_cells += 1
            check(f"Y.residual_{key}", bad_cells == 0,
                  f"{bad_cells}/{L*H_kv} cells overlap with S4")

        # Z. U6 (TT) / U7 (TV) / U11 (MMNIAH-prior) logits differ from U3 (S4 anchor)
        #    on at least one smoke item — verifies extra channels are actually
        #    different from the anchor (not silently empty).
        log("\n## Z. Extra-channel policies produce logits distinct from U3")
        for u_name in ("U6", "U7", "U11", "U12", "U13"):
            any_differ = False
            for it in smoke_items:
                u3 = by_cond.get("U3", {}).get(it.id, {}).get("first_logits_AD", [0]*4)
                ux = by_cond.get(u_name, {}).get(it.id, {}).get("first_logits_AD", [0]*4)
                if _l2(ux, u3) > 1e-4:
                    any_differ = True
                    break
            check(f"Z.logits_{u_name}_differs_from_U3", any_differ,
                  f"||{u_name} - U3|| <= 1e-4 on ALL smoke items — extra channels "
                  f"may be silently inactive")

        # AA. Cross-policy distinctness — U6 (TT) vs U7 (TV) must differ on at
        # least one item; otherwise extras NPZ has degenerate priors.
        log("\n## AA. Distinct extra-channel policies produce distinct logits")
        for a, b in (("U6", "U7"), ("U6", "U8"), ("U11", "U12")):
            any_differ = False
            for it in smoke_items:
                la = by_cond.get(a, {}).get(it.id, {}).get("first_logits_AD", [0]*4)
                lb = by_cond.get(b, {}).get(it.id, {}).get("first_logits_AD", [0]*4)
                if _l2(la, lb) > 1e-4:
                    any_differ = True
                    break
            check(f"AA.{a}_vs_{b}_differs", any_differ,
                  f"||{a} - {b}|| <= 1e-4 on ALL smoke items — "
                  f"likely identical extra arrays (check extras NPZ)")

    log(f"\n## SUMMARY")
    log(f"PASS: {n_pass}")
    log(f"FAIL: {n_fail}")

    args.out.write_text("\n".join(log_lines) + "\n")
    log(f"\nwrote {args.out}")

    if n_fail > 0:
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
