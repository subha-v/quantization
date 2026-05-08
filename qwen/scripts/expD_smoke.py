"""Smoke tests for Exp D0/D1 pipeline. Halts the pipeline if any check fails.

Checks:
  1. visual_token_range_detection — find_visual_token_span returns sane span
  2. window_token_mapping — partitioning covers v_start..v_end
  3. v3k_logits_differ — V3K mask (window 0 BF16, rest INT4) perturbs first-token
                         logits by ||Δ||∞ > 1e-3 (vs BF16 baseline)
  4. mask_cache_alignment — get_kv_bits_for_chunk slicing OK at offset=0
  5. frame_removal_correctness — format_mcq_messages_with_frames runs end-to-end

Writes pass/fail markdown to qwen/results/expD_smoke.md.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from data_longvideobench import (
    DEFAULT_SPLIT_FILE,
    answer_token_ids,
    filter_items,
    format_mcq_messages,
    format_mcq_messages_with_frames,
    load_all_items,
    load_split,
)
from fake_quant_kv_cache import BitController, FakeQuantKVCache
from frame_manip import decode_uniform_frames, select_frame_subset, window_indices
from visual_tokens import build_window_token_ranges, find_visual_token_span


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _option_logprobs(out, processor, n_options):
    answer_ids = answer_token_ids(processor, n=n_options)
    first_logits = out.scores[0]
    return torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist(), first_logits[0].float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--n_items", type=int, default=20)
    ap.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "expD_smoke.md")
    args = ap.parse_args()

    items_all = load_all_items()
    split = load_split(args.split_file)
    eval_items = filter_items(items_all, split["eval"])[: args.n_items]
    print(f"[smoke] n_items={len(eval_items)}", flush=True)

    from run_inference import load_model
    from qwen_vl_utils import process_vision_info  # type: ignore

    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    print(f"[smoke] num_layers={num_layers} num_kv_heads={num_kv_heads}", flush=True)

    results: list[dict] = []
    overall_pass = True

    # ---- Run on the first item only for the heavyweight checks ----
    first = eval_items[0]
    msgs = format_mcq_messages(first, n_frames=args.frames)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]

    # Check 1: visual-token-range detection
    try:
        v_start, v_end = find_visual_token_span(inputs["input_ids"], processor)
        ok = (v_start > 0 and v_end > v_start and (v_end - v_start) >= 1000
              and (v_end - v_start) <= 5000)
        results.append({
            "check": "visual_token_range_detection",
            "pass": ok,
            "v_start": int(v_start), "v_end": int(v_end),
            "n_visual": int(v_end - v_start), "seq_len": int(seq_len),
        })
        overall_pass &= ok
    except Exception as e:
        results.append({"check": "visual_token_range_detection", "pass": False, "error": str(e)})
        overall_pass = False
        v_start = v_end = None

    # Check 2: window-token mapping
    if v_start is not None and v_end is not None:
        try:
            ranges = build_window_token_ranges(v_start, v_end, n_windows=args.windows)
            covered_lo = ranges[0][0]
            covered_hi = ranges[-1][1]
            covers = (covered_lo == v_start) and (covered_hi == v_end) and len(ranges) == args.windows
            sizes = [b - a for a, b in ranges]
            results.append({
                "check": "window_token_mapping", "pass": covers,
                "covered_lo": int(covered_lo), "covered_hi": int(covered_hi),
                "n_windows": len(ranges), "sizes": sizes,
            })
            overall_pass &= covers
        except Exception as e:
            results.append({"check": "window_token_mapping", "pass": False, "error": str(e)})
            overall_pass = False
            ranges = None
    else:
        ranges = None

    # Check 3: V3K logits-differ — BF16 vs (window-0 BF16, rest INT4) on K
    bf16_logits = None
    v3k_logits = None
    try:
        # BF16 forward (no cache wrapper)
        cache_bf16 = DynamicCache()
        out_bf16 = model.generate(**inputs, past_key_values=cache_bf16, max_new_tokens=1,
                                  do_sample=False, return_dict_in_generate=True,
                                  output_scores=True, use_cache=True)
        _, bf16_logits = _option_logprobs(out_bf16, processor, len(first.candidates))
        del cache_bf16, out_bf16
        torch.cuda.empty_cache()

        # V3K forward: protect window 0 only on K, V at INT4 globally
        if ranges is not None:
            mask = torch.zeros(seq_len, dtype=torch.bool)
            a, b = ranges[0]
            mask[a:b] = True  # protect window 0 visual K only
            ctrl = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                                 mode="V3K", default_k_bits=4, default_v_bits=4)
            ctrl.set_global(k_bits=4, v_bits=4)
            for L in range(num_layers):
                ctrl.set_protected_mask(L, mask, hi_bits=16, lo_bits=4)
            cache = FakeQuantKVCache(ctrl)
            out_v3k = model.generate(**inputs, past_key_values=cache, max_new_tokens=1,
                                     do_sample=False, return_dict_in_generate=True,
                                     output_scores=True, use_cache=True)
            _, v3k_logits = _option_logprobs(out_v3k, processor, len(first.candidates))
            del cache, out_v3k
            torch.cuda.empty_cache()

            inf_norm = float((bf16_logits - v3k_logits).abs().max().item())
            ok = inf_norm > 1e-3
            results.append({
                "check": "v3k_logits_differ", "pass": ok,
                "logits_inf_norm": inf_norm, "threshold": 1e-3,
            })
            overall_pass &= ok
        else:
            results.append({"check": "v3k_logits_differ", "pass": False, "error": "no window ranges"})
            overall_pass = False
    except Exception as e:
        results.append({"check": "v3k_logits_differ", "pass": False, "error": str(e)})
        overall_pass = False

    # Check 4: mask-cache alignment — controller slicing at offset=0 returns expected shape
    try:
        if ranges is not None:
            mask = torch.zeros(seq_len, dtype=torch.bool)
            a, b = ranges[0]
            mask[a:b] = True
            ctrl = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                                 mode="V3K", default_k_bits=4, default_v_bits=4)
            ctrl.set_global(k_bits=4, v_bits=4)
            ctrl.set_protected_mask(0, mask, hi_bits=16, lo_bits=4)
            kb, vb = ctrl.get_kv_bits_for_chunk(0, new_chunk_len=seq_len, num_kv_heads=num_kv_heads,
                                                cache_offset=0)
            shape_ok = kb.shape == (num_kv_heads, seq_len) if hasattr(kb, "shape") else False
            v_scalar = isinstance(vb, int) and vb == 4
            n_protect = int(kb[0].eq(16).sum().item()) if hasattr(kb, "shape") else -1
            ok = bool(shape_ok and v_scalar and n_protect == (b - a))
            results.append({
                "check": "mask_cache_alignment", "pass": ok,
                "kb_shape": list(kb.shape) if hasattr(kb, "shape") else None,
                "vb": vb if isinstance(vb, int) else "tensor",
                "n_protected_in_chunk": n_protect, "expected": int(b - a),
            })
            overall_pass &= ok
        else:
            results.append({"check": "mask_cache_alignment", "pass": False, "error": "no ranges"})
            overall_pass = False
    except Exception as e:
        results.append({"check": "mask_cache_alignment", "pass": False, "error": str(e)})
        overall_pass = False

    # Check 5: frame-removal end-to-end
    try:
        frames = decode_uniform_frames(first.video_path, n_frames=args.frames)
        sub = select_frame_subset(frames, window_indices(0, frames_per_window=8))
        msgs2 = format_mcq_messages_with_frames(first, sub)
        prompt2 = processor.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)
        img2, vid2 = process_vision_info(msgs2)
        inputs2 = processor(text=[prompt2], images=img2, videos=vid2,
                            padding=True, return_tensors="pt").to(model.device)
        sl2 = int(inputs2["input_ids"].shape[1])
        # Should be substantially shorter than full-64 forward (we passed 8 frames)
        ok = sl2 < seq_len * 0.6  # 8/64 = 1/8; allow up to 60% to give text headroom
        out2 = model.generate(**inputs2, max_new_tokens=1, do_sample=False,
                              return_dict_in_generate=True, output_scores=True, use_cache=True)
        logp2, _ = _option_logprobs(out2, processor, len(first.candidates))
        ok = ok and len(logp2) == len(first.candidates) and all(
            isinstance(x, float) and not (x != x) for x in logp2  # not NaN
        )
        results.append({
            "check": "frame_removal_end_to_end", "pass": ok,
            "full_seq_len": int(seq_len), "frame_subset_seq_len": sl2,
            "logp_len": len(logp2),
        })
        overall_pass &= ok
        del out2, inputs2
        torch.cuda.empty_cache()
    except Exception as e:
        results.append({"check": "frame_removal_end_to_end", "pass": False, "error": str(e)})
        overall_pass = False

    # ---- Write smoke report ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Exp D Smoke Report\n\nGenerated: {_ts()}\n",
             f"Model: `{args.model}`  Frames: {args.frames}  Windows: {args.windows}  ",
             f"n_items_total: {len(eval_items)}  test_item: `{first.id}`\n",
             "## Checks\n",
             "| # | Check | Pass | Detail |",
             "|---|---|:-:|---|"]
    for i, r in enumerate(results, 1):
        det_keys = [k for k in r.keys() if k not in ("check", "pass")]
        det = ", ".join(f"{k}={r[k]}" for k in det_keys)
        marker = "✅" if r["pass"] else "❌"
        lines.append(f"| {i} | `{r['check']}` | {marker} | {det} |")
    lines.append("")
    lines.append(f"## Overall: {'PASS' if overall_pass else 'FAIL'}")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[smoke] wrote {args.out}", flush=True)
    print(f"[smoke] overall: {'PASS' if overall_pass else 'FAIL'}", flush=True)
    if not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
