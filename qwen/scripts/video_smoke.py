"""
1-item video-pathway smoke for Qwen2.5-VL.

Validates the full pipeline end-to-end on a single LongVideoBench item:
processor + qwen-vl-utils video decode + Qwen2.5-VL forward + FakeQuantKVCache
+ MCQ logprob scoring. Asserts BF16 vs INT2-KV first-token logits differ.

Picks the first item from lvb_val.json whose video has been extracted to
$LONGVIDEOBENCH_ROOT/videos. Useful while the full extraction is still running.

Usage:
    python video_smoke.py --model Qwen/Qwen2.5-VL-3B-Instruct --frames 32
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from data_longvideobench import answer_token_ids, format_mcq_messages, load_all_items
from fake_quant_kv_cache import BitController, FakeQuantKVCache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=1e-3)
    args = ap.parse_args()

    items = load_all_items()
    extracted = set(os.listdir("/data/subha2/longvideobench/videos"))
    present = [it for it in items if os.path.basename(it.video_path) in extracted]
    print(f"[vsmoke] items with extracted video: {len(present)}/{len(items)}")
    if not present:
        print("[vsmoke][FAIL] no extracted videos yet — wait for tar extraction")
        sys.exit(2)
    sample = present[0]
    print(f"[vsmoke] item: id={sample.id} dur={sample.duration_seconds:.1f}s n_cands={len(sample.candidates)}")
    print(f"[vsmoke] video: {sample.video_path}")
    print(f"[vsmoke] question: {sample.question[:120]}")
    print(f"[vsmoke] correct: {sample.correct_choice} ({sample.candidates[sample.correct_choice][:80]})")

    print(f"[vsmoke] loading {args.model} ...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    t0 = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"[vsmoke] loaded in {time.perf_counter() - t0:.1f}s")

    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = model.config.num_key_value_heads
    print(f"[vsmoke] num_layers={num_layers} num_kv_heads={num_kv_heads}")

    from qwen_vl_utils import process_vision_info

    msgs = format_mcq_messages(sample, n_frames=args.frames)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    print(f"[vsmoke] input_ids shape: {tuple(inputs.input_ids.shape)}")

    def run(k_bits: int, v_bits: int, label: str):
        ctrl = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads, mode="V1")
        ctrl.set_global(k_bits=k_bits, v_bits=v_bits)
        cache = FakeQuantKVCache(ctrl)
        t = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs, past_key_values=cache, max_new_tokens=1, do_sample=False,
                return_dict_in_generate=True, output_scores=True, use_cache=True,
            )
        dt = time.perf_counter() - t
        ans_ids = answer_token_ids(processor, n=len(sample.candidates))
        logp = torch.log_softmax(out.scores[0].float(), dim=-1)[0, ans_ids].tolist()
        pred = max(range(len(sample.candidates)), key=lambda i: logp[i])
        ok = "✓" if pred == sample.correct_choice else "✗"
        formatted_logp = "[" + ", ".join("{:.2f}".format(x) for x in logp) + "]"
        print(f"[vsmoke] {label}: latency={dt*1000:.0f}ms logp={formatted_logp} pred={pred} {ok}")
        return out.scores[0]

    bf16_logits = run(16, 16, "BF16")
    int2_logits = run(2, 2, "INT2")
    diff = (bf16_logits - int2_logits).abs().max().item()
    print(f"[vsmoke] ||Δ_logits||_inf = {diff:.4e}")
    if diff <= args.threshold:
        print(f"[vsmoke][FAIL] perturbation ≤ {args.threshold} — KV quant is not affecting prefill")
        sys.exit(1)
    print(f"[vsmoke][PASS] full video pathway working; KV quant active.")


if __name__ == "__main__":
    main()
