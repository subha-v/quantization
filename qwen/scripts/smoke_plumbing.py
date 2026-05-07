"""
Plumbing-only smoke for Qwen2.5-VL fake-quant KV cache.

Validates the critical correctness invariant *without* requiring LongVideoBench
videos: BF16 first-token logits and INT2-KV first-token logits MUST differ by
||Δ||_∞ > 1e-3 across multiple text prompts. If they don't, FakeQuantKVCache
is not being consumed by the attention matmul and the experiment is invalid.

Run on tambe-server-1 inside the qwen_venv:
    CUDA_VISIBLE_DEVICES=<idx> python smoke_plumbing.py [--model Qwen/Qwen2.5-VL-3B-Instruct]
"""
from __future__ import annotations

import argparse
import sys

import torch

from fake_quant_kv_cache import BitController, FakeQuantKVCache


PROMPTS = [
    "The capital of France is",
    "If I have 5 apples and give away 2, I have",
    "The boiling point of water at sea level is",
    "Photosynthesis converts sunlight, water, and carbon dioxide into",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--threshold", type=float, default=1e-3)
    args = ap.parse_args()

    print(f"[smoke] loading {args.model}")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = processor.tokenizer

    # Locate decoder layers (model.language_model.layers in Qwen2.5-VL)
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = model.config.num_key_value_heads if hasattr(model.config, "num_key_value_heads") else 4
    print(f"[smoke] num_layers={num_layers} num_kv_heads={num_kv_heads}")

    def score_first_logits(prompt: str, k_bits: int, v_bits: int):
        controller = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads, mode="V1")
        controller.set_global(k_bits=k_bits, v_bits=v_bits)
        cache = FakeQuantKVCache(controller)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, past_key_values=cache, max_new_tokens=1,
                do_sample=False, return_dict_in_generate=True, output_scores=True,
                use_cache=True,
            )
        return out.scores[0][0].float().cpu()  # [vocab]

    failures = []
    for i, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        seq_len = ids.shape[-1]
        bf16 = score_first_logits(prompt, 16, 16)
        int2 = score_first_logits(prompt, 2, 2)
        diff = (bf16 - int2).abs().max().item()
        argmax_bf16 = int(bf16.argmax())
        argmax_int2 = int(int2.argmax())
        decoded_bf16 = tokenizer.decode([argmax_bf16])
        decoded_int2 = tokenizer.decode([argmax_int2])
        ok = diff > args.threshold
        status = "OK" if ok else "FAIL"
        print(f"[smoke][{i+1}/{len(PROMPTS)}][{status}] seq_len={seq_len} ||Δ||_∞={diff:.4e}  "
              f"BF16->{decoded_bf16!r}  INT2->{decoded_int2!r}")
        if not ok:
            failures.append((prompt, diff))

    print()
    if failures:
        print(f"[smoke][FAIL] {len(failures)}/{len(PROMPTS)} prompts had ||Δ||_∞ ≤ {args.threshold}")
        print("              KV quantization is not being applied at prefill.")
        print("              Investigate fake_quant_kv_cache.update() / attention backend.")
        sys.exit(1)

    # Additional sanity: INT4 should differ less than INT2
    p = PROMPTS[0]
    bf16 = score_first_logits(p, 16, 16)
    int4 = score_first_logits(p, 4, 4)
    int2 = score_first_logits(p, 2, 2)
    d4 = (bf16 - int4).abs().max().item()
    d2 = (bf16 - int2).abs().max().item()
    print(f"[smoke] monotonicity check: ||BF16-INT4||_∞ = {d4:.4e}, ||BF16-INT2||_∞ = {d2:.4e}  "
          f"(INT4 should be ≤ INT2)")
    if not (d4 <= d2 * 1.5):  # allow some noise
        print(f"[smoke][WARN] INT4 perturbation > 1.5× INT2 — unexpected, investigate quant path")

    print(f"[smoke][PASS] all {len(PROMPTS)} prompts passed ||Δ||_∞ > {args.threshold}.")
    print("[smoke][PASS] KV quantization confirmed to affect first-token logits at prefill.")


if __name__ == "__main__":
    main()
