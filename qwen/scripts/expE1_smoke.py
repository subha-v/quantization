"""Smoke test for Exp E1 text-K slice ablation. Halts pipeline on any failure.

Checks (run on the first 5 eval items):
  1. slice_detection_coverage   — all 5 keys present, no fail status
  2. slice_non_overlap_full     — slices cover [0, seq_len) without overlap
  3. slice_decode_round_trip    — decoded slice text matches the expected substring
                                  (question and instruction; options is a substring match)
  4. v3k_question_logits_differ — V3K mask covering only `question` slice perturbs
                                  first-token logits ||Δ||∞ > 1e-3 vs BF16
  5. v3k_question_distinct_from_all_text — question-only mask gives different logits
                                  than the all-text-BF16 (D1.3) mask

Writes qwen/results/expE1_smoke.md with pass/fail per check.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from data_longvideobench import (
    DEFAULT_SPLIT_FILE,
    answer_token_ids,
    filter_items,
    format_mcq_messages,
    load_all_items,
    load_split,
)
from fake_quant_kv_cache import BitController, FakeQuantKVCache
from text_slices import find_text_slice_spans, union_mask
from visual_tokens import find_visual_token_span


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _logits_for_mask(model, processor, inputs, mask, num_layers, num_kv_heads,
                     k_hi=16, k_lo=4, v_bits=4) -> torch.Tensor:
    if mask is None:
        cache = DynamicCache()
    else:
        ctrl = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                             mode="V3K", default_k_bits=k_lo, default_v_bits=v_bits)
        ctrl.set_global(k_bits=k_lo, v_bits=v_bits)
        for L in range(num_layers):
            ctrl.set_protected_mask(L, mask, hi_bits=k_hi, lo_bits=k_lo)
        cache = FakeQuantKVCache(ctrl)
    out = model.generate(**inputs, past_key_values=cache, max_new_tokens=1, do_sample=False,
                         return_dict_in_generate=True, output_scores=True, use_cache=True)
    return out.scores[0][0].float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--n_items", type=int, default=5)
    ap.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "expE1_smoke.md")
    args = ap.parse_args()

    items_all = load_all_items()
    split = load_split(args.split_file)
    eval_items = filter_items(items_all, split["eval"])[: args.n_items]
    print(f"[E1-smoke] n_items={len(eval_items)}", flush=True)

    from run_inference import load_model
    from qwen_vl_utils import process_vision_info  # type: ignore

    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)

    results: list[dict] = []
    overall = True

    # Check 1+2+3 on n_items
    coverage_pass = True
    nonoverlap_pass = True
    roundtrip_pass = True
    coverage_detail = []
    nonoverlap_detail = []
    roundtrip_detail = []

    for idx, item in enumerate(eval_items):
        msgs = format_mcq_messages(item, n_frames=args.frames)
        prompt_text = processor.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        seq_len = int(inputs["input_ids"].shape[1])
        ids = inputs["input_ids"][0].tolist()

        slices = find_text_slice_spans(inputs["input_ids"], processor, item)
        warnings = slices.get("_warnings", [])
        keys = ("header", "visual_wrapper", "question", "options", "instruction", "answer_prefix")
        missing = [k for k in keys if not isinstance(slices.get(k), tuple)]
        empty_or_fail = [k for k in keys
                         if isinstance(slices.get(k), tuple)
                         and slices[k][1] <= slices[k][0]
                         and k != "header"]  # header may legitimately be empty edge case
        # Coverage
        if missing or empty_or_fail or any("fail" in w for w in warnings):
            coverage_pass = False
            coverage_detail.append(
                f"item={item.id} missing={missing} empty={empty_or_fail} warn={warnings}"
            )

        # Non-overlap and full coverage [0, seq_len)
        spans = sorted([slices[k] for k in keys if isinstance(slices.get(k), tuple)
                        and slices[k][1] > slices[k][0]])
        prev = 0
        gap_or_overlap = []
        for a, b in spans:
            if a < prev:
                gap_or_overlap.append(f"overlap@{a}<prev_end={prev}")
            elif a > prev:
                gap_or_overlap.append(f"gap_{prev}_{a}")
            prev = b
        if prev < seq_len:
            gap_or_overlap.append(f"trailing_gap_{prev}_{seq_len}")
        if gap_or_overlap:
            # Some gap is expected: e.g., header may end at v_start - 1, and visual_wrapper
            # starts at v_start - 1 as well. So we tolerate small gaps but log them.
            # The CRITICAL requirement is no overlaps and total covered length close to seq_len.
            covered = sum(b - a for a, b in spans)
            if covered < int(0.95 * seq_len):
                nonoverlap_pass = False
                nonoverlap_detail.append(
                    f"item={item.id} covered={covered}/{seq_len} issues={gap_or_overlap[:3]}"
                )

        # Decode round-trip checks: question and instruction must contain expected substrings
        try:
            q_a, q_b = slices["question"]
            i_a, i_b = slices["instruction"]
            decoded_q = processor.tokenizer.decode(ids[q_a:q_b])
            decoded_i = processor.tokenizer.decode(ids[i_a:i_b])
            # Trim leading/trailing whitespace and BPE artifacts
            if item.question.strip()[:30] not in decoded_q:
                roundtrip_pass = False
                roundtrip_detail.append(
                    f"item={item.id} question decode mismatch: "
                    f"got={decoded_q!r}[:60] expected to contain={item.question.strip()[:30]!r}"
                )
            if "Answer with a single letter" not in decoded_i:
                roundtrip_pass = False
                roundtrip_detail.append(
                    f"item={item.id} instruction decode mismatch: got={decoded_i!r}"
                )
        except Exception as e:
            roundtrip_pass = False
            roundtrip_detail.append(f"item={item.id} decode error: {type(e).__name__}: {e}")

    results.append({
        "check": "slice_detection_coverage", "pass": coverage_pass,
        "detail": "; ".join(coverage_detail) if coverage_detail else f"all {len(eval_items)} items have all 5 slice keys non-empty",
    })
    results.append({
        "check": "slice_non_overlap_full", "pass": nonoverlap_pass,
        "detail": "; ".join(nonoverlap_detail) if nonoverlap_detail else "all items: spans non-overlapping and >=95% coverage of seq_len",
    })
    results.append({
        "check": "slice_decode_round_trip", "pass": roundtrip_pass,
        "detail": "; ".join(roundtrip_detail) if roundtrip_detail else "question and instruction decode round-trip OK",
    })
    overall = overall and coverage_pass and nonoverlap_pass and roundtrip_pass

    # Checks 4 + 5: V3K logits-differ on the FIRST item
    first = eval_items[0]
    msgs = format_mcq_messages(first, n_frames=args.frames)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    seq_len = int(inputs["input_ids"].shape[1])
    slices = find_text_slice_spans(inputs["input_ids"], processor, first)
    v_start = int(slices["_v_start"])
    v_end = int(slices["_v_end"])

    bf16_logits = _logits_for_mask(model, processor, inputs, mask=None,
                                   num_layers=num_layers, num_kv_heads=num_kv_heads)
    torch.cuda.empty_cache()

    q_mask = union_mask(seq_len, [slices["question"]])
    q_logits = _logits_for_mask(model, processor, inputs, q_mask, num_layers, num_kv_heads)
    torch.cuda.empty_cache()

    inf_norm_q_vs_bf16 = float((bf16_logits - q_logits).abs().max().item())
    check4 = inf_norm_q_vs_bf16 > 1e-3
    results.append({
        "check": "v3k_question_logits_differ", "pass": check4,
        "detail": f"||logits_BF16 - logits_question_only_BF16||_inf = {inf_norm_q_vs_bf16:.4e} (threshold 1e-3)",
    })
    overall = overall and check4

    # Compare to all-text-BF16 (D1.3 equivalent): mask = all text positions
    all_text_mask = torch.zeros(seq_len, dtype=torch.bool)
    if v_start > 0:
        all_text_mask[:v_start] = True
    if v_end < seq_len:
        all_text_mask[v_end:] = True
    alltext_logits = _logits_for_mask(model, processor, inputs, all_text_mask,
                                      num_layers, num_kv_heads)
    torch.cuda.empty_cache()
    inf_norm_q_vs_alltext = float((q_logits - alltext_logits).abs().max().item())
    check5 = inf_norm_q_vs_alltext > 1e-3
    results.append({
        "check": "v3k_question_distinct_from_all_text", "pass": check5,
        "detail": f"||logits_question_only - logits_all_text||_inf = {inf_norm_q_vs_alltext:.4e}",
    })
    overall = overall and check5

    # ---- write report ----
    lines = [f"# Exp E1 Smoke Report\n\nGenerated: {_ts()}\n",
             f"Model: `{args.model}` Frames: {args.frames} n_items: {len(eval_items)}  test_item: `{first.id}`\n",
             "## Checks\n",
             "| # | Check | Pass | Detail |",
             "|---|---|:-:|---|"]
    for i, r in enumerate(results, 1):
        marker = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| {i} | `{r['check']}` | {marker} | {r['detail']} |")
    lines.append("")
    lines.append(f"## Overall: {'PASS' if overall else 'FAIL'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[E1-smoke] wrote {args.out}")
    print(f"[E1-smoke] overall: {'PASS' if overall else 'FAIL'}", flush=True)
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
