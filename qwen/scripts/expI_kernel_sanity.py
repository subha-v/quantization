"""Kernel-regression sanity check for Exp I.

Picks 3 item_ids from Exp G's stage-3 H6 rows, runs the current expI code
path (I3_TempWin2_128f, which uses the same H6_KIVI_TempWin2 cfg under the
extended _kivi_temporal_window kernel), and asserts the per-item
option_logprobs match expG's stored values within a tight tolerance.

If the outputs match, the seed=1 stage-1 result (I3 underperforming F4) is
real split noise, not a kernel regression. If they diverge, we have a bug
introduced by the Exp I kernel changes.

Usage:
  python expI_kernel_sanity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from data_longvideobench import (
    LVBItem, answer_token_ids, filter_items, format_mcq_messages, load_all_items,
)
from fake_quant_kv_cache import BitController, FakeQuantKVCache
from k_quantizers import build_f_conditions
from text_slices import find_text_slice_spans
from run_inference import load_model


SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPTS_DIR.parent / "results"
EXPG_STAGE3 = RESULTS_DIR / "expG_frame_stage3.jsonl"


def main():
    print("=" * 70)
    print("Exp I kernel-regression sanity check")
    print("=" * 70)

    # 1. Load expG H6 rows.
    rows = [json.loads(l) for l in open(EXPG_STAGE3) if l.strip()]
    h6_rows = [r for r in rows if r.get("condition") == "H6_KIVI_TempWin2_128f"
               and r.get("option_logprobs")
               and not r.get("error")
               and not r.get("skipped")]
    print(f"\nFound {len(h6_rows)} H6_KIVI_TempWin2_128f rows in expG stage 3")
    if len(h6_rows) < 3:
        raise SystemExit("not enough H6 rows for sanity check")

    # Take 3 from different buckets.
    by_bucket = {}
    for r in h6_rows:
        by_bucket.setdefault(r["duration_bucket"], []).append(r)
    selected = []
    for bucket in ("short", "mid", "long"):
        if by_bucket.get(bucket):
            selected.append(by_bucket[bucket][0])
    if not selected:
        selected = h6_rows[:3]
    selected = selected[:3]

    print(f"\nSelected {len(selected)} items for replay:")
    for r in selected:
        print(f"  item_id={r['item_id']!r} bucket={r['duration_bucket']!r}  "
              f"pred={r['pred_choice']} corr={r['is_correct']}  "
              f"logp[:3]={[f'{x:.4f}' for x in r['option_logprobs'][:3]]}")

    target_ids = {r["item_id"] for r in selected}
    expg_by_id = {r["item_id"]: r for r in selected}

    # 2. Load LVB items + filter.
    items_all = load_all_items()
    by_id = {it.id: it for it in items_all}
    items = [by_id[iid] for iid in target_ids if iid in by_id]
    print(f"\nResolved {len(items)} of {len(target_ids)} item_ids in LVB dataset")
    if len(items) != len(target_ids):
        missing = target_ids - set(by_id)
        raise SystemExit(f"missing items: {missing}")

    # 3. Load model.
    print("\nLoading model...")
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model, processor = load_model(model_id, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    n_layers = len(getattr(model.model, "layers", []) or model.model.language_model.layers)
    n_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    print(f"model loaded; n_layers={n_layers} n_kv_heads={n_kv_heads}")

    # 4. Build H6 cfg via build_f_conditions (this is the SAME path expI uses
    # for I3_TempWin2_128f via the f_cfg_name lookup).
    fc = {cfg.name: cfg for cfg in build_f_conditions(calib=None)}
    cfg_h6 = fc["H6_KIVI_TempWin2"]
    print(f"\ncfg_h6: kind={cfg_h6.kind} n_temporal_windows={cfg_h6.n_temporal_windows} "
          f"temporal_mode={cfg_h6.temporal_mode} n_outliers={cfg_h6.n_outliers} "
          f"v_per_channel_seq={cfg_h6.v_per_channel_seq}")

    # 5. Replay H6 forward on each item; compare logprobs.
    from qwen_vl_utils import process_vision_info  # type: ignore

    results = []
    for item in items:
        n_frames = 128
        msgs = format_mcq_messages(item, n_frames=n_frames)
        prompt_text = processor.apply_chat_template(msgs, tokenize=False,
                                                    add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        slices = find_text_slice_spans(inputs["input_ids"], processor, item)
        v_start = int(slices.get("_v_start", -1))
        v_end = int(slices.get("_v_end", -1))
        seq_len = int(inputs["input_ids"].shape[1])
        role_spans = {k: tuple(slices[k]) for k in
                      ("header", "question", "options", "instruction", "answer_prefix")
                      if isinstance(slices.get(k), tuple)}
        if v_start >= 0 and v_end > v_start:
            role_spans["visual"] = (v_start, v_end)
        slice_info = dict(v_start=v_start, v_end=v_end, seq_len=seq_len, role_spans=role_spans)

        ctrl = BitController(num_layers=n_layers, num_kv_heads=n_kv_heads,
                             mode="V1", default_k_bits=4, default_v_bits=4)
        cache = FakeQuantKVCache(ctrl, k_quantizer_config=cfg_h6)
        cache.set_slice_info(slice_info)

        with torch.no_grad():
            out = model.generate(**inputs, past_key_values=cache, max_new_tokens=1,
                                 do_sample=False, return_dict_in_generate=True,
                                 output_scores=True, use_cache=True)
        n_options = len(item.candidates)
        answer_ids = answer_token_ids(processor, n=n_options)
        first_logits = out.scores[0]
        new_logp = torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist()
        new_pred = int(max(range(n_options), key=lambda i: new_logp[i]))

        old = expg_by_id[item.id]
        old_logp = old["option_logprobs"]
        old_pred = old["pred_choice"]

        delta = max(abs(a - b) for a, b in zip(new_logp, old_logp))
        match = delta < 1e-3 and new_pred == old_pred
        results.append({
            "item_id": item.id, "bucket": item.duration_bucket,
            "old_pred": old_pred, "new_pred": new_pred,
            "old_logp": old_logp, "new_logp": new_logp,
            "max_abs_delta": delta, "match": match,
        })

    print("\n" + "=" * 70)
    print("Per-item comparison: expG H6 stage-3 vs current expI I3 path")
    print("=" * 70)
    all_match = True
    for r in results:
        status = "MATCH" if r["match"] else "MISMATCH"
        all_match = all_match and r["match"]
        print(f"\n[{status}] item_id={r['item_id']!r}  bucket={r['bucket']}")
        print(f"  old_pred={r['old_pred']}  new_pred={r['new_pred']}  "
              f"max_abs_delta={r['max_abs_delta']:.6f}")
        print(f"  old_logp = {[f'{x:.4f}' for x in r['old_logp']]}")
        print(f"  new_logp = {[f'{x:.4f}' for x in r['new_logp']]}")

    print("\n" + "=" * 70)
    if all_match:
        print("PASS: H6 kernel produces identical results in current code.")
        print("      The seed=1 stage-1 result (I3 underperforming F4) is real")
        print("      split noise, not a kernel regression. Safe to proceed to Stage 3.")
    else:
        print("FAIL: H6 kernel diverges from prior expG H6 outputs.")
        print("      A bug was introduced by the Exp I kernel changes.")
    print("=" * 70)


if __name__ == "__main__":
    main()
