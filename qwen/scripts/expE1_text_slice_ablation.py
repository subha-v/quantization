"""Experiment E1: Text-K slice ablation on Qwen2.5-VL × LongVideoBench.

V always at INT4. Visual K always at INT4. Only text K varied per condition via
the `BitController` V3K mode using a per-token mask.

Two-pass design:

  Pass A — fixed-slice conditions (E1.2-E1.8). One JSONL row per (item, condition).
    E1.2  HeaderOnly                = header
    E1.3  QuestionOnly              = question
    E1.4  OptionsOnly               = options
    E1.5  InstrAnsPrefix            = instruction + answer_prefix
    E1.6  QuestionOptions           = question + options
    E1.7  OptionsAnsPrefix          = options + answer_prefix
    E1.8  QuestionOptionsAnsPrefix  = question + options + answer_prefix

  Pass B — control conditions (E1.9 random ×3 seeds, E1.10 K-residual top-N).
    Budget N = global median across the 200 items of (per-item best-fixed-slice
    token count). Computed from Pass A's JSONL.

E1.0 (uniform INT4 K/V floor) and E1.1 (all-text-K BF16) are reused from
`expD1_crossmodal_kv.jsonl` — no re-run.

Output: qwen/results/expE1_text_slice_ablation.jsonl (Pass A and Pass B append to
the same file, distinguished by `condition` field).
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data_longvideobench import (
    DEFAULT_SPLIT_FILE,
    LVBItem,
    answer_token_ids,
    filter_items,
    format_mcq_messages,
    load_all_items,
    load_split,
)
from fake_quant_kv_cache import BitController, FakeQuantKVCache
from text_slices import (
    capture_text_k_residuals,
    find_text_slice_spans,
    positions_to_mask,
    text_positions,
    union_mask,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

K_HI, K_LO, V_BITS = 16, 4, 4

PASS_A_CONDITIONS: list[tuple[str, list[str]]] = [
    ("E1_2_HeaderOnly",                ["header"]),
    ("E1_3_QuestionOnly",              ["question"]),
    ("E1_4_OptionsOnly",               ["options"]),
    ("E1_5_InstrAnsPrefix",            ["instruction", "answer_prefix"]),
    ("E1_6_QuestionOptions",           ["question", "options"]),
    ("E1_7_OptionsAnsPrefix",          ["options", "answer_prefix"]),
    ("E1_8_QuestionOptionsAnsPrefix",  ["question", "options", "answer_prefix"]),
]


# ===================================================================
# Helpers
# ===================================================================


def _option_logprobs_and_pred(out, processor, n_options: int) -> tuple[list[float], int]:
    answer_ids = answer_token_ids(processor, n=n_options)
    first_logits = out.scores[0]
    logp = torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist()
    pred = int(max(range(n_options), key=lambda i: logp[i]))
    return logp, pred


def _answer_margin(logp: list[float], correct: int) -> float:
    if not logp:
        return float("nan")
    others = [v for i, v in enumerate(logp) if i != correct]
    if not others:
        return float("nan")
    return float(logp[correct] - max(others))


def _avg_kv_bits_for_mask(mask: torch.Tensor, k_hi: int, k_lo: int, v_bits: int) -> float:
    n = int(mask.numel())
    n_hi = int(mask.sum().item())
    n_lo = n - n_hi
    bits_K = (k_hi * n_hi + k_lo * n_lo) / max(1, n)
    return float((bits_K + v_bits) / 2.0)


def _append_progress(progress_log: Path, line: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_log, "a") as f:
        f.write(f"[{ts}] {line}\n")
        f.flush()


# ===================================================================
# Per-item Pass A
# ===================================================================


@torch.no_grad()
def _run_v3k_forward(model, inputs, mask: torch.Tensor, num_layers: int, num_kv_heads: int):
    """Run a single V3K forward; return (out, latency_ms)."""
    controller = BitController(
        num_layers=num_layers, num_kv_heads=num_kv_heads, mode="V3K",
        default_k_bits=K_LO, default_v_bits=V_BITS,
    )
    controller.set_global(k_bits=K_LO, v_bits=V_BITS)  # V always INT4
    for L in range(num_layers):
        controller.set_protected_mask(L, mask, hi_bits=K_HI, lo_bits=K_LO)
    cache = FakeQuantKVCache(controller)
    t0 = time.perf_counter()
    out = model.generate(
        **inputs, past_key_values=cache, max_new_tokens=1, do_sample=False,
        return_dict_in_generate=True, output_scores=True, use_cache=True,
    )
    return out, (time.perf_counter() - t0) * 1000.0


@torch.no_grad()
def run_item_passA(
    model, processor, item: LVBItem, n_frames: int,
    num_layers: int, num_kv_heads: int, bf16_pred: Optional[int] = None,
) -> list[dict]:
    """Run Pass A conditions on one item; return list of JSONL rows."""
    from qwen_vl_utils import process_vision_info  # type: ignore

    n_options = len(item.candidates)
    correct = item.correct_choice

    msgs = format_mcq_messages(item, n_frames=n_frames)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    seq_len = int(inputs["input_ids"].shape[1])

    slices = find_text_slice_spans(inputs["input_ids"], processor, item)
    warnings = list(slices.get("_warnings", []))
    v_start = int(slices["_v_start"])
    v_end = int(slices["_v_end"])

    # bf16_correct join field — caller may pass in (from D0/D1); fall back to NaN.
    bf16_correct = bool(bf16_pred == correct) if bf16_pred is not None else None

    rows: list[dict] = []
    for cond_name, slice_keys in PASS_A_CONDITIONS:
        ranges = []
        for k in slice_keys:
            sp = slices[k]
            if isinstance(sp, tuple) and sp[0] >= 0 and sp[1] > sp[0]:
                ranges.append(sp)
        mask = union_mask(seq_len, ranges)
        avg_bits = _avg_kv_bits_for_mask(mask, K_HI, K_LO, V_BITS)
        n_protected = int(mask.sum().item())

        out, latency_ms = _run_v3k_forward(model, inputs, mask, num_layers, num_kv_heads)
        logp, pred = _option_logprobs_and_pred(out, processor, n_options)
        margin = _answer_margin(logp, correct)
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rows.append({
            "phase": "E1_A",
            "item_id": item.id,
            "duration_bucket": item.duration_bucket,
            "duration_seconds": item.duration_seconds,
            "n_options": n_options,
            "correct_choice": correct,
            "n_frames": n_frames,
            "condition": cond_name,
            "k_text_slice_keys": list(slice_keys),
            "n_text_protected_tokens": n_protected,
            "avg_kv_bits": avg_bits,
            "k_hi": K_HI,
            "k_lo": K_LO,
            "v_bits": V_BITS,
            "seq_len": seq_len,
            "visual_token_start": v_start,
            "visual_token_end": v_end,
            "slice_spans": {k: list(slices[k]) for k in
                            ("header", "visual_wrapper", "question", "options",
                             "instruction", "answer_prefix")
                            if isinstance(slices[k], tuple)},
            "slice_match_warnings": warnings,
            "pred_choice": int(pred),
            "is_correct": bool(pred == correct),
            "option_logprobs": [float(x) for x in logp],
            "answer_margin": float(margin),
            "latency_ms": float(latency_ms),
            "bf16_pred": bf16_pred,
            "bf16_correct": bf16_correct,
        })
    return rows


# ===================================================================
# Per-item Pass B
# ===================================================================


def _global_median_budget(passA_jsonl: Path, fixed_cond_ids: list[str]) -> int:
    """Per-item best slice = max accuracy condition (ties broken by lower
    n_text_protected_tokens — prefer smaller slice). Token count of that slice
    becomes the per-item budget; return median across items.
    """
    rows = [json.loads(l) for l in passA_jsonl.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("phase") == "E1_A" and r["condition"] in fixed_cond_ids]
    by_item: dict[str, list[dict]] = {}
    for r in rows:
        by_item.setdefault(r["item_id"], []).append(r)
    best_counts: list[int] = []
    for iid, rs in by_item.items():
        # Pick highest accuracy; tiebreak smaller token budget.
        rs_sorted = sorted(rs, key=lambda r: (-int(r["is_correct"]), int(r["n_text_protected_tokens"])))
        best_counts.append(int(rs_sorted[0]["n_text_protected_tokens"]))
    if not best_counts:
        raise RuntimeError(f"No E1_A rows in {passA_jsonl} for fixed_cond_ids={fixed_cond_ids}")
    return int(np.median(np.asarray(best_counts)))


@torch.no_grad()
def run_item_passB(
    model, processor, item: LVBItem, n_frames: int, budget_N: int,
    num_layers: int, num_kv_heads: int, seeds: list[int],
    bf16_pred: Optional[int] = None,
) -> list[dict]:
    """Run Pass B conditions (E1.9 random x len(seeds) + E1.10 residual) for one item."""
    from qwen_vl_utils import process_vision_info  # type: ignore

    n_options = len(item.candidates)
    correct = item.correct_choice

    msgs = format_mcq_messages(item, n_frames=n_frames)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    seq_len = int(inputs["input_ids"].shape[1])

    slices = find_text_slice_spans(inputs["input_ids"], processor, item)
    warnings = list(slices.get("_warnings", []))
    v_start = int(slices["_v_start"])
    v_end = int(slices["_v_end"])
    text_pos = text_positions(seq_len, v_start, v_end)
    n_text = len(text_pos)
    bf16_correct = bool(bf16_pred == correct) if bf16_pred is not None else None

    budget = min(int(budget_N), n_text)
    rows: list[dict] = []

    # E1.9 random seeds
    for seed in seeds:
        rng = torch.Generator()
        rng.manual_seed(int(seed) * 1000003 + zlib.crc32(item.id.encode()) % 10_000)
        perm = torch.randperm(n_text, generator=rng).tolist()
        pick = sorted(text_pos[i] for i in perm[:budget])
        mask = positions_to_mask(seq_len, pick)
        avg_bits = _avg_kv_bits_for_mask(mask, K_HI, K_LO, V_BITS)
        out, latency_ms = _run_v3k_forward(model, inputs, mask, num_layers, num_kv_heads)
        logp, pred = _option_logprobs_and_pred(out, processor, n_options)
        margin = _answer_margin(logp, correct)
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rows.append({
            "phase": "E1_B",
            "item_id": item.id,
            "duration_bucket": item.duration_bucket,
            "duration_seconds": item.duration_seconds,
            "n_options": n_options,
            "correct_choice": correct,
            "n_frames": n_frames,
            "condition": f"E1_9_RandomTextK_seed{seed}",
            "k_text_slice_keys": [],
            "n_text_protected_tokens": int(mask.sum().item()),
            "budget_N": int(budget),
            "avg_kv_bits": avg_bits,
            "k_hi": K_HI, "k_lo": K_LO, "v_bits": V_BITS,
            "seq_len": seq_len,
            "visual_token_start": v_start, "visual_token_end": v_end,
            "slice_match_warnings": warnings,
            "seed": int(seed),
            "pred_choice": int(pred),
            "is_correct": bool(pred == correct),
            "option_logprobs": [float(x) for x in logp],
            "answer_margin": float(margin),
            "latency_ms": float(latency_ms),
            "bf16_pred": bf16_pred,
            "bf16_correct": bf16_correct,
        })

    # E1.10 K-residual-top text positions
    residuals = capture_text_k_residuals(model, processor, inputs,
                                         num_kv_heads=num_kv_heads, num_layers=num_layers)
    # residuals is [seq_len]; we only score text positions
    text_resid = torch.tensor([float(residuals[p].item()) for p in text_pos])
    top_idx = torch.argsort(text_resid, descending=True)[:budget].tolist()
    pick = sorted(text_pos[i] for i in top_idx)
    mask = positions_to_mask(seq_len, pick)
    avg_bits = _avg_kv_bits_for_mask(mask, K_HI, K_LO, V_BITS)
    out, latency_ms = _run_v3k_forward(model, inputs, mask, num_layers, num_kv_heads)
    logp, pred = _option_logprobs_and_pred(out, processor, n_options)
    margin = _answer_margin(logp, correct)
    del out, residuals, text_resid
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    rows.append({
        "phase": "E1_B",
        "item_id": item.id,
        "duration_bucket": item.duration_bucket,
        "duration_seconds": item.duration_seconds,
        "n_options": n_options,
        "correct_choice": correct,
        "n_frames": n_frames,
        "condition": "E1_10_KResidTopTextK",
        "k_text_slice_keys": [],
        "n_text_protected_tokens": int(mask.sum().item()),
        "budget_N": int(budget),
        "avg_kv_bits": avg_bits,
        "k_hi": K_HI, "k_lo": K_LO, "v_bits": V_BITS,
        "seq_len": seq_len,
        "visual_token_start": v_start, "visual_token_end": v_end,
        "slice_match_warnings": warnings,
        "seed": None,
        "pred_choice": int(pred),
        "is_correct": bool(pred == correct),
        "option_logprobs": [float(x) for x in logp],
        "answer_margin": float(margin),
        "latency_ms": float(latency_ms),
        "bf16_pred": bf16_pred,
        "bf16_correct": bf16_correct,
    })
    return rows


# ===================================================================
# Driver
# ===================================================================


def _load_d1_bf16_preds(d1_jsonl: Path) -> dict[str, int]:
    """Map item_id -> D1.0/D1.3's bf16_pred (= D0 full64 prediction)."""
    if not d1_jsonl.exists():
        return {}
    out = {}
    for line in d1_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("phase") == "D1" and "bf16_pred" in r and r["item_id"] not in out:
            out[r["item_id"]] = int(r["bf16_pred"])
    return out


def run_passA(model, processor, items, n_frames, num_layers, num_kv_heads,
              bf16_preds: dict[str, int], out_jsonl: Path, progress_every: int = 5):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_log = out_jsonl.with_name(out_jsonl.stem + ".progress.log")
    _append_progress(progress_log, f"START PassA n_items={len(items)} frames={n_frames}")
    t0 = time.perf_counter()
    n_done, n_failed, n_rows = 0, 0, 0
    with open(out_jsonl, "a") as f:
        for i, it in enumerate(items):
            try:
                rows = run_item_passA(model, processor, it, n_frames=n_frames,
                                      num_layers=num_layers, num_kv_heads=num_kv_heads,
                                      bf16_pred=bf16_preds.get(it.id))
            except Exception as e:
                n_failed += 1
                _append_progress(progress_log, f"WARN item={it.id} skipped: {type(e).__name__}: {e}")
                continue
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n_done += 1
            n_rows += len(rows)
            if (i + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            done = i + 1
            if done % progress_every == 0 or done == len(items):
                elapsed = time.perf_counter() - t0
                rate = elapsed / max(1, n_done)
                eta = max(0.0, rate * (len(items) - done))
                _append_progress(progress_log,
                                 f"PassA {done}/{len(items)} ok={n_done} rows={n_rows} "
                                 f"failed={n_failed} elapsed={timedelta(seconds=int(elapsed))} "
                                 f"ETA={timedelta(seconds=int(eta))}")
    _append_progress(progress_log, f"DONE PassA ok={n_done} rows={n_rows} failed={n_failed}")


def run_passB(model, processor, items, n_frames, num_layers, num_kv_heads,
              bf16_preds: dict[str, int], out_jsonl: Path, seeds: list[int],
              budget_N: int, progress_every: int = 5):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_log = out_jsonl.with_name(out_jsonl.stem + ".progress.log")
    _append_progress(progress_log, f"START PassB n_items={len(items)} N={budget_N} seeds={seeds}")
    t0 = time.perf_counter()
    n_done, n_failed, n_rows = 0, 0, 0
    with open(out_jsonl, "a") as f:
        for i, it in enumerate(items):
            try:
                rows = run_item_passB(model, processor, it, n_frames=n_frames,
                                      budget_N=budget_N,
                                      num_layers=num_layers, num_kv_heads=num_kv_heads,
                                      seeds=seeds, bf16_pred=bf16_preds.get(it.id))
            except Exception as e:
                n_failed += 1
                _append_progress(progress_log, f"WARN item={it.id} skipped: {type(e).__name__}: {e}")
                continue
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n_done += 1
            n_rows += len(rows)
            if (i + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            done = i + 1
            if done % progress_every == 0 or done == len(items):
                elapsed = time.perf_counter() - t0
                rate = elapsed / max(1, n_done)
                eta = max(0.0, rate * (len(items) - done))
                _append_progress(progress_log,
                                 f"PassB {done}/{len(items)} ok={n_done} rows={n_rows} "
                                 f"failed={n_failed} elapsed={timedelta(seconds=int(elapsed))} "
                                 f"ETA={timedelta(seconds=int(eta))}")
    _append_progress(progress_log, f"DONE PassB ok={n_done} rows={n_rows} failed={n_failed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--phase", choices=["passA", "passB"], required=True)
    ap.add_argument("--d1_jsonl", type=Path,
                    default=RESULTS_DIR / "expD1_crossmodal_kv.jsonl")
    ap.add_argument("--out", type=Path,
                    default=RESULTS_DIR / "expE1_text_slice_ablation.jsonl")
    ap.add_argument("--budget_N", type=int, default=0,
                    help="(passB only) override; default = global median from passA")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--progress_every", type=int, default=5)
    args = ap.parse_args()

    items = load_all_items()
    split = load_split(args.split_file)
    eval_items = filter_items(items, split["eval"])
    if args.limit:
        eval_items = eval_items[: args.limit]
    print(f"[expE1] phase={args.phase} eval_items={len(eval_items)}", flush=True)

    bf16_preds = _load_d1_bf16_preds(args.d1_jsonl)
    print(f"[expE1] loaded {len(bf16_preds)} D0/D1 BF16 preds for join", flush=True)

    if args.phase == "passA" and not args.append and args.out.exists():
        args.out.unlink()

    from run_inference import load_model
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    print(f"[expE1] model loaded; num_layers={num_layers} num_kv_heads={num_kv_heads}",
          flush=True)

    if args.phase == "passA":
        run_passA(model, processor, eval_items, n_frames=args.frames,
                  num_layers=num_layers, num_kv_heads=num_kv_heads,
                  bf16_preds=bf16_preds, out_jsonl=args.out,
                  progress_every=args.progress_every)
    else:  # passB
        if args.budget_N > 0:
            budget_N = args.budget_N
            print(f"[expE1] passB using --budget_N={budget_N} (override)", flush=True)
        else:
            budget_N = _global_median_budget(args.out, [c[0] for c in PASS_A_CONDITIONS])
            print(f"[expE1] passB computed global median budget N={budget_N} from {args.out}",
                  flush=True)
        run_passB(model, processor, eval_items, n_frames=args.frames,
                  num_layers=num_layers, num_kv_heads=num_kv_heads,
                  bf16_preds=bf16_preds, out_jsonl=args.out,
                  seeds=args.seeds, budget_N=budget_N,
                  progress_every=args.progress_every)


if __name__ == "__main__":
    main()
