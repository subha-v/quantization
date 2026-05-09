"""Experiment G — Frame-scaling under fixed KV memory budget.

Tests whether KIVI-style 4-bit KV (Exp F's F4 quantizer) lets Qwen2.5-VL
process more video frames on LongVideoBench under the same theoretical KV
memory budget as 64-frame BF16.

The headline test:
  G4 (256f F4 INT4)  vs  G0 (64f BF16)
  same theoretical KV memory: (256 * 4) / (64 * 16) == 1.00
  4x more frames at the same memory budget.

Stage-1 conditions (n=64 balanced 16/bucket; same split as F-suite Stage 1):

  ID  Frames   KV format         rel_kv_mem   Purpose
  G0  64       BF16              1.00x        Baseline ceiling (anchors A1=0.565)
  G1  64       F4 INT4           0.25x        Anchors F4=0.545 from F-suite Stage 3
  G2  128      BF16              2.00x        Upper-bound for what extra frames buy
  G3  128      F4 INT4           0.50x        Memory-saving point: 2x frames, half KV mem
  G4  256      F4 INT4           1.00x        HEADLINE: 4x frames at G0 budget
  G5  128      F9 (4.75 KV bits) 0.59x        Zero-loss point at 2x frames
  G6  256      F9 (4.75 KV bits) 1.19x        Zero-loss point at 4x frames

  G7 / G8 are post-processes (see expG_cascade.py + expG_type_adaptive.py),
  not separate forwards.

Compute structure:
  Outer loop over frame counts ({64, 128, 256}); inner loop over K-quantizer
  configs at that frame count. Visual prefill (image-token construction,
  position-id generation, find_text_slice_spans, inputs dict) is computed
  ONCE per (item, frame_count) and reused across the F0/F4/F9 conditions
  that share that frame count -- saves ~50% wall-time vs the naive nested
  loop.

Reused infrastructure:
  - k_quantizers.build_f_conditions: F0/F4/F9 KQuantizerConfig instances
  - fake_quant_kv_cache.{BitController, FakeQuantKVCache}
  - expF_kquant_screen._option_logprobs_and_pred / _answer_margin /
    _compute_three_bit_columns / _run_condition_forward / backfill_bf16_join /
    ensure_stage_split (canonical n=64 split file)
  - data_longvideobench.format_mcq_messages(item, n_frames=...): the only
    frame-count plumbing point
  - run_inference.load_model

Calibration reuse:
  F-suite calibration NPZ at frames=64 is reused as-is. K-channel outlier
  indices are post-RoPE and frame-count-independent in theory; smoke check 8
  (opt-in via EXPG_RIGOR_HIGH=1) verifies on 8 cal items at frames=256.

JSONL row schema (in addition to F-suite base fields):
  experiment            "G"
  condition             "G0_BF16" .. "G6_F9_256f"  (G7/G8 emitted by post-process)
  condition_class       "fixed_frame" | "cascade" | "type_adaptive"
  frames                concrete frames decoded for this row
  assigned_frames       same as frames for fixed_frame conditions
  relative_kv_memory    (frames * avg_kv_bits) / (64 * 16)

Usage:
  python expG_frame_scaling.py --stage 1 \
      --calib_npz qwen/calibration/expF_kcalib_Qwen2.5-VL-7B-Instruct_frames64.npz
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from data_longvideobench import (
    LVBItem,
    answer_token_ids,
    filter_items,
    format_mcq_messages,
    load_all_items,
    load_split,
    make_split,
    save_split,
)
from fake_quant_kv_cache import BitController, FakeQuantKVCache
from k_quantizers import KQuantizerConfig, build_f_conditions
from text_slices import find_text_slice_spans

# Reuse F-suite helpers verbatim.
from expF_kquant_screen import (
    STAGE_TARGETS,
    _answer_margin,
    _compute_three_bit_columns,
    _option_logprobs_and_pred,
    _run_condition_forward,
    backfill_bf16_join,
    ensure_stage_split,
    load_calib,
    stage_split_path,
)


SCRIPTS_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPTS_DIR.parent / "calibration"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"


# ===================================================================
# G condition specs
# ===================================================================


# Fixed-frame conditions only. G7 (cascade) and G8 (type-adaptive) are
# emitted by post-process scripts that re-stitch these rows.
FIXED_FRAME_CONDITIONS = [
    # (g_name, frames, f_cfg_name, mode)
    ("G0_BF16",      64,  "F0_BF16",              "bf16"),
    ("G1_F4_64f",    64,  "F4_KIVI_PerChannelSeq", "v1_kcfg"),
    ("G2_BF16_128f", 128, "F0_BF16",              "bf16"),
    ("G3_F4_128f",   128, "F4_KIVI_PerChannelSeq", "v1_kcfg"),
    ("G4_F4_256f",   256, "F4_KIVI_PerChannelSeq", "v1_kcfg"),
    ("G5_F9_128f",   128, "F9_KIVI_Outlier16",    "v1_kcfg"),
    ("G6_F9_256f",   256, "F9_KIVI_Outlier16",    "v1_kcfg"),
]


CONDITIONS_NEEDING_CALIB = {"G5_F9_128f", "G6_F9_256f"}  # F9 needs outlier indices


def build_g_conditions(calib: Optional[dict]) -> list[dict]:
    """Return G-suite condition specs as dicts matching the shape that
    `expF_kquant_screen._run_condition_forward` expects: {name, mode, cfg, frames}.

    `frames` is a new key on top of the F-suite shape; the G driver uses it to
    drive its outer-loop sort.
    """
    fc = build_f_conditions(calib=calib)
    by_name = {cfg.name: cfg for cfg in fc}
    out: list[dict] = []
    for g_name, frames, f_cfg_name, mode in FIXED_FRAME_CONDITIONS:
        cfg = by_name[f_cfg_name]
        out.append({
            "name": g_name,
            "mode": mode,
            "cfg": cfg,
            "frames": frames,
            "f_cfg_name": f_cfg_name,
        })
    return out


# ===================================================================
# Per-item runner (frame-aware)
# ===================================================================


def relative_kv_memory(frames: int, avg_kv_bits: float) -> float:
    """rel_mem = (frames * avg_kv_bits) / (64 * 16). G0 anchors at 1.00."""
    return (frames * avg_kv_bits) / (64.0 * 16.0)


@torch.no_grad()
def run_item_at_frames(model, processor, item: LVBItem, n_frames: int,
                       num_layers: int, num_kv_heads: int,
                       conditions_at_this_frame: list[dict],
                       stage: int, calibration_id: Optional[str]) -> list[dict]:
    """Build the inputs ONCE for this (item, n_frames) and run all conditions
    that need this frame count. Mirrors expF.run_item but factored to amortize
    the heavy visual prefill across K-quantizer configs."""
    from qwen_vl_utils import process_vision_info  # type: ignore

    n_options = len(item.candidates)
    correct = item.correct_choice

    msgs = format_mcq_messages(item, n_frames=n_frames)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    seq_len = int(inputs["input_ids"].shape[1])
    slices = find_text_slice_spans(inputs["input_ids"], processor, item)
    v_start = int(slices.get("_v_start", -1))
    v_end = int(slices.get("_v_end", -1))
    role_spans = {k: tuple(slices[k]) for k in
                  ("header", "question", "options", "instruction", "answer_prefix")
                  if isinstance(slices.get(k), tuple)}
    if v_start >= 0 and v_end > v_start:
        role_spans["visual"] = (v_start, v_end)
    slice_info = dict(v_start=v_start, v_end=v_end, seq_len=seq_len, role_spans=role_spans)

    rows: list[dict] = []
    for cond in conditions_at_this_frame:
        try:
            out, lat_ms, bits_tuple = _run_condition_forward(
                model, processor, item, n_frames=n_frames,
                num_layers=num_layers, num_kv_heads=num_kv_heads,
                cond=cond, slice_info=slice_info, inputs=inputs,
            )
            avg_k_bits, avg_v_bits, avg_kv_bits = bits_tuple
            logp, pred = _option_logprobs_and_pred(out, processor, n_options)
            margin = _answer_margin(logp, correct)
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            rows.append({
                "experiment": "G",
                "phase": f"G{stage}",
                "stage": stage,
                "item_id": item.id,
                "duration_bucket": item.duration_bucket,
                "duration_seconds": item.duration_seconds,
                "n_options": n_options,
                "correct_choice": correct,
                "n_frames": n_frames,
                "frames": n_frames,
                "assigned_frames": n_frames,
                "condition": cond["name"],
                "condition_class": "fixed_frame",
                "error": f"{type(e).__name__}: {e}",
                "is_correct": False,
            })
            continue

        cfg_obj = cond.get("cfg")
        rel_mem = relative_kv_memory(n_frames, float(avg_kv_bits))
        row = {
            "experiment": "G",
            "phase": f"G{stage}",
            "stage": stage,
            "item_id": item.id,
            "duration_bucket": item.duration_bucket,
            "duration_seconds": item.duration_seconds,
            "n_options": n_options,
            "correct_choice": correct,
            "n_frames": n_frames,
            "frames": n_frames,
            "assigned_frames": n_frames,
            "condition": cond["name"],
            "condition_class": "fixed_frame",
            "f_cfg_name": cond.get("f_cfg_name"),
            "k_quant_variant": cfg_obj.kind if cfg_obj is not None else cond["mode"],
            "calibration_id": calibration_id,
            "n_outlier_channels": (cfg_obj.n_outliers if cfg_obj is not None else None),
            "k_lo": (int(cfg_obj.bits) if cfg_obj is not None else 4),
            "v_bits": 4 if cond["mode"] != "bf16" else 16,
            "avg_k_bits": float(avg_k_bits),
            "avg_v_bits": float(avg_v_bits),
            "avg_kv_bits": float(avg_kv_bits),
            "relative_kv_memory": float(rel_mem),
            "pred_choice": int(pred),
            "is_correct": bool(pred == correct),
            "option_logprobs": [float(x) for x in logp],
            "answer_margin": float(margin),
            "latency_ms": float(lat_ms),
            "bf16_pred": None,
            "bf16_correct": None,
            "seq_len": seq_len,
            "visual_token_start": v_start,
            "visual_token_end": v_end,
        }
        rows.append(row)
    return rows


# ===================================================================
# Stage runner with outer-frame-loop / inner-condition-loop
# ===================================================================


def _append_progress(progress_log: Path, line: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_log, "a") as f:
        f.write(f"[{ts}] {line}\n")
        f.flush()


def run_stage_g(model, processor, items: list[LVBItem],
                num_layers: int, num_kv_heads: int, conditions: list[dict],
                stage: int, calibration_id: Optional[str],
                out_jsonl: Path, progress_every: int = 5,
                min_free_gb_for_256: int = 70) -> None:
    """Run all G conditions on all items.

    Frame-tier ordering: 64 -> 128 -> 256 (ascending). At each tier, run all
    conditions that need that frame count once per item, sharing the visual
    prefill. Lower-frame tiers run first so that early failures (anchor drift,
    OOM) surface before the expensive 256f tier.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_log = out_jsonl.with_name(out_jsonl.stem + ".progress.log")
    _append_progress(progress_log, f"START stage={stage} n_items={len(items)} "
                                   f"n_conditions={len(conditions)}")

    # Group conditions by frame count.
    by_frames: dict[int, list[dict]] = {}
    for c in conditions:
        by_frames.setdefault(int(c["frames"]), []).append(c)
    frame_tiers = sorted(by_frames.keys())
    _append_progress(progress_log, f"frame tiers: {frame_tiers}")

    t0 = time.perf_counter()
    n_done = 0
    n_failed = 0
    n_rows = 0

    with open(out_jsonl, "a") as f:
        for tier_frames in frame_tiers:
            tier_conds = by_frames[tier_frames]
            tier_names = [c["name"] for c in tier_conds]
            _append_progress(progress_log,
                             f"TIER frames={tier_frames} conditions={tier_names}")

            # Memory precheck for the 256f tier.
            if tier_frames >= 256 and torch.cuda.is_available():
                free_b, _ = torch.cuda.mem_get_info()
                free_gb = free_b / (1024 ** 3)
                _append_progress(progress_log,
                                 f"PRECHECK frames={tier_frames} free_gb={free_gb:.1f}")
                if free_gb < min_free_gb_for_256:
                    _append_progress(
                        progress_log,
                        f"WARN frames={tier_frames} free_gb={free_gb:.1f} < "
                        f"{min_free_gb_for_256}; skipping tier"
                    )
                    for cond in tier_conds:
                        for it in items:
                            f.write(json.dumps({
                                "experiment": "G",
                                "stage": stage,
                                "item_id": it.id,
                                "duration_bucket": it.duration_bucket,
                                "n_frames": tier_frames,
                                "frames": tier_frames,
                                "condition": cond["name"],
                                "condition_class": "fixed_frame",
                                "skipped": True,
                                "skip_reason": f"free_gb={free_gb:.1f} < {min_free_gb_for_256}",
                                "is_correct": False,
                            }) + "\n")
                    f.flush()
                    continue

            t_tier_start = time.perf_counter()
            tier_done = 0
            for i, it in enumerate(items):
                try:
                    rows = run_item_at_frames(
                        model, processor, it, n_frames=tier_frames,
                        num_layers=num_layers, num_kv_heads=num_kv_heads,
                        conditions_at_this_frame=tier_conds,
                        stage=stage, calibration_id=calibration_id,
                    )
                except Exception as e:
                    n_failed += 1
                    _append_progress(progress_log,
                                     f"WARN item={it.id} frames={tier_frames} "
                                     f"skipped: {type(e).__name__}: {e}")
                    continue
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                tier_done += 1
                n_done += 1
                n_rows += len(rows)
                if (i + 1) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                done = i + 1
                if done % progress_every == 0 or done == len(items):
                    elapsed = time.perf_counter() - t_tier_start
                    rate = elapsed / max(1, tier_done)
                    eta = max(0.0, rate * (len(items) - done))
                    _append_progress(
                        progress_log,
                        f"tier{tier_frames} {done}/{len(items)} ok={tier_done} "
                        f"rows_total={n_rows} failed={n_failed} "
                        f"elapsed={timedelta(seconds=int(elapsed))} "
                        f"ETA={timedelta(seconds=int(eta))}"
                    )

            tier_elapsed = time.perf_counter() - t_tier_start
            _append_progress(
                progress_log,
                f"TIER frames={tier_frames} DONE elapsed="
                f"{timedelta(seconds=int(tier_elapsed))} ok={tier_done}"
            )

    total_elapsed = time.perf_counter() - t0
    _append_progress(progress_log, f"DONE stage={stage} ok={n_done} rows={n_rows} "
                                   f"failed={n_failed} "
                                   f"total={timedelta(seconds=int(total_elapsed))}")


def backfill_bf16_join_g(out_jsonl: Path) -> None:
    """G-suite version of F-suite's BF16 join. Backfills bf16_pred and
    bf16_correct on every G row using the G0_BF16 row of the same item_id.

    G0_BF16 is the canonical baseline (64-frame BF16). All G conditions are
    paired against it, including 128f and 256f variants -- the comparison is
    accuracy-vs-G0 at matched / fewer / more KV memory.
    """
    if not out_jsonl.exists():
        return
    rows = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l.strip()]
    bf16_pred: dict[str, int] = {}
    for r in rows:
        if r.get("condition") == "G0_BF16" and "pred_choice" in r:
            bf16_pred[r["item_id"]] = int(r["pred_choice"])
    for r in rows:
        if r.get("condition") == "G0_BF16":
            r["bf16_pred"] = int(r.get("pred_choice", -1))
            r["bf16_correct"] = bool(r["bf16_pred"] == int(r["correct_choice"]))
            continue
        bp = bf16_pred.get(r["item_id"])
        if bp is None:
            continue
        r["bf16_pred"] = int(bp)
        r["bf16_correct"] = bool(bp == int(r["correct_choice"]))
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out_jsonl)
    print(f"[expG] backfilled BF16 join in {out_jsonl} ({len(rows)} rows)")


# ===================================================================
# Driver
# ===================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=1)
    ap.add_argument("--split_file", type=Path, default=None,
                    help="Override the auto-generated stage split file. Defaults to "
                         "the F-suite Stage-N split for paired-with-F analysis.")
    ap.add_argument("--calib_npz", type=Path, default=None,
                    help="Reuse F-suite calibration NPZ (frames=64). F9 needs this.")
    ap.add_argument("--calib_json", type=Path, default=None)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="Subset of G condition names to run. Default: all 7 fixed-frame.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Override: limit eval items count after split selection.")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing JSONL rather than truncating.")
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: results/expG_frame_stage{stage}.jsonl")
    ap.add_argument("--min_free_gb_256", type=int, default=70,
                    help="Skip the 256f tier if GPU free memory < this many GiB.")
    args = ap.parse_args()

    if args.out is None:
        args.out = RESULTS_DIR / f"expG_frame_stage{args.stage}.jsonl"
    if args.split_file is None:
        args.split_file = ensure_stage_split(args.stage)
    # Reuse F-suite calibration at frames=64. K-channel outlier indices are
    # post-RoPE and frame-count-independent, so the same NPZ works for the
    # 128f and 256f conditions.
    if args.calib_npz is None:
        model_short = args.model.split("/")[-1]
        cand = CALIBRATION_DIR / f"expF_kcalib_{model_short}_frames64.npz"
        if cand.exists():
            args.calib_npz = cand
    if args.calib_json is None and args.calib_npz is not None:
        cj = args.calib_npz.with_suffix(".json")
        if cj.exists():
            args.calib_json = cj

    items_all = load_all_items()
    split = load_split(args.split_file)
    eval_items = filter_items(items_all, split["eval"])
    if args.limit:
        eval_items = eval_items[: args.limit]
    print(f"[expG] stage={args.stage} eval_items={len(eval_items)} "
          f"split_file={args.split_file} calib_npz={args.calib_npz}", flush=True)

    calib, _ = load_calib(args.calib_npz, args.calib_json)
    calibration_id = (args.calib_npz.stem if args.calib_npz is not None else None)

    conditions = build_g_conditions(calib=calib)
    if args.conditions:
        wanted = set(args.conditions)
        conditions = [c for c in conditions if c["name"] in wanted]
        print(f"[expG] filtered conditions: {[c['name'] for c in conditions]}", flush=True)
    # Hard-fail-fast: F9 conditions need calib loaded.
    if calib is None and any(c["name"] in CONDITIONS_NEEDING_CALIB for c in conditions):
        missing = [c["name"] for c in conditions if c["name"] in CONDITIONS_NEEDING_CALIB]
        raise SystemExit(
            f"[expG] calibration missing but {missing} need it. "
            f"Run expF_calibrate.py first (frames=64 is fine; outlier indices are "
            f"frame-count-independent post-RoPE), or pass --conditions to exclude."
        )

    if not args.append and args.out.exists():
        args.out.unlink()

    from run_inference import load_model
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    print(f"[expG] model loaded; num_layers={num_layers} num_kv_heads={num_kv_heads}",
          flush=True)

    run_stage_g(model, processor, eval_items,
                num_layers=num_layers, num_kv_heads=num_kv_heads,
                conditions=conditions, stage=args.stage,
                calibration_id=calibration_id,
                out_jsonl=args.out, progress_every=args.progress_every,
                min_free_gb_for_256=args.min_free_gb_256)
    backfill_bf16_join_g(args.out)


if __name__ == "__main__":
    main()
