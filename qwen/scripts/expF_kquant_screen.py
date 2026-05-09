"""Experiment F — Tiered K-quantizer screening driver.

Stages:
  0:  n=16 (4/bucket)   smoke / wiring; do NOT interpret accuracy
  1:  n=64 (16/bucket)  screen 14 K-quantizer variants
  2:  n=100 (25/bucket) confirm borderline survivors
  3:  n=200 (50/bucket) final paired analysis

The same `seed=0` stratified split rules ensure Stage-0 ⊂ Stage-1 ⊂ Stage-2
⊂ Stage-3 — paired analysis works across stages.

Reads calibration data (NPZ + JSON) at startup, instantiates 14
KQuantizerConfig objects, then for each (item, condition) runs a
prefill+1-token forward and writes one JSONL row.

JSONL row schema:
  phase: "F1" | "F0" | "F2" | "F3"
  stage: 0 | 1 | 2 | 3
  item_id, duration_bucket, duration_seconds, n_options, correct_choice, n_frames
  condition, k_quant_variant, calibration_id, n_outlier_channels,
  percentile_clip, score_cal_weights, k_hi, k_lo, v_bits, avg_kv_bits
  pred_choice, is_correct, option_logprobs, answer_margin, latency_ms
  bf16_pred, bf16_correct  (backfilled after run)
  seq_len, visual_token_start, visual_token_end

Usage:
  python expF_kquant_screen.py --stage 1 --calib_file <npz>
  python expF_kquant_screen.py --stage 0  # 16-item smoke (no calib needed for F0/F1/F4/F7)
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
    DEFAULT_SPLIT_FILE,
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


SCRIPTS_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPTS_DIR.parent / "calibration"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"


# ===================================================================
# Subset config
# ===================================================================


# Stratified targets per stage. seed=0 ensures supersets across stages.
STAGE_TARGETS = {
    0: {"short": 4, "mid": 4, "long": 4, "very_long": 4},          # 16
    1: {"short": 16, "mid": 16, "long": 16, "very_long": 16},      # 64
    2: {"short": 25, "mid": 25, "long": 25, "very_long": 25},      # 100
    3: {"short": 50, "mid": 50, "long": 50, "very_long": 50},      # 200
}


def stage_split_path(stage: int) -> Path:
    n = sum(STAGE_TARGETS[stage].values())
    return CALIBRATION_DIR / f"split_seed0_n{n}.json"


def ensure_stage_split(stage: int) -> Path:
    """Generate or reuse the stratified split file for `stage`. Returns path."""
    sp = stage_split_path(stage)
    if sp.exists():
        return sp
    items = load_all_items()
    targets = STAGE_TARGETS[stage]
    # cal_fraction=0.0 puts everything in eval (we don't need a cal subset here).
    split = make_split(items, seed=0, targets=targets, cal_fraction=0.0)
    save_split(split, sp)
    print(f"[expF] wrote stage {stage} split (n_eval={len(split['eval'])}) -> {sp}",
          flush=True)
    return sp


# ===================================================================
# Calibration loader
# ===================================================================


def load_calib(calib_npz: Optional[Path], calib_json: Optional[Path]) -> tuple[Optional[dict], Optional[dict]]:
    """Return (calib, meta) tuple. calib is the dict consumed by KQuantizerConfig.calib."""
    if calib_npz is None or not calib_npz.exists():
        return None, None
    arrays = np.load(calib_npz)
    calib = {k: arrays[k] for k in arrays.files}
    meta = None
    if calib_json is not None and calib_json.exists():
        meta = json.loads(calib_json.read_text())
    return calib, meta


# ===================================================================
# Per-condition factory
# ===================================================================


def build_stage_conditions(calib: Optional[dict]) -> list[dict]:
    """Return the F-suite condition specs as dicts: {name, mode, ...}.

    F2 and F3 are NOT pure k_quantizer kinds — they need V3K mode + a per-token
    K mask. We special-case them here so the runner can route them to the
    right cache wiring.

    Other conditions use V1 mode + KQuantizerConfig.
    """
    fc = build_f_conditions(calib=calib)  # 14 KQuantizerConfig objects
    by_name = {cfg.name: cfg for cfg in fc}
    out: list[dict] = []

    out.append({"name": "F0_BF16", "mode": "bf16",
                "cfg": by_name["F0_BF16"]})
    out.append({"name": "F1_UniformInt4", "mode": "v1_kcfg",
                "cfg": by_name["F1_UniformInt4"]})
    out.append({"name": "F2_TextBF16_VisInt4", "mode": "v3k_text_bf16"})
    out.append({"name": "F3_AllKBF16_VInt4", "mode": "v3k_all_bf16"})
    out.append({"name": "F4_KIVI_PerChannelSeq", "mode": "v1_kcfg",
                "cfg": by_name["F4_KIVI_PerChannelSeq"]})
    out.append({"name": "F5_KIVI_TextVisualSplit", "mode": "v1_kcfg_with_slice",
                "cfg": by_name["F5_KIVI_TextVisualSplit"]})
    out.append({"name": "F6_KIVI_RoleSplit", "mode": "v1_kcfg_with_slice",
                "cfg": by_name["F6_KIVI_RoleSplit"]})
    out.append({"name": "F7_KIVI_P99_5", "mode": "v1_kcfg",
                "cfg": by_name["F7_KIVI_P99_5"]})
    out.append({"name": "F8_KIVI_Outlier8", "mode": "v1_kcfg",
                "cfg": by_name["F8_KIVI_Outlier8"]})
    out.append({"name": "F9_KIVI_Outlier16", "mode": "v1_kcfg",
                "cfg": by_name["F9_KIVI_Outlier16"]})
    out.append({"name": "F10_ScoreCal_Generic", "mode": "v1_kcfg",
                "cfg": by_name["F10_ScoreCal_Generic"]})
    out.append({"name": "F11_ScoreCal_Block_TTHeavy", "mode": "v1_kcfg_with_slice",
                "cfg": by_name["F11_ScoreCal_Block_TTHeavy"]})
    out.append({"name": "F12_ScoreCal_Block_Balanced", "mode": "v1_kcfg_with_slice",
                "cfg": by_name["F12_ScoreCal_Block_Balanced"]})
    out.append({"name": "F13_ScoreCal_TextOnly", "mode": "v1_kcfg_with_slice",
                "cfg": by_name["F13_ScoreCal_TextOnly"]})

    return out


# ===================================================================
# Per-item runner
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


@torch.no_grad()
def _run_condition_forward(model, processor, item: LVBItem, n_frames: int,
                           num_layers: int, num_kv_heads: int,
                           cond: dict, slice_info: dict, inputs):
    """Run one (item, condition) forward; return (out, latency_ms, avg_kv_bits)."""
    seq_len = int(inputs["input_ids"].shape[1])
    mode = cond["mode"]

    if mode == "bf16":
        # F0: vanilla DynamicCache, no quant.
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        avg_bits = 16.0
    elif mode == "v1_kcfg":
        cfg = cond["cfg"]
        ctrl = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                             mode="V1", default_k_bits=cfg.bits, default_v_bits=4)
        cache = FakeQuantKVCache(ctrl, k_quantizer_config=cfg)
        avg_bits = (float(cfg.bits) + 4.0) / 2.0
    elif mode == "v1_kcfg_with_slice":
        cfg = cond["cfg"]
        ctrl = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                             mode="V1", default_k_bits=cfg.bits, default_v_bits=4)
        cache = FakeQuantKVCache(ctrl, k_quantizer_config=cfg)
        cache.set_slice_info(slice_info)
        avg_bits = (float(cfg.bits) + 4.0) / 2.0
    elif mode == "v3k_text_bf16":
        # F2: text-K BF16 (mask=True at text positions), visual-K INT4.
        v_start = int(slice_info["v_start"])
        v_end = int(slice_info["v_end"])
        mask = torch.zeros(seq_len, dtype=torch.bool)
        if v_start > 0:
            mask[:v_start] = True
        if v_end < seq_len:
            mask[v_end:] = True
        ctrl = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                             mode="V3K", default_k_bits=4, default_v_bits=4)
        ctrl.set_global(k_bits=4, v_bits=4)
        for L in range(num_layers):
            ctrl.set_protected_mask(L, mask, hi_bits=16, lo_bits=4)
        cache = FakeQuantKVCache(ctrl)
        n_text = int(mask.sum().item())
        n_total = seq_len
        bits_K = (16.0 * n_text + 4.0 * (n_total - n_text)) / max(1, n_total)
        avg_bits = (bits_K + 4.0) / 2.0
    elif mode == "v3k_all_bf16":
        # F3: all-K BF16, V INT4.
        ctrl = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                             mode="V1", default_k_bits=16, default_v_bits=4)
        cache = FakeQuantKVCache(ctrl)  # K=16 from ctrl, V=4 from ctrl; no k_cfg
        avg_bits = (16.0 + 4.0) / 2.0
    else:
        raise ValueError(f"unknown condition mode: {mode}")

    t0 = time.perf_counter()
    out = model.generate(
        **inputs, past_key_values=cache, max_new_tokens=1, do_sample=False,
        return_dict_in_generate=True, output_scores=True, use_cache=True,
    )
    return out, (time.perf_counter() - t0) * 1000.0, avg_bits


@torch.no_grad()
def run_item(model, processor, item: LVBItem, n_frames: int,
             num_layers: int, num_kv_heads: int, conditions: list[dict],
             stage: int, calibration_id: Optional[str]) -> list[dict]:
    """Run all conditions for one item; return list of JSONL rows."""
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
    for cond in conditions:
        try:
            out, lat_ms, avg_bits = _run_condition_forward(
                model, processor, item, n_frames=n_frames,
                num_layers=num_layers, num_kv_heads=num_kv_heads,
                cond=cond, slice_info=slice_info, inputs=inputs,
            )
            logp, pred = _option_logprobs_and_pred(out, processor, n_options)
            margin = _answer_margin(logp, correct)
            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            rows.append({
                "phase": f"F{stage}",
                "stage": stage,
                "item_id": item.id,
                "duration_bucket": item.duration_bucket,
                "duration_seconds": item.duration_seconds,
                "n_options": n_options,
                "correct_choice": correct,
                "n_frames": n_frames,
                "condition": cond["name"],
                "error": f"{type(e).__name__}: {e}",
                "is_correct": False,
            })
            continue

        cfg_obj = cond.get("cfg")
        row = {
            "phase": f"F{stage}",
            "stage": stage,
            "item_id": item.id,
            "duration_bucket": item.duration_bucket,
            "duration_seconds": item.duration_seconds,
            "n_options": n_options,
            "correct_choice": correct,
            "n_frames": n_frames,
            "condition": cond["name"],
            "k_quant_variant": cfg_obj.kind if cfg_obj is not None else cond["mode"],
            "calibration_id": calibration_id,
            "n_outlier_channels": (cfg_obj.n_outliers if cfg_obj is not None else None),
            "percentile_clip": (cfg_obj.percentile if cfg_obj is not None else None),
            "score_cal_weights": (cfg_obj.score_cal_weights if cfg_obj is not None else None),
            "k_hi": (16 if cond["mode"] in ("v3k_text_bf16", "v3k_all_bf16", "bf16") else
                     (16 if cfg_obj is not None and cfg_obj.kind == "bf16" else int(cfg_obj.bits))),
            "k_lo": (int(cfg_obj.bits) if cfg_obj is not None else 4),
            "v_bits": 4,
            "avg_kv_bits": float(avg_bits),
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
# Driver
# ===================================================================


def _append_progress(progress_log: Path, line: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_log, "a") as f:
        f.write(f"[{ts}] {line}\n")
        f.flush()


def run_stage(model, processor, items: list[LVBItem], n_frames: int,
              num_layers: int, num_kv_heads: int, conditions: list[dict],
              stage: int, calibration_id: Optional[str],
              out_jsonl: Path, progress_every: int = 5) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_log = out_jsonl.with_name(out_jsonl.stem + ".progress.log")
    _append_progress(progress_log, f"START stage={stage} n_items={len(items)} "
                                   f"n_conditions={len(conditions)} frames={n_frames}")
    t0 = time.perf_counter()
    n_done, n_failed, n_rows = 0, 0, 0
    with open(out_jsonl, "a") as f:
        for i, it in enumerate(items):
            try:
                rows = run_item(model, processor, it, n_frames=n_frames,
                                num_layers=num_layers, num_kv_heads=num_kv_heads,
                                conditions=conditions, stage=stage,
                                calibration_id=calibration_id)
            except Exception as e:
                n_failed += 1
                _append_progress(progress_log,
                                 f"WARN item={it.id} skipped: {type(e).__name__}: {e}")
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
                _append_progress(
                    progress_log,
                    f"stage{stage} {done}/{len(items)} ok={n_done} rows={n_rows} "
                    f"failed={n_failed} elapsed={timedelta(seconds=int(elapsed))} "
                    f"ETA={timedelta(seconds=int(eta))}"
                )
    _append_progress(progress_log, f"DONE stage={stage} ok={n_done} rows={n_rows} "
                                   f"failed={n_failed}")


def backfill_bf16_join(out_jsonl: Path) -> None:
    """Walk the JSONL once; for each non-F0 row, fill bf16_pred/bf16_correct
    from the F0_BF16 row of the same item_id."""
    if not out_jsonl.exists():
        return
    rows = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l.strip()]
    # Find F0 prediction per item_id.
    bf16_pred: dict[str, int] = {}
    for r in rows:
        if r.get("condition") == "F0_BF16" and "pred_choice" in r:
            bf16_pred[r["item_id"]] = int(r["pred_choice"])
    # Backfill.
    for r in rows:
        if r.get("condition") == "F0_BF16":
            r["bf16_pred"] = int(r.get("pred_choice", -1))
            r["bf16_correct"] = bool(r["bf16_pred"] == int(r["correct_choice"]))
            continue
        bp = bf16_pred.get(r["item_id"])
        if bp is None:
            continue
        r["bf16_pred"] = int(bp)
        r["bf16_correct"] = bool(bp == int(r["correct_choice"]))
    # Rewrite atomically.
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out_jsonl)
    print(f"[expF] backfilled BF16 join in {out_jsonl} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--stage", type=int, choices=[0, 1, 2, 3], default=1)
    ap.add_argument("--split_file", type=Path, default=None,
                    help="Override the auto-generated stage split file.")
    ap.add_argument("--calib_npz", type=Path, default=None)
    ap.add_argument("--calib_json", type=Path, default=None)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="Subset of condition names to run. Default: all 14.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Override: limit eval items count after split selection.")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing JSONL rather than truncating.")
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: results/expF_kquant_stage{stage}.jsonl")
    args = ap.parse_args()

    # Resolve paths
    if args.out is None:
        args.out = RESULTS_DIR / f"expF_kquant_stage{args.stage}.jsonl"
    if args.split_file is None:
        args.split_file = ensure_stage_split(args.stage)
    if args.calib_npz is None:
        model_short = args.model.split("/")[-1]
        cand = CALIBRATION_DIR / f"expF_kcalib_{model_short}_frames{args.frames}.npz"
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
    print(f"[expF] stage={args.stage} eval_items={len(eval_items)} "
          f"split_file={args.split_file} calib_npz={args.calib_npz}", flush=True)

    calib, calib_meta = load_calib(args.calib_npz, args.calib_json)
    calibration_id = (args.calib_npz.stem if args.calib_npz is not None else None)

    # Build conditions; calib-dependent ones get the loaded calib injected.
    conditions = build_stage_conditions(calib=calib)
    if args.conditions:
        wanted = set(args.conditions)
        conditions = [c for c in conditions if c["name"] in wanted]
        print(f"[expF] filtered conditions: {[c['name'] for c in conditions]}", flush=True)
    # Hard-fail-fast: any calib-dependent condition with no calib loaded?
    needs_calib = {"F8_KIVI_Outlier8", "F9_KIVI_Outlier16", "F10_ScoreCal_Generic",
                   "F11_ScoreCal_Block_TTHeavy", "F12_ScoreCal_Block_Balanced",
                   "F13_ScoreCal_TextOnly"}
    if calib is None and any(c["name"] in needs_calib for c in conditions):
        missing = [c["name"] for c in conditions if c["name"] in needs_calib]
        raise SystemExit(
            f"[expF] calibration missing but {missing} need it. "
            f"Run expF_calibrate.py first or pass --conditions to exclude these."
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
    print(f"[expF] model loaded; num_layers={num_layers} num_kv_heads={num_kv_heads}",
          flush=True)

    run_stage(model, processor, eval_items, n_frames=args.frames,
              num_layers=num_layers, num_kv_heads=num_kv_heads,
              conditions=conditions, stage=args.stage,
              calibration_id=calibration_id,
              out_jsonl=args.out, progress_every=args.progress_every)
    backfill_bf16_join(args.out)


if __name__ == "__main__":
    main()
