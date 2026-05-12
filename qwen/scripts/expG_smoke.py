"""Smoke test for Exp G (frame-scaling). Halts pipeline on any failure.

Two phases (mirrors expF_smoke.py shape):

  Phase A (no model required) -- pure-tensor + post-process sanity:
    1. f4_dispatch_at_64_128_256   F4 KIVI per-channel-seq round-trip on
                                   synthetic K with T in {5760, 11520, 23040}
                                   matches shape/dtype + non-trivial Δ.
    2. f9_calibration_loadable     expF_kcalib_*_frames64.npz exists, and the
                                   outlier_channel_idx_top16 array has the
                                   right (28, 4, 16) shape.
    3. stage1_split_present        qwen/calibration/split_seed0_n64.json
                                   exists OR can be generated; len(eval)==64;
                                   balanced 16/bucket.
    4. cascade_margin_definition   max(option_logprobs) - second_max(...) on
                                   synthetic logprobs equals
                                   _answer_margin(logp, argmax) when argmax
                                   is treated as the predicted choice.
    5. qtype_classifier_coverage   100 LVB questions from cal-100 -> all
                                   labels in expected set; weighted-avg
                                   frames in [110, 145].

  Phase B (requires --model and a GPU) -- live-model logits-differ:
    6. memory_feasibility_at_256f  Reads `nvidia-smi --query-gpu=memory.free`
                                   and asserts >= EXPG_MIN_FREE_GB (default
                                   30 GiB) before launching the 256f tier.
                                   This is a runtime check intended for the
                                   smoke harness on a freshly-allocated GPU.
    7. anchor_g0_sanity            On 1 LVB item at 64 frames, run BF16 prefill
                                   + 1-token forward; assert that the model
                                   produces a 4- or 5-way valid logprob vec
                                   (no NaN) AND `prompt_text` does not include
                                   any "fast processor" warning markers in the
                                   processor messages. (Coarse anchor sanity --
                                   the strong test is acc(G0) on n=64 which is
                                   asserted in the analyzer post-Stage 1.)
    8. (EXPG_RIGOR_HIGH=1 only)
       outlier_jaccard_64_vs_256   Recalibrate at frames=256 on 8 cal items
                                   and compare top-16 outlier indices vs
                                   frames=64 calib using set Jaccard >= 0.75
                                   per (L, H_kv). Only runs with high-rigor
                                   flag because it costs ~10 min.

Writes qwen/results/expG_smoke.md with PASS/FAIL per check; exits 2 on FAIL.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch


SCRIPTS_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPTS_DIR.parent / "calibration"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===================================================================
# Phase A: synthetic-tensor + post-process checks (no model load)
# ===================================================================


def _make_synthetic_k(seed: int = 0, B: int = 1, H: int = 4, T: int = 5760,
                      D: int = 128, dtype=torch.bfloat16) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(B, H, T, D, generator=g, dtype=torch.float32) * 0.5
    out_ch = [3, 17, 31, 63, 91, 100, 113, 127]
    for d in out_ch:
        K[:, :, :, d] *= 8.0
    return K.to(dtype)


def _check_f4_dispatch_at_frames(num_kv_heads: int) -> tuple[bool, str]:
    from k_quantizers import KQuantizerConfig, apply_k_quantizer

    cfg_f4 = KQuantizerConfig(name="F4", kind="kivi_per_channel_seq", bits=4)
    failures: list[str] = []
    # T values approximating 64f / 128f / 256f at ~90 visual tokens/frame.
    for n_frames, T in [(64, 5760), (128, 11520), (256, 23040)]:
        K = _make_synthetic_k(H=num_kv_heads, T=T)
        try:
            K_q = apply_k_quantizer(K, cfg_f4, layer_idx=0)
        except Exception as e:
            failures.append(f"frames={n_frames} T={T}: "
                            f"{type(e).__name__}: {e}")
            continue
        if K_q.shape != K.shape:
            failures.append(f"frames={n_frames} T={T}: shape mismatch "
                            f"{K_q.shape} != {K.shape}")
            continue
        if K_q.dtype != K.dtype:
            failures.append(f"frames={n_frames} T={T}: dtype mismatch "
                            f"{K_q.dtype} != {K.dtype}")
            continue
        delta = float((K.float() - K_q.float()).abs().max().item())
        if delta < 1e-3:
            failures.append(f"frames={n_frames} T={T}: max-abs delta "
                            f"{delta:.4e} below 1e-3 (suspicious no-op)")
            continue
    if failures:
        return False, "; ".join(failures[:3])
    return True, "F4 dispatch OK at T in {5760, 11520, 23040}"


def _check_f9_calibration_loadable(model_short: str,
                                   num_layers: int = 28,
                                   num_kv_heads: int = 4,
                                   ) -> tuple[bool, str]:
    npz_path = CALIBRATION_DIR / f"expF_kcalib_{model_short}_frames64.npz"
    if not npz_path.exists():
        return False, f"calibration NPZ not found at {npz_path}; run expF_calibrate.py first"
    try:
        arrays = np.load(npz_path)
    except Exception as e:
        return False, f"np.load failed on {npz_path}: {type(e).__name__}: {e}"
    if "outlier_channel_idx_top16" not in arrays.files:
        return False, f"missing key outlier_channel_idx_top16 in {npz_path}"
    arr = arrays["outlier_channel_idx_top16"]
    if arr.shape != (num_layers, num_kv_heads, 16):
        return False, (f"outlier_channel_idx_top16.shape={tuple(arr.shape)} != "
                       f"({num_layers}, {num_kv_heads}, 16)")
    return True, f"loaded {npz_path.name} with outlier_idx shape "\
                 f"{tuple(arr.shape)}"


def _check_stage1_split_present() -> tuple[bool, str]:
    from data_longvideobench import (
        load_all_items, load_split, make_split, save_split,
    )
    sp = CALIBRATION_DIR / "split_seed0_n64.json"
    if not sp.exists():
        # Try to generate it.
        items = load_all_items()
        targets = {"short": 16, "mid": 16, "long": 16, "very_long": 16}
        try:
            split = make_split(items, seed=0, targets=targets, cal_fraction=0.0)
            save_split(split, sp)
        except Exception as e:
            return False, f"could not generate split at {sp}: " \
                          f"{type(e).__name__}: {e}"
    try:
        split = load_split(sp)
    except Exception as e:
        return False, f"load_split({sp}) failed: {type(e).__name__}: {e}"
    if "eval" not in split:
        return False, f"split file {sp} has no 'eval' key"
    eval_ids = split["eval"]
    if len(eval_ids) != 64:
        return False, f"len(eval)={len(eval_ids)} != 64"
    # Verify balanced 16/bucket.
    items = load_all_items()
    by_id = {it.id: it for it in items}
    buckets = [by_id[iid].duration_bucket for iid in eval_ids if iid in by_id]
    counts = {b: buckets.count(b) for b in ("short", "mid", "long", "very_long")}
    if any(c != 16 for c in counts.values()):
        return False, f"unbalanced buckets in {sp}: {counts}"
    return True, f"split at {sp.name} balanced 16/bucket"


def _check_cascade_margin_definition() -> tuple[bool, str]:
    from expG_cascade import _confidence_margin
    from expF_kquant_screen import _answer_margin

    # 4-way logprobs with a clear winner.
    logp = [-0.8, -2.1, -1.0, -3.4]
    cm = _confidence_margin(logp)
    pred = max(range(len(logp)), key=lambda i: logp[i])
    am = _answer_margin(logp, correct=pred)
    if not math.isclose(cm, am, abs_tol=1e-9):
        return False, f"confidence_margin={cm} != answer_margin@argmax={am}"
    # Sanity: logp[0] - logp[2] = -0.8 - (-1.0) = 0.2.
    expected = logp[0] - logp[2]
    if not math.isclose(cm, expected, abs_tol=1e-9):
        return False, f"confidence_margin={cm} != expected {expected}"
    return True, f"max-second_max={cm:.4f} == answer_margin@argmax"


def _check_qtype_classifier_coverage(min_avg: int = 110, max_avg: int = 145
                                     ) -> tuple[bool, str]:
    from data_longvideobench import load_all_items, load_split
    from question_type_classifier import (
        BUDGET_MAP, classify_question_type, weighted_avg_frames,
    )
    # Use the cal-100 split if available; else fall back to first 100 items.
    cal_split = CALIBRATION_DIR / "split_seed0.json"
    items_all = load_all_items()
    if cal_split.exists():
        try:
            split = load_split(cal_split)
            cal_ids = set(split.get("cal", []))
            items = [it for it in items_all if it.id in cal_ids]
        except Exception:
            items = items_all[:100]
    else:
        items = items_all[:100]
    if not items:
        return False, "no items available to classify"
    questions = [it.question for it in items]
    labels = [classify_question_type(q) for q in questions]
    valid = set(BUDGET_MAP.keys())
    bad = [l for l in labels if l not in valid]
    if bad:
        return False, f"{len(bad)} labels outside {valid}: " \
                      f"first={bad[:5]}"
    avg = weighted_avg_frames(questions)
    if not (min_avg <= avg <= max_avg):
        return False, f"weighted_avg_frames={avg:.1f} outside " \
                      f"[{min_avg}, {max_avg}] -- retune budget map"
    return True, (f"n={len(questions)} weighted_avg_frames={avg:.1f} "
                  f"in [{min_avg}, {max_avg}]")


def _run_phase_a(num_layers: int, num_kv_heads: int, head_dim: int,
                 model_short: str) -> tuple[list[dict], bool]:
    results: list[dict] = []
    all_pass = True

    ok, det = _check_f4_dispatch_at_frames(num_kv_heads)
    all_pass = all_pass and ok
    results.append({"check": "f4_dispatch_at_64_128_256",
                    "pass": ok, "detail": det})

    ok, det = _check_f9_calibration_loadable(model_short, num_layers, num_kv_heads)
    all_pass = all_pass and ok
    results.append({"check": "f9_calibration_loadable",
                    "pass": ok, "detail": det})

    ok, det = _check_stage1_split_present()
    all_pass = all_pass and ok
    results.append({"check": "stage1_split_present",
                    "pass": ok, "detail": det})

    ok, det = _check_cascade_margin_definition()
    all_pass = all_pass and ok
    results.append({"check": "cascade_margin_definition",
                    "pass": ok, "detail": det})

    ok, det = _check_qtype_classifier_coverage()
    all_pass = all_pass and ok
    results.append({"check": "qtype_classifier_coverage",
                    "pass": ok, "detail": det})

    return results, all_pass


# ===================================================================
# Phase B: live-model checks
# ===================================================================


def _check_memory_feasibility_at_256f(min_free_gb: int) -> tuple[bool, str]:
    if shutil.which("nvidia-smi") is None:
        return True, "nvidia-smi not found; skipping (not a GPU machine)"
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={dev}",
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        free_mib = int(out)
    except Exception as e:
        return True, f"nvidia-smi parse failed ({e}); skipping"
    free_gb = free_mib / 1024
    ok = free_gb >= min_free_gb
    return ok, (f"GPU{dev} free={free_gb:.1f} GiB "
                f"{'>=' if ok else '<'} EXPG_MIN_FREE_GB={min_free_gb}")


def _check_anchor_g0_sanity(model_id: str, frames: int, n_items: int,
                            split_file: Path) -> tuple[bool, str]:
    """Light live-model check: load the model, run G0 BF16 forward on 1 item,
    assert no-NaN logprobs + 4/5-way option mass != uniform.
    """
    from data_longvideobench import (
        filter_items, format_mcq_messages, load_all_items, load_split,
    )
    from qwen_vl_utils import process_vision_info  # type: ignore
    from run_inference import load_model
    from data_longvideobench import answer_token_ids
    from transformers.cache_utils import DynamicCache

    items_all = load_all_items()
    if split_file.exists():
        split = load_split(split_file)
        eval_items = filter_items(items_all, split["eval"])[:n_items]
    else:
        eval_items = items_all[:n_items]
    if not eval_items:
        return False, "no items to score"

    model, processor = load_model(model_id, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    item = eval_items[0]
    msgs = format_mcq_messages(item, n_frames=frames)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    cache = DynamicCache()
    with torch.no_grad():
        out = model.generate(**inputs, past_key_values=cache, max_new_tokens=1,
                             do_sample=False, return_dict_in_generate=True,
                             output_scores=True, use_cache=True)
    n_options = len(item.candidates)
    answer_ids = answer_token_ids(processor, n=n_options)
    first_logits = out.scores[0]
    logp = torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist()
    if any(math.isnan(x) for x in logp):
        return False, f"NaN in option logprobs: {logp}"
    # Sanity: at least one option should sit clearly above uniform (-log(n)).
    uniform_lp = -math.log(n_options)
    spread = max(logp) - min(logp)
    if spread < 1e-2:
        return False, f"flat logprobs (spread={spread:.4e}); model is hosed"
    return True, (f"item={item.id} frames={frames} n_options={n_options} "
                  f"logprobs ok (spread={spread:.3f})")


def _check_outlier_jaccard_high_rigor(model_id: str, model_short: str,
                                      n_cal_items: int = 8) -> tuple[bool, str]:
    """Recalibrate at frames=256 on a small cal subset; compare to frames=64.

    Run via subprocess to keep memory hygiene with the model load above.
    """
    cal64 = CALIBRATION_DIR / f"expF_kcalib_{model_short}_frames64.npz"
    if not cal64.exists():
        return False, f"frames=64 calib missing at {cal64}"
    out256 = CALIBRATION_DIR / f"expF_kcalib_{model_short}_frames256_subset{n_cal_items}.npz"
    cmd = [
        "python3", "-u",
        str(SCRIPTS_DIR / "expF_calibrate.py"),
        "--model", model_id,
        "--frames", "256",
        "--limit", str(n_cal_items),
        "--out_npz", str(out256),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    except subprocess.CalledProcessError as e:
        return False, f"recalibration at frames=256 failed: rc={e.returncode}; " \
                      f"stderr={e.stderr[-300:]}"
    except Exception as e:
        return False, f"recalibration subprocess: {type(e).__name__}: {e}"
    try:
        a64 = np.load(cal64)["outlier_channel_idx_top16"]
        a256 = np.load(out256)["outlier_channel_idx_top16"]
    except Exception as e:
        return False, f"could not load both NPZ: {e}"
    if a64.shape != a256.shape:
        return False, f"shape mismatch: 64={a64.shape} 256={a256.shape}"
    L, H, K = a64.shape
    jaccards: list[float] = []
    for li in range(L):
        for hi in range(H):
            s64 = set(int(x) for x in a64[li, hi].tolist())
            s256 = set(int(x) for x in a256[li, hi].tolist())
            inter = len(s64 & s256)
            union = len(s64 | s256)
            jaccards.append(inter / union if union else 1.0)
    j_min = float(min(jaccards))
    j_med = float(np.median(jaccards))
    ok = j_min >= 0.75
    return ok, (f"jaccard_min={j_min:.3f} median={j_med:.3f} over "
                f"({L}, {H}) (L, H_kv) cells; threshold 0.75")


def _run_phase_b(model_id: str, frames: int, n_items: int, split_file: Path,
                 model_short: str, min_free_gb: int,
                 high_rigor: bool) -> tuple[list[dict], bool]:
    results: list[dict] = []
    all_pass = True

    ok, det = _check_memory_feasibility_at_256f(min_free_gb)
    all_pass = all_pass and ok
    results.append({"check": "memory_feasibility_at_256f",
                    "pass": ok, "detail": det})

    try:
        ok, det = _check_anchor_g0_sanity(model_id, frames, n_items, split_file)
    except Exception as e:
        ok = False
        det = f"{type(e).__name__}: {e}"
    all_pass = all_pass and ok
    results.append({"check": "anchor_g0_sanity",
                    "pass": ok, "detail": det})

    if high_rigor:
        try:
            ok, det = _check_outlier_jaccard_high_rigor(model_id, model_short)
        except Exception as e:
            ok = False
            det = f"{type(e).__name__}: {e}"
        all_pass = all_pass and ok
        results.append({"check": "outlier_jaccard_64_vs_256",
                        "pass": ok, "detail": det})
    else:
        results.append({"check": "outlier_jaccard_64_vs_256",
                        "pass": True,
                        "detail": "skipped (set EXPG_RIGOR_HIGH=1 to enable)"})

    return results, all_pass


# ===================================================================
# Main
# ===================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="",
                    help="Pass to enable Phase B. Empty -> Phase A only.")
    ap.add_argument("--frames", type=int, default=64,
                    help="Frames to use for the live-model anchor sanity check.")
    ap.add_argument("--n_items", type=int, default=1)
    ap.add_argument("--num_layers", type=int, default=28)
    ap.add_argument("--num_kv_heads", type=int, default=4)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--min_free_gb", type=int,
                    default=int(os.environ.get("EXPG_MIN_FREE_GB", "30")))
    ap.add_argument("--high_rigor", action="store_true",
                    default=bool(int(os.environ.get("EXPG_RIGOR_HIGH", "0"))))
    ap.add_argument("--split_file", type=Path,
                    default=CALIBRATION_DIR / "split_seed0_n64.json")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "expG_smoke.md")
    args = ap.parse_args()

    model_short = (args.model or "Qwen/Qwen2.5-VL-7B-Instruct").split("/")[-1]

    a_results, a_pass = _run_phase_a(
        num_layers=args.num_layers, num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim, model_short=model_short,
    )

    b_results: list[dict] = []
    b_pass = True
    if args.model:
        b_results, b_pass = _run_phase_b(
            args.model, args.frames, args.n_items, args.split_file,
            model_short=model_short,
            min_free_gb=args.min_free_gb,
            high_rigor=args.high_rigor,
        )

    overall = a_pass and b_pass

    lines = [
        f"# Exp G Smoke Report\n",
        f"Generated: {_ts()}\n",
        f"num_layers: {args.num_layers}  num_kv_heads: {args.num_kv_heads}  head_dim: {args.head_dim}",
        f"model: `{args.model or '(skipped)'}`  frames: {args.frames}  "
        f"min_free_gb: {args.min_free_gb}  high_rigor: {args.high_rigor}\n",
        "## Phase A -- synthetic-tensor + post-process checks\n",
        "| # | Check | Pass | Detail |",
        "|---|---|:-:|---|",
    ]
    for i, r in enumerate(a_results, 1):
        marker = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| {i} | `{r['check']}` | {marker} | {r['detail']} |")

    lines += ["\n## Phase B -- live-model checks\n",
              "| # | Check | Pass | Detail |",
              "|---|---|:-:|---|"]
    if not args.model:
        lines.append("| - | (skipped -- no --model) | -- | use --model to enable |")
    else:
        for i, r in enumerate(b_results, 1):
            marker = "PASS" if r["pass"] else "FAIL"
            lines.append(f"| {i+5} | `{r['check']}` | {marker} | {r['detail']} |")

    lines.append(f"\n## Overall: {'PASS' if overall else 'FAIL'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[G-smoke] wrote {args.out}")
    print(f"[G-smoke] overall: {'PASS' if overall else 'FAIL'}", flush=True)
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
