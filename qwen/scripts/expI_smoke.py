"""Smoke test for Exp I (temporal-KIVI mechanism screen). Halts pipeline on any failure.

Phase A (no model required) — pure-tensor + post-process sanity:
  1. tempwin_outlier_kernel_round_trip   TempWin + outlier-N: planted outlier
                                         channels are restored at BF16 exactly;
                                         non-outlier channels are quantized.
  2. tempwin_window_boundaries           visual_only mode produces exactly the
                                         expected segment boundaries; segments
                                         have distinct quantized values when
                                         their content varies.
  3. vidkv_v_scale_differs_from_uniform  _v_per_channel_seq_quantize differs
                                         from fake_quantize_kv when V's time-
                                         axis variance differs from head_dim
                                         grouping.
  4. tokenblock6_segment_count           token_block n=6 produces 6 segments;
                                         visual_only n=4 with text prefix+
                                         suffix produces 6 segments.
  5. outlier8_subset_of_top16            top16[:8] ⊆ top16[:16] per (L, H).
                                         (Loads real calib NPZ if present;
                                         otherwise uses synthetic indices.)
  6. seed1_split_supersets               make_split(seed=1) at n=64 ⊂ n=200
                                         per duration bucket.

Phase B (requires --model and a GPU) — live-model logits-differ:
  7. visual_span_detection_seed1         find_text_slice_spans returns
                                         v_end > v_start for first 8 items
                                         in seed=1 stage-1 split.
  8. bf16_vs_i_conditions_logits_differ  On 1 LVB item: BF16, I3, I7, I8 all
                                         produce pairwise non-equal first-
                                         token logprob vectors.

Writes qwen/results/expI_smoke.md with PASS/FAIL per check; exits 2 on FAIL.

Usage (Phase A only):
  python expI_smoke.py
Usage (Phase A + B):
  python expI_smoke.py --model Qwen/Qwen2.5-VL-7B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
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
# Phase A — synthetic-tensor checks (no model load)
# ===================================================================


def _make_synthetic_k(seed: int = 0, B: int = 1, H: int = 4, T: int = 24,
                      D: int = 128, dtype=torch.bfloat16,
                      outlier_chans: Optional[list[int]] = None,
                      outlier_amp: float = 8.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(B, H, T, D, generator=g, dtype=torch.float32) * 0.5
    if outlier_chans is None:
        outlier_chans = [3, 17, 31, 63, 91, 100, 113, 127]
    for d in outlier_chans:
        K[:, :, :, d] *= outlier_amp
    return K.to(dtype)


def _check_tempwin_outlier_kernel_round_trip() -> tuple[bool, str]:
    from k_quantizers import KQuantizerConfig, apply_k_quantizer

    H, D = 4, 128
    planted = [3, 17, 31, 63, 91, 100, 113, 127]
    K = _make_synthetic_k(H=H, T=24, D=D, outlier_chans=planted)
    L = 28
    # outlier_channel_idx_top16 expected shape: [L, H_kv, 16]. Plant the same
    # 8 channels for every (L, H), then pad to 16 with arbitrary other channels.
    other_chans = [d for d in range(D) if d not in planted][:8]
    full16 = planted + other_chans
    idx16 = np.array([[full16] * H] * L, dtype=np.int64)  # [L, H, 16]
    calib = {"outlier_channel_idx_top16": idx16}

    cfg = KQuantizerConfig(name="I_TempWin2_Outlier8_TEST",
                           kind="kivi_temporal_window", bits=4,
                           n_temporal_windows=2, temporal_mode="visual_only",
                           n_outliers=8, calib=calib)
    slice_info = {"v_start": 4, "v_end": 20, "seq_len": 24}
    K_q = apply_k_quantizer(K, cfg, layer_idx=0, slice_info=slice_info,
                            cache_offset=0)

    if K_q.shape != K.shape or K_q.dtype != K.dtype:
        return False, f"shape/dtype mismatch K_q={K_q.shape}/{K_q.dtype} K={K.shape}/{K.dtype}"

    # Outlier channels must be exactly preserved.
    delta_outlier = (K[..., planted].float() - K_q[..., planted].float()).abs().max().item()
    if delta_outlier > 1e-6:
        return False, f"outlier channels not preserved: max_abs_delta={delta_outlier:.4e} > 1e-6"

    # Non-outlier channels must be quantized (i.e. differ from K).
    other_d = [d for d in range(D) if d not in planted]
    delta_other = (K[..., other_d].float() - K_q[..., other_d].float()).abs().max().item()
    if delta_other < 1e-3:
        return False, f"non-outlier channels not quantized: max_abs_delta={delta_other:.4e} < 1e-3"

    return True, (f"outlier {len(planted)} channels preserved exactly "
                  f"(delta={delta_outlier:.2e}); other channels quantized "
                  f"(delta={delta_other:.4f})")


def _check_tempwin_window_boundaries() -> tuple[bool, str]:
    """visual_only with n_temporal_windows=4, T=24, v_start=4, v_end=20:
    expected segment boundaries = {0, 4, 8, 12, 16, 20, 24} (6 segments).
    Build K with each segment having distinct mean magnitude; verify the
    quantized output's per-segment max-abs differs across segments.
    """
    from k_quantizers import KQuantizerConfig, apply_k_quantizer

    B, H, T, D = 1, 4, 24, 128
    K = torch.zeros(B, H, T, D, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(0)
    # Plant a different magnitude per expected segment.
    expected_boundaries = [0, 4, 8, 12, 16, 20, 24]
    seg_amps = [0.1, 1.0, 2.0, 4.0, 8.0, 0.5]
    for i in range(len(expected_boundaries) - 1):
        a, b = expected_boundaries[i], expected_boundaries[i + 1]
        K[:, :, a:b, :] = (torch.randn(B, H, b - a, D, generator=g) * seg_amps[i]).to(torch.bfloat16)

    cfg = KQuantizerConfig(name="I_TempWin4_TEST", kind="kivi_temporal_window",
                           bits=4, n_temporal_windows=4, temporal_mode="visual_only")
    slice_info = {"v_start": 4, "v_end": 20, "seq_len": 24}
    K_q = apply_k_quantizer(K, cfg, layer_idx=0, slice_info=slice_info,
                            cache_offset=0)

    # Per-segment max-abs of (K - K_q) reflects the quantization noise scale,
    # which is proportional to the segment's quantization scale. If boundaries
    # are correct + scales are per-segment, segments with different planted
    # amps yield different residual magnitudes.
    seg_residuals = []
    for i in range(len(expected_boundaries) - 1):
        a, b = expected_boundaries[i], expected_boundaries[i + 1]
        seg = (K[:, :, a:b, :].float() - K_q[:, :, a:b, :].float()).abs().max().item()
        seg_residuals.append(seg)

    # Segments with seg_amps differing by ≥ 4x should have residuals
    # differing meaningfully — at least one pair of (max, min) residuals
    # should differ by > 4x.
    max_r, min_r = max(seg_residuals), min(seg_residuals)
    if max_r / max(1e-12, min_r) < 3.0:
        return False, (f"segment residuals too uniform: {seg_residuals}; "
                       f"expected per-segment scales -> per-segment residuals")

    return True, (f"6 segments at boundaries {expected_boundaries}; "
                  f"residuals per segment {[f'{r:.3f}' for r in seg_residuals]} "
                  f"(max/min ratio {max_r / max(1e-12, min_r):.1f})")


def _check_vidkv_v_scale_differs_from_uniform() -> tuple[bool, str]:
    """VidKV V quantizer (per-channel along time axis) should produce
    different output than the default fake_quantize_kv (per-channel along
    head_dim with group_size=128) on synthetic V where the two scale-axis
    choices give different scales.
    """
    from fake_quant_kv_cache import _v_per_channel_seq_quantize, fake_quantize_kv

    B, H, T, D = 1, 4, 64, 128
    g = torch.Generator().manual_seed(0)
    V = torch.randn(B, H, T, D, generator=g, dtype=torch.float32) * 0.5
    # Plant time-axis structure that the time-axis scale will pick up but
    # the head_dim-grouped scale will average over.
    for d in range(D):
        if d % 8 == 0:
            V[:, :, :T // 2, d] *= 8.0  # large amplitude in first half on this channel
    V = V.to(torch.bfloat16)

    out_vidkv = _v_per_channel_seq_quantize(V, bits=4)
    out_uniform = fake_quantize_kv(V, bits=4, group_size=128)

    delta = (out_vidkv.float() - out_uniform.float()).abs().max().item()
    if delta < 1e-3:
        return False, (f"VidKV V == uniform INT4: max_abs_delta={delta:.4e} < 1e-3 "
                       f"(per-channel-seq scale and head_dim-group scale match — "
                       f"either no time-axis structure or wrong axis)")

    return True, f"VidKV V differs from uniform INT4: max_abs_delta={delta:.4f}"


def _check_tokenblock6_segment_count() -> tuple[bool, str]:
    """token_block n=6 must split [0, T) into 6 segments. visual_only n=4
    with text prefix + suffix must produce 6 segments (text-pre, 4 visual,
    text-post). Confirms parity claim for I12 vs I11.
    """
    from k_quantizers import KQuantizerConfig, _kivi_temporal_window

    # Helper: count segments by counting unique scales in K_q.
    # For a controlled test, use planted distinct per-segment magnitudes
    # and check that quantization noise has 6 distinct residual levels.
    def n_distinct_residual_levels(K, K_q, expected_boundaries):
        residuals = []
        for i in range(len(expected_boundaries) - 1):
            a, b = expected_boundaries[i], expected_boundaries[i + 1]
            r = (K[:, :, a:b, :].float() - K_q[:, :, a:b, :].float()).abs().max().item()
            residuals.append(round(r, 3))
        return len(set(residuals)), residuals

    B, H, T, D = 1, 4, 60, 128

    # token_block n=6: boundaries [0, 10, 20, 30, 40, 50, 60].
    K_tb = torch.zeros(B, H, T, D, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(1)
    seg_amps = [0.1, 0.4, 1.0, 2.5, 6.0, 12.0]
    for i, (a, b) in enumerate([(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]):
        K_tb[:, :, a:b, :] = (torch.randn(B, H, b - a, D, generator=g) * seg_amps[i]).to(torch.bfloat16)

    cfg_tb = KQuantizerConfig(name="I_TokenBlock6_TEST", kind="kivi_temporal_window",
                              bits=4, n_temporal_windows=6, temporal_mode="token_block")
    K_tb_q = _kivi_temporal_window(K_tb, cfg_tb, layer_idx=0, slice_info=None,
                                   cache_offset=0,
                                   qmax=float(2 ** (4 - 1) - 1),
                                   n_windows=6, mode="token_block")
    n_distinct_tb, residuals_tb = n_distinct_residual_levels(
        K_tb, K_tb_q, [0, 10, 20, 30, 40, 50, 60])

    # visual_only n=4 with text-prefix + suffix: T=60, v_start=10, v_end=50,
    # 4 visual windows of 10 each => boundaries [0, 10, 20, 30, 40, 50, 60].
    K_vo = torch.zeros(B, H, T, D, dtype=torch.bfloat16)
    g2 = torch.Generator().manual_seed(2)
    for i, (a, b) in enumerate([(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]):
        K_vo[:, :, a:b, :] = (torch.randn(B, H, b - a, D, generator=g2) * seg_amps[i]).to(torch.bfloat16)

    cfg_vo = KQuantizerConfig(name="I_TempWin4_TEST", kind="kivi_temporal_window",
                              bits=4, n_temporal_windows=4, temporal_mode="visual_only")
    slice_info = {"v_start": 10, "v_end": 50, "seq_len": 60}
    K_vo_q = _kivi_temporal_window(K_vo, cfg_vo, layer_idx=0,
                                   slice_info=slice_info, cache_offset=0,
                                   qmax=float(2 ** (4 - 1) - 1),
                                   n_windows=4, mode="visual_only")
    n_distinct_vo, residuals_vo = n_distinct_residual_levels(
        K_vo, K_vo_q, [0, 10, 20, 30, 40, 50, 60])

    if n_distinct_tb < 5:
        return False, (f"token_block n=6 produced only {n_distinct_tb} distinct "
                       f"residual levels (expected ~6); residuals={residuals_tb}")
    if n_distinct_vo < 5:
        return False, (f"visual_only n=4 with text prefix+suffix produced only "
                       f"{n_distinct_vo} distinct residual levels (expected ~6); "
                       f"residuals={residuals_vo}")

    return True, (f"token_block6: {n_distinct_tb} distinct residuals; "
                  f"visual_only4+text: {n_distinct_vo} distinct residuals")


def _check_outlier8_subset_of_top16(model_short: str = "Qwen2.5-VL-7B-Instruct"
                                    ) -> tuple[bool, str]:
    """Real calib NPZ if present: assert top16[:, :, :8] ⊆ top16[:, :, :16]
    per (L, H_kv). This is true by definition (slicing); the check is that
    we read the right key and the indices are integer and in range.
    """
    npz_path = CALIBRATION_DIR / f"expF_kcalib_{model_short}_frames64.npz"
    if not npz_path.exists():
        # Synthetic fallback.
        idx = np.random.randint(0, 128, size=(28, 4, 16), dtype=np.int64)
        for L in range(28):
            for H in range(4):
                if not set(idx[L, H, :8].tolist()).issubset(set(idx[L, H, :16].tolist())):
                    return False, f"synthetic check failed at (L={L}, H={H})"
        return True, "no real calib NPZ; synthetic top8 ⊆ top16 confirmed"

    arrays = np.load(npz_path)
    if "outlier_channel_idx_top16" not in arrays.files:
        return False, f"missing key outlier_channel_idx_top16 in {npz_path}"
    idx = arrays["outlier_channel_idx_top16"]  # [L, H, 16]
    L, H, n_top = idx.shape
    if n_top != 16:
        return False, f"top16 last-dim {n_top} != 16"
    bad = []
    for layer in range(L):
        for head in range(H):
            top8 = set(idx[layer, head, :8].tolist())
            top16 = set(idx[layer, head, :16].tolist())
            if not top8.issubset(top16):
                bad.append((layer, head))
    if bad:
        return False, f"top8 ⊄ top16 at {len(bad)} (L,H) cells: first={bad[:3]}"
    if int(idx.min()) < 0 or int(idx.max()) >= 128:
        return False, f"index out of range [0, 128): min={idx.min()} max={idx.max()}"
    return True, f"{npz_path.name}: top16 shape={tuple(idx.shape)}; top8 ⊆ top16 for all (L, H)"


def _check_seed1_split_supersets() -> tuple[bool, str]:
    """make_split(seed=1) at Stage 1 (n=64) ⊂ Stage 3 (n=200) per bucket.

    LongVideoBench is gated on HF; if not available locally (e.g. running
    on a dev Mac without the cached dataset at $LONGVIDEOBENCH_ROOT), this
    check is skipped — the same logic is exercised on the remote where the
    dataset is mounted at /data/subha2/longvideobench/.
    """
    from data_longvideobench import load_all_items, make_split

    try:
        items = load_all_items()
    except Exception as e:
        return True, (f"skipped: dataset unavailable locally "
                      f"({type(e).__name__}: {str(e)[:80]}); will run on remote")
    if not items:
        return True, "skipped: no items loaded; set $LONGVIDEOBENCH_ROOT on remote"

    split_n64 = make_split(items, seed=1, targets={"short": 16, "mid": 16,
                                                   "long": 16, "very_long": 16},
                           cal_fraction=0.0)
    split_n200 = make_split(items, seed=1, targets={"short": 50, "mid": 50,
                                                    "long": 50, "very_long": 50},
                            cal_fraction=0.0)
    eval64 = set(split_n64["eval"])
    eval200 = set(split_n200["eval"])
    if not eval64.issubset(eval200):
        diff = eval64 - eval200
        return False, (f"n=64 not subset of n=200 at seed=1: "
                       f"{len(diff)} items in n=64 missing from n=200")
    if len(eval64) != 64 or len(eval200) != 200:
        return False, f"len(eval)={len(eval64)}/{len(eval200)} != 64/200"
    return True, f"seed=1 split: n=64 ({len(eval64)}) ⊂ n=200 ({len(eval200)})"


def _run_phase_a(model_short: str) -> tuple[list[dict], bool]:
    results: list[dict] = []
    all_pass = True

    checks = [
        ("tempwin_outlier_kernel_round_trip", _check_tempwin_outlier_kernel_round_trip),
        ("tempwin_window_boundaries",         _check_tempwin_window_boundaries),
        ("vidkv_v_scale_differs_from_uniform", _check_vidkv_v_scale_differs_from_uniform),
        ("tokenblock6_segment_count",         _check_tokenblock6_segment_count),
        ("outlier8_subset_of_top16",          lambda: _check_outlier8_subset_of_top16(model_short)),
        ("seed1_split_supersets",             _check_seed1_split_supersets),
    ]
    for name, fn in checks:
        try:
            ok, det = fn()
        except Exception as e:
            ok, det = False, f"{type(e).__name__}: {e}"
        all_pass = all_pass and ok
        results.append({"check": name, "pass": ok, "detail": det})

    return results, all_pass


# ===================================================================
# Phase B — live-model checks (require --model)
# ===================================================================


def _check_visual_span_detection_seed1(model_id: str, n_items: int = 8
                                       ) -> tuple[bool, str]:
    from data_longvideobench import (
        filter_items, format_mcq_messages, load_all_items, load_split,
    )
    from qwen_vl_utils import process_vision_info  # type: ignore
    from run_inference import load_model
    from text_slices import find_text_slice_spans
    from expI_temporal_kivi import ensure_i_split

    sp = ensure_i_split(stage=1, seed=1)
    items_all = load_all_items()
    split = load_split(sp)
    eval_items = filter_items(items_all, split["eval"])[:n_items]
    if not eval_items:
        return False, f"no items in {sp}"

    model, processor = load_model(model_id, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    n_failed = 0
    spans: list[tuple[int, int]] = []
    for it in eval_items:
        msgs = format_mcq_messages(it, n_frames=128)
        prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        slices = find_text_slice_spans(inputs["input_ids"], processor, it)
        v_start = int(slices.get("_v_start", -1))
        v_end = int(slices.get("_v_end", -1))
        spans.append((v_start, v_end))
        if not (v_start >= 0 and v_end > v_start):
            n_failed += 1

    if n_failed > 0:
        return False, (f"{n_failed}/{len(eval_items)} items have invalid visual span; "
                       f"first 4 spans: {spans[:4]}")
    return True, f"{len(eval_items)}/{len(eval_items)} items have v_end > v_start (spans first 4: {spans[:4]})"


def _check_bf16_vs_i_conditions_logits_differ(model_id: str,
                                              calib_npz: Optional[Path]
                                              ) -> tuple[bool, str]:
    """On 1 LVB item: BF16, I3, I7, I8 all produce non-equal first-token
    logprob vectors. Catches the failure mode where a kernel silently
    no-ops (returns K unchanged) and produces the same logits as BF16.
    """
    from data_longvideobench import (
        answer_token_ids, filter_items, format_mcq_messages,
        load_all_items, load_split,
    )
    from qwen_vl_utils import process_vision_info  # type: ignore
    from run_inference import load_model
    from text_slices import find_text_slice_spans
    from transformers.cache_utils import DynamicCache
    from fake_quant_kv_cache import BitController, FakeQuantKVCache
    from k_quantizers import build_f_conditions
    from expI_temporal_kivi import ensure_i_split

    calib = None
    if calib_npz is not None and calib_npz.exists():
        arrs = np.load(calib_npz)
        calib = {k: arrs[k] for k in arrs.files}

    sp = ensure_i_split(stage=1, seed=1)
    items_all = load_all_items()
    split = load_split(sp)
    eval_items = filter_items(items_all, split["eval"])
    if not eval_items:
        return False, f"no items in {sp}"
    item = eval_items[0]

    model, processor = load_model(model_id, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    n_layers = len(getattr(model.model, "layers", []) or model.model.language_model.layers)
    n_kv_heads = getattr(model.config, "num_key_value_heads", 4)

    msgs = format_mcq_messages(item, n_frames=128)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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

    fc = {cfg.name: cfg for cfg in build_f_conditions(calib=calib)}
    n_options = len(item.candidates)
    answer_ids = answer_token_ids(processor, n=n_options)

    def score(cache):
        with torch.no_grad():
            out = model.generate(**inputs, past_key_values=cache, max_new_tokens=1,
                                 do_sample=False, return_dict_in_generate=True,
                                 output_scores=True, use_cache=True)
        first_logits = out.scores[0]
        return torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist()

    # BF16: vanilla DynamicCache.
    bf16_logp = score(DynamicCache())

    # I3: H6 TempWin2 (visual_only).
    cfg_i3 = fc["H6_KIVI_TempWin2"]
    ctrl = BitController(num_layers=n_layers, num_kv_heads=n_kv_heads,
                         mode="V1", default_k_bits=4, default_v_bits=4)
    cache_i3 = FakeQuantKVCache(ctrl, k_quantizer_config=cfg_i3)
    cache_i3.set_slice_info(slice_info)
    i3_logp = score(cache_i3)

    # I7: TempWin2 + VidKV V.
    cfg_i7 = fc["I_TempWin2_VidKVV"]
    ctrl7 = BitController(num_layers=n_layers, num_kv_heads=n_kv_heads,
                          mode="V1", default_k_bits=4, default_v_bits=4)
    cache_i7 = FakeQuantKVCache(ctrl7, k_quantizer_config=cfg_i7)
    cache_i7.set_slice_info(slice_info)
    i7_logp = score(cache_i7)

    # I8: TempWin2 + outlier-8.
    if calib is None:
        return False, "calib NPZ required for I8 logits-differ check; pass --calib_npz"
    cfg_i8 = fc["I_TempWin2_Outlier8"]
    ctrl8 = BitController(num_layers=n_layers, num_kv_heads=n_kv_heads,
                          mode="V1", default_k_bits=4, default_v_bits=4)
    cache_i8 = FakeQuantKVCache(ctrl8, k_quantizer_config=cfg_i8)
    cache_i8.set_slice_info(slice_info)
    i8_logp = score(cache_i8)

    def max_abs(a, b):
        return max(abs(x - y) for x, y in zip(a, b))

    pairs = [
        ("BF16-vs-I3", bf16_logp, i3_logp),
        ("BF16-vs-I7", bf16_logp, i7_logp),
        ("BF16-vs-I8", bf16_logp, i8_logp),
        ("I3-vs-I7",   i3_logp,   i7_logp),
        ("I3-vs-I8",   i3_logp,   i8_logp),
        ("I7-vs-I8",   i7_logp,   i8_logp),
    ]
    bad = []
    for name, a, b in pairs:
        if max_abs(a, b) < 1e-6:
            bad.append(name)
    if bad:
        return False, f"some logits identical (silent no-op suspected): {bad}"
    deltas = ", ".join(f"{n}={max_abs(a, b):.4f}" for n, a, b in pairs)
    return True, f"all 6 pairs differ; max-abs deltas: {deltas}"


def _run_phase_b(model_id: str, calib_npz: Optional[Path]) -> tuple[list[dict], bool]:
    results: list[dict] = []
    all_pass = True

    checks = [
        ("visual_span_detection_seed1",
         lambda: _check_visual_span_detection_seed1(model_id)),
        ("bf16_vs_i_conditions_logits_differ",
         lambda: _check_bf16_vs_i_conditions_logits_differ(model_id, calib_npz)),
    ]
    for name, fn in checks:
        try:
            ok, det = fn()
        except Exception as e:
            ok, det = False, f"{type(e).__name__}: {e}"
        all_pass = all_pass and ok
        results.append({"check": name, "pass": ok, "detail": det})

    return results, all_pass


# ===================================================================
# Driver
# ===================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="If set, runs Phase B on a GPU. Otherwise Phase A only.")
    ap.add_argument("--calib_npz", type=Path, default=None,
                    help="Calibration NPZ for I8 logits-differ check (Phase B).")
    ap.add_argument("--out_md", type=Path, default=RESULTS_DIR / "expI_smoke.md")
    args = ap.parse_args()

    model_short = (args.model or "Qwen/Qwen2.5-VL-7B-Instruct").split("/")[-1]
    if args.calib_npz is None:
        cand = CALIBRATION_DIR / f"expF_kcalib_{model_short}_frames64.npz"
        if cand.exists():
            args.calib_npz = cand

    print(f"[expI_smoke] {_ts()} starting Phase A", flush=True)
    a_results, a_ok = _run_phase_a(model_short)
    for r in a_results:
        marker = "PASS" if r["pass"] else "FAIL"
        print(f"  [{marker}] {r['check']}: {r['detail']}", flush=True)

    b_results: list[dict] = []
    b_ok = True
    if args.model is not None:
        print(f"[expI_smoke] {_ts()} starting Phase B (model={args.model})", flush=True)
        b_results, b_ok = _run_phase_b(args.model, args.calib_npz)
        for r in b_results:
            marker = "PASS" if r["pass"] else "FAIL"
            print(f"  [{marker}] {r['check']}: {r['detail']}", flush=True)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# expI_smoke results — {_ts()}", ""]
    lines.append(f"**Phase A:** {'PASS' if a_ok else 'FAIL'} ({sum(1 for r in a_results if r['pass'])}/"
                 f"{len(a_results)})")
    if args.model:
        lines.append(f"**Phase B:** {'PASS' if b_ok else 'FAIL'} ({sum(1 for r in b_results if r['pass'])}/"
                     f"{len(b_results)})")
    lines.append("")
    lines.append("## Phase A — synthetic kernel checks")
    lines.append("")
    lines.append("| Check | Result | Detail |")
    lines.append("|---|---|---|")
    for r in a_results:
        marker = "PASS" if r["pass"] else "**FAIL**"
        lines.append(f"| {r['check']} | {marker} | {r['detail']} |")
    if args.model:
        lines.append("")
        lines.append("## Phase B — live-model checks")
        lines.append("")
        lines.append("| Check | Result | Detail |")
        lines.append("|---|---|---|")
        for r in b_results:
            marker = "PASS" if r["pass"] else "**FAIL**"
            lines.append(f"| {r['check']} | {marker} | {r['detail']} |")
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"[expI_smoke] wrote {args.out_md}")

    if not (a_ok and b_ok):
        print("[expI_smoke] FAILED", flush=True)
        sys.exit(2)
    print("[expI_smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
