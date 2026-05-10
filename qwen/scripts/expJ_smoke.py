"""Smoke test for Exp J (cross-modal outlier-channel KV quantization).

Phase A (no model required) — synthetic-tensor checks:
  1. custom_outlier_idx_lookup     cfg.outlier_idx_key swaps in alternate
                                   indices; the right channels get protected.
  2. int_n_sidecode_round_trip     outlier_storage_bits=8 stores outlier
                                   channels at INT8 (not bit-identical to
                                   original; non-outlier channels match the
                                   default F4 quantization).
  3. layer_adaptive_budget_resolve _resolve_layer_adaptive_budget assigns
                                   exactly top_fraction × cells with
                                   n_per_cell, others 0.
  4. bits_accounting_J             cfg.avg_kv_bits via
                                   _compute_three_bit_columns matches the
                                   spec for J3-J14.
  5. seed2_split_supersets         make_split(seed=2, n=64) ⊂ n=200.

Phase B (live, requires --model + Exp J calib NPZ):
  6. visual_span_seed2             find_text_slice_spans on first 8 items
                                   of seed=2 split.
  7. logits_differ                 J6 (cross-modal), J9 (layer-adaptive),
                                   J12 (INT8 sidecode) each produce
                                   first-token logprobs different from BF16.

Output qwen/results/expJ_smoke.md; exit 2 on FAIL.
"""
from __future__ import annotations

import argparse
import json
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
# Phase A — synthetic checks
# ===================================================================


def _make_synthetic_k(seed: int = 0, B: int = 1, H: int = 4, T: int = 64,
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


def _check_custom_outlier_idx_lookup() -> tuple[bool, str]:
    from k_quantizers import KQuantizerConfig, apply_k_quantizer

    H, D = 4, 128
    L = 28
    # Generic top-16 plants channels [3, 17, 31, ...].
    generic_plants = [3, 17, 31, 63, 91, 100, 113, 127]
    K = _make_synthetic_k(H=H, T=64, D=D, outlier_chans=generic_plants)
    # Build calib with TWO different outlier-index arrays.
    generic_idx = np.tile(np.array(generic_plants + [d for d in range(D) if d not in generic_plants][:8],
                                   dtype=np.int32), (L, H, 1))
    custom_plants = [4, 18, 32, 64, 92, 101, 114, 126]  # different channels
    custom_idx = np.tile(np.array(custom_plants + [d for d in range(D) if d not in custom_plants][:8],
                                  dtype=np.int32), (L, H, 1))
    calib = {
        "outlier_channel_idx_top16": generic_idx,   # default key
        "outlier_idx_TT_top16": custom_idx,         # custom key
    }
    cfg_default = KQuantizerConfig(name="J3_test_default", kind="kivi_outlier8",
                                   bits=4, n_outliers=8, calib=calib)
    cfg_custom = KQuantizerConfig(name="J4_test_custom", kind="kivi_outlier8",
                                  bits=4, n_outliers=8, calib=calib,
                                  outlier_idx_key="outlier_idx_TT_top16")
    K_q_default = apply_k_quantizer(K, cfg_default, layer_idx=0)
    K_q_custom = apply_k_quantizer(K, cfg_custom, layer_idx=0)
    # Default: generic_plants exact-match, custom_plants quantized
    delta_g_def = (K[..., generic_plants].float() - K_q_default[..., generic_plants].float()).abs().max().item()
    delta_c_def = (K[..., custom_plants].float() - K_q_default[..., custom_plants].float()).abs().max().item()
    if delta_g_def > 1e-6 or delta_c_def < 1e-3:
        return False, (f"default-key restoration wrong: generic_delta={delta_g_def:.2e}, "
                       f"custom_delta={delta_c_def:.2e}")
    # Custom: custom_plants exact-match, generic_plants quantized
    delta_g_cus = (K[..., generic_plants].float() - K_q_custom[..., generic_plants].float()).abs().max().item()
    delta_c_cus = (K[..., custom_plants].float() - K_q_custom[..., custom_plants].float()).abs().max().item()
    if delta_c_cus > 1e-6 or delta_g_cus < 1e-3:
        return False, (f"custom-key restoration wrong: generic_delta={delta_g_cus:.2e}, "
                       f"custom_delta={delta_c_cus:.2e}")
    return True, (f"default & custom outlier_idx_key both restore correct channels: "
                  f"default(gen={delta_g_def:.2e},cus={delta_c_def:.4f}) "
                  f"custom(gen={delta_g_cus:.4f},cus={delta_c_cus:.2e})")


def _check_int_n_sidecode_round_trip() -> tuple[bool, str]:
    from k_quantizers import KQuantizerConfig, apply_k_quantizer

    H, D = 4, 128
    L = 28
    plants = [3, 17, 31, 63, 91, 100, 113, 127]
    K = _make_synthetic_k(H=H, T=64, D=D, outlier_chans=plants)
    calib = {
        "outlier_channel_idx_top16": np.tile(
            np.array(plants + [d for d in range(D) if d not in plants][:8], dtype=np.int32),
            (L, H, 1)),
    }
    cfg_bf16 = KQuantizerConfig(name="F8_BF16side", kind="kivi_outlier8", bits=4,
                                n_outliers=8, calib=calib, outlier_storage_bits=16)
    cfg_int8 = KQuantizerConfig(name="F8_INT8side", kind="kivi_outlier8", bits=4,
                                n_outliers=8, calib=calib, outlier_storage_bits=8)
    K_bf16 = apply_k_quantizer(K, cfg_bf16, layer_idx=0)
    K_int8 = apply_k_quantizer(K, cfg_int8, layer_idx=0)
    # Outlier channels: BF16 sidecode is bit-identical, INT8 sidecode is not but close.
    delta_bf16 = (K[..., plants].float() - K_bf16[..., plants].float()).abs().max().item()
    delta_int8 = (K[..., plants].float() - K_int8[..., plants].float()).abs().max().item()
    if delta_bf16 > 1e-6:
        return False, f"BF16 sidecode not bit-identical: delta={delta_bf16:.4e}"
    if delta_int8 < 1e-6 or delta_int8 > 5.0:
        return False, f"INT8 sidecode delta {delta_int8:.4f} out of expected range (0, 5)"
    # Non-outlier channels: should match between BF16 and INT8 (same F4 quantization).
    other = [d for d in range(D) if d not in plants]
    delta_other = (K_bf16[..., other].float() - K_int8[..., other].float()).abs().max().item()
    if delta_other > 1e-6:
        return False, f"non-outlier channels diverge between BF16 and INT8 sidecode: {delta_other:.4e}"
    return True, (f"BF16 sidecode delta={delta_bf16:.2e} (exact); "
                  f"INT8 sidecode delta={delta_int8:.4f} (lossy); "
                  f"non-outlier match across sidecodes (delta={delta_other:.2e})")


def _check_layer_adaptive_budget_resolve() -> tuple[bool, str]:
    from k_quantizers import _resolve_layer_adaptive_budget

    L, Hkv = 28, 4
    rng = np.random.default_rng(0)
    risk = rng.standard_normal((L, Hkv)).astype(np.float32) * 100 + 50
    calib = {"cell_risk_TT_TV": risk}
    budget = _resolve_layer_adaptive_budget(calib, "cell_risk_TT_TV", 0.50, n_per_cell=16)
    if budget is None or budget.shape != (L, Hkv):
        return False, f"budget shape={getattr(budget, 'shape', None)} != ({L}, {Hkv})"
    n_cells = L * Hkv
    expected_n_keep = int(round(0.50 * n_cells))
    actual_n_keep = int((budget == 16).sum())
    actual_n_zero = int((budget == 0).sum())
    if actual_n_keep != expected_n_keep:
        return False, (f"top-50% kept {actual_n_keep} cells, expected {expected_n_keep}")
    if actual_n_keep + actual_n_zero != n_cells:
        return False, (f"budget has unexpected values: {sorted(set(budget.flatten()))}")
    # Check that the kept cells are the actually-highest-risk ones.
    kept_mask = (budget == 16)
    kept_risk_min = risk[kept_mask].min()
    dropped_risk_max = risk[~kept_mask].max()
    if kept_risk_min < dropped_risk_max:
        return False, (f"kept cells include lower-risk than dropped cells: "
                       f"min(kept)={kept_risk_min:.3f} < max(dropped)={dropped_risk_max:.3f}")
    return True, (f"resolved budget: {actual_n_keep}/{n_cells} cells with budget=16, "
                  f"min(kept_risk)={kept_risk_min:.2f} >= max(dropped_risk)={dropped_risk_max:.2f}")


def _check_bits_accounting_J() -> tuple[bool, str]:
    from k_quantizers import build_f_conditions
    from expF_kquant_screen import _compute_three_bit_columns

    L, Hkv, D = 28, 4, 128
    # Build a calib with all required keys.
    calib = {
        "outlier_channel_idx_top16": np.zeros((L, Hkv, 16), dtype=np.int32),
        "k_channel_energy": np.ones((L, Hkv, D), dtype=np.float32),
        "outlier_idx_TT_top16": np.zeros((L, Hkv, 16), dtype=np.int32),
        "outlier_idx_TV_top16": np.zeros((L, Hkv, 16), dtype=np.int32),
        "outlier_idx_TT_TV_top16": np.zeros((L, Hkv, 16), dtype=np.int32),
        "outlier_idx_BAL_top16": np.zeros((L, Hkv, 16), dtype=np.int32),
        "outlier_idx_PIVOT_top16": np.zeros((L, Hkv, 16), dtype=np.int32),
        "cell_risk_TT_TV": np.random.default_rng(0).standard_normal((L, Hkv)).astype(np.float32),
        "cell_risk_all": np.random.default_rng(1).standard_normal((L, Hkv)).astype(np.float32),
    }
    fc = {cfg.name: cfg for cfg in build_f_conditions(calib=calib)}
    spec = {
        "F4_KIVI_PerChannelSeq": 4.000,
        "F8_KIVI_Outlier8":      4.375,
        "F9_KIVI_Outlier16":     4.750,
        "J4_Outlier8_TT":        4.375,
        "J5_Outlier8_TV":        4.375,
        "J6_Outlier8_TT_TV":     4.375,
        "J7_Outlier8_BAL":       4.375,
        "J8_Outlier8_PIVOT":     4.375,
        "J9_LA_TT_TV_50pct":     4.375,
        "J10_LA_ALL_50pct":      4.375,
        "J11_LA_TT_TV_75pct":    4.560,
        "J12_F9_INT8side":       4.250,
        "J13_F9_INT6side":       4.125,
        "J14_TT_TV_INT8side":    4.250,
    }
    bad = []
    for name, expected in spec.items():
        cfg = fc[name]
        n_out = int(cfg.n_outliers or 0)
        k, v, kv = _compute_three_bit_columns(cfg, "v1_kcfg", n_out, L, Hkv,
                                               head_dim=D, n_text=0, n_total=1)
        if abs(kv - expected) > 0.01:
            bad.append(f"{name}: got {kv:.4f}, expected {expected:.4f}")
    if bad:
        return False, "; ".join(bad[:5])
    return True, f"all {len(spec)} conditions match avg_kv_bits spec to ±0.01"


def _check_seed2_split_supersets() -> tuple[bool, str]:
    from data_longvideobench import load_all_items, make_split
    try:
        items = load_all_items()
    except Exception as e:
        return True, (f"skipped: dataset unavailable locally "
                      f"({type(e).__name__}: {str(e)[:80]}); will run on remote")
    if not items:
        return True, "skipped: no items loaded; will run on remote"
    s64 = make_split(items, seed=2,
                     targets={"short": 16, "mid": 16, "long": 16, "very_long": 16},
                     cal_fraction=0.0)
    s200 = make_split(items, seed=2,
                      targets={"short": 50, "mid": 50, "long": 50, "very_long": 50},
                      cal_fraction=0.0)
    e64 = set(s64["eval"])
    e200 = set(s200["eval"])
    if not e64.issubset(e200):
        return False, f"n=64 not subset of n=200 at seed=2: diff={len(e64 - e200)}"
    if len(e64) != 64 or len(e200) != 200:
        return False, f"sizes {len(e64)}/{len(e200)} != 64/200"
    return True, f"seed=2: n=64 ({len(e64)}) ⊂ n=200 ({len(e200)})"


def _run_phase_a() -> tuple[list[dict], bool]:
    results: list[dict] = []
    all_pass = True
    checks = [
        ("custom_outlier_idx_lookup",     _check_custom_outlier_idx_lookup),
        ("int_n_sidecode_round_trip",     _check_int_n_sidecode_round_trip),
        ("layer_adaptive_budget_resolve", _check_layer_adaptive_budget_resolve),
        ("bits_accounting_J",             _check_bits_accounting_J),
        ("seed2_split_supersets",         _check_seed2_split_supersets),
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
# Phase B — live-model checks
# ===================================================================


def _check_visual_span_seed2(model_id: str, n_items: int = 8) -> tuple[bool, str]:
    from data_longvideobench import filter_items, format_mcq_messages, load_all_items, load_split
    from qwen_vl_utils import process_vision_info  # type: ignore
    from run_inference import load_model
    from text_slices import find_text_slice_spans
    from expJ_xmodal_outlier import ensure_j_split

    sp = ensure_j_split(stage=1, seed=2)
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
        prompt_text = processor.apply_chat_template(msgs, tokenize=False,
                                                     add_generation_prompt=True)
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
        return False, f"{n_failed}/{len(eval_items)} items invalid v_span; first 4: {spans[:4]}"
    return True, f"{len(eval_items)}/{len(eval_items)} v_end > v_start; first 4: {spans[:4]}"


def _check_logits_differ(model_id: str, calib_npz: Path) -> tuple[bool, str]:
    """Run J0 (BF16), J6 (cross-modal), J9 (layer-adaptive), J12 (INT8 side)
    on one LVB item from the seed=2 split and assert all four pairwise differ.
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
    from expJ_xmodal_outlier import ensure_j_split

    if not calib_npz.exists():
        return False, f"calib NPZ missing: {calib_npz}"
    arrs = np.load(calib_npz)
    calib = {k: arrs[k] for k in arrs.files}

    sp = ensure_j_split(stage=1, seed=2)
    items_all = load_all_items()
    split = load_split(sp)
    eval_items = filter_items(items_all, split["eval"])
    if not eval_items:
        return False, f"no items in {sp}"
    item = eval_items[0]

    model, processor = load_model(model_id, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    n_layers = len(layers)
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

    bf16_logp = score(DynamicCache())

    def score_cfg(cfg):
        ctrl = BitController(num_layers=n_layers, num_kv_heads=n_kv_heads,
                             mode="V1", default_k_bits=4, default_v_bits=4)
        cache = FakeQuantKVCache(ctrl, k_quantizer_config=cfg)
        cache.set_slice_info(slice_info)
        return score(cache)

    j6_logp = score_cfg(fc["J6_Outlier8_TT_TV"])
    j9_logp = score_cfg(fc["J9_LA_TT_TV_50pct"])
    j12_logp = score_cfg(fc["J12_F9_INT8side"])

    def max_abs(a, b):
        return max(abs(x - y) for x, y in zip(a, b))

    pairs = [
        ("BF16-vs-J6",  bf16_logp, j6_logp),
        ("BF16-vs-J9",  bf16_logp, j9_logp),
        ("BF16-vs-J12", bf16_logp, j12_logp),
        ("J6-vs-J9",    j6_logp,   j9_logp),
        ("J6-vs-J12",   j6_logp,   j12_logp),
        ("J9-vs-J12",   j9_logp,   j12_logp),
    ]
    bad = [n for n, a, b in pairs if max_abs(a, b) < 1e-6]
    if bad:
        return False, f"some logits identical (silent no-op): {bad}"
    deltas = ", ".join(f"{n}={max_abs(a, b):.4f}" for n, a, b in pairs)
    return True, f"all 6 pairs differ: {deltas}"


def _run_phase_b(model_id: str, calib_npz: Path) -> tuple[list[dict], bool]:
    results: list[dict] = []
    all_pass = True
    checks = [
        ("visual_span_seed2", lambda: _check_visual_span_seed2(model_id)),
        ("logits_differ",     lambda: _check_logits_differ(model_id, calib_npz)),
    ]
    for name, fn in checks:
        try:
            ok, det = fn()
        except Exception as e:
            ok, det = False, f"{type(e).__name__}: {e}"
        all_pass = all_pass and ok
        results.append({"check": name, "pass": ok, "detail": det})
    return results, all_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--calib_npz", type=Path, default=None,
                    help="Exp J cross-modal calibration NPZ.")
    ap.add_argument("--out_md", type=Path, default=RESULTS_DIR / "expJ_smoke.md")
    args = ap.parse_args()

    print(f"[expJ_smoke] {_ts()} starting Phase A", flush=True)
    a_results, a_ok = _run_phase_a()
    for r in a_results:
        marker = "PASS" if r["pass"] else "FAIL"
        print(f"  [{marker}] {r['check']}: {r['detail']}", flush=True)

    b_results: list[dict] = []
    b_ok = True
    if args.model is not None:
        if args.calib_npz is None:
            model_short = args.model.split("/")[-1]
            cand = CALIBRATION_DIR / f"expJ_kcalib_{model_short}_frames128.npz"
            if cand.exists():
                args.calib_npz = cand
        if args.calib_npz is None or not args.calib_npz.exists():
            print(f"[expJ_smoke] WARN: --model set but no calib NPZ; skipping Phase B", flush=True)
        else:
            print(f"[expJ_smoke] {_ts()} starting Phase B (model={args.model})", flush=True)
            b_results, b_ok = _run_phase_b(args.model, args.calib_npz)
            for r in b_results:
                marker = "PASS" if r["pass"] else "FAIL"
                print(f"  [{marker}] {r['check']}: {r['detail']}", flush=True)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# expJ_smoke results — {_ts()}", ""]
    lines.append(f"**Phase A:** {'PASS' if a_ok else 'FAIL'} ({sum(1 for r in a_results if r['pass'])}/"
                 f"{len(a_results)})")
    if args.model:
        lines.append(f"**Phase B:** {'PASS' if b_ok else 'FAIL'} ({sum(1 for r in b_results if r['pass'])}/"
                     f"{len(b_results)})")
    lines.append("")
    lines.append("## Phase A — synthetic kernel + bits-accounting checks")
    lines.append("| Check | Result | Detail |")
    lines.append("|---|---|---|")
    for r in a_results:
        marker = "PASS" if r["pass"] else "**FAIL**"
        lines.append(f"| {r['check']} | {marker} | {r['detail']} |")
    if args.model and b_results:
        lines.append("")
        lines.append("## Phase B — live-model checks")
        lines.append("| Check | Result | Detail |")
        lines.append("|---|---|---|")
        for r in b_results:
            marker = "PASS" if r["pass"] else "**FAIL**"
            lines.append(f"| {r['check']} | {marker} | {r['detail']} |")
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"[expJ_smoke] wrote {args.out_md}")
    if not (a_ok and b_ok):
        print("[expJ_smoke] FAILED", flush=True)
        sys.exit(2)
    print("[expJ_smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
