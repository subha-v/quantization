"""Smoke test for Exp F K-quantizer screening. Halts pipeline on any failure.

Two phases:

  Phase A (no model required) — pure-tensor sanity:
    1. degenerate_bf16        bf16 kind returns K exactly (torch.equal)
    2. shape_dtype_invariance all 14 conditions return same shape/dtype as K
    3. f4_vs_f1_differ        on synthetic K, F4 (KIVI per-channel-seq) and F1
                              (uniform-INT4) produce different K_q (round-trip
                              error)
    4. calib_round_trip       (only if --calib_file given) load NPZ + JSON,
                              assert (num_layers, num_kv_heads, head_dim) =
                              (28, 4, 128) for the canonical setup
    5. outlier_preservation   F8 kind preserves protected channels exactly

  Phase B (requires --model and a GPU) — live-model logits-differ:
    6. f0_vs_f1_logits_differ  ||logits_BF16 - logits_INT4||_inf > 1e-3
                               (cache wiring assertion, mirrors E1 smoke).
    7. f4_vs_f1_logits_differ  proves new K-quantizer dispatch is invoked.
    8. f10_vs_f4_logits_differ (requires calib_file) proves score-cal scales
                               actually replace max-abs scales.

Writes qwen/results/expF_smoke.md with PASS/FAIL per check; exits 2 on FAIL.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===================================================================
# Phase A: synthetic-tensor checks (no model load)
# ===================================================================


def _make_synthetic_k(seed: int = 0, B: int = 1, H: int = 4, T: int = 200,
                      D: int = 128, dtype=torch.bfloat16) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    K = torch.randn(B, H, T, D, generator=g, dtype=torch.float32) * 0.5
    # Add a few outlier channels to make F8/F9 meaningful.
    out_ch = [3, 17, 31, 63, 91, 100, 113, 127]
    for d in out_ch:
        K[:, :, :, d] *= 8.0
    return K.to(dtype)


def _make_fake_calib(num_layers: int = 28, num_kv_heads: int = 4,
                     head_dim: int = 128, n_outliers: int = 16,
                     seed: int = 0) -> dict:
    """Synthesize plausible calibration data for Phase A correctness checks.
    The numbers are not meaningful — only shapes and selection logic are."""
    rng = np.random.default_rng(seed)
    k_chan_e = rng.gamma(shape=2.0, scale=0.5, size=(num_layers, num_kv_heads, head_dim)).astype(np.float32)
    # Make a few channels reliably "outlier" so F8 selection is deterministic.
    for L in range(num_layers):
        for H in range(num_kv_heads):
            top = rng.choice(head_dim, size=n_outliers, replace=False)
            k_chan_e[L, H, top] = 100.0 + rng.standard_normal(n_outliers).astype(np.float32) * 0.1
    # Precompute outlier indices top-16 per (L, H).
    top16 = np.argsort(k_chan_e, axis=-1)[..., -n_outliers:][..., ::-1].copy()
    q_e = rng.gamma(shape=2.0, scale=0.5, size=(num_layers, num_kv_heads, head_dim)).astype(np.float32)
    q_e_t = q_e * (1.0 + 0.3 * rng.standard_normal((num_layers, num_kv_heads, head_dim)).astype(np.float32))
    q_e_v = q_e * (1.0 + 0.3 * rng.standard_normal((num_layers, num_kv_heads, head_dim)).astype(np.float32))
    return {
        "k_channel_energy": k_chan_e,
        "outlier_channel_idx_top16": top16,
        "q_energy": q_e,
        "q_energy_text": np.clip(q_e_t, 1e-6, None),
        "q_energy_visual": np.clip(q_e_v, 1e-6, None),
    }


def _run_phase_a(num_layers: int, num_kv_heads: int, head_dim: int,
                 calib: Optional[dict]) -> tuple[list[dict], bool]:
    from k_quantizers import (
        KQuantizerConfig, apply_k_quantizer, build_f_conditions, KQUANTIZER_KINDS,
    )

    results: list[dict] = []
    all_pass = True

    K = _make_synthetic_k()
    seq_len = K.shape[-2]
    fake_slice_info = {
        "v_start": 30,
        "v_end": 150,
        "seq_len": seq_len,
        "role_spans": {
            "header": (0, 10),
            "visual": (30, 150),
            "question": (151, 170),
            "options": (170, 185),
            "instruction": (185, 195),
            "answer_prefix": (195, 200),
        },
    }

    # ---- Check 1: degenerate bf16 ----
    cfg_bf16 = KQuantizerConfig(name="bf16_check", kind="bf16", bits=16)
    K_q = apply_k_quantizer(K, cfg_bf16, layer_idx=0, slice_info=fake_slice_info)
    ok = torch.equal(K_q, K)
    all_pass = all_pass and ok
    results.append({
        "check": "degenerate_bf16",
        "pass": ok,
        "detail": ("bf16 kind returns K bit-exact" if ok
                   else f"BF16 kind altered K (max diff = {(K_q - K).abs().max().item():.4e})"),
    })

    # ---- Check 2: shape/dtype invariance for all 14 conditions ----
    conditions = build_f_conditions(calib=calib)
    shape_ok_count = 0
    shape_fails: list[str] = []
    for cfg in conditions:
        try:
            K_q = apply_k_quantizer(
                K, cfg, layer_idx=0, slice_info=fake_slice_info, cache_offset=0,
            )
            same_shape = K_q.shape == K.shape
            same_dtype = K_q.dtype == K.dtype
            if same_shape and same_dtype:
                shape_ok_count += 1
            else:
                shape_fails.append(
                    f"{cfg.name}: shape={K_q.shape} dtype={K_q.dtype}"
                )
        except Exception as e:
            shape_fails.append(f"{cfg.name}: {type(e).__name__}: {e}")
    ok = (shape_ok_count == len(conditions))
    all_pass = all_pass and ok
    results.append({
        "check": "shape_dtype_invariance",
        "pass": ok,
        "detail": (f"all {shape_ok_count}/{len(conditions)} conditions ok"
                   if ok else "; ".join(shape_fails[:5])),
    })

    # ---- Check 3: F4 vs F1 differ on synthetic K ----
    cfg_f1 = KQuantizerConfig(name="F1", kind="uniform_int4", bits=4)
    cfg_f4 = KQuantizerConfig(name="F4", kind="kivi_per_channel_seq", bits=4)
    K_f1 = apply_k_quantizer(K, cfg_f1, layer_idx=0)
    K_f4 = apply_k_quantizer(K, cfg_f4, layer_idx=0)
    delta = (K_f1.float() - K_f4.float()).abs().max().item()
    ok = (delta > 1e-3) and (not torch.equal(K_f1, K_f4))
    all_pass = all_pass and ok
    results.append({
        "check": "f4_vs_f1_differ_synthetic",
        "pass": ok,
        "detail": f"||K_F1 - K_F4||_inf = {delta:.4e} (threshold 1e-3)",
    })

    # ---- Check 4: calib round-trip (if provided) ----
    if calib is not None:
        keys_required = (
            "k_channel_energy", "outlier_channel_idx_top16",
            "q_energy", "q_energy_text", "q_energy_visual",
        )
        missing = [k for k in keys_required if k not in calib]
        shape_issues = []
        for k in keys_required:
            if k not in calib:
                continue
            arr = np.asarray(calib[k])
            if k == "outlier_channel_idx_top16":
                if arr.shape != (num_layers, num_kv_heads, 16):
                    shape_issues.append(f"{k}.shape={arr.shape}")
            else:
                if arr.shape != (num_layers, num_kv_heads, head_dim):
                    shape_issues.append(f"{k}.shape={arr.shape}")
        ok = not missing and not shape_issues
        all_pass = all_pass and ok
        results.append({
            "check": "calib_round_trip",
            "pass": ok,
            "detail": (f"all keys present with shapes ({num_layers}, {num_kv_heads}, *) ok"
                       if ok else f"missing={missing} shape_issues={shape_issues}"),
        })
    else:
        results.append({
            "check": "calib_round_trip",
            "pass": True,
            "detail": "skipped (no --calib_file given)",
        })

    # ---- Check 5: outlier preservation (F8 only, with calib) ----
    if calib is not None:
        cfg_f8 = KQuantizerConfig(
            name="F8", kind="kivi_outlier8", bits=4, n_outliers=8, calib=calib,
        )
        K_f8 = apply_k_quantizer(K, cfg_f8, layer_idx=0)
        outlier_idx = torch.as_tensor(calib["outlier_channel_idx_top16"][0, :, :8],
                                      dtype=torch.long)  # [H, 8]
        H = K.shape[1]
        all_outlier_ok = True
        bad_examples = []
        for h in range(H):
            ch = outlier_idx[h]
            if not torch.equal(K_f8[:, h, :, ch], K[:, h, :, ch]):
                all_outlier_ok = False
                diff = (K_f8[:, h, :, ch] - K[:, h, :, ch]).abs().max().item()
                bad_examples.append(f"h={h} max_diff={diff:.4e}")
        ok = all_outlier_ok
        all_pass = all_pass and ok
        results.append({
            "check": "outlier_preservation_f8",
            "pass": ok,
            "detail": ("F8 protected channels match K exactly across all 4 heads"
                       if ok else "; ".join(bad_examples)),
        })
    else:
        results.append({
            "check": "outlier_preservation_f8",
            "pass": True,
            "detail": "skipped (no --calib_file given)",
        })

    return results, all_pass


# ===================================================================
# Phase B: live-model logits-differ checks
# ===================================================================


def _run_phase_b(model_id: str, frames: int, n_items: int, split_file: Path,
                 calib: Optional[dict]) -> tuple[list[dict], bool]:
    """Runs assertions 6, 7, 8 with a real model and 1 sample item."""
    from data_longvideobench import (
        DEFAULT_SPLIT_FILE, filter_items, format_mcq_messages, load_all_items, load_split,
    )
    from fake_quant_kv_cache import BitController, FakeQuantKVCache
    from k_quantizers import KQuantizerConfig
    from qwen_vl_utils import process_vision_info  # type: ignore
    from text_slices import find_text_slice_spans
    from run_inference import load_model

    items_all = load_all_items()
    split = load_split(split_file)
    eval_items = filter_items(items_all, split["eval"])[:n_items]
    print(f"[F-smoke] phase B n_items={len(eval_items)}", flush=True)

    model, processor = load_model(model_id, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)

    item = eval_items[0]
    msgs = format_mcq_messages(item, n_frames=frames)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    seq_len = int(inputs["input_ids"].shape[1])
    slices = find_text_slice_spans(inputs["input_ids"], processor, item)
    v_start = int(slices["_v_start"])
    v_end = int(slices["_v_end"])
    role_spans = {k: tuple(slices[k]) for k in
                  ("header", "question", "options", "instruction", "answer_prefix")
                  if isinstance(slices.get(k), tuple)}
    role_spans["visual"] = (v_start, v_end)
    slice_info = dict(
        v_start=v_start, v_end=v_end, seq_len=seq_len, role_spans=role_spans,
    )

    def _logits_for_cfg(cfg: Optional[KQuantizerConfig]) -> torch.Tensor:
        ctrl = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                             mode="V1", default_k_bits=4, default_v_bits=4)
        if cfg is None:
            # F0 BF16 baseline — use a vanilla DynamicCache
            from transformers.cache_utils import DynamicCache
            cache = DynamicCache()
        else:
            cache = FakeQuantKVCache(ctrl, k_quantizer_config=cfg)
            cache.set_slice_info(slice_info)
        out = model.generate(**inputs, past_key_values=cache, max_new_tokens=1,
                             do_sample=False, return_dict_in_generate=True,
                             output_scores=True, use_cache=True)
        return out.scores[0][0].float().cpu()

    results: list[dict] = []
    all_pass = True

    cfg_f0 = None  # BF16 baseline
    cfg_f1 = KQuantizerConfig(name="F1", kind="uniform_int4", bits=4)
    cfg_f4 = KQuantizerConfig(name="F4", kind="kivi_per_channel_seq", bits=4)

    L_f0 = _logits_for_cfg(cfg_f0)
    torch.cuda.empty_cache()
    L_f1 = _logits_for_cfg(cfg_f1)
    torch.cuda.empty_cache()

    inf_norm_01 = float((L_f0 - L_f1).abs().max().item())
    ok = inf_norm_01 > 1e-3
    all_pass = all_pass and ok
    results.append({
        "check": "f0_vs_f1_logits_differ",
        "pass": ok,
        "detail": f"||logits_BF16 - logits_INT4||_inf = {inf_norm_01:.4e} (threshold 1e-3)",
    })

    L_f4 = _logits_for_cfg(cfg_f4)
    torch.cuda.empty_cache()
    inf_norm_14 = float((L_f1 - L_f4).abs().max().item())
    ok = inf_norm_14 > 1e-3
    all_pass = all_pass and ok
    results.append({
        "check": "f4_vs_f1_logits_differ",
        "pass": ok,
        "detail": f"||logits_F1 - logits_F4||_inf = {inf_norm_14:.4e} (proves KIVI dispatch)",
    })

    if calib is not None:
        cfg_f10 = KQuantizerConfig(name="F10", kind="score_cal_generic", bits=4, calib=calib)
        L_f10 = _logits_for_cfg(cfg_f10)
        torch.cuda.empty_cache()
        inf_norm_410 = float((L_f4 - L_f10).abs().max().item())
        ok = inf_norm_410 > 1e-3
        all_pass = all_pass and ok
        results.append({
            "check": "f10_vs_f4_logits_differ",
            "pass": ok,
            "detail": f"||logits_F4 - logits_F10||_inf = {inf_norm_410:.4e} (proves score-cal dispatch)",
        })
    else:
        results.append({
            "check": "f10_vs_f4_logits_differ",
            "pass": True,
            "detail": "skipped (no --calib_file given)",
        })

    return results, all_pass


# ===================================================================
# Calibration loader
# ===================================================================


def _load_calib(calib_file: Optional[Path]) -> Optional[dict]:
    if calib_file is None or not calib_file.exists():
        return None
    npz_file = calib_file.with_suffix(".npz")
    if not npz_file.exists():
        npz_file = calib_file
    arrays = np.load(npz_file)
    return {k: arrays[k] for k in arrays.files}


# ===================================================================
# Main
# ===================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="",
                    help="Pass to enable Phase B logits-differ checks. Empty -> skip Phase B.")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--n_items", type=int, default=1,
                    help="Phase B uses 1 item; --n_items reserved for future expansion.")
    ap.add_argument("--num_layers", type=int, default=28)
    ap.add_argument("--num_kv_heads", type=int, default=4)
    ap.add_argument("--head_dim", type=int, default=128)
    ap.add_argument("--calib_file", type=Path, default=None,
                    help="Path to expF_kcalib_*.npz file (or .json sibling). Optional.")
    ap.add_argument("--use_synthetic_calib", action="store_true",
                    help="If set, generate synthetic calib data for Phase A only "
                         "(useful for laptop-only smoke).")
    ap.add_argument("--split_file", type=Path,
                    default=Path(__file__).resolve().parents[1] / "calibration" / "split_seed0.json")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "expF_smoke.md")
    args = ap.parse_args()

    calib = _load_calib(args.calib_file)
    if calib is None and args.use_synthetic_calib:
        calib = _make_fake_calib(num_layers=args.num_layers, num_kv_heads=args.num_kv_heads,
                                 head_dim=args.head_dim)

    # Phase A
    a_results, a_pass = _run_phase_a(
        num_layers=args.num_layers, num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim, calib=calib,
    )

    # Phase B (only if --model specified)
    b_results: list[dict] = []
    b_pass = True
    if args.model:
        b_results, b_pass = _run_phase_b(
            args.model, args.frames, args.n_items, args.split_file,
            calib=calib if not args.use_synthetic_calib else None,  # don't use fake calib for live-model F10 check
        )

    overall = a_pass and b_pass

    # Write report
    lines = [
        f"# Exp F Smoke Report\n",
        f"Generated: {_ts()}\n",
        f"num_layers: {args.num_layers}  num_kv_heads: {args.num_kv_heads}  head_dim: {args.head_dim}",
        f"calib_file: {args.calib_file or 'N/A'}  synthetic_calib: {args.use_synthetic_calib}",
        f"model: `{args.model or '(skipped)'}`  frames: {args.frames}\n",
        "## Phase A — synthetic-tensor checks\n",
        "| # | Check | Pass | Detail |",
        "|---|---|:-:|---|",
    ]
    for i, r in enumerate(a_results, 1):
        marker = "PASS" if r["pass"] else "FAIL"
        lines.append(f"| {i} | `{r['check']}` | {marker} | {r['detail']} |")

    lines += ["\n## Phase B — live-model logits-differ\n",
              "| # | Check | Pass | Detail |",
              "|---|---|:-:|---|"]
    if not args.model:
        lines.append("| - | (skipped — no --model) | -- | use --model to enable |")
    else:
        for i, r in enumerate(b_results, 1):
            marker = "PASS" if r["pass"] else "FAIL"
            lines.append(f"| {i} | `{r['check']}` | {marker} | {r['detail']} |")

    lines.append(f"\n## Overall: {'PASS' if overall else 'FAIL'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[F-smoke] wrote {args.out}")
    print(f"[F-smoke] overall: {'PASS' if overall else 'FAIL'}", flush=True)
    if not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
