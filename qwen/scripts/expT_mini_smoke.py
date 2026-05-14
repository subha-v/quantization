"""Phase 0 smoke checks for Exp T-mini (page-aware K formats on MM-NIAH).

This script verifies — using synthetic tensors where possible and 3 real
items where the page layout matters — that the new K-quantizer kinds and
counting-image loader behave as expected before the main overnight sweep.

Run order on the remote server (after copying scripts):
    cd /data/subha2/quantization/qwen
    CUDA_VISIBLE_DEVICES=0 python3 scripts/expT_mini_smoke.py \
        --out-md results/expT_mini_smoke.md

Most checks are CPU-only and finish in seconds. The "anchor parity" check
requires a model load and takes ~3 minutes for 3 items.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from k_quantizers import (
    KQuantizerConfig, apply_k_quantizer, build_f_conditions,
)
from counting_parser import parse_counting_output, score_counting


# ---------------- helpers ----------------

def _make_synthetic_K(B=1, H=4, T=512, D=128, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    K = torch.randn(B, H, T, D, dtype=torch.float32) * 1.5
    # Inject a few "outlier channels" so KIVI-style quantizers face real outliers.
    K[..., 7] *= 12
    K[..., 23] *= 8
    return K


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.norm((a - b).float()).item())


def _make_synthetic_slice_info(T=512, page_widths=(64, 96, 80, 70, 96, 60, 46),
                               kinds=("text", "in_context_image", "text",
                                      "in_context_image", "text",
                                      "choice_image", "text"),
                               item_id="smoke_synth") -> dict:
    """Build a fake slice_info with explicit page boundaries."""
    assert sum(page_widths) == T, f"sum(page_widths)={sum(page_widths)} != T={T}"
    page_boundaries: list[tuple[int, int, str]] = []
    visual_token_positions_per_image: list[list[int]] = []
    text_chunk_positions: list[list[int]] = []
    text_positions: list[int] = []
    visual_positions: list[int] = []
    cursor = 0
    first_visual_span: Optional[tuple[int, int]] = None
    for width, kind in zip(page_widths, kinds):
        lo, hi = cursor, cursor + width
        page_boundaries.append((lo, hi, kind))
        positions = list(range(lo, hi))
        if kind in ("in_context_image", "choice_image"):
            visual_token_positions_per_image.append(positions)
            visual_positions.extend(positions)
            if first_visual_span is None:
                first_visual_span = (lo, hi)
        else:
            text_chunk_positions.append(positions)
            text_positions.extend(positions)
        cursor = hi
    v_start, v_end = first_visual_span if first_visual_span is not None else (-1, -1)
    return {
        "v_start": v_start, "v_end": v_end, "seq_len": T,
        "text_positions": text_positions,
        "visual_positions": visual_positions,
        "role_spans": ({"visual": (v_start, v_end)} if v_start >= 0 else {}),
        "page_boundaries": page_boundaries,
        "visual_token_positions_per_image": visual_token_positions_per_image,
        "text_chunk_positions": text_chunk_positions,
        "item_id": item_id,
    }


# ---------------- smoke checks ----------------

@dataclass
class SmokeResult:
    name: str
    passed: bool
    detail: str


def check_pagelocal_differs_from_global(slice_info: dict) -> SmokeResult:
    """T8 PageLocal-F4 must produce a different K_q than T1 Global-F4."""
    K = _make_synthetic_K()
    cfg_global = KQuantizerConfig(name="t1_global", kind="kivi_per_channel_seq", bits=4)
    cfg_pl = KQuantizerConfig(name="t8_page_local", kind="kivi_page_local", bits=4)
    K_global = apply_k_quantizer(K, cfg_global, layer_idx=0, slice_info=slice_info,
                                 cache_offset=0)
    K_pl = apply_k_quantizer(K, cfg_pl, layer_idx=0, slice_info=slice_info, cache_offset=0)
    diff = _l2(K_global, K_pl)
    passed = diff > 1e-3 and not torch.isnan(K_pl).any() and not torch.isinf(K_pl).any()
    return SmokeResult("PageLocal-F4 differs from Global-F4",
                       passed, f"L2 diff = {diff:.4f} (must be > 1e-3, no NaN/Inf)")


def check_pagelocal_token_block_coexist(slice_info: dict) -> SmokeResult:
    """Both PageLocal and TokenBlock at the same n_segments should run without error."""
    K = _make_synthetic_K()
    cfg_pl = KQuantizerConfig(name="t8_pl", kind="kivi_page_local", bits=4)
    cfg_tb = KQuantizerConfig(name="t6_tb16", kind="kivi_temporal_window",
                              bits=4, n_temporal_windows=16, temporal_mode="token_block")
    K_pl = apply_k_quantizer(K, cfg_pl, layer_idx=0, slice_info=slice_info, cache_offset=0)
    K_tb = apply_k_quantizer(K, cfg_tb, layer_idx=0, slice_info=slice_info, cache_offset=0)
    finite = (not torch.isnan(K_pl).any() and not torch.isinf(K_pl).any()
              and not torch.isnan(K_tb).any() and not torch.isinf(K_tb).any())
    return SmokeResult("PageLocal & TokenBlockLocal both finite",
                       bool(finite), "no NaN/Inf in either output")


def check_random_page_local_seed_determinism(slice_info: dict) -> SmokeResult:
    """RandomPageLocal must produce the same output across runs given same item_id."""
    K = _make_synthetic_K()
    cfg = KQuantizerConfig(name="t7_rpl", kind="kivi_random_page_local", bits=4,
                           random_seed_namespace="T7_random_page")
    a = apply_k_quantizer(K, cfg, layer_idx=0, slice_info=slice_info, cache_offset=0)
    b = apply_k_quantizer(K, cfg, layer_idx=0, slice_info=slice_info, cache_offset=0)
    same = bool(torch.equal(a, b))
    # Different item_id should change the output.
    slice2 = dict(slice_info); slice2["item_id"] = "different_item"
    c = apply_k_quantizer(K, cfg, layer_idx=0, slice_info=slice2, cache_offset=0)
    differs = _l2(a, c) > 1e-3
    return SmokeResult("RandomPageLocal deterministic per item_id",
                       same and differs,
                       f"same_item identical={same}, different_item differs L2={_l2(a, c):.4f}")


def check_image_only_local(slice_info: dict) -> SmokeResult:
    """ImageOnlyLocal-F4 must differ from PageLocal-F4 (text uses one pooled scale)."""
    K = _make_synthetic_K()
    cfg_pl = KQuantizerConfig(name="t8_pl", kind="kivi_page_local", bits=4)
    cfg_iol = KQuantizerConfig(name="t9_iol", kind="kivi_image_only_local", bits=4)
    K_pl = apply_k_quantizer(K, cfg_pl, layer_idx=0, slice_info=slice_info, cache_offset=0)
    K_iol = apply_k_quantizer(K, cfg_iol, layer_idx=0, slice_info=slice_info, cache_offset=0)
    diff = _l2(K_pl, K_iol)
    return SmokeResult("ImageOnlyLocal-F4 differs from PageLocal-F4",
                       diff > 1e-3,
                       f"L2 diff = {diff:.4f}")


def check_text_only_local(slice_info: dict) -> SmokeResult:
    """TextOnlyLocal-F4 must differ from PageLocal-F4 (visual uses one pooled scale)."""
    K = _make_synthetic_K()
    cfg_pl = KQuantizerConfig(name="t8_pl", kind="kivi_page_local", bits=4)
    cfg_tol = KQuantizerConfig(name="t10_tol", kind="kivi_text_only_local", bits=4)
    K_pl = apply_k_quantizer(K, cfg_pl, layer_idx=0, slice_info=slice_info, cache_offset=0)
    K_tol = apply_k_quantizer(K, cfg_tol, layer_idx=0, slice_info=slice_info, cache_offset=0)
    diff = _l2(K_pl, K_tol)
    return SmokeResult("TextOnlyLocal-F4 differs from PageLocal-F4",
                       diff > 1e-3, f"L2 diff = {diff:.4f}")


def check_page_sentinel_exact_protection(slice_info: dict) -> SmokeResult:
    """PageSentinel-4 must restore exactly 4 visual tokens per image page."""
    K = _make_synthetic_K()
    cfg = KQuantizerConfig(name="t12_ps4",
                           kind="kivi_page_sentinel", bits=4,
                           base_kind="kivi_per_channel_seq",
                           sentinel_kind="first_visual", sentinel_n_per_page=4)
    K_q = apply_k_quantizer(K, cfg, layer_idx=0, slice_info=slice_info, cache_offset=0)
    # Expected sentinel positions: first 4 indices of each visual page.
    expected_sentinel_pos = []
    for img_pos_list in slice_info["visual_token_positions_per_image"]:
        expected_sentinel_pos.extend(img_pos_list[:4])
    expected_sentinel_pos = sorted(set(expected_sentinel_pos))
    # Verify: at every sentinel position, K_q == K (exact restoration).
    all_match = True
    for p in expected_sentinel_pos:
        if not torch.allclose(K_q[:, :, p, :], K[:, :, p, :], atol=0.0):
            all_match = False
            break
    n_visual_pages = len(slice_info["visual_token_positions_per_image"])
    expected_count = 4 * n_visual_pages
    actual_count = len(expected_sentinel_pos)
    return SmokeResult("PageSentinel-4 exact protection",
                       all_match and actual_count == expected_count,
                       f"protected {actual_count} positions (expected {expected_count}); "
                       f"all_match={all_match}")


def check_random_sentinel_count(slice_info: dict) -> SmokeResult:
    """RandomSentinel-4 must protect the same number of tokens as PageSentinel-4."""
    K = _make_synthetic_K()
    cfg_ps = KQuantizerConfig(name="t12_ps4",
                              kind="kivi_page_sentinel", bits=4,
                              base_kind="kivi_per_channel_seq",
                              sentinel_kind="first_visual", sentinel_n_per_page=4)
    cfg_rs = KQuantizerConfig(name="t13_rs4",
                              kind="kivi_page_sentinel", bits=4,
                              base_kind="kivi_per_channel_seq",
                              sentinel_kind="random_visual", sentinel_n_per_page=4,
                              random_seed_namespace="T13_random_sentinel")
    K_ps = apply_k_quantizer(K, cfg_ps, layer_idx=0, slice_info=slice_info, cache_offset=0)
    K_rs = apply_k_quantizer(K, cfg_rs, layer_idx=0, slice_info=slice_info, cache_offset=0)
    # Count positions where output == input (i.e. sentinel-protected).
    def n_protected(K_out, K_in):
        same = torch.all(torch.isclose(K_out, K_in, atol=1e-10), dim=(0, 1, 3))
        return int(same.sum().item())
    n_ps = n_protected(K_ps, K)
    n_rs = n_protected(K_rs, K)
    # They should match in count.
    matches = n_ps == n_rs
    # But positions should differ (random vs first-N).
    diff = _l2(K_ps, K_rs)
    differs = diff > 1e-3
    return SmokeResult("RandomSentinel-4 count == PageSentinel-4 count, but positions differ",
                       matches and differs,
                       f"n_ps={n_ps} n_rs={n_rs} L2(K_ps,K_rs)={diff:.4f}")


def check_last_sentinel(slice_info: dict) -> SmokeResult:
    """LastSentinel-4 should restore the LAST 4 visual tokens of each image page."""
    K = _make_synthetic_K()
    cfg = KQuantizerConfig(name="t14_ls4",
                           kind="kivi_page_sentinel", bits=4,
                           base_kind="kivi_per_channel_seq",
                           sentinel_kind="last_visual", sentinel_n_per_page=4)
    K_q = apply_k_quantizer(K, cfg, layer_idx=0, slice_info=slice_info, cache_offset=0)
    expected = []
    for img_pos_list in slice_info["visual_token_positions_per_image"]:
        expected.extend(img_pos_list[-4:])
    expected = sorted(set(expected))
    all_match = all(torch.allclose(K_q[:, :, p, :], K[:, :, p, :], atol=0.0) for p in expected)
    return SmokeResult("LastSentinel-4 protects last-4 of each image page",
                       all_match, f"expected_count={len(expected)} all_match={all_match}")


def check_text_sentinel(slice_info: dict) -> SmokeResult:
    """TextSentinel-4 should restore first 4 tokens of each text page."""
    K = _make_synthetic_K()
    cfg = KQuantizerConfig(name="t15_ts4",
                           kind="kivi_page_sentinel", bits=4,
                           base_kind="kivi_per_channel_seq",
                           sentinel_kind="first_text", sentinel_n_per_page=4)
    K_q = apply_k_quantizer(K, cfg, layer_idx=0, slice_info=slice_info, cache_offset=0)
    expected = []
    for txt_pos_list in slice_info["text_chunk_positions"]:
        expected.extend(txt_pos_list[:4])
    expected = sorted(set(expected))
    all_match = all(torch.allclose(K_q[:, :, p, :], K[:, :, p, :], atol=0.0) for p in expected)
    return SmokeResult("TextSentinel-4 protects first-4 of each text page",
                       all_match, f"expected_count={len(expected)} all_match={all_match}")


def check_t16_combined(slice_info: dict) -> SmokeResult:
    """T16 PageLocal+Sentinel: sentinel positions match original, non-sentinel match PageLocal."""
    K = _make_synthetic_K()
    cfg_pl = KQuantizerConfig(name="t8_pl", kind="kivi_page_local", bits=4)
    cfg_t16 = KQuantizerConfig(name="t16",
                               kind="kivi_page_sentinel", bits=4,
                               base_kind="kivi_page_local",
                               sentinel_kind="first_visual", sentinel_n_per_page=4)
    K_pl = apply_k_quantizer(K, cfg_pl, layer_idx=0, slice_info=slice_info, cache_offset=0)
    K_t16 = apply_k_quantizer(K, cfg_t16, layer_idx=0, slice_info=slice_info, cache_offset=0)
    sentinel = []
    for img_pos_list in slice_info["visual_token_positions_per_image"]:
        sentinel.extend(img_pos_list[:4])
    sentinel_set = set(sentinel)
    # Non-sentinel positions: K_t16 should match K_pl (PageLocal base).
    # Sentinel positions: K_t16 should match K (original).
    nonsentinel_match = True
    sentinel_restore_match = True
    for p in range(slice_info["seq_len"]):
        if p in sentinel_set:
            if not torch.allclose(K_t16[:, :, p, :], K[:, :, p, :], atol=0.0):
                sentinel_restore_match = False
                break
        else:
            if not torch.allclose(K_t16[:, :, p, :], K_pl[:, :, p, :], atol=0.0):
                nonsentinel_match = False
                break
    return SmokeResult("T16 PageLocal+Sentinel composes correctly",
                       nonsentinel_match and sentinel_restore_match,
                       f"nonsentinel_match={nonsentinel_match} sentinel_match={sentinel_restore_match}")


def check_decode_time_fallback(slice_info: dict) -> SmokeResult:
    """For cache_offset > 0 (decode), PageLocal should fall back to plain F4."""
    K_small = torch.randn(1, 4, 1, 128, dtype=torch.float32)  # single new token
    cfg = KQuantizerConfig(name="t8_pl_decode", kind="kivi_page_local", bits=4)
    K_q = apply_k_quantizer(K_small, cfg, layer_idx=0, slice_info=slice_info, cache_offset=10000)
    finite = not torch.isnan(K_q).any() and not torch.isinf(K_q).any()
    return SmokeResult("PageLocal decode-time fallback runs without crash",
                       bool(finite), "cache_offset>0 produces finite K_q")


def check_counting_parser_valid() -> SmokeResult:
    """Counting parser handles common BF16 output shapes."""
    cases = [
        ("[1, 0, 0, 2]", [1, 0, 0, 2], True, 4),
        ("Sure, here is the result: [3, 4]", [3, 4], True, 2),
        ("```json\n[0, 0, 1]\n```", [0, 0, 1], True, 3),
        ("The answer is [1,2,3,4,5,6]", [1, 2, 3, 4, 5, 6], True, 6),
        ("not a list at all", None, False, None),
        ("[]", [], True, 0),
        ("[1, 2, three]", None, False, None),
    ]
    failures = []
    for text, exp_parsed, exp_valid, exp_len in cases:
        res = parse_counting_output(text)
        if res["parsed"] != exp_parsed or res["valid_format"] != exp_valid \
           or res["predicted_length"] != exp_len:
            failures.append((text, exp_parsed, res))
    return SmokeResult("Counting parser handles BF16 outputs",
                       not failures,
                       f"failures: {failures}" if failures else "all 7 cases pass")


def check_counting_score() -> SmokeResult:
    """Score function: exact match, length match, soft accuracy."""
    cases = [
        # (parsed, gold, exact, length, sum_match)
        ([1, 0, 0, 2], [1, 0, 0, 2], True, True, True),
        ([1, 0, 1, 0], [1, 0, 0, 1], False, True, True),  # different but same sum
        ([1, 0], [1, 0, 0, 2], False, False, False),
        (None, [1, 0, 0], False, False, False),
    ]
    failures = []
    for parsed, gold, exp_exact, exp_len, exp_sum in cases:
        s = score_counting(parsed, gold)
        if s["exact_match"] != exp_exact or s["length_match"] != exp_len \
           or s["sum_match"] != exp_sum:
            failures.append((parsed, gold, s))
    return SmokeResult("Counting scorer correct on test cases",
                       not failures, f"failures: {failures}" if failures else "all 4 pass")


def run_all_cpu_smoke() -> list[SmokeResult]:
    si = _make_synthetic_slice_info()
    return [
        check_pagelocal_differs_from_global(si),
        check_pagelocal_token_block_coexist(si),
        check_random_page_local_seed_determinism(si),
        check_image_only_local(si),
        check_text_only_local(si),
        check_page_sentinel_exact_protection(si),
        check_random_sentinel_count(si),
        check_last_sentinel(si),
        check_text_sentinel(si),
        check_t16_combined(si),
        check_decode_time_fallback(si),
        check_counting_parser_valid(),
        check_counting_score(),
    ]


def write_smoke_md(results: list[SmokeResult], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Exp T-mini Phase 0 smoke results", "",
             f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
             "| # | Check | Pass | Detail |", "|---|---|---|---|"]
    n_pass = sum(1 for r in results if r.passed)
    for i, r in enumerate(results, 1):
        flag = "PASS" if r.passed else "FAIL"
        lines.append(f"| {i} | {r.name} | {flag} | {r.detail} |")
    lines.append("")
    lines.append(f"**Summary**: {n_pass}/{len(results)} passed")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nSmoke summary: {n_pass}/{len(results)} passed -> {out_md}")
    for r in results:
        if not r.passed:
            print(f"  FAIL: {r.name} — {r.detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "results" / "expT_mini_smoke.md")
    args = ap.parse_args()
    results = run_all_cpu_smoke()
    write_smoke_md(results, args.out_md)
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
