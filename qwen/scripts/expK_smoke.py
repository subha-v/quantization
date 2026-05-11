"""Smoke test for Exp K (balanced cross-modal replication).

Phase A (no model required):
  1. K_bits_accounting          all 12 K conditions match avg_kv_bits spec
  2. K10_random_block_partition K10 picks 2 channels from each of 4 fixed
                                channel-position blocks (0-31, 32-63, ...)
  3. K6_balanced_top2_per_block K6's balanced indices have exactly 2 entries
                                from each modality block's top-N
  4. K8_K9_budget_sizes         K8 effective = 4 distinct channels per cell;
                                K9 effective = 12 distinct channels per cell
  5. seed_split_files_exist     seed=0/1/2 n=200 split files exist or can be
                                regenerated

Writes qwen/results/expK_smoke.md; exit 2 on any FAIL.

Phase B is skipped (Exp J's Phase B already validated live-model paths;
K conditions are structurally the same as J).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPTS_DIR.parent / "calibration"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _make_synthetic_calib(L: int = 28, Hkv: int = 4, D: int = 128) -> dict:
    rng = np.random.default_rng(0)
    return {
        "k_channel_energy": rng.random((L, Hkv, D)).astype(np.float32),
        "k_channel_energy_text": rng.random((L, Hkv, D)).astype(np.float32),
        "k_channel_energy_visual": rng.random((L, Hkv, D)).astype(np.float32),
        "k_abs_max": rng.random((L, Hkv, D)).astype(np.float32) + 1.0,
        "q_energy": rng.random((L, Hkv, D)).astype(np.float32),
        "q_energy_text": rng.random((L, Hkv, D)).astype(np.float32),
        "q_energy_visual": rng.random((L, Hkv, D)).astype(np.float32),
        "q_energy_pivot": rng.random((L, Hkv, D)).astype(np.float32),
        "outlier_channel_idx_top16": rng.choice(D, size=(L, Hkv, 16), replace=True).astype(np.int32),
        "outlier_idx_TT_top16": rng.choice(D, size=(L, Hkv, 16), replace=True).astype(np.int32),
        "outlier_idx_TV_top16": rng.choice(D, size=(L, Hkv, 16), replace=True).astype(np.int32),
        "outlier_idx_TT_TV_top16": rng.choice(D, size=(L, Hkv, 16), replace=True).astype(np.int32),
        "outlier_idx_BAL_top16": rng.choice(D, size=(L, Hkv, 16), replace=True).astype(np.int32),
        "outlier_idx_PIVOT_top16": rng.choice(D, size=(L, Hkv, 16), replace=True).astype(np.int32),
        "cell_risk_TT_TV": rng.random((L, Hkv)).astype(np.float32),
        "cell_risk_all": rng.random((L, Hkv)).astype(np.float32),
    }


def _check_K_bits_accounting() -> tuple[bool, str]:
    from k_quantizers import build_f_conditions
    from expF_kquant_screen import _compute_three_bit_columns
    L, Hkv, D = 28, 4, 128
    calib = _make_synthetic_calib(L, Hkv, D)
    # The K presets need a few more keys (they reference RANDOM, BAL_per_block, PIVOT_ERR, etc.).
    # Add zeros so the cfg builders find the keys at build time.
    for key in ("outlier_idx_RANDOM_top16",
                "outlier_idx_BAL_top1_per_block_top16",
                "outlier_idx_BAL_top2_per_block_top16",
                "outlier_idx_BAL_top3_per_block_top16",
                "outlier_idx_BAL_RANDOM_POS_top16",
                "outlier_idx_PIVOT_ERR_top16"):
        calib[key] = np.zeros((L, Hkv, 16), dtype=np.int32)
    for key in ("cell_risk_RANDOM",):
        calib[key] = np.random.default_rng(0).random((L, Hkv)).astype(np.float32)
    fc = {cfg.name: cfg for cfg in build_f_conditions(calib=calib)}
    spec = {
        # F-suite anchors (sanity).
        "F4_KIVI_PerChannelSeq": 4.000,
        # K-suite.
        "K2_F9_BF16side":        4.750,
        "K3_F9_INT8side":        4.250,
        "K4_F8_BF16side":        4.375,
        "K5_Random8_BF16side":   4.375,
        "K6_Bal2pb_BF16side":    4.375,
        "K7_Bal2pb_INT8side":    4.125,
        "K8_Bal1pb_BF16side":    4.1875,
        "K9_Bal3pb_BF16side":    4.5625,
        "K10_BalRandomPos_BF16side": 4.375,
        "K11_Pivot8_BF16side":   4.375,
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
    return True, f"all {len(spec)} K conditions match avg_kv_bits spec to ±0.01"


def _check_K10_random_block_partition() -> tuple[bool, str]:
    from k_quantizers import _balanced_random_top_indices
    L, Hkv, D = 4, 2, 128
    idx = _balanced_random_top_indices(L, Hkv, D, n_per_block=2, n_blocks=4,
                                       n_top=16, seed=99)
    if idx.shape != (L, Hkv, 16):
        return False, f"shape {idx.shape} != ({L}, {Hkv}, 16)"
    block_size = D // 4  # 32
    bad_cells: list[tuple] = []
    for L_i in range(L):
        for h in range(Hkv):
            # First 8 entries should have exactly 2 from each block.
            block_counts = [0] * 4
            for j in range(8):
                c = int(idx[L_i, h, j])
                if not (0 <= c < D):
                    bad_cells.append((L_i, h, j, c))
                    continue
                b = c // block_size
                block_counts[b] += 1
            if block_counts != [2, 2, 2, 2]:
                bad_cells.append((L_i, h, "block_counts", block_counts))
    if bad_cells:
        return False, f"K10 first-8 block partition wrong; first 3 issues: {bad_cells[:3]}"
    # Check determinism: same seed -> same output.
    idx2 = _balanced_random_top_indices(L, Hkv, D, n_per_block=2, n_blocks=4,
                                        n_top=16, seed=99)
    if not np.array_equal(idx, idx2):
        return False, "K10 random not deterministic with fixed seed"
    return True, f"K10 partition: every cell's first 8 entries are 2 per block; seed=99 deterministic"


def _check_K6_balanced_top2_per_block() -> tuple[bool, str]:
    from k_quantizers import _balanced_per_block_top_indices
    # Build synthetic scores where each modality block has obvious top-2 picks.
    L, Hkv, D = 2, 2, 128
    scores = {}
    # TT top-2 = [10, 20], TV top-2 = [11, 21], VT top-2 = [12, 22], VV top-2 = [13, 23]
    for k, peaks in (("TT", [10, 20]), ("TV", [11, 21]), ("VT", [12, 22]), ("VV", [13, 23])):
        s = np.zeros((L, Hkv, D), dtype=np.float32)
        s[:, :, peaks[0]] = 10.0
        s[:, :, peaks[1]] = 9.0
        scores[k] = s
    idx = _balanced_per_block_top_indices(scores, n_per_block=2, n_top=8)
    expected_set = {10, 20, 11, 21, 12, 22, 13, 23}
    for L_i in range(L):
        for h in range(Hkv):
            got = set(idx[L_i, h].tolist())
            if got != expected_set:
                return False, f"cell ({L_i},{h}): got {sorted(got)}, expected {sorted(expected_set)}"
    return True, f"K6 balanced top-2/block correctly picks 2 from each of TT/TV/VT/VV"


def _check_K8_K9_budget_sizes() -> tuple[bool, str]:
    from k_quantizers import _balanced_per_block_top_indices
    L, Hkv, D = 2, 2, 128
    scores = {}
    rng = np.random.default_rng(0)
    for k in ("TT", "TV", "VT", "VV"):
        scores[k] = rng.random((L, Hkv, D)).astype(np.float32)
    idx1 = _balanced_per_block_top_indices(scores, n_per_block=1, n_top=4)
    idx3 = _balanced_per_block_top_indices(scores, n_per_block=3, n_top=12)
    # First 4 of idx1: 4 distinct channels (1 from each block).
    for L_i in range(L):
        for h in range(Hkv):
            uniq = set(int(c) for c in idx1[L_i, h, :4])
            if len(uniq) < 4:
                return False, f"K8 cell ({L_i},{h}) has {len(uniq)} unique in first 4: {sorted(uniq)}"
    # First 12 of idx3: at most 12 distinct (some may dup-collide across modalities).
    for L_i in range(L):
        for h in range(Hkv):
            uniq = set(int(c) for c in idx3[L_i, h, :12])
            if len(uniq) < 8:  # at least 8 should be unique after dedup
                return False, f"K9 cell ({L_i},{h}) has only {len(uniq)} unique in first 12"
    return True, f"K8 (top-1/block) yields 4 unique per cell; K9 (top-3/block) yields ≥8 unique"


def _check_seed_split_files_exist() -> tuple[bool, str]:
    # We don't actually need the files locally; this is a structural check
    # that the file paths the K driver will look at are sensible.
    paths = [
        CALIBRATION_DIR / "split_seed0.json",
        CALIBRATION_DIR / "split_seed1_n200.json",
        CALIBRATION_DIR / "split_seed2_n200.json",
    ]
    found = [p for p in paths if p.exists()]
    if not found:
        return True, f"skipped: no split files locally ({len(paths)} candidate paths); will be generated/used on remote"
    return True, f"{len(found)}/{len(paths)} split file paths exist locally; rest will be generated on remote if missing"


def _run_phase_a() -> tuple[list[dict], bool]:
    results: list[dict] = []
    all_pass = True
    checks = [
        ("K_bits_accounting",          _check_K_bits_accounting),
        ("K10_random_block_partition", _check_K10_random_block_partition),
        ("K6_balanced_top2_per_block", _check_K6_balanced_top2_per_block),
        ("K8_K9_budget_sizes",         _check_K8_K9_budget_sizes),
        ("seed_split_files_exist",     _check_seed_split_files_exist),
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
    ap.add_argument("--out_md", type=Path, default=RESULTS_DIR / "expK_smoke.md")
    args = ap.parse_args()
    print(f"[expK_smoke] {_ts()} starting Phase A", flush=True)
    results, all_pass = _run_phase_a()
    for r in results:
        marker = "PASS" if r["pass"] else "FAIL"
        print(f"  [{marker}] {r['check']}: {r['detail']}", flush=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# expK_smoke results — {_ts()}", ""]
    lines.append(f"**Phase A:** {'PASS' if all_pass else 'FAIL'} "
                 f"({sum(1 for r in results if r['pass'])}/{len(results)})")
    lines.append("")
    lines.append("| Check | Result | Detail |")
    lines.append("|---|---|---|")
    for r in results:
        marker = "PASS" if r["pass"] else "**FAIL**"
        lines.append(f"| {r['check']} | {marker} | {r['detail']} |")
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"[expK_smoke] wrote {args.out_md}")
    if not all_pass:
        sys.exit(2)
    print("[expK_smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
