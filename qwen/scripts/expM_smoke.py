"""Smoke test for Exp M (matched-budget controls).

Phase A: synthetic bits-accounting check for the 5 new M presets.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPTS_DIR.parent / "results"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _check_M_bits_accounting() -> tuple[bool, str]:
    from k_quantizers import build_f_conditions
    from expF_kquant_screen import _compute_three_bit_columns
    L, Hkv, D = 28, 4, 128
    rng = np.random.default_rng(0)
    calib = {
        "outlier_channel_idx_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "k_channel_energy": rng.random((L, Hkv, D)).astype(np.float32),
        "k_channel_energy_text": rng.random((L, Hkv, D)).astype(np.float32),
        "k_channel_energy_visual": rng.random((L, Hkv, D)).astype(np.float32),
        "k_abs_max": rng.random((L, Hkv, D)).astype(np.float32) + 1.0,
        "q_energy": rng.random((L, Hkv, D)).astype(np.float32),
        "q_energy_text": rng.random((L, Hkv, D)).astype(np.float32),
        "q_energy_visual": rng.random((L, Hkv, D)).astype(np.float32),
        "q_energy_pivot": rng.random((L, Hkv, D)).astype(np.float32),
        "outlier_idx_TT_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "outlier_idx_TV_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "outlier_idx_TT_TV_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "outlier_idx_BAL_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "outlier_idx_PIVOT_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "cell_risk_TT_TV": rng.random((L, Hkv)).astype(np.float32),
        "cell_risk_all": rng.random((L, Hkv)).astype(np.float32),
        "outlier_idx_RANDOM_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "outlier_idx_BAL_top1_per_block_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "outlier_idx_BAL_top2_per_block_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "outlier_idx_BAL_top3_per_block_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "outlier_idx_BAL_RANDOM_POS_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "outlier_idx_BAL_RANDOM_POS_3pb_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "outlier_idx_PIVOT_ERR_top16": rng.choice(D, size=(L, Hkv, 16)).astype(np.int32),
        "cell_risk_RANDOM": rng.random((L, Hkv)).astype(np.float32),
    }
    fc = {cfg.name: cfg for cfg in build_f_conditions(calib=calib)}
    spec = {
        "M5_Generic12_BF16side":         4.5625,
        "M6_Random12_BF16side":          4.5625,
        "M7_BalRandomPos3pb_BF16side":   4.5625,
        "M10_Bal3pb_INT8side":           4.1875,
        "M12_Pivot12_BF16side":          4.5625,
        # Sanity that existing presets still resolve to right bits:
        "K2_F9_BF16side":                4.750,
        "K3_F9_INT8side":                4.250,
        "K9_Bal3pb_BF16side":            4.5625,
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
        return False, "; ".join(bad)
    return True, f"all {len(spec)} M+anchor presets match avg_kv_bits spec to ±0.01"


def _check_BAL_RANDOM_POS_3pb_partition() -> tuple[bool, str]:
    from k_quantizers import _balanced_random_top_indices
    L, Hkv, D = 4, 2, 128
    idx = _balanced_random_top_indices(L, Hkv, D, n_per_block=3, n_blocks=4,
                                       n_top=16, seed=99)
    if idx.shape != (L, Hkv, 16):
        return False, f"shape mismatch {idx.shape}"
    block_size = D // 4
    bad = []
    for L_i in range(L):
        for h in range(Hkv):
            block_counts = [0]*4
            for j in range(12):
                c = int(idx[L_i, h, j])
                if 0 <= c < D:
                    block_counts[c // block_size] += 1
            if block_counts != [3, 3, 3, 3]:
                bad.append((L_i, h, block_counts))
    if bad:
        return False, f"3-per-block partition wrong; first issue: {bad[0]}"
    return True, "BAL_RANDOM_POS_3pb_top16 yields exactly 3 channels per fixed position block"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", type=Path, default=RESULTS_DIR / "expM_smoke.md")
    args = ap.parse_args()
    print(f"[expM_smoke] {_ts()} starting Phase A", flush=True)
    checks = [
        ("M_bits_accounting",            _check_M_bits_accounting),
        ("BAL_RANDOM_POS_3pb_partition", _check_BAL_RANDOM_POS_3pb_partition),
    ]
    results = []
    all_pass = True
    for name, fn in checks:
        try:
            ok, det = fn()
        except Exception as e:
            ok, det = False, f"{type(e).__name__}: {e}"
        all_pass = all_pass and ok
        results.append((name, ok, det))
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}: {det}", flush=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# expM_smoke results — {_ts()}", ""]
    lines.append(f"**Phase A:** {'PASS' if all_pass else 'FAIL'} "
                 f"({sum(1 for _,ok,_ in results if ok)}/{len(results)})")
    lines.append("")
    lines.append("| Check | Result | Detail |")
    lines.append("|---|---|---|")
    for n, ok, d in results:
        m = "PASS" if ok else "**FAIL**"
        lines.append(f"| {n} | {m} | {d} |")
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"[expM_smoke] wrote {args.out_md}")
    if not all_pass:
        sys.exit(2)
    print("[expM_smoke] PASS", flush=True)


if __name__ == "__main__":
    main()
