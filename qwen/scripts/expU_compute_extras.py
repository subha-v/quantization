"""Exp U — Pre-compute residual extra-N channel-index arrays per calib NPZ.

For Experiment U1 (residual channel oracle/policy screen) we need to compose
S4's top-16 outlier set with N additional "extra" channels selected by one
of several policies, with the residual property: every extra channel is
NOT already in S4's top-16. This script reads each calibration NPZ
(retrieval-image / reasoning-image / counting-image / LVB-128f) and writes a
sibling NPZ ``expU_extras_<slice>.npz`` with the precomputed index arrays.

Policies (per (L, H_kv) cell):

  EXTRA_GEN_8         top-8 by raw k_channel_energy not in S4-top-16.
  EXTRA_RND_8         random-8 (seed=2026) from D \\ S4-top-16.
  EXTRA_TT_8          top-8 by D_TT = q_energy_text · k_channel_energy_text.
  EXTRA_TV_8          top-8 by D_TV = q_energy_text · k_channel_energy_visual.
  EXTRA_VT_8          top-8 by D_VT = q_energy_visual · k_channel_energy_text.
  EXTRA_VV_8          top-8 by D_VV = q_energy_visual · k_channel_energy_visual.
  EXTRA_BAL_8         balanced 2/block: 2 from each of TT/TV/VT/VV (residual,
                      composite-padded).
  EXTRA_MMNIAH_PRIOR_8  composite (TT+TV+VT+VV) score averaged across the
                      MM-NIAH calibs (retrieval+reasoning+counting), top-8
                      not in THIS calib's S4-top-16. The averaging happens
                      *across calibs* on the score arrays; the residual
                      filter uses THIS calib's S4.
  EXTRA_LVB_PRIOR_8   composite score from the LVB calib (single source),
                      top-8 not in this calib's S4.
  EXTRA_ALL16         top-16 by composite (TT+TV+VT+VV) score not in S4
                      (used for U13 — extra-16 condition).

The "S4 anchor set" for any calib is its precomputed
``outlier_channel_idx_top16`` (the generic top-16 by k_channel_energy that
expP_calibrate / expJ_calibrate emit).

This is a CPU-only one-shot helper. Run via:

    python expU_compute_extras.py --calib <path> --out <path>

or for the full Exp U slate:

    python expU_compute_extras.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPTS_DIR.parent / "calibration"


EXTRA_KEYS_8 = (
    "outlier_idx_EXTRA_GEN_8",
    "outlier_idx_EXTRA_RND_8",
    "outlier_idx_EXTRA_TT_8",
    "outlier_idx_EXTRA_TV_8",
    "outlier_idx_EXTRA_VT_8",
    "outlier_idx_EXTRA_VV_8",
    "outlier_idx_EXTRA_BAL_8",
    "outlier_idx_EXTRA_MMNIAH_PRIOR_8",
    "outlier_idx_EXTRA_LVB_PRIOR_8",
)
EXTRA_KEY_16 = "outlier_idx_EXTRA_ALL_16"

# Exp V additions:
# - Three deterministically-seeded RND arrays for robust paired-against-random test.
# - BAL budget ladder (per-block = 1/3/4 giving 4/12/16 residual channels).
EXTRA_RND_SEED_KEYS = {
    0: "outlier_idx_EXTRA_RND_8_s0",
    1: "outlier_idx_EXTRA_RND_8_s1",
    2: "outlier_idx_EXTRA_RND_8_s2",
}
EXTRA_BAL_LADDER = {
    4:  ("outlier_idx_EXTRA_BAL_4",  1),   # per_block=1, total=4
    12: ("outlier_idx_EXTRA_BAL_12", 3),
    16: ("outlier_idx_EXTRA_BAL_16", 4),
}


# ----------------------------- core helpers -----------------------------


def _s4_set(calib: dict, L: int, H_kv: int) -> np.ndarray:
    """Return S4 top-16 indices [L, H_kv, 16] — the *generic* top-16 by
    k_channel_energy. Prefers a precomputed ``outlier_channel_idx_top16``;
    falls back to online argsort if missing.
    """
    key = "outlier_channel_idx_top16"
    if key in calib:
        idx = np.asarray(calib[key], dtype=np.int32)
        if idx.shape[-1] >= 16:
            return idx[..., :16].copy()
    energy = np.asarray(calib["k_channel_energy"], dtype=np.float64)
    return np.argsort(energy, axis=-1)[..., -16:][..., ::-1].astype(np.int32).copy()


def _residual_topn(score: np.ndarray, banned: np.ndarray, n: int) -> np.ndarray:
    """Pick the top-n channels by score that are NOT in banned, per (L, H_kv).

    score: [L, H_kv, D] float
    banned: [L, H_kv, K] int — banned channel ids (the S4 top-16 set per cell)
    Returns [L, H_kv, n] int32.
    """
    L, H_kv, D = score.shape
    out = np.full((L, H_kv, n), -1, dtype=np.int32)
    order = np.argsort(score, axis=-1)[..., ::-1]  # [L, H_kv, D] descending
    for l in range(L):
        for h in range(H_kv):
            bset = set(int(x) for x in banned[l, h].tolist())
            picked = []
            for c in order[l, h].tolist():
                if c in bset:
                    continue
                picked.append(int(c))
                if len(picked) >= n:
                    break
            if len(picked) < n:
                raise RuntimeError(
                    f"_residual_topn: cell (L={l}, H={h}) only found "
                    f"{len(picked)} residual channels, need {n}. D={D}, "
                    f"|banned|={len(bset)}"
                )
            out[l, h] = np.array(picked, dtype=np.int32)
    return out


def _residual_balanced_8(block_scores: dict, banned: np.ndarray) -> np.ndarray:
    """Balanced 2/block residual pick. Takes per_block=2 from each of
    TT/TV/VT/VV that are not in banned and not already picked, then pads
    with composite (TT+TV+VT+VV) descending order.
    """
    keys = ("TT", "TV", "VT", "VV")
    L, H_kv, D = block_scores[keys[0]].shape
    composite = sum(block_scores[k] for k in keys)
    comp_order = np.argsort(composite, axis=-1)[..., ::-1]
    block_orders = {k: np.argsort(block_scores[k], axis=-1)[..., ::-1] for k in keys}
    out = np.full((L, H_kv, 8), -1, dtype=np.int32)
    for l in range(L):
        for h in range(H_kv):
            bset = set(int(x) for x in banned[l, h].tolist())
            seen: set[int] = set()
            picked: list[int] = []
            for k in keys:
                got = 0
                for c in block_orders[k][l, h].tolist():
                    if got >= 2:
                        break
                    c = int(c)
                    if c in bset or c in seen:
                        continue
                    seen.add(c)
                    picked.append(c)
                    got += 1
            # Pad with composite.
            for c in comp_order[l, h].tolist():
                if len(picked) >= 8:
                    break
                c = int(c)
                if c in bset or c in seen:
                    continue
                seen.add(c)
                picked.append(c)
            if len(picked) < 8:
                raise RuntimeError(
                    f"_residual_balanced_8: cell (L={l}, H={h}) only found "
                    f"{len(picked)} residual channels (need 8)."
                )
            out[l, h] = np.array(picked[:8], dtype=np.int32)
    return out


def _residual_random_n(L: int, H_kv: int, D: int, banned: np.ndarray,
                       n: int = 8, seed: int = 2026) -> np.ndarray:
    """Per (L, H_kv), pick n random channel ids uniformly from D \\ banned.
    Seeded deterministically for reproducibility across runs.
    """
    rng = np.random.default_rng(seed)
    out = np.full((L, H_kv, n), -1, dtype=np.int32)
    all_ids = np.arange(D, dtype=np.int32)
    for l in range(L):
        for h in range(H_kv):
            bset = set(int(x) for x in banned[l, h].tolist())
            pool = np.array([c for c in all_ids if int(c) not in bset], dtype=np.int32)
            if pool.size < n:
                raise RuntimeError(
                    f"_residual_random_n: cell (L={l}, H={h}) pool size {pool.size} "
                    f"< {n}."
                )
            sel = rng.choice(pool, size=n, replace=False)
            out[l, h] = sel.astype(np.int32)
    return out


# Backwards-compat alias for code outside this file.
_residual_random_8 = _residual_random_n


def _residual_balanced_n(block_scores: dict, banned: np.ndarray,
                         n_total: int, per_block: int) -> np.ndarray:
    """Balanced per_block-from-each residual pick. Takes top-`per_block` from each
    of TT/TV/VT/VV that are not in banned and not already picked, then pads with
    composite (TT+TV+VT+VV) descending order until n_total channels are chosen.

    With per_block * 4 >= n_total, padding is rarely needed; with
    per_block * 4 < n_total, composite padding kicks in.
    """
    keys = ("TT", "TV", "VT", "VV")
    L, H_kv, D = block_scores[keys[0]].shape
    composite = sum(block_scores[k] for k in keys)
    comp_order = np.argsort(composite, axis=-1)[..., ::-1]
    block_orders = {k: np.argsort(block_scores[k], axis=-1)[..., ::-1] for k in keys}
    out = np.full((L, H_kv, n_total), -1, dtype=np.int32)
    for l in range(L):
        for h in range(H_kv):
            bset = set(int(x) for x in banned[l, h].tolist())
            seen: set[int] = set()
            picked: list[int] = []
            for k in keys:
                got = 0
                for c in block_orders[k][l, h].tolist():
                    if got >= per_block or len(picked) >= n_total:
                        break
                    c = int(c)
                    if c in bset or c in seen:
                        continue
                    seen.add(c)
                    picked.append(c)
                    got += 1
            # Pad with composite descending.
            for c in comp_order[l, h].tolist():
                if len(picked) >= n_total:
                    break
                c = int(c)
                if c in bset or c in seen:
                    continue
                seen.add(c)
                picked.append(c)
            if len(picked) < n_total:
                raise RuntimeError(
                    f"_residual_balanced_n: cell (L={l}, H={h}) only found "
                    f"{len(picked)} residual channels (need {n_total})."
                )
            out[l, h] = np.array(picked[:n_total], dtype=np.int32)
    return out


def _residual_balanced_8(block_scores, banned):
    # Backwards-compat alias (per_block=2 for n=8).
    return _residual_balanced_n(block_scores, banned, n_total=8, per_block=2)


# ----------------------------- block scoring -----------------------------


def _block_scores(calib: dict) -> dict[str, np.ndarray]:
    """Compute D_TT/D_TV/D_VT/D_VV at full depth D=128 from raw energies.

    Falls back gracefully if a calib was generated without the modality-split
    arrays (raises an explicit error rather than silently using something else).
    """
    required = ("q_energy_text", "q_energy_visual",
                "k_channel_energy_text", "k_channel_energy_visual")
    missing = [k for k in required if k not in calib]
    if missing:
        raise RuntimeError(
            f"_block_scores: calib missing modality-split arrays: {missing}. "
            f"Available keys: {sorted(calib.keys())}"
        )
    qet = np.clip(np.asarray(calib["q_energy_text"], dtype=np.float64), 1e-12, None)
    qev = np.clip(np.asarray(calib["q_energy_visual"], dtype=np.float64), 1e-12, None)
    ke_t = np.clip(np.asarray(calib["k_channel_energy_text"], dtype=np.float64),
                   1e-12, None)
    ke_v = np.clip(np.asarray(calib["k_channel_energy_visual"], dtype=np.float64),
                   1e-12, None)
    return {
        "TT": qet * ke_t,
        "TV": qet * ke_v,
        "VT": qev * ke_t,
        "VV": qev * ke_v,
    }


def _composite_score(block: dict[str, np.ndarray]) -> np.ndarray:
    return block["TT"] + block["TV"] + block["VT"] + block["VV"]


# ----------------------------- prior aggregation -----------------------------


def _stack_composite_scores(calib_paths: list[Path]) -> np.ndarray:
    """Load each calib NPZ and return the *average* composite block score
    array across all of them. Requires the modality-split arrays in each NPZ.
    """
    accum: Optional[np.ndarray] = None
    n = 0
    for p in calib_paths:
        if not p.exists():
            print(f"[expU] WARN: prior source NPZ missing: {p}", flush=True)
            continue
        arr = np.load(p)
        calib = {k: arr[k] for k in arr.files}
        try:
            comp = _composite_score(_block_scores(calib))
        except RuntimeError as e:
            print(f"[expU] WARN: cannot use {p.name} as prior source ({e})", flush=True)
            continue
        if accum is None:
            accum = comp.astype(np.float64)
        else:
            accum = accum + comp.astype(np.float64)
        n += 1
        del arr, calib
    if accum is None or n == 0:
        return None
    return (accum / float(n)).astype(np.float64)


# ----------------------------- per-calib pipeline -----------------------------


def compute_extras(
    calib_path: Path,
    mmniah_prior: Optional[np.ndarray],
    lvb_prior: Optional[np.ndarray],
) -> dict[str, np.ndarray]:
    """Return all EXTRA index arrays for one calib NPZ."""
    arr = np.load(calib_path)
    calib = {k: arr[k] for k in arr.files}
    L, H_kv, D = calib["k_channel_energy"].shape
    print(f"[expU] {calib_path.name}: L={L} H_kv={H_kv} D={D}", flush=True)
    s4 = _s4_set(calib, L, H_kv)            # [L, H_kv, 16]
    block = _block_scores(calib)
    composite = _composite_score(block)
    energy_full = np.asarray(calib["k_channel_energy"], dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    # GEN = top-8 by k_channel_energy not in S4 (residual to S4 — these are
    # channels 17..24 by generic ranking).
    out["outlier_idx_EXTRA_GEN_8"] = _residual_topn(energy_full, s4, 8)
    # RND = random-8 from D \ S4 (seed=2026 keeps U1 back-compat).
    out["outlier_idx_EXTRA_RND_8"] = _residual_random_n(L, H_kv, D, s4, n=8, seed=2026)
    # Exp V: 3 deterministically-seeded RND arrays for robust paired-vs-random.
    for s, key in EXTRA_RND_SEED_KEYS.items():
        out[key] = _residual_random_n(L, H_kv, D, s4, n=8, seed=s)
    # Per-modality-block top-8 residuals.
    for k in ("TT", "TV", "VT", "VV"):
        out[f"outlier_idx_EXTRA_{k}_8"] = _residual_topn(block[k], s4, 8)
    # Balanced 2/block (n=8) — U1 baseline.
    out["outlier_idx_EXTRA_BAL_8"] = _residual_balanced_8(block, s4)
    # Exp V BAL ladder: 1/block (n=4), 3/block (n=12), 4/block (n=16).
    for n_total, (key, per_block) in EXTRA_BAL_LADDER.items():
        out[key] = _residual_balanced_n(block, s4, n_total=n_total, per_block=per_block)
    # Composite top-16 (used for U13 ALL-16 extra at 4.375 KV bits).
    out[EXTRA_KEY_16] = _residual_topn(composite, s4, 16)
    # Cross-dataset priors. The prior arrays are averaged composite scores
    # across the source dataset's calibs; residual is against THIS calib's S4.
    if mmniah_prior is not None:
        if mmniah_prior.shape != composite.shape:
            raise RuntimeError(
                f"mmniah_prior shape {mmniah_prior.shape} != composite shape "
                f"{composite.shape} — model/head_dim mismatch?"
            )
        out["outlier_idx_EXTRA_MMNIAH_PRIOR_8"] = _residual_topn(mmniah_prior, s4, 8)
    else:
        # Fall back to this calib's own composite (degenerate prior).
        out["outlier_idx_EXTRA_MMNIAH_PRIOR_8"] = _residual_topn(composite, s4, 8)
    if lvb_prior is not None:
        if lvb_prior.shape != composite.shape:
            raise RuntimeError(
                f"lvb_prior shape {lvb_prior.shape} != composite shape "
                f"{composite.shape}"
            )
        out["outlier_idx_EXTRA_LVB_PRIOR_8"] = _residual_topn(lvb_prior, s4, 8)
    else:
        out["outlier_idx_EXTRA_LVB_PRIOR_8"] = _residual_topn(composite, s4, 8)
    # Sanity: all 9 extra-8 arrays should be entirely outside S4.
    _verify_residual(out, s4)
    return out


def _verify_residual(extras: dict[str, np.ndarray], s4: np.ndarray) -> None:
    L, H_kv, _ = s4.shape
    for key, idx in extras.items():
        for l in range(L):
            for h in range(H_kv):
                s4_set = set(int(x) for x in s4[l, h].tolist())
                ex_set = set(int(x) for x in idx[l, h].tolist())
                inter = s4_set & ex_set
                if inter:
                    raise RuntimeError(
                        f"residual invariant violated: {key} cell (L={l}, H={h}) "
                        f"overlaps S4 on {sorted(inter)}"
                    )


# ----------------------------- self-test -----------------------------


def _self_test(extras: dict[str, np.ndarray]) -> None:
    """Print diagnostic stats to confirm the policies look distinct."""
    keys_to_compare = ("outlier_idx_EXTRA_TT_8",
                       "outlier_idx_EXTRA_TV_8",
                       "outlier_idx_EXTRA_VT_8",
                       "outlier_idx_EXTRA_VV_8")
    present = [k for k in keys_to_compare if k in extras]
    if len(present) < 2:
        return
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a = extras[present[i]]
            b = extras[present[j]]
            L, H_kv, _ = a.shape
            diff_cells = 0
            total = L * H_kv
            for l in range(L):
                for h in range(H_kv):
                    sa = set(int(x) for x in a[l, h].tolist())
                    sb = set(int(x) for x in b[l, h].tolist())
                    if sa != sb:
                        diff_cells += 1
            pct = 100.0 * diff_cells / total
            print(f"  diff: {present[i]} vs {present[j]}: "
                  f"{diff_cells}/{total} cells ({pct:.1f}%)", flush=True)


# ----------------------------- CLI -----------------------------


def _default_paths(model_short: str) -> dict[str, Path]:
    """Standard calibration NPZ paths used by Exp Q/T-mini/J for Qwen2.5-VL-7B."""
    return {
        "retrieval-image":
            CALIBRATION_DIR / f"expP_mmniah_kcalib_{model_short}_seed0.npz",
        "reasoning-image":
            CALIBRATION_DIR / f"expP_mmniah_kcalib_{model_short}_reasoning-image_seed0.npz",
        "counting-image":
            CALIBRATION_DIR / f"expP_mmniah_kcalib_{model_short}_counting-image_seed0.npz",
        "lvb-128f":
            CALIBRATION_DIR / f"expJ_kcalib_{model_short}_frames128.npz",
    }


def _extras_path(calib_path: Path) -> Path:
    return calib_path.with_name(calib_path.stem + "_expU_extras.npz")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", default="Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--calib", type=Path, default=None,
                    help="Single calib NPZ to process. Mutually exclusive with --all.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output NPZ. Default: sibling _expU_extras.npz.")
    ap.add_argument("--all", action="store_true",
                    help="Process all standard MM-NIAH + LVB calibs and compute "
                         "cross-dataset priors. Skips any NPZ that doesn't exist.")
    ap.add_argument("--mmniah-prior-sources", nargs="+", type=Path, default=None,
                    help="Override MM-NIAH prior source list (default: 3 MM-NIAH calibs).")
    ap.add_argument("--lvb-prior-source", type=Path, default=None,
                    help="Override LVB prior source NPZ (default: expJ_kcalib_*_frames128.npz).")
    ap.add_argument("--verify-only", action="store_true",
                    help="Recompute and verify residual invariant on an existing extras NPZ.")
    args = ap.parse_args()

    paths = _default_paths(args.model_short)

    # Prior source defaults.
    if args.mmniah_prior_sources is None:
        args.mmniah_prior_sources = [
            paths["retrieval-image"], paths["reasoning-image"], paths["counting-image"],
        ]
    if args.lvb_prior_source is None:
        args.lvb_prior_source = paths["lvb-128f"]

    # Compute the priors once across the source NPZs (if their modality-split
    # arrays are present).
    print("[expU] computing MM-NIAH prior (avg composite block score) from "
          f"{len(args.mmniah_prior_sources)} source(s)", flush=True)
    mm_prior = _stack_composite_scores(args.mmniah_prior_sources)
    if mm_prior is not None:
        print(f"  -> MM-NIAH prior shape={mm_prior.shape}", flush=True)
    else:
        print("  -> MM-NIAH prior unavailable (will fall back to per-calib composite)",
              flush=True)
    print(f"[expU] computing LVB prior from {args.lvb_prior_source}", flush=True)
    lvb_prior = _stack_composite_scores([args.lvb_prior_source])
    if lvb_prior is not None:
        print(f"  -> LVB prior shape={lvb_prior.shape}", flush=True)
    else:
        print("  -> LVB prior unavailable (will fall back to per-calib composite)",
              flush=True)

    if args.all:
        targets = list(paths.values())
    elif args.calib is not None:
        targets = [args.calib]
    else:
        ap.error("must pass --calib or --all")
        return 2

    ok = 0
    skip = 0
    for cp in targets:
        if not cp.exists():
            print(f"[expU] SKIP {cp.name} (not found)", flush=True)
            skip += 1
            continue
        # Decide which prior shape matches this calib. If MM-NIAH prior shape
        # mismatches (e.g., different head_dim layout), we'll surface a clear
        # error rather than silently pad.
        try:
            extras = compute_extras(cp, mmniah_prior=mm_prior, lvb_prior=lvb_prior)
        except RuntimeError as e:
            print(f"[expU] FAIL on {cp.name}: {e}", flush=True)
            return 3
        out_path = args.out if (args.out is not None and len(targets) == 1) else _extras_path(cp)
        np.savez_compressed(out_path, **extras)
        print(f"[expU] wrote {out_path} ({len(extras)} arrays)", flush=True)
        _self_test(extras)
        ok += 1

    print(f"[expU] DONE: wrote {ok} extras NPZs; skipped {skip}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
