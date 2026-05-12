"""Experiment M — Matched-budget sidecode controls.

After Exp L produced K9 (Balanced top-3/block, 12 BF16 outliers, 4.56 KV
bits) as the only paired-significant variant on seed=1 (χ²=8.07 vs K6,
matches BF16=0.615), Exp M asks the obvious follow-up:

  Is K9 good because it's CROSS-MODAL BALANCED, or just because it
  protects 12 channels instead of 8?

K9 was only compared to F9 (16 channels) and F8 (8 channels). The
missing controls at 12 channels are:
  - generic top-12 by magnitude
  - random top-12
  - balanced-random top-3/block (12 channels by structural balance only)
  - pivot top-12
  - K9 itself (replication)
  - K9 INT8 sidecode (can the budget winner be cheaper?)

13 conditions on seed=0 eval-200, the canonical F-suite split. Reuses
the existing expJ calibration NPZ (built on seed=0 cal-100, already
disjoint from seed=0 eval).

Conditions:
  M0  BF16 128f
  M1  F4 KIVI                                    4.00 KV bits
  M2  F9 generic top-16 BF16 sidecode            4.75 KV bits  (robust anchor)
  M3  F9 generic top-16 INT8 sidecode            4.25 KV bits
  M4  generic top-8 BF16                         4.375 KV bits (old J7 baseline)
  M5  generic top-12 BF16                        4.56 KV bits  (MATCHED K9 budget)
  M6  random top-12 BF16                         4.56 KV bits  (MATCHED random)
  M7  balanced-random top-3/block                4.56 KV bits  (BALANCE without scoring)
  M8  balanced TT/TV/VT/VV top-2 (= J7/K6)       4.375 KV bits (old J7)
  M9  balanced TT/TV/VT/VV top-3 (= K9)          4.56 KV bits  (current candidate)
  M10 balanced top-3 INT8 sidecode               4.19 KV bits  (cheaper K9)
  M11 pivot top-8 (= J8/K11)                     4.375 KV bits (current trending)
  M12 pivot top-12                               4.56 KV bits  (MATCHED pivot)

Decisive comparisons:
  M9 vs M5: does balanced top-3 beat generic top-12?
  M9 vs M6: does it beat random top-12?
  M9 vs M7: does cross-modal scoring beat balance-without-scoring?
  M9 vs M2: does top-3 match or beat F9 (16) at fewer bits?
  M10 vs M9: can balanced top-3 use INT8 sidecode?
  M12 vs M5: is pivot the simpler robust signal?

Usage:
  python expM_matched_budget.py --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from data_longvideobench import (
    LVBItem, filter_items, load_all_items, load_split, make_split, save_split,
)
from k_quantizers import (
    build_f_conditions,
    _balanced_random_top_indices,
)
from expF_kquant_screen import STAGE_TARGETS, load_calib
from expG_frame_scaling import relative_kv_memory, run_stage_g  # noqa: F401
from expK_balanced_replication import (
    resolve_split_for_seed,
    inject_k_calib_arrays,
    rewrite_experiment_tag_k as _rewrite_tag,
    backfill_bf16_join_k as _backfill_bf16,
)


SCRIPTS_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPTS_DIR.parent / "calibration"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"


# (m_name, frames, f_cfg_name, mode)
FIXED_FRAME_CONDITIONS_M = [
    ("M0_BF16_128f",                 128, "F0_BF16",                  "bf16"),
    ("M1_F4_128f",                   128, "F4_KIVI_PerChannelSeq",    "v1_kcfg"),
    ("M2_F9_BF16side_128f",          128, "K2_F9_BF16side",           "v1_kcfg"),
    ("M3_F9_INT8side_128f",          128, "K3_F9_INT8side",           "v1_kcfg"),
    ("M4_Generic8_BF16side_128f",    128, "K4_F8_BF16side",           "v1_kcfg"),
    ("M5_Generic12_BF16side_128f",   128, "M5_Generic12_BF16side",    "v1_kcfg"),
    ("M6_Random12_BF16side_128f",    128, "M6_Random12_BF16side",     "v1_kcfg"),
    ("M7_BalRandomPos3pb_BF16side_128f", 128, "M7_BalRandomPos3pb_BF16side", "v1_kcfg"),
    ("M8_Bal2pb_BF16side_128f",      128, "K6_Bal2pb_BF16side",       "v1_kcfg"),
    ("M9_Bal3pb_BF16side_128f",      128, "K9_Bal3pb_BF16side",       "v1_kcfg"),
    ("M10_Bal3pb_INT8side_128f",     128, "M10_Bal3pb_INT8side",      "v1_kcfg"),
    ("M11_Pivot8_BF16side_128f",     128, "K11_Pivot8_BF16side",      "v1_kcfg"),
    ("M12_Pivot12_BF16side_128f",    128, "M12_Pivot12_BF16side",     "v1_kcfg"),
]


def build_m_conditions(calib: Optional[dict]) -> list[dict]:
    fc = build_f_conditions(calib=calib)
    by_name = {cfg.name: cfg for cfg in fc}
    out: list[dict] = []
    for m_name, frames, f_cfg_name, mode in FIXED_FRAME_CONDITIONS_M:
        if f_cfg_name not in by_name:
            raise KeyError(
                f"[expM] f_cfg_name={f_cfg_name!r} not found; available: {sorted(by_name)}"
            )
        cfg = by_name[f_cfg_name]
        out.append({"name": m_name, "mode": mode, "cfg": cfg, "frames": frames,
                    "f_cfg_name": f_cfg_name})
    return out


def inject_m_calib_arrays(calib: dict) -> None:
    """Inject M-specific control arrays:
      - All K-suite arrays (via inject_k_calib_arrays)
      - outlier_idx_BAL_RANDOM_POS_3pb_top16: 3-per-block random by channel
        position partition (12 channels effective, padded to 16).
    """
    # First inject K-suite arrays (BAL_top1/2/3_per_block, BAL_RANDOM_POS_top16
    # at n_per_block=2, RANDOM_top16, PIVOT, etc.).
    inject_k_calib_arrays(calib)

    L, H_kv, D = calib["k_channel_energy"].shape
    n_top = int(calib["outlier_idx_TT_top16"].shape[-1])

    # Inject 3-per-block random version for M7.
    if "outlier_idx_BAL_RANDOM_POS_3pb_top16" not in calib:
        calib["outlier_idx_BAL_RANDOM_POS_3pb_top16"] = _balanced_random_top_indices(
            num_layers=L, num_kv_heads=H_kv, head_dim=D,
            n_per_block=3, n_blocks=4, n_top=n_top, seed=99)

    # Diagnostic overlap: balanced top-3/block first-12 vs generic top-12.
    bal3 = calib["outlier_idx_BAL_top3_per_block_top16"]
    generic = calib["outlier_channel_idx_top16"]
    overlap = []
    for L_i in range(L):
        for h in range(H_kv):
            s_bal = set(bal3[L_i, h, :12].tolist())
            s_gen = set(generic[L_i, h, :12].tolist())
            overlap.append(len(s_bal & s_gen))
    print(f"[expM] Bal3pb first-12 overlap with generic top-12: "
          f"mean={np.mean(overlap):.2f}/12 (over {L*H_kv} cells)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seed", type=int, default=0,
                    help="Eval split seed (default 0 = canonical F-suite split).")
    ap.add_argument("--split_file", type=Path, default=None)
    ap.add_argument("--calib_npz", type=Path, default=None,
                    help="Default expJ_kcalib_*_frames128.npz (seed=0 cal-100).")
    ap.add_argument("--calib_json", type=Path, default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min_free_gb_256", type=int, default=70)
    args = ap.parse_args()

    if args.out is None:
        args.out = RESULTS_DIR / f"expM_matched_stage3_seed{args.seed}.jsonl"
    if args.split_file is None:
        args.split_file = resolve_split_for_seed(args.seed)
    if args.calib_npz is None:
        model_short = args.model.split("/")[-1]
        cand = CALIBRATION_DIR / f"expJ_kcalib_{model_short}_frames128.npz"
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
    print(f"[expM] seed={args.seed} eval_items={len(eval_items)} "
          f"split_file={args.split_file} calib_npz={args.calib_npz}", flush=True)

    calib, _ = load_calib(args.calib_npz, args.calib_json)
    calibration_id = (args.calib_npz.stem if args.calib_npz is not None else None)
    if calib is None:
        raise SystemExit(
            "[expM] calibration NPZ not loaded; needs the Exp J cross-modal NPZ."
        )

    required_for_M = {
        "k_channel_energy", "k_channel_energy_text", "k_channel_energy_visual",
        "q_energy_text", "q_energy_visual", "q_energy_pivot",
        "outlier_idx_TT_top16", "outlier_idx_PIVOT_top16",
        "outlier_channel_idx_top16",
    }
    missing = required_for_M - set(calib)
    if missing:
        raise SystemExit(
            f"[expM] calibration NPZ missing required arrays: {sorted(missing)}. "
            f"Re-run expJ_calibrate.py."
        )

    inject_m_calib_arrays(calib)
    print(f"[expM] injected M-suite control arrays into calib", flush=True)

    conditions = build_m_conditions(calib=calib)

    if args.conditions:
        wanted = set(args.conditions)
        conditions = [c for c in conditions if c["name"] in wanted]
        print(f"[expM] filtered conditions: {[c['name'] for c in conditions]}", flush=True)

    if not args.append and args.out.exists():
        args.out.unlink()

    from run_inference import load_model
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    print(f"[expM] model loaded; num_layers={num_layers} num_kv_heads={num_kv_heads}",
          flush=True)

    run_stage_g(model, processor, eval_items,
                num_layers=num_layers, num_kv_heads=num_kv_heads,
                conditions=conditions, stage=3,
                calibration_id=calibration_id,
                out_jsonl=args.out, progress_every=args.progress_every,
                min_free_gb_for_256=args.min_free_gb_256)
    _rewrite_tag(args.out, stage=3)
    # Update experiment tag from "K" to "M".
    if args.out.exists():
        rows = [json.loads(l) for l in args.out.read_text().splitlines() if l.strip()]
        for r in rows:
            r["experiment"] = "M"
            if "phase" in r:
                r["phase"] = "M3"
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tmp.replace(args.out)
    _backfill_bf16(args.out, bf16_cond_name="M0_BF16_128f")


if __name__ == "__main__":
    main()
