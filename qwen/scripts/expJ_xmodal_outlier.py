"""Experiment J — Cross-modal outlier-channel KV quantization.

Tests whether F9 outlier-channel protection can be improved by:
  (a) selecting protected channels via cross-modal score distortion
      (TT/TV/VT/VV), instead of generic K-channel magnitude
  (b) layer-adaptive budgets that concentrate protection on highest-risk
      (L, H_kv) cells
  (c) compressing the side-channel (INT8/INT6 instead of BF16)

Conditions (15 forwards, 128f only):
  Anchors:
    J0  BF16 128f
    J1  F4 KIVI per-channel-seq                    4.00 KV bits
    J2  F9 generic top-16 BF16 outliers            4.75 KV bits
    J3  F8 generic top-8 BF16 outliers             4.375 KV bits

  Cross-modal selection (top-8 BF16, 4.375 KV bits):
    J4  TT-score top-8
    J5  TV-score top-8
    J6  TT+TV combined top-8
    J7  Balanced top-2 from each of TT/TV/VT/VV (deduped)
    J8  Pivot top-8 (E[Q²] at answer-prefix end only)

  Layer-adaptive budgets:
    J9   TT+TV top-50% cells × 16 outliers/cell    4.375 KV bits
    J10  All-token top-50% cells × 16 outliers/cell 4.375 KV bits
    J11  TT+TV top-75% cells × 16 outliers/cell    4.56 KV bits

  Side-channel compression:
    J12  F9 16 outliers stored INT8                4.25 KV bits
    J13  F9 16 outliers stored INT6                4.125 KV bits
    J14  TT+TV top-16 outliers stored INT8         4.25 KV bits

Stage 1: n=64, fresh seed=2 split (16/bucket). Stage 3: deferred to next-day
launch after analyzing Stage 1.

Stage-3 promotion (pre-registered):
  Always: {J0, J1, J2, J3}
  Data-driven from Stage 1 verdict: any of {J4-J14} that reaches
  promote_n200 / paper_strong / pareto_winner.

Reused infrastructure:
  - expG_frame_scaling.run_stage_g / run_item_at_frames
  - expF_kquant_screen._run_condition_forward / _option_logprobs_and_pred /
    _answer_margin / load_calib / STAGE_TARGETS
  - data_longvideobench.{LVBItem, load_all_items, load_split, save_split,
    make_split, format_mcq_messages}
  - k_quantizers.build_f_conditions (now includes 11 J-suite presets)

Calibration:
  Uses qwen/calibration/expJ_kcalib_<model>_frames128.npz which contains:
    - F-suite arrays (back-compat)
    - per-modality K and pivot Q energies
    - 7 cross-modal outlier-index arrays (top-16 each)
    - 2 cell-risk arrays (cell_risk_TT_TV, cell_risk_all)

Usage:
  python expJ_xmodal_outlier.py --stage 1 --seed 2 \\
      --calib_npz ../calibration/expJ_kcalib_Qwen2.5-VL-7B-Instruct_frames128.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from data_longvideobench import (
    LVBItem,
    filter_items,
    load_all_items,
    load_split,
    make_split,
    save_split,
)
from k_quantizers import build_f_conditions

from expF_kquant_screen import (
    STAGE_TARGETS,
    load_calib,
)
from expG_frame_scaling import (
    relative_kv_memory,  # noqa: F401
    run_stage_g,
)


SCRIPTS_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPTS_DIR.parent / "calibration"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"


# ===================================================================
# J-suite condition specs (single 128f tier)
# ===================================================================


# (j_name, frames, f_cfg_name, mode)
FIXED_FRAME_CONDITIONS_J = [
    ("J0_BF16_128f",            128, "F0_BF16",                  "bf16"),
    ("J1_F4_128f",              128, "F4_KIVI_PerChannelSeq",    "v1_kcfg"),
    ("J2_F9_128f",              128, "F9_KIVI_Outlier16",        "v1_kcfg"),
    ("J3_F8_128f",              128, "F8_KIVI_Outlier8",         "v1_kcfg"),
    ("J4_Outlier8_TT_128f",     128, "J4_Outlier8_TT",           "v1_kcfg"),
    ("J5_Outlier8_TV_128f",     128, "J5_Outlier8_TV",           "v1_kcfg"),
    ("J6_Outlier8_TT_TV_128f",  128, "J6_Outlier8_TT_TV",        "v1_kcfg"),
    ("J7_Outlier8_BAL_128f",    128, "J7_Outlier8_BAL",          "v1_kcfg"),
    ("J8_Outlier8_PIVOT_128f",  128, "J8_Outlier8_PIVOT",        "v1_kcfg"),
    ("J9_LA_TT_TV_50pct_128f",  128, "J9_LA_TT_TV_50pct",        "v1_kcfg"),
    ("J10_LA_ALL_50pct_128f",   128, "J10_LA_ALL_50pct",         "v1_kcfg"),
    ("J11_LA_TT_TV_75pct_128f", 128, "J11_LA_TT_TV_75pct",       "v1_kcfg"),
    ("J12_F9_INT8side_128f",    128, "J12_F9_INT8side",          "v1_kcfg"),
    ("J13_F9_INT6side_128f",    128, "J13_F9_INT6side",          "v1_kcfg"),
    ("J14_TT_TV_INT8side_128f", 128, "J14_TT_TV_INT8side",       "v1_kcfg"),
    # Stage-3 controls (random + error-weighted pivot).
    ("J15_Outlier8_RANDOM_128f",     128, "J15_Outlier8_RANDOM",     "v1_kcfg"),
    ("J16_LA_RANDOM_50pct_128f",     128, "J16_LA_RANDOM_50pct",     "v1_kcfg"),
    ("J17_Outlier8_PIVOT_ERR_128f",  128, "J17_Outlier8_PIVOT_ERR",  "v1_kcfg"),
]


CONDITIONS_NEEDING_CALIB_J = {
    "J2_F9_128f", "J3_F8_128f",
    "J4_Outlier8_TT_128f", "J5_Outlier8_TV_128f", "J6_Outlier8_TT_TV_128f",
    "J7_Outlier8_BAL_128f", "J8_Outlier8_PIVOT_128f",
    "J9_LA_TT_TV_50pct_128f", "J10_LA_ALL_50pct_128f", "J11_LA_TT_TV_75pct_128f",
    "J12_F9_INT8side_128f", "J13_F9_INT6side_128f", "J14_TT_TV_INT8side_128f",
    "J15_Outlier8_RANDOM_128f", "J16_LA_RANDOM_50pct_128f",
    "J17_Outlier8_PIVOT_ERR_128f",
}


# Exp U on LongVideoBench-128f — residual channel oracle/policy screen.
# Same condition semantics as expQ_driver.u_conditions_residual_screen but
# adapted to the J/G frame-scaling runner (single 128f tier, no PageLayout).
FIXED_FRAME_CONDITIONS_U = [
    ("U0_BF16_128f",                128, "F0_BF16",                    "bf16"),
    ("U1_F4_128f",                  128, "F4_KIVI_PerChannelSeq",      "v1_kcfg"),
    ("U2_F9_128f",                  128, "F9_KIVI_Outlier16",          "v1_kcfg"),
    ("U3_S4_INT7_128f",             128, "SL_Outlier16_INT7side",      "v1_kcfg"),
    ("U4_S4_GEN8_INT7_128f",        128, "U4_S4_plus_GEN8_INT7",       "v1_kcfg"),
    ("U5_S4_RND8_INT7_128f",        128, "U5_S4_plus_RND8_INT7",       "v1_kcfg"),
    ("U6_S4_TT8_INT7_128f",         128, "U6_S4_plus_TT8_INT7",        "v1_kcfg"),
    ("U7_S4_TV8_INT7_128f",         128, "U7_S4_plus_TV8_INT7",        "v1_kcfg"),
    ("U8_S4_VT8_INT7_128f",         128, "U8_S4_plus_VT8_INT7",        "v1_kcfg"),
    ("U9_S4_VV8_INT7_128f",         128, "U9_S4_plus_VV8_INT7",        "v1_kcfg"),
    ("U10_S4_BAL8_INT7_128f",       128, "U10_S4_plus_BAL8_INT7",      "v1_kcfg"),
    ("U11_S4_MMNIAH8_INT7_128f",    128, "U11_S4_plus_MMNIAH8_INT7",   "v1_kcfg"),
    ("U12_S4_LVB8_INT7_128f",       128, "U12_S4_plus_LVB8_INT7",      "v1_kcfg"),
    ("U13_S4_ALL16_INT7_128f",      128, "U13_S4_plus_ALL16_INT7",     "v1_kcfg"),
]

CONDITIONS_NEEDING_CALIB_U = {
    name for name, _, _, _ in FIXED_FRAME_CONDITIONS_U
    if name not in ("U0_BF16_128f", "U1_F4_128f")
}


# Pre-registered Stage-3 anchors (always run regardless of Stage-1 verdict).
# J15/J16/J17 are pre-registered controls added at Stage 3.
ANCHORS_ALWAYS_PROMOTED = {
    "J0_BF16_128f", "J1_F4_128f", "J2_F9_128f", "J3_F8_128f",
    "J15_Outlier8_RANDOM_128f", "J16_LA_RANDOM_50pct_128f",
    "J17_Outlier8_PIVOT_ERR_128f",
}
VARIANTS_DATA_DRIVEN = {
    "J4_Outlier8_TT_128f", "J5_Outlier8_TV_128f", "J6_Outlier8_TT_TV_128f",
    "J7_Outlier8_BAL_128f", "J8_Outlier8_PIVOT_128f",
    "J9_LA_TT_TV_50pct_128f", "J10_LA_ALL_50pct_128f", "J11_LA_TT_TV_75pct_128f",
    "J12_F9_INT8side_128f", "J13_F9_INT6side_128f", "J14_TT_TV_INT8side_128f",
}


def build_j_conditions(calib: Optional[dict]) -> list[dict]:
    """Build J-suite condition spec dicts in the shape expected by
    expG_frame_scaling.run_item_at_frames: {name, mode, cfg, frames, f_cfg_name}.
    """
    return _build_conditions_from_tuples(FIXED_FRAME_CONDITIONS_J, calib)


def build_u_conditions_lvb(calib: Optional[dict]) -> list[dict]:
    """Build U-suite condition spec dicts for LongVideoBench-128f.

    Same shape as build_j_conditions, sharing the calib dict so the F9 anchor
    and S4 / U-prime configs see both the precomputed top-16 outliers and the
    EXTRA index arrays merged from the sibling _expU_extras.npz.
    """
    return _build_conditions_from_tuples(FIXED_FRAME_CONDITIONS_U, calib)


def _build_conditions_from_tuples(specs, calib):
    fc = build_f_conditions(calib=calib)
    by_name = {cfg.name: cfg for cfg in fc}
    out: list[dict] = []
    for name, frames, f_cfg_name, mode in specs:
        if f_cfg_name not in by_name:
            raise KeyError(
                f"[expJ/U] f_cfg_name={f_cfg_name!r} not found in build_f_conditions; "
                f"available names: {sorted(by_name)}"
            )
        cfg = by_name[f_cfg_name]
        out.append({
            "name": name,
            "mode": mode,
            "cfg": cfg,
            "frames": frames,
            "f_cfg_name": f_cfg_name,
        })
    return out


# ===================================================================
# Fresh balanced split (seed=2, not used by F/G/H/I)
# ===================================================================


def j_split_path(stage: int, seed: int) -> Path:
    n = sum(STAGE_TARGETS[stage].values())
    return CALIBRATION_DIR / f"split_seed{seed}_n{n}.json"


def ensure_j_split(stage: int, seed: int = 2) -> Path:
    sp = j_split_path(stage, seed)
    if sp.exists():
        return sp
    items = load_all_items()
    targets = STAGE_TARGETS[stage]
    split = make_split(items, seed=seed, targets=targets, cal_fraction=0.0)
    save_split(split, sp)
    print(f"[expJ] wrote stage {stage} split (n_eval={len(split['eval'])}) -> {sp}",
          flush=True)
    return sp


# ===================================================================
# Post-write: rename experiment/phase tags + BF16 join on J0
# ===================================================================


def _rewrite_experiment_tag(out_jsonl: Path, exp: str, phase: str) -> None:
    """The reused run_stage_g/run_item_at_frames stamps each row with
    experiment="G". Rewrite to the given experiment/phase tag.
    """
    if not out_jsonl.exists():
        return
    rows = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l.strip()]
    for r in rows:
        r["experiment"] = exp
        if "phase" in r:
            r["phase"] = phase
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out_jsonl)


def rewrite_experiment_tag_j(out_jsonl: Path, stage: int) -> None:
    _rewrite_experiment_tag(out_jsonl, exp="J", phase=f"J{stage}")


def backfill_bf16_join_j(out_jsonl: Path,
                         bf16_cond_name: str = "J0_BF16_128f") -> None:
    """Backfill bf16_pred / bf16_correct from J0_BF16_128f rows."""
    if not out_jsonl.exists():
        return
    rows = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l.strip()]
    bf16_pred: dict[str, int] = {}
    for r in rows:
        if r.get("condition") == bf16_cond_name and "pred_choice" in r:
            bf16_pred[r["item_id"]] = int(r["pred_choice"])
    for r in rows:
        if r.get("skipped") or "correct_choice" not in r or r.get("error"):
            continue
        if r.get("condition") == bf16_cond_name:
            r["bf16_pred"] = int(r.get("pred_choice", -1))
            r["bf16_correct"] = bool(r["bf16_pred"] == int(r["correct_choice"]))
            continue
        bp = bf16_pred.get(r["item_id"])
        if bp is None:
            continue
        r["bf16_pred"] = int(bp)
        r["bf16_correct"] = bool(bp == int(r["correct_choice"]))
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out_jsonl)
    print(f"[expJ] backfilled BF16 join on {bf16_cond_name} in {out_jsonl} ({len(rows)} rows)")


# ===================================================================
# Stage-3 promotion gating
# ===================================================================


def filter_conditions_stage3(conditions: list[dict],
                             promotion_file: Optional[Path]) -> list[dict]:
    """For Stage 3: keep ANCHORS_ALWAYS_PROMOTED + variants in promotion_file."""
    promoted_variants: set[str] = set()
    if promotion_file is not None and promotion_file.exists():
        try:
            data = json.loads(promotion_file.read_text())
            if isinstance(data, list):
                promoted_variants = set(data)
            elif isinstance(data, dict) and "promoted" in data:
                promoted_variants = set(data["promoted"])
        except Exception as e:
            print(f"[expJ] WARN: could not parse {promotion_file}: {e}", flush=True)
    keep: list[dict] = []
    for c in conditions:
        n = c["name"]
        if n in ANCHORS_ALWAYS_PROMOTED:
            keep.append(c)
        elif n in VARIANTS_DATA_DRIVEN and n in promoted_variants:
            keep.append(c)
        elif n not in ANCHORS_ALWAYS_PROMOTED and n not in VARIANTS_DATA_DRIVEN:
            keep.append(c)
    return keep


# ===================================================================
# Driver
# ===================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--stage", type=int, choices=[1, 3], default=1)
    ap.add_argument("--seed", type=int, default=2,
                    help="Fresh split seed (default 2; F=0, I=1).")
    ap.add_argument("--split_file", type=Path, default=None)
    ap.add_argument("--calib_npz", type=Path, default=None,
                    help="Exp J cross-modal calibration NPZ at frames=128.")
    ap.add_argument("--calib_json", type=Path, default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min_free_gb_256", type=int, default=70,
                    help="Unused at 128f-only; kept for run_stage_g signature.")
    ap.add_argument("--stage3_promotion_file", type=Path, default=None)
    ap.add_argument("--exp-u", action="store_true",
                    help="Run Exp U1 (residual channel oracle/policy screen) on "
                         "LongVideoBench-128f instead of the J-suite. Requires "
                         "the sibling _expU_extras.npz alongside the calib.")
    ap.add_argument("--extras-npz", type=Path, default=None,
                    help="Override the auto-derived expU_extras NPZ path for --exp-u.")
    args = ap.parse_args()

    if args.out is None:
        if args.exp_u:
            args.out = RESULTS_DIR / f"expU_lvb_stage{args.stage}_seed{args.seed}.jsonl"
        else:
            args.out = RESULTS_DIR / f"expJ_xmodal_stage{args.stage}_seed{args.seed}.jsonl"
    if args.split_file is None:
        args.split_file = ensure_j_split(args.stage, seed=args.seed)
    if args.calib_npz is None:
        model_short = args.model.split("/")[-1]
        cand = CALIBRATION_DIR / f"expJ_kcalib_{model_short}_frames128.npz"
        if cand.exists():
            args.calib_npz = cand
    if args.calib_json is None and args.calib_npz is not None:
        cj = args.calib_npz.with_suffix(".json")
        if cj.exists():
            args.calib_json = cj
    if args.stage3_promotion_file is None:
        cand = RESULTS_DIR / "expJ_promote_stage1.json"
        if cand.exists():
            args.stage3_promotion_file = cand

    items_all = load_all_items()
    split = load_split(args.split_file)
    eval_items = filter_items(items_all, split["eval"])
    if args.limit:
        eval_items = eval_items[: args.limit]
    print(f"[expJ] stage={args.stage} seed={args.seed} eval_items={len(eval_items)} "
          f"split_file={args.split_file} calib_npz={args.calib_npz}", flush=True)

    calib, _ = load_calib(args.calib_npz, args.calib_json)
    calibration_id = (args.calib_npz.stem if args.calib_npz is not None else None)

    # Validate that the J-suite calibration arrays are present.
    if calib is None:
        raise SystemExit(
            "[expJ] calibration NPZ not loaded; J/U-suite needs cross-modal "
            "outlier indices. Run expJ_calibrate.py first."
        )

    if args.exp_u:
        # Exp U merges the residual extras NPZ alongside the calib. The
        # required keys are the EXTRA index arrays + k_channel_energy.
        extras_path = args.extras_npz
        if extras_path is None:
            extras_path = args.calib_npz.with_name(
                args.calib_npz.stem + "_expU_extras.npz")
        if not extras_path.exists():
            raise SystemExit(
                f"[expU] required extras NPZ not found: {extras_path}. "
                f"Run `python expU_compute_extras.py --all` first."
            )
        xarr = np.load(extras_path)
        for k in xarr.files:
            calib[k] = xarr[k]
        print(f"[expU] merged extras from {extras_path.name} "
              f"({len(xarr.files)} new keys)", flush=True)
        required_u_keys = {
            "k_channel_energy",
            "outlier_idx_EXTRA_GEN_8",
            "outlier_idx_EXTRA_RND_8",
            "outlier_idx_EXTRA_TT_8", "outlier_idx_EXTRA_TV_8",
            "outlier_idx_EXTRA_VT_8", "outlier_idx_EXTRA_VV_8",
            "outlier_idx_EXTRA_BAL_8",
            "outlier_idx_EXTRA_MMNIAH_PRIOR_8",
            "outlier_idx_EXTRA_LVB_PRIOR_8",
            "outlier_idx_EXTRA_ALL_16",
        }
        missing = required_u_keys - set(calib)
        if missing:
            raise SystemExit(
                f"[expU] calib+extras missing required arrays: {sorted(missing)}. "
                f"Available keys: {sorted(calib)}"
            )
    else:
        required_xmodal_keys = {
            "outlier_idx_TT_top16", "outlier_idx_TV_top16", "outlier_idx_TT_TV_top16",
            "outlier_idx_BAL_top16", "outlier_idx_PIVOT_top16",
            "cell_risk_TT_TV", "cell_risk_all",
        }
        missing = required_xmodal_keys - set(calib)
        if missing:
            raise SystemExit(
                f"[expJ] calibration NPZ missing cross-modal arrays: {sorted(missing)}. "
                f"Available keys: {sorted(calib)}. "
                f"Re-run expJ_calibrate.py."
            )

    # Inject Stage-3 control arrays (J15 random, J16 random-cell, J17
    # error-weighted pivot). Computed on the fly with fixed seeds so the
    # randomness is reproducible and doesn't require recalibration.
    # Skipped for Exp U (different control arrays — already in extras NPZ).
    if args.exp_u:
        conditions = build_u_conditions_lvb(calib=calib)
        if args.conditions:
            wanted = set(args.conditions)
            conditions = [c for c in conditions if c["name"] in wanted]
            print(f"[expU] filtered conditions: {[c['name'] for c in conditions]}",
                  flush=True)
        if not args.append and args.out.exists():
            args.out.unlink()
        from run_inference import load_model
        model, processor = load_model(args.model, dtype="bfloat16",
                                      attn_impl="sdpa", device_map="auto")
        lm = getattr(model, "language_model", None) or model.model.language_model
        layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
        num_layers = len(layers)
        num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
        print(f"[expU] model loaded; num_layers={num_layers} num_kv_heads={num_kv_heads}",
              flush=True)
        run_stage_g(model, processor, eval_items,
                    num_layers=num_layers, num_kv_heads=num_kv_heads,
                    conditions=conditions, stage=args.stage,
                    calibration_id=calibration_id,
                    out_jsonl=args.out, progress_every=args.progress_every,
                    min_free_gb_for_256=args.min_free_gb_256)
        # Rewrite the experiment/phase tag from G to U.
        _rewrite_experiment_tag(args.out, exp="U", phase=f"U{args.stage}")
        backfill_bf16_join_j(args.out, bf16_cond_name="U0_BF16_128f")
        return

    L, H_kv, D = calib["k_channel_energy"].shape
    n_top = int(calib["outlier_idx_TT_top16"].shape[-1])
    rng_chan = np.random.default_rng(42)
    # J15: random top-16 outlier indices per (L, H_kv).
    random_idx = np.empty((L, H_kv, n_top), dtype=np.int32)
    for L_i in range(L):
        for h in range(H_kv):
            random_idx[L_i, h] = rng_chan.choice(D, size=n_top, replace=False).astype(np.int32)
    calib["outlier_idx_RANDOM_top16"] = random_idx
    # J16: random per-(L, H_kv) cell-risk score, used by layer-adaptive resolver.
    rng_cell = np.random.default_rng(43)
    calib["cell_risk_RANDOM"] = rng_cell.standard_normal((L, H_kv)).astype(np.float32)
    # J17: error-weighted pivot — score(l, h, d) ∝ q_pivot · k_max² (uniform
    # INT4 quantization noise variance ∝ max²/588; constant drops out of argsort).
    q_pivot = np.asarray(calib["q_energy_pivot"], dtype=np.float64)  # [L, H_kv, D]
    k_max = np.asarray(calib["k_abs_max"], dtype=np.float64)
    score_pivot_err = q_pivot * (k_max ** 2)
    pivot_err_idx = np.argsort(score_pivot_err, axis=-1)[..., -n_top:][..., ::-1].copy().astype(np.int32)
    calib["outlier_idx_PIVOT_ERR_top16"] = pivot_err_idx
    print(f"[expJ] injected Stage-3 control arrays: outlier_idx_RANDOM_top16 "
          f"(seed=42), cell_risk_RANDOM (seed=43), outlier_idx_PIVOT_ERR_top16 "
          f"(q_pivot · k_max²)", flush=True)

    conditions = build_j_conditions(calib=calib)

    if args.stage == 3 and args.stage3_promotion_file is not None:
        before = [c["name"] for c in conditions]
        conditions = filter_conditions_stage3(conditions, args.stage3_promotion_file)
        after = [c["name"] for c in conditions]
        print(f"[expJ] stage3 promotion gating: kept {after} (dropped "
              f"{sorted(set(before) - set(after))})", flush=True)
    elif args.stage == 3 and args.stage3_promotion_file is None:
        print("[expJ] stage3 promotion file not found; running full slate (warning).",
              flush=True)

    if args.conditions:
        wanted = set(args.conditions)
        conditions = [c for c in conditions if c["name"] in wanted]
        print(f"[expJ] filtered conditions: {[c['name'] for c in conditions]}", flush=True)

    if not args.append and args.out.exists():
        args.out.unlink()

    from run_inference import load_model
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    print(f"[expJ] model loaded; num_layers={num_layers} num_kv_heads={num_kv_heads}",
          flush=True)

    run_stage_g(model, processor, eval_items,
                num_layers=num_layers, num_kv_heads=num_kv_heads,
                conditions=conditions, stage=args.stage,
                calibration_id=calibration_id,
                out_jsonl=args.out, progress_every=args.progress_every,
                min_free_gb_for_256=args.min_free_gb_256)
    rewrite_experiment_tag_j(args.out, args.stage)
    backfill_bf16_join_j(args.out, bf16_cond_name="J0_BF16_128f")


if __name__ == "__main__":
    main()
