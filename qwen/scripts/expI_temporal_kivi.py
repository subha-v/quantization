"""Experiment I — Temporal-KIVI mechanism screen.

Tests whether H6's win at 128f comes from visual temporal locality (vs
modality split alone, vs more scale groups, vs VidKV V, vs outlier protection)
and whether TempWin4 + outlier-8 closes the gap to F9 at 256f.

Conditions (15 forwards + 2 post-process):
  128f tier
    I0  BF16              4.00x rel mem (16 KV bits)
    I1  F4 KIVI           1.00x mem ref (4.00 KV bits)
    I2  F9 outlier-16     1.19x rel mem (4.75 KV bits)
    I3  H6 TempWin2       1.00x rel mem (current proposed method)
    I4  F5 text/visual    1.00x rel mem (modality split, no temporal)
    I5  TokenBlock4       1.00x rel mem (4 modality-blind blocks @128f)
    I6  TempWin4          1.00x rel mem (4 visual windows @128f)
    I7  TempWin2 + VidKVV 1.00x rel mem (V per-channel-seq instead of per-head_dim)
    I8  TempWin2 + Out8   1.09x rel mem (4.375 KV bits)
  256f tier
    I9  F4                2.00x rel mem
    I10 F9 outlier-16     2.38x rel mem
    I11 TempWin4          2.00x rel mem (= H3)
    I12 TokenBlock6       2.00x rel mem (6 modality-blind blocks; matches I11 segment count)
    I13 TempWin4 + Out8   2.19x rel mem
    I14 TempWin4 + VidKVV 2.00x rel mem
  Post-process (no new forwards)
    I15 Duration-Hybrid (mid -> F9, else -> H6)         expI_duration_hybrid.py
    I16 Random-Hybrid    (matched-rate random control)  expI_duration_hybrid.py

Stage-3 promotion (pre-registered):
  Always run: {I0, I1, I2, I3, I4, I5, I9, I10, I11}
  Data-driven: up to 2 winners from {I6, I7, I8, I12, I13, I14}, plus
  {I15, I16} if I15 beats H6 at Stage 1. Determined by expI_analyze.py;
  written to expI_promote_stage1.json and read by --stage3_promotion_file.

Reused infrastructure (verbatim):
  - expG_frame_scaling.run_stage_g / run_item_at_frames / relative_kv_memory
  - expF_kquant_screen._run_condition_forward / _option_logprobs_and_pred /
    _answer_margin / _compute_three_bit_columns / load_calib / STAGE_TARGETS
  - data_longvideobench.{LVBItem, load_all_items, load_split, save_split,
    make_split, format_mcq_messages}
  - k_quantizers.build_f_conditions (now includes 5 new Exp I presets)

Fresh balanced split:
  seed=1 (vs seed=0 used by F/G/H). 50/50/50/50 per duration bucket. Stage-1
  n=64 ⊂ Stage-3 n=200 by construction.

Calibration: reuse F-suite NPZ at frames=64 (frame-count-independent post-RoPE
outlier indices). For I8/I13, the n_outliers=8 indices are sliced from the
top-16 (verified by expI_smoke.py check 5).

Usage:
  python expI_temporal_kivi.py --stage 1 --seed 1 \\
      --calib_npz qwen/calibration/expF_kcalib_Qwen2.5-VL-7B-Instruct_frames64.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

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
    relative_kv_memory,  # noqa: F401  (re-exported for analyzer convenience)
    run_stage_g,
)


SCRIPTS_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPTS_DIR.parent / "calibration"
RESULTS_DIR = SCRIPTS_DIR.parent / "results"


# ===================================================================
# I-suite condition specs
# ===================================================================


# (i_name, frames, f_cfg_name, mode)
FIXED_FRAME_CONDITIONS_I = [
    # 128f tier
    ("I0_BF16_128f",                  128, "F0_BF16",                    "bf16"),
    ("I1_F4_128f",                    128, "F4_KIVI_PerChannelSeq",      "v1_kcfg"),
    ("I2_F9_128f",                    128, "F9_KIVI_Outlier16",          "v1_kcfg"),
    ("I3_TempWin2_128f",              128, "H6_KIVI_TempWin2",           "v1_kcfg_with_slice"),
    ("I4_TextVisualSplit_128f",       128, "F5_KIVI_TextVisualSplit",    "v1_kcfg_with_slice"),
    ("I5_TokenBlock4_128f",           128, "H5_KIVI_TokenBlock4",        "v1_kcfg_with_slice"),
    ("I6_TempWin4_128f",              128, "H3_KIVI_TempWin4",           "v1_kcfg_with_slice"),
    ("I7_TempWin2_VidKVV_128f",       128, "I_TempWin2_VidKVV",          "v1_kcfg_with_slice"),
    ("I8_TempWin2_Outlier8_128f",     128, "I_TempWin2_Outlier8",        "v1_kcfg_with_slice"),
    # 256f tier
    ("I9_F4_256f",                    256, "F4_KIVI_PerChannelSeq",      "v1_kcfg"),
    ("I10_F9_256f",                   256, "F9_KIVI_Outlier16",          "v1_kcfg"),
    ("I11_TempWin4_256f",             256, "H3_KIVI_TempWin4",           "v1_kcfg_with_slice"),
    ("I12_TokenBlock6_256f",          256, "I_TokenBlock6",              "v1_kcfg_with_slice"),
    ("I13_TempWin4_Outlier8_256f",    256, "I_TempWin4_Outlier8",        "v1_kcfg_with_slice"),
    ("I14_TempWin4_VidKVV_256f",      256, "I_TempWin4_VidKVV",          "v1_kcfg_with_slice"),
]


# These need calibration (outlier indices) loaded.
CONDITIONS_NEEDING_CALIB_I = {
    "I2_F9_128f",
    "I8_TempWin2_Outlier8_128f",
    "I10_F9_256f",
    "I13_TempWin4_Outlier8_256f",
}


# Stage-3 promotion gating (pre-registered).
ANCHORS_ALWAYS_PROMOTED = {
    "I0_BF16_128f",
    "I1_F4_128f",
    "I2_F9_128f",
    "I3_TempWin2_128f",
    "I4_TextVisualSplit_128f",
    "I5_TokenBlock4_128f",
    "I9_F4_256f",
    "I10_F9_256f",
    "I11_TempWin4_256f",
}
VARIANTS_DATA_DRIVEN = {
    "I6_TempWin4_128f",
    "I7_TempWin2_VidKVV_128f",
    "I8_TempWin2_Outlier8_128f",
    "I12_TokenBlock6_256f",
    "I13_TempWin4_Outlier8_256f",
    "I14_TempWin4_VidKVV_256f",
}


def build_i_conditions(calib: Optional[dict]) -> list[dict]:
    """Return I-suite condition specs as dicts in the shape that
    expG_frame_scaling.run_item_at_frames expects: {name, mode, cfg, frames,
    f_cfg_name}.
    """
    fc = build_f_conditions(calib=calib)
    by_name = {cfg.name: cfg for cfg in fc}
    out: list[dict] = []
    for i_name, frames, f_cfg_name, mode in FIXED_FRAME_CONDITIONS_I:
        if f_cfg_name not in by_name:
            raise KeyError(
                f"[expI] f_cfg_name={f_cfg_name!r} not found in build_f_conditions; "
                f"available names: {sorted(by_name)}"
            )
        cfg = by_name[f_cfg_name]
        out.append({
            "name": i_name,
            "mode": mode,
            "cfg": cfg,
            "frames": frames,
            "f_cfg_name": f_cfg_name,
        })
    return out


# ===================================================================
# Fresh balanced split (seed != 0, not used by F/G/H)
# ===================================================================


def i_split_path(stage: int, seed: int) -> Path:
    n = sum(STAGE_TARGETS[stage].values())
    return CALIBRATION_DIR / f"split_seed{seed}_n{n}.json"


def ensure_i_split(stage: int, seed: int = 1) -> Path:
    """Generate or reuse the stratified split file at this (stage, seed).

    Stage 1 (n=64, 16/bucket) ⊂ Stage 3 (n=200, 50/bucket) is guaranteed:
    `make_split` shuffles each bucket with the same seed and slices target-
    prefixes; with `targets_stage1[bucket] <= targets_stage3[bucket]`, the
    Stage-1 ids are a prefix of the Stage-3 ids.
    """
    sp = i_split_path(stage, seed)
    if sp.exists():
        return sp
    items = load_all_items()
    targets = STAGE_TARGETS[stage]
    split = make_split(items, seed=seed, targets=targets, cal_fraction=0.0)
    save_split(split, sp)
    print(f"[expI] wrote stage {stage} split (n_eval={len(split['eval'])}) -> {sp}",
          flush=True)
    return sp


# ===================================================================
# Post-write: rename experiment/phase tags + BF16 join on I0
# ===================================================================


def rewrite_experiment_tag_i(out_jsonl: Path, stage: int) -> None:
    """The reused run_stage_g/run_item_at_frames stamps each row with
    `experiment="G"` and `phase=f"G{stage}"`. Rewrite to "I" / "I{stage}".
    """
    if not out_jsonl.exists():
        return
    rows = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l.strip()]
    phase_tag = f"I{stage}"
    for r in rows:
        r["experiment"] = "I"
        if "phase" in r:
            r["phase"] = phase_tag
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out_jsonl)


def backfill_bf16_join_i(out_jsonl: Path,
                         bf16_cond_name: str = "I0_BF16_128f") -> None:
    """Backfill bf16_pred / bf16_correct on every row using the BF16 anchor
    row of the same item_id. The Exp I anchor is at 128f (not 64f like Exp G).
    """
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
    print(f"[expI] backfilled BF16 join on {bf16_cond_name} in {out_jsonl} ({len(rows)} rows)")


# ===================================================================
# Stage-3 promotion gating
# ===================================================================


def filter_conditions_stage3(conditions: list[dict],
                             promotion_file: Optional[Path]) -> list[dict]:
    """For Stage 3: keep ANCHORS_ALWAYS_PROMOTED, plus any variant in
    VARIANTS_DATA_DRIVEN that appears in promotion_file (a JSON list of
    promoted condition names written by expI_analyze.py).
    """
    promoted_variants: set[str] = set()
    if promotion_file is not None and promotion_file.exists():
        try:
            data = json.loads(promotion_file.read_text())
            if isinstance(data, list):
                promoted_variants = set(data)
            elif isinstance(data, dict) and "promoted" in data:
                promoted_variants = set(data["promoted"])
        except Exception as e:
            print(f"[expI] WARN: could not parse {promotion_file}: {e}", flush=True)
    keep: list[dict] = []
    for c in conditions:
        n = c["name"]
        if n in ANCHORS_ALWAYS_PROMOTED:
            keep.append(c)
        elif n in VARIANTS_DATA_DRIVEN and n in promoted_variants:
            keep.append(c)
        elif n not in ANCHORS_ALWAYS_PROMOTED and n not in VARIANTS_DATA_DRIVEN:
            # Not an Exp I condition; keep to be safe.
            keep.append(c)
    return keep


# ===================================================================
# Driver
# ===================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--stage", type=int, choices=[1, 3], default=1,
                    help="Stage 1 (n=64) or Stage 3 (n=200). No Stage 0/2 for Exp I.")
    ap.add_argument("--seed", type=int, default=1,
                    help="Fresh split seed (default 1; F/G/H all used 0).")
    ap.add_argument("--split_file", type=Path, default=None,
                    help="Override the auto-generated stage split file.")
    ap.add_argument("--calib_npz", type=Path, default=None,
                    help="F-suite calibration NPZ (frames=64); needed by I2/I8/I10/I13.")
    ap.add_argument("--calib_json", type=Path, default=None)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="Subset of I condition names to run. Default: all 15 fixed-frame.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Override: limit eval items count after split selection.")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing JSONL rather than truncating.")
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None,
                    help="Default: results/expI_tempkivi_stage{stage}_seed{seed}.jsonl")
    ap.add_argument("--min_free_gb_256", type=int, default=70,
                    help="Skip the 256f tier if GPU free memory < this many GiB.")
    ap.add_argument("--stage3_promotion_file", type=Path, default=None,
                    help="JSON list of promoted variant names from Stage-1 verdict; "
                         "default: results/expI_promote_stage1.json (auto-detected).")
    args = ap.parse_args()

    if args.out is None:
        args.out = RESULTS_DIR / f"expI_tempkivi_stage{args.stage}_seed{args.seed}.jsonl"
    if args.split_file is None:
        args.split_file = ensure_i_split(args.stage, seed=args.seed)
    if args.calib_npz is None:
        model_short = args.model.split("/")[-1]
        cand = CALIBRATION_DIR / f"expF_kcalib_{model_short}_frames64.npz"
        if cand.exists():
            args.calib_npz = cand
    if args.calib_json is None and args.calib_npz is not None:
        cj = args.calib_npz.with_suffix(".json")
        if cj.exists():
            args.calib_json = cj
    if args.stage3_promotion_file is None:
        cand = RESULTS_DIR / "expI_promote_stage1.json"
        if cand.exists():
            args.stage3_promotion_file = cand

    items_all = load_all_items()
    split = load_split(args.split_file)
    eval_items = filter_items(items_all, split["eval"])
    if args.limit:
        eval_items = eval_items[: args.limit]
    print(f"[expI] stage={args.stage} seed={args.seed} eval_items={len(eval_items)} "
          f"split_file={args.split_file} calib_npz={args.calib_npz}", flush=True)

    calib, _ = load_calib(args.calib_npz, args.calib_json)
    calibration_id = (args.calib_npz.stem if args.calib_npz is not None else None)

    conditions = build_i_conditions(calib=calib)

    # Stage-3 promotion gating: keep anchors + promoted variants only.
    if args.stage == 3 and args.stage3_promotion_file is not None:
        before = [c["name"] for c in conditions]
        conditions = filter_conditions_stage3(conditions, args.stage3_promotion_file)
        after = [c["name"] for c in conditions]
        print(f"[expI] stage3 promotion gating: kept {after} (dropped "
              f"{sorted(set(before) - set(after))})", flush=True)
    elif args.stage == 3 and args.stage3_promotion_file is None:
        print("[expI] stage3 promotion file not found; running full slate (warning).",
              flush=True)

    # User override (subset).
    if args.conditions:
        wanted = set(args.conditions)
        conditions = [c for c in conditions if c["name"] in wanted]
        print(f"[expI] filtered conditions: {[c['name'] for c in conditions]}", flush=True)

    if calib is None and any(c["name"] in CONDITIONS_NEEDING_CALIB_I for c in conditions):
        missing = [c["name"] for c in conditions if c["name"] in CONDITIONS_NEEDING_CALIB_I]
        raise SystemExit(
            f"[expI] calibration missing but {missing} need it. "
            f"Run expF_calibrate.py first or pass --conditions to exclude."
        )

    if not args.append and args.out.exists():
        args.out.unlink()

    from run_inference import load_model
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    print(f"[expI] model loaded; num_layers={num_layers} num_kv_heads={num_kv_heads}",
          flush=True)

    run_stage_g(model, processor, eval_items,
                num_layers=num_layers, num_kv_heads=num_kv_heads,
                conditions=conditions, stage=args.stage,
                calibration_id=calibration_id,
                out_jsonl=args.out, progress_every=args.progress_every,
                min_free_gb_for_256=args.min_free_gb_256)
    rewrite_experiment_tag_i(args.out, args.stage)
    backfill_bf16_join_i(args.out, bf16_cond_name="I0_BF16_128f")


if __name__ == "__main__":
    main()
