"""Experiment K — Balanced Cross-Modal Sidecode Replication.

After Exp J Stage 3 produced two findings on seed=2:
  - J7 (Balanced TT/TV/VT/VV top-2/block, BF16 sidecode) beats F9 by +3 pp
    at lower bits, paired McNemar significant against both generic and
    random controls.
  - J12 (F9 with INT8 sidecode) ties F9 at 4.25 vs 4.75 KV bits.

Exp K asks four narrower questions, each on n=200 of a different seed:

  Q1. Does J7 replicate on seed=1 (the harder split)?
  Q2. Does J7 still beat generic top-8 and random top-8?
  Q3. Can the J7 winner also use INT8 sidecode (K7) instead of BF16?
  Q4. Is top-2/block the right budget — does top-1 (K8) or top-3 (K9) win?

12 conditions, single 128f tier, no Stage 1 (this is a replication run on
already-promoted variants). Optionally chained across seed=1, seed=0, and
seed=2 reruns for cross-seed triangulation.

Conditions:
  K0  BF16 128f
  K1  F4 KIVI                                  4.000 KV bits
  K2  F9 generic top-16 BF16 sidecode          4.750 KV bits
  K3  F9 generic top-16 INT8 sidecode          4.250 KV bits
  K4  F8 generic top-8 BF16 sidecode           4.375 KV bits
  K5  Random top-8 BF16 sidecode               4.375 KV bits  (J15 control)
  K6  Balanced top-2/block BF16 sidecode       4.375 KV bits  (J7 replication)
  K7  Balanced top-2/block INT8 sidecode       4.125 KV bits  (J7 + sidecode)
  K8  Balanced top-1/block BF16 sidecode       4.1875 KV bits (smaller budget)
  K9  Balanced top-3/block BF16 sidecode       4.5625 KV bits (larger budget)
  K10 Balanced-random by channel-position      4.375 KV bits  (decouples balance from cross-modal)
  K11 Pivot top-8 BF16 sidecode                4.375 KV bits  (J8 replication)

Calibration:
  Reuses the existing seed=0 cal-100 calibration NPZ from Exp J:
  qwen/calibration/expJ_kcalib_<model>_frames128.npz
  Stage-3 control arrays (random/random-LA/pivot-err) and K-specific arrays
  (balanced-per-block at n_per_block ∈ {1,2,3}, balanced-random-by-position)
  are injected at driver startup.

Per-seed eval splits:
  --seed 0 -> qwen/calibration/split_seed0.json (eval field, n=200)
  --seed 1 -> qwen/calibration/split_seed1_n200.json
  --seed 2 -> qwen/calibration/split_seed2_n200.json

Output:
  qwen/results/expK_balanced_stage3_seed{S}.jsonl

Usage:
  python expK_balanced_replication.py --seed 1
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
from k_quantizers import (
    build_f_conditions,
    _balanced_per_block_top_indices,
    _balanced_random_top_indices,
)

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


# (k_name, frames, f_cfg_name, mode)
FIXED_FRAME_CONDITIONS_K = [
    ("K0_BF16_128f",                 128, "F0_BF16",                  "bf16"),
    ("K1_F4_128f",                   128, "F4_KIVI_PerChannelSeq",    "v1_kcfg"),
    ("K2_F9_BF16side_128f",          128, "K2_F9_BF16side",           "v1_kcfg"),
    ("K3_F9_INT8side_128f",          128, "K3_F9_INT8side",           "v1_kcfg"),
    ("K4_F8_BF16side_128f",          128, "K4_F8_BF16side",           "v1_kcfg"),
    ("K5_Random8_BF16side_128f",     128, "K5_Random8_BF16side",      "v1_kcfg"),
    ("K6_Bal2pb_BF16side_128f",      128, "K6_Bal2pb_BF16side",       "v1_kcfg"),
    ("K7_Bal2pb_INT8side_128f",      128, "K7_Bal2pb_INT8side",       "v1_kcfg"),
    ("K8_Bal1pb_BF16side_128f",      128, "K8_Bal1pb_BF16side",       "v1_kcfg"),
    ("K9_Bal3pb_BF16side_128f",      128, "K9_Bal3pb_BF16side",       "v1_kcfg"),
    ("K10_BalRandomPos_BF16side_128f", 128, "K10_BalRandomPos_BF16side", "v1_kcfg"),
    ("K11_Pivot8_BF16side_128f",     128, "K11_Pivot8_BF16side",      "v1_kcfg"),
]


def build_k_conditions(calib: Optional[dict]) -> list[dict]:
    fc = build_f_conditions(calib=calib)
    by_name = {cfg.name: cfg for cfg in fc}
    out: list[dict] = []
    for k_name, frames, f_cfg_name, mode in FIXED_FRAME_CONDITIONS_K:
        if f_cfg_name not in by_name:
            raise KeyError(
                f"[expK] f_cfg_name={f_cfg_name!r} not found in build_f_conditions; "
                f"available: {sorted(by_name)}"
            )
        cfg = by_name[f_cfg_name]
        out.append({
            "name": k_name, "mode": mode, "cfg": cfg, "frames": frames,
            "f_cfg_name": f_cfg_name,
        })
    return out


def resolve_split_for_seed(seed: int) -> Path:
    """Pick the existing n=200 eval split file for a given seed.

    seed=0: F-suite canonical split (cal+eval); we use the 'eval' field (~200 items).
    seed=1: split_seed1_n200.json (from Exp I).
    seed=2: split_seed2_n200.json (from Exp J).
    """
    if seed == 0:
        cand = CALIBRATION_DIR / "split_seed0.json"
        if cand.exists():
            return cand
        # Fallback: split_seed0_n200.json (regenerate if needed).
        cand2 = CALIBRATION_DIR / "split_seed0_n200.json"
        if cand2.exists():
            return cand2
        raise FileNotFoundError(f"no seed=0 split at {cand} or {cand2}")
    p = CALIBRATION_DIR / f"split_seed{seed}_n200.json"
    if not p.exists():
        # Generate it.
        items = load_all_items()
        split = make_split(items, seed=seed,
                           targets={"short": 50, "mid": 50, "long": 50, "very_long": 50},
                           cal_fraction=0.0)
        save_split(split, p)
        print(f"[expK] generated {p}", flush=True)
    return p


def inject_k_calib_arrays(calib: dict) -> None:
    """Mutate `calib` in place to add Exp-K control arrays:
      outlier_idx_RANDOM_top16              (J15 reuse — fully random)
      outlier_idx_PIVOT_top16               (already in calib from expJ_calibrate;
                                             present in NPZ — no-op if so)
      outlier_idx_BAL_top1_per_block_top16  (top-1 per modality block, deduped)
      outlier_idx_BAL_top2_per_block_top16  (top-2 per modality block, deduped)
      outlier_idx_BAL_top3_per_block_top16  (top-3 per modality block, deduped)
      outlier_idx_BAL_RANDOM_POS_top16      (2 random per channel-position block)
    """
    L, H_kv, D = calib["k_channel_energy"].shape
    n_top = int(calib["outlier_idx_TT_top16"].shape[-1])

    # Recompute D_TT/D_TV/D_VT/D_VV from per-modality energies in calib.
    qet = calib["q_energy_text"]
    qev = calib["q_energy_visual"]
    ke_t = calib["k_channel_energy_text"]
    ke_v = calib["k_channel_energy_visual"]
    D_TT = qet * ke_t
    D_TV = qet * ke_v
    D_VT = qev * ke_t
    D_VV = qev * ke_v
    scores = {"TT": D_TT, "TV": D_TV, "VT": D_VT, "VV": D_VV}

    # Balanced top-1/block (4 channels effective; pad to n_top=16).
    if "outlier_idx_BAL_top1_per_block_top16" not in calib:
        calib["outlier_idx_BAL_top1_per_block_top16"] = _balanced_per_block_top_indices(
            scores, n_per_block=1, n_top=n_top)
    # Balanced top-2/block (8 channels effective; pad to n_top=16). This is J7's intent.
    if "outlier_idx_BAL_top2_per_block_top16" not in calib:
        calib["outlier_idx_BAL_top2_per_block_top16"] = _balanced_per_block_top_indices(
            scores, n_per_block=2, n_top=n_top)
    # Balanced top-3/block (12 channels effective; pad to n_top=16).
    if "outlier_idx_BAL_top3_per_block_top16" not in calib:
        calib["outlier_idx_BAL_top3_per_block_top16"] = _balanced_per_block_top_indices(
            scores, n_per_block=3, n_top=n_top)

    # Balanced-random by channel-position blocks (control for K6 mechanism).
    if "outlier_idx_BAL_RANDOM_POS_top16" not in calib:
        calib["outlier_idx_BAL_RANDOM_POS_top16"] = _balanced_random_top_indices(
            num_layers=L, num_kv_heads=H_kv, head_dim=D,
            n_per_block=2, n_blocks=4, n_top=n_top, seed=99)

    # Fully-random top-16 (K5 control; same as J15). Fixed seed for reproducibility.
    if "outlier_idx_RANDOM_top16" not in calib:
        rng = np.random.default_rng(42)
        rand_idx = np.empty((L, H_kv, n_top), dtype=np.int32)
        for L_i in range(L):
            for h in range(H_kv):
                rand_idx[L_i, h] = rng.choice(D, size=n_top, replace=False).astype(np.int32)
        calib["outlier_idx_RANDOM_top16"] = rand_idx

    # Diagnostic overlap with generic top-16: how many of the K6 picks coincide with magnitude top-16?
    bal2 = calib["outlier_idx_BAL_top2_per_block_top16"]
    generic = calib["outlier_channel_idx_top16"]
    overlap = []
    for L_i in range(L):
        for h in range(H_kv):
            s_bal = set(bal2[L_i, h, :8].tolist())  # first 8 = balanced top-2/block
            s_gen = set(generic[L_i, h, :8].tolist())
            overlap.append(len(s_bal & s_gen))
    print(f"[expK] BAL_top2_per_block first-8 overlap with generic top-8: "
          f"mean={np.mean(overlap):.2f}/8 (over {L*H_kv} cells)", flush=True)


def rewrite_experiment_tag_k(out_jsonl: Path, stage: int) -> None:
    if not out_jsonl.exists():
        return
    rows = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l.strip()]
    phase_tag = f"K{stage}"
    for r in rows:
        r["experiment"] = "K"
        if "phase" in r:
            r["phase"] = phase_tag
    tmp = out_jsonl.with_suffix(out_jsonl.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(out_jsonl)


def backfill_bf16_join_k(out_jsonl: Path,
                         bf16_cond_name: str = "K0_BF16_128f") -> None:
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
    print(f"[expK] backfilled BF16 join on {bf16_cond_name} in {out_jsonl} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--seed", type=int, required=True,
                    help="Eval split seed: 0, 1, or 2.")
    ap.add_argument("--split_file", type=Path, default=None,
                    help="Override eval split file.")
    ap.add_argument("--calib_npz", type=Path, default=None,
                    help="Calibration NPZ; defaults to expJ_kcalib_<model>_frames128.npz.")
    ap.add_argument("--calib_json", type=Path, default=None)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min_free_gb_256", type=int, default=70,
                    help="Unused at 128f-only; kept for run_stage_g signature.")
    args = ap.parse_args()

    if args.out is None:
        args.out = RESULTS_DIR / f"expK_balanced_stage3_seed{args.seed}.jsonl"
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
    print(f"[expK] seed={args.seed} eval_items={len(eval_items)} "
          f"split_file={args.split_file} calib_npz={args.calib_npz}", flush=True)

    calib, _ = load_calib(args.calib_npz, args.calib_json)
    calibration_id = (args.calib_npz.stem if args.calib_npz is not None else None)
    if calib is None:
        raise SystemExit(
            "[expK] calibration NPZ not loaded; needs the Exp J cross-modal NPZ."
        )

    required_for_K = {
        "k_channel_energy", "k_channel_energy_text", "k_channel_energy_visual",
        "q_energy_text", "q_energy_visual", "q_energy_pivot",
        "outlier_idx_TT_top16", "outlier_idx_PIVOT_top16",
        "outlier_channel_idx_top16",
    }
    missing = required_for_K - set(calib)
    if missing:
        raise SystemExit(
            f"[expK] calibration NPZ missing required arrays: {sorted(missing)}. "
            f"Available: {sorted(calib)}. "
            f"Re-run expJ_calibrate.py."
        )

    # Inject Exp-K control arrays (balanced per_block ∈ {1,2,3} + balanced-random-pos).
    inject_k_calib_arrays(calib)
    print(f"[expK] injected K-suite control arrays into calib", flush=True)

    conditions = build_k_conditions(calib=calib)

    if args.conditions:
        wanted = set(args.conditions)
        conditions = [c for c in conditions if c["name"] in wanted]
        print(f"[expK] filtered conditions: {[c['name'] for c in conditions]}", flush=True)

    if not args.append and args.out.exists():
        args.out.unlink()

    from run_inference import load_model
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    print(f"[expK] model loaded; num_layers={num_layers} num_kv_heads={num_kv_heads}",
          flush=True)

    run_stage_g(model, processor, eval_items,
                num_layers=num_layers, num_kv_heads=num_kv_heads,
                conditions=conditions, stage=3,
                calibration_id=calibration_id,
                out_jsonl=args.out, progress_every=args.progress_every,
                min_free_gb_for_256=args.min_free_gb_256)
    rewrite_experiment_tag_k(args.out, stage=3)
    backfill_bf16_join_k(args.out, bf16_cond_name="K0_BF16_128f")


if __name__ == "__main__":
    main()
