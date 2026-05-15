#!/usr/bin/env bash
# Experiment V1 — Full-pool confirmation + budget-ladder residual screen.
#
# Builds on Exp U1: S4 (top-16 INT7 sidecode) + 8 residual extras at INT7
# beat the S4 anchor paired-significantly on retrieval-image (n=84), but only
# directionally beat F9. Exp V powers up the comparison on the FULL retrieval
# pool (n≈261), adds three deterministically-seeded RND variants for robust
# paired-vs-random, and tests a BAL budget ladder (4/8/12/16 residual
# channels) to confirm 8 is the sweet spot.
#
# Phases:
#   0  CPU extras compute (refresh with BAL ladder + RND seeds).
#   1  Smoke on retrieval (n=3 short, --exp-u assertions).
#   A  MM-NIAH retrieval-image full pool (n≈261, V0..V17).
#   B  MM-NIAH reasoning-image full pool (n≈120, R subset of V).
#   C  LongVideoBench-128f stage=3 n=200 (L0..L10).
#   5  Cross-phase aggregator.
#
# Env vars:
#   PIPELINE_MODEL          default Qwen/Qwen2.5-VL-7B-Instruct
#   CUDA_VISIBLE_DEVICES    required
#   QWEN_VENV               override venv; default /data/subha2/experiments/qwen_venv
#   EXPV_SKIP_SMOKE         set to 1 to bypass smoke
#   EXPV_SKIP_PHASE_A       set to 1 to bypass retrieval
#   EXPV_SKIP_PHASE_B       set to 1 to bypass reasoning
#   EXPV_SKIP_PHASE_C       set to 1 to bypass LVB
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_SHORT="${MODEL##*/}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QWEN_VENV="${QWEN_VENV:-/data/subha2/experiments/qwen_venv}"
if [ -f "$QWEN_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$QWEN_VENV/bin/activate"
  echo "Activated venv: $QWEN_VENV (python=$(which python3))"
fi

LOG="$QWEN_DIR/results/expV_overnight.progress.log"
RESULTS_DIR="$QWEN_DIR/results"
CALIB_DIR="$QWEN_DIR/calibration"
mkdir -p "$RESULTS_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

NPZ_RETRIEVAL="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_seed0.npz"
NPZ_REASONING="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_reasoning-image_seed0.npz"
NPZ_LVB="$CALIB_DIR/expJ_kcalib_${MODEL_SHORT}_frames128.npz"

EXTRAS_RETRIEVAL="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_seed0_expU_extras.npz"
EXTRAS_REASONING="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_reasoning-image_seed0_expU_extras.npz"
EXTRAS_LVB="$CALIB_DIR/expJ_kcalib_${MODEL_SHORT}_frames128_expU_extras.npz"

V_JSONL_RETRIEVAL="$RESULTS_DIR/expV_rollouts_sliceV_retrieval.jsonl"
V_JSONL_REASONING="$RESULTS_DIR/expV_rollouts_sliceV_reasoning.jsonl"
V_JSONL_LVB="$RESULTS_DIR/expV_lvb_stage3_seed2.jsonl"
V_JSONL_LVB_NORM="$RESULTS_DIR/expV_lvb_stage3_seed2_normalized.jsonl"

log "EXP V START model=$MODEL"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

if [ ! -f "$NPZ_RETRIEVAL" ]; then
  log "FAIL: retrieval calib NPZ missing: $NPZ_RETRIEVAL"
  exit 2
fi
log "  retrieval calib OK -> $NPZ_RETRIEVAL"
[ -f "$NPZ_REASONING" ] && log "  reasoning calib OK -> $NPZ_REASONING" || log "  WARN reasoning calib missing"
[ -f "$NPZ_LVB" ]       && log "  LVB calib OK -> $NPZ_LVB"             || log "  WARN LVB calib missing"

# ================================================================
# Phase 0 — refresh extras NPZs with BAL ladder + RND seed variants.
# ================================================================
log "--- PHASE 0: refresh Exp U/V extras NPZs (CPU only) ---"
python3 -u expU_compute_extras.py --all --model-short "$MODEL_SHORT" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "FAIL Phase 0 extras compute (rc=$rc)"
  exit $rc
fi
[ -f "$EXTRAS_RETRIEVAL" ] || { log "FAIL: retrieval extras NPZ missing after Phase 0"; exit 2; }
# Sanity: verify the new V-required keys are present in each extras NPZ.
python3 -u <<PYEOF 2>&1 | tee -a "$LOG"
import numpy as np
from pathlib import Path
required_v = {
    "outlier_idx_EXTRA_RND_8_s0",
    "outlier_idx_EXTRA_RND_8_s1",
    "outlier_idx_EXTRA_RND_8_s2",
    "outlier_idx_EXTRA_BAL_4",
    "outlier_idx_EXTRA_BAL_12",
    "outlier_idx_EXTRA_BAL_16",
}
for p in (
    "$EXTRAS_RETRIEVAL",
    "$EXTRAS_REASONING",
    "$EXTRAS_LVB",
):
    p = Path(p)
    if not p.exists():
        print(f"  WARN: extras NPZ missing: {p.name}")
        continue
    keys = set(np.load(p).files)
    missing = required_v - keys
    if missing:
        print(f"  FAIL: {p.name} missing V keys: {sorted(missing)}")
        raise SystemExit(2)
    print(f"  OK: {p.name} has all V keys ({len(keys)} total)")
PYEOF
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "FAIL Phase 0 sanity (rc=$rc)"
  exit $rc
fi

# ================================================================
# Phase 1 — Smoke (n=3 short bucket).
# ================================================================
if [ "${EXPV_SKIP_SMOKE:-0}" = "1" ]; then
  log "--- PHASE 1 SMOKE SKIPPED ---"
else
  log "--- PHASE 1 smoke: n=3 short bucket with --exp-u assertions (V uses same core) ---"
  python3 -u expQ_smoke.py \
      --model "$MODEL" \
      --n-items 3 \
      --bucket short \
      --task retrieval-image \
      --calib-npz "$NPZ_RETRIEVAL" \
      --extras-npz "$EXTRAS_RETRIEVAL" \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --exp-u \
      --out "$RESULTS_DIR/expV_smoke.md" \
      --out-jsonl "$RESULTS_DIR/expV_smoke.jsonl" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL smoke (rc=$rc)"
    exit $rc
  fi
  log "  smoke OK"
fi

# ================================================================
# Phase A — MM-NIAH retrieval-image full pool (n≈261, V0..V17).
# ================================================================
if [ "${EXPV_SKIP_PHASE_A:-0}" = "1" ]; then
  log "--- PHASE A retrieval-image SKIPPED ---"
else
  log "--- PHASE A: MM-NIAH retrieval-image V0..V17 × n≈261 full pool at 336° ---"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
  python3 -u expQ_driver.py \
      --model "$MODEL" \
      --task retrieval-image \
      --calib-npz "$NPZ_RETRIEVAL" \
      --extras-npz "$EXTRAS_RETRIEVAL" \
      --use-full-pool \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --exp-v \
      --include-choice-routing \
      --out-jsonl "$V_JSONL_RETRIEVAL" \
      --out-summary "$RESULTS_DIR/expV_summary_sliceV_retrieval.md" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL Phase A retrieval-image (rc=$rc)"
    exit $rc
  fi
  log "  Phase A OK -> $V_JSONL_RETRIEVAL"

  log "--- PHASE A analyze ---"
  python3 -u expQ_analyze.py --slice V \
      --in-jsonl "$V_JSONL_RETRIEVAL" \
      --out-prefix "expV_retrieval" \
      --out-summary "$RESULTS_DIR/expV_summary_sliceV_retrieval.md" \
      --out-paired  "$RESULTS_DIR/expV_paired_sliceV_retrieval.md" \
      --out-verdict "$RESULTS_DIR/expV_verdict_sliceV_retrieval.md" \
      --out-branch  "$RESULTS_DIR/expV_branch_sliceV_retrieval.json" \
      2>&1 | tee -a "$LOG"
fi

# ================================================================
# Phase B — MM-NIAH reasoning-image full pool (n≈120, R subset of V).
# R0=V0 R1=V1 R2=V2 R3=V3 R4=V4 R5=V5 R6=V6 R7=V7 R8=V10 (VT) R9=V11 R10=V12 R11=V13.
# ================================================================
if [ "${EXPV_SKIP_PHASE_B:-0}" = "1" ]; then
  log "--- PHASE B reasoning-image SKIPPED ---"
elif [ ! -f "$NPZ_REASONING" ]; then
  log "--- PHASE B reasoning-image SKIPPED (calib NPZ missing) ---"
elif [ ! -f "$EXTRAS_REASONING" ]; then
  log "--- PHASE B reasoning-image SKIPPED (extras NPZ missing) ---"
else
  log "--- PHASE B: MM-NIAH reasoning-image R-subset × n≈120 full pool at 336° ---"
  python3 -u expQ_driver.py \
      --model "$MODEL" \
      --task reasoning-image \
      --calib-npz "$NPZ_REASONING" \
      --extras-npz "$EXTRAS_REASONING" \
      --use-full-pool \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --exp-v \
      --conditions V0 V1 V2 V3 V4 V5 V6 V7 V10 V11 V12 V13 \
      --out-jsonl "$V_JSONL_REASONING" \
      --out-summary "$RESULTS_DIR/expV_summary_sliceV_reasoning.md" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL Phase B reasoning-image (rc=$rc)"
    exit $rc
  fi
  log "  Phase B OK -> $V_JSONL_REASONING"

  log "--- PHASE B analyze ---"
  python3 -u expQ_analyze.py --slice V \
      --in-jsonl "$V_JSONL_REASONING" \
      --out-prefix "expV_reasoning" \
      --out-summary "$RESULTS_DIR/expV_summary_sliceV_reasoning.md" \
      --out-paired  "$RESULTS_DIR/expV_paired_sliceV_reasoning.md" \
      --out-verdict "$RESULTS_DIR/expV_verdict_sliceV_reasoning.md" \
      --out-branch  "$RESULTS_DIR/expV_branch_sliceV_reasoning.json" \
      2>&1 | tee -a "$LOG"
fi

# ================================================================
# Phase C — LongVideoBench-128f stage=3 n=200 (L0..L10).
# ================================================================
if [ "${EXPV_SKIP_PHASE_C:-0}" = "1" ]; then
  log "--- PHASE C LVB-128f SKIPPED ---"
elif [ ! -f "$NPZ_LVB" ]; then
  log "--- PHASE C LVB-128f SKIPPED (calib NPZ missing) ---"
elif [ ! -f "$EXTRAS_LVB" ]; then
  log "--- PHASE C LVB-128f SKIPPED (extras NPZ missing) ---"
else
  log "--- PHASE C: LongVideoBench-128f V/L conditions × n=200 stage=3 seed=2 ---"
  python3 -u expJ_xmodal_outlier.py \
      --model "$MODEL" \
      --stage 3 \
      --seed 2 \
      --calib_npz "$NPZ_LVB" \
      --extras-npz "$EXTRAS_LVB" \
      --exp-v \
      --out "$V_JSONL_LVB" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL Phase C LVB-128f (rc=$rc)"
    exit $rc
  fi
  log "  Phase C OK -> $V_JSONL_LVB"

  # Normalize the V-suffix condition names back to bare V0..V14 for the analyzer.
  log "--- PHASE C normalize condition names + analyze ---"
  python3 -u <<PYEOF 2>&1 | tee -a "$LOG"
import json
from pathlib import Path
src = Path("$V_JSONL_LVB")
dst = Path("$V_JSONL_LVB_NORM")
n_in = n_out = 0
def short(c):
    if not c: return c
    return c.split("_", 1)[0]
with open(src) as fi, open(dst, "w") as fo:
    for line in fi:
        if not line.strip(): continue
        r = json.loads(line); n_in += 1
        c = r.get("condition")
        if c: r["condition"] = short(c)
        if "cond_name" in r and r["cond_name"]:
            r["cond_name"] = short(r["cond_name"])
        fo.write(json.dumps(r) + "\n"); n_out += 1
print(f"normalized {n_in} -> {n_out} rows")
PYEOF
  python3 -u expQ_analyze.py --slice V \
      --in-jsonl "$V_JSONL_LVB_NORM" \
      --out-prefix "expV_lvb" \
      --out-summary "$RESULTS_DIR/expV_summary_sliceV_lvb.md" \
      --out-paired  "$RESULTS_DIR/expV_paired_sliceV_lvb.md" \
      --out-verdict "$RESULTS_DIR/expV_verdict_sliceV_lvb.md" \
      --out-branch  "$RESULTS_DIR/expV_branch_sliceV_lvb.json" \
      2>&1 | tee -a "$LOG"
fi

# ================================================================
# Phase 5 — cross-phase aggregator.
# ================================================================
log "--- PHASE 5: cross-phase winning_policy + primary deployable summary ---"
export EXPV_RESULTS_DIR="$RESULTS_DIR"
python3 -u <<PYEOF 2>&1 | tee -a "$LOG"
import json, os
from pathlib import Path
R = Path(os.environ["EXPV_RESULTS_DIR"])
slices = [
    ("retrieval", R / "expV_branch_sliceV_retrieval.json"),
    ("reasoning", R / "expV_branch_sliceV_reasoning.json"),
    ("lvb",       R / "expV_branch_sliceV_lvb.json"),
]
print()
print("## Cross-phase Exp V1 verdict")
print()
print("| phase | winning_policy | V11_beats_F9 | V12_beats_F9 | V11_beats_S4 | V11_beats_random_robust | match_or_beat_F9 |")
print("|---|---|---|---|---|---|---|")
winners = {}
v11_vs_f9 = {}
for tag, p in slices:
    if not p.exists():
        print("| %s | (missing) | - | - | - | - | - |" % tag); continue
    b = json.loads(p.read_text())
    winners[tag] = b.get("winning_policy")
    v11_vs_f9[tag] = b.get("pass_v11_beats_f9")
    print("| %s | %s | %s | %s | %s | %s | %s |" % (
        tag, b.get("winning_policy"),
        b.get("pass_v11_beats_f9"),
        b.get("pass_v12_beats_f9"),
        b.get("pass_v11_beats_s4"),
        b.get("pass_v11_beats_random_robust"),
        b.get("pass_match_or_beat_f9"),
    ))
print()
print("winners by phase:", winners)
print("v11_beats_f9 by phase:", v11_vs_f9)
print()
print("=== TOP-3 per phase (acc) ===")
for tag, p in slices:
    if not p.exists(): continue
    b = json.loads(p.read_text())
    rank = b.get("candidate_ranking", [])[:3]
    accs = b.get("accuracy", {})
    s = "%s: " % tag
    s += ", ".join("%s=%.3f" % (r["cond"], r["acc"]) for r in rank)
    s += " | anchors V0 BF16=%.3f, V2 F9=%.3f, V3 S4=%.3f" % (
        accs.get("V0", float("nan")),
        accs.get("V2", float("nan")),
        accs.get("V3", float("nan"))
    )
    print(s)
print()
# Deployable headline
deployable_phases = [t for t in winners if v11_vs_f9.get(t)]
if deployable_phases:
    print("** DEPLOYABLE HEADLINE: V11 BAL8 paired-significantly beats F9 on:",
          deployable_phases, "**")
elif any(v11_vs_f9.values()):
    print("** PARTIAL: V11 BAL8 paired-significant on some phases:", v11_vs_f9, "**")
else:
    print("** NEGATIVE: V11 BAL8 did NOT paired-significantly beat F9 on any phase. "
          "Check directional + borderline pairs in the verdict matrices. **")
PYEOF

log "EXP V DONE — review $RESULTS_DIR/expV_*"
