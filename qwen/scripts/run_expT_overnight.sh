#!/usr/bin/env bash
# Experiment T — Seed=1 replication of the Exp S sidecode-ladder winner.
#
# After Exp S found S4 (top-16 INT7 sidecode, 4.1875 KV bits, acc=0.571)
# paired-tied with S3 SJ (top-16 INT8, 4.250 bits, 0.583) and decisively
# beat the matched-budget wider-lower-precision controls (S7/S8), the
# remaining question is: does this hold on a fresh seed?
#
# Past history (Exp J seed=2 → Exp K seed=1) shows one-seed wins can
# collapse. Exp T is the non-negotiable replication.
#
# Phase 0:
#   0a. Smoke with --exp-t: P/Q/R/U assertions (logits differ across
#       INT8/INT7/INT6, bit math for the ladder, sidecode is K-only,
#       BF16 dense V is 16-not-4 regression check).
#   0b. seed=1 split for MM-NIAH retrieval-image.
#   0c. seed=1 F9 calibration on the new cal-100 split.
#
# Phase 1:
#   T0 BF16, T1 F4, T2 F9, T3 SJ INT8, T4 INT7, T5 INT6 (cliff control)
#   on the same n=84 multi-image filter at 336° equal-resolution, but
#   with seed=1 split and seed=1 F9 calibration NPZ.
#
# Pass condition: T4 paired-ties T3 and T2 at fewer bits AND T5 is
# clearly worse than T4. If T4 fails on seed=1, S4 is not deployable.
#
# Env vars:
#   PIPELINE_MODEL       default Qwen/Qwen2.5-VL-7B-Instruct
#   CUDA_VISIBLE_DEVICES required
#   QWEN_VENV            override venv; default /data/subha2/experiments/qwen_venv
#   EXPT_SKIP_SMOKE      set to 1 to bypass smoke
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QWEN_VENV="${QWEN_VENV:-/data/subha2/experiments/qwen_venv}"
if [ -f "$QWEN_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$QWEN_VENV/bin/activate"
  echo "Activated venv: $QWEN_VENV (python=$(which python3))"
fi

LOG="$QWEN_DIR/results/expT_overnight.progress.log"
RESULTS_DIR="$QWEN_DIR/results"
CALIB_DIR="$QWEN_DIR/calibration"
mkdir -p "$RESULTS_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

MODEL_SHORT="${MODEL##*/}"
SEED0_NPZ="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_seed0.npz"
SEED1_NPZ="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_seed1.npz"
SEED1_SPLIT="$CALIB_DIR/mm_niah_retrieval-image_split_seed1.json"
T_JSONL="$RESULTS_DIR/expT_rollouts_seed1.jsonl"

log "EXP T START model=$MODEL"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

if [ ! -f "$SEED0_NPZ" ]; then
  log "FAIL seed=0 calibration NPZ missing: $SEED0_NPZ (needed for smoke)"
  exit 2
fi

# ================================================================
# Phase 0a — Smoke (--exp-t).
# ================================================================
if [ "${EXPT_SKIP_SMOKE:-0}" = "1" ]; then
  log "--- PHASE 0a SMOKE SKIPPED ---"
else
  log "--- PHASE 0a: smoke n=3 short bucket with --exp-t assertions ---"
  python3 -u expQ_smoke.py \
      --model "$MODEL" \
      --n-items 3 \
      --bucket short \
      --task retrieval-image \
      --calib-npz "$SEED0_NPZ" \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --out "$RESULTS_DIR/expT_smoke.md" \
      --out-jsonl "$RESULTS_DIR/expT_smoke.jsonl" \
      --exp-t \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL smoke (rc=$rc). Wiring or bit-math is broken — do not launch main."
    exit $rc
  fi
  log "  smoke OK -> $RESULTS_DIR/expT_smoke.md"
fi

# ================================================================
# Phase 0b — Generate seed=1 split for retrieval-image.
# ================================================================
log "--- PHASE 0b: seed=1 split for retrieval-image ---"
if [ -f "$SEED1_SPLIT" ]; then
  log "  seed=1 split already exists at $SEED1_SPLIT; skipping"
else
  python3 -u mm_niah_loader.py --seed 1 --task retrieval-image 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL seed=1 split (rc=$rc)"
    exit $rc
  fi
fi
if [ ! -f "$SEED1_SPLIT" ]; then
  log "FAIL seed=1 split not produced at $SEED1_SPLIT"
  exit 3
fi
log "  seed=1 split OK -> $SEED1_SPLIT"

# ================================================================
# Phase 0c — Fresh F9 calibration on the seed=1 cal-100 split.
# ================================================================
log "--- PHASE 0c: seed=1 F9 calibration on cal-100 ---"
if [ -f "$SEED1_NPZ" ]; then
  log "  seed=1 calib NPZ already exists at $SEED1_NPZ; skipping"
else
  python3 -u expP_calibrate.py --model "$MODEL" --seed 1 --split_file "$SEED1_SPLIT" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL seed=1 calibration (rc=$rc)"
    exit $rc
  fi
fi
if [ ! -f "$SEED1_NPZ" ]; then
  log "FAIL seed=1 calib NPZ not produced at $SEED1_NPZ"
  exit 4
fi
log "  seed=1 calibration OK -> $SEED1_NPZ"

# ================================================================
# Phase 1 — T0..T5 on seed=1 split with seed=1 cal.
# ================================================================
log "--- PHASE 1: T0..T5 (= S0..S5 subset of Exp S ladder) on seed=1 ---"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
python3 -u expQ_driver.py \
    --model "$MODEL" \
    --task retrieval-image \
    --seed 1 \
    --calib-npz "$SEED1_NPZ" \
    --split-path "$SEED1_SPLIT" \
    --use-full-pool \
    --min-num-images 8 \
    --max-pixels-context $((336*336)) \
    --max-pixels-choices $((336*336)) \
    --exp-s-ladder \
    --conditions S0 S1 S2 S3 S4 S5 \
    --include-choice-routing \
    --out-jsonl "$T_JSONL" \
    --out-summary "$RESULTS_DIR/expT_summary.md" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "FAIL Phase 1 main run (rc=$rc)"
  exit $rc
fi
log "  Phase 1 OK -> $T_JSONL"

# ================================================================
# Phase 1 analyze.
# ================================================================
log "--- PHASE 1 analyze (--slice S, pairs_slice_s) ---"
python3 -u expQ_analyze.py --slice S \
    --in-jsonl "$T_JSONL" \
    --out-summary "$RESULTS_DIR/expT_summary.md" \
    --out-paired  "$RESULTS_DIR/expT_paired.md" \
    --out-verdict "$RESULTS_DIR/expT_verdict.md" \
    --out-branch  "$RESULTS_DIR/expT_branch.json" \
    2>&1 | tee -a "$LOG"
log "  analyze OK"

log "EXP T DONE — review outputs in $RESULTS_DIR/expT_*"
log "Pass criteria for the deployable claim:"
log "  - T4 paired-ties T3 AND T2 in expT_paired.md (chi-sq < 3.84)"
log "  - T5 paired-WORSE than T4 (chi-sq > 3.84, T4 favored)"
log "If T4 fails on seed=1, S4 is not deployable — pivot back to F9/SJ."
