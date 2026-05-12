#!/usr/bin/env bash
# Experiment L — seed=1 calibration sanity check.
#
# Tests whether the J7 failure on seed=1 (Exp K) was because the J7
# mechanism is genuinely seed=2-specific, or because we used seed=0-derived
# calibration on seed=1 eval. Recalibrates on a FRESH seed=1 cal-100 split
# disjoint from the existing seed=1 eval-200, then re-evaluates the K-suite
# (12 conditions) on the same seed=1 eval-200 items.
#
# Phases:
#   Phase 1: Generate seed=1 cal split (disjoint from existing eval).
#   Phase 2: Run expJ_calibrate.py on seed=1 cal-100 (~15 min on contested GPU).
#   Phase 3: Run expK_balanced_replication.py on existing seed=1 eval-200 with the new calib NPZ.
#   Phase 4: Analyze.
#
# Env vars:
#   PIPELINE_MODEL       default Qwen/Qwen2.5-VL-7B-Instruct
#   CUDA_VISIBLE_DEVICES required
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG="$QWEN_DIR/results/expL_overnight.progress.log"
mkdir -p "$QWEN_DIR/results"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

MODEL_SHORT="${MODEL##*/}"
CAL_SPLIT="$QWEN_DIR/calibration/split_seed1_cal100_for_existing_eval.json"
CALIB_NPZ="$QWEN_DIR/calibration/expJ_kcalib_${MODEL_SHORT}_frames128_seed1cal.npz"
STAGE3_OUT="$QWEN_DIR/results/expL_seed1_recalib_stage3.jsonl"

log "EXP L SEED=1 RECALIB START model=$MODEL"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

# ================================================================
# Phase 1 — Generate the seed=1 cal split.
# ================================================================
log "--- PHASE 1: build seed=1 cal-100 split (disjoint from existing eval) ---"
if [ -f "$CAL_SPLIT" ]; then
  log "  cal split already exists at $CAL_SPLIT; skipping"
else
  python3 -u make_seed1_cal_split.py 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL split generation (rc=$rc)"
    exit $rc
  fi
fi
if [ ! -f "$CAL_SPLIT" ]; then
  log "FAIL cal split not produced at $CAL_SPLIT"
  exit 2
fi
log "  cal split OK -> $CAL_SPLIT"

# ================================================================
# Phase 2 — Calibration on seed=1 cal-100.
# ================================================================
log "--- PHASE 2: seed=1 cross-modal calibration ---"
if [ -f "$CALIB_NPZ" ]; then
  log "  calib NPZ already exists at $CALIB_NPZ; skipping"
else
  python3 -u expJ_calibrate.py --model "$MODEL" --frames 128 \
      --split_file "$CAL_SPLIT" \
      --out_npz "$CALIB_NPZ" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL calibration (rc=$rc)"
    exit $rc
  fi
fi
if [ ! -f "$CALIB_NPZ" ]; then
  log "FAIL calibration NPZ not produced at $CALIB_NPZ"
  exit 3
fi
log "  calibration OK -> $CALIB_NPZ"

# ================================================================
# Phase 3 — Stage 3 on existing seed=1 eval-200 with seed=1 calib.
# ================================================================
log "--- PHASE 3: Stage 3 on seed=1 eval-200, seed=1-calibrated channels ---"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
python3 -u expK_balanced_replication.py \
    --seed 1 \
    --calib_npz "$CALIB_NPZ" \
    --out "$STAGE3_OUT" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "FAIL Stage 3 (rc=$rc)"
  exit $rc
fi
if [ ! -f "$STAGE3_OUT" ]; then
  log "FAIL Stage 3 JSONL not produced at $STAGE3_OUT"
  exit 4
fi
log "  Stage 3 OK -> $STAGE3_OUT"

# ================================================================
# Phase 4 — Analyze.
# ================================================================
log "--- PHASE 4: analyze Exp L seed=1 recalib ---"
python3 -u expK_analyze.py --seed 1 \
    --in_jsonl "$STAGE3_OUT" \
    --summary  "$QWEN_DIR/results/expL_summary_seed1_recalib.md" \
    --paired   "$QWEN_DIR/results/expL_paired_seed1_recalib.md" \
    --verdict  "$QWEN_DIR/results/expL_verdict_matrix_seed1_recalib.md" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "WARN analyzer (rc=$rc)"
fi
log "  analyze OK -> see expL_*_seed1_recalib.md"

log "EXP L DONE — review outputs in $QWEN_DIR/results/expL_*"
log "Compare to Exp K seed=1 results in expK_*_seed1.md:"
log "  - If K3/L3 INT8 sidecode result holds, the engineering finding is robust."
log "  - If K6/L6 balanced result RECOVERS, J7 was calibration-dependent."
log "  - If K6/L6 stays low, J7 mechanism is genuinely seed=2-specific."
