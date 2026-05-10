#!/usr/bin/env bash
# Experiment J — Cross-modal outlier-channel overnight orchestrator.
#
# Runs the full overnight pipeline in sequence:
#   smoke (Phase A + Phase B with calibration)
#   calibrate
#   stage1
#   analyze
#
# Stage 3 is launched manually next day after reviewing the verdict matrix.
#
# Env vars:
#   PIPELINE_MODEL       default Qwen/Qwen2.5-VL-7B-Instruct
#   EXPJ_SEED            default 2
#   EXPJ_FRAMES          default 128
#   EXPJ_N_LIMIT         debug: limit cal/eval items count (default 0 -> full)
#   CUDA_VISIBLE_DEVICES required at launch
#   PYTORCH_CUDA_ALLOC_CONF defaults to expandable_segments:True
#
# Usage:
#   tmux new -s expJ
#   export CUDA_VISIBLE_DEVICES=0
#   bash run_expJ_overnight.sh
#   # Detach: Ctrl-b d. Reattach: tmux attach -t expJ.
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
SEED="${EXPJ_SEED:-2}"
FRAMES="${EXPJ_FRAMES:-128}"
N_LIMIT="${EXPJ_N_LIMIT:-0}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG="$QWEN_DIR/results/expJ_overnight.progress.log"
mkdir -p "$QWEN_DIR/results"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

MODEL_SHORT="${MODEL##*/}"
CALIB_NPZ="$QWEN_DIR/calibration/expJ_kcalib_${MODEL_SHORT}_frames${FRAMES}.npz"
EXISTING_F_CALIB="$QWEN_DIR/calibration/expF_kcalib_${MODEL_SHORT}_frames64.npz"
STAGE1_OUT="$QWEN_DIR/results/expJ_xmodal_stage1_seed${SEED}.jsonl"

log "EXP J OVERNIGHT START model=$MODEL seed=$SEED frames=$FRAMES limit=$N_LIMIT"
log "  GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

# ================================================================
# Phase 1 — smoke (with existing F-suite calib NPZ for kernel sanity)
# ================================================================
log "--- PHASE 1: smoke ---"
SMOKE_CALIB=""
if [ -f "$EXISTING_F_CALIB" ]; then
  SMOKE_CALIB="--calib_npz $EXISTING_F_CALIB"
  log "  using existing F-suite calib for Phase B kernel sanity: $EXISTING_F_CALIB"
fi
python3 -u expJ_smoke.py --model "$MODEL" $SMOKE_CALIB 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "FAIL smoke (rc=$rc) — aborting overnight pipeline"
  exit $rc
fi
log "  smoke OK"

# ================================================================
# Phase 2 — calibration
# ================================================================
log "--- PHASE 2: cross-modal calibration ---"
if [ -f "$CALIB_NPZ" ]; then
  log "  calib NPZ already exists at $CALIB_NPZ; skipping"
else
  CALIB_ARGS=(--model "$MODEL" --frames "$FRAMES")
  if [ "$N_LIMIT" -gt 0 ]; then
    CALIB_ARGS+=(--limit "$N_LIMIT")
  fi
  python3 -u expJ_calibrate.py "${CALIB_ARGS[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL calibration (rc=$rc) — aborting"
    exit $rc
  fi
fi
if [ ! -f "$CALIB_NPZ" ]; then
  log "FAIL calibration NPZ not produced at $CALIB_NPZ — aborting"
  exit 3
fi
log "  calibration OK -> $CALIB_NPZ"

# ================================================================
# Phase 3 — Stage 1 forward pass (n=64 fresh seed=2)
# ================================================================
log "--- PHASE 3: Stage 1 (n=64, seed=$SEED, 128f, 15 conditions) ---"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
STAGE_ARGS=(--stage 1 --seed "$SEED" --calib_npz "$CALIB_NPZ")
if [ "$N_LIMIT" -gt 0 ]; then
  STAGE_ARGS+=(--limit "$N_LIMIT")
fi
python3 -u expJ_xmodal_outlier.py "${STAGE_ARGS[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "FAIL Stage 1 (rc=$rc) — aborting"
  exit $rc
fi
if [ ! -f "$STAGE1_OUT" ]; then
  log "FAIL Stage 1 JSONL not produced at $STAGE1_OUT — aborting"
  exit 4
fi
log "  Stage 1 OK -> $STAGE1_OUT"

# ================================================================
# Phase 4 — Analyze Stage 1
# ================================================================
log "--- PHASE 4: analyze Stage 1 ---"
python3 -u expJ_analyze.py --stage 1 --seed "$SEED" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "WARN analyze (rc=$rc); manual review required"
fi
log "  analyze OK -> see expJ_summary_stage1.md, expJ_paired_stage1.md, "
log "                       expJ_verdict_matrix_stage1.md, expJ_promote_stage1.json"

log "EXP J OVERNIGHT DONE — review outputs in $QWEN_DIR/results/"
log "Stage 3 launch (next day):"
log "  python3 expJ_xmodal_outlier.py --stage 3 --seed $SEED --calib_npz $CALIB_NPZ"
