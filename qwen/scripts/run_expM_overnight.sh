#!/usr/bin/env bash
# Experiment M — Matched-budget sidecode controls orchestrator.
#
# Phases:
#   Phase 0: Phase A smoke (synthetic, no GPU)
#   Phase 1: Stage 3 on the chosen seed (default seed=0 canonical F-suite split)
#   Phase 2: Analyze
#
# Reuses existing expJ calibration NPZ (seed=0 cal-100). No new calibration.
#
# Env:
#   EXPM_SEED            default 0 (canonical F-suite split; harder than seed=2)
#   PIPELINE_MODEL       default Qwen/Qwen2.5-VL-7B-Instruct
#   CUDA_VISIBLE_DEVICES required
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
SEED="${EXPM_SEED:-0}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG="$QWEN_DIR/results/expM_overnight.progress.log"
mkdir -p "$QWEN_DIR/results"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

MODEL_SHORT="${MODEL##*/}"
CALIB_NPZ="$QWEN_DIR/calibration/expJ_kcalib_${MODEL_SHORT}_frames128.npz"
STAGE3_OUT="$QWEN_DIR/results/expM_matched_stage3_seed${SEED}.jsonl"

log "EXP M START model=$MODEL seed=$SEED"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
if [ ! -f "$CALIB_NPZ" ]; then
  log "FAIL calibration NPZ missing at $CALIB_NPZ"
  exit 2
fi
log "  using calibration NPZ: $CALIB_NPZ"

# Phase 0 — smoke
log "--- PHASE 0: smoke ---"
python3 -u expM_smoke.py 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then log "FAIL smoke (rc=$rc)"; exit $rc; fi
log "  smoke OK"

# Phase 1 — Stage 3
log "--- PHASE 1: Stage 3 on seed=$SEED ---"
python3 -u expM_matched_budget.py --seed "$SEED" --calib_npz "$CALIB_NPZ" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then log "FAIL Stage 3 (rc=$rc)"; exit $rc; fi
if [ ! -f "$STAGE3_OUT" ]; then log "FAIL Stage 3 JSONL not produced at $STAGE3_OUT"; exit 3; fi
log "  Stage 3 OK -> $STAGE3_OUT"

# Phase 2 — Analyze
log "--- PHASE 2: analyze ---"
python3 -u expM_analyze.py --seed "$SEED" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then log "WARN analyze (rc=$rc)"; fi
log "  analyze OK"

log "EXP M DONE — review outputs in $QWEN_DIR/results/expM_*"
