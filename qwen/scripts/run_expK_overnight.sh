#!/usr/bin/env bash
# Experiment K — Balanced Cross-Modal Sidecode Replication orchestrator.
#
# Runs the K-suite across 3 seeds (seed=1 first as the harder split, then
# seed=0, then seed=2 rerun) using the existing Exp J cross-modal calibration
# NPZ. No additional calibration needed.
#
# Phases:
#   Phase 0: Phase A smoke (synthetic kernel + bits accounting)
#   Phase 1: Stage 3 on seed=1
#   Phase 2: Analyze seed=1
#   Phase 3: Stage 3 on seed=0
#   Phase 4: Analyze seed=0
#   Phase 5: Stage 3 on seed=2 (rerun for triangulation)
#   Phase 6: Analyze seed=2
#
# Halts on any phase failure.
#
# Env vars:
#   PIPELINE_MODEL       default Qwen/Qwen2.5-VL-7B-Instruct
#   EXPK_N_LIMIT         debug: limit eval items per seed (default 0 -> full 200)
#   CUDA_VISIBLE_DEVICES required at launch
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
N_LIMIT="${EXPK_N_LIMIT:-0}"
SEEDS=(${EXPK_SEEDS:-1 0 2})

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG="$QWEN_DIR/results/expK_overnight.progress.log"
mkdir -p "$QWEN_DIR/results"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

MODEL_SHORT="${MODEL##*/}"
CALIB_NPZ="$QWEN_DIR/calibration/expJ_kcalib_${MODEL_SHORT}_frames128.npz"

log "EXP K OVERNIGHT START model=$MODEL seeds=(${SEEDS[*]}) limit=$N_LIMIT"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

if [ ! -f "$CALIB_NPZ" ]; then
  log "FAIL calibration NPZ missing at $CALIB_NPZ. Run expJ_calibrate.py first."
  exit 2
fi
log "  using calibration NPZ: $CALIB_NPZ"

# ================================================================
# Phase 0 — Phase A smoke (no model, no GPU)
# ================================================================
log "--- PHASE 0: smoke ---"
python3 -u expK_smoke.py 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "FAIL smoke (rc=$rc) — aborting"
  exit $rc
fi
log "  smoke OK"

# ================================================================
# Per-seed phases
# ================================================================
PHASE_IDX=0
for SEED in "${SEEDS[@]}"; do
  PHASE_IDX=$((PHASE_IDX + 1))
  log "--- PHASE $PHASE_IDX: Stage 3 on seed=$SEED ---"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
  STAGE_ARGS=(--seed "$SEED" --calib_npz "$CALIB_NPZ")
  if [ "$N_LIMIT" -gt 0 ]; then
    STAGE_ARGS+=(--limit "$N_LIMIT")
  fi
  python3 -u expK_balanced_replication.py "${STAGE_ARGS[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL seed=$SEED Stage 3 (rc=$rc) — aborting"
    exit $rc
  fi
  STAGE3_OUT="$QWEN_DIR/results/expK_balanced_stage3_seed${SEED}.jsonl"
  if [ ! -f "$STAGE3_OUT" ]; then
    log "FAIL Stage 3 JSONL not produced at $STAGE3_OUT — aborting"
    exit 3
  fi
  log "  Stage 3 seed=$SEED OK -> $STAGE3_OUT"

  PHASE_IDX=$((PHASE_IDX + 1))
  log "--- PHASE $PHASE_IDX: analyze seed=$SEED ---"
  python3 -u expK_analyze.py --seed "$SEED" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "WARN analyze seed=$SEED (rc=$rc); continuing"
  fi
  log "  analyze seed=$SEED OK"
done

log "EXP K OVERNIGHT DONE — review outputs in $QWEN_DIR/results/expK_*"
log "Per-seed verdict matrices:"
for SEED in "${SEEDS[@]}"; do
  log "  seed=$SEED: $QWEN_DIR/results/expK_verdict_matrix_seed${SEED}.md"
done
