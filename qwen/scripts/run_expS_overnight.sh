#!/usr/bin/env bash
# Experiment S — Sidecode bit-ladder.
#
# Driven by the Exp R finding that SJ (J12 = F9 + INT8 sidecode) numerically
# beat F9 dense by +3.5 pp at 4.25 vs 4.75 KV bits on the multi-image 336° slice.
# Exp S asks two follow-up questions:
#
#   Phase 0 (no-GPU reanalysis on existing expR_rollouts_C.jsonl):
#     - Is SJ vs F9 paired-significant or just aggregate-lucky?
#     - Is TextOnly vs F9 paired-significant?
#     - How does SJ stack up against BF16 ceiling / TextOnly / static S12?
#
#   Phase 1 (sidecode bit-ladder on the same n=84 multi-image slice at 336°):
#     S0 BF16, S1 F4, S2 F9, S3 SJ, S4 top16 INT7, S5 top16 INT6, S6 top16 INT5,
#     S7 top24 INT6, S8 top32 INT6, S9 TextOnly-SJ
#     -> tests whether sidecode can drop below SJ's 4.25 KV bits while keeping
#        F9-tier accuracy.
#
# Env vars:
#   PIPELINE_MODEL       default Qwen/Qwen2.5-VL-7B-Instruct
#   CUDA_VISIBLE_DEVICES required
#   QWEN_VENV            override venv; default /data/subha2/experiments/qwen_venv
#   EXPS_SKIP_SMOKE      set to 1 to bypass smoke
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

LOG="$QWEN_DIR/results/expS_overnight.progress.log"
RESULTS_DIR="$QWEN_DIR/results"
CALIB_DIR="$QWEN_DIR/calibration"
mkdir -p "$RESULTS_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

MODEL_SHORT="${MODEL##*/}"
SLICE_A_NPZ="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_seed0.npz"
EXPR_C_JSONL="$RESULTS_DIR/expR_rollouts_C.jsonl"
S_JSONL="$RESULTS_DIR/expS_rollouts_phase1.jsonl"

log "EXP S START model=$MODEL"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

if [ ! -f "$SLICE_A_NPZ" ]; then
  log "FAIL calibration NPZ missing: $SLICE_A_NPZ"
  exit 2
fi
log "  calib OK -> $SLICE_A_NPZ"

# ================================================================
# Phase 0 — Reanalyze existing expR_rollouts_C.jsonl with new SJ-anchored
# paired tests (added to pairs_slice_c in expQ_analyze.py). No GPU.
# ================================================================
log "--- PHASE 0: reanalyze Exp R Sub-exp C with new SJ-anchored pairs ---"
if [ -f "$EXPR_C_JSONL" ]; then
  python3 -u expQ_analyze.py --slice C \
      --in-jsonl "$EXPR_C_JSONL" \
      --out-summary "$RESULTS_DIR/expR_summary_sliceC.md" \
      --out-paired  "$RESULTS_DIR/expR_paired_sliceC.md" \
      --out-verdict "$RESULTS_DIR/expR_verdict_matrix_sliceC.md" \
      --out-branch  "$RESULTS_DIR/expR_branch_sliceC.json" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "WARN Phase 0 reanalyze (rc=$rc) — continuing"
  fi
  log "  Phase 0 OK -> updated expR_paired_sliceC.md with SJ-anchored pairs"
else
  log "WARN $EXPR_C_JSONL not present; skipping Phase 0 reanalyze"
fi

# ================================================================
# Phase 1 — Smoke + main ladder run.
# ================================================================
if [ "${EXPS_SKIP_SMOKE:-0}" = "1" ]; then
  log "--- PHASE 1 SMOKE SKIPPED ---"
else
  log "--- PHASE 1 smoke: n=3 short bucket (S0/S2/S3 + new ladder kcfgs build) ---"
  python3 -u expQ_smoke.py \
      --model "$MODEL" \
      --n-items 3 \
      --bucket short \
      --task retrieval-image \
      --calib-npz "$SLICE_A_NPZ" \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --out "$RESULTS_DIR/expS_smoke.md" \
      --out-jsonl "$RESULTS_DIR/expS_smoke.jsonl" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL smoke (rc=$rc)"
    exit $rc
  fi
  log "  smoke OK (reuses Exp Q assertions A-I; ladder kcfgs are exercised in main)"
fi

log "--- PHASE 1: sidecode bit-ladder S0..S9 × n=84 multi-image at 336° ---"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
python3 -u expQ_driver.py \
    --model "$MODEL" \
    --task retrieval-image \
    --calib-npz "$SLICE_A_NPZ" \
    --use-full-pool \
    --min-num-images 8 \
    --max-pixels-context $((336*336)) \
    --max-pixels-choices $((336*336)) \
    --exp-s-ladder \
    --include-choice-routing \
    --out-jsonl "$S_JSONL" \
    --out-summary "$RESULTS_DIR/expS_summary_phase1.md" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "FAIL Phase 1 main run (rc=$rc)"
  exit $rc
fi
log "  Phase 1 OK -> $S_JSONL"

# ================================================================
# Phase 1 analyze (--slice S uses pairs_slice_s with S-condition names).
# ================================================================
log "--- PHASE 1 analyze ---"
python3 -u expQ_analyze.py --slice S \
    --in-jsonl "$S_JSONL" \
    --out-summary "$RESULTS_DIR/expS_summary_phase1.md" \
    --out-paired  "$RESULTS_DIR/expS_paired_phase1.md" \
    --out-verdict "$RESULTS_DIR/expS_verdict_phase1.md" \
    --out-branch  "$RESULTS_DIR/expS_branch_phase1.json" \
    2>&1 | tee -a "$LOG"
log "  analyze OK -> see $RESULTS_DIR/expS_*"

log "EXP S DONE — review outputs in $RESULTS_DIR/expS_* and $RESULTS_DIR/expR_paired_sliceC.md"
