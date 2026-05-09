#!/usr/bin/env bash
# Experiment F — Tiered K-quantizer screening orchestrator.
#
# Subcommands:
#   smoke    — Phase A (synthetic-tensor) + Phase B (live-model) smoke checks
#   calib    — capture per-(L, H_kv, channel) K & Q stats over the cal-100 split
#   stage0   — n=16 wiring smoke (do NOT interpret accuracy)
#   stage1   — n=64 screening of all 14 K-quantizer variants
#   stage2   — n=100 confirmation (promoted variants only)
#   stage3   — n=200 final (top survivors only)
#   analyze  — produce expF_summary_stage{N}.md + expF_verdict_matrix_stage{N}.md
#   full     — smoke -> calib -> stage0 -> stage1 -> analyze (does NOT run stage2/3)
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
FRAMES="${EXPF_FRAMES:-64}"
N_LIMIT="${EXPF_N_LIMIT:-0}"     # 0 = use stage's full split
EXPF_STAGE_DEFAULT="${EXPF_STAGE_DEFAULT:-1}"
EXPF_MIN_FREE_GB="${EXPF_MIN_FREE_GB:-60}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG="$QWEN_DIR/results/expF_pipeline.progress.log"
mkdir -p "$QWEN_DIR/results"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

require_min_free_gb() {
  # Hard-fail if the chosen GPU has less than EXPF_MIN_FREE_GB free.
  # Reads memory.free for the device specified by CUDA_VISIBLE_DEVICES.
  local dev="${CUDA_VISIBLE_DEVICES%%,*}"
  local free_mib
  free_mib=$(nvidia-smi --id="${dev}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ' || echo 0)
  if [ -z "$free_mib" ] || ! [[ "$free_mib" =~ ^[0-9]+$ ]]; then
    log "WARN could not read memory.free for GPU $dev; skipping check"
    return 0
  fi
  local free_gb=$(( free_mib / 1024 ))
  if [ "$free_gb" -lt "$EXPF_MIN_FREE_GB" ]; then
    log "FAIL GPU $dev free=${free_gb}GiB < EXPF_MIN_FREE_GB=${EXPF_MIN_FREE_GB}GiB"
    return 2
  fi
  log "OK GPU $dev free=${free_gb}GiB >= ${EXPF_MIN_FREE_GB}GiB"
}

phase_smoke() {
  log "EXP F SMOKE START model=$MODEL frames=$FRAMES"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
  require_min_free_gb || return $?
  local args=(--model "$MODEL" --frames "$FRAMES")
  local calib_npz="$QWEN_DIR/calibration/expF_kcalib_${MODEL##*/}_frames${FRAMES}.npz"
  if [ -f "$calib_npz" ]; then
    args+=(--calib_file "$calib_npz")
  else
    args+=(--use_synthetic_calib)
    log "calibration NPZ missing; smoke uses synthetic Phase A only"
  fi
  python3 -u expF_smoke.py "${args[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "EXP F SMOKE FAIL (rc=$rc) — pipeline halted"
    return $rc
  fi
  log "EXP F SMOKE OK — see $QWEN_DIR/results/expF_smoke.md"
}

phase_calib() {
  log "EXP F CALIB START model=$MODEL frames=$FRAMES"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
  require_min_free_gb || return $?
  python3 -u expF_calibrate.py --model "$MODEL" --frames "$FRAMES" \
      --progress_every 5 2>&1 | tee -a "$LOG"
  log "EXP F CALIB DONE — see $QWEN_DIR/calibration/expF_kcalib_*.{json,npz}"
}

run_stage() {
  local stage="$1"
  log "EXP F STAGE $stage START model=$MODEL frames=$FRAMES limit=$N_LIMIT"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
  require_min_free_gb || return $?
  local args=(--model "$MODEL" --frames "$FRAMES" --stage "$stage")
  if [ "$N_LIMIT" != "0" ]; then args+=(--limit "$N_LIMIT"); fi
  python3 -u expF_kquant_screen.py "${args[@]}" --progress_every 5 2>&1 | tee -a "$LOG"
  log "EXP F STAGE $stage DONE"
}

phase_stage0() { run_stage 0; }
phase_stage1() { run_stage 1; }
phase_stage2() { run_stage 2; }
phase_stage3() { run_stage 3; }

phase_analyze() {
  local stage="${1:-$EXPF_STAGE_DEFAULT}"
  log "EXP F ANALYZE stage=$stage START"
  python3 -u expF_analyze.py --stage "$stage" 2>&1 | tee -a "$LOG"
  log "EXP F ANALYZE stage=$stage DONE — see $QWEN_DIR/results/expF_summary_stage${stage}.md, expF_verdict_matrix_stage${stage}.md"
}

case "${1:-full}" in
  smoke)    phase_smoke ;;
  calib)    phase_calib ;;
  stage0)   phase_stage0 ;;
  stage1)   phase_stage1 ;;
  stage2)   phase_stage2 ;;
  stage3)   phase_stage3 ;;
  analyze)  phase_analyze "${2:-$EXPF_STAGE_DEFAULT}" ;;
  full)
    phase_smoke || exit $?
    phase_calib || exit $?
    phase_smoke || exit $?         # re-run smoke now that calib exists; covers Phase B F10
    phase_stage0 && phase_analyze 0
    phase_stage1 && phase_analyze 1
    log "FULL pipeline complete — review verdict matrix to choose Stage 2 conditions"
    ;;
  *)
    echo "Usage: $0 {smoke|calib|stage0|stage1|stage2|stage3|analyze [stage]|full}" >&2
    exit 2
    ;;
esac
