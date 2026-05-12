#!/usr/bin/env bash
# Experiment G -- frame-scaling under fixed KV memory budget. Orchestrator.
#
# Subcommands:
#   smoke    -- Phase A (synthetic-tensor + post-process) + Phase B (live-model)
#               smoke checks. Requires CUDA_VISIBLE_DEVICES if model is set.
#   stage1   -- run G0..G6 on the n=64 balanced 16/bucket split (same items as
#               F-suite Stage 1 so the F4 anchor lines up exactly).
#   cascade  -- run G7 + G8 post-processes (no new model forwards). Requires
#               stage1 outputs to exist.
#   analyze  -- produce expG_summary_stage{N}.md, expG_paired_stage{N}.md,
#               expG_frontier_stage{N}.md, expG_verdict_matrix_stage{N}.md.
#   stage3   -- promoted-conditions n=200 run (manual; selected from verdict).
#   full     -- smoke -> stage1 -> cascade -> analyze
#
# Env vars:
#   PIPELINE_MODEL       default Qwen/Qwen2.5-VL-7B-Instruct
#   EXPG_N_LIMIT         debug: limit eval items count (default 0 -> full split)
#   EXPG_MIN_FREE_GB     min GPU free memory required before each phase (default 70)
#   EXPG_RIGOR_HIGH      set to 1 to enable smoke check 8 (~10 min subprocess)
#   CUDA_VISIBLE_DEVICES required at launch (must be set; we never default it)
#   PYTORCH_CUDA_ALLOC_CONF defaults to expandable_segments:True
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
N_LIMIT="${EXPG_N_LIMIT:-0}"
EXPG_MIN_FREE_GB="${EXPG_MIN_FREE_GB:-70}"
EXPG_RIGOR_HIGH="${EXPG_RIGOR_HIGH:-0}"
EXPG_STAGE_DEFAULT="${EXPG_STAGE_DEFAULT:-1}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG="$QWEN_DIR/results/expG_pipeline.progress.log"
mkdir -p "$QWEN_DIR/results"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

require_min_free_gb() {
  # Hard-fail if the chosen GPU has less than EXPG_MIN_FREE_GB free.
  local dev="${CUDA_VISIBLE_DEVICES%%,*}"
  local free_mib
  free_mib=$(nvidia-smi --id="${dev}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ' || echo 0)
  if [ -z "$free_mib" ] || ! [[ "$free_mib" =~ ^[0-9]+$ ]]; then
    log "WARN could not read memory.free for GPU $dev; skipping check"
    return 0
  fi
  local free_gb=$(( free_mib / 1024 ))
  if [ "$free_gb" -lt "$EXPG_MIN_FREE_GB" ]; then
    log "FAIL GPU $dev free=${free_gb}GiB < EXPG_MIN_FREE_GB=${EXPG_MIN_FREE_GB}GiB"
    return 2
  fi
  log "OK GPU $dev free=${free_gb}GiB >= ${EXPG_MIN_FREE_GB}GiB"
}

phase_smoke() {
  log "EXP G SMOKE START model=$MODEL min_free_gb=$EXPG_MIN_FREE_GB rigor_high=$EXPG_RIGOR_HIGH"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
  require_min_free_gb || return $?
  local args=(--model "$MODEL" --min_free_gb "$EXPG_MIN_FREE_GB")
  if [ "$EXPG_RIGOR_HIGH" = "1" ]; then
    args+=(--high_rigor)
  fi
  python3 -u expG_smoke.py "${args[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "EXP G SMOKE FAIL (rc=$rc) -- pipeline halted"
    return $rc
  fi
  log "EXP G SMOKE OK -- see $QWEN_DIR/results/expG_smoke.md"
}

run_stage() {
  local stage="$1"
  log "EXP G STAGE $stage START model=$MODEL limit=$N_LIMIT"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
  require_min_free_gb || return $?
  local args=(--model "$MODEL" --stage "$stage" --min_free_gb_256 "$EXPG_MIN_FREE_GB")
  if [ "$N_LIMIT" != "0" ]; then args+=(--limit "$N_LIMIT"); fi
  python3 -u expG_frame_scaling.py "${args[@]}" --progress_every 5 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  log "EXP G STAGE $stage DONE (rc=$rc)"
  return $rc
}

phase_stage1() { run_stage 1; }
phase_stage3() { run_stage 3; }

phase_cascade() {
  local stage="${1:-$EXPG_STAGE_DEFAULT}"
  log "EXP G CASCADE+TYPE_ADAPTIVE stage=$stage START"
  local in_jsonl="$QWEN_DIR/results/expG_frame_stage${stage}.jsonl"
  if [ ! -f "$in_jsonl" ]; then
    log "FAIL $in_jsonl missing -- run stage$stage first"
    return 2
  fi
  python3 -u expG_cascade.py \
    --in_jsonl "$in_jsonl" \
    --out_jsonl "$QWEN_DIR/results/expG_frame_stage${stage}_G7.jsonl" \
    --meta "$QWEN_DIR/results/expG_cascade_meta.json" \
    2>&1 | tee -a "$LOG"
  python3 -u expG_type_adaptive.py \
    --in_jsonl "$in_jsonl" \
    --out_jsonl "$QWEN_DIR/results/expG_frame_stage${stage}_G8.jsonl" \
    --meta "$QWEN_DIR/results/expG_qtype_meta.json" \
    --split_file "$QWEN_DIR/calibration/split_seed0_n64.json" \
    2>&1 | tee -a "$LOG"
  log "EXP G CASCADE+TYPE_ADAPTIVE stage=$stage DONE"
}

phase_analyze() {
  local stage="${1:-$EXPG_STAGE_DEFAULT}"
  log "EXP G ANALYZE stage=$stage START"
  python3 -u expG_analyze.py --stage "$stage" 2>&1 | tee -a "$LOG"
  log "EXP G ANALYZE stage=$stage DONE -- see "\
"expG_summary_stage${stage}.md, expG_paired_stage${stage}.md, "\
"expG_frontier_stage${stage}.md, expG_verdict_matrix_stage${stage}.md"
}

case "${1:-full}" in
  smoke)    phase_smoke ;;
  stage1)   phase_stage1 ;;
  stage3)   phase_stage3 ;;
  cascade)  phase_cascade "${2:-$EXPG_STAGE_DEFAULT}" ;;
  analyze)  phase_analyze "${2:-$EXPG_STAGE_DEFAULT}" ;;
  full)
    phase_smoke || exit $?
    phase_stage1 || exit $?
    phase_cascade 1 || exit $?
    phase_analyze 1
    log "FULL pipeline complete -- review verdict matrix to choose Stage 3 conditions"
    ;;
  *)
    echo "Usage: $0 {smoke|stage1|stage3|cascade [stage]|analyze [stage]|full}" >&2
    exit 2
    ;;
esac
