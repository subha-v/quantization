#!/usr/bin/env bash
# Experiment D — Evidence-Window + Cross-Modal Visual-Key Diagnostic
# Subcommands: smoke | d0 | d1 | analyze | full
#   smoke   — 20-item correctness checks (visual span, V3K mask, frame removal)
#   d0      — 200-item BF16 eager + 7 frame-restriction conditions per item
#   d1      — 200-item cross-modal K/V quantization conditions (uses D0 windows)
#   analyze — produce expD0_summary.md / expD1_summary.md / expD_combined_analysis.md
#   full    — runs all of the above sequentially, halting on smoke failure
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
FRAMES="${EXPD_FRAMES:-64}"
WINDOWS="${EXPD_WINDOWS:-8}"
SEEDS="${EXPD_SEEDS:-0 1 2}"
N_SMOKE="${EXPD_N_SMOKE:-20}"
N_LIMIT="${EXPD_N_LIMIT:-0}"     # 0 = full eval split (200)

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG="$QWEN_DIR/results/expD_pipeline.progress.log"
mkdir -p "$QWEN_DIR/results"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

phase_smoke() {
  log "EXP D SMOKE START model=$MODEL frames=$FRAMES windows=$WINDOWS n_items=$N_SMOKE"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
  python3 -u expD_smoke.py --model "$MODEL" --frames "$FRAMES" --windows "$WINDOWS" \
      --n_items "$N_SMOKE" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "EXP D SMOKE FAIL (rc=$rc) — pipeline halted"
    return $rc
  fi
  log "EXP D SMOKE OK — see $QWEN_DIR/results/expD_smoke.md"
}

phase_d0() {
  log "EXP D0 START model=$MODEL frames=$FRAMES windows=$WINDOWS limit=$N_LIMIT seeds=$SEEDS"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
  local args=(--model "$MODEL" --frames "$FRAMES" --windows "$WINDOWS" --seeds $SEEDS)
  if [ "$N_LIMIT" != "0" ]; then args+=(--limit "$N_LIMIT"); fi
  python3 -u expD0_evidence_diagnostic.py "${args[@]}" --progress_every 5 \
      2>&1 | tee -a "$LOG"
  log "EXP D0 DONE"
}

phase_d1() {
  log "EXP D1 START model=$MODEL frames=$FRAMES windows=$WINDOWS limit=$N_LIMIT seeds=$SEEDS"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
  local args=(--model "$MODEL" --frames "$FRAMES" --windows "$WINDOWS" --seeds $SEEDS)
  if [ "$N_LIMIT" != "0" ]; then args+=(--limit "$N_LIMIT"); fi
  python3 -u expD1_crossmodal_kv.py "${args[@]}" --progress_every 5 \
      2>&1 | tee -a "$LOG"
  log "EXP D1 DONE"
}

phase_analyze() {
  log "EXP D ANALYZE START"
  python3 -u expD_analyze.py 2>&1 | tee -a "$LOG"
  log "EXP D ANALYZE DONE — see $QWEN_DIR/results/expD0_summary.md, expD1_summary.md, expD_combined_analysis.md"
}

case "${1:-full}" in
  smoke)   phase_smoke ;;
  d0)      phase_d0 ;;
  d1)      phase_d1 ;;
  analyze) phase_analyze ;;
  full)
    phase_smoke || exit $?
    phase_d0
    phase_d1
    phase_analyze
    ;;
  *)
    echo "Usage: $0 {smoke|d0|d1|analyze|full}" >&2
    exit 2
    ;;
esac
