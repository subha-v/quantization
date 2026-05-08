#!/usr/bin/env bash
# Experiment E1 — Text-K slice ablation orchestrator.
#
# Subcommands:
#   smoke    — 5-item correctness checks; halts on failure
#   passA    — fixed-slice conditions E1.2..E1.8 on 200 items
#   passB    — controls E1.9 random + E1.10 K-residual at global median budget
#   analyze  — produce expE1_summary.md / expE1_pair_analysis.md
#   full     — smoke -> passA -> passB -> analyze
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
FRAMES="${EXPE1_FRAMES:-64}"
N_SMOKE="${EXPE1_N_SMOKE:-5}"
N_LIMIT="${EXPE1_N_LIMIT:-0}"     # 0 = full eval split (200)
SEEDS="${EXPE1_SEEDS:-0 1 2}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOG="$QWEN_DIR/results/expE1_pipeline.progress.log"
mkdir -p "$QWEN_DIR/results"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

phase_smoke() {
  log "EXP E1 SMOKE START model=$MODEL frames=$FRAMES n_items=$N_SMOKE"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
  python3 -u expE1_smoke.py --model "$MODEL" --frames "$FRAMES" --n_items "$N_SMOKE" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "EXP E1 SMOKE FAIL (rc=$rc) — pipeline halted"
    return $rc
  fi
  log "EXP E1 SMOKE OK — see $QWEN_DIR/results/expE1_smoke.md"
}

phase_passA() {
  log "EXP E1 PASS A START model=$MODEL frames=$FRAMES limit=$N_LIMIT"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
  local args=(--model "$MODEL" --frames "$FRAMES" --phase passA)
  if [ "$N_LIMIT" != "0" ]; then args+=(--limit "$N_LIMIT"); fi
  python3 -u expE1_text_slice_ablation.py "${args[@]}" --progress_every 5 \
      2>&1 | tee -a "$LOG"
  log "EXP E1 PASS A DONE"
}

phase_passB() {
  log "EXP E1 PASS B START model=$MODEL frames=$FRAMES limit=$N_LIMIT seeds=$SEEDS"
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"
  local args=(--model "$MODEL" --frames "$FRAMES" --phase passB --seeds $SEEDS --append)
  if [ "$N_LIMIT" != "0" ]; then args+=(--limit "$N_LIMIT"); fi
  python3 -u expE1_text_slice_ablation.py "${args[@]}" --progress_every 5 \
      2>&1 | tee -a "$LOG"
  log "EXP E1 PASS B DONE"
}

phase_analyze() {
  log "EXP E1 ANALYZE START"
  python3 -u expD_analyze.py 2>&1 | tee -a "$LOG"
  log "EXP E1 ANALYZE DONE — see $QWEN_DIR/results/expE1_summary.md, expE1_pair_analysis.md"
}

case "${1:-full}" in
  smoke)   phase_smoke ;;
  passA)   phase_passA ;;
  passB)   phase_passB ;;
  analyze) phase_analyze ;;
  full)
    phase_smoke || exit $?
    phase_passA
    phase_passB
    phase_analyze
    ;;
  *)
    echo "Usage: $0 {smoke|passA|passB|analyze|full}" >&2
    exit 2
    ;;
esac
