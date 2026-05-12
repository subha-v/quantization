#!/usr/bin/env bash
# Exp G overnight queue -- runs after the Stage 3 pipeline finishes.
#
# Phases:
#   A: Stage-3 G9_F9_192f at n=200 (~30 min)
#   B: Stage-3 cascade controls (3 random seeds + oracle), all post-process
#   C: Stage-3 re-analyze with all new conditions
#   D: Stage-1 H new forwards (H3/H4/H5/H6) at n=64 (~35 min)
#   E: Stage-1 re-analyze with all new conditions
#
# Failures in any phase emit a warning but do NOT halt later phases (uses ;
# chaining, not &&) so a single bad sidecar doesn't kill the H Stage-1 run.
#
# Env (inherits from the parent shell; the queueing wrapper sets these):
#   PIPELINE_MODEL          default Qwen/Qwen2.5-VL-7B-Instruct
#   EXPG_MIN_FREE_GB        default 30 (was lowered for the 256f tier)
#   CUDA_VISIBLE_DEVICES    must be set
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
EXPG_MIN_FREE_GB="${EXPG_MIN_FREE_GB:-30}"
PYTHON="${PYTHON:-/data/subha2/experiments/qwen_venv/bin/python3}"
RESULTS="$QWEN_DIR/results"
LOG="$RESULTS/expG_overnight.progress.log"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$RESULTS"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "=== overnight queue START model=$MODEL min_free_gb=$EXPG_MIN_FREE_GB ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

# Phase A: G9_F9_192f at n=200 ----------------------------------------------
phase_a() {
  log "PHASE A start -- G9_F9_192f at n=200"
  "$PYTHON" -u expG_frame_scaling.py \
    --model "$MODEL" \
    --stage 3 \
    --conditions G9_F9_192f \
    --append \
    --min_free_gb_256 "$EXPG_MIN_FREE_GB" \
    --progress_every 10 \
    2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  log "PHASE A done (rc=$rc)"
  return 0
}

# Phase B: Stage-3 cascade selection-mode controls --------------------------
phase_b() {
  log "PHASE B start -- F9 cascade controls (random x3 + oracle)"
  local in_jsonl="$RESULTS/expG_frame_stage3.jsonl"
  if [ ! -f "$in_jsonl" ]; then
    log "PHASE B skipped: $in_jsonl missing"
    return 0
  fi

  for seed in 0 1 2; do
    "$PYTHON" -u expG_cascade.py \
      --in_jsonl "$in_jsonl" \
      --first_pass G5_F9_128f --second_pass G6_F9_256f \
      --first_pass_frames 128 --second_pass_frames 256 \
      --target_avg_frames 192 \
      --selection_mode random --random_seed $seed \
      --stitched_name "G7_F9_CascadeRandomS${seed}" \
      --out_jsonl "$RESULTS/expG_frame_stage3_G7_random_s${seed}.jsonl" \
      --meta "$RESULTS/expG_cascade_f9_random_s${seed}_meta.json" \
      2>&1 | tee -a "$LOG"
  done

  "$PYTHON" -u expG_cascade.py \
    --in_jsonl "$in_jsonl" \
    --first_pass G5_F9_128f --second_pass G6_F9_256f \
    --first_pass_frames 128 --second_pass_frames 256 \
    --target_avg_frames 192 \
    --selection_mode oracle \
    --stitched_name "G7_F9_CascadeOracle" \
    --out_jsonl "$RESULTS/expG_frame_stage3_G7_oracle.jsonl" \
    --meta "$RESULTS/expG_cascade_f9_oracle_meta.json" \
    2>&1 | tee -a "$LOG"
  log "PHASE B done"
}

# Phase C: Stage-3 re-analyze with all new files ----------------------------
phase_c() {
  log "PHASE C start -- Stage 3 re-analyze with all new conditions"
  local extras=()
  for f in \
      expG_frame_stage3_G7f9.jsonl \
      expG_frame_stage3_G8f9.jsonl \
      expG_frame_stage3_G7_random_s0.jsonl \
      expG_frame_stage3_G7_random_s1.jsonl \
      expG_frame_stage3_G7_random_s2.jsonl \
      expG_frame_stage3_G7_oracle.jsonl \
      ; do
    if [ -f "$RESULTS/$f" ]; then
      extras+=("$RESULTS/$f")
    fi
  done
  "$PYTHON" -u expG_analyze.py --stage 3 --extra_jsonl "${extras[@]}" 2>&1 | tee -a "$LOG"
  log "PHASE C done; outputs: expG_summary_stage3.md, expG_paired_stage3.md, expG_frontier_stage3.md, expG_verdict_matrix_stage3.md"
}

# Phase D: H Stage-1 forwards ----------------------------------------------
phase_d() {
  log "PHASE D start -- H Stage 1 forwards (H3/H4/H5/H6) at n=64"
  "$PYTHON" -u expG_frame_scaling.py \
    --model "$MODEL" \
    --stage 1 \
    --conditions H3_KIVI_TempWin4_256f H4_KIVI_TempWin8_256f \
                 H5_KIVI_TokenBlock4_256f H6_KIVI_TempWin2_128f \
    --append \
    --min_free_gb_256 "$EXPG_MIN_FREE_GB" \
    --progress_every 5 \
    2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  log "PHASE D done (rc=$rc)"
}

# Phase E: Stage-1 re-analyze with H rows + existing F9 stitched files ------
phase_e() {
  log "PHASE E start -- Stage 1 re-analyze including H rows"
  local extras=()
  for f in \
      expG_frame_stage1_G7f9.jsonl \
      expG_frame_stage1_G8f9.jsonl \
      ; do
    if [ -f "$RESULTS/$f" ]; then
      extras+=("$RESULTS/$f")
    fi
  done
  "$PYTHON" -u expG_analyze.py --stage 1 --extra_jsonl "${extras[@]}" 2>&1 | tee -a "$LOG"
  log "PHASE E done; outputs: expG_summary_stage1.md, expG_paired_stage1.md, expG_frontier_stage1.md, expG_verdict_matrix_stage1.md"
}

# Run all phases sequentially. Use `;` so a fail in one phase doesn't kill the rest.
phase_a; phase_b; phase_c; phase_d; phase_e

log "=== overnight queue END ==="
