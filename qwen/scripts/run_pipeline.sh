#!/usr/bin/env bash
# Full Qwen2.5-VL × LongVideoBench pipeline:
#   1. ensure split exists
#   2. calibrate on 100 cal items @ 64 frames
#   3. Experiment A: 8 conditions × 200 eval items @ 64 frames
#   4. Experiment B: 9 conditions × 200 eval items @ 64 frames, target avg 3.0
#   5. Pareto plot from A+B JSONLs
#
# Designed to run in a tmux session that sticks around (sleep at end). All
# progress lands in:
#   results/<exp>_*.progress.log    timestamped milestones
#   results/<exp>_*_summary.md      regenerated every 25 items
#   results/<exp>_*_rollouts.jsonl  per-item rows (line-buffered)
#
# Env overrides:
#   PIPELINE_MODEL  (default Qwen/Qwen2.5-VL-7B-Instruct)
#   PIPELINE_FRAMES (default 64)
#   PIPELINE_AVG    (default 3.0)
#   CUDA_VISIBLE_DEVICES (required; the calling tmux session must export it)
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
FRAMES="${PIPELINE_FRAMES:-64}"
AVG="${PIPELINE_AVG:-3.0}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching pipeline}"

mkdir -p "$QWEN_DIR/results" "$QWEN_DIR/calibration" "$QWEN_DIR/plots"
PIPELINE_LOG="$QWEN_DIR/results/pipeline.progress.log"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log()  { echo "[$(ts)] $*" | tee -a "$PIPELINE_LOG"; }

log "PIPELINE START model=$MODEL frames=$FRAMES avg=$AVG cuda=$CUDA_VISIBLE_DEVICES"

# Show GPU state at start
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$PIPELINE_LOG" || true

# Step 1: ensure split exists
SPLIT_FILE="$QWEN_DIR/calibration/split_seed0.json"
if [ ! -f "$SPLIT_FILE" ]; then
    log "STEP 1 building stratified split (seed=0)"
    python3 data_longvideobench.py --seed 0
fi
log "STEP 1 done; split at $SPLIT_FILE"

# Step 2: calibration
THRESH_FILE="$QWEN_DIR/calibration/thresholds_${MODEL##*/}_avg${AVG}_frames${FRAMES}.json"
if [ ! -f "$THRESH_FILE" ]; then
    log "STEP 2 calibrating ($MODEL, $FRAMES frames, target avg $AVG)"
    python3 -u calibrate.py --model "$MODEL" --frames "$FRAMES" --target_avg_bits "$AVG" \
        --progress_every 5 --snapshot_every 20 \
        2>&1 | tee -a "$PIPELINE_LOG"
fi
log "STEP 2 done; thresholds at $THRESH_FILE"

# Step 3: Experiment A
log "STEP 3 starting Experiment A (8 conditions × $FRAMES frames × 200 eval items)"
python3 -u expA_baseline.py --model "$MODEL" --frames "$FRAMES" \
    --progress_every 5 --summary_every 20 \
    2>&1 | tee -a "$PIPELINE_LOG"
log "STEP 3 done"

# Step 4: Experiment B (skip B3 AttnMass and B8 V3 — they need eager attention
# at runtime for live entropy/attention-mass which is a bigger refactor; the 6
# remaining conditions cover the headline V1/V2 vs uniform/random/MEDA Pareto)
log "STEP 4 starting Experiment B (6 conditions × $FRAMES frames × 200 eval items, avg=$AVG)"
python3 -u expB_attnentropy.py --model "$MODEL" --frames "$FRAMES" --target_avg_bits "$AVG" \
    --thresholds "$THRESH_FILE" \
    --conditions B0 B1 B2 B4 B6 B7 \
    --progress_every 5 --summary_every 20 \
    2>&1 | tee -a "$PIPELINE_LOG"
log "STEP 4 done"

# Step 5: Pareto plot
log "STEP 5 generating Pareto plot"
python3 -u expB_pareto_plot.py \
    --jsonl "$QWEN_DIR/results/expA_rollouts_${MODEL##*/}.jsonl" \
            "$QWEN_DIR/results/expB_rollouts_${MODEL##*/}_avg${AVG}.jsonl" \
    --frames "$FRAMES" \
    2>&1 | tee -a "$PIPELINE_LOG"

log "PIPELINE DONE — all results in $QWEN_DIR/results/, plots in $QWEN_DIR/plots/"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$PIPELINE_LOG" || true
