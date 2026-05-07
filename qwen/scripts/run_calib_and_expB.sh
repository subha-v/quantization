#!/usr/bin/env bash
# Recovery script: run calibration (with the fixed chunked entropy hook) +
# Exp B. Called manually after Exp A finishes if the original pipeline's
# calibration OOM'd.
#
# Reduces calibration frames to 32 to keep a wider memory headroom against
# tlandeg's co-tenant on GPU 0.
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
EXP_FRAMES="${PIPELINE_FRAMES:-64}"
CAL_FRAMES="${CAL_FRAMES:-32}"
AVG="${PIPELINE_AVG:-3.0}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"

LOG="$QWEN_DIR/results/calib_expB.progress.log"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "RECOVERY START model=$MODEL cal_frames=$CAL_FRAMES exp_frames=$EXP_FRAMES avg=$AVG cuda=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

THRESH_FILE="$QWEN_DIR/calibration/thresholds_${MODEL##*/}_avg${AVG}_frames${CAL_FRAMES}.json"
if [ ! -f "$THRESH_FILE" ]; then
    log "STEP A calibrating (frames=$CAL_FRAMES, eager attention, chunked entropy)"
    python3 -u calibrate.py --model "$MODEL" --frames "$CAL_FRAMES" --target_avg_bits "$AVG" \
        --progress_every 5 --snapshot_every 20 \
        2>&1 | tee -a "$LOG"
fi
log "STEP A done; thresholds at $THRESH_FILE"

log "STEP B Experiment B (6 conditions, frames=$EXP_FRAMES, avg=$AVG)"
python3 -u expB_attnentropy.py --model "$MODEL" --frames "$EXP_FRAMES" --target_avg_bits "$AVG" \
    --thresholds "$THRESH_FILE" \
    --conditions B0 B1 B2 B4 B6 B7 \
    --progress_every 5 --summary_every 20 \
    2>&1 | tee -a "$LOG"
log "STEP B done"

log "STEP C Pareto plot"
python3 -u expB_pareto_plot.py \
    --jsonl "$QWEN_DIR/results/expA_rollouts_${MODEL##*/}.jsonl" \
            "$QWEN_DIR/results/expB_rollouts_${MODEL##*/}_avg${AVG}.jsonl" \
    --frames "$EXP_FRAMES" \
    2>&1 | tee -a "$LOG"

log "RECOVERY DONE — results in $QWEN_DIR/results/, plots in $QWEN_DIR/plots/"
