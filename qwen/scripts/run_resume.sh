#!/usr/bin/env bash
# Resume the pipeline after A1 succeeded but A2 OOM'd:
#   1. Run A2..A8 with --append (preserves the existing A1 rollouts JSONL)
#   2. Run calibration (32 frames for memory headroom)
#   3. Run Exp B (6 conditions, excluding B3/B8 which need eager attention)
#   4. Pareto plot
#
# Memory hygiene: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (avoids
# fragmentation OOMs against tlandeg's co-tenant on GPU 0) plus per-5-item
# torch.cuda.empty_cache() inside run_inference.py.
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
EXP_FRAMES="${PIPELINE_FRAMES:-64}"
CAL_FRAMES="${CAL_FRAMES:-32}"
AVG="${PIPELINE_AVG:-3.0}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LOG="$QWEN_DIR/results/resume.progress.log"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "RESUME START model=$MODEL exp_frames=$EXP_FRAMES cal_frames=$CAL_FRAMES cuda=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

# STEP A: A2..A8 with --append
log "STEP A Experiment A resume (conditions A2..A8, --append)"
python3 -u expA_baseline.py --model "$MODEL" --frames "$EXP_FRAMES" --append \
    --conditions A2_W4fake_BF16KV A3_AWQ_BF16KV A4_BF16_FP8KV A5_BF16_INT4KV \
                 A6_BF16_INT4K_INT8V A7_BF16_INT2KV A8_AWQ_INT4KV \
    --progress_every 5 --summary_every 20 \
    2>&1 | tee -a "$LOG"
log "STEP A done"

# STEP B: calibration with eager attention + chunked entropy hook + 32 frames
THRESH_FILE="$QWEN_DIR/calibration/thresholds_${MODEL##*/}_avg${AVG}_frames${CAL_FRAMES}.json"
if [ ! -f "$THRESH_FILE" ]; then
    log "STEP B calibrating ($CAL_FRAMES frames, eager attention, chunked entropy)"
    python3 -u calibrate.py --model "$MODEL" --frames "$CAL_FRAMES" --target_avg_bits "$AVG" \
        --progress_every 5 --snapshot_every 20 \
        2>&1 | tee -a "$LOG"
fi
log "STEP B done; thresholds at $THRESH_FILE"

# STEP C: Exp B (6 conditions)
log "STEP C Experiment B (6 conditions × $EXP_FRAMES frames × 200 eval items, avg=$AVG)"
python3 -u expB_attnentropy.py --model "$MODEL" --frames "$EXP_FRAMES" --target_avg_bits "$AVG" \
    --thresholds "$THRESH_FILE" \
    --conditions B0 B1 B2 B4 B6 B7 \
    --progress_every 5 --summary_every 20 \
    2>&1 | tee -a "$LOG"
log "STEP C done"

# STEP D: Pareto plot
log "STEP D Pareto plot"
python3 -u expB_pareto_plot.py \
    --jsonl "$QWEN_DIR/results/expA_rollouts_${MODEL##*/}.jsonl" \
            "$QWEN_DIR/results/expB_rollouts_${MODEL##*/}_avg${AVG}.jsonl" \
    --frames "$EXP_FRAMES" \
    2>&1 | tee -a "$LOG"

log "RESUME DONE — results in $QWEN_DIR/results/, plots in $QWEN_DIR/plots/"
