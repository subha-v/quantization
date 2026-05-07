#!/usr/bin/env bash
# Experiment B online precision-need routing pipeline.
#   Step 1: diagnostic pass on cal + eval (BF16 eager forward + uniform INT4/INT2 refs)
#   Step 2: aggregate static risk from cal-only entropy
#   Step 3: routed eval for B2 (3 seeds), B4, B6, B7, B8, B9 — and optionally B10
#   Step 4: build summary metrics
#
# Designed to run in a tmux session that sticks around.
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
EXP_FRAMES="${PIPELINE_FRAMES:-64}"
DIAG_FRAMES="${DIAG_FRAMES:-$EXP_FRAMES}"
TARGET_AVG="${PIPELINE_TARGET_AVG:-4.0}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LOG="$QWEN_DIR/results/expB_online.progress.log"
DIAG="$QWEN_DIR/results/diagnostic_signals.jsonl"
STATIC="$QWEN_DIR/calibration/static_entropy_risk.json"
ROUTED="$QWEN_DIR/results/expB_online_rollouts.jsonl"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

mkdir -p "$QWEN_DIR/results" "$QWEN_DIR/calibration" "$QWEN_DIR/plots"

log "EXPB_ONLINE START model=$MODEL diag_frames=$DIAG_FRAMES exp_frames=$EXP_FRAMES target_avg=$TARGET_AVG cuda=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG" || true

# ---- STEP 1: diagnostic pass (cal + eval) ----
if [ ! -f "$DIAG" ] || [ "${REDO_DIAG:-0}" = "1" ]; then
    log "STEP 1 diagnostic pass on cal+eval (BF16 eager + uniform INT4/INT2 refs)"
    [ "${REDO_DIAG:-0}" = "1" ] && rm -f "$DIAG"
    python3 -u diagnostic_pass.py --model "$MODEL" --frames "$DIAG_FRAMES" \
        --splits cal eval --out "$DIAG" --progress_every 5 \
        2>&1 | tee -a "$LOG"
fi
log "STEP 1 done; diagnostic at $DIAG"

# ---- STEP 2: aggregate static risk (cal-only) ----
if [ ! -f "$STATIC" ] || [ "${REDO_DIAG:-0}" = "1" ]; then
    log "STEP 2 aggregating static_entropy_risk from cal rows"
    python3 -u precision_need_scoring.py --diagnostic "$DIAG" --out "$STATIC" \
        2>&1 | tee -a "$LOG"
fi
log "STEP 2 done; static_risk at $STATIC"

# ---- STEP 3: routed eval for the 6 main conditions ----
log "STEP 3 routed eval (target_avg=$TARGET_AVG, eval_frames=$EXP_FRAMES)"
python3 -u expB_online.py --model "$MODEL" --frames "$EXP_FRAMES" \
    --diagnostic "$DIAG" --static_risk "$STATIC" --out "$ROUTED" \
    --target_avg_bits "$TARGET_AVG" \
    --conditions B2_Random B4_MEDA B6_StaticEntropy B7_FlippedEntropy \
                 B8_OnlineResidual B9_OnlineNeed_Static \
    --seeds 0 1 2 \
    --progress_every 5 \
    2>&1 | tee -a "$LOG"
log "STEP 3 done"

# ---- STEP 4 (optional): B10 OnlineNeed-AQ ----
if [ "${RUN_B10:-1}" = "1" ]; then
    log "STEP 4 B10 OnlineNeed-AQ (diagnostic upper bound)"
    python3 -u expB_online.py --model "$MODEL" --frames "$EXP_FRAMES" \
        --diagnostic "$DIAG" --static_risk "$STATIC" --out "$ROUTED" --append \
        --target_avg_bits "$TARGET_AVG" \
        --conditions B10_OnlineNeed_AQ \
        --progress_every 5 \
        2>&1 | tee -a "$LOG"
fi
log "STEP 4 done"

log "EXPB_ONLINE DONE — results in $QWEN_DIR/results/, plots in $QWEN_DIR/plots/"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG" || true
