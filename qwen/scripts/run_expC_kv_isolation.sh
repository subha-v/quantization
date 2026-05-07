#!/usr/bin/env bash
# Experiment C — K/V isolation mini-sweep on Qwen2.5-VL-7B + LongVideoBench.
#
# Four conditions, each leaving one of K or V at BF16 and quantizing the other:
#   C2.1 K=BF16, V=INT4
#   C2.2 K=INT4, V=BF16
#   C2.3 K=BF16, V=INT2
#   C2.4 K=INT2, V=BF16
#
# Run on the first 100 stratified eval items (proportional across short / mid /
# long / very_long buckets — yields 17/17/33/33 = 100). Appends to the existing
# expA rollouts JSONL so summarize() regenerates a single table containing
# A1-A8 (n=200) and the four C2 conditions (n=100).
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
EXP_FRAMES="${PIPELINE_FRAMES:-64}"
N_ITEMS="${EXPC_N_ITEMS:-100}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

LOG="$QWEN_DIR/results/expC_kv_isolation.progress.log"
ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

log "EXP C START model=$MODEL frames=$EXP_FRAMES n=$N_ITEMS cuda=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

python3 -u expA_baseline.py --model "$MODEL" --frames "$EXP_FRAMES" --append \
    --conditions C2_1_BF16K_INT4V C2_2_INT4K_BF16V C2_3_BF16K_INT2V C2_4_INT2K_BF16V \
    --stratified_limit "$N_ITEMS" \
    --progress_every 5 --summary_every 20 \
    2>&1 | tee -a "$LOG"

log "EXP C DONE — appended to expA_rollouts_${MODEL##*/}.jsonl"
