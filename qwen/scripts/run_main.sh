#!/usr/bin/env bash
# Main run: Qwen2.5-VL-7B on the 200-eval LongVideoBench split.
# Smoke test must pass first.
set -euo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-compute-apps=pid,used_memory,gpu_name --format=csv,noheader || true
fi
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an unused GPU before running main}"

MODEL="${MAIN_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
FRAMES="${MAIN_FRAMES:-64 128}"
TARGET_AVG="${MAIN_TARGET_AVG:-3.0}"

SPLIT_FILE="$QWEN_DIR/calibration/split_seed0.json"
if [ ! -f "$SPLIT_FILE" ]; then
  python data_longvideobench.py --seed 0
fi

# Calibration must complete before Exp B
THRESH_FILE="$QWEN_DIR/calibration/thresholds_${MODEL##*/}_avg${TARGET_AVG}_frames${FRAMES%% *}.json"
if [ ! -f "$THRESH_FILE" ]; then
  echo "[main] calibration missing -> running"
  python calibrate.py --model "$MODEL" --frames "${FRAMES%% *}" --target_avg_bits "$TARGET_AVG"
fi

PROGRESS_EVERY="${MAIN_PROGRESS:-10}"
SUMMARY_EVERY="${MAIN_SUMMARY_EVERY:-25}"

echo "[main] Experiment A: 8 conditions × frames=$FRAMES (progress every $PROGRESS_EVERY items, summary every $SUMMARY_EVERY)"
python expA_baseline.py --model "$MODEL" --frames $FRAMES \
    --progress_every "$PROGRESS_EVERY" --summary_every "$SUMMARY_EVERY"

echo "[main] Experiment B: 9 conditions @ avg=$TARGET_AVG × frames=$FRAMES"
python expB_attnentropy.py --model "$MODEL" --frames $FRAMES --target_avg_bits "$TARGET_AVG" \
    --progress_every "$PROGRESS_EVERY" --summary_every "$SUMMARY_EVERY"

# Pareto plot uses both expA and expB JSONLs at the larger frame budget
PLOT_FRAMES="${FRAMES##* }"
python expB_pareto_plot.py \
  --jsonl "$QWEN_DIR/results/expA_rollouts_${MODEL##*/}.jsonl" \
          "$QWEN_DIR/results/expB_rollouts_${MODEL##*/}_avg${TARGET_AVG}.jsonl" \
  --frames "$PLOT_FRAMES"

echo "[main][DONE] results/ + plots/ updated"
