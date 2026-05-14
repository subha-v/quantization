#!/usr/bin/env bash
# Experiment T-mini — VLM page-aware KV formats on MM-NIAH.
#
# Tests PageLocal (per-page K scales) and PageSentinel (image-identity register)
# as a paper-worthy VLM-specific axis beyond the sidecode width family that
# Exps Q/R/S/T already exhausted.
#
# Phases:
#   Phase 0  CPU smoke for the new K-quantizer kinds (10 s)
#   Phase 1  retrieval-image, multi-image (num_images >= 8), n ~ 84, T0..T16
#   Phase 2  reasoning-image, multi-image (num_images >= 5), n <= 84, T0..T16
#   Phase 3  counting-image, multi-image (num_images >= 5), n <= 64, C0..C12
#            (multi-token generation + list-output parsing)
#   Phase 4  analyzer → markdown summaries + paired McNemar matrices
#
# Env vars:
#   PIPELINE_MODEL       default Qwen/Qwen2.5-VL-7B-Instruct
#   CUDA_VISIBLE_DEVICES required
#   QWEN_VENV            override venv; default /data/subha2/experiments/qwen_venv
#   EXPT_MINI_SKIP_SMOKE set to 1 to bypass smoke
#   EXPT_MINI_PHASES     subset of phases to run (default "0 1 2 3 4")

set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PHASES="${EXPT_MINI_PHASES:-0 1 2 3 4}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QWEN_VENV="${QWEN_VENV:-/data/subha2/experiments/qwen_venv}"
if [ -f "$QWEN_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$QWEN_VENV/bin/activate"
  echo "Activated venv: $QWEN_VENV (python=$(which python3))"
fi

LOG="$QWEN_DIR/results/expT_mini_overnight.progress.log"
RESULTS_DIR="$QWEN_DIR/results"
CALIB_DIR="$QWEN_DIR/calibration"
mkdir -p "$RESULTS_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }
have_phase() { case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }

MODEL_SHORT="${MODEL##*/}"
NPZ_RETRIEVAL="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_seed0.npz"
NPZ_REASONING="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_reasoning-image_seed0.npz"
NPZ_COUNTING="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_counting-image_seed0.npz"

JSONL_T1="$RESULTS_DIR/expT_mini_rollouts_retrieval-image.jsonl"
JSONL_T2="$RESULTS_DIR/expT_mini_rollouts_reasoning-image.jsonl"
JSONL_T3="$RESULTS_DIR/expT_mini_rollouts_counting-image.jsonl"

MD_T1="$RESULTS_DIR/expT_mini_summary_retrieval-image.md"
MD_T2="$RESULTS_DIR/expT_mini_summary_reasoning-image.md"
MD_T3="$RESULTS_DIR/expT_mini_summary_counting-image.md"

log "EXP T-mini START model=$MODEL phases='$PHASES'"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

# ================================================================
# Phase 0 — CPU smoke for new K-quantizer kinds + counting parser.
# ================================================================
if have_phase 0; then
  if [ "${EXPT_MINI_SKIP_SMOKE:-0}" = "1" ]; then
    log "--- PHASE 0 smoke SKIPPED ---"
  else
    log "--- PHASE 0: CPU smoke (new K-quantizer kinds + counting parser) ---"
    python3 -u expT_mini_smoke.py --out-md "$RESULTS_DIR/expT_mini_smoke.md" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
      log "FAIL smoke (rc=$rc) — aborting"
      exit $rc
    fi
    log "  smoke OK -> $RESULTS_DIR/expT_mini_smoke.md"
  fi
fi

# ================================================================
# Phase 1 — retrieval-image multi-image, T0..T16, n ~ 84.
# Uses pre-existing seed=0 NPZ.
# ================================================================
if have_phase 1; then
  if [ ! -f "$NPZ_RETRIEVAL" ]; then
    log "FAIL retrieval-image calibration NPZ missing: $NPZ_RETRIEVAL"
    exit 2
  fi
  log "--- PHASE 1: retrieval-image T0..T16, num_images>=8, n~84, 336^2 ---"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
  python3 -u expQ_driver.py \
      --model "$MODEL" \
      --task retrieval-image \
      --seed 0 \
      --calib-npz "$NPZ_RETRIEVAL" \
      --use-full-pool \
      --min-num-images 8 \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --exp-t-mini \
      --include-choice-routing \
      --out-jsonl "$JSONL_T1" \
      --out-summary "$MD_T1" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL Phase 1 (rc=$rc)"
    exit $rc
  fi
  log "  Phase 1 OK -> $JSONL_T1"
fi

# ================================================================
# Phase 2 — reasoning-image multi-image, T0..T16.
# Generates a fresh reasoning-image cal NPZ if it doesn't exist.
# ================================================================
if have_phase 2; then
  if [ ! -f "$NPZ_REASONING" ]; then
    log "--- PHASE 2a: generate reasoning-image cal-100 NPZ ---"
    python3 -u expP_calibrate.py \
        --model "$MODEL" \
        --task reasoning-image \
        --seed 0 \
        --max_q_per_item 256 \
        --n_outliers_top 16 \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ] || [ ! -f "$NPZ_REASONING" ]; then
      log "WARN reasoning-image calibration failed (rc=$rc); skipping Phase 2"
    fi
  fi
  if [ -f "$NPZ_REASONING" ]; then
    log "--- PHASE 2: reasoning-image T0..T16, num_images>=5, n<=84 ---"
    python3 -u expQ_driver.py \
        --model "$MODEL" \
        --task reasoning-image \
        --seed 0 \
        --calib-npz "$NPZ_REASONING" \
        --use-full-pool \
        --min-num-images 5 \
        --n-items 84 \
        --max-pixels-context $((336*336)) \
        --max-pixels-choices $((336*336)) \
        --exp-t-mini \
        --include-choice-routing \
        --out-jsonl "$JSONL_T2" \
        --out-summary "$MD_T2" \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
      log "WARN Phase 2 returned rc=$rc; continuing to Phase 3"
    else
      log "  Phase 2 OK -> $JSONL_T2"
    fi
  fi
fi

# ================================================================
# Phase 3 — counting-image multi-image, C0..C12.
# Generates a fresh counting-image cal NPZ if it doesn't exist.
# ================================================================
if have_phase 3; then
  if [ ! -f "$NPZ_COUNTING" ]; then
    log "--- PHASE 3a: generate counting-image cal-100 NPZ ---"
    python3 -u expP_calibrate.py \
        --model "$MODEL" \
        --task counting-image \
        --seed 0 \
        --max_q_per_item 256 \
        --n_outliers_top 16 \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ] || [ ! -f "$NPZ_COUNTING" ]; then
      log "WARN counting-image calibration failed (rc=$rc); skipping Phase 3"
    fi
  fi
  if [ -f "$NPZ_COUNTING" ]; then
    log "--- PHASE 3: counting-image C0..C12, num_images>=5, n<=64, max_new=96 ---"
    python3 -u expQ_driver.py \
        --model "$MODEL" \
        --task counting-image \
        --seed 0 \
        --calib-npz "$NPZ_COUNTING" \
        --use-full-pool \
        --min-num-images 5 \
        --n-items 64 \
        --max-pixels-context $((336*336)) \
        --max-pixels-choices $((336*336)) \
        --exp-t-mini-counting \
        --max-new-tokens-counting 160 \
        --out-jsonl "$JSONL_T3" \
        --out-summary "$MD_T3" \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
      log "WARN Phase 3 returned rc=$rc; continuing"
    else
      log "  Phase 3 OK -> $JSONL_T3"
    fi
  fi
fi

# ================================================================
# Phase 4 — analyzer: bucketed summary + paired McNemar matrices.
# ================================================================
if have_phase 4; then
  log "--- PHASE 4: analyzer ---"
  if [ -f "$JSONL_T1" ]; then
    python3 -u expT_mini_analyze.py \
        --in-jsonl "$JSONL_T1" \
        --out-md   "$RESULTS_DIR/expT_mini_summary_retrieval-image.md" \
        --task retrieval-image 2>&1 | tee -a "$LOG"
  fi
  if [ -f "$JSONL_T2" ]; then
    python3 -u expT_mini_analyze.py \
        --in-jsonl "$JSONL_T2" \
        --out-md   "$RESULTS_DIR/expT_mini_summary_reasoning-image.md" \
        --task reasoning-image 2>&1 | tee -a "$LOG"
  fi
  if [ -f "$JSONL_T3" ]; then
    python3 -u expT_mini_analyze.py \
        --in-jsonl "$JSONL_T3" \
        --out-md   "$RESULTS_DIR/expT_mini_summary_counting-image.md" \
        --task counting-image 2>&1 | tee -a "$LOG"
  fi
  log "  Phase 4 OK"
fi

log "EXP T-mini DONE — see $RESULTS_DIR/expT_mini_*.md"
