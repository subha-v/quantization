#!/usr/bin/env bash
# Experiment Q — FormatBook v2 on MM-NIAH (multi-image retrieval + reasoning-image).
#
# Phases:
#   A. Smoke (n=3 short bucket) — fail-fast on wiring.
#   B. Slice A main run: Q0..Q11 on retrieval-image multi-image (full pool,
#      --min-num-images 8) at 336x336 equal-res.
#   C. Analyze → emit expQ_branch_sliceA.json with branch flags.
#   D. Branch-conditional reseed: launch random-seed reseed conditions only if
#      Quest-vs-Random gap triggers the rule (>=2 pp gap or paired_net >= 5).
#      Priority: top-25 (Q7/Q8) > top-50 (Q4/Q5) > INT2 (Q10/Q11).
#   E. 448² mini-check: Q0/Q2/Q4 on n=32 at max_pixels=200704.
#   F. Final analyze + Slice B recommendation written to summary.
#
# Slice B (reasoning-image) is NOT auto-launched. Pass --slice-b to:
#   G. Reasoning-image calibration on cal-100 (~5 min).
#   H. Slice B smoke + main (R0..R8) at 336x336.
#
# Env vars:
#   PIPELINE_MODEL       default Qwen/Qwen2.5-VL-7B-Instruct
#   CUDA_VISIBLE_DEVICES required (per CLAUDE.md GPU-safety rule)
#   EXPQ_SKIP_SMOKE      set to 1 to bypass Phase A (for resume runs)
#   EXPQ_FORCE_SLICE_B   set to 1 to ignore branch recommendation and run Slice B

set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
RUN_SLICE_B=0
for arg in "$@"; do
  case "$arg" in
    --slice-b) RUN_SLICE_B=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Activate the project venv if it exists; otherwise rely on the inherited PATH.
# Override with QWEN_VENV=/path/to/venv to point at a custom interpreter.
QWEN_VENV="${QWEN_VENV:-/data/subha2/experiments/qwen_venv}"
if [ -f "$QWEN_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$QWEN_VENV/bin/activate"
  echo "Activated venv: $QWEN_VENV (python=$(which python3))"
fi

LOG="$QWEN_DIR/results/expQ_overnight.progress.log"
RESULTS_DIR="$QWEN_DIR/results"
CALIB_DIR="$QWEN_DIR/calibration"
mkdir -p "$RESULTS_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

MODEL_SHORT="${MODEL##*/}"
SLICE_A_NPZ="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_seed0.npz"
SLICE_A_JSONL="$RESULTS_DIR/expQ_rollouts_sliceA.jsonl"
SLICE_A_RES448_JSONL="$RESULTS_DIR/expQ_rollouts_sliceA_res448.jsonl"
SLICE_A_BRANCH="$RESULTS_DIR/expQ_branch_sliceA.json"
SLICE_B_NPZ="$CALIB_DIR/expQ_mmniah_reasoning-image_kcalib_${MODEL_SHORT}_seed0.npz"
SLICE_B_JSONL="$RESULTS_DIR/expQ_rollouts_sliceB.jsonl"

log "EXP Q START model=$MODEL run_slice_b=$RUN_SLICE_B"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

# Calibration NPZ for Slice A is shared with Exp P; must exist.
if [ ! -f "$SLICE_A_NPZ" ]; then
  log "FAIL Slice A calibration NPZ missing: $SLICE_A_NPZ"
  log "  (Run Exp P calibration first: python expP_calibrate.py --model $MODEL)"
  exit 2
fi
log "  Slice A calib OK -> $SLICE_A_NPZ"

# ================================================================
# Phase A — Smoke (fail-fast on wiring).
# ================================================================
if [ "${EXPQ_SKIP_SMOKE:-0}" = "1" ]; then
  log "--- PHASE A: SMOKE SKIPPED (EXPQ_SKIP_SMOKE=1) ---"
else
  log "--- PHASE A: smoke n=3 short bucket ---"
  python3 -u expQ_smoke.py \
      --model "$MODEL" \
      --n-items 3 \
      --bucket short \
      --task retrieval-image \
      --calib-npz "$SLICE_A_NPZ" \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL smoke (rc=$rc). Wiring is broken — do not launch main."
    exit $rc
  fi
  log "  smoke OK"
fi

# ================================================================
# Phase B — Slice A main run.
# ================================================================
log "--- PHASE B: Slice A main run (Q0..Q11) 336x336 equal-res, full pool, num_images>=8 ---"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
python3 -u expQ_driver.py \
    --model "$MODEL" \
    --task retrieval-image \
    --calib-npz "$SLICE_A_NPZ" \
    --use-full-pool \
    --min-num-images 8 \
    --max-pixels-context $((336*336)) \
    --max-pixels-choices $((336*336)) \
    --include-int2-stretch \
    --out-jsonl "$SLICE_A_JSONL" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "FAIL Slice A main (rc=$rc)"
  exit $rc
fi
log "  Slice A main OK -> $SLICE_A_JSONL"

# ================================================================
# Phase C — Analyze + emit branch JSON.
# ================================================================
log "--- PHASE C: analyze Slice A + emit branch JSON ---"
python3 -u expQ_analyze.py --slice A \
    --in-jsonl "$SLICE_A_JSONL" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "WARN analyzer (rc=$rc) — continuing anyway"
fi
if [ ! -f "$SLICE_A_BRANCH" ]; then
  log "FAIL branch JSON not produced at $SLICE_A_BRANCH"
  exit 4
fi
log "  Slice A analyze OK -> branch JSON $SLICE_A_BRANCH"

# Read branch flags.
need_q5_seeds=$(python3 -c "import json; print(json.load(open('$SLICE_A_BRANCH')).get('need_q5_seeds', False))")
need_q8_seeds=$(python3 -c "import json; print(json.load(open('$SLICE_A_BRANCH')).get('need_q8_seeds', False))")
need_q11_seeds=$(python3 -c "import json; print(json.load(open('$SLICE_A_BRANCH')).get('need_q11_seeds', False))")
slice_b_rec=$(python3 -c "import json; print(json.load(open('$SLICE_A_BRANCH')).get('slice_b_recommendation', 'DEFER'))")
log "  branch flags: q5=$need_q5_seeds q8=$need_q8_seeds q11=$need_q11_seeds slice_b=$slice_b_rec"

# ================================================================
# Phase D — Branch-conditional reseed (top-25 priority over top-50).
# ================================================================
log "--- PHASE D: conditional reseed launches ---"
RESEED_CONDS=()
if [ "$need_q8_seeds" = "True" ]; then
  RESEED_CONDS+=("Q8_s1" "Q8_s2")
  log "  q8 trigger: appending Q8_s1 Q8_s2"
fi
if [ "$need_q11_seeds" = "True" ]; then
  RESEED_CONDS+=("Q11_s1" "Q11_s2")
  log "  q11 trigger: appending Q11_s1 Q11_s2"
fi
if [ "$need_q5_seeds" = "True" ]; then
  RESEED_CONDS+=("Q5_s1" "Q5_s2")
  log "  q5 trigger: appending Q5_s1 Q5_s2"
fi
if [ ${#RESEED_CONDS[@]} -gt 0 ]; then
  log "  running reseed conditions: ${RESEED_CONDS[*]}"
  python3 -u expQ_driver.py \
      --model "$MODEL" \
      --task retrieval-image \
      --calib-npz "$SLICE_A_NPZ" \
      --use-full-pool \
      --min-num-images 8 \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --include-reseed \
      --conditions "${RESEED_CONDS[@]}" \
      --out-jsonl "$SLICE_A_JSONL" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "WARN reseed (rc=$rc) — continuing to 448 mini-check"
  fi
else
  log "  no branch flags triggered; skipping reseed"
fi

# ================================================================
# Phase E — 448² mini-check (Q0/Q2/Q4 on n=32 at max_pixels=200704).
# ================================================================
log "--- PHASE E: 448² mini-check (Q0/Q2/Q4 on n=32) ---"
python3 -u expQ_driver.py \
    --model "$MODEL" \
    --task retrieval-image \
    --calib-npz "$SLICE_A_NPZ" \
    --use-full-pool \
    --min-num-images 8 \
    --n-items 32 \
    --max-pixels-context 200704 \
    --max-pixels-choices 200704 \
    --conditions Q0 Q2 Q4 \
    --out-jsonl "$SLICE_A_RES448_JSONL" \
    --out-summary "$RESULTS_DIR/expQ_summary_sliceA_res448.md" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "WARN 448² mini-check (rc=$rc)"
fi
log "  448² mini-check OK -> $SLICE_A_RES448_JSONL"

# ================================================================
# Phase F — Final analyze (re-run since reseed rows were appended).
# ================================================================
log "--- PHASE F: final analyze Slice A ---"
python3 -u expQ_analyze.py --slice A \
    --in-jsonl "$SLICE_A_JSONL" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "WARN final analyze (rc=$rc)"
fi
log "  Slice A FINAL outputs in $RESULTS_DIR/expQ_*_sliceA*"

# ================================================================
# Phase G/H — Slice B (only if --slice-b passed).
# ================================================================
if [ "$RUN_SLICE_B" -eq 1 ]; then
  if [ "$slice_b_rec" != "RUN" ] && [ "${EXPQ_FORCE_SLICE_B:-0}" != "1" ]; then
    log "--- Slice B SKIPPED — recommendation=$slice_b_rec (set EXPQ_FORCE_SLICE_B=1 to override) ---"
  else
    # Phase G: calibration on reasoning-image cal-100.
    log "--- PHASE G: reasoning-image calibration ---"
    if [ -f "$SLICE_B_NPZ" ]; then
      log "  Slice B calib already exists at $SLICE_B_NPZ; skipping"
    else
      python3 -u expQ_calibrate_reasoning.py \
          --model "$MODEL" \
          --task reasoning-image \
          --seed 0 \
          2>&1 | tee -a "$LOG"
      rc=${PIPESTATUS[0]}
      if [ $rc -ne 0 ]; then
        log "FAIL Slice B calibration (rc=$rc)"
        exit $rc
      fi
    fi
    if [ ! -f "$SLICE_B_NPZ" ]; then
      log "FAIL Slice B calib NPZ not produced at $SLICE_B_NPZ"
      exit 5
    fi
    log "  Slice B calib OK -> $SLICE_B_NPZ"

    # Phase H: Slice B smoke + main (R0..R8).
    log "--- PHASE H1: Slice B smoke ---"
    python3 -u expQ_smoke.py \
        --model "$MODEL" \
        --n-items 3 \
        --bucket short \
        --task reasoning-image \
        --calib-npz "$SLICE_B_NPZ" \
        --max-pixels-context $((336*336)) \
        --max-pixels-choices $((336*336)) \
        --out "$RESULTS_DIR/expQ_smoke_sliceB.md" \
        --out-jsonl "$RESULTS_DIR/expQ_smoke_sliceB.jsonl" \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
      log "FAIL Slice B smoke (rc=$rc)"
      exit $rc
    fi

    log "--- PHASE H2: Slice B main (R0..R8) ---"
    python3 -u expQ_driver.py \
        --model "$MODEL" \
        --task reasoning-image \
        --calib-npz "$SLICE_B_NPZ" \
        --use-full-pool \
        --max-pixels-context $((336*336)) \
        --max-pixels-choices $((336*336)) \
        --out-jsonl "$SLICE_B_JSONL" \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
      log "WARN Slice B main (rc=$rc)"
    fi
    log "  Slice B main OK -> $SLICE_B_JSONL"

    log "--- PHASE H3: analyze Slice B ---"
    python3 -u expQ_analyze.py --slice B \
        --in-jsonl "$SLICE_B_JSONL" \
        2>&1 | tee -a "$LOG"
  fi
fi

log "EXP Q DONE — review outputs in $RESULTS_DIR/expQ_*"
log "Slice A: $RESULTS_DIR/expQ_summary_sliceA.md, expQ_paired_sliceA.md, expQ_verdict_matrix_sliceA.md"
if [ "$RUN_SLICE_B" -eq 1 ]; then
  log "Slice B: $RESULTS_DIR/expQ_summary_sliceB.md (if recommendation was RUN)"
else
  log "Slice B recommendation: $slice_b_rec (re-run with --slice-b to launch)"
fi
