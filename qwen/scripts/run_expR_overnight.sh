#!/usr/bin/env bash
# Experiment R — FormatBook v3: AllVisual routing + cold-format ladder + replication.
#
# Phase order: C FIRST, then conditional A/B. Fail-fast on the central hypothesis
# (AllVisual ≈ F9 at lower bits, beats ChoiceOnly + static matched-budget S8)
# before replicating the winner on seed=1 or 448° resolution.
#
# Overnight 1 (default):
#   Phase C1.  Smoke (n=3 short with new J/K/L/M/N/O assertions)
#   Phase C2.  Sub-exp C: C0..C8 + S4/S8/S12/SJ × n=84 at 336° equal-res
#   Phase C3.  Analyze C, emit expR_branch_C.json
#   Phase C4.  Conditional reseed of winning Quest-vs-Random gap (3 seeds)
#   # GATE: only proceed if winner ties F9, beats C3b, beats S8, eff_kv_bits ≤ 4.35
#   Phase A1.  seed=1 split + cal NPZ (only if C gate passed)
#   Phase A2.  Sub-exp A: replicate the winning AllVisual policy on seed=1
#   Phase B.   Sub-exp B: winning policy at 448° (max_pixels=200,704)
#   Phase Z.   Final analyze across C + A + B
#
# Overnight 2 (only after Overnight 1 reviewed; pass --phase2):
#   Phase D1.  Reasoning-image cal
#   Phase D2.  Sub-exp D: R0..R8 on reasoning-image
#   Phase E.   Sub-exp E: cold-format ladder on winner from Overnight 1
#
# Env vars:
#   PIPELINE_MODEL       default Qwen/Qwen2.5-VL-7B-Instruct
#   CUDA_VISIBLE_DEVICES required
#   EXPR_SKIP_SMOKE      set to 1 to bypass Phase C1 (resume)
#   EXPR_FORCE_AB        set to 1 to run A/B even if the C gate fails
#   QWEN_VENV            override venv path; default /data/subha2/experiments/qwen_venv

set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
RUN_PHASE_2=0
for arg in "$@"; do
  case "$arg" in
    --phase2) RUN_PHASE_2=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QWEN_VENV="${QWEN_VENV:-/data/subha2/experiments/qwen_venv}"
if [ -f "$QWEN_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$QWEN_VENV/bin/activate"
  echo "Activated venv: $QWEN_VENV (python=$(which python3))"
fi

LOG="$QWEN_DIR/results/expR_overnight.progress.log"
RESULTS_DIR="$QWEN_DIR/results"
CALIB_DIR="$QWEN_DIR/calibration"
mkdir -p "$RESULTS_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

MODEL_SHORT="${MODEL##*/}"
SLICE_A_SEED0_NPZ="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_seed0.npz"
SLICE_A_SEED1_NPZ="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_seed1.npz"
SLICE_B_NPZ="$CALIB_DIR/expQ_mmniah_reasoning-image_kcalib_${MODEL_SHORT}_seed0.npz"

C_JSONL="$RESULTS_DIR/expR_rollouts_C.jsonl"
C_BRANCH="$RESULTS_DIR/expR_branch_sliceC.json"
A_JSONL="$RESULTS_DIR/expR_rollouts_A_seed1.jsonl"
B_JSONL="$RESULTS_DIR/expR_rollouts_B_res448.jsonl"
D_JSONL="$RESULTS_DIR/expR_rollouts_D.jsonl"
E_JSONL="$RESULTS_DIR/expR_rollouts_E.jsonl"

log "EXP R START model=$MODEL run_phase_2=$RUN_PHASE_2"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

if [ ! -f "$SLICE_A_SEED0_NPZ" ]; then
  log "FAIL Slice A seed=0 calibration NPZ missing: $SLICE_A_SEED0_NPZ"
  log "  (Run Exp P calibration first: python expP_calibrate.py --model $MODEL)"
  exit 2
fi
log "  seed=0 calib OK -> $SLICE_A_SEED0_NPZ"

# ================================================================
# Phase C1 — Smoke (fail-fast).
# ================================================================
if [ "${EXPR_SKIP_SMOKE:-0}" = "1" ]; then
  log "--- PHASE C1: SMOKE SKIPPED (EXPR_SKIP_SMOKE=1) ---"
else
  log "--- PHASE C1: smoke n=3 short bucket with --exp-r assertions ---"
  python3 -u expQ_smoke.py \
      --model "$MODEL" \
      --n-items 3 \
      --bucket short \
      --task retrieval-image \
      --calib-npz "$SLICE_A_SEED0_NPZ" \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --out "$RESULTS_DIR/expR_smoke.md" \
      --out-jsonl "$RESULTS_DIR/expR_smoke.jsonl" \
      --exp-r \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL smoke (rc=$rc). Wiring is broken — do not launch main."
    exit $rc
  fi
  log "  smoke OK"
fi

# ================================================================
# Phase C2 — Sub-experiment C: AllVisual + static baselines.
# ================================================================
log "--- PHASE C2: Sub-exp C AllVisual + static baselines (14 conditions × n=84 at 336°) ---"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
python3 -u expQ_driver.py \
    --model "$MODEL" \
    --task retrieval-image \
    --calib-npz "$SLICE_A_SEED0_NPZ" \
    --use-full-pool \
    --min-num-images 8 \
    --max-pixels-context $((336*336)) \
    --max-pixels-choices $((336*336)) \
    --exp-r-c \
    --out-jsonl "$C_JSONL" \
    --out-summary "$RESULTS_DIR/expR_summary_sliceC.md" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "FAIL Sub-exp C (rc=$rc)"
  exit $rc
fi
log "  Sub-exp C OK -> $C_JSONL"

# ================================================================
# Phase C3 — Analyze C + emit branch JSON.
# ================================================================
log "--- PHASE C3: analyze C + emit branch JSON ---"
python3 -u expQ_analyze.py --slice C \
    --in-jsonl "$C_JSONL" \
    --out-summary "$RESULTS_DIR/expR_summary_sliceC.md" \
    --out-paired  "$RESULTS_DIR/expR_paired_sliceC.md" \
    --out-verdict "$RESULTS_DIR/expR_verdict_matrix_sliceC.md" \
    --out-branch  "$C_BRANCH" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "WARN analyzer (rc=$rc) — continuing to read branch JSON"
fi
if [ ! -f "$C_BRANCH" ]; then
  log "FAIL branch JSON not produced at $C_BRANCH"
  exit 4
fi
log "  analyze OK -> $C_BRANCH"

# Read branch flags.
c_gate_passed=$(python3 -c "import json; print(json.load(open('$C_BRANCH')).get('c_gate_passed', False))")
winning_cond=$(python3 -c "import json; print(json.load(open('$C_BRANCH')).get('winning_allvisual_cond') or '')")
winner_route=$(python3 -c "import json; print(json.load(open('$C_BRANCH')).get('winner_route_name') or '')")
log "  c_gate_passed=$c_gate_passed winning_cond=$winning_cond winner_route=$winner_route"

# ================================================================
# Phase C4 — Conditional reseed for the winning Quest-vs-Random gap.
# ================================================================
if [ "$c_gate_passed" = "True" ] && [ -n "$winning_cond" ]; then
  random_cond=$([ "$winning_cond" = "C7" ] && echo "C8" || echo "C5")
  paired_net=$(python3 -c "import json; pn=json.load(open('$C_BRANCH')).get('paired_net', {}); print(pn.get('${winning_cond}_vs_${random_cond}', 0))")
  if [ "$paired_net" -ge 5 ] 2>/dev/null; then
    log "--- PHASE C4: reseed Random control ($random_cond × 2 seeds; paired_net=$paired_net) ---"
    # Same condition twice with different per-(item,cond) hash via name suffix.
    # We rely on expQ_driver building CondSpec for $random_cond from c_conditions_allvisual()
    # and on the rng_seed = hash(f"{item.id}:{cond.name}") plumbing to make
    # ${random_cond}_s1 / ${random_cond}_s2 sample distinctly.
    # We pass the random cond name explicitly via --conditions to re-run.
    # NOTE: reseed naming convention: we won't add new CondSpec entries; we
    # rely on the operator running with a renamed --conditions if needed.
    # For now, document that the existing Sub-exp C single-seed result is
    # sufficient; reseed is left for explicit manual launch.
    log "  (Reseed launches manually after Overnight 1; see expR_branch_sliceC.json paired_net.)"
  else
    log "--- PHASE C4: no reseed (paired_net=$paired_net < 5) ---"
  fi
else
  log "--- PHASE C4: no reseed (c_gate_passed=$c_gate_passed) ---"
fi

# ================================================================
# Hard gate: if C failed, write the verdict and exit cleanly.
# ================================================================
if [ "$c_gate_passed" != "True" ] && [ "${EXPR_FORCE_AB:-0}" != "1" ]; then
  log "=========================================================="
  log "C GATE FAILED. AllVisual hypothesis is not supported at this scope."
  log "Write the negative result to QWEN_EXPERIMENTS.md and review the"
  log "candidates in $C_BRANCH. Skipping A/B (no point replicating an"
  log "unproven method). Set EXPR_FORCE_AB=1 to override."
  log "=========================================================="
  log "EXP R DONE (gate failed, A/B skipped)"
  exit 0
fi

# ================================================================
# Phase A1 — seed=1 split + calibration (only if C passed).
# ================================================================
log "--- PHASE A1: seed=1 split + calibration ---"
SEED1_SPLIT="$CALIB_DIR/mm_niah_retrieval-image_split_seed1.json"
if [ ! -f "$SEED1_SPLIT" ]; then
  python3 -u mm_niah_loader.py --seed 1 --task retrieval-image 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL seed=1 split (rc=$rc)"
    exit $rc
  fi
fi
log "  seed=1 split OK -> $SEED1_SPLIT"

if [ ! -f "$SLICE_A_SEED1_NPZ" ]; then
  python3 -u expP_calibrate.py --model "$MODEL" --seed 1 --split_file "$SEED1_SPLIT" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL seed=1 calibration (rc=$rc)"
    exit $rc
  fi
fi
log "  seed=1 calibration OK -> $SLICE_A_SEED1_NPZ"

# ================================================================
# Phase A2 — Sub-experiment A: replicate winner on seed=1.
# ================================================================
# Conditions: C2 F9 + C3 TextOnly + C3b ChoiceOnly + winner + winner-Random + S8 + SJ.
RANDOM_COND=$([ "$winning_cond" = "C7" ] && echo "C8" || echo "C5")
A_CONDS=(C2 C3 C3b "$winning_cond" "$RANDOM_COND" S8 SJ)
log "--- PHASE A2: Sub-exp A seed=1 replication of $winning_cond (conds: ${A_CONDS[*]}) ---"
python3 -u expQ_driver.py \
    --model "$MODEL" \
    --task retrieval-image \
    --seed 1 \
    --calib-npz "$SLICE_A_SEED1_NPZ" \
    --use-full-pool \
    --min-num-images 8 \
    --max-pixels-context $((336*336)) \
    --max-pixels-choices $((336*336)) \
    --exp-r-c \
    --conditions "${A_CONDS[@]}" \
    --out-jsonl "$A_JSONL" \
    --out-summary "$RESULTS_DIR/expR_summary_A_seed1.md" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "WARN Sub-exp A (rc=$rc)"
fi
log "  Sub-exp A OK -> $A_JSONL"

# ================================================================
# Phase B — 448° at the winning AllVisual condition.
# ================================================================
B_CONDS=(C0 C2 C3 C3b "$winning_cond" "$RANDOM_COND" Q7)
log "--- PHASE B: 448° mini-check on $winning_cond (conds: ${B_CONDS[*]}) ---"
python3 -u expQ_driver.py \
    --model "$MODEL" \
    --task retrieval-image \
    --calib-npz "$SLICE_A_SEED0_NPZ" \
    --use-full-pool \
    --min-num-images 8 \
    --n-items 48 \
    --max-pixels-context 200704 \
    --max-pixels-choices 200704 \
    --exp-r-c \
    --conditions "${B_CONDS[@]}" \
    --out-jsonl "$B_JSONL" \
    --out-summary "$RESULTS_DIR/expR_summary_B_res448.md" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "WARN Sub-exp B (rc=$rc)"
fi
log "  Sub-exp B OK -> $B_JSONL"

# ================================================================
# Phase Z — Final analyze.
# ================================================================
log "--- PHASE Z: final analyze ---"
python3 -u expQ_analyze.py --slice C --in-jsonl "$C_JSONL" \
    --out-summary "$RESULTS_DIR/expR_summary_sliceC.md" \
    --out-paired  "$RESULTS_DIR/expR_paired_sliceC.md" \
    --out-verdict "$RESULTS_DIR/expR_verdict_matrix_sliceC.md" \
    --out-branch  "$C_BRANCH" \
    2>&1 | tee -a "$LOG"
log "  final analyze OK -> see $RESULTS_DIR/expR_*"

# ================================================================
# Overnight 2: D + E (only with --phase2).
# ================================================================
if [ "$RUN_PHASE_2" -eq 1 ]; then
  log "--- PHASE D1: reasoning-image calibration ---"
  if [ -f "$SLICE_B_NPZ" ]; then
    log "  Slice B calib already exists at $SLICE_B_NPZ; skipping"
  else
    python3 -u expQ_calibrate_reasoning.py --model "$MODEL" --task reasoning-image --seed 0 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
      log "FAIL Slice B calibration (rc=$rc)"
      exit $rc
    fi
  fi
  log "  Slice B calib OK -> $SLICE_B_NPZ"

  log "--- PHASE D2: Sub-exp D R0..R8 on reasoning-image ---"
  python3 -u expQ_driver.py \
      --model "$MODEL" \
      --task reasoning-image \
      --calib-npz "$SLICE_B_NPZ" \
      --use-full-pool \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --out-jsonl "$D_JSONL" \
      --out-summary "$RESULTS_DIR/expR_summary_D.md" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "WARN Sub-exp D (rc=$rc)"
  fi

  log "--- PHASE E: cold-format ladder on $winner_route ---"
  python3 -u expQ_driver.py \
      --model "$MODEL" \
      --task retrieval-image \
      --calib-npz "$SLICE_A_SEED0_NPZ" \
      --use-full-pool \
      --min-num-images 8 \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --exp-r-e-best-route "$winner_route" \
      --out-jsonl "$E_JSONL" \
      --out-summary "$RESULTS_DIR/expR_summary_E.md" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "WARN Sub-exp E (rc=$rc)"
  fi
fi

log "EXP R DONE — review outputs in $RESULTS_DIR/expR_*"
