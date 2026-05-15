#!/usr/bin/env bash
# Experiment U1 — Residual channel oracle/policy screen.
#
# Wonsuk's central question: does each query/task need DIFFERENT residual K
# channels on top of S4? We compose S4's top-16 INT7 sidecode (the 4.1875-bit
# Pareto anchor from Exp S) with extra-N residual channels selected by one of
# 9 policies (generic, random, TT/TV/VT/VV, balanced, MM-NIAH-prior,
# LVB-prior) plus an extra-16 "ALL" composite for U13. The bit budget across
# U4..U12 is fixed at KV=4.28125; U13 is at 4.375.
#
# Phases:
#   0  CPU-only: compute residual extras NPZs for every calib in scope.
#   1  Smoke (n=3 short) with U-assertions V/W/X/Y/Z/AA on MM-NIAH retrieval.
#   2  MM-NIAH retrieval-image: U0..U13 on n≈84 multi-image at 336°.
#   3  MM-NIAH reasoning-image: U0..U13 on n≈47 at 336°.
#   4  LongVideoBench-128f: U0..U13 on n=64 seed=2 stage-1 split.
#   5  Analyze: per-slice paired McNemar + verdict matrix + cross-slice diff.
#
# Env vars:
#   PIPELINE_MODEL          default Qwen/Qwen2.5-VL-7B-Instruct
#   CUDA_VISIBLE_DEVICES    required
#   QWEN_VENV               override venv; default /data/subha2/experiments/qwen_venv
#   EXPU_SKIP_SMOKE         set to 1 to bypass smoke
#   EXPU_SKIP_RETRIEVAL     set to 1 to bypass Phase 2
#   EXPU_SKIP_REASONING     set to 1 to bypass Phase 3
#   EXPU_SKIP_LVB           set to 1 to bypass Phase 4
set -uo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

MODEL="${PIPELINE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_SHORT="${MODEL##*/}"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

QWEN_VENV="${QWEN_VENV:-/data/subha2/experiments/qwen_venv}"
if [ -f "$QWEN_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$QWEN_VENV/bin/activate"
  echo "Activated venv: $QWEN_VENV (python=$(which python3))"
fi

LOG="$QWEN_DIR/results/expU_overnight.progress.log"
RESULTS_DIR="$QWEN_DIR/results"
CALIB_DIR="$QWEN_DIR/calibration"
mkdir -p "$RESULTS_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

NPZ_RETRIEVAL="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_seed0.npz"
NPZ_REASONING="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_reasoning-image_seed0.npz"
NPZ_COUNTING="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_counting-image_seed0.npz"
NPZ_LVB="$CALIB_DIR/expJ_kcalib_${MODEL_SHORT}_frames128.npz"

EXTRAS_RETRIEVAL="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_seed0_expU_extras.npz"
EXTRAS_REASONING="$CALIB_DIR/expP_mmniah_kcalib_${MODEL_SHORT}_reasoning-image_seed0_expU_extras.npz"
EXTRAS_LVB="$CALIB_DIR/expJ_kcalib_${MODEL_SHORT}_frames128_expU_extras.npz"

U_JSONL_RETRIEVAL="$RESULTS_DIR/expU_rollouts_sliceU_retrieval.jsonl"
U_JSONL_REASONING="$RESULTS_DIR/expU_rollouts_sliceU_reasoning.jsonl"
U_JSONL_LVB="$RESULTS_DIR/expU_lvb_stage1_seed2.jsonl"

log "EXP U START model=$MODEL"
log "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG"

# Validate at least the retrieval calib exists (the load-bearing slice).
if [ ! -f "$NPZ_RETRIEVAL" ]; then
  log "FAIL: retrieval-image calib NPZ missing: $NPZ_RETRIEVAL"
  exit 2
fi
log "  retrieval calib OK -> $NPZ_RETRIEVAL"
[ -f "$NPZ_REASONING" ] && log "  reasoning calib OK -> $NPZ_REASONING" || log "  WARN reasoning calib missing"
[ -f "$NPZ_LVB" ]       && log "  LVB calib OK -> $NPZ_LVB"             || log "  WARN LVB calib missing"

# ================================================================
# Phase 0 — Compute residual extras NPZs (CPU only).
# ================================================================
log "--- PHASE 0: compute Exp U residual extras NPZs (CPU only) ---"
python3 -u expU_compute_extras.py --all --model-short "$MODEL_SHORT" \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
  log "FAIL Phase 0 extras compute (rc=$rc)"
  exit $rc
fi
[ -f "$EXTRAS_RETRIEVAL" ] || { log "FAIL: retrieval extras NPZ missing after Phase 0"; exit 2; }

# ================================================================
# Phase 1 — Smoke (n=3 short bucket) on retrieval-image with U-assertions.
# ================================================================
if [ "${EXPU_SKIP_SMOKE:-0}" = "1" ]; then
  log "--- PHASE 1 SMOKE SKIPPED ---"
else
  log "--- PHASE 1 smoke: n=3 short bucket with --exp-u assertions ---"
  python3 -u expQ_smoke.py \
      --model "$MODEL" \
      --n-items 3 \
      --bucket short \
      --task retrieval-image \
      --calib-npz "$NPZ_RETRIEVAL" \
      --extras-npz "$EXTRAS_RETRIEVAL" \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --exp-u \
      --out "$RESULTS_DIR/expU_smoke.md" \
      --out-jsonl "$RESULTS_DIR/expU_smoke.jsonl" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL smoke (rc=$rc) — see $RESULTS_DIR/expU_smoke.md"
    exit $rc
  fi
  log "  smoke OK"
fi

# ================================================================
# Phase 2 — MM-NIAH retrieval-image (n≈84 multi-image at 336°).
# ================================================================
if [ "${EXPU_SKIP_RETRIEVAL:-0}" = "1" ]; then
  log "--- PHASE 2 retrieval-image SKIPPED ---"
else
  log "--- PHASE 2: MM-NIAH retrieval-image U0..U13 × n≈84 multi-image at 336° ---"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | tee -a "$LOG"
  python3 -u expQ_driver.py \
      --model "$MODEL" \
      --task retrieval-image \
      --calib-npz "$NPZ_RETRIEVAL" \
      --extras-npz "$EXTRAS_RETRIEVAL" \
      --use-full-pool \
      --min-num-images 8 \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --exp-u \
      --include-choice-routing \
      --out-jsonl "$U_JSONL_RETRIEVAL" \
      --out-summary "$RESULTS_DIR/expU_summary_sliceU_retrieval.md" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL Phase 2 retrieval-image (rc=$rc)"
    exit $rc
  fi
  log "  Phase 2 OK -> $U_JSONL_RETRIEVAL"

  log "--- PHASE 2 analyze ---"
  python3 -u expQ_analyze.py --slice U \
      --in-jsonl "$U_JSONL_RETRIEVAL" \
      --out-prefix "expU_retrieval" \
      --out-summary "$RESULTS_DIR/expU_summary_sliceU_retrieval.md" \
      --out-paired  "$RESULTS_DIR/expU_paired_sliceU_retrieval.md" \
      --out-verdict "$RESULTS_DIR/expU_verdict_sliceU_retrieval.md" \
      --out-branch  "$RESULTS_DIR/expU_branch_sliceU_retrieval.json" \
      2>&1 | tee -a "$LOG"
fi

# ================================================================
# Phase 3 — MM-NIAH reasoning-image (n≈47 at 336°).
# ================================================================
if [ "${EXPU_SKIP_REASONING:-0}" = "1" ]; then
  log "--- PHASE 3 reasoning-image SKIPPED ---"
elif [ ! -f "$NPZ_REASONING" ]; then
  log "--- PHASE 3 reasoning-image SKIPPED (calib NPZ missing) ---"
elif [ ! -f "$EXTRAS_REASONING" ]; then
  log "--- PHASE 3 reasoning-image SKIPPED (extras NPZ missing — re-run Phase 0) ---"
else
  log "--- PHASE 3: MM-NIAH reasoning-image U0..U13 × n≈47 at 336° ---"
  python3 -u expQ_driver.py \
      --model "$MODEL" \
      --task reasoning-image \
      --calib-npz "$NPZ_REASONING" \
      --extras-npz "$EXTRAS_REASONING" \
      --use-full-pool \
      --min-num-images 5 \
      --n-items 84 \
      --max-pixels-context $((336*336)) \
      --max-pixels-choices $((336*336)) \
      --exp-u \
      --out-jsonl "$U_JSONL_REASONING" \
      --out-summary "$RESULTS_DIR/expU_summary_sliceU_reasoning.md" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL Phase 3 reasoning-image (rc=$rc)"
    exit $rc
  fi
  log "  Phase 3 OK -> $U_JSONL_REASONING"

  log "--- PHASE 3 analyze ---"
  python3 -u expQ_analyze.py --slice U \
      --in-jsonl "$U_JSONL_REASONING" \
      --out-prefix "expU_reasoning" \
      --out-summary "$RESULTS_DIR/expU_summary_sliceU_reasoning.md" \
      --out-paired  "$RESULTS_DIR/expU_paired_sliceU_reasoning.md" \
      --out-verdict "$RESULTS_DIR/expU_verdict_sliceU_reasoning.md" \
      --out-branch  "$RESULTS_DIR/expU_branch_sliceU_reasoning.json" \
      2>&1 | tee -a "$LOG"
fi

# ================================================================
# Phase 4 — LongVideoBench-128f (n=64 seed=2 stage-1).
# ================================================================
if [ "${EXPU_SKIP_LVB:-0}" = "1" ]; then
  log "--- PHASE 4 LVB-128f SKIPPED ---"
elif [ ! -f "$NPZ_LVB" ]; then
  log "--- PHASE 4 LVB-128f SKIPPED (calib NPZ missing) ---"
elif [ ! -f "$EXTRAS_LVB" ]; then
  log "--- PHASE 4 LVB-128f SKIPPED (extras NPZ missing — re-run Phase 0) ---"
else
  log "--- PHASE 4: LongVideoBench-128f U0..U13 × n=64 (seed=2 stage-1) ---"
  python3 -u expJ_xmodal_outlier.py \
      --model "$MODEL" \
      --stage 1 \
      --seed 2 \
      --calib_npz "$NPZ_LVB" \
      --extras-npz "$EXTRAS_LVB" \
      --exp-u \
      --out "$U_JSONL_LVB" \
      2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    log "FAIL Phase 4 LVB-128f (rc=$rc)"
    exit $rc
  fi
  log "  Phase 4 OK -> $U_JSONL_LVB"

  log "--- PHASE 4 analyze ---"
  python3 -u expQ_analyze.py --slice U \
      --in-jsonl "$U_JSONL_LVB" \
      --out-prefix "expU_lvb" \
      --out-summary "$RESULTS_DIR/expU_summary_sliceU_lvb.md" \
      --out-paired  "$RESULTS_DIR/expU_paired_sliceU_lvb.md" \
      --out-verdict "$RESULTS_DIR/expU_verdict_sliceU_lvb.md" \
      --out-branch  "$RESULTS_DIR/expU_branch_sliceU_lvb.json" \
      2>&1 | tee -a "$LOG"
fi

# ================================================================
# Phase 5 — Cross-slice summary.
# ================================================================
log "--- PHASE 5: cross-slice winning_policy summary ---"
export EXPU_RESULTS_DIR="$RESULTS_DIR"
python3 -u - <<'PY' 2>&1 | tee -a "$LOG"
import json, os
from pathlib import Path
RESULTS = Path(os.environ["EXPU_RESULTS_DIR"])
slices = [
    ("retrieval", RESULTS / "expU_branch_sliceU_retrieval.json"),
    ("reasoning", RESULTS / "expU_branch_sliceU_reasoning.json"),
    ("lvb",       RESULTS / "expU_branch_sliceU_lvb.json"),
]
print("\n## Cross-slice Exp U1 verdict\n")
print("| slice | winning_policy | pass_any_extra_beats_s4 | pass_structured_beats_random | pass_match_or_beat_f9 | pass_same_prior_beats_foreign |")
print("|---|---|---|---|---|---|")
winners = {}
for tag, p in slices:
    if not p.exists():
        print(f"| {tag} | (no branch json) | — | — | — | — |")
        continue
    b = json.loads(p.read_text())
    winners[tag] = b.get("winning_policy")
    print(f"| {tag} | {b.get('winning_policy')} | {b.get('pass_any_extra_beats_s4')} | "
          f"{b.get('pass_structured_beats_random')} | {b.get('pass_match_or_beat_f9')} | "
          f"{b.get('pass_same_prior_beats_foreign')} |")
print()
print(f"\nwinning_policy by slice: {winners}")
distinct = {v for v in winners.values() if v is not None}
print(f"distinct winners across slices: {distinct}")
if len(distinct) >= 2:
    print("\n** WONSUK GATE PASSED: winning policy DIFFERS across datasets — "
          "residual channel allocation is task/dataset-specific. **")
else:
    print("\n** Wonsuk gate failed: same policy wins across slices (or insufficient data) — "
          "residual channel allocation may be universal. **")
PY

log "EXP U DONE — review $RESULTS_DIR/expU_*"
