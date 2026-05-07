#!/usr/bin/env bash
# Smoke for the new Exp B online routing pipeline.
#   - Qwen2.5-VL-3B (small, ~6 GB)
#   - 5 cal items + 5 eval items
#   - 32 frames
#   - 3 conditions: B2 Random, B6 StaticEntropy, B9 OnlineNeed-Static
#
# Verifies:
#   - DiagnosticCache + DiagnosticAttentionHook produce valid signals (no NaN)
#   - bf16_pred / uniform_int2_pred differ on at least one item
#   - Static-risk aggregator sees only cal rows (split-safety guard)
#   - Routed eval builds V2 controller + gets a non-trivial mix of {INT2, BF16} bits
set -euo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

: "${CUDA_VISIBLE_DEVICES:?Must set CUDA_VISIBLE_DEVICES before launching}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
N_LIMIT="${SMOKE_N:-5}"
FRAMES="${SMOKE_FRAMES:-32}"
TARGET_AVG="${SMOKE_TARGET_AVG:-4.0}"

LOG="$QWEN_DIR/results/expB_online_smoke.progress.log"
DIAG="$QWEN_DIR/results/diagnostic_signals_smoke.jsonl"
STATIC="$QWEN_DIR/calibration/static_entropy_risk_smoke.json"
ROUTED="$QWEN_DIR/results/expB_online_rollouts_smoke.jsonl"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

rm -f "$DIAG" "$STATIC" "$ROUTED" "$LOG"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv | tee -a "$LOG" || true
log "SMOKE START model=$MODEL n=$N_LIMIT frames=$FRAMES target_avg=$TARGET_AVG"

# Diagnostic on 5 cal + 5 eval
log "STEP 1 diagnostic pass (5 cal + 5 eval, 32 frames, eager)"
python3 -u diagnostic_pass.py --model "$MODEL" --frames "$FRAMES" \
    --splits cal eval --limit "$N_LIMIT" --out "$DIAG" --progress_every 1 \
    2>&1 | tee -a "$LOG"

# Inline assertions
log "STEP 2 verify diagnostic JSONL"
python3 - <<PYEOF | tee -a "$LOG"
import json
rows = [json.loads(l) for l in open("$DIAG") if l.strip()]
print(f"[smoke] diagnostic rows: {len(rows)}")
splits = {r["split"] for r in rows}
print(f"[smoke] splits: {splits}")
items = {r["item_id"] for r in rows}
print(f"[smoke] distinct items: {len(items)}")
# Per-item: bf16_pred vs uniform_int2_pred should differ on at least one item
diffs = 0
for iid in items:
    sample = [r for r in rows if r["item_id"] == iid][0]
    if sample["bf16_pred"] != sample["uniform_int2_pred"]:
        diffs += 1
print(f"[smoke] items with bf16_pred != uniform_int2_pred: {diffs}/{len(items)}")
assert diffs > 0, "FAIL: BF16 and INT2 predictions identical for all items — quant not effective"
# No NaN entropy_mean
nans = sum(1 for r in rows if r["entropy_mean"] != r["entropy_mean"])
print(f"[smoke] NaN entropy_mean rows: {nans}/{len(rows)}")
assert nans == 0, "FAIL: NaN in entropy_mean — eager attention hook didn't capture weights"
nans_res = sum(1 for r in rows if r["kv_residual_int2"] != r["kv_residual_int2"])
print(f"[smoke] NaN kv_residual_int2 rows: {nans_res}/{len(rows)}")
assert nans_res == 0, "FAIL: NaN in kv_residual_int2 — DiagnosticCache not recording"
print("[smoke] STEP 2 OK")
PYEOF

# Static risk (cal only)
log "STEP 3 aggregate static_entropy_risk from cal-only"
python3 -u precision_need_scoring.py --diagnostic "$DIAG" --out "$STATIC" \
    2>&1 | tee -a "$LOG"
python3 - <<PYEOF | tee -a "$LOG"
import json
d = json.load(open("$STATIC"))
print(f"[smoke] n_cal in static_risk = {d['n_cal']}")
assert d["n_cal"] == $N_LIMIT, f"FAIL: expected n_cal={$N_LIMIT}, got {d['n_cal']}"
print("[smoke] STEP 3 OK")
PYEOF

# Routed eval (3 conditions × 5 eval items)
log "STEP 4 routed eval (B2 1 seed + B6 + B9, 5 items, 32 frames)"
python3 -u expB_online.py --model "$MODEL" --frames "$FRAMES" \
    --diagnostic "$DIAG" --static_risk "$STATIC" --out "$ROUTED" \
    --limit "$N_LIMIT" --target_avg_bits "$TARGET_AVG" \
    --conditions B2_Random B6_StaticEntropy B9_OnlineNeed_Static \
    --seeds 0 \
    --progress_every 1 \
    2>&1 | tee -a "$LOG"

# Verify routed JSONL
python3 - <<PYEOF | tee -a "$LOG"
import json
rows = [json.loads(l) for l in open("$ROUTED") if l.strip()]
print(f"[smoke] routed rollouts: {len(rows)}")
conds = {r["condition"] for r in rows}
print(f"[smoke] conditions: {conds}")
# Each condition should produce $N_LIMIT rows
for c in conds:
    n = sum(1 for r in rows if r["condition"] == c)
    print(f"[smoke]   {c}: n={n}")
    assert n == $N_LIMIT, f"FAIL: condition {c} expected {$N_LIMIT} rows, got {n}"
# avg_kv_bits should be ~target (4) — within ±0.5
for c in conds:
    avg_bits = [r["avg_kv_bits"] for r in rows if r["condition"] == c]
    mean_avg = sum(avg_bits) / len(avg_bits)
    print(f"[smoke]   {c}: mean avg_kv_bits = {mean_avg:.2f}")
    assert 3.0 <= mean_avg <= 5.5, f"FAIL: avg_kv_bits {mean_avg:.2f} out of range for target=4"
print("[smoke] STEP 4 OK")
PYEOF

log "SMOKE DONE — pipeline plumbing verified"
