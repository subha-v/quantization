#!/usr/bin/env bash
# Smoke test: run Qwen2.5-VL-3B on 10 LongVideoBench items, conditions A1 (BF16),
# A7 (INT2-KV), B6 (AttnEntropy V1). Asserts that BF16 first-token logits and
# INT2 first-token logits MUST differ on at least one example.
#
# This assertion validates that K/V quantization is actually being applied at
# the prefill matmul (not silently bypassed). If it fails, full-scale Exp A/B
# runs are NOT safe to launch.
set -euo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$QWEN_DIR/scripts"

# ---- GPU safety check (per CLAUDE.md) ----
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[smoke] nvidia-smi compute-apps:"
  nvidia-smi --query-compute-apps=pid,used_memory,gpu_name --format=csv,noheader || true
fi
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an unused GPU index before running smoke}"
echo "[smoke] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
N_LIMIT="${SMOKE_N:-10}"
N_FRAMES="${SMOKE_FRAMES:-32}"

# ---- Build cal/eval split if missing ----
SPLIT_FILE="$QWEN_DIR/calibration/split_seed0.json"
if [ ! -f "$SPLIT_FILE" ]; then
  echo "[smoke] building stratified split"
  python data_longvideobench.py --seed 0
fi

# ---- Run smoke conditions, recording logits for the first 5 items per condition ----
SMOKE_JSONL="$QWEN_DIR/results/smoke_rollouts.jsonl"
rm -f "$SMOKE_JSONL"

echo "[smoke] Exp A subset: A1 + A7"
python expA_baseline.py \
  --model "$MODEL" \
  --frames "$N_FRAMES" \
  --limit "$N_LIMIT" \
  --conditions A1_BF16 A7_BF16_INT2KV \
  --record_logits_first_n 5
mv "$QWEN_DIR/results/expA_rollouts_${MODEL##*/}.jsonl" "$SMOKE_JSONL"

# ---- Hard correctness assertion ----
python - <<'PYEOF'
import json
import sys
from pathlib import Path

p = Path("../results/smoke_rollouts.jsonl")
rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
by_cond_item = {}
for r in rows:
    if r.get("first_token_logits") is None:
        continue
    by_cond_item[(r["condition"].split("_frames")[0], r["item_id"])] = r["first_token_logits"]

bf16, int2 = {}, {}
for (cond, iid), logits in by_cond_item.items():
    if cond == "A1_BF16":
        bf16[iid] = logits
    elif cond == "A7_BF16_INT2KV":
        int2[iid] = logits

shared = sorted(set(bf16) & set(int2))
if not shared:
    print("[smoke][FAIL] no shared items with logits between A1 and A7 -- record_logits_first_n=0?", file=sys.stderr)
    sys.exit(2)

import numpy as np
max_diffs = []
for iid in shared:
    a = np.asarray(bf16[iid], dtype=np.float32)
    b = np.asarray(int2[iid], dtype=np.float32)
    max_diffs.append(float(np.max(np.abs(a - b))))

print(f"[smoke] per-item ||Δlogits||_∞ across {len(shared)} items:")
for iid, d in zip(shared, max_diffs):
    print(f"        {iid}: {d:.4e}")

if max(max_diffs) <= 1e-3:
    print(f"[smoke][FAIL] BF16 vs INT2 first-token logits are ≤1e-3 across all items.", file=sys.stderr)
    print(f"[smoke][FAIL] KV quantization is NOT being applied at prefill.", file=sys.stderr)
    print(f"[smoke][FAIL] Investigate FakeQuantKVCache.update() / attention backend before running Exp A/B.", file=sys.stderr)
    sys.exit(1)

print(f"[smoke][OK] max ||Δlogits||_∞ = {max(max_diffs):.4e} > 1e-3")
PYEOF

# ---- Build a tiny calibration + B6 run (V1 AttnEntropy) ----
echo "[smoke] calibration on 10-item cal subset"
python calibrate.py --model "$MODEL" --frames "$N_FRAMES" --target_avg_bits 3.0

echo "[smoke] Exp B subset: B0 (uniform INT4), B6 (AttnEntropy V1)"
python expB_attnentropy.py \
  --model "$MODEL" \
  --frames "$N_FRAMES" \
  --limit "$N_LIMIT" \
  --target_avg_bits 3.0 \
  --conditions B0 B6

echo "[smoke][DONE] Inspect $QWEN_DIR/results/ and $QWEN_DIR/calibration/"
