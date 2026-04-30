#!/bin/bash
# Overnight orchestrator for ExpD Phase B + n=200 follow-ups.
#
# Runs ENTIRELY on tambe-server-1; survives user SSH disconnect via nohup.
# Watches the in-flight chunk logger and P1 (n=200) processes, runs the
# trial-gate analyses as each input becomes ready, and emits a final summary.
#
# Sequence:
#   1. Wait for chunk logger (n=50 chunks).
#   2. Run n=50 trial-gate analysis with chunks (Phase B headline).
#   3. Wait for P1 (n=200 main experiment).
#   4. Run n=200 trial-gate analysis (Phase A flavor; no chunks yet).
#   5. Log chunks for n=200 trials on GPU 1.
#   6. Run n=200 trial-gate analysis again (Phase B; chunks present).
#   7. Print final paths.
#
# All step outputs go under /data/subha2/experiments/results/.
# The console log goes to /data/subha2/experiments/logs/expD_overnight.log.

set -uo pipefail

LOGDIR=/data/subha2/experiments/logs
RESULTS=/data/subha2/experiments/results
EXP_DIR=/data/subha2/experiments
PYBIN=/data/subha2/openpi/.venv/bin/python

mkdir -p "$LOGDIR" "$RESULTS"

# Env required by all of LIBERO/openpi/MuJoCo paths
export LIBERO_CONFIG_PATH=/data/subha2/.libero
export PYTHONPATH=/data/subha2/openpi/third_party/libero
export MUJOCO_GL=egl

# In-flight PIDs to wait on. Override at invocation if PIDs differ.
CHUNK_PID="${CHUNK_PID:-105026}"
P1_PID="${P1_PID:-3909597}"

banner() {
    echo ""
    echo "=========================================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "=========================================================================="
}

wait_pid() {
    local pid=$1
    local label=$2
    local interval=${3:-60}
    banner "Waiting for $label (pid=$pid)"
    while ps -p "$pid" > /dev/null 2>&1; do
        sleep "$interval"
    done
    echo "[$(date '+%H:%M:%S')] $label exited"
}

# ---- Step 1: wait for chunk logger -------------------------------------------
wait_pid "$CHUNK_PID" "chunk logger" 60

CHUNK_FILE="$RESULTS/expD_chunks__libero_pro_obj_x0.2.jsonl"
if [ -f "$CHUNK_FILE" ]; then
    echo "Chunk file rows: $(wc -l < "$CHUNK_FILE")"
else
    echo "WARN: chunk file missing at $CHUNK_FILE"
fi

# ---- Step 2: n=50 trial-gate analysis with chunks ----------------------------
banner "Running n=50 trial-gate analysis (Phase B with chunks)"
"$PYBIN" "$EXP_DIR/exp_trialgate_analysis.py" \
    --data-tag libero_pro_obj_x0.2 \
    > "$LOGDIR/expD_trialgate_n50_phaseB.stdout.log" 2>&1
echo "n=50 Phase B summary -> $RESULTS/expD_trialgate_summary__libero_pro_obj_x0.2.md"

# ---- Step 3: wait for P1 (n=200) ---------------------------------------------
wait_pid "$P1_PID" "P1 n=200 main experiment" 300

ROLLOUTS_N200="$RESULTS/expB_w4__libero_pro_obj_x0.2_n200_rollouts.jsonl"
DIAG_N200="$RESULTS/expB_diagnostic_v3__libero_pro_obj_x0.2_n200.jsonl"
if [ -f "$ROLLOUTS_N200" ]; then
    echo "P1 rollouts rows: $(wc -l < "$ROLLOUTS_N200")"
fi
if [ -f "$DIAG_N200" ]; then
    echo "P1 diagnostic rows: $(wc -l < "$DIAG_N200")"
fi

# ---- Step 4: n=200 trial-gate analysis (no chunks) ---------------------------
banner "Running n=200 trial-gate analysis (Phase A, no chunks)"
"$PYBIN" "$EXP_DIR/exp_trialgate_analysis.py" \
    --data-tag libero_pro_obj_x0.2_n200 \
    > "$LOGDIR/expD_trialgate_n200_phaseA.stdout.log" 2>&1
echo "n=200 Phase A summary -> $RESULTS/expD_trialgate_summary__libero_pro_obj_x0.2_n200.md"

# ---- Step 5: log chunks for n=200 trials on GPU 1 ---------------------------
banner "Logging W4 chunks for n=200 trial set on GPU 1"
CUDA_VISIBLE_DEVICES=1 "$PYBIN" "$EXP_DIR/expD_log_chunks.py" \
    --pro-config "Object:x:0.2" \
    --match-rollouts "$ROLLOUTS_N200" \
    --out-tag libero_pro_obj_x0.2_n200 \
    > "$LOGDIR/expD_chunks_n200.stdout.log" 2>&1
CHUNK_FILE_N200="$RESULTS/expD_chunks__libero_pro_obj_x0.2_n200.jsonl"
if [ -f "$CHUNK_FILE_N200" ]; then
    echo "n=200 chunks rows: $(wc -l < "$CHUNK_FILE_N200")"
fi

# ---- Step 6: n=200 trial-gate with chunks (Phase B) -------------------------
banner "Running n=200 trial-gate analysis (Phase B with chunks)"
"$PYBIN" "$EXP_DIR/exp_trialgate_analysis.py" \
    --data-tag libero_pro_obj_x0.2_n200 \
    > "$LOGDIR/expD_trialgate_n200_phaseB.stdout.log" 2>&1
echo "n=200 Phase B summary -> $RESULTS/expD_trialgate_summary__libero_pro_obj_x0.2_n200.md"

# ---- Final report ------------------------------------------------------------
banner "ALL OVERNIGHT STEPS DONE"
echo "Summaries written:"
ls -la "$RESULTS"/expD_trialgate_summary__*.md 2>/dev/null
echo ""
echo "Logs:"
ls -la "$LOGDIR"/expD_*.log 2>/dev/null | tail -10
echo ""
echo "Done at $(date)"
