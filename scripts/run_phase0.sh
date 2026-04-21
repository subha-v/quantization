#!/bin/bash
# Phase 0 orchestrator — runs on tambe-server-1.
#
# Does the full sequence end-to-end:
#   1. Exports env vars matching this server's layout.
#   2. Syncs scripts from the pulled repo to $EXPERIMENT_DIR.
#   3. Installs LIBERO into the openpi venv (idempotent — skipped if present).
#   4. Smoke test A: headless MuJoCo render.
#   5. Smoke test B: 1 full rollout end-to-end.
#   6. Full 18-rollout FP16 reproduction sweep.
#
# Each step is idempotent — safe to re-run after a git pull.
#
# Usage:
#     cd /data/subha2/quantization
#     bash scripts/run_phase0.sh              # full sequence
#     bash scripts/run_phase0.sh --smoke-only # install + smoke tests, skip full run
#     bash scripts/run_phase0.sh --skip-install --skip-smoke # just the full run
#
# Recommended: run inside tmux so the 20-min full sweep survives SSH drops.
#     tmux new -s phase0
#     bash scripts/run_phase0.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
DO_INSTALL=1
DO_SMOKE=1
DO_FULL=1

for arg in "$@"; do
    case "$arg" in
        --skip-install) DO_INSTALL=0 ;;
        --skip-smoke)   DO_SMOKE=0 ;;
        --skip-full)    DO_FULL=0 ;;
        --smoke-only)   DO_FULL=0 ;;
        --install-only) DO_SMOKE=0; DO_FULL=0 ;;
        --help|-h)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1 — env vars (matching the tambe-server-1 layout)
# ---------------------------------------------------------------------------
export WORKSPACE="${WORKSPACE:-/data/subha2}"
export OPENPI_DIR="${OPENPI_DIR:-$WORKSPACE/openpi}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-$WORKSPACE/experiments}"
export REPO_DIR="${REPO_DIR:-$WORKSPACE/quantization}"

export HF_HOME="${HF_HOME:-$WORKSPACE/hf_cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$WORKSPACE/.cache/openpi}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$WORKSPACE/.uv-python}"
export XDG_DATA_HOME="${XDG_DATA_HOME:-$WORKSPACE/.local/share}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE/.pip-cache}"

# Headless MuJoCo — EGL on NVIDIA GPUs. Falls back below if render fails.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# LIBERO's resource path
export PYTHONPATH="${PYTHONPATH:-}:$OPENPI_DIR/third_party/libero"

mkdir -p "$EXPERIMENT_DIR"/{results,plots} \
         "$UV_CACHE_DIR" "$XDG_DATA_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

banner() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
}

banner "Step 1 — Environment"
echo "  WORKSPACE:      $WORKSPACE"
echo "  OPENPI_DIR:     $OPENPI_DIR"
echo "  EXPERIMENT_DIR: $EXPERIMENT_DIR"
echo "  REPO_DIR:       $REPO_DIR"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MUJOCO_GL=$MUJOCO_GL"

# ---------------------------------------------------------------------------
# Step 2 — Sync latest scripts from repo to $EXPERIMENT_DIR
# ---------------------------------------------------------------------------
banner "Step 2 — Sync scripts from repo to \$EXPERIMENT_DIR"

if [ ! -d "$REPO_DIR/scripts" ]; then
    echo "ERROR: $REPO_DIR/scripts not found. Did 'git pull' run? REPO_DIR=$REPO_DIR"
    exit 1
fi

SCRIPTS_TO_SYNC=(
    utils.py
    rollout.py
    exp0_rollout_reproduce.py
    exp1_activation_stats.py
    exp2_layer_sensitivity.py
    exp3_flow_step_sensitivity.py
    setup_libero.sh
    generate_plots.py
)
for f in "${SCRIPTS_TO_SYNC[@]}"; do
    src="$REPO_DIR/scripts/$f"
    if [ -f "$src" ]; then
        cp "$src" "$EXPERIMENT_DIR/"
        echo "  copied: $f"
    else
        echo "  (skipped, not in repo: $f)"
    fi
done
chmod +x "$EXPERIMENT_DIR"/*.sh 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 3 — LIBERO install (idempotent: skipped if import works)
# ---------------------------------------------------------------------------
if [ "$DO_INSTALL" = "1" ]; then
    banner "Step 3 — LIBERO install (idempotent)"
    cd "$OPENPI_DIR"

    if uv run python -c "import libero.libero" >/dev/null 2>&1 \
       && uv run python -c "from libero.libero.envs import OffScreenRenderEnv" >/dev/null 2>&1; then
        echo "  LIBERO already installed. Skipping install."
    else
        LIBERO_DIR="$OPENPI_DIR/third_party/libero"
        if [ ! -d "$LIBERO_DIR" ]; then
            echo "  third_party/libero missing — running git submodule update..."
            cd "$OPENPI_DIR" && git submodule update --init --recursive
        fi

        echo "  Installing LIBERO runtime deps..."
        if [ -f "$LIBERO_DIR/requirements.txt" ]; then
            uv pip install -r "$LIBERO_DIR/requirements.txt" || {
                echo "  WARNING: requirements.txt install had issues, trying minimal set"
                uv pip install robosuite bddl
            }
        else
            uv pip install robosuite bddl
        fi

        echo "  Installing LIBERO (editable)..."
        uv pip install -e "$LIBERO_DIR"

        echo "  Verifying import..."
        uv run python -c "
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bd = benchmark.get_benchmark_dict()
print(f'  benchmark suites: {list(bd.keys())}')
suite = bd['libero_object']()
print(f'  libero_object tasks: {suite.n_tasks}, task 0: {suite.get_task(0).language!r}')
print('  LIBERO import OK')
"
    fi
else
    banner "Step 3 — LIBERO install (skipped, --skip-install)"
fi

# ---------------------------------------------------------------------------
# Step 4 — Smoke test A: headless MuJoCo render
# ---------------------------------------------------------------------------
if [ "$DO_SMOKE" = "1" ]; then
    banner "Step 4 — Smoke test A: headless MuJoCo render"
    cd "$OPENPI_DIR"

    set +e
    uv run python "$EXPERIMENT_DIR/rollout.py" --smoke-render
    RC=$?
    set -e

    if [ $RC -ne 0 ]; then
        echo ""
        echo "  EGL render failed. Retrying with MUJOCO_GL=osmesa..."
        export MUJOCO_GL=osmesa
        export PYOPENGL_PLATFORM=osmesa
        set +e
        uv run python "$EXPERIMENT_DIR/rollout.py" --smoke-render
        RC=$?
        set -e
        if [ $RC -ne 0 ]; then
            echo ""
            echo "  Both egl and osmesa failed. Trying glx..."
            export MUJOCO_GL=glx
            uv run python "$EXPERIMENT_DIR/rollout.py" --smoke-render || {
                echo "ERROR: all MuJoCo backends failed. Check GPU drivers / X11."
                exit 1
            }
        fi
    fi
    echo ""
    echo "  Final MUJOCO_GL=$MUJOCO_GL"

    # ---------------------------------------------------------------------------
    # Step 5 — Smoke test B: 1 end-to-end rollout
    # ---------------------------------------------------------------------------
    banner "Step 5 — Smoke test B: 1 end-to-end rollout"
    uv run python "$EXPERIMENT_DIR/exp0_rollout_reproduce.py" --smoke
else
    banner "Steps 4-5 — Smoke tests (skipped, --skip-smoke)"
fi

# ---------------------------------------------------------------------------
# Step 6 — Full 18-rollout FP16 reproduction
# ---------------------------------------------------------------------------
if [ "$DO_FULL" = "1" ]; then
    banner "Step 6 — Full 18-rollout FP16 reproduction (~20 min)"
    cd "$OPENPI_DIR"
    STDOUT_LOG="$EXPERIMENT_DIR/results/exp0_stdout.log"
    uv run python "$EXPERIMENT_DIR/exp0_rollout_reproduce.py" 2>&1 | tee "$STDOUT_LOG"

    banner "Done."
    echo "  Tables: $EXPERIMENT_DIR/results/exp0_rollout_tables.md"
    echo "  JSONL:  $EXPERIMENT_DIR/results/exp0_rollouts.jsonl"
    echo "  Log:    $STDOUT_LOG"
    echo ""
    echo "Paste the contents of exp0_rollout_tables.md back so we can review."
else
    banner "Step 6 — Full run (skipped, --skip-full)"
fi
