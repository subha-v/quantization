#!/bin/bash
# Install LIBERO into the openpi main venv on tambe-server-1.
# Idempotent: safe to re-run.
#
# Prereq: $OPENPI_DIR is set and its venv is active (or use `uv run --` wrappers).
#         openpi was cloned with --recurse-submodules so third_party/libero exists.
#
# Usage:
#     cd $OPENPI_DIR && bash /data/subha2/experiments/setup_libero.sh

set -euo pipefail

WORKSPACE="${WORKSPACE:-/data/subha2}"
OPENPI_DIR="${OPENPI_DIR:-$WORKSPACE/openpi}"
LIBERO_DIR="$OPENPI_DIR/third_party/libero"

echo "=== LIBERO install ==="
echo "OPENPI_DIR: $OPENPI_DIR"
echo "LIBERO_DIR: $LIBERO_DIR"

if [ ! -d "$LIBERO_DIR" ]; then
    echo "ERROR: $LIBERO_DIR does not exist."
    echo "Did openpi clone with --recurse-submodules? Try:"
    echo "    cd $OPENPI_DIR && git submodule update --init --recursive"
    exit 1
fi

cd "$OPENPI_DIR"

# 1. Install LIBERO's runtime requirements (robosuite, etc).
echo ""
echo "--- Installing LIBERO runtime deps ---"
if [ -f "$LIBERO_DIR/requirements.txt" ]; then
    uv pip install -r "$LIBERO_DIR/requirements.txt"
else
    echo "Warning: $LIBERO_DIR/requirements.txt not found. Trying minimal set."
    uv pip install robosuite bddl
fi

# 2. Install LIBERO itself (editable).
echo ""
echo "--- Installing LIBERO (editable) ---"
uv pip install -e "$LIBERO_DIR"

# 3. PYTHONPATH — LIBERO expects its root on the path for some resource lookups.
echo ""
echo "--- Verifying import ---"
export PYTHONPATH="${PYTHONPATH:-}:$LIBERO_DIR"
uv run python -c "
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
bd = benchmark.get_benchmark_dict()
suites = list(bd.keys())
print(f'benchmark suites: {suites}')
suite = bd['libero_object']()
print(f'libero_object has {suite.n_tasks} tasks')
task = suite.get_task(0)
print(f'task 0 description: {task.language!r}')
print('LIBERO import OK')
"

echo ""
echo "=== LIBERO install done ==="
echo ""
echo "Add to your shell / .envrc so PYTHONPATH stays set:"
echo "  export PYTHONPATH=\"\${PYTHONPATH:-}:$LIBERO_DIR\""
echo ""
echo "Next: run the headless render smoke test"
echo "  cd $OPENPI_DIR && CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \\"
echo "      uv run python \$EXPERIMENT_DIR/rollout.py --smoke-render"
