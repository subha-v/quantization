#!/bin/bash
# Set up LIBERO-PRO on tambe-server-1 alongside the existing openpi LIBERO checkout.
# Idempotent: safe to re-run.
#
# Steps:
#   1. Clone Zxy-MLlab/LIBERO-PRO into $WORKSPACE/experiments/LIBERO-PRO (sibling).
#   2. Overlay LIBERO-PRO's updated benchmark/__init__.py + libero_suite_task_map.py
#      into the openpi LIBERO checkout. This registers libero_<suite>_temp suites
#      so benchmark.get_benchmark_dict() resolves them like standard suites.
#   3. Download the per-(suite, axis, magnitude) bddl + init bundles from
#      HuggingFace dataset zhouxueyang/LIBERO-Pro into the LIBERO checkout's
#      bddl_files/ and init_files/ directories. Each bundle goes in its own
#      named subdir (e.g., libero_object_temp_x0.2/) — the active dir
#      (libero_object_temp/) is left empty until rollout.py stages it.
#
# Constraints (per project CLAUDE.md):
#   - All writes confined to /data/subha2/.
#   - Never modifies anything under /home, /etc, etc.
#
# Usage (on remote):
#     bash /data/subha2/experiments/setup_libero_pro.sh

set -euo pipefail

WORKSPACE="${WORKSPACE:-/data/subha2}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$WORKSPACE/experiments}"
OPENPI_DIR="${OPENPI_DIR:-$WORKSPACE/openpi}"
LIBERO_DIR="$OPENPI_DIR/third_party/libero"
LIBERO_PRO_DIR="$EXPERIMENT_DIR/LIBERO-PRO"
HF_REPO="zhouxueyang/LIBERO-Pro"

echo "=== LIBERO-PRO setup ==="
echo "WORKSPACE:         $WORKSPACE"
echo "EXPERIMENT_DIR:    $EXPERIMENT_DIR"
echo "OPENPI_DIR:        $OPENPI_DIR"
echo "LIBERO_DIR:        $LIBERO_DIR"
echo "LIBERO_PRO_DIR:    $LIBERO_PRO_DIR"

if [ ! -d "$LIBERO_DIR" ]; then
    echo "ERROR: $LIBERO_DIR does not exist."
    echo "Run setup_libero.sh first to install the base LIBERO checkout."
    exit 1
fi

mkdir -p "$EXPERIMENT_DIR"

# 1. Clone LIBERO-PRO repo if missing.
if [ ! -d "$LIBERO_PRO_DIR/.git" ]; then
    echo ""
    echo "--- Cloning Zxy-MLlab/LIBERO-PRO ---"
    git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git "$LIBERO_PRO_DIR"
else
    echo ""
    echo "--- LIBERO-PRO already cloned; pulling latest ---"
    git -C "$LIBERO_PRO_DIR" pull --ff-only || true
fi

# 2. Overlay benchmark registration files.
# LIBERO-PRO's README §Quick Start: their updated benchmark/__init__.py +
# libero_suite_task_map.py register the libero_*_temp suites. We copy those two
# files (and only those two) into the openpi LIBERO checkout, leaving the
# rest of the openpi LIBERO install intact.
echo ""
echo "--- Overlaying LIBERO-PRO benchmark registration files ---"

PRO_BENCH_INIT="$LIBERO_PRO_DIR/libero/libero/benchmark/__init__.py"
PRO_TASK_MAP="$LIBERO_PRO_DIR/libero/libero/benchmark/libero_suite_task_map.py"
DST_BENCH_INIT="$LIBERO_DIR/libero/libero/benchmark/__init__.py"
DST_TASK_MAP="$LIBERO_DIR/libero/libero/benchmark/libero_suite_task_map.py"

# Save originals once if not already saved (so overlay is undoable).
for src_pair in "$DST_BENCH_INIT" "$DST_TASK_MAP"; do
    if [ -f "$src_pair" ] && [ ! -f "$src_pair.libero_pre_pro" ]; then
        cp "$src_pair" "$src_pair.libero_pre_pro"
        echo "saved original -> $src_pair.libero_pre_pro"
    fi
done

if [ -f "$PRO_BENCH_INIT" ]; then
    cp "$PRO_BENCH_INIT" "$DST_BENCH_INIT"
    echo "overlaid: $DST_BENCH_INIT"
else
    echo "WARN: $PRO_BENCH_INIT not found in cloned LIBERO-PRO repo"
fi
if [ -f "$PRO_TASK_MAP" ]; then
    cp "$PRO_TASK_MAP" "$DST_TASK_MAP"
    echo "overlaid: $DST_TASK_MAP"
else
    echo "WARN: $PRO_TASK_MAP not found in cloned LIBERO-PRO repo"
fi

# 3. Download bddl + init bundles from HuggingFace.
# The HuggingFace dataset has all bundles (per (suite, axis, magnitude) tag).
# We copy the relevant subset (libero_<suite>_temp_<axis><mag>/ for each suite)
# into the LIBERO checkout's bddl_files/ and init_files/ dirs.
LIBERO_BDDL_ROOT="$LIBERO_DIR/libero/libero/bddl_files"
LIBERO_INIT_ROOT="$LIBERO_DIR/libero/libero/init_files"
HF_CACHE_DIR="${HF_CACHE_DIR:-$EXPERIMENT_DIR/.libero_pro_hf_cache}"

mkdir -p "$LIBERO_BDDL_ROOT" "$LIBERO_INIT_ROOT" "$HF_CACHE_DIR"

echo ""
echo "--- Downloading LIBERO-PRO bundles from HuggingFace $HF_REPO ---"

# Use uv run + huggingface_hub.snapshot_download for a complete cached pull.
# The dataset is small (~hundreds of MB) so a full snapshot is simplest;
# it's idempotent (skips already-downloaded files).
cd "$OPENPI_DIR"
# huggingface_hub is required for the snapshot_download below. Most openpi
# venvs already have it; if not, install it via the venv python's ensurepip.
if ! "$OPENPI_DIR/.venv/bin/python" -c 'import huggingface_hub' 2>/dev/null; then
    echo "huggingface_hub not present; installing via venv python..."
    "$OPENPI_DIR/.venv/bin/python" -m ensurepip --upgrade
    "$OPENPI_DIR/.venv/bin/python" -m pip install --quiet 'huggingface_hub>=0.20'
fi

"$OPENPI_DIR/.venv/bin/python" -c "
import os, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download

cache_dir = os.environ['HF_CACHE_DIR']
local = snapshot_download(
    repo_id='$HF_REPO',
    repo_type='dataset',
    cache_dir=cache_dir,
)
print(f'snapshot at: {local}')

src_root = Path(local)
bddl_dst = Path('$LIBERO_BDDL_ROOT')
init_dst = Path('$LIBERO_INIT_ROOT')

# The HF repo layout (per LIBERO-PRO README §Quick Start step 2) is:
#   bddl_files/libero_<suite>_temp_<tag>/<...>
#   init_files/libero_<suite>_temp_<tag>/<...>
# We mirror those two subtrees into the LIBERO checkout, skipping over the
# active 'libero_<suite>_temp/' directories (those are populated at runtime by
# scripts/rollout.py:stage_libero_pro_files()).

def mirror(src_subdir: Path, dst_root: Path):
    if not src_subdir.exists():
        print(f'  no {src_subdir} in HF snapshot; skipping')
        return 0
    n = 0
    for child in src_subdir.iterdir():
        if not child.is_dir():
            continue
        # Skip the bare active dir; only copy magnitude-tagged subdirs.
        if child.name.endswith('_temp'):
            continue
        if not child.name.startswith('libero_') or '_temp_' not in child.name:
            continue
        target = dst_root / child.name
        if target.exists():
            # Idempotent: already mirrored; skip.
            continue
        shutil.copytree(child, target)
        n += 1
        print(f'  mirrored: {child.name}')
    print(f'mirrored {n} bundles into {dst_root}')

src_bddl = src_root / 'bddl_files'
src_init = src_root / 'init_files'
mirror(src_bddl, bddl_dst)
mirror(src_init, init_dst)
print('LIBERO-PRO bundle mirror done')
" || {
    echo ""
    echo "WARN: HuggingFace download failed. Fallback: copy bundles directly from"
    echo "the cloned LIBERO-PRO repo (if present) under $LIBERO_PRO_DIR/libero/."
    echo ""
    PRO_BDDL_SRC="$LIBERO_PRO_DIR/libero/libero/bddl_files"
    PRO_INIT_SRC="$LIBERO_PRO_DIR/libero/libero/init_files"
    if [ -d "$PRO_BDDL_SRC" ]; then
        for d in "$PRO_BDDL_SRC"/libero_*_temp_*; do
            [ -d "$d" ] || continue
            base=$(basename "$d")
            if [ ! -d "$LIBERO_BDDL_ROOT/$base" ]; then
                cp -r "$d" "$LIBERO_BDDL_ROOT/$base"
                echo "  fallback: copied $base"
            fi
        done
    fi
    if [ -d "$PRO_INIT_SRC" ]; then
        for d in "$PRO_INIT_SRC"/libero_*_temp_*; do
            [ -d "$d" ] || continue
            base=$(basename "$d")
            if [ ! -d "$LIBERO_INIT_ROOT/$base" ]; then
                cp -r "$d" "$LIBERO_INIT_ROOT/$base"
                echo "  fallback: copied $base"
            fi
        done
    fi
}

# 4. Sanity check: ensure at least one (suite, axis, magnitude) bundle exists.
echo ""
echo "--- Verifying bundles ---"
ls -d "$LIBERO_BDDL_ROOT"/libero_object_temp_x0.2 2>/dev/null \
    && echo "OK: object x0.2 bundle present in bddl_files" \
    || echo "WARN: libero_object_temp_x0.2 missing in $LIBERO_BDDL_ROOT — Step 1 smoke will fail"
ls -d "$LIBERO_INIT_ROOT"/libero_object_temp_x0.2 2>/dev/null \
    && echo "OK: object x0.2 bundle present in init_files" \
    || echo "WARN: libero_object_temp_x0.2 missing in $LIBERO_INIT_ROOT"

# 5. Verify benchmark registration loads the new suites.
echo ""
echo "--- Verifying libero_*_temp suite registration ---"
cd "$OPENPI_DIR"
export PYTHONPATH="${PYTHONPATH:-}:$LIBERO_DIR"
"$OPENPI_DIR/.venv/bin/python" -c "
from libero.libero import benchmark
bd = benchmark.get_benchmark_dict()
suites = sorted(bd.keys())
temp_suites = [s for s in suites if s.endswith('_temp')]
print(f'registered _temp suites: {temp_suites}')
needed = ['libero_object_temp', 'libero_goal_temp']
missing = [s for s in needed if s not in suites]
if missing:
    raise SystemExit(f'ERROR: missing suite registrations: {missing}')
print('libero_*_temp suite registration OK')
"

echo ""
echo "=== LIBERO-PRO setup done ==="
echo ""
echo "Next:"
echo "  CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl \\"
echo "      $OPENPI_DIR/.venv/bin/python \$EXPERIMENT_DIR/rollout.py \\"
echo "          --single-rollout --suite Object --task-id 20 --seed 0 \\"
echo "          --pro-config 'Object:x:0.2'"
