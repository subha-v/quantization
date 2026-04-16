#!/bin/bash
# Environment setup for VLA quantization experiments.
# Target: Stanford tambe-server-1 (2x H100 PCIe, shared server)
#
# Usage:
#   bash setup_env.sh
set -euo pipefail

# ---- Workspace on /data (4.6 TB free, local ZFS) ----
# Home dir (/home) is NFS-mounted and full — do NOT use it for data.
export WORKSPACE="${WORKSPACE:-/data/subha2}"
export OPENPI_DIR="$WORKSPACE/openpi"
export EXPERIMENT_DIR="$WORKSPACE/experiments"
export HF_HOME="$WORKSPACE/hf_cache"
export OPENPI_DATA_HOME="$WORKSPACE/.cache/openpi"

# Pin to GPU 0 (GPU 1 is occupied by other users)
export CUDA_VISIBLE_DEVICES=0

echo "=== VLA Quantization — Environment Setup ==="
echo "WORKSPACE:            $WORKSPACE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# ---- Disk check ----
echo "--- 1/9  Disk space ---"
AVAIL_GB=$(df --output=avail "$WORKSPACE" | tail -1 | awk '{print int($1/1048576)}')
echo "  Available on $WORKSPACE: ${AVAIL_GB} GB"
if [ "$AVAIL_GB" -lt 60 ]; then
    echo "  ERROR: Need at least 60 GB free. Aborting."
    exit 1
fi
echo "  OK"
echo ""

# ---- GPU ----
echo "--- 2/9  GPU check ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo "  Using GPU 0 only (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo ""

# ---- Clone openpi ----
echo "--- 3/9  openpi repo ---"
if [ -d "$OPENPI_DIR" ]; then
    echo "  Already exists at $OPENPI_DIR"
    cd "$OPENPI_DIR" && git log --oneline -3
else
    cd "$WORKSPACE"
    GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
fi
cd "$OPENPI_DIR"
echo ""

# ---- uv ----
echo "--- 4/9  uv package manager ---"
if ! command -v uv &>/dev/null; then
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
echo ""

# ---- Sync openpi deps ----
echo "--- 5/9  openpi dependencies ---"
cd "$OPENPI_DIR"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
echo ""

# ---- Patch transformers for PyTorch support ----
echo "--- 6/9  Patch transformers (required for PyTorch mode) ---"
SITE_PKGS=$(uv run python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
echo "  Site packages: $SITE_PKGS"
if [ -d "$SITE_PKGS/transformers" ]; then
    cp -r ./src/openpi/models_pytorch/transformers_replace/* "$SITE_PKGS/transformers/"
    echo "  Patched transformers"
else
    echo "  WARNING: transformers not found at $SITE_PKGS — may fail later"
fi
echo ""

# ---- Extra deps ----
echo "--- 7/9  Extra dependencies ---"
uv pip install datasets matplotlib scipy pillow
echo ""

# ---- Verify PyTorch + CUDA ----
echo "--- 8/9  PyTorch + CUDA ---"
uv run python -c "
import os, torch
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
print(f'PyTorch {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print('OK')
"
echo ""

# ---- Download checkpoint ----
echo "--- 9/9  Download pi0.5 LIBERO checkpoint ---"
uv run python -c "
from openpi.shared import download
path = download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')
print(f'Checkpoint: {path}')
"

# Try JAX → PyTorch conversion
PYTORCH_DIR="$WORKSPACE/pi05_libero_pytorch"
if [ -d "$PYTORCH_DIR" ] && [ -f "$PYTORCH_DIR/model.safetensors" ]; then
    echo "  PyTorch checkpoint already exists at $PYTORCH_DIR"
else
    echo "  Converting JAX → PyTorch..."
    CONVERT_SCRIPT=""
    for candidate in \
        "$OPENPI_DIR/examples/convert_jax_model_to_pytorch.py" \
        "$OPENPI_DIR/scripts/convert_jax_model_to_pytorch.py"; do
        [ -f "$candidate" ] && CONVERT_SCRIPT="$candidate" && break
    done
    if [ -n "$CONVERT_SCRIPT" ]; then
        uv run python "$CONVERT_SCRIPT" \
            --checkpoint_dir "$OPENPI_DATA_HOME/checkpoints/pi05_libero" \
            --config_name pi05_libero \
            --output_path "$PYTORCH_DIR" || echo "  WARNING: Conversion failed — will try loading JAX directly"
    else
        echo "  WARNING: Conversion script not found"
    fi
fi
echo ""

# ---- Create experiment dirs + copy scripts ----
mkdir -p "$EXPERIMENT_DIR"/{results,plots}
SCRIPT_SRC="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_SRC/utils.py" ]; then
    cp "$SCRIPT_SRC"/*.py "$EXPERIMENT_DIR/"
    echo "Copied experiment scripts to $EXPERIMENT_DIR/"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Add to your shell:"
echo "  export WORKSPACE=$WORKSPACE"
echo "  export OPENPI_DIR=$OPENPI_DIR"
echo "  export EXPERIMENT_DIR=$EXPERIMENT_DIR"
echo "  export HF_HOME=$HF_HOME"
echo "  export OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo "  export CUDA_VISIBLE_DEVICES=0"
echo ""
echo "Next:"
echo "  cd $OPENPI_DIR"
echo "  CUDA_VISIBLE_DEVICES=0 uv run python $EXPERIMENT_DIR/setup_and_verify.py"
echo ""
echo "Then kick off overnight:"
echo "  tmux new -s overnight"
echo "  CUDA_VISIBLE_DEVICES=0 uv run python $EXPERIMENT_DIR/run_all.py"
