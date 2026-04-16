#!/bin/bash
# Environment setup for VLA quantization experiments.
# Run ONCE on the GCP a3-highgpu-1g instance.
#
# Usage:
#   export WORKSPACE=/path/to/local/ssd   # or ~ if no SSD
#   bash setup_env.sh
set -euo pipefail

export WORKSPACE="${WORKSPACE:-$HOME}"
export OPENPI_DIR="$WORKSPACE/openpi"
export EXPERIMENT_DIR="$WORKSPACE/quantization_experiments"
export HF_HOME="$WORKSPACE/hf_cache"
export OPENPI_DATA_HOME="$WORKSPACE/.cache/openpi"

echo "=== VLA Quantization — Environment Setup ==="
echo "WORKSPACE:      $WORKSPACE"
echo "OPENPI_DIR:     $OPENPI_DIR"
echo "EXPERIMENT_DIR: $EXPERIMENT_DIR"
echo ""

# ---- GPU ----
echo "--- 1/8  GPU check ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# ---- Clone openpi ----
echo "--- 2/8  openpi repo ---"
if [ -d "$OPENPI_DIR" ]; then
    echo "Already exists at $OPENPI_DIR"
    cd "$OPENPI_DIR" && git log --oneline -3
else
    cd "$WORKSPACE"
    GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
fi
cd "$OPENPI_DIR"
echo ""

# ---- uv ----
echo "--- 3/8  uv package manager ---"
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
echo ""

# ---- Sync openpi deps ----
echo "--- 4/8  openpi dependencies ---"
cd "$OPENPI_DIR"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
echo ""

# ---- Patch transformers for PyTorch support ----
echo "--- 5/8  Patch transformers (required for PyTorch mode) ---"
SITE_PKGS=$(uv run python -c "import sysconfig; print(sysconfig.get_path('purelib'))")
echo "Site packages: $SITE_PKGS"
cp -r ./src/openpi/models_pytorch/transformers_replace/* "$SITE_PKGS/transformers/"
echo "Patched transformers at $SITE_PKGS/transformers/"
echo ""

# ---- Extra deps ----
echo "--- 6/8  Extra dependencies ---"
uv pip install datasets matplotlib scipy pillow
echo ""

# ---- Verify PyTorch + CUDA ----
echo "--- 7/8  PyTorch + CUDA ---"
uv run python -c "
import torch
print(f'PyTorch {torch.__version__}')
assert torch.cuda.is_available(), 'CUDA not available!'
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print('OK')
"
echo ""

# ---- Download checkpoint + convert to PyTorch ----
echo "--- 8/8  Download pi0.5 LIBERO checkpoint ---"
uv run python -c "
from openpi.shared import download
path = download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')
print(f'Checkpoint downloaded to: {path}')
"

CKPT_DIR="$OPENPI_DATA_HOME/checkpoints/pi05_libero"
PYTORCH_DIR="$WORKSPACE/pi05_libero_pytorch"
if [ -d "$PYTORCH_DIR" ] && [ -f "$PYTORCH_DIR/model.safetensors" ]; then
    echo "PyTorch checkpoint already exists at $PYTORCH_DIR"
else
    echo "Converting JAX → PyTorch..."
    # The conversion script path may vary; try the known locations
    CONVERT_SCRIPT=""
    for candidate in \
        "$OPENPI_DIR/examples/convert_jax_model_to_pytorch.py" \
        "$OPENPI_DIR/scripts/convert_jax_model_to_pytorch.py"; do
        if [ -f "$candidate" ]; then
            CONVERT_SCRIPT="$candidate"
            break
        fi
    done
    if [ -n "$CONVERT_SCRIPT" ]; then
        uv run python "$CONVERT_SCRIPT" \
            --checkpoint_dir "$CKPT_DIR" \
            --config_name pi05_libero \
            --output_path "$PYTORCH_DIR"
        echo "Converted to: $PYTORCH_DIR"
    else
        echo "WARNING: Conversion script not found. Will try loading JAX checkpoint directly."
        echo "Looked in: examples/ and scripts/ under $OPENPI_DIR"
    fi
fi
echo ""

# ---- Create experiment dirs ----
mkdir -p "$EXPERIMENT_DIR"/{results,plots}

# ---- Copy scripts ----
SCRIPT_SRC="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_SRC/utils.py" ]; then
    cp "$SCRIPT_SRC"/*.py "$EXPERIMENT_DIR/"
    echo "Copied experiment scripts to $EXPERIMENT_DIR/"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Add to your shell (or .bashrc):"
echo "  export WORKSPACE=$WORKSPACE"
echo "  export OPENPI_DIR=$OPENPI_DIR"
echo "  export EXPERIMENT_DIR=$EXPERIMENT_DIR"
echo "  export HF_HOME=$HF_HOME"
echo "  export OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
echo ""
echo "Next:"
echo "  cd $OPENPI_DIR"
echo "  uv run python $EXPERIMENT_DIR/setup_and_verify.py"
