#!/usr/bin/env bash
# Set up the Qwen2.5-VL experiment environment using uv.
#
# Creates a venv under /data/subha2/experiments/qwen_venv on the remote (or
# under qwen/.venv locally). Installs transformers + qwen-vl-utils + datasets +
# auto-awq + decord. Caches HF datasets/models under
# /data/subha2/longvideobench/ (videos) and /data/subha2/hf_cache/ (weights).
#
# Idempotent: re-running checks for an existing venv and only adds missing pkgs.
set -euo pipefail

QWEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -d /data/subha2 ] && [ -w /data/subha2 ]; then
  VENV_DIR="${QWEN_VENV:-/data/subha2/experiments/qwen_venv}"
  export HF_HOME="${HF_HOME:-/data/subha2/hf_cache}"
  export LONGVIDEOBENCH_ROOT="${LONGVIDEOBENCH_ROOT:-/data/subha2/longvideobench}"
else
  VENV_DIR="${QWEN_VENV:-${QWEN_DIR}/.venv}"
  export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
  export LONGVIDEOBENCH_ROOT="${LONGVIDEOBENCH_ROOT:-${QWEN_DIR}/longvideobench_data}"
fi
mkdir -p "$HF_HOME" "$LONGVIDEOBENCH_ROOT"

echo "[setup] VENV_DIR=$VENV_DIR HF_HOME=$HF_HOME LONGVIDEOBENCH_ROOT=$LONGVIDEOBENCH_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found; install from https://github.com/astral-sh/uv first."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  uv venv --python 3.11 "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

uv pip install --upgrade pip
uv pip install \
  "torch>=2.4" \
  "torchvision" \
  "transformers>=4.49.0,<4.55" \
  "qwen-vl-utils[decord]" \
  "accelerate>=0.30" \
  "datasets>=2.18" \
  "safetensors" \
  "numpy" \
  "pandas" \
  "matplotlib"

# Optional: auto-awq for the AWQ checkpoint conditions (A3, A8). Allowed to fail
# on systems without compatible CUDA kernels — the AWQ rows will be skipped.
uv pip install "autoawq>=0.2.6" || echo "[setup] autoawq install failed; AWQ conditions will be skipped"

echo "[setup] done. Activate with:  source $VENV_DIR/bin/activate"
echo "[setup] exports added to your shell:"
echo "        export HF_HOME=$HF_HOME"
echo "        export LONGVIDEOBENCH_ROOT=$LONGVIDEOBENCH_ROOT"
