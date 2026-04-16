# Common Errors on Stanford tambe-server-1

## NFS Home Quota (`Disk quota exceeded: '/home/subha2/...'`)

The `/home` partition is NFS-mounted and 100% full (182MB free). Many tools default to writing caches there.

**Fix:** Redirect all caches to `/data/subha2` via environment variables. These are set in `utils.py` but you may need them in your shell too:

```bash
export WORKSPACE=/data/subha2
export HF_HOME=/data/subha2/hf_cache
export TRITON_CACHE_DIR=/data/subha2/.triton
export TORCHINDUCTOR_CACHE_DIR=/data/subha2/.torch_inductor
export XDG_CACHE_HOME=/data/subha2/.cache
export XDG_DATA_HOME=/data/subha2/.local/share
export MPLCONFIGDIR=/data/subha2/.matplotlib
export UV_CACHE_DIR=/data/subha2/.uv-cache
export UV_PYTHON_INSTALL_DIR=/data/subha2/.uv-python
export PIP_CACHE_DIR=/data/subha2/.pip-cache
export OPENPI_DATA_HOME=/data/subha2/.cache/openpi
```

Specific instances encountered:
- `'/home/subha2/.triton'` — Triton kernel cache during torch.compile
- `'/home/subha2/.cache/huggingface'` — HuggingFace token storage during `huggingface-cli login`
- `'/home/subha2/.local/share/uv/python'` — uv Python installation
- `'/home/subha2/.config/matplotlib'` — matplotlib config/font cache

## `torch.compile` breaks hooks and weight modification

openpi's PI0Pytorch uses `torch.compile(mode="max-autotune")` on `sample_actions`. This causes:
- `nn.Module` forward hooks never fire (0/458 triggered)
- Weight modifications via `.data` assignment are not seen by the compiled graph
- First forward pass takes ~8 minutes for Triton autotuning

**Fix:** Set `TORCHDYNAMO_DISABLE=1` before importing torch (done in `utils.py`). This makes `torch.compile` a no-op. Eager mode is ~2x slower per forward pass but hooks and weight changes work correctly.

## `total_mem` vs `total_memory` (PyTorch 2.7+)

```
AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'
```

PyTorch 2.7 renamed `total_mem` to `total_memory` on CUDA device properties.

**Fix:** Use `torch.cuda.get_device_properties(0).total_memory`.

## `Feature type 'List' not found` (datasets library)

openpi pins `datasets==3.6.0` which doesn't support the `List` feature type used by newer HuggingFace datasets (like `physical-intelligence/libero`).

**Fix:** Don't use `datasets.load_dataset()` for this dataset. Load parquet files directly with `pyarrow.parquet` (which is what our scripts do). Images are embedded in the parquets as HF Image dicts `{bytes, path}`.

## LeRobotDataset version incompatibility

```
ForwardCompatibilityError: The dataset you requested (lerobot/libero) is only available in 3.0 format.
```

openpi pins an old lerobot version that only supports v2.x dataset format. The `lerobot/libero` dataset on HuggingFace has been updated to v3.0.

**Fix:** Don't use `LeRobotDataset`. Load data directly from parquets at `/data/subha2/libero_raw/`. The `physical-intelligence/libero` dataset has images embedded in parquet files (not as separate video files), so no video decoding is needed.

## JAX→PyTorch checkpoint conversion path mismatch

The openpi download utility saves checkpoints to `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero` but the conversion script may look in `~/.cache/openpi/checkpoints/pi05_libero` (without the `openpi-assets` nesting).

**Fix:** Check the actual download path and pass it explicitly:
```bash
uv run python examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir /data/subha2/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --config-name pi05_libero \
    --output-path /data/subha2/pi05_libero_pytorch
```

Also copy the assets directory (norm stats) which the conversion doesn't include:
```bash
cp -r /data/subha2/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets /data/subha2/pi05_libero_pytorch/
```

## Weight restore appears to fail (inference MSE != 0)

After quantize→restore, comparing inference outputs shows nonzero MSE even though weights are identical.

**Cause:** The pi0.5 denoising loop starts from random noise. Two forward passes with identical weights and identical input produce different actions because the noise initialization differs.

**Fix:** Verify weight restore by comparing weight tensors directly (`(w_restored - w_original).abs().max() == 0`), not by comparing inference outputs.

## `savefig.bbox_inches` invalid rcParam

The matplotlib version in openpi's venv (older) doesn't support `savefig.bbox_inches` as an rcParam.

**Fix:** Remove it from `plt.rcParams.update()`. Pass `bbox_inches='tight'` directly to `plt.savefig()` calls instead.

## GPU 1 occupied by other users

On tambe-server-1, GPU 1 is typically in use. Always pin to GPU 0:
```bash
export CUDA_VISIBLE_DEVICES=0
```

This is set automatically in `utils.py`.
