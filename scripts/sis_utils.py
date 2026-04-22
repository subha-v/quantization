"""
SIS (State Importance Score) utilities for the saliency-aware PTQ validation.

Three pieces:

1. `compute_sis(policy, openpi_obs, noise_np, ...)` — perturbation-based SIS
   from SQIL (arXiv:2505.15304):
        SIS(s) = E_k ||π(s) - π(φ(s, k))||²
   where φ Gaussian-blurs an N×N grid cell on the base camera image.

2. `PrecisionController(model)` — precomputes FP16 + W2-with-protection weight
   tensors for every quantizable VLM Linear, then swaps them in O(1) via
   `weight.data = ...` pointer assignment. Cheap per-cycle precision toggling.

3. `L12H2EntropyHook(model)` — minimal forward hook on
   `language_model.layers.12.self_attn` that captures per-head attention
   entropy on every forward pass; `.get_last_entropy_h2()` returns the head-2
   entropy from the most recent call. Used as the cheap-proxy detector for
   condition 7 (D2 finding: ρ = -0.294 between l12h2 entropy and W2
   sensitivity).

The module is self-contained: import once, instantiate, use in callbacks.
"""

import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import utils  # noqa: E402  — sets caches, paths, headless GL
from exp3_flow_step_sensitivity import infer_with_noise, make_noise, get_action_shape  # noqa: E402
from exp6_attention_predicts_quant import find_vlm_root, _get_bottleneck_protect_modules  # noqa: E402


# ===================================================================
# Image perturbation
# ===================================================================

def _gaussian_kernel_1d(sigma: float, radius: int) -> np.ndarray:
    """Discrete 1D Gaussian kernel, normalized."""
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _gaussian_blur_2d(patch: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 2D Gaussian blur on (H, W, C) float patch. Pure numpy.

    Uses 'edge' padding to avoid darkening the border.
    """
    radius = max(1, int(round(3.0 * sigma)))
    k1 = _gaussian_kernel_1d(sigma, radius)

    # Pad H, W with edge replication; conv1d along axis 0 then axis 1.
    padded = np.pad(patch, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    # axis=0 (rows)
    out = np.zeros_like(patch, dtype=np.float32)
    for i, w in enumerate(k1):
        out += w * padded[i : i + patch.shape[0], radius : radius + patch.shape[1], :]
    intermediate = out
    # axis=1 (cols)
    padded2 = np.pad(intermediate, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    out2 = np.zeros_like(patch, dtype=np.float32)
    for j, w in enumerate(k1):
        out2 += w * padded2[:, j : j + patch.shape[1], :]
    return out2


def gaussian_blur_patch(
    img_uint8: np.ndarray,
    grid_idx: tuple,
    n_grid: int,
    sigma: float,
) -> np.ndarray:
    """Apply Gaussian blur to one (i, j) grid cell of a uint8 (H, W, 3) image.

    Returns a copy. Untouched outside the cell.
    """
    H, W, _ = img_uint8.shape
    i, j = grid_idx
    y0 = (i * H) // n_grid
    y1 = ((i + 1) * H) // n_grid
    x0 = (j * W) // n_grid
    x1 = ((j + 1) * W) // n_grid

    out = img_uint8.copy()
    patch = out[y0:y1, x0:x1].astype(np.float32)
    blurred = _gaussian_blur_2d(patch, sigma=sigma)
    out[y0:y1, x0:x1] = np.clip(blurred, 0, 255).astype(np.uint8)
    return out


# ===================================================================
# SIS scoring
# ===================================================================

def compute_sis(
    policy,
    openpi_obs: dict,
    noise_np: np.ndarray,
    n_grid: int = 8,
    sigma: float = 8.0,
    a_clean: np.ndarray = None,
    image_key: str = "observation/image",
) -> tuple:
    """Compute SIS = mean_k ||π(s) - π(φ(s, k))||² over an N×N grid of patches.

    Caller must already have FP16 weights installed; `policy.infer` is called
    with the seeded `noise_np` to remove flow-matching noise variance from the
    score. The base-camera image (`image_key`) is the only thing perturbed —
    wrist image and state are untouched (matches the paper's single-image
    setup; cleaner attribution).

    If `a_clean` is provided (already-computed clean action chunk for this
    `noise_np`), it is reused — saves one forward pass per cycle.

    Returns (sis_scalar, a_clean) — caller can reuse a_clean as the FP16
    diagnostic action.
    """
    if a_clean is None:
        a_clean = infer_with_noise(policy, openpi_obs, noise_np)

    img = openpi_obs[image_key]
    sis_sum = 0.0
    n = 0
    for i in range(n_grid):
        for j in range(n_grid):
            perturbed_img = gaussian_blur_patch(img, (i, j), n_grid, sigma)
            obs_pert = dict(openpi_obs)
            obs_pert[image_key] = perturbed_img
            a_pert = infer_with_noise(policy, obs_pert, noise_np)
            sis_sum += float(np.mean((a_pert - a_clean) ** 2))
            n += 1

    return sis_sum / n, a_clean


# ===================================================================
# Precision controller — O(1) FP16 ↔ W2-with-protection swap
# ===================================================================

def _quantize_weight(w: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    """Symmetric, group-wise weight-only fake quantization. Returns a tensor of
    the same shape and dtype as `w`."""
    qmax = 2 ** (bits - 1) - 1
    wf = w.detach().float()
    if group_size > 0 and wf.shape[1] >= group_size and wf.shape[1] % group_size == 0:
        g = wf.reshape(wf.shape[0], -1, group_size)
        s = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        q = ((g / s).round().clamp(-qmax, qmax) * s).reshape_as(wf)
    else:
        s = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        q = (wf / s).round().clamp(-qmax, qmax) * s
    return q.to(w.dtype).contiguous()


class PrecisionController:
    """Maintains FP16 originals + W2-with-protection precomputed weights for
    every quantizable VLM Linear. Swaps via `weight.data = ...` (no copy).

    Memory cost: one extra weight tensor per quantized Linear (~equal to the
    VLM's quantizable weight footprint). On pi0.5 this is a few GB — fine
    on a server GPU.

    Usage:
        ctrl = PrecisionController(model, bits=2)
        ctrl.use_fp16()    # → original weights
        ctrl.use_quant()   # → W2-with-protection
        # ... inference happens normally; no other code changes needed.
    """

    def __init__(self, model: torch.nn.Module, bits: int = 2, group_size: int = 128):
        self.model = model
        self.bits = bits
        self.group_size = group_size

        self.vlm_name, self.vlm = find_vlm_root(model)
        protect = _get_bottleneck_protect_modules(model)
        self.protect_prefixes = [n for n, _ in protect]

        # name → module / fp16 weight tensor / quantized weight tensor
        self.linear_modules: dict = {}
        self.fp16_weights: dict = {}
        self.quant_weights: dict = {}

        n_quantized = 0
        n_protected = 0
        for n, m in self.vlm.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            global_name = f"{self.vlm_name}.{n}" if n else self.vlm_name
            if any(global_name.startswith(p) for p in self.protect_prefixes):
                n_protected += 1
                continue
            self.linear_modules[global_name] = m
            self.fp16_weights[global_name] = m.weight.data
            self.quant_weights[global_name] = _quantize_weight(
                m.weight.data, bits=bits, group_size=group_size
            )
            n_quantized += 1

        self.current = "fp16"
        utils.log(
            f"[PrecisionController] bits={bits} quantized={n_quantized} "
            f"protected={n_protected} (W2 keeps {self.protect_prefixes} at FP16)"
        )

    def use_fp16(self) -> None:
        if self.current == "fp16":
            return
        for name, mod in self.linear_modules.items():
            mod.weight.data = self.fp16_weights[name]
        self.current = "fp16"

    def use_quant(self) -> None:
        if self.current == "quant":
            return
        for name, mod in self.linear_modules.items():
            mod.weight.data = self.quant_weights[name]
        self.current = "quant"

    def restore_fp16_permanent(self) -> None:
        """Drop the quant cache and ensure FP16 is installed. Call before
        unrelated downstream work."""
        self.use_fp16()
        self.quant_weights.clear()


# ===================================================================
# Layer-12 head-2 attention entropy hook
# ===================================================================

class L12H2EntropyHook:
    """Targeted attention-entropy probe on `language_model.layers.12.self_attn`.

    Wraps that one module's forward to force `output_attentions=True`, then
    on every forward pass computes per-head Shannon entropy averaged across
    query positions. Only head 2 is exposed (the D2 finding's strongest
    predictor); full per-head vector is also stored for diagnostics.

    Usage:
        hook = L12H2EntropyHook(model)
        # ... policy.infer(obs) ...
        e = hook.get_last_entropy_h2()
        hook.uninstall()
    """

    TARGET_SUFFIX = "language_model.layers.12.self_attn"
    HEAD_INDEX = 2

    def __init__(self, model: torch.nn.Module):
        target_name = None
        target = None
        for name, mod in model.named_modules():
            if name.endswith(self.TARGET_SUFFIX):
                target_name = name
                target = mod
                break
        if target is None:
            raise RuntimeError(f"Could not find module ending in {self.TARGET_SUFFIX}")

        self.target = target
        self.target_name = target_name
        self._original_forward = target.forward
        self._last_entropy_per_head = None

        recorder = self
        original_forward = self._original_forward

        def wrapped(*args, **kwargs):
            kwargs["output_attentions"] = True
            try:
                result = original_forward(*args, **kwargs)
            except TypeError:
                kwargs.pop("output_attentions", None)
                return original_forward(*args, **kwargs)

            attn_weights = None
            if isinstance(result, tuple):
                for item in result:
                    if isinstance(item, torch.Tensor) and item.dim() == 4:
                        attn_weights = item
                        break

            if attn_weights is not None:
                with torch.no_grad():
                    a = attn_weights.detach().float()
                    if a.size(0) > 1:
                        a = a[:1]
                    eps = 1e-12
                    entropy_h = -(a * (a + eps).log()).sum(dim=-1).mean(dim=(0, 2))
                    recorder._last_entropy_per_head = entropy_h.cpu().numpy()

            return result

        target.forward = wrapped
        utils.log(f"[L12H2EntropyHook] hooked {target_name}")

    def get_last_entropy_h2(self) -> float:
        if self._last_entropy_per_head is None:
            return float("nan")
        if self.HEAD_INDEX >= len(self._last_entropy_per_head):
            return float("nan")
        return float(self._last_entropy_per_head[self.HEAD_INDEX])

    def get_last_entropy_per_head(self) -> np.ndarray:
        return None if self._last_entropy_per_head is None else self._last_entropy_per_head.copy()

    def reset(self) -> None:
        self._last_entropy_per_head = None

    def uninstall(self) -> None:
        self.target.forward = self._original_forward


# ===================================================================
# Convenience: deterministic noise for one cycle
# ===================================================================

def cycle_noise(model, base_seed: int, cycle_idx: int) -> np.ndarray:
    """Deterministic noise tensor for (rollout, cycle), as numpy float32.

    Used for SIS so that perturbed-vs-clean comparisons share noise.
    """
    ah, ad = get_action_shape(model)
    device = next(model.parameters()).device
    seed = int(base_seed) * 100_000 + int(cycle_idx)
    noise = make_noise(ah, ad, seed=seed, device=device)
    return noise.cpu().numpy().astype(np.float32)


# ===================================================================
# Smoke-test entry point
# ===================================================================

def _smoke():
    """Sanity-check the perturbation: blur a fake image, confirm cell coverage."""
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    # paint a vertical stripe so blur is visible
    img[:, 100:124] = 255
    out = gaussian_blur_patch(img, (3, 4), n_grid=8, sigma=8.0)
    diff = (out.astype(np.int32) - img.astype(np.int32))
    nz = (diff != 0).any(axis=2)
    ys, xs = np.where(nz)
    if ys.size == 0:
        print("[smoke] FAIL: blur had no effect")
        return
    print(f"[smoke] blur affected pixels rows {ys.min()}-{ys.max()} cols {xs.min()}-{xs.max()}")
    print(f"[smoke] expected cell rows {3*224//8}-{4*224//8} cols {4*224//8}-{5*224//8}")


if __name__ == "__main__":
    _smoke()
