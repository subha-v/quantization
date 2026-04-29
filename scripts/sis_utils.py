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
import re
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

def batched_sample_actions(policy, obs_list: list, noise_batched_np: np.ndarray) -> np.ndarray:
    """Run policy._sample_actions on a batch of observations with batched noise.

    Bypasses `policy.infer` (which hardcodes batch=1 via [None, ...] / [0, ...]).
    Applies `policy._input_transform` per obs (transforms are batch-1) then
    stacks leaves into a batched torch dict; calls `policy._sample_actions`
    with a leading batch dim; applies `policy._output_transform` per-row on
    the resulting actions (LiberoOutputs slices `[:, :7]` assuming a 2D
    (action_horizon, action_dim) shape, so we can't pass it the batched
    tensor directly).

    Args:
        obs_list: list of B obs dicts (same shapes/keys; e.g. perturbed
            copies of one base observation).
        noise_batched_np: (B, action_horizon, action_dim) numpy, one noise
            per batch element. Pass `np.broadcast_to(noise_np[None], (B, ah, ad))`
            to share noise across the batch.

    Returns: (B, action_horizon, 7) numpy float — same dtype as policy.infer.
    """
    import jax
    from openpi.models import model as _openpi_model

    device = policy._pytorch_device

    # Per-obs input_transform (batch-1 dicts of arrays/scalars)
    transformed = [policy._input_transform(jax.tree.map(lambda x: x, o)) for o in obs_list]

    # Stack each leaf into a (B, ...) torch tensor on device
    def _stack_to_torch(*xs):
        arrs = [np.asarray(x) for x in xs]
        return torch.from_numpy(np.stack(arrs)).to(device)

    batched_inputs = jax.tree.map(_stack_to_torch, *transformed)

    observation = _openpi_model.Observation.from_dict(batched_inputs)

    # Noise: (B, ah, ad) numpy → torch on device. ndim==3 means policy code
    # path won't add an extra batch dim (it only does so when ndim==2).
    noise_t = torch.from_numpy(noise_batched_np).to(device)

    sample_kwargs = dict(policy._sample_kwargs)
    sample_kwargs["noise"] = noise_t

    actions_batched = policy._sample_actions(device, observation, **sample_kwargs)
    actions_np = actions_batched.detach().cpu().numpy()  # (B, ah, ad_padded)

    state_t = batched_inputs["state"]
    state_np = state_t.detach().cpu().numpy() if isinstance(state_t, torch.Tensor) else np.asarray(state_t)

    # Per-row output_transform (LiberoOutputs returns {"actions": data["actions"][:, :7]})
    out_rows = []
    for i in range(actions_np.shape[0]):
        row = {"state": state_np[i], "actions": actions_np[i]}
        out = policy._output_transform(row)
        if isinstance(out, dict) and "actions" in out:
            out_rows.append(np.asarray(out["actions"]))
        else:
            out_rows.append(np.asarray(out))
    return np.stack(out_rows)


def compute_sis(
    policy,
    openpi_obs: dict,
    noise_np: np.ndarray,
    n_grid: int = 8,
    sigma: float = 8.0,
    a_clean: np.ndarray = None,
    image_key: str = "observation/image",
    batched: bool = True,
) -> tuple:
    """Compute SIS = mean_k ||π(s) - π(φ(s, k))||² over an N×N grid of patches.

    Caller must already have FP16 weights installed. The seeded `noise_np` is
    shared across clean and perturbed runs so the score reflects input
    sensitivity, not flow-matching noise variance. The base-camera image
    (`image_key`) is the only thing perturbed; wrist + state untouched
    (matches the paper's single-image setup; cleaner attribution).

    If `a_clean` is provided (already-computed clean action chunk for this
    `noise_np`), it is reused — saves one forward pass per cycle.

    `batched=True` (default) sends all N² perturbations through one forward
    call via `batched_sample_actions`. ~10× faster than sequential on H100.
    Set `batched=False` to fall back to sequential `infer_with_noise` calls
    (equivalent results within float-precision noise; useful for debugging).

    Returns (sis_scalar, a_clean) — caller can reuse a_clean as the FP16
    diagnostic action.
    """
    if a_clean is None:
        a_clean = infer_with_noise(policy, openpi_obs, noise_np)

    img = openpi_obs[image_key]
    perturbed = []
    for i in range(n_grid):
        for j in range(n_grid):
            perturbed_img = gaussian_blur_patch(img, (i, j), n_grid, sigma)
            obs_pert = dict(openpi_obs)
            obs_pert[image_key] = perturbed_img
            perturbed.append(obs_pert)
    B = len(perturbed)

    if batched:
        # Tile noise across batch dim with no extra memory; .copy() to satisfy
        # torch.from_numpy (broadcasted arrays aren't writable).
        noise_batched = np.broadcast_to(
            noise_np[None], (B,) + noise_np.shape
        ).copy()
        a_pert = batched_sample_actions(policy, perturbed, noise_batched)  # (B, ah, 7)
        # Mean over (B, ah, ad) == (1/B) * sum_i mean_per_patch — same as the
        # sequential implementation.
        sis_scalar = float(np.mean((a_pert - a_clean[None]) ** 2))
    else:
        sis_sum = 0.0
        for obs_pert in perturbed:
            a_pert = infer_with_noise(policy, obs_pert, noise_np)
            sis_sum += float(np.mean((a_pert - a_clean) ** 2))
        sis_scalar = sis_sum / B

    return sis_scalar, a_clean


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
    """Maintains FP16 originals + one or more quantized-with-protection precomputed
    weight tensors for every quantizable VLM Linear. Swaps via `weight.data = ...`
    (no copy).

    Memory cost: one extra weight tensor per quantized Linear per bits value
    (~equal to the VLM's quantizable weight footprint per tier). On pi0.5
    with bits_list=(2, 4) this is FP16 + W4 + W2 ≈ 8.25 GB total — fits on H100.

    Usage:
        ctrl = PrecisionController(model, bits_list=(2, 4))
        ctrl.use_fp16()        # → original weights
        ctrl.use_bits(2)       # → W2-with-protection (alias: use_quant())
        ctrl.use_bits(4)       # → W4-with-protection
        # ... inference happens normally; no other code changes needed.

    Backwards compat: `bits=N` (legacy scalar) is accepted and converted to
    `bits_list=(N,)`. Default remains W2-only to keep the legacy ExpB path's
    memory footprint unchanged.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        bits=None,
        bits_list=None,
        group_size: int = 128,
    ):
        if bits is not None and bits_list is not None:
            raise ValueError("Pass either `bits` (legacy scalar) or `bits_list`, not both")
        if bits is None and bits_list is None:
            bits_list = (2,)
        elif bits is not None:
            bits_list = (int(bits),)
        bits_list = tuple(int(b) for b in bits_list)
        if any(b <= 0 or b > 16 for b in bits_list):
            raise ValueError(f"bits_list must be in (0, 16]; got {bits_list}")

        self.model = model
        self.bits_list = bits_list
        self.group_size = group_size

        self.vlm_name, self.vlm = find_vlm_root(model)
        protect = _get_bottleneck_protect_modules(model)
        self.protect_prefixes = [n for n, _ in protect]

        # name → module / fp16 weight tensor / quantized weight tensor per bits
        self.linear_modules: dict = {}
        self.fp16_weights: dict = {}
        # bits → {name: quantized_weight_tensor}
        self.quant_weights_by_bits: dict = {b: {} for b in bits_list}

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
            for b in bits_list:
                self.quant_weights_by_bits[b][global_name] = _quantize_weight(
                    m.weight.data, bits=b, group_size=group_size
                )
            n_quantized += 1

        self.current = "fp16"
        # Per-name layer index lookup for use_bits_range (matches names like
        # "...language_model.layers.{i}..."). Computed once.
        self._layer_index_re = re.compile(r"language_model\.layers\.(\d+)\.")
        self._layer_index_by_name: dict = {}
        for name in self.linear_modules:
            m = self._layer_index_re.search(name)
            self._layer_index_by_name[name] = int(m.group(1)) if m else -1

        # Per-(layer_index, current_target) state for use_bits_range so adjacent
        # idempotent calls don't re-iterate the cache. Maps layer_idx -> str.
        self._range_current: dict = {}

        utils.log(
            f"[PrecisionController] bits_list={bits_list} quantized={n_quantized} "
            f"protected={n_protected} (protect prefixes kept at FP16: {self.protect_prefixes})"
        )

    @property
    def bits(self) -> int:
        """Legacy alias: the smallest cached bits value. Used to be a scalar."""
        return min(self.bits_list)

    @property
    def quant_weights(self) -> dict:
        """Legacy alias for the smallest-bits cache. Preserves use_quant() semantics."""
        return self.quant_weights_by_bits[min(self.bits_list)]

    def use_fp16(self) -> None:
        if self.current == "fp16":
            return
        for name, mod in self.linear_modules.items():
            mod.weight.data = self.fp16_weights[name]
        self.current = "fp16"
        # All layers now consistent at FP16; mark per-layer state.
        for i in self._layer_index_by_name.values():
            if i >= 0:
                self._range_current[i] = "fp16"

    def use_bits(self, bits: int) -> None:
        """Swap to the cached weights for the given bits value.

        Raises if `bits` was not in the constructor's bits_list (so we don't
        silently quantize on the hot path)."""
        bits = int(bits)
        if bits not in self.quant_weights_by_bits:
            raise KeyError(
                f"bits={bits} not cached; controller was built with bits_list={self.bits_list}"
            )
        target = f"w{bits}"
        if self.current == target:
            return
        cache = self.quant_weights_by_bits[bits]
        for name, mod in self.linear_modules.items():
            mod.weight.data = cache[name]
        self.current = target
        # All layers now consistent at `target`; mark per-layer state.
        for i in self._layer_index_by_name.values():
            if i >= 0:
                self._range_current[i] = target

    def use_quant(self) -> None:
        """Legacy alias: swap to the smallest-bits cache (typically W2)."""
        self.use_bits(min(self.bits_list))

    def use_bits_range(self, start_layer: int, end_layer: int, target) -> None:
        """Swap weights for `language_model.layers.{i}.*` for i ∈ [start, end]
        (inclusive on both ends) to `target` ∈ {"fp16", 2, 4}.

        Vision tower / projector / protected layer 0 are untouched (they were
        excluded at controller construction). Layer indices outside the cached
        Linear set (e.g. start_layer > MAX or layer 0 since it's protected) are
        silently skipped.

        O(n_linears_in_range). Use_bits_range invalidates `self.current` (since
        not all layers are at the same precision after a partial swap).
        """
        # Resolve target to lookup table.
        if isinstance(target, str):
            t = target.lower()
            if t == "fp16":
                weight_dict = self.fp16_weights
                target_tag = "fp16"
            elif t in ("w2", "w4"):
                bits = int(t[1:])
                if bits not in self.quant_weights_by_bits:
                    raise KeyError(f"bits={bits} not cached; bits_list={self.bits_list}")
                weight_dict = self.quant_weights_by_bits[bits]
                target_tag = f"w{bits}"
            else:
                raise ValueError(f"target str must be fp16/w2/w4; got {target!r}")
        else:
            bits = int(target)
            if bits not in self.quant_weights_by_bits:
                raise KeyError(f"bits={bits} not cached; bits_list={self.bits_list}")
            weight_dict = self.quant_weights_by_bits[bits]
            target_tag = f"w{bits}"

        start = int(start_layer)
        end = int(end_layer)
        for name, mod in self.linear_modules.items():
            i = self._layer_index_by_name.get(name, -1)
            if i < 0:
                continue
            if start <= i <= end:
                # Idempotent fast path: skip if already at target for this layer.
                if self._range_current.get(i) == target_tag:
                    continue
                mod.weight.data = weight_dict[name]
                self._range_current[i] = target_tag

        # Self.current is no longer authoritative across layers; invalidate.
        self.current = "mixed"

    def restore_fp16_permanent(self) -> None:
        """Drop all quant caches and ensure FP16 is installed. Call before
        unrelated downstream work."""
        self.use_fp16()
        for cache in self.quant_weights_by_bits.values():
            cache.clear()
        self.quant_weights_by_bits.clear()


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
# Generalized attention-metric hook (any layer, any head, any metric)
# ===================================================================

class AttentionMetricHook:
    """Generalized attention-metric probe at `language_model.layers.{layer_idx}.self_attn`.

    Wraps the target module's forward to force `output_attentions=True`, then on
    every forward pass computes per-head {entropy, top1, top5, sparsity, sink}
    averaged across query positions. `metric` selects what `get_last()` returns.

    Mirrors L12H2EntropyHook's pattern; both classes coexist (legacy hook kept
    for back-compat with existing scripts).
    """

    METRICS = ("entropy", "top1", "top5", "sparsity", "sink")

    def __init__(self, model: torch.nn.Module, layer_idx: int, head_idx: int, metric: str):
        if metric not in self.METRICS:
            raise ValueError(f"metric must be one of {self.METRICS}; got {metric!r}")
        self.layer_idx = int(layer_idx)
        self.head_idx = int(head_idx)
        self.metric = metric

        target_suffix = f"language_model.layers.{layer_idx}.self_attn"
        target_name = None
        target = None
        for name, mod in model.named_modules():
            if name.endswith(target_suffix):
                target_name = name
                target = mod
                break
        if target is None:
            raise RuntimeError(f"Could not find module ending in {target_suffix}")

        self.target = target
        self.target_name = target_name
        self._original_forward = target.forward
        # Keyed by metric name → np.ndarray (H,) of per-head scalars on most recent call
        self._last_per_head: dict = {m: None for m in self.METRICS}

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
                    # All five metrics, computed once per forward (cheap reduction).
                    entropy_h = -(a * (a + eps).log()).sum(dim=-1).mean(dim=(0, 2))
                    top1_h = a.amax(dim=-1).mean(dim=(0, 2))
                    top5_h = a.topk(min(5, a.size(-1)), dim=-1).values.sum(dim=-1).mean(dim=(0, 2))
                    row_max = a.amax(dim=-1, keepdim=True).clamp(min=1e-12)
                    sparsity_h = (a < 0.01 * row_max).float().mean(dim=(0, 2, 3))
                    sink_h = a[..., 0].mean(dim=(0, 2))
                    recorder._last_per_head["entropy"] = entropy_h.cpu().numpy()
                    recorder._last_per_head["top1"] = top1_h.cpu().numpy()
                    recorder._last_per_head["top5"] = top5_h.cpu().numpy()
                    recorder._last_per_head["sparsity"] = sparsity_h.cpu().numpy()
                    recorder._last_per_head["sink"] = sink_h.cpu().numpy()

            return result

        target.forward = wrapped
        utils.log(
            f"[AttentionMetricHook] hooked {target_name} (layer={layer_idx}, "
            f"head={head_idx}, metric={metric})"
        )

    def get_last(self) -> float:
        """Return the (head, metric) scalar from the most recent forward."""
        v = self._last_per_head.get(self.metric)
        if v is None:
            return float("nan")
        if self.head_idx >= len(v):
            return float("nan")
        return float(v[self.head_idx])

    def get_last_metric(self, metric: str) -> float:
        """Return the (head, alt_metric) scalar — useful for diagnostics that
        want all five metrics from one hook installation."""
        v = self._last_per_head.get(metric)
        if v is None:
            return float("nan")
        if self.head_idx >= len(v):
            return float("nan")
        return float(v[self.head_idx])

    def get_last_per_head(self) -> dict:
        return {m: (None if v is None else v.copy()) for m, v in self._last_per_head.items()}

    def reset(self) -> None:
        for k in self._last_per_head:
            self._last_per_head[k] = None

    def uninstall(self) -> None:
        self.target.forward = self._original_forward


# Tag conventions used in JSONL diagnostics and condition names.
# Format: "l{layer}h{head}-{metric}" e.g. "l1h7-top1", "l12h2-ent".
# `parse_probe_tag` and `format_probe_tag` provide the canonical mapping.
def parse_probe_tag(tag: str) -> tuple:
    """'l1h7-top1' -> (layer=1, head=7, metric='top1').
    Accepts 'ent' as an alias for 'entropy'."""
    if "-" not in tag:
        raise ValueError(f"probe tag must contain '-' separating lh and metric; got {tag!r}")
    lh, metric = tag.split("-", 1)
    if not lh.startswith("l") or "h" not in lh:
        raise ValueError(f"probe tag prefix must look like 'l<L>h<H>'; got {lh!r}")
    layer_str, head_str = lh[1:].split("h", 1)
    layer = int(layer_str)
    head = int(head_str)
    metric = {"ent": "entropy"}.get(metric, metric)
    if metric not in AttentionMetricHook.METRICS:
        raise ValueError(f"metric must be one of {AttentionMetricHook.METRICS}; got {metric!r}")
    return layer, head, metric


def format_probe_tag(layer: int, head: int, metric: str) -> str:
    """(1, 7, 'top1') -> 'l1h7-top1'. (12, 2, 'entropy') -> 'l12h2-ent'."""
    metric_short = "ent" if metric == "entropy" else metric
    return f"l{layer}h{head}-{metric_short}"


# Direction map: which side of the percentile predicts HIGH quantization sensitivity.
# Derived from MEETING_5 top-15 ρ signs (negative ρ → low metric = high sensitivity → "bottom"
# of the rank should be escalated; positive ρ → high metric = high sensitivity → "top").
# Used by build_masks to decide largest=True/False in _topk_indices.
PROBE_DIRECTION_BY_TAG = {
    "l1h7-top1":   "top",      # ρ = +0.264 → high top1 = high sensitivity
    "l9h2-ent":    "bottom",   # ρ = -0.268 → low entropy = high sensitivity
    "l9h2-top5":   "top",      # ρ = +0.277
    "l12h2-ent":   "bottom",   # ρ = -0.294 (D2 winner)
    "l12h2-top5":  "top",      # ρ = +0.294
    "l12h2-sparsity": "top",   # ρ = +0.284
    "l3h4-top5":   "bottom",   # ρ = -0.258
    "l17h4-top1":  "bottom",   # ρ = -0.232
    "l4h2-sparsity": "bottom", # ρ = -0.230
    "l11h7-top1":  "top",      # ρ = +0.230
}


# ===================================================================
# Intra-pass controller — mid-forward weight swap driven by attention
# ===================================================================

class IntraPassController:
    """Forward-hook-driven mid-pass weight swap.

    Wraps `language_model.layers.{layer_L}.self_attn.forward`. After the
    original forward returns and the attention metric is computed, calls
    `ctrl.use_bits_range(layer_L + 1, MAX_LAYER, decision_prec)` so layers
    L+1..MAX_LAYER use the new precision for the rest of THIS forward pass.

    Three-tier supported via `decision_low_prec`. Decision logic per cycle:
      - bottom-frac_high by metric (predicted-most-sensitive) → escalate to high
      - top-frac_low by metric (predicted-least-sensitive) → de-escalate to low
      - middle → stays at base_prec (which `pre_infer` re-installed at cycle start)

    `direction="bottom"` means LOW metric values predict HIGH sensitivity (the
    common case for entropy: ρ < 0). `direction="top"` flips this (top1 with ρ > 0).

    Per-rollout state: `history` is a list of metric values seen so far in the
    current rollout. `cycle_decisions` records (metric_value, decision_str) per
    cycle for downstream cost/avg-bits calculation.

    Uses an internal `AttentionMetricHook` for the metric extraction; that hook
    is uninstalled in `uninstall()`.
    """

    MAX_LAYER = 17  # PaliGemma decoder layer count is 18 (indices 0..17)

    PRECISION_BITS = {"fp16": 16, "w4": 4, "w2": 2}

    def __init__(
        self,
        model: torch.nn.Module,
        ctrl: "PrecisionController",
        layer_L: int,
        head: int,
        metric: str,
        base_prec: str = "w4",
        decision_high_prec: str = "fp16",
        decision_low_prec=None,
        direction: str = "bottom",
        frac_high: float = 0.4,
        frac_low: float = 0.0,
    ):
        if base_prec not in self.PRECISION_BITS:
            raise ValueError(f"base_prec must be one of {list(self.PRECISION_BITS)}")
        if decision_high_prec not in self.PRECISION_BITS:
            raise ValueError(f"decision_high_prec must be one of {list(self.PRECISION_BITS)}")
        if decision_low_prec is not None and decision_low_prec not in self.PRECISION_BITS:
            raise ValueError(f"decision_low_prec must be one of {list(self.PRECISION_BITS)} or None")
        if direction not in ("bottom", "top"):
            raise ValueError(f"direction must be 'bottom' or 'top'; got {direction!r}")
        if not (0.0 <= frac_high <= 1.0) or not (0.0 <= frac_low <= 1.0):
            raise ValueError("frac_high and frac_low must be in [0, 1]")
        if frac_high + frac_low > 1.0 + 1e-6:
            raise ValueError(f"frac_high + frac_low must be ≤ 1; got {frac_high} + {frac_low}")
        if int(layer_L) < 0 or int(layer_L) >= self.MAX_LAYER:
            # We need at least one layer downstream of L to swap.
            raise ValueError(f"layer_L must be in [0, {self.MAX_LAYER - 1}]; got {layer_L}")

        self.ctrl = ctrl
        self.layer_L = int(layer_L)
        self.head_idx = int(head)
        self.metric = metric
        self.base_prec = base_prec
        self.decision_high_prec = decision_high_prec
        self.decision_low_prec = decision_low_prec
        self.direction = direction
        self.frac_high = float(frac_high)
        self.frac_low = float(frac_low)

        self.history = []
        self.cycle_decisions = []  # list of (metric_value, decision_str)

        # Install the metric hook on the chosen layer.
        self.probe = AttentionMetricHook(model, layer_L, head, metric)

        # Wrap the metric hook so it ALSO fires the precision swap after recording.
        # AttentionMetricHook already replaced target.forward; we wrap that wrapper.
        target = self.probe.target
        outer_forward = target.forward
        controller = self

        def wrapped_with_swap(*args, **kwargs):
            result = outer_forward(*args, **kwargs)
            # AttentionMetricHook has now populated _last_per_head. Read & decide.
            v = controller.probe.get_last()
            controller._post_attn(v)
            return result

        target.forward = wrapped_with_swap
        self._target = target
        self._outer_forward = outer_forward
        utils.log(
            f"[IntraPassController] L={layer_L} head={head} metric={metric} "
            f"base={base_prec} high={decision_high_prec} low={decision_low_prec} "
            f"dir={direction} frac_high={frac_high} frac_low={frac_low}"
        )

    def reset_per_rollout(self) -> None:
        """Call once at the start of every rollout (clears running history)."""
        self.history.clear()
        self.cycle_decisions.clear()

    def pre_infer(self, _t=None) -> None:
        """Reset all post-protection layers to base precision for a new cycle.
        Wire this into the rollout's `pre_infer_callback`. Without it, the
        previous cycle's escalation leaks across cycles."""
        if self.base_prec == "fp16":
            self.ctrl.use_bits_range(1, self.MAX_LAYER, "fp16")
        elif self.base_prec == "w2":
            self.ctrl.use_bits_range(1, self.MAX_LAYER, 2)
        elif self.base_prec == "w4":
            self.ctrl.use_bits_range(1, self.MAX_LAYER, 4)
        # Reset the probe hook so we know any subsequent get_last() reflects this cycle.
        self.probe.reset()

    def _post_attn(self, v: float) -> None:
        """Called immediately after the attention forward returns at layer L.
        Computes the precision decision and swaps layers L+1..MAX in O(1)."""
        if not (v == v):  # NaN — no signal
            self.cycle_decisions.append((v, self.base_prec))
            return
        self.history.append(v)
        n = len(self.history)
        # Per-rollout running quantile rank: rank in [0, 1]; 0 = smallest.
        rank = sum(1 for x in self.history if x <= v) / n

        if self.direction == "bottom":
            escalate = rank <= self.frac_high
            de_escalate = (
                self.decision_low_prec is not None
                and self.frac_low > 0
                and rank >= 1.0 - self.frac_low
            )
        else:  # "top"
            escalate = rank >= 1.0 - self.frac_high
            de_escalate = (
                self.decision_low_prec is not None
                and self.frac_low > 0
                and rank <= self.frac_low
            )

        if escalate:
            chosen = self.decision_high_prec
        elif de_escalate:
            chosen = self.decision_low_prec
        else:
            chosen = self.base_prec

        if chosen != self.base_prec:
            target = "fp16" if chosen == "fp16" else (2 if chosen == "w2" else 4)
            self.ctrl.use_bits_range(self.layer_L + 1, self.MAX_LAYER, target)
        # else: stays at base, already installed by pre_infer
        self.cycle_decisions.append((float(v), chosen))

    def avg_bits(self) -> float:
        """Average bits across cycles, accounting for the protected prefix
        (layers 0..L always at base) and the post-decision tail."""
        if not self.cycle_decisions:
            return float("nan")
        # Protected layers (vision tower, layer 0, projector) and layers 1..L
        # always run at base. Layers L+1..MAX run at the per-cycle decision.
        # Approximate: weight equally across layers 1..MAX (this ignores
        # parameter-count differences between layers, but gives a clean
        # comparison metric).
        n_layers_base = self.layer_L  # layers 1..L (L layers)
        n_layers_tail = self.MAX_LAYER - self.layer_L  # layers L+1..MAX
        n_total = n_layers_base + n_layers_tail
        bits_base = self.PRECISION_BITS[self.base_prec]
        total = 0.0
        for _, dec in self.cycle_decisions:
            bits_tail = self.PRECISION_BITS[dec]
            total += (n_layers_base * bits_base + n_layers_tail * bits_tail) / n_total
        return total / len(self.cycle_decisions)

    def n_escalations(self) -> int:
        return sum(1 for _, d in self.cycle_decisions if d == self.decision_high_prec)

    def n_de_escalations(self) -> int:
        if self.decision_low_prec is None:
            return 0
        return sum(1 for _, d in self.cycle_decisions if d == self.decision_low_prec)

    def uninstall(self) -> None:
        # Restore the AttentionMetricHook's wrapper (which itself restores in turn).
        self._target.forward = self._outer_forward
        self.probe.uninstall()


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
