"""
Multi-layer attention-entropy hook for Qwen2.5-VL.

Ported from scripts/sis_utils.py:L12H2EntropyHook (single-layer probe on pi0.5)
to all 28 decoder layers of Qwen2.5-VL. Forces output_attentions=True per layer,
which drops those layers to eager attention internally — meaningful wallclock
cost on 7B + thousands of visual tokens. Only enable for calibration runs and
V3 online-token-mask updates.

Per-layer entropy is normalized by log(seq_len) so values are comparable across
calibration examples with varying input lengths.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import math
import torch


def _entropy_from_attn(attn_weights: torch.Tensor) -> torch.Tensor:
    """attn_weights: [B, num_q_heads, q_len, k_len] -> [num_q_heads] entropy per head.

    Normalized by log(k_len) so values are in [0, 1] and comparable across runs.
    """
    eps = 1e-12
    a = attn_weights.detach().float()
    if a.size(0) > 1:
        a = a[:1]
    k_len = max(2, a.size(-1))
    norm = math.log(k_len)
    h = -(a * (a + eps).log()).sum(dim=-1).mean(dim=(0, 2)) / norm
    return h.cpu()


class MultiLayerEntropyHook:
    """Wraps every decoder layer's self_attn.forward to capture per-head entropy.

    Writes into `cache.entropy_log[layer_idx]` (a list of [num_q_heads] tensors,
    one per forward pass). The cache is a FakeQuantKVCache (or any object with a
    matching .entropy_log list).

    Usage:
        hook = MultiLayerEntropyHook(model, cache)
        # ... model.generate(...) ...
        hook.uninstall()
    """

    def __init__(self, model: torch.nn.Module, cache, layer_indices: Optional[list[int]] = None):
        from fake_quant_kv_cache import _find_decoder_layers
        layers = _find_decoder_layers(model)
        if layer_indices is not None:
            wanted = set(layer_indices)
            layers = [(i, m) for i, m in layers if i in wanted]
        self.cache = cache
        self._restores: list[tuple[torch.nn.Module, callable]] = []

        for layer_idx, attn in layers:
            original_forward = attn.forward

            def make_wrapped(orig, lidx):
                def wrapped(*args, **kwargs):
                    kwargs["output_attentions"] = True
                    try:
                        result = orig(*args, **kwargs)
                    except TypeError:
                        kwargs.pop("output_attentions", None)
                        return orig(*args, **kwargs)
                    attn_weights = None
                    if isinstance(result, tuple):
                        for item in result:
                            if isinstance(item, torch.Tensor) and item.dim() == 4:
                                attn_weights = item
                                break
                    if attn_weights is not None:
                        with torch.no_grad():
                            h = _entropy_from_attn(attn_weights)
                            cache.record_entropy(lidx, h)
                    return result
                return wrapped

            attn.forward = make_wrapped(original_forward, layer_idx)
            self._restores.append((attn, original_forward))

    def uninstall(self) -> None:
        for module, original in self._restores:
            module.forward = original
        self._restores = []


@contextmanager
def entropy_hook(model, cache, layer_indices: Optional[list[int]] = None):
    """Context-manager wrapper around MultiLayerEntropyHook."""
    hook = MultiLayerEntropyHook(model, cache, layer_indices=layer_indices)
    try:
        yield hook
    finally:
        hook.uninstall()


def aggregate_layer_entropy(cache) -> torch.Tensor:
    """Mean per-layer entropy across (heads, forward-passes) -> [num_layers] tensor.

    NaN for layers with no recorded entropy.
    """
    means = []
    for layer_log in cache.entropy_log:
        if not layer_log:
            means.append(torch.tensor(float("nan")))
            continue
        stacked = torch.stack([h.float() for h in layer_log], dim=0)  # [steps, num_heads]
        means.append(stacked.mean())
    return torch.stack(means)


def aggregate_layer_head_entropy(cache) -> torch.Tensor:
    """Mean per-(layer, head) entropy across forward passes -> [num_layers, num_heads]."""
    rows = []
    max_h = 0
    for layer_log in cache.entropy_log:
        if not layer_log:
            rows.append(None)
            continue
        stacked = torch.stack([h.float() for h in layer_log], dim=0)
        rows.append(stacked.mean(dim=0))
        max_h = max(max_h, rows[-1].numel())
    out = torch.full((len(rows), max_h), float("nan"))
    for i, r in enumerate(rows):
        if r is not None:
            out[i, : r.numel()] = r
    return out
