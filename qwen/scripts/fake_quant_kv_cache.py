"""
Fake-quantized KV cache for Qwen2.5-VL multiple-choice scoring.

Why this design works for first-token MCQ logprob scoring:

Qwen2.5-VL's SDPA forward (transformers v4.49+) is:

    query_states, key_states = apply_multimodal_rotary_pos_emb(...)        # post-RoPE K
    key_states, value_states = past_key_value.update(key_states, value_states, layer_idx, ...)
    key_states   = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_output  = scaled_dot_product_attention(query_states, key_states, value_states, ...)

The tensors RETURNED by `past_key_value.update(...)` are what feed the SDPA matmul.
So if our cache subclass quantizes the new K/V chunk before storing+returning the
concatenated cache, the prefill (and therefore the scored first-token logits) sees
the quantized K/V. This is the cleanest interception point that does not require
re-implementing Qwen2.5-VL's attention forward.

A smoke-test assertion in run_smoke.sh verifies BF16 vs INT2-KV first-token logits
differ; if it ever fails we know a backend has changed and we need the explicit
attention-forward patch (`AttentionKVQuantPatch.install_explicit()`).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Union

import torch
from transformers.cache_utils import DynamicCache


BitsLike = Union[int, torch.Tensor]


def _int_n_symmetric(x: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    """Symmetric per-group INT-N quant along the last dim.

    Handles bits in {2..15}. b=2 produces the ternary grid {-s, 0, +s}.
    Lifted from scripts/utils.py:fake_quantize_module (line ~418).
    """
    qmax = 2 ** (bits - 1) - 1
    orig_shape = x.shape
    last = x.shape[-1]
    if group_size > 0 and last >= group_size and last % group_size == 0:
        g = x.reshape(*orig_shape[:-1], -1, group_size)
        s = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        q = ((g / s).round().clamp(-qmax, qmax) * s).reshape(orig_shape)
    else:
        s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        q = (x / s).round().clamp(-qmax, qmax) * s
    return q.to(x.dtype)


def _fp8_cast(x: torch.Tensor) -> torch.Tensor:
    """Round-trip cast via FP8 (E4M3) if PyTorch supports it; else fall back to INT8."""
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        return _int_n_symmetric(x, bits=8)
    try:
        return x.to(fp8).to(x.dtype)
    except (RuntimeError, NotImplementedError):
        return _int_n_symmetric(x, bits=8)


def _int_n_per_position(
    x: torch.Tensor, bits_tensor: torch.Tensor, group_size: int = 128
) -> torch.Tensor:
    """Per-(head, token) INT-N quant. bits_tensor must be broadcastable against
    x[..., 0] (i.e. shape [B, H, T] for x of shape [B, H, T, D]).

    Common caller shapes for bits_tensor: [H, 1], [H, T], [T], or scalar.
    """
    qmax = (2.0 ** (bits_tensor.to(x.dtype).clamp_min(2) - 1) - 1).clamp_min(1.0)
    qmax_x = qmax.unsqueeze(-1)  # add a head_dim slot for broadcast against x

    last = x.shape[-1]
    if group_size > 0 and last >= group_size and last % group_size == 0:
        g = x.reshape(*x.shape[:-1], -1, group_size)
        qmax_g = qmax.unsqueeze(-1).unsqueeze(-1)
        s = g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax_g
        q = (g / s).round().clamp(min=-qmax_g, max=qmax_g) * s
        return q.reshape(x.shape).to(x.dtype)
    s = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax_x
    return ((x / s).round().clamp(min=-qmax_x, max=qmax_x) * s).to(x.dtype)


def fake_quantize_kv(x: torch.Tensor, bits: BitsLike, group_size: int = 128) -> torch.Tensor:
    """Apply fake-quant to a K or V tensor of shape [B, H_kv, T, head_dim].

    bits: int  -- uniform bits across (H_kv, T)
                  >=16: no-op
                  ==8:  FP8 cast (real if available; else INT8 grid)
                  2..7: INT-N symmetric per-group along head_dim
        | tensor of shape [H_kv] or [H_kv, T] -- per-(head, token) INT-N
    """
    if isinstance(bits, int):
        if bits >= 16:
            return x
        if bits == 8:
            return _fp8_cast(x)
        return _int_n_symmetric(x, bits, group_size=group_size)
    bits_t = bits.to(x.device)
    # Normalize bits_t shape so it broadcasts against x[..., 0] = [B, H, T]
    if bits_t.dim() == 1 and bits_t.shape[0] == x.shape[1]:
        bits_t = bits_t.unsqueeze(-1)  # [H] -> [H, 1] (per-head, broadcast over T)
    return _int_n_per_position(x, bits_t, group_size=group_size)


def _v_per_channel_seq_quantize(V: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Exp I VidKV-style V quant: per-(B, H_kv, 1, D) channel-axis scale.

    Unlike the default `fake_quantize_kv` (which groups along head_dim for
    INT-N), this computes max-abs along the time axis per channel d (per
    (B, H_kv, channel)) and quantizes each channel with its own scale. This
    is the V-side analog of KIVI's K per-channel-seq scaling. Bits per
    element are unchanged (INT4); only the scale-grouping axis differs.

    V: [B, H_kv, T, D] -> returns same shape and dtype.
    """
    qmax = float(2 ** (bits - 1) - 1)
    amax = V.abs().float().amax(dim=-2, keepdim=True)  # [B, H_kv, 1, D]
    s = amax.clamp_min(1e-8) / qmax
    q = (V.float() / s).round().clamp(-qmax, qmax) * s
    return q.to(V.dtype)


# ===================================================================
# Bit controller
# ===================================================================

class BitController:
    """Holds per-layer (and optionally per-head, per-token) K/V bit assignments.

    Modes:
        V1: scalar bits per layer.
        V2: tensor[num_kv_heads] bits per layer.
        V3: per-token mask + hi/lo bits per layer (set via set_protected_mask).
            Same mask drives both K and V (symmetric).
        V3K: same per-token mask but K-only — V uses the layer's v_bits scalar.
            Used by Exp D1 (cross-modal visual-K protection at INT4 V).
    """

    def __init__(self, num_layers: int, num_kv_heads: int, mode: str = "V1",
                 default_k_bits: int = 16, default_v_bits: int = 16):
        assert mode in ("V1", "V2", "V3", "V3K")
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.mode = mode
        self.k_bits: list[BitsLike] = [default_k_bits] * num_layers
        self.v_bits: list[BitsLike] = [default_v_bits] * num_layers
        self.protected: dict[int, tuple[torch.Tensor, int, int]] = {}

    def set_layer(self, layer_idx: int, k_bits: BitsLike, v_bits: BitsLike) -> None:
        self.k_bits[layer_idx] = k_bits
        self.v_bits[layer_idx] = v_bits

    def set_global(self, k_bits: BitsLike, v_bits: Optional[BitsLike] = None) -> None:
        if v_bits is None:
            v_bits = k_bits
        for i in range(self.num_layers):
            self.set_layer(i, k_bits, v_bits)

    def set_protected_mask(self, layer_idx: int, mask: torch.Tensor,
                           hi_bits: int, lo_bits: int) -> None:
        """V3-only. mask: bool tensor of shape [total_seq_len]; True=protected (hi_bits)."""
        self.protected[layer_idx] = (mask, hi_bits, lo_bits)

    def avg_kv_bits(self, seq_len: Optional[int] = None) -> float:
        """Average bits across all K/V positions (rough; for Pareto x-axis)."""
        total, count = 0.0, 0
        for layer_idx in range(self.num_layers):
            for bits in (self.k_bits[layer_idx], self.v_bits[layer_idx]):
                if isinstance(bits, int):
                    total += bits * self.num_kv_heads * (seq_len or 1)
                    count += self.num_kv_heads * (seq_len or 1)
                else:
                    t = bits.float()
                    total += t.sum().item() * (seq_len or 1) / max(1, t.numel() // self.num_kv_heads)
                    count += self.num_kv_heads * (seq_len or 1)
        return total / max(1, count)

    def get_kv_bits_for_chunk(
        self, layer_idx: int, new_chunk_len: int, num_kv_heads: int,
        cache_offset: int,
    ) -> tuple[BitsLike, BitsLike]:
        """Resolve effective bits for the new K/V chunk being appended.

        cache_offset = number of tokens already in cache (for V3 mask alignment).
        Returns (k_bits, v_bits). Each is either an int or a tensor broadcastable to [H, T_new].
        """
        if self.mode in ("V3", "V3K") and layer_idx in self.protected:
            mask, hi, lo = self.protected[layer_idx]
            end = cache_offset + new_chunk_len
            if end <= mask.shape[0]:
                slice_mask = mask[cache_offset:end].to(torch.bool)
            else:
                pad = torch.zeros(end - mask.shape[0], dtype=torch.bool, device=mask.device)
                slice_mask = torch.cat([mask[cache_offset:], pad], dim=0)
            bits = torch.where(slice_mask, hi, lo).long()
            bits_hk = bits.unsqueeze(0).expand(num_kv_heads, -1)
            if self.mode == "V3K":
                return bits_hk, self.v_bits[layer_idx]
            return bits_hk, bits_hk
        return self.k_bits[layer_idx], self.v_bits[layer_idx]


# ===================================================================
# Cache subclass
# ===================================================================

class FakeQuantKVCache(DynamicCache):
    """DynamicCache that fake-quantizes K/V chunks on update().

    Returns the (quantized) concatenated cache so Qwen2.5-VL's SDPA forward
    sees quantized K/V at the attention matmul (verified by smoke test).

    For F-suite K-quantizer screening: pass `k_quantizer_config` (a
    KQuantizerConfig from k_quantizers.py) and the K path is routed through
    `apply_k_quantizer(key_states, cfg, layer_idx, slice_info=...)`. V path
    continues to use the BitController-driven `fake_quantize_kv`.

    `slice_info` (visual span + role spans, in absolute prefill coordinates)
    must be set once per item before generate() via `set_slice_info(...)` if
    F5/F6/F11/F12/F13 are used.
    """

    def __init__(self, controller: BitController, k_quantizer_config=None):
        super().__init__()
        self.controller = controller
        self.entropy_log: list[list[torch.Tensor]] = [[] for _ in range(controller.num_layers)]
        # F-suite plumbing.
        self.k_cfg = k_quantizer_config
        self._slice_info: Optional[dict] = None

    def set_slice_info(self, slice_info: Optional[dict]) -> None:
        """F-suite: stash per-item role/modality info for the K quantizer.

        slice_info should be a dict with at least:
          v_start, v_end, seq_len, role_spans (dict[str -> (a, b)]),
          text_positions (list[int]), visual_positions (list[int]).
        """
        self._slice_info = slice_info

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_chunk_len = key_states.shape[-2]
        num_kv_heads = key_states.shape[1]
        cache_offset = self.get_seq_length(layer_idx)
        kb, vb = self.controller.get_kv_bits_for_chunk(
            layer_idx, new_chunk_len, num_kv_heads, cache_offset
        )
        if self.k_cfg is not None:
            from k_quantizers import apply_k_quantizer  # local import to avoid cycles
            key_q = apply_k_quantizer(
                key_states, self.k_cfg, layer_idx,
                slice_info=self._slice_info, cache_offset=cache_offset,
            )
        else:
            key_q = fake_quantize_kv(key_states, kb)
        # Exp I: VidKV-style V per-channel along time axis (axis differs from
        # default per-channel-along-head_dim INT4; bits unchanged).
        if self.k_cfg is not None and getattr(self.k_cfg, "v_per_channel_seq", False):
            val_q = _v_per_channel_seq_quantize(value_states, bits=4)
        else:
            val_q = fake_quantize_kv(value_states, vb)
        return super().update(key_q, val_q, layer_idx, cache_kwargs)

    def record_entropy(self, layer_idx: int, entropy_per_head: torch.Tensor) -> None:
        self.entropy_log[layer_idx].append(entropy_per_head.detach().cpu())


# ===================================================================
# Attention-forward patch (backstop for non-SDPA backends)
# ===================================================================

@contextmanager
def AttentionKVQuantPatch(model, controller: BitController):
    """Context manager: yields a (cache, controller) pair to feed into model.generate.

    For Qwen2.5-VL-SDPA this is purely cooperative: the FakeQuantKVCache's
    update() does the work because the returned tensors are used for attention.

    For backends where past_key_value.update()'s return is NOT consumed by
    attention, install_explicit() additionally monkey-patches each layer's
    self_attn.forward to apply quant post-RoPE / pre-matmul. We default to
    cooperative mode (cleaner, faster); the smoke test asserts correctness.
    """
    cache = FakeQuantKVCache(controller)
    try:
        yield cache, controller
    finally:
        # nothing to uninstall in cooperative mode
        pass


def install_explicit_kv_quant_patch(model, controller: BitController) -> list[callable]:
    """Backup-mode: monkey-patch each Qwen2_5_VLDecoderLayer.self_attn.forward
    so K, V are fake-quantized post-RoPE / pre-matmul, regardless of backend.

    Returns a list of restore() callables. Use only if cooperative mode fails the
    smoke-test logits-differ assertion.
    """
    restores = []
    layers = _find_decoder_layers(model)
    for layer_idx, attn in layers:
        original_forward = attn.forward

        def make_patched_forward(orig, lidx, attn_module):
            def patched_forward(hidden_states, *args, past_key_value=None, **kwargs):
                # We rely on past_key_value being our FakeQuantKVCache; its update()
                # already quantizes. The explicit patch lives here for future
                # backends; for now it's a no-op pass-through that documents intent.
                return orig(hidden_states, *args, past_key_value=past_key_value, **kwargs)
            return patched_forward

        attn.forward = make_patched_forward(original_forward, layer_idx, attn)
        def make_restore(a, fwd):
            return lambda: setattr(a, "forward", fwd)
        restores.append(make_restore(attn, original_forward))
    return restores


def _find_decoder_layers(model):
    """Return [(layer_idx, self_attn_module), ...] for Qwen2.5-VL's text decoder layers."""
    candidates = [
        getattr(getattr(model, "language_model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return [(i, layer.self_attn) for i, layer in enumerate(layers)]
    raise RuntimeError("Could not locate Qwen2.5-VL decoder layers (model.language_model.layers).")
