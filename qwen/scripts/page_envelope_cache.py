"""Page-aware KV cache for Exp P (Quest + FormatBook).

PageAwareFakeQuantKVCache subclasses FakeQuantKVCache. On each layer's update():
  1. Applies the K-quantizer config (e.g. J12 = F9 + INT8 sidecode), via the
     existing apply_k_quantizer dispatch.
  2. Computes per-page (k_min, k_max) envelopes from the quantized K (free —
     we already touch K once).
  3. Stores the full envelope under self.envelopes[layer_idx] for later analysis,
     and writes self.most_recent_envelope / most_recent_layer_idx so the SDPA
     wrapper can read the just-computed envelope without a global counter.

The cache does NOT make per-page format decisions (that's the SDPA wrapper's job
for the "formatbook" route policy — it downgrades cold pages by re-quantizing in
place at SDPA time). This keeps the cache simple and reusable across P2-P6.
"""
from __future__ import annotations

from typing import Optional

import torch

from fake_quant_kv_cache import BitController, FakeQuantKVCache, fake_quantize_kv
from k_quantizers import apply_k_quantizer
from page_layout import PageLayout
from quest_scorer import compute_page_envelope


class PageAwareFakeQuantKVCache(FakeQuantKVCache):
    """FakeQuantKVCache that also captures per-page K envelopes per layer.

    Constructor extras:
      page_layout: PageLayout for the current item. Set via `set_page_layout`
                   *before* `model.generate` so envelopes are computed correctly
                   on the prefill chunk.
      needle_page_idx: optional; recorded for diagnostics in routing_log.
    """

    def __init__(self, controller: BitController, k_quantizer_config=None,
                 page_layout: Optional[PageLayout] = None,
                 needle_page_idx: Optional[int] = None):
        super().__init__(controller, k_quantizer_config=k_quantizer_config)
        self.page_layout: Optional[PageLayout] = page_layout
        self.envelopes: dict[int, torch.Tensor] = {}  # layer_idx -> [H_kv, n_pages, D, 2]
        self.most_recent_envelope: Optional[torch.Tensor] = None
        self.most_recent_layer_idx: Optional[int] = None
        self.needle_page_idx: Optional[int] = needle_page_idx
        # Filled by the SDPA wrapper after a routing decision; the driver reads
        # this to write per-item diagnostics.
        self.routing_log: dict[int, dict] = {}  # layer_idx -> {policy, active, cold, scores, ...}

    def set_page_layout(self, layout: PageLayout, needle_page_idx: Optional[int] = None) -> None:
        """Plumb a fresh PageLayout for the next item. Resets envelopes and routing logs."""
        self.page_layout = layout
        self.needle_page_idx = (
            needle_page_idx if needle_page_idx is not None else layout.needle_page_idx
        )
        self.envelopes = {}
        self.routing_log = {}
        self.most_recent_envelope = None
        self.most_recent_layer_idx = None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # 1) Quantize K via the configured K-quantizer (e.g. J12).
        new_chunk_len = key_states.shape[-2]
        num_kv_heads = key_states.shape[1]
        cache_offset = self.get_seq_length(layer_idx)
        kb, vb = self.controller.get_kv_bits_for_chunk(
            layer_idx, new_chunk_len, num_kv_heads, cache_offset
        )
        if self.k_cfg is not None:
            key_q = apply_k_quantizer(
                key_states, self.k_cfg, layer_idx,
                slice_info=self._slice_info, cache_offset=cache_offset,
            )
        else:
            key_q = fake_quantize_kv(key_states, kb)
        val_q = fake_quantize_kv(value_states, vb)

        # 2) Compute per-page envelope on the quantized K (only meaningful during
        # prefill when cache_offset == 0; on later updates the chunk is small
        # and we re-merge into self.envelopes).
        if self.page_layout is not None:
            env = compute_page_envelope(key_q, self.page_layout)  # [H_kv, n_pages, D, 2]
            if cache_offset == 0:
                self.envelopes[layer_idx] = env
            else:
                prev = self.envelopes.get(layer_idx)
                if prev is not None:
                    # Merge: min(prev_min, env_min), max(prev_max, env_max)
                    prev[..., 0] = torch.minimum(prev[..., 0], env[..., 0])
                    prev[..., 1] = torch.maximum(prev[..., 1], env[..., 1])
                else:
                    self.envelopes[layer_idx] = env
            self.most_recent_envelope = self.envelopes[layer_idx]
            self.most_recent_layer_idx = layer_idx

        # 3) Delegate to grandparent for the actual concat (skip FakeQuantKVCache.update
        # since we already quantized above).
        from transformers.cache_utils import DynamicCache
        return DynamicCache.update(self, key_q, val_q, layer_idx, cache_kwargs)
