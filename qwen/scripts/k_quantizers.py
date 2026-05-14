"""F-suite K quantizer kernels for Exp F (K-quantizer repair screening).

Single responsibility: given K of shape [B, H_kv, T, D] (post-RoPE) plus a
KQuantizerConfig, return K_q of the same shape and dtype, fake-quantized
according to the chosen kind. No I/O. V is untouched (the caller continues
to call `fake_quantize_kv` for V).

Kinds:

  bf16                      pass-through (= F0 anchor with bits=16)
  uniform_int4              current per-channel-along-head_dim INT4 (= F1 floor;
                            forwards to fake_quant_kv_cache.fake_quantize_kv)
  kivi_per_channel_seq      KIVI-lite. Per-(B, H_kv, D) scale computed from
                            seq-dim max-abs (scale shape [B, H_kv, 1, D]).
                            (= F4)
  kivi_text_visual_split    F4 + separate per-channel scales for text-K vs
                            visual-K, computed at runtime from slice_info.
                            (= F5)
  kivi_role_split           F4 + separate per-channel scales for prompt roles
                            (header / question / options / instruction /
                             answer_prefix / visual). (= F6)
  kivi_p99_5                F4 + 99.5-percentile clipping replaces max-abs. (= F7)
  kivi_outlier8             F4 + top-8 outlier channels per (L, H_kv) at BF16. (= F8)
  kivi_outlier16            F4 + top-16 outlier channels per (L, H_kv) at BF16. (= F9)
  score_cal_generic         F4 + per-channel scale reweighted by sqrt(q_energy)
                            from calibration. (= F10)
  score_cal_block_tt_heavy  F4 + block-score reweighting w_TT=4, w_TV=1,
                            w_VT=1, w_VV=0.5. Separate text/visual scales. (= F11)
  score_cal_block_balanced  F4 + balanced block-score reweighting (all w=1).
                            Separate text/visual scales. (= F12)
  score_cal_text_only       F4 modified: text-K with score-cal scales, visual-K
                            and V at current INT4. (= F13)

Calibration data (when needed) is supplied via cfg.calib, a dict-like:
  {
    "k_channel_energy": np.ndarray [L, H_kv, D]    # for outlier ranking
    "outlier_channel_idx": np.ndarray [L, H_kv, n] # precomputed top-N indices
    "q_energy":           np.ndarray [L, H_kv, D]  # E[Q_d**2] all positions
    "q_energy_text":      np.ndarray [L, H_kv, D]  # E[Q_d**2] text positions
    "q_energy_visual":    np.ndarray [L, H_kv, D]  # E[Q_d**2] visual positions
  }
The caller (`expF_kquant_screen.py`) loads the NPZ once at startup and
attaches it to each KQuantizerConfig that needs it.

Slice info (for F5/F6/F13/T-mini) is passed at runtime via `slice_info` argument:
  {
    "v_start": int, "v_end": int,        # visual span (absolute positions)
    "text_positions": list[int],          # absolute positions of non-visual tokens
    "visual_positions": list[int],        # absolute positions of visual tokens
    "role_spans": {role: (a, b)} for     # role boundaries (absolute positions)
                  role in {header, question, options, instruction,
                           answer_prefix, visual}
    "seq_len": int,
    # T-mini fields (PageLocal / PageSentinel / Random*):
    "page_boundaries": list[(start, end, kind)],   # absolute; kind in
                                                   #   {"text",
                                                   #    "in_context_image",
                                                   #    "choice_image"}
    "visual_token_positions_per_image": list[list[int]],  # one list per visual
                                                          #   page (absolute pos)
    "text_chunk_positions": list[list[int]],       # one list per text page
                                                   #   (absolute pos)
    "item_id": str,                                # used to seed per-item RNG
                                                   #   for kivi_random_page_local
                                                   #   and random_visual sentinel
  }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


KQUANTIZER_KINDS = (
    "bf16",
    "uniform_int4",
    "kivi_per_channel_seq",
    "kivi_text_visual_split",
    "kivi_role_split",
    "kivi_p99_5",
    "kivi_outlier8",
    "kivi_outlier16",
    "score_cal_generic",
    "score_cal_block_tt_heavy",
    "score_cal_block_balanced",
    "score_cal_text_only",
    # H-suite: temporal-windowed KIVI. One per-channel scale per temporal
    # segment instead of one scale shared across the whole sequence. Stays at
    # TRUE 4.00 KV bits (no outlier spend; scale metadata is negligible).
    "kivi_temporal_window",
    # T-mini: VLM page-aware K formats.
    # PageLocal-F4: one per-channel scale per (Page.start, Page.end) from PageLayout.
    "kivi_page_local",
    # RandomPageLocal-F4: same page-count budget, randomly chosen boundary positions
    # (seeded per item). Stronger control than TokenBlock.
    "kivi_random_page_local",
    # ImageOnlyLocal-F4: page-local scales only on visual pages; text gets one global scale.
    "kivi_image_only_local",
    # TextOnlyLocal-F4: page-local scales only on text pages; visual gets one global scale.
    "kivi_text_only_local",
    # PageSentinel composite: base kind (F4 or PageLocal-F4) + sentinel positions kept at
    # original BF16 (keep-from-original, like F9 outlier sidecode but on POSITIONS).
    "kivi_page_sentinel",
)


@dataclass
class KQuantizerConfig:
    name: str
    kind: str
    bits: int = 4
    percentile: Optional[float] = None
    n_outliers: Optional[int] = None
    score_cal_weights: Optional[dict] = None  # {w_TT, w_TV, w_VT, w_VV}
    calib: Optional[dict] = field(default=None, repr=False)
    group_size: int = 0  # 0 means "no head_dim grouping" — pure per-channel-seq
    text_only: bool = False
    # H-suite (kivi_temporal_window): number of K-scale segments inside the
    # visual span (or across the whole sequence in token_block mode).
    n_temporal_windows: int = 0
    # "visual_only" -> text-prefix + N visual windows + text-suffix, each with
    # its own per-channel scale.
    # "token_block" -> N equal-token segments across the whole sequence,
    # ignoring modality boundaries (control for the visual-time hypothesis).
    temporal_mode: str = "visual_only"
    # Exp I: VidKV-style V quantization. When True, the cache subclass routes
    # V through a per-(B, H_kv, 1, D) channel-axis scale (computed via time-
    # axis max-abs) instead of the default per-channel-along-head_dim INT4.
    # Bits per element unchanged (still INT4); axis is what changes.
    v_per_channel_seq: bool = False
    # Exp J: alternate calib key for outlier-channel index lookup.
    # When set, _outlier_channel_indices reads cfg.calib[outlier_idx_key]
    # instead of the default 'outlier_channel_idx_top16'. Used to swap in
    # cross-modal scoring schemes (TT/TV/VT/VV/BAL/PIVOT/TT+TV).
    outlier_idx_key: Optional[str] = None
    # Exp J: storage bits for protected outlier channels.
    # 16 (default, F8/F9 path) = BF16 keep-from-original; 8/6 = INT-N
    # quantize-then-restore (cheaper sidecode at the cost of small noise).
    outlier_storage_bits: int = 16
    # Exp J: layer-adaptive outlier budget. Resolved at build time from a
    # tuple (cell_risk_calib_key, top_fraction) into a [num_layers, num_kv_heads]
    # int tensor with budget=n_outliers in the top fraction of cells (by risk)
    # and 0 elsewhere. When set as a tensor, _kivi_outlier honors it per-cell.
    layer_adaptive_outlier_budget: Optional[object] = None
    # T-mini: PageSentinel composite — base K format applied first, then sentinel
    # positions are restored to original (BF16). base_kind must be one of the
    # non-sentinel KQUANTIZER_KINDS. sentinel_kind in {first_visual, last_visual,
    # random_visual, first_text} chooses how to pick the sentinel positions from
    # the page boundaries in slice_info. sentinel_n_per_page is the count per page.
    base_kind: Optional[str] = None
    sentinel_kind: Optional[str] = None
    sentinel_n_per_page: int = 0
    # T-mini: random seed namespace for kivi_random_page_local and the
    # random_visual sentinel pattern. Combined with item-id at runtime so the
    # same item produces the same random boundaries / sentinels each pass.
    random_seed_namespace: Optional[str] = None

    def __post_init__(self):
        if self.kind not in KQUANTIZER_KINDS:
            raise ValueError(f"unknown KQuantizerConfig.kind={self.kind!r}")

    def avg_kv_bits(self, n_outlier_chan_total: int = 0,
                    head_dim: int = 128, n_layers: int = 28,
                    n_kv_heads: int = 4) -> float:
        """Effective avg KV bits (K + V averaged), assuming V = INT4 by F-suite convention.
        For outlier-channel kinds, n_outlier_chan_total = sum over (L, H_kv) of n
        protected BF16 channels.
        """
        if self.kind == "bf16":
            k_bits = 16.0
        else:
            total_chan = n_layers * n_kv_heads * head_dim
            if n_outlier_chan_total > 0:
                bf16_frac = n_outlier_chan_total / total_chan
                k_bits = 16.0 * bf16_frac + float(self.bits) * (1.0 - bf16_frac)
            else:
                k_bits = float(self.bits)
        v_bits = 4.0  # F-suite always V = INT4
        return (k_bits + v_bits) / 2.0


# ===================================================================
# Low-level quant primitives
# ===================================================================


def _clamp_min(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.clamp_min(eps)


def _per_channel_seq_scale(K: torch.Tensor, qmax: float,
                           percentile: Optional[float] = None) -> torch.Tensor:
    """Per-(B, H_kv, channel) scale across seq dim.

    K: [B, H_kv, T, D]  ->  scale: [B, H_kv, 1, D]

    If percentile is None, use max-abs. Else use the given percentile of |K|
    along seq, where percentile in (0, 100].
    """
    abs_K = K.abs().float()
    if percentile is None or percentile >= 100.0:
        amax = abs_K.amax(dim=-2, keepdim=True)
    else:
        # quantile expects fraction in [0, 1]
        q = max(0.0, min(1.0, percentile / 100.0))
        # torch.quantile is along a single dim; flatten across non-target dims
        amax = torch.quantile(abs_K, q=q, dim=-2, keepdim=True)
    return _clamp_min(amax / qmax).to(K.dtype)


def _quantize_with_scale(K: torch.Tensor, scale: torch.Tensor,
                         qmax: float) -> torch.Tensor:
    """Round-to-nearest INT-N with given scale; same shape and dtype as K.

    scale broadcasts over the appropriate dims. Standard symmetric round-clamp.
    """
    Kf = K.float()
    sf = scale.float()
    q = (Kf / sf).round().clamp(-qmax, qmax)
    return (q * sf).to(K.dtype)


def _qmax_from_bits(bits: int) -> float:
    return float(2 ** (bits - 1) - 1)


def _uniform_int_n_head_dim(x: torch.Tensor, bits: int,
                            group_size: int = 128) -> torch.Tensor:
    """Local copy of fake_quant_kv_cache._int_n_symmetric for the F1 path.

    Per-channel symmetric INT-N quantization along head_dim with grouping.
    Inlined here so this module has no cross-script imports (the smoke
    test can therefore run on machines without `transformers` installed).
    """
    qmax = float(2 ** (bits - 1) - 1)
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


# ===================================================================
# Kind dispatch
# ===================================================================


def apply_k_quantizer(K: torch.Tensor, cfg: KQuantizerConfig,
                      layer_idx: int,
                      slice_info: Optional[dict] = None,
                      cache_offset: int = 0) -> torch.Tensor:
    """Apply the configured K quantizer to a chunk of K.

    K: [B, H_kv, T, D] post-RoPE keys for one layer's chunk.
    layer_idx: integer layer index (0..num_layers-1) for calib lookup.
    slice_info: dict (see module docstring) describing role/modality positions
                in absolute coordinates over [0, seq_len). Required for F5/F6/F13.
    cache_offset: number of tokens already in cache (0 during prefill); used
                  to align slice_info absolute positions with the new chunk.

    Returns K_q of the same shape and dtype.
    """
    if K.numel() == 0:
        return K

    kind = cfg.kind
    if kind == "bf16":
        return K  # F0: pass-through

    if kind == "uniform_int4":
        # F1 floor — current per-channel-along-head_dim quantizer.
        return _uniform_int_n_head_dim(K, int(cfg.bits))

    qmax = _qmax_from_bits(cfg.bits)

    if kind == "kivi_per_channel_seq":
        # F4: per-(B, H_kv, channel) scale across seq.
        scale = _per_channel_seq_scale(K, qmax, percentile=None)
        return _quantize_with_scale(K, scale, qmax)

    if kind == "kivi_p99_5":
        # F7: same as F4 but use 99.5 percentile instead of max-abs.
        pct = float(cfg.percentile) if cfg.percentile is not None else 99.5
        scale = _per_channel_seq_scale(K, qmax, percentile=pct)
        return _quantize_with_scale(K, scale, qmax)

    if kind == "kivi_text_visual_split":
        # F5: separate per-channel scales for text/visual at runtime.
        return _kivi_modality_split(K, cfg, slice_info, cache_offset, qmax)

    if kind == "kivi_role_split":
        # F6: separate per-channel scales per prompt role.
        return _kivi_role_split(K, cfg, slice_info, cache_offset, qmax)

    if kind in ("kivi_outlier8", "kivi_outlier16"):
        # F8/F9: top-N outlier channels per (L, H_kv) at BF16; rest at INT4.
        return _kivi_outlier(K, cfg, layer_idx, qmax)

    if kind == "score_cal_generic":
        # F10: per-channel scale reweighted by sqrt(E[Q_d^2]).
        return _score_cal_generic(K, cfg, layer_idx, qmax)

    if kind in ("score_cal_block_tt_heavy", "score_cal_block_balanced"):
        # F11/F12: block-score reweighting with separate text/visual scales.
        return _score_cal_block(K, cfg, layer_idx, slice_info, cache_offset, qmax)

    if kind == "score_cal_text_only":
        # F13: text-K with score-cal scales, visual-K with current INT4.
        return _score_cal_text_only(K, cfg, layer_idx, slice_info, cache_offset, qmax)

    if kind == "kivi_temporal_window":
        # H3/H4/H6: per-channel scales applied separately to (text-prefix,
        # visual-window-1, ..., visual-window-N, text-suffix).
        # H5: per-channel scales applied to N equal-token blocks across the
        # whole sequence, ignoring modality.
        # Exp I: when cfg.n_outliers > 0, restore top-N outlier channels at
        # BF16 after the per-segment quantization (TempWin + outlier-N).
        return _kivi_temporal_window(
            K, cfg, layer_idx, slice_info, cache_offset, qmax,
            n_windows=int(cfg.n_temporal_windows or 0),
            mode=str(cfg.temporal_mode or "visual_only"),
        )

    if kind == "kivi_page_local":
        # T8/C6: one per-channel scale per Page from slice_info["page_boundaries"].
        return _kivi_page_local(K, cfg, slice_info, cache_offset, qmax,
                                page_filter=None)

    if kind == "kivi_random_page_local":
        # T7: same page-count as PageLocal but boundaries randomly chosen per item.
        return _kivi_random_page_local(K, cfg, slice_info, cache_offset, qmax)

    if kind == "kivi_image_only_local":
        # T9: page-local scales only on visual pages; text gets one global scale.
        return _kivi_page_local(K, cfg, slice_info, cache_offset, qmax,
                                page_filter="visual_only")

    if kind == "kivi_text_only_local":
        # T10: page-local scales only on text pages; visual gets one global scale.
        return _kivi_page_local(K, cfg, slice_info, cache_offset, qmax,
                                page_filter="text_only")

    if kind == "kivi_page_sentinel":
        # T11-T16: base K format + sentinel positions restored to original BF16.
        return _kivi_page_sentinel(K, cfg, layer_idx, slice_info, cache_offset, qmax)

    raise ValueError(f"apply_k_quantizer: unhandled kind={kind!r}")


# ===================================================================
# Modality / role split helpers
# ===================================================================


def _chunk_position_indices(slice_info: Optional[dict], cache_offset: int,
                            T: int, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (text_idx, visual_idx) — chunk-local position indices (in [0, T)).

    Falls back to "all text" if slice_info is missing (defensive: append step
    after prefill is generated text).
    """
    if slice_info is None:
        all_text = torch.arange(T, dtype=torch.long)
        empty = torch.empty(0, dtype=torch.long)
        return all_text, empty
    v_start = int(slice_info["v_start"])
    v_end = int(slice_info["v_end"])
    abs_lo = cache_offset
    abs_hi = cache_offset + T
    text_local: list[int] = []
    vis_local: list[int] = []
    for i in range(T):
        a = abs_lo + i
        if v_start <= a < v_end:
            vis_local.append(i)
        else:
            text_local.append(i)
    return torch.tensor(text_local, dtype=torch.long), torch.tensor(vis_local, dtype=torch.long)


def _kivi_modality_split(K: torch.Tensor, cfg: KQuantizerConfig,
                         slice_info: Optional[dict], cache_offset: int,
                         qmax: float) -> torch.Tensor:
    """F5: separate per-channel scales for text vs visual K positions.

    K: [B, H_kv, T, D]. Compute scale_text from K[..., text_idx, :] and
    scale_visual from K[..., visual_idx, :]; quantize each subset with its
    own scale.
    """
    B, H, T, D = K.shape
    text_idx, vis_idx = _chunk_position_indices(slice_info, cache_offset, T, cfg.kind)
    text_idx = text_idx.to(K.device)
    vis_idx = vis_idx.to(K.device)

    K_q = K.clone()
    if text_idx.numel() > 0:
        K_text = K.index_select(dim=-2, index=text_idx)  # [B, H, n_text, D]
        s_text = _per_channel_seq_scale(K_text, qmax, percentile=None)  # [B, H, 1, D]
        K_text_q = _quantize_with_scale(K_text, s_text, qmax)
        K_q.index_copy_(dim=-2, index=text_idx, source=K_text_q)
    if vis_idx.numel() > 0:
        K_vis = K.index_select(dim=-2, index=vis_idx)
        s_vis = _per_channel_seq_scale(K_vis, qmax, percentile=None)
        K_vis_q = _quantize_with_scale(K_vis, s_vis, qmax)
        K_q.index_copy_(dim=-2, index=vis_idx, source=K_vis_q)
    return K_q


def _role_local_indices(slice_info: Optional[dict], cache_offset: int,
                        T: int) -> dict[str, torch.Tensor]:
    """Map each role -> chunk-local position indices."""
    roles = ("header", "question", "options", "instruction", "answer_prefix", "visual")
    out: dict[str, list[int]] = {r: [] for r in roles}
    if slice_info is None or "role_spans" not in slice_info:
        all_local = list(range(T))
        out["question"] = all_local  # fallback: bucket everything as text/question
        return {r: torch.tensor(out[r], dtype=torch.long) for r in roles}
    role_spans = slice_info["role_spans"]
    abs_lo = cache_offset
    abs_hi = cache_offset + T
    for i in range(T):
        a = abs_lo + i
        for r in roles:
            sp = role_spans.get(r)
            if sp is None:
                continue
            ra, rb = int(sp[0]), int(sp[1])
            if ra <= a < rb:
                out[r].append(i)
                break
        else:
            # No role matched. Bucket into question by default to keep behavior
            # consistent (this happens for tokens that fall in gaps like the
            # visual_wrapper boundary).
            out["question"].append(i)
    return {r: torch.tensor(out[r], dtype=torch.long) for r in roles}


def _kivi_role_split(K: torch.Tensor, cfg: KQuantizerConfig,
                     slice_info: Optional[dict], cache_offset: int,
                     qmax: float) -> torch.Tensor:
    """F6: per-prompt-role per-channel K scales."""
    B, H, T, D = K.shape
    role_idx = _role_local_indices(slice_info, cache_offset, T)
    K_q = K.clone()
    for r, idx in role_idx.items():
        if idx.numel() == 0:
            continue
        idx = idx.to(K.device)
        K_r = K.index_select(dim=-2, index=idx)
        s_r = _per_channel_seq_scale(K_r, qmax, percentile=None)
        K_r_q = _quantize_with_scale(K_r, s_r, qmax)
        K_q.index_copy_(dim=-2, index=idx, source=K_r_q)
    return K_q


# ===================================================================
# Temporal-windowed KIVI (H suite)
# ===================================================================


def _kivi_temporal_window(K: torch.Tensor, cfg: KQuantizerConfig,
                          layer_idx: int,
                          slice_info: Optional[dict], cache_offset: int,
                          qmax: float, n_windows: int,
                          mode: str = "visual_only") -> torch.Tensor:
    """One per-channel KIVI scale per temporal segment.

    visual_only mode:
      - text-prefix span [0, v_start): one scale (per-channel)
      - visual span [v_start, v_end): split into n_windows equal-token segments;
        one scale per segment.
      - text-suffix span [v_end, T): one scale (per-channel)
      Each segment is quantized with its own [B, H_kv, 1, D] scale.

    token_block mode:
      [0, T) split into n_windows equal-token segments. Modality is ignored.
      Used as a control to test whether the visual-time structure is the
      load-bearing thing or just generic local scaling.

    Exp I: when cfg.n_outliers > 0, after per-segment quantization, restore
    the top-N outlier channels per (L, H_kv) at BF16 from the original K.
    This composes TempWin K with F8/F9-style outlier protection.

    cache_offset > 0 (decode-time, not used in MCQ scoring) falls back to plain
    F4 per-channel-seq scale on the small new chunk. Matches F5/F6 pattern.
    """
    B, H, T, D = K.shape
    if T == 0:
        return K

    # Decode-time fallback: small new chunk, no temporal structure to exploit.
    if cache_offset > 0:
        scale = _per_channel_seq_scale(K, qmax, percentile=None)
        return _quantize_with_scale(K, scale, qmax)

    # Degenerate window count: behave like F4.
    if n_windows is None or n_windows <= 1:
        scale = _per_channel_seq_scale(K, qmax, percentile=None)
        K_q = _quantize_with_scale(K, scale, qmax)
    else:
        if mode == "token_block":
            boundaries = [int(round(T * i / n_windows)) for i in range(n_windows + 1)]
        else:
            # visual_only: text-prefix + N visual windows + text-suffix.
            v_start = 0
            v_end = T
            if slice_info is not None and "v_start" in slice_info and "v_end" in slice_info:
                v_start = max(0, min(T, int(slice_info["v_start"])))
                v_end = max(v_start, min(T, int(slice_info["v_end"])))
            if v_end <= v_start:
                # Degenerate: no visual span detected. Fall back to F4.
                scale = _per_channel_seq_scale(K, qmax, percentile=None)
                K_q = _quantize_with_scale(K, scale, qmax)
                # Outlier restoration still applies in degenerate path so I8/I13
                # don't silently degrade if visual span detection fails.
                n_out = int(cfg.n_outliers or 0)
                if n_out > 0:
                    outlier_idx = _outlier_channel_indices(cfg, layer_idx, H, D, n_out).to(K.device)
                    for h in range(H):
                        ch = outlier_idx[h]
                        if ch.numel() > 0:
                            K_q[:, h, :, ch] = K[:, h, :, ch]
                return K_q
            v_boundaries = [v_start + int(round((v_end - v_start) * i / n_windows))
                            for i in range(n_windows + 1)]
            boundaries = [0] + v_boundaries + [T]
            # Dedup adjacents (handles cases where v_start==0 or v_end==T or
            # window boundary equals v_start/v_end).
            boundaries = sorted(set(boundaries))

        K_q = K.clone()
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            if b <= a:
                continue
            seg = K[:, :, a:b, :]
            s = _per_channel_seq_scale(seg, qmax, percentile=None)
            K_q[:, :, a:b, :] = _quantize_with_scale(seg, s, qmax)

    # Exp I: outlier protection on top of TempWin (I8/I13).
    n_out = int(cfg.n_outliers or 0)
    if n_out > 0:
        outlier_idx = _outlier_channel_indices(cfg, layer_idx, H, D, n_out).to(K.device)
        for h in range(H):
            ch = outlier_idx[h]
            if ch.numel() > 0:
                K_q[:, h, :, ch] = K[:, h, :, ch]
    return K_q


# ===================================================================
# T-mini: page-aware K quantizers (PageLocal, RandomPageLocal,
# ImageOnlyLocal, TextOnlyLocal, PageSentinel)
# ===================================================================


_VISUAL_PAGE_KINDS = ("in_context_image", "choice_image")


def _kivi_page_local(K: torch.Tensor, cfg: KQuantizerConfig,
                     slice_info: Optional[dict], cache_offset: int,
                     qmax: float, page_filter: Optional[str] = None) -> torch.Tensor:
    """T8/T9/T10: per-page per-channel K scales.

    slice_info["page_boundaries"] is a list of (start, end, kind) tuples in
    absolute coordinates where kind ∈ {"text", "in_context_image", "choice_image"}.

    page_filter:
      None           -> every page gets a local scale (T8 PageLocal-F4)
      "visual_only"  -> visual pages get local scales; text spans share one global
                        scale (T9 ImageOnlyLocal-F4)
      "text_only"    -> text pages get local scales; visual spans share one global
                        scale (T10 TextOnlyLocal-F4)

    Decode-time fallback (cache_offset > 0): plain F4 per-channel-seq on the
    small new chunk — generated tokens lie after the prefill page coverage.
    """
    B, H, T, D = K.shape
    if T == 0:
        return K

    # Fallback when slice_info is missing or has no page_boundaries.
    page_boundaries = None
    if slice_info is not None:
        page_boundaries = slice_info.get("page_boundaries")
    if not page_boundaries or cache_offset > 0:
        scale = _per_channel_seq_scale(K, qmax, percentile=None)
        return _quantize_with_scale(K, scale, qmax)

    abs_lo = cache_offset
    abs_hi = cache_offset + T
    K_q = K.clone()

    # Page intersections that get their own per-channel scale.
    local_positions: list[int] = []  # chunk-local indices receiving local scales
    # Positions that belong to the modality-pooled (single-scale) bucket.
    pooled_positions: list[int] = []

    for (lo, hi, kind) in page_boundaries:
        chunk_lo = max(int(lo) - abs_lo, 0)
        chunk_hi = min(int(hi) - abs_lo, T)
        if chunk_hi <= chunk_lo:
            continue
        is_visual = kind in _VISUAL_PAGE_KINDS
        if page_filter is None:
            give_local_scale = True
        elif page_filter == "visual_only":
            give_local_scale = is_visual
        elif page_filter == "text_only":
            give_local_scale = (not is_visual)
        else:
            raise ValueError(f"unknown page_filter={page_filter!r}")

        if give_local_scale:
            seg = K[:, :, chunk_lo:chunk_hi, :]
            scale = _per_channel_seq_scale(seg, qmax, percentile=None)
            K_q[:, :, chunk_lo:chunk_hi, :] = _quantize_with_scale(seg, scale, qmax)
            local_positions.extend(range(chunk_lo, chunk_hi))
        else:
            pooled_positions.extend(range(chunk_lo, chunk_hi))

    # Single pooled global scale for whichever modality was not page-locally scaled.
    if pooled_positions:
        idx = torch.tensor(pooled_positions, dtype=torch.long, device=K.device)
        K_pool = K.index_select(dim=-2, index=idx)
        scale = _per_channel_seq_scale(K_pool, qmax, percentile=None)
        K_pool_q = _quantize_with_scale(K_pool, scale, qmax)
        K_q.index_copy_(dim=-2, index=idx, source=K_pool_q)

    # Catch-all for positions not covered by any page boundary (rare, but
    # generated decode-time tokens or unparsed prefill ranges). Fall back to F4.
    covered = set(local_positions) | set(pooled_positions)
    leftover = [i for i in range(T) if i not in covered]
    if leftover:
        idx = torch.tensor(leftover, dtype=torch.long, device=K.device)
        K_left = K.index_select(dim=-2, index=idx)
        scale = _per_channel_seq_scale(K_left, qmax, percentile=None)
        K_left_q = _quantize_with_scale(K_left, scale, qmax)
        K_q.index_copy_(dim=-2, index=idx, source=K_left_q)

    return K_q


def _kivi_random_page_local(K: torch.Tensor, cfg: KQuantizerConfig,
                            slice_info: Optional[dict], cache_offset: int,
                            qmax: float) -> torch.Tensor:
    """T7: same number of pages as PageLocal, randomly chosen boundaries.

    Reads slice_info["page_boundaries"] for n_pages and slice_info["item_id"]
    for the per-item RNG seed. If page_boundaries is missing, falls back to F4.
    """
    B, H, T, D = K.shape
    if T == 0:
        return K
    page_boundaries = None
    item_id = None
    if slice_info is not None:
        page_boundaries = slice_info.get("page_boundaries")
        item_id = slice_info.get("item_id")
    if not page_boundaries or cache_offset > 0:
        scale = _per_channel_seq_scale(K, qmax, percentile=None)
        return _quantize_with_scale(K, scale, qmax)

    n_pages = len(page_boundaries)
    if n_pages <= 1:
        scale = _per_channel_seq_scale(K, qmax, percentile=None)
        return _quantize_with_scale(K, scale, qmax)

    # Deterministic per-item random boundary positions across [0, T).
    import random
    seed_basis = f"{cfg.random_seed_namespace or 'random_page_local'}:{item_id or 0}"
    rng = random.Random(abs(hash(seed_basis)) % (2 ** 31))
    # Pick n_pages-1 unique interior boundaries.
    if T - 1 <= n_pages - 1:
        # Not enough positions to split. Fallback.
        scale = _per_channel_seq_scale(K, qmax, percentile=None)
        return _quantize_with_scale(K, scale, qmax)
    interior = rng.sample(range(1, T), n_pages - 1)
    interior.sort()
    boundaries = [0] + interior + [T]

    K_q = K.clone()
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        if b <= a:
            continue
        seg = K[:, :, a:b, :]
        scale = _per_channel_seq_scale(seg, qmax, percentile=None)
        K_q[:, :, a:b, :] = _quantize_with_scale(seg, scale, qmax)
    return K_q


def _resolve_sentinel_positions(slice_info: Optional[dict],
                                sentinel_kind: str,
                                n_per_page: int,
                                seed_namespace: Optional[str]) -> list[int]:
    """Resolve sentinel token positions in absolute coordinates.

    sentinel_kind options:
      "first_visual" — first n_per_page visual tokens of each image page
      "last_visual"  — last n_per_page visual tokens of each image page
      "random_visual" — random n_per_page positions per image page (seeded per item)
      "first_text"   — first n_per_page tokens of each text page/chunk
    """
    if slice_info is None or n_per_page <= 0:
        return []
    pos: list[int] = []
    item_id = slice_info.get("item_id") or 0
    if sentinel_kind in ("first_visual", "last_visual", "random_visual"):
        per_image = slice_info.get("visual_token_positions_per_image") or []
        for img_idx, pos_list in enumerate(per_image):
            if not pos_list:
                continue
            if sentinel_kind == "first_visual":
                pos.extend(pos_list[:n_per_page])
            elif sentinel_kind == "last_visual":
                pos.extend(pos_list[-n_per_page:])
            else:  # random_visual
                import random
                seed_basis = f"{seed_namespace or 'random_sentinel'}:{item_id}:{img_idx}"
                rng = random.Random(abs(hash(seed_basis)) % (2 ** 31))
                n = min(n_per_page, len(pos_list))
                pos.extend(rng.sample(list(pos_list), n))
    elif sentinel_kind == "first_text":
        per_chunk = slice_info.get("text_chunk_positions") or []
        for pos_list in per_chunk:
            if not pos_list:
                continue
            pos.extend(pos_list[:n_per_page])
    else:
        raise ValueError(f"unknown sentinel_kind={sentinel_kind!r}")
    return sorted(set(pos))


def _kivi_page_sentinel(K: torch.Tensor, cfg: KQuantizerConfig,
                        layer_idx: int,
                        slice_info: Optional[dict], cache_offset: int,
                        qmax: float) -> torch.Tensor:
    """T11-T16: base K format + sentinel positions restored to original BF16.

    cfg.base_kind selects the base K format ("kivi_per_channel_seq" for Global-F4
    base in T11-T15; "kivi_page_local" for the combined T16). cfg.sentinel_kind
    chooses which positions to protect. cfg.sentinel_n_per_page controls count.

    Sentinel positions are kept at original BF16 (lossless on those positions),
    mirroring the F9 outlier sidecode pattern but on POSITIONS instead of
    CHANNELS.
    """
    base_kind = cfg.base_kind or "kivi_per_channel_seq"
    # Construct a transient base cfg. Reuse calib / outlier-storage fields if
    # the base needs them (e.g. PageLocal base with outlier sidecode is not in
    # T-mini, but keep extensibility).
    base_cfg = KQuantizerConfig(
        name=f"{cfg.name}:base",
        kind=base_kind,
        bits=cfg.bits,
        calib=cfg.calib,
        random_seed_namespace=cfg.random_seed_namespace,
    )
    K_q = apply_k_quantizer(K, base_cfg, layer_idx,
                            slice_info=slice_info, cache_offset=cache_offset)

    # Restore sentinel positions from original K (BF16 keep-from-original).
    if cache_offset > 0:
        # Sentinels are defined on prefill positions only; decode-time chunks
        # contain generated tokens which are never sentinel.
        return K_q
    sentinel_abs = _resolve_sentinel_positions(
        slice_info, cfg.sentinel_kind or "first_visual",
        int(cfg.sentinel_n_per_page or 0),
        cfg.random_seed_namespace,
    )
    if not sentinel_abs:
        return K_q
    B, H, T, D = K.shape
    abs_lo = cache_offset
    abs_hi = cache_offset + T
    sentinel_local = [p - abs_lo for p in sentinel_abs if abs_lo <= p < abs_hi]
    if not sentinel_local:
        return K_q
    idx = torch.tensor(sentinel_local, dtype=torch.long, device=K.device)
    K_q.index_copy_(dim=-2, index=idx, source=K.index_select(dim=-2, index=idx))
    return K_q


# ===================================================================
# Outlier-channel helpers (F8 / F9)
# ===================================================================


def _outlier_channel_indices(cfg: KQuantizerConfig, layer_idx: int,
                             num_kv_heads: int, head_dim: int,
                             n: int) -> torch.Tensor:
    """Return [H_kv, n] long tensor of channel indices to protect at BF16.

    Source priority:
      1. cfg.outlier_idx_key (Exp J: cross-modal schemes like
         'outlier_idx_TT_top16', 'outlier_idx_TT_TV_top16', etc.)
      2. cfg.calib['outlier_channel_idx_top16'] (F-suite default)
      3. Online argsort from cfg.calib['k_channel_energy'] (fallback)

    The selected key must be a numpy array of shape [L, H_kv, K] where K >= n;
    we return the first `n` columns sliced for the given layer.
    """
    if cfg.calib is None:
        raise RuntimeError(
            f"{cfg.name}: outlier kind requires cfg.calib with an outlier-index "
            f"array or 'k_channel_energy'."
        )
    key = cfg.outlier_idx_key or "outlier_channel_idx_top16"
    if key in cfg.calib:
        idx = cfg.calib[key]  # numpy [L, H_kv, K]
        if idx.shape[-1] >= n:
            return torch.as_tensor(idx[layer_idx, :, :n], dtype=torch.long)
        # Stored key has fewer channels than requested — fall through to
        # k_channel_energy argsort. Exp S Phase 1 needs top-24 / top-32 with
        # only top-16 precomputed.
    elif cfg.outlier_idx_key is not None:
        # User explicitly asked for a key that's missing — fail loudly rather
        # than silently falling back to the generic top-16.
        raise RuntimeError(
            f"{cfg.name}: cfg.outlier_idx_key={cfg.outlier_idx_key!r} not present in "
            f"cfg.calib (keys: {sorted(cfg.calib)})."
        )
    energy = cfg.calib.get("k_channel_energy")
    if energy is None:
        raise RuntimeError(
            f"{cfg.name}: cfg.calib missing 'k_channel_energy' (needed for n={n} "
            f"channels when precomputed key has fewer)."
        )
    e = torch.as_tensor(energy[layer_idx])  # [H_kv, D]
    return torch.argsort(e, dim=-1, descending=True)[:, :n]


def _kivi_outlier(K: torch.Tensor, cfg: KQuantizerConfig,
                  layer_idx: int, qmax: float) -> torch.Tensor:
    """F8/F9 + Exp J extensions: per-(L, H_kv) outlier-channel protection.

    Pipeline:
      1. Quantize all channels with KIVI per-channel-seq scale (F4 baseline).
      2. For each head h, look up its outlier-channel indices and the per-cell
         budget (uniform `cfg.n_outliers` or layer-adaptive from
         `cfg.layer_adaptive_outlier_budget`).
      3. Restore the protected channels from the original K (BF16) OR
         re-quantize them at `cfg.outlier_storage_bits` (INT8/INT6 sidecode).

    Layer-adaptive budget (Exp J): `cfg.layer_adaptive_outlier_budget` is a
    pre-resolved [num_layers, num_kv_heads] int tensor. Cells with budget=0
    contribute zero protected channels; cells with budget>0 use that count as
    the per-cell n. The default `cfg.n_outliers` field is ignored when this
    tensor is set.

    Sidecode storage (Exp J): `cfg.outlier_storage_bits` controls how protected
    channels are encoded:
      16: BF16 keep-from-original (F8/F9 default; lossless-on-protected).
      8/6: INT-N quantize-then-restore (cheaper sidecode; small added noise).
    """
    B, H, T, D = K.shape
    n_uniform = int(cfg.n_outliers or 0)

    # Resolve per-head budget. layer_adaptive_outlier_budget overrides n_uniform.
    la_budget = cfg.layer_adaptive_outlier_budget
    if la_budget is not None:
        # Expect a [L, H_kv] int tensor / numpy array. Must already be resolved
        # (via build-time cell-risk argsort) before kernel dispatch.
        budget_row = la_budget[layer_idx]  # length H
        # Allow numpy array or torch tensor.
        if not isinstance(budget_row, torch.Tensor):
            budget_row = torch.as_tensor(budget_row, dtype=torch.long)
        per_head_n = [int(budget_row[h].item()) for h in range(H)]
    else:
        per_head_n = [n_uniform] * H

    # If every head has budget=0, this degenerates to F4.
    if all(n <= 0 for n in per_head_n):
        scale = _per_channel_seq_scale(K, qmax, percentile=None)
        return _quantize_with_scale(K, scale, qmax)

    # The outlier-index lookup uses the largest per-head n; we slice per head later.
    n_max = max(per_head_n)
    outlier_idx_full = _outlier_channel_indices(cfg, layer_idx, H, D, n_max).to(K.device)
    # outlier_idx_full: [H, n_max]

    # Quantize all channels with KIVI per-channel-seq scale (F4 baseline).
    scale = _per_channel_seq_scale(K, qmax, percentile=None)  # [B, H, 1, D]
    K_q = _quantize_with_scale(K, scale, qmax)

    storage_bits = int(cfg.outlier_storage_bits or 16)
    storage_qmax = float(2 ** (storage_bits - 1) - 1) if storage_bits < 16 else None

    for h in range(H):
        n_h = per_head_n[h]
        if n_h <= 0:
            continue
        ch = outlier_idx_full[h, :n_h]  # [n_h]
        if storage_bits >= 16:
            # F8/F9 path: BF16 keep-from-original.
            K_q[:, h, :, ch] = K[:, h, :, ch]
        else:
            # Exp J INT-N sidecode: quantize protected channels with their own
            # per-channel-seq scale at the smaller bit width, then restore.
            K_outliers = K[:, h:h+1, :, ch]  # [B, 1, T, n_h]
            s = K_outliers.abs().float().amax(dim=-2, keepdim=True).clamp_min(1e-8) / storage_qmax
            s = s.to(K.dtype)
            K_outliers_q = (K_outliers.float() / s.float()).round().clamp(
                -storage_qmax, storage_qmax) * s.float()
            K_q[:, h, :, ch] = K_outliers_q[:, 0, :, :].to(K.dtype)
    return K_q


# ===================================================================
# Score-cal helpers (F10 / F11 / F12 / F13)
# ===================================================================


def _score_cal_scale(K: torch.Tensor, q_energy: torch.Tensor,
                     qmax: float) -> torch.Tensor:
    """Per-channel scale = max(|K|) per channel, reweighted by sqrt(q_energy_d).

    Centered so that the mean reweighting factor across channels equals 1
    (preserves the overall scale-magnitude regime — this is how the closed-
    form Q-energy reweighting is anchored).

    K: [B, H_kv, T, D]  (a single layer's chunk)
    q_energy: [H_kv, D] tensor (cal-derived E[Q_d^2] per channel per head;
              float32 numpy or torch tensor accepted)

    Returns scale: [B, H_kv, 1, D].
    """
    qe = torch.as_tensor(q_energy, dtype=torch.float32, device=K.device)  # [H_kv, D]
    qe = _clamp_min(qe, eps=1e-12)
    sqrt_qe = qe.sqrt()  # [H_kv, D]
    # Center per (head): factor / mean(factor) so that mean rewight = 1.
    mean_per_head = sqrt_qe.mean(dim=-1, keepdim=True).clamp_min(1e-8)  # [H_kv, 1]
    factor = sqrt_qe / mean_per_head  # [H_kv, D], per-head mean=1
    # Base per-channel max-abs scale (KIVI).
    base = K.abs().float().amax(dim=-2, keepdim=True)  # [B, H_kv, 1, D]
    # Apply factor — broadcast factor [H_kv, D] -> [1, H_kv, 1, D].
    scaled_amax = base * factor.unsqueeze(0).unsqueeze(-2)
    return _clamp_min(scaled_amax / qmax).to(K.dtype)


def _score_cal_generic(K: torch.Tensor, cfg: KQuantizerConfig,
                       layer_idx: int, qmax: float) -> torch.Tensor:
    """F10: closed-form Q-energy reweighting using cfg.calib['q_energy']."""
    if cfg.calib is None or "q_energy" not in cfg.calib:
        raise RuntimeError(f"{cfg.name}: requires cfg.calib['q_energy'].")
    qe = cfg.calib["q_energy"][layer_idx]  # [H_kv, D]
    scale = _score_cal_scale(K, qe, qmax)
    return _quantize_with_scale(K, scale, qmax)


def _score_cal_block(K: torch.Tensor, cfg: KQuantizerConfig, layer_idx: int,
                     slice_info: Optional[dict], cache_offset: int,
                     qmax: float) -> torch.Tensor:
    """F11/F12: block-score reweighting with separate text/visual scales.

    Text-K channel reweighting factor:
      w_text(d) = w_TT * E[Q_text_d^2] + w_VT * E[Q_vis_d^2]
    Visual-K channel reweighting factor:
      w_vis(d) = w_TV * E[Q_text_d^2] + w_VV * E[Q_vis_d^2]
    """
    if cfg.calib is None:
        raise RuntimeError(f"{cfg.name}: requires cfg.calib with q_energy_text/visual.")
    qet = cfg.calib.get("q_energy_text")
    qev = cfg.calib.get("q_energy_visual")
    if qet is None or qev is None:
        raise RuntimeError(f"{cfg.name}: cfg.calib missing q_energy_text/q_energy_visual.")
    w = cfg.score_cal_weights or {}
    w_TT = float(w.get("w_TT", 1.0))
    w_TV = float(w.get("w_TV", 1.0))
    w_VT = float(w.get("w_VT", 1.0))
    w_VV = float(w.get("w_VV", 1.0))

    qet_l = torch.as_tensor(qet[layer_idx], dtype=torch.float32, device=K.device)
    qev_l = torch.as_tensor(qev[layer_idx], dtype=torch.float32, device=K.device)
    w_text = w_TT * qet_l + w_VT * qev_l
    w_vis = w_TV * qet_l + w_VV * qev_l

    B, H, T, D = K.shape
    text_idx, vis_idx = _chunk_position_indices(slice_info, cache_offset, T, cfg.kind)
    text_idx = text_idx.to(K.device)
    vis_idx = vis_idx.to(K.device)

    K_q = K.clone()
    if text_idx.numel() > 0:
        K_text = K.index_select(dim=-2, index=text_idx)
        scale = _score_cal_scale(K_text, w_text, qmax)
        K_text_q = _quantize_with_scale(K_text, scale, qmax)
        K_q.index_copy_(dim=-2, index=text_idx, source=K_text_q)
    if vis_idx.numel() > 0:
        K_vis = K.index_select(dim=-2, index=vis_idx)
        scale = _score_cal_scale(K_vis, w_vis, qmax)
        K_vis_q = _quantize_with_scale(K_vis, scale, qmax)
        K_q.index_copy_(dim=-2, index=vis_idx, source=K_vis_q)
    return K_q


def _score_cal_text_only(K: torch.Tensor, cfg: KQuantizerConfig, layer_idx: int,
                         slice_info: Optional[dict], cache_offset: int,
                         qmax: float) -> torch.Tensor:
    """F13: text-K with score-cal scales; visual-K with current INT4 (per-channel
    along head_dim, group_size=128, like F1 floor).
    """
    if cfg.calib is None or "q_energy" not in cfg.calib:
        raise RuntimeError(f"{cfg.name}: requires cfg.calib['q_energy'].")
    qe = cfg.calib["q_energy"][layer_idx]  # [H_kv, D]

    B, H, T, D = K.shape
    text_idx, vis_idx = _chunk_position_indices(slice_info, cache_offset, T, cfg.kind)
    text_idx = text_idx.to(K.device)
    vis_idx = vis_idx.to(K.device)

    K_q = K.clone()
    if text_idx.numel() > 0:
        K_text = K.index_select(dim=-2, index=text_idx)
        s_text = _score_cal_scale(K_text, qe, qmax)
        K_text_q = _quantize_with_scale(K_text, s_text, qmax)
        K_q.index_copy_(dim=-2, index=text_idx, source=K_text_q)
    if vis_idx.numel() > 0:
        K_vis = K.index_select(dim=-2, index=vis_idx)
        K_vis_q = _uniform_int_n_head_dim(K_vis, int(cfg.bits))
        K_q.index_copy_(dim=-2, index=vis_idx, source=K_vis_q)
    return K_q


# ===================================================================
# Convenience: F-suite condition definitions
# ===================================================================


def _balanced_random_top_indices(num_layers: int, num_kv_heads: int,
                                 head_dim: int, n_per_block: int = 2,
                                 n_blocks: int = 4, n_top: int = 16,
                                 seed: int = 99):
    """Exp K K10 control: partition channel-dim into `n_blocks` equal
    contiguous slices (e.g. [0,32), [32,64), [64,96), [96,128) for D=128
    and 4 blocks) and randomly pick `n_per_block` channels from each slice
    per (L, H_kv). Result: [L, H_kv, n_top] int32 with the first
    n_blocks*n_per_block entries holding the random-per-block selections;
    padded with random remainder channels (not in earlier picks) up to n_top.

    This is the structural-balance-without-cross-modal-scoring control: it
    tests whether the J7 win is about cross-modal channel relevance, or just
    about enforcing balanced coverage across some arbitrary partition.
    """
    import numpy as np
    if head_dim % n_blocks != 0:
        raise ValueError(f"head_dim={head_dim} not divisible by n_blocks={n_blocks}")
    block_size = head_dim // n_blocks
    rng = np.random.default_rng(seed)
    out = np.full((num_layers, num_kv_heads, n_top), -1, dtype=np.int32)
    for L_i in range(num_layers):
        for h in range(num_kv_heads):
            picked: list[int] = []
            seen = set()
            for b in range(n_blocks):
                pool = list(range(b * block_size, (b + 1) * block_size))
                chosen = rng.choice(pool, size=n_per_block, replace=False)
                for c in chosen.tolist():
                    if c not in seen and len(picked) < n_top:
                        seen.add(int(c))
                        picked.append(int(c))
            # Pad with random remaining channels.
            remainder = [d for d in range(head_dim) if d not in seen]
            rng.shuffle(remainder)
            for c in remainder:
                if len(picked) >= n_top:
                    break
                picked.append(c)
                seen.add(c)
            out[L_i, h] = np.array(picked[:n_top], dtype=np.int32)
    return out


def _balanced_per_block_top_indices(scores_by_block: dict, n_per_block: int,
                                    n_top: int):
    """Exp K K8/K6/K9: take top-`n_per_block` from each modality block in
    scores_by_block (keys typically 'TT','TV','VT','VV'; values [L, H_kv, D]),
    dedup per (L, H_kv), pad to n_top from a composite score.

    Returns [L, H_kv, n_top] int32 (descending priority within picks).
    """
    import numpy as np
    keys = list(scores_by_block.keys())
    first = scores_by_block[keys[0]]
    L, Hkv, D = first.shape
    out = np.full((L, Hkv, n_top), -1, dtype=np.int32)
    composite = sum(scores_by_block[k] for k in keys)
    composite_sort = np.argsort(composite, axis=-1)[..., ::-1]
    for L_i in range(L):
        for h in range(Hkv):
            picked: list[int] = []
            seen = set()
            for k in keys:
                k_top = np.argsort(scores_by_block[k][L_i, h])[-n_per_block:][::-1]
                for c in k_top.tolist():
                    if c not in seen and len(picked) < n_top:
                        seen.add(int(c))
                        picked.append(int(c))
            for c in composite_sort[L_i, h].tolist():
                if len(picked) >= n_top:
                    break
                if c not in seen:
                    seen.add(int(c))
                    picked.append(int(c))
            out[L_i, h] = np.array(picked[:n_top], dtype=np.int32)
    return out


def _resolve_layer_adaptive_budget(calib: Optional[dict], cell_risk_key: str,
                                   top_fraction: float, n_per_cell: int = 16,
                                   ) -> Optional[object]:
    """Exp J: resolve a (cell_risk_key, top_fraction) budget spec into a
    [L, H_kv] int numpy array.

    Reads cfg.calib[cell_risk_key] (shape [L, H_kv], higher = riskier),
    sorts cells globally, and assigns budget=n_per_cell to the top
    `top_fraction × L × H_kv` cells (rounded). Other cells get 0.

    Returns None if calib is None or the key is missing — callers should
    guard.
    """
    import numpy as np
    if calib is None or cell_risk_key not in calib:
        return None
    risk = np.asarray(calib[cell_risk_key], dtype=np.float32)  # [L, H_kv]
    L, H_kv = risk.shape
    n_cells = L * H_kv
    n_keep = int(round(top_fraction * n_cells))
    if n_keep <= 0:
        return np.zeros((L, H_kv), dtype=np.int64)
    if n_keep >= n_cells:
        return np.full((L, H_kv), int(n_per_cell), dtype=np.int64)
    flat = risk.reshape(-1)
    # Indices of top-k by risk (descending).
    idx_keep = np.argpartition(-flat, n_keep - 1)[:n_keep]
    budget_flat = np.zeros(n_cells, dtype=np.int64)
    budget_flat[idx_keep] = int(n_per_cell)
    return budget_flat.reshape(L, H_kv)


def build_f_conditions(calib: Optional[dict] = None) -> list[KQuantizerConfig]:
    """Return the canonical 14-condition F-suite list.

    Conditions that need calibration data fetch it from the supplied `calib`
    dict (per the keys listed in the module docstring). Anchors and per-batch
    KIVI variants don't need calib.
    """
    return [
        # --- Anchors ---
        KQuantizerConfig(name="F0_BF16", kind="bf16", bits=16),
        KQuantizerConfig(name="F1_UniformInt4", kind="uniform_int4", bits=4),
        # F2: text-K BF16, visual-K INT4 — handled by V3K mode in driver
        # (this entry is a marker; driver special-cases it via slice_info + V3K)
        KQuantizerConfig(name="F2_TextBF16_VisInt4", kind="uniform_int4", bits=4),
        # F3: all-K BF16, V INT4 — driver sets cfg=bf16 for K; V is INT4 in cache
        KQuantizerConfig(name="F3_AllKBF16_VInt4", kind="bf16", bits=16),
        # --- Literature-aligned K repairs ---
        KQuantizerConfig(name="F4_KIVI_PerChannelSeq", kind="kivi_per_channel_seq", bits=4),
        KQuantizerConfig(name="F5_KIVI_TextVisualSplit", kind="kivi_text_visual_split", bits=4),
        KQuantizerConfig(name="F6_KIVI_RoleSplit", kind="kivi_role_split", bits=4),
        KQuantizerConfig(name="F7_KIVI_P99_5", kind="kivi_p99_5", bits=4, percentile=99.5),
        KQuantizerConfig(name="F8_KIVI_Outlier8", kind="kivi_outlier8", bits=4,
                         n_outliers=8, calib=calib),
        KQuantizerConfig(name="F9_KIVI_Outlier16", kind="kivi_outlier16", bits=4,
                         n_outliers=16, calib=calib),
        # --- VLM-specific score-space repairs ---
        KQuantizerConfig(name="F10_ScoreCal_Generic", kind="score_cal_generic", bits=4,
                         calib=calib),
        KQuantizerConfig(name="F11_ScoreCal_Block_TTHeavy", kind="score_cal_block_tt_heavy",
                         bits=4, calib=calib,
                         score_cal_weights={"w_TT": 4.0, "w_TV": 1.0, "w_VT": 1.0, "w_VV": 0.5}),
        KQuantizerConfig(name="F12_ScoreCal_Block_Balanced", kind="score_cal_block_balanced",
                         bits=4, calib=calib,
                         score_cal_weights={"w_TT": 1.0, "w_TV": 1.0, "w_VT": 1.0, "w_VV": 1.0}),
        KQuantizerConfig(name="F13_ScoreCal_TextOnly", kind="score_cal_text_only", bits=4,
                         calib=calib, text_only=True),
        # --- H-suite: temporal-windowed KIVI (genuine VLM novelty) ---
        # Stays at TRUE 4.00 KV bits; scale metadata is negligible vs cache.
        KQuantizerConfig(name="H3_KIVI_TempWin4", kind="kivi_temporal_window", bits=4,
                         n_temporal_windows=4, temporal_mode="visual_only"),
        KQuantizerConfig(name="H4_KIVI_TempWin8", kind="kivi_temporal_window", bits=4,
                         n_temporal_windows=8, temporal_mode="visual_only"),
        KQuantizerConfig(name="H5_KIVI_TokenBlock4", kind="kivi_temporal_window", bits=4,
                         n_temporal_windows=4, temporal_mode="token_block"),
        KQuantizerConfig(name="H6_KIVI_TempWin2", kind="kivi_temporal_window", bits=4,
                         n_temporal_windows=2, temporal_mode="visual_only"),
        # --- Exp I: TempWin + outlier-N composition (TempWin K + F8-style restoration) ---
        # I_TempWin2_Outlier8: H6 + 8 outlier channels at BF16 (avg KV bits 4.375).
        KQuantizerConfig(name="I_TempWin2_Outlier8", kind="kivi_temporal_window", bits=4,
                         n_temporal_windows=2, temporal_mode="visual_only",
                         n_outliers=8, calib=calib),
        # I_TempWin4_Outlier8: H3 + 8 outlier channels at BF16 (avg KV bits 4.375).
        KQuantizerConfig(name="I_TempWin4_Outlier8", kind="kivi_temporal_window", bits=4,
                         n_temporal_windows=4, temporal_mode="visual_only",
                         n_outliers=8, calib=calib),
        # I_TokenBlock6: 6 equal-token segments across the whole sequence,
        # modality-blind. Same number of segments as H3's text-prefix + 4
        # visual windows + text-suffix at 256f. Exact control for the visual-
        # time hypothesis at the 256f tier.
        KQuantizerConfig(name="I_TokenBlock6", kind="kivi_temporal_window", bits=4,
                         n_temporal_windows=6, temporal_mode="token_block"),
        # --- Exp I: VidKV-style V per-channel ---
        # I_TempWin2_VidKVV: H6 K-side + per-(B, H_kv, 1, D) V-side scale.
        # Bits unchanged (still INT4 for V); axis is what changes.
        KQuantizerConfig(name="I_TempWin2_VidKVV", kind="kivi_temporal_window", bits=4,
                         n_temporal_windows=2, temporal_mode="visual_only",
                         v_per_channel_seq=True),
        # I_TempWin4_VidKVV: H3 K-side + VidKV V-side.
        KQuantizerConfig(name="I_TempWin4_VidKVV", kind="kivi_temporal_window", bits=4,
                         n_temporal_windows=4, temporal_mode="visual_only",
                         v_per_channel_seq=True),
        # --- Exp J: cross-modal outlier-channel selection (top-8 BF16) ---
        # All J4-J8 use kind="kivi_outlier8" (n_outliers=8) with different
        # outlier_idx_key pointing at calib-precomputed cross-modal indices.
        KQuantizerConfig(name="J4_Outlier8_TT", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_TT_top16",
                         calib=calib),
        KQuantizerConfig(name="J5_Outlier8_TV", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_TV_top16",
                         calib=calib),
        KQuantizerConfig(name="J6_Outlier8_TT_TV", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_TT_TV_top16",
                         calib=calib),
        KQuantizerConfig(name="J7_Outlier8_BAL", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_BAL_top16",
                         calib=calib),
        KQuantizerConfig(name="J8_Outlier8_PIVOT", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_PIVOT_top16",
                         calib=calib),
        # --- Exp J: layer-adaptive outlier budget ---
        # Top X% of (L, H_kv) cells (by cell-risk score) get 16 BF16 outliers;
        # remaining cells get 0. The budget is resolved at build time from a
        # cell-risk array in calib.
        KQuantizerConfig(name="J9_LA_TT_TV_50pct", kind="kivi_outlier16", bits=4,
                         n_outliers=16, outlier_idx_key="outlier_idx_TT_TV_top16",
                         calib=calib,
                         layer_adaptive_outlier_budget=_resolve_layer_adaptive_budget(
                             calib, "cell_risk_TT_TV", 0.50, n_per_cell=16)),
        KQuantizerConfig(name="J10_LA_ALL_50pct", kind="kivi_outlier16", bits=4,
                         n_outliers=16, calib=calib,
                         layer_adaptive_outlier_budget=_resolve_layer_adaptive_budget(
                             calib, "cell_risk_all", 0.50, n_per_cell=16)),
        KQuantizerConfig(name="J11_LA_TT_TV_75pct", kind="kivi_outlier16", bits=4,
                         n_outliers=16, outlier_idx_key="outlier_idx_TT_TV_top16",
                         calib=calib,
                         layer_adaptive_outlier_budget=_resolve_layer_adaptive_budget(
                             calib, "cell_risk_TT_TV", 0.75, n_per_cell=16)),
        # --- Exp J: side-channel compression (INT-N storage of outliers) ---
        # Same selected channels as F9 but stored at INT8 / INT6 instead of BF16.
        KQuantizerConfig(name="J12_F9_INT8side", kind="kivi_outlier16", bits=4,
                         n_outliers=16, calib=calib, outlier_storage_bits=8),
        KQuantizerConfig(name="J13_F9_INT6side", kind="kivi_outlier16", bits=4,
                         n_outliers=16, calib=calib, outlier_storage_bits=6),
        KQuantizerConfig(name="J14_TT_TV_INT8side", kind="kivi_outlier16", bits=4,
                         n_outliers=16, outlier_idx_key="outlier_idx_TT_TV_top16",
                         calib=calib, outlier_storage_bits=8),
        # --- Exp J Stage-3 controls (added after Stage 1 results) ---
        # J15: random top-8 BF16 sidecode. Reads `outlier_idx_RANDOM_top16` from
        # calib (the driver injects this array with a fixed seed at startup).
        # Purpose: defend against "any 8-channel side-channel would work on an
        # easy split" — must show J7/J8/J9 beat random, not just generic.
        KQuantizerConfig(name="J15_Outlier8_RANDOM", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_RANDOM_top16",
                         calib=calib),
        # J16: random layer-adaptive 50% cells, generic top-16 within cells.
        # Reads `cell_risk_RANDOM` from calib (driver injects). Control for J9/J10.
        KQuantizerConfig(name="J16_LA_RANDOM_50pct", kind="kivi_outlier16", bits=4,
                         n_outliers=16, calib=calib,
                         layer_adaptive_outlier_budget=_resolve_layer_adaptive_budget(
                             calib, "cell_risk_RANDOM", 0.50, n_per_cell=16)),
        # J17: error-weighted Pivot top-8. Score(l,h,d) = q_pivot²·(k_d−Q4_d)²
        # ≈ q_pivot · k_max² (uniform quantization noise variance ∝ max²/588).
        # Reads `outlier_idx_PIVOT_ERR_top16` from calib (driver injects).
        # More literal KVQuant-style "model-visible quantization distortion".
        KQuantizerConfig(name="J17_Outlier8_PIVOT_ERR", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_PIVOT_ERR_top16",
                         calib=calib),
        # ============================================================
        # Exp K: balanced cross-modal sidecode replication
        # ============================================================
        # K2 / K3: F9 generic top-16 sidecode at BF16 vs INT8.
        # K2 same as J2_F9_128f (already in F-suite as F9). Re-emit for K-only
        # condition lists; identical config.
        KQuantizerConfig(name="K2_F9_BF16side", kind="kivi_outlier16", bits=4,
                         n_outliers=16, calib=calib, outlier_storage_bits=16),
        KQuantizerConfig(name="K3_F9_INT8side", kind="kivi_outlier16", bits=4,
                         n_outliers=16, calib=calib, outlier_storage_bits=8),
        # K4: F8 generic top-8 BF16 (same as F8/J3).
        KQuantizerConfig(name="K4_F8_BF16side", kind="kivi_outlier8", bits=4,
                         n_outliers=8, calib=calib, outlier_storage_bits=16),
        # K5: random top-8 BF16 (same as J15). Reads outlier_idx_RANDOM_top16.
        KQuantizerConfig(name="K5_Random8_BF16side", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_RANDOM_top16",
                         calib=calib, outlier_storage_bits=16),
        # K6: J7 balanced top-2/block BF16. Reads outlier_idx_BAL_top2_per_block_top16.
        KQuantizerConfig(name="K6_Bal2pb_BF16side", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_BAL_top2_per_block_top16",
                         calib=calib, outlier_storage_bits=16),
        # K7: J7 balanced top-2/block INT8. Same indices as K6; INT8 sidecode.
        KQuantizerConfig(name="K7_Bal2pb_INT8side", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_BAL_top2_per_block_top16",
                         calib=calib, outlier_storage_bits=8),
        # K8: balanced top-1/block BF16 (4 channels total).
        KQuantizerConfig(name="K8_Bal1pb_BF16side", kind="kivi_outlier8", bits=4,
                         n_outliers=4, outlier_idx_key="outlier_idx_BAL_top1_per_block_top16",
                         calib=calib, outlier_storage_bits=16),
        # K9: balanced top-3/block BF16 (12 channels total).
        KQuantizerConfig(name="K9_Bal3pb_BF16side", kind="kivi_outlier16", bits=4,
                         n_outliers=12, outlier_idx_key="outlier_idx_BAL_top3_per_block_top16",
                         calib=calib, outlier_storage_bits=16),
        # K10: balanced-random by channel-position blocks (decouples balance
        # structure from cross-modal scoring). Reads outlier_idx_BAL_RANDOM_POS_top16.
        KQuantizerConfig(name="K10_BalRandomPos_BF16side", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_BAL_RANDOM_POS_top16",
                         calib=calib, outlier_storage_bits=16),
        # K11: pivot top-8 BF16 (same as J8).
        KQuantizerConfig(name="K11_Pivot8_BF16side", kind="kivi_outlier8", bits=4,
                         n_outliers=8, outlier_idx_key="outlier_idx_PIVOT_top16",
                         calib=calib, outlier_storage_bits=16),
        # ============================================================
        # Exp M: matched-budget sidecode controls
        # Tests whether K9 (Balanced top-3/block, 12 BF16 outliers, 4.56 KV
        # bits) wins because of cross-modal balance, or just because it
        # protects more channels than F8/J7. All M5-M7 and M9-M12 are at
        # 12 channels (4.56 KV bits) for direct matched-budget paired McNemar.
        # ============================================================
        # M5: generic top-12 BF16 — uses default outlier_channel_idx_top16,
        # sliced to first 12 channels. Direct matched-budget control vs K9.
        KQuantizerConfig(name="M5_Generic12_BF16side", kind="kivi_outlier16", bits=4,
                         n_outliers=12, calib=calib, outlier_storage_bits=16),
        # M6: random top-12 BF16 — uses outlier_idx_RANDOM_top16 (driver-injected
        # with seed=42), sliced to first 12. Strong control for "any 12 channels
        # work at this budget".
        KQuantizerConfig(name="M6_Random12_BF16side", kind="kivi_outlier16", bits=4,
                         n_outliers=12, outlier_idx_key="outlier_idx_RANDOM_top16",
                         calib=calib, outlier_storage_bits=16),
        # M7: balanced-random by channel-position, 3/block. Reads driver-injected
        # outlier_idx_BAL_RANDOM_POS_3pb_top16. Tests "balance structure without
        # cross-modal scoring".
        KQuantizerConfig(name="M7_BalRandomPos3pb_BF16side", kind="kivi_outlier16", bits=4,
                         n_outliers=12,
                         outlier_idx_key="outlier_idx_BAL_RANDOM_POS_3pb_top16",
                         calib=calib, outlier_storage_bits=16),
        # M10: balanced TT/TV/VT/VV top-3 INT8 sidecode (cheaper K9).
        KQuantizerConfig(name="M10_Bal3pb_INT8side", kind="kivi_outlier16", bits=4,
                         n_outliers=12,
                         outlier_idx_key="outlier_idx_BAL_top3_per_block_top16",
                         calib=calib, outlier_storage_bits=8),
        # M12: pivot top-12 BF16 — matched-budget pivot control. Same
        # outlier_idx_PIVOT_top16 as K11/J8 but with 12 channels protected
        # instead of 8.
        KQuantizerConfig(name="M12_Pivot12_BF16side", kind="kivi_outlier16", bits=4,
                         n_outliers=12, outlier_idx_key="outlier_idx_PIVOT_top16",
                         calib=calib, outlier_storage_bits=16),
        # ============================================================
        # Exp R: static matched-budget baselines for the AllVisual claim.
        # Defends against the reviewer question "why not just use a uniform
        # sidecode at this bit count?" S8/SJ already exist as F8/J12.
        # ============================================================
        # S4: top-4 BF16 outlier channels everywhere.
        #     K-bits = (16·4 + 124·4)/128 = 4.375 → KV avg 4.1875
        KQuantizerConfig(name="S4_Outlier4_BF16side", kind="kivi_outlier8", bits=4,
                         n_outliers=4, calib=calib, outlier_storage_bits=16),
        # S12: top-12 BF16 outlier channels everywhere.
        #      K-bits = (16·12 + 116·4)/128 = 5.125 → KV avg 4.5625
        KQuantizerConfig(name="S12_Outlier12_BF16side", kind="kivi_outlier16", bits=4,
                         n_outliers=12, calib=calib, outlier_storage_bits=16),
        # ============================================================
        # Exp S: sidecode bit-ladder. Same protected-channel identities as
        # F9 (top-16 by k_channel_energy) but with INT-N sidecode instead
        # of BF16. Tests whether F9's BF16 sidecode is overkill.
        # ============================================================
        # SL_INT7: top-16 outliers at INT7 sidecode.
        #          K-bits = (7·16 + 4·112)/128 = 4.375 → KV avg 4.1875
        KQuantizerConfig(name="SL_Outlier16_INT7side", kind="kivi_outlier16", bits=4,
                         n_outliers=16, calib=calib, outlier_storage_bits=7),
        # SL_INT6: top-16 outliers at INT6 sidecode (alias of existing J13).
        #          K-bits = (6·16 + 4·112)/128 = 4.250 → KV avg 4.125
        KQuantizerConfig(name="SL_Outlier16_INT6side", kind="kivi_outlier16", bits=4,
                         n_outliers=16, calib=calib, outlier_storage_bits=6),
        # SL_INT5: top-16 outliers at INT5 sidecode (aggressive).
        #          K-bits = (5·16 + 4·112)/128 = 4.125 → KV avg 4.0625
        KQuantizerConfig(name="SL_Outlier16_INT5side", kind="kivi_outlier16", bits=4,
                         n_outliers=16, calib=calib, outlier_storage_bits=5),
        # SL_top24_INT6: top-24 outliers at INT6 sidecode.
        #                K-bits = (6·24 + 4·104)/128 = 4.375 → KV avg 4.1875
        # (same bit budget as SL_INT7, different "width × depth" tradeoff)
        KQuantizerConfig(name="SL_Outlier24_INT6side", kind="kivi_outlier16", bits=4,
                         n_outliers=24, calib=calib, outlier_storage_bits=6),
        # SL_top32_INT6: top-32 outliers at INT6 sidecode.
        #                K-bits = (6·32 + 4·96)/128 = 4.500 → KV avg 4.250
        # (same bit budget as SJ = J12 INT8 top-16; tests "wider lower-precision")
        KQuantizerConfig(name="SL_Outlier32_INT6side", kind="kivi_outlier16", bits=4,
                         n_outliers=32, calib=calib, outlier_storage_bits=6),
        # ============================================================
        # T-mini: VLM page-aware K formats (PageLocal + PageSentinel).
        # All variants stay at TRUE ≈4.00 KV bits — scale metadata and
        # tiny sentinel restoration overhead are negligible vs cache.
        # ============================================================
        # T5: TextVisualLocal-F4 — already covered by F5_KIVI_TextVisualSplit.
        # T6: TokenBlockLocal-F4 — token-block control with 16 equal segments.
        KQuantizerConfig(name="T6_TokenBlock16_F4", kind="kivi_temporal_window",
                         bits=4, n_temporal_windows=16, temporal_mode="token_block"),
        # T7: RandomPageLocal-F4 — same per-item page count as PageLocal,
        # randomly chosen boundary positions.
        KQuantizerConfig(name="T7_RandomPageLocal_F4", kind="kivi_random_page_local",
                         bits=4, random_seed_namespace="T7_random_page"),
        # T8: PageLocal-F4 — main hypothesis. Per-page per-channel K scales.
        KQuantizerConfig(name="T8_PageLocal_F4", kind="kivi_page_local", bits=4),
        # T9: ImageOnlyLocal-F4 — page-local scales only on visual pages.
        KQuantizerConfig(name="T9_ImageOnlyLocal_F4", kind="kivi_image_only_local", bits=4),
        # T10: TextOnlyLocal-F4 — page-local scales only on text pages.
        KQuantizerConfig(name="T10_TextOnlyLocal_F4", kind="kivi_text_only_local", bits=4),
        # T11: PageSentinel-1 — first visual token of each image page kept at BF16.
        KQuantizerConfig(name="T11_PageSentinel1_F4base",
                         kind="kivi_page_sentinel", bits=4,
                         base_kind="kivi_per_channel_seq",
                         sentinel_kind="first_visual", sentinel_n_per_page=1),
        # T12: PageSentinel-4 — first 4 visual tokens of each image page at BF16.
        KQuantizerConfig(name="T12_PageSentinel4_F4base",
                         kind="kivi_page_sentinel", bits=4,
                         base_kind="kivi_per_channel_seq",
                         sentinel_kind="first_visual", sentinel_n_per_page=4),
        # T13: RandomSentinel-4 — same number of protected visual tokens, random
        # positions within each image page (seeded per-item).
        KQuantizerConfig(name="T13_RandomSentinel4_F4base",
                         kind="kivi_page_sentinel", bits=4,
                         base_kind="kivi_per_channel_seq",
                         sentinel_kind="random_visual", sentinel_n_per_page=4,
                         random_seed_namespace="T13_random_sentinel"),
        # T14: LastSentinel-4 — last 4 visual tokens of each image page at BF16.
        KQuantizerConfig(name="T14_LastSentinel4_F4base",
                         kind="kivi_page_sentinel", bits=4,
                         base_kind="kivi_per_channel_seq",
                         sentinel_kind="last_visual", sentinel_n_per_page=4),
        # T15: TextSentinel-4 — first 4 tokens of each text page/chunk at BF16.
        KQuantizerConfig(name="T15_TextSentinel4_F4base",
                         kind="kivi_page_sentinel", bits=4,
                         base_kind="kivi_per_channel_seq",
                         sentinel_kind="first_text", sentinel_n_per_page=4),
        # T16: PageLocal-F4 + PageSentinel-4 (combined).
        KQuantizerConfig(name="T16_PageLocal_PageSentinel4",
                         kind="kivi_page_sentinel", bits=4,
                         base_kind="kivi_page_local",
                         sentinel_kind="first_visual", sentinel_n_per_page=4),
    ]
