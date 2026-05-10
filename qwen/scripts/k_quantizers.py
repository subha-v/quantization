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

Slice info (for F5/F6/F13) is passed at runtime via `slice_info` argument:
  {
    "v_start": int, "v_end": int,        # visual span (absolute positions)
    "text_positions": list[int],          # absolute positions of non-visual tokens
    "visual_positions": list[int],        # absolute positions of visual tokens
    "role_spans": {role: (a, b)} for     # role boundaries (absolute positions)
                  role in {header, question, options, instruction,
                           answer_prefix, visual}
    "seq_len": int,
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
        if idx.shape[-1] < n:
            raise RuntimeError(
                f"{cfg.name}: cfg.calib[{key!r}].shape[-1]={idx.shape[-1]} < n={n}; "
                f"recalibrate with at least n protected channels per (L, H_kv)."
            )
        return torch.as_tensor(idx[layer_idx, :, :n], dtype=torch.long)
    if cfg.outlier_idx_key is not None:
        # User explicitly asked for a key that's missing — fail loudly rather
        # than silently falling back to the generic top-16.
        raise RuntimeError(
            f"{cfg.name}: cfg.outlier_idx_key={cfg.outlier_idx_key!r} not present in "
            f"cfg.calib (keys: {sorted(cfg.calib)})."
        )
    energy = cfg.calib.get("k_channel_energy")
    if energy is None:
        raise RuntimeError(f"{cfg.name}: cfg.calib missing 'k_channel_energy'.")
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
    ]
