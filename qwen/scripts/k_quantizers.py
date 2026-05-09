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
# Outlier-channel helpers (F8 / F9)
# ===================================================================


def _outlier_channel_indices(cfg: KQuantizerConfig, layer_idx: int,
                             num_kv_heads: int, head_dim: int,
                             n: int) -> torch.Tensor:
    """Return [H_kv, n] long tensor of channel indices to protect at BF16.

    Source: cfg.calib['outlier_channel_idx_top16'] (precomputed at calib time
    with n=16, then we slice the first `n`). If absent, ranks online from
    cfg.calib['k_channel_energy'][L, H_kv, D].
    """
    if cfg.calib is None:
        raise RuntimeError(
            f"{cfg.name}: outlier kind requires cfg.calib with 'outlier_channel_idx_top16' "
            f"or 'k_channel_energy'."
        )
    if "outlier_channel_idx_top16" in cfg.calib:
        idx = cfg.calib["outlier_channel_idx_top16"]  # numpy [L, H_kv, 16]
        return torch.as_tensor(idx[layer_idx, :, :n], dtype=torch.long)
    energy = cfg.calib.get("k_channel_energy")
    if energy is None:
        raise RuntimeError(f"{cfg.name}: cfg.calib missing 'k_channel_energy'.")
    e = torch.as_tensor(energy[layer_idx])  # [H_kv, D]
    return torch.argsort(e, dim=-1, descending=True)[:, :n]


def _kivi_outlier(K: torch.Tensor, cfg: KQuantizerConfig,
                  layer_idx: int, qmax: float) -> torch.Tensor:
    """F8/F9: top-N outlier channels at BF16; rest at INT4 per-channel-seq.

    Build a [H_kv, D] bool mask of "protected channels", quantize all channels
    with KIVI-style per-channel-seq scale, then restore the protected channel
    values from the original K.
    """
    B, H, T, D = K.shape
    n = int(cfg.n_outliers or 0)
    if n <= 0:
        # Degenerate: no outliers. Falls back to F4.
        scale = _per_channel_seq_scale(K, qmax, percentile=None)
        return _quantize_with_scale(K, scale, qmax)

    outlier_idx = _outlier_channel_indices(cfg, layer_idx, H, D, n).to(K.device)  # [H, n]

    # Quantize everything with KIVI per-channel-seq scale.
    scale = _per_channel_seq_scale(K, qmax, percentile=None)  # [B, H, 1, D]
    K_q = _quantize_with_scale(K, scale, qmax)

    # Restore outlier channels from original K. Build a [B, H, T, D] index expansion.
    # Use scatter along the last dim for selected outlier channels.
    # For each head h, channel index list outlier_idx[h] is what we restore.
    for h in range(H):
        ch = outlier_idx[h]  # [n]
        if ch.numel() == 0:
            continue
        K_q[:, h, :, ch] = K[:, h, :, ch]
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
    ]
