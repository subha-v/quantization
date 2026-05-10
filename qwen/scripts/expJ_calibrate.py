"""Exp J calibration pass — cross-modal outlier-channel scoring.

Extends `expF_calibrate.py` with:
  - per-modality K stats (k_channel_energy_text, k_channel_energy_visual)
  - pivot-Q stats (q_energy_pivot at the prompt's last position — the actual
    answer-query that produces the first-token MCQ logits)
  - precomputed cross-modal outlier-channel index arrays (TT, TV, VT, VV,
    BAL, TT+TV, PIVOT — each top-16 per (L, H_kv))
  - cell-risk arrays for layer-adaptive selection (cell_risk_TT_TV,
    cell_risk_all)

Output NPZ is a SUPERSET of expF_kcalib's keys, so it can drive F8/F9 too
(via outlier_channel_idx_top16 — kept for backward compat).

Runs at frames=128 by default (Exp J is 128f-only). Writes
`qwen/calibration/expJ_kcalib_<model>_frames128.npz` + matching JSON meta.

Usage:
  python expJ_calibrate.py --model Qwen/Qwen2.5-VL-7B-Instruct --frames 128
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from data_longvideobench import (
    DEFAULT_SPLIT_FILE,
    filter_items,
    format_mcq_messages,
    load_all_items,
    load_split,
)
from text_slices import find_text_slice_spans


CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"


# ===================================================================
# K-stats cache: per-modality split
# ===================================================================


class KStatsCacheXModal(DynamicCache):
    """DynamicCache that records K stats split by modality (text vs visual).

    Bind per-item modality info via `set_modality_context(v_start, v_end,
    cache_offset_at_start)` BEFORE generate(). The cache uses cache_offset_at_start
    + the size of each new chunk to figure out which positions are text vs
    visual.

    Accumulators (float32):
      k_sumsq_total[L, H_kv, D]   sum_t K_d² over all positions
      k_sumsq_text[L, H_kv, D]    sum_t K_d² over positions outside [v_start, v_end)
      k_sumsq_vis[L, H_kv, D]     sum_t K_d² over positions inside [v_start, v_end)
      k_count_total[L]            number of positions counted
      k_count_text[L]             text positions counted
      k_count_visual[L]           visual positions counted
      k_max[L, H_kv, D]           running max(|K|)
    """

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # Total stats (back-compat with KStatsCache).
        self.k_sumsq = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
        self.k_count = torch.zeros(num_layers, dtype=torch.int64)
        self.k_max = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
        # Per-modality split (Exp J).
        self.k_sumsq_text = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
        self.k_count_text = torch.zeros(num_layers, dtype=torch.int64)
        self.k_sumsq_vis = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
        self.k_count_vis = torch.zeros(num_layers, dtype=torch.int64)
        # Per-item context (set via set_modality_context before generate).
        self._v_start = -1
        self._v_end = -1

    def set_modality_context(self, v_start: int, v_end: int) -> None:
        self._v_start = int(v_start)
        self._v_end = int(v_end)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        with torch.no_grad():
            B, H, T, D = key_states.shape
            kf = key_states.float()
            sumsq_full = (kf * kf).sum(dim=(0, 2)).cpu()  # [H, D]
            mx = kf.abs().amax(dim=(0, 2)).cpu()
            # Total accumulators.
            self.k_sumsq[layer_idx] += sumsq_full
            self.k_count[layer_idx] += int(B * T)
            self.k_max[layer_idx] = torch.maximum(self.k_max[layer_idx], mx)
            # Per-modality split (only on prefill chunk: cache_offset==0 means
            # this chunk covers absolute positions [0, T)).
            cache_offset = self.get_seq_length(layer_idx)
            if (self._v_start >= 0 and self._v_end > self._v_start and
                    cache_offset == 0):
                # Build text/visual masks over [0, T).
                text_pos = []
                vis_pos = []
                for t in range(T):
                    abs_pos = cache_offset + t
                    if self._v_start <= abs_pos < self._v_end:
                        vis_pos.append(t)
                    else:
                        text_pos.append(t)
                if text_pos:
                    text_idx = torch.tensor(text_pos, dtype=torch.long, device=kf.device)
                    kf_text = kf.index_select(dim=2, index=text_idx)
                    sumsq_text = (kf_text * kf_text).sum(dim=(0, 2)).cpu()
                    self.k_sumsq_text[layer_idx] += sumsq_text
                    self.k_count_text[layer_idx] += int(B * len(text_pos))
                if vis_pos:
                    vis_idx = torch.tensor(vis_pos, dtype=torch.long, device=kf.device)
                    kf_vis = kf.index_select(dim=2, index=vis_idx)
                    sumsq_vis = (kf_vis * kf_vis).sum(dim=(0, 2)).cpu()
                    self.k_sumsq_vis[layer_idx] += sumsq_vis
                    self.k_count_vis[layer_idx] += int(B * len(vis_pos))
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


# ===================================================================
# Q-stats hook: per-modality + pivot
# ===================================================================


class QStatsHookXModal:
    """Q-side hook that captures total / text / visual / pivot stats.

    Pivot Q = Q at the LAST input position (seq_len - 1), the position whose
    next-token logits give the MCQ answer letter. Bind via
    `set_pivot_position(pos)` before generate().
    """

    def __init__(self, model: torch.nn.Module, num_layers: int,
                 num_kv_heads: int, num_q_heads: int, head_dim: int,
                 max_q_per_item: int = 256):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.group = num_q_heads // num_kv_heads
        self.max_q_per_item = max_q_per_item

        zeros_k = lambda: torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
        zeros_l = lambda: torch.zeros(num_layers, dtype=torch.int64)
        self.q_sumsq = zeros_k()
        self.q_count = zeros_l()
        self.q_sumsq_text = zeros_k()
        self.q_count_text = zeros_l()
        self.q_sumsq_vis = zeros_k()
        self.q_count_vis = zeros_l()
        self.q_sumsq_pivot = zeros_k()
        self.q_count_pivot = zeros_l()

        self._v_start = -1
        self._v_end = -1
        self._pivot = -1

        self._handles = []
        layers = self._find_decoder_layers(model)
        if len(layers) != num_layers:
            print(f"[J-Qhook][warn] decoder_layers found={len(layers)} expected={num_layers}")
        for layer_idx, attn in layers:
            q_proj = getattr(attn, "q_proj", None)
            if q_proj is None:
                print(f"[J-Qhook][warn] layer {layer_idx} has no q_proj; skip")
                continue
            handle = q_proj.register_forward_hook(self._make_hook(layer_idx))
            self._handles.append(handle)

    @staticmethod
    def _find_decoder_layers(model):
        candidates = [
            getattr(getattr(model, "language_model", None), "layers", None),
            getattr(getattr(getattr(model, "model", None), "language_model", None), "layers", None),
            getattr(getattr(model, "model", None), "layers", None),
        ]
        for layers in candidates:
            if layers is not None:
                return [(i, layer.self_attn) for i, layer in enumerate(layers)]
        raise RuntimeError("Could not locate decoder layers")

    def set_item_context(self, v_start: int, v_end: int, pivot: int) -> None:
        self._v_start = int(v_start)
        self._v_end = int(v_end)
        self._pivot = int(pivot)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            with torch.no_grad():
                q = output
                if q.dim() != 3:
                    return
                B, T, total = q.shape
                if total != self.num_q_heads * self.head_dim:
                    return
                qf = q.detach().float().view(B, T, self.num_q_heads, self.head_dim)
                qf = qf.view(B, T, self.num_kv_heads, self.group, self.head_dim)
                # Pivot capture FIRST (before subsampling, so we don't miss the
                # exact last-position Q).
                if 0 <= self._pivot < T:
                    qsq_pivot = (qf[:, self._pivot:self._pivot + 1, :, :, :] ** 2).sum(dim=3)
                    self.q_sumsq_pivot[layer_idx] += qsq_pivot.sum(dim=(0, 1)).cpu()
                    self.q_count_pivot[layer_idx] += int(B * 1 * self.group)
                # Subsample T to cap memory for the standard text/visual binning.
                t_take = min(T, self.max_q_per_item)
                if T > t_take:
                    idx = torch.linspace(0, T - 1, steps=t_take, dtype=torch.long, device=q.device)
                    qf_sub = qf.index_select(dim=1, index=idx)
                    pos_abs = idx.cpu().tolist()
                else:
                    qf_sub = qf
                    pos_abs = list(range(T))
                qsq = (qf_sub * qf_sub).sum(dim=3)  # [B, t_take, H_kv, D]
                # Total
                self.q_sumsq[layer_idx] += qsq.sum(dim=(0, 1)).cpu()
                self.q_count[layer_idx] += int(B * len(pos_abs) * self.group)
                # Text vs visual
                if self._v_start >= 0 and self._v_end > self._v_start:
                    text_idx = []
                    vis_idx = []
                    for j, p in enumerate(pos_abs):
                        if self._v_start <= p < self._v_end:
                            vis_idx.append(j)
                        else:
                            text_idx.append(j)
                    if text_idx:
                        t_t = torch.tensor(text_idx, dtype=torch.long, device=q.device)
                        self.q_sumsq_text[layer_idx] += qsq.index_select(dim=1, index=t_t).sum(dim=(0, 1)).cpu()
                        self.q_count_text[layer_idx] += int(B * len(text_idx) * self.group)
                    if vis_idx:
                        v_t = torch.tensor(vis_idx, dtype=torch.long, device=q.device)
                        self.q_sumsq_vis[layer_idx] += qsq.index_select(dim=1, index=v_t).sum(dim=(0, 1)).cpu()
                        self.q_count_vis[layer_idx] += int(B * len(vis_idx) * self.group)
        return hook

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def finalize(self) -> dict[str, np.ndarray]:
        def _mean_per_layer(sumsq: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(sumsq)
            for L in range(sumsq.shape[0]):
                c = float(counts[L].item())
                if c > 0:
                    out[L] = sumsq[L] / c
            return out
        return {
            "q_energy":         _mean_per_layer(self.q_sumsq,       self.q_count).numpy().astype(np.float32),
            "q_energy_text":    _mean_per_layer(self.q_sumsq_text,  self.q_count_text).numpy().astype(np.float32),
            "q_energy_visual":  _mean_per_layer(self.q_sumsq_vis,   self.q_count_vis).numpy().astype(np.float32),
            "q_energy_pivot":   _mean_per_layer(self.q_sumsq_pivot, self.q_count_pivot).numpy().astype(np.float32),
            "q_count_total":    self.q_count.numpy().astype(np.int64),
            "q_count_text":     self.q_count_text.numpy().astype(np.int64),
            "q_count_visual":   self.q_count_vis.numpy().astype(np.int64),
            "q_count_pivot":    self.q_count_pivot.numpy().astype(np.int64),
        }


# ===================================================================
# Cross-modal outlier-index helpers
# ===================================================================


def _topn_indices(score: np.ndarray, n: int) -> np.ndarray:
    """Top-n channel indices per (L, H_kv) by score descending.

    score: [L, H_kv, D] float; returns [L, H_kv, n] int32 (descending order).
    """
    # argsort ascending, take last n, reverse to descending. Use np.argsort
    # for full ordering rather than argpartition to keep the descending order.
    L, Hkv, D = score.shape
    sorted_idx = np.argsort(score, axis=-1)[..., -n:][..., ::-1].copy()
    return sorted_idx.astype(np.int32)


def _balanced_top_indices(scores: dict[str, np.ndarray], n_total: int = 16,
                          per_block: int = 4) -> np.ndarray:
    """Take top-`per_block` from each of the 4 score arrays (TT/TV/VT/VV),
    deduplicate per (L, H_kv) and pad with the next channels by (TT+TV+VT+VV)
    sum until we have exactly `n_total`.

    scores: dict with keys 'TT', 'TV', 'VT', 'VV' each shape [L, H_kv, D].
    """
    keys = ("TT", "TV", "VT", "VV")
    L, Hkv, D = scores[keys[0]].shape
    out = np.full((L, Hkv, n_total), -1, dtype=np.int32)
    # Composite score for tie-breaking and padding.
    composite = sum(scores[k] for k in keys)
    composite_sort = np.argsort(composite, axis=-1)[..., ::-1]  # descending
    for L_i in range(L):
        for h in range(Hkv):
            picked: list[int] = []
            seen = set()
            # Take per-block top-K from each block.
            for k in keys:
                k_top = np.argsort(scores[k][L_i, h])[-per_block:][::-1]
                for c in k_top.tolist():
                    if c not in seen:
                        seen.add(c)
                        picked.append(c)
            # Pad with composite top until we have n_total.
            for c in composite_sort[L_i, h].tolist():
                if len(picked) >= n_total:
                    break
                if c not in seen:
                    seen.add(c)
                    picked.append(c)
            out[L_i, h] = np.array(picked[:n_total], dtype=np.int32)
    return out


def _build_xmodal_outlier_arrays(stats: dict[str, np.ndarray],
                                 n_top: int = 16) -> dict[str, np.ndarray]:
    """Build the 7 cross-modal outlier-index arrays + 2 cell-risk arrays.

    stats must contain:
      q_energy_text, q_energy_visual, q_energy_pivot     [L, H_kv, D]
      k_channel_energy, k_channel_energy_text,
      k_channel_energy_visual                            [L, H_kv, D]
    """
    qet = stats["q_energy_text"]
    qev = stats["q_energy_visual"]
    qep = stats["q_energy_pivot"]
    ke_full = stats["k_channel_energy"]
    ke_t = stats["k_channel_energy_text"]
    ke_v = stats["k_channel_energy_visual"]

    # Distortion scores D_B(l, h, d) = q_energy_a · k_energy_b
    D_TT = qet * ke_t
    D_TV = qet * ke_v
    D_VT = qev * ke_t
    D_VV = qev * ke_v
    D_TT_TV = D_TT + D_TV
    D_PIVOT = qep * ke_full  # pivot uses full K (single position has limited modality info)

    out = {
        "outlier_idx_TT_top16":      _topn_indices(D_TT, n_top),
        "outlier_idx_TV_top16":      _topn_indices(D_TV, n_top),
        "outlier_idx_VT_top16":      _topn_indices(D_VT, n_top),
        "outlier_idx_VV_top16":      _topn_indices(D_VV, n_top),
        "outlier_idx_TT_TV_top16":   _topn_indices(D_TT_TV, n_top),
        "outlier_idx_PIVOT_top16":   _topn_indices(D_PIVOT, n_top),
        "outlier_idx_BAL_top16":     _balanced_top_indices(
            {"TT": D_TT, "TV": D_TV, "VT": D_VT, "VV": D_VV},
            n_total=n_top, per_block=n_top // 4),
        # Cell-risk arrays for layer-adaptive selection: sum over d.
        "cell_risk_TT_TV":           D_TT_TV.sum(axis=-1).astype(np.float32),
        "cell_risk_all":             (D_TT + D_TV + D_VT + D_VV).sum(axis=-1).astype(np.float32),
    }
    return out


# ===================================================================
# Driver
# ===================================================================


@torch.no_grad()
def run_calibration(model_id: str, frames: int, n_outliers_top: int,
                    split_file: Path, out_json: Path, out_npz: Path,
                    max_q_per_item: int = 256, progress_every: int = 5,
                    limit: int = 0) -> None:
    from run_inference import load_model
    from qwen_vl_utils import process_vision_info  # type: ignore

    items = load_all_items()
    split = load_split(split_file)
    cal_items = filter_items(items, split["cal"])
    if limit:
        cal_items = cal_items[: limit]
    print(f"[J-calib] cal_items={len(cal_items)} model={model_id} frames={frames}", flush=True)

    model, processor = load_model(model_id, awq=False, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", 4))
    num_q_heads = int(getattr(model.config, "num_attention_heads", 28))
    q_proj_weight = layers[0].self_attn.q_proj.weight
    head_dim = int(q_proj_weight.shape[0] // num_q_heads)
    print(f"[J-calib] model={model_id} num_layers={num_layers} num_kv_heads={num_kv_heads} "
          f"num_q_heads={num_q_heads} head_dim={head_dim}", flush=True)

    progress_log = out_json.with_name(out_json.stem + ".progress.log")
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    def _log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[J-calib] {msg}", flush=True)
        with open(progress_log, "a") as f:
            f.write(f"[{ts}] {msg}\n"); f.flush()

    _log(f"START n_cal={len(cal_items)} max_q_per_item={max_q_per_item}")
    t_start = time.perf_counter()

    q_hook = QStatsHookXModal(model, num_layers=num_layers, num_kv_heads=num_kv_heads,
                              num_q_heads=num_q_heads, head_dim=head_dim,
                              max_q_per_item=max_q_per_item)

    # K-stats accumulators (totals across all items).
    zeros_k = lambda: torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
    zeros_l = lambda: torch.zeros(num_layers, dtype=torch.int64)
    k_sumsq_total = zeros_k()
    k_count_total = zeros_l()
    k_max_total = zeros_k()
    k_sumsq_text_total = zeros_k()
    k_count_text_total = zeros_l()
    k_sumsq_vis_total = zeros_k()
    k_count_vis_total = zeros_l()

    n_done, n_failed = 0, 0
    try:
        for i, it in enumerate(cal_items):
            try:
                msgs = format_mcq_messages(it, n_frames=frames)
                prompt_text = processor.apply_chat_template(msgs, tokenize=False,
                                                            add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(msgs)
                inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                                   padding=True, return_tensors="pt").to(model.device)
                seq_len = int(inputs["input_ids"].shape[1])
                slices = find_text_slice_spans(inputs["input_ids"], processor, it)
                v_start = int(slices.get("_v_start", -1))
                v_end = int(slices.get("_v_end", -1))
                pivot = seq_len - 1  # last input position = answer-query

                q_hook.set_item_context(v_start, v_end, pivot)

                cache = KStatsCacheXModal(num_layers=num_layers,
                                          num_kv_heads=num_kv_heads,
                                          head_dim=head_dim)
                cache.set_modality_context(v_start, v_end)
                _ = model.generate(**inputs, past_key_values=cache,
                                   max_new_tokens=1, do_sample=False,
                                   return_dict_in_generate=True, output_scores=True,
                                   use_cache=True)
                # Aggregate.
                k_sumsq_total += cache.k_sumsq
                k_count_total += cache.k_count
                k_max_total = torch.maximum(k_max_total, cache.k_max)
                k_sumsq_text_total += cache.k_sumsq_text
                k_count_text_total += cache.k_count_text
                k_sumsq_vis_total += cache.k_sumsq_vis
                k_count_vis_total += cache.k_count_vis

                n_done += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (i + 1) % progress_every == 0 or (i + 1) == len(cal_items):
                    elapsed = time.perf_counter() - t_start
                    rate = elapsed / max(1, n_done)
                    eta = max(0.0, rate * (len(cal_items) - (i + 1)))
                    _log(f"{i+1}/{len(cal_items)} ok={n_done} fail={n_failed} "
                         f"elapsed={timedelta(seconds=int(elapsed))} "
                         f"ETA={timedelta(seconds=int(eta))}")
            except Exception as e:
                n_failed += 1
                _log(f"WARN item={it.id} skipped: {type(e).__name__}: {e}")
                continue
    finally:
        q_hook.uninstall()

    # Finalize K stats: per-channel energies (mean of K²).
    def _mean_per_layer(sumsq: torch.Tensor, counts: torch.Tensor) -> np.ndarray:
        out = torch.zeros_like(sumsq)
        for L in range(sumsq.shape[0]):
            c = float(counts[L].item())
            if c > 0:
                out[L] = sumsq[L] / c
        return out.numpy().astype(np.float32)

    k_channel_energy = _mean_per_layer(k_sumsq_total, k_count_total)
    k_channel_energy_text = _mean_per_layer(k_sumsq_text_total, k_count_text_total)
    k_channel_energy_visual = _mean_per_layer(k_sumsq_vis_total, k_count_vis_total)
    k_max_np = k_max_total.numpy().astype(np.float32)

    # Generic outlier indices (top-N by k_channel_energy) — keep for back-compat.
    n_top = int(n_outliers_top)
    outlier_idx_generic = np.argsort(k_channel_energy, axis=-1)[..., -n_top:][..., ::-1].copy().astype(np.int32)

    # Q stats.
    q_data = q_hook.finalize()
    for k in ("q_energy", "q_energy_text", "q_energy_visual", "q_energy_pivot"):
        q_data[k] = np.clip(q_data[k], 1e-12, None).astype(np.float32)

    # Cross-modal outlier indices and cell-risk arrays.
    xmodal = _build_xmodal_outlier_arrays(
        {
            "q_energy_text":          q_data["q_energy_text"],
            "q_energy_visual":        q_data["q_energy_visual"],
            "q_energy_pivot":         q_data["q_energy_pivot"],
            "k_channel_energy":       k_channel_energy,
            "k_channel_energy_text":  np.clip(k_channel_energy_text, 1e-12, None),
            "k_channel_energy_visual": np.clip(k_channel_energy_visual, 1e-12, None),
        },
        n_top=n_top,
    )

    # Sanity check: count consistency. Per-layer text + visual should equal total.
    counts_total = k_count_total.numpy()
    counts_text = k_count_text_total.numpy()
    counts_vis = k_count_vis_total.numpy()
    diff = counts_total - (counts_text + counts_vis)
    if np.any(diff != 0):
        print(f"[J-calib] WARN: K count mismatch per-layer (total - text - vis): "
              f"max_diff={diff.max()}, min_diff={diff.min()}", flush=True)
    q_diff = q_data["q_count_total"] - (q_data["q_count_text"] + q_data["q_count_visual"])
    if np.any(q_diff != 0):
        print(f"[J-calib] WARN: Q count mismatch per-layer (total - text - vis): "
              f"max_diff={q_diff.max()}, min_diff={q_diff.min()}", flush=True)

    # Sanity: TT_top16 vs generic top16 overlap (for diagnostics).
    overlap_TT = []
    overlap_TT_TV = []
    for L_i in range(k_channel_energy.shape[0]):
        for h in range(k_channel_energy.shape[1]):
            gen = set(outlier_idx_generic[L_i, h].tolist())
            tt = set(xmodal["outlier_idx_TT_top16"][L_i, h].tolist())
            tt_tv = set(xmodal["outlier_idx_TT_TV_top16"][L_i, h].tolist())
            overlap_TT.append(len(gen & tt))
            overlap_TT_TV.append(len(gen & tt_tv))
    print(f"[J-calib] outlier-index overlap with generic-top16: "
          f"TT mean={np.mean(overlap_TT):.1f}/16, TT_TV mean={np.mean(overlap_TT_TV):.1f}/16",
          flush=True)

    # Meta + write.
    meta = {
        "model": model_id,
        "frames": frames,
        "n_cal_items": len(cal_items),
        "n_done": n_done,
        "n_failed": n_failed,
        "num_layers": int(k_channel_energy.shape[0]),
        "num_kv_heads": int(k_channel_energy.shape[1]),
        "head_dim": int(k_channel_energy.shape[2]),
        "max_q_per_item": max_q_per_item,
        "k_outlier_n_top": n_top,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": int(time.perf_counter() - t_start),
        "outlier_overlap_TT_vs_generic_mean": float(np.mean(overlap_TT)),
        "outlier_overlap_TT_TV_vs_generic_mean": float(np.mean(overlap_TT_TV)),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(meta, indent=2))
    np.savez_compressed(
        out_npz,
        # F-suite back-compat.
        k_channel_energy=k_channel_energy,
        k_abs_max=k_max_np,
        outlier_channel_idx_top16=outlier_idx_generic,
        q_energy=q_data["q_energy"],
        q_energy_text=q_data["q_energy_text"],
        q_energy_visual=q_data["q_energy_visual"],
        q_count_total=q_data["q_count_total"],
        q_count_text=q_data["q_count_text"],
        q_count_visual=q_data["q_count_visual"],
        # Exp J additions.
        k_channel_energy_text=k_channel_energy_text,
        k_channel_energy_visual=k_channel_energy_visual,
        q_energy_pivot=q_data["q_energy_pivot"],
        q_count_pivot=q_data["q_count_pivot"],
        outlier_idx_TT_top16=xmodal["outlier_idx_TT_top16"],
        outlier_idx_TV_top16=xmodal["outlier_idx_TV_top16"],
        outlier_idx_VT_top16=xmodal["outlier_idx_VT_top16"],
        outlier_idx_VV_top16=xmodal["outlier_idx_VV_top16"],
        outlier_idx_TT_TV_top16=xmodal["outlier_idx_TT_TV_top16"],
        outlier_idx_PIVOT_top16=xmodal["outlier_idx_PIVOT_top16"],
        outlier_idx_BAL_top16=xmodal["outlier_idx_BAL_top16"],
        cell_risk_TT_TV=xmodal["cell_risk_TT_TV"],
        cell_risk_all=xmodal["cell_risk_all"],
    )
    _log(f"DONE wrote meta -> {out_json} arrays -> {out_npz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--max_q_per_item", type=int, default=256)
    ap.add_argument("--n_outliers_top", type=int, default=16)
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_npz", type=Path, default=None)
    ap.add_argument("--out_json", type=Path, default=None)
    args = ap.parse_args()

    model_short = args.model.split("/")[-1]
    if args.out_npz is None:
        args.out_npz = CALIBRATION_DIR / f"expJ_kcalib_{model_short}_frames{args.frames}.npz"
    if args.out_json is None:
        args.out_json = args.out_npz.with_suffix(".json")

    run_calibration(model_id=args.model, frames=args.frames,
                    n_outliers_top=args.n_outliers_top,
                    split_file=args.split_file,
                    out_json=args.out_json, out_npz=args.out_npz,
                    max_q_per_item=args.max_q_per_item,
                    progress_every=args.progress_every,
                    limit=args.limit)


if __name__ == "__main__":
    main()
