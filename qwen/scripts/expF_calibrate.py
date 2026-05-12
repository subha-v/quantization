"""Exp F calibration pass.

Captures the per-(layer, kv-head, channel) statistics needed by the F-suite
score-calibrated and outlier-channel quantizers. One forward pass per cal
item, BF16 weights + BF16 KV cache, eager attention not required (only
`q_proj` outputs are needed; we hook them via forward hooks).

Captured:
  k_channel_energy[L, H_kv, D]   = sum over (item, t) of K_d^2 (then meaned)
  k_abs_max[L, H_kv, D]          = max over (item, t) of |K_d|
  outlier_channel_idx_top16[L, H_kv, 16]  = top-16 channels by k_channel_energy
  q_energy[L, H_kv, D]           = mean over (item, t, q_in_kv_group) of Q_d^2
  q_energy_text[L, H_kv, D]      = mean only over text positions
  q_energy_visual[L, H_kv, D]    = mean only over visual positions

Q is captured via a forward hook on each layer's `q_proj` (pre-RoPE). RoPE
preserves per-pair-of-channels (d, d+1) energy, so per-channel Q-energy is
approximately preserved after RoPE — sufficient for a closed-form score-cal
reweighting heuristic.

K is captured via a custom DynamicCache subclass that records K stats on
update() before delegating to the parent.

Output: two-file pair at `qwen/calibration/expF_kcalib_{model}_frames{F}.{json,npz}`.
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
# K-stats-capturing cache
# ===================================================================


class KStatsCache(DynamicCache):
    """DynamicCache that accumulates per-(layer, kv-head, channel) K stats.

    On update() we sum K**2 and track running max(|K|) across (item, t)
    before delegating to DynamicCache.update().
    """

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # accumulators in float32
        self.k_sumsq = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
        self.k_count = torch.zeros(num_layers, dtype=torch.int64)
        self.k_max = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # K shape: [B, H_kv, T, D]
        with torch.no_grad():
            B, H, T, D = key_states.shape
            # Sum-of-squares and max-abs over (B, T)
            kf = key_states.float()
            sumsq = (kf * kf).sum(dim=(0, 2)).cpu()  # [H, D]
            mx = kf.abs().amax(dim=(0, 2)).cpu()     # [H, D]
            self.k_sumsq[layer_idx] += sumsq
            self.k_count[layer_idx] += int(B * T)
            self.k_max[layer_idx] = torch.maximum(self.k_max[layer_idx], mx)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


# ===================================================================
# Q-stats forward hook
# ===================================================================


class QStatsHook:
    """Captures per-(layer, kv-head, channel) Q stats via q_proj forward hooks.

    Per item we know visual-token span [v_start, v_end). Within that span we
    accumulate Q^2 separately for text (positions outside [v_start, v_end))
    and visual (inside). To cap memory, we sub-sample query positions per
    item per layer to `max_q_per_item` (default 256) — giving 100 items x
    256 positions per layer = 25600 sample positions per layer for the
    aggregated Q-energy estimates (more than enough for stable diagonals).

    The hook stores accumulators on `self`. After all items are run, call
    `finalize()` to get the per-(L, H_kv, D) tensors.
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

        self.q_sumsq = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
        self.q_count = torch.zeros(num_layers, dtype=torch.int64)
        self.q_sumsq_text = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
        self.q_count_text = torch.zeros(num_layers, dtype=torch.int64)
        self.q_sumsq_vis = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
        self.q_count_vis = torch.zeros(num_layers, dtype=torch.int64)

        # Per-item state: set before each item via set_item_context().
        self._v_start = -1
        self._v_end = -1

        # Find layer modules and register hooks on q_proj.
        self._handles = []
        layers = self._find_decoder_layers(model)
        if len(layers) != num_layers:
            print(f"[Q-hook][warn] decoder_layers found={len(layers)} expected={num_layers}")
        for layer_idx, attn in layers:
            q_proj = getattr(attn, "q_proj", None)
            if q_proj is None:
                print(f"[Q-hook][warn] layer {layer_idx} has no q_proj; skip")
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

    def set_item_context(self, v_start: int, v_end: int) -> None:
        self._v_start = int(v_start)
        self._v_end = int(v_end)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            with torch.no_grad():
                # output: [B, T, H_q * D]
                q = output
                if q.dim() != 3:
                    return
                B, T, total = q.shape
                if total != self.num_q_heads * self.head_dim:
                    return
                qf = q.detach().float().view(B, T, self.num_q_heads, self.head_dim)
                # collapse q_heads -> kv_heads by averaging within each group of size `self.group`.
                # That preserves per-pair-of-channels energy and reduces noise.
                qf = qf.view(B, T, self.num_kv_heads, self.group, self.head_dim)
                # Compute Q^2 per kv_head per channel, summed over q_in_kv_group then over batch.
                # Subsample T to cap memory.
                t_take = min(T, self.max_q_per_item)
                # Deterministic subsample: uniform stride.
                if T > t_take:
                    idx = torch.linspace(0, T - 1, steps=t_take, dtype=torch.long, device=q.device)
                    qf_sub = qf.index_select(dim=1, index=idx)
                    pos_abs = idx.cpu().tolist()
                else:
                    qf_sub = qf
                    pos_abs = list(range(T))
                qsq = (qf_sub * qf_sub).sum(dim=3)  # sum over group dim -> [B, t_take, H_kv, D]
                # Aggregate
                # Total
                self.q_sumsq[layer_idx] += qsq.sum(dim=(0, 1)).cpu()  # [H_kv, D]
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
                        text_t = torch.tensor(text_idx, dtype=torch.long, device=q.device)
                        self.q_sumsq_text[layer_idx] += qsq.index_select(
                            dim=1, index=text_t,
                        ).sum(dim=(0, 1)).cpu()
                        self.q_count_text[layer_idx] += int(B * len(text_idx) * self.group)
                    if vis_idx:
                        vis_t = torch.tensor(vis_idx, dtype=torch.long, device=q.device)
                        self.q_sumsq_vis[layer_idx] += qsq.index_select(
                            dim=1, index=vis_t,
                        ).sum(dim=(0, 1)).cpu()
                        self.q_count_vis[layer_idx] += int(B * len(vis_idx) * self.group)
        return hook

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def finalize(self) -> dict[str, np.ndarray]:
        """Compute means and return as numpy arrays."""
        def _mean_per_layer(sumsq: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(sumsq)
            for L in range(sumsq.shape[0]):
                c = float(counts[L].item())
                if c > 0:
                    out[L] = sumsq[L] / c
            return out
        q_e = _mean_per_layer(self.q_sumsq, self.q_count)
        q_e_t = _mean_per_layer(self.q_sumsq_text, self.q_count_text)
        q_e_v = _mean_per_layer(self.q_sumsq_vis, self.q_count_vis)
        return {
            "q_energy": q_e.numpy().astype(np.float32),
            "q_energy_text": q_e_t.numpy().astype(np.float32),
            "q_energy_visual": q_e_v.numpy().astype(np.float32),
            "q_count_total": self.q_count.numpy().astype(np.int64),
            "q_count_text": self.q_count_text.numpy().astype(np.int64),
            "q_count_visual": self.q_count_vis.numpy().astype(np.int64),
        }


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
    print(f"[F-calib] cal_items={len(cal_items)} model={model_id} frames={frames}", flush=True)

    # Use SDPA for speed; we don't need eager attention here.
    model, processor = load_model(model_id, awq=False, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", 4))
    num_q_heads = int(getattr(model.config, "num_attention_heads", 28))
    # Probe head_dim from a layer's q_proj weight: shape [H_q*D, hidden]
    q_proj_weight = layers[0].self_attn.q_proj.weight
    head_dim = int(q_proj_weight.shape[0] // num_q_heads)
    print(f"[F-calib] model={model_id} num_layers={num_layers} num_kv_heads={num_kv_heads} "
          f"num_q_heads={num_q_heads} head_dim={head_dim}", flush=True)

    progress_log = out_json.with_name(out_json.stem + ".progress.log")
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    def _log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[F-calib] {msg}", flush=True)
        with open(progress_log, "a") as f:
            f.write(f"[{ts}] {msg}\n"); f.flush()

    _log(f"START n_cal={len(cal_items)} max_q_per_item={max_q_per_item}")
    t_start = time.perf_counter()

    q_hook = QStatsHook(model, num_layers=num_layers, num_kv_heads=num_kv_heads,
                        num_q_heads=num_q_heads, head_dim=head_dim,
                        max_q_per_item=max_q_per_item)

    # K-stats accumulators across all items (one shared cache per item is fine).
    k_sumsq_total = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
    k_count_total = torch.zeros(num_layers, dtype=torch.int64)
    k_max_total = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)

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
                q_hook.set_item_context(v_start, v_end)

                cache = KStatsCache(num_layers=num_layers, num_kv_heads=num_kv_heads,
                                    head_dim=head_dim)
                _ = model.generate(**inputs, past_key_values=cache,
                                   max_new_tokens=1, do_sample=False,
                                   return_dict_in_generate=True, output_scores=True,
                                   use_cache=True)
                # Aggregate per-item K stats into totals.
                k_sumsq_total += cache.k_sumsq
                k_count_total += cache.k_count
                k_max_total = torch.maximum(k_max_total, cache.k_max)

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

    # Finalize K stats: mean of K^2 per channel (k_channel_energy).
    k_channel_energy = torch.zeros_like(k_sumsq_total)
    for L in range(num_layers):
        c = float(k_count_total[L].item())
        if c > 0:
            k_channel_energy[L] = k_sumsq_total[L] / c
    k_channel_energy_np = k_channel_energy.numpy().astype(np.float32)
    k_max_np = k_max_total.numpy().astype(np.float32)

    # Outlier channels: top-N by k_channel_energy per (L, H_kv).
    n_top = int(n_outliers_top)
    outlier_idx = np.argsort(k_channel_energy_np, axis=-1)[..., -n_top:][..., ::-1].copy()
    outlier_idx = outlier_idx.astype(np.int32)

    # Q stats from hook.
    q_data = q_hook.finalize()

    # Sanity: clamp tiny zeros so downstream sqrt is stable.
    q_data["q_energy"] = np.clip(q_data["q_energy"], 1e-12, None).astype(np.float32)
    q_data["q_energy_text"] = np.clip(q_data["q_energy_text"], 1e-12, None).astype(np.float32)
    q_data["q_energy_visual"] = np.clip(q_data["q_energy_visual"], 1e-12, None).astype(np.float32)

    # Write JSON metadata.
    meta = {
        "model": model_id,
        "frames": frames,
        "n_cal_items": len(cal_items),
        "n_done": n_done,
        "n_failed": n_failed,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "num_q_heads": num_q_heads,
        "head_dim": head_dim,
        "max_q_per_item": max_q_per_item,
        "k_outlier_n_top": n_top,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": int(time.perf_counter() - t_start),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(meta, indent=2))
    np.savez_compressed(
        out_npz,
        k_channel_energy=k_channel_energy_np,
        k_abs_max=k_max_np,
        outlier_channel_idx_top16=outlier_idx,
        q_energy=q_data["q_energy"],
        q_energy_text=q_data["q_energy_text"],
        q_energy_visual=q_data["q_energy_visual"],
        q_count_total=q_data["q_count_total"],
        q_count_text=q_data["q_count_text"],
        q_count_visual=q_data["q_count_visual"],
    )
    _log(f"DONE wrote meta -> {out_json} arrays -> {out_npz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--max_q_per_item", type=int, default=256,
                    help="Cap on query positions sampled per item per layer.")
    ap.add_argument("--n_outliers_top", type=int, default=16,
                    help="Top-N outlier channels per (L, H_kv) to record.")
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_npz", type=Path, default=None)
    args = ap.parse_args()
    model_short = args.model.split("/")[-1]
    if args.out_json is None:
        args.out_json = CALIBRATION_DIR / f"expF_kcalib_{model_short}_frames{args.frames}.json"
    if args.out_npz is None:
        args.out_npz = CALIBRATION_DIR / f"expF_kcalib_{model_short}_frames{args.frames}.npz"
    run_calibration(
        model_id=args.model, frames=args.frames, n_outliers_top=args.n_outliers_top,
        split_file=args.split_file, out_json=args.out_json, out_npz=args.out_npz,
        max_q_per_item=args.max_q_per_item, progress_every=args.progress_every,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
