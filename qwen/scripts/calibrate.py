"""
Calibration pass: BF16 weights + BF16 KV + entropy hook over the cal split.

Computes per-(layer, head) attention entropy averaged over forward passes and
examples, then derives bit-allocation thresholds for V1/V2/V3 controllers
targeting a specified average KV-bit budget. Writes a frozen JSON file under
qwen/calibration/.

Usage:
    python calibrate.py --model Qwen/Qwen2.5-VL-7B-Instruct \
                        --frames 64 --target_avg_bits 3.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from attn_entropy_hook import (
    aggregate_layer_entropy,
    aggregate_layer_head_entropy,
    entropy_hook,
)
from data_longvideobench import (
    DEFAULT_SPLIT_FILE,
    filter_items,
    load_all_items,
    load_split,
)
from fake_quant_kv_cache import BitController, FakeQuantKVCache
from run_inference import load_model, score_item


CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"


# ---------------- bit-allocation policies ----------------

BIT_LEVELS = (2, 4, 8)  # bottom -> top tertile


def _allocate_by_tertile(values: torch.Tensor, target_avg_bits: float,
                         levels: tuple[int, ...] = BIT_LEVELS) -> torch.Tensor:
    """Tertile-bin `values` (low->high) into `levels`; rebalance to hit target_avg_bits.

    Strategy: start with strict tertiles (bottom->levels[0], mid->levels[1], top->levels[-1]).
    Then move the boundary thresholds in unit steps until the resulting average is within
    0.05 of `target_avg_bits` or boundaries collapse.
    """
    flat = values.flatten().clone()
    flat[torch.isnan(flat)] = flat[~torch.isnan(flat)].median()
    n = flat.numel()
    sorted_vals, sorted_idx = torch.sort(flat)

    def assign(t1: int, t2: int) -> torch.Tensor:
        # t1, t2 are indices into the sorted values: <t1 -> levels[0], <t2 -> levels[1], else levels[-1]
        t1 = max(1, min(n - 2, t1))
        t2 = max(t1 + 1, min(n - 1, t2))
        bits_sorted = torch.full((n,), levels[-1], dtype=torch.long)
        bits_sorted[:t1] = levels[0]
        bits_sorted[t1:t2] = levels[1]
        out = torch.zeros_like(bits_sorted)
        out[sorted_idx] = bits_sorted
        return out

    t1, t2 = n // 3, 2 * n // 3
    best = assign(t1, t2)
    best_diff = abs(best.float().mean().item() - target_avg_bits)
    for shift in range(0, n // 6):
        for ds1, ds2 in ((-shift, -shift), (-shift, +shift), (+shift, -shift), (+shift, +shift)):
            cand = assign(t1 + ds1, t2 + ds2)
            diff = abs(cand.float().mean().item() - target_avg_bits)
            if diff < best_diff:
                best, best_diff = cand, diff
        if best_diff < 0.05:
            break
    return best.reshape(values.shape)


# ---------------- main ----------------

def run(args):
    items = load_all_items()
    split = load_split(args.split_file)
    cal_items = filter_items(items, split["cal"])
    print(f"[calibrate] cal_items={len(cal_items)} target_avg_bits={args.target_avg_bits}")

    model, processor = load_model(args.model, awq=False, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    num_layers = len(getattr(getattr(model, "language_model", None), "layers",
                             getattr(model.model, "layers", [])))
    print(f"[calibrate] num_layers={num_layers}")

    bf16_controller = BitController(num_layers=num_layers, num_kv_heads=4, mode="V1",
                                    default_k_bits=16, default_v_bits=16)
    cache = FakeQuantKVCache(bf16_controller)

    with entropy_hook(model, cache) as _hook:
        for i, it in enumerate(cal_items):
            r = score_item(model, processor, it, n_frames=args.frames,
                           controller=bf16_controller,
                           condition=f"cal_bf16_frames{args.frames}",
                           model_id=args.model)
            if i % 10 == 0:
                print(f"[calibrate] {i}/{len(cal_items)} acc-so-far={int(r.is_correct)}")

    layer_ent = aggregate_layer_entropy(cache)         # [L]
    layer_head_ent = aggregate_layer_head_entropy(cache)  # [L, num_q_heads]
    print(f"[calibrate] layer entropy: min={layer_ent.min():.3f} max={layer_ent.max():.3f}")

    # V1: per-layer bits
    v1_bits = _allocate_by_tertile(layer_ent, args.target_avg_bits)
    # V2: per-(layer, kv_head) bits — collapse Q-head entropy to KV-head granularity by max
    num_kv_heads = 4
    num_q_heads = layer_head_ent.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise RuntimeError(f"num_q_heads ({num_q_heads}) not divisible by num_kv_heads ({num_kv_heads})")
    group = num_q_heads // num_kv_heads
    layer_kvhead_ent = layer_head_ent.reshape(layer_head_ent.shape[0], num_kv_heads, group).max(dim=-1).values
    v2_bits = _allocate_by_tertile(layer_kvhead_ent, args.target_avg_bits)
    # V3: per-layer protected fraction p_l = sigmoid-ish of layer entropy
    layer_norm_ent = (layer_ent - layer_ent.min()) / (layer_ent.max() - layer_ent.min() + 1e-8)
    p_per_layer = (0.1 + layer_norm_ent * 0.4).tolist()  # protected fraction in [0.1, 0.5]

    out = {
        "model": args.model,
        "frames": args.frames,
        "target_avg_bits": args.target_avg_bits,
        "num_layers": int(num_layers),
        "num_kv_heads": int(num_kv_heads),
        "layer_entropy": layer_ent.tolist(),
        "layer_kvhead_entropy": layer_kvhead_ent.tolist(),
        "v1_bits_per_layer": v1_bits.tolist(),
        "v2_bits_per_layer_kvhead": v2_bits.tolist(),
        "v3_protected_fraction_per_layer": p_per_layer,
        "v3_hi_bits": 4,
        "v3_lo_bits": 2,
        "n_cal_items": len(cal_items),
    }
    out_path = CALIBRATION_DIR / f"thresholds_{args.model.split('/')[-1]}_avg{args.target_avg_bits:.1f}_frames{args.frames}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calibrate] wrote thresholds -> {out_path}")
    print(f"[calibrate] V1 avg bits: {np.mean(v1_bits.tolist()):.2f}")
    print(f"[calibrate] V2 avg bits: {np.mean(v2_bits.tolist()):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--target_avg_bits", type=float, default=3.0)
    ap.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
