"""
Score-combination utilities for the Experiment B online precision-need router.

Reads `qwen/results/diagnostic_signals.jsonl` (produced by diagnostic_pass.py).
Aggregates static priors from cal-only rows (split safety enforced). Provides
per-item online signals and combined scores. Returns a [num_layers, num_kv_heads]
boolean mask of which blocks should be allocated BF16 at the given budget.

The `BitController(mode="V2")` already supports per-(layer, KV-head) bits; this
module just decides the assignment.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch


# ===================================================================
# Loading + percentile rank
# ===================================================================

def load_diagnostic_jsonl(path: Path) -> dict:
    """Group rows by item_id. Returns {item_id: {'split', 'rows': [row, ...] sorted by (layer, kv_head)}}."""
    by_item: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            iid = r["item_id"]
            if iid not in by_item:
                by_item[iid] = {
                    "split": r["split"],
                    "rows": [],
                    "n_frames": r["n_frames"],
                    "duration_bucket": r["duration_bucket"],
                    "correct_choice": r["correct_choice"],
                    "n_options": r.get("n_options", 4),
                    "bf16_pred": r["bf16_pred"],
                    "uniform_int4_pred": r["uniform_int4_pred"],
                    "uniform_int2_pred": r["uniform_int2_pred"],
                }
            by_item[iid]["rows"].append(r)
    for iid in by_item:
        by_item[iid]["rows"].sort(key=lambda r: (r["layer"], r["kv_head"]))
    return by_item


def item_signals(item_record: dict, num_layers: int, num_kv_heads: int) -> dict[str, torch.Tensor]:
    """Reshape per-item rows into per-(layer, KV-head) [L, H] tensors."""
    out = {
        "entropy_mean": torch.full((num_layers, num_kv_heads), float("nan")),
        "entropy_answer_query": torch.full((num_layers, num_kv_heads), float("nan")),
        "aq_topk_mass": torch.full((num_layers, num_kv_heads), float("nan")),
        "aq_frame_entropy": torch.full((num_layers, num_kv_heads), float("nan")),
        "kv_residual_int2": torch.full((num_layers, num_kv_heads), float("nan")),
        "kv_residual_int4": torch.full((num_layers, num_kv_heads), float("nan")),
    }
    for r in item_record["rows"]:
        L, h = r["layer"], r["kv_head"]
        if L >= num_layers or h >= num_kv_heads:
            continue
        for k in out:
            v = r.get(k, float("nan"))
            out[k][L, h] = float(v) if v is not None else float("nan")
    return out


def percentile_rank(scores: torch.Tensor) -> torch.Tensor:
    """Element-wise percentile rank in [0, 1] across the flattened tensor.

    NaN inputs are mapped to 0 (lowest rank). Stable for ties via average rank.
    """
    flat = scores.flatten().clone()
    nan_mask = torch.isnan(flat)
    if nan_mask.any():
        # Replace NaN with very small value so they sort to the bottom
        flat[nan_mask] = float("-inf")
    n = flat.numel()
    sorted_idx = torch.argsort(flat)
    ranks = torch.empty(n, dtype=torch.float32)
    ranks[sorted_idx] = torch.arange(n, dtype=torch.float32) / max(1, n - 1)
    ranks[nan_mask] = 0.0
    return ranks.reshape(scores.shape)


# ===================================================================
# Static-risk aggregation (cal only — eval leakage guard)
# ===================================================================

def infer_dims_from_jsonl(diagnostic_path: Path) -> tuple[int, int]:
    """Returns (num_layers, num_kv_heads) inferred from max(layer)+1 / max(kv_head)+1 in the JSONL."""
    max_L, max_h = 0, 0
    with open(diagnostic_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            max_L = max(max_L, int(r.get("layer", 0)))
            max_h = max(max_h, int(r.get("kv_head", 0)))
    return max_L + 1, max_h + 1


def aggregate_static_risk(diagnostic_path: Path, num_layers: Optional[int] = None,
                          num_kv_heads: Optional[int] = None) -> dict:
    """Mean per-(layer, KV-head) entropy from cal items only. Returns:
        {
          "n_cal": int,
          "mean_entropy_LH": [L, H] tensor,
          "static_low_risk": [L, H] tensor (rank of -mean_entropy, low entropy => high risk),
          "static_high_risk": [L, H] tensor (rank of +mean_entropy, high entropy => high risk),
        }
    Asserts that no eval row contributes — strict split safety.
    """
    if num_layers is None or num_kv_heads is None:
        nL, nH = infer_dims_from_jsonl(diagnostic_path)
        num_layers = num_layers or nL
        num_kv_heads = num_kv_heads or nH
    sums = torch.zeros(num_layers, num_kv_heads)
    counts = torch.zeros(num_layers, num_kv_heads)
    n_cal = 0
    seen_ids: set[str] = set()
    with open(diagnostic_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r["split"] != "cal":
                continue
            L, h = r["layer"], r["kv_head"]
            if L >= num_layers or h >= num_kv_heads:
                continue
            v = r.get("entropy_mean")
            if v is None or (isinstance(v, float) and v != v):
                continue
            sums[L, h] += float(v)
            counts[L, h] += 1
            if r["item_id"] not in seen_ids:
                seen_ids.add(r["item_id"])
    n_cal = len(seen_ids)
    mean_LH = sums / counts.clamp_min(1)
    mean_LH[counts == 0] = float("nan")
    return {
        "n_cal": n_cal,
        "mean_entropy_LH": mean_LH,
        "static_low_risk": percentile_rank(-mean_LH),
        "static_high_risk": percentile_rank(mean_LH),
    }


def save_static_risk(static: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_cal": int(static["n_cal"]),
        "mean_entropy_LH": static["mean_entropy_LH"].tolist(),
        "static_low_risk": static["static_low_risk"].tolist(),
        "static_high_risk": static["static_high_risk"].tolist(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_static_risk(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    return {
        "n_cal": d["n_cal"],
        "mean_entropy_LH": torch.tensor(d["mean_entropy_LH"]),
        "static_low_risk": torch.tensor(d["static_low_risk"]),
        "static_high_risk": torch.tensor(d["static_high_risk"]),
    }


# ===================================================================
# Per-condition score functions
# ===================================================================

def compute_score(
    method: str,
    *,
    num_layers: int,
    num_kv_heads: int,
    static: Optional[dict] = None,            # cal-only static map (from aggregate_static_risk)
    item_sig: Optional[dict[str, torch.Tensor]] = None,   # per-item online signals
    seed: int = 0,
) -> torch.Tensor:
    """Returns a [num_layers, num_kv_heads] score tensor. Higher = more BF16 priority.

    Methods:
      'random'           — random scores per (L, h)
      'meda_layer'       — layer-grain entropy (high entropy → high priority); broadcast across heads
      'static_low'       — frozen cal: low entropy → high priority
      'static_high'      — frozen cal: high entropy → high priority (flipped control)
      'online_residual'  — per-item residual under INT2 (high residual → high priority)
      'online_need_static' — percentile_rank(static_low) × percentile_rank(online residual)
      'online_need_aq'   — percentile_rank(answer-query risk: -aq_topk_mass) × percentile_rank(online residual)
    """
    if method == "random":
        g = torch.Generator()
        g.manual_seed(seed)
        return torch.rand(num_layers, num_kv_heads, generator=g)

    if method == "meda_layer":
        assert static is not None
        layer_score = torch.nan_to_num(static["mean_entropy_LH"], nan=0.0).mean(dim=-1)  # [L]
        # Broadcast: all 4 KV-heads in a layer share the layer's score (so picking top-K layers picks all heads)
        return layer_score.unsqueeze(-1).expand(num_layers, num_kv_heads).contiguous()

    if method == "static_low":
        assert static is not None
        return static["static_low_risk"]

    if method == "static_high":
        assert static is not None
        return static["static_high_risk"]

    if method == "online_residual":
        assert item_sig is not None
        return torch.nan_to_num(item_sig["kv_residual_int2"], nan=0.0)

    if method == "online_need_static":
        assert static is not None and item_sig is not None
        a = static["static_low_risk"]
        b = percentile_rank(torch.nan_to_num(item_sig["kv_residual_int2"], nan=0.0))
        return a * b

    if method == "online_need_aq":
        assert item_sig is not None
        # Low aq_topk_mass = diffuse attention = harder to recover from quant -> higher priority
        aq_risk = -torch.nan_to_num(item_sig["aq_topk_mass"], nan=0.0)
        a = percentile_rank(aq_risk)
        b = percentile_rank(torch.nan_to_num(item_sig["kv_residual_int2"], nan=0.0))
        return a * b

    raise ValueError(f"unknown method: {method}")


# ===================================================================
# Mask + bits assignment
# ===================================================================

def top_k_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Returns a [num_layers, num_kv_heads] bool mask with exactly k True entries."""
    flat = scores.flatten()
    n = flat.numel()
    k = min(k, n)
    if k <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    # In case of ties, argsort gives a deterministic break
    top_idx = torch.topk(flat, k, dim=0).indices
    mask_flat = torch.zeros(n, dtype=torch.bool)
    mask_flat[top_idx] = True
    return mask_flat.reshape(scores.shape)


def bits_from_mask(mask: torch.Tensor, hi_bits: int = 16, lo_bits: int = 2) -> torch.Tensor:
    """Returns a [num_layers, num_kv_heads] int tensor with hi_bits where mask is True, else lo_bits."""
    return torch.where(mask, torch.tensor(hi_bits, dtype=torch.long),
                       torch.tensor(lo_bits, dtype=torch.long))


def apply_to_controller(controller, bits_LH: torch.Tensor) -> None:
    """Set per-layer per-KV-head bits on a V2 BitController."""
    num_layers, num_kv_heads = bits_LH.shape
    for L in range(num_layers):
        bits_L = bits_LH[L].long()
        controller.set_layer(L, k_bits=bits_L, v_bits=bits_L)


def avg_bits_from_LH(bits_LH: torch.Tensor) -> float:
    return float(bits_LH.float().mean().item())


# ===================================================================
# CLI: aggregate static risk + dump
# ===================================================================

def _cli():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnostic", type=Path, required=True)
    ap.add_argument("--num_layers", type=int, default=None,
                    help="Default: inferred from JSONL max(layer)+1")
    ap.add_argument("--num_kv_heads", type=int, default=None,
                    help="Default: inferred from JSONL max(kv_head)+1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    static = aggregate_static_risk(args.diagnostic, args.num_layers, args.num_kv_heads)
    save_static_risk(static, args.out)
    print(f"[scoring] n_cal={static['n_cal']} -> {args.out}")
    valid = static["mean_entropy_LH"][~torch.isnan(static["mean_entropy_LH"])]
    if valid.numel() > 0:
        print(f"  mean_entropy_LH range: [{valid.min():.3f}, {valid.max():.3f}] "
              f"(n_valid={valid.numel()}/{static['mean_entropy_LH'].numel()})")


if __name__ == "__main__":
    _cli()
