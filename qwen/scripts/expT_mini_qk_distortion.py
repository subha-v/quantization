"""QK score-distortion diagnostic for Exp T-mini.

Compares the attention-score matrices QK_bf16 vs QK_quant per layer and
per modality-block (TT/TV/VT/VV) under each candidate K format. For each
item and each condition, we monkey-patch torch SDPA to capture (Q, K)
just before SDPA fires, then externally compute and compare
QK = (Q @ K.T) * scale on a subsampled set of query positions.

Metrics (per item, condition, layer, kv_head, modality_block):
  qk_mse     = mean((QK_bf16 - QK_quant)^2)  over (query, key) in the block
               and after causal masking
  qk_corr    = Pearson correlation between the two flattened blocks
  top1_pres  = fraction of sampled queries where argmax(QK_bf16) == argmax(QK_quant)
  top10_ovr  = mean over sampled queries of |top10(QK_bf16) ∩ top10(QK_quant)| / 10
  margin_chg = (margin_quant - margin_bf16) per item

If F9 / SJ reduce QK distortion in *T blocks (queries attending to text/choice
keys) while PageLocal only reduces distortion in *V blocks (visual keys), the
modality breakdown will show it directly.

This is the most important missing diagnostic — it tells us *why* PageLocal
doesn't close the gap to F9 on aggregate accuracy: it's repairing the wrong
side of the attention pattern.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 expT_mini_qk_distortion.py \
        --n-items 4 --conditions BF16 F4 F9 PageLocal \
        --out-jsonl results/expT_mini_qk_distortion.jsonl \
        --out-summary results/expT_mini_qk_distortion.md
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# --------------- SDPA monkey-patch capture ----------------

_CAPTURE: dict = {"q": [], "k": [], "active": False}
_ORIG_SDPA = None


def install_capture(num_layers: int):
    """Monkey-patch torch.nn.functional.scaled_dot_product_attention so each
    call appends (Q, K) onto _CAPTURE in arrival order. We rely on the fact
    that Qwen2.5-VL calls SDPA exactly once per layer per forward pass.
    """
    global _ORIG_SDPA
    import torch.nn.functional as F
    if _ORIG_SDPA is None:
        _ORIG_SDPA = F.scaled_dot_product_attention

    def patched(query, key, value, *args, **kwargs):
        if _CAPTURE["active"]:
            # Move to CPU + bf16 immediately to free GPU memory.
            _CAPTURE["q"].append(query.detach().to("cpu", torch.bfloat16))
            _CAPTURE["k"].append(key.detach().to("cpu", torch.bfloat16))
        return _ORIG_SDPA(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = patched


def reset_capture():
    _CAPTURE["q"].clear()
    _CAPTURE["k"].clear()


def set_capture_active(active: bool):
    _CAPTURE["active"] = active


# --------------- per-item processing ----------------

def _sample_query_positions(text_positions: list[int],
                            visual_positions: list[int],
                            n_per_modality: int = 32,
                            seq_len: int = 0,
                            rng: Optional[np.random.Generator] = None) -> dict:
    """Return {"text_qpos": [...], "vis_qpos": [...]} — subsampled query
    positions. We sample uniformly without replacement; for very short
    text/visual spans we take all positions.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    text_pool = [p for p in text_positions if p < seq_len]
    vis_pool = [p for p in visual_positions if p < seq_len]
    n_text = min(n_per_modality, len(text_pool))
    n_vis = min(n_per_modality, len(vis_pool))
    text_q = sorted(rng.choice(text_pool, size=n_text, replace=False).tolist()) if n_text > 0 else []
    vis_q = sorted(rng.choice(vis_pool, size=n_vis, replace=False).tolist()) if n_vis > 0 else []
    return {"text_qpos": text_q, "vis_qpos": vis_q}


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between two flat vectors. Returns NaN if degenerate."""
    x = x.float().flatten()
    y = y.float().flatten()
    if x.numel() < 2:
        return float("nan")
    xm = x.mean()
    ym = y.mean()
    xc = x - xm
    yc = y - ym
    denom = (torch.sqrt((xc * xc).sum()) * torch.sqrt((yc * yc).sum())).item()
    if denom == 0.0:
        return float("nan")
    return float((xc * yc).sum().item() / denom)


def _compute_block_metrics(Q_bf: torch.Tensor, K_bf: torch.Tensor,
                           Q_q: torch.Tensor, K_q: torch.Tensor,
                           q_pos: list[int], k_pos: list[int],
                           seq_len: int, head_dim: int) -> Optional[dict]:
    """Compute MSE/corr/top1/top10 between QK_bf and QK_q on a (q_pos, k_pos) block.

    Q tensors: [n_qh, T, D]
    K tensors: [T, D]
    q_pos / k_pos: indices into [0, T)
    Returns dict of metrics, or None if empty.
    """
    if not q_pos or not k_pos:
        return None
    q_idx = torch.tensor(q_pos, dtype=torch.long)
    k_idx = torch.tensor(k_pos, dtype=torch.long)

    # Subselect query rows. Q shape [n_qh, len(q_pos), D].
    Q_bf_sub = Q_bf.index_select(dim=-2, index=q_idx).float()
    Q_q_sub = Q_q.index_select(dim=-2, index=q_idx).float()
    K_bf_sub = K_bf.index_select(dim=-2, index=k_idx).float()
    K_q_sub = K_q.index_select(dim=-2, index=k_idx).float()

    scale = 1.0 / math.sqrt(head_dim)
    # [n_qh, len(q), len(k)]
    QK_bf = torch.matmul(Q_bf_sub, K_bf_sub.transpose(-1, -2)) * scale
    QK_q = torch.matmul(Q_q_sub, K_q_sub.transpose(-1, -2)) * scale

    # Apply causal mask: keep only entries where k_pos[j] <= q_pos[i].
    qi = q_idx.unsqueeze(-1)  # [len(q), 1]
    kj = k_idx.unsqueeze(0)   # [1, len(k)]
    mask = (kj <= qi)         # [len(q), len(k)]
    if not mask.any():
        return None

    # Flatten the masked region.
    mask_b = mask.unsqueeze(0).expand_as(QK_bf)
    QK_bf_flat = QK_bf[mask_b]
    QK_q_flat = QK_q[mask_b]

    mse = ((QK_bf_flat - QK_q_flat) ** 2).mean().item()
    corr = _pearson(QK_bf_flat, QK_q_flat)
    return {"qk_mse": mse, "qk_corr": corr, "n_entries": int(QK_bf_flat.numel())}


def _compute_topk_preservation(Q_bf: torch.Tensor, K_bf: torch.Tensor,
                               Q_q: torch.Tensor, K_q: torch.Tensor,
                               q_pos: list[int], head_dim: int) -> dict:
    """For each query position in q_pos, compute the top-1/top-10 key preservation.
    Keys considered: all positions with index <= query (causal). Returns
    average top-1 match rate and average top-10 overlap fraction.

    Q tensors: [n_qh, T, D]
    K tensors: [T, D]
    """
    if not q_pos:
        return {"top1_pres": float("nan"), "top10_ovr": float("nan"), "n_queries": 0}
    scale = 1.0 / math.sqrt(head_dim)
    top1_matches = []
    top10_ovr = []
    for qi in q_pos:
        T = K_bf.shape[0]
        if qi <= 0:
            continue
        keys_lo, keys_hi = 0, qi + 1  # causal: keys are positions [0, qi]
        K_bf_q = K_bf[keys_lo:keys_hi].float()
        K_q_q = K_q[keys_lo:keys_hi].float()
        # Average over query heads (GQA-pool). Use first qh for speed
        # (averaging across heads doesn't add much information beyond per-head).
        for h in range(Q_bf.shape[0]):
            q_bf = Q_bf[h, qi].float()  # [D]
            q_q = Q_q[h, qi].float()    # [D]
            scores_bf = (K_bf_q @ q_bf) * scale  # [keys_hi]
            scores_q = (K_q_q @ q_q) * scale    # [keys_hi]
            top1_bf = int(scores_bf.argmax().item())
            top1_q = int(scores_q.argmax().item())
            top1_matches.append(top1_bf == top1_q)
            if scores_bf.numel() >= 10:
                top10_bf = set(scores_bf.topk(10).indices.tolist())
                top10_q = set(scores_q.topk(10).indices.tolist())
                top10_ovr.append(len(top10_bf & top10_q) / 10.0)
    return {
        "top1_pres": float(np.mean(top1_matches)) if top1_matches else float("nan"),
        "top10_ovr": float(np.mean(top10_ovr)) if top10_ovr else float("nan"),
        "n_queries": len(top1_matches),
    }


# --------------- driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--task", default="retrieval-image",
                    choices=("retrieval-image",))  # only retrieval-image for now
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=4,
                    help="Number of items to diagnose (default 4).")
    ap.add_argument("--n-queries-per-modality", type=int, default=32,
                    help="Subsampled query positions per modality (text, visual).")
    ap.add_argument("--conditions", nargs="+",
                    default=["BF16", "F4", "F9", "SJ", "S4", "PageLocal"],
                    help="Condition names to evaluate. BF16 is mandatory (reference).")
    ap.add_argument("--calib-npz", type=Path,
                    default=Path(__file__).resolve().parents[1] / "calibration"
                    / "expP_mmniah_kcalib_Qwen2.5-VL-7B-Instruct_seed0.npz")
    ap.add_argument("--max-pixels-context", type=int, default=336 * 336)
    ap.add_argument("--max-pixels-choices", type=int, default=336 * 336)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-jsonl", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results"
                    / "expT_mini_qk_distortion.jsonl")
    ap.add_argument("--out-summary", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results"
                    / "expT_mini_qk_distortion.md")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    # Lazy imports — after CUDA env is set
    from mm_niah_loader import (
        SUPPORTED_TASKS, answer_token_ids, filter_items, format_mcq_messages,
        load_all_items, load_split, make_split, save_split, split_file_for_task,
    )
    from fake_quant_kv_cache import BitController, FakeQuantKVCache
    from k_quantizers import build_f_conditions
    from page_envelope_cache import PageAwareFakeQuantKVCache
    from page_layout import build_page_layout
    from run_inference import load_model
    from attention_router import RoutePolicy, page_routing_sdpa_context

    # ---- map condition name -> k_cfg name (or None for BF16) ----
    NAME_TO_KCFG = {
        "BF16": None,
        "F4": "F4_KIVI_PerChannelSeq",
        "F9": "F9_KIVI_Outlier16",
        "SJ": "J12_F9_INT8side",
        "S4": "SL_Outlier16_INT7side",
        "PageLocal": "T8_PageLocal_F4",
    }
    for c in args.conditions:
        if c not in NAME_TO_KCFG:
            raise ValueError(f"unknown condition {c!r}; choose from {list(NAME_TO_KCFG)}")

    # Items
    items = load_all_items(task=args.task)
    split_path = split_file_for_task(args.task)
    if not split_path.exists():
        split = make_split(items, seed=args.seed)
        save_split(split, split_path)
    else:
        split = load_split(split_path)
    cal_ids = set(split.get("cal", []))
    eval_items = [it for it in items if it.id not in cal_ids and it.num_images >= 8]
    eval_items = eval_items[: args.n_items]
    print(f"diagnosing {len(eval_items)} items, conditions={args.conditions}", flush=True)

    # Calibration
    calib = None
    if args.calib_npz.exists():
        arr = np.load(args.calib_npz)
        calib = {k: arr[k] for k in arr.files}
        print(f"loaded calib {args.calib_npz}", flush=True)
    else:
        print(f"[warn] calib NPZ not found at {args.calib_npz}", flush=True)

    # Resolve k_cfgs once
    cfgs = build_f_conditions(calib=calib)
    name_to_cfg = {c.name: c for c in cfgs}
    cond_to_cfg = {n: (name_to_cfg.get(NAME_TO_KCFG[n]) if NAME_TO_KCFG[n] else None)
                   for n in args.conditions}

    # Model
    print(f"loading model {args.model}...", flush=True)
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    cfg = model.config
    num_layers = int(getattr(cfg, "num_hidden_layers", None)
                     or cfg.text_config.num_hidden_layers)
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", None)
                       or cfg.text_config.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", None)
                   or cfg.text_config.hidden_size // cfg.text_config.num_attention_heads)
    print(f"  num_layers={num_layers} num_kv_heads={num_kv_heads} head_dim={head_dim}",
          flush=True)
    install_capture(num_layers)
    answer_ids = answer_token_ids(processor, n=4)

    from qwen_vl_utils import process_vision_info  # type: ignore

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out_jsonl, "a")
    t_start = time.perf_counter()

    for item_idx, item in enumerate(eval_items):
        print(f"\n=== item {item_idx + 1}/{len(eval_items)} id={item.id} "
              f"num_images={item.num_images} ctx={item.context_length} ===", flush=True)

        # Build inputs
        messages = format_mcq_messages(item,
                                       max_pixels_context=args.max_pixels_context,
                                       max_pixels_choices=args.max_pixels_choices)
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        seq_len = int(input_ids.shape[1])

        # Page layout for modality classification
        layout = build_page_layout(
            input_ids, processor,
            n_in_context_images=item.num_images, n_choice_images=4,
            needle_idx_in_images=item.needle_idx_in_images,
            include_choice_routing=False,
        )
        text_positions: list[int] = []
        visual_positions: list[int] = []
        for p in layout.pages:
            positions = list(range(int(p.start), int(p.end)))
            if p.kind in ("in_context_image", "choice_image"):
                visual_positions.extend(positions)
            else:
                text_positions.extend(positions)

        # Sampled queries (held constant across conditions for fair comparison)
        rng = np.random.default_rng(abs(hash(item.id)) % (2 ** 31))
        sampled = _sample_query_positions(
            text_positions, visual_positions,
            n_per_modality=args.n_queries_per_modality, seq_len=seq_len, rng=rng,
        )
        print(f"  sampled_queries: text={len(sampled['text_qpos'])} "
              f"visual={len(sampled['vis_qpos'])} | text_pool={len(text_positions)} "
              f"visual_pool={len(visual_positions)}", flush=True)

        # Build slice_info (for T-mini K kinds)
        slice_info = {
            "v_start": (visual_positions[0] if visual_positions else -1),
            "v_end": (visual_positions[-1] + 1 if visual_positions else -1),
            "seq_len": seq_len,
            "text_positions": text_positions,
            "visual_positions": visual_positions,
            "role_spans": {},
            "page_boundaries": [(p.start, p.end, p.kind) for p in layout.pages],
            "visual_token_positions_per_image": [
                list(range(p.start, p.end)) for p in layout.pages
                if p.kind in ("in_context_image", "choice_image")
            ],
            "text_chunk_positions": [
                list(range(p.start, p.end)) for p in layout.pages if p.kind == "text"
            ],
            "item_id": item.id,
        }

        # ----- BF16 reference pass -----
        print("  [BF16 reference forward...]", flush=True)
        reset_capture()
        set_capture_active(True)
        with torch.no_grad():
            out_bf16 = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                      return_dict_in_generate=True, output_scores=True,
                                      use_cache=True)
        set_capture_active(False)
        Q_layers_bf16 = list(_CAPTURE["q"])
        K_layers_bf16 = list(_CAPTURE["k"])
        if len(Q_layers_bf16) != num_layers:
            print(f"  [warn] captured {len(Q_layers_bf16)} layers, expected {num_layers}",
                  flush=True)
        logp_bf16 = torch.log_softmax(out_bf16.scores[0].float(), dim=-1)[0, answer_ids].tolist()
        pred_bf16 = int(max(range(len(answer_ids)), key=lambda i: logp_bf16[i]))
        wrong_lp = [v for i, v in enumerate(logp_bf16) if i != item.correct_choice]
        margin_bf16 = logp_bf16[item.correct_choice] - max(wrong_lp)
        del out_bf16
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----- each quant condition -----
        for cond_name in args.conditions:
            if cond_name == "BF16":
                continue
            cfg_obj = cond_to_cfg[cond_name]
            if cfg_obj is None:
                continue
            print(f"  [{cond_name} forward...]", flush=True)
            controller = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads,
                                       mode="V1", default_k_bits=4, default_v_bits=4)
            cache = PageAwareFakeQuantKVCache(controller, k_quantizer_config=cfg_obj)
            cache.set_page_layout(layout, rng_seed=abs(hash(f"{item.id}:{cond_name}")) % (2 ** 31))
            cache.correct_choice_idx = int(item.correct_choice)
            cache.set_slice_info(slice_info)

            reset_capture()
            set_capture_active(True)
            with torch.no_grad():
                with page_routing_sdpa_context(cache, RoutePolicy("none")):
                    out_q = model.generate(**inputs, past_key_values=cache,
                                           max_new_tokens=1, do_sample=False,
                                           return_dict_in_generate=True,
                                           output_scores=True, use_cache=True)
            set_capture_active(False)
            Q_layers_q = list(_CAPTURE["q"])
            K_layers_q = list(_CAPTURE["k"])
            logp_q = torch.log_softmax(out_q.scores[0].float(), dim=-1)[0, answer_ids].tolist()
            pred_q = int(max(range(len(answer_ids)), key=lambda i: logp_q[i]))
            wrong_lp_q = [v for i, v in enumerate(logp_q) if i != item.correct_choice]
            margin_q = logp_q[item.correct_choice] - max(wrong_lp_q)
            del out_q, cache, controller
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ---- per-layer per-kv_head metrics ----
            n_layers_eff = min(len(Q_layers_bf16), len(Q_layers_q), num_layers)
            for L in range(n_layers_eff):
                Q_bf_L = Q_layers_bf16[L][0]   # [n_qh, T, D] (drop batch)
                K_bf_L = K_layers_bf16[L][0]   # [n_kv, T, D]
                Q_q_L = Q_layers_q[L][0]
                K_q_L = K_layers_q[L][0]
                n_qh = Q_bf_L.shape[0]
                n_kv = K_bf_L.shape[0]
                qh_per_kv = max(1, n_qh // n_kv)

                # Per kv_head, pool the query-heads that map to it
                for h_kv in range(n_kv):
                    h_lo = h_kv * qh_per_kv
                    h_hi = (h_kv + 1) * qh_per_kv
                    Q_bf_h = Q_bf_L[h_lo:h_hi]   # [qh_per_kv, T, D]
                    Q_q_h = Q_q_L[h_lo:h_hi]
                    K_bf_h = K_bf_L[h_kv]        # [T, D]
                    K_q_h = K_q_L[h_kv]

                    # Per modality block: TT, TV, VT, VV
                    for q_mod, q_pos in (("T", sampled["text_qpos"]),
                                         ("V", sampled["vis_qpos"])):
                        for k_mod, k_pos in (("T", text_positions),
                                             ("V", visual_positions)):
                            block = q_mod + k_mod
                            m = _compute_block_metrics(
                                Q_bf_h, K_bf_h, Q_q_h, K_q_h,
                                q_pos, k_pos, seq_len, head_dim,
                            )
                            if m is None:
                                continue
                            row = {
                                "item_id": item.id,
                                "condition": cond_name,
                                "layer": L,
                                "kv_head": h_kv,
                                "modality_block": block,
                                **m,
                                "n_qh_pooled": int(h_hi - h_lo),
                                "seq_len": seq_len,
                                "num_images": item.num_images,
                                "context_length_bucket": item.context_length_bucket,
                            }
                            out_f.write(json.dumps(row) + "\n")
                            out_f.flush()

                    # Top-k preservation per modality of query (uses full key set)
                    for q_mod, q_pos in (("T", sampled["text_qpos"]),
                                         ("V", sampled["vis_qpos"])):
                        topk = _compute_topk_preservation(
                            Q_bf_h, K_bf_h, Q_q_h, K_q_h,
                            q_pos, head_dim,
                        )
                        row = {
                            "item_id": item.id,
                            "condition": cond_name,
                            "layer": L,
                            "kv_head": h_kv,
                            "modality_block": f"{q_mod}*",
                            "metric_kind": "topk_preservation",
                            **topk,
                            "seq_len": seq_len,
                            "num_images": item.num_images,
                        }
                        out_f.write(json.dumps(row) + "\n")
                        out_f.flush()

            # Margin change row (per item, per condition)
            row = {
                "item_id": item.id,
                "condition": cond_name,
                "metric_kind": "margin_change",
                "margin_bf16": float(margin_bf16),
                "margin_quant": float(margin_q),
                "margin_change": float(margin_q - margin_bf16),
                "pred_bf16": pred_bf16,
                "pred_quant": pred_q,
                "correct_choice": int(item.correct_choice),
                "bf16_correct": bool(pred_bf16 == item.correct_choice),
                "quant_correct": bool(pred_q == item.correct_choice),
                "seq_len": seq_len,
                "num_images": item.num_images,
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

            elapsed = time.perf_counter() - t_start
            print(f"    {cond_name}: margin_bf16={margin_bf16:.3f} "
                  f"margin_quant={margin_q:.3f} change={margin_q - margin_bf16:+.3f} | "
                  f"elapsed={timedelta(seconds=int(elapsed))}", flush=True)

        # Free per-item captures
        del Q_layers_bf16, K_layers_bf16
        gc.collect()

    out_f.close()

    # ---- summary ----
    rows = []
    with open(args.out_jsonl) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    block_rows = [r for r in rows if "modality_block" in r and "qk_mse" in r]
    topk_rows = [r for r in rows if r.get("metric_kind") == "topk_preservation"]
    margin_rows = [r for r in rows if r.get("metric_kind") == "margin_change"]

    # Aggregate by (condition, modality_block)
    from collections import defaultdict
    agg_block: dict = defaultdict(lambda: {"qk_mse": [], "qk_corr": []})
    for r in block_rows:
        key = (r["condition"], r["modality_block"])
        agg_block[key]["qk_mse"].append(r["qk_mse"])
        agg_block[key]["qk_corr"].append(r["qk_corr"])

    agg_topk: dict = defaultdict(lambda: {"top1_pres": [], "top10_ovr": []})
    for r in topk_rows:
        key = (r["condition"], r["modality_block"])
        if r.get("top1_pres") is not None and not (isinstance(r.get("top1_pres"), float)
                                                   and math.isnan(r["top1_pres"])):
            agg_topk[key]["top1_pres"].append(r["top1_pres"])
        if r.get("top10_ovr") is not None and not (isinstance(r.get("top10_ovr"), float)
                                                    and math.isnan(r["top10_ovr"])):
            agg_topk[key]["top10_ovr"].append(r["top10_ovr"])

    lines = ["# Exp T-mini QK distortion diagnostic", "",
             f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_", "",
             f"Items: {len(margin_rows) // max(1, (len(args.conditions) - 1))} (n_per_modality_q={args.n_queries_per_modality})", ""]

    lines.append("## Margin change per condition")
    lines.append("")
    lines.append("| condition | n_items | mean margin_bf16 | mean margin_quant | mean Δ | flips (BF16✓→quant✗) | flips (BF16✗→quant✓) |")
    lines.append("|---|---|---|---|---|---|---|")
    by_cond_m = defaultdict(list)
    for r in margin_rows:
        by_cond_m[r["condition"]].append(r)
    for c in sorted(by_cond_m):
        rs = by_cond_m[c]
        nb = float(np.mean([r["margin_bf16"] for r in rs]))
        nq = float(np.mean([r["margin_quant"] for r in rs]))
        nd = float(np.mean([r["margin_change"] for r in rs]))
        f_down = sum(1 for r in rs if r["bf16_correct"] and not r["quant_correct"])
        f_up = sum(1 for r in rs if not r["bf16_correct"] and r["quant_correct"])
        lines.append(f"| {c} | {len(rs)} | {nb:.3f} | {nq:.3f} | {nd:+.3f} | {f_down} | {f_up} |")

    lines.append("")
    lines.append("## QK-MSE and QK-Corr per (condition, modality_block)")
    lines.append("(modality_block = QueryModality + KeyModality; e.g. TT = text query, text key)")
    lines.append("")
    lines.append("| condition | modality_block | mean qk_mse | mean qk_corr |")
    lines.append("|---|---|---|---|")
    for (cond, block), agg in sorted(agg_block.items()):
        mse = float(np.mean(agg["qk_mse"])) if agg["qk_mse"] else float("nan")
        corr = float(np.mean(agg["qk_corr"])) if agg["qk_corr"] else float("nan")
        lines.append(f"| {cond} | {block} | {mse:.4f} | {corr:.3f} |")

    lines.append("")
    lines.append("## Top-1 / Top-10 key preservation per (condition, query_modality)")
    lines.append("")
    lines.append("| condition | query_modality | mean top1_pres | mean top10_ovr |")
    lines.append("|---|---|---|---|")
    for (cond, block), agg in sorted(agg_topk.items()):
        t1 = float(np.mean(agg["top1_pres"])) if agg["top1_pres"] else float("nan")
        t10 = float(np.mean(agg["top10_ovr"])) if agg["top10_ovr"] else float("nan")
        lines.append(f"| {cond} | {block} | {t1:.3f} | {t10:.3f} |")

    args.out_summary.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out_summary}")
    print(f"wrote {args.out_jsonl}")


if __name__ == "__main__":
    main()
