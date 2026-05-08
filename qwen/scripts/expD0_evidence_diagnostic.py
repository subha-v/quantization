"""Experiment D0: Evidence-window diagnostic for Qwen2.5-VL on LongVideoBench.

For each of 200 eval items, runs 8 BF16 forwards and saves one JSONL row per
item:

  D0.1  Full-64 BF16 (eager + EvidenceWindowAttentionHook)
        -> evidence_attn_all/mid/maxhead, top1_window, top2_windows,
           visual_mass_total, full64 pred + margin
  D0.2  Uniform-16 BF16
  D0.3  Top-1-window-only BF16  (frames in the top-attended 8-frame window)
  D0.4  Top-2-windows-only BF16 (frames in the top-2 attended windows)
  D0.5  Top-1-window-removed BF16 (drop the top-attended window's 8 frames)
  D0.6a/b/c  Random-window-removed BF16 x 3 seeds

Frame manipulation strategy: v1 frame removal (simpler — pass the modified
frame subset directly to qwen_vl_utils via format_mcq_messages_with_frames).
Sequence length and temporal positions change; flagged in JSONL with
mode="frame_removal_v1".

Output: qwen/results/expD0_evidence_diagnostic.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache

from data_longvideobench import (
    DEFAULT_SPLIT_FILE,
    LVBItem,
    answer_token_ids,
    filter_items,
    format_mcq_messages,
    format_mcq_messages_with_frames,
    load_all_items,
    load_split,
)
from fake_quant_kv_cache import _find_decoder_layers
from frame_manip import (
    all_indices_except,
    decode_uniform_frames,
    select_frame_subset,
    window_indices,
    windows_indices,
)
from visual_tokens import build_window_token_ranges, find_visual_token_span


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

N_FRAMES_DEFAULT = 64
N_WINDOWS_DEFAULT = 8


# ===================================================================
# EvidenceWindowAttentionHook — captures last-query attention per (L, h)
# ===================================================================


class EvidenceWindowAttentionHook:
    """Wraps each Qwen2.5-VL decoder layer's self_attn.forward to capture
    the last-query attention row (`attn_weights[:1, :, -1, :]`) at each layer,
    pooled from H_q query heads to H_kv KV-heads via mean over the GQA group.

    Stores `signals[layer_idx]` as a CPU FP32 tensor of shape [H_kv, K] where
    K is the total key length (== prefill seq_len). Memory ~ 28 layers x 4 heads
    x ~3500 keys x 4 bytes = ~1.5 MB per item — fine.
    """

    def __init__(self, model: torch.nn.Module, num_kv_heads: int = 4, num_kv_groups: int = 7):
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_kv_groups
        self.signals: dict[int, torch.Tensor] = {}
        self._restores: list[tuple[torch.nn.Module, callable]] = []
        layers = _find_decoder_layers(model)
        for layer_idx, attn in layers:
            original_forward = attn.forward

            def make_wrapped(orig, lidx, recorder):
                def wrapped(*args, **kwargs):
                    kwargs["output_attentions"] = True
                    try:
                        result = orig(*args, **kwargs)
                    except TypeError:
                        kwargs.pop("output_attentions", None)
                        return orig(*args, **kwargs)
                    attn_weights = None
                    if isinstance(result, tuple):
                        for item in result:
                            if isinstance(item, torch.Tensor) and item.dim() == 4:
                                attn_weights = item
                                break
                    if attn_weights is not None:
                        recorder._process(lidx, attn_weights)
                    return result

                return wrapped

            attn.forward = make_wrapped(original_forward, layer_idx, self)
            self._restores.append((attn, original_forward))

    def _process(self, layer_idx: int, attn_weights: torch.Tensor) -> None:
        # attn_weights: [B, H_q, Q, K] (eager)
        with torch.no_grad():
            B, Hq, Q, K = attn_weights.shape
            Hkv = self.num_kv_heads
            groups = max(1, Hq // Hkv if Hkv > 0 else 1)
            usable_Hq = Hkv * groups
            if usable_Hq != Hq:
                attn_weights = attn_weights[:, :usable_Hq, :, :]
            # Take last query position only, then pool Q-heads -> KV-heads by averaging.
            # Use reshape (not view) — the -1 indexing yields a non-contiguous tensor.
            last_q = attn_weights[:1, :, -1, :].contiguous()  # [1, Hq, K]
            last = last_q.reshape(1, Hkv, groups, K).mean(dim=2)  # [1, Hkv, K]
            self.signals[layer_idx] = last[0].float().cpu()  # [Hkv, K]
        del attn_weights

    def uninstall(self) -> None:
        for module, original in self._restores:
            module.forward = original
        self._restores = []


# ===================================================================
# Per-window mass pooling
# ===================================================================


def _per_layer_head_window_mass(
    signals: dict[int, torch.Tensor], window_token_ranges: list[tuple[int, int]]
) -> torch.Tensor:
    """signals[L]: [Hkv, K_total] -> returns [num_layers, Hkv, n_windows] raw mass."""
    layer_ids = sorted(signals.keys())
    L = len(layer_ids)
    Hkv = signals[layer_ids[0]].shape[0]
    nW = len(window_token_ranges)
    out = torch.zeros(L, Hkv, nW, dtype=torch.float32)
    for li, l in enumerate(layer_ids):
        row = signals[l]  # [Hkv, K]
        for k, (a, b) in enumerate(window_token_ranges):
            out[li, :, k] = row[:, a:b].sum(dim=-1)
    return out


def pool_evidence(
    m_lhk: torch.Tensor,
    layer_subset: Optional[list[int]] = None,
) -> tuple[torch.Tensor, float]:
    """Raw-visual-mass pooling.

    m_lhk : [L, Hkv, nW]
    layer_subset : optional subset of layer indices (within m_lhk's L axis)
    Returns:
      evidence_attn : [nW] normalized over windows (sums to 1)
      visual_mass_total : mean over (L, h) of total visual mass on that head
    """
    m = m_lhk if layer_subset is None else m_lhk[torch.tensor(layer_subset, dtype=torch.long)]
    raw_per_window = m.sum(dim=(0, 1))  # [nW]
    total = raw_per_window.sum().clamp_min(1e-12)
    evidence_attn = (raw_per_window / total).cpu()
    L_eff, Hkv, _ = m.shape
    per_lh_total_mass = m.sum(dim=-1)  # [L, Hkv]
    visual_mass_total = float(per_lh_total_mass.mean().item())
    return evidence_attn, visual_mass_total


def pool_maxhead(m_lhk: torch.Tensor) -> tuple[torch.Tensor, int, int, float]:
    """Per-(L, h) normalize over windows, then pick the (L, h) with highest top-1 mass.

    Returns:
      evidence_attn : [nW] normalized window distribution of the winning head
      maxhead_layer : int
      maxhead_kv_head : int
      maxhead_top_mass : float (the top-1 window mass of that head, in [0, 1])
    """
    L, Hkv, nW = m_lhk.shape
    sums = m_lhk.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # [L, Hkv, 1]
    norm = m_lhk / sums  # [L, Hkv, nW] each row sums to 1
    top_mass = norm.max(dim=-1).values  # [L, Hkv]
    # argmax over flattened (L, Hkv)
    flat_idx = int(top_mass.argmax().item())
    l_star = flat_idx // Hkv
    h_star = flat_idx % Hkv
    evidence_attn = norm[l_star, h_star].cpu()
    return evidence_attn, l_star, h_star, float(top_mass[l_star, h_star].item())


def evidence_width_90(evidence_attn: torch.Tensor) -> int:
    """Number of windows needed (sorted desc) to cover 90% cumulative mass."""
    sorted_desc, _ = torch.sort(evidence_attn, descending=True)
    csum = torch.cumsum(sorted_desc, dim=0)
    return int((csum >= 0.9).nonzero()[0, 0].item() + 1) if (csum >= 0.9).any() else len(evidence_attn)


# ===================================================================
# Inference helpers
# ===================================================================


def _option_logprobs_and_pred(out, processor, n_options: int) -> tuple[list[float], int]:
    answer_ids = answer_token_ids(processor, n=n_options)
    first_logits = out.scores[0]
    logp = torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist()
    pred = int(max(range(n_options), key=lambda i: logp[i]))
    return logp, pred


def _answer_margin(logp: list[float], correct: int) -> float:
    """margin = logp[correct] - max(logp[other])."""
    if not logp:
        return float("nan")
    others = [v for i, v in enumerate(logp) if i != correct]
    if not others:
        return float("nan")
    return float(logp[correct] - max(others))


@torch.no_grad()
def _forward_messages(model, processor, messages: list[dict], n_options: int,
                      use_attn_hook: bool = False, num_kv_heads: int = 4):
    """Run a single BF16 forward; return (logp, pred, margin_input_dict, hook_signals).

    margin_input_dict is None unless caller wants the raw inputs back.
    """
    from qwen_vl_utils import process_vision_info  # type: ignore

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    cache = DynamicCache()
    hook = None
    if use_attn_hook:
        hook = EvidenceWindowAttentionHook(model, num_kv_heads=num_kv_heads)
    try:
        out = model.generate(
            **inputs, past_key_values=cache, max_new_tokens=1, do_sample=False,
            return_dict_in_generate=True, output_scores=True, use_cache=True,
        )
    finally:
        if hook is not None:
            hook.uninstall()
    logp, pred = _option_logprobs_and_pred(out, processor, n_options)
    signals = dict(hook.signals) if hook is not None else None
    return logp, pred, inputs, signals


# ===================================================================
# Per-item D0 pipeline
# ===================================================================


@torch.no_grad()
def diagnose_item_d0(
    model, processor, item: LVBItem, n_frames: int, n_windows: int,
    num_kv_heads: int, random_seeds: list[int],
) -> dict:
    """Run the 8 D0 conditions for a single item; return one packed JSONL row."""
    n_options = len(item.candidates)
    correct = item.correct_choice

    # ---- D0.1 Full-64 BF16 + attention hook ----
    msgs_full = format_mcq_messages(item, n_frames=n_frames)
    logp_full, pred_full, inputs_full, signals = _forward_messages(
        model, processor, msgs_full, n_options, use_attn_hook=True, num_kv_heads=num_kv_heads
    )
    margin_full = _answer_margin(logp_full, correct)

    # Visual-token span + per-window mass
    v_start, v_end = find_visual_token_span(inputs_full["input_ids"], processor)
    window_ranges = build_window_token_ranges(v_start, v_end, n_windows=n_windows)
    m_lhk = _per_layer_head_window_mass(signals, window_ranges)  # [L, Hkv, nW]
    L_total = m_lhk.shape[0]

    evidence_attn_all, vmass_all = pool_evidence(m_lhk, layer_subset=None)
    # Mid layers: 8..20 inclusive (13 layers). Clamp to model's actual layer count.
    mid_layers = [li for li in range(8, 21) if li < L_total]
    evidence_attn_mid, vmass_mid = pool_evidence(m_lhk, layer_subset=mid_layers)
    evidence_attn_maxhead, mh_layer, mh_head, mh_topmass = pool_maxhead(m_lhk)

    top1_all = int(evidence_attn_all.argmax().item())
    top2_all = sorted([int(i) for i in torch.topk(evidence_attn_all, 2).indices.tolist()])
    width_all = evidence_width_90(evidence_attn_all)

    top1_mid = int(evidence_attn_mid.argmax().item())
    top2_mid = sorted([int(i) for i in torch.topk(evidence_attn_mid, 2).indices.tolist()])
    width_mid = evidence_width_90(evidence_attn_mid)

    top1_maxhead = int(evidence_attn_maxhead.argmax().item())

    del inputs_full, signals, m_lhk
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- D0.2 Uniform-16 BF16 ----
    msgs_u16 = format_mcq_messages(item, n_frames=16)
    logp_u16, pred_u16, _, _ = _forward_messages(
        model, processor, msgs_u16, n_options, use_attn_hook=False
    )
    margin_u16 = _answer_margin(logp_u16, correct)

    # ---- For D0.3-D0.6 we need a precomputed frame list ----
    # Decode 64 frames once and reuse.
    try:
        frames_64 = decode_uniform_frames(item.video_path, n_frames=n_frames)
    except Exception as e:
        # If decode fails, return what we have with NaNs in restricted-evidence slots.
        return _pack_partial_row(item, n_frames, n_windows, v_start, v_end,
                                 evidence_attn_all, evidence_attn_mid, evidence_attn_maxhead,
                                 vmass_all, vmass_mid, mh_topmass,
                                 top1_all, top2_all, width_all,
                                 top1_mid, top2_mid, width_mid,
                                 top1_maxhead, mh_layer, mh_head,
                                 logp_full, pred_full, margin_full, n_options,
                                 logp_u16, pred_u16, margin_u16,
                                 decode_error=str(e))

    fpw = max(1, n_frames // n_windows)  # frames per window

    # ---- D0.3 Top-1-window-only BF16 ----
    keep1 = window_indices(top1_all, frames_per_window=fpw)
    sub1 = select_frame_subset(frames_64, keep1)
    logp_t1, pred_t1, _, _ = _forward_messages(
        model, processor, format_mcq_messages_with_frames(item, sub1), n_options
    )
    margin_t1 = _answer_margin(logp_t1, correct)

    # ---- D0.4 Top-2-windows-only BF16 ----
    keep2 = windows_indices(top2_all, frames_per_window=fpw)
    sub2 = select_frame_subset(frames_64, keep2)
    logp_t2, pred_t2, _, _ = _forward_messages(
        model, processor, format_mcq_messages_with_frames(item, sub2), n_options
    )
    margin_t2 = _answer_margin(logp_t2, correct)

    # ---- D0.5 Top-1-window-removed BF16 ----
    keep_remove1 = all_indices_except(top1_all, n_windows=n_windows, frames_per_window=fpw)
    sub_r = select_frame_subset(frames_64, keep_remove1)
    logp_r1, pred_r1, _, _ = _forward_messages(
        model, processor, format_mcq_messages_with_frames(item, sub_r), n_options
    )
    margin_r1 = _answer_margin(logp_r1, correct)

    # ---- D0.6 Random-window-removed BF16 x len(random_seeds) ----
    pred_rand: list[int] = []
    margin_rand: list[float] = []
    logp_rand: list[list[float]] = []
    rand_window_ids: list[int] = []
    for seed in random_seeds:
        rng = torch.Generator()
        rng.manual_seed(int(seed) * 1000003 + zlib.crc32(item.id.encode()) % 10_000)
        # Pick a random window != top1_all so it's a fair "removal of a non-evidence window".
        # If top1 is the only window, fall back to any.
        candidates = [w for w in range(n_windows) if w != top1_all] or list(range(n_windows))
        wsel = candidates[int(torch.randint(0, len(candidates), (1,), generator=rng).item())]
        rand_window_ids.append(int(wsel))
        keep_r = all_indices_except(wsel, n_windows=n_windows, frames_per_window=fpw)
        sub_rr = select_frame_subset(frames_64, keep_r)
        lp, pr, _, _ = _forward_messages(
            model, processor, format_mcq_messages_with_frames(item, sub_rr), n_options
        )
        pred_rand.append(int(pr))
        logp_rand.append(lp)
        margin_rand.append(_answer_margin(lp, correct))

    # Causal effects
    twce = float(margin_full - margin_r1)
    rwce = float(margin_full - (sum(margin_rand) / len(margin_rand))) if margin_rand else float("nan")
    egap = float(twce - rwce) if not math.isnan(rwce) else float("nan")

    # ---- Pack JSONL row ----
    row = {
        "phase": "D0",
        "item_id": item.id,
        "split": "eval",
        "duration_bucket": item.duration_bucket,
        "duration_seconds": item.duration_seconds,
        "n_options": n_options,
        "correct_choice": correct,
        "n_frames": n_frames,
        "n_windows": n_windows,
        "mode": "frame_removal_v1",
        "visual_token_start": int(v_start),
        "visual_token_end": int(v_end),
        # Primary selector
        "evidence_attn_all": [float(x) for x in evidence_attn_all.tolist()],
        "visual_mass_total_all": float(vmass_all),
        "top1_window_all": top1_all,
        "top2_windows_all": top2_all,
        "evidence_width_90_all": int(width_all),
        # Mid-layer diagnostic
        "evidence_attn_mid": [float(x) for x in evidence_attn_mid.tolist()],
        "visual_mass_total_mid": float(vmass_mid),
        "top1_window_mid": top1_mid,
        "top2_windows_mid": top2_mid,
        "evidence_width_90_mid": int(width_mid),
        # Max-head diagnostic
        "evidence_attn_maxhead": [float(x) for x in evidence_attn_maxhead.tolist()],
        "top1_window_maxhead": top1_maxhead,
        "maxhead_layer": int(mh_layer),
        "maxhead_kv_head": int(mh_head),
        "maxhead_top_mass": float(mh_topmass),
        # Predictions per condition
        "pred_full64": int(pred_full),
        "pred_uniform16": int(pred_u16),
        "pred_top1_only": int(pred_t1),
        "pred_top2_only": int(pred_t2),
        "pred_top1_removed": int(pred_r1),
        "pred_random_removed": pred_rand,
        "random_removed_window_ids": rand_window_ids,
        # Margins per condition
        "margin_full64": float(margin_full),
        "margin_uniform16": float(margin_u16),
        "margin_top1_only": float(margin_t1),
        "margin_top2_only": float(margin_t2),
        "margin_top1_removed": float(margin_r1),
        "margin_random_removed": [float(m) for m in margin_rand],
        # Logprobs (small, useful for re-analysis)
        "option_logprobs_full64": [float(x) for x in logp_full],
        "option_logprobs_uniform16": [float(x) for x in logp_u16],
        "option_logprobs_top1_only": [float(x) for x in logp_t1],
        "option_logprobs_top2_only": [float(x) for x in logp_t2],
        "option_logprobs_top1_removed": [float(x) for x in logp_r1],
        "option_logprobs_random_removed": [[float(x) for x in lp] for lp in logp_rand],
        # Causal effects
        "top_window_causal_effect": twce,
        "random_window_causal_effect_mean": rwce,
        "evidence_causal_gap": egap,
    }
    return row


def _pack_partial_row(item, n_frames, n_windows, v_start, v_end,
                     evidence_attn_all, evidence_attn_mid, evidence_attn_maxhead,
                     vmass_all, vmass_mid, mh_topmass,
                     top1_all, top2_all, width_all,
                     top1_mid, top2_mid, width_mid,
                     top1_maxhead, mh_layer, mh_head,
                     logp_full, pred_full, margin_full, n_options,
                     logp_u16, pred_u16, margin_u16,
                     decode_error: str) -> dict:
    return {
        "phase": "D0",
        "item_id": item.id,
        "split": "eval",
        "duration_bucket": item.duration_bucket,
        "duration_seconds": item.duration_seconds,
        "n_options": n_options,
        "correct_choice": item.correct_choice,
        "n_frames": n_frames,
        "n_windows": n_windows,
        "mode": "frame_removal_v1",
        "visual_token_start": int(v_start),
        "visual_token_end": int(v_end),
        "evidence_attn_all": [float(x) for x in evidence_attn_all.tolist()],
        "visual_mass_total_all": float(vmass_all),
        "top1_window_all": top1_all,
        "top2_windows_all": top2_all,
        "evidence_width_90_all": int(width_all),
        "evidence_attn_mid": [float(x) for x in evidence_attn_mid.tolist()],
        "visual_mass_total_mid": float(vmass_mid),
        "top1_window_mid": top1_mid,
        "top2_windows_mid": top2_mid,
        "evidence_width_90_mid": int(width_mid),
        "evidence_attn_maxhead": [float(x) for x in evidence_attn_maxhead.tolist()],
        "top1_window_maxhead": top1_maxhead,
        "maxhead_layer": int(mh_layer),
        "maxhead_kv_head": int(mh_head),
        "maxhead_top_mass": float(mh_topmass),
        "pred_full64": int(pred_full),
        "pred_uniform16": int(pred_u16),
        "margin_full64": float(margin_full),
        "margin_uniform16": float(margin_u16),
        "option_logprobs_full64": [float(x) for x in logp_full],
        "option_logprobs_uniform16": [float(x) for x in logp_u16],
        "decode_error": decode_error,
        "partial": True,
    }


# ===================================================================
# Driver
# ===================================================================


def _append_progress(progress_log: Path, line: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_log, "a") as f:
        f.write(f"[{ts}] {line}\n")
        f.flush()


def run_d0(
    model,
    processor,
    items: list[LVBItem],
    n_frames: int,
    n_windows: int,
    num_kv_heads: int,
    random_seeds: list[int],
    out_jsonl: Path,
    progress_every: int = 5,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_log = out_jsonl.with_name(out_jsonl.stem + ".progress.log")
    _append_progress(progress_log, f"START D0 n_items={len(items)} frames={n_frames} windows={n_windows}")
    t_start = time.perf_counter()
    n_done = 0
    n_failed = 0
    with open(out_jsonl, "a") as f:
        for i, it in enumerate(items):
            try:
                row = diagnose_item_d0(
                    model, processor, it,
                    n_frames=n_frames, n_windows=n_windows,
                    num_kv_heads=num_kv_heads, random_seeds=random_seeds,
                )
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_done += 1
            except Exception as e:
                n_failed += 1
                _append_progress(progress_log, f"WARN item={it.id} skipped: {type(e).__name__}: {e}")
                continue
            if (i + 1) % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            done = i + 1
            if done % progress_every == 0 or done == len(items):
                elapsed = time.perf_counter() - t_start
                rate = elapsed / max(1, n_done)
                eta = max(0.0, rate * (len(items) - done))
                _append_progress(
                    progress_log,
                    f"D0 {done}/{len(items)} ok={n_done} failed={n_failed} "
                    f"elapsed={timedelta(seconds=int(elapsed))} ETA={timedelta(seconds=int(eta))}"
                )
    _append_progress(progress_log, f"DONE D0 ok={n_done} failed={n_failed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=N_FRAMES_DEFAULT)
    ap.add_argument("--windows", type=int, default=N_WINDOWS_DEFAULT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--out", type=Path,
                    default=RESULTS_DIR / "expD0_evidence_diagnostic.jsonl")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--progress_every", type=int, default=5)
    args = ap.parse_args()

    items = load_all_items()
    split = load_split(args.split_file)
    eval_items = filter_items(items, split["eval"])
    if args.limit:
        eval_items = eval_items[: args.limit]
    print(f"[expD0] eval_items={len(eval_items)} frames={args.frames} windows={args.windows}",
          flush=True)

    if not args.append and args.out.exists():
        args.out.unlink()

    from run_inference import load_model
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="eager", device_map="auto")
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    print(f"[expD0] model loaded; num_kv_heads={num_kv_heads}", flush=True)

    run_d0(
        model, processor, eval_items,
        n_frames=args.frames, n_windows=args.windows,
        num_kv_heads=num_kv_heads, random_seeds=args.seeds,
        out_jsonl=args.out, progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
