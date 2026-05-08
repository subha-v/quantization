"""Experiment D1: Cross-modal K/V quantization on Qwen2.5-VL × LongVideoBench.

V is fixed at INT4 everywhere (Exp C: V at INT4 is essentially free if K is BF16).
K is varied: text/visual/protected windows masked between BF16 and INT4 via
the BitController V3K mode.

Per item, run the conditions whose top-window/top2-windows D1 needs from D0:
  D1.3  Text-K BF16, Visual-K INT4, V INT4
  D1.4  Text-K INT4, Visual-K BF16, V INT4
  D1.5a Text-K BF16, top-1 visual-window K BF16, rest visual K INT4, V INT4
  D1.5b Text-K BF16, top-2 visual-window K BF16, rest visual K INT4, V INT4
  D1.6a Text-K BF16, random-1 visual-window K BF16 (×3 seeds)
  D1.6b Text-K BF16, random-2 visual-windows K BF16 (×3 seeds)
  D1.7a Text-K BF16, uniform-1 visual-window K BF16 (window 4 = mid)
  D1.7b Text-K BF16, uniform-2 visual-windows K BF16 (windows 0, 4)

D1.0 (BF16 ceiling = A1), D1.1 (INT4 K/V floor = A5), D1.2 (BF16-K + INT4-V
upper bound = C2.1) reuse existing rollouts in expA_rollouts JSONL — not re-run.

Output: qwen/results/expD1_crossmodal_kv.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import torch

from data_longvideobench import (
    DEFAULT_SPLIT_FILE,
    LVBItem,
    answer_token_ids,
    filter_items,
    format_mcq_messages,
    load_all_items,
    load_split,
)
from fake_quant_kv_cache import BitController, FakeQuantKVCache
from visual_tokens import build_window_token_ranges, find_visual_token_span


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

N_FRAMES_DEFAULT = 64
N_WINDOWS_DEFAULT = 8


# ===================================================================
# Mask construction
# ===================================================================


def build_mask(
    seq_len: int,
    v_start: int,
    v_end: int,
    *,
    text_protect: bool,
    visual_default_protect: bool,
    visual_protect_windows: list[int],
    window_token_ranges: list[tuple[int, int]],
) -> torch.Tensor:
    """[seq_len] bool mask: True = K=BF16, False = K=INT4."""
    mask = torch.zeros(seq_len, dtype=torch.bool)
    if text_protect:
        if v_start > 0:
            mask[:v_start] = True
        if v_end < seq_len:
            mask[v_end:] = True
    if visual_default_protect:
        mask[v_start:v_end] = True
    for w in visual_protect_windows:
        a, b = window_token_ranges[w]
        mask[a:b] = True
    return mask


def avg_kv_bits_for_mask(
    mask: torch.Tensor, k_hi: int, k_lo: int, v_bits: int
) -> float:
    """Effective avg KV bits given the K-only mask."""
    n = mask.numel()
    n_protect = int(mask.sum().item())
    n_lo = n - n_protect
    bits_K = (k_hi * n_protect + k_lo * n_lo) / max(1, n)
    return float((bits_K + v_bits) / 2.0)


# ===================================================================
# Condition spec
# ===================================================================


def conditions_for_item(top1: int, top2: list[int],
                        top1_mh: int, top2_mh: list[int],
                        n_windows: int, seeds: list[int]):
    """Yield (name, build_kwargs) for each D1 condition this item runs.

    build_kwargs has keys: text_protect, visual_default_protect,
    visual_protect_windows, k_visual_policy (string label), seed (or None).

    `top1` / `top2` are from D0's `evidence_attn_all` (raw-mass-pooled across all
    L, h) — the primary selector. `top1_mh` / `top2_mh` are from D0's
    `evidence_attn_maxhead` — the per-(L, h)-normalized variant that picks the
    sharpest single head. Both are tested in parallel because in D0 the all-
    pooled selector exhibited an attention-sink pathology (top1=window-0 in
    97.5% of items), so D1.5a-mh is the cleaner test of "does attention pick
    causal evidence windows."
    """
    middle = n_windows // 2  # uniform-1 fallback
    uniform2 = sorted({0, middle})

    yield ("D1_3_TextBF16_VisInt4_VInt4", dict(
        text_protect=True, visual_default_protect=False, visual_protect_windows=[],
        k_visual_policy="all_visual_INT4", seed=None,
    ))
    yield ("D1_4_TextInt4_VisBF16_VInt4", dict(
        text_protect=False, visual_default_protect=True, visual_protect_windows=[],
        k_visual_policy="all_visual_BF16", seed=None,
    ))
    yield ("D1_5a_TextBF16_Top1VisBF16_VInt4", dict(
        text_protect=True, visual_default_protect=False,
        visual_protect_windows=[int(top1)],
        k_visual_policy=f"top1_all_BF16_w{int(top1)}", seed=None,
    ))
    yield ("D1_5b_TextBF16_Top2VisBF16_VInt4", dict(
        text_protect=True, visual_default_protect=False,
        visual_protect_windows=[int(w) for w in top2],
        k_visual_policy=f"top2_all_BF16_{sorted(int(w) for w in top2)}", seed=None,
    ))
    yield ("D1_5a_mh_TextBF16_Top1MaxheadVisBF16_VInt4", dict(
        text_protect=True, visual_default_protect=False,
        visual_protect_windows=[int(top1_mh)],
        k_visual_policy=f"top1_maxhead_BF16_w{int(top1_mh)}", seed=None,
    ))
    yield ("D1_5b_mh_TextBF16_Top2MaxheadVisBF16_VInt4", dict(
        text_protect=True, visual_default_protect=False,
        visual_protect_windows=[int(w) for w in top2_mh],
        k_visual_policy=f"top2_maxhead_BF16_{sorted(int(w) for w in top2_mh)}", seed=None,
    ))
    for s in seeds:
        yield (f"D1_6a_TextBF16_Rand1VisBF16_VInt4_seed{s}", dict(
            text_protect=True, visual_default_protect=False,
            visual_protect_windows=_random_windows(n_windows, k=1, seed=s, exclude_top=int(top1)),
            k_visual_policy=f"rand1_BF16_seed{s}", seed=s,
        ))
        yield (f"D1_6b_TextBF16_Rand2VisBF16_VInt4_seed{s}", dict(
            text_protect=True, visual_default_protect=False,
            visual_protect_windows=_random_windows(n_windows, k=2, seed=s, exclude_top=int(top1)),
            k_visual_policy=f"rand2_BF16_seed{s}", seed=s,
        ))
    yield ("D1_7a_TextBF16_UniformMidVisBF16_VInt4", dict(
        text_protect=True, visual_default_protect=False,
        visual_protect_windows=[middle],
        k_visual_policy=f"uniform1_window{middle}_BF16", seed=None,
    ))
    yield ("D1_7b_TextBF16_Uniform2VisBF16_VInt4", dict(
        text_protect=True, visual_default_protect=False,
        visual_protect_windows=uniform2,
        k_visual_policy=f"uniform2_windows{uniform2}_BF16", seed=None,
    ))


def _random_windows(n_windows: int, k: int, seed: int, exclude_top: int) -> list[int]:
    """Pick k distinct windows from [0..n_windows) excluding exclude_top."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    candidates = [w for w in range(n_windows) if w != exclude_top]
    if len(candidates) < k:
        candidates = list(range(n_windows))
    perm = torch.randperm(len(candidates), generator=g).tolist()
    return sorted(candidates[i] for i in perm[:k])


# ===================================================================
# Per-item D1 inference
# ===================================================================


def _option_logprobs_and_pred(out, processor, n_options: int) -> tuple[list[float], int]:
    answer_ids = answer_token_ids(processor, n=n_options)
    first_logits = out.scores[0]
    logp = torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist()
    pred = int(max(range(n_options), key=lambda i: logp[i]))
    return logp, pred


def _answer_margin(logp: list[float], correct: int) -> float:
    if not logp:
        return float("nan")
    others = [v for i, v in enumerate(logp) if i != correct]
    if not others:
        return float("nan")
    return float(logp[correct] - max(others))


@torch.no_grad()
def run_item_d1(
    model, processor, item: LVBItem, d0_row: dict,
    n_frames: int, n_windows: int, num_layers: int, num_kv_heads: int,
    seeds: list[int],
) -> list[dict]:
    """Run all D1 conditions for this item; return list of JSONL rows."""
    from qwen_vl_utils import process_vision_info  # type: ignore

    n_options = len(item.candidates)
    correct = item.correct_choice

    msgs = format_mcq_messages(item, n_frames=n_frames)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    seq_len = inputs["input_ids"].shape[1]

    v_start, v_end = find_visual_token_span(inputs["input_ids"], processor)
    window_ranges = build_window_token_ranges(v_start, v_end, n_windows=n_windows)

    top1 = int(d0_row["top1_window_all"])
    top2 = list(d0_row["top2_windows_all"])
    # Maxhead-derived windows: top1_window_maxhead is in D0; top2 isn't saved
    # explicitly but we can derive it from evidence_attn_maxhead (saved [8]).
    top1_mh = int(d0_row["top1_window_maxhead"])
    ev_mh = d0_row.get("evidence_attn_maxhead", [])
    if isinstance(ev_mh, list) and len(ev_mh) == n_windows:
        sorted_idx = sorted(range(n_windows), key=lambda i: -ev_mh[i])
        top2_mh = sorted(sorted_idx[:2])
    else:
        # Fallback: top1_mh + middle as a stand-in
        top2_mh = sorted({top1_mh, n_windows // 2})
    bf16_pred = int(d0_row["pred_full64"])
    bf16_correct = bool(bf16_pred == correct)

    rows: list[dict] = []
    K_HI, K_LO, V_BITS = 16, 4, 4

    for name, kw in conditions_for_item(top1, top2, top1_mh, top2_mh, n_windows, seeds):
        mask = build_mask(
            seq_len=seq_len, v_start=v_start, v_end=v_end,
            text_protect=kw["text_protect"],
            visual_default_protect=kw["visual_default_protect"],
            visual_protect_windows=kw["visual_protect_windows"],
            window_token_ranges=window_ranges,
        )
        # Build controller fresh per condition
        controller = BitController(
            num_layers=num_layers, num_kv_heads=num_kv_heads, mode="V3K",
            default_k_bits=K_LO, default_v_bits=V_BITS,
        )
        controller.set_global(k_bits=K_LO, v_bits=V_BITS)  # V always INT4
        for L in range(num_layers):
            controller.set_protected_mask(L, mask, hi_bits=K_HI, lo_bits=K_LO)

        cache = FakeQuantKVCache(controller)
        t0 = time.perf_counter()
        out = model.generate(
            **inputs, past_key_values=cache, max_new_tokens=1, do_sample=False,
            return_dict_in_generate=True, output_scores=True, use_cache=True,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logp, pred = _option_logprobs_and_pred(out, processor, n_options)
        margin = _answer_margin(logp, correct)
        avg_bits = avg_kv_bits_for_mask(mask, k_hi=K_HI, k_lo=K_LO, v_bits=V_BITS)

        # K bit characterization
        text_n = (v_start) + (seq_len - v_end)
        text_bits = K_HI if kw["text_protect"] else K_LO

        rows.append({
            "phase": "D1",
            "item_id": item.id,
            "duration_bucket": item.duration_bucket,
            "duration_seconds": item.duration_seconds,
            "n_options": n_options,
            "correct_choice": correct,
            "n_frames": n_frames,
            "n_windows": n_windows,
            "condition": name,
            "k_text_bits": int(text_bits),
            "k_visual_policy": kw["k_visual_policy"],
            "v_bits": V_BITS,
            "avg_kv_bits": float(avg_bits),
            "visual_token_start": int(v_start),
            "visual_token_end": int(v_end),
            "seq_len": int(seq_len),
            "visual_protect_windows": list(kw["visual_protect_windows"]),
            "pred_choice": int(pred),
            "is_correct": bool(pred == correct),
            "option_logprobs": [float(x) for x in logp],
            "answer_margin": float(margin),
            "latency_ms": float(latency_ms),
            "seed": kw.get("seed"),
            # D0 join fields (read at runtime so analyze can post-hoc re-stratify)
            "top1_window_all": top1,
            "top2_windows_all": top2,
            "top1_window_maxhead": top1_mh,
            "top2_windows_maxhead": top2_mh,
            "bf16_pred": bf16_pred,
            "bf16_correct": bf16_correct,
        })

        del cache, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


# ===================================================================
# Driver
# ===================================================================


def _load_d0_rows(d0_jsonl: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    if not d0_jsonl.exists():
        raise FileNotFoundError(f"D0 JSONL not found: {d0_jsonl}. Run expD0 first.")
    for line in d0_jsonl.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") != "D0":
            continue
        by_id[row["item_id"]] = row
    return by_id


def _append_progress(progress_log: Path, line: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_log, "a") as f:
        f.write(f"[{ts}] {line}\n")
        f.flush()


def run_d1(
    model, processor, items: list[LVBItem], d0_rows_by_id: dict[str, dict],
    n_frames: int, n_windows: int, num_layers: int, num_kv_heads: int,
    seeds: list[int], out_jsonl: Path, progress_every: int = 5,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_log = out_jsonl.with_name(out_jsonl.stem + ".progress.log")
    _append_progress(progress_log, f"START D1 n_items={len(items)} frames={n_frames}")
    t_start = time.perf_counter()
    n_done = 0
    n_rows_total = 0
    n_failed = 0
    with open(out_jsonl, "a") as f:
        for i, it in enumerate(items):
            d0 = d0_rows_by_id.get(it.id)
            if d0 is None:
                n_failed += 1
                _append_progress(progress_log, f"WARN item={it.id} no D0 row, skipped")
                continue
            try:
                rows = run_item_d1(
                    model, processor, it, d0,
                    n_frames=n_frames, n_windows=n_windows,
                    num_layers=num_layers, num_kv_heads=num_kv_heads,
                    seeds=seeds,
                )
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                n_done += 1
                n_rows_total += len(rows)
            except Exception as e:
                n_failed += 1
                _append_progress(progress_log, f"WARN item={it.id} failed: {type(e).__name__}: {e}")
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
                    f"D1 {done}/{len(items)} ok={n_done} rows={n_rows_total} failed={n_failed} "
                    f"elapsed={timedelta(seconds=int(elapsed))} ETA={timedelta(seconds=int(eta))}"
                )
    _append_progress(progress_log, f"DONE D1 ok={n_done} rows={n_rows_total} failed={n_failed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=N_FRAMES_DEFAULT)
    ap.add_argument("--windows", type=int, default=N_WINDOWS_DEFAULT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--d0_jsonl", type=Path,
                    default=RESULTS_DIR / "expD0_evidence_diagnostic.jsonl")
    ap.add_argument("--out", type=Path,
                    default=RESULTS_DIR / "expD1_crossmodal_kv.jsonl")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--progress_every", type=int, default=5)
    args = ap.parse_args()

    items = load_all_items()
    split = load_split(args.split_file)
    eval_items = filter_items(items, split["eval"])
    if args.limit:
        eval_items = eval_items[: args.limit]
    print(f"[expD1] eval_items={len(eval_items)}", flush=True)

    d0_rows = _load_d0_rows(args.d0_jsonl)
    print(f"[expD1] loaded {len(d0_rows)} D0 rows from {args.d0_jsonl}", flush=True)

    if not args.append and args.out.exists():
        args.out.unlink()

    from run_inference import load_model
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    print(f"[expD1] model loaded; num_layers={num_layers} num_kv_heads={num_kv_heads}", flush=True)

    run_d1(
        model, processor, eval_items, d0_rows,
        n_frames=args.frames, n_windows=args.windows,
        num_layers=num_layers, num_kv_heads=num_kv_heads,
        seeds=args.seeds, out_jsonl=args.out,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
