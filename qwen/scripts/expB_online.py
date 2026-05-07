"""
Experiment B online precision-need routing driver.

For each LongVideoBench eval item, runs each routing condition by:
  1. Reading per-item online signals from diagnostic_signals.jsonl (eval rows).
  2. Computing per-(layer, KV-head) score from the chosen method.
  3. Picking the top-k blocks → BF16; rest → INT2 (V2 BitController).
  4. Calling score_item() to get the routed prediction.
  5. Appending per-item row to qwen/results/expB_online_<cond>.jsonl.

Conditions covered (target_avg_bits=4 → k=16 of 112 (layer, KV-head) blocks at BF16):
  B2  Random-V2       (3 seeds; rolled into a single condition with seed in row)
  B4  MEDA-style layer entropy
  B6  StaticEntropy-V2 (low entropy → BF16)
  B7  FlippedEntropy-V2 (high entropy → BF16; symmetry control)
  B8  OnlineResidual-V2
  B9  OnlineNeed-Static-V2 (= percentile(static_low) × percentile(online_residual))
  B10 OnlineNeed-AQ-V2  (= percentile(-aq_topk_mass) × percentile(online_residual))

Reuses A1 (BF16) / A5 (Uniform INT4) / A7 (Uniform INT2) predictions baked into
the diagnostic JSONL — no need to re-run them.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
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
from precision_need_scoring import (
    aggregate_static_risk,
    apply_to_controller,
    avg_bits_from_LH,
    bits_from_mask,
    compute_score,
    item_signals,
    load_diagnostic_jsonl,
    load_static_risk,
    save_static_risk,
    top_k_mask,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"


# ===================================================================
# Per-condition spec
# ===================================================================

CONDITIONS_SPEC = {
    # condition_id -> (method, requires_static, requires_per_item)
    "B2_Random":            ("random",              False, False),
    "B4_MEDA":              ("meda_layer",          True,  False),
    "B6_StaticEntropy":     ("static_low",          True,  False),
    "B7_FlippedEntropy":    ("static_high",         True,  False),
    "B8_OnlineResidual":    ("online_residual",     False, True),
    "B9_OnlineNeed_Static": ("online_need_static",  True,  True),
    "B10_OnlineNeed_AQ":    ("online_need_aq",      False, True),
}


# ===================================================================
# Score one item under a given condition+seed
# ===================================================================

@torch.no_grad()
def score_item_routed(
    model, processor, item: LVBItem, n_frames: int, controller: BitController,
) -> tuple[list[float], int, float]:
    """Single forward pass with the given V2 controller. Returns (logprobs, pred, latency_ms)."""
    from qwen_vl_utils import process_vision_info  # type: ignore
    msgs = format_mcq_messages(item, n_frames=n_frames)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    cache = FakeQuantKVCache(controller)
    t0 = time.perf_counter()
    out = model.generate(
        **inputs, past_key_values=cache, max_new_tokens=1, do_sample=False,
        return_dict_in_generate=True, output_scores=True, use_cache=True,
    )
    dt_ms = (time.perf_counter() - t0) * 1000
    n_options = len(item.candidates)
    answer_ids = answer_token_ids(processor, n=n_options)
    first_logits = out.scores[0]
    logp = torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist()
    pred = int(max(range(n_options), key=lambda i: logp[i]))
    return logp, pred, dt_ms


# ===================================================================
# Driver
# ===================================================================

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(progress_log: Path, msg: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[expB_online] {msg}", flush=True)
    with open(progress_log, "a") as f:
        f.write(f"[{_ts()}] {msg}\n"); f.flush()


def run_condition(
    cond_id: str, method: str, *,
    model, processor,
    eval_items: list[LVBItem],
    diag_records: dict[str, dict],
    static: Optional[dict],
    num_layers: int, num_kv_heads: int, k_bf16_blocks: int,
    seeds: list[int],
    out_jsonl: Path, progress_log: Path,
    n_frames: int,
    progress_every: int = 5,
) -> None:
    """Run one condition over all eval items. For random methods, runs all seeds in sequence."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    requires_per_item = CONDITIONS_SPEC[cond_id][2]
    seeds_to_use = seeds if cond_id == "B2_Random" else [0]

    for seed in seeds_to_use:
        full_cond_label = f"{cond_id}_seed{seed}" if cond_id == "B2_Random" else cond_id
        _log(progress_log, f"START {full_cond_label} k_bf16_blocks={k_bf16_blocks}/{num_layers*num_kv_heads}")
        t_start = time.perf_counter()
        n_correct, n_done = 0, 0
        with open(out_jsonl, "a") as f:
            for i, it in enumerate(eval_items):
                if it.id not in diag_records:
                    continue
                rec = diag_records[it.id]
                if rec["split"] != "eval":
                    continue
                # Build per-(L, H) score
                if requires_per_item:
                    item_sig = item_signals(rec, num_layers, num_kv_heads)
                else:
                    item_sig = None
                score = compute_score(
                    method=method,
                    num_layers=num_layers, num_kv_heads=num_kv_heads,
                    static=static, item_sig=item_sig, seed=seed,
                )
                mask = top_k_mask(score, k_bf16_blocks)
                bits_LH = bits_from_mask(mask, hi_bits=16, lo_bits=2)
                avg_bits = avg_bits_from_LH(bits_LH)

                # Build V2 controller
                ctrl = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads, mode="V2")
                apply_to_controller(ctrl, bits_LH)

                try:
                    logp, pred, latency_ms = score_item_routed(model, processor, it,
                                                                n_frames=n_frames, controller=ctrl)
                except Exception as e:
                    _log(progress_log, f"WARN item={it.id}: {type(e).__name__}: {e}")
                    continue

                is_correct = (pred == it.correct_choice)
                n_correct += int(is_correct); n_done += 1
                row = {
                    "item_id": it.id,
                    "condition": full_cond_label,
                    "method": method,
                    "seed": seed,
                    "n_frames": n_frames,
                    "duration_bucket": it.duration_bucket,
                    "correct_choice": it.correct_choice,
                    "n_options": len(it.candidates),
                    "pred_choice": pred,
                    "is_correct": is_correct,
                    "option_logprobs": logp,
                    "latency_ms": latency_ms,
                    "avg_kv_bits": avg_bits,
                    "k_bf16_blocks": int(mask.sum().item()),
                    # bake in the references for offline metrics
                    "bf16_pred": rec["bf16_pred"],
                    "uniform_int4_pred": rec["uniform_int4_pred"],
                    "uniform_int2_pred": rec["uniform_int2_pred"],
                }
                f.write(json.dumps(row) + "\n"); f.flush()

                done = i + 1
                if done % progress_every == 0 or done == len(eval_items):
                    elapsed = time.perf_counter() - t_start
                    rate = elapsed / max(1, n_done)
                    eta = max(0.0, rate * (len(eval_items) - done))
                    _log(progress_log,
                         f"{full_cond_label} {done}/{len(eval_items)} acc={n_correct/max(1,n_done):.3f} "
                         f"avg_bits={avg_bits:.2f} elapsed={timedelta(seconds=int(elapsed))} "
                         f"ETA={timedelta(seconds=int(eta))}")
                # Memory hygiene every 5 items
                if (i + 1) % 5 == 0:
                    import gc as _gc
                    _gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        wall = time.perf_counter() - t_start
        _log(progress_log,
             f"DONE {full_cond_label} acc={n_correct}/{n_done}={n_correct/max(1,n_done):.3f} "
             f"wall={timedelta(seconds=int(wall))}")


def summarize(out_jsonl: Path) -> None:
    """Build summary.md with: condition, n, acc + 95% CI, BF16-correct preservation,
    flip recovery vs INT4 / INT2, damage rate, per-bucket acc."""
    import numpy as np
    rows = [json.loads(l) for l in out_jsonl.read_text().splitlines() if l.strip()]
    if not rows:
        return
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    def boot_ci(arr, n_boot=2000, seed=0):
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0: return float("nan"), float("nan"), float("nan")
        rng = np.random.default_rng(seed)
        means = [a[rng.integers(0, a.size, a.size)].mean() for _ in range(n_boot)]
        return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    out = ["# Experiment B Online — Precision-Need Routing\n"]
    out.append("| Condition | n | acc | 95% CI | avg KV bits | BF16-pres | flip-rec INT4 | flip-rec INT2 | damage |")
    out.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for cond, rs in sorted(by_cond.items()):
        accs = [int(r["is_correct"]) for r in rs]
        mean, lo, hi = boot_ci(accs)
        avg_bits = sum(r.get("avg_kv_bits", 16.0) for r in rs) / max(1, len(rs))
        # BF16-correct preservation
        bf16_correct = [r for r in rs if r.get("bf16_pred") == r["correct_choice"]]
        pres = sum(1 for r in bf16_correct if r["is_correct"]) / max(1, len(bf16_correct))
        # Flip recovery vs INT4: BF16 correct AND uniform_int4 wrong
        flip4 = [r for r in rs if r.get("bf16_pred") == r["correct_choice"]
                                  and r.get("uniform_int4_pred") != r["correct_choice"]]
        rec4 = sum(1 for r in flip4 if r["is_correct"]) / max(1, len(flip4))
        # Flip recovery vs INT2
        flip2 = [r for r in rs if r.get("bf16_pred") == r["correct_choice"]
                                  and r.get("uniform_int2_pred") != r["correct_choice"]]
        rec2 = sum(1 for r in flip2 if r["is_correct"]) / max(1, len(flip2))
        # Damage: BF16 correct AND uniform_int4 correct AND method wrong
        clean = [r for r in rs if r.get("bf16_pred") == r["correct_choice"]
                                  and r.get("uniform_int4_pred") == r["correct_choice"]]
        damage = sum(1 for r in clean if not r["is_correct"]) / max(1, len(clean))
        out.append(f"| {cond} | {len(rs)} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] | {avg_bits:.2f} | "
                   f"{pres:.3f} | {rec4:.3f} (n={len(flip4)}) | {rec2:.3f} (n={len(flip2)}) | {damage:.3f} (n={len(clean)}) |")

    out.append("")
    out.append("## Per-duration-bucket accuracy")
    out.append("| Condition | short | mid | long | very_long |")
    out.append("|---|---:|---:|---:|---:|")
    for cond, rs in sorted(by_cond.items()):
        bk = defaultdict(list)
        for r in rs: bk[r["duration_bucket"]].append(int(r["is_correct"]))
        cells = []
        for b in ("short", "mid", "long", "very_long"):
            arr = bk.get(b, [])
            if arr:
                cells.append(f"{sum(arr)/len(arr):.3f} (n={len(arr)})")
            else:
                cells.append("—")
        out.append(f"| {cond} | " + " | ".join(cells) + " |")

    summary_path = out_jsonl.with_name(out_jsonl.stem.replace("rollouts", "summary") + ".md")
    if "rollouts" not in out_jsonl.stem:
        summary_path = out_jsonl.with_name(out_jsonl.stem + "_summary.md")
    summary_path.write_text("\n".join(out) + "\n")
    print(f"[expB_online] summary -> {summary_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--diagnostic", type=Path,
                    default=RESULTS_DIR / "diagnostic_signals.jsonl")
    ap.add_argument("--static_risk", type=Path,
                    default=CALIBRATION_DIR / "static_entropy_risk.json")
    ap.add_argument("--out", type=Path,
                    default=RESULTS_DIR / "expB_online_rollouts.jsonl")
    ap.add_argument("--target_avg_bits", type=float, default=4.0)
    ap.add_argument("--lo_bits", type=int, default=2)
    ap.add_argument("--hi_bits", type=int, default=16)
    ap.add_argument("--num_layers", type=int, default=28)
    ap.add_argument("--num_kv_heads", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--conditions", nargs="+",
                    default=["B2_Random", "B4_MEDA", "B6_StaticEntropy",
                             "B7_FlippedEntropy", "B8_OnlineResidual",
                             "B9_OnlineNeed_Static"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    args = ap.parse_args()

    # 1. Load diagnostic JSONL
    print(f"[expB_online] loading diagnostic from {args.diagnostic}", flush=True)
    diag_records = load_diagnostic_jsonl(args.diagnostic)
    n_cal = sum(1 for r in diag_records.values() if r["split"] == "cal")
    n_eval = sum(1 for r in diag_records.values() if r["split"] == "eval")
    print(f"[expB_online] diagnostic: cal={n_cal} eval={n_eval}", flush=True)

    # 2. Static risk (cal-only) — aggregate or load from cache
    if args.static_risk.exists():
        static = load_static_risk(args.static_risk)
        print(f"[expB_online] loaded static_risk (n_cal={static['n_cal']}) <- {args.static_risk}",
              flush=True)
    else:
        static = aggregate_static_risk(args.diagnostic, args.num_layers, args.num_kv_heads)
        save_static_risk(static, args.static_risk)
        print(f"[expB_online] aggregated static_risk (n_cal={static['n_cal']}) -> {args.static_risk}",
              flush=True)

    # 3. Compute block budget
    total_blocks = args.num_layers * args.num_kv_heads
    if args.hi_bits <= args.lo_bits:
        raise ValueError("hi_bits must be > lo_bits")
    p_hi = (args.target_avg_bits - args.lo_bits) / (args.hi_bits - args.lo_bits)
    k_hi = int(round(total_blocks * p_hi))
    print(f"[expB_online] target_avg_bits={args.target_avg_bits}: "
          f"k_bf16_blocks={k_hi}/{total_blocks} ({100*p_hi:.1f}% BF16)", flush=True)

    # 4. Load eval items
    items = load_all_items()
    split = load_split(args.split_file)
    eval_items = filter_items(items, split["eval"])
    if args.limit:
        eval_items = eval_items[: args.limit]
    print(f"[expB_online] eval items: {len(eval_items)}", flush=True)

    # 5. Load model (SDPA — fast, no entropy hooks needed at routed eval time)
    from run_inference import load_model
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")

    # 6. Run conditions
    if not args.append and args.out.exists():
        args.out.unlink()
    progress_log = args.out.with_name(args.out.stem + ".progress.log")

    for cond_id in args.conditions:
        if cond_id not in CONDITIONS_SPEC:
            print(f"[expB_online] unknown condition {cond_id}; skipping", flush=True)
            continue
        method, _need_static, _need_pi = CONDITIONS_SPEC[cond_id]
        run_condition(
            cond_id=cond_id, method=method,
            model=model, processor=processor,
            eval_items=eval_items, diag_records=diag_records, static=static,
            num_layers=args.num_layers, num_kv_heads=args.num_kv_heads,
            k_bf16_blocks=k_hi, seeds=args.seeds,
            out_jsonl=args.out, progress_log=progress_log,
            n_frames=args.frames, progress_every=args.progress_every,
        )
        summarize(args.out)

    summarize(args.out)


if __name__ == "__main__":
    main()
