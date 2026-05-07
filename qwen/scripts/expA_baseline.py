"""
Experiment A: baseline KV-quant sensitivity for Qwen2.5-VL on LongVideoBench.

8 conditions × {64, 128} frame budgets, 200 eval examples each. Identifies the
"rescuable regime" = examples where BF16 is correct and uniform KV-quant flips
wrong; that's the population AttnEntropy needs to recover in Experiment B.

Conditions (see plan):
  A1 BF16-W + BF16-KV          (ceiling)
  A2 W4-fakequant + BF16-KV    (weight-only, LIBERO methodology)
  A3 AWQ + BF16-KV             (weight-only, real)
  A4 BF16-W + FP8-KV           (mild)
  A5 BF16-W + INT4-KV          (uniform aggressive)
  A6 BF16-W + INT4-K/INT8-V    (asymmetric)
  A7 BF16-W + INT2-KV          (sub-2-bit stress)
  A8 AWQ + INT4-KV             (combined realistic)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data_longvideobench import (
    DEFAULT_SPLIT_FILE,
    filter_items,
    load_all_items,
    load_split,
)
from fake_quant_kv_cache import BitController
from run_inference import fake_quantize_weights_w4, load_model, run_condition


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


CONDITIONS = [
    # (name, weights, k_bits, v_bits)  -- weights in {bf16, w4_fake, awq}
    ("A1_BF16",            "bf16",   16, 16),
    ("A2_W4fake_BF16KV",   "w4_fake", 16, 16),
    ("A3_AWQ_BF16KV",      "awq",    16, 16),
    ("A4_BF16_FP8KV",      "bf16",    8,  8),
    ("A5_BF16_INT4KV",     "bf16",    4,  4),
    ("A6_BF16_INT4K_INT8V", "bf16",   4,  8),
    ("A7_BF16_INT2KV",     "bf16",    2,  2),
    ("A8_AWQ_INT4KV",      "awq",     4,  4),
    # Experiment C — K/V isolation: each leaves one side at BF16, quantizes the other.
    ("C2_1_BF16K_INT4V",   "bf16",   16,  4),
    ("C2_2_INT4K_BF16V",   "bf16",    4, 16),
    ("C2_3_BF16K_INT2V",   "bf16",   16,  2),
    ("C2_4_INT2K_BF16V",   "bf16",    2, 16),
]


def avg_bits_for(k: int, v: int, num_layers: int, num_kv_heads: int) -> float:
    return (k + v) / 2.0


def run(args):
    items = load_all_items()
    split = load_split(args.split_file)
    eval_items = filter_items(items, split["eval"])
    if args.limit:
        eval_items = eval_items[: args.limit]
    if args.stratified_limit:
        # Bucket-proportional sub-sample preserving short/mid/long/very_long ratios.
        # The split list is grouped by bucket, so plain --limit slices unevenly.
        by_bucket: dict[str, list] = {}
        for it in eval_items:
            by_bucket.setdefault(it.duration_bucket, []).append(it)
        total = len(eval_items)
        chosen, picks = [], []
        for bucket, pool in by_bucket.items():
            n_pick = round(args.stratified_limit * len(pool) / total)
            picks.append((bucket, n_pick, pool))
        # Fix rounding off-by-one against the largest bucket
        diff = args.stratified_limit - sum(n for _, n, _ in picks)
        if diff != 0:
            largest_idx = max(range(len(picks)), key=lambda i: len(picks[i][2]))
            b, n, p = picks[largest_idx]
            picks[largest_idx] = (b, n + diff, p)
        for bucket, n_pick, pool in picks:
            chosen.extend(pool[:n_pick])
        eval_items = chosen
        bucket_counts = {b: sum(1 for it in eval_items if it.duration_bucket == b)
                         for b in {it.duration_bucket for it in eval_items}}
        print(f"[expA] stratified_limit={args.stratified_limit} bucket_counts={bucket_counts}")
    print(f"[expA] eval_items={len(eval_items)}")

    out_jsonl = RESULTS_DIR / f"expA_rollouts_{args.model.split('/')[-1]}.jsonl"
    if out_jsonl.exists() and not args.append:
        out_jsonl.unlink()

    last_weight_mode = None
    model, processor = None, None
    saved_w4 = None

    def need_model_for(weight_mode: str):
        nonlocal model, processor, last_weight_mode, saved_w4
        if last_weight_mode == weight_mode:
            return
        # Tear down previous
        if model is not None:
            del model
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
        if weight_mode == "awq":
            model_id = args.model + "-AWQ"
            model, processor = load_model(model_id, awq=True, dtype="float16",
                                          attn_impl="sdpa")
        else:
            model, processor = load_model(args.model, dtype="bfloat16", attn_impl="sdpa")
            if weight_mode == "w4_fake":
                saved_w4 = fake_quantize_weights_w4(model)
        last_weight_mode = weight_mode

    num_layers = None
    for name, weights, kb, vb in CONDITIONS:
        if name not in args.conditions:
            continue
        need_model_for(weights)
        if num_layers is None:
            lm = getattr(model, "language_model", None)
            num_layers = len(getattr(lm, "layers", None) or model.model.layers)

        for n_frames in args.frames:
            controller = BitController(num_layers=num_layers, num_kv_heads=4, mode="V1",
                                       default_k_bits=kb, default_v_bits=vb)
            controller.set_global(k_bits=kb, v_bits=vb)
            avg_bits = avg_bits_for(kb, vb, num_layers, 4)
            print(f"[expA] {name} frames={n_frames} kb={kb} vb={vb} avg={avg_bits:.2f}",
                  flush=True)
            run_condition(
                model, processor, eval_items, n_frames=n_frames,
                controller=controller,
                condition=f"{name}_frames{n_frames}",
                model_id=args.model + ("-AWQ" if weights == "awq" else ""),
                out_jsonl=out_jsonl,
                avg_kv_bits=avg_bits,
                record_logits_first_n=args.record_logits_first_n,
                progress_every=args.progress_every,
                summary_every=args.summary_every,
                summary_callback=summarize,
            )
            # Final summary regen after each (condition × frames) so file is current
            summarize(out_jsonl)

    summarize(out_jsonl)


# ---------------- summary ----------------

def _bootstrap_ci(arr: list[bool], n_boot: int = 2000, seed: int = 0):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.size, arr.size)
        means.append(arr[idx].mean())
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(jsonl_path: Path) -> None:
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    bf16_set = next(
        (rows_i for cond, rows_i in by_cond.items() if cond.startswith("A1_BF16")), []
    )
    bf16_correct_by_id = {(r["item_id"], r["n_frames"]): r["is_correct"] for r in bf16_set}

    out_lines = ["# Experiment A — KV-quant sensitivity\n"]
    out_lines.append("| Condition | n | acc | 95% CI | avg KV bits | BF16-correct preserved |")
    out_lines.append("|---|---:|---:|---|---:|---:|")
    for cond, rs in sorted(by_cond.items()):
        accs = [r["is_correct"] for r in rs]
        mean, lo, hi = _bootstrap_ci(accs)
        avg_bits = next((r["avg_kv_bits"] for r in rs if r.get("avg_kv_bits") is not None), float("nan"))
        # BF16-correct preservation
        if not cond.startswith("A1_BF16"):
            shared = [r for r in rs if bf16_correct_by_id.get((r["item_id"], r["n_frames"])) is True]
            preserved = sum(1 for r in shared if r["is_correct"]) / max(1, len(shared))
        else:
            preserved = 1.0
        out_lines.append(f"| {cond} | {len(rs)} | {mean:.3f} | [{lo:.3f}, {hi:.3f}] | {avg_bits:.2f} | {preserved:.3f} |")

    summary_path = jsonl_path.with_name(jsonl_path.stem.replace("rollouts", "summary") + ".md")
    summary_path.write_text("\n".join(out_lines) + "\n")
    print(f"[expA] summary -> {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap eval items (0 = all). Slices the bucket-grouped list "
                         "as-is, so usually only first N items by bucket order.")
    ap.add_argument("--stratified_limit", type=int, default=0,
                    help="Cap eval items to N total, proportionally stratified across "
                         "duration buckets. 0 = disabled.")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing JSONL instead of overwriting.")
    ap.add_argument("--conditions", nargs="+", default=[c[0] for c in CONDITIONS])
    ap.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--record_logits_first_n", type=int, default=0,
                    help="For the first N items per condition, record full first-token logits "
                         "(used by smoke test for the BF16-vs-INT2 logits-differ assertion).")
    ap.add_argument("--progress_every", type=int, default=10,
                    help="Print stdout progress + ETA every N items.")
    ap.add_argument("--summary_every", type=int, default=25,
                    help="Regenerate summary.md every N items so user can tail mid-run.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
