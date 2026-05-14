"""Build a unified per-item flip matrix CSV across Exp Q / R / S / T-mini.

One row per (dataset, task_type, seed, item_id). For each method (BF16, F4,
F9, SJ, S4, Q7, PageLocal-F4) the row carries pred / correct / margin.
Routing diagnostics (Quest needle-hit / needle-rank) come from FormatBook
conditions when available.

Margin definition:
  - MCQ tasks: logprob(correct) - max(logprob(non-correct))
  - counting-image: soft_accuracy (proxy)

Inputs (all in qwen/results/):
  expT_mini_rollouts_retrieval-image.jsonl  -- T0..T16 + T5b on retrieval-image
  expT_mini_rollouts_reasoning-image.jsonl  -- T0..T16 + T5b on reasoning-image
  expT_mini_rollouts_counting-image.jsonl   -- C0..C12 on counting-image
  expQ_rollouts_sliceA.jsonl                -- Q0..Q11 on retrieval-image (FormatBook)
  expR_rollouts_C.jsonl                     -- C0..C8 + S4/S8/S12/SJ on retrieval-image
  expS_rollouts_phase1.jsonl                -- S0..S9 sidecode ladder on retrieval-image

Output:
  expT_mini_flip_matrix.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional


# Maps the "logical method name" -> list of (jsonl-file-name, condition-name) candidates
# in priority order. First match wins for a given (item_id, source-task).
METHOD_SOURCES = {
    "bf16": [
        ("expT_mini_rollouts_{task}.jsonl", "T0"),
        ("expT_mini_rollouts_{task}.jsonl", "C0"),  # counting-image
        ("expR_rollouts_C.jsonl",            "C0"),
        ("expS_rollouts_phase1.jsonl",       "S0"),
        ("expQ_rollouts_sliceA.jsonl",       "Q0"),
    ],
    "f4": [
        ("expT_mini_rollouts_{task}.jsonl", "T1"),
        ("expT_mini_rollouts_{task}.jsonl", "C1"),
        ("expR_rollouts_C.jsonl",            "C1"),
        ("expS_rollouts_phase1.jsonl",       "S1"),
        ("expQ_rollouts_sliceA.jsonl",       "Q1"),
    ],
    "f9": [
        ("expT_mini_rollouts_{task}.jsonl", "T2"),
        ("expT_mini_rollouts_{task}.jsonl", "C2"),
        ("expR_rollouts_C.jsonl",            "C2"),
        ("expS_rollouts_phase1.jsonl",       "S2"),
        ("expQ_rollouts_sliceA.jsonl",       "Q2"),
    ],
    "sj": [
        ("expT_mini_rollouts_{task}.jsonl", "T3"),
        ("expT_mini_rollouts_{task}.jsonl", "C3"),
        ("expR_rollouts_C.jsonl",            "SJ"),
        ("expS_rollouts_phase1.jsonl",       "S3"),
    ],
    "s4": [
        ("expT_mini_rollouts_{task}.jsonl", "T4"),
        ("expT_mini_rollouts_{task}.jsonl", "C4"),
        ("expS_rollouts_phase1.jsonl",       "S4"),
    ],
    "q7": [
        # Q7 = Exp Q FormatBook Quest top-25% (also has routing diagnostics)
        ("expQ_rollouts_sliceA.jsonl",       "Q7"),
    ],
    "pagelocal": [
        ("expT_mini_rollouts_{task}.jsonl", "T8"),
        ("expT_mini_rollouts_{task}.jsonl", "C6"),
    ],
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def compute_margin(r: dict) -> Optional[float]:
    """logprob(correct) - max(logprob(wrong)) for MCQ; soft_accuracy for counting."""
    # Counting-image: soft_accuracy is our proxy margin.
    if "soft_accuracy" in r and r.get("task") == "counting-image":
        return float(r["soft_accuracy"])
    logp = r.get("option_logprobs")
    correct = r.get("correct_choice")
    if logp is None or correct is None:
        return None
    n = len(logp)
    if not (0 <= int(correct) < n):
        return None
    correct_lp = float(logp[int(correct)])
    wrong = [float(x) for i, x in enumerate(logp) if i != int(correct)]
    if not wrong:
        return None
    return correct_lp - max(wrong)


def extract_per_item(rows: list[dict], cond_name: str) -> dict[str, dict]:
    """item_id -> {pred, correct, margin, plus diagnostics if present}."""
    out: dict[str, dict] = {}
    for r in rows:
        if r.get("condition") != cond_name:
            continue
        iid = str(r.get("item_id"))
        rec = {
            "pred": (r.get("pred_choice") if "pred_choice" in r
                     else (r.get("parsed") if "parsed" in r else None)),
            "correct": bool(r.get("is_correct", False)),
            "margin": compute_margin(r),
            "exact_match": r.get("exact_match"),
            "valid_format": r.get("valid_format"),
            "length_match": r.get("length_match"),
            "sum_match": r.get("sum_match"),
            "soft_accuracy": r.get("soft_accuracy"),
            "predicted_length": r.get("predicted_length"),
            "effective_k_bits": r.get("effective_k_bits"),
            "effective_v_bits": r.get("effective_v_bits"),
            "effective_kv_bits": r.get("effective_kv_bits"),
            "f9_sidecode_token_fraction": r.get("f9_sidecode_token_fraction"),
            "needle_in_active_layer_mean": r.get("needle_in_active_layer_mean"),
            "needle_rank_median": r.get("needle_rank_median"),
            "page_read_fraction": r.get("page_read_fraction"),
            "latency_ms": r.get("latency_ms"),
        }
        out[iid] = rec
    return out


def fetch_method(method: str, task: str, results_dir: Path) -> dict[str, dict]:
    """Try each (file, cond) source for `method` in priority order."""
    for fname_tmpl, cond_name in METHOD_SOURCES[method]:
        fname = fname_tmpl.format(task=task)
        path = results_dir / fname
        rows = load_jsonl(path)
        per = extract_per_item(rows, cond_name)
        if per:
            return per
    return {}


def gather_for_task(task: str, results_dir: Path) -> tuple[list[dict], dict]:
    """Return (rows, anchor_rows) for one task. anchor_rows are BF16 rollouts
    used to seed the item_id list and base item-level metadata."""
    anchor = fetch_method("bf16", task, results_dir)
    if not anchor:
        return [], {}

    methods = ["bf16", "f4", "f9", "sj", "s4", "q7", "pagelocal"]
    method_data: dict[str, dict[str, dict]] = {}
    for m in methods:
        method_data[m] = fetch_method(m, task, results_dir)

    # Item-level metadata from the BF16 JSONL row (any condition would do; BF16
    # is the canonical reference for the per-item context).
    anchor_jsonl_path: Optional[Path] = None
    for fname_tmpl, cond_name in METHOD_SOURCES["bf16"]:
        fname = fname_tmpl.format(task=task)
        path = results_dir / fname
        rows = load_jsonl(path)
        if any(r.get("condition") == cond_name for r in rows):
            anchor_jsonl_path = path
            break
    anchor_rows = []
    if anchor_jsonl_path is not None:
        anchor_rows = [r for r in load_jsonl(anchor_jsonl_path)
                       if r.get("condition") == METHOD_SOURCES["bf16"][0][1]
                       or r.get("condition") == METHOD_SOURCES["bf16"][1][1]
                       or r.get("condition") in ("T0", "C0", "Q0", "S0")]
    meta_by_id: dict[str, dict] = {}
    for r in anchor_rows:
        iid = str(r.get("item_id"))
        meta_by_id[iid] = {
            "context_length": r.get("context_length"),
            "context_length_bucket": r.get("context_length_bucket"),
            "num_images": r.get("num_images"),
            "needle_idx_in_images": r.get("needle_idx_in_images"),
            "placed_depth": r.get("placed_depth"),
            "correct_choice": r.get("correct_choice"),
            "num_choices": len(r.get("option_logprobs") or [])
                            if r.get("option_logprobs") else 0,
            "seq_len": r.get("seq_len"),
            "gold_counts": r.get("gold_counts"),
            "gold_sum": r.get("gold_sum"),
        }

    out_rows: list[dict] = []
    for iid, meta in meta_by_id.items():
        row = {
            "item_id": iid,
            "dataset": "MM-NIAH",
            "task_type": task,
            "seed": 0,
            **meta,
        }
        # Per-method columns: pred, correct, margin
        for m in methods:
            d = method_data[m].get(iid, {})
            row[f"{m}_pred"] = d.get("pred")
            row[f"{m}_correct"] = d.get("correct")
            row[f"{m}_margin"] = d.get("margin")
        # Counting-image extras
        if task == "counting-image":
            for m in methods:
                d = method_data[m].get(iid, {})
                row[f"{m}_valid_format"] = d.get("valid_format")
                row[f"{m}_length_match"] = d.get("length_match")
                row[f"{m}_sum_match"] = d.get("sum_match")
                row[f"{m}_soft_accuracy"] = d.get("soft_accuracy")
                row[f"{m}_predicted_length"] = d.get("predicted_length")
        # Routing diagnostics from Q7
        q7d = method_data["q7"].get(iid, {})
        row["q7_needle_in_active_layer_mean"] = q7d.get("needle_in_active_layer_mean")
        row["q7_needle_rank_median"] = q7d.get("needle_rank_median")
        row["q7_page_read_fraction"] = q7d.get("page_read_fraction")
        # Bit accounting (take from F9 if available; otherwise BF16 row)
        for src in ("f9", "bf16"):
            d = method_data[src].get(iid, {})
            if d.get("effective_kv_bits") is not None:
                row["f9_effective_kv_bits"] = d.get("effective_kv_bits")
                row["f9_sidecode_token_frac"] = d.get("f9_sidecode_token_fraction")
                break
        # PageLocal bit accounting
        pl = method_data["pagelocal"].get(iid, {})
        row["pagelocal_effective_kv_bits"] = pl.get("effective_kv_bits")
        out_rows.append(row)

    return out_rows, meta_by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results")
    ap.add_argument("--out-csv", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "results" / "expT_mini_flip_matrix.csv")
    args = ap.parse_args()

    all_rows: list[dict] = []
    for task in ("retrieval-image", "reasoning-image", "counting-image"):
        rows, _meta = gather_for_task(task, args.results_dir)
        print(f"{task}: {len(rows)} rows")
        all_rows.extend(rows)

    if not all_rows:
        print("[warn] no rows assembled")
        return

    # Collect all keys for a stable column order
    all_keys: list[str] = []
    seen = set()
    for r in all_rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"wrote {args.out_csv} ({len(all_rows)} rows, {len(all_keys)} columns)")


if __name__ == "__main__":
    main()
