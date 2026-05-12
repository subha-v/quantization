"""Exp Q calibration on MM-NIAH reasoning-image cal-100.

Clone of expP_calibrate.py with task=reasoning-image and a task-specific split
file + NPZ output. Recomputes F9 outlier channel indices on the reasoning-image
subtask — the Exp P retrieval-image NPZ would be the wrong distribution for
this slice (per the LVB → MM-NIAH overlap result of 11/16 from Exp P, expect
similar reasoning-image vs retrieval-image divergence).

Output: qwen/calibration/expQ_mmniah_reasoning_kcalib_{model}_seed{S}.{json,npz}
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from expF_calibrate import KStatsCache, QStatsHook
from expP_calibrate import _mmniah_visual_envelope  # reuse for visual-region detection
from mm_niah_loader import (
    SUPPORTED_TASKS, filter_items, format_mcq_messages,
    load_all_items, load_split, make_split, save_split, split_file_for_task,
)


CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration"


@torch.no_grad()
def run_calibration(model_id: str, task: str, n_outliers_top: int,
                    split_file: Path, out_json: Path, out_npz: Path,
                    seed: int = 0, max_q_per_item: int = 256,
                    progress_every: int = 5, limit: int = 0) -> None:
    from run_inference import load_model
    from qwen_vl_utils import process_vision_info  # type: ignore

    items = load_all_items(task=task)
    if split_file.exists():
        split = load_split(split_file)
        if "cal" not in split:
            print(f"[mmniah-calib] split at {split_file} missing 'cal'; regenerating")
            split = make_split(items, seed=seed)
            save_split(split, split_file)
    else:
        split = make_split(items, seed=seed)
        save_split(split, split_file)
    cal_items = filter_items(items, split["cal"])
    if limit:
        cal_items = cal_items[:limit]
    print(f"[mmniah-calib] task={task} cal_items={len(cal_items)} model={model_id} seed={seed}",
          flush=True)

    model, processor = load_model(model_id, awq=False, dtype="bfloat16",
                                  attn_impl="sdpa", device_map="auto")
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", 4))
    num_q_heads = int(getattr(model.config, "num_attention_heads", 28))
    q_proj_weight = layers[0].self_attn.q_proj.weight
    head_dim = int(q_proj_weight.shape[0] // num_q_heads)
    print(f"[mmniah-calib] num_layers={num_layers} num_kv_heads={num_kv_heads} "
          f"num_q_heads={num_q_heads} head_dim={head_dim}", flush=True)

    progress_log = out_json.with_name(out_json.stem + ".progress.log")
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    def _log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[mmniah-calib] {msg}", flush=True)
        with open(progress_log, "a") as f:
            f.write(f"[{ts}] {msg}\n"); f.flush()

    _log(f"START task={task} n_cal={len(cal_items)} max_q_per_item={max_q_per_item}")
    t_start = time.perf_counter()

    q_hook = QStatsHook(model, num_layers=num_layers, num_kv_heads=num_kv_heads,
                        num_q_heads=num_q_heads, head_dim=head_dim,
                        max_q_per_item=max_q_per_item)

    k_sumsq_total = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)
    k_count_total = torch.zeros(num_layers, dtype=torch.int64)
    k_max_total = torch.zeros(num_layers, num_kv_heads, head_dim, dtype=torch.float32)

    n_done, n_failed = 0, 0
    try:
        for i, it in enumerate(cal_items):
            try:
                msgs = format_mcq_messages(it)
                prompt_text = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(msgs)
                inputs = processor(text=[prompt_text], images=image_inputs,
                                   videos=video_inputs, padding=True,
                                   return_tensors="pt").to(model.device)
                v_start, v_end = _mmniah_visual_envelope(inputs["input_ids"], processor)
                q_hook.set_item_context(v_start, v_end)

                cache = KStatsCache(num_layers=num_layers, num_kv_heads=num_kv_heads,
                                    head_dim=head_dim)
                _ = model.generate(**inputs, past_key_values=cache,
                                   max_new_tokens=1, do_sample=False,
                                   return_dict_in_generate=True, output_scores=True,
                                   use_cache=True)
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
                         f"seq_len={int(inputs['input_ids'].shape[1])} "
                         f"elapsed={timedelta(seconds=int(elapsed))} "
                         f"ETA={timedelta(seconds=int(eta))}")
            except Exception as e:
                n_failed += 1
                _log(f"WARN item={it.id} skipped: {type(e).__name__}: {e}")
                continue
    finally:
        q_hook.uninstall()

    k_channel_energy = torch.zeros_like(k_sumsq_total)
    for L in range(num_layers):
        c = float(k_count_total[L].item())
        if c > 0:
            k_channel_energy[L] = k_sumsq_total[L] / c
    k_channel_energy_np = k_channel_energy.numpy().astype(np.float32)
    k_max_np = k_max_total.numpy().astype(np.float32)

    n_top = int(n_outliers_top)
    outlier_idx = np.argsort(k_channel_energy_np, axis=-1)[..., -n_top:][..., ::-1].copy()
    outlier_idx = outlier_idx.astype(np.int32)

    q_data = q_hook.finalize()
    q_data["q_energy"] = np.clip(q_data["q_energy"], 1e-12, None).astype(np.float32)
    q_data["q_energy_text"] = np.clip(q_data["q_energy_text"], 1e-12, None).astype(np.float32)
    q_data["q_energy_visual"] = np.clip(q_data["q_energy_visual"], 1e-12, None).astype(np.float32)

    meta = {
        "model": model_id,
        "dataset": f"MM-NIAH val {task}",
        "task": task,
        "seed": seed,
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
    ap.add_argument("--task", choices=SUPPORTED_TASKS, default="reasoning-image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_file", type=Path, default=None,
                    help="Defaults to per-task path via split_file_for_task(task).")
    ap.add_argument("--max_q_per_item", type=int, default=256)
    ap.add_argument("--n_outliers_top", type=int, default=16)
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_npz", type=Path, default=None)
    args = ap.parse_args()

    model_short = args.model.split("/")[-1]
    if args.split_file is None:
        args.split_file = split_file_for_task(args.task)
    if args.out_json is None:
        args.out_json = CALIBRATION_DIR / f"expQ_mmniah_{args.task}_kcalib_{model_short}_seed{args.seed}.json"
    if args.out_npz is None:
        args.out_npz = CALIBRATION_DIR / f"expQ_mmniah_{args.task}_kcalib_{model_short}_seed{args.seed}.npz"

    run_calibration(
        model_id=args.model, task=args.task, n_outliers_top=args.n_outliers_top,
        split_file=args.split_file, out_json=args.out_json, out_npz=args.out_npz,
        seed=args.seed, max_q_per_item=args.max_q_per_item,
        progress_every=args.progress_every, limit=args.limit,
    )


if __name__ == "__main__":
    main()
