"""
Unified diagnostic pass for Experiment B online precision-need routing.

For each LongVideoBench item, runs three forwards through Qwen2.5-VL-7B:
  1. BF16 KV (eager attention) — captures per-(layer, KV-head) signals:
        entropy_mean         (averaged over query positions)
        entropy_answer_query (entropy of last query position's attention row)
        aq_topk_mass         (top-32 attention mass at last query position)
        aq_frame_entropy     (placeholder=0 for v1; can be filled later)
        kv_residual_int2     (Frobenius residual under simulated INT2 quant)
        kv_residual_int4     (under simulated INT4 quant)
        bf16_pred            (argmax MCQ answer under BF16 KV)
  2. Uniform INT4 KV (SDPA, fast) — uniform_int4_pred
  3. Uniform INT2 KV (SDPA, fast) — uniform_int2_pred

Writes one JSONL row per (item, layer, kv_head). Eval-leakage guarded by an
explicit `split` field consumed by the static-risk aggregator in
precision_need_scoring.py.

Memory budget at 7B + 64 frames + eager attention is ~25-28 GB on top of
tlandeg's 50 GB co-tenant. The chunked entropy in attn_entropy_hook avoids
materializing the full FP32 attention tensor.

Usage:
    python diagnostic_pass.py --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --frames 64 --splits cal eval \\
        --out qwen/results/diagnostic_signals.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
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
    load_all_items,
    load_split,
)
from fake_quant_kv_cache import (
    BitController,
    FakeQuantKVCache,
    fake_quantize_kv,
    _find_decoder_layers,
)


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


# ===================================================================
# Diagnostic cache: BF16 storage, computes residuals against simulated quant
# ===================================================================

class DiagnosticCache(DynamicCache):
    """BF16 cache that records per-(layer, KV-head) Frobenius residuals against
    simulated INT2 and INT4 fake-quantization on the fly.

    Storage stays BF16; this just measures "how much would quantization perturb
    K, V at each (layer, KV-head)?". Used in the diagnostic pass before any
    routing decision is made.
    """

    def __init__(self, num_layers: int):
        super().__init__()
        # residual[layer_idx] = {'int2_K': [H], 'int2_V': [H], 'int4_K': [H], 'int4_V': [H]}
        self.residual: list[Optional[dict]] = [None] * num_layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # key_states, value_states: [B, H_kv, T_new, D]
        with torch.no_grad():
            for bits, tag in [(2, "int2"), (4, "int4")]:
                Kq = fake_quantize_kv(key_states, bits)
                Vq = fake_quantize_kv(value_states, bits)
                # Per-(B, H_kv) Frobenius norm of residual / norm of original
                num_K = (key_states - Kq).pow(2).sum(dim=(-2, -1)).sqrt()  # [B, H]
                den_K = key_states.pow(2).sum(dim=(-2, -1)).sqrt().clamp_min(1e-8)
                rk = (num_K / den_K)[0].detach().cpu()  # [H]
                num_V = (value_states - Vq).pow(2).sum(dim=(-2, -1)).sqrt()
                den_V = value_states.pow(2).sum(dim=(-2, -1)).sqrt().clamp_min(1e-8)
                rv = (num_V / den_V)[0].detach().cpu()
                if self.residual[layer_idx] is None:
                    self.residual[layer_idx] = {}
                self.residual[layer_idx][f"{tag}_K"] = rk
                self.residual[layer_idx][f"{tag}_V"] = rv
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


# ===================================================================
# Diagnostic attention hook: captures per-(layer, KV-head) attention signals
# ===================================================================

def _chunked_entropy_mean(kv_attn: torch.Tensor, chunk_size: int = 256) -> torch.Tensor:
    """kv_attn: [B, H_kv, Q, K] -> [H_kv] mean entropy over query positions, normalized by log(K)."""
    eps = 1e-12
    B, H, Q, K = kv_attn.shape
    norm = math.log(max(2, K))
    h_acc = torch.zeros(H, dtype=torch.float32, device=kv_attn.device)
    for q_start in range(0, Q, chunk_size):
        q_end = min(q_start + chunk_size, Q)
        chunk = kv_attn[:1, :, q_start:q_end, :].float()
        ent_chunk = -(chunk * (chunk + eps).log()).sum(dim=-1)  # [1, H, chunk]
        h_acc += ent_chunk.sum(dim=(0, 2))
    return (h_acc / max(1, Q) / norm).cpu()


class DiagnosticAttentionHook:
    """Captures per-(layer, KV-head) attention signals during a single eager
    BF16 forward pass. Pools 7 Q-heads per KV-head by averaging.

    Signals captured (per layer):
        entropy_mean         [H_kv]   averaged over query positions
        entropy_answer_query [H_kv]   entropy of last-query attention row
        aq_topk_mass         [H_kv]   top-32 mass at last query position
        aq_frame_entropy     [H_kv]   placeholder zeros for v1
    """

    def __init__(self, model: torch.nn.Module, num_kv_heads: int = 4, num_kv_groups: int = 7,
                 top_k: int = 32):
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_kv_groups
        self.top_k = top_k
        self.signals: dict[int, dict[str, torch.Tensor]] = {}
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
                        recorder._process_attn(lidx, attn_weights)
                    return result
                return wrapped
            attn.forward = make_wrapped(original_forward, layer_idx, self)
            self._restores.append((attn, original_forward))

    def _process_attn(self, layer_idx: int, attn_weights: torch.Tensor) -> None:
        # attn_weights: [B, H_q, Q, K] (eager)
        with torch.no_grad():
            B, Hq, Q, K = attn_weights.shape
            Hkv = self.num_kv_heads
            groups = Hq // Hkv if Hkv > 0 else 1
            if groups < 1:
                groups = 1
            # Truncate Hq to Hkv*groups in case it isn't exactly divisible
            usable_Hq = Hkv * groups
            if usable_Hq != Hq:
                attn_weights = attn_weights[:, :usable_Hq, :, :]
            # Pool Q-heads -> KV-heads by averaging in BF16 (saves memory)
            kv_attn = attn_weights[:1].view(1, Hkv, groups, Q, K).mean(dim=2)  # [1, Hkv, Q, K]
            # entropy_mean over all queries
            entropy_mean = _chunked_entropy_mean(kv_attn)  # [Hkv]
            # entropy_answer_query: last query position
            eps = 1e-12
            log_K = math.log(max(2, K))
            last_q = kv_attn[:, :, -1, :].float()  # [1, Hkv, K]
            entropy_aq = (-(last_q * (last_q + eps).log()).sum(dim=-1) / log_K)[0].cpu()  # [Hkv]
            # aq_topk_mass at last query position
            top_k = min(self.top_k, K)
            top_mass = last_q.topk(top_k, dim=-1).values.sum(dim=-1)[0].cpu()  # [Hkv]
            self.signals[layer_idx] = {
                "entropy_mean": entropy_mean,
                "entropy_answer_query": entropy_aq,
                "aq_topk_mass": top_mass,
                "aq_frame_entropy": torch.zeros(Hkv),  # v1 placeholder
            }
        del attn_weights

    def uninstall(self) -> None:
        for module, original in self._restores:
            module.forward = original
        self._restores = []


# ===================================================================
# Per-item diagnostic: 3 forwards (BF16 eager + uniform INT4 + uniform INT2)
# ===================================================================

def _option_logprobs_and_pred(out, processor, n_options: int) -> tuple[list[float], int]:
    answer_ids = answer_token_ids(processor, n=n_options)
    first_logits = out.scores[0]
    logp = torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist()
    pred = int(max(range(n_options), key=lambda i: logp[i]))
    return logp, pred


@torch.no_grad()
def diagnose_item(
    model_eager,           # model loaded with attn_implementation="eager"
    model_sdpa,            # model loaded with attn_implementation="sdpa" (can be same instance if eager)
    processor,
    item: LVBItem,
    n_frames: int,
    num_layers: int,
    num_kv_heads: int,
    split: str,
) -> list[dict]:
    """Returns a list of (num_layers * num_kv_heads) JSONL rows for this item."""
    from qwen_vl_utils import process_vision_info  # type: ignore

    msgs = format_mcq_messages(item, n_frames=n_frames)
    prompt_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(msgs)
    inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model_eager.device)

    n_options = len(item.candidates)

    # ---- Forward 1: BF16 + DiagnosticCache + DiagnosticAttentionHook (eager) ----
    diag_cache = DiagnosticCache(num_layers=num_layers)
    diag_hook = DiagnosticAttentionHook(model_eager, num_kv_heads=num_kv_heads)
    try:
        out_bf16 = model_eager.generate(
            **inputs, past_key_values=diag_cache, max_new_tokens=1, do_sample=False,
            return_dict_in_generate=True, output_scores=True, use_cache=True,
        )
    finally:
        diag_hook.uninstall()
    _, bf16_pred = _option_logprobs_and_pred(out_bf16, processor, n_options)

    # Free the BF16 cache — we have all the residual data we need
    bf16_residual = list(diag_cache.residual)  # snapshot
    bf16_signals = dict(diag_hook.signals)
    del diag_cache, diag_hook, out_bf16
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Forward 2: Uniform INT4 KV (SDPA, fast) ----
    ctrl_int4 = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads, mode="V1")
    ctrl_int4.set_global(k_bits=4, v_bits=4)
    cache_int4 = FakeQuantKVCache(ctrl_int4)
    out_int4 = model_sdpa.generate(
        **inputs, past_key_values=cache_int4, max_new_tokens=1, do_sample=False,
        return_dict_in_generate=True, output_scores=True, use_cache=True,
    )
    _, uniform_int4_pred = _option_logprobs_and_pred(out_int4, processor, n_options)
    del cache_int4, out_int4
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Forward 3: Uniform INT2 KV ----
    ctrl_int2 = BitController(num_layers=num_layers, num_kv_heads=num_kv_heads, mode="V1")
    ctrl_int2.set_global(k_bits=2, v_bits=2)
    cache_int2 = FakeQuantKVCache(ctrl_int2)
    out_int2 = model_sdpa.generate(
        **inputs, past_key_values=cache_int2, max_new_tokens=1, do_sample=False,
        return_dict_in_generate=True, output_scores=True, use_cache=True,
    )
    _, uniform_int2_pred = _option_logprobs_and_pred(out_int2, processor, n_options)
    del cache_int2, out_int2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Pack into per-(layer, KV-head) rows ----
    rows: list[dict] = []
    for L in range(num_layers):
        sig = bf16_signals.get(L, None)
        res = bf16_residual[L]
        for h in range(num_kv_heads):
            row = {
                "item_id": item.id,
                "split": split,
                "n_frames": n_frames,
                "duration_bucket": item.duration_bucket,
                "duration_seconds": item.duration_seconds,
                "correct_choice": item.correct_choice,
                "n_options": n_options,
                "layer": L,
                "kv_head": h,
                "entropy_mean": float(sig["entropy_mean"][h].item()) if sig else float("nan"),
                "entropy_answer_query": float(sig["entropy_answer_query"][h].item()) if sig else float("nan"),
                "aq_topk_mass": float(sig["aq_topk_mass"][h].item()) if sig else float("nan"),
                "aq_frame_entropy": float(sig["aq_frame_entropy"][h].item()) if sig else 0.0,
                "kv_residual_int2": (float((res["int2_K"][h] + res["int2_V"][h]).item()) if res else float("nan")),
                "kv_residual_int4": (float((res["int4_K"][h] + res["int4_V"][h]).item()) if res else float("nan")),
                "bf16_pred": bf16_pred,
                "uniform_int4_pred": uniform_int4_pred,
                "uniform_int2_pred": uniform_int2_pred,
            }
            rows.append(row)
    return rows


# ===================================================================
# Driver
# ===================================================================

def _append_progress(progress_log: Path, line: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_log, "a") as f:
        f.write(f"[{ts}] {line}\n"); f.flush()


def run_diagnostic(
    model,                    # eager attention model
    processor,
    items: list[LVBItem],
    split: str,
    n_frames: int,
    num_layers: int,
    num_kv_heads: int,
    out_jsonl: Path,
    progress_every: int = 5,
) -> None:
    """For now we use a single eager-loaded model for all 3 forwards (eager
    works for SDPA-style code; just slower). Caller can pass two model handles
    if memory allows the dual-load."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_log = out_jsonl.with_name(out_jsonl.stem + ".progress.log")
    _append_progress(progress_log, f"START split={split} n_items={len(items)} frames={n_frames}")
    t_start = time.perf_counter()
    n_done = 0
    with open(out_jsonl, "a") as f:
        for i, it in enumerate(items):
            try:
                rows = diagnose_item(
                    model_eager=model, model_sdpa=model, processor=processor, item=it,
                    n_frames=n_frames, num_layers=num_layers, num_kv_heads=num_kv_heads,
                    split=split,
                )
            except Exception as e:
                _append_progress(progress_log, f"WARN item={it.id} skipped: {type(e).__name__}: {e}")
                continue
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            n_done += 1
            done = i + 1
            if done % progress_every == 0 or done == len(items):
                elapsed = time.perf_counter() - t_start
                rate = elapsed / max(1, n_done)
                eta = max(0.0, rate * (len(items) - done))
                _append_progress(progress_log, f"{done}/{len(items)} ok={n_done} "
                                              f"elapsed={timedelta(seconds=int(elapsed))} "
                                              f"ETA={timedelta(seconds=int(eta))}")
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _append_progress(progress_log, f"DONE split={split} n_items={n_done}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--splits", nargs="+", default=["cal", "eval"], choices=["cal", "eval"])
    ap.add_argument("--out", type=Path,
                    default=RESULTS_DIR / "diagnostic_signals.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--progress_every", type=int, default=5)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    items = load_all_items()
    split_data = load_split(args.split_file)

    from run_inference import load_model
    model, processor = load_model(args.model, dtype="bfloat16",
                                  attn_impl="eager", device_map="auto")
    # Locate decoder layers + KV head count
    lm = getattr(model, "language_model", None) or model.model.language_model
    layers = lm.layers if hasattr(lm, "layers") else lm.model.layers
    num_layers = len(layers)
    num_kv_heads = getattr(model.config, "num_key_value_heads", 4)
    print(f"[diag] num_layers={num_layers} num_kv_heads={num_kv_heads} attn=eager", flush=True)

    if not args.append and args.out.exists():
        args.out.unlink()
    for split in args.splits:
        ids = split_data[split]
        sub = filter_items(items, ids)
        if args.limit:
            sub = sub[: args.limit]
        run_diagnostic(model, processor, sub, split=split, n_frames=args.frames,
                       num_layers=num_layers, num_kv_heads=num_kv_heads,
                       out_jsonl=args.out, progress_every=args.progress_every)


if __name__ == "__main__":
    main()
