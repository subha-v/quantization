"""
Shared MCQ scorer for Qwen2.5-VL on LongVideoBench.

Loads model + processor (BF16 weights or AWQ), runs prefill with the supplied
FakeQuantKVCache, scores the four answer-token logits, returns prediction +
per-sample diagnostics. Writes per-sample JSONL to results/.

Per-item progress is flushed to stdout every `--progress-every` items (default
10) with running accuracy + ETA. The `summary_callback` (set by drivers) is
invoked every `--summary-every` items (default 25) so summary.md stays current
while a long run is still going. A timestamped milestone is appended to the
progress.log alongside the JSONL.
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import torch

from data_longvideobench import LVBItem, answer_token_ids, format_mcq_messages
from fake_quant_kv_cache import BitController, FakeQuantKVCache


# ---------------- model loading ----------------

def load_model(model_id: str, awq: bool = False, dtype: str = "bfloat16",
               attn_impl: str = "sdpa", device_map: str = "auto"):
    """Load Qwen2.5-VL model + processor.

    awq=True selects the quantized AWQ checkpoint variant. attn_impl is forwarded
    to from_pretrained; SDPA is required for FakeQuantKVCache to intercept K/V
    via the cache.update() return path.
    """
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
    )
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    kwargs = dict(torch_dtype=torch_dtype, attn_implementation=attn_impl, device_map=device_map)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def fake_quantize_weights_w4(model, group_size: int = 128):
    """Apply fake-quant W4 to all Linear layers in the language_model decoder.

    Reuses the lifted symmetric per-group formula. Returns a dict of saved
    originals so callers can restore() if desired.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from utils import fake_quantize_module  # type: ignore

    decoder = getattr(getattr(model, "language_model", None), "model", None) or getattr(model, "language_model", None) or model
    saved = fake_quantize_module(decoder, bits=4, group_size=group_size)
    return saved


# ---------------- per-sample scoring ----------------

@dataclass
class ScoreResult:
    item_id: str
    duration_bucket: str
    duration_seconds: float
    n_frames: int
    correct_choice: int
    pred_choice: int
    is_correct: bool
    option_logprobs: list[float]
    first_token_logits: Optional[list[float]] = None  # full slice for diagnostics
    latency_ms: float = 0.0
    avg_kv_bits: Optional[float] = None
    condition: str = ""
    model_id: str = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))


@torch.no_grad()
def score_item(
    model,
    processor,
    item: LVBItem,
    n_frames: int,
    controller: Optional[BitController],
    *,
    condition: str,
    model_id: str,
    record_logits: bool = False,
    avg_kv_bits: Optional[float] = None,
    entropy_hook_factory=None,
) -> ScoreResult:
    """Score a single LongVideoBench MCQ item; return ScoreResult.

    controller may be None for the BF16 baseline (no cache wrapper).
    entropy_hook_factory: callable(model, cache) -> ContextManager — used by
    calibration runs and V3 online updates; pass None to skip.
    """
    from qwen_vl_utils import process_vision_info  # type: ignore

    messages = format_mcq_messages(item, n_frames=n_frames)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    cache = FakeQuantKVCache(controller) if controller is not None else None
    cm = entropy_hook_factory(model, cache) if (entropy_hook_factory is not None and cache is not None) else nullcontext()

    t0 = time.perf_counter()
    with cm:
        out = model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    n_options = len(item.candidates)
    answer_ids = answer_token_ids(processor, n=n_options)
    first_logits = out.scores[0]  # [1, vocab]
    logprobs = torch.log_softmax(first_logits.float(), dim=-1)[0, answer_ids].tolist()
    pred = int(max(range(n_options), key=lambda i: logprobs[i]))

    return ScoreResult(
        item_id=item.id,
        duration_bucket=item.duration_bucket,
        duration_seconds=item.duration_seconds,
        n_frames=n_frames,
        correct_choice=item.correct_choice,
        pred_choice=pred,
        is_correct=(pred == item.correct_choice),
        option_logprobs=logprobs,
        first_token_logits=first_logits[0].float().cpu().tolist() if record_logits else None,
        latency_ms=latency_ms,
        avg_kv_bits=avg_kv_bits,
        condition=condition,
        model_id=model_id,
    )


# ---------------- batch driver ----------------

def _append_progress(progress_log: Path, line: str) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_log, "a") as f:
        f.write(f"[{ts}] {line}\n")
        f.flush()


def run_condition(
    model,
    processor,
    items: list[LVBItem],
    n_frames: int,
    controller: Optional[BitController],
    *,
    condition: str,
    model_id: str,
    out_jsonl: Path,
    avg_kv_bits: Optional[float] = None,
    entropy_hook_factory=None,
    record_logits_first_n: int = 0,
    progress_every: int = 10,
    summary_every: int = 25,
    summary_callback: Optional[Callable[[Path], None]] = None,
) -> list[ScoreResult]:
    """Run one condition over a list of items; append per-sample rows to out_jsonl.

    Periodically:
      - prints `[condition] i/N acc=X.XXX latency=Yms ETA=Zm` every `progress_every` items
      - calls `summary_callback(out_jsonl)` every `summary_every` items so the
        markdown summary stays current while the run is in flight
      - appends a timestamped milestone to <out_jsonl>.progress.log
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    progress_log = out_jsonl.with_name(out_jsonl.stem + ".progress.log")

    n = len(items)
    _append_progress(progress_log, f"START {condition} frames={n_frames} n_items={n}")
    t_start = time.perf_counter()
    results: list[ScoreResult] = []
    n_correct = 0

    with open(out_jsonl, "a") as f:
        for i, it in enumerate(items):
            r = score_item(
                model, processor, it, n_frames=n_frames, controller=controller,
                condition=condition, model_id=model_id,
                record_logits=(i < record_logits_first_n),
                avg_kv_bits=avg_kv_bits,
                entropy_hook_factory=entropy_hook_factory,
            )
            f.write(r.to_jsonl() + "\n")
            f.flush()
            results.append(r)
            n_correct += int(r.is_correct)

            done = i + 1
            if done % progress_every == 0 or done == n:
                elapsed = time.perf_counter() - t_start
                rate = elapsed / done
                eta = max(0.0, rate * (n - done))
                acc = n_correct / done
                eta_str = str(timedelta(seconds=int(eta)))
                line = (f"[{condition}] frames={n_frames} {done}/{n} "
                        f"acc={acc:.3f} latency={r.latency_ms:.0f}ms "
                        f"elapsed={timedelta(seconds=int(elapsed))} ETA={eta_str}")
                print(line, flush=True)
                _append_progress(progress_log, line)

            if summary_callback is not None and done % summary_every == 0:
                try:
                    summary_callback(out_jsonl)
                except Exception as e:
                    _append_progress(progress_log, f"WARN summary_callback failed: {e!r}")

    if summary_callback is not None:
        try:
            summary_callback(out_jsonl)
        except Exception as e:
            _append_progress(progress_log, f"WARN final summary_callback failed: {e!r}")

    final_acc = n_correct / max(1, n)
    final_line = (f"DONE {condition} frames={n_frames} acc={n_correct}/{n}={final_acc:.3f} "
                  f"wall={timedelta(seconds=int(time.perf_counter() - t_start))}")
    print(f"[{condition}] {final_line}", flush=True)
    _append_progress(progress_log, final_line)
    return results
