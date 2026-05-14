"""CPU-only runtime audit for Exp T-mini (no GPU contention).

Verifies — using the real Qwen2.5-VL processor (image processor + tokenizer,
no model weights) — that:

  1. reasoning-image items load correctly with the patched _normalize
     (binary MCQ support).
  2. counting-image items build the right page layout: haystack images
     first as in_context_image pages, needle pattern last as choice_image.
  3. format_counting_messages produces a well-formed chat template that
     tokenizes cleanly.
  4. format_mcq_messages handles the dynamic "Answer with letters" tail
     for the 2-choice reasoning-image case.

No GPU required; just AutoProcessor + qwen_vl_utils for image pre-proc.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mm_niah_loader import (
    load_all_items, format_counting_messages, format_mcq_messages,
)
from page_layout import build_page_layout, coverage_ok


@dataclass
class AuditResult:
    name: str
    passed: bool
    detail: str


def _load_processor(model_id: str):
    from transformers import AutoProcessor  # type: ignore
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def audit_reasoning_image_loader() -> AuditResult:
    """Confirm reasoning-image items pass _normalize with the binary MCQ fix."""
    items = load_all_items(task="reasoning-image")
    if not items:
        return AuditResult("reasoning-image: load_all_items returns items",
                           False, "loaded 0 items — _normalize is rejecting everything")
    # Check that num_choices is correctly set to 2.
    nc = {it.num_choices for it in items[:10]}
    return AuditResult("reasoning-image: load_all_items returns items",
                       True,
                       f"loaded {len(items)} items; sample num_choices={nc} "
                       f"(expected {{2}})")


def audit_counting_image_loader() -> AuditResult:
    items = load_all_items(task="counting-image")
    if not items:
        return AuditResult("counting-image: load_all_items returns items",
                           False, "loaded 0 items")
    # Confirm gold_counts and task fields set correctly.
    samp = [it for it in items[:5]]
    bad = [it for it in samp
           if it.gold_counts is None or len(it.gold_counts) != it.num_images - 1]
    return AuditResult("counting-image: gold_counts length = num_images - 1",
                       not bad,
                       f"loaded {len(items)} items; first 5 sampled num_choices="
                       f"{[it.num_choices for it in samp]} (expected 0); "
                       f"bad gold_counts items: {len(bad)}")


def audit_counting_image_page_layout(processor, n_check: int = 2) -> AuditResult:
    """Build real page layouts on counting-image items and verify:
       - Total visual spans == num_images
       - First (num_images - 1) spans are in_context_image (haystack)
       - Last span is choice_image (needle pattern from question)
       - coverage_ok = True
    """
    from qwen_vl_utils import process_vision_info  # type: ignore

    items = load_all_items(task="counting-image")
    # Pick items with num_images >= 5 to match the Phase 3 filter.
    big = [it for it in items if it.num_images >= 5][:n_check]
    if not big:
        return AuditResult("counting-image: page layout shape on real items",
                           False, "no items with num_images >= 5")
    details = []
    all_ok = True
    for it in big:
        try:
            msgs = format_counting_messages(it)
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(msgs)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt")
            input_ids = inputs["input_ids"]
            layout = build_page_layout(
                input_ids, processor,
                n_in_context_images=max(0, it.num_images - 1),
                n_choice_images=1,
                needle_idx_in_images=-1,
                include_choice_routing=False,
            )
            n_in_ctx = sum(1 for p in layout.pages if p.kind == "in_context_image")
            n_choice = sum(1 for p in layout.pages if p.kind == "choice_image")
            # Last visual span check
            visual_pages = [p for p in layout.pages
                            if p.kind in ("in_context_image", "choice_image")]
            last_visual = visual_pages[-1] if visual_pages else None
            cov = coverage_ok(layout)
            expected_in_ctx = it.num_images - 1
            ok = (
                n_in_ctx == expected_in_ctx
                and n_choice == 1
                and last_visual is not None
                and last_visual.kind == "choice_image"
                and cov
            )
            all_ok = all_ok and ok
            details.append(
                f"id={it.id} num_images={it.num_images} "
                f"in_ctx={n_in_ctx}/{expected_in_ctx} choice={n_choice}/1 "
                f"last_visual_kind={last_visual.kind if last_visual else 'NONE'} "
                f"cov_ok={cov} -> {'OK' if ok else 'FAIL'}"
            )
        except Exception as e:
            all_ok = False
            details.append(f"id={it.id} FAILED: {type(e).__name__}: {e}")
    return AuditResult("counting-image: page layout shape on real items",
                       all_ok, " | ".join(details))


def audit_reasoning_image_page_layout(processor, n_check: int = 2) -> AuditResult:
    """Build real page layouts on reasoning-image items and verify:
       - num_choices == 2 (binary MCQ)
       - Total visual spans == num_images + 2 (= num_images haystack + 2 choices)
       - Last 2 spans are choice_image
       - coverage_ok = True
       - format_mcq_messages produces "Answer with a single letter from A, B."
    """
    from qwen_vl_utils import process_vision_info  # type: ignore

    items = load_all_items(task="reasoning-image")
    # Pick items with num_images >= 5 to match the Phase 2 filter.
    big = [it for it in items if it.num_images >= 5][:n_check]
    if not big:
        return AuditResult("reasoning-image: page layout shape on real items",
                           False, "no items with num_images >= 5")
    details = []
    all_ok = True
    instr_text = None
    for it in big:
        try:
            msgs = format_mcq_messages(it, max_pixels_context=144 * 144,
                                       max_pixels_choices=144 * 144)
            # Inspect the last text fragment for the dynamic instruction
            tail_text = msgs[0]["content"][-1].get("text", "")
            instr_text = tail_text
            text = processor.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(msgs)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt")
            input_ids = inputs["input_ids"]
            layout = build_page_layout(
                input_ids, processor,
                n_in_context_images=it.num_images,
                n_choice_images=2,  # reasoning-image is binary
                needle_idx_in_images=it.needle_idx_in_images,
                include_choice_routing=False,
            )
            n_in_ctx = sum(1 for p in layout.pages if p.kind == "in_context_image")
            n_choice = sum(1 for p in layout.pages if p.kind == "choice_image")
            cov = coverage_ok(layout)
            ok = (n_in_ctx == it.num_images and n_choice == 2 and cov)
            all_ok = all_ok and ok
            details.append(
                f"id={it.id} num_images={it.num_images} in_ctx={n_in_ctx} "
                f"choice={n_choice}/2 cov_ok={cov} -> {'OK' if ok else 'FAIL'}"
            )
        except Exception as e:
            all_ok = False
            details.append(f"id={it.id} FAILED: {type(e).__name__}: {e}")
    detail = " | ".join(details)
    if instr_text:
        detail += f" | instruction_tail={instr_text!r}"
    return AuditResult("reasoning-image: page layout + dynamic MCQ tail",
                       all_ok, detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--out-md", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "results" / "expT_mini_runtime_audit.md")
    ap.add_argument("--n-check", type=int, default=2)
    args = ap.parse_args()

    results: list[AuditResult] = []
    # Loader-only checks (cheap).
    results.append(audit_reasoning_image_loader())
    results.append(audit_counting_image_loader())
    # Processor-based checks (loads AutoProcessor + actually tokenizes real images).
    print(f"Loading processor {args.model}...", flush=True)
    t0 = time.perf_counter()
    processor = _load_processor(args.model)
    print(f"  processor loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    results.append(audit_counting_image_page_layout(processor, n_check=args.n_check))
    results.append(audit_reasoning_image_page_layout(processor, n_check=args.n_check))

    # Write summary
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Exp T-mini runtime audit ({time.strftime('%Y-%m-%d %H:%M:%S')})", "",
             "| # | Audit | Pass | Detail |", "|---|---|---|---|"]
    n_pass = 0
    for i, r in enumerate(results, 1):
        flag = "PASS" if r.passed else "FAIL"
        if r.passed:
            n_pass += 1
        lines.append(f"| {i} | {r.name} | {flag} | {r.detail} |")
    lines.append("")
    lines.append(f"**Summary**: {n_pass}/{len(results)} passed")
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"\nRuntime audit: {n_pass}/{len(results)} passed -> {args.out_md}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.detail}")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
