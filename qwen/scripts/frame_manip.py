"""Frame-level manipulation for Exp D0 evidence-restriction conditions.

D0 needs to feed the Qwen2.5-VL processor a controlled subset of frames per
condition (Top-1-window-only, Top-1-window-removed, Random-window-removed,
Uniform-16). v1 ships with the simpler "frame removal" path: pre-decode 64
frames at the same uniform indices qwen_vl_utils would use, then select a
subset and pass it back to the processor as a list of PIL Images.

Sequence length and temporal RoPE positions change as a result. We mark this
with `mode="frame_removal_v1"` in the JSONL row. The seq_len-preserving
"blank-in-place" v2 (replace dropped frames with black/mean RGB) is a stretch
goal kept behind a flag.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from PIL import Image


def _uniform_indices(total_frames: int, n_sample: int) -> list[int]:
    """Mimic qwen_vl_utils' uniform sampling: floor((i+0.5) * total / n_sample)."""
    if n_sample <= 0 or total_frames <= 0:
        return []
    if n_sample >= total_frames:
        return list(range(total_frames))
    return [min(total_frames - 1, int(math.floor((i + 0.5) * total_frames / n_sample)))
            for i in range(n_sample)]


def decode_uniform_frames(video_path: str, n_frames: int = 64) -> list[Image.Image]:
    """Decode `video_path` and uniformly sample n_frames as RGB PIL Images.

    Uses decord if available (matches qwen_vl_utils' default backend); falls
    back to imageio/cv2 otherwise. Returns a list of PIL.Image of length
    n_frames (or fewer if the source has fewer frames).
    """
    # Try decord first (most consistent with qwen_vl_utils sampling)
    try:
        import decord  # type: ignore

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path, num_threads=1)
        total = len(vr)
        idx = _uniform_indices(total, n_frames)
        batch = vr.get_batch(idx).asnumpy()  # [N, H, W, 3] uint8
        return [Image.fromarray(batch[i]) for i in range(batch.shape[0])]
    except Exception:
        pass
    # Fallback: imageio
    try:
        import imageio.v3 as iio  # type: ignore
        meta = iio.immeta(video_path)
        # imageio doesn't expose total frames cleanly; iterate.
        frames_all = list(iio.imiter(video_path))
        total = len(frames_all)
        idx = _uniform_indices(total, n_frames)
        return [Image.fromarray(frames_all[i]) for i in idx]
    except Exception:
        pass
    # Last-resort fallback: cv2
    import cv2  # type: ignore
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = _uniform_indices(total, n_frames)
    frames: list[Image.Image] = []
    target = sorted(set(idx))
    target_iter = iter(target)
    next_target = next(target_iter, None)
    cur = 0
    while next_target is not None:
        ret, bgr = cap.read()
        if not ret:
            break
        if cur == next_target:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            next_target = next(target_iter, None)
        cur += 1
    cap.release()
    # Reorder to the original idx ordering (handles duplicates)
    by_index = {t: frames[i] for i, t in enumerate(target[: len(frames)])}
    return [by_index[i] for i in idx if i in by_index]


def select_frame_subset(frames: list[Image.Image], indices: Iterable[int]) -> list[Image.Image]:
    """Return frames[i] for i in indices, preserving order. Skips out-of-range."""
    n = len(frames)
    return [frames[i] for i in indices if 0 <= i < n]


def window_indices(window_id: int, frames_per_window: int = 8) -> list[int]:
    """Frame indices for a single window (0..7)."""
    a = window_id * frames_per_window
    return list(range(a, a + frames_per_window))


def windows_indices(window_ids: Iterable[int], frames_per_window: int = 8) -> list[int]:
    """Frame indices for multiple windows, sorted ascending."""
    out: list[int] = []
    for w in window_ids:
        out.extend(window_indices(int(w), frames_per_window))
    return sorted(set(out))


def all_indices_except(window_id: int, n_windows: int = 8, frames_per_window: int = 8) -> list[int]:
    """Frame indices excluding a single window."""
    keep: list[int] = []
    for w in range(n_windows):
        if w == window_id:
            continue
        keep.extend(window_indices(w, frames_per_window))
    return keep


def blank_frames_in_place(
    frames: list[Image.Image], blank_indices: Iterable[int], color: tuple[int, int, int] = (0, 0, 0)
) -> list[Image.Image]:
    """Return a new list where frames at blank_indices are solid-color images
    matching the source frame size. Shape-preserving alternative to frame removal.
    Used by the v2 stretch-goal path; not invoked in v1.
    """
    blank_set = set(int(i) for i in blank_indices)
    out: list[Image.Image] = []
    for i, f in enumerate(frames):
        if i in blank_set:
            out.append(Image.new("RGB", f.size, color))
        else:
            out.append(f)
    return out
