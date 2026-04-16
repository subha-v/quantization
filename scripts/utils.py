"""
Shared utilities for VLA quantization experiments.

Covers: model loading (openpi), data loading (lerobot HuggingFace → openpi format),
fake quantization, activation hooks, metrics, and incremental result I/O.
"""

import os
import sys
import json
import time
import logging
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# GPU pinning — must happen before any CUDA call
# ---------------------------------------------------------------------------
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# ---------------------------------------------------------------------------
# Paths — all configurable via env vars.  Set these before importing.
# ---------------------------------------------------------------------------
WORKSPACE = os.environ.get("WORKSPACE", "/data/subha2")
OPENPI_DIR = os.environ.get("OPENPI_DIR", os.path.join(WORKSPACE, "openpi"))
EXPERIMENT_DIR = os.environ.get(
    "EXPERIMENT_DIR", os.path.join(WORKSPACE, "experiments")
)
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
PLOTS_DIR = os.path.join(EXPERIMENT_DIR, "plots")

os.environ.setdefault("HF_HOME", os.path.join(WORKSPACE, "hf_cache"))

# Make openpi importable
for p in [os.path.join(OPENPI_DIR, "src"), OPENPI_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)


# ===================================================================
# Model loading
# ===================================================================

def download_checkpoint(config_name="pi05_libero"):
    """Download the JAX checkpoint via openpi's GCS downloader.

    Returns the local path to the downloaded checkpoint directory.
    """
    from openpi.shared import download

    # The fine-tuned LIBERO checkpoint
    gcs_paths = [
        f"gs://openpi-assets/checkpoints/{config_name}",
        f"gs://openpi-assets/checkpoints/{config_name}/params",
    ]
    for gcs in gcs_paths:
        try:
            local = download.maybe_download(gcs)
            print(f"[download] {gcs} → {local}")
            return str(local)
        except Exception as e:
            print(f"[download] {gcs} failed: {e}")

    raise RuntimeError(
        f"Could not download checkpoint for {config_name}. "
        "Tried: " + ", ".join(gcs_paths)
    )


def _find_local_checkpoint(config_name="pi05_libero"):
    """Search for an already-downloaded checkpoint on disk.

    Prefers PyTorch (model.safetensors) over JAX so we get nn.Module.
    """
    cache_base = os.environ.get("OPENPI_DATA_HOME", os.path.join(WORKSPACE, ".cache/openpi"))

    # PyTorch checkpoints first — these give us nn.Module
    pytorch_candidates = [
        os.path.join(WORKSPACE, f"{config_name}_pytorch"),
        os.path.join(WORKSPACE, "pi05_libero_pytorch"),
    ]
    for c in pytorch_candidates:
        if os.path.isdir(c) and (
            os.path.exists(os.path.join(c, "model.safetensors"))
            or os.path.exists(os.path.join(c, "model.pt"))
        ):
            print(f"[checkpoint] Found PyTorch: {c}")
            return c

    # Fall back to JAX
    jax_candidates = [
        os.path.join(cache_base, "openpi-assets/checkpoints", config_name),
        os.path.join(cache_base, "checkpoints", config_name),
    ]
    for c in jax_candidates:
        if os.path.isdir(c):
            print(f"[checkpoint] Found JAX: {c}")
            return c
    return None


def load_policy(config_name="pi05_libero", checkpoint_dir=None):
    """Load an openpi policy ready for inference.

    Returns (policy, model) where:
        policy — the Policy object (call policy.infer(obs))
        model  — the underlying torch.nn.Module (for hooks / weight surgery)
    """
    from openpi.training import config as train_config
    from openpi.policies import policy_config as pc

    config = train_config.get_config(config_name)
    print(f"[load_policy] config={config_name}")

    if checkpoint_dir is None:
        # Try local first to avoid re-downloading
        checkpoint_dir = _find_local_checkpoint(config_name)
    if checkpoint_dir is None:
        checkpoint_dir = download_checkpoint(config_name)

    print(f"[load_policy] checkpoint_dir={checkpoint_dir}")

    # create_trained_policy handles transforms + weight loading
    policy = pc.create_trained_policy(config, checkpoint_dir)
    print(f"[load_policy] policy type: {type(policy).__name__}")

    # Extract the underlying nn.Module
    model = _extract_model(policy)
    return policy, model


def _extract_model(policy):
    """Walk policy attributes to find the torch.nn.Module."""
    for attr in ["model", "_model", "net", "_net"]:
        obj = getattr(policy, attr, None)
        if isinstance(obj, torch.nn.Module):
            print(f"[load_policy] model at .{attr}: {type(obj).__name__}")
            return obj
    # Fallback: first nn.Module attribute
    for attr in dir(policy):
        if attr.startswith("_"):
            continue
        obj = getattr(policy, attr, None)
        if isinstance(obj, torch.nn.Module):
            print(f"[load_policy] model at .{attr}: {type(obj).__name__}")
            return obj
    print("[load_policy] WARNING: could not find nn.Module inside policy")
    return None


def print_model_summary(model, max_depth=3):
    """Print layer names/types/shapes; return summary dict."""
    total_params = sum(p.numel() for p in model.parameters())
    linear_layers = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            linear_layers.append((name, m.weight.shape))

    print(f"\n[model] Parameters: {total_params:,}")
    print(f"[model] Linear layers: {len(linear_layers)}")
    for name, shape in linear_layers[:60]:
        print(f"  {name}: Linear({shape[1]}→{shape[0]})")
    if len(linear_layers) > 60:
        print(f"  ... {len(linear_layers) - 60} more")

    return {
        "total_params": total_params,
        "linear_count": len(linear_layers),
        "linear_layers": [(n, list(s)) for n, s in linear_layers],
    }


def get_layer_groups(model):
    """Identify quantizable layer groups at the decoder-layer granularity.

    Returns list of {"name", "module", "group_type", "linear_count"}.
    """
    groups = []
    claimed = set()

    for name, module in model.named_modules():
        if any(name.startswith(c + ".") or name == c for c in claimed):
            continue
        cls = type(module).__name__
        n_lin = sum(1 for _, m in module.named_modules() if isinstance(m, torch.nn.Linear))

        if "DecoderLayer" in cls and n_lin > 0:
            gtype = "action_expert" if any(k in name for k in ("expert", "action")) else "vlm"
            groups.append({"name": name, "module": module, "group_type": gtype, "linear_count": n_lin})
            claimed.add(name)

    # Non-decoder blocks: vision, projector, action head
    for name, module in model.named_modules():
        if any(name.startswith(c + ".") or name == c for c in claimed):
            continue
        n_lin = sum(1 for _, m in module.named_modules() if isinstance(m, torch.nn.Linear))
        if n_lin == 0:
            continue
        lower = name.lower()
        if any(k in lower for k in ("vision", "siglip", "vit")):
            groups.append({"name": name, "module": module, "group_type": "vision", "linear_count": n_lin})
            claimed.add(name)
        elif any(k in lower for k in ("projector", "multi_modal")):
            groups.append({"name": name, "module": module, "group_type": "projector", "linear_count": n_lin})
            claimed.add(name)
        elif any(k in lower for k in ("action_out", "action_in", "state_proj", "time_mlp")):
            groups.append({"name": name, "module": module, "group_type": "action_head", "linear_count": n_lin})
            claimed.add(name)

    print(f"[layer_groups] {len(groups)} groups found:")
    for g in groups:
        print(f"  {g['name']}: {g['group_type']} ({g['linear_count']} linears)")
    return groups


# ===================================================================
# Data loading  (lerobot HuggingFace → openpi observation format)
# ===================================================================

# LIBERO task suites.  The task_index→suite mapping depends on the dataset
# version; we identify suites from the language instruction at runtime.
SPATIAL_KEYWORDS = ["spatial", "left of", "right of", "behind", "in front of", "next to"]
LONG_KEYWORDS = ["and then", "after that", "first.*then"]  # multi-step instructions

def classify_suite(instruction: str) -> str:
    """Heuristic suite classification from the language instruction.

    Returns one of: spatial, object, goal, long, unknown.
    The definitive mapping is printed by setup_and_verify.py; this is a fallback.
    """
    inst_lower = instruction.lower()
    import re
    for kw in LONG_KEYWORDS:
        if re.search(kw, inst_lower):
            return "long"
    for kw in SPATIAL_KEYWORDS:
        if kw in inst_lower:
            return "spatial"
    return "unknown"


def load_libero_observations(n_easy=128, n_hard=128, seed=42, suite_map=None):
    """Load observations from the LIBERO HuggingFace dataset.

    Args:
        n_easy: samples from easy suite (LIBERO-Spatial or first 10 tasks)
        n_hard: samples from hard suite (LIBERO-Long or last 10 tasks)
        seed: RNG seed for reproducibility
        suite_map: optional dict {task_index: suite_name} from setup_and_verify

    Returns (observations, metadata) where:
        observations: list of openpi-formatted observation dicts
        metadata: list of per-sample metadata dicts for post-hoc analysis
    """
    from datasets import load_dataset

    rng = np.random.RandomState(seed)
    # Try multiple dataset sources — the primary one may need a newer datasets lib
    ds = None
    for repo_id in [
        "physical-intelligence/libero",
        "lerobot/libero",
        "HuggingFaceVLA/libero",
    ]:
        try:
            print(f"[data] Trying {repo_id}...")
            ds = load_dataset(repo_id, split="train")
            print(f"[data] Loaded {repo_id}: {len(ds)} frames, columns: {ds.column_names}")
            break
        except Exception as e:
            print(f"[data] {repo_id} failed: {e}")
            continue

    if ds is None:
        raise RuntimeError(
            "Could not load LIBERO dataset from any source. "
            "Try: uv pip install --upgrade datasets huggingface-hub"
        )

    # Get task descriptions for suite classification + prompts
    task_descriptions = _load_task_descriptions(ds)
    print(f"[data] {len(task_descriptions)} unique tasks")

    # Build suite map if not provided
    if suite_map is None:
        suite_map = _build_suite_map(task_descriptions)

    # Identify easy (spatial) and hard (long) task indices
    easy_tasks = [t for t, s in suite_map.items() if s in ("spatial", "easy")]
    hard_tasks = [t for t, s in suite_map.items() if s in ("long", "hard")]

    # Fallback: if classification didn't work, use first 10 / last 10
    if not easy_tasks:
        print("[data] WARNING: no spatial tasks identified; using task_index 0-9")
        easy_tasks = list(range(10))
    if not hard_tasks:
        print("[data] WARNING: no long tasks identified; using task_index 30-39")
        hard_tasks = list(range(30, 40))

    print(f"[data] Easy tasks: {easy_tasks}")
    print(f"[data] Hard tasks: {hard_tasks}")

    # Get task indices from dataset
    task_col = _find_column(ds, ["task_index", "task_id", "task"])
    all_task_indices = np.array(ds[task_col])

    easy_mask = np.isin(all_task_indices, easy_tasks)
    hard_mask = np.isin(all_task_indices, hard_tasks)
    easy_frame_indices = np.where(easy_mask)[0]
    hard_frame_indices = np.where(hard_mask)[0]

    print(f"[data] Easy frames: {len(easy_frame_indices)}, Hard frames: {len(hard_frame_indices)}")

    n_e = min(n_easy, len(easy_frame_indices))
    n_h = min(n_hard, len(hard_frame_indices))
    sampled = np.concatenate([
        rng.choice(easy_frame_indices, size=n_e, replace=False),
        rng.choice(hard_frame_indices, size=n_h, replace=False),
    ])
    print(f"[data] Sampled {n_e} easy + {n_h} hard = {len(sampled)} observations")

    # Build observations + metadata
    observations = []
    metadata = []
    ep_col = _find_column(ds, ["episode_index", "episode_id"])
    frame_col = _find_column(ds, ["frame_index", "index"])

    for i, idx in enumerate(sampled):
        idx = int(idx)
        sample = ds[idx]

        task_id = int(sample[task_col])
        episode_id = int(sample[ep_col]) if ep_col else -1
        frame_idx = int(sample[frame_col]) if frame_col else -1
        suite = suite_map.get(task_id, "unknown")
        prompt = task_descriptions.get(task_id, "do something")

        obs = _format_observation(sample, prompt)
        observations.append(obs)

        metadata.append({
            "sample_idx": i,
            "dataset_idx": idx,
            "task_id": task_id,
            "episode_id": episode_id,
            "frame_idx": frame_idx,
            "suite": suite,
            "prompt": prompt,
            # phase_bin will be computed if episode_length is known
            "phase_bin": "unknown",
        })

        if i % 50 == 0 and i > 0:
            print(f"  loaded {i}/{len(sampled)}")

    # Try to compute phase bins from episode lengths
    _compute_phase_bins(metadata, ds, ep_col, frame_col)

    print(f"[data] Done. {len(observations)} observations loaded.")
    return observations, metadata


def _load_task_descriptions(ds):
    """Extract task_index → language instruction mapping."""
    task_col = _find_column(ds, ["task_index", "task_id"])
    desc_col = _find_column(ds, [
        "language_instruction", "task_description", "prompt",
        "observation.language_instruction",
    ])
    if not task_col or not desc_col:
        # Try the meta/tasks.jsonl file via the dataset API
        print("[data] No language instruction column; using generic prompts")
        return {}

    # Sample a few rows per task to get descriptions
    descriptions = {}
    seen = set()
    for i in range(0, len(ds), max(1, len(ds) // 200)):
        tid = int(ds[i][task_col])
        if tid not in seen:
            seen.add(tid)
            desc = ds[i][desc_col]
            if isinstance(desc, str) and len(desc) > 2:
                descriptions[tid] = desc
    return descriptions


def _build_suite_map(task_descriptions):
    """Classify tasks into suites from their language instructions."""
    suite_map = {}
    for tid, desc in task_descriptions.items():
        suite_map[tid] = classify_suite(desc)
    return suite_map


def _find_column(ds, candidates):
    """Find the first matching column name in the dataset."""
    for c in candidates:
        if c in ds.column_names:
            return c
    return None


def _format_observation(sample, prompt):
    """Convert a lerobot HuggingFace sample to openpi observation format."""
    obs = {"prompt": prompt}

    # State: observation.state → observation/state
    for src in ["observation.state", "state"]:
        if src in sample:
            val = sample[src]
            obs["observation/state"] = np.array(val, dtype=np.float32)
            break

    # Top camera: observation.images.image → observation/image
    for src in ["observation.images.image", "image", "observation.image"]:
        if src in sample:
            img = _to_numpy_image(sample[src])
            obs["observation/image"] = img
            break

    # Wrist camera: observation.images.image2 → observation/wrist_image
    for src in ["observation.images.image2", "wrist_image", "image2", "observation.wrist_image"]:
        if src in sample:
            img = _to_numpy_image(sample[src])
            obs["observation/wrist_image"] = img
            break

    return obs


def _to_numpy_image(img):
    """Convert various image formats to (H, W, 3) uint8 numpy array."""
    if isinstance(img, np.ndarray):
        return img.astype(np.uint8)
    if isinstance(img, torch.Tensor):
        arr = img.numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:  # CHW → HWC
            arr = arr.transpose(1, 2, 0)
        return arr.astype(np.uint8)
    # PIL Image
    try:
        return np.array(img, dtype=np.uint8)
    except Exception:
        pass
    return np.zeros((256, 256, 3), dtype=np.uint8)


def _compute_phase_bins(metadata, ds, ep_col, frame_col):
    """Compute early/mid/late phase bins from episode structure."""
    if not ep_col or not frame_col:
        return

    # Get max frame per episode (approximate episode length)
    ep_ids = set(m["episode_id"] for m in metadata if m["episode_id"] >= 0)
    if not ep_ids:
        return

    # Use dataset to find episode lengths
    all_eps = np.array(ds[ep_col])
    all_frames = np.array(ds[frame_col])

    ep_lengths = {}
    for eid in ep_ids:
        mask = all_eps == eid
        if mask.any():
            ep_lengths[eid] = int(all_frames[mask].max()) + 1

    for m in metadata:
        eid = m["episode_id"]
        fid = m["frame_idx"]
        if eid in ep_lengths and fid >= 0:
            m["episode_length"] = ep_lengths[eid]
            progress = fid / ep_lengths[eid]
            m["phase_bin"] = "early" if progress < 0.25 else ("mid" if progress < 0.75 else "late")


# ===================================================================
# Inference
# ===================================================================

def run_inference(policy, observation):
    """Run policy.infer() on a single observation. Returns action as numpy array."""
    with torch.no_grad():
        result = policy.infer(observation)

    if isinstance(result, dict):
        for key in ["actions", "action", "raw_actions"]:
            if key in result:
                val = result[key]
                return val.cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
        # First array-valued entry
        for v in result.values():
            if isinstance(v, (np.ndarray, torch.Tensor)):
                return v.cpu().numpy() if isinstance(v, torch.Tensor) else v
        raise ValueError(f"No action in result keys: {list(result.keys())}")

    if isinstance(result, torch.Tensor):
        return result.cpu().numpy()
    return np.asarray(result)


def compute_reference_actions(policy, observations):
    """Compute FP16 reference actions for all observations."""
    actions = []
    for i, obs in enumerate(observations):
        if i % 50 == 0:
            print(f"  reference actions: {i}/{len(observations)}")
        actions.append(run_inference(policy, obs))
    return actions


# ===================================================================
# Quantization
# ===================================================================

def fake_quantize_module(module, bits=4, group_size=128):
    """Fake symmetric weight quantization on all Linear layers in module.

    Modifies weights in-place.  Returns dict of cloned originals for restoration.
    """
    qmax = 2 ** (bits - 1) - 1
    saved = {}
    for name, child in module.named_modules():
        if not isinstance(child, torch.nn.Linear):
            continue
        saved[name] = child.weight.data.clone()
        w = child.weight.data.float()
        if group_size > 0 and w.shape[1] >= group_size and w.shape[1] % group_size == 0:
            g = w.reshape(w.shape[0], -1, group_size)
            s = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
            child.weight.data = ((g / s).round().clamp(-qmax, qmax) * s).reshape_as(w).to(child.weight.dtype)
        else:
            s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
            child.weight.data = ((w / s).round().clamp(-qmax, qmax) * s).to(child.weight.dtype)
    return saved


def restore_weights(module, saved):
    """Restore weights from a dict produced by fake_quantize_module."""
    for name, child in module.named_modules():
        if isinstance(child, torch.nn.Linear) and name in saved:
            child.weight.data = saved[name]


def precompute_quantized_weights(module, bits=4, group_size=128):
    """Pre-compute quantized weight tensors for O(1) pointer swapping.

    Returns (orig_ptrs, quant_tensors).
    """
    qmax = 2 ** (bits - 1) - 1
    orig, quant = {}, {}
    for name, child in module.named_modules():
        if not isinstance(child, torch.nn.Linear):
            continue
        w = child.weight.data
        orig[name] = w  # pointer, not copy
        wf = w.float()
        if group_size > 0 and wf.shape[1] >= group_size and wf.shape[1] % group_size == 0:
            g = wf.reshape(wf.shape[0], -1, group_size)
            s = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
            quant[name] = ((g / s).round().clamp(-qmax, qmax) * s).reshape_as(wf).to(w.dtype)
        else:
            s = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
            quant[name] = ((wf / s).round().clamp(-qmax, qmax) * s).to(w.dtype)
    return orig, quant


def swap_weights(module, weight_dict):
    """Pointer-swap all Linear weights in module to weight_dict entries."""
    for name, child in module.named_modules():
        if isinstance(child, torch.nn.Linear) and name in weight_dict:
            child.weight.data = weight_dict[name]


# ===================================================================
# Activation hooks
# ===================================================================

def compute_kurtosis(t):
    x = t.flatten().float()
    if x.numel() < 4:
        return 0.0
    std = x.std()
    if std < 1e-8:
        return 0.0
    return float(((x - x.mean()) / std).pow(4).mean() - 3.0)


def register_activation_hooks(model, layer_filter=None):
    """Register hooks on Linear layers.  Returns (hooks, stats_dict)."""
    stats, hooks = {}, []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if layer_filter and not layer_filter(name):
            continue

        def _hook(layer_name):
            def fn(mod, inp, out):
                with torch.no_grad():
                    x = out.detach().float()
                    s = x.std()
                    stats.setdefault(layer_name, []).append({
                        "max_abs": float(x.abs().max()),
                        "mean_abs": float(x.abs().mean()),
                        "l2_norm": float(x.norm(2)),
                        "std": float(s),
                        "kurtosis": compute_kurtosis(x),
                        "outlier_6s": float((x.abs() > 6 * s).float().mean()) if s > 1e-8 else 0.0,
                    })
            return fn

        hooks.append(module.register_forward_hook(_hook(name)))
    print(f"[hooks] registered {len(hooks)} hooks")
    return hooks, stats


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


# ===================================================================
# Metrics
# ===================================================================

def action_mse(a, b):
    return float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))


# ===================================================================
# Result I/O  (incremental, fsync'd)
# ===================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data, path):
    """Atomic JSON write with fsync."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        data["_saved_at"] = datetime.now().isoformat()
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(p)


def append_jsonl(entry, path):
    """Append one JSON object to a JSONL file.  Fsync'd for durability."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(entry, cls=NumpyEncoder) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def save_npz(path, **arrays):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(p), **arrays)


# ===================================================================
# Plotting
# ===================================================================

def setup_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (14, 6),
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 150,
        "savefig.bbox_inches": "tight",
    })
    return plt


# ===================================================================
# Timer
# ===================================================================

class Timer:
    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *_):
        self.elapsed = time.time() - self._t0
        if self.name:
            log(f"[timer] {self.name}: {self.elapsed:.1f}s")


# ===================================================================
# Logging — dual output to console + file
# ===================================================================

_log_file = None

def setup_logging(log_path=None):
    """Set up dual logging to console and file.  Call once at script start."""
    global _log_file
    if log_path is None:
        log_path = os.path.join(RESULTS_DIR, "experiment.log")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(log_path, "a")
    _log_file.write(f"\n{'='*60}\n")
    _log_file.write(f"Session started: {datetime.now().isoformat()}\n")
    _log_file.write(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}\n")
    _log_file.write(f"{'='*60}\n")
    _log_file.flush()


def log(msg):
    """Print to console AND append to log file."""
    print(msg)
    if _log_file is not None:
        _log_file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        _log_file.flush()


def gpu_mem_str():
    """Return a short string with current GPU memory usage."""
    if not torch.cuda.is_available():
        return "no-gpu"
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    return f"GPU mem: {a:.1f}GB alloc / {r:.1f}GB reserved"


# ===================================================================
# Smoke test wrapper
# ===================================================================

def run_smoke_test(name, fn, n_samples=2):
    """Run fn on a small sample.  Returns True if it passes, False + prints error if not."""
    log(f"\n[smoke test] {name} ({n_samples} samples)...")
    try:
        t0 = time.time()
        fn()
        dt = time.time() - t0
        log(f"[smoke test] {name}: PASSED ({dt:.1f}s).  {gpu_mem_str()}")
        return True
    except Exception as e:
        import traceback
        log(f"[smoke test] {name}: FAILED — {e}")
        log(traceback.format_exc())
        return False
