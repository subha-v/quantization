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
# Data loading  (direct parquet → openpi observation format)
# ===================================================================
#
# Loads images + state + prompts directly from the downloaded parquet
# files at /data/subha2/libero_raw/.  No LeRobotDataset needed — avoids
# version compatibility issues entirely.

import io
import random as _random
from PIL import Image as _PILImage
import pyarrow.parquet as _pq

# Default data root — set via env var or override in function calls
LIBERO_DATA_ROOT = Path(os.environ.get("LIBERO_DATA_ROOT", os.path.join(WORKSPACE, "libero_raw")))


def suite_of(task_index: int) -> str:
    """Map task_index to suite name.

    Verified from tasks.jsonl: 0-9=Long, 10-19=Goal, 20-29=Object, 30-39=Spatial.
    """
    if task_index < 10:
        return "Long"
    if task_index < 20:
        return "Goal"
    if task_index < 30:
        return "Object"
    return "Spatial"


def load_task_prompts(data_root=None):
    """Load {task_index: language_instruction} from tasks.jsonl."""
    root = Path(data_root) if data_root else LIBERO_DATA_ROOT
    prompts = {}
    tasks_path = root / "meta" / "tasks.jsonl"
    with open(tasks_path) as f:
        for line in f:
            obj = json.loads(line)
            prompts[obj["task_index"]] = obj["task"]
    print(f"[data] {len(prompts)} task prompts loaded from {tasks_path}")
    return prompts


def _decode_img(cell) -> np.ndarray:
    """Decode a HuggingFace Image cell {bytes, path} to (H,W,3) uint8."""
    im = _PILImage.open(io.BytesIO(cell["bytes"])).convert("RGB")
    return np.array(im)


def load_libero_observations(n_easy=128, n_hard=128, seed=42, suite_map=None, data_root=None):
    """Load real LIBERO observations with decoded images from local parquets.

    Easy = Object (task 20-29), Hard = Long (task 0-9).
    Samples frames spread across episodes (early/mid/late per episode).

    Returns (observations, metadata) where:
        observations: list of openpi-formatted dicts with real images
        metadata: list of per-sample metadata for post-hoc analysis
    """
    root = Path(data_root) if data_root else LIBERO_DATA_ROOT
    rng = _random.Random(seed)
    prompts = load_task_prompts(root)

    # Index parquet files by suite (peek one row for task_index)
    files_by_suite = {"Long": [], "Object": []}
    pq_files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    print(f"[data] Found {len(pq_files)} parquet files in {root / 'data'}")

    for pq_file in pq_files:
        t = _pq.read_table(str(pq_file), columns=["task_index"])
        task_index = int(t["task_index"][0].as_py())
        suite = suite_of(task_index)
        if suite in files_by_suite:
            files_by_suite[suite].append((pq_file, task_index))

    print(f"[data] Long: {len(files_by_suite['Long'])} episodes, "
          f"Object: {len(files_by_suite['Object'])} episodes")

    def _sample_from(suite, n):
        eps = list(files_by_suite[suite])
        rng.shuffle(eps)
        obs_list, meta_list = [], []
        for pq_file, task_index in eps:
            if len(obs_list) >= n:
                break
            table = _pq.read_table(str(pq_file))
            ep_len = table.num_rows
            ep_idx = int(table["episode_index"][0].as_py())
            # Sample up to 8 frames spread across the episode
            n_frames = min(8, ep_len, n - len(obs_list))
            frame_idxs = sorted(rng.sample(range(ep_len), n_frames))
            for fi in frame_idxs:
                img = _decode_img(table["image"][fi].as_py())
                wrist = _decode_img(table["wrist_image"][fi].as_py())
                state = np.array(table["state"][fi].as_py(), dtype=np.float32)
                phase = fi / max(1, ep_len - 1)
                phase_bin = "early" if phase < 1 / 3 else ("mid" if phase < 2 / 3 else "late")
                obs_list.append({
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": state,
                    "prompt": prompts.get(task_index, "do something"),
                })
                meta_list.append({
                    "sample_idx": -1,  # filled below
                    "suite": suite,
                    "task_id": task_index,
                    "episode_id": ep_idx,
                    "frame_idx": fi,
                    "episode_length": ep_len,
                    "phase_bin": phase_bin,
                    "prompt": prompts.get(task_index, "do something"),
                })
        return obs_list, meta_list

    print(f"[data] Sampling {n_hard} hard (Long) + {n_easy} easy (Object)...")
    hard_obs, hard_meta = _sample_from("Long", n_hard)
    easy_obs, easy_meta = _sample_from("Object", n_easy)

    observations = hard_obs + easy_obs
    metadata = hard_meta + easy_meta
    for i, m in enumerate(metadata):
        m["sample_idx"] = i

    print(f"[data] Done. {len(observations)} observations with real images "
          f"({len(hard_obs)} Long + {len(easy_obs)} Object).")
    return observations, metadata


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
