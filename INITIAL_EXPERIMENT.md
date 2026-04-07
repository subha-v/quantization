# Initial Experiment: pi0 Activation Distribution Profiling

## Purpose

Profile activation distributions across all modules of pi0 (a flow-matching VLA) to determine whether different modules — vision encoder, projector, VLM backbone, action expert, action projections — have meaningfully different distributions that would motivate module-specific quantization. This is the prerequisite for our research direction: does the flow-matching VLA's indirect error propagation pathway require action-aware quantization?

**Time budget:** < 1 hour on 1x H100 80GB

## Background

**pi0** is a flow-matching VLA by Physical Intelligence. Its architecture:

```
Image + Language → [SigLip Vision Encoder] → [Projector] → [PaliGemma VLM Backbone]
                                                                    ↓
                                                            KV cache (prefix)
                                                                    ↓ (read 10x during denoising)
Robot state + Noisy actions + Timestep → [Action Expert (Gemma)] → denoised actions
```

Key architectural detail: The VLM backbone and action expert share attention at every layer. At each transformer layer, Q/K/V from both modules are concatenated and processed through a single attention computation. The VLM's KV cache is computed once and then `deepcopy()`-ed and read by the action expert at each of 10 denoising steps. This means quantization errors in the VLM's KV cache get amplified through 10 repeated reads.

**QVLA (ICLR 2026)** showed that in AR VLAs (OpenVLA), the vision encoder is robust to quantization, the LLM backbone is moderate, and the projector/action head are extremely sensitive. We want to see if pi0's module distributions suggest similar or different patterns — especially for the action expert, which has no analogue in AR VLAs.

## Model Details

**Model:** `lerobot/pi0_base` (4B params, based on PaliGemma 3B + Gemma action expert)
**Also available:** `lerobot/pi0_libero` (fine-tuned on LIBERO benchmark)
**Source code:** `lerobot` library (pip install lerobot) — policy at `lerobot.policies.pi0`
**Alternative source:** https://github.com/Physical-Intelligence/openpi

### Module hierarchy and hook targets

```
PI0Policy
  └── model: PI0Pytorch
        ├── paligemma_with_expert: PaliGemmaWithExpertModel
        │     ├── paligemma (VLM backbone)
        │     │     └── model
        │     │           ├── vision_tower              ← HOOK: vision encoder
        │     │           ├── multi_modal_projector      ← HOOK: projector
        │     │           └── language_model             ← HOOK: LLM backbone
        │     │                 └── layers[0..N]
        │     │                       ├── self_attn (q_proj, k_proj, v_proj, o_proj)
        │     │                       └── mlp (up_proj, gate_proj, down_proj)
        │     └── gemma_expert (Action expert)           ← HOOK: action expert
        │           └── model
        │                 └── layers[0..N]
        │                       ├── self_attn (q_proj, k_proj, v_proj, o_proj)
        │                       └── mlp (up_proj, gate_proj, down_proj)
        ├── action_in_proj: nn.Linear                    ← HOOK: action input projection
        ├── action_out_proj: nn.Linear                   ← HOOK: action output projection
        ├── state_proj: nn.Linear                        ← HOOK: state projection
        ├── action_time_mlp_in: nn.Linear                ← HOOK: time embedding
        └── action_time_mlp_out: nn.Linear               ← HOOK: time embedding
```

## Steps

### Step 0: Environment setup

```bash
pip install lerobot torch numpy matplotlib scipy
# LeRobot will pull in transformers, accelerate, etc.
```

### Step 1: Load pi0

```python
import torch
import numpy as np
from collections import defaultdict
from lerobot.policies.pi0 import PI0Policy, PI0Config

# Option A: Load from HuggingFace
policy = PI0Policy.from_pretrained("lerobot/pi0_base")
policy.to("cuda")
policy.eval()
model = policy.model  # PI0Pytorch instance

# If Option A fails (e.g., config issues), try Option B:
# Option B: Load with explicit config
# config = PI0Config()
# policy = PI0Policy(config)
# policy.load_state_dict(torch.load("path/to/checkpoint"))
# policy.to("cuda")

print("Model loaded successfully")

# Verify module structure
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(f"  {name}: {module.in_features} -> {module.out_features}")
```

### Step 2: Register activation hooks

```python
activation_stats = defaultdict(list)

def compute_kurtosis(x):
    """Compute excess kurtosis of a tensor."""
    x_flat = x.flatten().float()
    mean = x_flat.mean()
    std = x_flat.std()
    if std < 1e-8:
        return 0.0
    z = (x_flat - mean) / std
    return (z ** 4).mean().item() - 3.0

def compute_outlier_fraction(x, sigma=3):
    """Fraction of values beyond sigma standard deviations."""
    x_flat = x.flatten().float()
    mean = x_flat.mean()
    std = x_flat.std()
    if std < 1e-8:
        return 0.0
    return ((x_flat - mean).abs() > sigma * std).float().mean().item()

def classify_pi0_module(name):
    """Classify a pi0 module by its functional role."""
    if 'vision_tower' in name:
        return 'vision_encoder'
    elif 'multi_modal_projector' in name:
        return 'projector'
    elif 'paligemma' in name and 'language_model' in name:
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name:
            return 'vlm_attention'
        elif 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name:
            return 'vlm_mlp'
        else:
            return 'vlm_other'
    elif 'gemma_expert' in name:
        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name or 'o_proj' in name:
            return 'action_expert_attention'
        elif 'gate_proj' in name or 'up_proj' in name or 'down_proj' in name:
            return 'action_expert_mlp'
        else:
            return 'action_expert_other'
    elif 'action_in_proj' in name:
        return 'action_in_proj'
    elif 'action_out_proj' in name:
        return 'action_out_proj'
    elif 'state_proj' in name:
        return 'state_proj'
    elif 'action_time_mlp' in name:
        return 'time_embedding'
    else:
        return 'other'

def make_hook(name):
    def hook_fn(module, input, output):
        x = input[0].detach().float()
        stats = {
            'mean_magnitude': x.abs().mean().item(),
            'max_magnitude': x.abs().max().item(),
            'variance': x.var().item(),
            'kurtosis': compute_kurtosis(x),
            'outlier_fraction_3sigma': compute_outlier_fraction(x, sigma=3),
            'outlier_fraction_6sigma': compute_outlier_fraction(x, sigma=6),
            'dynamic_range': (x.abs().max() / (x.abs().mean() + 1e-8)).item(),
            'num_elements': x.numel(),
        }
        activation_stats[name].append(stats)
    return hook_fn

# Register hooks on all linear layers
hooks = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        hooks.append(module.register_forward_hook(make_hook(name)))

print(f"Registered {len(hooks)} hooks")
```

### Step 3: Run inference

pi0 expects specific input format. We need to construct proper observations.

```python
# Option A: Use LIBERO environment for realistic inputs
# This requires: pip install libero
# See https://github.com/Lifelong-Robot-Learning/LIBERO

# Option B: Use synthetic inputs matching pi0's expected format
# pi0 expects observations with images and robot state

# Try to run with dummy data first to verify the pipeline works
# pi0's sample_actions() expects a dict with observation and task info

# The simplest approach: use lerobot's built-in data loading
from lerobot.datasets import LeRobotDataset

# Load a small dataset that pi0 was trained on
try:
    dataset = LeRobotDataset("lerobot/libero_10_demos")
    print(f"Loaded dataset with {len(dataset)} samples")
except:
    print("Could not load dataset, will use synthetic inputs")
    dataset = None

if dataset is not None:
    # Run inference on a few real samples
    for i in range(min(10, len(dataset))):
        batch = dataset[i]
        # Convert to proper format and add batch dimension
        batch = {k: v.unsqueeze(0).to("cuda") if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        with torch.no_grad():
            try:
                actions = policy.select_action(batch)
                print(f"Sample {i}: Generated action shape {actions.shape}")
            except Exception as e:
                print(f"Sample {i}: Error - {e}")
                # Try alternative input format
                break
else:
    # Synthetic inputs — construct what pi0 expects
    # pi0 takes images (B, C, H, W) and robot state (B, state_dim)
    # Check policy.config for expected dimensions
    print("Using synthetic inputs")
    print(f"Config: {policy.config}")
    
    # Construct dummy batch — adjust dimensions based on config
    for i in range(10):
        dummy_image = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float16)
        dummy_state = torch.randn(1, 14, device="cuda", dtype=torch.float16)  # typical robot state dim
        
        # You may need to adjust this based on pi0's expected input format
        # Check policy.config.input_shapes for the correct format
        with torch.no_grad():
            try:
                # Direct model call for profiling — we just need activations, not valid actions
                # The exact input format depends on the LeRobot version
                pass  # Will be filled in based on actual API
            except Exception as e:
                print(f"Error: {e}")
                break

print(f"\nCollected activation stats for {len(activation_stats)} layers")
print(f"Total forward passes recorded: {max(len(v) for v in activation_stats.values()) if activation_stats else 0}")
```

**IMPORTANT NOTE FOR THE IMPLEMENTING AGENT:** The exact input format for pi0 depends on the LeRobot version and dataset. If the above doesn't work:

1. Check `policy.config` to see expected input shapes
2. Look at `policy.select_action()` or `policy.forward()` signature
3. Check LeRobot docs: https://huggingface.co/docs/lerobot/en/pi0
4. As a fallback, directly call `model.paligemma_with_expert.paligemma` (the VLM backbone) with standard image+text inputs — this still gives you the VLM activation profile. Then separately profile the action expert with dummy inputs.
5. Another fallback: use the OpenPI codebase (https://github.com/Physical-Intelligence/openpi) which has more explicit inference examples.

### Step 4: Aggregate and analyze

```python
import json

# Aggregate stats across forward passes
aggregated = {}
for layer_name, stats_list in activation_stats.items():
    if not stats_list:
        continue
    agg = {}
    for key in stats_list[0].keys():
        if key == 'num_elements':
            agg[key] = stats_list[0][key]
            continue
        values = [s[key] for s in stats_list]
        agg[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
        }
    aggregated[layer_name] = agg

# Group by module type
module_groups = defaultdict(list)
for layer_name, stats in aggregated.items():
    module_type = classify_pi0_module(layer_name)
    module_groups[module_type].append({
        'name': layer_name,
        'stats': stats
    })

# Print summary table
print(f"\n{'Module Type':<30} {'Count':<8} {'Mean Mag':<12} {'Max Mag':<12} {'Dyn Range':<12} {'Kurtosis':<12} {'Outlier 3σ':<12}")
print("=" * 98)
for module_type in ['vision_encoder', 'projector', 'vlm_attention', 'vlm_mlp',
                     'action_expert_attention', 'action_expert_mlp', 
                     'action_in_proj', 'action_out_proj', 'state_proj', 'time_embedding']:
    if module_type not in module_groups:
        continue
    layers = module_groups[module_type]
    mean_mag = np.mean([l['stats']['mean_magnitude']['mean'] for l in layers])
    max_mag = np.mean([l['stats']['max_magnitude']['mean'] for l in layers])
    dyn_range = np.mean([l['stats']['dynamic_range']['mean'] for l in layers])
    kurtosis = np.mean([l['stats']['kurtosis']['mean'] for l in layers])
    outlier = np.mean([l['stats']['outlier_fraction_3sigma']['mean'] for l in layers])
    print(f"{module_type:<30} {len(layers):<8} {mean_mag:<12.4f} {max_mag:<12.4f} {dyn_range:<12.2f} {kurtosis:<12.2f} {outlier:<12.4f}")

# KEY COMPARISON: VLM backbone vs Action Expert
# This is the core question — do these two modules have different distributions?
if 'vlm_attention' in module_groups and 'action_expert_attention' in module_groups:
    print("\n" + "=" * 60)
    print("KEY COMPARISON: VLM Backbone vs Action Expert")
    print("=" * 60)
    
    for stat_name in ['mean_magnitude', 'max_magnitude', 'dynamic_range', 'kurtosis']:
        vlm_vals = [l['stats'][stat_name]['mean'] for l in module_groups['vlm_attention'] + module_groups.get('vlm_mlp', [])]
        expert_vals = [l['stats'][stat_name]['mean'] for l in module_groups['action_expert_attention'] + module_groups.get('action_expert_mlp', [])]
        
        vlm_mean = np.mean(vlm_vals)
        expert_mean = np.mean(expert_vals)
        ratio = expert_mean / (vlm_mean + 1e-8)
        
        print(f"  {stat_name}:")
        print(f"    VLM backbone:  {vlm_mean:.4f}")
        print(f"    Action expert: {expert_mean:.4f}")
        print(f"    Ratio (expert/VLM): {ratio:.2f}x")
```

### Step 5: Generate plots

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define consistent module order and colors
MODULE_ORDER = [
    'vision_encoder', 'projector', 
    'vlm_attention', 'vlm_mlp',
    'action_expert_attention', 'action_expert_mlp',
    'action_in_proj', 'action_out_proj', 'state_proj', 'time_embedding'
]
MODULE_COLORS = {
    'vision_encoder': '#2196F3',         # blue
    'projector': '#F44336',              # red
    'vlm_attention': '#4CAF50',          # green
    'vlm_mlp': '#8BC34A',               # light green
    'action_expert_attention': '#9C27B0', # purple
    'action_expert_mlp': '#CE93D8',      # light purple
    'action_in_proj': '#FF9800',         # orange
    'action_out_proj': '#FFB74D',        # light orange
    'state_proj': '#795548',             # brown
    'time_embedding': '#607D8B',         # gray
}

present_modules = [m for m in MODULE_ORDER if m in module_groups]

# Plot 1: Module-level comparison (box plots)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
stat_configs = [
    ('mean_magnitude', 'Mean Activation Magnitude'),
    ('dynamic_range', 'Dynamic Range (max/mean)'),
    ('kurtosis', 'Excess Kurtosis'),
    ('outlier_fraction_3sigma', 'Outlier Fraction (3σ)')
]

for ax, (stat_name, title) in zip(axes.flatten(), stat_configs):
    data = []
    labels = []
    colors = []
    for mod in present_modules:
        vals = [l['stats'][stat_name]['mean'] for l in module_groups[mod]]
        data.append(vals)
        labels.append(mod.replace('_', '\n'))
        colors.append(MODULE_COLORS.get(mod, 'gray'))
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.set_ylabel(stat_name)

plt.suptitle('pi0 Activation Distribution by Module Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('pi0_module_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pi0_module_comparison.png")

# Plot 2: Per-layer magnitude profile across full model
fig, ax = plt.subplots(figsize=(20, 6))

layer_names = list(aggregated.keys())
magnitudes = [aggregated[n]['mean_magnitude']['mean'] for n in layer_names]
colors = [MODULE_COLORS.get(classify_pi0_module(n), 'gray') for n in layer_names]

ax.bar(range(len(magnitudes)), magnitudes, color=colors, width=1.0, alpha=0.8)
ax.set_xlabel('Layer index')
ax.set_ylabel('Mean activation magnitude')
ax.set_title('pi0: Activation Magnitude Across All Linear Layers')

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=MODULE_COLORS[m], alpha=0.7, label=m) for m in present_modules]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('pi0_layer_magnitude_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pi0_layer_magnitude_profile.png")

# Plot 3: VLM vs Action Expert layer-by-layer comparison
# Extract per-layer stats for VLM and action expert attention layers
vlm_attn_layers = sorted(
    [l for l in module_groups.get('vlm_attention', [])],
    key=lambda x: x['name']
)
expert_attn_layers = sorted(
    [l for l in module_groups.get('action_expert_attention', [])],
    key=lambda x: x['name']
)

if vlm_attn_layers and expert_attn_layers:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, stat_name, title in zip(axes,
        ['mean_magnitude', 'dynamic_range', 'kurtosis'],
        ['Mean Magnitude', 'Dynamic Range', 'Kurtosis']):
        
        vlm_vals = [l['stats'][stat_name]['mean'] for l in vlm_attn_layers]
        expert_vals = [l['stats'][stat_name]['mean'] for l in expert_attn_layers]
        
        # Align by layer index (they may have different numbers of layers)
        ax.plot(range(len(vlm_vals)), vlm_vals, 'g-o', label='VLM backbone', markersize=4)
        ax.plot(range(len(expert_vals)), expert_vals, 'm-s', label='Action expert', markersize=4)
        ax.set_xlabel('Layer index')
        ax.set_ylabel(stat_name)
        ax.set_title(title)
        ax.legend()
    
    plt.suptitle('VLM Backbone vs Action Expert: Per-Layer Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pi0_vlm_vs_expert.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pi0_vlm_vs_expert.png")

# Plot 4: Quantization difficulty score per module
fig, ax = plt.subplots(figsize=(12, 6))

difficulty_data = []
difficulty_labels = []
difficulty_colors = []

for mod in present_modules:
    layers = module_groups[mod]
    scores = [l['stats']['dynamic_range']['mean'] * (1 + abs(l['stats']['kurtosis']['mean'])) 
              for l in layers]
    difficulty_data.append(scores)
    difficulty_labels.append(mod.replace('_', '\n'))
    difficulty_colors.append(MODULE_COLORS.get(mod, 'gray'))

bp = ax.boxplot(difficulty_data, labels=difficulty_labels, patch_artist=True)
for patch, color in zip(bp['boxes'], difficulty_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title('Estimated Quantization Difficulty by Module\n(dynamic_range × (1 + |kurtosis|))', fontweight='bold')
ax.set_ylabel('Difficulty score (higher = harder to quantize uniformly)')
ax.tick_params(axis='x', rotation=45, labelsize=8)

plt.tight_layout()
plt.savefig('pi0_quantization_difficulty.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pi0_quantization_difficulty.png")
```

### Step 6: Save results

```python
# Save raw data
output = {}
for layer_name, stats in aggregated.items():
    output[layer_name] = {
        'module_type': classify_pi0_module(layer_name),
        'stats': stats
    }

with open('pi0_activation_profile.json', 'w') as f:
    json.dump(output, f, indent=2)
print("Saved: pi0_activation_profile.json")

# Save summary
summary = {}
for mod in present_modules:
    layers = module_groups[mod]
    summary[mod] = {
        'num_layers': len(layers),
        'mean_magnitude': float(np.mean([l['stats']['mean_magnitude']['mean'] for l in layers])),
        'max_magnitude': float(np.mean([l['stats']['max_magnitude']['mean'] for l in layers])),
        'dynamic_range': float(np.mean([l['stats']['dynamic_range']['mean'] for l in layers])),
        'kurtosis': float(np.mean([l['stats']['kurtosis']['mean'] for l in layers])),
        'outlier_3sigma': float(np.mean([l['stats']['outlier_fraction_3sigma']['mean'] for l in layers])),
    }

with open('pi0_module_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved: pi0_module_summary.json")

# Clean up hooks
for h in hooks:
    h.remove()
```

## Interpreting Results

### Strong signal (direction is promising):

1. **VLM and action expert have very different distributions:** The action expert layers show significantly different magnitude, dynamic range, or kurtosis compared to VLM backbone layers. This means uniform quantization across both modules is suboptimal — you'd want different precision or different BlockDialect formatbooks for each.

2. **Action-interface layers are outlier-heavy:** The `action_in_proj`, `action_out_proj`, and `state_proj` layers have high dynamic range or kurtosis. These are the exact layers that bridge the VLM to the action expert. If they're hard to quantize, that explains why standard LLM quantization breaks VLAs (matching QVLA's finding about projectors).

3. **Heterogeneity within the action expert across denoising steps:** If you ran multiple denoising steps, check whether the action expert's activation distributions change across steps. If early denoising steps have different distributions than late steps, this motivates denoising-step-aware precision.

### Weak signal (reconsider):

1. **All modules look roughly the same:** VLM backbone and action expert have similar distributions. This would suggest that standard LLM quantization might transfer to flow-matching VLAs without modification.

## Fallback Plan

If loading pi0 via LeRobot proves too complex within the time budget:

1. **Try the OpenPI codebase:** Clone https://github.com/Physical-Intelligence/openpi and follow their inference example. It has more explicit model loading.

2. **Try GR00T-N1.6 instead:** Available at `nvidia/GR00T-N1.6-3B` on HuggingFace. It's another flow-matching VLA with a DiT-based action head. The profiling methodology is identical — just change the module classification function.

3. **Profile just the PaliGemma backbone:** Load `google/paligemma-3b-pt-224` directly and profile it. This gives you the VLM side of the story. Then argue to Wonsuk that the next step is profiling the full pi0 with the action expert included.

The key deliverable is the **plots and summary table** comparing activation distributions across VLA modules. Even partial results (e.g., just the VLM backbone) are useful for the meeting.
