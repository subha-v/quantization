# Initial Experiment: Does Standard Quantization Break pi0?

## The Question

**Does applying off-the-shelf weight quantization (AWQ W4A16) to pi0's VLM backbone — while leaving the action expert in full precision — degrade task success rate on LIBERO?**

- If success drops >5%: standard LLM quantization is insufficient for flow-matching VLAs, motivating action-aware quantization research.
- If success drops <2%: the action expert buffers VLM quantization errors, meaning flow-matching VLAs are more robust than AR VLAs (also interesting, but different paper).

## Setup

**Model:** `lerobot/pi0_libero` (pi0 fine-tuned on LIBERO)
**Benchmark:** LIBERO (4 suites: Spatial, Object, Goal, Long), 10-20 episodes per suite
**Hardware:** 1x H100 80GB
**Time budget:** < 1 hour compute (main time is environment setup)

### pi0 Architecture Recap

```
Image + Language → [SigLip Vision Encoder] → [Projector] → [PaliGemma VLM (Gemma 3B)]
                                                                    ↓
                                                            KV cache (computed once)
                                                                    ↓ (deepcopy'd and read 10x)
Robot state + Noisy actions + Timestep → [Gemma Action Expert] → denoised trajectory
```

We quantize only the VLM backbone (vision encoder + projector + PaliGemma language model). The action expert stays in full precision. This isolates the question: does VLM quantization error propagate through the KV cache interface and degrade trajectories?

## Steps

### Step 0: Environment

```bash
# Core dependencies
pip install lerobot torch

# LIBERO simulation environment
pip install libero

# Quantization
pip install autoawq
# or: pip install auto-gptq
```

### Step 1: Baseline — Full Precision

Run pi0 on LIBERO in full precision. Record per-suite success rates and per-episode trajectories.

```python
import torch
import numpy as np
import json
from lerobot.policies.pi0 import PI0Policy

# Load pi0 fine-tuned on LIBERO
policy = PI0Policy.from_pretrained("lerobot/pi0_libero")
policy.to("cuda")
policy.eval()

# NOTE: The exact LIBERO evaluation loop depends on the LeRobot/LIBERO integration.
# The implementing agent should check:
#   1. LeRobot's evaluation scripts: `lerobot/scripts/eval.py`
#   2. LIBERO's evaluation API: https://github.com/Lifelong-Robot-Learning/LIBERO
#   3. OpenPI's LIBERO evaluation: https://github.com/Physical-Intelligence/openpi
#
# The general pattern is:

from libero.libero import benchmark

benchmark_dict = benchmark.get_benchmark_dict()
num_episodes = 20  # per suite

results = {}
trajectories_fp16 = {}

for suite_name in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_long']:
    task_suite = benchmark_dict[suite_name]()
    successes = []
    suite_trajectories = []
    
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        env = task_suite.get_env(task_id)  # MuJoCo environment
        
        for episode in range(num_episodes):
            obs = env.reset()
            trajectory = []
            done = False
            
            while not done:
                # Convert obs to pi0's expected format
                # (check policy.config.input_shapes for exact format)
                action = policy.select_action(obs)
                obs, reward, done, info = env.step(action)
                trajectory.append(action.cpu().numpy())
            
            success = info.get('success', reward > 0)
            successes.append(success)
            suite_trajectories.append(np.array(trajectory))
    
    success_rate = np.mean(successes)
    results[suite_name] = success_rate
    trajectories_fp16[suite_name] = suite_trajectories
    print(f"{suite_name}: {success_rate:.1%} ({sum(successes)}/{len(successes)})")

results['average'] = np.mean(list(results.values()))
print(f"\nAverage: {results['average']:.1%}")

with open('baseline_fp16_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Step 2: Quantize VLM Backbone

Apply AWQ W4A16 to the VLM backbone only. The action expert remains in full precision.

```python
from awq import AutoAWQForCausalLM

# The VLM backbone in pi0 is a PaliGemma model.
# We need to quantize it in-place within the pi0 model.
#
# Approach A: Quantize the VLM backbone weights directly
# Access the VLM backbone:
vlm_backbone = policy.model.paligemma_with_expert.paligemma

# Apply AWQ quantization to the language model weights
# AWQ needs a calibration dataset — use a small set of LIBERO observations
#
# Approach B: If AWQ doesn't directly support in-place quantization of a submodule,
# manually apply uniform INT4 quantization to all linear layers in the VLM backbone:

def quantize_linear_layer_w4(layer):
    """Apply simple W4 symmetric quantization to a linear layer's weights."""
    with torch.no_grad():
        w = layer.weight.data.float()
        # Per-channel symmetric quantization (4-bit = 16 levels)
        max_val = w.abs().amax(dim=1, keepdim=True)
        scale = max_val / 7.0  # symmetric int4: [-7, 7]
        scale = scale.clamp(min=1e-8)
        w_q = (w / scale).round().clamp(-7, 7)
        w_dequant = (w_q * scale).to(layer.weight.dtype)
        layer.weight.data = w_dequant

# Quantize all linear layers in the VLM backbone
vlm_modules = [
    policy.model.paligemma_with_expert.paligemma.model.language_model,
    policy.model.paligemma_with_expert.paligemma.model.multi_modal_projector,
]

quantized_count = 0
for parent_module in vlm_modules:
    for name, module in parent_module.named_modules():
        if isinstance(module, torch.nn.Linear):
            quantize_linear_layer_w4(module)
            quantized_count += 1

print(f"Quantized {quantized_count} linear layers in VLM backbone to W4")

# Verify action expert is untouched
expert = policy.model.paligemma_with_expert.gemma_expert
for name, module in expert.named_modules():
    if isinstance(module, torch.nn.Linear):
        # These should still be in original precision
        assert module.weight.dtype in [torch.float16, torch.bfloat16], \
            f"Action expert layer {name} was accidentally quantized!"

print("Action expert verified: still in full precision")
```

### Step 3: Evaluate Quantized Model

Run the exact same episodes with the quantized model.

```python
results_w4 = {}
trajectories_w4 = {}

for suite_name in ['libero_spatial', 'libero_object', 'libero_goal', 'libero_long']:
    task_suite = benchmark_dict[suite_name]()
    successes = []
    suite_trajectories = []
    
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        env = task_suite.get_env(task_id)
        
        for episode in range(num_episodes):
            obs = env.reset()
            trajectory = []
            done = False
            
            while not done:
                action = policy.select_action(obs)
                obs, reward, done, info = env.step(action)
                trajectory.append(action.cpu().numpy())
            
            success = info.get('success', reward > 0)
            successes.append(success)
            suite_trajectories.append(np.array(trajectory))
    
    success_rate = np.mean(successes)
    results_w4[suite_name] = success_rate
    trajectories_w4[suite_name] = suite_trajectories
    print(f"{suite_name}: {success_rate:.1%} ({sum(successes)}/{len(successes)})")

results_w4['average'] = np.mean([v for k, v in results_w4.items() if k != 'average'])
print(f"\nAverage: {results_w4['average']:.1%}")

with open('quantized_w4_results.json', 'w') as f:
    json.dump(results_w4, f, indent=2)
```

### Step 4: Compare and Compute Trajectory Divergence

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Success rate comparison
print("\n" + "=" * 60)
print("RESULTS: FP16 vs W4A16 (VLM backbone only)")
print("=" * 60)
print(f"{'Suite':<20} {'FP16':<10} {'W4A16':<10} {'Drop':<10}")
print("-" * 50)

suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_long']
for suite in suites:
    fp16 = results[suite]
    w4 = results_w4[suite]
    drop = fp16 - w4
    print(f"{suite:<20} {fp16:<10.1%} {w4:<10.1%} {drop:<+10.1%}")

avg_drop = results['average'] - results_w4['average']
print(f"{'AVERAGE':<20} {results['average']:<10.1%} {results_w4['average']:<10.1%} {avg_drop:<+10.1%}")

# Trajectory ADE (Average Displacement Error) between FP16 and W4 trajectories
print("\nTrajectory Divergence (ADE between FP16 and W4 trajectories):")
for suite in suites:
    ades = []
    for traj_fp16, traj_w4 in zip(trajectories_fp16[suite], trajectories_w4[suite]):
        min_len = min(len(traj_fp16), len(traj_w4))
        if min_len > 0:
            ade = np.mean(np.linalg.norm(traj_fp16[:min_len] - traj_w4[:min_len], axis=-1))
            ades.append(ade)
    if ades:
        print(f"  {suite}: mean ADE = {np.mean(ades):.4f}, std = {np.std(ades):.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart: success rates
x = np.arange(len(suites))
width = 0.35
axes[0].bar(x - width/2, [results[s] for s in suites], width, label='FP16', color='#2196F3')
axes[0].bar(x + width/2, [results_w4[s] for s in suites], width, label='W4A16 (VLM only)', color='#F44336')
axes[0].set_xlabel('LIBERO Suite')
axes[0].set_ylabel('Success Rate')
axes[0].set_title('Task Success: FP16 vs W4A16 VLM Backbone')
axes[0].set_xticks(x)
axes[0].set_xticklabels([s.replace('libero_', '') for s in suites])
axes[0].legend()
axes[0].set_ylim(0, 1.05)

# Bar chart: drop per suite
drops = [results[s] - results_w4[s] for s in suites]
colors = ['#F44336' if d > 0.05 else '#FF9800' if d > 0.02 else '#4CAF50' for d in drops]
axes[1].bar(range(len(suites)), drops, color=colors)
axes[1].set_xlabel('LIBERO Suite')
axes[1].set_ylabel('Success Rate Drop')
axes[1].set_title('Performance Drop from W4A16 Quantization')
axes[1].set_xticks(range(len(suites)))
axes[1].set_xticklabels([s.replace('libero_', '') for s in suites])
axes[1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% threshold')
axes[1].axhline(y=0.02, color='orange', linestyle='--', alpha=0.5, label='2% threshold')
axes[1].legend()

plt.suptitle(f'Overall drop: {avg_drop:+.1%}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('pi0_quantization_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: pi0_quantization_results.png")

# Save combined results
combined = {
    'fp16': results,
    'w4a16_vlm_only': results_w4,
    'average_drop': float(avg_drop),
    'verdict': 'PROMISING' if avg_drop > 0.05 else 'MARGINAL' if avg_drop > 0.02 else 'ROBUST'
}
with open('experiment_results.json', 'w') as f:
    json.dump(combined, f, indent=2)
print(f"\nVerdict: {combined['verdict']}")
```

## Interpreting Results

| Average Drop | Verdict | What it means for the research |
|---|---|---|
| >10% | Strong signal | Standard quantization catastrophically fails. Action-aware quantization is essential. Direct analogue to QVLA's findings, but for flow-matching. Strong paper. |
| 5-10% | Promising | Meaningful degradation. Room for action-aware methods to improve. Good motivation for BlockDialect extension. |
| 2-5% | Marginal | Flow-matching VLAs are partially robust. May need to push to W3 or W2 to find the breaking point. Story shifts to "how far can you push it?" |
| <2% | Robust | The action expert effectively buffers VLM errors. Standard LLM quantization works for flow-matching VLAs. Still publishable (surprising negative result) but different framing. |

**Also pay attention to per-suite variation.** If Spatial/Object tasks survive but Long tasks break, that's evidence of error accumulation over long horizons — the action expert amplifies small KV cache errors through many timesteps.

## What to Tell Wonsuk

Show the success rate comparison table and the bar chart. The conversation is:

- **If it breaks:** "Standard LLM quantization hurts pi0 by X%. The error propagates through the KV cache to the action expert. This motivates action-aware format selection — BlockDialect's formatbook could be tuned for trajectory quality instead of reconstruction error."

- **If it's robust:** "Flow-matching VLAs are surprisingly robust to W4 backbone quantization, unlike AR VLAs where QVLA showed >5% drops. The decoupled architecture (KV cache + separate action expert) acts as an error buffer. We can push to more aggressive quantization (W3, W2) to find the limit."

Either outcome gives you a clear next step.

## Practical Notes

- **Seed control:** Use the same random seeds for FP16 and W4 evaluations so you're comparing equivalent episodes.
- **Episode count:** 20 episodes per suite (80 total) is minimal but sufficient for a go/no-go signal. For a paper you'd want 50+ per suite.
- **If LIBERO setup is too complex:** Run pi0 inference without the environment — just feed it observations and compare the raw action outputs (trajectory ADE) between FP16 and W4. You lose the closed-loop success rate metric but still get the trajectory divergence signal.
- **If pi0_libero isn't available:** Use `pi0_base` and evaluate on any dataset/environment it supports. The key is quantize → measure actions, not the specific benchmark.
