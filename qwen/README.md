# Qwen2.5-VL × LongVideoBench — AttnEntropy KV-cache experiments

Pivot from the pi0.5/LIBERO line (saturated benchmark; static W2 protection
already wins) to a setting where AttnEntropy can have headroom: long-video VLM
KV-cache quantization. The novelty thesis becomes:

> Attention entropy can allocate per-(layer, head, token) KV-cache **precision**
> — not just cache size — better than uniform / random / attention-mass / MEDA
> baselines.

## Layout

```
qwen/
  scripts/
    setup_qwen_env.sh              # one-time venv + deps install
    data_longvideobench.py         # loader + stratified 100-cal / 200-eval split
    fake_quant_kv_cache.py         # FakeQuantKVCache + BitController (the core primitive)
    attn_entropy_hook.py           # multi-layer attention-entropy hook
    run_inference.py               # MCQ scorer (4-way logprob over A/B/C/D)
    calibrate.py                   # frozen layer/head/token thresholds from cal split
    expA_baseline.py               # 8 conditions × frame budgets
    expB_attnentropy.py            # 9 conditions @ matched avg KV bits
    expB_pareto_plot.py            # avg KV bits vs accuracy Pareto curve
    run_smoke.sh                   # 3B + 10 items, with hard logits-differ assertion
    run_main.sh                    # 7B + 200-eval set
  calibration/                     # split_seed0.json + frozen threshold JSONs
  results/                         # per-sample JSONL + summary markdown
  plots/                           # Pareto PNGs
```

## Quickstart

On `tambe-server-1`, inside `/data/subha2/quantization/qwen/scripts`:

```bash
# 1. one-time setup (creates venv at /data/subha2/experiments/qwen_venv)
bash setup_qwen_env.sh
source /data/subha2/experiments/qwen_venv/bin/activate

# 2. download LongVideoBench-val under /data/subha2/longvideobench/
#    (datasets library will cache automatically on first run; or place
#     lvb_val.json + videos/<id>.mp4 manually)

# 3. nvidia-smi gate, then smoke test
nvidia-smi
export CUDA_VISIBLE_DEVICES=<unused-gpu>
bash run_smoke.sh
# -> the smoke test asserts BF16 vs INT2 first-token logits differ. If that
#    assertion fails, do not proceed. See "Correctness invariant" below.

# 4. main runs
bash run_main.sh
```

## Conditions

### Experiment A — baseline KV-quant sensitivity

| # | Weights | KV | Purpose |
|---|---|---|---|
| A1 | BF16 | BF16 | Accuracy ceiling |
| A2 | W4 fake-quant | BF16 | Weight-only (LIBERO methodology) |
| A3 | AWQ checkpoint | BF16 | Weight-only (real, end-to-end) |
| A4 | BF16 | FP8 | Mild KV |
| A5 | BF16 | INT4 | Aggressive uniform KV |
| A6 | BF16 | INT4-K / INT8-V | Asymmetric |
| A7 | BF16 | INT2 (ternary) | Sub-2-bit stress |
| A8 | AWQ | INT4 | Realistic combined |

### Experiment B — controllers at matched ~3 avg KV bits

| # | Method | Granularity |
|---|---|---|
| B0 | Uniform INT4 | — (Pareto anchor) |
| B1 | Uniform INT2 | — (Pareto anchor) |
| B2 | Random mixed INT2/INT4 | layer (3 seeds) |
| B3 | Attention-mass token protection | per-token mask |
| B4 | MEDA-style entropy | layer (cache-eviction analogue → bit budget) |
| B5 | Oracle (logit-KL) | layer — privileged reference |
| **B6** | **AttnEntropy V1** | **layer** |
| **B7** | **AttnEntropy V2** | **(layer, KV-head)** |
| **B8** | **AttnEntropy V3** | **(layer, token) via online attention-mass mask** |

## Correctness invariant

For multiple-choice scoring with `max_new_tokens=1`, the scored logit is produced
by the prefill forward pass. For our KV-quant to actually affect the score,
quantization must happen **before** the attention matmul.

We rely on Qwen2.5-VL's SDPA forward routing: `past_key_value.update(...)`
returns the (now-quantized) concatenated cache, and the returned tensors feed
the SDPA matmul. So our `FakeQuantKVCache.update()` quantizing on write *does*
affect first-token logits.

`run_smoke.sh` asserts this invariant by computing
`||logits_BF16 − logits_INT2||_∞` per item; the test fails loud if max diff
≤ 1e-3. If the upstream attention backend ever changes (e.g. flash-attn-2
default), `install_explicit_kv_quant_patch()` in `fake_quant_kv_cache.py`
provides a fallback that monkey-patches each layer's `self_attn.forward`.

## GQA note

Qwen2.5-VL-7B has 28 Q-heads, 4 KV-heads, head_dim=128, 28 layers. The cache
sees post-RoPE K/V at `[B, 4, seq, 128]`. **V2 operates over KV-heads (length 4),
not Q-heads (28)** — each KV-head is shared by 7 Q-heads under GQA. Per-Q-head
precision is not directly expressible without re-quantizing post-`repeat_kv`,
which is out of scope.
