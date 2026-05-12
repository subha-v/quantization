# expI_smoke results — 2026-05-10 04:01:15

**Phase A:** PASS (6/6)
**Phase B:** PASS (2/2)

## Phase A — synthetic kernel checks

| Check | Result | Detail |
|---|---|---|
| tempwin_outlier_kernel_round_trip | PASS | outlier 8 channels preserved exactly (delta=0.00e+00); other channels quantized (delta=0.1436) |
| tempwin_window_boundaries | PASS | 6 segments at boundaries [0, 4, 8, 12, 16, 20, 24]; residuals per segment ['0.024', '0.227', '0.484', '0.906', '2.297', '0.095'] (max/min ratio 94.1) |
| vidkv_v_scale_differs_from_uniform | PASS | VidKV V differs from uniform INT4: max_abs_delta=1.7656 |
| tokenblock6_segment_count | PASS | token_block6: 6 distinct residuals; visual_only4+text: 6 distinct residuals |
| outlier8_subset_of_top16 | PASS | expF_kcalib_Qwen2.5-VL-7B-Instruct_frames64.npz: top16 shape=(28, 4, 16); top8 ⊆ top16 for all (L, H) |
| seed1_split_supersets | PASS | seed=1 split: n=64 (64) ⊂ n=200 (200) |

## Phase B — live-model checks

| Check | Result | Detail |
|---|---|---|
| visual_span_detection_seed1 | PASS | 8/8 items have v_end > v_start (spans first 4: [(15, 11535), (15, 11535), (15, 11535), (15, 11535)]) |
| bf16_vs_i_conditions_logits_differ | PASS | all 6 pairs differ; max-abs deltas: BF16-vs-I3=1.1452, BF16-vs-I7=0.6389, BF16-vs-I8=0.7218, I3-vs-I7=0.5063, I3-vs-I8=1.1377, I7-vs-I8=0.9886 |
