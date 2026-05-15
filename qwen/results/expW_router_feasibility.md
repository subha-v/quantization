# Exp W — Router feasibility postprocess on Exp V1 rollouts

Loaded retrieval: 261 items, tier-2 portfolio = ['V15', 'V4', 'V9', 'V10', 'V11', 'V12', 'V5', 'V6', 'V7']
  Stage-A label distribution: escalate=1: 47, escalate=0: 214
  S4 alone acc: 0.579; oracle (S4 + extras): 0.759; oracle lift over S4: +18.0 pp

Loaded reasoning: 120 items, tier-2 portfolio = ['V4', 'V10', 'V11', 'V12', 'V5', 'V6', 'V7']
  Stage-A label distribution: escalate=1: 24, escalate=0: 96
  S4 alone acc: 0.567; oracle (S4 + extras): 0.767; oracle lift over S4: +20.0 pp

Loaded lvb: 200 items, tier-2 portfolio = ['V4', 'V9', 'V11', 'V12', 'V5']
  Stage-A label distribution: escalate=1: 13, escalate=0: 187
  S4 alone acc: 0.680; oracle (S4 + extras): 0.745; oracle lift over S4: +6.5 pp


## Split: retrieval 70/30 (seed=0)  (n_train=182, n_test=79)
  baselines (test): V10=0.557, V11=0.608, V12=0.595, V15=0.608, V3=0.582, V4=0.608, V5=0.608, V6=0.608, V7=0.595, V9=0.582
  oracle (test): 0.722
  oracle lift over best static: +11.4 pp

  | router | classifier | stage_b | thresh | test_acc | mean_KV | lift_vs_best_static | frac_oracle_lift_recovered |
  |---|---|---|---:|---:|---:|---:|---:|
  | always-S4 | — | — | — | 0.582 | 4.188 | -2.5pp | -22.2% |
  | always-V15 | — | — | — | 0.608 | 4.234 | +0.0pp | 0.0% |
  | always-V11 | — | — | — | 0.608 | 4.281 | +0.0pp | 0.0% |
  | always-best-static | — | — | — | 0.608 | 4.234 | +0.0pp | 0.0% |
  | uniform-random | — | — | — | 0.595 | 4.265 | -2.5pp | -22.2% |
  | one-pass | gbm | default_v15 | 0.3 | 0.595 | 4.196 | -1.3pp | -11.1% |
  | one-pass | gbm | default_v15 | 0.5 | 0.582 | 4.193 | -2.5pp | -22.2% |
  | one-pass | gbm | default_v15 | 0.7 | 0.582 | 4.192 | -2.5pp | -22.2% |
  | one-pass | gbm | default_best | 0.3 | 0.595 | 4.204 | -1.3pp | -11.1% |
  | one-pass | gbm | default_best | 0.5 | 0.582 | 4.199 | -2.5pp | -22.2% |
  | one-pass | gbm | default_best | 0.7 | 0.582 | 4.197 | -2.5pp | -22.2% |
  | one-pass | gbm | classifier | 0.3 | 0.595 | 4.201 | -1.3pp | -11.1% |
  | one-pass | gbm | classifier | 0.5 | 0.582 | 4.198 | -2.5pp | -22.2% |
  | one-pass | gbm | classifier | 0.7 | 0.582 | 4.195 | -2.5pp | -22.2% |
  | one-pass | lr | default_v15 | 0.3 | 0.582 | 4.193 | -2.5pp | -22.2% |
  | one-pass | lr | default_v15 | 0.5 | 0.582 | 4.188 | -2.5pp | -22.2% |
  | one-pass | lr | default_v15 | 0.7 | 0.582 | 4.188 | -2.5pp | -22.2% |
  | one-pass | lr | default_best | 0.3 | 0.582 | 4.199 | -2.5pp | -22.2% |
  | one-pass | lr | default_best | 0.5 | 0.582 | 4.188 | -2.5pp | -22.2% |
  | one-pass | lr | default_best | 0.7 | 0.582 | 4.188 | -2.5pp | -22.2% |
  | one-pass | lr | classifier | 0.3 | 0.582 | 4.196 | -2.5pp | -22.2% |
  | one-pass | lr | classifier | 0.5 | 0.582 | 4.188 | -2.5pp | -22.2% |
  | one-pass | lr | classifier | 0.7 | 0.582 | 4.188 | -2.5pp | -22.2% |
  | two-pass | gbm | default_v15 | 0.3 | 0.595 | 4.193 | -1.3pp | -11.1% |
  | two-pass | gbm | default_v15 | 0.5 | 0.595 | 4.192 | -1.3pp | -11.1% |
  | two-pass | gbm | default_v15 | 0.7 | 0.582 | 4.190 | -2.5pp | -22.2% |
  | two-pass | gbm | default_best | 0.3 | 0.582 | 4.199 | -2.5pp | -22.2% |
  | two-pass | gbm | default_best | 0.5 | 0.582 | 4.196 | -2.5pp | -22.2% |
  | two-pass | gbm | default_best | 0.7 | 0.570 | 4.193 | -3.8pp | -33.3% |
  | two-pass | gbm | classifier | 0.3 | 0.595 | 4.195 | -1.3pp | -11.1% |
  | two-pass | gbm | classifier | 0.5 | 0.582 | 4.193 | -2.5pp | -22.2% |
  | two-pass | gbm | classifier | 0.7 | 0.582 | 4.190 | -2.5pp | -22.2% |
  | two-pass | lr | default_v15 | 0.3 | 0.595 | 4.197 | -1.3pp | -11.1% |
  | two-pass | lr | default_v15 | 0.5 | 0.570 | 4.190 | -3.8pp | -33.3% |
  | two-pass | lr | default_v15 | 0.7 | 0.582 | 4.188 | -2.5pp | -22.2% |
  | two-pass | lr | default_best | 0.3 | 0.582 | 4.206 | -2.5pp | -22.2% |
  | two-pass | lr | default_best | 0.5 | 0.570 | 4.193 | -3.8pp | -33.3% |
  | two-pass | lr | default_best | 0.7 | 0.582 | 4.188 | -2.5pp | -22.2% |
  | two-pass | lr | classifier | 0.3 | 0.608 | 4.200 | +0.0pp | 0.0% |
  | two-pass | lr | classifier | 0.5 | 0.595 | 4.192 | -1.3pp | -11.1% |
  | two-pass | lr | classifier | 0.7 | 0.582 | 4.188 | -2.5pp | -22.2% |


## Split: retrieval 70/30 (seed=1)  (n_train=182, n_test=79)
  baselines (test): V10=0.582, V11=0.608, V12=0.595, V15=0.633, V3=0.582, V4=0.570, V5=0.620, V6=0.646, V7=0.646, V9=0.608
  oracle (test): 0.772
  oracle lift over best static: +12.7 pp

  | router | classifier | stage_b | thresh | test_acc | mean_KV | lift_vs_best_static | frac_oracle_lift_recovered |
  |---|---|---|---:|---:|---:|---:|---:|
  | always-S4 | — | — | — | 0.582 | 4.188 | -6.3pp | -50.0% |
  | always-V15 | — | — | — | 0.633 | 4.234 | -1.3pp | -10.0% |
  | always-V11 | — | — | — | 0.608 | 4.281 | -3.8pp | -30.0% |
  | always-best-static | — | — | — | 0.646 | 4.281 | +0.0pp | 0.0% |
  | uniform-random | — | — | — | 0.582 | 4.265 | -5.1pp | -40.0% |
  | one-pass | gbm | default_v15 | 0.3 | 0.608 | 4.198 | -3.8pp | -30.0% |
  | one-pass | gbm | default_v15 | 0.5 | 0.608 | 4.194 | -3.8pp | -30.0% |
  | one-pass | gbm | default_v15 | 0.7 | 0.582 | 4.191 | -6.3pp | -50.0% |
  | one-pass | gbm | default_best | 0.3 | 0.608 | 4.209 | -3.8pp | -30.0% |
  | one-pass | gbm | default_best | 0.5 | 0.608 | 4.201 | -3.8pp | -30.0% |
  | one-pass | gbm | default_best | 0.7 | 0.582 | 4.195 | -6.3pp | -50.0% |
  | one-pass | gbm | classifier | 0.3 | 0.608 | 4.201 | -3.8pp | -30.0% |
  | one-pass | gbm | classifier | 0.5 | 0.608 | 4.195 | -3.8pp | -30.0% |
  | one-pass | gbm | classifier | 0.7 | 0.582 | 4.191 | -6.3pp | -50.0% |
  | one-pass | lr | default_v15 | 0.3 | 0.582 | 4.190 | -6.3pp | -50.0% |
  | one-pass | lr | default_v15 | 0.5 | 0.582 | 4.188 | -6.3pp | -50.0% |
  | one-pass | lr | default_v15 | 0.7 | 0.582 | 4.188 | -6.3pp | -50.0% |
  | one-pass | lr | default_best | 0.3 | 0.582 | 4.192 | -6.3pp | -50.0% |
  | one-pass | lr | default_best | 0.5 | 0.582 | 4.188 | -6.3pp | -50.0% |
  | one-pass | lr | default_best | 0.7 | 0.582 | 4.188 | -6.3pp | -50.0% |
  | one-pass | lr | classifier | 0.3 | 0.582 | 4.190 | -6.3pp | -50.0% |
  | one-pass | lr | classifier | 0.5 | 0.582 | 4.188 | -6.3pp | -50.0% |
  | one-pass | lr | classifier | 0.7 | 0.582 | 4.188 | -6.3pp | -50.0% |
  | two-pass | gbm | default_v15 | 0.3 | 0.608 | 4.199 | -3.8pp | -30.0% |
  | two-pass | gbm | default_v15 | 0.5 | 0.595 | 4.194 | -5.1pp | -40.0% |
  | two-pass | gbm | default_v15 | 0.7 | 0.595 | 4.191 | -5.1pp | -40.0% |
  | two-pass | gbm | default_best | 0.3 | 0.595 | 4.211 | -5.1pp | -40.0% |
  | two-pass | gbm | default_best | 0.5 | 0.582 | 4.201 | -6.3pp | -50.0% |
  | two-pass | gbm | default_best | 0.7 | 0.582 | 4.195 | -6.3pp | -50.0% |
  | two-pass | gbm | classifier | 0.3 | 0.582 | 4.204 | -6.3pp | -50.0% |
  | two-pass | gbm | classifier | 0.5 | 0.582 | 4.198 | -6.3pp | -50.0% |
  | two-pass | gbm | classifier | 0.7 | 0.582 | 4.194 | -6.3pp | -50.0% |
  | two-pass | lr | default_v15 | 0.3 | 0.633 | 4.199 | -1.3pp | -10.0% |
  | two-pass | lr | default_v15 | 0.5 | 0.608 | 4.191 | -3.8pp | -30.0% |
  | two-pass | lr | default_v15 | 0.7 | 0.595 | 4.188 | -5.1pp | -40.0% |
  | two-pass | lr | default_best | 0.3 | 0.620 | 4.210 | -2.5pp | -20.0% |
  | two-pass | lr | default_best | 0.5 | 0.595 | 4.195 | -5.1pp | -40.0% |
  | two-pass | lr | default_best | 0.7 | 0.595 | 4.189 | -5.1pp | -40.0% |
  | two-pass | lr | classifier | 0.3 | 0.608 | 4.204 | -3.8pp | -30.0% |
  | two-pass | lr | classifier | 0.5 | 0.595 | 4.194 | -5.1pp | -40.0% |
  | two-pass | lr | classifier | 0.7 | 0.595 | 4.188 | -5.1pp | -40.0% |


## Split: train retrieval → test reasoning  (cross-task)  (n_train=261, n_test=120)
  baselines (test): V10=0.642, V11=0.617, V12=0.592, V3=0.567, V4=0.600, V5=0.542, V6=0.592, V7=0.608
  oracle (test): 0.767
  oracle lift over best static: +12.5 pp

  | router | classifier | stage_b | thresh | test_acc | mean_KV | lift_vs_best_static | frac_oracle_lift_recovered |
  |---|---|---|---:|---:|---:|---:|---:|
  | always-S4 | — | — | — | 0.567 | 4.188 | -7.5pp | -60.0% |
  | always-V11 | — | — | — | 0.617 | 4.281 | -2.5pp | -20.0% |
  | always-best-static | — | — | — | 0.642 | 4.281 | +0.0pp | 0.0% |
  | uniform-random | — | — | — | 0.567 | 4.267 | -4.2pp | -33.3% |
  | one-pass | gbm | default_v15 | 0.3 | 0.608 | 4.208 | -3.3pp | -26.7% |
  | one-pass | gbm | default_v15 | 0.5 | 0.592 | 4.197 | -5.0pp | -40.0% |
  | one-pass | gbm | default_v15 | 0.7 | 0.583 | 4.191 | -5.8pp | -46.7% |
  | one-pass | gbm | default_best | 0.3 | 0.625 | 4.208 | -1.7pp | -13.3% |
  | one-pass | gbm | default_best | 0.5 | 0.600 | 4.197 | -4.2pp | -33.3% |
  | one-pass | gbm | default_best | 0.7 | 0.583 | 4.191 | -5.8pp | -46.7% |
  | one-pass | gbm | classifier | 0.3 | 0.617 | 4.208 | -2.5pp | -20.0% |
  | one-pass | gbm | classifier | 0.5 | 0.592 | 4.197 | -5.0pp | -40.0% |
  | one-pass | gbm | classifier | 0.7 | 0.583 | 4.191 | -5.8pp | -46.7% |
  | one-pass | lr | default_v15 | 0.3 | 0.592 | 4.193 | -5.0pp | -40.0% |
  | one-pass | lr | default_v15 | 0.5 | 0.567 | 4.188 | -7.5pp | -60.0% |
  | one-pass | lr | default_v15 | 0.7 | 0.567 | 4.188 | -7.5pp | -60.0% |
  | one-pass | lr | default_best | 0.3 | 0.567 | 4.193 | -7.5pp | -60.0% |
  | one-pass | lr | default_best | 0.5 | 0.567 | 4.188 | -7.5pp | -60.0% |
  | one-pass | lr | default_best | 0.7 | 0.567 | 4.188 | -7.5pp | -60.0% |
  | one-pass | lr | classifier | 0.3 | 0.592 | 4.193 | -5.0pp | -40.0% |
  | one-pass | lr | classifier | 0.5 | 0.567 | 4.188 | -7.5pp | -60.0% |
  | one-pass | lr | classifier | 0.7 | 0.567 | 4.188 | -7.5pp | -60.0% |
  | two-pass | gbm | default_v15 | 0.3 | 0.583 | 4.222 | -5.8pp | -46.7% |
  | two-pass | gbm | default_v15 | 0.5 | 0.575 | 4.213 | -6.7pp | -53.3% |
  | two-pass | gbm | default_v15 | 0.7 | 0.583 | 4.194 | -5.8pp | -46.7% |
  | two-pass | gbm | default_best | 0.3 | 0.592 | 4.222 | -5.0pp | -40.0% |
  | two-pass | gbm | default_best | 0.5 | 0.575 | 4.213 | -6.7pp | -53.3% |
  | two-pass | gbm | default_best | 0.7 | 0.583 | 4.194 | -5.8pp | -46.7% |
  | two-pass | gbm | classifier | 0.3 | 0.600 | 4.222 | -4.2pp | -33.3% |
  | two-pass | gbm | classifier | 0.5 | 0.575 | 4.213 | -6.7pp | -53.3% |
  | two-pass | gbm | classifier | 0.7 | 0.583 | 4.194 | -5.8pp | -46.7% |
  | two-pass | lr | default_v15 | 0.3 | 0.583 | 4.210 | -5.8pp | -46.7% |
  | two-pass | lr | default_v15 | 0.5 | 0.575 | 4.191 | -6.7pp | -53.3% |
  | two-pass | lr | default_v15 | 0.7 | 0.567 | 4.188 | -7.5pp | -60.0% |
  | two-pass | lr | default_best | 0.3 | 0.583 | 4.210 | -5.8pp | -46.7% |
  | two-pass | lr | default_best | 0.5 | 0.583 | 4.191 | -5.8pp | -46.7% |
  | two-pass | lr | default_best | 0.7 | 0.567 | 4.188 | -7.5pp | -60.0% |
  | two-pass | lr | classifier | 0.3 | 0.600 | 4.210 | -4.2pp | -33.3% |
  | two-pass | lr | classifier | 0.5 | 0.592 | 4.191 | -5.0pp | -40.0% |
  | two-pass | lr | classifier | 0.7 | 0.567 | 4.188 | -7.5pp | -60.0% |


## Split: train retrieval+reasoning → test LVB  (cross-dataset)  (n_train=381, n_test=200)
  baselines (test): V11=0.700, V12=0.715, V3=0.680, V4=0.705, V5=0.680
  oracle (test): 0.745
  oracle lift over best static: +3.0 pp

  | router | classifier | stage_b | thresh | test_acc | mean_KV | lift_vs_best_static | frac_oracle_lift_recovered |
  |---|---|---|---:|---:|---:|---:|---:|
  | always-S4 | — | — | — | 0.680 | 4.188 | -3.5pp | -116.7% |
  | always-V11 | — | — | — | 0.700 | 4.281 | -1.5pp | -50.0% |
  | always-best-static | — | — | — | 0.715 | 4.281 | +0.0pp | 0.0% |
  | uniform-random | — | — | — | 0.675 | 4.263 | -1.0pp | -33.3% |
  | one-pass | gbm | default_v15 | 0.3 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | gbm | default_v15 | 0.5 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | gbm | default_v15 | 0.7 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | gbm | default_best | 0.3 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | gbm | default_best | 0.5 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | gbm | default_best | 0.7 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | gbm | classifier | 0.3 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | gbm | classifier | 0.5 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | gbm | classifier | 0.7 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | lr | default_v15 | 0.3 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | lr | default_v15 | 0.5 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | lr | default_v15 | 0.7 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | lr | default_best | 0.3 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | lr | default_best | 0.5 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | lr | default_best | 0.7 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | lr | classifier | 0.3 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | lr | classifier | 0.5 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | one-pass | lr | classifier | 0.7 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | two-pass | gbm | default_v15 | 0.3 | 0.695 | 4.199 | -2.0pp | -66.7% |
  | two-pass | gbm | default_v15 | 0.5 | 0.675 | 4.192 | -4.0pp | -133.3% |
  | two-pass | gbm | default_v15 | 0.7 | 0.675 | 4.189 | -4.0pp | -133.3% |
  | two-pass | gbm | default_best | 0.3 | 0.695 | 4.199 | -2.0pp | -66.7% |
  | two-pass | gbm | default_best | 0.5 | 0.675 | 4.192 | -4.0pp | -133.3% |
  | two-pass | gbm | default_best | 0.7 | 0.675 | 4.189 | -4.0pp | -133.3% |
  | two-pass | gbm | classifier | 0.3 | 0.705 | 4.199 | -1.0pp | -33.3% |
  | two-pass | gbm | classifier | 0.5 | 0.680 | 4.192 | -3.5pp | -116.7% |
  | two-pass | gbm | classifier | 0.7 | 0.680 | 4.189 | -3.5pp | -116.7% |
  | two-pass | lr | default_v15 | 0.3 | 0.685 | 4.196 | -3.0pp | -100.0% |
  | two-pass | lr | default_v15 | 0.5 | 0.680 | 4.191 | -3.5pp | -116.7% |
  | two-pass | lr | default_v15 | 0.7 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | two-pass | lr | default_best | 0.3 | 0.685 | 4.196 | -3.0pp | -100.0% |
  | two-pass | lr | default_best | 0.5 | 0.680 | 4.191 | -3.5pp | -116.7% |
  | two-pass | lr | default_best | 0.7 | 0.680 | 4.188 | -3.5pp | -116.7% |
  | two-pass | lr | classifier | 0.3 | 0.685 | 4.196 | -3.0pp | -100.0% |
  | two-pass | lr | classifier | 0.5 | 0.675 | 4.191 | -4.0pp | -133.3% |
  | two-pass | lr | classifier | 0.7 | 0.680 | 4.188 | -3.5pp | -116.7% |

