# Exp V1 — Oracle-over-policies diagnostic

## Dataset: retrieval  (expV_rollouts_sliceV_retrieval.jsonl)
  Loaded 261 items; conds per item (first item): ['V0', 'V1', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']

### Policy set: extra8_structured  (|P|=7, n_items=261)
  policies: ['V4', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13']
  best_static: V9 (TV-8) acc=0.6322
  oracle_acc: 0.7011
  lift oracle vs best_static: +0.0690 (6.90 pp)
  F9 (V2) acc: 0.6054
  lift oracle vs F9: +0.0958 (9.58 pp)
  cheapest_correct_avg_kv_bits: 4.2812
  pick histogram: {'V4': 161, 'V8': 12, 'V9': 6, 'V11': 3, 'V12': 1}
  F9_wrong_lowbit_correct: 30  F9_correct_lowbit_wrong: 5  both_wrong: 73  both_right: 153
  per-context bucket oracle lift:
          long: n= 110  oracle=0.673  best=V11=0.618  lift=+5.5pp
           mid: n=  57  oracle=0.614  best=V9=0.561  lift=+5.3pp
         short: n=  94  oracle=0.787  best=V9=0.734  lift=+5.3pp
  per-num_images bucket oracle lift:
           1-4: n= 136  oracle=0.750  best=V9=0.691  lift=+5.9pp
         12-19: n=  38  oracle=0.763  best=V4=0.711  lift=+5.3pp
           5-7: n=  41  oracle=0.585  best=V9=0.537  lift=+4.9pp
          8-11: n=  46  oracle=0.609  best=V11=0.587  lift=+2.2pp

### Policy set: budget_ladder  (|P|=4, n_items=261)
  policies: ['V15', 'V11', 'V16', 'V17']
  best_static: V15 (BAL-4 (1/blk)) acc=0.6360
  oracle_acc: 0.6897
  lift oracle vs best_static: +0.0536 (5.36 pp)
  F9 (V2) acc: 0.6054
  lift oracle vs F9: +0.0843 (8.43 pp)
  cheapest_correct_avg_kv_bits: 4.2404
  pick histogram: {'V15': 166, 'V11': 8, 'V17': 3, 'V16': 3}
  F9_wrong_lowbit_correct: 28  F9_correct_lowbit_wrong: 6  both_wrong: 75  both_right: 152
  per-context bucket oracle lift:
          long: n= 110  oracle=0.691  best=V11=0.618  lift=+7.3pp
           mid: n=  57  oracle=0.561  best=V15=0.544  lift=+1.8pp
         short: n=  94  oracle=0.766  best=V15=0.734  lift=+3.2pp
  per-num_images bucket oracle lift:
           1-4: n= 136  oracle=0.713  best=V15=0.684  lift=+2.9pp
         12-19: n=  38  oracle=0.737  best=V16=0.684  lift=+5.3pp
           5-7: n=  41  oracle=0.610  best=V17=0.585  lift=+2.4pp
          8-11: n=  46  oracle=0.652  best=V11=0.587  lift=+6.5pp

### Policy set: lowbit_all  (|P|=12, n_items=261)
  policies: ['V3', 'V4', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17']
  best_static: V15 (BAL-4 (1/blk)) acc=0.6360
  oracle_acc: 0.7318
  lift oracle vs best_static: +0.0958 (9.58 pp)
  F9 (V2) acc: 0.6054
  lift oracle vs F9: +0.1264 (12.64 pp)
  cheapest_correct_avg_kv_bits: 4.2032
  pick histogram: {'V3': 151, 'V15': 22, 'V4': 6, 'V8': 3, 'V9': 3, 'V11': 2, 'V17': 2, 'V14': 1, 'V12': 1}
  F9_wrong_lowbit_correct: 37  F9_correct_lowbit_wrong: 4  both_wrong: 66  both_right: 154
  per-context bucket oracle lift:
          long: n= 110  oracle=0.718  best=V11=0.618  lift=+10.0pp
           mid: n=  57  oracle=0.667  best=V9=0.561  lift=+10.5pp
         short: n=  94  oracle=0.787  best=V9=0.734  lift=+5.3pp
  per-num_images bucket oracle lift:
           1-4: n= 136  oracle=0.765  best=V9=0.691  lift=+7.4pp
         12-19: n=  38  oracle=0.763  best=V4=0.711  lift=+5.3pp
           5-7: n=  41  oracle=0.659  best=V17=0.585  lift=+7.3pp
          8-11: n=  46  oracle=0.674  best=V11=0.587  lift=+8.7pp

### Policy set: lowbit_plus_f9  (|P|=13, n_items=261)
  policies: ['V2', 'V3', 'V4', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17']
  best_static: V15 (BAL-4 (1/blk)) acc=0.6360
  oracle_acc: 0.7471
  lift oracle vs best_static: +0.1111 (11.11 pp)
  F9 (V2) acc: 0.6054
  lift oracle vs F9: +0.1418 (14.18 pp)
  cheapest_correct_avg_kv_bits: 4.2144
  pick histogram: {'V3': 151, 'V15': 22, 'V4': 6, 'V2': 4, 'V8': 3, 'V9': 3, 'V11': 2, 'V17': 2, 'V14': 1, 'V12': 1}
  F9_wrong_lowbit_correct: 37  F9_correct_lowbit_wrong: 0  both_wrong: 66  both_right: 158
  per-context bucket oracle lift:
          long: n= 110  oracle=0.745  best=V11=0.618  lift=+12.7pp
           mid: n=  57  oracle=0.684  best=V9=0.561  lift=+12.3pp
         short: n=  94  oracle=0.787  best=V9=0.734  lift=+5.3pp
  per-num_images bucket oracle lift:
           1-4: n= 136  oracle=0.772  best=V9=0.691  lift=+8.1pp
         12-19: n=  38  oracle=0.763  best=V4=0.711  lift=+5.3pp
           5-7: n=  41  oracle=0.707  best=V17=0.585  lift=+12.2pp
          8-11: n=  46  oracle=0.696  best=V11=0.587  lift=+10.9pp

### Policy set: random8  (|P|=3, n_items=261)
  policies: ['V5', 'V6', 'V7']
  best_static: V7 (RND-s2) acc=0.6169
  oracle_acc: 0.6973
  lift oracle vs best_static: +0.0805 (8.05 pp)
  F9 (V2) acc: 0.6054
  lift oracle vs F9: +0.0920 (9.20 pp)
  cheapest_correct_avg_kv_bits: 4.2812
  pick histogram: {'V5': 158, 'V6': 14, 'V7': 10}
  F9_wrong_lowbit_correct: 33  F9_correct_lowbit_wrong: 9  both_wrong: 70  both_right: 149
  per-context bucket oracle lift:
          long: n= 110  oracle=0.682  best=V7=0.591  lift=+9.1pp
           mid: n=  57  oracle=0.579  best=V6=0.509  lift=+7.0pp
         short: n=  94  oracle=0.787  best=V7=0.734  lift=+5.3pp
  per-num_images bucket oracle lift:
           1-4: n= 136  oracle=0.735  best=V5=0.669  lift=+6.6pp
         12-19: n=  38  oracle=0.711  best=V7=0.684  lift=+2.6pp
           5-7: n=  41  oracle=0.610  best=V6=0.512  lift=+9.8pp
          8-11: n=  46  oracle=0.652  best=V5=0.565  lift=+8.7pp

### Policy set: structured3_blocks  (|P|=3, n_items=261)
  policies: ['V8', 'V9', 'V10']
  best_static: V9 (TV-8) acc=0.6322
  oracle_acc: 0.6667
  lift oracle vs best_static: +0.0345 (3.45 pp)
  F9 (V2) acc: 0.6054
  lift oracle vs F9: +0.0613 (6.13 pp)
  cheapest_correct_avg_kv_bits: 4.2812
  pick histogram: {'V8': 163, 'V9': 10, 'V10': 1}
  F9_wrong_lowbit_correct: 25  F9_correct_lowbit_wrong: 9  both_wrong: 78  both_right: 149
  per-context bucket oracle lift:
          long: n= 110  oracle=0.627  best=V8=0.591  lift=+3.6pp
           mid: n=  57  oracle=0.596  best=V9=0.561  lift=+3.5pp
         short: n=  94  oracle=0.755  best=V9=0.734  lift=+2.1pp
  per-num_images bucket oracle lift:
           1-4: n= 136  oracle=0.721  best=V9=0.691  lift=+2.9pp
         12-19: n=  38  oracle=0.711  best=V10=0.711  lift=+0.0pp
           5-7: n=  41  oracle=0.561  best=V9=0.537  lift=+2.4pp
          8-11: n=  46  oracle=0.565  best=V8=0.543  lift=+2.2pp

### CROSS-CUT — structured-3 (TT/TV/VT) oracle vs random-3 (V5/V6/V7) oracle
  structured-3 oracle: 0.6667   best_static=V9 (0.6322)  lift=3.45pp
  random-3 oracle:     0.6973   best_static=V7 (0.6169)  lift=8.05pp
  Δ(structured-3 oracle − random-3 oracle): +-0.0307 (-3.07 pp)

### CROSS-CUT — budget-ladder oracle vs V15 BAL4 static
  V15 BAL4 static: 0.6360
  budget-ladder oracle: 0.6897
  Δ(oracle − V15 static): +0.0536 (5.36 pp)
  pick histogram (oracle picks cheapest-correct in budget ladder):
    V15 (BAL-4 (1/blk)): 166
    V11 (BAL-8 (2/blk)): 8
    V17 (BAL-16 (4/blk)): 3
    V16 (BAL-12 (3/blk)): 3

### CROSS-CUT — extra8_structured oracle vs V11 BAL8 static
  V11 BAL8 static: 0.6322
  extra8_structured oracle: 0.7011
  Δ(oracle − V11 static): +0.0690 (6.90 pp)
  pick histogram:
    V4 (GEN-8): 161
    V8 (TT-8): 12
    V9 (TV-8): 6
    V11 (BAL-8 (2/blk)): 3
    V12 (MMNIAH-prior-8): 1


## Dataset: reasoning  (expV_rollouts_sliceV_reasoning.jsonl)
  Loaded 120 items; conds per item (first item): ['V0', 'V1', 'V10', 'V11', 'V12', 'V13', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']

### Policy set: extra8_structured  (|P|=5, n_items=120)
  policies: ['V4', 'V10', 'V11', 'V12', 'V13']
  best_static: V10 (VT-8) acc=0.6417
  oracle_acc: 0.7250
  lift oracle vs best_static: +0.0833 (8.33 pp)
  F9 (V2) acc: 0.6167
  lift oracle vs F9: +0.1083 (10.83 pp)
  cheapest_correct_avg_kv_bits: 4.2812
  pick histogram: {'V4': 72, 'V10': 10, 'V11': 3, 'V12': 1, 'V13': 1}
  F9_wrong_lowbit_correct: 17  F9_correct_lowbit_wrong: 4  both_wrong: 29  both_right: 70
  per-context bucket oracle lift:
          long: n=  35  oracle=0.657  best=V10=0.571  lift=+8.6pp
           mid: n=  19  oracle=0.632  best=V4=0.579  lift=+5.3pp
         short: n=  66  oracle=0.788  best=V11=0.758  lift=+3.0pp
  per-num_images bucket oracle lift:
           1-4: n=  73  oracle=0.767  best=V11=0.740  lift=+2.7pp
         12-19: n=  13  oracle=0.692  best=V4=0.692  lift=+0.0pp
           5-7: n=  16  oracle=0.688  best=V4=0.500  lift=+18.8pp
          8-11: n=  18  oracle=0.611  best=V10=0.556  lift=+5.6pp

### Policy set: budget_ladder  (|P|=1, n_items=120)
  policies: ['V11']
  best_static: V11 (BAL-8 (2/blk)) acc=0.6167
  oracle_acc: 0.6167
  lift oracle vs best_static: +0.0000 (0.00 pp)
  F9 (V2) acc: 0.6167
  lift oracle vs F9: +0.0000 (0.00 pp)
  cheapest_correct_avg_kv_bits: 4.2812
  pick histogram: {'V11': 74}
  F9_wrong_lowbit_correct: 9  F9_correct_lowbit_wrong: 9  both_wrong: 37  both_right: 65
  per-context bucket oracle lift:
          long: n=  35  oracle=0.400  best=V11=0.400  lift=+0.0pp
           mid: n=  19  oracle=0.526  best=V11=0.526  lift=+0.0pp
         short: n=  66  oracle=0.758  best=V11=0.758  lift=+0.0pp
  per-num_images bucket oracle lift:
           1-4: n=  73  oracle=0.740  best=V11=0.740  lift=+0.0pp
         12-19: n=  13  oracle=0.462  best=V11=0.462  lift=+0.0pp
           5-7: n=  16  oracle=0.438  best=V11=0.438  lift=+0.0pp
          8-11: n=  18  oracle=0.389  best=V11=0.389  lift=+0.0pp

### Policy set: lowbit_all  (|P|=6, n_items=120)
  policies: ['V3', 'V4', 'V10', 'V11', 'V12', 'V13']
  best_static: V10 (VT-8) acc=0.6417
  oracle_acc: 0.7417
  lift oracle vs best_static: +0.1000 (10.00 pp)
  F9 (V2) acc: 0.6167
  lift oracle vs F9: +0.1250 (12.50 pp)
  cheapest_correct_avg_kv_bits: 4.2096
  pick histogram: {'V3': 68, 'V4': 14, 'V10': 3, 'V11': 2, 'V12': 1, 'V13': 1}
  F9_wrong_lowbit_correct: 18  F9_correct_lowbit_wrong: 3  both_wrong: 28  both_right: 71
  per-context bucket oracle lift:
          long: n=  35  oracle=0.657  best=V10=0.571  lift=+8.6pp
           mid: n=  19  oracle=0.684  best=V4=0.579  lift=+10.5pp
         short: n=  66  oracle=0.803  best=V11=0.758  lift=+4.5pp
  per-num_images bucket oracle lift:
           1-4: n=  73  oracle=0.781  best=V11=0.740  lift=+4.1pp
         12-19: n=  13  oracle=0.692  best=V4=0.692  lift=+0.0pp
           5-7: n=  16  oracle=0.750  best=V4=0.500  lift=+25.0pp
          8-11: n=  18  oracle=0.611  best=V10=0.556  lift=+5.6pp

### Policy set: lowbit_plus_f9  (|P|=7, n_items=120)
  policies: ['V2', 'V3', 'V4', 'V10', 'V11', 'V12', 'V13']
  best_static: V10 (VT-8) acc=0.6417
  oracle_acc: 0.7667
  lift oracle vs best_static: +0.1250 (12.50 pp)
  F9 (V2) acc: 0.6167
  lift oracle vs F9: +0.1500 (15.00 pp)
  cheapest_correct_avg_kv_bits: 4.2272
  pick histogram: {'V3': 68, 'V4': 14, 'V10': 3, 'V2': 3, 'V11': 2, 'V12': 1, 'V13': 1}
  F9_wrong_lowbit_correct: 18  F9_correct_lowbit_wrong: 0  both_wrong: 28  both_right: 74
  per-context bucket oracle lift:
          long: n=  35  oracle=0.714  best=V10=0.571  lift=+14.3pp
           mid: n=  19  oracle=0.737  best=V4=0.579  lift=+15.8pp
         short: n=  66  oracle=0.803  best=V11=0.758  lift=+4.5pp
  per-num_images bucket oracle lift:
           1-4: n=  73  oracle=0.795  best=V11=0.740  lift=+5.5pp
         12-19: n=  13  oracle=0.769  best=V4=0.692  lift=+7.7pp
           5-7: n=  16  oracle=0.750  best=V2=0.500  lift=+25.0pp
          8-11: n=  18  oracle=0.667  best=V10=0.556  lift=+11.1pp

### Policy set: random8  (|P|=3, n_items=120)
  policies: ['V5', 'V6', 'V7']
  best_static: V7 (RND-s2) acc=0.6083
  oracle_acc: 0.7000
  lift oracle vs best_static: +0.0917 (9.17 pp)
  F9 (V2) acc: 0.6167
  lift oracle vs F9: +0.0833 (8.33 pp)
  cheapest_correct_avg_kv_bits: 4.2812
  pick histogram: {'V5': 65, 'V6': 10, 'V7': 9}
  F9_wrong_lowbit_correct: 15  F9_correct_lowbit_wrong: 5  both_wrong: 31  both_right: 69
  per-context bucket oracle lift:
          long: n=  35  oracle=0.629  best=V6=0.514  lift=+11.4pp
           mid: n=  19  oracle=0.579  best=V6=0.474  lift=+10.5pp
         short: n=  66  oracle=0.773  best=V7=0.742  lift=+3.0pp
  per-num_images bucket oracle lift:
           1-4: n=  73  oracle=0.753  best=V7=0.726  lift=+2.7pp
         12-19: n=  13  oracle=0.615  best=V6=0.538  lift=+7.7pp
           5-7: n=  16  oracle=0.625  best=V6=0.562  lift=+6.2pp
          8-11: n=  18  oracle=0.611  best=V7=0.500  lift=+11.1pp

### Policy set: structured3_blocks  (|P|=1, n_items=120)
  policies: ['V10']
  best_static: V10 (VT-8) acc=0.6417
  oracle_acc: 0.6417
  lift oracle vs best_static: +0.0000 (0.00 pp)
  F9 (V2) acc: 0.6167
  lift oracle vs F9: +0.0250 (2.50 pp)
  cheapest_correct_avg_kv_bits: 4.2812
  pick histogram: {'V10': 77}
  F9_wrong_lowbit_correct: 13  F9_correct_lowbit_wrong: 10  both_wrong: 33  both_right: 64
  per-context bucket oracle lift:
          long: n=  35  oracle=0.571  best=V10=0.571  lift=+0.0pp
           mid: n=  19  oracle=0.526  best=V10=0.526  lift=+0.0pp
         short: n=  66  oracle=0.712  best=V10=0.712  lift=+0.0pp
  per-num_images bucket oracle lift:
           1-4: n=  73  oracle=0.699  best=V10=0.699  lift=+0.0pp
         12-19: n=  13  oracle=0.692  best=V10=0.692  lift=+0.0pp
           5-7: n=  16  oracle=0.438  best=V10=0.438  lift=+0.0pp
          8-11: n=  18  oracle=0.556  best=V10=0.556  lift=+0.0pp

### CROSS-CUT — structured-3 (TT/TV/VT) oracle vs random-3 (V5/V6/V7) oracle
  structured-3 oracle: 0.6417   best_static=V10 (0.6417)  lift=0.00pp
  random-3 oracle:     0.7000   best_static=V7 (0.6083)  lift=9.17pp
  Δ(structured-3 oracle − random-3 oracle): +-0.0583 (-5.83 pp)

### CROSS-CUT — budget-ladder oracle vs V15 BAL4 static
  V15 BAL4 static: 0.0000
  budget-ladder oracle: 0.6167
  Δ(oracle − V15 static): +0.6167 (61.67 pp)
  pick histogram (oracle picks cheapest-correct in budget ladder):
    V11 (BAL-8 (2/blk)): 74

### CROSS-CUT — extra8_structured oracle vs V11 BAL8 static
  V11 BAL8 static: 0.6167
  extra8_structured oracle: 0.7250
  Δ(oracle − V11 static): +0.1083 (10.83 pp)
  pick histogram:
    V4 (GEN-8): 72
    V10 (VT-8): 10
    V11 (BAL-8 (2/blk)): 3
    V12 (MMNIAH-prior-8): 1
    V13 (LVB-prior-8): 1


## Dataset: lvb  (expV_lvb_stage3_seed2_normalized.jsonl)
  Loaded 200 items; conds per item (first item): ['V0', 'V1', 'V11', 'V12', 'V13', 'V14', 'V2', 'V3', 'V4', 'V5', 'V9']

### Policy set: extra8_structured  (|P|=5, n_items=200)
  policies: ['V4', 'V9', 'V11', 'V12', 'V13']
  best_static: V9 (TV-8) acc=0.7150
  oracle_acc: 0.7350
  lift oracle vs best_static: +0.0200 (2.00 pp)
  F9 (V2) acc: 0.6950
  lift oracle vs F9: +0.0400 (4.00 pp)
  cheapest_correct_avg_kv_bits: 4.2812
  pick histogram: {'V4': 141, 'V9': 6}
  F9_wrong_lowbit_correct: 11  F9_correct_lowbit_wrong: 3  both_wrong: 50  both_right: 136
  per-context bucket oracle lift:
             ?: n= 200  oracle=0.735  best=V9=0.715  lift=+2.0pp
  per-num_images bucket oracle lift:
             ?: n= 200  oracle=0.735  best=V9=0.715  lift=+2.0pp

### Policy set: budget_ladder  (|P|=1, n_items=200)
  policies: ['V11']
  best_static: V11 (BAL-8 (2/blk)) acc=0.7000
  oracle_acc: 0.7000
  lift oracle vs best_static: +0.0000 (0.00 pp)
  F9 (V2) acc: 0.6950
  lift oracle vs F9: +0.0050 (0.50 pp)
  cheapest_correct_avg_kv_bits: 4.2812
  pick histogram: {'V11': 140}
  F9_wrong_lowbit_correct: 6  F9_correct_lowbit_wrong: 5  both_wrong: 55  both_right: 134
  per-context bucket oracle lift:
             ?: n= 200  oracle=0.700  best=V11=0.700  lift=+0.0pp
  per-num_images bucket oracle lift:
             ?: n= 200  oracle=0.700  best=V11=0.700  lift=+0.0pp

### Policy set: lowbit_all  (|P|=7, n_items=200)
  policies: ['V3', 'V4', 'V9', 'V11', 'V12', 'V13', 'V14']
  best_static: V9 (TV-8) acc=0.7150
  oracle_acc: 0.7450
  lift oracle vs best_static: +0.0300 (3.00 pp)
  F9 (V2) acc: 0.6950
  lift oracle vs F9: +0.0500 (5.00 pp)
  cheapest_correct_avg_kv_bits: 4.1957
  pick histogram: {'V3': 136, 'V4': 9, 'V9': 4}
  F9_wrong_lowbit_correct: 13  F9_correct_lowbit_wrong: 3  both_wrong: 48  both_right: 136
  per-context bucket oracle lift:
             ?: n= 200  oracle=0.745  best=V9=0.715  lift=+3.0pp
  per-num_images bucket oracle lift:
             ?: n= 200  oracle=0.745  best=V9=0.715  lift=+3.0pp

### Policy set: lowbit_plus_f9  (|P|=8, n_items=200)
  policies: ['V2', 'V3', 'V4', 'V9', 'V11', 'V12', 'V13', 'V14']
  best_static: V9 (TV-8) acc=0.7150
  oracle_acc: 0.7600
  lift oracle vs best_static: +0.0450 (4.50 pp)
  F9 (V2) acc: 0.6950
  lift oracle vs F9: +0.0650 (6.50 pp)
  cheapest_correct_avg_kv_bits: 4.2066
  pick histogram: {'V3': 136, 'V4': 9, 'V9': 4, 'V2': 3}
  F9_wrong_lowbit_correct: 13  F9_correct_lowbit_wrong: 0  both_wrong: 48  both_right: 139
  per-context bucket oracle lift:
             ?: n= 200  oracle=0.760  best=V9=0.715  lift=+4.5pp
  per-num_images bucket oracle lift:
             ?: n= 200  oracle=0.760  best=V9=0.715  lift=+4.5pp

### Policy set: random8  (|P|=1, n_items=200)
  policies: ['V5']
  best_static: V5 (RND-s0) acc=0.6800
  oracle_acc: 0.6800
  lift oracle vs best_static: +0.0000 (0.00 pp)
  F9 (V2) acc: 0.6950
  lift oracle vs F9: +-0.0150 (-1.50 pp)
  cheapest_correct_avg_kv_bits: 4.2812
  pick histogram: {'V5': 136}
  F9_wrong_lowbit_correct: 5  F9_correct_lowbit_wrong: 8  both_wrong: 56  both_right: 131
  per-context bucket oracle lift:
             ?: n= 200  oracle=0.680  best=V5=0.680  lift=+0.0pp
  per-num_images bucket oracle lift:
             ?: n= 200  oracle=0.680  best=V5=0.680  lift=+0.0pp

### Policy set: structured3_blocks  (|P|=1, n_items=200)
  policies: ['V9']
  best_static: V9 (TV-8) acc=0.7150
  oracle_acc: 0.7150
  lift oracle vs best_static: +0.0000 (0.00 pp)
  F9 (V2) acc: 0.6950
  lift oracle vs F9: +0.0200 (2.00 pp)
  cheapest_correct_avg_kv_bits: 4.2812
  pick histogram: {'V9': 143}
  F9_wrong_lowbit_correct: 8  F9_correct_lowbit_wrong: 4  both_wrong: 53  both_right: 135
  per-context bucket oracle lift:
             ?: n= 200  oracle=0.715  best=V9=0.715  lift=+0.0pp
  per-num_images bucket oracle lift:
             ?: n= 200  oracle=0.715  best=V9=0.715  lift=+0.0pp

### CROSS-CUT — structured-3 (TT/TV/VT) oracle vs random-3 (V5/V6/V7) oracle
  structured-3 oracle: 0.7150   best_static=V9 (0.7150)  lift=0.00pp
  random-3 oracle:     0.6800   best_static=V5 (0.6800)  lift=0.00pp
  Δ(structured-3 oracle − random-3 oracle): +0.0350 (3.50 pp)

### CROSS-CUT — budget-ladder oracle vs V15 BAL4 static
  V15 BAL4 static: 0.0000
  budget-ladder oracle: 0.7000
  Δ(oracle − V15 static): +0.7000 (70.00 pp)
  pick histogram (oracle picks cheapest-correct in budget ladder):
    V11 (BAL-8 (2/blk)): 140

### CROSS-CUT — extra8_structured oracle vs V11 BAL8 static
  V11 BAL8 static: 0.7000
  extra8_structured oracle: 0.7350
  Δ(oracle − V11 static): +0.0350 (3.50 pp)
  pick histogram:
    V4 (GEN-8): 141
    V9 (TV-8): 6



## Cross-dataset oracle-over-policies summary

| dataset | set | n | best_static | best_acc | oracle_acc | lift | cheapest_correct_KV |
|---|---|---:|---|---:|---:|---:|---:|
| retrieval | extra8_structured | 261 | V9 | 0.632 | 0.701 | +6.9pp | 4.281 |
| retrieval | budget_ladder | 261 | V15 | 0.636 | 0.690 | +5.4pp | 4.240 |
| retrieval | lowbit_all | 261 | V15 | 0.636 | 0.732 | +9.6pp | 4.203 |
| retrieval | lowbit_plus_f9 | 261 | V15 | 0.636 | 0.747 | +11.1pp | 4.214 |
| retrieval | random8 | 261 | V7 | 0.617 | 0.697 | +8.0pp | 4.281 |
| retrieval | structured3_blocks | 261 | V9 | 0.632 | 0.667 | +3.4pp | 4.281 |
| reasoning | extra8_structured | 120 | V10 | 0.642 | 0.725 | +8.3pp | 4.281 |
| reasoning | budget_ladder | 120 | V11 | 0.617 | 0.617 | +0.0pp | 4.281 |
| reasoning | lowbit_all | 120 | V10 | 0.642 | 0.742 | +10.0pp | 4.210 |
| reasoning | lowbit_plus_f9 | 120 | V10 | 0.642 | 0.767 | +12.5pp | 4.227 |
| reasoning | random8 | 120 | V7 | 0.608 | 0.700 | +9.2pp | 4.281 |
| reasoning | structured3_blocks | 120 | V10 | 0.642 | 0.642 | +0.0pp | 4.281 |
| lvb | extra8_structured | 200 | V9 | 0.715 | 0.735 | +2.0pp | 4.281 |
| lvb | budget_ladder | 200 | V11 | 0.700 | 0.700 | +0.0pp | 4.281 |
| lvb | lowbit_all | 200 | V9 | 0.715 | 0.745 | +3.0pp | 4.196 |
| lvb | lowbit_plus_f9 | 200 | V9 | 0.715 | 0.760 | +4.5pp | 4.207 |
| lvb | random8 | 200 | V5 | 0.680 | 0.680 | +0.0pp | 4.281 |
| lvb | structured3_blocks | 200 | V9 | 0.715 | 0.715 | +0.0pp | 4.281 |
