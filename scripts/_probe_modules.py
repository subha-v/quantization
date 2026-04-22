import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils

policy, model = utils.load_policy("pi05_libero")
names = [n for n, _ in model.named_modules()]
print("TOP-LEVEL SUBTREE (depth<=3, filtered):")
for n in names:
    depth = n.count(".")
    if depth <= 3:
        print(" ", n)

print("\nSAMPLE language_model paths:")
lm = [n for n in names if "language_model" in n][:20]
for n in lm: print(" ", n)

print("\nSAMPLE vision_tower paths:")
vt = [n for n in names if "vision_tower" in n][:10]
for n in vt: print(" ", n)

print("\nSAMPLE gemma_expert paths:")
ge = [n for n in names if "gemma_expert" in n][:20]
for n in ge: print(" ", n)

print("\nPARAM BUCKETS (by top module):")
import collections
buck = collections.Counter()
for name, p in model.named_parameters():
    parts = name.split(".")
    prefix = ".".join(parts[:4])
    buck[prefix] += p.numel()
for k, v in sorted(buck.items(), key=lambda kv: -kv[1])[:15]:
    print(f"  {v:>14,}  {k}")
