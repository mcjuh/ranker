"""
Where does the missing 31% live?

Decompose the top-10 of the shipped model against the oracle: how many grade-3
and grade-2 items exist in the pool, how many each system actually surfaces, and
therefore which grade band the error is concentrated in. A cross-encoder can only
help if the loss is in the ORDERING of already-retrieved items, not in what was
retrieved at all.
"""
import json

import numpy as np

gt = json.loads(open("data/data_set_v5/ground_truth_llm.json").read())
graded = json.loads(open("data/data_set_v5/llm_judgments_merged.json").read())
q = sorted(gt)

sys = {
    "shipped LambdaMART": json.loads(open("results_data/data_set_v5/v5_lin.json").read()),
    "CE zero-shot alone": json.loads(open("results_data/data_set_v5/ce_v5_minilm.json").read()),
}
try:
    sys["CE fine-tuned alone"] = json.loads(open("results_data/data_set_v5/ce_v5_ft.json").read())
except FileNotFoundError:
    pass

avail3 = {h: sum(1 for v in graded[h].values() if v == 3) for h in q}
avail2 = {h: sum(1 for v in graded[h].values() if v == 2) for h in q}
print(f"  pool composition per query (median): grade-3 {int(np.median(list(avail3.values())))}, "
      f"grade-2 {int(np.median(list(avail2.values())))}, total graded {np.median([len(graded[h]) for h in q]):.0f}")
print()
print(f"  {'system':<24}{'g3 in top10':>13}{'g2 in top10':>13}{'recall g3':>11}{'recall g2':>11}")
for name, r in sys.items():
    a3 = np.mean([sum(1 for p in r.get(h, [])[:10] if graded[h].get(str(p), graded[h].get(p, 0)) == 3) for h in q])
    a2 = np.mean([sum(1 for p in r.get(h, [])[:10] if graded[h].get(str(p), graded[h].get(p, 0)) == 2) for h in q])
    r3 = a3 / np.mean(list(avail3.values()))
    r2 = a2 / np.mean(list(avail2.values()))
    print(f"  {name:<24}{a3:>13.2f}{a2:>13.2f}{r3:>11.1%}{r2:>11.1%}")

# oracle: perfect order puts every available high grade first
o3 = np.mean([min(10, avail3[h]) for h in q])
o2 = np.mean([avail2[h] for h in q])
print(f"  {'ORACLE':<24}{o3:>13.2f}{min(10, np.mean(list(avail2.values()))):>13.2f}"
      f"{o3/np.mean(list(avail3.values())):>11.1%}{o2/np.mean(list(avail2.values())):>11.1%}")
print()
print(f"  mean grade-3 available per query : {np.mean(list(avail3.values())):.2f}")
print(f"  -> the shipped model surfaces {np.mean([sum(1 for p in sys['shipped LambdaMART'].get(h, [])[:10] if graded[h].get(str(p),0)==3) for h in q]):.2f} of them in its top-10")
