"""
Is the DATA the bottleneck, or the modelling?

Corrected diagnostic. ground_truth_llm.json holds ONLY the relevant pairs, so
shuffling it gives a meaningless floor. The real pool is every graded candidate,
which lives in llm_judgments_merged.json. Random-shuffle that to get the true
floor; the oracle is the ceiling; the gap is the only headroom that exists.
"""
import json

import numpy as np

import evaluate as ev

gt = json.loads(open("data/data_set_v5/ground_truth_llm.json").read())
graded = json.loads(open("data/data_set_v5/llm_judgments_merged.json").read())
q = sorted(gt)
print(f"  sample merged entry: {list(graded[q[0]].items())[:3]}")
rng = np.random.default_rng(20261011)

pool = {h: sorted(graded[h]) for h in q}                       # every graded candidate
oracle = {h: [p for p, _ in sorted(graded[h].items(), key=lambda kv: (-kv[1], kv[0]))] for h in q}

allg = [v for h in q for v in graded[h].values()]
print(f"  graded pairs            : {len(allg)}")
print(f"  grade>=2 (relevant)     : {sum(1 for v in allg if v >= 2)/len(allg):.1%}  <- base rate IN THE POOL")
print(f"  graded candidates/query : {len(allg)/len(q):.1f}")
print(f"  relevant/query          : {np.mean([sum(1 for v in graded[h].values() if v>=2) for h in q]):.1f}")
print()

K = 10
orac = float(np.mean([ev.ndcg_at_k(oracle[h], graded[h], K) or 0.0 for h in q]))
rand = float(np.mean([np.mean([ev.ndcg_at_k(list(rng.permutation(pool[h])), graded[h], K) or 0.0
                               for h in q]) for _ in range(20)]))
print(f"  RANDOM ordering of the graded pool, NDCG@{K} : {rand:.4f}   <- the real floor")
print(f"  ORACLE ordering,                    NDCG@{K} : {orac:.4f}")
print(f"  -> actual headroom available in the task     : {orac - rand:.4f}")
print()
for name, path in (("LambdaMART (shipped)", "results_data/data_set_v5/v5_lin.json"),
                   ("+ full taxonomy", "results_data/data_set_v5_tax/v5tax_lin.json")):
    try:
        r = json.loads(open(path).read())
    except FileNotFoundError:
        print(f"  {name:<28} (absent - regenerable)")
        continue
    v = float(np.mean([ev.ndcg_at_k(r.get(h, []), graded[h], K) or 0.0 for h in q]))
    print(f"  {name:<28} NDCG@{K} {v:.4f}  = {(v-rand)/(orac-rand):5.1%} of the headroom")
