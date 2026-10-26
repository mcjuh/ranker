"""Why is recall low? R@K is capped by K against the number of relevant items."""
import json

import numpy as np

from evaluate import RELEVANCE_THRESHOLD, recall_at_k

gt = json.loads(open("data/data_set_v5/ground_truth_llm.json").read())
R = json.loads(open("results_data/data_set_v5/v5_lin.json").read())
raw = json.loads(open("results_data/data_set_v5/rrf_k60.json").read())
q = sorted(gt)

print(f"  exported v5_lin list length : {len(R[q[0]])}")
print(f"  rrf_k60 list length         : {len(raw[q[0]])}")

rq = {h: sum(1 for v in gt[h].values() if v >= RELEVANCE_THRESHOLD) for h in q}
vals = list(rq.values())
defined = [h for h in q if rq[h] > 0]
print(f"  queries {len(q)} | with >=1 relevant {len(defined)}")
print(f"  relevant/query: median {int(np.median(vals))}, mean {np.mean(vals):.1f}, max {max(vals)}")

oracle = {h: [p for p, _ in sorted(r.items(), key=lambda kv: -kv[1])] for h, r in gt.items()}


def mean_skip(xs):
    v = [x for x in xs if x is not None]
    return float(np.mean(v)), len(v)


print("\n=== R@K: observed vs oracle (over queries where recall is defined) ===")
print(f"  {'K':>5}{'observed':>12}{'oracle':>10}{'% of achievable':>18}{'n':>7}")
for K in (5, 10, 20, 50):
    o, n = mean_skip([recall_at_k(R[h], gt[h], K) for h in q])
    oc, _ = mean_skip([recall_at_k(oracle[h], gt[h], K) for h in q])
    print(f"  {K:>5}{o:>12.4f}{oc:>10.4f}{o/oc:>17.1%}{n:>7}")

print("\n=== the structural cap: mean over queries of min(R_q, K) / R_q ===")
print("    (this is what a PERFECT ranker scores, for the same candidate set)")
for K in (5, 10, 20, 50):
    v = [min(x, K) / x for x in vals if x > 0]
    print(f"    R@{K:<4} ceiling {np.mean(v):.4f}")
