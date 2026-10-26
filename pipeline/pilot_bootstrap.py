"""
Paired bootstrap over the pilot's 30 queries: is the taxonomy channel's gain real?

Queries are the unit of resampling (the standard paired test for IR metrics).
At n=30 the intervals are wide -- that is the point of running it.
"""
import json
import random
from pathlib import Path

import numpy as np

from evaluate import ndcg_at_k, precision_at_k, reciprocal_rank

BASE = Path(__file__).parent
gt = json.loads((BASE / "data_pilot/ground_truth_llm.json").read_text())
SYS = {
    "RRF k=60": "results_data_pilot/rrf_k60.json",
    "LambdaMART (9 feat)": "results_data_pilot/pilot_linz.json",
    "LambdaMART +tax (15 feat)": "results_data_pilot/pilot_tax.json",
}
METRICS = ("ndcg@10", "p@5", "mrr")

per = {}
for name, rel in SYS.items():
    ranked = json.loads((BASE / rel).read_text())
    per[name] = {h: {
        "ndcg@10": ndcg_at_k(r, gt.get(h, {}), 10),
        "p@5": precision_at_k(r, gt.get(h, {}), 5),
        "mrr": reciprocal_rank(r, gt.get(h, {})),
    } for h, r in ranked.items()}

q = sorted(gt)
print(f"{len(q)} queries\n")
print(f"{'system':<28} " + "".join(f"{m:>10}" for m in METRICS))
for name in SYS:
    print(f"{name:<28} " + "".join(f"{np.mean([per[name][h][m] for h in q]):>10.4f}" for m in METRICS))

rng = np.random.default_rng(20261011)
B = 10_000
pairs = [
    ("RRF k=60", "LambdaMART (9 feat)"),
    ("RRF k=60", "LambdaMART +tax (15 feat)"),
    ("LambdaMART (9 feat)", "LambdaMART +tax (15 feat)"),
]
print(f"\npaired bootstrap, {B:,} resamples over queries (delta = B - A):")
for a, b in pairs:
    print(f"\n  {b}  vs  {a}")
    for m in METRICS:
        x = np.array([per[a][h][m] for h in q])
        y = np.array([per[b][h][m] for h in q])
        d = y - x
        boot = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        crosses = "crosses 0" if lo <= 0 <= hi else "excludes 0"
        print(f"    {m:<8} d={d.mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   {crosses}")

print("\nfor reference, the same test on the 1,023-query corpus had a noise floor of ~0.007")
