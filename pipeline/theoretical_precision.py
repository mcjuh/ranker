"""
Theoretical precision ceilings, and where each system sits against them.

Three numbers matter for any precision@K:

  ceiling   = mean_q  min(R_q, K) / K        -- a perfect ranker, same candidate set
  random    = mean_q  R_q / N_q              -- expected P@K over a random permutation
  observed  = mean_q  (relevant in top K)/K  -- what the system actually got

where R_q = judged-relevant items for query q, N_q = size of the pool being ranked.

IMPORTANT SUBTLETY: R_q is counted from the JUDGED pairs only. Where the candidate
pool is larger than the judged set (data_all: 3,024 providers, ~28 judged per gig),
the true R_q is unknown and the ceiling below UNDERSTATES it -- unjudged providers
may also be relevant. On data_pilot every candidate is judged, so its ceiling is exact.
"""
import json
from pathlib import Path

import numpy as np

from evaluate import RELEVANCE_THRESHOLD, evaluate_all_hirers, precision_at_k

BASE = Path(__file__).parent


def analyse(tag, results=None):
    gt = json.loads((BASE / tag / "ground_truth_llm.json").read_text())
    pools = {h: len({int(p) for p in row}) for h, row in gt.items()}
    rel = {h: sum(1 for v in row.values() if v >= RELEVANCE_THRESHOLD) for h, row in gt.items()}

    print(f"\n{'='*76}\n{tag}   {len(gt)} labelled queries, "
          f"{sum(len(v) for v in gt.values())} judged pairs\n{'='*76}")
    print(f"  judged candidates per query : min {min(pools.values())}, "
          f"max {max(pools.values())}, median {int(np.median(list(pools.values())))}")
    print(f"  judged-relevant per query   : {dict(sorted({r: list(rel.values()).count(r) for r in set(rel.values())}.items()))}")

    n = len(gt)
    print(f"\n  {'K':>4}{'ceiling':>11}{'random':>11}", end="")
    if results:
        for name in results:
            print(f"{name[:14]:>17}", end="")
    print()
    for K in (1, 3, 5, 10, 25):
        ceil = np.mean([min(rel[h], K) / K for h in gt])
        rnd = np.mean([rel[h] / pools[h] for h in gt])
        print(f"  {K:>4}{ceil:>11.4f}{rnd:>11.4f}", end="")
        if results:
            for name, ranked in results.items():
                obs = np.mean([precision_at_k(ranked.get(h, []), gt[h], K) for h in gt])
                print(f"{obs:>17.4f}", end="")
        print()

    if results:
        print(f"\n  mean metrics:")
        print(f"  {'system':<22}" + "".join(f"{m:>11}" for m in ("P@5", "R@10", "NDCG@10", "MRR")))
        for name, ranked in results.items():
            m = evaluate_all_hirers(ranked, gt)
            print(f"  {name:<22}{m['precision@5']:>11.4f}{m['recall@10']:>11.4f}"
                  f"{m['ndcg@10']:>11.4f}{m['mrr']:>11.4f}")


def load(tag, names):
    out = {}
    for n in names:
        p = BASE / f"results_{tag}" / n
        if p.exists():
            out[n] = json.loads(p.read_text())
    return out


analyse("data_pilot", load("data_pilot", ["rrf_k60.json", "pilot_linz.json", "pilot_tax.json"]))
analyse("data_all", load("data_all", ["rrf_k60.json", "all_linz.json", "all_tax.json"]))
print("\nreminder: on data_all the ceiling is computed from judged pairs only;")
print("the pool is 3,024 providers wide, so the true ceiling is >= what is shown.")
