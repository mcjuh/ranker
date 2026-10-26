"""
Compare cross-encoder reranked lists against the shipped LambdaMART/XGBoost ranker.

Both systems are scored against the SAME ground truth (data/<ds>/ground_truth_llm.json)
and only over queries present in that ground truth, so the comparison is apples-to-apples.

Usage:
    python compare_ce_xgb.py --baseline results_data/data_set_v5_tax/v5tax_tags.json \
        results_data/data_set_v5/ce_v5ce_minilm.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
import evaluate as E

GT = "data/data_set_v5/ground_truth_llm.json"
METRICS = ["p@5", "r@5", "ndcg@5", "p@10", "r@10", "ndcg@10", "mrr"]


def per_query(lists, queries, gt):
    out = {m: [] for m in METRICS}
    for q in queries:
        lst, g = lists.get(str(q), []), gt.get(str(q), {})
        out["p@5"].append(E.precision_at_k(lst, g, 5))
        out["ndcg@5"].append(E.ndcg_at_k(lst, g, 5))
        out["p@10"].append(E.precision_at_k(lst, g, 10))
        rc = E.recall_at_k(lst, g, 10)
        out["r@5"].append(E.recall_at_k(lst, g, 5) or 0.0)
        out["r@10"].append(np.nan if rc is None else rc)
        out["ndcg@10"].append(E.ndcg_at_k(lst, g, 10))
        out["mrr"].append(E.reciprocal_rank(lst, g))
    return {k: np.array(v) for k, v in out.items()}


def boot(a, b, n=10000):
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    s = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", nargs="?", default="results_data/data_set_v5_tax/v5tax_tags.json")
    ap.add_argument("systems", nargs="*")
    args = ap.parse_args()

    gt = json.loads((BASE / GT).read_text())
    queries = sorted(int(q) for q in gt)

    # baseline is given as a path-like first positional; argparse grabbed it as `baseline`
    paths = [args.baseline] + args.systems
    paths = [p for p in paths if p.endswith(".json")]
    if not paths:
        paths = ["results_data/data_set_v5_tax/v5tax_tags.json"]

    per, names = {}, {}
    print(f"{'system':<42}" + "".join(f"{m:>9}" for m in METRICS))
    for p in paths:
        d = json.loads((BASE / p).read_text())
        name = Path(p).stem
        names[p] = name
        pq = per_query(d, queries, gt)
        per[p] = pq
        row = []
        for m in METRICS:
            v = pq[m]
            v = v[~np.isnan(v)]
            row.append(f"{v.mean():.4f}")
        print(f"{name:<42}" + "".join(f"{x:>9}" for x in row))

    base = paths[0]
    print(f"\npaired bootstrap vs {names[base]} (10k resamples over {len(queries)} queries):")
    for p in paths[1:]:
        print(f"\n  {names[p]} vs {names[base]}")
        for m in METRICS:
            d, lo, hi = boot(per[p][m], per[base][m])
            flag = "crosses 0" if lo <= 0 <= hi else "EXCLUDES 0"
            print(f"    {m:<8} d={d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {flag}")


if __name__ == "__main__":
    main()
