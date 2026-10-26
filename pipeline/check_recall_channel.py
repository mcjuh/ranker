"""
Does adding the third (industry/taxonomy) channel to the RECALL fusion help?

run_pipeline.py calls rrf_fuse(sparse, dense, k=k) only -- the taxonomy channel is
never passed, so previous --taxonomy runs exercised the match FEATURES, not the
recall channel. rrf_fuse already supports the third term; this rebuilds both
variants from the saved ranked lists, which is valid because RRF consumes RANKS
only (no re-embedding needed).

Acceptance test: the 2-channel rebuild must reproduce the saved rrf_k60.json
metrics before any 3-channel delta is believed.
"""
import json
from pathlib import Path

import numpy as np

from evaluate import evaluate_all_hirers, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from retrieval_rrf import rrf_fuse

T, R = "data/data_set_v5", Path("results_data/data_set_v5")
gt = json.loads(Path(f"{T}/ground_truth_llm.json").read_text())
load = lambda f: json.loads((R / f).read_text())
bm25, dense, tax, saved = load("bm25.json"), load("dense_full.json"), load("taxonomy.json"), load("rrf_k60.json")


def scored(d, h):
    return [(int(p), 0.0) for p in d.get(h, [])]


fuse2, fuse3 = {}, {}
for h in gt:
    s, d, t = scored(bm25, h), scored(dense, h), scored(tax, h)
    fuse2[h] = [p for p, _ in rrf_fuse(s, d, k=60)]
    fuse3[h] = [p for p, _ in rrf_fuse(s, d, k=60, taxonomy_ranked=t)]

M = [("precision@1", "P@1"), ("precision@5", "P@5"), ("precision@10", "P@10"),
     ("recall@10", "R@10"), ("ndcg@10", "NDCG@10"), ("mrr", "MRR")]
S = {"saved rrf_k60.json": saved, "rebuilt 2-channel": fuse2, "rebuilt 3-channel (+industry)": fuse3}
res = {n: evaluate_all_hirers(r, gt, ks=(1, 5, 10)) for n, r in S.items()}

print("=== recall fusion: 2-channel vs 3-channel ===")
print(f"  {'system':<32}" + "".join(f"{l:>9}" for _, l in M))
for n in S:
    print(f"  {n:<32}" + "".join(f"{res[n][k]:>9.4f}" for k, _ in M))

print("\n  acceptance: rebuilt 2-channel vs saved rrf_k60.json")
for k, l in M:
    d = res["rebuilt 2-channel"][k] - res["saved rrf_k60.json"][k]
    print(f"    {l:<8} delta {d:+.4f}" + ("  OK" if abs(d) < 5e-4 else "  <-- does not reproduce"))

q = sorted(gt)
F = {"NDCG@10": lambda r, g: ndcg_at_k(r, g, 10), "P@5": lambda r, g: precision_at_k(r, g, 5),
     "P@10": lambda r, g: precision_at_k(r, g, 10), "R@10": lambda r, g: recall_at_k(r, g, 10),
     "MRR": lambda r, g: reciprocal_rank(r, g)}
rng = np.random.default_rng(20261011)
print("\n  paired bootstrap: 3-channel - 2-channel, 10,000 resamples over queries")
print(f"  {'metric':<9}{'delta':>10}{'95% CI':>26}   verdict")
for n, fn in F.items():
    x = np.array([fn(fuse2.get(h, []), gt[h]) or 0.0 for h in q])
    y = np.array([fn(fuse3.get(h, []), gt[h]) or 0.0 for h in q])
    d = y - x
    boot = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {n:<9}{d.mean():>+10.4f}   [{lo:+.4f}, {hi:+.4f}]   "
          f"{'crosses 0' if lo <= 0 <= hi else 'SIGNIFICANT'}")

# where does the channel actually add candidates?
added = np.mean([len(set(fuse3[h]) - set(fuse2[h])) for h in q])
in10 = np.mean([len(set(fuse3[h][:10]) - set(fuse2[h][:10])) for h in q])
print(f"\n  candidates the 3rd channel adds: {added:.0f} per query in the pool, {in10:.2f} inside the top-10")
