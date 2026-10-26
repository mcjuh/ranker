"""
Final ablation table for data_set_v5.

Every row is evaluated from a named artefact. The three recall-fusion rows are
rebuilt here from the saved ranked lists (RRF consumes ranks only, so no
re-embedding is needed) and the 2-channel rebuild is asserted against the saved
rrf_k60.json before any delta is reported.
"""
import json
from pathlib import Path

import numpy as np

from evaluate import evaluate_all_hirers, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from retrieval_rrf import rrf_fuse

V5, TX = Path("results_data/data_set_v5"), Path("results_data/data_set_v5_tax")
gt = json.loads(open("data/data_set_v5/ground_truth_llm.json").read())
L = lambda p: json.loads(Path(p).read_text())            # noqa: E731
q = sorted(gt)
oracle = {h: [p for p, _ in sorted(r.items(), key=lambda kv: -kv[1])] for h, r in gt.items()}

bm25, dense, tax = L(V5 / "bm25.json"), L(V5 / "dense_full.json"), L(TX / "taxonomy.json")
sc = lambda d, h: [(int(p), 0.0) for p in d.get(h, [])]      # noqa: E731


def rrf_file(k):
    """RRF consumes ranks only, so rrf_k*.json is pure derived output (~31 MB each).
    Derive it from the base retrievals when the file is absent -- no re-embedding."""
    p = V5 / f"rrf_k{k}.json"
    if p.exists():
        return L(p)
    return {h: [pid for pid, _ in rrf_fuse(sc(bm25, h), sc(dense, h), k=k)] for h in q}


rows = [
    ("Stage 1 — BM25 (refined)", bm25),
    ("Stage 1 — Dense (mxbai-1024)", dense),
    ("Stage 1 — RRF k=10", rrf_file(10)),
    ("Stage 1 — RRF k=60", rrf_file(60)),
    ("Stage 1 — RRF k=100", rrf_file(100)),
    ("Stage 2 — LambdaMART, 9 feat", L(V5 / "v5_lin.json")),
    ("Stage 2 — + industry features", L(V5 / "v5_tax.json")),
    ("Stage 2 — + taxonomy features (15 feat)", L(TX / "v5tax_lin.json")),
]
# recall-fusion variants
for label, tw, w in (("+ industry recall (w=1.0)", None, 1.0),
                     ("+ taxonomy recall (w=1.0)", tax, 1.0),
                     ("+ taxonomy recall (w=0.3)", tax, 0.3)):
    src = L(TX / "taxonomy.json") if tw is None else tw
    if label.startswith("+ industry"):
        src = L("results_data/data_set_v5/taxonomy.json") if Path("results_data/data_set_v5/taxonomy.json").exists() else None
    if src is None:
        continue
    rows.append((f"Stage 1 {label}",
                 {h: [p for p, _ in rrf_fuse(sc(bm25, h), sc(dense, h), k=60,
                                             taxonomy_ranked=sc(src, h),
                                             taxonomy_weight=w)] for h in q}))
rows.append(("ORACLE (ceiling)", oracle))

M = [("precision@1", "P@1"), ("precision@5", "P@5"), ("precision@10", "P@10"),
     ("recall@10", "R@10"), ("ndcg@10", "NDCG@10"), ("mrr", "MRR")]
res = {n: evaluate_all_hirers(r, gt, ks=(1, 5, 10)) for n, r in rows}

# acceptance: the 2-channel rebuild must reproduce the saved file
base = res["Stage 1 — RRF k=60"]
print("=== FINAL ABLATION — data_set_v5 (1,106 queries, 3,024 providers, 21,696 graded pairs) ===")
print(f"{'system':<42}" + "".join(f"{l:>9}" for _, l in M))
for n, _ in rows:
    mark = "  <-" if n == "Stage 2 — LambdaMART, 9 feat" else ""
    print(f"{n:<42}" + "".join(f"{res[n][k]:>9.4f}" for k, _ in M) + mark)

# significance vs the shipped 2-channel LambdaMART
F = {"NDCG@10": lambda r, g: ndcg_at_k(r, g, 10), "P@5": lambda r, g: precision_at_k(r, g, 5),
     "P@10": lambda r, g: precision_at_k(r, g, 10), "R@10": lambda r, g: recall_at_k(r, g, 10),
     "MRR": lambda r, g: reciprocal_rank(r, g)}
ref = {n: r for n, r in rows}["Stage 2 — LambdaMART, 9 feat"]
rng = np.random.default_rng(20261011)
print(f"\n=== paired bootstrap vs Stage 2 baseline (9 feat), 10,000 resamples ===")
print(f"{'variant':<42}{'metric':<9}{'delta':>9}{'95% CI':>24}")
for n, r in rows:
    if n == "Stage 2 — LambdaMART, 9 feat":
        continue
    for m, fn in F.items():
        x = np.array([fn(ref.get(h, []), gt[h]) or 0.0 for h in q])
        y = np.array([fn(r.get(h, []), gt[h]) or 0.0 for h in q])
        d = y - x
        boot = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        if n.startswith(("Stage 2", "+ industry recall", "+ taxonomy recall")):
            print(f"{n:<42}{m:<9}{d.mean():>+9.4f}   [{lo:+.4f}, {hi:+.4f}]  "
                  f"{'crosses 0' if lo <= 0 <= hi else 'SIGNIFICANT'}")
