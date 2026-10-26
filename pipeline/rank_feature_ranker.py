"""
Per-query rank features for the tag / graph signals.

A tree splits on an ABSOLUTE feature value, but relevance is usually relative:
"this provider has the highest skill coverage for this gig" matters more than
"skill_cov = 0.40". We already carry bm25_rank / dense_rank / rrf_rank; this adds
the same treatment to the Taxonomy-graph features, plus a combined tag score.

    python pipeline/rank_feature_ranker.py --data-dir data/data_set_v5_tax
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

import evaluate as E
import rerank_ltr as R
import tier1_rank as T
import hpo_ltr as H

RANKED = ["tag_level", "spec_cov", "skill_cov", "skill_idf_cov", "skill_soft",
          "skill_jaccard", "spec_cat_cov", "graph_dist_min", "graph_dist_mean"]


def add_ranks(rows, by_query):
    new = [f"r_{f}" for f in RANKED]
    for q, rs in by_query.items():
        df = pd.DataFrame({f: [r[f] for r in rs] for f in RANKED})
        for f in RANKED:
            rs_order = df[f].rank(method="min", ascending=False).to_numpy()
            for r, v in zip(rs, rs_order):
                r[f"r_{f}"] = float(v)
        # combined tag score: z-sum of coverage, then rank it
        cov = df[["spec_cov", "skill_cov", "skill_idf_cov", "skill_soft"]]
        z = (cov - cov.mean()) / (cov.std(ddof=0).replace(0, 1))
        comb = z.sum(axis=1).rank(method="min", ascending=False).to_numpy()
        for r, v in zip(rs, comb):
            r["r_tag_comb"] = float(v)
    return new + ["r_tag_comb"]


def prior_by_fold(rows, tr_q, alpha=5.0):
    tot_s, tot_c = defaultdict(float), defaultdict(int)
    pq_s, pq_c = defaultdict(float), defaultdict(int)
    gsum = gcnt = 0
    for r in rows:
        if r["label"] is None or r["hire_id"] not in tr_q:
            continue
        p, q, y = r["provider_id"], r["hire_id"], r["label"]
        tot_s[p] += y; tot_c[p] += 1; pq_s[(p, q)] += y; pq_c[(p, q)] += 1
        gsum += y; gcnt += 1
    gm = gsum / gcnt if gcnt else 0.0
    for r in rows:
        p, q = r["provider_id"], r["hire_id"]
        s, c = tot_s[p], tot_c[p]
        if q in tr_q:
            s -= pq_s[(p, q)]; c -= pq_c[(p, q)]
        r["prov_prior"] = (s + alpha * gm) / (c + alpha)
        r["prov_ct"] = float(np.log1p(c))


def oof(rows, by_query, feats, params, folds, seed):
    queries = sorted(by_query)
    out = {}
    for tr, te in GroupKFold(n_splits=folds).split(queries, groups=queries):
        tr_q = {queries[i] for i in tr}
        te_q = {queries[i] for i in te}
        prior_by_fold(rows, tr_q)
        tr_rows = [r for q in sorted(tr_q) for r in by_query[q] if r["label"] is not None]
        X = np.array([[r[f] for f in feats] for r in tr_rows], np.float32)
        y = np.array([r["label"] for r in tr_rows], np.float32)
        groups = [sum(1 for r in tr_rows if r["hire_id"] == q) for q in sorted(tr_q)]
        m = xgb.XGBRanker(**params, random_state=seed, n_jobs=8)
        m.fit(X, y, group=groups)
        te_rows = [r for q in sorted(te_q) for r in by_query[q]]
        Xt = np.array([[r[f] for f in feats] for r in te_rows], np.float32)
        for r, p in zip(te_rows, m.predict(Xt)):
            out[(r["hire_id"], r["provider_id"])] = float(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    gt = json.loads((R.BASE / args.data_dir / "ground_truth_llm.json").read_text())
    T.gt_global = gt
    rows, by_query, feats = H.build_rows(args.data_dir)
    new = add_ranks(rows, by_query)
    feats = feats + new
    print(f"features={len(feats)} (+{len(new)} per-query ranks)")

    best = json.loads((Path(R.RESULTS_DIR) / "hpo_best.json").read_text())["params"]
    params = dict(H.BASE_PARAMS, **best)
    o = oof(rows, by_query, feats, params, args.folds, 7)
    lists = T.ranked_from_oof(rows, by_query, o)
    m = E.evaluate_all_hirers({str(q): v for q, v in lists.items()}, gt, ks=(5, 10))
    print(f"OOF: P@5={m['precision@5']:.4f} NDCG@5={m['ndcg@5']:.4f} "
          f"P@10={m['precision@10']:.4f} NDCG@10={m['ndcg@10']:.4f} MRR={m['mrr']:.4f}")
    out = Path(R.RESULTS_DIR) / "rank_features.json"
    out.write_text(json.dumps({str(q): [int(p) for p in v] for q, v in lists.items()}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
