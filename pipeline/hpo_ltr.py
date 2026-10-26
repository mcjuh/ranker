"""
Randomized hyperparameter search for the Stage-2 LambdaMART, OOF by query.

Evaluates on NDCG@10 (the headline metric) using the same GroupKFold protocol as
tier1_rank.py, over the 28-feature set (17 shipped + 11 Tier-1). Also tests the
top-k pairwise construction, which focuses the lambdarank gradient on the head of
the list -- the region NDCG@10 actually reads.

    python pipeline/hpo_ltr.py --data-dir data/data_set_v5_tax --trials 40
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np
import xgboost as xgb

import evaluate as E
import rerank_ltr as R
import tier1_rank as T

BASE_PARAMS = dict(objective="rank:ndcg", n_estimators=300, learning_rate=0.05,
                   max_depth=4, subsample=0.9, colsample_bytree=0.9,
                   min_child_weight=2, reg_lambda=1.0, ndcg_exp_gain=False)


def build_rows(data_dir, bm25_fields=False):
    R.set_dataset(data_dir)
    rows, base = R.load_candidates(50, None, tag_features=True)
    if bm25_fields:
        import csv
        from rerank_ltr import BM25_FIELD_FEATURES
        bf = {(r["hire_id"], r["provider_id"]): [float(r[c]) for c in BM25_FIELD_FEATURES]
              for r in csv.DictReader((R.FEAT_DIR / "bm25_fields.csv").open())}
        for r in rows:
            vals = bf.get((str(r["hire_id"]), str(r["provider_id"])), [0.0] * len(BM25_FIELD_FEATURES))
            for c, v in zip(BM25_FIELD_FEATURES, vals):
                r[c] = v
        base = list(base) + list(BM25_FIELD_FEATURES)
        R.normalize_by_query(rows, R.SCORE_FEATURES + list(BM25_FIELD_FEATURES), "zscore")
    else:
        R.normalize_by_query(rows, R.SCORE_FEATURES, "zscore")

    gig, prov, sk2spec, sk2cat = T.load_index(__import__("pathlib").Path(R.DATA_DIR))
    for r in rows:
        r.update(T.graph_features(gig[r["hire_id"]], prov[r["provider_id"]], sk2spec, sk2cat))
        tl = r["tag_level"]
        r["tl_x_rank"] = tl * (50.0 - r["rrf_rank"]) / 49.0
        r["speccov_x_tl"] = r["spec_cov"] * tl
        r["skillcov_x_tl"] = r["skill_cov"] * tl

    by_query = defaultdict(list)
    for r in rows:
        by_query[r["hire_id"]].append(r)
    feats = list(base) + T.NEW_GRAPH + T.NEW_INTER + T.NEW_PRIOR
    return rows, by_query, feats


def evaluate_oof(rows, by_query, feats, params, folds=5, seed=7):
    old = dict(R.MODEL_PARAMS)
    R.MODEL_PARAMS = dict(BASE_PARAMS, **{k: v for k, v in params.items() if v is not None})
    try:
        oof, _, _ = T.oof_scores(rows, by_query, feats, folds, seed)
    finally:
        R.MODEL_PARAMS = old
    lists = T.ranked_from_oof(rows, by_query, oof)
    return E.evaluate_all_hirers({str(q): v for q, v in lists.items()}, T.gt_global, ks=(10,))["ndcg@10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bm25-fields", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    gt = json.loads((__import__("pathlib").Path(R.BASE) / args.data_dir / "ground_truth_llm.json").read_text())
    T.gt_global = gt
    rows, by_query, feats = build_rows(args.data_dir, args.bm25_fields)
    print(f"rows={len(rows)} queries={len(by_query)} features={len(feats)}")

    rng = random.Random(1234)
    configs = [dict()]
    for _ in range(args.trials):
        cfg = dict(
            max_depth=rng.choice([3, 4, 5, 6]),
            n_estimators=rng.choice([300, 500, 800, 1200]),
            learning_rate=rng.choice([0.02, 0.03, 0.05, 0.08]),
            min_child_weight=rng.choice([1, 2, 4, 8]),
            subsample=rng.choice([0.8, 0.9, 1.0]),
            colsample_bytree=rng.choice([0.6, 0.8, 1.0]),
            reg_lambda=rng.choice([0.5, 1.0, 2.0, 5.0]),
        )
        if rng.random() < 0.5:
            cfg["lambdarank_pair_method"] = "topk"
            cfg["lambdarank_num_pair_per_sample"] = rng.choice([4, 8, 16])
        configs.append(cfg)

    results = []
    for i, cfg in enumerate(configs):
        try:
            nd = evaluate_oof(rows, by_query, feats, cfg, args.folds, args.seed)
        except Exception as e:
            print(f"  trial {i:2d} failed: {type(e).__name__}: {e}")
            continue
        results.append((nd, cfg))
        print(f"  trial {i:2d}  NDCG@10={nd:.4f}  {cfg}")

    results.sort(key=lambda x: -x[0])
    print("\n=== top 8 ===")
    for nd, cfg in results[:8]:
        print(f"  {nd:.4f}  {cfg}")

    out = __import__("pathlib").Path(R.RESULTS_DIR) / "hpo_best.json"
    out.write_text(json.dumps({"ndcg@10": results[0][0], "params": results[0][1]}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
