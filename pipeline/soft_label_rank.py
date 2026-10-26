"""
Soft-label ranker: train on the grader's CONTINUOUS score, not the 4 bins.

`llm_scores_v5.json` (+ `judgments_topup_tags.jsonl`) carries the grader's raw
score in [0,1] for every graded pair. The shipped LambdaMART trains on the
coarse grade 0-3; within a bin (e.g. all the grade-1s at 0.33) every order is
equally good to the loss, even though the continuous score distinguishes them.
A regressor on the continuous score is a different, finer signal -- useful both
on its own and as a fusion input.

OUT-OF-FOLD by query (GroupKFold), provider prior rebuilt per fold from training
queries only, evaluated NDCG@10 against the same graded ground truth.

    python pipeline/soft_label_rank.py --data-dir data/data_set_v5_tax
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

import evaluate as E
import rerank_ltr as R
import tier1_rank as T
import hpo_ltr as H


def load_soft(data_root: Path):
    soft = {}
    scores = json.loads((data_root / "llm_scores_v5.json").read_text())
    for h, v in scores.items():
        for p, s in v.items():
            soft[(int(h), int(p))] = float(s)
    topup = data_root / "judgments_topup_tags.jsonl"
    if topup.exists():
        for line in topup.read_text().splitlines():
            d = json.loads(line)
            soft[(int(d["hire_id"]), int(d["provider_id"]))] = float(d["score"])
    return soft


def prior_by_fold(rows, by_query, tr_q, alpha=5.0):
    tot_s, tot_c = defaultdict(float), defaultdict(int)
    pq_s, pq_c = defaultdict(float), defaultdict(int)
    gsum = gcnt = 0
    for r in rows:
        if r.get("soft_label") is None or r["hire_id"] not in tr_q:
            continue
        p, q, y = r["provider_id"], r["hire_id"], r["soft_label"]
        tot_s[p] += y; tot_c[p] += 1
        pq_s[(p, q)] += y; pq_c[(p, q)] += 1
        gsum += y; gcnt += 1
    gm = gsum / gcnt if gcnt else 0.0
    for r in rows:
        p, q = r["provider_id"], r["hire_id"]
        s, c = tot_s[p], tot_c[p]
        if q in tr_q:
            s -= pq_s[(p, q)]; c -= pq_c[(p, q)]
        r["prov_prior"] = (s + alpha * gm) / (c + alpha)
        r["prov_ct"] = math.log1p(c)


def oof_reg(rows, by_query, feats, params, folds, seed):
    queries = sorted(by_query)
    oof = {}
    for tr, te in GroupKFold(n_splits=folds).split(queries, groups=queries):
        tr_q = {queries[i] for i in tr}
        te_q = {queries[i] for i in te}
        prior_by_fold(rows, by_query, tr_q)
        tr_rows = [r for q in sorted(tr_q) for r in by_query[q] if r.get("soft_label") is not None]
        X = np.array([[r[f] for f in feats] for r in tr_rows], dtype=np.float32)
        y = np.array([r["soft_label"] for r in tr_rows], dtype=np.float32)
        m = xgb.XGBRegressor(**params, random_state=seed, n_jobs=8)
        m.fit(X, y)
        te_rows = [r for q in sorted(te_q) for r in by_query[q]]
        Xt = np.array([[r[f] for f in feats] for r in te_rows], dtype=np.float32)
        for r, pr in zip(te_rows, m.predict(Xt)):
            oof[(r["hire_id"], r["provider_id"])] = float(pr)
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax")
    ap.add_argument("--source-data", default="data/data_set_v5",
                    help="where llm_scores_v5.json / judgments_topup_tags live")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    gt = json.loads((R.BASE / args.data_dir / "ground_truth_llm.json").read_text())
    T.gt_global = gt
    rows, by_query, feats = H.build_rows(args.data_dir)

    soft = load_soft(R.BASE / args.source_data)
    n = 0
    for r in rows:
        r["soft_label"] = soft.get((r["hire_id"], r["provider_id"]))
        n += r["soft_label"] is not None
    print(f"soft labels attached: {n} / {len(rows)} candidate rows; features={len(feats)}")

    params = dict(objective="reg:squarederror", n_estimators=600, learning_rate=0.03,
                  max_depth=5, subsample=0.9, colsample_bytree=0.8,
                  min_child_weight=2, reg_lambda=1.0)
    oof = oof_reg(rows, by_query, feats, params, args.folds, args.seed)
    lists = T.ranked_from_oof(rows, by_query, oof)
    m = E.evaluate_all_hirers({str(q): v for q, v in lists.items()}, gt, ks=(5, 10))
    print(f"regression-on-soft-score: P@5={m['precision@5']:.4f} NDCG@5={m['ndcg@5']:.4f} "
          f"P@10={m['precision@10']:.4f} NDCG@10={m['ndcg@10']:.4f} MRR={m['mrr']:.4f}")

    out = Path(R.RESULTS_DIR) / "soft_label.json"
    out.write_text(json.dumps({str(q): [int(p) for p in v] for q, v in lists.items()}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
