"""
Does adding the SkillsFuture tag features to the Stage-2 LambdaMART help?

The protocol that makes the answer trustworthy (ablation.md §5):

  1. Train on the ORIGINAL labels -- the 21,696 pairs graded from the text
     retrievers' pools, i.e. the top-up grades held out of training.
  2. Evaluate on ALL labels, original + top-up. The top-up graded every pair that the
     tag rankers put in their top-10 and nobody had graded, so both systems'
     top-10s are ~99% graded and evaluate.py's "ungraded = grade 0" no longer biases
     the comparison toward the system whose candidates were pooled.

Training on the top-up too is reported as a second block, but it is the weaker test:
the top-up was chosen from where the TAG rankers look, so it hands the baseline
labelled examples in regions it would never rank highly and changes what it learns,
and the retrained baseline's own top-10 is then only ~94% graded.

Every number is out-of-fold. Split 0 is rerank_ltr.py's GroupKFold; the others are
shuffled 5-fold splits, so an effect has to replicate across fold assignments.
Paired bootstrap over queries, 10k resamples.

    python pipeline/merge_topup_grades.py --data-dir data/data_set_v5_tax   # once
    python pipeline/tag_features.py       --data-dir data/data_set_v5_tax   # separate process
    python pipeline/tag_ablation.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold, KFold

import evaluate as E
import rerank_ltr as R
from merge_topup_grades import load_topup
from tag_features import TAG_FEATURES

BASE = Path(__file__).parent
METRICS = ["ndcg@10", "ndcg@5", "p@5", "p@10", "r@10", "mrr", "judged@10"]


def oof_lists(rows, by_query, queries, features, split_seed, use_label):
    """Out-of-fold ranked lists. `use_label(row)` decides which labels training may see."""
    folds = (GroupKFold(n_splits=5).split(queries, groups=queries) if split_seed == 0
             else KFold(n_splits=5, shuffle=True, random_state=split_seed).split(queries))
    scores = {}
    for tr, te in folds:
        tr_q = [queries[i] for i in sorted(tr)]
        tr_rows, groups = [], []
        for q in tr_q:
            rs = [r for r in by_query[q] if use_label(r)]
            tr_rows += rs
            groups.append(len(rs))
        X = np.array([[r[f] for f in features] for r in tr_rows], dtype=np.float32)
        model = xgb.XGBRanker(**R.MODEL_PARAMS, random_state=7, n_jobs=4)
        model.fit(X, R.train_labels(tr_rows), group=groups)
        te_rows = [r for i in te for r in by_query[queries[i]]]
        Xt = np.array([[r[f] for f in features] for r in te_rows], dtype=np.float32)
        for r, p in zip(te_rows, model.predict(Xt)):
            scores[(r["hire_id"], r["provider_id"])] = float(p)
    return {q: [r["provider_id"] for r in sorted(by_query[q], key=lambda r: -scores[(q, r["provider_id"])])]
            for q in queries}


def per_query(lists, queries, gt, judged):
    out = defaultdict(list)
    for q in queries:
        lst, g = lists[q], gt.get(str(q), {})
        out["ndcg@10"].append(E.ndcg_at_k(lst, g, 10))
        out["ndcg@5"].append(E.ndcg_at_k(lst, g, 5))
        out["p@5"].append(E.precision_at_k(lst, g, 5))
        out["p@10"].append(E.precision_at_k(lst, g, 10))
        rc = E.recall_at_k(lst, g, 10)
        out["r@10"].append(np.nan if rc is None else rc)
        out["mrr"].append(E.reciprocal_rank(lst, g))
        out["judged@10"].append(np.mean([str(p) in judged.get(str(q), {}) for p in lst[:10]]))
    return {k: np.array(v) for k, v in out.items()}


def boot(a, b, n=10000):
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    s = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax")
    ap.add_argument("--splits", default="0,11,23", help="fold-split seeds; 0 = rerank_ltr.py's GroupKFold")
    args = ap.parse_args()

    R.set_dataset(args.data_dir)
    R.MODEL_PARAMS["ndcg_exp_gain"] = False                    # shipped: --linear-gain
    gt = json.loads((BASE / args.data_dir / "ground_truth_llm.json").read_text())
    judged = json.loads((BASE / args.data_dir / "llm_judgments_merged.json").read_text())
    topup = load_topup()
    if not any(p in judged.get(h, {}) for h, p in topup):
        raise SystemExit("top-up grades are not merged -- run merge_topup_grades.py first")

    rows, base_feats = R.load_candidates(50, None, tag_features=True)
    base_feats = [f for f in base_feats if f not in TAG_FEATURES]
    R.normalize_by_query(rows, R.SCORE_FEATURES, "zscore")     # shipped: --normalize-scores zscore
    by_query = defaultdict(list)
    for r in rows:
        by_query[r["hire_id"]].append(r)
    queries = sorted(by_query)

    systems = {"LambdaMART (9 feat, shipped)": base_feats,
               "LambdaMART + tag features (17 feat)": base_feats + TAG_FEATURES}
    rrf = {q: [r["provider_id"] for r in sorted(by_query[q], key=lambda r: r["rrf_rank"])] for q in queries}
    m_rrf = per_query(rrf, queries, gt, judged)
    seeds = [int(s) for s in args.splits.split(",")]
    regimes = {
        "trained on ORIGINAL labels (primary test)":
            lambda r: r["label"] is not None and (str(r["hire_id"]), str(r["provider_id"])) not in topup,
        "trained on ALL labels incl. top-up":
            lambda r: r["label"] is not None,
    }

    out_dir = BASE / args.data_dir.replace("data/", "results_data/", 1)
    for regime, use_label in regimes.items():
        res = {name: {s: per_query(oof_lists(rows, by_query, queries, feats, s, use_label), queries, gt, judged)
                      for s in seeds} for name, feats in systems.items()}
        if regime.startswith("trained on ORIGINAL"):
            for name, feats in systems.items():
                tag = "v5tax_lin_orig" if feats == base_feats else "v5tax_tags_orig"
                lists = oof_lists(rows, by_query, queries, feats, 0, use_label)
                (out_dir / f"{tag}.json").write_text(json.dumps({str(q): [int(p) for p in v] for q, v in lists.items()}, indent=2))

        print(f"\n=== {regime} -- evaluated on all {sum(map(len, judged.values()))} graded pairs, "
              f"mean over splits {seeds} ===")
        print(f"{'system':38s}" + "".join(f"{c:>10s}" for c in METRICS))
        print(f"{'RRF k=60 (no training)':38s}" + "".join(f"{np.nanmean(m_rrf[c]):10.4f}" for c in METRICS))
        for name in systems:
            print(f"{name:38s}" + "".join(
                f"{np.mean([np.nanmean(res[name][s][c]) for s in seeds]):10.4f}" for c in METRICS))
        a, b = list(systems)
        print(f"\n  tags vs shipped, paired bootstrap on split {seeds[0]} (delta on every split):")
        for c in ["ndcg@10", "ndcg@5", "p@5", "p@10", "r@10", "mrr"]:
            d, lo, hi = boot(res[b][seeds[0]][c], res[a][seeds[0]][c])
            verdict = "better" if lo > 0 else "worse" if hi < 0 else "crosses 0"
            per = " ".join(f"{np.nanmean(res[b][s][c] - res[a][s][c]):+.4f}" for s in seeds)
            print(f"    {c:8s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict:9s}  per split: {per}")


if __name__ == "__main__":
    main()
