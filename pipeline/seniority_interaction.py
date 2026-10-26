"""Does `seniority_needed` pay off as an INTERACTION feature?

It is a gig attribute, so on its own it is constant across a query's 50 candidates
and a within-query ranker cannot use it (0 gain). This tests whether multiplying it
by a *varying* provider signal recovers anything.

Reuses rerank_ltr's loader / params / OOF harness so the baseline here must
reproduce the shipped 0.8962 before the interaction is believed.

    python pipeline/seniority_interaction.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import rerank_ltr as R
from evaluate import evaluate_all_hirers
from tag_features import TAG_FEATURES

DATA = "data/data_set_v5_tax"
SEN = {"mid": 0, "senior": 1, "expert": 2}
INTERACTIONS = ["seniority_enc", "sen_x_skill_cov", "sen_x_spec_cov",
                "sen_x_skill_idf_cov", "sen_x_dense", "sen_x_avail"]


def load():
    R.set_dataset(DATA)
    rows, features = R.load_candidates(50, None)
    features = list(features)

    tf = {(R.as_id(r["hire_id"]), R.as_id(r["provider_id"])): r
          for r in csv.DictReader((R.FEAT_DIR / "tag_features.csv").open())}
    for r in rows:
        src = tf[(r["hire_id"], r["provider_id"])]
        for f in TAG_FEATURES:
            r[f] = src[f]
    features.extend(TAG_FEATURES)

    # z-score first, so the interactions multiply like-for-like scales
    R.normalize_by_query(rows, list(R.SCORE_FEATURES), "zscore")

    sen = {str(h["hire_id"]): SEN.get(h.get("seniority_needed"))
           for h in json.loads((R.DATA_DIR / "hirers.json").read_text())}
    for r in rows:
        s = sen.get(str(r["hire_id"]))
        r["seniority_enc"] = float(s) if s is not None else np.nan
        for name, col in (("sen_x_skill_cov", "skill_cov"), ("sen_x_spec_cov", "spec_cov"),
                          ("sen_x_skill_idf_cov", "skill_idf_cov"), ("sen_x_dense", "dense_cosine"),
                          ("sen_x_avail", "avail_immediacy")):
            r[name] = r["seniority_enc"] * float(r[col])
    return rows, features


def run(rows, features, folds=5, seed=7):
    gt = json.loads((R.DATA_DIR / "ground_truth_llm.json").read_text())
    by_query = defaultdict(list)
    for r in rows:
        by_query[r["hire_id"]].append(r)

    oof = R.train_oof(rows, features, folds, seed, gt, {})
    ranked = {str(q): [p for p, _ in sorted(
        [(r["provider_id"], oof.get((q, r["provider_id"]), float("-inf"))) for r in rs],
        key=lambda x: -x[1])] for q, rs in by_query.items()}
    return evaluate_all_hirers(ranked, gt, ks=(5, 10)), ranked


def show(tag, m, ranked):
    print(f"  {tag:<28} P@5={m['precision@5']:.4f} NDCG@10={m['ndcg@10']:.4f} MRR={m['mrr']:.4f}")
    out = R.RESULTS_DIR / f"sen_{tag.replace(' ', '_')}.json"
    out.write_text(json.dumps({str(q): [int(p) for p in v] for q, v in ranked.items()}, indent=2))
    print(f"    -> {out.name}")


if __name__ == "__main__":
    rows, base_features = load()
    print(f"\n=== baseline ({len(base_features)} features) ===")
    m0, r0 = run(rows, base_features)
    show("base", m0, r0)

    all_features = base_features + INTERACTIONS
    print(f"\n=== + seniority interactions ({len(all_features)} features) ===")
    m1, r1 = run(rows, all_features)
    show("inter", m1, r1)

    print(f"\n  delta NDCG@10 = {m1['ndcg@10'] - m0['ndcg@10']:+.4f}")
