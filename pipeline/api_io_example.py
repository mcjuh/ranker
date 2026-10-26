"""
Emit one REAL request/response pair for the 0.89 model, with the signals block
aligned to the features the model actually uses.

Two things this pins down that the old contract got wrong:

1. SIGNALS. api_sample_final.json advertises tax_search_tag / tax_specialisation /
   tax_category / tax_skill / tax_n_shared_skills / taxonomy_score. Those are from
   a 30-gig pilot and exist nowhere in the shipped model, which uses the 8
   TAG_FEATURES below. Anyone coding against the old shape reads dead fields.

2. NORMALISATION IS PART OF THE CONTRACT. The model was trained on per-query
   z-scored SCORE_FEATURES (rerank_ltr.py --normalize-scores zscore). A server
   that skips this feeds the model features on a different scale and silently
   gets worse rankings. That is a serving requirement, not an option.

Note the committed results artifact (v5tax_tags.json) holds OUT-OF-FOLD rankings
-- each gig scored by a model that did not train on it -- while v5tax_tags_xgb.json
is the all-data serving model. They cannot match exactly, and that is correct.

    python pipeline/api_io_example.py --hire-id 1 --top-k 10
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import xgboost as xgb

from rerank_ltr import DEFAULT_FEATURES, SCORE_FEATURES, normalize_by_query
from tag_features import TAG_FEATURES

BASE = Path(__file__).parent
DATA = BASE / "data" / "data_set_v5_tax"
FEAT = BASE / "features_data" / "data_set_v5_tax"
MODEL = BASE / "models_data" / "data_set_v5_tax" / "v5tax_tags_xgb.json"
RESULTS = BASE / "results_data" / "data_set_v5_tax" / "v5tax_tags.json"

FEATURES = list(DEFAULT_FEATURES) + list(TAG_FEATURES)   # 17, in declared order


def load_rows(hire_id):
    base = {}
    for r in csv.DictReader((FEAT / "candidates_top50.csv").open()):
        if str(r["hire_id"]) == str(hire_id):
            base[str(r["provider_id"])] = {k: (float(v) if k in DEFAULT_FEATURES else v)
                                           for k, v in r.items()}
    tag = {}
    for r in csv.DictReader((FEAT / "tag_features.csv").open()):
        k = (str(r["hire_id"]), str(r["provider_id"]))
        if k[0] == str(hire_id) and k[1] in base:
            tag[k[1]] = r

    assert base, f"no candidate rows for hire_id={hire_id}"
    assert len(tag) == len(base), f"tag join incomplete: {len(tag)}/{len(base)}"

    rows = [base[p] for p in base]
    normalize_by_query(rows, list(SCORE_FEATURES), "zscore")   # required, see docstring
    for p, r in zip(base, rows):
        for f in TAG_FEATURES:
            r[f] = float(tag[p][f])
    return [r for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hire-id", default="1")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    rows = load_rows(args.hire_id)
    assert len(rows) == 50, f"expected a 50-candidate pool, got {len(rows)}"
    X = [[r[f] for f in FEATURES] for r in rows]
    assert all(v == v for row in X for v in row), "NaN in feature matrix"

    model = xgb.XGBRanker()
    model.load_model(MODEL)
    raw = model.predict(X)

    lo, span = float(raw.min()), (float(raw.max()) - float(raw.min())) or 1.0
    scored = sorted(zip([r["provider_id"] for r in rows], raw, rows), key=lambda x: -x[1])

    # diagnostic: serving model vs the OOF artifact. Equal sets are expected; the
    # order differs because one is cross-validated and one is not.
    ref = json.load(open(RESULTS))[str(args.hire_id)]
    overlap = len(set(int(p) for p, _, _ in scored[: len(ref)]) & set(ref))
    print(f"# OOF-artifact overlap: {overlap}/{len(ref)} of the same providers",
          file=sys.stderr, flush=True)

    provider = {str(p["provider_id"]): p for p in json.load(open(DATA / "providers.json"))}
    hire = next(h for h in json.load(open(DATA / "hirers.json"))
                if str(h["hire_id"]) == str(args.hire_id))

    results = [{
        "rank": rank,
        "provider_id": int(pid),
        "score": round((float(r) - lo) / span * 100, 2),
        "raw_margin": round(float(r), 6),
        "provider": {
            "display_title": provider[str(pid)].get("about_title"),
            "industry": provider[str(pid)].get("industry"),
            "bio_excerpt": (provider[str(pid)].get("about_description") or "")[:200],
        },
        "signals": {
            "retrieval": {f: round(row[f], 6) for f in DEFAULT_FEATURES[:6]},
            "business_fit": {f: round(row[f], 6) for f in DEFAULT_FEATURES[6:]},
            "taxonomy": {f: round(row[f], 4) for f in TAG_FEATURES},
        },
    } for rank, (pid, r, row) in enumerate(scored[: args.top_k], start=1)]

    print(json.dumps({
        "request": {
            "hire_id": hire["hire_id"],
            "hire_title": hire["hire_title"],
            "hire_description": hire["hire_description"],
            "industry": hire.get("industry"),
            "taxonomy": (hire.get("tags") or {}).get("categories"),
            "top_k": args.top_k,
        },
        "response": {
            "results": results,
            "meta": {
                "pipeline": "refined BM25 || mxbai-embed-large-v1 (1024-d) -> RRF(k=60) -> LambdaMART",
                "normalization": "per-query zscore on SCORE_FEATURES (required at serve time)",
                "n_features": len(FEATURES),
                "features": FEATURES,
                "candidate_pool": len(rows),
                "returned": len(results),
                "score_scale": "0-100, higher is better; ranker score, NOT a calibrated probability",
                "ranking_is": "serving model (all-data); committed artifact is out-of-fold",
            },
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
