"""
Serve ONE gig end-to-end: retrieval -> features -> LambdaMART -> ranked list.

This is the piece that was missing. api_io_example.py only re-read precomputed
CSVs, so it worked for the 1,106 gigs already in the corpus and failed
immediately on anything else. This runs the actual pipeline for an arbitrary gig.

    python pipeline/serve.py --hire-id 1                 # a corpus gig (control)
    python pipeline/serve.py --gig-json request.json     # any gig, including new

First run embeds 3,024 providers (~1-2 min) and caches to cache_data/. Later runs
reuse the cache.

Contract notes (see API_IO_SPEC.md):
  - per-query zscore on SCORE_FEATURES is applied; the model was trained with it
  - feature order is positional (the model stores feature_names: [])
  - provider_id is a corpus row index; platform_id comes from id_map.json
  - score is per-query min-max of the ranker margin, NOT a probability
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

from corpus import hirer_text, provider_text
from rerank_ltr import DEFAULT_FEATURES, SCORE_FEATURES, normalize_by_query
from retrieval_bm25 import BM25Retriever
from retrieval_dense import encode_docs, encode_queries, rank_from_raw
from retrieval_rrf import rrf_fuse
from tag_features import TAG_FEATURES, TagMatcher, embed_skills, tag_axes
from tier import load_tier

BASE = Path(__file__).parent
DATA = BASE / "data" / "data_set_v5_tax"
CACHE = BASE / "cache_data" / "data_set_v5_tax"

# Swapped by --tier in main(). Both tiers answer the same contract; only the model
# behind the ranking changes (see configs/*.json and API_IO_SPEC.md).
TIER = load_tier("lightweight")
MODEL = BASE / TIER["model_path"]
FEATURES = list(TIER["features"])
TOP_K_POOL = TIER["retrieval"]["pool_size"]

_state: dict = {}


def _load():
    """Corpus + retrievers, built once per process."""
    if _state:
        return _state
    providers = json.loads((DATA / "providers.json").read_text(encoding="utf-8"))
    hirers = json.loads((DATA / "hirers.json").read_text(encoding="utf-8"))

    emb_path = CACHE / "provider_emb_1024.npy"
    if emb_path.exists():
        doc_raw = np.load(emb_path)
    else:
        doc_raw = encode_docs([provider_text(p) for p in providers])
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, doc_raw)

    _state.update(
        providers=providers,
        hirers=hirers,
        by_id={int(p["provider_id"]): p for p in providers},
        provider_ids=[int(p["provider_id"]) for p in providers],
        doc_raw=doc_raw,
        bm25=BM25Retriever(providers),
        model=(lambda m: (m.load_model(MODEL), m)[1])(xgb.XGBRanker()),
    )
    return _state


def _tag_matcher(gig):
    """TagMatcher over the corpus plus this gig, so a new gig scores too."""
    s = _load()
    hirers = s["hirers"] + [gig]
    # the vocabulary must cover PROVIDER tags too: features() looks up provider-side
    # skills, and a hirer-only vocabulary KeyErrors on any skill unique to a provider
    skills = sorted({k for r in hirers for k in tag_axes(r)[2]}
                    | {k for p in s["providers"] for k in tag_axes(p)[2]})
    vec = embed_skills(skills, CACHE / "skill_vectors.json") if skills else {}
    return TagMatcher(hirers, s["providers"], vec)


def serve(gig: dict, top_k: int = 10) -> dict:
    s = _load()
    hid = gig["hire_id"]

    # ---- stage 1: retrieval -------------------------------------------------
    q_text = hirer_text(gig)
    bm25_ranked = s["bm25"].rank(q_text, query_title=gig.get("hire_title"))
    q_vec = encode_queries([q_text])[0]
    dense_ranked = rank_from_raw(s["doc_raw"], q_vec, s["provider_ids"], truncate_dim=1024)
    fused = rrf_fuse(bm25_ranked, dense_ranked, k=60)

    bm25_score = dict(bm25_ranked)
    dense_score = dict(dense_ranked)
    bm25_pos = {p: i + 1 for i, (p, _) in enumerate(bm25_ranked)}
    dense_pos = {p: i + 1 for i, (p, _) in enumerate(dense_ranked)}

    # ---- stage 2: features --------------------------------------------------
    pool = fused[:TOP_K_POOL]
    matcher = _tag_matcher(gig)
    rows = []
    for rrf_rank, (pid, rrf_score) in enumerate(pool, start=1):
        pid = int(pid)
        row = {
            "hire_id": hid,
            "provider_id": pid,
            "bm25_score": float(bm25_score.get(pid, 0.0)),
            "bm25_rank": float(bm25_pos.get(pid, 0)),
            "dense_cosine": float(dense_score.get(pid, 0.0)),
            "dense_rank": float(dense_pos.get(pid, 0)),
            "rrf_score": float(rrf_score),
            "rrf_rank": float(rrf_rank),
            # no corpus field supplies these; see API_IO_SPEC.md "Known-dead fields"
            "budget_fit": 0.0,
            "seniority_fit": 0.0,
            "avail_immediacy": 0.0,
        }
        row.update(matcher.features(hid, pid))
        rows.append(row)

    normalize_by_query(rows, list(TIER["score_features"]), "zscore")

    # ---- stage 3: rank ------------------------------------------------------
    X = [[r[f] for f in FEATURES] for r in rows]
    assert all(v == v for row in X for v in row), "NaN in feature matrix"
    raw = s["model"].predict(X)

    lo = float(raw.min())
    span = (float(raw.max()) - lo) or 1.0
    order = sorted(zip([r["provider_id"] for r in rows], raw, rows), key=lambda x: -x[1])

    results = []
    for rank, (pid, margin, row) in enumerate(order[:top_k], start=1):
        p = s["by_id"][int(pid)]
        results.append({
            "rank": rank,
            "provider_id": int(pid),
            "score": round((float(margin) - lo) / span * 100, 2),
            "raw_margin": round(float(margin), 6),
            "provider": {
                "display_title": p.get("about_title"),
                "industry": p.get("industry"),
                "bio_excerpt": (p.get("about_description") or "")[:200],
            },
            "signals": {
                "retrieval": {f: round(row[f], 6) for f in DEFAULT_FEATURES[:6]},
                "business_fit": {f: round(row[f], 6) for f in DEFAULT_FEATURES[6:]},
                "taxonomy": {f: round(row[f], 4) for f in TAG_FEATURES},
            },
        })

    return {
        "hire_id": hid,
        "results": results,
        "meta": {
            "tier": TIER["tier"],
            "model": MODEL.name,
            "sota_ndcg@10": TIER["sota"]["ndcg@10"],
            "n_features": len(FEATURES),
            "candidate_pool": len(rows),
            "returned": len(results),
            "normalization": "per-query zscore on score_features",
            "score_scale": "0-100 ranker score, NOT a probability",
        },
    }


def main():
    global TIER, MODEL, FEATURES
    ap = argparse.ArgumentParser()
    ap.add_argument("--hire-id")
    ap.add_argument("--gig-json", help="path to a gig record, or '-' for stdin")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--tier", default="lightweight", choices=["lightweight", "pro"],
                    help="lightweight = GBDT only; pro = cross-encoders + set-wise transformer")
    a = ap.parse_args()

    TIER = load_tier(a.tier)
    if not TIER.get("servable"):
        raise SystemExit(f"tier '{a.tier}' has no serving artifact.\n  blocked on: {TIER['blocked_on']}")
    MODEL = BASE / TIER["model_path"]
    FEATURES = list(TIER["features"])
    # the response's signals{retrieval,business_fit,taxonomy} blocks slice this list
    assert FEATURES == list(DEFAULT_FEATURES) + list(TAG_FEATURES), \
        "tier features must match the declared signal blocks"

    if a.gig_json:
        raw = sys.stdin.read() if a.gig_json == "-" else Path(a.gig_json).read_text(encoding="utf-8")
        gig = json.loads(raw)
    elif a.hire_id:
        gig = next(h for h in json.loads((DATA / "hirers.json").read_text(encoding="utf-8"))
                   if str(h["hire_id"]) == str(a.hire_id))
    else:
        raise SystemExit("pass --hire-id or --gig-json")

    print(json.dumps(serve(gig, a.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
