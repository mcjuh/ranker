"""
Emit a sample recommendation-API response built from real pipeline output.

Every value here comes from the actual corpus, retrieval and taxonomy features --
nothing is invented. Fields the serving layer must supply itself (latency, request
id, timestamp) are marked so the SWE team knows what is theirs to fill.

Usage: python make_api_sample.py [--hire-id H-...] [--top-k 5] [--out api_sample_response.json]
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent


def load_features(tag):
    """candidates_top50.csv holds the stage-1 signals; the taxonomy match features
    live in their own file, so merge them in on (hire_id, provider_id)."""
    rows = {}
    p = BASE / f"features_{tag}" / "candidates_top50.csv"
    if p.exists():
        for r in csv.DictReader(p.open()):
            rows.setdefault(r["hire_id"], []).append(r)
    t = BASE / f"features_{tag}" / "taxonomy_scores.csv"
    if t.exists():
        tax = {(r["hire_id"], r["provider_id"]): r for r in csv.DictReader(t.open())}
        for hid, lst in rows.items():
            for r in lst:
                for k, v in tax.get((hid, r["provider_id"]), {}).items():
                    if k.startswith("tax"):
                        r[k] = v
    return rows


def tax_block(tags):
    """Corpus tags are already the nested SkillsFuture shape."""
    out = []
    for c in tags or []:
        out.append({
            "category": c.get("name"),
            "specialisations": [s.get("name") for s in c.get("specialisations") or []],
            "search_tags": [s.get("search_tag") for s in c.get("specialisations") or []],
        })
    return out


def facets(tags):
    """Split corpus tags into (search_tags, specialisations, skills)."""
    st, sp, sk = set(), set(), set()
    for c in tags or []:
        for s in c.get("specialisations") or []:
            if s.get("search_tag"):
                st.add(s["search_tag"])
            if s.get("name"):
                sp.add(s["name"])
            for x in s.get("skills") or []:
                sk.add(x)
    return st, sp, sk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="data_all")
    ap.add_argument("--hire-id", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="api_sample_response.json")
    args = ap.parse_args()

    hirers = {h["hire_id"]: h for h in json.loads((BASE / args.tag / "hirers.json").read_text())}
    provs = {int(p["provider_id"]): p for p in json.loads((BASE / args.tag / "providers.json").read_text())}
    feats = load_features(args.tag)

    ranked, ranker_used = None, None
    suffix = args.tag.split("_")[-1]                 # data_pilot -> pilot, data_all -> all
    for cand in (f"{suffix}_tax.json", f"{suffix}_linz.json"):
        p = BASE / f"results_{args.tag}" / cand
        if p.exists():
            ranked = json.loads(p.read_text())
            ranker_used = cand
            break
    if ranked is None:
        ranked = json.loads((BASE / f"results_{args.tag}" / "rrf_k60.json").read_text())
        ranker_used = "rrf_k60.json (no LambdaMART trained yet -- stage-1 ranking)"

    # rerank_ltr --export-scores writes <tag>_scores.json: {query: [{provider_id,
    # semantic_score, rank}]} on a fixed per-query 0-100 scale.
    #
    # IMPORTANT: those scores come from the FULL-data model, whereas the ranked
    # lists in <tag>.json are out-of-fold predictions. Mixing them puts a higher
    # score at a worse rank. So when the score export exists we take BOTH the
    # order and the scores from it, keeping rank and score consistent.
    score_map, score_file, score_order = {}, None, {}
    cands = sorted((BASE / f"results_{args.tag}").glob("*_scores.json"))
    if cands:
        score_file = cands[-1].name
        for q, lst in json.loads(cands[-1].read_text()).items():
            for e in lst:
                score_map[(q, e["provider_id"])] = e["semantic_score"]
            score_order[q] = [e["provider_id"] for e in sorted(lst, key=lambda x: x["rank"])]

    hid = args.hire_id
    if hid is None:
        # pick the query with the most candidate features, so the sample is complete
        hid = max((h for h in ranked if feats.get(h)), key=lambda h: len(feats[h]))
    h = hirers[hid]
    frow = {r["provider_id"]: r for r in feats.get(hid, [])}
    g_st, g_sp, g_sk = facets(h.get("tags"))

    results = []
    order = score_order.get(hid) or ranked[hid]
    for rank, pid in enumerate(order[: args.top_k], start=1):
        p = provs[pid]
        f = frow.get(str(pid), {})
        p_st, p_sp, p_sk = facets(p.get("tags"))

        def num(k):
            v = f.get(k)
            try:
                return round(float(v), 6)
            except (TypeError, ValueError):
                return None

        results.append({
            "rank": rank,
            "provider_id": pid,
            "score": score_map.get((hid, pid)),
            "provider": {
                "display_title": p.get("about_title"),
                "headline": (p.get("about_title") or "")[:120],
                "bio_excerpt": (p.get("about_description") or "")[:280],
                "years_experience": p.get("years_experience"),
                "services": [p.get("services_offered_title")] if p.get("services_offered_title") else [],
                "taxonomy": tax_block(p.get("tags")),
                # 0% populated in the source schema -- the endpoint must return null,
                # not 0, or downstream filters will treat them as free/available
                "rate": None,
                "availability": None,
            },
            "signals": {
                "stage1": {
                    "bm25_rank": num("bm25_rank"), "bm25_score": num("bm25_score"),
                    "dense_rank": num("dense_rank"), "dense_cosine": num("dense_cosine"),
                    "rrf_rank": num("rrf_rank"), "rrf_score": num("rrf_score"),
                },
                "taxonomy_match": {
                    f: num(f) for f in ("tax_search_tag", "tax_specialisation", "tax_category",
                                        "tax_skill", "tax_n_shared_skills", "taxonomy_score")
                },
                "provider_attrs": {"exp_years": num("exp_years")},
            },
            "explanation": {
                "matched_search_tags": sorted(g_st & p_st),
                "matched_specialisations": sorted(g_sp & p_sp),
                "matched_skills": sorted(g_sk & p_sk),
            },
        })

    payload = {
        "api_version": "1.0",
        "request_id": "<generate per request, e.g. req_01J9Z8...>",
        "generated_at": "<ISO-8601 with timezone, set by the service>",
        "query": {
            "hire_id": hid,
            "title": h.get("hire_title"),
            "description": h.get("hire_description"),
            "duration_weeks": {"min": h.get("duration_weeks_min"), "max": h.get("duration_weeks_max")},
            "taxonomy": tax_block(h.get("tags")),
        },
        "results": results,
        "meta": {
            "pipeline": "refined BM25 || mxbai-embed-large-v1 (Matryoshka 1024) -> RRF(k=60) "
                        "-> LambdaMART (taxonomy variant)",
            "ranker_artifact": ranker_used,
            "score_artifact": score_file,
            "corpus_size": len(provs),
            "candidate_pool": 50,
            "returned": len(results),
            "latency_ms": "<measured by the service>",
            "score_scale": "0-100, higher is better; ranker output, not a probability",
        },
    }

    out = BASE / args.out
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:2600])


if __name__ == "__main__":
    main()
