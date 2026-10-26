"""
Tier-1 upgrade of the Stage-2 LambdaMART ranker.

Adds three families of features on top of the shipped 17-feature model
(9 retrieval + 8 SkillsFuture tag features), keeping the identical OOF protocol
(GroupKFold by query, linear gain, per-query z-scored score features):

  1. Taxonomy-graph partial credit. `tag_features.py` matches skills by exact
     string and by name-embedding cosine. The sf_v1 tree has real edges, so a
     gig skill the provider does not hold exactly can still be *reached*:
       - sibling  : the provider holds another skill under the SAME specialisation
       - category : ... under the SAME category
     Features: skill_sib_cov, skill_cat_cov, graph_dist_mean/min, spec_cat_cov,
     skill_jaccard.

  2. Interaction crosses the trees cannot easily construct at this sample size:
     tag_level x rank, spec/skill coverage x tag_level.

  3. Provider-prior quality: the mean grade a provider earns on OTHER queries
     (leave-one-query-out target encoding, alpha-smoothed). Fold-aware: for each
     fold the prior is built ONLY from that fold's training queries, so a test
     query's own labels never enter it.

Then a small rank-fusion (weighted RRF) of the base and the upgraded ranker.

    python pipeline/tier1_rank.py --data-dir data/data_set_v5_tax
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

import evaluate as E
import rerank_ltr as R
from rerank_ltr import BM25_FIELD_FEATURES

BASE = Path(__file__).parent
METRICS = ["p@5", "p@10", "r@5", "r@10", "ndcg@5", "ndcg@10", "mrr"]


# --------------------------------------------------------------------------
# taxonomy index
# --------------------------------------------------------------------------
def load_index(data_dir: Path):
    hirers = json.loads((data_dir / "hirers.json").read_text(encoding="utf-8"))
    providers = json.loads((data_dir / "providers.json").read_text(encoding="utf-8"))

    sk2spec, sk2cat, spec2cat = defaultdict(set), defaultdict(set), {}

    def axes(rec):
        cats, specs, skills = set(), set(), set()
        for c in (rec.get("tags") or {}).get("categories") or []:
            cats.add(c["name"])
            for s in c.get("specialisations") or []:
                specs.add((c["name"], s["name"]))
                spec2cat[(c["name"], s["name"])] = c["name"]
                for k in s.get("skills") or []:
                    skills.add(k)
                    sk2spec[k].add((c["name"], s["name"]))
                    sk2cat[k].add(c["name"])
        return cats, specs, skills

    gig = {r["hire_id"]: axes(r) for r in hirers}
    prov = {r["provider_id"]: axes(r) for r in providers}
    return gig, prov, sk2spec, sk2cat


def graph_features(g, p, sk2spec, sk2cat):
    """Partial-credit coverage of the gig's skills/specs by the provider."""
    gc, gs, gk = g
    pc, ps, pk = p
    shared = gk & pk

    # best hop distance per gig skill: 0 exact, 1 sibling (same spec), 2 same cat, 3 none
    dists = []
    for k in gk:
        if k in pk:
            dists.append(0)
        elif any(sk2spec[k] & sk2spec[k2] for k2 in pk):
            dists.append(1)
        elif any(sk2cat[k] & sk2cat[k2] for k2 in pk):
            dists.append(2)
        else:
            dists.append(3)
    if dists:
        d = np.asarray(dists)
        sib_cov = float((d <= 1).mean())
        cat_cov = float((d <= 2).mean())
        dist_mean = float(d.mean() / 3.0)
        dist_min = float(d.min() / 3.0)
    else:
        sib_cov = cat_cov = dist_mean = dist_min = 0.0

    # share of the gig's (cat, spec) pairs whose CATEGORY the provider covers
    spec_cat_cov = float(sum(1 for (c, _s) in gs if c in pc) / len(gs)) if gs else 0.0
    jacc = float(len(shared) / len(gk | pk)) if (gk | pk) else 0.0

    return {
        "skill_sib_cov": sib_cov,
        "skill_cat_cov": cat_cov,
        "graph_dist_mean": dist_mean,
        "graph_dist_min": dist_min,
        "spec_cat_cov": spec_cat_cov,
        "skill_jaccard": jacc,
    }


NEW_GRAPH = ["skill_sib_cov", "skill_cat_cov", "graph_dist_mean", "graph_dist_min",
             "spec_cat_cov", "skill_jaccard"]
NEW_INTER = ["tl_x_rank", "speccov_x_tl", "skillcov_x_tl"]
NEW_PRIOR = ["prov_prior", "prov_ct"]


# --------------------------------------------------------------------------
# OOF LambdaMART with a fold-aware provider prior
# --------------------------------------------------------------------------
def oof_scores(rows, by_query, features, folds, seed, alpha=5.0):
    queries = sorted(by_query)
    oof, imps, fold_nd = {}, [], []
    for tr_idx, te_idx in GroupKFold(n_splits=folds).split(queries, groups=queries):
        tr_q = {queries[i] for i in tr_idx}
        te_q = {queries[i] for i in te_idx}

        # ---- provider prior from TRAIN queries only, leave-own-query-out ----
        tot_s, tot_c = defaultdict(float), defaultdict(int)
        pq_s, pq_c = defaultdict(float), defaultdict(int)          # per (provider, query)
        gsum = gcnt = 0
        for r in rows:
            if r["label"] is None or r["hire_id"] not in tr_q:
                continue
            p, q, y = r["provider_id"], r["hire_id"], r["label"]
            tot_s[p] += y; tot_c[p] += 1
            pq_s[(p, q)] += y; pq_c[(p, q)] += 1
            gsum += y; gcnt += 1
        gm = gsum / gcnt if gcnt else 0.0
        for r in rows:
            p, q = r["provider_id"], r["hire_id"]
            s, c = tot_s[p], tot_c[p]
            if q in tr_q:                                          # drop the row's own query
                s -= pq_s[(p, q)]; c -= pq_c[(p, q)]
            r["prov_prior"] = (s + alpha * gm) / (c + alpha)
            r["prov_ct"] = math.log1p(c)

        tr_rows = [r for q in sorted(tr_q) for r in by_query[q] if r["label"] is not None]
        X = np.array([[r[f] for f in features] for r in tr_rows], dtype=np.float32)
        y = R.train_labels(tr_rows)
        groups = [sum(1 for r in tr_rows if r["hire_id"] == q) for q in sorted(tr_q)]
        model = xgb.XGBRanker(**R.MODEL_PARAMS, random_state=seed, n_jobs=8)
        model.fit(X, y, group=groups)
        imps.append(model.feature_importances_)

        te_rows = [r for q in sorted(te_q) for r in by_query[q]]
        Xt = np.array([[r[f] for f in features] for r in te_rows], dtype=np.float32)
        for r, pr in zip(te_rows, model.predict(Xt)):
            oof[(r["hire_id"], r["provider_id"])] = float(pr)

        fold_rank = {}
        for r in te_rows:
            fold_rank.setdefault(r["hire_id"], []).append((r["provider_id"], oof[(r["hire_id"], r["provider_id"])]))
        fl = {str(q): [p for p, _ in sorted(v, key=lambda x: -x[1])] for q, v in fold_rank.items()}
        fold_nd.append(E.evaluate_all_hirers(fl, gt_global, ks=(10,))["ndcg@10"])
    return oof, np.mean(imps, axis=0), fold_nd


def ranked_from_oof(rows, by_query, oof):
    return {q: [r["provider_id"] for r in sorted(rs, key=lambda r: -oof[(q, r["provider_id"])])]
            for q, rs in by_query.items()}


def per_query(lists, queries, gt):
    out = {m: [] for m in METRICS}
    for q in queries:
        lst, g = lists.get(q, []), gt.get(str(q), {})
        out["p@5"].append(E.precision_at_k(lst, g, 5))
        out["p@10"].append(E.precision_at_k(lst, g, 10))
        out["r@5"].append(E.recall_at_k(lst, g, 5) or 0.0)
        rc = E.recall_at_k(lst, g, 10)
        out["r@10"].append(np.nan if rc is None else rc)
        out["ndcg@5"].append(E.ndcg_at_k(lst, g, 5))
        out["ndcg@10"].append(E.ndcg_at_k(lst, g, 10))
        out["mrr"].append(E.reciprocal_rank(lst, g))
    return {k: np.array(v) for k, v in out.items()}


def boot(a, b, n=10000):
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    s = d[np.random.default_rng(0).integers(0, len(d), (n, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def rrf_fuse(list_a, list_b, w=0.5, k=60):
    out = {}
    for q in list_a:
        sc = defaultdict(float)
        for rank, p in enumerate(list_a[q], 1):
            sc[p] += w / (k + rank)
        for rank, p in enumerate(list_b.get(q, []), 1):
            sc[p] += (1 - w) / (k + rank)
        out[q] = sorted(sc, key=lambda p: -sc[p])
    return out


def show(name, lists, queries, gt):
    pq = per_query(lists, queries, gt)
    row = "".join(f"{np.nanmean(pq[m]):>9.4f}" for m in METRICS)
    print(f"  {name:<30}{row}")
    return pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bm25-fields", action="store_true",
                    help="also merge features_data/<ds>/bm25_fields.csv (per-section BM25)")
    args = ap.parse_args()

    R.set_dataset(args.data_dir)
    R.MODEL_PARAMS["ndcg_exp_gain"] = False

    global gt_global
    gt = json.loads((BASE / args.data_dir / "ground_truth_llm.json").read_text())
    gt_global = gt

    rows, base_feats = R.load_candidates(50, None, tag_features=True)
    if args.bm25_fields:
        bf_path = R.FEAT_DIR / "bm25_fields.csv"
        if not bf_path.exists():
            raise SystemExit(f"{bf_path} not found -- run `python pipeline/bm25_fields.py "
                             f"--data-dir {args.data_dir}`")
        bf = {(r["hire_id"], r["provider_id"]): [float(r[c]) for c in BM25_FIELD_FEATURES]
              for r in csv.DictReader(bf_path.open())}
        miss = 0
        for r in rows:
            vals = bf.get((str(r["hire_id"]), str(r["provider_id"])))
            if vals is None:
                vals = [0.0] * len(BM25_FIELD_FEATURES)
                miss += 1
            for c, v in zip(BM25_FIELD_FEATURES, vals):
                r[c] = v
        base_feats = list(base_feats) + list(BM25_FIELD_FEATURES)
        print(f"  bm25 field features merged ({len(BM25_FIELD_FEATURES)} cols); {miss} rows missing -> 0.0")

    R.normalize_by_query(rows, R.SCORE_FEATURES + (list(BM25_FIELD_FEATURES) if args.bm25_fields else []),
                         "zscore")

    gig, prov, sk2spec, sk2cat = load_index(BASE / args.data_dir)
    print("building taxonomy-graph + interaction features ...")
    for r in rows:
        r.update(graph_features(gig[r["hire_id"]], prov[r["provider_id"]], sk2spec, sk2cat))
        tl = r["tag_level"]
        r["tl_x_rank"] = tl * (50.0 - r["rrf_rank"]) / 49.0
        r["speccov_x_tl"] = r["spec_cov"] * tl
        r["skillcov_x_tl"] = r["skill_cov"] * tl

    by_query = defaultdict(list)
    for r in rows:
        by_query[r["hire_id"]].append(r)
    queries = sorted(by_query)

    base17 = list(base_feats)
    tier1 = base17 + NEW_GRAPH + NEW_INTER + NEW_PRIOR
    print(f"base features ({len(base17)}): {base17}")
    print(f"tier-1 additions ({len(tier1) - len(base17)}): {NEW_GRAPH + NEW_INTER + NEW_PRIOR}\n")

    print(f"{'system':<32}" + "".join(f"{m:>9}" for m in METRICS))
    rrf = {q: [r["provider_id"] for r in sorted(rs, key=lambda r: r["rrf_rank"])] for q, rs in by_query.items()}
    pq_rrf = show("RRF k=60", rrf, queries, gt)

    oof_b, imp_b, fold_b = oof_scores(rows, by_query, base17, args.folds, args.seed)
    lb = ranked_from_oof(rows, by_query, oof_b)
    pq_b = show("LambdaMART base (17 feat)", lb, queries, gt)

    oof_t, imp_t, fold_t = oof_scores(rows, by_query, tier1, args.folds, args.seed)
    lt = ranked_from_oof(rows, by_query, oof_t)
    pq_t = show("LambdaMART + tier1", lt, queries, gt)

    fused = rrf_fuse(lb, lt, w=0.5)
    pq_f = show("RRF-fuse base+tier1 (0.5)", fused, queries, gt)

    out_dir = R.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suf = "_bm25f" if args.bm25_fields else ""
    (out_dir / f"tier1_base{suf}.json").write_text(json.dumps({str(q): [int(p) for p in v] for q, v in lb.items()}))
    (out_dir / f"tier1_model{suf}.json").write_text(json.dumps({str(q): [int(p) for p in v] for q, v in lt.items()}))
    (out_dir / f"tier1_fuse{suf}.json").write_text(json.dumps({str(q): [int(p) for p in v] for q, v in fused.items()}))

    print(f"\n  per-fold NDCG@10  base: {np.mean(fold_b):.4f} +- {np.std(fold_b):.4f}"
          f"   tier1: {np.mean(fold_t):.4f} +- {np.std(fold_t):.4f}")

    print("\n  mean feature importance (tier1), top 15:")
    for f, v in sorted(zip(tier1, imp_t), key=lambda x: -x[1])[:15]:
        print(f"    {f:18s} {v:.4f}")

    print(f"\npaired bootstrap vs LambdaMART base (10k resamples, {len(queries)} queries):")
    for name, pq in (("+ tier1", pq_t), ("RRF-fuse", pq_f)):
        print(f"\n  {name} vs base")
        for m in METRICS:
            d, lo, hi = boot(pq[m], pq_b[m])
            flag = "crosses 0" if lo <= 0 <= hi else "EXCLUDES 0"
            print(f"    {m:<8} d={d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {flag}")
    print("\n  (RRF / base vs shipped xgboost for reference:)")
    for m in METRICS:
        d, lo, hi = boot(pq_b[m], pq_rrf[m])
        print(f"    base - RRF  {m:<8} d={d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
