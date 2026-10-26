"""
LambdaMART learning-to-rank re-ranker (Stage 2b) for the Senseigigs matching search.

This is the component that turns *non-text* signals into ordering: budget fit,
seniority fit and availability are invisible to any embedding, yet they are
exactly what separates the same-category-wrong-speciality profiles that clog
ranks 2-5.

Training is mandatory here -- there is no pretrained LambdaMART, the model IS
the training. It is cheap: gradient-boosted trees over ~1-3k judged rows and
130 queries trains in seconds on CPU.

Method
------
- candidates from `features/candidates_topK.csv` (frozen RRF pool)
- labels from the LLM judgments (0-3) -- never the retired taxonomy formula
- GroupKFold by hire_id: every query is held out exactly once, so the reported
  metrics are out-of-fold (OOF) and no query is ever scored by a model that
  saw it. With 130 queries, a random split would leak and flatter the result.
- optional ce_score column merged from `features/ce_scores_<tag>.csv`

Run (fi-bench env, from the repo root):
    python pipeline/rerank_ltr.py --mode ablation
    python pipeline/rerank_ltr.py --mode ablation --ce-tag minilm
    python pipeline/rerank_ltr.py --mode train --ce-tag minilm --export-scores
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from evaluate import evaluate_all_hirers
from tag_features import TAG_FEATURES
from taxonomy import TAXONOMY_FEATURES

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
FEAT_DIR = BASE / "features"
RESULTS_DIR = BASE / "results"
MODELS_DIR = BASE / "models"


def set_dataset(tag: str):
    """Namespace every path by dataset so data_sat runs never touch the
    frozen synthetic-data baseline (features/, results/, models/)."""
    global DATA_DIR, FEAT_DIR, RESULTS_DIR, MODELS_DIR
    DATA_DIR = BASE / tag
    if tag == "data":
        FEAT_DIR, RESULTS_DIR, MODELS_DIR = BASE / "features", BASE / "results", BASE / "models"
    else:
        FEAT_DIR = BASE / f"features_{tag}"
        RESULTS_DIR = BASE / f"results_{tag}"
        MODELS_DIR = BASE / f"models_{tag}"


ID_COLS = {"hire_id", "provider_id", "label", "judged"}


def as_id(value):
    """A query/document id, kept as it appears in the corpus.

    data_sat uses numeric ids ('1', '2'); the v2 corpus uses string refs
    ('H-26499bf5'). Ints stay ints so existing datasets are byte-identical, and
    anything non-numeric is preserved as a string instead of crashing.
    """
    s = str(value).strip()
    return int(s) if s.isdigit() else s


# Ranking objective params, shared by the CV models and the final model so the
# two can never drift apart. `rank:ndcg` uses EXPONENTIAL gain (2^label - 1)
# by default, but evaluate.py scores NDCG with LINEAR gain on the 0-100
# relevance scores. Exponential gain treats grade-3 as 7x grade-1; linear
# treats it as 3x -- so by default the model optimises a different ranking
# than the one being measured. --linear-gain sets ndcg_exp_gain=False to
# align training with evaluation.
MODEL_PARAMS = dict(
    objective="rank:ndcg", n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=2, reg_lambda=1.0,
)
# Set by --binary-relevance in main(); read by train_labels().
BINARY_LABELS = False
# Set by --gain-curve in main(); read by train_labels().
GAIN_CURVE = "none"
DEFAULT_FEATURES = ["bm25_score", "bm25_rank", "dense_cosine", "dense_rank",
                    "rrf_score", "rrf_rank", "budget_fit", "seniority_fit", "avail_immediacy"]
# Features whose raw scale drifts per query, so --normalize-scores rescales them
# within each query. The *_rank features are already monotone within a query, so
# normalising them changes nothing a tree can use.
SCORE_FEATURES = ["bm25_score", "dense_cosine", "rrf_score",
                  "budget_fit", "seniority_fit", "avail_immediacy"]

# Field-split BM25 columns from `bm25_fields.py`: each is an Okapi score against
# ONE provider section rather than the blended whole, so the model can tell WHERE
# a term matched instead of only how much. Raw scale drifts per query for the
# same reason bm25_score's does, so these join SCORE_FEATURES for normalisation.
BM25_FIELD_FEATURES = ["bm25_title_title", "bm25_desc_about", "bm25_desc_services",
                       "bm25_desc_exp", "bm25_field_max", "bm25_field_gap"]


def normalize_by_query(rows, keys, how):
    """Rescale feature(s) WITHIN each query.

    The score features are the ones whose raw scale drifts per query: bm25_score
    depends on query length and term rarity, dense_cosine on the query's
    embedding neighbourhood, rrf_score on how many lists agreed. A tree splits
    on absolute values, so one threshold means a different thing for every
    query. Per-query normalisation makes a split mean the same thing everywhere:
    "how does this candidate rank among its siblings for the same gig".

    This also fixes a train/serve skew on `ce_score`: the OOF scores the ranker
    trains on come from 5 different fold models, while the saved serving model
    scores every query itself.

    Measured on data_sat: +0.006 to +0.010 NDCG@10 (paired bootstrap, 10k
    resamples, significant in both harnesses tried).
    """
    if how == "none":
        return
    if isinstance(keys, str):
        keys = [keys]
    by_query = {}
    for r in rows:
        by_query.setdefault(r["hire_id"], []).append(r)
    for group in by_query.values():
        for key in keys:
            vals = [r[key] for r in group if r[key] == r[key]]  # skip NaN
            if len(vals) < 2:
                continue
            if how == "zscore":
                mu = sum(vals) / len(vals)
                sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
                for r in group:
                    if r[key] == r[key]:
                        r[key] = (r[key] - mu) / sd
            elif how == "rank":
                # 1.0 for the query's best candidate, down to ~0.02 at rank 50;
                # ties receive the identical value.
                rank_of = {v: i + 1 for i, v in enumerate(sorted(set(vals), reverse=True))}
                for r in group:
                    if r[key] == r[key]:
                        r[key] = 1.0 / rank_of[r[key]]


def train_labels(rows):
    """Training target for a row list.

    With --binary-relevance this is grade>=2 (the bar P@5/R@10/MRR use); otherwise
    the raw 0-3 grade. Only ever applied to TRAINING labels -- every reported metric
    is computed against the graded ground truth, so the columns stay comparable.
    """
    y = np.array([r["label"] for r in rows], dtype=np.float32)
    if BINARY_LABELS:
        return (y >= 2).astype(np.float32)
    if GAIN_CURVE == "asym":
        # Asymmetric curve: grade 1 collapsed onto "irrelevant" (0), grades 2/3 kept at
        # the exponential-gain magnitudes 3 and 7, so the trees spend their splits on the
        # 5.5% of pairs that are strong matches. Requires ndcg_exp_gain=False -- otherwise
        # XGBoost re-expands these to 2^label - 1 and the curve is not the one asked for.
        table = np.array([0.0, 0.0, 3.0, 7.0], dtype=np.float32)
        return table[np.clip(y.astype(np.int64), 0, 3)]
    return y


def load_json(p):
    return json.loads(Path(p).read_text())


def load_candidates(top_k: int, ce_tag: str | None, ce_normalize: str = "none",
                    taxonomy: bool = False, tag_features: bool = False,
                    bm25_fields: bool = False):
    path = FEAT_DIR / f"candidates_top{top_k}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `python pipeline/features.py --top-k {top_k}` first")
    rows = list(csv.DictReader(path.open()))

    features = list(DEFAULT_FEATURES)
    if ce_tag:
        ce_path = FEAT_DIR / f"ce_scores_{ce_tag}.csv"
        if not ce_path.exists():
            raise SystemExit(f"{ce_path} not found -- run `python pipeline/rerank_crossencoder.py --mode score --tag {ce_tag}`")
        ce = {(r["hire_id"], r["provider_id"]): float(r["ce_score"]) for r in csv.DictReader(ce_path.open())}
        missing = 0
        for r in rows:
            key = (r["hire_id"], r["provider_id"])
            if key in ce:
                r["ce_score"] = ce[key]
            else:
                r["ce_score"] = float("nan")
                missing += 1
        if missing:
            print(f"  note: {missing} candidate rows had no cross-encoder score (filled with NaN)")
        if ce_normalize != "none":
            normalize_by_query(rows, "ce_score", ce_normalize)
            print(f"  ce_score normalised per query: {ce_normalize}")
        features.append("ce_score")

    if bm25_fields:
        # A missing row here is a REAL zero (the pair simply is not in the
        # top-k pool the field scores were computed for), not an unavailable
        # score -- same distinction the taxonomy merge makes.
        bf_path = FEAT_DIR / "bm25_fields.csv"
        if not bf_path.exists():
            raise SystemExit(f"{bf_path} not found -- run `python pipeline/bm25_fields.py "
                             f"--data-dir {DATA_DIR} --tag {DATA_DIR.name}`")
        bf = {(r["hire_id"], r["provider_id"]): [float(r[c]) for c in BM25_FIELD_FEATURES]
              for r in csv.DictReader(bf_path.open())}
        unmatched = 0
        for r in rows:
            vals = bf.get((str(r["hire_id"]), str(r["provider_id"])))
            if vals is None:
                vals = [0.0] * len(BM25_FIELD_FEATURES)
                unmatched += 1
            for c, v in zip(BM25_FIELD_FEATURES, vals):
                r[c] = v
        if unmatched:
            print(f"  note: {unmatched} candidate rows had no field-BM25 score (filled 0.0)")
        features.extend(BM25_FIELD_FEATURES)

    if taxonomy:
        # Card 3's "Recall Channel Matching Scores", third channel. Unlike the
        # cross-encoder, a missing row here is a REAL zero (no shared tag), not an
        # unavailable score, so it is filled with 0.0 rather than NaN.
        tx_path = FEAT_DIR / "taxonomy_scores.csv"
        if not tx_path.exists():
            raise SystemExit(f"{tx_path} not found -- build it with `python pipeline/build_v2_corpus.py "
                             f"--data-dir data/new --tag {DATA_DIR.name}`")
        tx = {(r["hire_id"], r["provider_id"]): [float(r[f]) for f in TAXONOMY_FEATURES]
              for r in csv.DictReader(tx_path.open())}
        unmatched = 0
        for r in rows:
            vals = tx.get((str(r["hire_id"]), str(r["provider_id"])))
            if vals is None:
                vals = [0.0] * len(TAXONOMY_FEATURES)
                unmatched += 1
            for f, v in zip(TAXONOMY_FEATURES, vals):
                r[f] = v
        print(f"  taxonomy features merged ({len(TAXONOMY_FEATURES)} cols); "
              f"{unmatched} candidate rows had no tag overlap (filled 0.0)")
        features.extend(TAXONOMY_FEATURES)

    if tag_features:
        # SkillsFuture tag-match features (tag_features.py). Built in a separate process:
        # it loads torch, which segfaults alongside xgboost on macOS.
        tf_path = FEAT_DIR / "tag_features.csv"
        if not tf_path.exists():
            raise SystemExit(f"{tf_path} not found -- build it with "
                             f"`python pipeline/tag_features.py --data-dir {DATA_DIR.relative_to(BASE)}`")
        tf = {(r["hire_id"], r["provider_id"]): r for r in csv.DictReader(tf_path.open())}
        missing = [r for r in rows if (r["hire_id"], r["provider_id"]) not in tf]
        if missing:
            raise SystemExit(f"{tf_path.name} is stale: {len(missing)} candidate rows have no tag "
                             f"features -- rebuild it after features.py")
        for r in rows:
            src = tf[(r["hire_id"], r["provider_id"])]
            for f in TAG_FEATURES:
                r[f] = src[f]
        print(f"  tag features merged ({len(TAG_FEATURES)} cols): {TAG_FEATURES}")
        features.extend(TAG_FEATURES)

    for r in rows:
        for f in features:
            r[f] = float(r[f]) if r.get(f) not in ("", None) else float("nan")
        r["label"] = int(r["label"]) if r["label"] not in ("", None) else None
        r["hire_id"] = as_id(r["hire_id"])
        r["provider_id"] = as_id(r["provider_id"])
    return rows, features


def order_by(rows, key, descending=True):
    """Rank a query's candidates by `key`. Scores sort descending; rank
    columns (rrf_rank) sort ascending so rank 1 wins."""
    return [r["provider_id"] for r in sorted(rows, key=lambda r: (-r[key] if descending else r[key]))]


def full_lists(ranked_head, rrf_full):
    out = {}
    for hid, head in ranked_head.items():
        seen = set(head)
        out[str(hid)] = head + [pid for pid in rrf_full.get(str(hid), []) if pid not in seen]
    return out


def report(name, ranked_by_query, gt, rrf_full):
    m = evaluate_all_hirers(full_lists(ranked_by_query, rrf_full), gt, ks=(5, 10))
    print(f"  {name:34s} P@5={m['precision@5']:.3f} R@5={m['recall@5']:.3f} NDCG@5={m['ndcg@5']:.3f} "
          f"P@10={m['precision@10']:.3f} R@10={m['recall@10']:.3f} NDCG@10={m['ndcg@10']:.4f} MRR={m['mrr']:.3f}")
    return m


def train_oof(rows, features, folds, seed, gt, rrf_full):
    """Return out-of-fold LambdaMART scores keyed by (hire_id, provider_id)."""
    by_query = defaultdict(list)
    for r in rows:
        by_query[r["hire_id"]].append(r)
    queries = sorted(by_query)
    oof = {}
    importances = []
    fold_metrics = []

    gkf = GroupKFold(n_splits=folds)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(queries, groups=queries), start=1):
        tr_q = sorted(queries[i] for i in tr_idx)
        te_q = sorted(queries[i] for i in te_idx)

        tr_rows = [r for q in tr_q for r in by_query[q] if r["label"] is not None]
        if not tr_rows:
            continue
        X = np.array([[r[f] for f in features] for r in tr_rows], dtype=np.float32)
        y = train_labels(tr_rows)
        groups = [sum(1 for r in tr_rows if r["hire_id"] == q) for q in tr_q]

        model = xgb.XGBRanker(**MODEL_PARAMS, random_state=seed, n_jobs=4)
        model.fit(X, y, group=groups)

        te_rows = [r for q in te_q for r in by_query[q]]
        Xt = np.array([[r[f] for f in features] for r in te_rows], dtype=np.float32)
        preds = model.predict(Xt)
        for r, p in zip(te_rows, preds):
            oof[(r["hire_id"], r["provider_id"])] = float(p)

        importances.append(model.feature_importances_)
        # per-fold sanity metric (fold-local ranking, tail appended)
        fold_rank = {}
        for r, p in zip(te_rows, preds):
            fold_rank.setdefault(r["hire_id"], []).append((r["provider_id"], float(p)))
        ranked = {q: [pid for pid, _ in sorted(v, key=lambda x: -x[1])] for q, v in fold_rank.items()}
        fold_metrics.append(evaluate_all_hirers(full_lists(ranked, rrf_full), gt, ks=(10,))["ndcg@10"])
        print(f"  fold {fold}: {len(tr_q)} train q / {len(te_q)} held-out q  NDCG@10={fold_metrics[-1]:.4f}")

    imp = np.mean(importances, axis=0)
    print("\n  mean feature importance:")
    for f, v in sorted(zip(features, imp), key=lambda x: -x[1]):
        print(f"    {f:18s} {v:.4f}")
    if fold_metrics:
        print(f"  fold NDCG@10: mean={np.mean(fold_metrics):.4f} std={np.std(fold_metrics):.4f}")
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "ablation"], default="ablation")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--ce-tag", default=None, help="include features/ce_scores_<tag>.csv as a feature")
    ap.add_argument("--bm25-fields", action="store_true",
                    help="include features/bm25_fields.csv (per-section BM25) as features")
    ap.add_argument("--drop-features", default=None,
                    help="comma-separated feature names to drop (ablation), e.g. rrf_rank,rrf_score")
    ap.add_argument("--taxonomy", action="store_true",
                    help="include the taxonomy recall channel's match scores (Card 3) from "
                         "features_<dataset>/taxonomy_scores.csv")
    ap.add_argument("--tag-features", action="store_true",
                    help="add the SkillsFuture tag-match features from features_<dataset>/tag_features.csv "
                         "(build with tag_features.py). Shipped configuration on data_set_v5_tax")
    ap.add_argument("--ce-normalize", default="none", choices=["none", "zscore", "rank"],
                    help="rescale ce_score within each query. The raw logit's per-query mean "
                         "spans ~9 units, and OOF scores come from 5 fold models while serving "
                         "uses one -- so an un-normalised ce_score means different things per query")
    ap.add_argument("--normalize-scores", default="none", choices=["none", "zscore", "rank"],
                    help="rescale the score features within each query (recommended: zscore). "
                         "Their raw scale drifts per query, so an absolute tree split means "
                         "something different for every gig")
    ap.add_argument("--objective", default="rank:ndcg",
                    choices=["rank:ndcg", "rank:map", "rank:pairwise"],
                    help="ranking objective. rank:map/rank:pairwise REJECT graded labels, so "
                         "pair them with --binary-relevance")
    ap.add_argument("--binary-relevance", action="store_true",
                    help="binarise the TRAINING labels to grade>=2, which is the relevance bar "
                         "P@5/R@10/MRR already score against. Evaluation still uses the graded "
                         "ground truth, so NDCG stays comparable -- but the model now optimises "
                         "membership of the top-5 rather than graded ordering, and the two pull "
                         "in opposite directions (measured: best P@5/MRR of anything tried, "
                         "worst NDCG@10)")
    ap.add_argument("--linear-gain", action="store_true",
                    help="rank:ndcg with ndcg_exp_gain=False, matching evaluate.py's linear-gain "
                         "NDCG instead of XGBoost's default exponential (2^label-1) gain")
    ap.add_argument("--gain-curve", default="none", choices=["none", "asym"],
                    help="training-label remap for rank:ndcg: 'asym' maps grades 0/1 -> 0 and "
                         "2/3 -> 3/7 (use with --linear-gain so XGBoost does not re-expand "
                         "them). Metrics always use the graded ground truth")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--export-scores", action="store_true",
                    help="write results/ltr_<tag>.json ranked lists + 0-100 calibrated scores")
    ap.add_argument("--tag", default="ltr")
    ap.add_argument("--data-dir", default="data",
                    help="dataset folder under pipeline/ (e.g. data_sat); outputs are namespaced accordingly")
    args = ap.parse_args()
    set_dataset(args.data_dir)
    global BINARY_LABELS, GAIN_CURVE
    BINARY_LABELS = args.binary_relevance
    GAIN_CURVE = args.gain_curve
    if args.objective != MODEL_PARAMS["objective"]:
        MODEL_PARAMS["objective"] = args.objective
        print(f"objective: {args.objective}")
    if args.binary_relevance:
        print("training labels: BINARY (grade>=2) -- metrics still use the graded ground truth")
    if args.linear_gain:
        MODEL_PARAMS["ndcg_exp_gain"] = False
        print("gain: LINEAR (ndcg_exp_gain=False), aligned with evaluate.py")
    if args.gain_curve != "none":
        if MODEL_PARAMS.get("ndcg_exp_gain", True):
            print("WARNING: --gain-curve asym without --linear-gain; XGBoost will re-expand "
                  "the remapped labels via 2^label - 1")
        print("training labels: ASYM (0/1 -> 0, 2 -> 3, 3 -> 7) -- metrics still use the "
              "graded ground truth")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # The tail beyond the top-k pool only pads each ranked list to a full
    # permutation; no metric at K<=10 reads it. Tolerate the dump being absent
    # so training runs from candidates_top50.csv alone (it is ~31 MB and
    # regenerable with `run_pipeline.py --data-dir data/data_set_v5`).
    _rrf_path = RESULTS_DIR / "rrf_k60.json"
    rrf_full = load_json(_rrf_path) if _rrf_path.exists() else {}
    gt = load_json(DATA_DIR / "ground_truth_llm.json")

    rows, features = load_candidates(args.top_k, args.ce_tag, args.ce_normalize, args.taxonomy,
                                     args.tag_features, args.bm25_fields)

    if args.drop_features:
        _drop = {f.strip() for f in args.drop_features.split(",") if f.strip()}
        unknown = _drop - set(features)
        if unknown:
            raise SystemExit(f"--drop-features names not in the feature set: {sorted(unknown)}")
        features = [f for f in features if f not in _drop]
        print(f"dropped features: {sorted(_drop)}  -> {len(features)} remain")

    # Fail loudly rather than silently training on an empty target. Grades must key on
    # the SAME ids the corpus uses -- for the v2 corpus that is `hirer_ref` and the
    # 1-based row index of providers.csv; any other convention joins to nothing, and
    # a silent zero-overlap otherwise looks like a broken model rather than a bad join.
    q_ids = {str(r["hire_id"]) for r in rows}
    n_labelled = sum(1 for r in rows if r["label"] is not None)
    gt_pairs = sum(len(v) for v in gt.values())
    gt_match_q = sum(1 for q in gt if str(q) in q_ids)
    gt_match_rows = sum(1 for r in rows if str(r["provider_id"]) in gt.get(str(r["hire_id"]), {}))
    print(f"ground truth: {len(gt)} queries / {gt_pairs} pairs -> {gt_match_q} queries and "
          f"{gt_match_rows} candidate rows match this corpus; {n_labelled} rows carry a label")
    if not n_labelled:
        raise SystemExit(
            "FAIL: no candidate row carries a training label, so there is nothing to fit.\n"
            f"  Build {FEAT_DIR.name}/candidates_top50.csv against a ground truth keyed on\n"
            f"  hire_id from {DATA_DIR.name}/hirers.json and provider_id from {DATA_DIR.name}/providers.json."
        )
    if gt_pairs and not gt_match_rows:
        raise SystemExit(
            "FAIL: ground truth exists but none of its pairs match a candidate row -- the grades\n"
            "  are keyed on different ids than the corpus. Check the two id spaces above."
        )

    if args.normalize_scores != "none":
        normalize_by_query(rows, SCORE_FEATURES + (BM25_FIELD_FEATURES if args.bm25_fields else []),
                           args.normalize_scores)
        print(f"score features normalised per query: {args.normalize_scores}")
    print(f"loaded {len(rows)} candidate rows / {len({r['hire_id'] for r in rows})} queries")
    print(f"features ({len(features)}): {features}\n")

    by_query = defaultdict(list)
    for r in rows:
        by_query[r["hire_id"]].append(r)

    # ---------------- baselines ------------------------------------------
    print("baselines (same candidate universe, full lists):")
    base_rrf = {q: order_by(rs, "rrf_rank", descending=False) for q, rs in by_query.items()}
    m_rrf = report("RRF (frozen baseline)", base_rrf, gt, rrf_full)
    if args.ce_tag:
        base_ce = {q: order_by(rs, "ce_score") for q, rs in by_query.items()}
        m_ce = report(f"cross-encoder ({args.ce_tag})", base_ce, gt, rrf_full)

    # ---------------- LambdaMART (out-of-fold) ---------------------------
    print("\nLambdaMART (GroupKFold by query, out-of-fold predictions):")
    oof = train_oof(rows, features, args.folds, args.seed, gt, rrf_full)

    ltr_rank = {}
    for q, rs in by_query.items():
        scored = [(r["provider_id"], oof.get((q, r["provider_id"]), float("-inf"))) for r in rs]
        ltr_rank[q] = [pid for pid, _ in sorted(scored, key=lambda x: -x[1])]
    m_ltr = report("LambdaMART (" + ", ".join(f for f in features if f != "rrf_rank") + ")", ltr_rank, gt, rrf_full)

    # Dump the OUT-OF-FOLD ranked lists so the reported row is traceable to an
    # artifact, exactly like results/rrf_k60.json and results/ce_<tag>.json.
    # This is the OOF ranking (not the full-data model's), so it stays honest.
    oof_path = RESULTS_DIR / f"{args.tag}.json"
    oof_path.write_text(json.dumps({str(q): [int(pid) for pid in rs] for q, rs in ltr_rank.items()}, indent=2))
    print(f"wrote out-of-fold ranked lists to {oof_path}")

    # ---------------- final trained model on all queries ------------------
    all_rows = sorted([r for r in rows if r["label"] is not None], key=lambda r: r["hire_id"])
    X = np.array([[r[f] for f in features] for r in all_rows], dtype=np.float32)
    y = train_labels(all_rows)
    all_q = sorted({r["hire_id"] for r in all_rows})
    groups = [sum(1 for r in all_rows if r["hire_id"] == q) for q in all_q]
    final = xgb.XGBRanker(**MODEL_PARAMS, random_state=args.seed, n_jobs=4)
    final.fit(X, y, group=groups)
    model_path = MODELS_DIR / f"{args.tag}_xgb.json"
    final.save_model(str(model_path))
    print(f"\nsaved full-data model to {model_path}")

    if args.export_scores:
        # NOTE: min-max per query -> 0..100 so the value can leave our service as a
        # semantic_score. Ranking is what matters; the absolute scale is a contract
        # detail and must be documented to the sponsor (fixed 0-100 normalisation).
        out = {}
        for q, rs in by_query.items():
            Xq = np.array([[r[f] for f in features] for r in rs], dtype=np.float32)
            preds = final.predict(Xq)
            order = np.argsort(-preds)
            pmin, pmax = float(preds.min()), float(preds.max())
            span = (pmax - pmin) or 1.0
            out[str(q)] = [
                {"provider_id": int(rs[i]["provider_id"]),
                 "semantic_score": round(100.0 * (float(preds[i]) - pmin) / span, 2),
                 "rank": int(np.where(order == i)[0][0]) + 1}
                for i in range(len(rs))
            ]
        path = RESULTS_DIR / f"{args.tag}_scores.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote calibrated 0-100 scores to {path}")

    print("\n--- ablation summary ---")
    print(f"  RRF baseline        NDCG@10={m_rrf['ndcg@10']:.4f}  P@5={m_rrf['precision@5']:.3f}  MRR={m_rrf['mrr']:.3f}")
    if args.ce_tag:
        print(f"  + cross-encoder     NDCG@10={m_ce['ndcg@10']:.4f}  P@5={m_ce['precision@5']:.3f}  MRR={m_ce['mrr']:.3f}")
    print(f"  + LambdaMART (OOF)  NDCG@10={m_ltr['ndcg@10']:.4f}  P@5={m_ltr['precision@5']:.3f}  MRR={m_ltr['mrr']:.3f}")
    # The ~0.02 figure comes from the 130-query synthetic corpus. It is NOT
    # valid at data_sat's scale: standard error scales as 1/sqrt(n), so at
    # 1,023 queries the bar is ~2.8x tighter (~0.007). Printing a flat 0.02
    # here caused a real NDCG@10 regression to be read as "flat vs RRF".
    n_q = len({r["hire_id"] for r in rows}) if rows else 0
    threshold = 0.02
    if n_q:
        threshold = 0.02 * (130.0 / n_q) ** 0.5
    print(f"\nReminder: {n_q or '?'} queries -- treat any delta under "
          f"~{threshold:.3f} as noise. Report paired bootstrap CIs before claiming "
          f"an improvement.")


if __name__ == "__main__":
    main()
