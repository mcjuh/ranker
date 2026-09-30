"""
Downstream effect of the tag channel: the full Stage-1 -> Stage-2 path, with and without it.

    RRF(bm25, dense) pool        -> RRF order, LambdaMART "linz", LambdaMART "noce", fused 0.7/0.3
    RRF(bm25, dense, tag) pool   -> the same four, with tag_score/tag_rank added as features

"linz" and "noce" are the two rankers the shipped system fuses (RERANK_README.md): linear gain with
per-query z-scored score features, and exponential gain on raw features; the shipped list is their
weighted RRF (0.7 linz + 0.3 noce, k = 60). Everything else is shared by both systems: GroupKFold by gig
(5 folds, out-of-fold predictions only), seed 7, the same 50-candidate cut, the same metrics.

Metrics (evaluate.py, relevant = score >= 40 = grade >= 2): P@K, R@K, NDCG@K for K = 5, 10, 20, and MRR.
P and NDCG average over all gigs, as README.md does; R averages over the gigs that have a relevant provider.
Lists are the 50 candidates only, so K <= 20 never depends on the tail. Paired bootstrap CIs over gigs.

Two label sets, because the tag channel surfaces providers nobody graded:
    original   llm_judgments_merged.json: unjudged counts as irrelevant (biased against the tag channel)
    extended   + llm_judgments_merged_tag.json (--extra-grades): tag-only pairs graded by labeller.py, which is
               more lenient than the original grader and only covers what the tag channel adds
and two treatments of unjudged candidates: "standard" (irrelevant) and "condensed" (dropped from every list).
Training rows are the judged candidates. By default both systems train on the ORIGINAL grades only, whatever
label set they are scored on (--train-labels same trains on the extended grades too; that comparison is unfair,
see the option's help).

Run from the repo root, after features.py (with and without --tag-channel):
    python pipeline/eval_tag_downstream.py --extra-grades
The baseline candidates default to the committed features_data_sat/candidates_top50.csv; pass a freshly
regenerated copy with --baseline-csv when the tag file was generated on a different day (avail_immediacy
depends on the date) so both systems share every non-tag feature.
"""
import argparse
import contextlib
import csv
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import rerank_ltr as ltr
from evaluate import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from fuse_rankers import rrf_fuse as weighted_rrf

BASE = Path(__file__).parent
KS = (5, 10, 20)
METRICS = [f"{m}@{k}" for k in KS for m in ("P", "R", "NDCG")] + ["MRR"]
TAG_FEATURES = ["tag_score", "tag_rank"]
RECIPES = ("linz", "noce")          # RERANK_README.md
FUSE_WEIGHTS = {"linz": 0.7, "noce": 0.3}
SEED = 7


def rng_for(seed: int, purpose: str) -> np.random.Generator:
    import zlib
    return np.random.default_rng([seed, zlib.crc32(purpose.encode())])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_rows(path: Path, features: list[str], grades: dict) -> list[dict]:
    """Candidate rows with numeric features and the label taken from `grades` ({hire: {provider: grade}}),
    None when the pair was never graded. The CSV's own label column is ignored so a label set can be swapped."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for f in features:
            r[f] = float(r[f]) if r.get(f) not in ("", None) else float("nan")
        g = grades.get(r["hire_id"], {}).get(r["provider_id"])
        r["label"] = int(g) if g is not None else None
        r["hire_id"], r["provider_id"] = int(r["hire_id"]), int(r["provider_id"])
    return rows


def by_query(rows: list[dict]) -> dict[int, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[r["hire_id"]].append(r)
    return out


# ---------------------------------------------------------------------------
# Rankers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def recipe_params(name: str):
    """Set rerank_ltr.MODEL_PARAMS for a recipe and restore it afterwards (it is a module global)."""
    saved = dict(ltr.MODEL_PARAMS)
    try:
        if name == "linz":
            ltr.MODEL_PARAMS["ndcg_exp_gain"] = False
        else:
            ltr.MODEL_PARAMS.pop("ndcg_exp_gain", None)
        yield
    finally:
        ltr.MODEL_PARAMS.clear()
        ltr.MODEL_PARAMS.update(saved)


def ranked_lists(csv_path: Path, features: list[str], grades: dict, folds: int, seed: int) -> dict[str, dict[int, list[int]]]:
    """{system: {gig: [provider ids, best first]}} for the RRF order, the two rankers and their fusion."""
    lists = {}
    rows = load_rows(csv_path, features, grades)
    lists["rrf"] = {q: ltr.order_by(rs, "rrf_rank", descending=False) for q, rs in by_query(rows).items()}
    for name in RECIPES:
        rows = load_rows(csv_path, features, grades)                # fresh copy: normalisation mutates rows
        if name == "linz":
            ltr.normalize_by_query(rows, [f for f in ltr.SCORE_FEATURES + ["tag_score"] if f in features], "zscore")
        queries = by_query(rows)
        with recipe_params(name), contextlib.redirect_stdout(io.StringIO()):
            oof = ltr.train_oof(rows, features, folds, seed, {}, {})
        lists[name] = {q: [r["provider_id"] for r in sorted(
            rs, key=lambda r: -oof.get((q, r["provider_id"]), float("-inf")))] for q, rs in queries.items()}
    weights = [FUSE_WEIGHTS[n] for n in RECIPES]
    lists["fused"] = weighted_rrf([lists[n] for n in RECIPES], weights)
    return lists


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_gig(lists: dict[int, list[int]], gt: dict, gigs: list[int], judged: dict | None = None) -> dict[str, np.ndarray]:
    """{metric: array over `gigs`} (NaN where undefined: R for a gig with no relevant provider). With `judged`,
    providers without a grade are dropped from each list first."""
    out = {m: [] for m in METRICS}
    for q in gigs:
        ids = lists[q]
        if judged is not None:
            ids = [p for p in ids if str(p) in judged.get(str(q), {})]
        row = gt.get(str(q), {})
        for k in KS:
            out[f"P@{k}"].append(precision_at_k(ids, row, k))
            r = recall_at_k(ids, row, k)
            out[f"R@{k}"].append(np.nan if r is None else r)
            out[f"NDCG@{k}"].append(ndcg_at_k(ids, row, k))
        out["MRR"].append(reciprocal_rank(ids, row))
    return {m: np.array(v, dtype=float) for m, v in out.items()}


def boot_ci(arr: np.ndarray, idx: np.ndarray):
    means = np.nanmean(arr[idx], axis=1)
    return float(np.nanmean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(title: str, systems: dict[str, dict[str, np.ndarray]], idx: np.ndarray) -> dict:
    """Print mean metrics per system and paired tag-minus-baseline differences with 95% CIs; return both."""
    print(f"\n=== {title} ===")
    print(f"{'system':14}" + "".join(f"{m:>9}" for m in METRICS))
    means = {}
    for name, ms in systems.items():
        means[name] = {m: float(np.nanmean(v)) for m, v in ms.items()}
        print(f"{name:14}" + "".join(f"{means[name][m]:9.4f}" for m in METRICS))
    diffs = {}
    print("paired difference, tag system minus baseline (95% CI over gigs; * = CI excludes 0):")
    for stage in ("rrf", "linz", "noce", "fused"):
        diffs[stage] = {}
        cells = []
        for m in METRICS:
            mean, lo, hi = boot_ci(systems[f"tag_{stage}"][m] - systems[f"base_{stage}"][m], idx)
            diffs[stage][m] = [mean, lo, hi]
            cells.append(f"{m} {mean:+.4f}{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"  {stage:6}" + "  ".join(cells))
    return {"means": means, "tag_minus_baseline": diffs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data_sat")
    ap.add_argument("--baseline-csv", type=Path, default=None, help="default: features_<data-dir>/candidates_top50.csv")
    ap.add_argument("--tag-csv", type=Path, default=None, help="default: features_<data-dir>/candidates_top50_tag.csv")
    ap.add_argument("--extra-grades", action="store_true", help="also report on the extended labels")
    ap.add_argument("--train-labels", choices=["same", "original"], default="original",
                    help="labels the rankers TRAIN on when evaluating on the extended labels. 'original' (default) "
                         "trains both systems on the original grades only. 'same' trains on the extended grades too, "
                         "which is not a fair comparison: which candidates got an extra grade depends on the tag "
                         "channel, a signal the baseline's features cannot see, so the baseline learns that deep-ranked "
                         "graded candidates are positive and then ranks ungraded ones high")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=BASE / "results_tag" / "downstream.json")
    args = ap.parse_args()

    t0 = time.time()
    data_dir = BASE / args.data_dir
    feat_dir = BASE / f"features_{args.data_dir}"
    base_csv = args.baseline_csv or feat_dir / "candidates_top50.csv"
    tag_csv = args.tag_csv or feat_dir / "candidates_top50_tag.csv"
    load = lambda name: json.loads((data_dir / name).read_text(encoding="utf-8"))
    label_sets = {"original": (load("llm_judgments_merged.json"), load("ground_truth_llm.json"))}
    if args.extra_grades:
        label_sets["extended"] = (load("llm_judgments_merged_tag.json"), load("ground_truth_llm_tag.json"))

    result = {"seed": args.seed, "folds": args.folds, "train_labels": args.train_labels,
              "baseline_csv": base_csv.name, "tag_csv": tag_csv.name,
              "recipes": {"fuse_weights": FUSE_WEIGHTS, "k": 60}}
    for label_name, (judged, gt) in label_sets.items():
        print(f"\n################ labels: {label_name} ################", flush=True)
        systems = {}
        for system, path, feats in (("base", base_csv, list(ltr.DEFAULT_FEATURES)),
                                    ("tag", tag_csv, list(ltr.DEFAULT_FEATURES) + TAG_FEATURES)):
            train_grades = label_sets["original"][0] if args.train_labels == "original" else judged
            for stage, lst in ranked_lists(path, feats, train_grades, args.folds, args.seed).items():
                systems[f"{system}_{stage}"] = lst
        gigs = sorted(systems["base_rrf"])
        assert gigs == sorted(systems["tag_rrf"]), "baseline and tag candidates cover different gigs"
        idx = rng_for(args.seed, "downstream/boot").integers(0, len(gigs), size=(args.n_boot, len(gigs)))
        n_pos = sum(1 for q in gigs if any(s >= 40 for s in gt.get(str(q), {}).values()))
        print(f"{len(gigs)} gigs, {n_pos} with a relevant provider; baseline {base_csv.name}, tag {tag_csv.name}")
        result[label_name] = {"n_gigs": len(gigs), "n_gigs_with_relevant": n_pos}
        for treatment, j in (("standard", None), ("condensed", judged)):
            ms = {name: per_gig(lst, gt, gigs, j) for name, lst in systems.items()}
            result[label_name][treatment] = report(f"{label_name} labels, {treatment} lists", ms, idx)
        shares = {name: float(np.mean([np.mean([str(p) in judged.get(str(q), {}) for p in lst[q][:10]]) for q in gigs]))
                  for name, lst in systems.items() if name.endswith("_fused")}
        result[label_name]["judged_share_of_fused_top10"] = shares
        print("share of the fused top-10 that has a grade: " + ", ".join(f"{k} {v:.3f}" for k, v in shares.items()))

    result["seconds"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out} ({result['seconds']}s)")


if __name__ == "__main__":
    main()
