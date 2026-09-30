"""
Same-grader comparison of what the tag channel changes, on the test gigs, at K <= 10.

Why: the original labels were graded by one grader, the tag-only pairs by labeller.py's reproduction, which is
more lenient (results_tag/calibration.json). Comparing a system whose top-10 was graded mostly by the original
with one graded mostly by the reproduction mixes the two scales. Here every pair in every compared top-10 is
regraded by the reproduction, so the comparison uses one grader and every list is fully judged.

Compared lists, per test gig, all cut at 10:
    channels alone          bm25, dense, tag           (each channel's own top-10)
    Stage-1 fusion          rrf2 = RRF(bm25, dense),  rrf3 = RRF(bm25, dense, tag)
    shipped Stage-2 ranker  fused2 / fused3 = 0.7 linz + 0.3 noce trained without / with tag_score, tag_rank
The Stage-2 rankers train on the ORIGINAL grades only (same for both systems; see eval_tag_downstream.py).

Metrics at K = 5 and 10 (P, R, NDCG) and MRR@10. Recall's denominator is the relevant pairs among those graded
(the union of the compared top-10s), identical for every list of a gig, so it ranks systems fairly but is not a
recall over all providers. Grade >= 2 is relevant; NDCG uses the 0/33/67/100 scale of evaluate.py.

Run from the repo root:
    python pipeline/eval_tag_samegrader.py pool      # writes judging_pools_regrade.json, seeds judgments_regrade.jsonl
    python pipeline/labeller.py pairs --data-dir pipeline/data_sat --pools judging_pools_regrade.json --out judgments_regrade.jsonl
    python pipeline/eval_tag_samegrader.py eval      # writes results_tag/samegrader.json
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import eval_tag_downstream as ds
import rerank_ltr as ltr
from evaluate import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

BASE = Path(__file__).parent
TOP = 10
KS = (5, 10)
METRICS = [f"{m}@{k}" for k in KS for m in ("P", "R", "NDCG")] + ["MRR@10"]
SCORE = {1: 33, 2: 67, 3: 100}
CHANNELS = ("bm25", "dense", "tag")


def test_gigs(data_dir: Path, seed: int) -> list[int]:
    """The dev/test split of eval_tag_channel.py sat: gigs with an original grade >= 2 provider, permuted with
    the same seeded stream, second half."""
    hirers = json.loads((data_dir / "hirers.json").read_text(encoding="utf-8"))
    gt = json.loads((data_dir / "ground_truth_llm.json").read_text(encoding="utf-8"))
    gigs = [str(h["hire_id"]) for h in hirers if any(s >= 40 for s in gt.get(str(h["hire_id"]), {}).values())]
    perm = ds.rng_for(seed, "sat/split").permutation(len(gigs))
    return [int(gigs[i]) for i in sorted(perm[len(gigs) // 2:])]


def channel_lists(data_dir: Path, top: int = TOP) -> dict[str, dict[int, list[int]]]:
    """Each channel's own top `top` per gig, exactly as eval_tag_channel.py ranks them: refined BM25 and dense
    from the cached embeddings, the tag channel with TagChannel's defaults (30 tags per text, b 0.75, k1 1.2).
    (The RRF candidate CSVs would miss channel hits that fall outside the fused top-50.)"""
    from corpus import hirer_text, provider_text
    from eval_tag_channel import cached_embeddings, existing_channels
    from tag_channel import TagChannel

    load = lambda name: json.loads((data_dir / name).read_text(encoding="utf-8"))
    hirers, providers = load("hirers.json"), load("providers.json")
    ex = existing_channels(hirers, providers,
                           cached_embeddings("sat_docs_mxbai_by_text", [provider_text(p) for p in providers]),
                           cached_embeddings("sat_queries_mxbai_by_text", [hirer_text(h) for h in hirers]))
    tag = TagChannel(data_dir)
    out = {c: {} for c in CHANNELS}
    for h in hirers:
        q = int(h["hire_id"])
        out["bm25"][q] = [int(p) for p in ex["bm25"][str(q)].ids[:top]]
        out["dense"][q] = [int(p) for p in ex["dense"][str(q)].ids[:top]]
        out["tag"][q] = [int(p) for p, _ in tag.rank(h["hire_id"], top_k=top)]
    return out


def compared_lists(data_dir: Path, base_csv: Path, tag_csv: Path, original_grades: dict, args) -> dict[str, dict[int, list[int]]]:
    """Every list compared, cut at TOP: channels, rrf2/rrf3, fused2/fused3."""
    out = dict(channel_lists(data_dir))
    base = ds.ranked_lists(base_csv, list(ltr.DEFAULT_FEATURES), original_grades, args.folds, args.seed)
    tag = ds.ranked_lists(tag_csv, list(ltr.DEFAULT_FEATURES) + ds.TAG_FEATURES, original_grades, args.folds, args.seed)
    out["rrf2"], out["rrf3"] = base["rrf"], tag["rrf"]
    out["fused2"], out["fused3"] = base["fused"], tag["fused"]
    return {name: {q: ids[:TOP] for q, ids in per_q.items()} for name, per_q in out.items()}


def pool_pairs(lists: dict[str, dict[int, list[int]]], gigs: list[int]) -> dict[str, list[int]]:
    pools = {}
    for q in gigs:
        ids = sorted({p for per_q in lists.values() for p in per_q.get(q, [])})
        pools[str(q)] = ids
    return pools


def metrics_per_gig(lists: dict[int, list[int]], gt: dict, gigs: list[int]) -> dict[str, np.ndarray]:
    out = {m: [] for m in METRICS}
    for q in gigs:
        ids, row = lists.get(q, [])[:TOP], gt.get(str(q), {})
        for k in KS:
            out[f"P@{k}"].append(precision_at_k(ids, row, k))
            r = recall_at_k(ids, row, k)
            out[f"R@{k}"].append(np.nan if r is None else r)
            out[f"NDCG@{k}"].append(ndcg_at_k(ids, row, k))
        out["MRR@10"].append(reciprocal_rank(ids, row))
    return {m: np.array(v, dtype=float) for m, v in out.items()}


def grades_from_records(records: list[dict]) -> dict[str, dict[str, int]]:
    grades = {}
    for r in records:
        if r.get("status") == "ok":
            grades.setdefault(r["hire_id"], {})[r["provider_id"]] = r["grade"]
    return grades


def read_records(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["pool", "eval"])
    ap.add_argument("--data-dir", default="data_sat")
    ap.add_argument("--baseline-csv", type=Path, default=None)
    ap.add_argument("--tag-csv", type=Path, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=ds.SEED)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    data_dir = BASE / args.data_dir
    feat_dir = BASE / f"features_{args.data_dir}"
    base_csv = args.baseline_csv or feat_dir / "candidates_top50.csv"
    tag_csv = args.tag_csv or feat_dir / "candidates_top50_tag.csv"
    original = json.loads((data_dir / "llm_judgments_merged.json").read_text(encoding="utf-8"))
    gigs = test_gigs(data_dir, args.seed)
    lists = compared_lists(data_dir, base_csv, tag_csv, original, args)
    pools = pool_pairs(lists, gigs)
    regrade_file = data_dir / "judgments_regrade.jsonl"

    if args.mode == "pool":
        (data_dir / "judging_pools_regrade.json").write_text(json.dumps(pools, indent=1), encoding="utf-8")
        wanted = {(h, str(p)) for h, ps in pools.items() for p in ps}
        have = {(r["hire_id"], r["provider_id"]) for r in read_records(regrade_file) if r.get("status") == "ok"}
        # seed with grades the reproduction already produced for these pairs (no point paying for them twice)
        seeded = 0
        with regrade_file.open("a", encoding="utf-8") as fh:
            for source in ("judgments_tag.jsonl", "calibration.jsonl"):
                for r in read_records(data_dir / source):
                    key = (r["hire_id"], r["provider_id"])
                    if r.get("status") == "ok" and key in wanted and key not in have:
                        fh.write(json.dumps(r) + "\n")
                        have.add(key)
                        seeded += 1
        print(f"{len(gigs)} test gigs, {len(wanted)} pairs in the compared top-10s; {seeded} seeded from earlier "
              f"reproduction grades, {len(wanted - have)} still to grade -> judging_pools_regrade.json")
        for name, per_q in lists.items():
            print(f"  {name:7} mean list length on test gigs {np.mean([len(per_q.get(q, [])) for q in gigs]):.2f}")
        return

    t0 = time.time()
    grades = grades_from_records(read_records(regrade_file))
    missing = [(h, p) for h, ps in pools.items() for p in ps if str(p) not in grades.get(h, {})]
    if missing:
        raise SystemExit(f"{len(missing)} pairs in the pool have no reproduction grade yet; run labeller.py pairs first")
    gt = {h: {p: SCORE[g] for p, g in ps.items() if g > 0} for h, ps in grades.items()}
    idx = ds.rng_for(args.seed, "samegrader/boot").integers(0, len(gigs), size=(args.n_boot, len(gigs)))
    per = {name: metrics_per_gig(lst, gt, gigs) for name, lst in lists.items()}
    n_pos = sum(1 for q in gigs if gt.get(str(q)))
    print(f"{len(gigs)} test gigs, {n_pos} with a relevant pair among the graded; one grader (rubric_0_3.v2-repro)")
    print(f"{'list':8}" + "".join(f"{m:>9}" for m in METRICS))
    means = {}
    for name, ms in per.items():
        means[name] = {m: float(np.nanmean(v)) for m, v in ms.items()}
        print(f"{name:8}" + "".join(f"{means[name][m]:9.4f}" for m in METRICS))
    diffs = {}
    print("paired differences over test gigs (95% CI; * = excludes 0):")
    for label, a, b in (("rrf3 - rrf2", "rrf3", "rrf2"), ("fused3 - fused2", "fused3", "fused2"),
                        ("tag - bm25", "tag", "bm25"), ("tag - dense", "tag", "dense")):
        diffs[label] = {}
        cells = []
        for m in METRICS:
            mean, lo, hi = ds.boot_ci(per[a][m] - per[b][m], idx)
            diffs[label][m] = [mean, lo, hi]
            cells.append(f"{m} {mean:+.4f}{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"  {label:16}" + "  ".join(cells))
    out = {"n_test_gigs": len(gigs), "n_with_relevant": n_pos, "grader": "rubric_0_3.v2-repro", "means": means,
           "paired_differences": diffs, "pairs_graded": sum(len(v) for v in pools.values()),
           "seconds": round(time.time() - t0, 1)}
    path = BASE / "results_tag" / "samegrader.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
