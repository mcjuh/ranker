"""
Same-grader comparison of tag-channel variants, on the dev or the test gigs, at K <= 10.

The hubness-corrected variants (tag_corpus.py --hubness center, see TAG_CHANNEL.md) are compared with the current
channel the way eval_tag_samegrader.py compares the current channel with nothing: every pair in every compared
top-10 is graded by one grader (labeller.py, rubric_0_3.v2-repro), so every list is fully judged and the numbers do
not depend on which retriever happened to be pooled. With --stage2 the shipped Stage-2 ranker (eval_tag_downstream.py: linz + noce LambdaMART, fused 0.7 / 0.3, 5-fold by
gig, trained on the ORIGINAL grades only, for both systems) is added: fused2 from the candidate CSV without tag features,
fused3_hc from the hc candidate CSV with tag_score / tag_rank. Both CSVs must come from the same day and cache.

Compared lists, per gig, all cut at 10:
    channels alone   bm25, dense, tag (raw cosine, BM25), tag_hc (centred tags, BM25), tag_hcw (centred tags, weighted cosine)
    fusions          rrf2 = RRF(bm25, dense),  rrf3 / rrf3_hc / rrf3_hcw = RRF(bm25, dense, <tag channel>), weight 1

Choose settings on --split dev, then confirm once on --split test. Run from the repo root:
    python pipeline/tag_corpus.py --data-dir data_sat --hubness center        # once
    python pipeline/eval_tag_variants.py pool --split dev                     # writes judging_pools_variants_dev.json
    python pipeline/labeller.py pairs --data-dir pipeline/data_sat --pools judging_pools_variants_dev.json --out judgments_variants.jsonl
    python pipeline/eval_tag_variants.py eval --split dev                     # writes results_tag/variants_dev.json

Stage 2 (after features.py --data-dir data_sat, once without and once with --tag-channel --tag-variant hc, the
baseline copied to candidates_top50_regen.csv so the tracked CSV stays as committed):
    python pipeline/eval_tag_variants.py pool --split test --stage2           # judging_pools_variants_test_stage2.json
    python pipeline/labeller.py pairs --data-dir pipeline/data_sat --pools judging_pools_variants_test_stage2.json --out judgments_variants.jsonl
    python pipeline/eval_tag_variants.py eval --split test --stage2           # writes results_tag/variants_test_stage2.json

Robustness to the reproduced grader's leniency (audit/RESULTS.md): `eval --relevance all` repeats every comparison
with grade >= 2 kept only if the grader's own P(>= 2) is high and/or the pair has no serious term mismatch.
No new grades are needed.

`pool` seeds judgments_variants.jsonl with every reproduction grade that already exists for a pooled pair
(judgments_tag.jsonl, judgments_regrade.jsonl, calibration.jsonl), so only new pairs are sent to the grader. The
original grader's files are never used: mixing the two graders is what made the extended labels misleading.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import eval_tag_downstream as ds
import eval_tag_samegrader as sg

BASE = Path(__file__).parent
TOP = sg.TOP
SEED_FILES = ("judgments_regrade.jsonl", "judgments_tag.jsonl", "calibration.jsonl")
OUT_FILE = "judgments_variants.jsonl"
# name -> TagChannel keyword arguments
TAG_CHANNELS = {"tag": {}, "tag_hc": {"variant": "hc"}, "tag_hcw": {"variant": "hc", "scorer": "wcos"}}
FUSED = {"rrf3": "tag", "rrf3_hc": "tag_hc", "rrf3_hcw": "tag_hcw"}
COMPARISONS = (
    ("rrf3 - rrf2", "rrf3", "rrf2"),
    ("rrf3_hc - rrf2", "rrf3_hc", "rrf2"),
    ("rrf3_hcw - rrf2", "rrf3_hcw", "rrf2"),
    ("rrf3_hc - rrf3", "rrf3_hc", "rrf3"),
    ("rrf3_hcw - rrf3", "rrf3_hcw", "rrf3"),
    ("tag_hc - tag", "tag_hc", "tag"),
    ("tag_hcw - tag", "tag_hcw", "tag"),
    ("tag_hcw - dense", "tag_hcw", "dense"),
    ("fused2 - rrf2", "fused2", "rrf2"),
    ("fused3_hc - fused2", "fused3_hc", "fused2"),
    ("fused3_hc - rrf3_hc", "fused3_hc", "rrf3_hc"),
)
STAGE2_BASELINE_CSV = "candidates_top50_regen.csv"
STAGE2_TAG_CSV = "candidates_top50_tag_hc.csv"

# What counts as relevant. "repro" is the published definition (grade >= 2 from the reproduced labeller). The
# others keep a grade >= 2 only if it passes the check, and demote it to grade 1 (partial) otherwise, because
# pipeline/audit/RESULTS.md found that the reproduction over-credits: min_p keeps pairs whose own
# P(grade >= 2) is high (the audit's confirmation rate rises with it), no_mismatch drops pairs the rubric itself
# forbids from scoring >= 2 (a serious term mismatch). A sweep, not one post-hoc cut-off.
RELEVANCE = {
    "repro": {},
    "p60": {"min_p": 0.6},
    "p80": {"min_p": 0.8},
    "term": {"no_mismatch": True},
    "p80+term": {"min_p": 0.8, "no_mismatch": True},
}
DEMOTED_SCORE = sg.SCORE[1]


def truth_from_records(records: list[dict], min_p: float = 0.0, flagged=None) -> dict[str, dict[str, int]]:
    """{hire_id: {provider_id: score}} on the 0/33/67/100 scale, grade 0 left out. With the defaults it equals the
    published ground truth. A grade >= 2 is demoted to grade 1 if P(grade >= 2) < `min_p` (P is the sum of the last
    two entries of the record's `probs`) or if `flagged(hire_id, provider_id)` says the pair has a serious term
    mismatch. Later records win, as in eval_tag_samegrader.grades_from_records."""
    out: dict[str, dict[str, int]] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        grade = r["grade"]
        if grade >= 2:
            low_confidence = min_p > 0 and sum(r["probs"][2:]) < min_p
            if low_confidence or (flagged is not None and flagged(r["hire_id"], r["provider_id"])):
                grade = 1
        row = out.setdefault(r["hire_id"], {})
        if grade > 0:
            row[r["provider_id"]] = sg.SCORE[grade]
        else:
            row.pop(r["provider_id"], None)
    return out


def mismatch_checker(data_dir: Path):
    """`flagged(hire_id, provider_id)`: the rubric's serious-term-mismatch rule from claude_audit.py. The real
    encoder module is imported first so claude_audit's torch-free stub of it is never installed here."""
    import features  # noqa: F401
    from claude_audit import _serious_mismatch

    load = lambda name: json.loads((data_dir / name).read_text(encoding="utf-8"))
    hirers = {str(h["hire_id"]): h for h in load("hirers.json")}
    providers = {str(p["provider_id"]): p for p in load("providers.json")}
    return lambda h, p: bool(_serious_mismatch(hirers[str(h)], providers[str(p)]))


def split_gigs(data_dir: Path, seed: int, split: str) -> list[int]:
    """The dev / test split of eval_tag_channel.py sat: gigs with an original grade >= 2 provider, permuted with the
    same seeded stream, first half = dev, second half = test (the test half is eval_tag_samegrader.test_gigs)."""
    hirers = json.loads((data_dir / "hirers.json").read_text(encoding="utf-8"))
    gt = json.loads((data_dir / "ground_truth_llm.json").read_text(encoding="utf-8"))
    gigs = [str(h["hire_id"]) for h in hirers if any(s >= 40 for s in gt.get(str(h["hire_id"]), {}).values())]
    perm = ds.rng_for(seed, "sat/split").permutation(len(gigs))
    half = len(gigs) // 2
    chosen = perm[:half] if split == "dev" else perm[half:]
    return [int(gigs[i]) for i in sorted(chosen)]


def stage2_lists(data_dir: Path, gigs: list[int], folds: int, seed: int, top: int = TOP) -> dict[str, dict[int, list[int]]]:
    """The shipped Stage-2 ranker (linz + noce, fused) out-of-fold, without tag features (baseline CSV) and with the
    hc channel's tag_score / tag_rank (hc CSV), cut at `top` for `gigs`. Both train on the original grades."""
    import rerank_ltr as ltr

    feat_dir = BASE / f"features_{data_dir.name}"
    original = json.loads((data_dir / "llm_judgments_merged.json").read_text(encoding="utf-8"))
    out = {}
    for name, csv_name, features in (("fused2", STAGE2_BASELINE_CSV, list(ltr.DEFAULT_FEATURES)),
                                     ("fused3_hc", STAGE2_TAG_CSV, list(ltr.DEFAULT_FEATURES) + ds.TAG_FEATURES)):
        fused = ds.ranked_lists(feat_dir / csv_name, features, original, folds, seed)["fused"]
        out[name] = {q: fused[q][:top] for q in gigs if q in fused}
    return out


def compared_lists(data_dir: Path, gigs: list[int], top: int = TOP) -> dict[str, dict[int, list[int]]]:
    """Every list compared, cut at `top`, from the same full rankings the channels and RRF produce."""
    from corpus import hirer_text, provider_text
    from eval_tag_channel import cached_embeddings, existing_channels
    from retrieval_rrf import rrf_fuse, rrf_fuse_n
    from tag_channel import TagChannel

    load = lambda name: json.loads((data_dir / name).read_text(encoding="utf-8"))
    wanted = set(gigs)
    hirers = [h for h in load("hirers.json") if int(h["hire_id"]) in wanted]
    providers = load("providers.json")
    ex = existing_channels(hirers, providers,
                           cached_embeddings("sat_docs_mxbai_by_text", [provider_text(p) for p in providers]),
                           cached_embeddings("sat_queries_mxbai_by_text", [hirer_text(h) for h in hirers]))
    channels = {name: TagChannel(data_dir, **kwargs) for name, kwargs in TAG_CHANNELS.items()}
    names = ["bm25", "dense", "rrf2", *TAG_CHANNELS, *FUSED]
    out = {name: {} for name in names}
    cut = lambda ranked: [int(p) for p, _ in ranked[:top]]
    for h in hirers:
        q = int(h["hire_id"])
        bm25, dense = ex["bm25"][str(q)].pairs(), ex["dense"][str(q)].pairs()
        out["bm25"][q], out["dense"][q] = cut(bm25), cut(dense)
        out["rrf2"][q] = cut(rrf_fuse(bm25, dense, k=60))
        for name, channel in channels.items():
            tag = channel.rank(q)
            out[name][q] = cut(tag)
            fused = next(f for f, t in FUSED.items() if t == name)
            out[fused][q] = cut(rrf_fuse_n([bm25, dense, tag], k=60, weights=[1.0, 1.0, 1.0]))
    return out


def seed_grades(data_dir: Path, wanted: set[tuple[str, str]], out_file: Path) -> tuple[int, int]:
    """Append to `out_file` the reproduction grades that already exist for wanted pairs. Returns (seeded, have)."""
    have = {(r["hire_id"], r["provider_id"]) for r in sg.read_records(out_file) if r.get("status") == "ok"}
    before, seeded = len(have), 0
    with out_file.open("a", encoding="utf-8") as fh:
        for source in SEED_FILES:
            for r in sg.read_records(data_dir / source):
                key = (r["hire_id"], r["provider_id"])
                if r.get("status") == "ok" and key in wanted and key not in have:
                    fh.write(json.dumps(r) + "\n")
                    have.add(key)
                    seeded += 1
    return seeded, before


def evaluate_lists(lists: dict, gt: dict, gigs: list[int], idx: np.ndarray, split: str, relevance: str) -> dict:
    """Per-list means and the paired differences of COMPARISONS (95% bootstrap CI over gigs) for one ground truth;
    prints both. `idx` holds the bootstrap resamples of gig positions."""
    per = {name: sg.metrics_per_gig(lst, gt, gigs) for name, lst in lists.items()}
    n_pos = sum(1 for q in gigs if gt.get(str(q)))
    print(f"\n[{relevance}] {split}: {len(gigs)} gigs, {n_pos} with a relevant pair among the graded; one grader "
          f"(rubric_0_3.v2-repro)")
    print(f"{'list':9}" + "".join(f"{m:>9}" for m in sg.METRICS))
    means = {}
    for name, ms in per.items():
        means[name] = {m: float(np.nanmean(v)) for m, v in ms.items()}
        print(f"{name:9}" + "".join(f"{means[name][m]:9.4f}" for m in sg.METRICS))
    diffs = {}
    print("paired differences over gigs (95% CI; * = excludes 0):")
    for label, a, b in COMPARISONS:
        if a not in per or b not in per:
            continue
        diffs[label] = {}
        cells = []
        for m in sg.METRICS:
            mean, lo, hi = ds.boot_ci(per[a][m] - per[b][m], idx)
            diffs[label][m] = [mean, lo, hi]
            cells.append(f"{m} {mean:+.4f}{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"  {label:16}" + "  ".join(cells))
    return {"means": means, "paired_differences": diffs, "n_with_relevant": n_pos}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["pool", "eval", "diff"])
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--data-dir", default="data_sat")
    ap.add_argument("--seed", type=int, default=ds.SEED)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--relevance", nargs="+", choices=[*RELEVANCE, "all"], default=["repro"],
                    help="eval: also score under stricter definitions of relevant (see RELEVANCE); 'repro', the "
                         "published one, is always run first. 'all' = every definition")
    ap.add_argument("--diff", nargs=2, metavar=("WITH", "WITHOUT"), default=["rrf3_hc", "rrf2"],
                    help="diff: the two compared lists; writes the pairs entering (in WITH's top-10 only) and "
                         "leaving (in WITHOUT's only) to audit/diff_pairs_<split>.json")
    ap.add_argument("--stage2", action="store_true",
                    help="also compare the shipped Stage-2 ranker without / with the hc tag features (see above)")
    ap.add_argument("--folds", type=int, default=5, help="--stage2: GroupKFold folds, as eval_tag_downstream.py")
    args = ap.parse_args()
    if "all" in args.relevance:
        args.relevance = list(RELEVANCE)

    data_dir = BASE / args.data_dir
    gigs = split_gigs(data_dir, args.seed, args.split)
    lists = compared_lists(data_dir, gigs)
    if args.stage2:
        lists.update(stage2_lists(data_dir, gigs, args.folds, args.seed))
    stem = f"{args.split}_stage2" if args.stage2 else args.split
    pools = sg.pool_pairs(lists, gigs)
    pool_file = data_dir / f"judging_pools_variants_{stem}.json"
    out_file = data_dir / OUT_FILE

    if args.mode == "pool":
        pool_file.write_text(json.dumps(pools, indent=1), encoding="utf-8")
        wanted = {(h, str(p)) for h, ps in pools.items() for p in ps}
        seeded, before = seed_grades(data_dir, wanted, out_file)
        have = {(r["hire_id"], r["provider_id"]) for r in sg.read_records(out_file) if r.get("status") == "ok"}
        print(f"{args.split}: {len(gigs)} gigs, {len(wanted)} pairs in the compared top-{TOP}s; {seeded} seeded from "
              f"earlier reproduction grades ({before} already in {OUT_FILE}), {len(wanted - have)} still to grade "
              f"-> {pool_file.name}")
        for name, per_q in lists.items():
            print(f"  {name:9} mean list length {np.mean([len(per_q.get(q, [])) for q in gigs]):.2f}")
        return

    if args.mode == "diff":
        with_, without = args.diff
        entering, leaving = [], []
        for q in gigs:
            a, b = lists[with_].get(q, []), lists[without].get(q, [])
            entering += [[str(q), str(p)] for p in a if p not in b]
            leaving += [[str(q), str(p)] for p in b if p not in a]
        changed = sum(1 for q in gigs if set(lists[with_].get(q, [])) != set(lists[without].get(q, [])))
        path = BASE / "audit" / f"diff_pairs_{stem}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"split": args.split, "with": with_, "without": without, "top": TOP,
                                    "entering": entering, "leaving": leaving}, indent=0), encoding="utf-8")
        print(f"{with_} vs {without}, top-{TOP}: {changed}/{len(gigs)} gigs change; {len(entering)} pairs enter, "
              f"{len(leaving)} leave ({len(entering) / len(gigs):.2f} per gig) -> {path}")
        return

    t0 = time.time()
    grades = sg.grades_from_records(sg.read_records(out_file))
    missing = [(h, p) for h, ps in pools.items() for p in ps if str(p) not in grades.get(h, {})]
    if missing:
        raise SystemExit(f"{len(missing)} pairs in the {args.split} pool have no reproduction grade yet; run "
                         f"`eval_tag_variants.py pool --split {args.split}` and labeller.py pairs first")
    # only this evaluation's pool: NDCG and R are normalised by the graded pairs of a gig, so grades for other pools
    # (the file grows with every --stage2 or dev run) must not leak in and shift the published numbers
    in_pool = {(h, str(p)) for h, ps in pools.items() for p in ps}
    records = [r for r in sg.read_records(out_file) if (r["hire_id"], str(r["provider_id"])) in in_pool]
    wanted =["repro"] + [r for r in args.relevance if r != "repro"]          # repro is the published definition
    flagged = mismatch_checker(data_dir) if any(RELEVANCE[r].get("no_mismatch") for r in wanted) else None
    # one bootstrap draw for every definition, so differences are paired across definitions as well as across lists
    idx = ds.rng_for(args.seed, f"variants/boot/{args.split}").integers(0, len(gigs), size=(args.n_boot, len(gigs)))
    by_relevance = {}
    for rel in wanted:
        cfg = RELEVANCE[rel]
        gt = truth_from_records(records, cfg.get("min_p", 0.0), flagged if cfg.get("no_mismatch") else None)
        by_relevance[rel] = evaluate_lists(lists, gt, gigs, idx, args.split, rel)
    means, diffs, n_pos = (by_relevance["repro"][k] for k in ("means", "paired_differences", "n_with_relevant"))
    out = {"split": args.split, "n_gigs": len(gigs), "n_with_relevant": n_pos, "grader": "rubric_0_3.v2-repro",
           "means": means, "paired_differences": diffs, "pairs_graded": sum(len(v) for v in pools.values()),
           "relevance": {rel: v for rel, v in by_relevance.items() if rel != "repro"},
           "seconds": round(time.time() - t0, 1)}
    path = BASE / "results_tag" / f"variants_{stem}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
