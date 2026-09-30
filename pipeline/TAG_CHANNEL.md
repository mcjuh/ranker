# Tag-ID BM25 recall channel: what was built, what it measured, and the verdict

**Verdict: do not turn it on. Keep the code behind `--tag-channel` (default off).** On the labels that
were graded before the channel existed, adding it to the two-channel RRF either does nothing or hurts. On
labels extended to cover what it surfaces, it adds candidates, but those labels are graded by a more lenient
reproduction of the grader and are pooled in the channel's favour, so they cannot carry a keep decision. The
open question is whether it widens the candidate pool enough for a retrained Stage-2 ranker to use; that was not
tested. Details and caveats below.

## 1. What was built

A third recall channel: BM25 over **taxonomy tag IDs** (the SkillsFuture taxonomy in the GreyGigz schema,
`taxonomy_greygigz/`: 247 tracks, 2,001 roles, 2,088 tags, 43,958 role-tag links).

| File | Role |
|---|---|
| `greygigz.py` | parses the SQL dumps directly; asserts the counts above |
| `retrieval_tagbm25.py` | `TagBM25`: `score = coverage * (B @ (idf * q))`, `idf = ln(1 + (N-df+0.5)/(df+0.5))`, binary tf, zero scores dropped. Baselines: `coverage`, `idf_overlap`, `sum_levels` |
| `tag_corpus.py` | the tagger: for each gig or provider text, the 30 taxonomy tags nearest by cosine (mxbai-embed-large-v1). `data_sat` has no tags of its own, so they are predicted |
| `tag_channel.py` | `TagChannel`: providers are the documents, a gig's predicted tags are the query |
| `retrieval_rrf.py` | `rrf_fuse_n` added (N-way RRF); `rrf_fuse` untouched |
| `--tag-channel` | in `run_pipeline.py`, `features.py`, `build_judging_pools_sat.py`. Default off; a flagged run writes `*_tag` files and never overwrites a baseline file |
| `eval_tag_channel.py` | `roles` (taxonomy only), `tagger`, `sat` (real data). Seed 7, bootstrap CIs. Results in `results_tag/` |
| `labeller.py` | zero-shot 0-3 grader used to label pairs only this channel surfaces (section 5) |

Only `numpy` and `scipy` are used; no new dependency.

## 2. Part A: the channel on the taxonomy alone (an upper bound)

Queries are 5 gold tags sampled from a role; test split; 95% CIs over tag-set-equivalence classes (2,001 roles
fall into 1,606 classes, because 619 roles share a tag set). `results_tag/roles.json`.

| Method | role@1 | role@5 | role@10 | track@1 | track@3 |
|---|---|---|---|---|---|
| **BM25** (k1 1.2, b 0.25 chosen on dev) | **0.642** | 0.956 | 0.992 | 0.766 | 0.968 |
| IDF-overlap | 0.424 | | | | |
| coverage | 0.424 | | | | |
| sum-of-levels | 0.282 | | | | |
| LSA alone | 0.546 | | | | |
| BM25 + LSA, RRF | 0.609 | | | | |

Ablations (role@1 on test):

- **b** (length normalisation): 0.0 gives 0.424, 0.25 gives 0.642, 0.5 gives 0.634, 0.75 gives 0.612,
  1.0 gives 0.590. **k1** does not matter: 0.5 to 3.0 moves role@1 by 0.001.
- **Generic-tag queries** (a role's 5 most widespread tags): role@1 0.343.
- **Wrong-tag queries**: a random wrong tag costs about nothing; a wrong tag borrowed from a sibling role in the
  same track costs 0.27 role@1 while track@1 is unchanged. Sibling-tag errors are exactly what a nearest-tag
  tagger makes.
- **Track-first funnel** (pick the track, then rank roles inside it) loses: the track scorer is 0.762 accurate,
  and it would need 0.90 (role@1) or 0.96 (role@5) to break even.

These numbers use gold tags, so they are an upper bound on what a tagger-fed channel can do.

## 3. The tagger check (how much the upper bound shrinks)

400 role descriptions from the taxonomy (200 dev, 200 test classes), tagged exactly as providers are
(`results_tag/tagger.json`). Precision 0.289 at 5 tags per text, falling to 0.135 at 30 (chance 0.011); recall
0.076 to 0.199. Retrieving roles from the *predicted* tags (tuned on dev: 5 tags, b 0.75): role@1 **0.130**,
role@5 0.350, role@10 0.512, track@1 0.415, track@3 0.640. Tagging noise removes most of Part A's advantage.
Role descriptions are cleaner than gig and provider text, so real-data tagging is probably worse, and the tagger
shares the dense channel's embedding model, which caps how much the channel can add beyond dense.

## 4. Part B: real data (`data_sat`, 1,023 gigs, 2,165 providers)

Relevant means grade >= 2. 542 gigs have at least one relevant provider; dev and test are 271 gigs each. The
existing two channels and their RRF are recomputed from cached embeddings and agree with the committed
`candidates_top50.csv` (mean top-50 overlap 0.9957, identical for 78.9% of gigs, worst 0.96; I did not investigate the gap,
which is probably tie-order and embedding numerics). `results_tag/sat.json`.

**Tuning is degenerate.** Dev tuning over (tags per gig, tags per provider, b) picked 30/30/0.75, but every b
scored identically (R@50 0.6536), because every provider carries exactly 30 tags, so the length normalisation is
a constant. Fewer tags were worse. The "b = 0.75" in the result is an arbitrary tie-break.

### 4a. Original labels (22,289 pairs, pooled from BM25 + dense + 3 random per gig)

Channels alone, test gigs:

| Channel | R@10 | R@50 | NDCG@10 | MRR |
|---|---|---|---|---|
| refined BM25 | 0.542 | 0.855 | 0.568 | 0.417 |
| dense (mxbai) | 0.665 | 0.900 | 0.641 | 0.453 |
| RRF(bm25, dense) | 0.765 | 0.999 | 0.713 | 0.523 |
| **tag** | 0.357 | 0.625 | 0.337 | 0.309 |

Control: with each gig's tags swapped for another gig's, tag R@50 falls to 0.064 (random: 0.023), so the channel
is using real signal. It is simply the weakest channel.

Marginal recall (relevant providers in a channel's top-K that neither other channel has in theirs):
K=10 **0.021** of positives (13 hits), K=20 0.006, **K=50 0.000**. Only 5 positives lie outside the existing
RRF top-50, and the tag channel finds none of them. This number is a lower bound: 93% of the tag channel's
tag-only top-10 pairs were never graded and count as irrelevant.

Fusion, RRF(bm25, dense) versus RRF(bm25, dense, tag), paired 95% CI over test gigs:

| | weight | R@10 | R@50 | NDCG@10 |
|---|---|---|---|---|
| standard | 1.0 | **-0.092** [-0.143, -0.044] | **-0.021** [-0.033, -0.011] | **-0.054** [-0.070, -0.038] |
| standard | 0.5 | -0.028 [-0.066, 0.009] | -0.003 [-0.008, -0.000] | -0.012 [-0.023, -0.000] |
| condensed (unjudged dropped) | 1.0 | -0.025 [-0.069, 0.020] | 0.000 | +0.007 [-0.008, 0.021] |
| condensed | 0.5 | +0.003 [-0.031, 0.037] | 0.000 | +0.011 [-0.000, 0.021] |

At full weight it clearly hurts. At half weight it is indistinguishable from nothing on the condensed lists.
In the pipeline's 3-way pool (`features.py --tag-channel`), 44 relevant pairs fall outside the top-50, against 5
for the two-channel pool.

### 4b. Extended labels (+5,492 graded pairs, see section 5)

The 5,500 pairs that only the tag channel's top 8 adds (minus 8 already graded) were graded with the reproduced
labeller. 19.4% came back grade >= 2 (1,067 pairs on 534 gigs; 2,657 zeros, 1,776 ones, 1,040 twos, 27 threes),
against 5.5% in the original pool. Dev and test split unchanged.

| Channel | R@10 | R@50 | NDCG@10 |
|---|---|---|---|
| refined BM25 | 0.397 | 0.684 | 0.524 |
| dense | 0.480 | 0.771 | 0.593 |
| RRF(bm25, dense) | 0.540 | 0.839 | 0.654 |
| tag | 0.387 | 0.688 | 0.411 |

- Marginal recall of the tag channel: K=10 **0.147** of positives (144 hits), K=20 0.157, K=50 0.112. BM25 and
  dense: 0.116 / 0.156 at K=10. Positives outside the existing RRF top-50: 454 (from 5), of which the tag channel
  has 326 in its top-50.
- Fusion at tag weight 1.0: R@50 **+0.071** [0.049, 0.092], R@100 +0.073, NDCG@10 **-0.025** [-0.040, -0.010],
  R@10 -0.012 (not significant). At weight 0.5: R@50 +0.058, NDCG@10 -0.002 (not significant). On condensed
  lists NDCG@10 is +0.016 [0.002, 0.030] (weight 1.0) and +0.012 [0.002, 0.022] (weight 0.5).

So on extended labels the channel does surface relevant providers the other two miss, and the fused list gains
pool recall but not top-of-list quality. **Why this cannot decide the question:**

1. The grader is more lenient (section 5), so some of the +1,065 positives are not relevant by the original
   standard.
2. Only the tag channel's extra pairs were graded. BM25 and dense were pooled to depth 8 originally, and their
   pairs below that are still ungraded, so the labels are now biased in the tag channel's favour, just as the
   original labels were biased against it. Neither label set is neutral.
3. Even so, top-of-list quality (NDCG@10) does not improve, which is what a Stage-2 ranker starts from.

## 5. The labeller and its calibration

The original grader prompt (`rubric_0_3_v2.md`, in the BT4103-Scrape-and-Tag repo) is not on this machine, so
`labeller.py` reconstructs it from the rubric in `label.md` and is versioned `rubric_0_3.v2-repro`. Scoring
matches the original: temperature 0, one token, grade = argmax of the log-probabilities over "0".."3",
`expected_grade` their mean. The endpoint serves a reasoning model, so requests must send
`reasoning_effort: "none"` or the answer never appears. Sentinel dates (`now`, `asap`) resolve to the fixed
2026-10-01 anchor from `label.md`, not `date.today()`, so a prompt does not depend on the day it is sent.

Calibration (`results_tag/calibration.json`): 300 already-graded pairs, 75 per original grade, seed 7, regraded.

| Measure | Value |
|---|---|
| exact agreement | 0.69 |
| within one grade | 0.973 |
| quadratic weighted kappa | 0.818 |
| Cohen's kappa on grade >= 2 | 0.733 |

Confusion matrix (rows original, columns new): 0: [53, 14, 8, 0]; 1: [4, 39, 32, 0]; 2: [0, 0, 75, 0];
3: [0, 0, 35, 40]. Every original grade >= 2 stays >= 2, but 40 of 150 original grade 0-1 pairs move up to 2,
and grade 2 is over-assigned (150 against 75). Mean `expected_grade` per assigned grade is 0.30 / 1.08 / 1.88 /
2.67, against the original 0.20 / 0.83 / 1.47 / 2.26. The gate proposed before running (kappa on grade >= 2 of at
least 0.6) is met, but the **reproduction is lenient**, and the extended-label positives should be read as an
upper bound. Among the new tag-only grade 2-3 pairs the mean `expected_grade` is closer to the original scale
(grade 2: 1.61, grade 3: 2.56).

The original label files are untouched; new grades live in `judgments_tag.jsonl`,
`llm_judgments_merged_tag.json`, `ground_truth_llm_tag.json`. `labeller.merge` never overwrites an existing label.

## 6. What did not work

- **LSA** over the tag-role matrix, alone (role@1 0.546) or fused with BM25 (0.609): both below BM25 (0.642).
- **Sum-of-levels scoring** (0.282): worse than counting tags.
- **Coverage and IDF-overlap** scoring (0.424 each): BM25's saturation and length terms matter.
- **The track-first funnel**: the track scorer is not accurate enough to pay for itself.
- **The tagger**: precision 0.135-0.289. Its track@3 (0.64) is far better than its role@1 (0.13), which fits
  errors landing near the right role, the costly kind in section 2, though I did not measure that directly.
- **Fusing at full weight**: hurts on the original labels, and hurts NDCG@10 even on the extended ones.
- **Dev tuning**: b and tags-per-text carry almost no information on this data (section 4).

## 7. Keep or drop

- **Default pipeline: drop** (leave the flag off). The channel alone is weaker than BM25 and dense on the original
  labels, and the fused list is worse or unchanged there. The marginal recall measured on extended labels is real
  signal but cannot be separated from grader leniency and pool bias, and top-of-list quality does not improve.
- **Keep the code** behind `--tag-channel`: it is default-off, tested, and leaves baseline outputs
  byte-identical (verified by running the pre-change `features.py` and the flag-off `features.py` on the same
  inputs: identical md5).
- **What would change this:** (a) a tagger that does not share the dense model's embedding space, or
  predicted tags that keep hierarchy (track-level agreement is much better than role-level: track@3 0.64 from
  predicted tags); (b) a human-graded sample of the tag-only positives to replace the leniency assumption;
  (c) retraining the LTR ranker with `tag_score`/`tag_rank` features on the widened pool and measuring final
  NDCG@10, which is the number that matters and was not run. Extended-label R@50 gains only matter if Stage 2
  can turn them into better top-10s.

## 8. Notes and caveats

- Everything is seeded (seed 7) and reproducible: `eval_tag_channel.py roles|tagger|sat [--extra-grades]`.
  Tagging embeddings are cached under `pipeline/cache/` (git-ignored).
- The tagger's tags are predicted, not gold: the channel inherits every error. Gig and provider text is noisier
  than role descriptions.
- Corrections to the initial brief: 9 track names are shared by 22 category IDs (234 distinct names over 247
  IDs); the 2/4/6 level mapping applies only to the 346 CCS rows, while the 43,612 TSC rows keep native levels 1-6;
  category 80's track name is truncated in the export (234 of 366 characters).
- `features.py` reads JSON with the platform default encoding, which fails on Windows (cp1252); run with
  `PYTHONUTF8=1`. It also keeps its own embedding cache (`cache_data_sat/`), separate from the one
  `build_judging_pools_sat.py` fills, so the first run re-embeds everything (hours on CPU) unless seeded.
- The committed baseline feature CSVs do not regenerate byte for byte today: `avail_immediacy` depends on
  `date.today()` (one day off since they were built) and dense cosines differ by up to 1.5e-3 between embedding
  runs. This is independent of the flag.
- The README's "23,873 pairs" is the old v1 figure; the current count is 22,289.
