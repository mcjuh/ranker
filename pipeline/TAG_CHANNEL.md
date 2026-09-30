# Tag-ID BM25 recall channel: what was built, what it measured, and the verdict

**Verdict: do not turn it on. Keep the code behind `--tag-channel` (default off).**

The one comparison with a single grader and fully judged lists (section 5) finds:

- **Alone,** the tag channel is clearly weaker than dense (NDCG@10 0.610 against 0.717) and level with refined
  BM25 (0.618).
- **In the RRF** (BM25 + dense + tag) it changes nothing significant (NDCG@10 +0.010, CI includes 0).
- **In the shipped Stage-2 ranker** it changes nothing significant either (NDCG@10 -0.003, CI includes 0) and
  MRR@10 is significantly worse (-0.034).

Every comparison that looked better for the channel mixed two graders (section 4c) and is explained by the
reproduced grader being more lenient than the original.

## 1. What was built

A third recall channel: BM25 over **taxonomy tag IDs** (the SkillsFuture taxonomy in the GreyGigz schema,
`taxonomy_greygigz/`: 247 tracks, 2,001 roles, 2,088 tags, 43,958 role-tag links).

| File | Role |
|---|---|
| `greygigz.py` | parses the SQL dumps directly; asserts the counts above |
| `retrieval_tagbm25.py` | `TagBM25`: `score = c_d * sum_t idf_t * q_t * B[d,t]`, `idf = ln(1 + (N-df+0.5)/(df+0.5))`, `c_d = (k1+1) / (1 + k1*(1-b+b*\|d\|/avgdl))`, binary tf, zero scores dropped. Baselines: `coverage`, `idf_overlap`, `sum_levels` |
| `tag_corpus.py` | the tagger: for each gig or provider text, the 30 taxonomy tags nearest by cosine (mxbai-embed-large-v1). `data_sat` has no tags of its own, so they are predicted |
| `tag_channel.py` | `TagChannel`: providers are the documents, a gig's predicted tags are the query. Defaults: 30 tags per text, k1 1.2, b 0.75 |
| `retrieval_rrf.py` | `rrf_fuse_n` added (N-way RRF); `rrf_fuse` untouched |
| `--tag-channel` | in `run_pipeline.py`, `features.py`, `build_judging_pools_sat.py`. Default off; a flagged run writes `*_tag` files and never overwrites a baseline file |
| `eval_tag_channel.py` | `roles` (taxonomy only), `tagger`, `sat` (channel alone and RRF on real data) |
| `eval_tag_downstream.py` | RRF, then the two LambdaMART rankers and their 0.7/0.3 fusion, with and without the channel |
| `eval_tag_samegrader.py` | regrades every compared top-10 with one grader (section 5) |
| `labeller.py` | zero-shot 0-3 grader for pairs only this channel surfaces (section 6) |

Only `numpy` and `scipy` are used by the channel itself. The downstream evaluation uses the project's existing
`xgboost` requirement (`requirements-rerank.txt`).

### For whoever owns the final fusion

`TagChannel(data_dir).rank(hire_id)` returns `[(provider_id, score), ...]`, best first, zero scores dropped: the
same shape as `BM25Retriever.rank` and `rank_from_raw`. Fuse it with `rrf_fuse_n([bm25, dense, tag], k=60,
weights=[1, 1, w])`. `features.py --tag-channel` adds `tag_score` and `tag_rank` (0 = the channel did not return
the provider) to the candidate and training CSVs. Nothing changes unless the flag is passed.

## 2. How it works, step by step

1. **Tag the texts.** Embed each gig as a query and each provider profile as a passage, and each taxonomy tag
   title in the opposite role; keep the 30 tags with the highest `cos(u, v) = u.v / (|u||v|)`.
2. **Index providers.** `B[d, t] = 1` if provider d carries predicted tag t. `df_t` counts providers carrying t.
3. **Score a gig.** With `q` its 0/1 tag vector: `score(d) = c_d * sum_t idf_t * q_t * B[d,t]`. Providers with
   score 0 are not returned.
4. **Fuse** (only when a script is run with `--tag-channel`): `RRF(d) = sum_c w_c / (60 + rank_c(d))` over the
   channels that return d.
5. **Stage 2** (existing code): LambdaMART over `bm25_score/rank, dense_cosine/rank, rrf_score/rank, budget_fit,
   seniority_fit, avail_immediacy` (+ `tag_score, tag_rank` with the channel), two recipes, fused 0.7 + 0.3.
6. **Metrics** (`evaluate.py`): relevant = score >= 40 = grade >= 2. `P@K = relevant in top K / K`,
   `R@K = relevant in top K / all relevant`, `DCG@K = sum_i gain_i / log2(i+1)` with gain the 0/33/67/100 score,
   `NDCG = DCG / IDCG`, `MRR = 1 / rank of the first relevant`. Unjudged pairs count as irrelevant unless a list is
   "condensed" (unjudged dropped). Bootstrap CIs (seed 7) over gigs; differences are paired.

## 3. On the taxonomy alone, and the tagger check

**Part A** (queries are 5 gold tags sampled from a role; test split; 95% CIs over tag-set-equivalence classes,
2,001 roles in 1,606 classes; `results_tag/roles.json`). This is an upper bound: it uses gold tags.

| Method | role@1 | role@5 | role@10 | track@1 | track@3 |
|---|---|---|---|---|---|
| **BM25** (k1 1.2, b 0.25 chosen on dev) | **0.642** | 0.956 | 0.992 | 0.766 | 0.968 |
| IDF-overlap / coverage | 0.424 | | | | |
| sum-of-levels | 0.282 | | | | |
| LSA alone / BM25 + LSA RRF | 0.546 / 0.609 | | | | |

- **b:** 0.0 gives 0.424, 0.25 gives 0.642, 0.5 gives 0.634, 0.75 gives 0.612, 1.0 gives 0.590. **k1** does not
  matter (0.5 to 3.0 moves role@1 by 0.001).
- **Generic-tag queries** (a role's 5 most widespread tags): role@1 0.343.
- **Wrong-tag queries:** a random wrong tag costs about nothing; a wrong tag borrowed from a sibling role in the
  same track costs 0.27 role@1 while track@1 is unchanged.
- **Track-first funnel:** loses. The track scorer is 0.762 accurate; break-even would be 0.90 (role@1) or 0.96 (role@5).

**Tagger check** (400 role descriptions, 200 dev / 200 test classes; `results_tag/tagger.json`): precision 0.289
at 5 tags falling to 0.135 at 30 (chance 0.011), recall 0.076 to 0.199. Roles retrieved from the *predicted* tags
(5 tags, b 0.75): role@1 **0.130**, role@5 0.350, role@10 0.512, track@1 0.415, track@3 0.640. Tagging noise
removes most of Part A's advantage. Role descriptions are cleaner than gig and provider text, and the tagger shares
the dense channel's embedding model, which caps what the channel can add beyond dense.

## 4. Real data (`data_sat`): channel alone and RRF

1,023 gigs, 2,165 providers; 542 gigs have a grade >= 2 provider; dev and test are 271 gigs each. The existing
channels are recomputed from cached embeddings and agree with the committed `candidates_top50.csv` (mean top-50
overlap 0.9957, identical for 78.9% of gigs, worst 0.96; I did not investigate the gap, which is probably
tie-order and embedding numerics). Dev tuning (tags per gig, tags per provider, b) selected **30 / 30 / 0.75**,
but every b scored identically (R@50 0.6536), because every provider carries exactly 30 tags, so the length
normalisation is a constant; "b = 0.75" is an arbitrary tie-break. A control that shuffles gig tags across gigs
falls to R@50 0.064 (random: 0.023), so the channel uses real signal. `results_tag/sat.json`.

### 4a. Original labels (22,289 pairs, pooled from BM25 + dense + 3 random per gig), 271 test gigs

| Channel | P@5 | P@10 | P@20 | R@10 | R@20 | R@50 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| refined BM25 | 0.164 | 0.121 | 0.080 | 0.542 | 0.690 | 0.855 | 0.543 | 0.568 | 0.580 | 0.417 |
| dense | 0.199 | 0.149 | 0.090 | 0.665 | 0.802 | 0.900 | 0.598 | 0.641 | 0.649 | 0.453 |
| RRF(bm25, dense) | 0.222 | 0.170 | 0.107 | 0.765 | 0.943 | 0.999 | 0.646 | 0.713 | 0.749 | 0.523 |
| **tag** | 0.108 | 0.074 | 0.050 | 0.357 | 0.455 | 0.625 | 0.332 | 0.337 | 0.368 | 0.309 |

These are **lower bounds for the tag channel**: only 37.6% of its top-10 was ever graded, against 88% for BM25
and dense, and unjudged pairs count as irrelevant. Marginal recall (relevant providers in a channel's top-K that
neither other channel has in theirs): tag 0.021 of positives at K=10, 0.006 at K=20, 0.000 at K=50.

### 4b. Extended labels (+12,579 pairs graded), 271 test gigs

Every pair outside the original pool that BM25, dense or the 30-tag channel ranks in its top 10 was graded by the
reproduced labeller (section 6), so **the top-10 of every channel is fully judged** and metrics at K <= 10 are
exact for that label set (top-20 is 65-78% judged and top-50 35-46%, so R@20 and R@50 remain lower bounds).

| Channel | P@5 | P@10 | P@20 | R@10 | R@20 | R@50 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|---|---|---|---|---|
| refined BM25 | 0.164 | 0.142 | 0.094 | 0.339 | 0.424 | 0.575 | 0.478 | 0.510 | 0.484 | 0.419 |
| dense | 0.199 | 0.190 | 0.128 | 0.428 | 0.561 | 0.721 | 0.527 | 0.588 | 0.567 | 0.457 |
| RRF(bm25, dense) | 0.222 | 0.170 | 0.131 | 0.409 | 0.580 | 0.777 | 0.567 | 0.603 | 0.618 | 0.523 |
| **tag** | 0.262 | 0.245 | 0.153 | 0.482 | 0.582 | 0.722 | 0.519 | 0.578 | 0.548 | 0.496 |

Marginal recall of the tag channel on these labels: 0.317 of positives at K=10 (on 69.7% of gigs), 0.286 at K=20,
0.178 at K=50 (BM25: 0.116 / 0.104 / 0.064; dense: 0.165 / 0.127 / 0.071). Positives outside the existing RRF
top-50: 912 (from 5), of which the tag channel has 781 in its top-50.

**Read this table with 4c.** It looks favourable to the tag channel, but its top-10 was graded mostly by the
reproduced labeller (62% of it was ungraded before), while BM25's and dense's were graded 88% by the original.
Scores from the two graders are not on one scale.

### 4c. Why 4b is inflated: the two graders differ

Calibration (section 6) shows the reproduced grader promotes about 27% of original grade 0-1 pairs to grade 2. The
tag channel's top-10 contains many more newly graded pairs than the other channels', so the lenient scale helps
it most. Section 5 removes the difference by regrading everything with one grader, and the tag channel then drops
back to the level of BM25.

## 5. One grader, fully judged lists: the comparison to trust

`eval_tag_samegrader.py`: on the 271 test gigs, every pair in the top-10 of BM25, dense, the tag channel, both
RRFs and both shipped Stage-2 rankers (7,345 pairs) was graded by the reproduced labeller, reusing grades it had
already produced for 2,420 of them. Every compared list is fully judged, by one grader. The Stage-2 rankers train
on the original grades only, for both systems. Recall's denominator is the relevant pairs among those graded (the
union of the compared top-10s): it ranks lists fairly but is not a recall over all providers, and the absolute
precision values are high because the graded set is enriched with good candidates. `results_tag/samegrader.json`.

| List | P@5 | P@10 | R@5 | R@10 | NDCG@5 | NDCG@10 | MRR@10 |
|---|---|---|---|---|---|---|---|
| refined BM25 | 0.448 | 0.389 | 0.255 | 0.415 | 0.623 | 0.618 | 0.712 |
| dense | 0.527 | 0.479 | 0.291 | 0.493 | 0.701 | 0.717 | 0.765 |
| **tag** | 0.416 | 0.379 | 0.225 | 0.398 | 0.594 | 0.610 | 0.657 |
| RRF(bm25, dense) | 0.540 | 0.481 | 0.305 | 0.509 | 0.722 | 0.729 | 0.787 |
| RRF(bm25, dense, tag) | 0.540 | 0.491 | 0.302 | 0.521 | 0.723 | 0.739 | 0.771 |
| shipped ranker, without tag | 0.638 | 0.564 | 0.362 | 0.601 | 0.791 | 0.794 | 0.892 |
| shipped ranker, with tag | 0.630 | 0.558 | 0.362 | 0.602 | 0.785 | 0.790 | 0.859 |

Paired differences over the 271 test gigs (95% CI over gigs):

- **tag minus dense:** NDCG@10 -0.107, P@10 -0.099, R@10 -0.095, MRR@10 -0.108; all CIs exclude 0.
- **tag minus BM25:** NDCG@10 -0.008, P@10 -0.009, R@10 -0.018; CIs include 0 (MRR@10 -0.055 and R@5 -0.030 exclude it).
- **RRF with tag minus RRF without:** NDCG@10 +0.010, P@10 +0.010, R@10 +0.011, MRR@10 -0.017; all CIs include 0.
- **Shipped ranker with tag minus without:** NDCG@10 -0.003, P@10 -0.007, R@10 +0.001; CIs include 0.
  MRR@10 **-0.034** excludes 0.

## 6. The labeller and its calibration

The original grader prompt (`rubric_0_3_v2.md`, in the BT4103-Scrape-and-Tag repo) is not on this machine, so
`labeller.py` reconstructs it from the rubric in `label.md` and is versioned `rubric_0_3.v2-repro`. Scoring
matches the original: temperature 0, one token, `grade = argmax_d p(d)` with
`p(d) = exp(logprob_d) / sum over d' in 0..3 of exp(logprob_d')`, and `expected_grade = sum_d d * p(d)`. The
endpoint serves a reasoning model, so requests send `reasoning_effort: "none"` or no answer appears. Sentinel dates
(`now`, `asap`) resolve to the fixed 2026-10-01 anchor from `label.md`, not `date.today()`.

Calibration (`results_tag/calibration.json`): 300 already-graded pairs, 75 per original grade, seed 7, regraded.

| Measure | Value |
|---|---|
| exact agreement | 0.69 |
| within one grade | 0.973 |
| quadratic weighted kappa | 0.818 |
| Cohen's kappa on grade >= 2 | 0.733 |

Confusion matrix (rows original, columns new): 0: [53, 14, 8, 0]; 1: [4, 39, 32, 0]; 2: [0, 0, 75, 0];
3: [0, 0, 35, 40]. Every original grade >= 2 stays >= 2, but 40 of 150 original grade 0-1 pairs move up to 2, and
grade 2 is over-assigned (150 against 75). Mean `expected_grade` per assigned grade is 0.30 / 1.08 / 1.88 / 2.67,
against the original 0.20 / 0.83 / 1.47 / 2.26. The kappa gate proposed before running (>= 0.6 on grade >= 2) is
met, but **the reproduction is lenient**, which is why section 5 regrades everything with it.

The original label files are untouched. New grades: `judgments_tag.jsonl` (tag-related pairs, 12,616),
`judgments_regrade.jsonl` (the section 5 pool), merged into `llm_judgments_merged_tag.json` and
`ground_truth_llm_tag.json`. `labeller.merge` never overwrites an existing label.

## 7. Downstream, all 1,023 gigs (RRF, then LambdaMART, then the 0.7/0.3 fusion)

`eval_tag_downstream.py`: the shipped recipe (`linz`: linear gain with per-query z-scored scores; `noce`:
exponential gain, raw; fused 0.7 + 0.3), 5-fold GroupKFold by gig, seed 7, with and without `tag_score/tag_rank`.
The baseline reproduces the README (RRF 0.671, linz 0.685, fused 0.684 NDCG@10). P and NDCG average over all gigs
as in the README; R over the 807 (extended) or 542 (original) gigs with a relevant provider. Both systems train on
the original grades. `results_tag/downstream.json`.

**Shipped (fused) ranker, tag system minus baseline** (`*` = 95% CI excludes 0):

| Labels, lists | NDCG@5 | NDCG@10 | P@5 | P@10 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|
| Original, standard | -0.014* | **-0.029*** | -0.009* | -0.009* | -0.035* | **-0.065*** | -0.008 |
| Original, condensed | +0.013* | +0.016* | -0.005* | -0.003* | -0.020 | -0.020 | -0.002 |
| Extended, standard | +0.014* | +0.023* | +0.011* | +0.021* | +0.018* | +0.081* | +0.016* |
| Extended, condensed | +0.019* | +0.032* | +0.012* | +0.025* | +0.020* | +0.098* | +0.017* |

The rows disagree, and none of them is trustworthy on its own: the original labels count everything the tag
channel surfaces as irrelevant (biased against it), the extended labels mix two graders (biased toward it), and
"condensed" drops unjudged pairs, which favours whichever system ranks fewer judged items high. The table in
section 5 is the controlled version: same grader, fully judged lists, and it shows no significant NDCG@10, P@10 or
R@10 effect from the channel in either the RRF or the shipped ranker.

**A trap in the extended labels.** Training the rankers on the extended grades collapses the baseline (fused
NDCG@10 0.22 against 0.56): which baseline candidates received an extra grade depends on the tag channel, which the
baseline's features cannot see, so its LambdaMART learns that deep-ranked graded candidates are positive and then
ranks ungraded ones high. The evaluation therefore trains both systems on the original grades by default
(`--train-labels original`).

In the 3-way pool, 35 relevant pairs fall outside the top-50 (original labels), against 5 for the two-channel pool.

## 8. What did not work

- **LSA** over the tag-role matrix, alone (role@1 0.546) or fused with BM25 (0.609): both below BM25 (0.642).
- **Sum-of-levels scoring** (0.282), **coverage** and **IDF-overlap** (0.424 each): BM25's saturation and length
  terms matter.
- **The track-first funnel:** the track scorer is not accurate enough to pay for itself.
- **The tagger:** precision 0.135-0.289. Its track@3 (0.64) is far better than its role@1 (0.13), which fits
  errors landing near the right role, the costly kind in section 3, though I did not measure that directly.
- **Tuning on `data_sat`:** b and tags-per-text carry almost no information there (section 4).
- **The channel in fusion:** no significant gain in the RRF or the shipped ranker under a single grader.

## 9. Keep or drop

- **Default pipeline: drop** (leave the flag off). Alone it is weaker than dense and level with BM25; fused, it adds
  nothing significant to NDCG@10, P@10 or R@10 and worsens MRR@10 in the shipped ranker.
- **Keep the code** behind `--tag-channel`: default-off, tested, and flag-off output is byte-identical to the
  pre-change `features.py` (same inputs, identical md5).
- **What would change this:** a tagger that does not share the dense model's embedding space, or predicted tags
  that keep hierarchy (track-level agreement is much better than role-level); a human-graded sample to replace the
  reproduced grader, since every label beyond the original 22,289 pairs rests on it; or a use of the channel that
  needs its different errors more than its accuracy (for example a candidate-pool widener, where the extended
  label sets do show relevant providers the other channels miss, but whose value depends on the grader).

## 10. Notes and caveats

- **Corrected during the work:** the first 5,500 graded tag pairs came from the flagged pool builder, whose
  `TagChannel` defaulted to 10 tags per text, while the evaluation uses the dev-tuned 30. The default is now 30
  (`tag_channel.py`), the pools were rebuilt, and the top-10 of every channel was then graded, so the final numbers
  use the 30-tag channel throughout. The first 5,500 grades remain valid and are in `judgments_tag.jsonl`.
- **Sample and scope:** section 5 covers the 271 test gigs at K <= 10 only; section 7 covers all 1,023 gigs but its
  labels are incomplete beyond the original pool (see the judged share printed by the script).
- Everything is seeded (seed 7) and reproducible: `eval_tag_channel.py roles|tagger|sat [--extra-grades]`,
  `eval_tag_downstream.py --extra-grades`, `eval_tag_samegrader.py pool|eval`. Tagging embeddings are cached under
  `pipeline/cache/` (git-ignored); regenerate the tag candidate CSV with `features.py --data-dir data_sat
  --tag-channel` (it is not committed).
- The baseline and tag candidate files must come from the same day and cache: `avail_immediacy` uses
  `date.today()`, and dense cosines differ by up to 1.5e-3 between embedding runs. The downstream and same-grader
  results use a baseline regenerated alongside the tag file (`--baseline-csv`), not the committed CSV.
- `features.py` reads JSON with the platform default encoding, which fails on Windows (cp1252); run with
  `PYTHONUTF8=1`. It also keeps its own embedding cache (`cache_data_sat/`), separate from the one
  `build_judging_pools_sat.py` fills, so the first run re-embeds everything (hours on CPU) unless seeded.
- Corrections to the initial brief: 9 track names are shared by 22 category IDs (234 distinct names over 247 IDs);
  the 2/4/6 level mapping applies only to the 346 CCS rows, while the 43,612 TSC rows keep native levels 1-6;
  category 80's track name is truncated in the export (234 of 366 characters).
- The README's "23,873 pairs" is the old v1 figure; the current count is 22,289.
