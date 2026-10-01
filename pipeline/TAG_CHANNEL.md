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

**Follow-up (sections 11 and 12):** the channel's tagger had a hubness problem. With hubness-corrected tags (`--hubness center`) the same one-grader, fully judged comparison gives a significant RRF gain (NDCG@10 +0.021, P@10 +0.026). Section 12 re-checks that against the independent Claude audit (`pipeline/audit/RESULTS.md`): the gain keeps its sign but shrinks to about +0.010 to +0.015 P@10 once the reproduced grader's over-credit is calibrated away, and in the shipped Stage-2 ranker only NDCG@10 moves (+0.012 to +0.013). The default stays off.

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
  `pipeline/cache/` (git-ignored); regenerate the raw tag candidate CSV with `features.py --data-dir data_sat
  --tag-channel` (it is not committed). The two CSVs the Stage-2 comparison of section 12 needs (`candidates_top50_regen.csv`,
  `candidates_top50_tag_hc.csv`) are committed, because a rebuild on another day gives different values.
- The baseline and tag candidate files must come from the same day and cache: `avail_immediacy` uses
  `date.today()`, and dense cosines differ by up to 1.5e-3 between embedding runs. The downstream and same-grader
  results use a baseline regenerated alongside the tag file (`--baseline-csv`), not the committed CSV.
- `features.py` reads JSON with the platform default encoding, which fails on Windows (cp1252); run with
  `PYTHONUTF8=1`. It also keeps its own embedding cache (`cache_data_sat/`), separate from the one
  `build_judging_pools_sat.py` fills, so the first run re-embeds everything (hours on CPU) unless seeded. [^hw]
- Corrections to the initial brief: 9 track names are shared by 22 category IDs (234 distinct names over 247 IDs);
  the 2/4/6 level mapping applies only to the 346 CCS rows, while the 43,612 TSC rows keep native levels 1-6;
  category 80's track name is truncated in the export (234 of 366 characters).
- The README's "23,873 pairs" is the old v1 figure; the current count is 22,289.

## 11. Follow-up: hubness-corrected tags (`tag_corpus.py --hubness center`)

**Why the channel was weak: tag hubness.** The tagger picks the 30 tags with the highest raw cosine, so generic tag
titles near the centre of a domain win for almost every text. Measured on `data_sat`: the correlation between a
tag's mean cosine and how often it is picked is 0.57 (providers) and 0.60 (gigs); 50 tags fill 35.5% of all
provider slots; "Personal Finance Advisory" is on 813 of 2,165 providers (auditors, valuers and credit specialists
alike) and 547 provider-side tags are never picked. Raw cosine also spans only about 0.10 from rank 1 to rank 30
(0.68 to 0.59), and the channel keeps neither that gap nor the tag's strength. The tag-channel false positives
that dense does not share are mostly same-domain, different-task pairs (a "map the record-to-report process" gig
matched with a valuation specialist on "Financial Reporting" tags), not cross-industry ones: the share of false
positives in a clearly different industry is 0.57 for tag, 0.53 for dense, 0.62 for BM25. A hard industry gate
would cost good providers: relevance is 0.446 for the same industry, 0.326 for a different one, 0.426 for
cross-industry.

**Fix.** Subtract each tag's mean cosine over the corpus before taking the top tags (gig and provider sides have
their own means, since the two directions use different embedding spaces). This is written to new
`tags_*_hc.json` files; the raw files and every earlier result are untouched. Provider top-50 slot share falls to
14.7% (max tag frequency 813 to 282, tags used 1,541 to 2,029); gig side 16.4% to 10.1%. Two scorers are
available: BM25 over the corrected top-30 (`variant="hc"`, compared as `tag_hc`) and cosine of the weighted tags
`relu(score - 0.05)` over the top-100 (`scorer="wcos"`, `tag_hcw`). Dividing by the tag's sd as well (z-score, `--hubness z`)
was tried and works about as well, but adds off-topic picks on the gig side (top-10 tags whose raw cosine rank is
above 200: 7.4% for z against 1.3% centred; provider side 2.7% against 2.5%), so centring is the default.

**Result, one grader, fully judged top-10, 271 test gigs** (`eval_tag_variants.py eval --split test`,
`results_tag/variants_test.json`; 8,938 pairs graded by `rubric_0_3.v2-repro`, 1,585 of them new; paired 95% CI over
gigs, `*` = excludes 0):

| List | P@5 | P@10 | NDCG@5 | NDCG@10 | MRR@10 |
|---|---|---|---|---|---|
| refined BM25 | 0.448 | 0.389 | 0.624 | 0.613 | 0.712 |
| dense | 0.527 | 0.479 | 0.702 | 0.712 | 0.765 |
| tag (current) | 0.416 | 0.379 | 0.595 | 0.605 | 0.657 |
| tag_hc (centred, BM25) | 0.438 | 0.393 | 0.627 | 0.633 | 0.657 |
| tag_hcw (centred, weighted cosine) | 0.456 | 0.408 | 0.651 | 0.657 | 0.681 |
| RRF(bm25, dense) | 0.540 | 0.481 | 0.723 | 0.724 | 0.787 |
| RRF + tag (current) | 0.540 | 0.491 | 0.724 | 0.733 | 0.771 |
| **RRF + tag_hc** | 0.551 | 0.506 | 0.733 | 0.745 | 0.797 |
| **RRF + tag_hcw** | 0.555 | 0.503 | 0.734 | 0.744 | 0.781 |

- RRF + tag_hc minus RRF(bm25, dense): NDCG@10 **+0.021\*** [+0.009, +0.034], P@10 **+0.026\*** [+0.010, +0.040],
  MRR@10 +0.010 (CI includes 0). RRF + tag_hcw: NDCG@10 **+0.020\*** [+0.007, +0.033], P@10 **+0.022\***.
- Against the current channel in the same RRF: tag_hc NDCG@10 +0.012\* [+0.001, +0.022], P@10 +0.015\*. The current
  channel's own gain here is +0.009 (CI includes 0), the same as section 5.
- Alone: tag_hcw minus tag NDCG@10 +0.052\* [+0.031, +0.074]; it is still 0.055 below dense (\*).
- Absolute NDCG and R are normalised by the graded pool, which is the union of every compared top-10, so they shift
  slightly when lists are added (dense NDCG@10 is 0.717 in section 5 and 0.712 here, with identical lists and
  grades). Only paired differences within one run mean anything.

**What this does and does not establish.**
- The Stage-1 gate is met. Stage 2 (the shipped ranker with `features.py --tag-channel --tag-variant hc`) was run
  afterwards and is in section 12; the flag stays off by default.
- **Selection on the test gigs.** About 20 tag scorers (centring vs z-score, truncation, tau, soft kernels) were
  compared on a fixed pool built from the test gigs' top-10s before the two finalists were graded, and "hc"
  versus "z" was decided there. Only the finalists were graded prospectively, and the realised gain (+0.021) is
  below that proxy's (+0.026), but the test split is not untouched. The dev gigs (271) were not graded (6,498 new
  pairs, about 1.8 h of grader time); `eval_tag_variants.py pool --split dev` builds that pool if a clean
  confirmation is wanted.
- Every grade still comes from the lenient reproduced grader (section 6); section 12 tests how much that matters.

**Tried on the fixed pool and not built:** soft tag-tag kernels (title-embedding and role-co-occurrence
soft-cosine: no better than plain centring), role-level re-scoring and track-mass scoring (AUC 0.57, worse than tag
level, consistent with the track-first funnel result in section 3), softmax-weighted tags, and a per-text tag count
set by a score threshold (worse). The tuning objective in section 4 (channel-alone R@50 on BM25 + dense pooled
labels) rewards agreeing with those two channels, which is why every `b` tied; `run_sat` still uses it, to keep
the published numbers reproducible, and the variants are compared with `eval_tag_variants.py` instead.

**Not done from the plan:** per-gig overlap and track-mass features for the ranker; `features.py --tag-variant`
only adds the variants and an empty (NaN) `tag_score`/`tag_rank` for providers the channel did not return, in place of the 0 sentinel
(raw output is unchanged).

Reproduce: `tag_corpus.py --data-dir data_sat --hubness center`, then `eval_tag_variants.py pool --split test`,
`labeller.py pairs --pools judging_pools_variants_test.json --out judgments_variants.jsonl` (it resumes; only the
1,585 new pairs are sent), then `eval_tag_variants.py eval --split test`.

## 12. Re-validation against the Claude audit

`pipeline/audit/RESULTS.md` found that the reproduced grader (`rubric_0_3.v2-repro`) over-credits, and every number in
section 11 rests on it. This section asks how much of the hubness gain survives, using the same 271 test gigs, the
same fully judged top-10s and no new qwen grades except the 30 pairs missing from the Stage-2 pool. `eval_tag_variants.py
eval --relevance all` repeats every comparison under stricter definitions of relevant (grade >= 2 is kept only if the
grader's own P(>= 2) is at least 0.6 / 0.8, and/or the pair has no serious term mismatch; a demoted pair counts as
grade 1). The `repro` rows reproduce section 11 exactly. `*` = 95% CI over gigs excludes 0.

**Stage 1, RRF with the corrected channel minus RRF(bm25, dense):**

| Relevance | NDCG@10 | P@10 | MRR@10 |
|---|---|---|---|
| repro (section 11) | +0.0210* | +0.0255* | +0.0096 |
| P(>= 2) >= 0.6 | +0.0202* | +0.0214* | +0.0085 |
| P(>= 2) >= 0.8 | +0.0172* | +0.0018 | -0.0003 |
| no serious term mismatch | +0.0196* | +0.0232* | -0.0014 |
| both of the last two | +0.0172* | +0.0018 | -0.0003 |

The current (uncorrected) channel stays insignificant under every definition (NDCG@10 +0.007 to +0.010). The NDCG
gain is robust; the P@10 gain comes from positives the grader itself is unsure of.

**Stage 2 (shipped fused ranker, trained on the original grades), with the corrected channel minus without:**
NDCG@10 +0.0132 [+0.0017, +0.0250]* under `repro` and +0.0120 [+0.0004, +0.0235]* under the strictest definition
(0.784 to 0.797 absolute); P@10 (-0.006 to +0.010), R@10 (-0.022 to +0.011) and MRR@10 (-0.001 to +0.012) are not
significant under any definition.

**A grader-free check of the mechanism** (`eval_tag_channel.py tagger --hubness center`, 400 role descriptions with gold
tags, `results_tag/tagger_hubness_center.json`). Centring with the per-tag mean taken over the texts being tagged raises
gold-tag precision against raw cosine by +0.032 [+0.011, +0.055] at 5 tags, +0.022 at 10, +0.011 at 20 and +0.007 at 30
(all CIs exclude 0). With the means stored from the provider texts instead (a domain-mismatch control: role
descriptions are cleaner than provider profiles) there is no gain (-0.001 at 5, -0.007* at 30), so the per-tag mean must
come from the same kind of text it corrects. The raw rows reproduce section 3. CSLS (a local penalty, the mean of each
tag's k nearest texts) was tried on the same check as the one variant section 11 had not covered: +0.035 (k = 20) and
+0.043 (k = 50) at 5 tags against +0.032 for centring, intervals overlapping, so it was not built.

**Independent adjudication** (`claude_audit.py`, round 3, 195 pairs graded blind, same rubric; one rater):

- *Prompt or population?* Of 40 pairs the original grader put below 2 and the reproduction put at 2 or more, Claude
  sided with the reproduction on 22 (0.55 [0.40, 0.69]) and with the original on 18; a concordant control was
  confirmed 15/15. Across the whole repository, 5,004 pairs were graded under both prompts: the reproduction never
  lowers a grade and raises 34% of the original's negatives, and among the 1,624 with a serious term mismatch the
  original gave grade >= 2 to 0.4% and the reproduction to 9.7% (the share of pairs with a mismatch is 0.385 and 0.383 in
  the two populations, so this is the prompt, not the pairs).
- *Does the gain rest on over-credited pairs?* Of the 140 reproduction positives that enter or leave the top-10 when the
  corrected channel is added, Claude confirmed 44/70 of the entering and 44/70 of the leaving ones (0.63 each;
  difference 0.000 [-0.156, +0.156]). Confirmation rises with the grader's own confidence: 16/50 below 0.6, 30/48 from 0.6 to
  0.8 and 42/42 from 0.8 up, which is why the P(>= 2) >= 0.8 row is a conservative lower bound.
- *Calibrated effect.* Weighting each reproduction positive by Claude's confirmation rate for its confidence bin gives a
  P@10 gain of +0.0154 [+0.0048, +0.0262] (against +0.0255 uncalibrated). Using side-specific rates, the worst case
  for the channel, it is +0.0095 [-0.0114, +0.0290] (82% of bootstrap draws above 0).

**Reading.** The over-credit is common to both sides of the comparison, so it shrinks the gain (by roughly 40% in P@10, up to
60% in the worst case, and less in NDCG) without reversing it; the audit does not contradict a small positive effect, but at these sample sizes
(70 per side) it could not have detected a gap as small as the 0.08 that the Stage-1 P@10 gain implies, and the
side-specific worst case is not distinguishable from zero. Remaining limits: one rater with no human labels; the
test gigs were used to choose the two finalists (section 11); the dev gigs are still ungraded (6,498 pairs). The
verdict for the shipped pipeline stands: keep `--tag-channel` off by default. If the channel is ever adopted,
`TagChannel.rank_text(gig_vec, tag_vecs, tag_ids)` ranks a gig that is not in the tag files from its embedding (the
stored per-tag statistics supply the correction; identical to `rank` for the BM25 variants, and for the weighted scorer
the top-10 is identical and deeper ranks differ only by the 4-decimal rounding of the stored scores).

Reproduce: `eval_tag_variants.py eval --split test [--stage2] --relevance all`, `eval_tag_variants.py diff --split test`,
`claude_audit.py sample|show|score --round 3` and `claude_audit.py termcheck`.

[^hw]: This work was run on Windows 11 with Python 3.14, on a machine that also has an NVIDIA RTX 5070 (12 GB, driver
596.49); the pipeline itself runs on CPU and needs no GPU.
