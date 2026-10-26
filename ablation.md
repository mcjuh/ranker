# Ablation — Stage-1 retrieval and Stage-2 re-ranking

**Corpus:** `pipeline/data/data_set_v5` — 1,106 gig queries, 3,024 provider showcases,
**21,696 LLM-graded pairs**, graded 0–3 by `qwen3.8:27b` on prompt `rubric_01.v5`
(0: 28.1% · 1: 10.5% · 2: 44.6% · 3: 16.9%). Relevance for P/R/MRR is **grade ≥ 2**,
so **61.5%** of judged pairs are relevant, median **13** relevant per query.

> **Update — §5 supersedes the shipped configuration and finding 5.** Re-grading the pairs
> the tag rankers surfaced showed the "+ taxonomy features" regression in §1–§2 was an
> artefact of ungraded candidates being scored as grade 0. With them graded, LambdaMART +
> SkillsFuture tag features beats the 9-feature model by **+0.0188 NDCG@10
> [+0.0145, +0.0231]** and is now shipped. §1–§4 are kept as they were measured: on the
> original 21,696 labels, which the committed label files no longer equal (they now include
> the 2,374 top-up grades, so re-running `final_ablation.py` scores against the larger set).

Reproduce with:

```bash
python pipeline/run_pipeline.py   --data-dir data/data_set_v5
python pipeline/features.py       --data-dir data/data_set_v5
python pipeline/rerank_ltr.py     --data-dir data/data_set_v5 \
                                  --linear-gain --normalize-scores zscore --tag v5_lin
python pipeline/final_ablation.py
```

All metrics are **out-of-fold** (`GroupKFold` by query — no query is scored by a model
that saw it). Significance is a **paired bootstrap over queries** (10,000 resamples);
the noise floor at n=1,106 is ≈ **0.007**.

---

## 1. Main results

| system | P@1 | P@5 | P@10 | R@10 | NDCG@10 | MRR |
|---|---|---|---|---|---|---|
| Stage 1 — BM25 (refined) | 0.9159 | 0.7937 | 0.6683 | 0.5693 | 0.7717 | 0.9466 |
| Stage 1 — Dense (mxbai-embed-large, 1024-d) | 0.9340 | 0.8420 | 0.7155 | 0.6157 | 0.8160 | 0.9597 |
| Stage 1 — RRF k=10 | 0.9439 | 0.8727 | 0.7980 | 0.6855 | 0.8812 | 0.9682 |
| Stage 1 — RRF k=60 | 0.9485 | 0.8745 | **0.8038** | 0.6887 | 0.8861 | 0.9698 |
| Stage 1 — RRF k=100 | 0.9494 | 0.8727 | 0.8025 | 0.6870 | 0.8848 | 0.9703 |
| **Stage 2 — LambdaMART (9 feat)** | **0.9548** | 0.8756 | 0.8020 | **0.6896** | **0.8867** | 0.9724 |
| Stage 2 — + industry features | 0.9575 | 0.8812 | 0.7826 | 0.6751 | 0.8729 | 0.9740 |
| Stage 2 — + taxonomy features (15 feat) | 0.9584 | 0.8807 | 0.7451 | 0.6459 | 0.8473 | 0.9746 |
| Stage 1 + industry recall (w=1.0) | 0.8626 | 0.7986 | 0.7149 | 0.6184 | 0.7952 | 0.9206 |
| Stage 1 + taxonomy recall (w=1.0) | 0.8969 | 0.7745 | 0.6611 | 0.5781 | 0.7526 | 0.9387 |
| Stage 1 + taxonomy recall (w=0.3) | 0.9584 | **0.8859** | 0.7883 | 0.6794 | 0.8756 | **0.9749** |
| *ORACLE (ceiling)* | *0.9991* | *0.9856* | *0.9365* | *0.8081* | *1.0000* | *0.9991* |

**Shipped configuration at the time: row 6** (superseded by §5) — refined BM25 ⊕ dense → RRF(k=60) → LambdaMART,
9 features, `rank:ndcg` with linear gain and per-query z-score normalisation.
It is the best NDCG@10 and every alternative is either a significant regression or a
non-significant trade.

---

## 2. Significance vs the shipped Stage-2 baseline

| variant | metric | Δ | 95% CI | verdict |
|---|---|---|---|---|
| + industry features | NDCG@10 | −0.0138 | [−0.0173, −0.0102] | **worse** |
| | P@5 | +0.0056 | [+0.0005, +0.0107] | better |
| | P@10 | −0.0193 | [−0.0245, −0.0143] | **worse** |
| | R@10 | −0.0145 | [−0.0197, −0.0094] | **worse** |
| | MRR | +0.0016 | [−0.0026, +0.0059] | crosses 0 |
| + taxonomy features (15 feat) | NDCG@10 | −0.0394 | [−0.0443, −0.0345] | **worse** |
| | P@5 | +0.0051 | [−0.0014, +0.0118] | crosses 0 |
| | P@10 | −0.0569 | [−0.0637, −0.0503] | **worse** |
| | R@10 | −0.0436 | [−0.0500, −0.0371] | **worse** |
| | MRR | +0.0022 | [−0.0032, +0.0077] | crosses 0 |

Every recall-fusion row is significantly worse than the baseline on NDCG@10.

---

## 3. Findings

**1. Hybrid fusion is the largest single win.** BM25 0.7717 → dense 0.8160 →
RRF 0.8861 NDCG@10. Fusing the two ranked lists is worth **+0.070 NDCG@10** over the
better single retriever, and it improves every metric. `k=60` is the optimum; k=10 is
worse (0.8812) and k=100 is level (0.8848).

**2. Stage 2 does not earn its place on this corpus.** The LambdaMART beats RRF by
+0.0006 NDCG@10, +0.0011 P@5, +0.0026 MRR — and **every paired CI crosses zero**
against a 0.007 noise floor. Feature importance sits **80% on `rrf_rank` + `rrf_score`**,
i.e. the model largely re-expresses the retriever. This is a consequence of the corpus
shape: with ~13 relevant providers per query out of a 50-candidate pool, ~26% of the
pool is relevant, so almost any ordering scores well and there is very little for a
ranker to fix.

Reproduce that check from the artifact `results_data/data_set_v5/v5_lin.json`.

**3. The third (taxonomy) recall channel adds nothing, because there is nothing to add.**
`run_pipeline.py` calls `rrf_fuse(sparse, dense, k=k)` — the taxonomy channel is never
passed, although `retrieval_rrf.rrf_fuse` has always supported it. Wiring it in was
tested here and:

```
candidates the third channel adds: 0 per query in the pool, ~1–2 inside the top-10
```

BM25 and dense already rank **all 3,024 providers**, so the candidate pool is
exhaustive and a third channel can only reorder. Reordering loses: at w=1.0 the
taxonomy fusion costs **−0.13 NDCG@10**; at w=0.3 it costs −0.011.

**4. A coarse label makes a degenerate channel.** `data_set_v5` ships only an
`industry` field (36 values). Mapped onto the taxonomy category axis it produces a
**binary** feature — one distinct value in the whole file:

```
tax_category    distinct: ['1.000000']    444,104 rows
taxonomy_score  distinct: ['0.100000']    444,104 rows
largest tied group per query: median 660
```

~660 candidates tie at the same score, so RRF converts an arbitrary tie order into
ranks 1…660 and industry-matched noise floods the top-10. Cosine similarity would not
fix this — 2,772 providers share 36 vectors, so the ties survive.

**5. The full SkillsFuture tree was recovered and still did not help** — *overturned in §5:
the regression was an evaluation artefact.* v5 dropped the
taxonomy, but the `mcjuh/BT4103-Scrape-and-Tag` records carry the complete `sf_v1`
tree at 100% coverage and join to v5 by `source_file` with zero loss (1,106/1,106 gigs,
3,024/3,024 providers). `enrich_v5_with_tags.py` grafts it on, giving 33/37 categories,
103/134 specialisations, 108/143 search tags and **500/745 skills**. The features are
genuinely graded (means 0.11–0.78) and the model uses them at ~30% importance — yet
NDCG@10 falls **−0.0394** (significant). The signal is real but redundant: it trades
graded ordering for head membership (P@5 and MRR up, NDCG@10/P@10/R@10 down).

**6. Recall is capped by K, not by the system.** With a median of 13 relevant items per
query, a top-5 list can hold at most 5 of them, so R@5 is arithmetically capped at
≈ 0.385 — and the system sits at 84.6% of the *oracle* ceiling:

| K | observed | oracle | % of achievable |
|---|---|---|---|
| 5 | 0.3924 | 0.4638 | 84.6% |
| 10 | 0.6896 | 0.8081 | 85.3% |
| 20 | 0.9268 | 0.9998 | 92.7% |
| 50 | **0.9908** | 1.0000 | **99.1%** |

Report the curve, never R@5 alone.

**7. Three features are structurally dead.** `budget_fit`, `seniority_fit` and
`avail_immediacy` read `budget_lo/hi`, `rate_per_hour`, `seniority_needed`,
`seniority` and `availability` — none of which the v5 schema carries. They contribute
`0.0000` importance in every run. They are retained so the column contract is stable,
but they carry no information on this corpus.

---

## 4. Caveats (§1–§3)

- **Out-of-fold only.** The served model is trained on all queries
  (`models_data/data_set_v5/v5_lin_xgb.json`); the reported numbers come from OOF
  predictions (`results_data/data_set_v5/v5_lin.json`). Do not mix them.
- **The exported ranked lists are top-50**, so R@50 here is measured on the top-50
  only; deep recall is higher.
- **One query in 1,106 has no relevant item**, so `recall_at_k` returns `None` for it;
  all recall figures average over the 1,105 queries where recall is defined.
- **`data_set_v5_tax` shares v5's text byte-for-byte** — only `tags` was added — so its
  BM25/dense rankings are identical to v5's and are reused rather than recomputed.
- The recall-fusion rows were rebuilt from saved ranked lists, which is valid because
  RRF consumes **ranks only**. The 2-channel rebuild was asserted to reproduce the
  saved `rrf_k60.json` exactly before any 3-channel delta was reported.

---

## 5. Tag features, re-evaluated with the ungraded pairs graded (shipped)

**What went wrong in §1–§2.** The judging pool came from the text retrievers' top results (RRF, BM25,
dense) plus a few randoms per gig, so beyond RRF rank 10 most candidates were never
graded: 40% of ranks 11–20, 14% of 21–30, 5% of 31–50. `evaluate.py` scores an ungraded pair
as grade 0. A feature that promotes providers the text retrievers ranked low is therefore
penalised whether or not those providers are good — and that is exactly what tag features do.
The 15-feature taxonomy ranker's top-10 was only **87.4% graded** (the 9-feature model's:
98.7%). Dropping ungraded pairs from both lists instead flipped the sign (0.9079 vs 0.8923).

**Evidence the tags carry real signal** (top-50 pool, graded pairs only, Spearman with the grade
after removing log RRF rank within each gig): match level +0.245, specialisation coverage
+0.216, skill overlap +0.21, category +0.20, industry +0.19, "same category, different
specialisation" −0.03. The share graded ≥ 2 by match level (none / category / specialisation /
specialisation + skill) is monotone in every rank band, e.g. ranks 31–50: 27% / 37% / 52% / 74%.

**The top-up.** Every ungraded pair in the top-10 of any tag variant (out-of-fold, split 0) —
**2,383 pairs** — was graded with the original grader (`qwen3.8:27b`, `rubric_01.v5`,
temperature 0, via the Scrape-and-Tag repo's `labeller/grade_pairs.py`). A calibration re-grade
of 60 already-graded pairs (15 per grade) reproduced **60/60** exactly. 2,374 were graded
(9 failed: the model answered in prose or with a bare score, deterministically):
0: 427 · 1: 346 · 2: 1,420 · 3: 181 — **67% relevant**, against 61.5% for the original pool.
Stored in `data/data_set_v5/judgments_topup_tags.jsonl`; `merge_topup_grades.py` merges them.

**Features** (`tag_features.py`): `tag_level` (0 none · 1 category · 2 specialisation ·
3 specialisation + skill), `has_tags_gig`, `has_tags_prov`, `cat_any`, `spec_cov`, `skill_cov`,
`skill_idf_cov` (rarity-weighted), `skill_soft` (mean best cosine between skill names,
mxbai-embed-large). Category/specialisation are matched exactly (closed vocabularies of 37 /
144). Industry is excluded — a provider's industry is always one of its tag categories, and the
two agree on 89% of candidate pairs.

**Protocol** (`tag_ablation.py`): train on the original 21,696 labels, evaluate on all 24,070.
Out-of-fold over three fold splits (GroupKFold + two shuffled KFolds). Both systems' top-10s
are ≥ 99% graded.

| system (trained on original labels) | NDCG@10 | NDCG@5 | P@5 | P@10 | R@10 | MRR | top-10 graded |
|---|---|---|---|---|---|---|---|
| RRF k=60 | 0.8757 | 0.8817 | 0.8745 | 0.8039 | 0.6197 | 0.9698 | 99.9% |
| LambdaMART, 9 feat | 0.8768 | 0.8842 | 0.8751 | 0.8042 | 0.6223 | 0.9705 | 99.1% |
| **LambdaMART + 8 tag features** | **0.8927** | **0.9000** | **0.8966** | **0.8245** | **0.6409** | **0.9782** | 99.0% |

*(mean over the three splits)*

| tags vs 9-feat (split 0) | Δ | 95% CI | per split |
|---|---|---|---|
| NDCG@10 | **+0.0188** | [+0.0145, +0.0231] | +0.0188 / +0.0169 / +0.0119 |
| NDCG@5 | **+0.0162** | [+0.0115, +0.0208] | +0.0162 / +0.0149 / +0.0164 |
| P@5 | **+0.0217** | [+0.0150, +0.0286] | +0.0217 / +0.0193 / +0.0235 |
| P@10 | **+0.0251** | [+0.0190, +0.0315] | +0.0251 / +0.0208 / +0.0151 |
| R@10 | **+0.0215** | [+0.0153, +0.0278] | +0.0215 / +0.0200 / +0.0145 |
| MRR | +0.0055 | [−0.0001, +0.0115] | +0.0055 / +0.0090 / +0.0086 |

**Variants tried** (exploration harness, same protocol, split 0, NDCG@10 on all labels):
`tag_level` alone 0.8951 · exact-match features only 0.8965 · **exact + `skill_soft` 0.8969
(shipped)** · with monotone constraints 0.8949 · `tag_level` + `skill_soft` monotone 0.8947.
`tag_level` alone carries most of the gain; `skill_soft` adds a small, consistent P@5 lift
(+0.006 over exact-only); monotone constraints do not help.

**Shipped model**: the 17-feature ranker retrained on all 24,070 labels —
`rerank_ltr.py --data-dir data/data_set_v5_tax --linear-gain --normalize-scores zscore
--tag-features --tag v5tax_tags`, out-of-fold NDCG@10 **0.8982**, P@5 0.8991, MRR 0.9774
(`results_data/data_set_v5_tax/v5tax_tags.json`, model `models_data/data_set_v5_tax/v5tax_tags_xgb.json`).
Importance: `rrf_rank` 0.33, `rrf_score` 0.24, `tag_level` 0.07, then `bm25_score`,
`dense_cosine`, `skill_cov`, `spec_cov`, `skill_soft` (0.04–0.06 each); tag features 28% in total.

**Caveats.**
- The top-up was chosen from the tag rankers' split-0 top-10s, so it cannot favour the
  baseline. On the other two splits a few ungraded pairs remain in the tag ranker's top-10
  (99.0% graded on average), which, if anything, understates the tags there.
- **Worst case for the remaining ungraded slots** (split 0): 97 slots in the baseline's
  top-10s and 30 only in the tag ranker's are still ungraded. Scoring every baseline one as
  grade 3 and every tag-only one as grade 0 still leaves the tags ahead:
  **+0.0122 NDCG@10 [+0.0080, +0.0165]** (baseline 0.8800, tags 0.8922).
- Retraining the 9-feature model on the top-up as well is *not* a fair baseline: the top-up is
  mostly relevant providers at low RRF ranks, so that model learns to promote low-ranked
  candidates generally and its top-10 falls to 94% graded (NDCG@10 0.8583). `tag_ablation.py`
  prints it for completeness.
- All NDCG values in this section are on the 24,070-pair label set. The added grades raise the
  ideal DCG of the gigs where they found strong providers, so RRF reads 0.8757 here vs 0.8861
  in §1. Do not compare across the two label sets.
- The grader saw only the text, never the tags, so the tag features are not being scored by
  a judge that read them.

---

## 6. Tier-1 features + rank fusion (the new best)

Reproduce:

```bash
python pipeline/tier1_rank.py --data-dir data/data_set_v5_tax
```

Same protocol as §5 (out-of-fold, GroupKFold by query, `--linear-gain --normalize-scores zscore`,
evaluated on all 24,070 labels; paired bootstrap over queries, 10k resamples).

**Three feature families on top of the 17 shipped features.**

1. **Taxonomy-graph partial credit.** §5's `skill_cov` is exact string match and `skill_soft` is
   a name-embedding cosine; neither uses the `sf_v1` tree's edges. A gig skill the provider does
   not hold exactly can still be *reached*: a **sibling** skill under the same specialisation, or
   any provider skill under the same **category**. Adds `skill_sib_cov`, `skill_cat_cov`,
   `graph_dist_mean`, `graph_dist_min` (mean/min hop distance 0 exact → 1 sibling → 2 category →
   3 none), `spec_cat_cov`, `skill_jaccard`.
2. **Interaction crosses** the trees cannot cheaply build at this sample size: `tl_x_rank`
   (`tag_level` × normalised RRF rank), `speccov_x_tl`, `skillcov_x_tl`.
3. **Provider-prior quality.** Leave-one-query-out, α-smoothed (α=5) mean grade of a provider.
   **Fold-aware:** built inside each fold from that fold's training queries only, and a training
   row's own query is subtracted from its provider's totals, so no query ever sees its own label.

Then a **weighted-RRF fusion** of the rankers' OOF lists (the `fuse_rankers.py` recipe).

| System (OOF, 1,106 queries) | NDCG@10 | NDCG@5 | P@5 | P@10 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|
| RRF k=60 (Stage 1) | 0.8757 | 0.8817 | 0.8745 | 0.8039 | 0.3517 | 0.6197 | 0.9698 |
| Shipped: LambdaMART + tags (17 feat) | 0.8982 | 0.9018 | 0.8991 | 0.8304 | 0.3638 | 0.6449 | 0.9774 |
| + Tier-1 features (28 feat, alone) | 0.8913 | 0.9127 | 0.9085 | 0.8162 | 0.3692 | 0.6360 | 0.9844 |
| RRF-fuse base + Tier-1 (0.6/0.4) | 0.9039 | 0.9095 | 0.9069 | 0.8357 | 0.3678 | 0.6511 | 0.9811 |
| **RRF-fuse base + Tier-1 + cross-encoder (0.4/0.4/0.2)** | **0.9121** | **0.9243** | **0.9226** | **0.8429** | **0.3762** | **0.6587** | **0.9878** |

Paired bootstrap, fusion vs the shipped 17-feature ranker (10k resamples):

| Metric | Δ | 95% CI | Verdict |
|---|---|---|---|
| NDCG@10 | **+0.0139** | [+0.0109, +0.0170] | better |
| NDCG@5 | +0.0225 | [+0.0187, +0.0263] | better |
| P@5 | **+0.0235** | [+0.0179, +0.0293] | better |
| P@10 | +0.0125 | [+0.0080, +0.0169] | better |
| R@5 | +0.0123 | [+0.0087, +0.0161] | better |
| R@10 | +0.0138 | [+0.0096, +0.0180] | better |
| MRR | +0.0104 | [+0.0057, +0.0153] | better |

Every metric improves with a CI excluding 0. The +0.0139 NDCG@10 is ~2× the 0.007 noise floor.

**Findings.**

1. **The Tier-1 features alone do not beat the shipped model on NDCG@10** (0.8913 vs 0.8982, a
   significant −0.0068), but they move the *opposite* way on the head: P@5 +0.0094, MRR +0.0070,
   NDCG@5 +0.0109 (all significant). This is the same graded-order vs head-membership trade seen
   with `rank:map` in the earlier corpus, and it is what makes them worth fusing.
2. **The features genuinely fire**: top importances are `tl_x_rank` 0.060, `prov_prior` 0.043,
   `graph_dist_min` 0.037, `skillcov_x_tl` 0.025, `skill_cov` 0.022, `skill_soft` 0.022.
3. **Rank fusion turns a trade into a win.** Because the 17-feature model and the 28-feature model
   are strong on different metrics, weighted RRF beats *both* on *every* metric — the same
   mechanism as §4 of `RERANK_README.md`.
4. **The cross-encoder is a good ensemble input and a bad ranker.** `ms-marco-MiniLM-L6-v2`
   fine-tuned under per-query OOF CV scores NDCG@10 **0.7706** (zero-shot 0.6165), and folding
   `ce_score` into LambdaMART as a *feature* costs **−0.082 NDCG@10** (0.8781 → 0.7966) for
   +0.011 MRR. Yet adding its OOF list as the third fusion input is worth **+0.008 NDCG@10** and
   **+0.007 MRR** over the two-way fusion. Its errors are decorrelated from the GBDT's; its
   ordering is not good enough to stand alone. This is the resolution of the "cross-encoder line
   of work" that was dropped: it belongs in the ensemble, not in the ranker.

**Caveats.**

- **Fusion weights are selected on the OOF split** (0.4/0.4/0.2). That is a mild selection; the
  direction is robust — every weight in 0.30–0.50 beats the baseline.
- **The fusion's inputs are re-trained models, not the shipped artifact.** `tier1_base` is a fresh
  fit of the 17-feature configuration (0.8956 vs the artifact's 0.8982), so its inclusion is not
  circular. Adding the shipped `v5tax_tags.json` as a fourth input changes nothing
  (0.9121 → 0.9126, and P@5/MRR drop slightly), confirming redundancy.
- **The cross-encoder OOF scores** come from `rerank_crossencoder.py --mode finetune` (5-fold
  per-query CV, 3 epochs, positives = grade ≥ 3, negatives = grade ≤ 1; grade-2 excluded).
  A larger `Alibaba-NLP/gte-reranker-modernbert-base` run was started and cancelled after fold 1
  (0.7840 vs MiniLM's 0.7592 on the same fold) — it was tracking to ~0.79, still far below the
  ranker, so it was not carried further.
- All numbers are OOF; the fusion lists are committed at
  `results_data/data_set_v5_tax/tier1_fuse{,2,3}.json` and the component lists alongside them.
