# The Ranker and Its Third Recall Channel: A Primer

*For readers who know embeddings but have not yet worked inside a search pipeline or a retrieval experiment.*

**Scope and provenance.** This was written against branch `claude/bold-franklin-e7tk9p` at commit `19c7249` (2026-10-02), after reading the code in `pipeline/`, the repo's own write-ups (`README.md`, `RERANK_README.md`, `label.md`, `TAG_CHANNEL.md`, `TAG_CHANNEL_HOW_IT_WORKS.md`, `audit/*`), the result files in `results_tag/`, and the 55-commit history. Numbers are of two kinds:

- **(repo)** means quoted from the repo's docs or result files. I cross-checked a sample of them against the committed JSON and found no mismatches.
- **(recomputed)** means I re-derived it from committed data while writing this primer. The code for the important ones is shown, so you can run it yourself.

Where my reading of the code differs from a sentence in the repo's docs, I say so.

**How the document is organised.** It builds knowledge first and only then assembles the system.

| Part | Purpose |
|---|---|
| 0 | The whole system on one page, so every later concept has a place to land |
| I | Foundations: search pipelines, BM25, dense retrieval, rank fusion, metrics, learning to rank, experimental design and statistics, softmax and LLM judges, hubness |
| II | This repo's data and its first two channels, briefly |
| III | The third channel (predicted taxonomy tags), step by step |
| IV | How we know what we know: the sequence of experiments, from the commit history |
| V | Putting it together: one gig end to end, what a "strong" number looks like, limits, how to run things |

Each concept in Part I follows one pattern: **Idea**, **Formula**, **Intuition**, **In this repo**, **What a good value looks like**. If you already know a topic, jump to its *In this repo* line.

---

## Contents

- [0. The whole system on one page](#0-the-whole-system-on-one-page)
- **Part I: Foundations**
  - [1. Search as two stages](#1-search-as-two-stages)
  - [2. Matching text: lexical and dense](#2-matching-text-lexical-and-dense)
  - [3. Merging ranked lists: reciprocal rank fusion](#3-merging-ranked-lists-reciprocal-rank-fusion)
  - [4. Measuring a ranking](#4-measuring-a-ranking)
  - [5. Stage 2: learning to rank](#5-stage-2-learning-to-rank)
  - [6. Experimental design and statistics](#6-experimental-design-and-statistics)
  - [7. Softmax, and using an LLM as the judge](#7-softmax-and-using-an-llm-as-the-judge)
  - [8. Hubness](#8-hubness)
- **Part II: This repo's setting**
  - [9. The data and the labels](#9-the-data-and-the-labels)
  - [10. The first two channels and the shipped ranker](#10-the-first-two-channels-and-the-shipped-ranker)
- **Part III: The third channel**
  - [11. Why a third channel, and what error it should cover](#11-why-a-third-channel-and-what-error-it-should-cover)
  - [12. The taxonomy](#12-the-taxonomy)
  - [13. Step 1: the tagger](#13-step-1-the-tagger)
  - [14. Step 2: the hubness correction](#14-step-2-the-hubness-correction)
  - [15. Step 3: scoring in tag space](#15-step-3-scoring-in-tag-space)
  - [16. Step 4: fusion and the Stage-2 features](#16-step-4-fusion-and-the-stage-2-features)
  - [17. How it fails](#17-how-it-fails)
  - [18. Interfaces, flags, files](#18-interfaces-flags-files)
- **Part IV: How we know what we know**
  - [19. The history, from the commits](#19-the-history-from-the-commits)
  - [20. The evidence ladder](#20-the-evidence-ladder)
  - [21. The results, in order](#21-the-results-in-order)
  - [22. Re-deriving the headline yourself](#22-re-deriving-the-headline-yourself)
  - [23. Threats to validity](#23-threats-to-validity)
- **Part V: Putting it together**
  - [24. One gig, end to end](#24-one-gig-end-to-end)
  - [25. What a strong value looks like](#25-what-a-strong-value-looks-like)
  - [26. Open questions and next steps](#26-open-questions-and-next-steps)
  - [27. Running things, and a file map](#27-running-things-and-a-file-map)
- [Glossary](#glossary) · [Further reading](#further-reading)

---

## 0. The whole system on one page

The product matches **gigs** (a client's request for a piece of consulting work) to **providers** (a consultant's profile). For each gig it must return the providers worth showing, best first.

```
                          one gig (title + description + notes)
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          ▼                              ▼                              ▼
   CHANNEL 1                       CHANNEL 2                      CHANNEL 3   <- this primer
   refined BM25                    dense embeddings               predicted taxonomy tags
   matches WORDS                   matches MEANING                matches SKILL IDs
   (stemming, title x3,            (mxbai cosine,                 (tag BM25 or weighted cosine
    synonyms)                       1024 dims)                     over 30 predicted tags)
          │                              │                              │
   ranked list of all 2,165       ranked list of all 2,165       ranked list of the ~500
   providers                      providers                      providers with evidence
          └──────────────────────────────┼──────────────────────────────┘
                                         ▼
                      Reciprocal Rank Fusion, k = 60   ──┐
                                         ▼               │  STAGE 1: RECALL
                         top-50 candidates per gig     ──┘  "is the right provider in the pool?"
                                         ▼
              add business features: budget fit, seniority fit, availability
                                         ▼
              LambdaMART (gradient-boosted trees that learn to rank)      ──┐
              two recipes, "linz" and "noce", fused 0.7 / 0.3 by RRF        │  STAGE 2: ORDER
                                         ▼                                  │  "which first?"
                       ranked list + a 0-100 score per provider           ──┘
```

**The third channel in five sentences.**

1. Neither gigs nor providers carry skill tags, but an official skills taxonomy exists (2,088 tags such as *Vendor Management* or *Debt Restructuring*), so we **predict** each text's tags by finding the tag titles nearest to the text in embedding space.
2. Raw nearest-tag lists are dominated by generic "hub" tags that sit near the middle of whole domains, so each tag's average similarity is subtracted first (**centring**), and the 30 best tags per text are kept.
3. A provider becomes a document made of tag IDs, a gig becomes a query made of tag IDs, and providers are scored by **IDF-weighted overlap** (BM25 over IDs) or by weighted cosine.
4. Its ranked list is merged with the other two by RRF, and its score and rank can be fed to Stage 2 as two extra features.
5. Verdict so far: it is **off by default** (`--tag-channel`); the plain mxbai version gives small, partly audited gains, and the same recipe with a Qwen3 encoder gives larger ones (+0.058 NDCG@10 in the fused list, +0.043 in the shipped ranker, on 271 held-out gigs) whose size depends on one LLM grader.

**Numbers worth knowing before we start** (repo, with the arithmetic recomputed where noted):

| Fact | Value |
|---|---|
| Gigs × providers | 1,023 × 2,165 = 2,214,795 possible pairs |
| Pairs graded by an LLM | 22,289 (about 1% of all pairs: only a pool per gig was graded) |
| Grades | 50.8% grade 0, 43.7% grade 1, 2.5% grade 2, 3.0% grade 3 |
| "Relevant" means | grade ≥ 2, only **5.5%** of graded pairs |
| Gigs with at least one relevant provider ("answerable") | 542 of 1,023 (53.0%) |
| Mean relevant providers per answerable gig | 2.26 (recomputed) |
| Taxonomy | 247 tracks, 2,001 roles, 2,088 tags, 43,958 role-tag links |
| Candidates per gig passed to Stage 2 | 50 (51,150 rows in total) |

The scarcity of relevant pairs (5.5% of graded pairs, 2.26 per answerable gig) matters for everything below: with so few relevant providers per gig, many metrics have low *ceilings*, and "good" must be read against the ceiling, not against 1.0 (section 25).

---

# Part I: Foundations

## 1. Search as two stages

**Idea.** Scoring every item against every query with the best available model is too slow, and the best models are slow because they read both texts together. So search systems split the job. A cheap first stage looks at everything and returns a short candidate list. An expensive second stage reorders only that list.

**Recall** is the fraction of the relevant items that were found:

```
Recall@K = (number of relevant items in the top K) / (total number of relevant items)
```

It is the metric of the **first stage** of search engines, recommender candidate generators and retrieval-augmented generation (RAG) systems, because an item the first stage drops cannot be recovered by anything after it. A first-stage component is therefore called a **recall channel** or *candidate generator*: its job is to make sure the right items are *somewhere* in the pool, and ordering them is someone else's job.

| | Stage 1 (recall) | Stage 2 (ranking) |
|---|---|---|
| Question | Is the right provider anywhere in my short list? | In what order should the short list be shown? |
| Search space per gig | all 2,165 providers | the 50 candidates |
| Cost per pair | tiny (a dot product, a sparse sum) | can afford a learned model and extra features |
| Metrics | R@K of the pool, *marginal* recall of each channel | NDCG@10, P@5, MRR |
| Failure mode | a miss: nothing downstream can fix it | a misordering |
| In this repo | BM25, dense, tag, fused by RRF into a top-50 | LambdaMART (two recipes), fused |

**Why several channels.** A channel is a *view* of the data, and every view has blind spots. Using several views and merging them raises the chance that at least one of them sees the right provider. Whether that works depends entirely on whether their blind spots **differ**. Two channels that fail on the same gigs add nothing to each other; this is the central question for the third channel and is developed in section 11.

**Ceiling of Stage 2.** The Stage-2 ranker only sees the top 50. In this repo, of the 1,225 grade ≥ 2 pairs, 1,220 are inside the 50 candidates of the two-channel pool and 5 outside (recomputed from `features_data_sat/candidates_top50_regen.csv`). That looks like 99.6% recall, but it is flattered: the grades themselves were collected from the top of these same channels, so a relevant provider that no channel surfaced was never graded and never counted. The repo's own wording is that this is "a pool-coverage statement, not a claim that the catalog lacks a strong match" (section 6.5 explains the mechanism).

---

## 2. Matching text: lexical and dense

### 2.1 Lexical matching with BM25

**Idea.** Represent each text as a bag of words. A provider scores well for a gig if it contains the gig's words, especially the *rare* ones, and not merely by being long.

**Formula** (Okapi BM25, as implemented in the `rank_bm25` library the repo uses, with `k1 = 1.5`, `b = 0.75`):

```
score(D, Q) = sum over query terms t of   IDF(t) * f(t,D) * (k1 + 1)
                                                  ─────────────────────────────────────
                                                  f(t,D) + k1 * (1 - b + b * |D| / avgdl)

IDF(t) = ln( (N - n_t + 0.5) / (n_t + 0.5) )        N = number of documents,
                                                    n_t = documents containing t
f(t,D) = how many times t occurs in document D      |D| = length of D, avgdl = average length
```

(The library floors a negative IDF at `0.25 × the average IDF`, so a word in more than half the documents still contributes a little.)

**Intuition**, one factor at a time:

- **IDF** rewards rarity. A word in 3 of 2,165 documents is strong evidence; "experience" is none.
- **Term-frequency saturation.** `f(k1+1)/(f+k1)` rises quickly and then flattens: with `k1 = 1.5`, one occurrence gives 1.0, two give 1.43, three give 1.67, ten give 2.17, and no count can exceed `k1 + 1 = 2.5`. Repeating a word 50 times does not make a document 50 times more relevant.
- **Length normalisation.** The `b` term shrinks the contribution of a term in a document that is longer than average, because a long document contains many words by chance. `b = 0` switches this off.

**In this repo** (`corpus.py`, `retrieval_bm25.py`). "Refined" BM25 adds three things an embedding model does not need: a light hand-written suffix stemmer, **title weighting** (title tokens are repeated 3× in the document), and **query-side synonym expansion** from a hand-written `synonyms.json` (to bridge, for example, "M&A" and "mergers and acquisitions"). A provider's text is built from `about_title`, `about_description`, `services_offered_title`, `services_offered_description` and `relevant_experience`; a gig's from `hire_title`, `hire_description` and `hire_description_additional_notes`. A known, documented quirk: the query side counts the title 4× rather than 3× (the title is also inside the description string it is built from). It affects every run equally, so relative comparisons are unaffected.

**What it misses.** Vocabulary mismatch ("cash transfer programme" against a provider who writes "social protection") and **polysemy**. The repo's own worked example (`RERANK_README.md`, gig 745): for a gig on a *refugee cash transfer* trial, the fused top 5 contained two *Transfer Pricing* specialists, a lexical collision on the word "transfer" that BM25 introduced. Every one of that top 5 was graded irrelevant.

### 2.2 Dense matching with embeddings

**Idea.** A *bi-encoder* turns each text into one vector, once, offline. Matching is a dot product at query time.

**Formula.** For vectors `u` and `v`:

```
cos(u, v) = (u · v) / (‖u‖ ‖v‖)        for unit-length vectors this is just  u · v
```

**In this repo** (`retrieval_dense.py`). The model is `mixedbread-ai/mxbai-embed-large-v1`, which produces 1024-dimensional vectors. Three details matter later:

1. **Asymmetry.** The model was trained for *query to passage* search, so a query is prefixed with the instruction `Represent this sentence for searching relevant passages: ` and a passage is not. In this repo a gig is a query and a provider is a passage. The tag channel inherits this (section 13).
2. **Matryoshka truncation.** The model was trained so that the first 256 or 512 coordinates are themselves a usable embedding: you truncate and re-normalise to trade quality for size. The repo's retrieval class defaults to 256 dimensions, but the Stage-2 feature builder, the judging pools and the tagger all use the full 1024.
3. **Cosine ranges are narrow.** Cosines between texts of one domain sit in a narrow band (the `dense_cosine` values of a gig's top candidates are around 0.7), so absolute differences are small and the scale shifts from query to query. This is why Stage 2 normalises features per query (section 5).

**What it misses.** One vector must summarise a whole profile, so rare, specific skills can be blurred into the topic as a whole.

**What a good value looks like.** On the repo's fully judged comparison, dense alone reaches NDCG@10 0.717 against BM25's 0.618 (repo, same-grader table, reproduced in section 10). Dense is the strongest single channel here.

### 2.3 Why two channels beat one, and when they do not

Suppose a relevant provider is missed by channel A with probability `a` and by channel B with probability `b`. If their misses were independent, the chance both miss is `a × b` (for example `0.4 × 0.4 = 0.16`). If they miss *the same* providers (fully correlated), it stays at `min(a, b) = 0.4` and the second channel is wasted. Real channels sit in between, and the quantity that tells you where is **marginal recall**:

```
marginal recall of channel X at K =
    (relevant providers in X's top K that no other channel has in its top K)
    / (all relevant providers)
```

BM25 and dense are complementary because one reads *surface words* and the other *meaning*: RRF of the two beats both (NDCG@10 0.729 against 0.717 and 0.618; repo). A third channel must justify itself the same way: **it has to cover some error that the first two share.** Section 11 asks which error that could be.

---

## 3. Merging ranked lists: reciprocal rank fusion

**RRF** stands for **Reciprocal Rank Fusion**. It merges several ranked lists into one using only each item's *position* in each list.

**Formula.**

```
RRF(d) = sum over channels c in which d appears of     w_c / (k + rank_c(d))

rank = 1 for the best item;  k = 60 (default);  w_c = channel weight (1.0 here)
```

**Why ranks and not scores.** The three channels produce numbers on unrelated scales. From one real gig in the candidate file: BM25 scores around 100, dense cosines around 0.72, tag scores around 30. Averaging them would let whichever scale is largest dominate. Ranks are comparable by construction: "third best" means the same thing in every list.

**Why `k = 60`.** The `1/(k + rank)` curve is the *discount* given to lower ranks. Its steepness is controlled by `k`:

| rank | 1 | 2 | 5 | 10 | 50 | 100 | 300 |
|---|---|---|---|---|---|---|---|
| `1/(60+rank)` | 0.01639 | 0.01613 | 0.01538 | 0.01429 | 0.00909 | 0.00625 | 0.00278 |

The best item gets only 1.15× the credit of the tenth (`k = 1` would give 5.5×; `k = 1000` almost 1.0×). At `k = 60` the curve is flat, so **agreement across channels matters more than one channel's enthusiasm**: an item ranked 10th in all three lists (`3 × 0.01429 = 0.0429`) beats an item ranked 1st in one list and absent from the other two (`0.0164`).

**Missing items.** If a channel does not list an item, that channel contributes nothing for it. This is why the tag channel **drops providers with a zero score** instead of giving them an arbitrary tie-broken rank that RRF would then reward (`retrieval_tagbm25.py`).

**A worked example** (toy numbers, run through the repo's `rrf_fuse_n`):

```
BM25 list : p1 p2 p3 p4         dense list: p3 p5 p2 p1         tag list: p5 p3 p6

provider   BM25 rank  dense rank  RRF of two channels      tag rank  RRF of three channels
p3             3          1       1/63 + 1/61 = 0.03227       2      0.03227 + 1/62 = 0.04840
p5            none        2       1/62        = 0.01613       1      0.01613 + 1/61 = 0.03252
p1             1          4       1/61 + 1/64 = 0.03202      none    0.03202
p2             2          3       1/62 + 1/63 = 0.03200      none    0.03200

two-channel order:    p3  p1  p2  p5  p4
three-channel order:  p3  p5  p1  p2  p6  p4
```

`p5` is missed by BM25 and only second in dense, so with two channels it lands fourth. A third channel that ranks it first lifts it to second. The third channel did not need to be the best channel to change the outcome; it needed to *disagree usefully*.

**A real example** (recomputed from `features_data_sat/candidates_top50_*.csv`, qwen3 tags). Gig 1010, *IT Vendor Contract Review for Singapore Beverage Firm*, provider 274, a procurement specialist graded 3 (excellent):

```
BM25 rank 309     dense rank 3     tag rank 6

two-channel RRF  = 1/(60+309) + 1/(60+3)             = 0.002710 + 0.015873 = 0.018583   -> rank 19
three-channel    = 0.018583 + 1/(60+6)               = 0.018583 + 0.015152 = 0.033735   -> rank 4
```

BM25 buried it (rank 309, because the gig says "outsourcing agreements" and the profile says "supplier relationship management"); the tag channel recognised that both texts are about the same procurement and vendor-management skills.

**Variants in the repo.** `rrf_fuse_n` merges any number of lists with weights. `fuse_rankers.py` fuses the two LambdaMART output lists with weights 0.7 / 0.3 (it normalises the weights to sum to 1 and uses the same `k = 60`). The tag channel enters with weight 1, the same as BM25 and dense.

**Origin.** RRF comes from the information-retrieval literature (Cormack, Clarke and Büttcher, 2009, which also proposed `k = 60`); the pointer is from memory, see *Further reading*.

---

## 4. Measuring a ranking

**Idea.** To compare two rankers you need two things: a table of **relevance judgments** (for each gig, which providers are good, and how good), and a function that turns a ranked list plus those judgments into one number. Each metric below is computed **per gig** and then averaged over gigs. The gig, not the (gig, provider) pair, is the unit of analysis (section 6.1 explains why).

**In this repo** (`evaluate.py`). Each graded pair has a grade 0 to 3, stored as a score of 0, 33, 67 or 100 (the file holds 33 and 67; a few comments in the repo write 66). For the binary metrics, a provider is **relevant** if its score is at least 40, which means grade ≥ 2. A pair that was never graded counts as score 0 (irrelevant). That last rule is the source of the most important bias in this project (section 6.5).

### 4.1 The metrics

**Precision@K** answers "of the K providers shown, how many are good?"

```
P@K = (relevant providers in the top K) / K
```

It is the natural metric when a user sees a fixed first page. Its ceiling is lowered by scarcity: a gig with one relevant provider can score at most `1/5 = 0.2` at K = 5, however good the ranker.

**Recall@K** answers "of all the good providers, how many did we show?"

```
R@K = (relevant providers in the top K) / (all relevant providers for this gig)
```

It is undefined for a gig with no relevant provider, so those gigs are left out of the average (the code returns `None`). This is how R can be averaged over 542 gigs while P is averaged over 1,023.

**MRR** (mean reciprocal rank) answers "how far down is the first good one?"

```
RR = 1 / (rank of the first relevant provider)      (0 if there is none)
MRR = the mean of RR over gigs
```

It is used where one good answer is enough: question answering, "I'm feeling lucky", known-item search. It is very sensitive to the top few positions (rank 1 gives 1.0, rank 2 gives 0.5, rank 10 gives 0.1).

**NDCG@K** (normalised discounted cumulative gain) answers "how close is the order to the best possible order, counting better items as worth more and early positions as worth more?"

```
DCG@K  = sum for i = 1..K of   gain_i / log2(i + 1)         gain = the pair's 0 / 33 / 67 / 100 score
IDCG@K = the DCG of the ideal ordering of ALL graded providers for the gig
NDCG@K = DCG@K / IDCG@K                                      between 0 and 1
```

- The **discount** `1/log2(i+1)` is 1.0 at rank 1, 0.63 at rank 2, 0.5 at rank 3, 0.39 at rank 5 and 0.29 at rank 10: a good provider is worth less the lower it is shown.
- **Normalising** by the ideal makes gigs comparable. A gig with a single weak match and a gig with five excellent ones can both score 1.0.
- The **gain** here is *linear* in the score (0, 33, 67, 100). The textbook form uses `2^grade - 1` (0, 1, 3, 7), which punishes missing a grade-3 item much harder. The choice matters: section 5 shows the repo once trained on one and evaluated on the other.
- NDCG uses **every** graded provider, including grade 1 ("plausible, not excellent"), whereas P, R and MRR only count grade ≥ 2.

### 4.2 A worked example

One gig, ten ranked providers, `A` first. Grades: `B` = 3, `E` = 2, `D` = 1, `H` = 1, and a grade-3 provider `X` that the system never retrieved. All others are 0.

```
rank  provider  grade  score  gain / log2(rank+1)
  1      A        0      0        0.00
  2      B        3    100       63.09      <- relevant
  3      C        0      0        0.00
  4      D        1     33       14.21
  5      E        2     67       25.92      <- relevant
  6      F        0      0        0.00
  7      G        0      0        0.00
  8      H        1     33       10.41
  9      I        0      0        0.00
 10      J        0      0        0.00                      DCG@10 = 113.63
                                                            ideal order = 100, 100, 67, 33, 33
                                                            IDCG@10 = 100 + 63.09 + 33.50 + 14.21 + 12.77 = 223.57
```

| Metric | Value | Reading |
|---|---|---|
| P@5 | 0.40 | 2 relevant (B, E) in the top 5 |
| P@10 | 0.20 | still 2 relevant in the top 10 |
| R@5 = R@10 | 0.67 | 2 of the 3 relevant providers (B, E, X); X was never retrieved |
| MRR | 0.50 | first relevant provider at rank 2 |
| NDCG@10 | 0.508 | 113.63 / 223.57 |
| NDCG@5 | 0.462 | |

Swap `A` and `B` (put the grade-3 provider first): P@5 and R@10 do not change, MRR becomes 1.0 and NDCG@10 rises from 0.508 to 0.673. Each metric is blind to some changes and sensitive to others, which is why the repo reports several together. (All values were produced by running the toy through `evaluate.py`.)

### 4.3 Which metric answers which question

| Metric | Question | Where it is the right tool |
|---|---|---|
| R@K of the pool (R@50) | Did the candidate generator include the good ones? | **Stage 1 / recall channels** |
| marginal recall | Did this channel add good ones the others lack? | judging a **new channel** |
| P@5, P@10 | Is the first page clean? | user-facing quality |
| MRR | Is the best answer at the very top? | "one good hit is enough" tasks |
| NDCG@10 | Is the whole top of the list well ordered, with graded relevance? | **Stage 2 / overall ranking quality** |

### 4.4 Ceilings: reading a number against what is achievable

With only about 1.2 relevant providers per gig on average (2.26 among the 542 answerable gigs), the metrics have low ceilings (recomputed from `ground_truth_llm.json`):

- **P@5 ceiling = 0.230** over all gigs (mean of `min(R, 5)/5`); P@10 ceiling = 0.120. The shipped ranker's P@5 of 0.146 is therefore about **63% of the best any ranker could do**, not 15% of anything.
- **MRR over answerable gigs** is the headline 0.341 rescaled: `0.341 × 1023 / 542 = 0.644`, since the 481 gigs with no relevant provider contribute exactly 0. A value of 0.64 means the first relevant provider is typically at rank 1 or 2.
- Some of the repo's docs say NDCG is "a hard 0" for the 481 gigs with no grade ≥ 2 provider. By the code it is not: those gigs still have grade-1 providers (467 of them), and NDCG gives grade 1 a gain of 33. Only 14 gigs with nothing above grade 0 score 0. This is a reading of `ndcg_at_k`, not a number the repo reports.
- **Absolute NDCG and recall depend on which lists were graded.** The same list scores 0.717 in one comparison and 0.712 in another in the repo, because the normaliser is the set of graded pairs and that set changes when lists are added. Only *paired differences within one run* are meaningful.

---

## 5. Stage 2: learning to rank

**Idea.** Once Stage 1 has produced 50 candidates per gig, Stage 2 is a supervised learning problem: for each (gig, candidate) there is a feature vector, the label is the LLM grade, and the goal is to order each gig's candidates well.

### 5.1 The features (`features.py`)

| Feature | Meaning |
|---|---|
| `bm25_score`, `bm25_rank` | the BM25 channel's score and its rank among all 2,165 providers |
| `dense_cosine`, `dense_rank` | the same for the dense channel |
| `rrf_score`, `rrf_rank` | the fusion score and the position in the pool |
| `budget_fit` | `1` if the provider's hourly rate is inside the gig's budget band; otherwise `max(0, 1 − gap / width)` with `width = max(budget_hi − budget_lo, 1)` |
| `seniority_fit` | levels are mid < senior < expert; `1 − abs(difference) / 2`, so 1, 0.5 or 0 |
| `avail_immediacy` | `1` if the provider is available by the gig's start date, otherwise `max(0, 1 − days_late / 60)`; the sentinels "now" and "asap" resolve to today's date, so **the value depends on the day it is built** |
| `tag_score`, `tag_rank` | *only with the tag channel on*: the third channel's score and rank |

The business features are the point of Stage 2. **No embedding can see a budget**: an excellent profile with a rate 50% over budget is graded 1, and a text model has no way to know that. (The structured terms in this dataset are synthetic; section 9.)

### 5.2 LambdaMART in four steps

Three families of learning-to-rank exist. *Pointwise* predicts each item's grade independently. *Pairwise* learns which of two items is better. *Listwise* optimises a list metric directly. **LambdaMART** is the practical hybrid, and the repo uses it through XGBoost (`XGBRanker`, `objective="rank:ndcg"`).

1. **Pairwise probability (RankNet).** Give each candidate a score `s`. Model the chance that `i` beats `j` as `P(i ≻ j) = 1 / (1 + e^(−(s_i − s_j)))`, a logistic function of the score gap, and train to raise it whenever `i` is truly better.
2. **Lambda gradients (LambdaRank).** Multiply each pair's gradient by `|ΔNDCG|`, the change in NDCG if the two were swapped. Swapping the items at ranks 1 and 2 matters more than at ranks 40 and 41, so the model spends its effort at the top. This turns a pairwise loss into something that follows a list metric without differentiating it.
3. **Boosted trees (MART).** The scores `s` come from gradient-boosted regression trees: each new small tree is fitted to the current lambda gradients. The repo uses 300 trees, depth 4, learning rate 0.05, row and column subsampling 0.9.
4. **Why trees here.** They need no feature scaling, capture interactions ("a good text score matters more when seniority fits"), and handle **missing values natively**: XGBoost learns which branch a `NaN` should take. That last property is why the tag features are left empty (NaN) for providers the tag channel did not return (section 16).

### 5.3 Two details that were worth +0.03 NDCG@10 together

- **Training objective versus evaluation metric.** XGBoost's `rank:ndcg` uses exponential gain by default (grades 0, 1, 2, 3 become 0, 1, 3, 7), while `evaluate.py` scores NDCG with linear gain. The model was optimising a different ranking from the one reported. Setting `ndcg_exp_gain=False` (`--linear-gain`) gave NDCG@10 0.658 to 0.681 (repo).
- **Per-query normalisation.** A tree splits on absolute values, but `bm25_score`, `dense_cosine` and `rrf_score` drift in scale from gig to gig. So each is rescaled **within the gig's 50 candidates**:

  ```
  z = (x − mean over the gig's candidates) / (standard deviation over the gig's candidates)
  ```

  After this, a split means "how does this candidate compare with its siblings for this gig". It added a further +0.006 to +0.010 NDCG@10 (repo). Rank features are already within-query and are left alone.

### 5.4 How the shipped ranker is produced

- **Training rows** are only the *judged* candidates (an unjudged pair is not silently treated as a negative; that would teach the model noise).
- **Out-of-fold (OOF) evaluation.** The gigs are split into 5 folds with `GroupKFold` so that **all candidates of a gig stay together**. Each fold is predicted by a model trained on the other four, so every reported ranking comes from a model that never saw that gig. A random split over rows would put the same gig on both sides and flatter the result.
- **Two recipes.** `linz` (linear gain, per-query z-scores) maximises NDCG@10. `noce` (exponential gain, raw features) is better at P@5. Their lists are merged by weighted RRF, `0.7 × linz + 0.3 × noce`: the shipped ranker keeps nearly all the NDCG gain and recovers about 40% of the P@5 the linear model gives up (repo).
- **Output.** `rerank_ltr.py --export-scores` min-max scales a single model's scores per gig to a 0 to 100 `semantic_score`, `100 × (score − min) / (max − min)`. The repo notes that this has not yet been run for `data_sat` and that the normalisation method is still an open item with the sponsor; `fuse_rankers.py` itself writes only ranked lists.
- **What is deliberately not in it.** A cross-encoder (a model that reads gig and provider *together* and is usually more accurate than a bi-encoder, at much higher cost) scored 0.365 NDCG@10 zero-shot and 0.555 fine-tuned against RRF's 0.671, and as a LambdaMART feature it lowered NDCG@10, so it was dropped (repo).

**What a good value looks like.** On all 1,023 gigs the shipped ranker reaches NDCG@10 0.684 against RRF's 0.671, with paired bootstrap difference +0.0132 [+0.0039, +0.0227], and R@10 0.858 against 0.774 (repo). The gain is modest in NDCG and large in recall within the top 10; the repo attributes the low absolute values largely to the corpus ceiling (section 4.4).

---

## 6. Experimental design and statistics

A metric value on its own says little. Whether system B really beats system A depends on how the data were split, how the labels were made, and how much the number would move if you had drawn different gigs. This section builds the vocabulary the rest of the document uses.

### 6.1 The unit of analysis, and why one number is not enough

**Idea.** Every metric above is a mean over gigs, so it is an *estimate* of how the system would perform on gigs in general. A different sample of gigs would give a different mean. The size of that wobble is the **standard error**:

```
SE of a mean = (standard deviation of the per-gig values) / sqrt(number of gigs)
```

It shrinks only with the square root of the sample, so four times as many gigs halve it. The repo encodes this as a rule of thumb: the noise bar of `0.02` found at 130 gigs becomes `0.02 × sqrt(130 / n)`, which is **0.007 at 1,023 gigs** and **0.014 at the 271 test gigs**. A difference smaller than the bar is within noise.

**Why the gig is the unit.** All the (median 22) graded pairs of a gig share the same query text, so they are strongly correlated. Treating each pair as an independent observation (**pseudo-replication**) would make intervals far too narrow. The gig is the independent draw, so every average and every resampling in this repo is over gigs.

### 6.2 Splits, leakage, and "no cross-contamination"

**Idea.** A model or a setting chosen by looking at data must be evaluated on *other* data, or the evaluation is optimistic. We use three roles for data:

| Role | Purpose | In this repo |
|---|---|---|
| training | fit model parameters | LambdaMART rows (judged candidates) |
| **dev** (validation) | choose settings: how many tags, which encoder, which scorer | 271 gigs |
| **test** | one final, honest estimate of the chosen setting | 271 gigs |

The dev and test sets are the two halves of the 542 answerable gigs, split by a seeded random permutation (seed 7). The discipline is to **choose on dev and confirm once on test**, which ensures no cross-contamination between selection and measurement: if you look at test, change something, and look again, test has become a second dev set and its number is biased upward.

Other places where leakage was designed out:

- **Split by the right unit.** Stage 2 uses `GroupKFold` by gig, so one gig is never half in training and half in evaluation. In the taxonomy-only experiments (Part A, section 20), 619 of the 2,001 roles share an identical tag set with another role, so roles are first collapsed into 1,606 *equivalence classes* and the classes are split, so that a role's twin never sits on the other side.
- **Out-of-fold predictions.** The LambdaMART number reported is always from a model that did not see that gig.
- **Unsupervised statistics.** The per-tag means used by the hubness correction (section 14) are computed over *all* gigs and providers, test gigs included. No label is involved, so this is not label leakage. It does mean the statistic is fitted on the same unlabelled texts it corrects, and that a deployed system must store the means (it does: `rank_text`, section 18).
- **A disclosed imperfection.** About 20 tag scorers were compared on a pool built from the *test* gigs during the hubness work (section 11 of `TAG_CHANNEL.md`), and the encoder was chosen on dev among about 30 candidates but the test half was looked at once before the graded run. Both are listed in the repo's caveats. This is what honest bookkeeping of selection looks like, and it is why section 23 treats the test numbers as strong evidence and not as final proof.

### 6.3 Confidence intervals by paired bootstrap

**Idea.** A confidence interval (CI) for a difference between two systems says which differences are consistent with the data. The **bootstrap** gets one without formulas by re-running the experiment on resampled data:

```
repeat B times (2,000 here):
    draw n gigs WITH replacement from the n gigs          (same draw used for both systems)
    compute  mean over the drawn gigs of (metric_B − metric_A)
the 95% CI is the 2.5th to the 97.5th percentile of the B values
```

**Why paired.** The same gigs score both systems, and a hard gig is hard for both. Resampling *the per-gig difference* cancels that shared difficulty. A toy shows how large the effect can be. Twelve gigs, per-gig NDCG for A and B (the standard deviation of A across gigs is 0.21, but the standard deviation of the per-gig *differences* is only 0.036):

| | Mean difference B − A | 95% CI | Reading |
|---|---|---|---|
| paired (resample the differences) | +0.044 | [+0.023, +0.063] | excludes 0: B wins on 10 of 12 gigs consistently |
| unpaired (resample A and B separately) | +0.044 | [−0.116, +0.206] | includes 0: the gig-to-gig spread swamps the effect |

**Reading a CI.** In the repo's tables `+0.039 [+0.027, +0.052]*` means a mean difference of +0.039, interval from +0.027 to +0.052, and `*` marks "the interval excludes 0". Excluding 0 says the data would be surprising if there were no difference; it does not give the probability that the effect is real, and it says nothing about whether the effect is *large enough to matter*. Compare the interval with the noise bar and with what the metric can express (section 25).

**What the bootstrap cannot repair.** It captures only the sampling variability of gigs. It does not fix a biased label set, a lenient grader, or a setting chosen on the same data. Those are the subject of the next three subsections.

### 6.4 Multiple comparisons and the winner's curse

**Idea.** If you try 30 candidates on the same dev gigs and keep the best, its measured advantage is inflated even if none is truly better, because you selected the luckiest draw of noise. The expected maximum of `n` independent standard normal draws is about 1.5 for `n = 10`, 2.0 for `n = 25` and **2.04 for `n = 30`** (simulated).

**In this repo.** About 30 tagger candidates were compared on the 271 dev gigs. The within-pool AUC advantage of the winner, Qwen3, was +0.065 with an interval of [+0.041, +0.091] (from `results_tag/encoder_bench_dev_step3a.json`; half-width 0.025, so SE ≈ 0.013). If *all 30* candidates had zero true effect, the expected best would be about `2.04 × 0.013 ≈ +0.026`. The observed +0.065 is about two and a half times that, and it was then confirmed on test at +0.058 [+0.031, +0.085]. This is a back-of-envelope argument of mine, not a calculation in the repo; its role is to show why confirming the winner on untouched data is mandatory, and why the confirmation counts for more than the dev number.

### 6.5 Ground truth: pooling, and why a new channel is judged unfairly

**Idea.** Nobody can grade 2.2 million pairs. The standard technique, **pooling** (from the TREC evaluations), grades only the union of the top results of the systems being studied. Every other pair is *unjudged*. Standard metrics then treat unjudged as irrelevant.

**In this repo.** Each gig's pool is: RRF top 12 + BM25 top 8 + dense top 8 + 3 random providers, 15 to 28 pairs after overlap (median 22), 22,289 graded pairs in all (`label.md`).

**The bias this creates.** Any system that retrieves things the pooling systems did *not* is penalised for them: a relevant provider that only it finds was never graded, so it counts as a miss. A new channel is by definition a system that retrieves different things. The numbers:

| Share of the top 10 that carries an original grade (271 test gigs) | |
|---|---|
| RRF of BM25 and dense | about 100% (it was the pool) |
| BM25 / dense top 10 | about 88% to 93% (repo and recomputed) |
| raw tag channel alone | 37.6% (repo) |
| qwen3 tag channel alone | 30.5% (recomputed) |

**The effect on a conclusion** (recomputed; code in section 22). Same 271 test gigs, same two ranked lists (RRF of BM25 and dense, against the same RRF with the qwen3 tag channel added), two label sets:

| Label set | NDCG@10 without the tag channel | with it | Difference [95% CI] |
|---|---|---|---|
| original grades, unjudged counted as irrelevant | 0.713 | 0.697 | **−0.016 [−0.034, +0.002]** (looks neutral or harmful) |
| every top-10 pair graded by one LLM prompt | 0.716 | 0.774 | **+0.058 [+0.046, +0.071]** (a clear gain) |

The sign flips. The two label sets answer different questions, and only the second is fair to a new channel.

**Remedies, and their traps.**

| Remedy | What it does | Trap |
|---|---|---|
| "Condensed" lists | drop unjudged providers from every list before scoring | favours whichever system ranks *fewer judged* items highly |
| Extended labels | grade the new channel's unjudged pairs with an LLM and merge | the new grader differs from the original, so the scales mix; here the second grader was more lenient, which inflated the new channel |
| **One grader, fully judged top-K** | grade *every* pair in *every* compared top-10 with *one* grader | cost: about 10,000 grader calls for 271 gigs; exact only for K ≤ 10 |

The third is the comparison the repo trusts (sections 5 and 13 of `TAG_CHANNEL.md`). A closely related trap appeared when *training* Stage 2: if the extended grades are used as training labels, which candidates received an extra grade depends on the tag channel, a signal the baseline's features cannot see. The baseline then learns that deep-ranked candidates with a grade tend to be positive and ranks ungraded ones high; its fused NDCG@10 fell to 0.22 against 0.56 (repo). Both systems are therefore trained on the original grades only.

A last pooling effect, also recomputed: with the tag channel in the fusion, the 50-slot candidate pool must share its places among three channels. Of the 1,225 original relevant pairs, 5 fall outside the two-channel top-50, 25 outside the pool with mxbai tags, and 20 outside the pool with qwen3 tags. The tag channel displaces a few providers whose grades came from the old pools.

### 6.6 Agreement statistics, and how many samples you need

When an LLM is the judge (next section) you must measure how well it agrees with a reference. The toolbox, with the conventional reading of each:

| Statistic | Definition | Reading |
|---|---|---|
| **Accuracy** | share of pairs with the same grade | simple, but inflated when one grade dominates (50.8% of pairs are grade 0) |
| **Within one** | share of pairs within one grade of each other | tolerant of near misses |
| **Cohen's κ** | `(p_o − p_e) / (1 − p_e)`, observed agreement `p_o` corrected for the agreement `p_e` expected by chance from each rater's base rates | 0 is chance, 1 is perfect. Conventional bands: 0.41 to 0.60 moderate, 0.61 to 0.80 substantial, above 0.80 almost perfect (Landis and Koch; a heuristic, not a law) |
| **Quadratic weighted κ** | κ in which a disagreement of two grades costs four times one of a single grade | right for ordinal grades |
| **AUC** | the probability that a randomly chosen positive outscores a randomly chosen negative (ties count half); 0.5 is chance | measures *ordering* independent of any cut-off |
| **Spearman ρ** | the correlation between the two rankings | monotone agreement |
| **Wilson interval** | a CI for a proportion that behaves for small `n` and for proportions near 0 or 1 | used for audit rates such as 30 of 30 = 1.00 [0.89, 1.00] |
| **Fisher's exact test** | tests whether two proportions differ, exact for small counts | used to compare confirmation rates between strata |

A toy for κ: two raters, 100 pairs, both say "relevant" on 40, both "not" on 45, and disagree on 15 (10 + 5). Observed agreement is 0.85, but each rater says "relevant" about half the time, so chance agreement is 0.50 and `κ = (0.85 − 0.50) / (1 − 0.50) = 0.70`. Raw agreement of 85% becomes a substantial but not near-perfect 0.70.

**Power: is the sample big enough to see the effect?** A test that cannot detect a plausible effect cannot support "no effect". For two proportions near `p` and a true gap `δ`, the sample needed per group for 80% power at the 5% level is roughly:

```
n per group ≈ 2 * (1.96 + 0.84)^2 * p * (1 - p) / δ^2
```

For the blind audits, `p ≈ 0.63` and the gap in confirmation rate that the main result implies is `δ ≈ 0.08` (repo), giving about **570 pairs per group**. The audits used 70 (round 3) or 120 (round 4) per group; at 70 the interval half-width is about ±0.16. The repo says exactly this about itself: the audit "is underpowered for a gap of about 0.08", so it can confirm a *direction* but not a *size*.

---

## 7. Softmax, and using an LLM as the judge

### 7.1 Softmax

**Idea.** The softmax turns any list of real numbers `z_1 ... z_m` into probabilities:

```
softmax(z)_i = exp(z_i / T) / sum over j of exp(z_j / T)
```

Larger inputs get larger shares, exponentiation makes the largest dominate, and the **temperature** `T` sets how sharply (small `T` approaches picking the maximum; large `T` approaches a uniform distribution). It is how classifiers turn scores into class probabilities, how attention weights are formed, and how a language model chooses its next token.

### 7.2 The softmax that grades every pair

The relevance labels in this repo are written by a language model (Qwen, `qwen3.8:27b` in the repo's notation) that is shown the gig, the provider profile and the structured terms, and asked for one digit, 0 to 3. The model's reply is a distribution over its vocabulary, and the API returns the log-probabilities of the likeliest first tokens. The labeller (`labeller.py`) then:

1. Takes the log-probabilities of the four tokens `"0"`, `"1"`, `"2"`, `"3"`.
2. **Applies a softmax over just those four**, which renormalises them to sum to 1: `p(d) = exp(logprob_d) / sum over d' in 0..3 of exp(logprob_d')`.
3. Sets the **grade** to the most probable digit (decoding at temperature 0), the **expected grade** to `sum of d × p(d)`, and uses `P(grade ≥ 2) = p(2) + p(3)` as a confidence.

A toy: log-probabilities `[−3.2, −1.1, −0.9, −2.8]` give `p = [0.048, 0.396, 0.483, 0.072]`, so the grade is 2, the expected grade 1.58 and `P(≥ 2) = 0.556`: a borderline positive. The repo's robustness sweeps keep a grade-2 pair only if `P(≥ 2) ≥ 0.6` or `≥ 0.8`.

**The rubric** (summarised from `label.md` and `labeller.py`): judge *content fit first*, then check the terms. Budget: up to about 15% over the top of the range is a minor mismatch, more than about 40% a serious one; a rate below the range is fine. Seniority (mid < senior < expert): one level apart is minor, mid against expert is serious. Availability: starting up to 2 weeks late, or 1 day a week short, is minor; over a month late or 2 days short is serious. Terms can only lower a grade. Grade 3 is excellent content with at most one minor mismatch; grade 2 is relevant but imperfect; grade 1 is surface relevance, or relevant expertise with a serious mismatch; grade 0 is no genuine relevance. So **a grade is overall fit, not content fit**.

### 7.3 A judge is an instrument that needs calibrating

Everything downstream rests on these grades, so they are themselves measured. Three layers of checking exist in the repo.

**Layer 1: calibration against the original grader.** The original grading prompt was not available, so `labeller.py` reconstructs it (`rubric_0_3.v2-repro`). On 300 already-graded pairs (75 per original grade) the reconstruction agrees exactly on 69%, within one grade on 97.3%, with quadratic weighted κ 0.818 and Cohen's κ on "grade ≥ 2" of 0.733 (repo). The confusion matrix shows the flaw: every original grade ≥ 2 stays ≥ 2, but **40 of 150 original grade 0 or 1 pairs are promoted to grade 2**. The AUC for "original ≥ 2" is 0.980, so the *ordering* is fine and the *cut-off* is lenient.

**Layer 2: better prompts.** Two more prompts were built and calibrated on the same 300 pairs: the original owner's text as a 0 to 3 prompt (`--prompt orig`, κ 0.86), and a continuous 0 to 1 prompt with anchor bands (`--prompt cont`, `rubric_0_1.v2-cont`, κ 0.90 banded, scored by reading the decoded number, so it has no `probs`). Bands are cut into grades at 0.175, 0.50 and 0.825, so **grade ≥ 2 means score ≥ 0.5**. Promotions of original 0/1 pairs to ≥ 2 fall from 40 to 8 and 13.

**Layer 3: a blind second rater from another model family.** Claude graded stratified samples blind (rounds 1 to 4; 40, 160, 195 and 240 pairs). The mechanics are the standard ones for a trustworthy audit: items shown with only the gig, the provider and their terms, in shuffled order under neutral ids; the key (stratum, Qwen grade, probabilities) held in a separate file that is not opened until the grades are saved; for round 4 a decision rule fixed *before* any grade was read. A different family was chosen because the tagger and the grader are both Qwen models and could share biases. Findings (repo):

- Of the reconstruction's positives that only the tag channel surfaced, Claude agreed on **60%** (133 of 220); for the original-text prompt **86%**, for the continuous prompt **85%**. The reconstruction over-credited, and the continuous prompt repaired that.
- The price: both new prompts are *stricter* than Claude and miss about a third of what Claude confirms. Claude is itself one rater with its own threshold, so neither is ground truth.
- All three prompts *order* tag-surfaced pairs almost identically (AUC about 0.86 against Claude), so they differ in **where the line is**, not in who ranks first.

**The convention that follows.** One prompt per comparison, never mixed (the scripts refuse a mixed file). From section 13 of `TAG_CHANNEL.md` on, the headline grader is the continuous prompt with **relevant = score ≥ 0.5**, with 0.6 and 0.7 as sensitivity checks and graded NDCG (gain = 100 × score) as a further one.

---

## 8. Hubness

**Idea.** In high-dimensional nearest-neighbour search, a few items, the **hubs**, appear among the nearest neighbours of a disproportionate share of all queries, while many others, **anti-hubs**, appear in none. Formally, count for each item `x` the number `N_k(x)` of queries whose `k` nearest neighbours include `x`. In low dimensions `N_k` is roughly bell-shaped. In high dimensions it becomes strongly right-skewed. The effect is a known consequence of the *concentration of distances* in high dimensions: points close to the centre of the data are moderately close to *everything*.

**Why it applies to our tagger.** The "items" are the 2,088 tag titles and the "queries" are texts. A short generic title such as *Financial Planning and Analysis* sits near the middle of the whole finance cloud, so it has a moderately high cosine with almost every finance text. A specific title such as *Debt Restructuring* is very close to a few texts and far from the rest. A ranking by raw cosine therefore favours the generic tags for every text.

**Measured on this data** (recomputed from the stored tag files, matching the repo's section 11):

| Raw cosine tagger, provider side | Value |
|---|---|
| tag with the highest mean cosine to a provider text | *Business Insights*, 0.564 |
| tag with the lowest mean cosine | *Western Cold Dish Preparation*, 0.320 |
| most frequent tag in the top 30 | *Personal Finance Advisory* on **813 of 2,165** providers (auditors, valuers and credit specialists alike) |
| share of all tag slots held by the 50 most frequent tags | **35.5%** |
| tags never picked for any provider | 547 of 2,088 |
| correlation of a tag's mean cosine with how often it is picked | 0.57 (providers), 0.60 (gigs) (repo) |

**Fixes**, each a per-tag adjustment `s'(x, t) = cos(x, t) − (something about tag t)`:

| Fix | The "something about tag t" | Status here |
|---|---|---|
| **Centring** | the tag's mean cosine over the corpus of texts, `μ_t` | **used** |
| z-scoring | `μ_t`, and divide by the tag's standard deviation `σ_t` | tried; adds off-topic picks (top-10 tags ranked beyond 200 by raw cosine: 7.4% against 1.3% for centring, gig side) |
| CSLS (cross-domain similarity local scaling) | half the mean cosine of the tag to its `k` nearest texts | tried; on the gold-tag check it was no better than centring (intervals overlap), and in the encoder bench it made provider-side hubness worse |
| Dual softmax | a *soft maximum* `T · log mean exp(s / T)` of the tag's scores in place of the mean; as `T` grows it tends to centring | tried, parked as too many layers |

For Qwen3 the repo centres and then divides by **one pooled standard deviation for the whole matrix** (0.0559 gig side, 0.0580 provider side), not a different one per tag. That puts all scores in units of one standard deviation without re-weighting individual tags, so the weighted scorer's threshold is simply 1.0.

**A toy of centring.** Two tags and one finance text about credit risk. The hub *Financial Analysis* has mean cosine 0.56 over all texts, and the specific *Credit Risk Modelling* has mean 0.40. The text's cosines are 0.62 and 0.58.

```
                       raw cosine    minus the tag's mean     centred
Financial Analysis        0.62          0.62 - 0.56            +0.06
Credit Risk Modelling     0.58          0.58 - 0.40            +0.18     <- now first
```

Raw cosine ranks the hub first; centred, the text is *unusually* close to the specific tag, which is the information we want.

**A subtle point worth knowing.** Centring the *tag vectors* (subtracting their mean vector) changes nothing: for a text `q`, the term `q · mean` is the same for every tag, so the order of tags for that text is unchanged. What helps is centring against the *texts*, which subtracts a different constant for each tag. The repo's session notes make exactly this point.

**Same kind of text.** The per-tag mean must be computed from texts of the kind being tagged. In a grader-free test, tagging 400 role descriptions with gold tags, centring with the mean over those descriptions raised gold-tag precision by **+0.032 [+0.011, +0.055] at 5 tags**; using means taken from provider profiles instead gave no gain (−0.001). Role descriptions are cleaner than provider profiles, so the profile means were wrong for them.

**After the fix** (recomputed): the most frequent tag falls from 813 to 282 providers (*Financial Closing*); the 50 most frequent tags hold **14.7%** of slots; tags never picked fall from 547 to 59; *Personal Finance Advisory* falls from 813 providers to 183. The gig side goes from 16.4% to 10.1%.

---

# Part II: This repo's setting

## 9. The data and the labels

**Provenance.** Gigs and provider profiles come from a sibling project, BT4103 *Scrape-and-Tag*, which extracts and enriches them from web sources; this repo's copy is `pipeline/data_sat/` ("sat" for Scrape-and-Tag). Of 1,216 extracted gigs, 1,023 pass a quality bar (grounded extraction, a `Deliverable:` line, a non-`OTHER` industry, a unique title, 300 to 1,200 characters); of 2,193 providers, 2,165 have a headline and an About section.

**What a record looks like** (gig 1010 and provider 274, truncated; both from the committed JSON):

```
GIG 1010  "IT Vendor Contract Review for Singapore Beverage Firm"      industry: Food & Beverage
  A Singapore-based food and beverage company faces rising IT costs and service disruptions due to a
  fragmented vendor landscape. We need a specialist to review existing outsourcing agreements and
  performance benchmarks to identify consolidation opportunities ...
  Deliverable: gap analysis report with prioritised vendor consolidation recommendations.

PROVIDER 274  "Procurement and Supply Chain Strategy Specialist for Singapore and Regional Organisations"
  I am a procurement and supply chain specialist with over 15 years of experience, based in Singapore ...
  My expertise covers supplier relationship management, competitive negotiations, digital procurement ...
  services: Procurement and Supply Chain Transformation
```

**Text fields used for matching.** Gig: `hire_title`, `hire_description`, `hire_description_additional_notes`. Provider: `about_title`, `about_description`, `services_offered_title`, `services_offered_description`, `relevant_experience` (credentials are excluded on purpose, to avoid double counting).

**Structured fields** (not text; used by Stage 2 and by the grader):

| Gig | Provider |
|---|---|
| `budget_lo`, `budget_hi` (S$ per hour) | `rate_per_hour` |
| `seniority_needed` (mid, senior, expert) | `seniority` (same scale) |
| `start_by` (a date or `asap`) | `available_from` (a date or `now`) |
| `commitment` (days a week) | `capacity` (days a week) |

A caveat the labelling notes state plainly: **these fields are synthetic.** An LLM inferred seniority, price tier and urgency from each record's text, and code then generated budgets, rates, dates and capacity from those categories using a rate card. Stage 2 therefore partly learns the generator's rules, not how real clients trade price against fit.

**No tags.** The records carry only a coarse `industry` field. There are **no skill tags** on gigs or providers, which is why the third channel has to predict them (section 11).

**The labels** (`label.md`). Each gig has a pool of graded providers, and an LLM assigned every pair a grade 0 to 3 for overall fit. The numbers to hold in mind:

| | |
|---|---|
| graded pairs | 22,289 over 1,023 gigs, a pool of 15 to 28 per gig |
| grades | 0: 11,315 (50.8%) · 1: 9,749 (43.7%) · 2: 555 (2.5%) · 3: 670 (3.0%) |
| gigs with no provider at grade ≥ 2 | 481 (47%) |
| structured terms matter | the share graded ≥ 2 is 9.1% for in-budget rates but 0.1% for rates over 40% above budget |
| grader | `qwen3.8:27b`, prompt `rubric_0_3.v2`, temperature 0; no human spot-check has been done |

The 47% is a statement about the *pool*, not about the catalogue: only about 1% of the 2,165 providers were graded for any gig.

**Label sets you will meet by name** (never mixed in one comparison):

| Name | What it is |
|---|---|
| **original** | the 22,289 grades above (`judgments.jsonl`, `llm_judgments_merged.json`) |
| **repro** | a reconstruction of the original prompt (`rubric_0_3.v2-repro`), used to grade 12,616 extra pairs the tag channel surfaced; found lenient |
| **extended** | original grades plus repro grades for the tag-only pairs (a mixture of two graders) |
| **orig-text** | the original prompt's text as supplied by its owner (`--prompt orig`) |
| **cont** | the continuous 0 to 1 prompt (`rubric_0_1.v2-cont`); relevant means score ≥ 0.5 |

**Splits.** Of the 542 answerable gigs, a seeded permutation (seed 7) assigns 271 to **dev** and 271 to **test** (section 6.2).

---

## 10. The first two channels and the shipped ranker

The channels were described in section 2. Here is how they perform and where the third attaches.

**Results on all 1,023 gigs, original labels, out-of-fold** (repo, `README.md`; P and NDCG average over 1,023 gigs, R over the 542 answerable):

| Stage | NDCG@10 | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| RRF of BM25 and dense (Stage 1 baseline) | 0.671 | 0.116 | 0.531 | 0.774 | 0.278 |
| LambdaMART, exponential gain (previous ranker) | 0.658 | 0.153 | 0.682 | 0.860 | 0.338 |
| LambdaMART, linear gain, per-query z-scores (`linz`) | 0.685 | 0.141 | 0.640 | 0.854 | 0.333 |
| **Fused 0.7 `linz` + 0.3 previous: the shipped ranker** | **0.684** | 0.146 | 0.657 | 0.858 | 0.341 |

**The channels compared fairly** (repo, section 5 of `TAG_CHANNEL.md`: 271 test gigs, one grader, every top-10 fully judged; P@10 is high because the graded set is enriched with the union of good candidates, so only compare rows):

| List | P@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| refined BM25 | 0.389 | 0.618 | 0.712 |
| dense (mxbai) | 0.479 | 0.717 | 0.765 |
| RRF of BM25 and dense | 0.481 | 0.729 | 0.787 |
| shipped ranker (no tags) | 0.564 | 0.794 | 0.892 |

Reading: dense beats BM25 by 0.10 NDCG@10, fusing the two adds 0.012 more, and Stage 2 adds a further 0.065. That is the ladder a third channel has to climb.

**Where the third channel can attach.** There are three insertion points, in increasing depth, and the repo uses the first and the third:

```
1. as a CHANNEL in the Stage-1 fusion        rrf_fuse_n([bm25, dense, tag])             <- done
2. as a WIDENER of the candidate pool        widen the top-50 with tag-only hits        <- discussed, untested
3. as two FEATURES for Stage 2               tag_score, tag_rank beside the 9 others    <- done
```

---

# Part III: The third channel

## 11. Why a third channel, and what error it should cover

Start from the principle in section 2.3: a new channel earns its place only if it covers errors the existing ones share. So, what do BM25 and dense get wrong *together*?

1. **Neither has a notion of "skill".** Both treat a profile as one blob, of words in one case and of meaning in the other. A business that curates a skills taxonomy has already decided what the units of "being able to do this work" are. A channel that maps both sides into that controlled vocabulary can match "outsourcing agreements" with "supplier relationship management" because both map onto *Vendor Management*, even though no word is shared (gig 1010 and provider 274 share seven tags; section 24).
2. **Rare, shared skills are decisive.** With IDF weighting over tags, two texts that both carry a tag found on only three providers receive a large score; a dense similarity has no such explicit emphasis.
3. **A different, discrete representation** brings a different failure profile: it can be wrong in ways a cosine score is not (wrong tags, section 17).

That is the intuition: **the third channel must cover some error that the other two do not.** Two things in this repo keep it honest.

**The tags are predicted, so we evaluate "channel plus tagger".** The channel was designed to match on tag IDs (its docstring describes a query as "the tag IDs a user picked from the same vocabulary"). The data has no tags, so `tag_corpus.py` predicts them by embedding. Every result in this document is therefore a statement about the channel *and* the tagger together; the channel inherits every tagging error.

**A fair objection, recorded in the project's own notes.** The tagger embeds with the same mxbai model as the dense channel and then discards most of the information (30 tags out of 2,088), so it could be a *lossy projection of dense*: anything it knows, dense already knows. The early evidence was consistent with this: with the original labels its marginal recall over BM25 and dense was 0.000 at K = 50. The reply the project pursued, in two parts:

- **Decorrelate the errors.** Swap the tagger's encoder for one from a different model family (Qwen3-Embedding) so the tag channel stops reproducing dense's mistakes. Two measures from the encoder bench quantify the decorrelation (repo, test half): the **Spearman correlation** of the channel's pair scores with dense falls from 0.459 (mxbai tags) to 0.395 (Qwen3 tags), and the **novelty@50**, the share of the channel's top 50 that is *not* in the RRF top 50, rises from 0.626 to 0.673.
- **Correct the tagger's known bias** (hubness, section 14), so the channel stops returning generic, domain-wide matches that dense already finds.

Whether this worked is measured, not argued, in Part IV.

**The channel at a glance.** What happens once, ahead of time, and what happens for each gig:

```
 OFFLINE, once per dataset and encoder
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │ embed the 2,088 tag titles twice (as passages and as queries)                       │
 │ embed the 1,023 gigs as queries and the 2,165 providers as passages                 │
 │ cosines: gigs against tag-passages, providers against tag-queries                   │
 │ per side: subtract each tag's mean (Qwen3: and divide by one pooled sd)             │
 │ keep the best 100 tags per text  ->  tags_hirers_*.json, tags_providers_*.json      │
 │     (the files also store the per-tag mean and sd, which serving needs)             │
 │ provider index: each provider's best 30 tags  ->  matrix B, df_t, idf_t             │
 └────────────────────────────────────────────────────────────────────────────────────┘
                                          │
 ONLINE, per gig (no API call)            ▼
 ┌────────────────────────────────────────────────────────────────────────────────────┐
 │ gig already in the files:  read its best 30 tags                                    │
 │ new gig (rank_text):  embed as a query -> cosines with the tag-passages             │
 │                       -> subtract the STORED mean -> best 30 tags                   │
 │ score every provider (IDF-overlap or weighted cosine); drop the zero scores         │
 │ -> a ranked list of the providers it has evidence for                               │
 └───────────────────────────────────────────┬────────────────────────────────────────┘
                                             ▼
                       RRF with the BM25 and dense lists   and / or   tag_score, tag_rank as Stage-2 features
```

---

## 12. The taxonomy

The tags come from the **SkillsFuture** skills taxonomy as exported in the GreyGigz database schema, kept in `pipeline/taxonomy_greygigz/` as SQL dumps. `greygigz.py` parses them directly (no database), asserts the expected row counts, and raises if an export is truncated.

```
 247 TRACKS (categories)            e.g.  "Engineering Procurement"   in sector "Engineering Services"
      │ contain
 2,001 ROLES (specialities)         e.g.  "Assistant Engineer / Officer (Engineering Procurement)"
      │ require, at a proficiency level 1-6          43,958 role-tag links, median 20 tags per role
 2,088 TAGS (skills)                e.g.  "Procurement Coordination and Policy Development"  (level 3 for that role)
```

| Fact | Value |
|---|---|
| roles sharing an identical tag set with another role | 619 (so 1,606 distinct tag sets, called equivalence classes) |
| most widespread tags (roles containing them) | *Stakeholder Management* (1,060), *Change Management* (672), *Continuous Improvement Management* (586) |
| tags carried by exactly one role | 39 |
| track names that repeat across different category IDs | 9 names over 22 IDs, so tracks are keyed by ID, never by name |

**What the channel uses.** Only the **tag titles**: 2,088 short strings, embedded. The role-tag links, proficiency levels and tracks are *not used by the tagger*. They are used only by the taxonomy-only experiments in Part IV (to build queries with known gold tags), and a design that used the curated structure (soft-map a text to roles and take their curated tag sets) is listed as untested future work.

**An important property of tag IDs.** They are matched by **ID**, never by title or embedding. After tagging, "Vendor Management" is an opaque identifier. Two texts match to the extent that they received the same identifiers.

---

## 13. Step 1: the tagger

**Goal.** Give every gig and every provider a set of taxonomy tags, with no labels and no training.

**Method (`tag_corpus.py`).** Embed the text, embed every tag title, and take the tags whose vectors are closest. In matrix form, with `T` the 2,088 tag-title vectors:

```
  text vectors            tag-title vectors                 similarity matrix
  G  (1,023 × d)   ·   (T as passages)ᵀ  (d × 2,088)   =   S_gig   (1,023 × 2,088)
  P  (2,165 × d)   ·   (T as queries)ᵀ   (d × 2,088)   =   S_prov  (2,165 × 2,088)

  every vector first scaled to unit length, so each entry is a cosine; d = 1024 for mxbai
  for each row, keep the m best columns: the text's tags, best first
```

**Two matrices, because the encoder is asymmetric.** As in section 2.2, the model wants queries and passages embedded differently. A gig plays the *query*, so it is compared with tag titles embedded as *passages*. A provider plays the *passage*, so it is compared with tag titles embedded as *queries*:

| Side | Text embedded as | Tag titles embedded as |
|---|---|---|
| gig | query (with the instruction prefix) | passage |
| provider | passage | query (with the instruction prefix) |

Because the two sides live in different similarity spaces, **each side gets its own statistics** in the next step.

**Encoders** (`tag_encoders.py` holds a registry of ten):

- **mxbai**: `mixedbread-ai/mxbai-embed-large-v1`, the same model as the dense channel; it reuses the dense channel's cached embeddings, so tagging costs only one pass over the 2,088 titles.
- **qwen3-0.6b**: `Qwen/Qwen3-Embedding-0.6B`, with the instruction `Instruct: Given a text, retrieve the skills it describes` on the query side. This instruction was written for tagging, not for gig-to-provider matching. Its vectors came from a GPU; a CPU rebuild differs by about 1e-3 in cosine, so never mix devices inside one comparison.

**Cost.** On CPU about 1.4 s per gig and 2.9 s per provider, roughly two hours for the whole dataset, once (commit `bb6ca3d`). Query-time tagging of one new gig is a single embedding plus a 2,088-wide dot product, with no API call.

**Quality of the raw tagger** (grader-free; 400 role descriptions with known gold tags; repo): precision at 5 predicted tags is **0.289**, falling to 0.135 at 30, against a chance level of 0.011; recall rises from 0.076 to 0.199. Roles retrieved from the *predicted* tags (5 tags) reach role@1 0.130 and track@1 0.415, against 0.642 and 0.766 from *gold* tags. So the tagging step is the bottleneck, and its errors tend to land in the right track (track@3 0.640) and the wrong role. Precision 0.29 is weak in absolute terms and 26 times chance, which is why it supports a recall helper and not a classifier.

---

## 14. Step 2: the hubness correction

The raw top tags are dominated by hubs (section 8). The tagger therefore scores each (text, tag) pair by a **corrected** score before taking the best `m`:

```
  mxbai      score(x, t) =  cos(x, t) − μ_t                      centring
  qwen3      score(x, t) = (cos(x, t) − μ_t) / σ                  centring, then one pooled scale
  where  μ_t = the tag's mean cosine over ALL texts of that side (1,023 gigs, or 2,165 providers)
         σ   = one standard deviation for the whole centred matrix (0.0559 gig side, 0.0580 provider side)
```

The tagger stores each text's **best 100** tags with their corrected scores (4 decimals), while the channel itself uses the best **30**. It also stores `tag_ids`, `mean` and `sd` for each side, which is what lets a new gig be corrected later (section 18). Ties keep tag order, so the output is deterministic.

**A real before and after** (gig 1010, the beverage firm's IT vendor review; recomputed from the three committed tag files):

| rank | raw cosine (mxbai) | centred (mxbai) | centred, in sd units (Qwen3) |
|---|---|---|---|
| 1 | Contract / Vendor Management 0.707 | Contract and Vendor Management 0.198 | Contract and Vendor Management 4.35 |
| 2 | Contract and Vendor Management 0.698 | Supplier Performance 0.197 | Contract / Vendor Management 4.13 |
| 3 | Vendor Management 0.679 | Vendor Management 0.195 | Contract Management 3.52 |
| 4 | Supplier Performance 0.661 | Contract / Vendor Management 0.191 | Procurement Performance Monitoring 3.49 |
| 5 | Supplier Sourcing 0.655 | Supplier Sourcing 0.179 | Vendor Management 3.45 |
| 6 | **Business Needs Analysis 0.655** | Food and Beverage Services 0.169 | Audit and Review Management 3.17 |
| 7 | Procurement Performance Monitoring 0.631 | Food and Beverage Service 0.158 | Procurement Management 3.04 |

Read three things from it. (1) The raw list carries *Business Needs Analysis* at rank 6, the single most frequent tag on the gig side (277 of 1,023 gigs): a hub. Centring removes it. (2) Centring is not a pure win: with mxbai, the gig's industry context (a beverage firm) now pulls in *Food and Beverage* tags, which are about the client, not the work. The Qwen3 list stays on procurement and contract skills. (3) Raw cosines span only about 0.10 from rank 1 to rank 30 on average (0.68 to 0.59; repo), and the BM25 scorer keeps neither that gap nor each tag's strength; the weighted scorer in section 15 does.

**Hubness before and after, by variant** (recomputed from the committed files; "slots" is the share of all 30-tag slots held by the 50 most frequent tags):

| Variant | Side | Most frequent tag | Providers/gigs carrying it | Slots held by top 50 | Tags never picked |
|---|---|---|---|---|---|
| raw cosine (mxbai) | provider | Personal Finance Advisory | 813 of 2,165 | 35.5% | 547 |
| centred (mxbai) | provider | Financial Closing | 282 | 14.7% | 59 |
| centred (Qwen3) | provider | Digital Technology Adoption and Innovation | 300 | 14.4% | 63 |
| raw cosine (mxbai) | gig | Business Needs Analysis | 277 of 1,023 | 16.4% | 292 |
| centred (mxbai) | gig | Artificial Intelligence Application | 95 | 10.1% | 36 |
| centred (Qwen3) | gig | Artificial Intelligence Application | 108 | 12.6% | 144 |

The Qwen3 gig side is somewhat *more* hub-prone than mxbai's (12.6% against 10.1%); the repo records this as a trade-off it accepted, because the fused ranking improved anyway.

**Files, per variant** (`pipeline/data_sat/`): `tags_{hirers,providers}.json` (raw, 30 tags), `…_hc.json` (centred mxbai, 100 tags), `…_hz.json` (z-scored mxbai, not committed), `…_hc-qwen3-0.6b.json` (Qwen3). Each is `{"model", "top_m", "tags": {id: [[tag_id, score], ...]}}`, plus the per-tag statistics for corrected variants.

**Why 30 tags per text?** Chosen on the dev gigs by the objective *channel-alone R@50 against the original labels*. The repo is frank that this is a weak objective: those labels were pooled from BM25 and dense, so it rewards agreeing with them. Further, with 30 tags on every provider the BM25 length normaliser is a constant (next section), so every setting of `b` tied exactly (R@50 0.6536) and "b = 0.75" is an arbitrary tie-break. A control that shuffles gig tags across gigs falls to R@50 0.064 (a random ranking gives 0.023), so the channel does use real signal.

---

## 15. Step 3: scoring in tag space

After the tagger, every provider is a **set of 30 tag IDs** and a gig is a set of 30 tag IDs. Matching is a set problem, and there are two scorers over the same sets.

```
     providers (documents)                      taxonomy tags                    one gig (query)
   ┌────────────────────────┐              ┌────────────────────┐          ┌──────────────────────┐
   │ p274  ─────────────────┼──── 30 ──────┤ Vendor Management  ├──────────┤ the gig's 30 tags    │
   │ p115  ─────────────────┼──── 30 ──────┤ Supplier Sourcing  ├──────────┤ (best first, with    │
   │ p1375 ─────────────────┼──── 30 ──────┤ Procurement Mgmt   ├──────────┤  corrected scores)   │
   │  ...  2,165 providers  │              │  ...  2,088 tags   │          │                      │
   └────────────────────────┘              └────────────────────┘          └──────────────────────┘
        B[d, t] = 1 if provider d carries tag t               a provider scores by the tags it shares
        df_t    = how many providers carry tag t              with the gig, weighted by how rare each is
```

### 15.1 Scorer 1: BM25 over tag IDs (the default)

**Formula** (`retrieval_tagbm25.py`). With binary term frequency (a provider either carries a tag or not), `N` providers, `df_t` of them carrying tag `t`, `|d|` the number of tags on provider `d`, and `q_t = 1` if the gig carries `t`:

```
idf_t   = ln( 1 + (N − df_t + 0.5) / (df_t + 0.5) )            never negative, even if every provider has the tag
c_d     = (k1 + 1) / ( 1 + k1 * (1 − b + b * |d| / avgdl) )     k1 = 1.2, b = 0.75
score(d)= c_d * sum over tags t of  idf_t * q_t * B[d, t]
```

This is the text BM25 of section 2.1 with `f = 1` for every present term. Two differences from the text version are worth noticing. The IDF here is the `ln(1 + …)` form used by Lucene, which cannot go negative, whereas the text version uses `ln((N − n + 0.5)/(n + 0.5))` with a floor. And the "document length" is the number of tags, not words.

**Intuition.** Count the tags a provider shares with the gig, but let each count in proportion to how *rare* the tag is among providers. A shared tag carried by 3 providers is worth 6.4; one carried by 300 is worth 2.0.

| providers carrying the tag (`df`) | 1 | 3 | 10 | 50 | 183 | 300 | 813 (raw variant only) |
|---|---|---|---|---|---|---|---|
| `idf` with `N = 2,165` | 7.275 | 6.428 | 5.329 | 3.759 | 2.468 | 1.975 | 0.979 |

**A simplification that follows from the data.** Every provider carries exactly 30 tags, so `|d| = avgdl = 30` and

```
c_d = (k1 + 1) / (1 + k1 * (1 − b + b * 1)) = (k1 + 1) / (1 + k1) = 1    exactly, for every provider
```

(recomputed: the set of all 2,165 values of `c_d` is `{1.0}`). The score is therefore the plain **IDF-weighted overlap** of two 30-tag sets, and `k1` and `b` have no effect on the channel. This is the reason every `b` tied during tuning. BM25's length term earns its keep only when documents have different numbers of tags, as in the taxonomy-only experiments where roles carry between 2 and 78 tags and `b = 0.25` was best.

**A hand-worked fixture** (from `tests/test_tag_bm25.py`, run by the unit tests). Four documents over five tags: doc 10 = {101, 102}, doc 20 = {101, 102, 103, 104}, doc 30 = {103, 104}, doc 40 = {105}. Then `N = 4`, `avgdl = 2.25`, tags 101 to 104 each appear in 2 documents (`idf = ln 2`), tag 105 in one (`idf = ln(10/3)`), and `c = 2.2/2.1`, `2.2/2.9`, `2.2/2.1`, `2.2/1.7` for the four documents. For the query {101, 102, 103}: doc 20 overlaps on three tags and scores `3 ln2 × 2.2/2.9`; doc 10 on two, `2 ln2 × 2.2/2.1`; doc 30 on one; **doc 40 on none, so it is absent from the output**, not ranked last.

**A real score** (recomputed with the repo's `TagChannel`, Qwen3 tags). Gig 1010 against provider 274 share seven tags:

| shared tag | providers carrying it | `idf` |
|---|---|---|
| Vendor Management | 7 | 5.666 |
| Contract and Vendor Management | 15 | 4.940 |
| Contract / Vendor Management | 15 | 4.940 |
| Procurement Performance Monitoring | 26 | 4.403 |
| Procurement Management | 29 | 4.296 |
| Supplier Performance | 29 | 4.296 |
| Supplier Sourcing | 34 | 4.140 |
| **sum = the channel's score** | | **32.681** |

That is exactly `tag_score = 32.680994` in the candidate file, with `c_d = 1.0`. Provider 115, another supply-chain specialist, shares exactly the same seven tags, so the two **tie** at 32.681 and the stable sort breaks the tie by provider order. Discrete scoring produces coarse ties, which the rank-based RRF absorbs without comment.

### 15.2 Scorer 2: weighted cosine

The BM25 scorer throws away *how strongly* each tag matched, keeping only whether it made the top 30. The weighted scorer (`TagWeightedCosine`) keeps the strength:

```
w_t(x) = max(0, s(x, t) − τ)      for the text's 30 best tags (s = the corrected score), 0 for every other tag
score(g, p) = ( w(g) · w(p) ) / ( ‖w(g)‖ ‖w(p)‖ )                 cosine of the two sparse weight vectors
τ = 0.05 for centred mxbai scores (about one per-tag sd),  τ = 1.0 for Qwen3 scores (one pooled sd)
```

There is **no IDF and no length term**: the hubness correction has already down-weighted generic tags, and the cosine normalises vector length. It requires a corrected variant.

A toy, τ = 1: a gig with tag scores A 4.3, B 3.5, C 3.0 has weights (3.3, 2.5, 2.0); provider 1 has A 4.9, C 2.0, D 3.8, so weights (A 3.9, C 1.0, D 2.8); provider 2 has B 3.1, E 2.6, so weights (B 2.1, E 1.6). Provider 1: `(3.3×3.9 + 2.0×1.0) / (4.598 × 4.904) = 0.659`. Provider 2: `(2.5×2.1) / (4.598 × 2.640) = 0.433`. (Checked by running the class.) Provider 1 shares the gig's strongest tag; provider 2 shares a middle one.

On the same gig, provider 274 is again 6th of 854 returned, so the two scorers largely agree on a clear case. Alone, the weighted scorer is stronger (NDCG@10 0.657 against 0.633 for BM25 with mxbai tags; repo), and once fused the two perform about the same.

### 15.3 What the channel returns: only providers it has evidence for

Both scorers return **every provider whose score is above zero**, best first, with no cut-off. A provider that shares no tag with the gig is not listed, and **RRF gives it nothing from this channel**. The list is long but not complete (recomputed over all 1,023 gigs):

| Variant | providers returned per gig, mean (min to max) |
|---|---|
| raw cosine tags | 730 (63 to 1,827) |
| centred, mxbai | 525 (140 to 1,095) |
| centred, Qwen3 | 505 (118 to 1,070) |

(The repo's how-it-works note says "about 480", measured on the first 300 gigs.) This "no opinion" semantics is the right behaviour for a *recall* channel: an absent provider is an abstention, not a vote against.

---

## 16. Step 4: fusion and the Stage-2 features

**In the Stage-1 fusion.** The three lists are merged with the formula of section 3:

```
RRF(d) = 1/(60 + rank_BM25(d)) + 1/(60 + rank_dense(d)) + 1/(60 + rank_tag(d))        weights 1, 1, 1
```

with the tag term simply absent for providers the channel did not list. The top 50 per gig become the Stage-2 candidates.

**As Stage-2 features.** `features.py --tag-channel --tag-variant V` adds two columns to every candidate row:

- `tag_score`: the channel's score; `tag_rank`: its rank in the channel's own list.
- **For a provider the channel did not return, the value is left empty (read as NaN)** for every corrected variant, so LambdaMART can distinguish "no evidence" from "ranked first". The original raw variant keeps a `0 / 0` sentinel so its committed files stay reproducible.
- In the `linz` recipe, `tag_score` is z-scored within each gig's candidates, like the other score features.

**Three real cases** (recomputed from `candidates_top50_regen.csv` and `candidates_top50_tag_hcq.csv`; all three providers are graded 3, excellent). In each, one of the two older channels buries the right provider and the tag channel breaks the tie:

| Gig | Provider | BM25 rank | dense rank | tag rank | RRF rank without tag | RRF rank with tag |
|---|---|---|---|---|---|---|
| 820, HIV sustainability roadmap | 1891, health policy and development finance | 118 | 4 | 3 | 17 | **5** |
| 1010, IT vendor contract review | 274, procurement and supply chain | 309 | 3 | 6 | 19 | **4** |
| 446, restructuring opinion | 110, financial restructuring MD | 1 | 80 | 2 | 16 | **2** |

Gig 820 and gig 1010 are cases where BM25 fails (the profile uses different words); gig 446 is a case where *dense* fails (rank 80) and BM25 and the tag channel agree. The channel is not "a better channel"; it is a channel whose agreement pattern differs.

---

## 17. How it fails

The same machinery produces confident wrong answers. Knowing their shape is part of understanding the channel.

**1. Same domain, different task.** Tags describe the *domain and skill family*, not the deliverable. Gig 63 needs *energy modelling simulations for rooftop solar*. Provider 1964, an energy-market economist who works on competition litigation, shares **13** of the gig's 30 tags (*Solar Photovoltaic Energy Assessment*, *Power Generation System Design*, *Electrical Systems Design*, and so on). It was graded 0, and the tag channel ranked it **1st** (BM25 6, dense 70; fused rank 13 without the tag channel, 5 with it). Gig 23 (commissioning ELV and BMS building systems for a hotel) likewise pulls a building-enclosure engineer, with 9 shared mechanical, fire-protection and plumbing tags, graded 0 and ranked 1st by the channel. The repo measured this: the false positives the tag channel produces that dense does not are mostly same-domain, different-task pairs, not cross-industry ones (the share of false positives in a clearly different industry is 0.57 for tag, 0.53 for dense, 0.62 for BM25). A hard industry gate would cost good providers: the share relevant is 0.446 for the same industry, 0.326 for a different one, 0.426 across industries.

**2. Generic overlap, and the taxonomy's gaps.** Gig 130 (an investment prospectus template for offshore wind projects) is tagged *Business Proposal Writing*, *Proposal Writing* and *Transaction Documentation for Prospectus Development*, which are right. The energy side is not captured: the taxonomy has no wind tag, and the only "energy" tag in the gig's top 100 is at rank 56, outside the 30 the channel uses. The channel's first-ranked provider is a healthcare consulting partner who shares generic proposal and document tags. The channel can only express what the vocabulary can.

**3. Context leaking into tags, and tagger noise.** The tagger embeds the *whole* gig text, so the client's industry leaks into the tags. Gig 1010 acquired *Food and Beverage* tags from the mxbai tagger (section 14). Gig 446 is an automotive supplier needing a restructuring opinion; it and provider 110 share twelve tags, including *Engine Cleaning* (2.70 sd on the gig side) and *Engine Disassembly and Assembly*, which describe the client's industry and not the work requested, and which sit on the provider side at only 1.16 and 1.00 sd, barely above the noise floor. *Ship Financing* is shared too. The repo does not discuss these tags, so treat this reading as mine. They did no harm here, because the provider is right for good reasons (*Debt Restructuring*, *Restructuring Insolvency Advisory*), but a rare tag that is context or noise is amplified by IDF.

**4. A wrong tag from a sibling role is costly; a random wrong tag is not.** In the taxonomy-only experiment (queries are 5 gold tags of a role), adding one random wrong tag costs `role@1` about 0.004. Adding one wrong tag borrowed from a *sibling role in the same track* costs **0.267**, while the track is still found (`track@1` unchanged). This matters because the tagger's errors land near the right role, which is exactly the costly kind.

**5. The bench that disagrees.** On gold-tag role descriptions, Qwen3 is slightly *worse* than mxbai on the test half (precision@5 −0.016, role@10 −0.050), while on real gigs it is clearly better. The repo reports both and goes with the real-gig evidence; the disagreement is a reminder that role descriptions are cleaner text than gigs and profiles.

**6. It is not independent of dense.** The tagger and the dense channel use related embedding models. The channel adds value through the taxonomy vocabulary and the correction; the Qwen3 swap reduces but does not remove the relationship (Spearman 0.395 with dense).

**7. Gig-side hubness remains higher for Qwen3** (12.6% against 10.1%), accepted as a trade-off.

---

## 18. Interfaces, flags, files

**Everything is behind `--tag-channel`, default off.** With the flag off, every code path is what it was before the channel existed, and the flag-off feature file is byte-identical to the pre-change one (same inputs, identical md5; repo).

| Where | Flag | Effect |
|---|---|---|
| `run_pipeline.py` | `--tag-channel` | also writes `tagbm25.json` and `rrf3_k60.json` (raw variant only) |
| `features.py` | `--tag-channel --tag-variant {raw,hc,hcw,hcq,hcqw}` | 3-way pool, `tag_score` and `tag_rank`; writes `candidates_top50_tag_<variant>.csv` and `train_pairs_tag_<variant>.csv`, never overwriting a baseline file |
| `build_judging_pools_sat.py` | `--tag-channel` | adds the tag channel's top hits to the pools (raw variant) |

The variant names: `raw` (raw cosine, BM25), `hc` (centred mxbai, BM25), `hcw` (centred mxbai, weighted cosine), `hcq` (Qwen3, BM25), `hcqw` (Qwen3, weighted cosine). Only `features.py` and the `eval_tag_*` scripts accept a variant; the two other scripts always use raw.

**The class** (`tag_channel.py`):

```python
channel = TagChannel(DATA_DIR, variant="hc-qwen3-0.6b", scorer="bm25")   # also top_m_*, k1, b, tau, max_returned
channel.rank(hire_id)                                  # [(provider_id, score), ...] best first, zero scores dropped
channel.rank_text(gig_vec, tag_vecs, tag_ids)          # the same for a gig that is NOT in the tag files
```

`rank_text` is what makes the channel deployable. It takes the new gig embedded as a query, the 2,088 titles embedded as passages and their sorted IDs, re-applies the **stored** per-tag mean (and sd) of the gig side, keeps the top 30, and scores against the existing provider index, with no API call. For a gig that is in the files it equals `rank()` up to the 4-decimal rounding of the stored scores. It needs the gig embedding from the same encoder and device family that built the files.

**What to change where** (from `TAG_CHANNEL_HOW_IT_WORKS.md`):

| To change | Edit | Then rebuild |
|---|---|---|
| tags kept per text | `DEFAULT_TOP_M_*` in `tag_channel.py` (files store 100) | nothing, but pools and labels must use the same value |
| BM25 `k1`, `b` | `TagChannel(k1=, b=)` | nothing (with 30 tags per provider they have no effect) |
| weighted-scorer threshold | `VARIANT_TAU` or `TagChannel(tau=)` | nothing |
| encoder | add an `EncoderSpec` in `tag_encoders.py` | `tag_corpus.py --encoder NAME`, then the Stage-2 CSVs |
| fusion weights | `weights=` in the `rrf_fuse_n` call | the candidate CSVs |

**Dependencies.** The channel itself uses only `numpy` and `scipy`; nothing in `retrieval_tagbm25.py` or `tag_channel.py` imports a deep-learning library or calls an LLM. Embedding needs the same local encoders the dense channel already requires. The only LLM calls in the whole project are in the offline grader, `labeller.py`.

**Tests.** `pipeline/tests/` holds unit tests with hand-worked fixtures for the scorer, RRF, the loader, the evaluation scripts and the labeller; run them with `python -m unittest discover -s pipeline/tests -t pipeline`.

---

# Part IV: How we know what we know

## 19. The history, from the commits

The git history is a record of an experiment being corrected as its measuring instruments were found wanting. Reading it in order explains why the evaluation has the shape it does. (55 commits; dates are the commit dates.)

| When | Commits | What was done | What was learned |
|---|---|---|---|
| 09-11 to 09-12 | `67e965d` `1ba5ee0` `8a8e10c` `8d687f7` | Stage 2 built: feature builder, cross-encoder, LambdaMART; dense fine-tune with a Matryoshka-aware loss | the two-stage skeleton, first on a 130-gig synthetic corpus |
| 09-29 | `5b586a1` `3e02d8b` `c31b7fa` `1311004` and docs | Re-trained on the real `data_sat`; a date-sentinel bug found in review (47.5% of rows silently fell to a default); cross-encoder fine-tuned, then dropped; training-gain mismatch and per-query scaling fixed (+0.027 NDCG@10) | **the baseline the channel must beat**: the shipped ranker, NDCG@10 0.684 |
| 09-30 | `bead3e2` `afcfa5b` `913a1a0` `5b7cfce` `b135803` | Taxonomy imported; `TagBM25` (numpy and scipy only, hand-worked tests); `rrf_fuse_n`; the loader that asserts row counts | components with unit tests before any experiment |
| 09-30 | `bb6ca3d` `57ef7c3` `4d6729f` `286cb75` `6e294f1` | Tagger and `TagChannel`; wired behind **default-off** flags; Part A (taxonomy alone), Part B (real data), tagger check | BM25 over tags is sound when tags are right (role@1 0.642), but tagging is weak (precision 0.289) |
| 09-30 | `d16e232` `72cf6f6` `c879049` `35b2e3e` `36a694f` `6d8a9f5` `a1c8174` `332bb70` | A zero-shot grader reconstructed; calibrated; tags predicted for `data_sat`; Part B on original labels; 5,500 tag-only pairs graded; Part B on extended labels | the original labels **cannot judge** the channel (pool bias), the extended labels **flatter** it (lenient grader) |
| 09-30 | `42087e4` `b2024fc` `5ebad20` `954934f` `3063e2d` `25c7725` `7911298` | First `TAG_CHANNEL.md` (verdict: do not turn on); the default fixed to the dev-tuned 30 tags (an earlier pool had used 10); pools that fully grade every channel's top K; more metrics; downstream evaluation | a mid-course correction of the project's own bug, recorded in its caveats |
| 10-01 | `6f2fa15` `768ca1a` `d0c7e89` `bbbc5e7` | Train Stage 2 on original grades only; grade the top 10 of every channel (12,616 pairs); same-grader comparison; report rewritten | **the fair comparison**: the raw channel adds nothing significant |
| 10-01 | `9c5afaa` `9449ec1` `0760599` | Blind audit by Claude, rounds 1 and 2; session notes with design ideas | the reconstructed grader over-credits tag-only positives (56% confirmed, against 100% for original ones) |
| 10-01 | `a6d44fe` `e0fbd00` `788a3fa` | **Hubness** diagnosed and corrected; weighted scorer; stricter relevance definitions; audit round 3; docs | a significant gain that shrinks, but survives, under stricter definitions |
| 10-02 | `57d4d4e` | Two further grader prompts, calibrated, compared with Claude | the continuous prompt repairs the over-credit |
| 10-02 | `fb039e1` `d1354b2` `4c02a40` `465abda` | **Encoder bench**; the Qwen3 tagger; score-based evaluation; Stage 2 with Qwen3 tags; audit round 4; section 13 | the largest gains yet (section 21) |
| 10-02 | `19c7249` | `TAG_CHANNEL_HOW_IT_WORKS.md` | the mechanism written down as it now stands |

A theme to notice: **each major step was a fix to the evaluation or to a diagnosed weakness, not a new idea.** Pool bias led to extended labels; the lenient grader led to same-grader regrading; the weak channel led to the hubness diagnosis; the grader audit led to a better prompt; the encoder bench came last. There was also a deliberate narrowing. The session notes record a "course correction: too many layers, keep it explainable": a supervised head, dual-softmax and CSLS corrections, encoder ensembles and tag prototypes were all tried on the dev gigs (about 30 candidates in all), found hard to explain, and **parked**. The plain swap of the encoder, with the same centring and the same BM25 over 30 tags, is what was kept and confirmed.

---

## 20. The evidence ladder

Each experiment below answers one question and has a blind spot that the next one is designed to remove. Climbing the ladder, trust in the *direction* of the result grows; none of the rungs, taken alone, would justify the conclusion.

```
 rung 7   better encoder + better grader + Stage 2 + audit by another model     <- strongest, current
 rung 6   fix the diagnosed weakness (hubness); re-validate against the audit
 rung 5   audit the grader itself with a rater from a different model family
 rung 4   ONE grader, EVERY top-10 pair graded            <- the first fair comparison
 rung 3   extended labels: grade what the channel surfaces (but with a lenient second grader)
 rung 2   real data, original labels (pooled from the other channels: biased against the channel)
 rung 1   the tagger alone, against gold tags (no LLM, no pooling)
 rung 0   the taxonomy alone, gold tags, retrieve roles (an upper bound)
```

### Rung 0: the taxonomy alone (`eval_tag_channel.py roles`, "Part A")

*Question.* If the tags were perfectly right, is BM25 over tag IDs a good way to match? *Design.* The documents are the 2,001 roles; a query is **5 tags sampled from one role's own tags**; success is retrieving that role. Because 619 roles are exact twins, rankings are scored at the level of the 1,606 tag-set equivalence classes, and ties are broken in expectation so that no method benefits from ID order. The classes are split into dev and test halves; `b` is tuned on dev; results are on test; intervals are bootstrapped **over classes**, not queries (queries from one class are not independent). *Result* (repo): BM25 `role@1` **0.642**, `role@5` 0.956, `track@1` 0.766. Simpler scorers fall short: IDF-overlap or plain coverage 0.424, sum-of-levels 0.282, latent semantic analysis 0.546, and BM25 fused with it 0.609. *Stress tests*: queries made of a role's 5 most widespread tags give `role@1` 0.343; and a **track-first funnel** (predict the track, then rank only its roles) loses, because the track predictor is 0.762 accurate and break-even would need 0.90 (`role@1`). *Blind spot*: gold tags and clean role text make it an **upper bound**; nothing here involves a tagger, a gig or a provider.

### Rung 1: the tagger alone (`eval_tag_channel.py tagger`)

*Question.* How good are the predicted tags, with no LLM grader involved? *Design.* 400 role descriptions (name plus description) are tagged exactly as providers are, and compared with each role's gold tags; precision and recall at 5, 10, 20, 30 tags, with 95% intervals over roles. *Result* (repo): precision 0.289 at 5 tags (chance 0.011), role@1 from predicted tags 0.130. The same harness is the grader-free test of the hubness fix: **+0.032 [+0.011, +0.055]** precision at 5. *Blind spot*: role descriptions are cleaner text than gigs or profiles.

### Rung 2: real data, original labels (`eval_tag_channel.py sat`, "Part B")

*Question.* On 271 held-out gigs, how does the channel do alone and fused? *Design.* The channel's dev tuning (tags per text, `b`) uses R@50 on dev gigs. Reports on test: channel alone, **marginal recall**, a shuffle control (gig tags swapped across gigs: R@50 falls to 0.064 against a random 0.023), and the fused ranking. *Result*: alone, NDCG@10 0.337 against dense 0.641; marginal recall 0.021 of positives at K = 10 and 0.000 at K = 50. *Blind spot*: **the labels were pooled from BM25 and dense**, so only 37.6% of the channel's top 10 was ever graded, and unjudged counts as irrelevant. These are lower bounds for the channel and, per the repo, "biased against it".

### Rung 3: extended labels

*Question.* What if the channel's unjudged pairs were graded? *Design.* An LLM prompt reconstructed from the rubric (`rubric_0_3.v2-repro`) graded the tag-only pairs (first 5,500, then every top-10 pair of every channel: 12,616 in all). *Result*: alone, NDCG@10 0.578 against dense 0.588; marginal recall now 0.317 at K = 10. *Blind spot*: **two graders on two scales.** The reconstruction promotes about 27% of original grade 0 or 1 pairs to grade 2 (calibration, section 7.3), and the tag channel's top 10 contains many more newly graded pairs than the others, so the lenient scale flatters it most.

The same channel, three label regimes, the same 271 test gigs, NDCG@10 alone:

| Label regime | Tag channel alone NDCG@10 | Dense alone | The channel is... |
|---|---|---|---|
| original grades (rung 2) | **0.337** | 0.641 | far behind |
| extended, two graders (rung 3) | **0.578** | 0.588 | level with dense |
| one grader, every top-10 graded (rung 4) | **0.610** | 0.717 | clearly behind dense, level with BM25 |

A swing of 0.27 for one system, from the labels alone. This is the core lesson of experimental design in retrieval: **before asking whether a system is good, ask whether the measuring instrument is fair to it.**

### Rung 4: one grader, every top-10 pair graded (`eval_tag_samegrader.py`)

*Question.* Remove the grader mismatch and the pool bias together: does the channel help? *Design.* On the 271 test gigs, **every pair in the top 10 of every compared list** (BM25, dense, tag, both RRFs, both shipped rankers; 7,345 pairs) is graded by one grader, reusing earlier reproduction grades for 2,420. Every list is *fully judged*, so metrics at K ≤ 10 are exact for that label set. Stage 2 trains on original grades only, for both systems. *Result* (repo): the raw channel alone is behind dense (NDCG@10 −0.107, CI excludes 0) and level with BM25 (−0.008, includes 0); in the RRF it adds **+0.010 (CI includes 0)**; in the shipped ranker **−0.003 (includes 0)** with MRR@10 **−0.034 (excludes 0)**. Verdict: do not turn it on. *Blind spot*: one grader, and the grader is known to be lenient.

### Rung 5: auditing the grader (`claude_audit.py`)

*Question.* Does the grader, and so every conclusion, survive a rater from another model family? *Design and result*: section 7.3. Rounds 1 and 2: of the reconstruction's positives, original-pool positives are confirmed 30 of 30, tag-only positives only **45 of 80 (0.56)**, so the reconstruction over-credits exactly the pairs that matter to the channel. *Blind spot*: one rater, with its own threshold.

### Rung 6: fix the diagnosed weakness, and re-check (`eval_tag_variants.py`)

*Question.* Does removing hubness help, and does the help survive the audit? *Design.* Compare the corrected channel against the uncorrected on the same fully judged top-10 (8,938 pairs), then repeat under **stricter definitions of relevant**: keep a grade ≥ 2 only if the grader's own `P(≥ 2)` is at least 0.6 or 0.8, and/or the pair has no serious term mismatch. Add the blind audit of exactly the pairs that decide the gain. *Result* (repo): RRF with centred mxbai tags beats RRF by NDCG@10 **+0.021 [+0.009, +0.034]** and P@10 **+0.026**. Under stricter definitions the NDCG gain holds (+0.017 to +0.020) while the P@10 gain shrinks (to +0.002 at `P ≥ 0.8`), meaning part of it came from positives the grader itself was unsure of. Claude confirmed 44 of 70 pairs entering and 44 of 70 leaving the top 10 (0.63 each), so the over-credit is common to both sides and shrinks the gain without reversing it: a **calibrated** P@10 gain of +0.0154 [+0.0048, +0.0262], and +0.0095 [−0.0114, +0.0290] in the worst case for the channel. In the shipped ranker only NDCG@10 moves (+0.012 to +0.013). *Blind spot*: about 20 scorers were compared on the test gigs' pool before the two finalists were graded, so the test split is not untouched.

### Rung 7: a better encoder, a better grader, Stage 2, another audit

*Question.* Does replacing the tagger's encoder help, in the fused ranking and in Stage 2, under a grader that does not over-credit? *Design.* Three stages, each guarding against a different mistake:

1. **The encoder bench** (`eval_tag_encoder.py`; no LLM grader). *Bench A*: gold tags on role descriptions (precision, role@k). *Bench B*: real gigs; the dev (or test) gigs' **original-graded pairs** are ranked by the cosine of the two weighted tag vectors, scored by per-gig **AUC** and NDCG within that pool. *Bench C*: diagnostics (hubness; correlation with dense; novelty). Candidates are centred per tag and divided by their pooled standard deviation so encoders on different cosine scales are comparable. About 25 to 30 candidates were compared on the dev gigs; Qwen3-0.6B won (AUC +0.065 [+0.041, +0.091] over mxbai); it was then checked **once** on test (AUC +0.058 [+0.031, +0.085], NDCG@10 +0.037), also above the next-best single encoder (mpnet, +0.030).
2. **Graded confirmation** with the continuous prompt (relevant = score ≥ 0.5; all 9,806 pairs of the compared top-10s regraded with that one prompt; section 22 re-derives it). The RRF gain with Qwen3 tags is **+0.058 NDCG@10**; against the mxbai tags **+0.039 [+0.027, +0.052]**; the mxbai gain reproduces under the new grader (+0.019 against +0.021 before), which is a sanity check on the grader.
3. **Stage 2** (LambdaMART with the tag features, trained on original grades only): **+0.043 NDCG@10 [+0.029, +0.057]** over no tags.
4. **Blind audit, round 4** (120 gigs, one entering and one leaving pair each, drawn whatever the Qwen grade, decision rule fixed beforehand): Claude's rate of grade ≥ 2 is 50/120 for entering pairs and 36/120 for leaving, a per-gig difference of **+0.117 [+0.008, +0.225]**. The rule calls the gain *supported*, but the interval only just excludes zero, the Qwen grader gives no difference on the same pairs (0.000 [−0.092, +0.092]), and the audit is underpowered for a gap of 0.08 (section 6.6).

*Blind spot.* One grader and one rater; no human labels; Stage 2 was not audited; the test gigs had already been used once.

**A cheap sanity check you can run without any grader** (recomputed). On the 542 answerable gigs, compute the BM25 tag-overlap score of every originally graded pair. If the tags carry signal, the mean score must rise with the grade:

| Tagger | Mean tag score, grade 0 / 1 / 2 / 3 | Per-gig AUC, grade ≥ 2 against < 2 |
|---|---|---|
| raw cosine (mxbai) | 9.2 / 20.5 / 24.7 / 29.0 | 0.693 |
| centred (mxbai) | 10.6 / 24.8 / 29.7 / 34.4 | 0.702 |
| centred (Qwen3) | 8.9 / 24.8 / 32.9 / 35.6 | **0.758** |

The scores rise monotonically with grade for every tagger, and Qwen3 ranks the good pairs better (AUC 0.758 against 0.702). This is a *within-pool* test: the pairs were chosen by BM25 and dense, so it measures how well the channel discriminates inside a pool that is already good. It says little about recall, which is why the centring gain, visible in the retrieval-wide results, is almost invisible here (0.693 to 0.702).

---

## 21. The results, in order

One table, with the warning that rows use different graders and pools, so **compare within a row, not between rows**. `*` means the paired 95% CI excludes 0.

| Tagger and scorer | Grader | Alone: NDCG@10 (dense in same run) | RRF with tag minus RRF: NDCG@10 | Shipped ranker with tag minus without: NDCG@10 |
|---|---|---|---|---|
| raw cosine, BM25 | repro, 0 to 3 | 0.610 (0.717) | +0.010 (CI includes 0) | −0.003 (includes 0); MRR@10 **−0.034*** |
| centred mxbai, BM25 | repro | 0.633 (0.712) | **+0.021*** [+0.009, +0.034] | **+0.013*** [+0.002, +0.025] |
| centred mxbai, BM25 | cont, ≥ 0.5 | 0.616 (0.691) | **+0.019*** | +0.010 [+0.000, +0.020] |
| **centred Qwen3, BM25** | cont, ≥ 0.5 | 0.661 (0.691); weighted scorer 0.681 | **+0.058*** [+0.046, +0.071] | **+0.043*** [+0.029, +0.057] |

The row that appears under two graders (centred mxbai) gives +0.021 and +0.019: the grader change did not move that result. Reading the table:

- **Raw tags**: a weaker channel than dense that adds nothing detectable on top of BM25 and dense.
- **Centring**: a small, significant Stage-1 gain, concentrated in NDCG; it shrinks when the grader's over-credit is calibrated away, and adds about +0.01 NDCG@10 in Stage 2.
- **Qwen3**: the channel alone comes within 0.010 of dense (weighted scorer; the gap was 0.056 with mxbai) and the gains are several times larger. Absolute values for the comparison that matters: RRF NDCG@10 0.716 to 0.774 (P@10 0.300 to 0.334); shipped ranker NDCG@10 0.771 to 0.814 (P@10 0.363 to 0.394). In Stage 2 the gain also holds at the stricter cut-off (score ≥ 0.7: NDCG@10 +0.039 over no tags, +0.029 over mxbai tags) and under graded NDCG (+0.041, +0.034).

**How large is +0.058?** Against the ladder of section 10 (different grader, so only an order of magnitude): fusing BM25 with dense added about 0.012, and the whole of Stage 2 about 0.065. A third channel that adds 0.058 in Stage 1 is of the same size as the entire second stage, which is why the Qwen3 result is taken seriously despite its caveats.

**The repo's verdict** (section 13 of `TAG_CHANNEL.md`): the plain encoder swap is a real gain over the earlier tagger, in the fused ranking and in Stage 2, and the Stage-1 direction is supported by an independent audit, but the size rests on one grader. **`--tag-channel` stays off by default and nothing shipped changed.** Adopting it (selecting `--tag-variant hcq` and the matching tag files for serving) is a separate decision.

---

## 22. Re-deriving the headline yourself

The best way to trust a number is to produce it from raw files. This script (about 45 lines; it needs only `numpy` and the repo) re-derives the Stage-1 result of section 21: the 271 test gigs, the top 10 of RRF without and with the Qwen3 tag channel, the grades from the continuous prompt, per-gig metrics, and a paired bootstrap. I ran it; its output is shown below it.

```python
# run from the repository root:  python headline.py
import csv, json, sys, zlib, collections
import numpy as np
sys.path.insert(0, "pipeline")
from evaluate import ndcg_at_k, precision_at_k, reciprocal_rank

USE_ORIGINAL_LABELS = False            # True reproduces the biased view of section 6.5
D, F = "pipeline/data_sat/", "pipeline/features_data_sat/"
hirers = json.load(open(D + "hirers.json"))
original = json.load(open(D + "ground_truth_llm.json"))      # {gig: {provider: 33 | 67 | 100}}, original grades

# 1. the split: answerable gigs, a seeded permutation, the second half is "test"
rng = lambda purpose: np.random.default_rng([7, zlib.crc32(purpose.encode())])
gigs = [str(h["hire_id"]) for h in hirers if any(s >= 40 for s in original[str(h["hire_id"])].values())]
test = [gigs[i] for i in sorted(rng("sat/split").permutation(len(gigs))[len(gigs) // 2:])]

# 2. the two ranked lists per gig: the top 10 of the RRF pool without and with the Qwen3 tag channel
def top10(csv_name):
    out = collections.defaultdict(list)
    rows = sorted(csv.DictReader(open(F + csv_name)), key=lambda r: (int(r["hire_id"]), int(r["rrf_rank"])))
    for r in rows:
        if int(r["rrf_rank"]) <= 10:
            out[r["hire_id"]].append(int(r["provider_id"]))
    return out
without_tag, with_tag = top10("candidates_top50_regen.csv"), top10("candidates_top50_tag_hcq.csv")

# 3. ground truth: the continuous grader's grades, restricted to the evaluation's own pool of graded pairs
pool = {h: set(map(int, ps)) for h, ps in json.load(open(D + "judging_pools_variants_test_qwen.json")).items()}
band_score = {1: 33, 2: 67, 3: 100}
truth = {}
for line in open(D + "judgments_variants_cont.jsonl"):
    r = json.loads(line)
    if r["status"] == "ok" and r["grade"] > 0 and int(r["provider_id"]) in pool.get(r["hire_id"], ()):
        truth.setdefault(r["hire_id"], {})[r["provider_id"]] = band_score[r["grade"]]
if USE_ORIGINAL_LABELS:
    truth = original

# 4. per-gig metrics, then a paired bootstrap over gigs
metrics = {"NDCG@10": lambda ids, row: ndcg_at_k(ids, row, 10),
           "P@10":    lambda ids, row: precision_at_k(ids, row, 10),
           "MRR@10":  lambda ids, row: reciprocal_rank(ids[:10], row)}
resample = rng("demo/boot").integers(0, len(test), size=(2000, len(test)))     # the same gigs for both systems
for name, metric in metrics.items():
    a = np.array([metric(without_tag[h], truth.get(h, {})) for h in test])
    b = np.array([metric(with_tag[h], truth.get(h, {})) for h in test])
    lo, hi = np.percentile((b - a)[resample].mean(axis=1), [2.5, 97.5])
    print(f"{name:8} {a.mean():.3f} -> {b.mean():.3f}   diff {np.mean(b - a):+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
```

Output with the continuous grader (`USE_ORIGINAL_LABELS = False`):

```
NDCG@10  0.716 -> 0.774   diff +0.058   95% CI [+0.045, +0.071]
P@10     0.300 -> 0.334   diff +0.035   95% CI [+0.024, +0.046]
MRR@10   0.664 -> 0.751   diff +0.088   95% CI [+0.052, +0.123]
```

These match the repo's reported absolute values (0.716 to 0.774, 0.300 to 0.334) and its differences (+0.058, +0.035, +0.088). With `USE_ORIGINAL_LABELS = True` the same two lists give NDCG@10 0.713 to 0.697, difference **−0.016 [−0.034, +0.002]**, and P@10 0.170 to 0.163 (**−0.006** [−0.014, +0.001]), while MRR@10 moves the other way (**+0.052** [+0.017, +0.089]). The metrics no longer agree with one another, which is itself a warning sign: this is the pooling bias of section 6.5, produced by changing four lines.

Things worth trying: (a) replace `candidates_top50_tag_hcq.csv` by `candidates_top50_tag_hc.csv` to reproduce the mxbai-tag comparison (+0.019 [+0.007, +0.032]); (b) compare the two tag lists with each other (+0.039 [+0.027, +0.052]); (c) change step 3 to count only the continuous prompt's top band as relevant (`r["grade"] >= 3` in place of `r["grade"] > 0`). I tried it: NDCG@10 0.241 to 0.249, difference **+0.008 [−0.009, +0.026]**, and P@10 +0.002 [−0.002, +0.007], so with only "excellent" counted the gain is no longer distinguishable from zero. This is not in the repo; section 23 says what to make of it.

---

## 23. Threats to validity

A list of what could make the conclusions wrong, with the direction of the likely bias where one is known. Most are acknowledged in the repo's own caveats.

| Threat | Direction | State |
|---|---|---|
| **One LLM grader** for every label, and the tagger and the grader are both Qwen models | could inflate the Qwen3 channel if they share biases | partly addressed by the blind Claude audit (rounds 3 and 4), which is itself one rater and underpowered for the size |
| **No human labels** | unknown | the repo lists a human-graded sample as what would most change confidence |
| **The test gigs were used during selection** (about 20 scorers compared on their pool; Qwen3's bench B on test looked at once before the graded run) | inflates the test numbers slightly | disclosed; the 271 *dev* gigs (6,498 pairs, about 1.8 hours of grading) are still ungraded and could give a clean confirmation |
| **Winner's curse** among about 30 dev candidates | inflates the dev gain | the test confirmation (+0.058 AUC against +0.065) and the back-of-envelope in section 6.4 argue it is small here |
| **Pool bias** | against the new channel at K > 10, and in Recall's denominator | exact for K ≤ 10 only; R@20 and R@50 remain lower bounds |
| **Synthetic structured terms** (budget, seniority, availability) | the labels reward the generator's rules, so Stage 2 and the grades partly encode them | structural; noted in `label.md` |
| **Scarce positives**: 47% of gigs have none | wide intervals, ceilings, and evaluation on the 542 answerable gigs only | structural |
| **A single fold split and seed** | the repo estimates the NDCG@10 spread across splits at about ±0.008 | effects near that size are unproven until they replicate across splits |
| **Stage 2 not audited**; its Qwen3 gain (+0.043) is larger than the mxbai gain (+0.010) | one comparison on test gigs already used once | consistent with Stage 1, but a single result |
| **Transductive per-tag means** (fitted on all texts including test) | none from labels; but needs stored means at serving time | handled by `rank_text` |
| **Embedding numerics**: GPU and CPU vectors differ by about 1e-3 in cosine; `avail_immediacy` depends on the day | small shifts in near-tied ranks and features | every compared CSV must be built the same day on the same device |
| **Dependence on where "relevant" starts** (recomputed, section 22) | the Qwen3 gain is detectable when grade ≥ 2 (score ≥ 0.5) counts as relevant, shrinks at score ≥ 0.7 (still significant), and is not detectable when only the top band counts (+0.008 [−0.009, +0.026]) | the gain lives in "good but not perfect" providers, the very pairs on which graders disagree most; the top band is also sparse and under-assigned by this prompt, so the null is weak evidence, but the sensitivity is real |
| **One domain, one corpus** (1,023 consulting gigs) | a channel built on a skills taxonomy may transfer poorly | untested |

**How to hold the conclusion.** The direction (the Qwen3 tag channel improves the fused list and the Stage-2 ranker) is supported by four independent kinds of evidence: the encoder bench on original grades, a graded confirmation on held-out gigs, a Stage-2 run, and a blind audit that agrees in sign. The *size* (+0.058 and +0.043 NDCG@10) rests on one LLM grader and is the part to treat with care.

---

# Part V: Putting it together

## 24. One gig, end to end

Everything in the previous parts, applied to one gig and one provider. All numbers below are real (recomputed from the committed files).

**The request.** Gig 1010, *IT Vendor Contract Review for Singapore Beverage Firm* (industry Food & Beverage; budget S$90 to 125 an hour; senior; start by 3 December 2026; 2 days a week). The client wants a specialist to review outsourcing agreements and performance benchmarks to find vendor consolidation opportunities.

**The provider.** Provider 274, *Procurement and Supply Chain Strategy Specialist for Singapore and Regional Organisations* (S$90 an hour; senior; available now; 2 days a week). Graded **3** by the original grader: excellent fit.

```
STAGE 1: three channels look at all 2,165 providers

  CHANNEL 1  refined BM25         the gig says "outsourcing agreements", the profile says "supplier
                                   relationship management": few shared words           -> rank 309
  CHANNEL 2  dense (mxbai)        same subject area in meaning                           -> rank 3
  CHANNEL 3  tags (Qwen3)         gig's top tags   : Contract and Vendor Management 4.35, Contract / Vendor
                                                     Management 4.13, Contract Management 3.52, ...
                                   provider's top   : Procurement 4.90, Procurement for Production Operations
                                                     4.80, Supplier Sourcing 4.59, Procurement Management 4.54, ...
                                   shared among each text's best 30: 7 tags
                                     Vendor Management (7 providers carry it) 5.67 | Contract and Vendor Mgmt
                                     (15) 4.94 | Contract / Vendor Mgmt (15) 4.94 | Procurement Performance
                                     Monitoring (26) 4.40 | Procurement Mgmt (29) 4.30 | Supplier Performance
                                     (29) 4.30 | Supplier Sourcing (34) 4.14                 sum = 32.68
                                   this puts provider 274 6th among 866 providers listed    -> rank 6

  RRF (k = 60)
     without the tag channel   1/(60+309) + 1/(60+3)             = 0.002710 + 0.015873 = 0.018583   -> rank 19
     with the tag channel      0.018583 + 1/(60+6)               = 0.018583 + 0.015152 = 0.033735   -> rank 4
     (the pool that goes on to Stage 2 is the top 50: this provider is inside it either way)

STAGE 2: one candidate row, the features LambdaMART sees
     bm25_score 22.43   bm25_rank 309   dense_cosine 0.689   dense_rank 3   rrf_score 0.0337   rrf_rank 4
     budget_fit 1.0 (rate S$90 inside S$90-125)   seniority_fit 1.0 (senior and senior)
     avail_immediacy 1.0 (available now)          tag_score 32.68   tag_rank 6
     (z-scored per gig for the "linz" recipe; the other recipe uses them raw; the lists are fused 0.7 / 0.3)

THE LABEL    grade 3
```

**What the example shows.**

1. **Complementary errors.** BM25 buried the right provider; dense and the tag channel found it. Without the third channel, RRF ranked it 19th; with it, 4th. Either way it is inside the 50-candidate pool, so Stage 2 can still reorder it, but the third channel hands Stage 2 a better starting order and one more feature.
2. **Why the match is not an accident.** The seven shared tags are all about vendor, supplier and procurement management, and every one is rare (carried by fewer than 35 of 2,165 providers, six of them by fewer than 30), so each is worth more than 4 points of IDF.
3. **Where the channel was wrong at the same time.** For this gig the channel's own first-ranked provider, *Government Contracts Compliance Expert* (score 42.0), was never graded; its second, provider 435, a former IT sourcing leader, is a grade-3 match. The channel is a good net, not a good judge.
4. **The business features matter too.** Had provider 274's rate been S$180, `budget_fit` would drop to `max(0, 1 − 55/35) = 0` (a gap of S$55 over a band 35 wide) and Stage 2 would push it down regardless of its tags, which no text-based channel could have foreseen.

---

## 25. What a strong value looks like

"Good" depends on the chance level, the ceiling, and the noise. This table collects them. Rows marked (repo) are quoted; the **judgement** column is mine, a rule of thumb and not the repo's.

| Quantity | Chance level | Ceiling in this repo | Values seen | A judgement |
|---|---|---|---|---|
| **R@50 of a channel** | 0.023 (50 of 2,165) | 1.0 | tag channel alone ≈ 0.65 on dev (repo) | it only needs to be high *and different* from the others |
| **R@10** (over the 542 answerable gigs) | 0.005 | 1.0 | RRF 0.774, shipped 0.858 (repo) | 0.86 is strong |
| **P@5** | about 0.0006 over all 2,165 providers (1.2 relevant on average); 0.024 inside the top-50 pool (1,220 relevant in 51,150 rows) | **0.230** (all gigs) | RRF 0.116, shipped 0.146 (repo) | 0.146 is 63% of the best possible |
| **P@10** | as above | **0.120** with original labels | 0.30 to 0.56 on enriched, fully judged sets | compare rows within a run only |
| **NDCG@10** | no simple chance value | 1.0 | 0.67 to 0.73 (RRF), 0.68 (shipped, 1,023 gigs), 0.77 to 0.81 (shipped, 271 test gigs, fully judged) | absolute values depend on which lists were graded |
| **MRR** | small | 0.644 over answerable gigs | 0.341 over all gigs = 0.644 over answerable | the first relevant provider is typically at rank 1 to 2 |
| **A difference in NDCG@10** | 0 | | noise bar 0.007 (1,023 gigs), 0.014 (271 gigs) | below the bar: nothing; +0.01 to +0.02 with a CI excluding 0: small but real; **+0.04 and above: large here** |
| **Marginal recall** | 0 | | 0.021 (original labels, biased low) to 0.317 (extended, biased high) | the truth lies between; use fully judged lists |
| **AUC** (ordering) | 0.5 | 1.0 | tag scores against grades 0.69 to 0.76; the grader prompts against original grades about 0.98 | rule of thumb: 0.7 acceptable, 0.8 good, 0.9 excellent |
| **Cohen's κ** (grader agreement) | 0 | 1.0 | 0.73 reconstruction, 0.86 original text, 0.90 continuous (banded) against the original grader; 0.30 to 0.55 prompt against Claude on tag-surfaced pairs | 0.6 to 0.8 substantial, above 0.8 almost perfect |
| **Gold-tag precision@5** | 0.011 | 1.0 | 0.289 raw; about 0.32 centred | 26 times chance but weak absolute: a recall helper, not a classifier |
| **Hubness: slot share of the 50 most frequent tags** | **2.4%** (50 of 2,088, perfectly even use) | | 35.5% raw, 14.7% centred (providers) | lower is more even; the repo's gate was "no worse than 10.1% / 14.7%" |
| **Tags never picked** | 0 | | 547 raw, 59 centred (providers) | fewer is better |
| **A 95% CI** | | | `+0.043 [+0.029, +0.057]` | excludes 0 and the lower end is still above the noise bar: a convincing effect size |

Three habits follow from the table. Compare a value with its **ceiling and chance level**, not with 1.0. Compare **paired differences with their intervals and the noise bar**, not absolute values across runs. And always ask **which labels and which grader** produced the number.

---

## 26. Open questions and next steps

Collected from the repo's own notes, in rough priority order.

1. **The adoption decision.** Whether to serve the Qwen3 tag channel (`--tag-variant hcq`, with the matching tag files and `rank_text`) is explicitly left open. Everything else here is evidence for that decision.
2. **A clean selection set.** Grade the 271 dev gigs' pairs (6,498 new pairs, about 1.8 hours at one call per second) so settings can be chosen on dev and confirmed on test with no winner's curse and no touched test data.
3. **Human labels.** A hand-graded sample, even a small one, would replace the reconstruction-versus-Claude arguments with a ground truth, and would show where the continuous prompt's 0.5 cut-off sits relative to people.
4. **Audit Stage 2.** The Stage-2 gain with Qwen3 tags (+0.043) has not been audited; round 4 covers only the Stage-1 entering and leaving pairs.
5. **A pool-widener test.** Using the channel to add candidates to the top 50 (insertion point 2 in section 10) is plausible, since the extended labels show relevant providers that the other channels miss, but its value depends on the grader and is untested.
6. **Tagger ideas parked for explainability**: a supervised head trained on the taxonomy's role-tag links, dual-softmax hubness correction, encoder ensembles, tag prototypes; and ideas never tried: soft-mapping a text to roles and taking their curated tag sets, an LLM-written tag description, learned sparse retrieval over the tag vocabulary.
7. **Replication across splits and seeds.** All Stage-2 numbers come from one fold split (seed 7).
8. **Pre-existing issues the repo flags and does not fix**: the BM25 query-side title is counted four times, not three; `budget_fit` penalises a below-budget rate as harshly as an above-budget one (a cheaper provider is usually fine); and the 0 to 100 calibration of the fused ranker has not been run for `data_sat`.
9. **Deployment.** Confirm the production gig payload carries budget, seniority and availability (otherwise Stage 2 learns on features absent at serving), and decide the fixed 0 to 100 normalisation with the sponsor.

---

## 27. Running things, and a file map

Everything below is run from the repository root. The recomputations in this primer need only `numpy` plus the committed files; building embeddings from scratch needs the encoders (GPU recommended; a CPU pass over all texts takes hours).

```bash
# Stage 1 and the baseline feature table
python pipeline/run_pipeline.py --data-dir data_sat
python pipeline/features.py --data-dir data_sat --top-k 50

# tags: once per variant
python pipeline/tag_corpus.py --data-dir data_sat                          # raw, mxbai, 30 tags
python pipeline/tag_corpus.py --data-dir data_sat --hubness center         # centred mxbai ("hc")
python pipeline/tag_corpus.py --data-dir data_sat --encoder qwen3-0.6b     # centred Qwen3 ("hc-qwen3-0.6b")

# the channel inside the pipeline (build the baseline and tag CSVs on the SAME day: avail_immediacy uses today's date)
python pipeline/features.py --data-dir data_sat --top-k 50 --tag-channel --tag-variant hcq

# evaluation, in the order of the evidence ladder
python pipeline/eval_tag_channel.py {roles,tagger,sat} [--extra-grades]      # rungs 0, 1, 2, 3
python pipeline/eval_tag_samegrader.py {pool,eval}                         # rung 4
python pipeline/eval_tag_variants.py {pool,eval,diff} --split test [--prompt cont] [--relevance all] [--stage2]   # rungs 6, 7
python pipeline/eval_tag_encoder.py --encoders mxbai qwen3-0.6b --split dev  # the encoder bench
python pipeline/eval_tag_downstream.py --extra-grades                        # Stage 2 with and without the channel

# grading and auditing (the grader needs SOCLAAS_* in .env; see .env.example)
python pipeline/labeller.py {pairs,calibrate,merge} [--prompt {orig,cont}] ...
python pipeline/claude_audit.py {sample,show,score,termcheck} [--round N]

python -m unittest discover -s pipeline/tests -t pipeline                    # the unit tests
```

**Where things live.**

| Path | What |
|---|---|
| `README.md`, `pipeline/RERANK_README.md` | the shipped ranker, its results and its runbook |
| `pipeline/label.md` | how `data_sat` was built and graded |
| `pipeline/TAG_CHANNEL.md` | the tag channel's results and verdict, sections 1 to 13 (the history of the work) |
| `pipeline/TAG_CHANNEL_HOW_IT_WORKS.md` | the mechanism as it now stands |
| `pipeline/audit/RESULTS.md`, `SESSION_NOTES*.md` | the Claude audit; session notes with design ideas and the encoder work |
| `pipeline/corpus.py`, `retrieval_bm25.py`, `retrieval_dense.py`, `retrieval_rrf.py` | text builders and the first two channels, RRF |
| `pipeline/greygigz.py` | the taxonomy loader |
| `pipeline/tag_encoders.py`, `tag_corpus.py` | the tagger (encoders, centring, tag files) |
| `pipeline/retrieval_tagbm25.py`, `tag_channel.py` | the third channel (BM25 over tag IDs, weighted cosine, `TagChannel`) |
| `pipeline/features.py`, `rerank_ltr.py`, `fuse_rankers.py` | Stage-2 features, LambdaMART, the 0.7 / 0.3 fusion |
| `pipeline/evaluate.py` | P, R, NDCG, MRR |
| `pipeline/eval_tag_*.py`, `labeller.py`, `claude_audit.py` | the experiments of Part IV |
| `pipeline/data_sat/` | the dataset, grades, pools, tag files, grader records |
| `pipeline/features_data_sat/`, `results_tag/` | candidate CSVs; the result JSON behind every table in `TAG_CHANNEL.md` |
| `pipeline/tests/` | unit tests with hand-worked fixtures |

**Suggested reading order.** This primer; then `TAG_CHANNEL_HOW_IT_WORKS.md` for the mechanism; then `TAG_CHANNEL.md` sections 5, 11, 12 and 13 for the decisive comparisons; then `audit/RESULTS.md`.

---

## Glossary

- **Anti-hub**: an item that appears in nobody's nearest-neighbour list (section 8).
- **Answerable gig**: a gig with at least one provider graded ≥ 2 (542 of 1,023); recall is defined only for these.
- **Asymmetric encoder**: an embedding model trained for query-to-passage search, which embeds a query and a passage differently (a prefix on the query).
- **AUC**: probability that a random positive outscores a random negative (section 6.6).
- **Bi-encoder**: embeds each text separately to one vector; matching is a dot product. Contrast **cross-encoder**, which reads both texts together.
- **BM25**: the standard lexical ranking function (section 2.1).
- **Bootstrap (paired)**: resampling gigs to estimate the sampling spread of a metric difference (section 6.3).
- **Candidate pool**: the top 50 per gig that Stage 2 reorders.
- **Centring**: subtracting each tag's mean cosine over the corpus, to remove hubs (section 14).
- **Channel**: one independent retriever that produces a ranked list (BM25, dense, tag).
- **Condensed list**: a list with unjudged items removed before scoring (a biased remedy for pool bias).
- **Cosine similarity**: the dot product of two unit vectors.
- **Dev / test**: the two halves of the answerable gigs; choose on dev, confirm once on test.
- **DCG / NDCG**: discounted cumulative gain, and its normalisation by the ideal (section 4).
- **df, idf**: the number of documents containing a term, and the rarity weight derived from it.
- **Equivalence class (of roles)**: roles with an identical tag set, indistinguishable from tags alone.
- **Extended labels**: original grades plus a second grader's grades for the tag-only pairs (mixes two scales).
- **GroupKFold**: cross-validation that keeps all rows of one group (here, a gig) in the same fold.
- **Hub, hubness**: an item that is among the nearest neighbours of many queries; the phenomenon (section 8).
- **κ (Cohen's kappa)**: chance-corrected agreement between two raters (section 6.6).
- **LambdaMART**: gradient-boosted trees trained with LambdaRank gradients to optimise a ranking metric (section 5).
- **Marginal recall**: relevant items a channel finds that no other channel finds (section 2.3).
- **Matryoshka embedding**: an embedding whose leading coordinates form a smaller usable embedding.
- **MRR**: mean reciprocal rank of the first relevant item.
- **Out-of-fold (OOF)**: predictions for a row made by a model that did not train on its group.
- **Pooling**: grading only the union of several systems' top results; the rest is unjudged (section 6.5).
- **Pseudo-replication**: treating correlated observations (pairs of one gig) as independent.
- **Query / passage prefix**: the instruction string added to a query before embedding.
- **Recall channel**: a first-stage component whose job is to include the relevant items in the pool.
- **repro / orig-text / cont**: the three grader prompt variants (section 9).
- **RRF**: reciprocal rank fusion, `sum w / (k + rank)` (section 3).
- **Same-grader, fully judged**: every pair in every compared top-K is graded by one grader (section 6.5).
- **Softmax**: turns scores into probabilities (section 7.1).
- **Stage 1 / Stage 2**: candidate generation and learning-to-rank reordering.
- **Tag, tagger, taxonomy**: a skill ID from the SkillsFuture-based vocabulary; the embedding procedure that assigns tags to a text; the whole vocabulary of tracks, roles and tags.
- **Transductive**: using unlabelled test-time data to fit a statistic (here, the per-tag means).
- **Weighted cosine (wcos)**: the second tag scorer, a cosine of `max(0, score − τ)` weight vectors (section 15.2).
- **Wilson interval**: a confidence interval for a proportion that works for small samples.
- **Winner's curse**: the inflated advantage of the best of many noisy candidates (section 6.4).
- **z-score**: `(x − mean) / sd`; used per gig in Stage 2 and, with one pooled sd, in the Qwen3 tagger.

---

## Further reading

These pointers are from memory and have not been checked against the sources; verify titles and years before citing.

- S. Robertson and H. Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009).
- G. Cormack, C. Clarke and S. Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*, SIGIR 2009.
- C. Burges, *From RankNet to LambdaRank to LambdaMART: An Overview*, Microsoft Research technical report, 2010.
- K. Järvelin and J. Kekäläinen, *Cumulated gain-based evaluation of IR techniques*, 2002 (NDCG).
- M. Radovanović, A. Nanopoulos and M. Ivanović, *Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data*, JMLR 2010.
- A. Conneau et al., *Word Translation Without Parallel Data*, 2018 (CSLS).
- A. Kusupati et al., *Matryoshka Representation Learning*, NeurIPS 2022.
- B. Efron and R. Tibshirani, *An Introduction to the Bootstrap*, 1993.
- J. R. Landis and G. Koch, *The Measurement of Observer Agreement for Categorical Data*, 1977 (the κ bands).
- C. Manning, P. Raghavan and H. Schütze, *Introduction to Information Retrieval*, 2008 (a general textbook for sections 1 to 4).
- For evaluation methodology with pooled judgments and significance testing, the TREC literature (Voorhees and Harman) and Smucker, Allan and Carterette's comparison of significance tests (2007).
