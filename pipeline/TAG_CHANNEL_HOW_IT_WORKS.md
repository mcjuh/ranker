# How the tag channel works, exactly

> **Status note:** the front-end schema now carries user-selected `search_tags`. For those records see `EXPLICIT_TAG_CHANNEL.md`; this predicted-tag channel remains as the fallback. What is active, shelved and borrowable is in `CHANNEL_STATUS_AND_ARCHIVE.md`.

This describes the code as it is on branch `tag-encoder-test` (commit `465abda`). For what the channel measured and whether to
use it, see `TAG_CHANNEL.md` (results and verdict) and `audit/SESSION_NOTES_encoder_and_grader.md` (the encoder work). This file
is only the mechanism. Every number below was read from the code or from the files in `data_sat/`, not from memory.

## In one paragraph

The third channel ranks providers by how many **taxonomy tags** they share with a gig. Neither side has real tags, so both are
**predicted**: every gig and every provider text is embedded, compared with the embedding of each of the 2,088 taxonomy tag
titles, and given its 30 closest tags (after a per-tag correction that stops generic tags from winning everywhere). A provider's
document is then its set of tags, a gig's query is its set of tags, and providers are ranked by a BM25 or weighted-cosine score
over tag IDs. The resulting list is fused with refined BM25 and dense by reciprocal rank fusion (RRF), and its score and rank can
also be fed to the Stage-2 ranker as two extra features. It is **off by default** everywhere.

## Data flow

```
taxonomy_greygigz/ (SQL dumps)      hirers.json            providers.json
  2,088 tag titles                     |                        |
        |                      hirer_text(h)             provider_text(p)
        |                       title + description      about title/description + services
        |                       + notes                  title/description + experience
        v                              v                        v
  embed titles twice          embed as QUERY            embed as PASSAGE
  (as passage / as query)     (gig side)                (provider side)
        |                              |                        |
        |        cosine(gig, tag-as-passage)    cosine(provider, tag-as-query)
        |                              |                        |
        |            subtract each tag's mean over all gigs / all providers   (centring)
        |                       (qwen3: also divide by one pooled sd)
        |                              |                        |
        |                  keep best 100 per text, stored in tags_*.json
        |                              |                        |
        |                  channel uses the best 30 per text
        |                              |                        |
        |                      gig = query               provider = document
        |                              \______ BM25 over tag IDs  or  weighted cosine ______/
        |                                               |
        |                          ranked list of providers with a score > 0   -> RRF with BM25 + dense
        |                                                                      -> tag_score, tag_rank for Stage 2
```

## Step by step

### 1. The texts and the taxonomy
- A gig is `hirer_text(h)` in `corpus.py`: `hire_title`, `hire_description` and `hire_description_additional_notes`, joined with
  spaces. Budget, seniority and dates are not part of it.
- A provider is `provider_text(p)`: `about_title`, `about_description`, `services_offered_title`,
  `services_offered_description` and `relevant_experience`, joined with spaces (credentials excluded on purpose).
- The taxonomy is the SkillsFuture / GreyGigz export parsed by `greygigz.py`: 2,088 tags, each with an ID and a title. Only the
  **titles** are embedded; the role-tag links of the taxonomy are not used by the tagger. Tags are processed in sorted ID order.

### 2. Embedding (`tag_corpus.py`, `tag_encoders.py`)
Both sides are embedded with the same model, in the asymmetric direction the model was trained for. Gig and provider vectors are
in different spaces from each other's tag vectors, so each side gets its own tag matrix:

| Side | Text role | Tag-title role it is compared with |
|---|---|---|
| gig | query (with the model's query prefix) | tag titles embedded as **passages** |
| provider | passage | tag titles embedded as **queries** (with the query prefix) |

- **mxbai** (the original tagger and the dense channel's model): `mixedbread-ai/mxbai-embed-large-v1`, 1024 dims, query prefix
  `Represent this sentence for searching relevant passages: `, no passage prefix. Vectors are cached under `pipeline/cache/`
  (git-ignored) with the same names the dense channel's pooling script uses, so tagging reuses them.
- **qwen3-0.6b** (`--encoder qwen3-0.6b`): `Qwen/Qwen3-Embedding-0.6B`, query prefix
  `Instruct: Given a text, retrieve the skills it describes\nQuery: `, no passage prefix. This instruction was written for tagging;
  it is not the instruction a gig-to-provider dense channel would use. Vectors are cached under `cache/tagenc/qwen3-0.6b/`, keyed
  by a hash of each text, and the vectors used for the committed tag files came from the GPU (a CPU rebuild differs by about
  1e-3 in cosine).
- All cosines are computed on unit-normalised vectors, so a score is `u.v / (|u||v|)`: one matrix product of texts against tags.

### 3. The per-tag correction ("hubness") and picking the top tags
Raw cosine favours generic tags ("Financial Analysis") that sit near the middle of a whole domain, so the same tags win for most
texts. The fix is to subtract each tag's **mean cosine over the whole corpus of that side** (all 1,023 gigs for the gig side, all
2,165 providers for the provider side):

```
score(text, tag) = cos(text, tag) - mean_tag          # centring (variants hc and hc-<encoder>)
score(text, tag) = (cos - mean_tag) / sd_tag          # z-scoring (variant hz; not used by the current lead)
```

- **mxbai `hc`:** centred only. Scores are cosine differences; the typical per-tag sd is about 0.05, which is why the weighted
  scorer's threshold for this variant is 0.05.
- **qwen3 (`hc-qwen3-0.6b`):** centred, then divided by **one** pooled sd for the whole matrix (0.0559 on the gig side, 0.0580 on
  the provider side), the same for every tag. Scores are therefore in units of one standard deviation, and the weighted scorer's
  threshold is 1.0. For the gig side the 30th best tag of the example gig below scores 2.53, the 100th 1.99.
- Each text keeps its **100** best tags (`DEFAULT_TOP_M_HUBNESS`), best first, with the corrected score (4 decimals). The raw
  files (no correction) keep 30 tags with the plain cosine.
- Ties keep tag order (stable sort), so the output is deterministic.
- The files also store `tag_ids`, `mean` and `sd` for each side, which is what lets a new gig be corrected later (step 8).

Files written to `pipeline/data_sat/`:

| Variant name | Files | Correction | Weighted-scorer tau |
|---|---|---|---|
| `""` (raw) | `tags_hirers.json`, `tags_providers.json` | none, cosine, 30 tags | not available |
| `hc` | `tags_*_hc.json` | centring (mxbai) | 0.05 |
| `hz` | `tags_*_hz.json` | z-score (mxbai) | 1.0 |
| `hc-qwen3-0.6b` | `tags_*_hc-qwen3-0.6b.json` | centring + pooled sd (qwen3) | 1.0 |

Each file is `{"model", "top_m", "tags": {id: [[tag_id, score], ...]}}` plus, for corrected variants, `"score"`, `"tag_ids"`,
`"mean"`, `"sd"`.

### 4. Building the channel (`tag_channel.py`)
`TagChannel(data_dir, variant=..., scorer=...)` reads the two files of a variant and keeps each text's best **30** tags (30 for
providers and 30 for gigs; `DEFAULT_TOP_M_*`, chosen on the dev gigs). Providers are the documents; a gig's tags are the query.
Gig and provider tags are matched by **ID**, never by name or by embedding at this point.

### 5. Scoring: two scorers over the same tags
**`scorer="bm25"` (default; `TagBM25` in `retrieval_tagbm25.py`).** Only *which* tags a text carries matters, not their scores.
With `B[d, t] = 1` if provider `d` carries tag `t`, `N` providers and `df_t` providers carrying `t`:

```
idf_t = ln(1 + (N - df_t + 0.5) / (df_t + 0.5))              never negative
c_d   = (k1 + 1) / (1 + k1 * (1 - b + b * |d| / avgdl))     |d| = tags on provider d (30 for every provider here)
score(d) = c_d * sum over t in the gig's tags of idf_t * B[d, t]
```

with `k1 = 1.2`, `b = 0.75` and binary term frequency. Because every provider has exactly 30 tags, `c_d` is the same constant for
all of them (`avgdl` is 30.0), so the score is the **IDF-weighted overlap** of the two 30-tag sets. A tag carried by many providers
counts for little (the most common tag is on 300 of 2,165 providers; `idf` falls accordingly), and a rare one counts for a lot.

**`scorer="wcos"` (weighted cosine; `TagWeightedCosine`).** Keeps the strength of each tag. A text is a sparse vector over tag
IDs with weight `max(0, score - tau)` for each of its 30 tags, and providers are ranked by the **cosine** of the gig vector and
the provider vector. There is no IDF and no length term: the correction already down-weights generic tags and cosine normalises
length. Needs a corrected variant (it refuses `raw`).

In both cases the list returned is **every provider with a score above 0**, best first, ties in provider order (stable), with no
cut-off unless `max_returned` is set. On the qwen3 files this is about 480 of the 2,165 providers per gig (131 to 981 over the
first 300 gigs): the channel has no opinion about the rest, and they get no rank from it.

### 6. Fusion and the Stage-2 ranker
- **RRF** (`retrieval_rrf.rrf_fuse_n`): `score(d) = sum over channels of w / (60 + rank_in_channel(d))`, over the channels that
  returned `d`. The pool is `[refined BM25, dense, tag]` with weights 1, 1, 1 and `k = 60`. A provider the tag channel did not
  return simply gets nothing from it.
- **Candidates for Stage 2** (`features.py --tag-channel --tag-variant V`): the top 50 of that three-way RRF per gig. Each row
  gets `tag_score` and `tag_rank`. A provider the channel did not return gets **0 / 0** for `raw`, and an **empty value (read as
  NaN)** for every other variant, so LambdaMART can tell "no evidence" from "ranked first". The other features are
  `bm25_score/rank`, `dense_cosine/rank`, `rrf_score/rank`, `budget_fit`, `seniority_fit` and `avail_immediacy`
  (this one depends on today's date, so build every compared CSV on the same day).
- `--tag-variant`: `raw`, `hc`, `hcw` (hc + weighted), `hcq` (qwen3 + BM25), `hcqw` (qwen3 + weighted). Output files are
  `candidates_top50_tag_<variant>.csv` (`_tag.csv` for raw).
- Only `features.py` and the `eval_tag_*.py` scripts take a variant. `run_pipeline.py --tag-channel` and
  `build_judging_pools_sat.py --tag-channel` build `TagChannel(DATA_DIR)` and so always use the **raw** variant.

### 7. Building the tag files
```
python pipeline/tag_corpus.py --data-dir data_sat                       # raw, mxbai, 30 tags
python pipeline/tag_corpus.py --data-dir data_sat --hubness center      # hc, mxbai
python pipeline/tag_corpus.py --data-dir data_sat --encoder qwen3-0.6b  # hc-qwen3-0.6b (needs the tagenc cache; use the same device for a whole comparison)
```
The per-tag means are taken over the gigs and providers **being tagged** (here the whole dataset, including the test gigs), so the
correction is fitted on unlabelled text of the same kind, with no labels involved. In the grader-free check, a mean taken from provider texts and applied to role descriptions gave
no gain, so the mean has to come from the same kind of text it corrects (`TAG_CHANNEL.md` section 12).

### 8. Ranking a gig that is not in the files
`TagChannel.rank_text(gig_vec, tag_vecs, tag_ids)` takes the gig embedded **as a query**, the taxonomy titles embedded **as
passages** and their IDs (sorted, as the files were built), re-applies the stored per-tag mean (and sd) of the gig side, takes the
top 30 and scores them against the existing provider index. For a gig that is in the files it equals `rank()` up to the 4-decimal
rounding of stored scores. It needs the gig embedding from the same encoder and device family as the files; no API call.

## A worked example (qwen3 files, gig 130)

Gig: *Create Investment Prospectus Template for Wind Projects*. Best gig-side tags (centred, in sd units): Business Proposal
Writing 5.08, Transaction Documentation for Prospectus Development 4.67, Proposal Writing 4.45, Proposal Writing Development 4.27,
Contract Drafting 3.77, Solar Photovoltaic Project Financing and Risk Analysis 3.59, Project Plan 3.32, Deal Structuring 3.09.
Note what is and is not captured: document and finance tags are found, but the **energy side of the gig is not represented**: the
taxonomy has no wind tag, and the only tag with "energy" in its title that reaches this gig's top 100 is Energy Product Advisory,
at rank 56, outside the 30 the channel uses.

Providers ranked first by the channel:

| Scorer | 1st | 2nd | 3rd |
|---|---|---|---|
| BM25 | Healthcare Strategy and Advisory Services (26.3) | Construction Legal Advisory Services (25.7) | Performance Improvement and Strategy Advisory (22.8) |
| weighted cosine | Private Equity Investment Advisory (0.236) | Construction Legal Advisory Services (0.192) | M&A Advisory and Fairness Opinions (0.186) |

The BM25 list is a weak result for this gig: the winners share many generic document and proposal tags with the gig and few
energy ones. This is the channel's known failure shape (a generic-tag overlap), which the correction reduces and does not remove.
Fusion with BM25 and dense, and Stage 2, are what turn such lists into the measured gains.

## What to change where

| To change | Edit | Then rebuild |
|---|---|---|
| number of tags used per text | `DEFAULT_TOP_M_*` in `tag_channel.py` (the files hold 100) | nothing; pools and labels must use the same value |
| BM25 `k1`, `b` | `TagChannel(k1=, b=)` | nothing (with 30 tags per provider they barely matter) |
| weighted-scorer threshold | `VARIANT_TAU` or `TagChannel(tau=)` | nothing |
| encoder | add an `EncoderSpec` in `tag_encoders.py` | `tag_corpus.py --encoder NAME`, then the Stage-2 CSVs |
| fusion weights | `weights=` in the `rrf_fuse_n` call | the candidate CSVs |

## Properties to keep in mind
- **The tags are predictions.** The channel inherits every tagging error; gold-tag precision is low (precision@5 about 0.29 for
  mxbai) and the bench on role descriptions does not predict the fused result (`TAG_CHANNEL.md` section 3, section 13).
- **It is not independent of dense.** The tagger and the dense channel use closely related embeddings; the channel adds value
  through the taxonomy vocabulary and the correction, not through an independent signal.
- **Gig-side hubness is a trade-off for qwen3:** 12.6% of gig-side slots in the top-50 tags against 10.1% for mxbai (provider
  side 14.4% against 14.7%).
- **Nothing here uses labels.** Tagging, centring and scoring are unsupervised; only the Stage-2 ranker is trained, on graded pairs.
