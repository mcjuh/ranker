# Third-channel status, shelved work and what can be borrowed

Written when the front-end-aligned samples arrived (`data_frontend_sample/`), so that if the implementation changes
again nobody has to rediscover what was tried. Nothing was deleted or moved: the predicted-tag code is still the
fallback for records without tags, and eight files import it (`explicit_tag_channel.py`, `tag_channel.py`,
`tag_corpus.py`, `eval_tag_channel.py`, `eval_tag_encoder.py`, `claude_audit.py`, and their tests), so relocating it would
break imports and the tests for no gain. If a file is ever retired, move it to `pipeline/deprecated/` in one commit that
fixes the imports, and update the table below.

## Status by component

| Component | Status | Why | Where to look |
|---|---|---|---|
| Explicit-tag channel (`explicit_tag_channel.py`, `frontend_schema.py`) | **Active, not yet wired into `run_pipeline.py` / `features.py`** | Users pick `search_tags` from the back-end taxonomy; no prediction needed | `EXPLICIT_TAG_CHANNEL.md` |
| Predicted-tag channel (`tag_channel.py`, `tag_corpus.py`, `tag_encoders.py`) | **Fallback**, off by default | Still needed for hirers/providers with no tags and for `data_sat`, which has none | `TAG_CHANNEL.md`, `TAG_CHANNEL_HOW_IT_WORKS.md` |
| `retrieval_tagbm25.py` (`TagBM25`) | **Shared** | Both channels score with it | `tests/test_tag_bm25.py` |
| `retrieval_rrf.rrf_fuse_n` | **Shared** | N-way fusion for either tag channel | `tests/test_rrf_n.py` |
| Hubness correction, qwen3 encoder, weighted-cosine scorer | **Shelved for tagged records** (kept for the fallback) | They repair the tagger's precision; explicit tags have no tagger | `TAG_CHANNEL.md` s11-s13 |
| `rank_text` (rank a new gig from its embedding) | **Shelved for tagged records** | A new gig arrives with its tags; needed only for the fallback | `tag_channel.py` |
| Role/track experiments (LSA, sum-of-levels, track-first funnel, soft kernels) | **Closed negative results** | Did not beat plain BM25 on tags | `TAG_CHANNEL.md` s3, s8, s11 |
| Graders and audit (`labeller.py`, `claude_audit.py`, `audit/`) | **Active** | Needed to grade any new channel; calibrated on `data_sat` text, so recalibrate on the new schema | `TAG_CHANNEL.md` s6, `audit/RESULTS.md` |
| `data_sat/` structured fields (`budget_lo/hi`, `seniority_needed`, `start_by`, `available_from`, `capacity`) and the three Stage-2 fit features | **Active for `data_sat`; inputs missing in the new schema** | Front-end payload keeps rate and availability as free text; hirers carry no budget, seniority or start date | `features.py` |

## What the old work established (still true, borrow freely)

- A tag channel is only as good as the tags. On predicted tags it was weaker than dense alone (NDCG@10 0.610 vs 0.717)
  and added little to RRF until hubness was corrected (+0.021) and the encoder swapped to qwen3 (+0.033 more in Stage 2).
- Evaluate with one grader and fully judged top-10s. Mixing graders made the channel look better than it was
  (`TAG_CHANNEL.md` s4c, s5). Never mix prompts in one comparison.
- Do not train the Stage-2 ranker on labels whose coverage depends on the channel under test (`TAG_CHANNEL.md` s7).
- A hard industry gate loses good providers (relevance 0.446 same industry, 0.326 different, 0.426 cross-industry).
- Per-gig overlap features for the ranker were planned but never built (`TAG_CHANNEL.md` s11); `ExplicitTagChannel.features`
  now provides a version of them.

## What did not transfer, and why

- All absolute and paired numbers in `TAG_CHANNEL.md`: measured on predicted tags over 1,023 gigs x 2,165 providers.
- The tag-count and `b` tuning (30 tags, `b = 0.75`): `b` was a constant only because every provider had exactly 30 tags.
  With 1-3 explicit tags it is not, and the explicit channel defaults to `b = 0`.
- Gold-tag precision numbers: they describe the tagger, which explicit tags bypass.

## Reviving or borrowing

- To run the predicted-tag channel as before: `--tag-channel` (default off); files and commands in
  `TAG_CHANNEL_HOW_IT_WORKS.md`.
- To use predicted tags as the fallback for a hirer with no tags: `ExplicitTagChannel(..., fallback=TagChannel(...))`.
  The fallback is keyed by the id `TagChannel.rank` expects (int hire ids today), so a string-ID dataset needs its own
  tag files built first.
- To compare the two on the same hirers once a mapping exists from `"Specialisation (Category)"` strings to GreyGigz tag
  IDs: check how often the tagger's top-30 contains a user's picks, then run the same-grader protocol
  (`eval_tag_samegrader.py`).

## Open items

- Mapping from the back-end taxonomy list to the 2,088 GreyGigz tag IDs (needs the back-end file).
- Wire the explicit channel into `run_pipeline.py` and `features.py` behind a flag, with the Stage-2 fit features
  computed from the parsed free text (`frontend_schema.py`) or dropped.
- Full front-end-aligned corpus and new grades before any quality claim.
