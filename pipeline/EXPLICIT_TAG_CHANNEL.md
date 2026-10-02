# Explicit-tag channel (front-end schema)

The front-end-aligned samples (`data_frontend_sample/hirers_sample.json`, `providers_sample.json`) carry `search_tags`
("Specialisation (Category)") that users picked from the back-end taxonomy. For those records the predicted-tag channel
(`TAG_CHANNEL.md`, `tag_channel.py`) is not needed. It is wired into `run_pipeline.py` behind `--explicit-tag-channel` (see
below) but not into `features.py`, and nothing measured in `TAG_CHANNEL.md` transfers to it: those results are for predicted
tags on `data_sat`.

| File | Role |
|---|---|
| `frontend_schema.py` | `adapt_hirer` / `adapt_provider`: new schema to the internal record shape, string IDs, metadata and credentials dropped, "Not focused on ..." sentences removed from provider text, free-text `rate` / `availability` parsed |
| `taxonomy_sf.py`, `taxonomy_sf/` | Loader for the taxonomy as the front end keys it (Category = Sector, Specialisation = Track, Skills = TSC): `resolve_tag`, `track_tscs`, `track_similarity`, `bridge_to_greygigz` |
| `explicit_tag_channel.py` | `ExplicitTagChannel`: IDF overlap on composite tags plus a category back-off, optional tie-break, per-pair overlap features, predicted-tag fallback for hirers with no tags |
| `tests/test_explicit_tag_channel.py` | hand-worked fixture and the two sample files |

## Running it

    python pipeline/run_pipeline.py --data-dir <dataset> --explicit-tag-channel [--track-sim-weight 0.5]

`<dataset>` is a folder under `pipeline/` holding `hirers.json` and `providers.json`. Front-end records (`gig_title`,
`about_headline`, ...) are converted on load by `frontend_schema.adapt_records`; pipeline-schema records pass through
unchanged. The stage takes the refined BM25 and full-dimension dense lists the script has already computed, ranks every hirer
with the dense cosine as the tie-break and writes, next to the other results:

| File | Content |
|---|---|
| `explicit_tags.json` | hire id -> provider ids from the explicit channel, best first |
| `rrf3_explicit_k60.json` | hire id -> provider ids from RRF(BM25, dense, explicit), k = 60 |
| `explicit_tags_report.json` | hirers/providers without tags, records with unknown tags, hirers whose explicit list is empty |

A tag outside the taxonomy stops the run. Without the flag nothing is read or written for this stage. The logic is
`explicit_tag_channel.run_explicit_channel`, a pure function that the tests exercise with stub scores
(`tests/test_run_explicit_channel.py`); `run_pipeline.py` itself needs the embedding libraries. On the 30 x 30 sample, 10 hirers
share no exact tag with any provider; with `--track-sim-weight 0.5` only H030 (a Legal Services lease review) has an empty
list, because no sample provider has a Legal Services tag or a similar track.

## Using the taxonomy

    tax = taxonomy_sf.load_taxonomy()
    ch = ExplicitTagChannel(hirers, providers, taxonomy=tax, track_sim_weight=0.5, strict=True)
    ch.report()        # hirers/providers without tags, records with tags not in the taxonomy

With `taxonomy=`, a tag resolves to a track ID (all 109 tag occurrences in the samples do), unknown tags are listed in
`unknown_tags` (`strict=True` raises) and matching is by ID. A track is a set of TSCs (3 to 112, via its roles), so two tracks
have a graded similarity (Jaccard of the TSC sets): Financial Accounting / Management Accounting 0.54, Business Valuation / M&A
0.35, Financial Forensics / Cyber Security 0.02, Operations in Healthcare / in Financial Services 0.04. The optional back-off
(`track_sim_weight`, 0 to 1, off by default) gives a hirer track the provider lacks a credit from that similarity
(`min_track_sim` = 0.1 zeroes the near-unrelated pairs). It is capped at the smallest exact-match score, so an exact tag
always outranks a back-off. Weight and threshold are untuned.

**The integer IDs here are not the IDs in `taxonomy_greygigz/` or in `data_sat/tags_*.json`.** The content is identical, the
numbering is not. `bridge_to_greygigz()` translates by TSC title and by (track, sector) name and refuses a mismatch.

## Facts that drive the design
- `search_tags` is the authoritative pairing. `category` and `specialisation` are not parallel lists; never zip them.
- 1-3 tags per record, so scores take few distinct values and ties are common. Pass `tiebreak=` (e.g. dense cosine) to
  `rank`, or use `features()` in Stage 2 instead of an RRF list.
- A specialisation name is only meaningful inside its category ("Operations" exists under Financial Services and Healthcare).
- 10 of 30 sample hirers share no exact tag with any of the 30 sample providers, hence the category back-off
  (`category_weight`, default 0.25, untuned).

## Assumptions to confirm
- `HOURS_PER_DAY = 8`, `HOURS_PER_MONTH = 160` for turning day rates and retainers into `rate_per_hour`. A retainer is not
  comparable to a day rate (P008: S$8,000 per month becomes S$50 per hour).
- The first S$ amount with a unit is the rate; fixed fees alone give `None`.
- `b = 0` (plain IDF overlap) and `category_weight = 0.25` are defaults, not tuned.
- Hirers in the samples carry no budget, seniority or start date, so `budget_fit`, `seniority_fit` and the date part of
  `avail_immediacy` have no hirer-side input yet.
