# Explicit-tag channel (front-end schema)

The front-end-aligned samples (`data_frontend_sample/hirers_sample.json`, `providers_sample.json`) carry `search_tags`
("Specialisation (Category)") that users picked from the back-end taxonomy. For those records the predicted-tag channel
(`TAG_CHANNEL.md`, `tag_channel.py`) is not needed. Nothing here is wired into `run_pipeline.py` or `features.py` yet, and
nothing measured in `TAG_CHANNEL.md` transfers to it: those results are for predicted tags on `data_sat`.

| File | Role |
|---|---|
| `frontend_schema.py` | `adapt_hirer` / `adapt_provider`: new schema to the internal record shape, string IDs, metadata and credentials dropped, "Not focused on ..." sentences removed from provider text, free-text `rate` / `availability` parsed |
| `explicit_tag_channel.py` | `ExplicitTagChannel`: IDF overlap on composite tags plus a category back-off, optional tie-break, per-pair overlap features, predicted-tag fallback for hirers with no tags |
| `tests/test_explicit_tag_channel.py` | hand-worked fixture and the two sample files |

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
