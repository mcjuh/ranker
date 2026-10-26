# Recommendation API — aligned I/O

Supersedes `api_sample_final.json`, which was generated from `pilot_tax.json` on a
30-gig corpus and describes a `signals` block whose fields no longer exist.

Regenerate a live example at any time:

```bash
python pipeline/api_io_example.py --hire-id 1 --top-k 10
```

It scores with the saved model, so the numbers in the output are real.

---

## Tiers

Two tiers, one contract. **The request and response shapes below are identical for
both** — a client switches tiers by changing one config value, never its parsing code.

| tier | what runs | NDCG@10 | P@5 | MRR | servable |
|---|---|---|---|---|---|
| `lightweight` | BM25 + dense → RRF → LambdaMART (17 features) | **0.8982** | 0.8991 | 0.9774 | yes |
| `pro` | same retrieval → 2 cross-encoders → set-wise transformer | **0.9508** | 0.9541 | 0.9940 | **no** |

Configs live in `configs/lightweight.json` and `configs/pro.json`; `pipeline/tier.py
--check` asserts they still agree on the response shape and feature vector.

```bash
python pipeline/serve.py --hire-id 1 --tier lightweight
python pipeline/serve.py --hire-id 1 --tier pro   # exits with the blocking reason
```

**`pro` has no serving artifact yet.** `listwise_ranker_v2.py` emits out-of-fold
ranked lists only (it never calls `torch.save`), and its cross-encoder inputs are
precomputed OOF CSVs rather than a live scorer. The 0.9508 is an offline evaluation,
not a service. Serving it needs: fit on all queries + `torch.save`, a live CE pass
over the candidate pool, and a response builder — the last one already exists in
`serve.py` and can be reused verbatim.

---

## Request

```json
{
  "hire_id": 1,
  "hire_title": "FinOps Maturity Assessment for Tech Team",
  "hire_description": "We are the IT team at a mid-sized ...",
  "industry": "Infocomm Technology",
  "taxonomy": [
    { "name": "Infocomm Technology", "group": null,
      "specialisations": [
        { "name": "Data and Artificial Intelligence",
          "search_tag": "Data and Artificial Intelligence (Infocomm Technology)",
          "skills": ["Cloud Computing", "Data Analytics", "Business Performance Management"] } ] } ],
  "top_k": 10
}
```

Field names follow the corpus (`data/data_set_v5_tax/hirers.json`) exactly, so a
request can be built by passing a corpus record through unchanged. `hire_id` is an
**integer**.

> **Do not use the old contract's `title` / `description` names.** They do not
> correspond to anything in the data; the corpus uses `hire_title` /
> `hire_description`, and the client would have to rename on every call.

## Response

```json
{
  "results": [
    {
      "rank": 1,
      "provider_id": 2720,
      "score": 100.0,
      "raw_margin": 4.292988,
      "provider": { "display_title": "...", "industry": "...", "bio_excerpt": "..." },
      "signals": {
        "retrieval":    { "bm25_score": 4.455356, "bm25_rank": 1.0,
                          "dense_cosine": 2.359133, "dense_rank": 1.0,
                          "rrf_score": 2.811146, "rrf_rank": 1.0 },
        "business_fit": { "budget_fit": 0.0, "seniority_fit": 0.0, "avail_immediacy": 0.0 },
        "taxonomy":     { "tag_level": 1.0, "has_tags_gig": 1.0, "has_tags_prov": 1.0,
                          "cat_any": 1.0, "spec_cov": 0.0, "skill_cov": 0.0,
                          "skill_idf_cov": 0.0, "skill_soft": 0.6344 }
      }
    }
  ],
  "meta": { "...": "see the script" }
}
```

`signals` is the model's actual feature vector — all 17 columns, in declared order.
It is emitted so a wrong answer can be debugged without re-running the pipeline.

---

## Serving requirements (these are the work items)

1. **Per-query z-score normalisation is mandatory.** The model was trained on
   `SCORE_FEATURES` rescaled within each query
   (`bm25_score, dense_cosine, rrf_score, budget_fit, seniority_fit, avail_immediacy`).
   Serve the raw values and the model receives features on a different scale —
   no error, just worse rankings. Use `rerank_ltr.normalize_by_query(rows, keys, "zscore")`;
   it is population std (divide by n, not n−1).

2. **Feature column order is positional.** The saved model has
   `feature_names: []`, so assembly order is the only thing keeping it correct:
   `DEFAULT_FEATURES` (9) then `TAG_FEATURES` (8). Do not reorder, do not omit the
   dead columns — they must be present as `0.0`.

3. **`provider_id` is a corpus row index (1–3024), not a platform id.** The
   platform identifier is in `data/data_set_v5_tax/id_map.json`. The service must
   join before returning anything user-facing.

4. **The score is not a probability.** `score` is per-query min-max of the ranker
   margin onto 0–100, so rank 1 is always exactly 100.0 and the spacing depends on
   the pool. `raw_margin` is the model output; it can be negative (ranks 5–10 in
   the sample are). Do not present `score` to users as a confidence.

## On "scoring confidence"

There is no calibrated confidence today, and the honest label for `score` is
"ranking score". If a user-facing confidence is required, the work is:

- take the out-of-fold predictions (`results_data/data_set_v5_tax/v5tax_tags.json`
  ranks + the model's margins),
- fit isotonic regression from margin → P(grade ≥ 2),
- serve `confidence = calibrator.predict(margin)` alongside `score`.

Until that exists, label the field as a score, not a probability.

## Known-dead fields

`seniority_fit` is emitted as `0.5` at train time and `0.0` at serve time on every
request, and will stay that way: `providers.json` has **no `seniority` field at all**
(0/3,024 — the key does not exist, and neither does `years_experience`). The gig side
has `seniority_needed` (100% populated: senior 75%, mid 21%, expert 4%), but a
*fit* needs both halves, and a within-query ranker cannot use the gig value alone —
it is constant across all 50 candidates. Do not build UI that depends on it.

`budget_fit` and `avail_immediacy` **are now supported by the data** (`budget_lo` /
`budget_hi` / `rate_per_hour` and `available_from` / `start_by`, all 100%), but the
shipped model was trained with them zeroed, so they are still emitted as `0.0`:
feeding live values to a model that never saw them is out-of-distribution. Retraining
on live values scored 0.8962 vs 0.8962 — identical — so the zeros stayed. A future
model can turn them on by setting `dead_features` aside and retraining.

## Out-of-fold vs serving

`results_data/data_set_v5_tax/v5tax_tags.json` holds **out-of-fold** rankings — each
gig scored by a fold model that never saw it — and is what the reported NDCG@10
0.8955 is computed from. `models_data/data_set_v5_tax/v5tax_tags_xgb.json` is the
all-data serving model. They will not agree on ordering, and that is correct; do not
"fix" the discrepancy.
