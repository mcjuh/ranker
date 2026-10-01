# Session notes 2: tag-encoder bench and the grader prompt

Written at the end of a second working session with Claude (Claude Code) on branch `tag-encoder-test`, continuing
`SESSION_NOTES.md` (tag channel review, LLM usage, grader audit). It records what was done, what was found, what is parked
and what is left. Numbers come from files in the repo (`results_tag/`, `data_sat/`); anything not measured is marked as a
hypothesis. **Nothing from this session is committed.**

## 1. The question and the plan

The tag channel is a matmul: `S = unit(text) @ unit(tag_vecs).T` over 2,088 skill tags (the TSC/CCS `tags` table, not
tracks or roles), top-30 per text, then BM25 over tag IDs. Everything inherits the encoder's quality, and no tag-encoder
experiment had been run (earlier work only post-processed the same mxbai scores: centring, weights, scorers). The plan
(approved) was: build a bench, then try (1) other encoders and ensembles, (2) richer tag vectors from the taxonomy,
(3) a supervised head on frozen vectors, (4) a contrastive fine-tune, (5) optional LLM-written tag descriptions. Baseline to
beat throughout: **mxbai + per-tag centring** (the current best, `tag_hc`).

Where the channel stood at the start (TAG_CHANNEL.md s11-12, one grader, 271 test gigs): RRF + `tag_hc` over RRF(bm25, dense)
NDCG@10 +0.021, P@10 +0.025, shrinking to about +0.010..+0.015 P@10 once the lenient grader is calibrated away; Stage-2
NDCG@10 +0.013; alone NDCG@10 0.633 (`tag_hc`) / 0.657 (`tag_hcw`) against dense 0.712.

## 2. What was built

| File | What |
|---|---|
| `tag_encoders.py` | registry of 10 encoders (mxbai, bge-large, e5-large, gte-large, arctic-l, mpnet, minilm-l12, all-roberta, nomic, qwen3-0.6b) with query/passage prefixes and a text-keyed embedding cache under `pipeline/cache/tagenc/<encoder>/` (git-ignored) |
| `eval_tag_encoder.py` | the bench: any encoder, recipe (`enc`, `enc:proto0.7`, `enc:knn0.05`, `enc:mix1.0_0.05`) or ensemble (`a+b`) on benches A, B, C below |
| `tests/test_eval_tag_encoder.py` | 18 tests of the pure-numpy parts, incl. a leakage-style check of the kNN and prototype recipes |
| `results_tag/encoder_bench_dev.json` | the all-encoder comparison on the dev half |
| `labeller.py` (edited) | `--prompt orig` and `--prompt cont` variants; default (`repro`) unchanged; outputs get `_orig` / `_cont` suffixes |
| `tests/test_labeller.py` (edited) | 12 tests added for the two variants (183 tests pass in total) |
| `data_sat/calibration_orig.jsonl`, `calibration_cont.jsonl`, `results_tag/calibration_orig.json`, `calibration_cont.json` | the two new calibrations |
| `kiv/rubric_0_3_ordinal_orig_text.md` | the parked 0-3 prompt (section 5) |
| `audit/prompt_vs_claude.py`, `results_tag/prompt_vs_claude.json`, `data_sat/judging_pools_audit.json`, `judgments_audit_cont.jsonl`, `judgments_audit_orig.jsonl` | the 355-pair comparison of the prompts with the blind Claude grades (2026-10-02) |

**The three benches** (none calls the LLM grader):
- **A, gold tags.** Role name + description tagged and compared with the role's gold tags (precision/recall at m, role@k,
  track@k); the split is the RoleWorld dev/test halves of the 1,606 role equivalence classes (803 each). Upper bound: role
  text is cleaner than gig/provider text.
- **B, real gigs.** The dev (or test) gigs' **original-graded** pairs ranked by the cosine of the gig's and provider's weighted
  tag vectors (as `wcos`): AUC, NDCG@5, NDCG@10. Pool is small (about 21 pairs and 2 positives per gig), so AUC is the
  sensitive metric and CIs are wide.
- **C, diagnostics.** Hubness (top-50 tag slot share, tags never picked, most frequent tag) and complementarity (Spearman of
  the pair scores with dense; share of the top-50 not in the RRF top-50). Marginal recall against the original labels was
  tried and dropped: the original pool was built from the RRF top 20, so it is zero by construction.

Candidates are centred per tag and divided by their pooled sd, so encoders on different cosine scales are comparable and
the weighted-cosine threshold (1 sd, which is the tuned 0.05 for mxbai) means the same for all. An ensemble `a+b` averages
those unit-free matrices. Recipes that learn from roles (prototype, kNN) use only the other half's classes for bench A,
so an evaluated role never informs its own tags.

**The baseline reproduces** (mxbai, GPU embeddings): raw precision@5 0.285 (TAG_CHANNEL.md 0.289), top-50 slot shares 10.1%
(gigs) and 14.7% (providers) and max tag frequency 282, all matching s11. Embeddings were first computed on the local RTX 5070
(memory note: GPU setup is local-only and not part of the repo).

## 3. Encoder bench results (dev half, baseline = mxbai + centring; `*` = paired 95% CI excludes 0)

| Candidate | A: precision@5 | B: AUC | B: NDCG@10 | hubness slots gigs/providers | never picked (prov.) |
|---|---|---|---|---|---|
| mxbai (baseline) | 0.305 | 0.695 | 0.690 | 10.1% / 14.7% | 59 |
| bge-large | -0.004 | -0.008 | -0.021* | 9.1 / 13.4 | 39 |
| e5-large | -0.007 | +0.009 | -0.006 | 9.3 / 13.0 | 29 |
| gte-large | **+0.011\*** | +0.002 | +0.001 | 9.9 / 14.3 | 76 |
| arctic-l | +0.001 | +0.017 | -0.001 | 10.2 / 13.2 | 59 |
| **mpnet** | -0.002 | **+0.052\*** | **+0.029\*** | 10.3 / 15.9 | 131 |
| minilm-l12 | -0.022* | -0.026* | -0.020* | 9.2 / 14.5 | 78 |
| all-roberta | -0.026* | +0.012 | +0.011 | 10.0 / 15.3 | 112 |
| mxbai+gte-large | +0.010* | -0.003 | -0.005 | 10.0 / 14.5 | 67 |
| six-encoder ensemble | +0.016* | +0.021* | +0.007 | 10.1 / 14.9 | 91 |
| **mxbai+mpnet** | +0.013* | +0.033* | +0.021* | 10.4 / 16.1 | 115 |
| mxbai:proto0.7 (tag prototypes) | +0.089* | +0.021* | +0.009* | 10.6 / 15.5 | 55 |
| mxbai:proto0.7+mpnet:proto0.7 | +0.110* | +0.046* | +0.029* | 11.1 / 16.5 | 116 |
| mxbai:mix0.5_0.03 (title + 0.5 x kNN) | +0.366* | +0.034* | +0.016 | 15.6 / 19.7 | 171 |
| mxbai:knn0.05 (kNN alone) | +0.467* | **-0.041\*** | -0.040* | 31.0 / 33.3 | 1,046 |

(Columns A and B are differences from the baseline except the baseline row. Dense on the same dev pool: AUC 0.717,
NDCG@10 0.691.)

- **Swapping the encoder alone does not help on gold tags.** Only gte-large and ensembles are above mxbai there, by
  +0.01..+0.016 precision@5.
- **mpnet is the exception on real gigs** (AUC +0.052 [+0.026, +0.077]; NDCG@5 +0.044*), and it is not a family effect (MiniLM
  is worse, all-roberta is not significant). It held on the test half in the same original-grade bench: mpnet AUC +0.030
  [+0.006, +0.055], mxbai+mpnet AUC +0.033 [+0.016, +0.049], NDCG@10 +0.019*. On gold tags it is neutral (test precision@5
  -0.009, ns), and mxbai+mpnet is +0.004 (ns) on test A.
- **Tag prototypes** (title blended with the mean of the roles carrying the tag) lift gold-tag precision@5 by +0.09 and real
  gigs modestly (AUC +0.021).
- **The role-kNN tagger is a trap on bench A.** Precision@5 jumps to 0.77 because near-twin roles in the other class half share
  their tags; role@1 halves (0.126 to 0.058) and real-gig AUC falls. Bench A overstates any recipe that learns from roles;
  benches B and C are the check.

**Caveats that apply to all of it.**
- About 12 candidates were compared on the same dev gigs (winner's curse), and bench B's pool is small.
- **Hubness gate not met.** The plan said "no worse than 10.1% / 14.7%". `mpnet`, `mxbai+mpnet` and the prototype ensembles are
  slightly worse on the provider side (15.9-16.5%); this was reported, not relaxed.
- **The test half was looked at once** (original grades only, B and A, as a sanity check). The graded test-gig confirmation has
  not been run, but the test gigs are no longer untouched.
- None of this is a fused-ranking result. B ranks within a small pool, so it says nothing yet about recall or about RRF.

## 4. The grader work

The grader is `qwen3.8:27b` through SOCLaaS, scoring 0-3 (relevant = 2 or more) with the first generated token's log-probabilities.
Everything for the tag channel's extra pairs (12.6k) and the hubness and variants pools rests on the **reconstruction**
(`rubric_0_3.v2-repro`), which `audit/RESULTS.md` found lenient.

- **The original prompt text was supplied by its owner** (it is the term-aware rubric in `label.md`, not the one-liner first
  quoted; that one-liner is probably a wrapper). Compared with the reconstruction it adds anchors the reconstruction lacks:
  grade 2 = "an adjacent sub-focus", grade 1 = "an adjacent area or a transferable skill". The reconstruction's grade 2 is only
  "relevant but not perfect", which invites adjacent-domain providers to be graded 2; this matches the audit's finding that most
  disagreements were content-only, adjacent-domain.
- **Calibration on the same 300 pairs** (75 per original grade; sample is stratified, so these are not population rates):

| | Reconstruction | Original text (0-3) | Continuous 0-1 |
|---|---|---|---|
| Kappa on grade >= 2 | 0.733 | 0.86 | 0.90 (banded) |
| AUC for original >= 2 | 0.980 | 0.987 | 0.984 |
| Spearman with the original grade | 0.900 | 0.919 | 0.917 |
| Original 0-1 pairs promoted to >= 2 (of 150) | 40 | 8 | 13 |
| Original >= 2 pairs demoted below 2 (of 150) | 0 | 13 | 2 |
| Pairs graded >= 2 (original: 150) | 190 | 145 | 161 |

- **The reconstruction's problem was the cut, not the ordering.** All three rank the pairs almost identically (continuous vs
  original-text expected grade Spearman 0.978; AUC >= 2 0.980-0.987).
- **Continuous prompt (`--prompt cont`, `rubric_0_1.v2-cont`).** The user's 0-1 version with anchor bands, scored by
  greedy-decoding the number (not a log-probability read). Bands map to grades by cutting the gaps at 0.175 / 0.50 / 0.825;
  those cut points are ours. It used only 22 distinct values over 300 pairs, did not pile on round numbers (24% end in 0 or 5),
  and almost nothing fell in the band gaps (0.3%). Grade 3 is under-assigned (45 of 75 original 3s; 30 land in the 2 band).
  It has no `probs`, so the existing `--relevance p60/p80` sweeps cannot run on it.
- **Checked on the tag-surfaced pairs (2026-10-02, `audit/prompt_vs_claude.py`).** The 300 calibration pairs come from the
  original pools, so they could not show whether a prompt still over-credits the pairs **only the tag channel surfaces**. The 355
  audited pairs of rounds 2 and 3 were regraded with the continuous and the original-text prompt and compared with the blind
  Claude grades (one rater; strata were drawn by the reconstruction's grade, so rates are not population rates):

| On the 260 tag-surfaced pairs (Claude grades >= 2 on 52%) | Reconstruction | Original text | Continuous >= 0.5 |
|---|---|---|---|
| Pairs the prompt rates >= 2 | 220 | 92 | 104 |
| ...of which Claude also rates >= 2 | 133 (60%) | 79 (86%) | 88 (85%) |
| Claude >= 2 pairs the prompt misses | 2 | 56 | 47 |
| Kappa with Claude (binary) | 0.30 | 0.48 | **0.52** |
| AUC vs Claude >= 2 (ordering) | 0.859 | 0.865 | 0.859 |

  Over all 355 pairs: kappa 0.31 / 0.50 / **0.55** and AUC 0.843 / 0.879 / 0.881.
  - **The over-credit is fixed.** Of the reconstruction's 105 positives that Claude rejected, the original-text prompt still
    credits 17 (0.16 [0.10, 0.24]) and the continuous prompt at 0.5 credits 21 (0.20 [0.13, 0.29]); at 0.7 only 1 (0.01).
  - **The cost is that both are now stricter than Claude.** Of the reconstruction's 170 positives that Claude confirmed, the
    original-text prompt keeps 102 (60%) and the continuous prompt at 0.5 keeps 115 (68%), at 0.7 only 57. On the 80 tag-only
    positives Claude rates 56% >= 2, the new prompts 31-36%. This matches the earlier note that Claude is itself one rater with its
    own threshold; neither side is ground truth.
  - **The ordering is unchanged** (tag-surfaced AUC about 0.86 for all three), so the prompts differ in where the line is, not
    in who ranks first. Thresholds 0.5 and 0.6 behave the same; 0.7 is a conservative lower bound.
  - **Original pools:** both new prompts keep the original positives Claude confirmed (original text 28/30, continuous 29/30).
  - **Between the two new prompts:** continuous is marginally closer to Claude (kappa 0.55 vs 0.50, same AUC). KIV condition 2
    is not triggered; the ordinal prompt stays parked.

## 5. Parked (KIV): the 0-3 ordinal prompt

`kiv/rubric_0_3_ordinal_orig_text.md` parks the original-text 0-3 prompt (`--prompt orig`), with the calibration table and
the conditions for coming back to it: try it if the continuous prompt does not help significantly, meaning (1) its scores are
noisy or tied where it matters (graded NDCG tells no different story from binary, or flips with small changes), (2) it
over-credits tag-only pairs (agrees with the blind Claude audit grades no better than the ordinal prompt does), (3) the
probability-based robustness sweeps turn out to be the more useful check, (4) decode wobble matters. The code stays in
`labeller.py`; the reconstruction stays there too as the default, because every existing grade depends on it.

## 6. Assessment: does the continuous prompt change the plan?

**Unchanged.**
- The encoder work itself. Benches A (gold tags) and B (original grades) do not use the new prompt, so Stages 0-2 stand as
  reported. Stage 3 (supervised head on gold tags) and Stage 4 (contrastive fine-tune, only if Stage 3 transfers) are
  grader-free and still wait to be run; the Stage-2 LambdaMART trains on the original grades, also unchanged.
- The decision to do a graded confirmation on at most two finalists, prospectively, with a Claude blind adjudication of the
  pairs entering and leaving the top-10.

**Changed.**
1. **The confirmation cannot reuse the reconstruction's cached grades.** s4c of TAG_CHANNEL.md showed that mixing graders is what
   made the extended labels misleading, so with a new prompt **every pair in every compared top-10 must be regraded**, not just
   the new ones. Plan: about 1.5k new pairs per finalist (about an hour each) reusing cached grades. Now: the test pool is 8,938
   pairs (9,080 with Stage 2) plus the finalists' new pairs, **about 10-11k calls, roughly 3 hours at 1 call/s**, roughly three
   times the plan. In exchange the confounder that shrank every earlier gain is gone (the calibrated prompt over-credits
   about 5-9% of original 0-1 pairs rather than 27%), so the audit-based discount becomes much less necessary. A small Claude audit
   of the entering/leaving pairs is still worth doing as an independent check.
2. **The relevance sweep changes.** `--relevance p60/p80/p80+term` depend on `probs`, which continuous records do not have, and
   `eval_tag_variants.truth_from_records` would raise on them. The replacement is a **score threshold sweep** (for example >= 0.5,
   0.6, 0.7) plus the term-mismatch demotion. This is arguably cleaner, and "solid gain" becomes: RRF + new tag minus RRF + `tag_hc`
   has an NDCG@10 CI excluding 0 at thresholds 0.5 and 0.6/0.7 **and** under graded NDCG, while the channel alone closes part of
   the 0.055 gap to dense.
3. **Evaluation code needs changes before any regrade:** `eval_tag_samegrader.SCORE` maps a grade to 33/67/100 and
   `grades_from_records` reads only `grade`; they need to read `score` (gain = 100 x score for graded NDCG, `score >= threshold`
   for binary P/R/MRR). The convention proposed: **binary at 0.5 as the headline** (comparable with every earlier number) and
   graded NDCG as the robustness check.
4. **Order of work.** Since the regrade is the expensive step, finish candidate selection first (more dev work below), then
   spend it once on the final pool. Do the cheap tag-only check (item 1 below) before that.
5. **A bonus the new grader allows:** the dev gigs (6,498 ungraded pairs, about 1.8 h) can now be graded too, to give a clean
   dev/test selection and remove the winner's curse, if the extra time is wanted.

**Net:** the improvement plan stays; the evaluation of it gets more expensive and more trustworthy, and a handful of code
changes sit in front of it. The KIV prompt is the fallback if the continuous grader does not hold up on tag-only pairs.

## 7. Left to do

In order:
1. ~~Check the continuous prompt on tag-only pairs against the blind Claude grades.~~ **Done 2026-10-02** (section 4): it no longer
   over-credits (85% of its tag-surfaced positives are Claude-confirmed, against 60% for the reconstruction), but it is
   stricter than Claude and misses about 35% of what Claude confirms. Decision for the evaluation: headline at score >= 0.5,
   sensitivity at 0.6, and 0.7 as a conservative bound.
2. ~~**Settle the evaluation convention** and adapt `eval_tag_samegrader.py` / `eval_tag_variants.py` to read `score`.~~
   **Done 2026-10-02** (192 tests pass in `.venv`; 9 new in `test_eval_tag_variants.py`; not yet run on a real pool):
   - `eval_tag_variants.py --prompt cont` reads, seeds and writes only continuous grades (`judgments_variants_cont.jsonl`,
     seeded from `judgments_audit_cont.jsonl` / `calibration_cont.jsonl`, results to `variants_<split>_cont.json`); the
     reconstruction's cached grades are never seeded into it, and a file that mixes graders is refused.
   - Headline = the record's `grade`, which for the continuous prompt is its band, so relevant is exactly score >= 0.5 and
     the 33/67/100 NDCG scale stays comparable with earlier numbers. Sweep: `s60`, `s70`, `term`, `s60+term`, and `graded`
     (same binary P/R/MRR, NDCG on gain = 100 x score). `--relevance all` expands to what exists for the prompt; `p60/p80`
     on `cont` (no `probs`) and `s*` on the 0-3 prompts are refused, not silently ignored.
   - `eval_tag_samegrader.py` already worked on the band `grade`; it now reports the grader from the records and refuses
     mixed graders. Its `metrics_per_gig` takes an optional `gain` for graded NDCG.
   - Next command once the finalists exist: `eval_tag_variants.py pool --split test --prompt cont`, then
     `labeller.py pairs --prompt cont ... --out judgments_variants_cont.jsonl`, then `eval --prompt cont --relevance all`.
3. **More encoder work on dev: run 2026-10-02 (results below; hubness correction added and Stage 4 skipped, both on 2026-10-02).**
   Dev half, baseline mxbai + centring, original grades (bench B), files `results_tag/encoder_bench_dev_step3{a,b,c,d}.json`
   (`*` = paired 95% CI excludes 0; AUC / NDCG@10 differences; hubness = top-50 slot share gigs / providers):

   | Candidate | B: AUC | B: NDCG@10 | hubness g / p | never picked (prov.) |
   |---|---|---|---|---|
   | mxbai (baseline) | 0.695 | 0.690 | 10.1 / 14.7% | 59 |
   | **qwen3-0.6b** | +0.065* | +0.038* | 12.6 / 14.4 | 63 |
   | mxbai+mpnet+qwen3-0.6b | +0.056* | +0.036* | 11.2 / 15.7 | 128 |
   | mxbai:head1 / head10 (Stage 3) | +0.028* / +0.029* | +0.016* / +0.011* | 10.6 / 15.8; 10.2 / 15.2 | 67; 66 |
   | **qwen3-0.6b:head10** | **+0.081\*** | **+0.054\*** | 13.2 / 14.3 | 55 |
   | mxbai:head1+mpnet:head1+qwen3-0.6b:head1 | +0.072* | +0.050* | 12.6 / 16.4 | 131 |
   | (dense on the same pool) | 0.717 | 0.691 | | |

   - **qwen3-0.6b is the best single encoder on real gigs**, better than mpnet (+0.052). On gold tags (bench A) it is neutral
     (precision@5 -0.005, ns), like mpnet; the three-way ensemble is +0.020* on A and +0.056 AUC on B.
   - **Stage 3 (supervised residual head, `name:head<lam>`; `fit_tag_head` in `eval_tag_encoder.py`) transfers, modestly.**
     Trained on the other half's roles (listwise softmax loss, penalty toward the titles). Gold-tag precision@5 goes
     0.305 -> 0.66, which bench A overstates (near-twin roles, as with the kNN tagger). On real gigs it adds AUC +0.028 to mxbai
     and, paired against qwen3 alone, +0.016 [+0.003, +0.030] (NDCG@10 +0.015*); the penalty is smooth (qwen3 head3 +0.011 ns,
     head10 +0.016*, head30 +0.002 ns). It is smaller than choosing the encoder, and on top of the ensembles it adds little.
   - **Hubness gate (10.1% / 14.7%) is not met by the winners:** qwen3 raises the gig side (12.6%; with head10 13.2%) while
     the provider side improves (14.3-14.4%); mxbai+mpnet+qwen3 is 11.2 / 15.7. Reported, not relaxed. The decision (accept a
     gig-side rise, or add a correction) is the user's.
   - **nomic was not run:** its remote modeling code fails on the installed transformers (`NomicBertModel` has no
     `get_extended_attention_mask`); fixing it means patching downloaded code or pinning transformers in the local GPU venv
     (einops was installed there for it; nothing in the repo changed). qwen3-0.6b ran with the registry entry as is.
   - **Hubness correction added (2026-10-02, user's call) and Stage 4 skipped (user's call).** The existing fixes (centring,
     z-scoring) were already tried, so the new ones target the upper tail a hub tag keeps even when its mean is ordinary:
     `name~dsm<t>` = dual softmax, `s - t log mean exp(s/t)` with t in pooled-sd units of the centred matrix (t -> infinity is
     centring), and `name~csls<k>` = `s - r_tag/2`, r_tag the mean of the tag's k closest texts. They replace centring
     (`hub_correct` in `eval_tag_encoder.py`; a per-tag shift followed by centring would be cancelled, which a test pins), and
     `name~center` reproduces the old recipe exactly. Dev, bench B, vs mxbai (hubness g / p, gate 10.1 / 14.7%):

     | Candidate | AUC | NDCG@10 | hubness g / p | never picked |
     |---|---|---|---|---|
     | qwen3-0.6b:head10 (uncorrected) | 0.776 | 0.743 | 13.2 / 14.3 | 55 |
     | ...~dsm3 | 0.769 | 0.742 | 11.5 / 12.6 | 34 |
     | ...~dsm2 | 0.767 | 0.738 | 10.6 / 11.8 | 29 |
     | **...~dsm1.5** | **0.762** | **0.735** | **9.6 / 11.1** | 20 |
     | ...~dsm1 / ~dsm0.5 | 0.752 / 0.719 | 0.722 / 0.694 | 8.1 / 9.9; 6.7 / 9.3 | 4; 1 |
     | ...~csls10 | 0.763 | 0.734 | 13.5 / 19.1 | 98 |
     | mxbai (baseline) | 0.695 | 0.690 | 10.1 / 14.7 | 59 |

     - **dsm trades ranking quality for evenness smoothly; csls fails** (it makes the provider side worse, 19.1%).
     - **`qwen3-0.6b:head10~dsm1.5` meets the gate on both sides** and keeps most of the gain: AUC +0.067 and NDCG@10 +0.045
       over mxbai, at a paired cost against the uncorrected head of AUC -0.014 [-0.023, -0.005] and NDCG@10 -0.008
       [-0.016, -0.000]. `~dsm2` is the middle option (gigs 10.6%, a hair over the gate).
     - **Correcting an ensemble costs more** (mxbai+mpnet+qwen3 with dsm1.5: AUC 0.733), so it is not a candidate.
     - Caveats: t was chosen on the dev gigs (one more tuned degree of freedom); the weighted-cosine threshold (1 sd) was not
       re-tuned for the corrected scores; and the statistics are taken over the texts being tagged, like centring, so adopting it
       in the pipeline needs `tag_corpus.py` to store the per-tag shift (as it stores the means today).
   - **Stage 4 (contrastive fine-tune) skipped** (recommendation accepted): Stage 3 helps less than the encoder choice and a
     fine-tune would face the same near-twin leakage on bench A.
   - **Winner's curse is larger now:** about 25 candidates have been compared on the same 271 dev gigs. The test-half check
     is still pending for qwen3-0.6b (and for any head recipe).
   - Tests: 199 pass (2 new for the head, 5 for the corrections).
3b. **Course correction (user, 2026-10-02): too many layers, keep it explainable.** Head, dsm, csls, nomic, prototypes and the
   ensembles are parked: all were tuned on the same 271 dev gigs (about 30 candidates) and are hard to explain. The one lead kept
   is the plain swap `qwen3-0.6b` with the current per-tag centring. **Checked once on the test half** (original grades,
   `results_tag/encoder_bench_test_qwen3.json`): vs mxbai AUC +0.058 [+0.031, +0.085], NDCG@5 +0.041*, NDCG@10 +0.037*
   (dev: +0.065 / +0.043 / +0.038), better than mpnet (+0.030 / +0.023). Pair scores are less like dense (Spearman 0.395 vs 0.459)
   and more novel (top-50 not in RRF: 0.673 vs 0.626). Caveats: on gold tags it is slightly worse on test (precision@5 -0.016*,
   role@10 -0.050*), gig-side hubness is 12.6% vs 10.1% (provider side 14.4% vs 14.7%), and bench B is a within-pool rerank of
   a pool built from BM25 + dense, so it does not yet show a gain in the fused ranking.
3c. **Graded confirmation of plain qwen3-0.6b on the 271 test gigs (2026-10-02, continuous prompt, headline score >= 0.5).**
   Wiring: `tag_corpus.py --encoder qwen3-0.6b` writes `tags_*_hc-qwen3-0.6b.json` (centred, scaled to unit pooled sd, so the
   weighted scorer's tau is 1.0); `TagChannel(variant="hc-qwen3-0.6b")` ranks exactly like the bench (same nonzero providers,
   scores equal up to a per-gig constant). `eval_tag_variants.py --prompt cont --lists qwen` regraded all 9,806 pairs in the compared
   top-10s with one grader (2 pairs the grader answered in prose are filled worst/best as bounds; conclusions are identical).
   Fused ranking RRF(bm25, dense, tag), paired over gigs (NDCG@10 / P@10; `*` = CI excludes 0):

   | Comparison | NDCG@10 | P@10 | MRR@10 |
   |---|---|---|---|
   | RRF + mxbai tag (`rrf3_hc`) - RRF | +0.019* | +0.012* | +0.008 |
   | RRF + qwen3 tag (`rrf3_hcq`) - RRF | **+0.058\*** | +0.035* | +0.088* |
   | **`rrf3_hcq` - `rrf3_hc`** | **+0.039 [+0.027, +0.052]\*** | **+0.023 [+0.011, +0.034]\*** | +0.080 [+0.043, +0.120]* |
   | same, relevant = score >= 0.6 / 0.7 | +0.038* / +0.035* | +0.022* / +0.013* | +0.074* / +0.051* |
   | same, graded NDCG (gain = 100 x score) | +0.037* | | |
   | channel alone `tag_hcq` - `tag_hc` | +0.046* | +0.036* | +0.068* |
   | `tag_hcqw` - dense | -0.010 (ns) | -0.026* | -0.047 (ns) |

   - **The mxbai tagger's RRF gain reproduces under the new grader (+0.019 NDCG@10 vs +0.021 before)**, a sanity check on the grader.
   - **Absolute:** P@10 0.300 -> 0.334 and NDCG@10 0.716 -> 0.774 (RRF alone -> RRF + qwen3 tag). The qwen3 channel alone closes the
     NDCG@10 gap to dense from 0.056 to 0.010 (weighted scorer).
   - Same result with the weighted scorer (`rrf3_hcqw` - `rrf3_hcw`: NDCG@10 +0.039*), and unchanged however the 2 ungradable pairs are filled.
   - **Blind Claude audit, round 4 (`claude_audit.py --round 4`, `audit/claude_grades4.jsonl`)**, run because the grader and the encoder
     are both Qwen models: 120 gigs, one entering and one leaving pair each, drawn whatever the qwen grade, graded blind by Claude in
     session (one rater; rule fixed in `ROUNDS[4]` before reading any grade). Claude >= 2 rate: entering 50/120 = 0.42, leaving
     36/120 = 0.30; per-gig difference **+0.117 [+0.008, +0.225]**, so by the rule the gain is **supported** (the interval only just
     excludes 0). On the same pairs the qwen grader gives 0.22 vs 0.22 (difference 0.000 [-0.092, +0.092]), so the audit does not
     reproduce its size, only its direction; Claude is more lenient overall (36% >= 2 against 22%). Underpowered for a gap of about 0.08, as noted.
   - **Caveats:** one grader plus one rater; the audit's lower bound is close to 0; bench B on these test gigs was looked at once
     before; the qwen3 embeddings came from the GPU run (a CPU rebuild would differ by about 1e-3 in cosine); the channel's
     gig-side hubness is 12.6% vs 10.1% (see 3 above); **Stage 2 (shipped LambdaMART ranker) has not been run with the qwen3 tags.**

4. **Pick at most two finalists** (after 3b: one finalist, plain `qwen3-0.6b`; the head / dsm / ensemble candidates above are parked; the earlier `mxbai+mpnet` and prototype candidates are now dominated on dev).
5. **Confirmation on the test gigs** with the continuous prompt (about 3 hours of grading), the score-threshold sweep, a Stage-2
   run (`features.py --tag-channel --tag-variant <new>` for both CSVs the same day), and a blind Claude audit of entering/leaving pairs.
   **Done for Stage 1 on 2026-10-02 (section 3c); the Stage-2 run is still open.**
6. **Write up** as TAG_CHANNEL.md section 13, with the grader caveats, and decide what to commit (the user's call).
7. **Housekeeping:** nothing is committed; `tag_corpus.py` / `tag_channel.py` would need a small edit (new variant names and
   the `wcos` gate) only if a finalist is adopted into the pipeline.

## 8. Files and how to reproduce

- Bench: `python pipeline/eval_tag_encoder.py --encoders mxbai mpnet mxbai+mpnet --split dev` (first embedding pass in the GPU venv;
  later runs read the cache). Candidates are compared with `--baseline` (first by default).
- Calibration: `python pipeline/labeller.py calibrate --prompt orig|cont` (about 5 minutes; needs `SOCLAAS_*` in `.env`).
- Tests: `python -m unittest discover -s pipeline/tests -t pipeline` (183 pass).
- Memory notes kept for later sessions: `tag-encoder-bench-findings.md`, and updates in `tag-channel-audit-findings.md`.
