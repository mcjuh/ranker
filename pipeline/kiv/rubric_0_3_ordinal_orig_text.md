# KIV: the 0-3 ordinal grading prompt (original text)

**Status: parked, not in use.** Keep in view and try later if the continuous 0-1 prompt (`labeller.py --prompt cont`)
does not help significantly in the end (criteria below).

- Prompt version: `rubric_0_3.v2-orig-text`. Run it with `python pipeline/labeller.py pairs|calibrate --prompt orig`; output
  files get an `_orig` suffix and nothing earlier is overwritten. The code stays in `labeller.py` (so this file and the code
  must change together); the text below is copied from `labeller.RUBRIC_ORIG` and `labeller.OUTPUT_INSTRUCTION`.
- Source: the rubric text was pasted by the original grader's owner (BT4103-Scrape-and-Tag `rubric_0_3_v2.md`). It was **not**
  read from that repo, so it is unverified against the file. The output instruction at the end is **ours**: the pasted text had none.
- Scoring: one generated token; the grade is the argmax of the log-probabilities over the tokens `0`..`3`, and
  `expected_grade` is their probability-weighted mean (same as every earlier grade).
- Not to be confused with the *reconstruction* (`rubric_0_3.v2-repro`, `labeller.RUBRIC`), which is still the default and
  which every existing tag-channel, hubness and encoder-bench grade came from. It stays in `labeller.py`.

## How it compared (300 already-graded pairs, 75 per original grade; `results_tag/calibration_orig.json`)

| | Reconstruction | This prompt | Continuous 0-1 |
|---|---|---|---|
| Kappa on grade >= 2 | 0.733 | **0.86** | 0.90 (banded) |
| Quadratic kappa | 0.818 | 0.89 | 0.876 (banded) |
| AUC for original >= 2 | 0.980 | 0.987 | 0.984 |
| Original 0-1 pairs promoted to >= 2 (of 150) | 40 | 8 | 13 |
| Original >= 2 pairs demoted below 2 (of 150) | 0 | 13 | 2 |
| Pairs graded >= 2 (original has 150) | 190 | 145 | 161 |

Known shifts of this prompt against the original grades: 21 of 75 original 1s went to 0 and 24 of 75 original 2s went to 3,
so grade 3 is over-assigned (95 against 75) and grade 2 under-assigned (50 against 75).

## Status of the conditions (checked 2026-10-02)

Condition 2 was tested (`audit/prompt_vs_claude.py`, 355 audited pairs, 260 of them tag-surfaced): on Claude agreement the
continuous prompt is about level with this one (binary kappa 0.55 against 0.50 over all pairs, 0.52 against 0.48 on the
tag-surfaced pairs; AUC 0.881 against 0.879). **Not triggered; this prompt stays parked.** Conditions 1, 3 and 4 are open until
the continuous score has been used in a graded evaluation.

## Why it might be worth coming back to

Try it if the continuous prompt does not help significantly, meaning any of:
1. **Its scores are noisy or tied where it matters.** The continuous prompt used only 22 distinct values over 300 pairs. If
   graded NDCG (continuous gain) tells no different story from the binary one, or flips with small score changes, the extra
   resolution is not buying anything.
2. **It over-credits pairs only the tag channel surfaces.** The calibration pairs come from the original pools, so they do not
   test that. If the continuous prompt agrees with the blind Claude audit grades (`audit/claude_grades2.jsonl`,
   `claude_grades3.jsonl`) no better than this prompt does, switch.
3. **Probability-based tooling is needed.** This prompt yields `probs` and `expected_grade`, which `eval_tag_variants.py`'s
   `--relevance p60 / p80 / p80+term` sweeps use. The continuous prompt has no `probs`; its sweep is by score threshold instead.
   If those sweeps turn out to be the more useful robustness check, this is the prompt that supports them without code changes.
4. **Decode wobble.** The continuous score is a greedy-decoded number; this one reads a probability distribution from the first
   token, so it is less sensitive to a single decoding path.

## The prompt (exactly as sent, after the gig and provider blocks)

Prompt text:

```text
You are matching a gig (a piece of work a client wants done) to a provider profile (an experienced professional's showcase). Score how well the provider fits this gig on an ordinal scale from 0 to 3: both what they have done and whether the practical terms work.

First judge the content fit: does the provider's expertise address what the gig needs? Then check the terms:

Budget: the provider's hourly rate against the gig's budget range. Up to about 15% over the top of the range is a minor mismatch; more than about 40% over is serious. A rate below the range is fine.
Seniority: the provider's level against the level the gig asks for (mid < senior < expert). One level apart is a minor mismatch; mid against expert is serious.
Availability: whether the provider can start by the gig's start date and offers the days per week the gig needs. Starting up to 2 weeks late, or offering 1 day a week fewer, is minor; starting more than a month late, or 2 or more days a week fewer, is serious.
Judgement: 3 = excellent match: the provider's specific expertise directly addresses this gig's specific need, and the terms work (at most one minor mismatch). 2 = good match: genuinely relevant domain and skill set, but not a perfect fit (for example an adjacent sub-focus), with at most minor mismatches in the terms; OR an excellent content fit with several minor mismatches. 1 = weak match: some surface relevance (an adjacent area or a transferable skill), but not a real fit for this need; OR relevant expertise with a serious mismatch in the terms. 0 = not relevant: no genuine content-level relevance, whatever the terms.
```

Layout built by `labeller.build_prompt(..., variant="orig")`:

```text
<the prompt text above>

Gig: <gig title + description + notes>
Budget: S$<lo>-<hi> per hour
Seniority needed: <mid|senior|expert>
Start date: <d Mon YYYY>
Commitment: <n> days a week

Provider profile: <about + services + experience>
Rate: S$<rate> per hour
Seniority: <mid|senior|expert>
Availability: from <d Mon YYYY>, <n> days a week

Output only a single integer between 0 and 3 inclusive.
```
