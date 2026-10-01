# Claude audit of the qwen grader (rounds 1 and 2)

Independent regrade by Claude (a different model family) of stratified (gig, provider) pairs, graded blind with
the rubric in `labeller.py`. One rater, so "agreement" is agreement with Claude, not with ground truth.
Reproduce: `python pipeline/claude_audit.py score [--round 2]` and `... termcheck`.

## Round 2: 160 pairs, one per gig, drawn at random within each qwen grade

| Stratum | n | Claude also grades >= 2 (of qwen's >= 2) | qwen < 2 but Claude >= 2 |
|---|---|---|---|
| Original grades, qwen >= 2 | 30 | 30/30 = 1.00 [0.89, 1.00] | |
| Original grades, qwen < 2 | 10 | | 2/10 [0.06, 0.51] |
| Tag-only pairs (v2-repro), qwen >= 2 | 80 | 45/80 = 0.56 [0.45, 0.67] | |
| Tag-only pairs (v2-repro), qwen < 2 | 40 | | 2/40 [0.01, 0.17] |

Fisher exact, tag-only vs original positives: p < 0.0001.

Matched on qwen's own confidence (mean P(grade >= 2) is 0.63 in both positive strata), Claude confirms original
positives in every bin (13/13, 13/13, 3/3, 1/1) but tag-only positives only 12/34, 25/37, 6/7, 2/2 for
P < 0.6, 0.6-0.8, 0.8-0.95, >= 0.95.

## Rule-based term-cap check (no second grader; all graded pairs)

Share of grade >= 2 given despite a serious term mismatch (rate > 1.4x the top of the budget, mid vs expert,
start > 30 days late, or >= 2 days a week short): original 10/1225 = 0.8%, v2-repro 146/2426 = 6.0% (109 budget).
Of the 35 sampled tag-only positives Claude downgraded, 3 had a serious mismatch and 32 were content-only.

## Round 1 (40 pairs, enriched for borderline pairs)

Confident pairs agree on the >= 2 line 12/12. Borderline original pairs split 8/16 with no lean (Claude lower 3,
higher 6). Tag-only positives agree 3/8.

## Round 3: what the hubness-corrected channel's gain rests on (195 pairs, graded blind)

Drawn from the pairs that decide the paired P@10 delta of RRF + `tag_hc` against RRF(bm25, dense) on the 271 test
gigs, plus pairs graded under both prompts (`eval_tag_variants.py diff`, `claude_audit.py sample --round 3`). Only
reproduction positives are drawn from the two diff strata. One rater; the key was not opened until all 195 grades were
saved. `python pipeline/claude_audit.py score --round 3`.

| Stratum | n | Claude also grades >= 2 |
|---|---|---|
| Entering the top-10 (repro >= 2) | 70 | 44/70 = 0.63 [0.51, 0.73] |
| Leaving the top-10 (repro >= 2) | 70 | 44/70 = 0.63 [0.51, 0.73] |
| Original < 2, repro >= 2 (same pairs) | 40 | 22/40 = 0.55 [0.40, 0.69] |
| Original >= 2 and repro >= 2 (control) | 15 | 15/15 = 1.00 [0.80, 1.00] |

- Entering minus leaving: 0.000 [-0.156, +0.156], Fisher p = 1. Not evidence that the entering pairs are over-credited
  more than the leaving ones; underpowered for a gap of 0.08, which is what the Stage-1 P@10 gain implies.
- On the 40 pairs where the prompts disagree, Claude sides with the reproduction 22 times and the original 18 times:
  the reproduction is lenient, but the original is not simply right either.
- Confirmation by the grader's own P(>= 2), entering and leaving pooled: 16/50 below 0.6, 30/48 from 0.6 to 0.8,
  42/42 from 0.8 up. Weighting each reproduction positive by that rate gives a P@10 gain of +0.0154 [+0.0048, +0.0262]
  (uncalibrated +0.0255); with side-specific rates, the worst case, +0.0095 [-0.0114, +0.0290].

Paired term check (`claude_audit.py termcheck`, no second grader): of the 5,004 pairs graded under both prompts, 1,624
have a serious term mismatch. Among those, the original grade is >= 2 for 7 (0.4%) and the reproduction's for 157
(9.7%); among the 3,380 without a mismatch, 20% and 59%. The share of pairs with a mismatch is 0.385 among the
original's pairs and 0.383 among the tag-only pairs, so the 0.8% against 6.0% above reflects the prompt, not a
different base rate of mismatched pairs. This answers the confound noted for round 2: prompt and population are
separable, and the prompt is lenient.
