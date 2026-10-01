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
