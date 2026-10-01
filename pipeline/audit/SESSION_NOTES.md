# Session notes: tag channel review, LLM usage, and the grader audit

Written at the end of a working session with Claude (Claude Code, on the `mcjuh/ranker` fork). It records what was
asked, what was found, what was done, and what is suggested next. Numbers come from the repo and from
`pipeline/audit/RESULTS.md`; anything not measured is marked as a hypothesis. Literature pointers were recalled from
memory and have not been checked against the papers.

## 1. What was asked, in order

1. What is RRF fusion, and is the combination of channels the task owner's decision or only the channel's recall?
2. Is the proposed tag-embedding channel just another dense channel? Alternatives, literature, and whether
   centering or other geometry tricks can help.
3. Where is the qwen model called through SOCLaaS, why, and how often? Is pooling other LLM graders worth it, can
   Claude do a small reranking audit, and is the deployed ranker LLM-independent?
4. Run a blind audit of the qwen grader (40 pairs, then 160 more, mostly tag-only).
5. Verify the repo is the fork, push, and write these notes.

## 2. RRF and who owns the fusion

Reciprocal Rank Fusion ignores raw scores and uses only ranks: `score(d) = sum_c w_c / (60 + rank_c(d))`. BM25,
cosine and tag-BM25 scores are not comparable, ranks are. `k = 60` flattens the curve. A provider several channels
rank decently beats one a single channel loves.

- The combiner (`rrf_fuse`, two channels, k=60) is upstream code and is unchanged. `rrf_fuse_n` (N-way, same
  arithmetic, bit-identical for two lists) was added in this fork behind `--tag-channel` (default off).
- Choosing the weight of your own channel and reporting its sensitivity is in scope. Changing `k` or the fusion method
  changes every downstream result (LambdaMART features, the shipped fused ranker), so that is a team decision.
- Maximising a channel's standalone NDCG is the wrong target. Stage 2 reorders the top 50, so the metric that
  matters is recall of the fused top-50, and the useful property is complementary errors, not accuracy alone.

## 3. Problems observed in the tag channel

(From `tag_corpus.py`, `tag_channel.py`, `retrieval_tagbm25.py` and `pipeline/TAG_CHANNEL.md` on
`feat/tag-channel-eval`.)

- **It is a lossy projection of the dense channel.** Texts are embedded with the same mxbai model, matched by cosine
  against 2,088 short tag titles, the top 30 kept, and BM25 run over the tag IDs. Anything it knows, dense already
  knows. Marginal recall over BM25+dense on the original labels was 0.000 at K=50.
- **The cosines are thrown away.** `load_tags` keeps `[t for t, _cos in tags[:m]]`, so rank 1 and rank 30 weigh the
  same.
- **Length normalisation is a constant.** Every provider carries exactly 30 tags, so `b` cannot matter
  ("b = 0.75" is an arbitrary tie-break in the doc).
- **The tagger is weak.** Precision 0.29 at 5 tags falling to 0.135 at 30 on 400 role descriptions with gold tags
  (chance 0.011). Role@1 from predicted tags is 0.13, against 0.64 from gold tags, so the curated taxonomy is good and
  the tagging is the bottleneck. Hard track-first funnelling loses because the track scorer (0.762) is below the
  break-even (0.90).
- **Not deployable for unseen gigs as written.** `TagChannel.rank(hire_id)` indexes a precomputed
  `tags_hirers.json`; a new gig raises a `KeyError`. `tag_corpus.top_tags` already does the matmul and could back a
  `rank_text(text_vec)` method with no API call.
- **Evaluation is confounded both ways.** The original labels count everything the tag channel surfaces as
  irrelevant; the extended labels were graded by a reconstructed, more lenient prompt. Verdict in the doc: do not
  turn the channel on.

## 4. Where the LLM is used, and the deployment boundary

- One model, `qwen3.8:27b`, through SOCLaaS (`/chat/completions`), as a 0-3 relevance grader (temperature 0, one
  token, grade = argmax of logprobs). It is offline labelling only.
- Volume (counts from the repo and arithmetic): 22,289 original grades (about 6.2 h at 1 call/s, about 19.7M prompt
  tokens), 12,616 tag-pair grades, 7,345 same-grader regrades (2,420 reused), 300 calibration calls, so about 40k
  calls.
- The only HTTP client is in `labeller.py`. `retrieval_tagbm25.py` and `tag_channel.py` import only numpy and scipy,
  and the LambdaMART features call no LLM. Deployment needs the local mxbai encoder (already required by the dense
  channel) but no API. The one gap is the `KeyError` above.
- Pooling other graders: not recommended broadly. The repo's own pilot found `qwen3.6:35b` agreed exactly on 45% of
  pairs and graded far higher ("grade scales are model-specific, never mix models"). A 1.3B model adds noise, and
  same-family qwens share biases. If anything, run one second qwen on only the 7,345-pair same-grader pool and check
  whether the sign of the channel's effect holds under within-model ranks.

## 5. The audit

**Why.** All labels come from one model, and the tag-channel pairs came from a reconstructed prompt. A grader from a
different model family tests whether the verdict on the channel survives.

**How.** `pipeline/claude_audit.py`. Claude graded pairs blind (gig, provider and terms only, shuffled under neutral
ids) with the rubric from `labeller.py`. The key (stratum, qwen grade, probabilities) sat in a separate file that was
not read until the grades were saved. Round 1: 40 pairs, enriched for borderline cases. Round 2: 160 pairs, one per
gig, drawn at random within each qwen grade (80 tag-only positives, 40 tag-only negatives, 30 original positives,
10 original negatives).

**Results** (full table and method in `RESULTS.md`):

- Original positives: Claude also grades >= 2 on 30/30 [0.89, 1.00].
- Tag-only positives (reconstructed prompt): 45/80 = 0.56 [0.45, 0.67]. Fisher p < 0.0001.
- Matched on qwen's own confidence (mean P(>=2) = 0.63 in both), the gap remains: 13/13 against 12/34 for P < 0.6,
  13/13 against 25/37 for 0.6 to 0.8.
- The reconstructed prompt's error is mostly false positives (2 of 40 negatives upgraded).
- A rule-based term-cap check over all graded pairs found a grade >= 2 despite a serious term mismatch in 0.8% of the
  original positives and 6.0% of the reconstructed ones. Only 3 of the 35 downgraded tag-only positives had one; the
  rest were content-only, mostly adjacent-domain providers graded 2. An early remark (from one pair) that term caps
  were the main cause was too strong.

**Reading it.** Original grades look sound. Reconstructed grades, and the results that depend on them (extended
labels, the +0.023 NDCG@10 downstream row), should be discounted: roughly 56% of the 2,426 reconstructed positives
would survive, a one-rater estimate. The "do not turn it on" verdict is not weakened, because every comparison that
favoured the channel relied on those grades. The same-grader table also uses the reconstructed prompt, so its
absolute levels are uncertain.

**Limits.** One rater with its own threshold (strict on adjacent-domain matches). No pair was graded by both prompts
in this sample, so prompt leniency and pair population are not fully separated here; the repo's 300-pair calibration
(27% of original 0-1 pairs promoted to 2) points the same way. Only 10 original negatives were sampled, so false
negatives in the original labels are unbounded (2/10 upgraded, CI [0.06, 0.51]).

## 6. Suggested design improvements (none implemented)

Test these first on the tagger check (400 role descriptions with gold tags), which has no LLM-grader confound.

1. **Keep the cosines.** Weight tags by a softmax over cosines and score with a sparse weighted matmul instead of
   binarising the top 30. Expected to help the channel's accuracy, not its diversity.
2. **Hubness correction on the text side.** Subtracting a mean from the tag vectors does not change a text's ranking
   of tags (`q.mu` is constant); subtracting the text-side mean does (`q.t - mu_q.t`, a per-tag penalty). CSLS
   (Conneau et al. 2018) and Radovanovic et al. 2010 are the references.
3. **Use the curated taxonomy structure.** Embed roles, map a text softly to its nearest roles, and take their
   curated tag sets (43,958 role-tag links), instead of nearest tag titles. Hypothesis: cleaner tag sets than title
   cosine.
4. **Supervise the tagger.** 2,001 roles with gold tags are supervised data for extreme multi-label classification
   (Parabel, AttentionXML lineage); contrastive fine-tuning on (role description, tags) fits the embedding-geometry
   work.
5. **Decorrelate the errors.** A different encoder for the tagger, or an offline retrieve-then-LLM-rerank tagger
   (as in ESCO skill-extraction work, e.g. SkillSpan, Decorte et al.), run once to produce tag artifacts. The LLM
   stays out of the serving path. Risk: an LLM tagger scored by an LLM grader shares biases.
6. **Other options:** SPLADE-style learned sparse over the tag vocabulary; ColBERT-style multi-vector matching of
   requirements to provider entries (only if provider data has separable entries).
7. **Make it deployable:** add a `rank_text(text_vec)` path on top of `top_tags`.
8. **Fix the evaluation:** stop using the reconstructed prompt's grades as positives (or recalibrate against the audit
   labels), and grade the pairs that matter for a pool-widener claim (positives the two-way RRF top-50 misses) with an
   independent grader. A pool-widener use remains plausible but is untested here.

## 7. Process notes

- The local checkout started behind `origin/feat/tag-channel-eval`. Work was done on `audit/claude-grader`, created
  from that branch. It tracked `feat/tag-channel-eval` by default, so the push was made to a new remote branch name
  to avoid touching it.
- To import the labeller helpers without torch, `claude_audit.py` stubs `sentence_transformers`; numpy, scipy and
  rank-bm25 were installed in the session container (not repo changes).
- Grading cost about 90k tokens of context for the 160 pairs.

## 8. Files added on this branch

`pipeline/claude_audit.py`, `pipeline/audit/sample_key.json`, `sample2_key.json`, `claude_grades.jsonl`,
`claude_grades2.jsonl`, `RESULTS.md`, and this file. Reproduce with
`python pipeline/claude_audit.py score [--round 2]` and `python pipeline/claude_audit.py termcheck`.
