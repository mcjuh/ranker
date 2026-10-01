"""
Tests for labeller.py with a stub `complete()` (no network, no key needed).

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v
"""
import json
import math
import tempfile
import unittest
import urllib.error
from datetime import date
from pathlib import Path

import labeller
from labeller import (ANCHOR_DATE, Labeller, build_prompt, calibration_report, cohen_kappa, grade_distribution,
                      ground_truth_from, load_env, merge_labels, read_records, run_pairs)

HIRER = {"hire_id": "7", "hire_title": "Audit data room", "hire_description": "Review contracts for an acquisition.",
         "hire_description_additional_notes": "", "budget_lo": "165", "budget_hi": "235", "seniority_needed": "senior",
         "start_by": "asap", "commitment": "3"}
PROVIDER = {"provider_id": "9", "about_title": "Corporate lawyer", "about_description": "Ten years of M&A diligence.",
            "services_offered_title": "Due diligence", "services_offered_description": "Contract review.",
            "relevant_experience": "", "rate_per_hour": "310", "seniority": "expert", "available_from": "2026-11-19",
            "capacity": "2", "availability": "Available from 19 Nov 2026, 2 days a week"}


def lp(**probs):
    """A response whose first token has these probabilities: lp(**{"0": .1, "3": .9})."""
    top = [{"token": t, "logprob": math.log(p)} for t, p in probs.items()]
    return {"choices": [{"message": {"content": max(probs, key=probs.get)}, "logprobs": {"content": [{"top_logprobs": top}]}}],
            "usage": {"prompt_tokens": 800, "completion_tokens": 2}}


def fake_clock():
    t = [1_700_000_000.0]
    def clock():
        t[0] += 0.25
        return t[0]
    return clock


class GradeDistribution(unittest.TestCase):
    def test_renormalises_over_the_four_digits_and_ignores_other_tokens(self):
        top = [{"token": "2", "logprob": math.log(0.3)}, {"token": "3", "logprob": math.log(0.1)},
               {"token": "We", "logprob": math.log(0.5)}]   # 0.4 of the mass is on digits
        probs = grade_distribution(top)
        self.assertAlmostEqual(probs[2], 0.75)
        self.assertAlmostEqual(probs[3], 0.25)
        self.assertEqual(probs[0], 0.0)
        self.assertAlmostEqual(sum(probs), 1.0)

    def test_whitespace_variants_of_a_digit_add(self):
        top = [{"token": "1", "logprob": math.log(0.2)}, {"token": " 1", "logprob": math.log(0.2)},
               {"token": "0", "logprob": math.log(0.2)}]
        probs = grade_distribution(top)
        self.assertAlmostEqual(probs[1], 2 / 3)
        self.assertAlmostEqual(probs[0], 1 / 3)

    def test_no_digit_at_all_is_none(self):
        self.assertIsNone(grade_distribution([{"token": "We", "logprob": -0.1}]))
        self.assertIsNone(grade_distribution([]))


class GradePair(unittest.TestCase):
    def label(self, response, **kw):
        calls = []
        def complete(payload):
            calls.append(payload)
            return response(payload) if callable(response) else response
        rec = Labeller(complete, "m", clock=fake_clock()).grade_pair(HIRER, PROVIDER)
        return rec, calls

    def test_record_schema_grade_and_expected_grade(self):
        rec, calls = self.label(lp(**{"1": 0.25, "2": 0.5, "3": 0.25}))
        self.assertEqual(list(rec), ["hire_id", "provider_id", "model", "prompt_version", "timestamp", "status", "grade",
                                     "expected_grade", "probs", "generated", "scoring", "usage", "elapsed"])
        self.assertEqual((rec["status"], rec["grade"]), ("ok", 2))
        self.assertAlmostEqual(rec["expected_grade"], 0.25 * 1 + 0.5 * 2 + 0.25 * 3)
        self.assertEqual(rec["probs"], [0.0, 0.25, 0.5, 0.25])
        self.assertEqual(rec["prompt_version"], "rubric_0_3.v2-repro")
        self.assertEqual(rec["scoring"], "logprobs")
        self.assertEqual(rec["usage"], {"prompt_tokens": 800, "completion_tokens": 2})
        self.assertEqual((rec["hire_id"], rec["provider_id"]), ("7", "9"))

    def test_grade_is_the_argmax_not_the_rounded_mean(self):
        rec, _ = self.label(lp(**{"0": 0.45, "3": 0.4, "2": 0.15}))   # mean 1.5 would round to 2; the mode is 0
        self.assertEqual(rec["grade"], 0)
        self.assertAlmostEqual(rec["expected_grade"], 0.4 * 3 + 0.15 * 2)

    def test_request_is_deterministic_one_token_with_reasoning_off(self):
        _, calls = self.label(lp(**{"0": 0.9, "1": 0.1}))
        (p,) = calls
        self.assertEqual((p["temperature"], p["max_tokens"], p["logprobs"], p["top_logprobs"]), (0, 2, True, 20))
        self.assertEqual(p["reasoning_effort"], "none")
        self.assertNotIn("chat_template_kwargs", p)

    def test_empty_content_retries_once_with_thinking_off_the_other_way(self):
        leaked = {"choices": [{"message": {"content": "", "reasoning": "We need to"}, "logprobs": {"content": [{"top_logprobs": [{"token": "We", "logprob": -0.1}]}]}}]}
        rec, calls = self.label(lambda p: lp(**{"3": 0.9, "2": 0.1}) if "chat_template_kwargs" in p else leaked)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("reasoning_effort", calls[1])
        self.assertEqual((rec["status"], rec["grade"]), ("ok", 3))

    def test_empty_content_twice_is_an_error_not_a_grade(self):
        leaked = {"choices": [{"message": {"content": ""}, "logprobs": {"content": [{"top_logprobs": []}]}}]}
        rec, calls = self.label(leaked)
        self.assertEqual(len(calls), 2)
        self.assertTrue(rec["status"].startswith("error"))
        self.assertIsNone(rec["grade"])

    def test_answer_without_grade_tokens_is_an_error(self):
        resp = lp(**{"We": 0.9})
        rec, _ = self.label(resp)
        self.assertTrue(rec["status"].startswith("error"))
        self.assertIsNone(rec["grade"])


class Prompt(unittest.TestCase):
    def test_contains_every_field_and_the_thresholds(self):
        text = build_prompt(HIRER, PROVIDER)
        for needle in ("Audit data room", "Review contracts for an acquisition.", "Corporate lawyer", "Ten years of M&A diligence.",
                       "Due diligence", "Contract review.", "Budget: S$165-235 per hour", "Seniority needed: senior",
                       "Commitment: 3 days a week", "Rate: S$310 per hour", "Seniority: expert",
                       "Availability: from 19 Nov 2026, 2 days a week", "15%", "40%", "single digit"):
            self.assertIn(needle, text)

    def test_sentinels_resolve_to_the_anchor_date_not_today(self):
        text = build_prompt(HIRER, {**PROVIDER, "available_from": "now"})
        self.assertIn("Start date: 1 Oct 2026", text)
        self.assertIn("Availability: from 1 Oct 2026, 2 days a week", text)
        self.assertNotIn("asap", text)
        other = build_prompt(HIRER, {**PROVIDER, "available_from": "now"}, today=date(2026, 12, 25))
        self.assertIn("Start date: 25 Dec 2026", other)
        self.assertEqual(ANCHOR_DATE, date(2026, 10, 1))

    def test_availability_falls_back_to_the_display_text(self):
        text = build_prompt(HIRER, {**PROVIDER, "available_from": None, "capacity": None})
        self.assertIn("Availability: Available from 19 Nov 2026, 2 days a week", text)

    def test_singular_day(self):
        self.assertIn("1 day a week", build_prompt({**HIRER, "commitment": "1"}, PROVIDER))


class OriginalTextPrompt(unittest.TestCase):
    def test_default_variant_is_the_reconstruction(self):
        self.assertEqual(build_prompt(HIRER, PROVIDER), build_prompt(HIRER, PROVIDER, variant="repro"))
        self.assertEqual(Labeller(lambda p: {}, "m").prompt_version, "rubric_0_3.v2-repro")

    def test_orig_carries_the_adjacency_anchors_the_terms_and_the_output_instruction(self):
        text = build_prompt(HIRER, PROVIDER, variant="orig")
        for needle in ("for example an adjacent sub-focus", "an adjacent area or a transferable skill",
                       "Up to about 15% over the top of the range", "mid against expert is serious",
                       "Gig: Audit data room", "Provider profile: Corporate lawyer",
                       "Budget: S$165-235 per hour", "Rate: S$310 per hour",
                       "Availability: from 19 Nov 2026, 2 days a week"):
            self.assertIn(needle, text)
        self.assertTrue(text.endswith("Output only a single integer between 0 and 3 inclusive."))
        self.assertNotIn("Terms can only lower a grade", text)       # the reconstruction's own addition

    def test_orig_is_versioned_apart_from_the_reconstruction(self):
        self.assertEqual(Labeller(lambda p: {}, "m", variant="orig").prompt_version, "rubric_0_3.v2-orig-text")

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            build_prompt(HIRER, PROVIDER, variant="bogus")


def text_response(content):
    return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 800, "completion_tokens": 4}}


class ContinuousVariant(unittest.TestCase):
    def test_prompt_has_the_bands_the_date_the_terms_and_the_output_instruction(self):
        text = build_prompt(HIRER, PROVIDER, variant="cont")
        for needle in ("continuous scale from 0 to 1", "Today is 1 October 2026.", "0.85-1.00 = excellent match",
                       "0.20-0.45 = weak match", "for example an adjacent sub-focus", "Gig: Audit data room",
                       "Provider profile: Corporate lawyer", "Budget: S$165-235 per hour", "Rate: S$310 per hour"):
            self.assertIn(needle, text)
        self.assertTrue(text.endswith("(for example 0.07, 0.38, 0.64 or 0.91)."))
        self.assertIn("Today is 25 December 2026.", build_prompt(HIRER, PROVIDER, date(2026, 12, 25), "cont"))

    def test_parse_score(self):
        for text, want in (("0.64", 0.64), (" 0.07\n", 0.07), ("1", 1.0), ("0", 0.0), (".5", 0.5), ("Score: 0.38", 0.38)):
            self.assertEqual(labeller.parse_score(text), want, text)
        for text in ("", "high", "1.5", "7", "0.64.2", "12"):
            self.assertIsNone(labeller.parse_score(text), text)

    def test_band_grade_cuts_the_gaps_between_the_anchor_bands(self):
        got = [labeller.band_grade(s) for s in (0.0, 0.15, 0.17, 0.2, 0.45, 0.5, 0.55, 0.8, 0.82, 0.85, 1.0)]
        self.assertEqual(got, [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3])

    def label(self, content):
        calls = []
        def complete(payload):
            calls.append(payload)
            return text_response(content)
        return Labeller(complete, "m", clock=fake_clock(), variant="cont").grade_pair(HIRER, PROVIDER), calls

    def test_record_carries_the_decoded_score_and_its_band(self):
        rec, calls = self.label("0.64")
        self.assertEqual((rec["status"], rec["score"], rec["grade"], rec["scoring"]), ("ok", 0.64, 2, "generated"))
        self.assertEqual(rec["prompt_version"], "rubric_0_1.v2-cont")
        self.assertIsNone(rec["probs"])
        self.assertEqual(calls[0]["max_tokens"], 8)

    def test_unparseable_answer_is_an_error(self):
        rec, _ = self.label("very good")
        self.assertTrue(rec["status"].startswith("error"))
        self.assertNotIn("score", rec)

    def test_continuous_report_on_a_perfectly_ordered_regrade(self):
        original = [0, 0, 1, 1, 2, 2, 3, 3]
        scores = [0.03, 0.10, 0.25, 0.40, 0.60, 0.72, 0.88, 0.95]
        r = labeller.continuous_calibration_report(original, scores)
        self.assertGreater(r["spearman_with_original_grade"], 0.97)      # grades tie in pairs, so just under 1
        self.assertEqual((r["auc_original_ge1"], r["auc_original_ge2"], r["auc_original_ge3"]), (1.0, 1.0, 1.0))
        self.assertEqual(r["banded_exact_agreement"], 1.0)
        self.assertEqual(r["banded_kappa_grade_ge2"], 1.0)
        self.assertEqual(r["score_by_original_grade"][3]["mean"], (0.88 + 0.95) / 2)
        self.assertEqual(r["share_of_scores_in_band_gaps"], 0.0)
        self.assertEqual(r["distinct_scores"], 8)

    def test_auc_counts_ties_half(self):
        self.assertEqual(labeller._auc([0.5], [0.5]), 0.5)
        self.assertEqual(labeller._auc([0.9, 0.1], [0.5]), 0.5)
        self.assertIsNone(labeller._auc([], [0.5]))

    def test_spearman_handles_ties_and_reversal(self):
        self.assertAlmostEqual(labeller.spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertAlmostEqual(labeller.spearman([1, 1, 2], [5, 5, 9]), 1.0)


class Runner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "j.jsonl"
        self.hirers = {str(i): {**HIRER, "hire_id": str(i)} for i in range(1, 4)}
        self.providers = {str(i): {**PROVIDER, "provider_id": str(i)} for i in range(1, 4)}
        self.pairs = [(h, p) for h in self.hirers for p in self.providers]
        self.calls = []

    def labeller(self, responder=None):
        def complete(payload):
            self.calls.append(payload)
            return responder(payload) if responder else lp(**{"2": 0.9, "1": 0.1})
        return Labeller(complete, "m", clock=fake_clock())

    def run_it(self, labeller=None, **kw):
        sleeps = []
        stats = run_pairs(self.pairs, self.hirers, self.providers, labeller or self.labeller(), self.out,
                          sleep=sleeps.append, log=lambda *_: None, **kw)
        return stats, sleeps

    def test_appends_one_record_per_pair_in_seeded_order(self):
        stats, _ = self.run_it(seed=3)
        self.assertEqual((stats["graded"], stats["errors"]), (9, 0))
        first = [(r["hire_id"], r["provider_id"]) for r in read_records(self.out)]
        self.assertEqual(sorted(first), sorted(self.pairs))
        self.assertEqual(first, labeller.seeded_order(self.pairs, 3))
        self.assertNotEqual(first, sorted(self.pairs))   # shuffled, not sorted

    def test_resume_skips_finished_pairs(self):
        self.run_it(limit=4)
        self.assertEqual(len(read_records(self.out)), 4)
        self.calls.clear()
        stats, _ = self.run_it()
        self.assertEqual((stats["already_done"], stats["graded"]), (4, 5))
        self.assertEqual(len(self.calls), 5)
        done = [(r["hire_id"], r["provider_id"]) for r in read_records(self.out)]
        self.assertEqual(len(done), len(set(done)))   # nothing graded twice

    def test_errors_are_retried_on_the_next_run(self):
        leaked = {"choices": [{"message": {"content": ""}, "logprobs": {"content": [{"top_logprobs": []}]}}]}
        self.run_it(self.labeller(lambda p: leaked), limit=3)
        self.assertTrue(all(r["status"] != "ok" for r in read_records(self.out)))
        self.calls.clear()
        stats, _ = self.run_it()
        self.assertEqual(len(self.calls), 9)     # the 3 errors are not counted as done
        self.assertEqual(stats["graded"], 9)
        self.assertEqual(len(labeller.finished_pairs(read_records(self.out))), 9)

    def test_rate_limit_spaces_calls(self):
        sleeps = []
        ticks = iter(range(1000))
        run_pairs(self.pairs[:3], self.hirers, self.providers, self.labeller(), self.out,
                  rps=0.5, sleep=sleeps.append, clock=lambda: float(next(ticks)) * 0.1, log=lambda *_: None)
        self.assertEqual(len(sleeps), 2)           # none before the first call
        self.assertTrue(all(s > 1.0 for s in sleeps))   # ~2 s interval minus the 0.1-0.2 s already elapsed

    def test_network_failure_backs_off_then_recovers(self):
        state = {"n": 0}
        def flaky(payload):
            state["n"] += 1
            if state["n"] <= 2:
                raise urllib.error.URLError("down")
            return lp(**{"1": 1.0})
        stats, sleeps = self.run_it(self.labeller(flaky), limit=1)
        self.assertEqual((stats["graded"], stats["errors"]), (1, 0))
        self.assertEqual([s for s in sleeps if s >= 2], [2.0, 4.0])   # exponential backoff

    def test_persistent_failure_is_recorded_and_eventually_aborts(self):
        self.hirers = {str(i): {**HIRER, "hire_id": str(i)} for i in range(1, 6)}
        self.providers = {str(i): {**PROVIDER, "provider_id": str(i)} for i in range(1, 6)}
        self.pairs = [(h, p) for h in self.hirers for p in self.providers]   # 25 pairs > the abort threshold
        def down(payload):
            raise urllib.error.URLError("down")
        with self.assertRaises(RuntimeError):
            self.run_it(self.labeller(down))
        recs = read_records(self.out)
        self.assertEqual(len(recs), labeller.ABORT_AFTER_CONSECUTIVE_ERRORS)
        self.assertTrue(all(r["status"].startswith("error") for r in recs))


class Merge(unittest.TestCase):
    def rec(self, h, p, g, status="ok"):
        return {"hire_id": h, "provider_id": p, "status": status, "grade": g}

    def test_never_overwrites_an_existing_label(self):
        base = {"1": {"10": 2}}
        merged, counts = merge_labels(base, [self.rec("1", "10", 0), self.rec("1", "11", 3), self.rec("2", "12", 1)])
        self.assertEqual(merged, {"1": {"10": 2, "11": 3}, "2": {"12": 1}})
        self.assertEqual(counts, {"added": 2, "kept_existing": 1, "skipped_errors": 0})
        self.assertEqual(base, {"1": {"10": 2}})     # input untouched

    def test_errors_add_nothing(self):
        merged, counts = merge_labels({}, [self.rec("1", "10", None, status="error: x")])
        self.assertEqual(merged, {})
        self.assertEqual(counts["skipped_errors"], 1)

    def test_ground_truth_keeps_positive_grades_only_with_the_score_scale(self):
        gt = ground_truth_from({"1": {"10": 0, "11": 1, "12": 2, "13": 3}, "2": {"20": 0}})
        self.assertEqual(gt, {"1": {"11": 33, "12": 67, "13": 100}})


class Calibration(unittest.TestCase):
    # original:  0 0 1 1 2 2 3 3     regraded: 0 1 1 1 2 3 3 3  -> 6/8 exact, all within one
    ORIG = [0, 0, 1, 1, 2, 2, 3, 3]
    NEW = [0, 1, 1, 1, 2, 3, 3, 3]

    def test_confusion_matrix_rows_are_original(self):
        m = labeller.confusion_matrix(self.ORIG, self.NEW)
        self.assertEqual(m, [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1], [0, 0, 0, 2]])

    def test_report_statistics(self):
        r = calibration_report(self.ORIG, self.NEW, [0.1, 0.9, 0.8, 0.8, 1.5, 2.1, 2.3, 2.3])
        self.assertAlmostEqual(r["exact_agreement"], 6 / 8)
        self.assertAlmostEqual(r["within_one"], 1.0)
        # grade >= 2: original 0 0 0 0 1 1 1 1, new 0 0 0 0 1 1 1 1 -> perfect
        self.assertAlmostEqual(r["kappa_grade_ge2"], 1.0)
        # hand-computed: observed disagreement 2/72, expected 20/72 -> 1 - 0.1
        self.assertAlmostEqual(r["quadratic_weighted_kappa"], 0.9)
        self.assertAlmostEqual(r["by_assigned_grade"][1]["mean_expected_grade"], (0.9 + 0.8 + 0.8) / 3)
        self.assertEqual(r["by_assigned_grade"][3]["original_mean_expected_grade"], 2.26)
        self.assertEqual(r["by_assigned_grade"][0]["n"], 1)

    def test_kappa_known_values(self):
        a, b = [0, 1, 0, 1], [0, 1, 1, 0]          # 50% agreement, chance 50% -> kappa 0
        self.assertAlmostEqual(cohen_kappa(a, b, labels=(0, 1)), 0.0)
        self.assertAlmostEqual(cohen_kappa(a, a, labels=(0, 1)), 1.0)
        self.assertAlmostEqual(cohen_kappa([0, 1, 0, 1], [1, 0, 1, 0], labels=(0, 1)), -1.0)

    def test_quadratic_weights_forgive_near_misses_more_than_far_ones(self):
        orig = [0, 1, 2, 3] * 5
        near = [min(g + 1, 3) if i % 2 else g for i, g in enumerate(orig)]
        far = [3 - g if i % 2 else g for i, g in enumerate(orig)]
        self.assertGreater(cohen_kappa(orig, near, weights="quadratic"), cohen_kappa(orig, far, weights="quadratic"))

    def test_report_with_a_grade_never_assigned(self):
        r = calibration_report([0, 1], [0, 1], [0.1, 0.9])
        self.assertIsNone(r["by_assigned_grade"][3]["mean_expected_grade"])

    def test_sample_is_balanced_seeded_and_capped(self):
        merged = {str(h): {str(p): (h + p) % 4 for p in range(10)} for h in range(10)}
        s = labeller.calibration_sample(merged, per_grade=5, seed=7)
        self.assertEqual([sum(1 for *_, g in s if g == k) for k in range(4)], [5, 5, 5, 5])
        self.assertTrue(all(merged[h][p] == g for h, p, g in s))
        self.assertEqual(s, labeller.calibration_sample(merged, per_grade=5, seed=7))
        self.assertNotEqual(s, labeller.calibration_sample(merged, per_grade=5, seed=8))
        scarce = labeller.calibration_sample({"1": {"1": 0, "2": 3}}, per_grade=5, seed=7)
        self.assertEqual(len(scarce), 2)


class LoadEnv(unittest.TestCase):
    def test_parses_quotes_comments_and_splits_on_first_equals(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / ".env"
            p.write_text('# comment\n\nA=plain\nB="quoted value"\nC=\'single\'\nD=x=y=z\n  E = spaced \nnoequals\n', encoding="utf-8")
            self.assertEqual(load_env(p), {"A": "plain", "B": "quoted value", "C": "single", "D": "x=y=z", "E": "spaced"})

    def test_missing_file_is_empty(self):
        self.assertEqual(load_env(Path("definitely/not/here/.env")), {})


if __name__ == "__main__":
    unittest.main()
