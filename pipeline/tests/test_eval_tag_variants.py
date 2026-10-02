"""
Tests for eval_tag_variants.py: the dev/test split must be exactly the one eval_tag_channel.py sat and
eval_tag_samegrader.py use, and seeding must only ever reuse reproduction grades for the pairs wanted.

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v
"""
import json
import tempfile
import unittest
from pathlib import Path

try:
    import eval_tag_samegrader as sg
    import eval_tag_variants as ev
except ImportError as exc:                       # xgboost is only in requirements-rerank.txt
    raise unittest.SkipTest(f"variant evaluation needs the rerank requirements: {exc}")


def _write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class SplitGigs(unittest.TestCase):
    def test_dev_and_test_partition_the_gigs_and_test_matches_samegrader(self):
        hirers = [{"hire_id": str(h)} for h in range(1, 13)]
        gt = {str(h): {"7": 100} for h in range(1, 11)}          # 11 and 12 have no grade >= 2
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "hirers.json").write_text(json.dumps(hirers), encoding="utf-8")
            (Path(d) / "ground_truth_llm.json").write_text(json.dumps(gt), encoding="utf-8")
            dev, test = (ev.split_gigs(Path(d), 7, s) for s in ("dev", "test"))
            self.assertEqual(test, sg.test_gigs(Path(d), 7))
        self.assertEqual(sorted(dev + test), list(range(1, 11)))
        self.assertFalse(set(dev) & set(test))


class SeedGrades(unittest.TestCase):
    def test_only_wanted_reproduction_grades_are_seeded_and_never_twice(self):
        ok = lambda h, p, g: {"hire_id": h, "provider_id": p, "grade": g, "status": "ok"}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write(d / "judgments_tag.jsonl", [ok("1", "10", 2), ok("1", "11", 0), {**ok("2", "10", 1), "status": "error"}])
            _write(d / "judgments_regrade.jsonl", [ok("1", "10", 3), ok("3", "30", 1)])   # 1/10 is in both: SEED_FILES order, so regrade wins
            out = d / "out.jsonl"
            seeded, before = ev.seed_grades(d, {("1", "10"), ("2", "10"), ("3", "30")}, out)
            self.assertEqual((seeded, before), (2, 0))
            grades = sg.grades_from_records(sg.read_records(out))
            self.assertEqual(grades, {"1": {"10": 3}, "3": {"30": 1}})       # 1/11 not wanted, 2/10 errored
            self.assertEqual(ev.seed_grades(d, {("1", "10"), ("3", "30")}, out), (0, 2))   # idempotent


class TruthFromRecords(unittest.TestCase):
    @staticmethod
    def rec(h, p, grade, probs, status="ok"):
        return {"hire_id": h, "provider_id": p, "grade": grade, "probs": probs, "status": status}

    def setUp(self):
        self.records = [
            self.rec("1", "10", 3, [0.0, 0.05, 0.15, 0.80]),     # confident positive
            self.rec("1", "11", 2, [0.1, 0.30, 0.50, 0.10]),     # P(>=2) = 0.6
            self.rec("1", "12", 1, [0.1, 0.60, 0.25, 0.05]),     # grade 1 is never touched
            self.rec("1", "13", 0, [0.9, 0.10, 0.00, 0.00]),     # grade 0 is left out
            self.rec("2", "10", 2, [0.0, 0.40, 0.50, 0.10], status="error"),
        ]

    def test_default_equals_the_published_ground_truth(self):
        published = {h: {p: sg.SCORE[g] for p, g in ps.items() if g > 0}
                     for h, ps in sg.grades_from_records(self.records).items()}
        got = {h: row for h, row in ev.truth_from_records(self.records).items() if row}
        self.assertEqual(got, published)
        self.assertEqual(got, {"1": {"10": 100, "11": 67, "12": 33}})

    def test_low_confidence_positives_are_demoted_to_grade_one(self):
        self.assertEqual(ev.truth_from_records(self.records, min_p=0.8)["1"], {"10": 100, "11": 33, "12": 33})
        self.assertEqual(ev.truth_from_records(self.records, min_p=0.6)["1"]["11"], 67)   # P is 0.6: kept

    def test_flagged_pairs_are_demoted_but_only_when_positive(self):
        flagged = lambda h, p: p in ("10", "12", "13")
        self.assertEqual(ev.truth_from_records(self.records, flagged=flagged)["1"], {"10": 33, "11": 67, "12": 33})

    def test_a_later_record_replaces_an_earlier_one(self):
        later = self.records + [self.rec("1", "10", 0, [1.0, 0.0, 0.0, 0.0])]
        self.assertNotIn("10", ev.truth_from_records(later)["1"])


class ContinuousPrompt(unittest.TestCase):
    """--prompt cont: records carry `score` (and the band `grade`), never `probs`."""

    @staticmethod
    def rec(h, p, score, status="ok"):
        grade = sum(score >= cut for cut in (0.175, 0.50, 0.825))       # labeller.band_grade
        return {"hire_id": h, "provider_id": p, "grade": grade, "score": score, "probs": None, "status": status,
                "prompt_version": "rubric_0_1.v2-cont"}

    def setUp(self):
        self.records = [self.rec("1", "10", 0.90), self.rec("1", "11", 0.50), self.rec("1", "12", 0.65),
                        self.rec("1", "13", 0.30), self.rec("1", "14", 0.05), self.rec("2", "10", 0.9, status="error")]

    def test_headline_is_score_at_least_half_and_equals_the_band_grade(self):
        got = ev.truth_from_records(self.records)
        self.assertEqual(got["1"], {"10": 100, "11": 67, "12": 67, "13": 33})
        self.assertNotIn("14", got["1"])
        self.assertNotIn("2", got)

    def test_score_cut_demotes_positives_below_it_to_grade_one(self):
        self.assertEqual(ev.truth_from_records(self.records, min_score=0.6)["1"], {"10": 100, "11": 33, "12": 67, "13": 33})
        self.assertEqual(ev.truth_from_records(self.records, min_score=0.7)["1"], {"10": 100, "11": 33, "12": 33, "13": 33})
        self.assertEqual(ev.truth_from_records(self.records, min_score=0.5)["1"]["11"], 67)       # 0.50 is kept

    def test_cuts_demote_only_positives_and_flagged_stacks_with_them(self):
        flagged = lambda h, p: p == "10"
        self.assertEqual(ev.truth_from_records(self.records, flagged=flagged, min_score=0.6)["1"],
                         {"10": 33, "11": 33, "12": 67, "13": 33})

    def test_a_cut_whose_field_is_missing_raises_rather_than_keeping_the_pair(self):
        with self.assertRaises(ValueError):
            ev.truth_from_records(self.records, min_p=0.6)                       # no probs on the continuous prompt
        repro = [{"hire_id": "1", "provider_id": "10", "grade": 3, "probs": [0, 0, 0.1, 0.9], "status": "ok"}]
        with self.assertRaises(ValueError):
            ev.truth_from_records(repro, min_score=0.6)
        with self.assertRaises(ValueError):
            ev.gains_from_records(repro)

    def test_gains_are_one_hundred_times_the_score(self):
        gains = ev.gains_from_records(self.records + [self.rec("1", "10", 0.0)])       # a later 0 removes the pair
        self.assertEqual(set(gains["1"]), {"11", "12", "13", "14"})
        self.assertAlmostEqual(gains["1"]["12"], 65.0)

    def test_graded_ndcg_changes_only_ndcg(self):
        gt = {"1": {"10": 100, "11": 67, "12": 67, "13": 33}}
        gain = ev.gains_from_records(self.records)
        lists = {1: [13, 10, 11]}
        banded, graded = sg.metrics_per_gig(lists, gt, [1]), sg.metrics_per_gig(lists, gt, [1], gain)
        for m in ("P@5", "P@10", "R@5", "R@10", "MRR@10"):
            self.assertEqual(banded[m][0], graded[m][0])
        self.assertNotAlmostEqual(banded["NDCG@5"][0], graded["NDCG@5"][0])
        self.assertTrue(0 < graded["NDCG@5"][0] < 1)

    def test_seeding_reads_only_the_given_graders_files(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write(d / "judgments_tag.jsonl", [{"hire_id": "1", "provider_id": "10", "grade": 3, "status": "ok",
                                                "prompt_version": "rubric_0_3.v2-repro"}])
            _write(d / "calibration_cont.jsonl", [self.rec("1", "10", 0.9), self.rec("1", "11", 0.5)])
            out = d / "judgments_variants_cont.jsonl"
            seeded, _ = ev.seed_grades(d, {("1", "10"), ("1", "11")}, out, ev.PROMPTS["cont"]["seed"])
            self.assertEqual(seeded, 2)
            self.assertEqual(sg.prompt_version(sg.read_records(out)), "rubric_0_1.v2-cont")

    def test_prompt_version_refuses_a_mix_of_graders(self):
        mixed = [self.rec("1", "10", 0.9), {"hire_id": "1", "provider_id": "11", "grade": 2, "status": "ok",
                                             "prompt_version": "rubric_0_3.v2-repro"}]
        with self.assertRaises(ValueError):
            sg.prompt_version(mixed)
        self.assertEqual(sg.prompt_version([]), "rubric_0_3.v2-repro")

    def test_each_prompt_offers_only_relevance_definitions_it_can_compute(self):
        self.assertTrue(set(ev.PROMPT_RELEVANCE["repro"]) <= set(ev.RELEVANCE))
        self.assertTrue(set(ev.PROMPT_RELEVANCE["cont"]) <= set(ev.RELEVANCE))
        for name in ev.PROMPT_RELEVANCE["cont"]:
            self.assertNotIn("min_p", ev.RELEVANCE[name])
        for name in ev.PROMPT_RELEVANCE["repro"]:
            self.assertNotIn("min_score", ev.RELEVANCE[name])
            self.assertFalse(ev.RELEVANCE[name].get("graded"))


class EncoderVariant(unittest.TestCase):
    def test_variant_tau_per_tagger(self):
        import tag_channel as tc
        self.assertEqual((tc.variant_tau("hc"), tc.variant_tau("hz")), (0.05, 1.0))
        self.assertEqual(tc.variant_tau(ev.QWEN_VARIANT), 1.0)        # scores are in pooled-sd units
        self.assertIsNone(tc.variant_tau(""))                         # raw cosine has no weighted scorer

    def test_the_qwen_list_set_names_only_lists_that_exist_and_every_comparison_resolves(self):
        stage2 = {"fused2", "fused3_hc", "fused3_hcq"}                # exist only with --stage2
        known = {"bm25", "dense", "rrf2", *ev.TAG_CHANNELS, *ev.FUSED, *stage2}
        self.assertTrue(set(ev.LIST_SETS["qwen"]) <= known)
        self.assertEqual(set(ev.FUSED.values()) | {"tag"}, set(ev.TAG_CHANNELS))
        for label, a, b in ev.COMPARISONS:
            if a.startswith(("rrf3_hcq", "tag_hcq", "fused3_hcq")):
                self.assertIn(a, ev.LIST_SETS["qwen"], label)
                self.assertIn(b, ev.LIST_SETS["qwen"], label)


if __name__ == "__main__":
    unittest.main()
