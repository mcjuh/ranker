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


if __name__ == "__main__":
    unittest.main()
