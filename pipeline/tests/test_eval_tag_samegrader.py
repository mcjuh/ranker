"""
Tests for eval_tag_samegrader.py: the test split must be exactly the one eval_tag_channel.py sat uses, plus the
small pure helpers (metrics at K <= 10, pools, reading the regrade records).

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import eval_tag_samegrader as sg
except ImportError as exc:                       # xgboost is only in requirements-rerank.txt
    raise unittest.SkipTest(f"same-grader evaluation needs the rerank requirements: {exc}")

from eval_tag_channel import rng_for as channel_rng_for


class TestGigs(unittest.TestCase):
    def test_matches_the_split_used_by_eval_tag_channel(self):
        hirers = [{"hire_id": str(h)} for h in (30, 4, 17, 9, 22, 1, 8, 15, 12, 40)]
        gt = {str(h): {"7": 100} for h in (30, 17, 22, 1, 8, 15, 12, 40)}      # 4 and 9 have no grade >= 2
        gt["4"] = {"7": 33}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "hirers.json").write_text(json.dumps(hirers), encoding="utf-8")
            (Path(d) / "ground_truth_llm.json").write_text(json.dumps(gt), encoding="utf-8")
            got = sg.test_gigs(Path(d), seed=7)
        # eval_tag_channel.run_sat: gigs in hirers.json order, permuted with rng_for(seed, "sat/split"), second half
        gigs = ["30", "17", "22", "1", "8", "15", "12", "40"]
        perm = channel_rng_for(7, "sat/split").permutation(len(gigs))
        self.assertEqual(got, [int(gigs[i]) for i in sorted(perm[len(gigs) // 2:])])
        self.assertEqual(len(got), 4)
        self.assertNotIn(4, got)
        self.assertNotIn(9, got)


class Metrics(unittest.TestCase):
    def test_cut_at_ten_and_hand_checked_values(self):
        gt = {"1": {"5": 100}, "2": {}}
        lists = {1: [9, 5] + list(range(100, 120)), 2: [1, 2]}
        m = sg.metrics_per_gig(lists, gt, [1, 2])
        self.assertAlmostEqual(m["P@5"][0], 1 / 5)
        self.assertAlmostEqual(m["P@10"][0], 1 / 10)
        self.assertAlmostEqual(m["R@5"][0], 1.0)
        self.assertAlmostEqual(m["NDCG@5"][0], 1 / math.log2(3))
        self.assertAlmostEqual(m["MRR@10"][0], 0.5)
        self.assertTrue(math.isnan(m["R@10"][1]))                               # no relevant pair for gig 2
        long_tail = sg.metrics_per_gig({1: list(range(100, 120)) + [5]}, gt, [1])   # relevant at rank 21: beyond the cut
        self.assertEqual((long_tail["P@10"][0], long_tail["MRR@10"][0]), (0.0, 0.0))

    def test_a_missing_gig_scores_as_an_empty_list(self):
        m = sg.metrics_per_gig({}, {"1": {"5": 100}}, [1])
        self.assertEqual((m["P@5"][0], m["NDCG@10"][0], m["MRR@10"][0]), (0.0, 0.0, 0.0))


class Records(unittest.TestCase):
    def test_grades_ignore_errors_and_the_latest_ok_record_wins(self):
        recs = [{"hire_id": "1", "provider_id": "5", "status": "ok", "grade": 1},
                {"hire_id": "1", "provider_id": "5", "status": "ok", "grade": 3},
                {"hire_id": "1", "provider_id": "6", "status": "error: x", "grade": None}]
        self.assertEqual(sg.grades_from_records(recs), {"1": {"5": 3}})

    def test_pool_is_the_sorted_union_over_lists_for_the_requested_gigs(self):
        lists = {"a": {1: [9, 5], 2: [7]}, "b": {1: [5, 3], 3: [8]}}
        self.assertEqual(sg.pool_pairs(lists, [1, 2]), {"1": [3, 5, 9], "2": [7]})

    def test_read_records_handles_missing_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            self.assertEqual(sg.read_records(p), [])
            p.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
            self.assertEqual([r["a"] for r in sg.read_records(p)], [1, 2])

    def test_score_scale_matches_evaluate_py(self):
        self.assertEqual(sg.SCORE, {1: 33, 2: 67, 3: 100})


if __name__ == "__main__":
    unittest.main()
