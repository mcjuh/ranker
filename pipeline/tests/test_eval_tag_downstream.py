"""
Tests for eval_tag_downstream.py: per-gig metrics (hand-checked), the NaN-aware bootstrap, label swapping,
the MODEL_PARAMS context manager, and a tiny end-to-end ranker run.

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v
"""
import csv
import math
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

try:
    import eval_tag_downstream as ds
    import rerank_ltr as ltr
except ImportError as exc:                       # xgboost is only in requirements-rerank.txt
    raise unittest.SkipTest(f"downstream evaluation needs the rerank requirements: {exc}")

FIELDS = ["hire_id", "provider_id", "label", "judged", "bm25_score", "bm25_rank", "dense_cosine", "dense_rank",
          "rrf_score", "rrf_rank", "budget_fit", "seniority_fit", "avail_immediacy", "tag_score", "tag_rank"]


def write_candidates(path: Path, rng: np.random.Generator, n_gigs=12, n_cands=8):
    """Synthetic candidates where dense_cosine tracks relevance, so any sane ranker beats random."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for h in range(1, n_gigs + 1):
            cos = rng.random(n_cands)
            order = np.argsort(-cos)
            for rank, i in enumerate(order, start=1):
                w.writerow({"hire_id": h, "provider_id": 100 + i, "label": "", "judged": 1, "bm25_score": rng.random(),
                            "bm25_rank": rng.integers(1, 50), "dense_cosine": cos[i], "dense_rank": rank,
                            "rrf_score": 1 / (60 + rank), "rrf_rank": rank, "budget_fit": rng.random(),
                            "seniority_fit": rng.random(), "avail_immediacy": rng.random(),
                            "tag_score": rng.random(), "tag_rank": rng.integers(0, 30)})


class PerGig(unittest.TestCase):
    GT = {"1": {"5": 100, "6": 33}, "2": {}}

    def test_hand_checked_metrics_and_undefined_recall(self):
        lists = {1: [9, 5, 6], 2: [1, 2, 3]}
        m = ds.per_gig(lists, self.GT, [1, 2])
        self.assertAlmostEqual(m["P@5"][0], 1 / 3)                 # only provider 5 is >= 40 among 3 returned
        self.assertAlmostEqual(m["R@5"][0], 1.0)
        self.assertAlmostEqual(m["MRR"][0], 0.5)
        ideal = 100 / math.log2(2) + 33 / math.log2(3)
        self.assertAlmostEqual(m["NDCG@5"][0], (100 / math.log2(3) + 33 / math.log2(4)) / ideal)
        self.assertTrue(math.isnan(m["R@5"][1]))                   # gig 2 has no relevant provider
        self.assertEqual((m["NDCG@10"][1], m["P@10"][1], m["MRR"][1]), (0.0, 0.0, 0.0))

    def test_condensed_drops_unjudged_before_scoring(self):
        lists = {1: [9, 5, 6]}
        self.assertAlmostEqual(ds.per_gig(lists, self.GT, [1])["MRR"][0], 0.5)
        judged = {"1": {"5": 3, "6": 1}}                           # provider 9 was never graded
        self.assertAlmostEqual(ds.per_gig(lists, self.GT, [1], judged)["MRR"][0], 1.0)

    def test_all_reported_metrics_present(self):
        self.assertEqual(set(ds.per_gig({1: [5]}, self.GT, [1])), set(ds.METRICS))
        self.assertEqual(len(ds.METRICS), 10)


class BootCI(unittest.TestCase):
    def test_ignores_nan_and_brackets_the_mean(self):
        arr = np.array([1.0, 0.0, np.nan, 1.0, 0.0, np.nan, 1.0, 1.0])
        idx = np.random.default_rng(0).integers(0, len(arr), size=(500, len(arr)))
        mean, lo, hi = ds.boot_ci(arr, idx)
        self.assertAlmostEqual(mean, 4 / 6)
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)

    def test_zero_difference_has_a_zero_interval(self):
        idx = np.random.default_rng(0).integers(0, 10, size=(100, 10))
        self.assertEqual(ds.boot_ci(np.zeros(10), idx), (0.0, 0.0, 0.0))


class LoadRows(unittest.TestCase):
    def test_label_comes_from_the_grades_argument_not_the_csv(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.csv"
            write_candidates(path, np.random.default_rng(1), n_gigs=2, n_cands=3)
            grades = {"1": {"100": 3, "101": 0}}                   # provider 102 and all of gig 2: ungraded
            rows = ds.load_rows(path, ltr.DEFAULT_FEATURES, grades)
            labels = {(r["hire_id"], r["provider_id"]): r["label"] for r in rows}
            self.assertEqual((labels[(1, 100)], labels[(1, 101)], labels[(1, 102)]), (3, 0, None))
            self.assertTrue(all(labels[(2, p)] is None for p in (100, 101, 102)))
            self.assertIsInstance(rows[0]["hire_id"], int)


class Recipes(unittest.TestCase):
    def test_model_params_are_restored(self):
        before = dict(ltr.MODEL_PARAMS)
        with ds.recipe_params("linz"):
            self.assertIs(ltr.MODEL_PARAMS["ndcg_exp_gain"], False)
        self.assertEqual(ltr.MODEL_PARAMS, before)
        ltr.MODEL_PARAMS["ndcg_exp_gain"] = False                  # a leaked linear-gain setting must not reach noce
        try:
            with ds.recipe_params("noce"):
                self.assertNotIn("ndcg_exp_gain", ltr.MODEL_PARAMS)
            self.assertIs(ltr.MODEL_PARAMS["ndcg_exp_gain"], False)
        finally:
            ltr.MODEL_PARAMS.pop("ndcg_exp_gain", None)

    def test_only_linz_zscores_the_score_features_including_tag_score(self):
        seen = {}

        def capture(rows, features, folds, seed, gt, rrf_full):
            seen[len(seen)] = [dict(r) for r in rows]
            return {}

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.csv"
            write_candidates(path, np.random.default_rng(3))
            feats = list(ltr.DEFAULT_FEATURES) + ds.TAG_FEATURES
            with unittest.mock.patch.object(ltr, "train_oof", side_effect=capture):
                ds.ranked_lists(path, feats, {}, folds=3, seed=7)
        linz_rows, noce_rows = seen[0], seen[1]                     # RECIPES order
        for key in ("dense_cosine", "tag_score"):
            per_gig = {}
            for r in linz_rows:
                per_gig.setdefault(r["hire_id"], []).append(r[key])
            self.assertTrue(all(abs(np.mean(v)) < 1e-9 for v in per_gig.values()), key)
        self.assertTrue(all(0 <= r["dense_cosine"] <= 1 for r in noce_rows))       # noce stays raw
        self.assertTrue(all(r["tag_rank"] == int(r["tag_rank"]) for r in linz_rows))  # *_rank never normalised

    def test_end_to_end_lists_cover_every_candidate_and_fusion_is_weighted(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.csv"
            write_candidates(path, np.random.default_rng(2))
            grades = {str(h): {str(100 + i): (3 if i < 2 else 0) for i in range(8)} for h in range(1, 13)}
            feats = list(ltr.DEFAULT_FEATURES) + ds.TAG_FEATURES
            lists = ds.ranked_lists(path, feats, grades, folds=3, seed=7)
            with path.open(newline="", encoding="utf-8") as f:
                gig1 = sorted((r for r in csv.DictReader(f) if r["hire_id"] == "1"), key=lambda r: int(r["rrf_rank"]))
        self.assertEqual(set(lists), {"rrf", "linz", "noce", "fused"})
        for name, per_query in lists.items():
            self.assertEqual(len(per_query), 12, name)
            for q, ids in per_query.items():
                self.assertEqual(sorted(ids), [100 + i for i in range(8)], (name, q))
        self.assertEqual(lists["rrf"][1], [int(r["provider_id"]) for r in gig1])   # rrf list = rrf_rank order
        expected = ds.weighted_rrf([lists["linz"], lists["noce"]], [0.7, 0.3])
        self.assertEqual(lists["fused"], expected)


if __name__ == "__main__":
    unittest.main()
