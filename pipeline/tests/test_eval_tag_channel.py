"""
Tests for the pieces of eval_tag_channel.py that decide what the numbers mean: tie handling, class
collapsing, track credit, query generation, and rank-based fusion. Uses a 5-role synthetic taxonomy.

    roles:  r1 {1,2} track 1    r2 {1,2} track 1 (twin of r1)    r3 {1,2,3} track 2
            r4 {4,5} track 2    r5 {1,6} track 1
    classes (by smallest role id): A = {r1, r2}, B = {r3}, C = {r4}, D = {r5}

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v
"""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from eval_tag_channel import (CONDITIONS, GIG_METRICS, Ranked, RoleWorld, cached_embeddings, ci, class_means,
                              gig_metrics, hit_metrics, judged_share, lsa_basis, rng_for, rrf_matrix,
                              unique_positives)
from greygigz import Taxonomy


def make_world(seed=7):
    role_tags = {1: {1, 2}, 2: {1, 2}, 3: {1, 2, 3}, 4: {4, 5}, 5: {1, 6}}
    role_track = {1: 1, 2: 1, 3: 2, 4: 2, 5: 1}
    tax = Taxonomy(
        tracks={1: {"name": "T1", "service_id": 1}, 2: {"name": "T2", "service_id": 1}},
        roles={r: {"name": f"R{r}", "description": "", "track_id": t} for r, t in role_track.items()},
        tags={t: f"tag{t}" for t in range(1, 7)},
        role_tags={r: frozenset(t) for r, t in role_tags.items()},
        levels={(r, t): 2 for r, tags in role_tags.items() for t in tags},
    )
    return RoleWorld(tax, seed)


class ClassCollapse(unittest.TestCase):
    def test_classes_and_tracks(self):
        w = make_world()
        self.assertEqual(w.reps, [1, 3, 4, 5])
        self.assertEqual(w.n_classes, 4)
        self.assertEqual(w.class_track.tolist(), [1, 2, 2, 1])

    def test_class_score_is_the_max_over_members(self):
        w = make_world()
        S = np.array([[0.9, 0.7, 0.4, 0.3, 0.1]])
        np.testing.assert_allclose(w.class_scores(S), [[0.9, 0.4, 0.3, 0.1]])

    def test_split_is_disjoint_and_covers_every_class(self):
        w = make_world()
        dev, test = w.split_of_class["dev"], w.split_of_class["test"]
        self.assertFalse((dev & test).any())
        self.assertTrue((dev | test).all())


class HitMetrics(unittest.TestCase):
    def setUp(self):
        self.w = make_world()

    def metrics(self, rows, qcls, **kw):
        return hit_metrics(self.w, np.array(rows), np.array(qcls), **kw)

    def test_twin_roles_do_not_compete_but_a_tied_class_does(self):
        # truth class A (r1, r2) ties with class B (r3) at 0.9: it gets the top slot half the time
        m = self.metrics([[0.9, 0.9, 0.9, 0.5, 0.1]], [0])
        self.assertAlmostEqual(m["role@1"][0], 0.5)
        self.assertAlmostEqual(m["role@5"][0], 1.0)  # 4 classes, all inside the top 5

    def test_strictly_beaten_truth(self):
        # truth class D (r5) is 4th of 4 classes
        m = self.metrics([[0.9, 0.9, 0.8, 0.7, 0.6]], [3])
        self.assertEqual(m["role@1"][0], 0.0)
        self.assertEqual(m["role@5"][0], 1.0)

    def test_everything_tied(self):
        m = self.metrics([[0.5] * 5], [2])  # 4 classes tied: a 1-in-4 chance at rank 1
        self.assertAlmostEqual(m["role@1"][0], 0.25)
        self.assertEqual(m["role@5"][0], 1.0)

    def test_a_zero_score_truth_is_not_retrieved(self):
        m = self.metrics([[0.5, 0.5, 0.0, 0.0, 0.0]], [1])
        self.assertEqual((m["role@1"][0], m["role@5"][0], m["role@10"][0]), (0.0, 0.0, 0.0))
        kept = self.metrics([[0.5, 0.5, 0.0, 0.0, 0.0]], [1], drop_nonpositive=False)
        self.assertAlmostEqual(kept["role@1"][0], 0.0)  # one class scores higher, so no chance at rank 1
        self.assertEqual(kept["role@5"][0], 1.0)        # and it is tied with the other two zeros for ranks 2-4

    def test_track_credit_uses_score_mass_of_the_top_roles(self):
        # track 1 holds r1 + r2 + r5 = 0.9 + 0.9 + 0.1, track 2 holds r3 + r4 = 0.9 + 0.5
        m = self.metrics([[0.9, 0.9, 0.9, 0.5, 0.1]], [0])
        self.assertEqual((m["track@1"][0], m["track@3"][0]), (1.0, 1.0))
        # for a class-C query (truth track 2) the same scores put track 1 first: track@1 misses, track@3 hits
        m = self.metrics([[0.9, 0.9, 0.9, 0.5, 0.1]], [2])
        self.assertEqual((m["track@1"][0], m["track@3"][0]), (0.0, 1.0))

    def test_no_positive_scores_means_no_track(self):
        m = self.metrics([[0.0] * 5], [0])
        self.assertEqual((m["track@1"][0], m["track@3"][0], m["track@1_samename"][0]), (0.0, 0.0, 0.0))

    def test_precomputed_class_scores_are_used_for_the_funnel(self):
        # masking r1/r2 out of class A leaves A scoring 0 -> not retrieved
        S = np.array([[0.9, 0.9, 0.4, 0.3, 0.1]])
        masked = np.array([[0.0, 0.4, 0.3, 0.1]])
        m = hit_metrics(self.w, S, np.array([0]), class_scores=masked)
        self.assertEqual(m["role@5"][0], 0.0)


class Aggregation(unittest.TestCase):
    def test_class_means_average_queries_within_a_class(self):
        ids, cm = class_means({"x": np.array([1.0, 0.0, 1.0, 1.0])}, np.array([5, 5, 9, 9]))
        self.assertEqual(ids.tolist(), [5, 9])
        np.testing.assert_allclose(cm["x"], [0.5, 1.0])

    def test_ci_brackets_the_mean_and_collapses_when_constant(self):
        rng = np.random.default_rng(0)
        cm = rng.random(50)
        boot = rng.integers(0, 50, size=(200, 50))
        mean, lo, hi = ci(cm, boot)
        self.assertAlmostEqual(mean, cm.mean())
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)
        const = ci(np.full(10, 0.3), rng.integers(0, 10, size=(50, 10)))  # no spread across classes, no interval
        self.assertTrue(all(abs(v - 0.3) < 1e-12 for v in const))


class Queries(unittest.TestCase):
    def setUp(self):
        self.w = make_world()
        idx = self.w.index()
        self.df = dict(zip(idx.tag_ids, idx.df))
        self.vocab = np.array(idx.tag_ids)

    def gen(self, cond):
        return self.w.make_queries(cond, self.df, self.vocab)

    def test_random_queries_come_from_the_role_and_repeat_exactly(self):
        cond = ("t", 2, "random", 0, 3, "vocab")
        queries, qcls = self.gen(cond)
        self.assertEqual(len(queries), 4 * 3)
        for q, c in zip(queries, qcls):
            self.assertEqual(len(q), 2)
            self.assertTrue(set(q) <= self.w.tax.role_tags[self.w.reps[c]])
        again, again_cls = self.gen(cond)  # seeded: identical on every call
        self.assertEqual(again, queries)
        self.assertTrue(np.array_equal(again_cls, qcls))

    def test_a_role_with_fewer_tags_than_asked_contributes_all_of_them(self):
        queries, qcls = self.gen(("t", 5, "random", 0, 1, "vocab"))
        by_class = {c: q for q, c in zip(queries, qcls)}
        self.assertEqual(sorted(by_class[2]), [4, 5])   # r4 has only two tags

    def test_wrong_tags_are_not_the_roles_own(self):
        queries, qcls = self.gen(("t", 3, "random", 1, 4, "vocab"))
        for q, c in zip(queries, qcls):
            own = self.w.tax.role_tags[self.w.reps[c]]
            self.assertEqual(len([t for t in q if t not in own]), 1)

    def test_sibling_wrong_tags_come_from_the_same_track(self):
        queries, qcls = self.gen(("t", 3, "random", 1, 4, "track"))
        for q, c in zip(queries, qcls):
            own = self.w.tax.role_tags[self.w.reps[c]]
            (wrong,) = [t for t in q if t not in own]
            self.assertIn(wrong, self.w.track_tags[int(self.w.class_track[c])])

    def test_generic_and_specific_pick_by_document_frequency(self):
        # r5 (class 3) = {1, 6}: tag 1 is on four roles, tag 6 on one
        def for_class_3(cond):
            queries, qcls = self.gen(cond)
            return [q for q, c in zip(queries, qcls) if c == 3]
        self.assertEqual(for_class_3(("s", 1, "specific", 0, 1, "vocab")), [[6]])
        self.assertEqual(for_class_3(("g", 1, "generic", 0, 1, "vocab")), [[1]])

    def test_every_declared_condition_is_well_formed(self):
        names = [c[0] for c in CONDITIONS]
        self.assertEqual(len(names), len(set(names)))
        for name, size, mode, n_wrong, draws, wrong_from in CONDITIONS:
            self.assertIn(mode, ("random", "generic", "specific"))
            self.assertIn(wrong_from, ("vocab", "track"))
            self.assertLess(n_wrong, size)


class Scorers(unittest.TestCase):
    def test_rrf_matrix_gives_tied_columns_equal_fused_scores(self):
        a = np.array([[0.9, 0.9, 0.5, 0.0]])   # columns 0 and 1 tie for first; column 3 is dropped
        b = np.array([[0.1, 0.1, 0.8, 0.3]])
        fused = rrf_matrix([a, b], [True, False], k=60)
        self.assertEqual(fused[0, 0], fused[0, 1])
        self.assertAlmostEqual(fused[0, 0], 1 / 61 + 1 / 63)    # ranks: a -> 1 (tied), b -> 3 (tied)
        self.assertAlmostEqual(fused[0, 2], 1 / 63 + 1 / 61)    # a -> 3, b -> 1
        self.assertAlmostEqual(fused[0, 3], 1 / 62)             # a contributes nothing; b -> 2

    def test_lsa_basis_is_orthonormal(self):
        w = make_world()
        V = lsa_basis(w.index(), 4)
        np.testing.assert_allclose(V.T @ V, np.eye(V.shape[1]), atol=1e-9)

    def test_rng_is_reproducible_and_purpose_specific(self):
        self.assertEqual(rng_for(7, "a").integers(0, 10**9), rng_for(7, "a").integers(0, 10**9))
        self.assertNotEqual(rng_for(7, "a").integers(0, 10**9), rng_for(7, "b").integers(0, 10**9))


class PartBHelpers(unittest.TestCase):
    def test_ranked_top_and_pairs(self):
        r = Ranked([7, 3, 9], [0.9, 0.5, 0.1])
        self.assertEqual(r.top(2), {7, 3})
        self.assertEqual(r.pairs(), [(7, 0.9), (3, 0.5), (9, 0.1)])

    def test_unique_positives_are_relevant_and_found_by_no_other_channel(self):
        target = Ranked([1, 2, 3, 4], [4, 3, 2, 1])
        others = [Ranked([2, 8, 9], [3, 2, 1]), Ranked([3, 5, 6], [3, 2, 1])]
        # at K=3 the others hold {2, 8, 9} and {3, 5, 6}; the target's {1, 2, 3} minus those is {1}
        self.assertEqual(unique_positives(target, others, positives={1, 3, 7}, k=3), {1})
        self.assertEqual(unique_positives(target, others, positives={2, 3}, k=3), set())   # both found elsewhere
        self.assertEqual(unique_positives(target, others, positives={4}, k=3), set())      # outside the target's top 3
        self.assertEqual(unique_positives(target, [], positives={1, 4}, k=4), {1, 4})

    def test_gig_metrics_hand_checked(self):
        # provider 5 is graded relevant (100), provider 6 is not (33 < 40); the ranking finds 5 second
        gt = {"1": {"5": 100, "6": 33}}
        m = gig_metrics({"1": Ranked([9, 5, 6], [3, 2, 1])}, ["1"], gt)
        self.assertEqual((m["R@10"][0], m["R@50"][0], m["R@100"][0]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(m["MRR"][0], 0.5)
        # evaluate.py uses the raw 0-100 grade as the NDCG gain, so the 33 counts too (only R and MRR threshold at 40)
        ideal = 100 / math.log2(2) + 33 / math.log2(3)
        self.assertAlmostEqual(m["NDCG@10"][0], (100 / math.log2(3) + 33 / math.log2(4)) / ideal)

    def test_cutoffs_at_5_10_20(self):
        gt = {"1": {"500": 100}}
        filler = list(range(1, 20))
        at = lambda rank: {"1": Ranked(filler[: rank - 1] + [500] + filler[rank - 1:], list(range(20, 0, -1)))}
        m = gig_metrics(at(7), ["1"], gt)              # relevant provider at rank 7
        self.assertEqual(set(m), set(GIG_METRICS))
        self.assertEqual((m["P@5"][0], m["P@10"][0], m["P@20"][0]), (0.0, 0.1, 0.05))
        self.assertEqual((m["NDCG@5"][0], m["R@10"][0], m["R@20"][0]), (0.0, 1.0, 1.0))
        self.assertAlmostEqual(m["NDCG@10"][0], 1 / math.log2(8))
        self.assertAlmostEqual(m["NDCG@20"][0], 1 / math.log2(8))
        m = gig_metrics(at(12), ["1"], gt)             # rank 12: outside the top 10, inside the top 20
        self.assertEqual((m["R@10"][0], m["R@20"][0], m["NDCG@10"][0]), (0.0, 1.0, 0.0))
        self.assertAlmostEqual(m["NDCG@20"][0], 1 / math.log2(13))
        m = gig_metrics(at(3), ["1"], gt)
        self.assertAlmostEqual(m["NDCG@5"][0], 1 / math.log2(4))

    def test_judged_share_of_top_k(self):
        ranking = {"1": Ranked([1, 2, 3, 4], [4, 3, 2, 1]), "2": Ranked([1, 2], [2, 1])}
        judged = {"1": {"1": 0, "2": 1}, "2": {"1": 0, "2": 0}}
        self.assertAlmostEqual(judged_share(ranking, ["1"], judged, 4), 0.5)
        self.assertAlmostEqual(judged_share(ranking, ["1", "2"], judged, 4), (0.5 + 1.0) / 2)
        self.assertAlmostEqual(judged_share(ranking, ["1"], judged, 2), 1.0)

    def test_condensed_lists_drop_unjudged_providers(self):
        gt = {"1": {"5": 100}}
        ranking = {"1": Ranked([9, 5], [2, 1])}       # 9 was never graded
        self.assertAlmostEqual(gig_metrics(ranking, ["1"], gt)["MRR"][0], 0.5)
        judged = {"1": {"5": 3, "6": 1}}
        self.assertAlmostEqual(gig_metrics(ranking, ["1"], gt, judged)["MRR"][0], 1.0)

    def test_cached_embeddings_reads_by_text_hash_and_never_encodes(self):
        texts = ["alpha", "beta", "gamma"]
        vecs = np.arange(9, dtype=float).reshape(3, 3)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d / "c.npy", vecs[[2, 0]])          # cache holds gamma and alpha only
            (d / "c.keys.json").write_text(json.dumps(
                [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in ("gamma", "alpha")]))
            np.testing.assert_allclose(cached_embeddings("c", ["alpha", "gamma"], d), vecs[[0, 2]])
            with self.assertRaises(SystemExit):
                cached_embeddings("c", texts, d)         # beta is missing


if __name__ == "__main__":
    unittest.main()
