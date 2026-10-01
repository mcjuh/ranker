"""
Tests for the pure-numpy parts of eval_tag_encoder.py and the prefix logic of tag_encoders.py (tiny hand-made matrices;
no model is loaded).

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v
"""
import unittest

import numpy as np

import eval_tag_encoder as ee
import tag_encoders as te


class Scores(unittest.TestCase):
    def test_centring_removes_a_hub_tag_and_gives_unit_pooled_sd(self):
        # tag 0 is a hub: high for every text; tag 1 distinguishes the texts
        sims = np.array([[0.9, 0.5, 0.1], [0.9, 0.3, 0.2], [0.9, 0.4, 0.5]])
        c = ee.centred_unit(sims)
        self.assertTrue(np.allclose(c.mean(axis=0), 0.0))
        self.assertAlmostEqual(float(c.std()), 1.0)
        self.assertEqual(int(np.argmax(sims[0])), 0)          # raw: the hub wins
        self.assertEqual(int(np.argmax(c[0])), 1)             # centred: the tag that is high for this text wins

    def test_ensemble_has_unit_sd_and_averages_before_scaling(self):
        a, b = np.array([[1.0, -1.0]]), np.array([[3.0, -3.0]])
        e = ee.ensemble([a, b])
        self.assertAlmostEqual(float(e.std()), 1.0)
        self.assertGreater(e[0, 0], 0)

    def test_top_indices_break_ties_by_tag_order(self):
        top = ee.top_indices(np.array([[0.5, 0.9, 0.5, 0.9]]), 3)
        self.assertEqual(top[0].tolist(), [1, 3, 0])


class GoldTagMetrics(unittest.TestCase):
    def test_precision_recall_hit(self):
        scores = np.array([[0.9, 0.8, 0.7, 0.1, 0.0, 0.0]])
        gold = np.array([[True, False, True, False, False, True]])
        ms = ee.gold_tag_metrics(scores, gold, ms=(1, 2, 3))
        self.assertEqual(ms["precision@1"][0], 1.0)
        self.assertEqual(ms["precision@2"][0], 0.5)
        self.assertAlmostEqual(ms["precision@3"][0], 2 / 3)
        self.assertAlmostEqual(ms["recall@3"][0], 2 / 3)
        self.assertEqual(ms["hit@1"][0], 1.0)

    def test_no_hit(self):
        ms = ee.gold_tag_metrics(np.array([[0.9, 0.1]]), np.array([[False, True]]), ms=(1,))
        self.assertEqual(ms["hit@1"][0], 0.0)
        self.assertEqual(ms["recall@1"][0], 0.0)


class WeightedVectors(unittest.TestCase):
    def test_only_top_m_above_tau_carry_weight(self):
        w = ee.weighted_vectors(np.array([[3.0, 2.0, 0.5, 4.0]]), m=3, tau=1.0)
        self.assertEqual(w[0].tolist(), [2.0, 1.0, 0.0, 3.0])      # top-3 = tags 3, 0, 1; tag 2 is outside
        w2 = ee.weighted_vectors(np.array([[3.0, 2.0, 0.5, 4.0]]), m=2, tau=1.0)
        self.assertEqual(w2[0].tolist(), [2.0, 0.0, 0.0, 3.0])

    def test_tau_zeroes_weak_tags(self):
        w = ee.weighted_vectors(np.array([[0.5, 0.2]]), m=2, tau=1.0)
        self.assertEqual(w.tolist(), [[0.0, 0.0]])

    def test_unit_rows_leaves_zero_rows(self):
        u = ee.unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
        self.assertTrue(np.allclose(u, [[0.6, 0.8], [0.0, 0.0]]))


class Diagnostics(unittest.TestCase):
    def test_slot_stats(self):
        # 4 texts, 5 tags, top-1 pick: tag 0 three times, tag 3 once
        scores = np.array([[9, 0, 0, 1, 0], [9, 0, 0, 1, 0], [9, 0, 0, 1, 0], [0, 0, 0, 9, 0]], dtype=float)
        s = ee.slot_stats(scores, m=1, top=1)
        self.assertEqual(s["slot_share_top50"], 0.75)
        self.assertEqual(s["tags_used"], 2)
        self.assertEqual(s["tags_never_picked"], 3)
        self.assertEqual(s["max_tag_freq"], 3)

    def test_auc(self):
        self.assertEqual(ee.auc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([True, True, False, False])), 1.0)
        self.assertEqual(ee.auc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([True, True, False, False])), 0.0)
        self.assertEqual(ee.auc(np.array([0.5, 0.5]), np.array([True, False])), 0.5)     # a tie counts half
        self.assertTrue(np.isnan(ee.auc(np.array([0.5, 0.4]), np.array([True, True]))))

    def test_rank_pool_breaks_ties_with_the_key_not_the_position(self):
        order = ee.rank_pool(np.array([0.0, 0.0, 1.0]), np.array([0.9, 0.1, 0.5]))
        self.assertEqual(order.tolist(), [2, 1, 0])


class RoleRecipes(unittest.TestCase):
    ROLES = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])             # 3 role vectors in 2-d
    GOLD = np.array([[True, False, False], [False, True, False], [True, True, False]])   # tag 2 is carried by no role

    def test_prototype_blends_title_with_the_carrying_roles_and_keeps_orphan_titles(self):
        titles = np.array([[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])
        proto = ee.tag_prototypes(titles, self.ROLES, self.GOLD, a=0.5)
        self.assertTrue(np.allclose(proto[2], titles[2]))                  # no role carries tag 2: title only
        roles_u = ee.unit_rows(self.ROLES)
        want0 = 0.5 * titles[0] + 0.5 * ee.unit_rows((roles_u[0] + roles_u[2])[None])[0]   # tag 0: roles 0 and 2
        self.assertTrue(np.allclose(proto[0], want0))
        self.assertTrue(np.allclose(ee.tag_prototypes(titles, self.ROLES, self.GOLD, a=1.0), titles))

    def test_knn_inherits_the_tag_set_of_the_nearest_role(self):
        s = ee.role_knn_scores(np.array([[1.0, 0.01]]), self.ROLES, self.GOLD, tau=0.01)
        self.assertAlmostEqual(s[0, 0], 1.0, places=3)                     # nearest role is role 0, which has tag 0 only
        self.assertAlmostEqual(s[0, 1], 0.0, places=3)
        self.assertEqual(s[0, 2], 0.0)
        flat = ee.role_knn_scores(np.array([[1.0, 0.0]]), self.ROLES, self.GOLD, tau=100.0)
        self.assertTrue(np.allclose(flat[0], self.GOLD.mean(axis=0), atol=1e-2))   # a hot softmax averages every role

    def test_knn_rows_are_probability_weighted(self):
        s = ee.role_knn_scores(np.random.default_rng(0).normal(size=(4, 2)), self.ROLES, self.GOLD, tau=0.1)
        self.assertTrue(((s >= 0) & (s <= 1)).all())

    def test_parse_name(self):
        self.assertEqual(ee.parse_name("bge-large"), ("bge-large", "title", []))
        self.assertEqual(ee.parse_name("mxbai:proto0.5"), ("mxbai", "proto", [0.5]))
        self.assertEqual(ee.parse_name("mxbai:mix1.0_0.05"), ("mxbai", "mix", [1.0, 0.05]))
        self.assertEqual(ee.parse_name("mxbai:head0.1"), ("mxbai", "head", [0.1]))
        with self.assertRaises(ValueError):
            ee.parse_name("mxbai:bogus1")


@unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "the supervised head needs torch")
class SupervisedHead(unittest.TestCase):
    TITLES = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.2], [0.2, 0.0, 1.0]])     # tag j's title leans on axis j
    ROLES = np.array([[0.0, 1.0, 0.5], [0.1, 0.9, 0.6]])                        # both roles lean on axes 1 and 2
    GOLD = np.array([[True, False, False], [True, False, False]])               # ...yet both carry tag 0

    def cos(self, tags, j):
        return float((ee.unit_rows(self.ROLES) @ ee.unit_rows(tags).T)[:, j].mean())

    def test_a_carried_tag_moves_toward_its_roles_and_a_large_penalty_keeps_the_titles(self):
        free = ee.head_tags(self.TITLES, ee.fit_tag_head(self.TITLES, self.ROLES, self.GOLD, lam=0.0))
        held = ee.head_tags(self.TITLES, ee.fit_tag_head(self.TITLES, self.ROLES, self.GOLD, lam=1e4))
        self.assertGreater(self.cos(free, 0), self.cos(self.TITLES, 0) + 0.2)
        self.assertTrue(np.allclose(held, ee.unit_rows(self.TITLES), atol=1e-2))
        self.assertTrue(np.allclose(np.linalg.norm(free, axis=1), 1.0))

    def test_roles_without_gold_tags_are_ignored_and_the_fit_is_deterministic(self):
        gold = np.vstack([self.GOLD, np.zeros((1, 3), dtype=bool)])
        roles = np.vstack([self.ROLES, [[5.0, -3.0, 1.0]]])
        a = ee.fit_tag_head(self.TITLES, roles, gold, lam=0.01)
        self.assertTrue(np.allclose(a, ee.fit_tag_head(self.TITLES, self.ROLES, self.GOLD, lam=0.01), atol=1e-5))
        self.assertTrue(np.array_equal(a, ee.fit_tag_head(self.TITLES, roles, gold, lam=0.01)))


class HubnessCorrection(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.sims = rng.normal(0, 0.05, size=(200, 6)) + 0.3
        self.sims[:, 0] += 0.1                                   # tag 0: high everywhere (centring already removes this)
        self.sims[:, 1] = 0.3 + rng.normal(0, 0.15, size=200)    # tag 1: ordinary mean, 3x the spread: a fat upper tail

    def slots(self, scores, m=2):
        return np.bincount(ee.top_indices(scores, m).ravel(), minlength=scores.shape[1])

    def test_center_is_the_current_recipe_and_dsm_with_a_huge_temperature_is_centring(self):
        want = ee.centred_unit(self.sims)
        self.assertTrue(np.allclose(ee.global_unit(ee.hub_correct(self.sims, "center")), want))
        self.assertTrue(np.allclose(ee.global_unit(ee.hub_correct(self.sims, "dsm", 1e6)), want, atol=1e-3))

    def test_a_per_tag_shift_is_cancelled_by_a_later_centring_but_not_by_global_unit(self):
        shifted = self.sims + np.linspace(0, 0.2, 6)
        self.assertTrue(np.allclose(ee.centred_unit(shifted), ee.centred_unit(self.sims)))
        self.assertFalse(np.allclose(ee.global_unit(shifted), ee.global_unit(self.sims)))

    def test_dsm_penalises_the_high_tail_that_centring_keeps(self):
        centred = self.slots(ee.centred_unit(self.sims))
        for kind, arg in (("dsm", 0.5), ("csls", 10)):
            fixed = self.slots(ee.global_unit(ee.hub_correct(self.sims, kind, arg)))
            self.assertLess(fixed[1], centred[1], kind)           # the tail tag is picked less
            self.assertEqual(fixed.sum(), centred.sum())

    def test_csls_uses_the_mean_of_the_k_closest_texts(self):
        s = np.array([[0.9, 0.1], [0.5, 0.3], [0.1, 0.2]])
        got = ee.hub_correct(s, "csls", 2)
        self.assertTrue(np.allclose(got[:, 0], s[:, 0] - 0.5 * 0.7))        # tag 0: mean of 0.9 and 0.5
        self.assertTrue(np.allclose(got[:, 1], s[:, 1] - 0.5 * 0.25))       # tag 1: mean of 0.3 and 0.2

    def test_names(self):
        self.assertEqual(ee.split_hub("qwen3-0.6b:head10~dsm0.5"), ("qwen3-0.6b:head10", ("dsm", 0.5)))
        self.assertEqual(ee.split_hub("mxbai~csls"), ("mxbai", ("csls", None)))
        self.assertEqual(ee.split_hub("mxbai"), ("mxbai", None))
        self.assertEqual(ee.split_hub("mxbai~center"), ("mxbai", ("center", None)))
        with self.assertRaises(ValueError):
            ee.split_hub("mxbai~bogus")
        with self.assertRaises(ValueError):
            ee.hub_correct(self.sims, "bogus")


class Encoders(unittest.TestCase):
    def test_prefixes_per_role(self):
        spec = te.get_spec("e5-large")
        self.assertEqual(te.prefixed(spec, ["x"], "query"), ["query: x"])
        self.assertEqual(te.prefixed(spec, ["x"], "doc"), ["passage: x"])
        self.assertEqual(te.prefixed(te.get_spec("gte-large"), ["x"], "doc"), ["x"])

    def test_mxbai_matches_the_dense_channel_prefix(self):
        from retrieval_dense import QUERY_PREFIX_GENERIC
        self.assertEqual(te.get_spec("mxbai").query_prefix, QUERY_PREFIX_GENERIC)

    def test_unknown_role_and_encoder(self):
        with self.assertRaises(ValueError):
            te.prefixed(te.get_spec("mxbai"), ["x"], "passage")
        with self.assertRaises(KeyError):
            te.get_spec("nope")


if __name__ == "__main__":
    unittest.main()
