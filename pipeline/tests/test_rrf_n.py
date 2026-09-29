"""
Tests for retrieval_rrf.rrf_fuse_n, including parity with the existing two-way rrf_fuse.

Hand-checked three-list case, k = 60 (rank r contributes 1/(60 + r)):

    A = [1, 2, 3]   B = [2, 3, 4]   C = [3, 5]
    id 1: 1/61                      (A1)
    id 2: 1/62 + 1/61               (A2, B1)
    id 3: 1/63 + 1/62 + 1/61        (A3, B2, C1)
    id 4: 1/63                      (B3)
    id 5: 1/62                      (C2)
    order by score: 3, 2, 1, 5, 4
"""
import random
import unittest

from retrieval_rrf import rrf_fuse, rrf_fuse_n


def scored(ids):
    """A ranked list in the shape the channels return: best first, with a descending dummy score."""
    return [(i, float(len(ids) - n)) for n, i in enumerate(ids)]


class RrfFuseN(unittest.TestCase):
    def test_hand_checked_three_lists(self):
        fused = rrf_fuse_n([scored([1, 2, 3]), scored([2, 3, 4]), scored([3, 5])])
        self.assertEqual([i for i, _ in fused], [3, 2, 1, 5, 4])
        expected = {1: 1 / 61, 2: 1 / 62 + 1 / 61, 3: 1 / 63 + 1 / 62 + 1 / 61, 4: 1 / 63, 5: 1 / 62}
        for pid, score in fused:
            self.assertAlmostEqual(score, expected[pid])

    def test_two_lists_match_rrf_fuse_exactly(self):
        rng = random.Random(7)
        for _ in range(50):
            pool = list(range(1, 40))
            a = scored(rng.sample(pool, rng.randint(0, 30)))
            b = scored(rng.sample(pool, rng.randint(0, 30)))
            k, sw, dw = rng.choice([10, 60, 100]), rng.choice([1.0, 0.7, 1.5]), rng.choice([1.0, 1.3, 1.5])
            old = rrf_fuse(a, b, k=k, sparse_weight=sw, dense_weight=dw)
            new = rrf_fuse_n([a, b], k=k, weights=[sw, dw])
            self.assertEqual(dict(old), dict(new))  # bit-identical scores, same id set
            self.assertEqual([s for _, s in old], [s for _, s in new])  # same score sequence (ties may reorder)

    def test_weights_scale_each_list(self):
        a, b = scored([1, 2]), scored([2, 3])
        fused = dict(rrf_fuse_n([a, b], weights=[1.0, 2.0]))
        self.assertAlmostEqual(fused[1], 1 / 61)
        self.assertAlmostEqual(fused[2], 1 / 62 + 2 / 61)
        self.assertAlmostEqual(fused[3], 2 / 62)

    def test_zero_weight_list_changes_no_scores(self):
        a, b, c = scored([1, 2, 3]), scored([3, 1]), scored([9, 8])
        two = dict(rrf_fuse_n([a, b]))
        three = dict(rrf_fuse_n([a, b, c], weights=[1.0, 1.0, 0.0]))
        self.assertEqual({p: s for p, s in three.items() if p in two}, two)
        self.assertEqual(three[9], 0.0)  # still present, with no score

    def test_empty_list_contributes_nothing(self):
        self.assertEqual(dict(rrf_fuse_n([scored([4, 5]), []])), {4: 1 / 61, 5: 1 / 62})
        self.assertEqual(rrf_fuse_n([]), [])

    def test_single_list_keeps_its_order(self):
        ids = [7, 3, 9, 1]
        self.assertEqual([i for i, _ in rrf_fuse_n([scored(ids)])], ids)

    def test_exact_ties_break_by_first_appearance(self):
        # ids 1 and 2 swap places between the lists, so their scores are exactly equal
        fused = rrf_fuse_n([scored([1, 2]), scored([2, 1])])
        self.assertEqual(fused[0][1], fused[1][1])
        self.assertEqual([i for i, _ in fused], [1, 2])
        again = rrf_fuse_n([scored([2, 1]), scored([1, 2])])
        self.assertEqual([i for i, _ in again], [2, 1])

    def test_weight_count_must_match(self):
        with self.assertRaises(ValueError):
            rrf_fuse_n([scored([1]), scored([2])], weights=[1.0])


if __name__ == "__main__":
    unittest.main()
