"""
Unit tests for retrieval_tagbm25.TagBM25 on a small matrix whose numbers are worked out by hand.

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v

Fixture (doc ids and tag ids are deliberately non-contiguous, so an implementation that confused
an ID with a row/column position would fail):

    doc 10 = {101, 102}              |d| = 2
    doc 20 = {101, 102, 103, 104}    |d| = 4
    doc 30 = {103, 104}              |d| = 2
    doc 40 = {105}                   |d| = 1

    N = 4, sum of |d| = 9, avgdl = 2.25
    df: tags 101..104 appear in 2 docs, tag 105 in 1 doc
    idf(df=2) = ln(1 + 2.5/2.5) = ln 2
    idf(df=1) = ln(1 + 3.5/1.5) = ln(10/3)

    With k1 = 1.2, b = 0.75:  c_d = 2.2 / (1 + 1.2 * (0.25 + 0.75 * |d| / 2.25))
        |d| = 2 -> 1 + 1.2 * 0.91667 = 2.1  -> c = 2.2/2.1
        |d| = 4 -> 1 + 1.2 * 1.58333 = 2.9  -> c = 2.2/2.9
        |d| = 1 -> 1 + 1.2 * 0.58333 = 1.7  -> c = 2.2/1.7
"""
import math
import unittest

import numpy as np

from retrieval_tagbm25 import TagBM25

LN2 = math.log(2)
IDF_RARE = math.log(10 / 3)
C2, C4, C1 = 2.2 / 2.1, 2.2 / 2.9, 2.2 / 1.7

DOC_IDS = [10, 20, 30, 40]
DOC_TAGS = [{101, 102}, {101, 102, 103, 104}, {103, 104}, {105}]
LEVELS = {
    (10, 101): 2, (10, 102): 4,
    (20, 101): 2, (20, 102): 6, (20, 103): 4, (20, 104): 2,
    (30, 103): 4, (30, 104): 2,
    (40, 105): 6,
}


def build(**kwargs):
    return TagBM25(DOC_IDS, DOC_TAGS, **kwargs)


class TagBM25HandChecked(unittest.TestCase):
    def test_idf_and_length_factor(self):
        idx = build()
        col = idx.tag_ids.index
        for t in (101, 102, 103, 104):
            self.assertAlmostEqual(idx.idf[col(t)], LN2)
        self.assertAlmostEqual(idx.idf[col(105)], IDF_RARE)
        self.assertAlmostEqual(idx.avgdl, 2.25)
        np.testing.assert_allclose(idx.c, [C2, C4, C2, C1])

    def test_scores_for_three_tag_query(self):
        # q = {101, 102, 103}: doc 10 overlaps on 2 tags, doc 20 on 3, doc 30 on 1, doc 40 on none
        s = build().scores([101, 102, 103])
        np.testing.assert_allclose(s, [2 * LN2 * C2, 3 * LN2 * C4, LN2 * C2, 0.0])

    def test_rank_orders_descending_and_drops_zero_scores(self):
        ranked = build().rank([101, 102, 103])
        self.assertEqual([d for d, _ in ranked], [20, 10, 30])  # doc 40 has no evidence: absent
        self.assertEqual([s for _, s in ranked], sorted((s for _, s in ranked), reverse=True))
        self.assertAlmostEqual(ranked[0][1], 3 * LN2 * C4)

    def test_rare_tag_scores_only_its_document(self):
        self.assertEqual(len(build().rank([105])), 1)
        (doc, score), = build().rank([105])
        self.assertEqual(doc, 40)
        self.assertAlmostEqual(score, C1 * IDF_RARE)

    def test_top_k_truncates(self):
        self.assertEqual([d for d, _ in build().rank([101, 102, 103], top_k=2)], [20, 10])

    def test_b_zero_removes_length_normalisation(self):
        # q = {101, 102} hits docs 10 (|d|=2) and 20 (|d|=4) on the same two tags.
        flat = build(b=0.0).rank([101, 102])
        self.assertAlmostEqual(flat[0][1], 2 * LN2)  # c == 1 exactly at b = 0
        self.assertAlmostEqual(flat[1][1], 2 * LN2)
        self.assertEqual([d for d, _ in flat], [10, 20])  # exact tie -> document order
        normed = build(b=0.75).rank([101, 102])
        self.assertAlmostEqual(normed[0][1], 2 * LN2 * C2)  # the shorter doc wins
        self.assertAlmostEqual(normed[1][1], 2 * LN2 * C4)

    def test_b_one_and_k1_change_the_constant_as_derived(self):
        # b = 1: c = 2.2 / (1 + 1.2 * |d| / 2.25); |d| = 2 -> 2.2 / (1 + 1.2 * 2/2.25)
        s = build(b=1.0).scores([101, 102])
        self.assertAlmostEqual(s[0], 2 * LN2 * 2.2 / (1 + 1.2 * 2 / 2.25))
        # k1 = 0 makes c == 1 for every document whatever b is
        np.testing.assert_allclose(build(k1=0.0, b=0.75).c, np.ones(4))


class TagBM25Queries(unittest.TestCase):
    def test_unknown_and_repeated_ids_are_ignored(self):
        idx = build()
        self.assertEqual(idx.rank([101, 999]), idx.rank([101]))
        self.assertEqual(idx.rank([101, 101, 102]), idx.rank([101, 102]))
        self.assertEqual(idx.rank([999]), [])
        self.assertEqual(idx.rank([]), [])

    def test_matching_is_by_id_not_position(self):
        relabel = {101: 105, 102: 104, 103: 103, 104: 102, 105: 101}
        permuted = TagBM25(DOC_IDS, [{relabel[t] for t in tags} for tags in DOC_TAGS])
        q = [101, 102, 103]
        np.testing.assert_allclose(permuted.scores([relabel[t] for t in q]), build().scores(q))
        self.assertFalse(np.allclose(permuted.scores(q), build().scores(q)))  # old IDs mean something else

    def test_batch_equals_single_queries(self):
        idx = build()
        queries = [[101, 102, 103], [105], [999], [], [103, 104]]
        batch = idx.score_matrix(queries)
        self.assertEqual(batch.shape, (5, 4))
        for row, q in zip(batch, queries):
            np.testing.assert_allclose(row, idx.scores(q))


class TagBM25Baselines(unittest.TestCase):
    Q = [101, 102, 103]

    def test_coverage(self):
        np.testing.assert_allclose(build().scores(self.Q, "coverage"), [2 / 3, 1.0, 1 / 3, 0.0])

    def test_idf_overlap_has_no_length_normalisation(self):
        np.testing.assert_allclose(build().scores(self.Q, "idf_overlap"), [2 * LN2, 3 * LN2, LN2, 0.0])

    def test_idf_overlap_is_bm25_with_b_zero(self):
        np.testing.assert_allclose(build(b=0.0).scores(self.Q), build().scores(self.Q, "idf_overlap"))

    def test_sum_levels(self):
        np.testing.assert_allclose(build(levels=LEVELS).scores(self.Q, "sum_levels"), [6, 12, 4, 0])

    def test_sum_levels_needs_levels(self):
        with self.assertRaises(ValueError):
            build().scores(self.Q, "sum_levels")

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            build().scores(self.Q, "tfidf")


class TagBM25EdgeCases(unittest.TestCase):
    def test_idf_stays_positive_when_a_tag_is_on_every_doc(self):
        idx = TagBM25([1, 2, 3], [{7}, {7, 8}, {7, 9}])
        self.assertAlmostEqual(idx.idf[idx.tag_ids.index(7)], math.log1p(0.5 / 3.5))
        self.assertGreater(idx.idf[idx.tag_ids.index(7)], 0)

    def test_document_without_tags_never_ranks(self):
        idx = TagBM25([1, 2], [set(), {5}])
        self.assertEqual([d for d, _ in idx.rank([5])], [2])

    def test_all_documents_empty_does_not_divide_by_zero(self):
        idx = TagBM25([1], [set()])
        self.assertEqual(idx.rank([5]), [])

    def test_from_pairs_matches_direct_construction(self):
        pairs = [(d, t) for d, tags in zip(DOC_IDS, DOC_TAGS) for t in sorted(tags)]
        a = TagBM25.from_pairs(pairs)
        np.testing.assert_allclose(a.scores([101, 102, 103]), build().scores([101, 102, 103]))
        b = TagBM25.from_pairs(pairs, doc_ids=[40, 10, 99])  # 99 has no pairs, order follows doc_ids
        self.assertEqual(b.doc_ids, [40, 10, 99])
        self.assertEqual(b.dl.tolist(), [1, 2, 0])

    def test_parameter_validation(self):
        for kwargs in ({"b": 1.5}, {"b": -0.1}, {"k1": -1.0}):
            with self.assertRaises(ValueError):
                build(**kwargs)
        with self.assertRaises(ValueError):
            TagBM25([1, 2], [{1}])


if __name__ == "__main__":
    unittest.main()
