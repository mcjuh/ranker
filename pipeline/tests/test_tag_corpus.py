"""
Tests for tag_corpus.top_tags (hand-made 3-d embeddings) and tag_channel.TagChannel (tiny tag files).

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v
"""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tag_channel import TagChannel, load_tags
from tag_corpus import top_tags

TAG_IDS = [501, 502, 503]
TAG_VECS = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class TopTags(unittest.TestCase):
    def test_orders_by_cosine_and_reports_it(self):
        # [3, 4, 0] has cosines 0.6 / 0.8 / 0 with the three axes: unit-normalising it must not change the order
        (tags,) = top_tags(np.array([[3.0, 4.0, 0.0]]), TAG_VECS, TAG_IDS, m=3)
        self.assertEqual([t for t, _ in tags], [502, 501, 503])
        self.assertAlmostEqual(tags[0][1], 0.8)
        self.assertAlmostEqual(tags[1][1], 0.6)
        self.assertAlmostEqual(tags[2][1], 0.0)

    def test_cosine_ignores_vector_length(self):
        a = top_tags(np.array([[1.0, 2.0, 0.0]]), TAG_VECS, TAG_IDS, m=2)
        b = top_tags(np.array([[10.0, 20.0, 0.0]]), TAG_VECS * 7, TAG_IDS, m=2)
        self.assertEqual([t for t, _ in a[0]], [t for t, _ in b[0]])
        self.assertAlmostEqual(a[0][0][1], 2 / math.sqrt(5))

    def test_m_truncates_and_is_capped_at_the_vocabulary(self):
        self.assertEqual(len(top_tags(np.array([[1.0, 1.0, 1.0]]), TAG_VECS, TAG_IDS, m=2)[0]), 2)
        self.assertEqual(len(top_tags(np.array([[1.0, 1.0, 1.0]]), TAG_VECS, TAG_IDS, m=99)[0]), 3)

    def test_equal_cosines_keep_tag_order(self):
        (tags,) = top_tags(np.array([[1.0, 1.0, 1.0]]), TAG_VECS, TAG_IDS, m=3)
        self.assertEqual([t for t, _ in tags], [501, 502, 503])

    def test_one_row_per_text_and_zero_vector_is_safe(self):
        rows = top_tags(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), TAG_VECS, TAG_IDS, m=1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0][0], 501)
        self.assertEqual(rows[1][0][1], 0.0)


def _payload(tagged):
    return {"model": "test", "top_m": 3, "tags": {str(i): [[t, c] for t, c in tags] for i, tags in tagged.items()}}


class TagChannelTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        providers = {1: [(501, .9), (502, .8), (503, .1)], 2: [(502, .9), (503, .8)], 3: [(503, .9)]}
        hirers = {7: [(501, .9), (502, .5), (503, .1)], 8: [(503, .9)]}
        (self.dir / "tags_providers.json").write_text(json.dumps(_payload(providers)))
        (self.dir / "tags_hirers.json").write_text(json.dumps(_payload(hirers)))

    def test_load_tags_keeps_the_best_m_and_int_keys(self):
        self.assertEqual(load_tags(self.dir / "tags_providers.json", 2), {1: [501, 502], 2: [502, 503], 3: [503]})

    def test_rank_uses_the_gigs_top_m_tags_only(self):
        channel = TagChannel(self.dir, top_m_provider=3, top_m_hirer=1)  # gig 7 queries with 501 alone
        self.assertEqual([p for p, _ in channel.rank(7)], [1])
        channel = TagChannel(self.dir, top_m_provider=3, top_m_hirer=2)  # gig 7 queries with 501 and 502
        self.assertEqual([p for p, _ in channel.rank(7)], [1, 2])

    def test_rank_returns_scored_pairs_and_drops_providers_without_evidence(self):
        ranked = TagChannel(self.dir, top_m_provider=3, top_m_hirer=3).rank(8)  # 503: providers 1, 2 and 3
        self.assertEqual(sorted(p for p, _ in ranked), [1, 2, 3])
        self.assertTrue(all(score > 0 for _, score in ranked))
        # keeping one tag per provider leaves only provider 3 holding tag 503, so only it has evidence
        narrow = TagChannel(self.dir, top_m_provider=1, top_m_hirer=1).rank(8)
        self.assertEqual([p for p, _ in narrow], [3])

    def test_missing_tag_files_explain_what_to_run(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(FileNotFoundError, "tag_corpus.py"):
                TagChannel(Path(empty))


if __name__ == "__main__":
    unittest.main()
