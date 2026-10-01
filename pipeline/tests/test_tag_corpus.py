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
from tag_corpus import hubness_stats, tag_similarities, top_tags

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


class Hubness(unittest.TestCase):
    """Tag 501 is a hub: every text is fairly close to it. Raw cosine picks it first for everyone; z-scoring each
    tag's column over the corpus picks, for each text, the tag it is unusually close to."""
    TEXTS = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.4, 0.0, 0.0], [1.0, 0.0, 0.4, 0.0], [1.0, 0.0, 0.0, 0.4]])
    TAGS = np.eye(4)
    IDS = [501, 502, 503, 504]

    def test_raw_cosine_ranks_the_hub_first_for_every_text(self):
        self.assertEqual({t[0][0] for t in top_tags(self.TEXTS, self.TAGS, self.IDS, m=1)}, {501})

    def test_standardising_prefers_each_texts_distinctive_tag(self):
        stats = hubness_stats(tag_similarities(self.TEXTS, self.TAGS))
        firsts = [t[0][0] for t in top_tags(self.TEXTS[1:], self.TAGS, self.IDS, m=1, standardise=stats)]
        self.assertEqual(firsts, [502, 503, 504])

    def test_centring_keeps_the_scale_of_the_cosine(self):
        sims = tag_similarities(self.TEXTS, self.TAGS)
        mean, _sd = hubness_stats(sims)
        (tags,) = top_tags(self.TEXTS[1:2], self.TAGS, self.IDS, m=4, standardise=(mean, np.ones_like(mean)))
        self.assertAlmostEqual(dict(tags)[502], sims[1, 1] - mean[1])

    def test_reported_score_is_the_z_score(self):
        sims = tag_similarities(self.TEXTS, self.TAGS)
        mean, sd = hubness_stats(sims)
        (tags, *_) = top_tags(self.TEXTS[1:2], self.TAGS, self.IDS, m=4, standardise=(mean, sd))
        by_tag = dict(tags)
        self.assertAlmostEqual(by_tag[502], (sims[1, 1] - mean[1]) / sd[1])

    def test_constant_column_does_not_divide_by_zero(self):
        mean, sd = hubness_stats(np.array([[0.3, 0.1], [0.3, 0.9]]))
        self.assertGreater(sd[0], 0)
        self.assertTrue(np.isfinite(top_tags(np.array([[1.0, 0.0]]), np.eye(2), [1, 2], m=2, standardise=(mean, sd))[0][0][1]))

    def test_slot_share_of_the_most_used_tags_drops_on_hubby_data(self):
        rng = np.random.default_rng(0)
        topics = rng.normal(size=(40, 16))
        texts = topics[rng.integers(0, 40, 400)] + 0.3 * rng.normal(size=(400, 16))
        tags = np.vstack([topics + 0.3 * rng.normal(size=(40, 16)), 3.0 * np.ones((3, 16)) + rng.normal(size=(3, 16))])
        ids = list(range(len(tags)))
        share = lambda rows: np.bincount([t for r in rows for t, _ in r], minlength=len(tags))
        raw = share(top_tags(texts, tags, ids, m=5))
        fixed = share(top_tags(texts, tags, ids, m=5, standardise=hubness_stats(tag_similarities(texts, tags))))
        self.assertLess(fixed.max(), raw.max())


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

    def test_variant_reads_its_own_files_and_default_is_unchanged(self):
        z = {1: [(501, 3.0), (502, 2.0)], 2: [(502, 3.0), (503, 2.5)], 3: [(503, 1.2)]}
        for side, tagged in (("providers", z), ("hirers", {7: [(501, 4.0), (502, 1.5)], 8: [(503, 3.0)]})):
            (self.dir / f"tags_{side}_hz.json").write_text(json.dumps(_payload(tagged)))
        raw = TagChannel(self.dir, top_m_provider=3, top_m_hirer=3).rank(8)
        hz = TagChannel(self.dir, top_m_provider=3, top_m_hirer=3, variant="hz").rank(8)
        self.assertEqual(sorted(p for p, _ in raw), [1, 2, 3])      # raw files: 503 is on all three providers
        self.assertEqual(sorted(p for p, _ in hz), [2, 3])          # hz files: only providers 2 and 3 carry 503

    def test_weighted_scorer_ranks_by_z_strength_and_needs_z_files(self):
        z = {1: [(501, 3.0), (502, 1.1)], 2: [(501, 1.2), (502, 3.0)]}
        (self.dir / "tags_providers_hz.json").write_text(json.dumps(_payload(z)))
        (self.dir / "tags_hirers_hz.json").write_text(json.dumps(_payload({7: [(501, 3.0), (502, 1.1)]})))
        ranked = TagChannel(self.dir, variant="hz", scorer="wcos").rank(7)   # tau defaults to 1.0 for z-scores
        self.assertEqual([p for p, _ in ranked], [1, 2])
        self.assertGreater(ranked[0][1], ranked[1][1])
        with self.assertRaisesRegex(ValueError, "hubness center"):
            TagChannel(self.dir, scorer="wcos")

    def _hc_files(self, gigs, tag_vecs, tag_ids, hirer_ids):
        """tags_*_hc.json built the way tag_corpus.py builds them (centred scores, stats stored in the gig file)."""
        from tag_corpus import _write, hubness_stats, tag_similarities, top_tags
        mean, _sd = hubness_stats(tag_similarities(gigs, tag_vecs))
        standardise = (mean, np.ones_like(mean))
        tagged = dict(zip(hirer_ids, top_tags(gigs, tag_vecs, tag_ids, 4, standardise)))
        _write(self.dir / "tags_hirers_hc.json", "test", 4, tagged,
               {"score": "center", "tag_ids": tag_ids, "mean": [round(float(x), 6) for x in mean], "sd": [1.0] * len(mean)})
        providers = {1: [(501, .9), (502, .5), (503, .2)], 2: [(502, .6), (503, .5), (504, .4)], 3: [(504, .3), (501, .2)]}
        (self.dir / "tags_providers_hc.json").write_text(json.dumps(_payload(providers)))

    def test_rank_text_equals_rank_for_a_gig_in_the_file_and_works_for_one_that_is_not(self):
        tag_vecs = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [1., 1., 0]])
        tag_ids = [501, 502, 503, 504]
        gigs = np.array([[1., .2, 0], [0, 1., .3], [.1, 0, 1.], [.5, .5, .1]])
        self._hc_files(gigs, tag_vecs, tag_ids, [7, 8, 9, 10])
        for kwargs in ({}, {"scorer": "wcos", "tau": 0.0}):
            channel = TagChannel(self.dir, variant="hc", top_m_provider=3, top_m_hirer=3, **kwargs)
            for hire_id, vec in zip((7, 8, 9, 10), gigs):
                from_file, from_text = channel.rank(hire_id), channel.rank_text(vec, tag_vecs, tag_ids)
                self.assertEqual([p for p, _ in from_text], [p for p, _ in from_file])
                for (_, a), (_, b) in zip(from_file, from_text):
                    self.assertAlmostEqual(a, b, places=2)            # stored scores are rounded to 4 decimals
            with self.assertRaises(KeyError):
                channel.rank(99)                                     # not in the file ...
            self.assertTrue(channel.rank_text(np.array([.3, .2, 1.]), tag_vecs, tag_ids))   # ... but rank_text copes

    def test_rank_text_refuses_a_mismatched_tag_list_and_a_file_without_statistics(self):
        tag_vecs = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [1., 1., 0]])
        gigs = np.array([[1., .2, 0], [0, 1., .3], [.1, 0, 1.], [.5, .5, .1]])
        self._hc_files(gigs, tag_vecs, [501, 502, 503, 504], [7, 8, 9, 10])
        channel = TagChannel(self.dir, variant="hc", top_m_provider=3, top_m_hirer=3)
        with self.assertRaisesRegex(ValueError, "tag_ids differ"):
            channel.rank_text(gigs[0], tag_vecs, [1, 2, 3, 4])
        (self.dir / "tags_hirers_hc.json").write_text(json.dumps(_payload({7: [(501, .9)]})))   # a file with no stats
        with self.assertRaisesRegex(ValueError, "per-tag statistics"):
            TagChannel(self.dir, variant="hc", top_m_provider=3, top_m_hirer=3).rank_text(gigs[0], tag_vecs, [501, 502, 503, 504])

    def test_rank_text_on_the_raw_channel_uses_plain_cosine(self):
        tag_vecs = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        channel = TagChannel(self.dir, top_m_provider=3, top_m_hirer=1)       # raw files from setUp: gig 7 -> tag 501
        self.assertEqual([p for p, _ in channel.rank_text(np.array([1., .1, 0]), tag_vecs, [501, 502, 503])], [1])

    def test_max_returned_caps_the_list(self):
        self.assertEqual(len(TagChannel(self.dir, top_m_provider=3, top_m_hirer=3, max_returned=2).rank(8)), 2)

    def test_missing_tag_files_explain_what_to_run(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaisesRegex(FileNotFoundError, "tag_corpus.py"):
                TagChannel(Path(empty))


if __name__ == "__main__":
    unittest.main()
