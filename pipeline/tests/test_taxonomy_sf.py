"""
Tests for taxonomy_sf (the SkillsFuture taxonomy keyed as the front end keys it) and for ExplicitTagChannel when it is
given that taxonomy.

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v

The numbers asserted for the real taxonomy were computed from the files (Jaccard of the tracks' TSC sets).
"""
import json
import unittest
from pathlib import Path

import greygigz
import taxonomy_sf as sf
from explicit_tag_channel import ExplicitTagChannel
from frontend_schema import adapt_hirer, adapt_provider

SAMPLES = Path(__file__).resolve().parents[1] / "data_frontend_sample"


def rec(rid, kind, tags, cats=()):
    key = "provider_id" if kind == "p" else "hire_id"
    return {key: rid, "search_tags": tags, "category": [{"group": g, "name": n} for g, n in cats]}


class TestTaxonomy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tax = sf.load_taxonomy()

    def tid(self, tag):
        t = self.tax.resolve_tag(tag)
        self.assertIsNotNone(t, tag)
        return t

    def test_counts_and_mapping(self):
        t = self.tax
        self.assertEqual((len(t.sectors), len(t.tracks), len(t.tscs), len(t.roles)), (39, 247, 2088, 2001))
        self.assertEqual(sum(len(v) for v in t.role_tscs.values()), 43958)

    def test_resolve_is_by_track_and_sector(self):
        a, b = self.tid("Operations (Healthcare)"), self.tid("Operations (Financial Services)")
        self.assertNotEqual(a, b)
        self.assertEqual(self.tax.sector_of(a), "Healthcare")
        self.assertEqual(self.tid("  business   VALUATION (accountancy) "), self.tid("Business Valuation (Accountancy)"))
        self.assertEqual(self.tax.track_key(a), "Operations (Healthcare)")

    def test_unknown_tags_are_reported_not_dropped(self):
        ids, unknown = self.tax.resolve_tags(["Tax (Accountancy)", "Tax (Accountancy)", "Nope (Accountancy)", "no sector"])
        self.assertEqual(len(ids), 1)
        self.assertEqual(unknown, ["Nope (Accountancy)", "no sector"])
        self.assertIsNone(self.tax.resolve_tag("Tax (Healthcare)"))

    def test_track_tsc_sets_and_similarity(self):
        t = self.tax
        sizes = [len(t.track_tscs(x)) for x in t.tracks]
        self.assertEqual((min(sizes), max(sizes)), (3, 112))
        fa, ma = self.tid("Financial Accounting (Accountancy)"), self.tid("Management Accounting (Accountancy)")
        self.assertAlmostEqual(t.track_similarity(fa, ma), 0.537, places=3)
        self.assertAlmostEqual(t.track_similarity(fa, fa), 1.0)
        val, mna = self.tid("Business Valuation (Accountancy)"), self.tid("Mergers and Acquisitions (Accountancy)")
        self.assertAlmostEqual(t.track_similarity(val, mna), 0.353, places=3)
        ff, cs = self.tid("Financial Forensics (Accountancy)"), self.tid("Cyber Security (Infocomm Technology)")
        self.assertLess(t.track_similarity(ff, cs), 0.02)

    def test_similarity_matrix_matches_pairwise(self):
        ids, sim = self.tax.similarity_matrix()
        a, b = self.tid("Tax (Accountancy)"), self.tid("Internal Audit (Accountancy)")
        self.assertAlmostEqual(float(sim[ids.index(a), ids.index(b)]), self.tax.track_similarity(a, b))
        self.assertAlmostEqual(float(sim[ids.index(a), ids.index(a)]), 1.0)
        self.assertIs(self.tax.similarity_matrix(), self.tax.similarity_matrix())

    def test_bridge_to_greygigz_is_lossless(self):
        old = greygigz.load_taxonomy(verify=False)
        br = sf.bridge_to_greygigz(self.tax, old)
        self.assertEqual((len(br["track"]), len(br["tsc"])), (247, 2088))
        self.assertEqual((len(set(br["track"].values())), len(set(br["tsc"].values()))), (247, 2088))
        self.assertEqual(br["tsc_old"][br["tsc"][5]], 5)
        # the role-to-TSC links agree under the bridge: every old role's tag set maps onto a new role of the same name
        new_by_key = {(r["name"], r["track_id"]): rid for rid, r in self.tax.roles.items()}
        checked = 0
        for rid, role in old.roles.items():
            new_track = br["track_old"][role["track_id"]]
            new_rid = new_by_key[(role["name"], new_track)]
            self.assertEqual({br["tsc_old"][x] for x in old.role_tags[rid]}, set(self.tax.role_tscs[new_rid]))
            checked += 1
        self.assertEqual(checked, 2001)

    def test_ids_differ_between_the_two_id_spaces(self):
        # the reason the bridge exists: the integer IDs are not comparable
        old = greygigz.load_taxonomy(verify=False)
        br = sf.bridge_to_greygigz(self.tax, old)
        self.assertGreater(sum(1 for n, o in br["track"].items() if n != o), 200)

    def test_bridge_refuses_a_changed_export(self):
        old = greygigz.load_taxonomy(verify=False)
        old.tags[1] = "A title that is not in the new export"
        with self.assertRaises(ValueError):
            sf.bridge_to_greygigz(self.tax, old)


class TestChannelWithTaxonomy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tax = sf.load_taxonomy()
        cls.providers = [
            rec("P1", "p", ["Management Accounting (Accountancy)"], [("Function", "Accountancy")]),
            rec("P2", "p", ["Cyber Security (Infocomm Technology)"], [("Function", "Infocomm Technology")]),
            rec("P3", "p", ["Operations (Financial Services)"], [("Industry", "Financial Services")]),
        ]
        cls.hirers = [
            rec("H1", "h", ["Financial Accounting (Accountancy)"], [("Function", "Accountancy")]),
            rec("H2", "h", ["Tax (Accountancy)", "Not a track (Accountancy)"], [("Function", "Accountancy")]),
        ]

    def channel(self, **kw):
        kw.setdefault("category_weight", 0.0)
        return ExplicitTagChannel(self.hirers, self.providers, taxonomy=self.tax, **kw)

    def test_matching_is_by_track_id_and_unknown_tags_are_reported(self):
        ch = self.channel()
        self.assertEqual(ch.unknown_tags, {"H2": ["Not a track (Accountancy)"]})
        self.assertEqual(ch.report()["records_with_unknown_tags"], 1)
        self.assertEqual(ch.report()["unknown_tags"], ["Not a track (Accountancy)"])
        self.assertEqual(ch.rank("H1"), [])  # no provider carries Financial Accounting, no back-off asked for
        with self.assertRaises(ValueError):
            self.channel(strict=True)

    def test_track_backoff_reaches_a_similar_track_and_ignores_unrelated_ones(self):
        ch = self.channel(track_sim_weight=1.0)
        ranked = dict(ch.rank("H1"))
        fa, ma = (self.tax.resolve_tag(t) for t in ("Financial Accounting (Accountancy)",
                                                    "Management Accounting (Accountancy)"))
        unit = float(ch.tag_index.idf.min())
        self.assertAlmostEqual(ranked["P1"], unit * self.tax.track_similarity(fa, ma))
        self.assertNotIn("P2", ranked)  # similarity ~0.016 is under min_track_sim
        self.assertNotIn("P3", ranked)

    def test_exact_match_outranks_backoff_and_earns_no_backoff(self):
        providers = self.providers + [rec("P4", "p", ["Financial Accounting (Accountancy)"])]
        ch = ExplicitTagChannel(self.hirers, providers, taxonomy=self.tax, category_weight=0.0, track_sim_weight=1.0)
        ranked = ch.rank("H1")
        self.assertEqual(ranked[0][0], "P4")
        self.assertEqual(ch.track_backoff("H1", "P4"), 0.0)
        self.assertGreater(ranked[0][1], dict(ranked)["P1"])

    def test_backoff_never_exceeds_the_smallest_exact_match_score(self):
        ch = self.channel(track_sim_weight=1.0)
        unit = float(ch.tag_index.idf.min())
        for p in ch.provider_ids:
            self.assertLessEqual(ch.track_sim_weight * unit * ch.track_backoff("H1", p), unit)
        with self.assertRaises(ValueError):
            self.channel(track_sim_weight=1.5)

    def test_weight_zero_leaves_scores_unchanged(self):
        base = ExplicitTagChannel(self.hirers, self.providers, taxonomy=self.tax)
        also = ExplicitTagChannel(self.hirers, self.providers, taxonomy=self.tax, track_sim_weight=0.0)
        self.assertEqual(base.rank("H1"), also.rank("H1"))

    def test_weight_without_taxonomy_is_an_error(self):
        with self.assertRaises(ValueError):
            ExplicitTagChannel(self.hirers, self.providers, track_sim_weight=1.0)

    def test_features_include_the_backoff(self):
        ch = self.channel(track_sim_weight=1.0)
        self.assertGreater(ch.features("H1", "P1")["track_backoff"], 0.5 * 0.537 * 2 * 0.5)
        self.assertEqual(ch.features("H1", "P2")["track_backoff"], 0.0)


@unittest.skipUnless((SAMPLES / "hirers_sample.json").exists(), "sample files not present")
class TestSamplesAgainstTaxonomy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tax = sf.load_taxonomy()
        cls.hirers = [adapt_hirer(x) for x in json.loads((SAMPLES / "hirers_sample.json").read_text(encoding="utf-8"))]
        cls.providers = [adapt_provider(x) for x in
                         json.loads((SAMPLES / "providers_sample.json").read_text(encoding="utf-8"))]

    def test_every_sample_tag_resolves(self):
        ch = ExplicitTagChannel(self.hirers, self.providers, taxonomy=self.tax, strict=True)
        self.assertEqual(ch.report(), {"hirers_without_tags": 0, "providers_without_tags": 0,
                                       "records_with_unknown_tags": 0, "unknown_tags": []})

    def test_taxonomy_mode_ranks_like_string_mode_with_no_backoff(self):
        by_id = ExplicitTagChannel(self.hirers, self.providers, taxonomy=self.tax)
        by_str = ExplicitTagChannel(self.hirers, self.providers)
        for h in self.hirers:
            self.assertEqual(by_id.rank(h["hire_id"]), by_str.rank(h["hire_id"]), h["hire_id"])

    def test_backoff_fills_hirers_that_share_no_tag_with_any_provider(self):
        plain = ExplicitTagChannel(self.hirers, self.providers, taxonomy=self.tax, category_weight=0.0)
        soft = ExplicitTagChannel(self.hirers, self.providers, taxonomy=self.tax, category_weight=0.0,
                                  track_sim_weight=1.0)
        empty = [h["hire_id"] for h in self.hirers if not plain.rank(h["hire_id"])]
        self.assertTrue(empty)
        self.assertTrue(any(soft.rank(h) for h in empty))
        for h in self.hirers:  # back-off never removes a provider that already matched
            self.assertTrue({p for p, _ in plain.rank(h["hire_id"])} <= {p for p, _ in soft.rank(h["hire_id"])})


if __name__ == "__main__":
    unittest.main()
