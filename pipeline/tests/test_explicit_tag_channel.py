"""
Tests for frontend_schema (adapter and free-text parsers) and explicit_tag_channel (the third channel over
user-selected tags).

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v

The channel fixture is small enough to work out by hand. Providers (b = 0, so a score is the IDF-weighted overlap):

    P1 = {Tax (Accountancy)}                                   categories {accountancy}
    P2 = {Tax (Accountancy), Operations (Healthcare)}          categories {accountancy, healthcare}
    P3 = {Operations (Financial Services)}                     categories {financial services}
    P4 = {}                                                    no tags

    N = 4. df(tax) = 2, df(ops healthcare) = 1, df(ops fin services) = 1.
    idf(df=2) = ln(1 + 2.5/2.5) = ln 2;  idf(df=1) = ln(1 + 3.5/1.5) = ln(10/3)
    category df: accountancy 2, healthcare 1, financial services 1 -> the same IDFs.
"""
import json
import math
import unittest
from pathlib import Path

from explicit_tag_channel import ExplicitTagChannel, category_groups, parse_tag, tag_keys
from frontend_schema import (adapt_hirer, adapt_provider, parse_available_from, parse_capacity_days,
                             parse_duration_weeks, parse_rate_per_hour, strip_exclusions)

LN2 = math.log(2)
IDF_RARE = math.log(10 / 3)
SAMPLES = Path(__file__).resolve().parents[1] / "data_frontend_sample"


def _rec(kind, rid, tags, cats):
    key = "provider_id" if kind == "p" else "hire_id"
    return {key: rid, "search_tags": tags, "category": [{"group": g, "name": n} for g, n in cats]}


PROVIDERS = [
    _rec("p", "P1", ["Tax (Accountancy)"], [("Function", "Accountancy")]),
    _rec("p", "P2", ["Tax (Accountancy)", "Operations (Healthcare)"],
         [("Function", "Accountancy"), ("Industry", "Healthcare")]),
    _rec("p", "P3", ["Operations (Financial Services)"], [("Industry", "Financial Services")]),
    _rec("p", "P4", [], []),
]
HIRERS = [
    _rec("h", "H1", ["Tax (Accountancy)"], [("Function", "Accountancy")]),
    _rec("h", "H2", ["Operations (Healthcare)"], [("Industry", "Healthcare")]),
    _rec("h", "H3", ["Corporate Finance (Financial Services)"], [("Industry", "Financial Services")]),
    _rec("h", "H4", [], []),
]


class TestParsers(unittest.TestCase):
    def test_parse_tag(self):
        self.assertEqual(parse_tag("Risk, Compliance and Legal (Financial Services)"),
                         ("Risk, Compliance and Legal", "Financial Services"))
        self.assertEqual(parse_tag("No brackets"), ("No brackets", None))

    def test_rate(self):
        self.assertEqual(parse_rate_per_hour("From S$1,400 per day; fixed fees for defined reviews."), 175.0)
        self.assertEqual(parse_rate_per_hour("From S$8,000 per month on retainer."), 50.0)
        self.assertEqual(parse_rate_per_hour("S$120 per hour"), 120.0)
        self.assertIsNone(parse_rate_per_hour("Fixed fee per study."))
        self.assertIsNone(parse_rate_per_hour(None))

    def test_availability(self):
        self.assertEqual(parse_available_from("Available immediately for 2-8 week engagements in Singapore."), "now")
        self.assertEqual(parse_available_from("Available from mid-November 2026 for short engagements."), "2026-11-15")
        self.assertEqual(parse_available_from("Up to 2 days a week from December 2026; board work."), "2026-12-01")
        self.assertIsNone(parse_available_from("Ask me."))
        self.assertEqual(parse_capacity_days("Available from November 2026, up to 3 days a week; on-site."), 3)
        self.assertEqual(parse_capacity_days("Available from November 2026, 1-2 days a week on a retainer."), 2)
        self.assertIsNone(parse_capacity_days("Available immediately for 2-8 week engagements."))

    def test_duration(self):
        self.assertEqual(parse_duration_weeks("... Deliverable: x. Engagement duration: 4-6 weeks."), (4, 6))
        self.assertEqual(parse_duration_weeks("Engagement duration: 3 weeks"), (3, 3))
        self.assertIsNone(parse_duration_weeks("no duration"))

    def test_strip_exclusions(self):
        text = "Valuation for SMEs. Not focused on brokerage. Includes reports. Does not include court advocacy."
        self.assertEqual(strip_exclusions(text), "Valuation for SMEs. Includes reports.")
        self.assertEqual(strip_exclusions(""), "")


class TestAdapter(unittest.TestCase):
    def test_hirer_keeps_string_id_and_tags_and_drops_metadata(self):
        raw = {"hirer_id": "H001", "gig_id": "G001", "gig_title": "T", "short_gig_description":
               "Do it. Engagement duration: 4-6 weeks.", "category": [{"group": "Function", "name": "Accountancy"}],
               "specialisation": ["Tax", "Operations"], "search_tags": ["Tax (Accountancy)", "Operations (Accountancy)"],
               "review_flag": "bad", "source_row": 1, "source_file": "firm.com__x.md"}
        h = adapt_hirer(raw)
        self.assertEqual(h["hire_id"], "H001")
        self.assertEqual(h["hire_title"], "T")
        self.assertEqual(h["search_tags"], ["Tax (Accountancy)", "Operations (Accountancy)"])
        self.assertEqual((h["duration_weeks_lo"], h["duration_weeks_hi"]), (4, 6))
        for leaked in ("review_flag", "source_row", "source_file"):
            self.assertNotIn(leaked, h)
        self.assertNotIn("firm.com", json.dumps(h))

    def test_provider_text_fields_and_exclusions(self):
        raw = {"provider_id": "P001", "about_headline": "Head", "about_bio": "Bio.", "credentials": "CPA",
               "services_i_offer": [{"service_title": "Svc", "service_detail": "Does tax. Not focused on audit."}],
               "relevant_achievements": ["Won case."], "technical_proficiency": [{"category": "Tax", "skills": ["A", "B"]}],
               "how_i_work": "I do things.", "rate": "From S$800 per day", "availability": "Available immediately.",
               "search_tags": ["Tax (Accountancy)"], "localisation_changes": ["x -> y"], "source_file": "firm.com__p.md"}
        p = adapt_provider(raw)
        self.assertEqual(p["provider_id"], "P001")
        self.assertEqual(p["about_title"], "Head")
        self.assertEqual(p["services_offered_description"], "Does tax.")
        self.assertEqual(p["relevant_experience"], "Won case. Tax: A, B")
        self.assertEqual((p["rate_per_hour"], p["available_from"], p["capacity"]), (100.0, "now", None))
        text = json.dumps(p)
        for leaked in ("CPA", "x -> y", "firm.com", "I do things"):
            self.assertNotIn(leaked, text)
        self.assertIn("I do things.", adapt_provider(raw, include_how_i_work=True)["relevant_experience"])
        self.assertIn("Not focused on audit", adapt_provider(raw, drop_exclusions=False)["services_offered_description"])


class TestExplicitTagChannel(unittest.TestCase):
    def setUp(self):
        self.ch = ExplicitTagChannel(HIRERS, PROVIDERS, category_weight=0.25)

    def test_exact_match_scores_by_idf_plus_category_credit(self):
        ranked = dict(self.ch.rank("H1"))
        # P1 and P2 share the tag and the category: idf(tax) + 0.25 * idf(accountancy)
        self.assertAlmostEqual(ranked["P1"], LN2 + 0.25 * LN2)
        self.assertAlmostEqual(ranked["P2"], LN2 + 0.25 * LN2)
        self.assertNotIn("P3", ranked)
        self.assertNotIn("P4", ranked)

    def test_composite_tag_is_not_the_bare_specialisation(self):
        # H2 asks for Operations (Healthcare); P3 has Operations (Financial Services) and must not match on the tag
        ranked = dict(self.ch.rank("H2"))
        self.assertAlmostEqual(ranked["P2"], IDF_RARE + 0.25 * IDF_RARE)
        self.assertNotIn("P3", ranked)

    def test_category_backoff_when_no_exact_tag(self):
        # nobody carries Corporate Finance (Financial Services), but P3 is in the category
        ranked = self.ch.rank("H3")
        self.assertEqual([p for p, _ in ranked], ["P3"])
        self.assertAlmostEqual(ranked[0][1], 0.25 * IDF_RARE)

    def test_zero_category_weight_disables_backoff(self):
        self.assertEqual(ExplicitTagChannel(HIRERS, PROVIDERS, category_weight=0).rank("H3"), [])

    def test_tiebreak_orders_ties_and_never_changes_scores(self):
        plain = self.ch.rank("H1")
        self.assertEqual([p for p, _ in plain], ["P1", "P2"])  # tie: provider order
        tied = self.ch.rank("H1", tiebreak={"P2": 0.9, "P1": 0.1})
        self.assertEqual([p for p, _ in tied], ["P2", "P1"])
        self.assertEqual(dict(plain), dict(tied))

    def test_top_k(self):
        self.assertEqual(len(self.ch.rank("H1", top_k=1)), 1)

    def test_hirer_without_tags_uses_fallback_or_returns_empty(self):
        self.assertEqual(self.ch.rank("H4"), [])

        class Fallback:
            def rank(self, hire_id, top_k=None):
                return [("PX", 1.0)]

        self.assertEqual(ExplicitTagChannel(HIRERS, PROVIDERS, fallback=Fallback()).rank("H4"), [("PX", 1.0)])

    def test_b_penalises_longer_provider_tag_lists(self):
        flat = dict(ExplicitTagChannel(HIRERS, PROVIDERS, b=0.0, category_weight=0).rank("H1"))
        norm = dict(ExplicitTagChannel(HIRERS, PROVIDERS, b=0.75, category_weight=0).rank("H1"))
        self.assertAlmostEqual(flat["P1"], flat["P2"])
        self.assertGreater(norm["P1"], norm["P2"])

    def test_features(self):
        f = self.ch.features("H1", "P2")
        self.assertEqual((f["tag_overlap"], f["category_overlap"], f["function_match"], f["industry_match"]),
                         (1, 1, 1, 0))
        self.assertAlmostEqual(f["tag_idf_overlap"], LN2, places=5)
        self.assertAlmostEqual(f["tag_jaccard"], 1 / 2, places=5)
        g = self.ch.features("H2", "P2")
        self.assertEqual((g["function_match"], g["industry_match"]), (0, 1))
        self.assertEqual(self.ch.features("H1", "P3")["tag_overlap"], 0)

    def test_helpers(self):
        self.assertEqual(tag_keys({"search_tags": ["  Tax   (Accountancy) "]}), frozenset({"tax (accountancy)"}))
        self.assertEqual(category_groups({"category": [{"group": "Function", "name": "Accountancy"}],
                                          "search_tags": ["Operations (Healthcare)"]}),
                         {"accountancy": "Function", "healthcare": ""})


@unittest.skipUnless((SAMPLES / "hirers_sample.json").exists(), "sample files not present")
class TestSampleFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hirers = [adapt_hirer(x) for x in json.loads((SAMPLES / "hirers_sample.json").read_text(encoding="utf-8"))]
        cls.providers = [adapt_provider(x) for x in
                         json.loads((SAMPLES / "providers_sample.json").read_text(encoding="utf-8"))]

    def test_counts_ids_and_tags(self):
        self.assertEqual((len(self.hirers), len(self.providers)), (30, 30))
        self.assertTrue(all(isinstance(h["hire_id"], str) and h["search_tags"] for h in self.hirers))
        self.assertTrue(all(isinstance(p["provider_id"], str) and p["search_tags"] for p in self.providers))

    def test_two_specialisations_under_one_category_are_both_kept(self):
        p001 = next(p for p in self.providers if p["provider_id"] == "P001")
        self.assertEqual(len(p001["category"]), 1)
        self.assertEqual(p001["search_tags"], ["Financial Forensics (Accountancy)", "Internal Audit (Accountancy)"])

    def test_no_metadata_reaches_retrieval_text(self):
        for rec in self.hirers + self.providers:
            blob = json.dumps({k: v for k, v in rec.items() if k not in ("rate", "availability")})
            self.assertNotIn("source_file", blob)
            self.assertNotIn("localisation", blob)
            self.assertNotIn(".com__", blob)

    def test_channel_ranks_every_hirer_with_overlap_and_ties_are_deterministic(self):
        ch = ExplicitTagChannel(self.hirers, self.providers)
        h = ch.rank("H026")  # Business Valuation + Mergers and Acquisitions (both Accountancy)
        self.assertEqual(h[0][0], "P013")  # the only provider with the rare M&A tag
        self.assertEqual(h, ch.rank("H026"))
        for p in ("P007", "P015", "P023", "P024"):  # Business Valuation (Accountancy)
            self.assertIn(p, dict(h))


if __name__ == "__main__":
    unittest.main()
