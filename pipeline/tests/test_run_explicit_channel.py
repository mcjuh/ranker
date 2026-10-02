"""
Tests for the explicit-tag stage of run_pipeline.py: run_explicit_channel (pure, stub scores), adapt_records, and a
source-level check that the CLI flag is opt-in. run_pipeline.py itself cannot be imported without the embedding libraries,
so it is parsed with `ast`.

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v
"""
import ast
import json
import unittest
from pathlib import Path

import taxonomy_sf as sf
from explicit_tag_channel import ExplicitTagChannel, run_explicit_channel
from frontend_schema import adapt_hirer, adapt_provider, adapt_records, is_frontend_schema

PIPELINE = Path(__file__).resolve().parents[1]
SAMPLES = PIPELINE / "data_frontend_sample"


def rec(rid, kind, tags):
    return {("provider_id" if kind == "p" else "hire_id"): rid, "search_tags": tags,
            "category": [{"group": "Function", "name": "Accountancy"}]}


PROVIDERS = [rec("P1", "p", ["Tax (Accountancy)"]), rec("P2", "p", ["Tax (Accountancy)"]),
             rec("P3", "p", ["Internal Audit (Accountancy)"])]
HIRERS = [rec("H1", "h", ["Tax (Accountancy)"]), rec("H2", "h", [])]
BM25 = {"H1": [("P3", 3.0), ("P2", 2.0), ("P1", 1.0)], "H2": [("P1", 2.0), ("P2", 1.0)]}
DENSE = {"H1": [("P2", 0.9), ("P3", 0.8), ("P1", 0.7)], "H2": [("P2", 0.9), ("P1", 0.8)]}


class TestRunExplicitChannel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tax = sf.load_taxonomy()

    def run_it(self, **kw):
        return run_explicit_channel(HIRERS, PROVIDERS, BM25, DENSE, taxonomy=self.tax, category_weight=0.0, **kw)

    def test_dense_breaks_the_tie_between_equal_explicit_scores(self):
        explicit, _, _ = self.run_it()
        # P1 and P2 both carry Tax; plain order would put P1 first, the dense cosine puts P2 first
        self.assertEqual([p for p, _ in explicit["H1"]], ["P2", "P1"])

    def test_fusion_is_three_way_rrf_and_keeps_string_ids(self):
        _, fused, _ = self.run_it()
        ranked = [p for p, _ in fused["H1"]]
        self.assertEqual(set(ranked), {"P1", "P2", "P3"})
        # P2: rank 2 in bm25, 1 in dense, 1 in explicit; P3: 1, 2 and absent; P1: 3, 3, 2
        self.assertEqual(ranked[0], "P2")
        score = dict(fused["H1"])
        self.assertAlmostEqual(score["P2"], 1 / 62 + 1 / 61 + 1 / 61)
        self.assertAlmostEqual(score["P3"], 1 / 61 + 1 / 62)

    def test_hirer_without_tags_gets_an_empty_explicit_list_and_still_fuses(self):
        explicit, fused, report = self.run_it()
        self.assertEqual(explicit["H2"], [])
        self.assertEqual({p for p, _ in fused["H2"]}, {"P1", "P2"})
        self.assertEqual(report["hirers_without_tags"], 1)
        self.assertEqual(report["hirers_with_empty_explicit_list"], 1)

    def test_fallback_is_used_for_a_hirer_without_tags(self):
        class Fallback:
            def rank(self, hire_id, top_k=None):
                return [("P3", 1.0)]

        explicit, _, _ = self.run_it(fallback=Fallback())
        self.assertEqual(explicit["H2"], [("P3", 1.0)])

    def test_strict_raises_on_a_tag_outside_the_taxonomy(self):
        bad = HIRERS + [rec("H3", "h", ["Made Up (Accountancy)"])]
        with self.assertRaises(ValueError):
            run_explicit_channel(bad, PROVIDERS, {**BM25, "H3": []}, {**DENSE, "H3": []}, taxonomy=self.tax)
        _, _, report = run_explicit_channel(bad, PROVIDERS, {**BM25, "H3": []}, {**DENSE, "H3": []},
                                            taxonomy=self.tax, strict=False)
        self.assertEqual(report["unknown_tags"], ["Made Up (Accountancy)"])

    def test_without_taxonomy_matches_by_string(self):
        explicit, _, report = run_explicit_channel(HIRERS, PROVIDERS, BM25, DENSE, category_weight=0.0)
        self.assertEqual([p for p, _ in explicit["H1"]], ["P2", "P1"])
        self.assertEqual(report["records_with_unknown_tags"], 0)


@unittest.skipUnless((SAMPLES / "hirers_sample.json").exists(), "sample files not present")
class TestOnTheSampleFiles(unittest.TestCase):
    def test_every_hirer_is_fused_with_stub_scores(self):
        hirers = adapt_records(json.loads((SAMPLES / "hirers_sample.json").read_text(encoding="utf-8")), "hirer")
        providers = adapt_records(json.loads((SAMPLES / "providers_sample.json").read_text(encoding="utf-8")), "provider")
        ids = [p["provider_id"] for p in providers]
        stub = {h["hire_id"]: [(p, 1.0 - i / 100) for i, p in enumerate(ids)] for h in hirers}
        explicit, fused, report = run_explicit_channel(hirers, providers, stub, stub, taxonomy=sf.load_taxonomy(),
                                                       track_sim_weight=0.5)
        self.assertEqual(set(fused), {h["hire_id"] for h in hirers})
        self.assertTrue(all(len(v) == 30 for v in fused.values()))
        self.assertEqual((report["hirers_without_tags"], report["records_with_unknown_tags"]), (0, 0))
        # without the back-off 10 sample hirers share no exact tag with any provider; with it only H030 stays empty: a
        # Legal Services lease review, no Legal Services provider in the sample and no track above min_track_sim
        self.assertEqual(report["hirers_with_empty_explicit_list"], 1)
        self.assertEqual([h for h, v in explicit.items() if not v], ["H030"])
        plain, _, plain_report = run_explicit_channel(hirers, providers, stub, stub, taxonomy=sf.load_taxonomy(),
                                                      category_weight=0.0)
        self.assertGreaterEqual(plain_report["hirers_with_empty_explicit_list"], 10)


class TestAdaptRecords(unittest.TestCase):
    def test_frontend_records_are_converted(self):
        raw = [{"hirer_id": "H1", "gig_title": "T", "short_gig_description": "D", "search_tags": ["Tax (Accountancy)"]}]
        self.assertTrue(is_frontend_schema(raw))
        out = adapt_records(raw, "hirer")
        self.assertEqual((out[0]["hire_id"], out[0]["hire_title"]), ("H1", "T"))

    def test_pipeline_schema_records_pass_through_unchanged(self):
        old_h = [{"hire_id": 1, "hire_title": "T", "hire_description": "D", "budget_lo": 1}]
        old_p = [{"provider_id": 1, "about_title": "A", "about_description": "B"}]
        self.assertFalse(is_frontend_schema(old_h))
        self.assertIs(adapt_records(old_h, "hirer"), old_h)
        self.assertIs(adapt_records(old_p, "provider"), old_p)
        self.assertEqual(adapt_records([], "hirer"), [])

    def test_real_data_sat_records_pass_through(self):
        path = PIPELINE / "data_sat" / "hirers.json"
        if path.exists():
            records = json.loads(path.read_text(encoding="utf-8"))
            self.assertIs(adapt_records(records, "hirer"), records)

    def test_bad_kind(self):
        with self.assertRaises(ValueError):
            adapt_records([], "gig")


class TestCliFlagIsOptIn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((PIPELINE / "run_pipeline.py").read_text(encoding="utf-8"))

    def test_flag_is_store_true_without_a_true_default(self):
        found = False
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                    and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--explicit-tag-channel"):
                found = True
                kw = {k.arg: k.value for k in node.keywords}
                self.assertEqual(ast.literal_eval(kw["action"]), "store_true")
                self.assertNotIn("default", kw)
        self.assertTrue(found)

    def test_track_sim_weight_defaults_to_off(self):
        for node in ast.walk(self.tree):
            if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "--track-sim-weight"):
                kw = {k.arg: k.value for k in node.keywords}
                self.assertEqual(ast.literal_eval(kw["default"]), 0.0)
                return
        self.fail("--track-sim-weight not found")

    def test_the_stage_only_runs_under_the_flag(self):
        guarded = [n for n in ast.walk(self.tree) if isinstance(n, ast.If)
                   and isinstance(n.test, ast.Attribute) and n.test.attr == "explicit_tag_channel"]
        self.assertEqual(len(guarded), 1)
        calls = {c.func.id for c in ast.walk(guarded[0]) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)}
        self.assertIn("run_explicit_channel", calls)
        # and nothing outside that block calls it
        outside = [c for c in ast.walk(self.tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                   and c.func.id == "run_explicit_channel" and c not in set(ast.walk(guarded[0]))]
        self.assertEqual(outside, [])


if __name__ == "__main__":
    unittest.main()
