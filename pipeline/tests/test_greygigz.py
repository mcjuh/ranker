"""
Tests for greygigz.py: the MySQL-dump scanner, the Taxonomy helpers on a tiny synthetic export,
and a check that the real export in pipeline/taxonomy_greygigz/ still matches what we measured.

Run from the repo root:
    python -m unittest discover -s pipeline/tests -t pipeline -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import greygigz
from greygigz import load_taxonomy, parse_inserts, track_distribution

# CRLF on purpose: working copies of the dumps are CRLF on Windows (core.autocrlf).
SQL = "\r\n".join([
    "-- a comment mentioning INSERT INTO `t` is fine",
    "INSERT INTO `t` (`id`, `name`, `note`, `score`) VALUES",
    r"(1, 'Plain', NULL, 1.5),",
    r"(2, 'It\'s (odd), yes; really', 'a''b', -3),",
    r"(3, 'INSERT INTO `t` (`id`) VALUES (99);', 'back\\slash \n end', 0);",
    "INSERT INTO `u` (`id`) VALUES (7);",
    "INSERT INTO `t` (`id`, `name`, `note`, `score`) VALUES (4, 'Last', '', 2);",
])


class ParseInserts(unittest.TestCase):
    def test_rows_types_escapes_and_multiple_statements(self):
        rows = parse_inserts(SQL, "t")
        self.assertEqual(rows, [
            {"id": 1, "name": "Plain", "note": None, "score": 1.5},
            {"id": 2, "name": "It's (odd), yes; really", "note": "a'b", "score": -3},
            {"id": 3, "name": "INSERT INTO `t` (`id`) VALUES (99);", "note": "back\\slash \n end", "score": 0},
            {"id": 4, "name": "Last", "note": "", "score": 2},
        ])

    def test_other_tables_are_kept_apart(self):
        self.assertEqual(parse_inserts(SQL, "u"), [{"id": 7}])

    def test_missing_table_is_an_error_not_an_empty_list(self):
        with self.assertRaises(ValueError):
            parse_inserts(SQL, "nope")

    def test_malformed_and_ragged_rows_raise(self):
        with self.assertRaises(ValueError):
            parse_inserts("INSERT INTO `t` (`id`) VALUES (1, 2);", "t")  # 2 values, 1 column
        with self.assertRaises(ValueError):
            parse_inserts("INSERT INTO `t` (`id`) VALUES (1) (2);", "t")


# ---------------------------------------------------------------------------
# A tiny synthetic export
#   tracks: 1 'Alpha', 2 'Beta', 3 'Alpha' (same name as 1, another sector)
#   roles:  r1 {1,2} in track 1   r2 {1,2} in track 1 (twin of r1)   r3 {2,3} in track 2
#           r4 {3} in track 3     r5 {1} in track 3
# ---------------------------------------------------------------------------

def _insert(table, cols, rows):
    def lit(v):
        return "NULL" if v is None else str(v) if isinstance(v, (int, float)) else "'" + v.replace("'", "''") + "'"
    head = f"INSERT INTO `{table}` ({', '.join('`' + c + '`' for c in cols)}) VALUES\n"
    return head + ",\n".join("(" + ", ".join(lit(v) for v in r) + ")" for r in rows) + ";\n"


def write_export(d: Path, bridge=None, levels=None):
    (d / "categories.sql").write_text(_insert("categories", ["id", "name", "service_id"],
                                              [(1, "Alpha", 1), (2, "Beta", 1), (3, "Alpha", 1)]))
    (d / "specialities.sql").write_text(_insert("specialities", ["id", "name", "description", "category_id"], [
        (1, "R1", "d1", 1), (2, "R2", "d2", 1), (3, "R3", "d3", 2), (4, "R4", "d4", 3), (5, "R5", "d5", 3)]))
    (d / "tags.sql").write_text(_insert("tags", ["id", "name"], [(1, "T1"), (2, "T2"), (3, "T3")]))
    bridge = bridge if bridge is not None else [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (5, 1)]
    (d / "speciality_tags.sql").write_text(_insert("speciality_tags", ["id", "speciality_id", "tag_id"],
                                                   [(i, r, t) for i, (r, t) in enumerate(bridge, 1)]))
    levels = levels if levels is not None else [(r, t, 2) for r, t in bridge]
    (d / "speciality_tag_levels.csv").write_text(
        "speciality_id,tag_id,level,raw_levels,tsc_ccs_type,tsc_ccs_codes\n"
        + "".join(f"{r},{t},{lv},{lv},tsc,X\n" for r, t, lv in levels))
    (d / "sector_by_category.csv").write_text("category_id,sector,track\n1,S1,Alpha\n2,S1,Beta\n3,S2,Alpha\n")
    (d / "truncated_names.csv").write_text("field,original,stored\ncategories.name [S2],Alpha and much more,Alpha\n")


SYNTHETIC_COUNTS = {"categories": 3, "specialities": 5, "tags": 3, "speciality_tags": 8, "speciality_tag_levels": 8}


class SyntheticTaxonomy(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        write_export(self.dir)
        self.tax = load_taxonomy(self.dir, verify=False)

    def test_tables_and_bridge(self):
        self.assertEqual(self.tax.tags, {1: "T1", 2: "T2", 3: "T3"})
        self.assertEqual(self.tax.role_tags[1], frozenset({1, 2}))
        self.assertEqual(self.tax.role_tags[4], frozenset({3}))
        self.assertEqual(self.tax.role_track(), {1: 1, 2: 1, 3: 2, 4: 3, 5: 3})
        self.assertEqual(self.tax.levels[(3, 3)], 2)
        self.assertEqual(self.tax.sectors, {1: "S1", 2: "S1", 3: "S2"})

    def test_equivalence_classes_group_identical_tag_sets(self):
        self.assertEqual(self.tax.equivalence_classes(), {1: 1, 2: 1, 3: 3, 4: 4, 5: 5})
        self.assertEqual(self.tax.n_roles_sharing_tagset(), 2)

    def test_duplicate_track_names_are_reported_by_category_id(self):
        self.assertEqual(self.tax.duplicate_track_names(), {"Alpha": [1, 3]})

    def test_truncated_track_name_is_restored_for_display(self):
        self.assertEqual(self.tax.track_display_name(3), "Alpha and much more")
        self.assertEqual(self.tax.track_display_name(2), "Beta")

    def test_track_distribution(self):
        rt = self.tax.role_track()
        ranked = [(1, 3.0), (3, 1.0), (4, 1.0)]
        dist = track_distribution(ranked, rt, top_r=3)
        self.assertEqual([c for c, _ in dist], [1, 2, 3])
        self.assertAlmostEqual(dist[0][1], 0.6)
        self.assertAlmostEqual(sum(m for _, m in dist), 1.0)
        self.assertEqual(track_distribution(ranked, rt, top_r=1), [(1, 1.0)])
        self.assertEqual(track_distribution([], rt), [])

    def test_verify_checks_counts(self):
        with self.assertRaisesRegex(ValueError, r"categories: expected 247 rows, found 3"):
            load_taxonomy(self.dir, verify=True)

    def test_verify_passes_and_catches_dangling_references(self):
        with mock.patch.dict(greygigz.EXPECTED_COUNTS, SYNTHETIC_COUNTS, clear=True):
            load_taxonomy(self.dir, verify=True)  # consistent export: no error
            write_export(self.dir, bridge=[(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (5, 99)])
            with self.assertRaisesRegex(ValueError, r"unknown tags"):
                load_taxonomy(self.dir, verify=True)

    def test_verify_catches_levels_bridge_disagreement(self):
        with mock.patch.dict(greygigz.EXPECTED_COUNTS, SYNTHETIC_COUNTS, clear=True):
            bridge = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3), (5, 1)]
            write_export(self.dir, bridge=bridge, levels=[(r, t, 2) for r, t in bridge[:-1]] + [(5, 3, 2)])
            with self.assertRaisesRegex(ValueError, r"levels CSV and bridge table disagree"):
                load_taxonomy(self.dir, verify=True)


class RealExport(unittest.TestCase):
    """Pins what we measured on the export committed in pipeline/taxonomy_greygigz/. If the export is
    regenerated these should fail loudly, and the README there should be updated with the new numbers."""

    @classmethod
    def setUpClass(cls):
        cls.tax = load_taxonomy()  # verify=True: counts, no dangling refs, levels == bridge pairs

    def test_counts(self):
        self.assertEqual(self.tax.raw_counts, greygigz.EXPECTED_COUNTS)
        self.assertEqual((len(self.tax.tracks), len(self.tax.roles), len(self.tax.tags)), (247, 2001, 2088))

    def test_every_role_has_tags(self):
        self.assertGreaterEqual(min(len(t) for t in self.tax.role_tags.values()), 1)

    def test_identical_tag_sets(self):
        self.assertEqual(self.tax.n_roles_sharing_tagset(), 619)
        self.assertEqual(len(set(self.tax.equivalence_classes().values())), 1606)

    def test_track_names_repeat_but_ids_are_unique(self):
        dup = self.tax.duplicate_track_names()
        self.assertEqual((len(dup), sum(len(v) for v in dup.values())), (9, 22))
        self.assertEqual(len({t["name"] for t in self.tax.tracks.values()}), 247 - 13)
        self.assertEqual({t["service_id"] for t in self.tax.tracks.values()}, {1})

    def test_truncated_track_name(self):
        cut = [cid for cid, t in self.tax.tracks.items() if t["name"] in self.tax.truncated]
        self.assertEqual(cut, [80])
        self.assertGreater(len(self.tax.track_display_name(80)), len(self.tax.tracks[80]["name"]))


if __name__ == "__main__":
    unittest.main()
