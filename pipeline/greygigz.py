"""
Loader for the SkillsFuture taxonomy exported in the GreyGigz schema (pipeline/taxonomy_greygigz/).

The export is MySQL dump text plus a few side CSVs. This module parses the dumps directly with a
small scanner (no MySQL, no live database, nothing is executed) and exposes:

    categories      = Track          (id -> name, service_id)
    specialities    = Job Role       (id -> name, description, category_id)
    tags            = TSC/CCS skill  (id -> title)
    speciality_tags = role-skill bridge  ->  Taxonomy.role_tags  {role id: frozenset(tag ids)}
    speciality_tag_levels.csv        ->  Taxonomy.levels     {(role id, tag id): level}
    sector_by_category.csv           ->  Taxonomy.sectors    {category id: sector}
    truncated_names.csv              ->  display names for values cut to fit a column

Tracks are always keyed by category ID: track names repeat across sectors.

Because some roles have exactly the same tag set as another role, retrieval of a role cannot be
scored by exact role ID. `Taxonomy.equivalence_classes()` maps every role to the smallest role ID
that shares its tag set; score at that level (or at track level).

Run `python pipeline/greygigz.py` for a summary of the export (counts, quirks, level distribution).
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EXPORT_DIR = Path(__file__).parent / "taxonomy_greygigz"

# What the export is documented to contain. load_taxonomy(verify=True) fails loudly on a mismatch,
# so a re-generated or truncated export can't silently change results.
EXPECTED_COUNTS = {
    "categories": 247,
    "specialities": 2001,
    "tags": 2088,
    "speciality_tags": 43958,
    "speciality_tag_levels": 43958,
}

# ---------------------------------------------------------------------------
# MySQL dump scanner
# ---------------------------------------------------------------------------

_INSERT = re.compile(r"INSERT INTO `(?P<table>\w+)` \((?P<cols>[^)]*)\) VALUES", re.IGNORECASE)
_TOKEN = re.compile(
    r"\s*(?:(?P<null>NULL)|(?P<num>-?\d+(?:\.\d+)?)|'(?P<str>(?:[^'\\]|\\.|'')*)'|(?P<punct>[(),;]))",
    re.DOTALL,
)
_ESCAPES = {"0": "\0", "n": "\n", "r": "\r", "t": "\t", "b": "\b", "Z": "\x1a"}
_UNESCAPE = re.compile(r"\\(.)|''", re.DOTALL)


def _unescape(s: str) -> str:
    return _UNESCAPE.sub(lambda m: "'" if m.group(0) == "''" else _ESCAPES.get(m.group(1), m.group(1)), s)


def _next(text: str, pos: int):
    m = _TOKEN.match(text, pos)
    if m is None:
        line = text.count("\n", 0, pos) + 1
        raise ValueError(f"unparseable SQL near line {line}: {text[pos:pos + 60]!r}")
    return m, m.end()


def _parse_statement(text: str, pos: int, cols: list[str]) -> tuple[list[dict], int]:
    """Parse `(v, v, ...), (v, ...);` starting just after VALUES. Returns (rows, position after ';')."""
    rows = []
    while True:
        m, pos = _next(text, pos)
        if m.group("punct") != "(":
            raise ValueError(f"expected '(' near line {text.count(chr(10), 0, pos) + 1}")
        values = []
        while True:
            m, pos = _next(text, pos)
            if m.group("null"):
                values.append(None)
            elif m.group("num") is not None:
                num = m.group("num")
                values.append(float(num) if "." in num else int(num))
            elif m.group("str") is not None:
                values.append(_unescape(m.group("str")))
            else:
                raise ValueError(f"expected a value near line {text.count(chr(10), 0, pos) + 1}")
            m, pos = _next(text, pos)
            if m.group("punct") == ")":
                break
            if m.group("punct") != ",":
                raise ValueError(f"expected ',' or ')' near line {text.count(chr(10), 0, pos) + 1}")
        if len(values) != len(cols):
            raise ValueError(f"row has {len(values)} values for {len(cols)} columns: {values[:3]}...")
        rows.append(dict(zip(cols, values)))
        m, pos = _next(text, pos)
        if m.group("punct") == ";":
            return rows, pos
        if m.group("punct") != ",":
            raise ValueError(f"expected ',' or ';' near line {text.count(chr(10), 0, pos) + 1}")


def parse_inserts(text: str, table: str) -> list[dict]:
    """All rows inserted into `table` by the dump text, as {column: value} dicts.
    Handles several INSERT statements per table, NULL, ints/decimals, and MySQL string escapes."""
    text = text.replace("\r\n", "\n")  # working copies are CRLF on Windows (core.autocrlf)
    rows, pos, seen = [], 0, False
    while (m := _INSERT.search(text, pos)) is not None:
        cols = [c.strip().strip("`") for c in m.group("cols").split(",")]
        stmt_rows, pos = _parse_statement(text, m.end(), cols)
        if m.group("table") == table:
            rows.extend(stmt_rows)
            seen = True
    if not seen:
        raise ValueError(f"no INSERT INTO `{table}` found")
    return rows


def read_table(path: Path, table: str) -> list[dict]:
    return parse_inserts(Path(path).read_text(encoding="utf-8-sig"), table)


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

@dataclass
class Taxonomy:
    tracks: dict[int, dict]                 # category id -> {"name", "service_id"}
    roles: dict[int, dict]                  # speciality id -> {"name", "description", "track_id"}
    tags: dict[int, str]                    # tag id -> title
    role_tags: dict[int, frozenset]         # role id -> tag ids (every role has an entry)
    levels: dict[tuple, int]                # (role id, tag id) -> proficiency level
    sectors: dict[int, str] = field(default_factory=dict)      # category id -> sector
    truncated: dict[str, str] = field(default_factory=dict)    # stored (cut) value -> original value
    raw_counts: dict[str, int] = field(default_factory=dict)   # rows read per source, before dedup

    def role_track(self) -> dict[int, int]:
        """role id -> track (category) id."""
        return {rid: r["track_id"] for rid, r in self.roles.items()}

    def equivalence_classes(self) -> dict[int, int]:
        """role id -> smallest role id whose tag set is identical (its own id if the set is unique).
        Score role retrieval at this level: identical roles cannot be told apart from tags alone."""
        by_set: dict[frozenset, list[int]] = defaultdict(list)
        for rid in sorted(self.role_tags):
            by_set[self.role_tags[rid]].append(rid)
        return {rid: members[0] for members in by_set.values() for rid in members}

    def n_roles_sharing_tagset(self) -> int:
        """Roles whose tag set is shared with at least one other role."""
        sizes = Counter(self.role_tags.values())
        return sum(1 for tags in self.role_tags.values() if sizes[tags] > 1)

    def duplicate_track_names(self) -> dict[str, list[int]]:
        """Track names used by more than one category id (why tracks are keyed by id)."""
        ids_by_name: dict[str, list[int]] = defaultdict(list)
        for cid, t in sorted(self.tracks.items()):
            ids_by_name[t["name"]].append(cid)
        return {name: ids for name, ids in ids_by_name.items() if len(ids) > 1}

    def track_display_name(self, category_id: int) -> str:
        """Track name for display; restores the original where the stored one was cut to fit its column."""
        stored = self.tracks[category_id]["name"]
        return self.truncated.get(stored, stored)


def track_distribution(ranked_roles, role_track: dict, top_r: int = 10) -> list[tuple]:
    """Aggregate the top_r ranked roles into a distribution over tracks (category ids).

    `ranked_roles` is [(role_id, score), ...] best first, as TagBM25.rank returns. Each role adds its
    score to its track; masses are normalised to sum to 1. Returns [(category_id, mass), ...]
    descending, ties by category id. Empty when nothing was retrieved."""
    mass: dict[int, float] = defaultdict(float)
    for rid, score in list(ranked_roles)[:top_r]:
        mass[role_track[rid]] += score
    total = sum(mass.values())
    if total <= 0:
        return []
    return sorted(((cid, m / total) for cid, m in mass.items()), key=lambda x: (-x[1], x[0]))


def load_taxonomy(export_dir=DEFAULT_EXPORT_DIR, verify: bool = True) -> Taxonomy:
    d = Path(export_dir)
    categories = read_table(d / "categories.sql", "categories")
    specialities = read_table(d / "specialities.sql", "specialities")
    tags = read_table(d / "tags.sql", "tags")
    bridge = read_table(d / "speciality_tags.sql", "speciality_tags")
    level_rows = _read_csv(d / "speciality_tag_levels.csv")
    sector_rows = _read_csv(d / "sector_by_category.csv") if (d / "sector_by_category.csv").exists() else []
    trunc_rows = _read_csv(d / "truncated_names.csv") if (d / "truncated_names.csv").exists() else []

    tax = Taxonomy(
        tracks={c["id"]: {"name": c["name"], "service_id": c["service_id"]} for c in categories},
        roles={s["id"]: {"name": s["name"], "description": s["description"], "track_id": s["category_id"]}
               for s in specialities},
        tags={t["id"]: t["name"] for t in tags},
        role_tags={},
        levels={(int(r["speciality_id"]), int(r["tag_id"])): int(r["level"]) for r in level_rows},
        sectors={int(r["category_id"]): r["sector"] for r in sector_rows},
        truncated={r["stored"]: r["original"] for r in trunc_rows if r["field"].startswith("categories.name")},
        raw_counts={
            "categories": len(categories), "specialities": len(specialities), "tags": len(tags),
            "speciality_tags": len(bridge), "speciality_tag_levels": len(level_rows),
        },
    )
    role_tags: dict[int, set] = {rid: set() for rid in tax.roles}
    for row in bridge:
        role_tags.setdefault(row["speciality_id"], set()).add(row["tag_id"])
    tax.role_tags = {rid: frozenset(t) for rid, t in role_tags.items()}

    if verify:
        problems = _check(tax, {(r["speciality_id"], r["tag_id"]) for r in bridge})
        if problems:
            raise ValueError("greygigz export failed verification:\n  - " + "\n  - ".join(problems))
    return tax


def _check(tax: Taxonomy, bridge_pairs: set) -> list[str]:
    problems = [
        f"{name}: expected {want} rows, found {tax.raw_counts[name]}"
        for name, want in EXPECTED_COUNTS.items() if tax.raw_counts[name] != want
    ]
    if len(bridge_pairs) != tax.raw_counts["speciality_tags"]:
        problems.append(f"speciality_tags has {tax.raw_counts['speciality_tags'] - len(bridge_pairs)} duplicate pairs")
    unknown_roles = {r for r, _ in bridge_pairs} - set(tax.roles)
    unknown_tags = {t for _, t in bridge_pairs} - set(tax.tags)
    unknown_tracks = {r["track_id"] for r in tax.roles.values()} - set(tax.tracks)
    if unknown_roles:
        problems.append(f"bridge references {len(unknown_roles)} unknown roles, e.g. {sorted(unknown_roles)[:3]}")
    if unknown_tags:
        problems.append(f"bridge references {len(unknown_tags)} unknown tags, e.g. {sorted(unknown_tags)[:3]}")
    if unknown_tracks:
        problems.append(f"roles reference {len(unknown_tracks)} unknown tracks, e.g. {sorted(unknown_tracks)[:3]}")
    if set(tax.levels) != bridge_pairs:
        problems.append(
            f"levels CSV and bridge table disagree: {len(set(tax.levels) - bridge_pairs)} pairs only in levels, "
            f"{len(bridge_pairs - set(tax.levels))} only in bridge"
        )
    return problems


# ---------------------------------------------------------------------------
# CLI: summary of the export
# ---------------------------------------------------------------------------

def summarise(tax: Taxonomy) -> str:
    lines = ["rows read: " + ", ".join(f"{k}={v}" for k, v in tax.raw_counts.items())]
    sizes = sorted(len(t) for t in tax.role_tags.values())
    lines.append(f"tags per role: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}, "
                 f"mean {sum(sizes) / len(sizes):.1f}")
    df = Counter(t for tags in tax.role_tags.values() for t in tags)
    unused = len(tax.tags) - len(df)
    top = ", ".join(f"{tax.tags[t]} ({n})" for t, n in df.most_common(5))
    lines.append(f"tags used by no role: {unused}; most generic tags (roles containing them): {top}")
    classes = tax.equivalence_classes()
    lines.append(f"roles sharing an identical tag set with another role: {tax.n_roles_sharing_tagset()} "
                 f"({len(set(classes.values()))} equivalence classes for {len(classes)} roles)")
    dup = tax.duplicate_track_names()
    lines.append(f"track names used by more than one category id: {len(dup)} names, "
                 f"{sum(len(v) for v in dup.values())} category ids")
    lines.append(f"service ids on categories: {dict(Counter(t['service_id'] for t in tax.tracks.values()))}")
    lines.append(f"proficiency levels: {dict(sorted(Counter(tax.levels.values()).items()))}")
    cut = [cid for cid, t in tax.tracks.items() if t["name"] in tax.truncated]
    lines.append(f"tracks whose stored name was truncated: {cut}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    ap.add_argument("--no-verify", action="store_true", help="skip the expected-count checks")
    args = ap.parse_args()
    print(summarise(load_taxonomy(args.export_dir, verify=not args.no_verify)))


if __name__ == "__main__":
    main()
