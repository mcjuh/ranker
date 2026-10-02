"""
Loader for the SkillsFuture taxonomy as the front end and back end key it (pipeline/taxonomy_sf/).

    Category = Sector,  Specialisation = Track,  Skills = TSC        (see taxonomy_sf/README.md)

    tax = load_taxonomy()
    tax.resolve_tag("Business Valuation (Accountancy)")      # -> track id
    tax.track_tscs(track_id)                                   # frozenset of TSC ids reached through the track's roles
    tax.track_similarity(a, b)                                 # Jaccard of the two TSC sets, 0..1
    tax.resolve_tags(tags)                                     # (track ids, unknown tag strings)
    bridge_to_greygigz(tax)                                    # {"track": {new: old}, "tsc": {new: old}, ...}

A tag is "<track> (<sector>)". Track names repeat across sectors, so tags are keyed by the pair, compared case- and
whitespace-insensitively. These integer IDs are NOT the IDs in taxonomy_greygigz/ (the content is identical); use
bridge_to_greygigz to translate by name, never compare IDs across the two.
"""
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path(__file__).parent / "taxonomy_sf"

EXPECTED_COUNTS = {"sector": 39, "track": 247, "tsc": 2088, "job_role": 2001, "job_role_tsc": 43958}

_TAG = re.compile(r"^(?P<track>.+?)\s*\((?P<sector>[^()]+)\)\s*$")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def parse_tag(tag: str) -> tuple[str, str] | None:
    """'Business Valuation (Accountancy)' -> ('Business Valuation', 'Accountancy'); None when there is no '(sector)'."""
    m = _TAG.match(tag.strip())
    return (m.group("track").strip(), m.group("sector").strip()) if m else None


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


@dataclass
class Taxonomy:
    sectors: dict[int, str]                   # sector id -> name
    tracks: dict[int, dict]                   # track id -> {"name", "sector_id"}
    tscs: dict[int, str]                      # tsc id -> title
    roles: dict[int, dict]                    # role id -> {"name", "track_id"}
    role_tscs: dict[int, frozenset]           # role id -> tsc ids
    _tag_index: dict[tuple[str, str], int]
    _track_tscs: dict[int, frozenset]
    _sim: tuple | None = None

    def sector_of(self, track_id: int) -> str:
        return self.sectors[self.tracks[track_id]["sector_id"]]

    def track_key(self, track_id: int) -> str:
        """The tag string the front end shows for a track: 'Business Valuation (Accountancy)'."""
        return f"{self.tracks[track_id]['name']} ({self.sector_of(track_id)})"

    def resolve_tag(self, tag: str) -> int | None:
        """Track id for '<track> (<sector>)', or None if the string is not in the taxonomy."""
        parsed = parse_tag(tag)
        if parsed is None:
            return None
        return self._tag_index.get((norm(parsed[0]), norm(parsed[1])))

    def resolve_tags(self, tags) -> tuple[list[int], list[str]]:
        """(track ids in order, de-duplicated; tag strings that did not resolve). Nothing is dropped silently."""
        ids, unknown = [], []
        for t in tags:
            tid = self.resolve_tag(t)
            if tid is None:
                unknown.append(t)
            elif tid not in ids:
                ids.append(tid)
        return ids, unknown

    def track_tscs(self, track_id: int) -> frozenset:
        """TSC ids on any role of the track (3 to 112 of them)."""
        return self._track_tscs[track_id]

    def track_similarity(self, a: int, b: int) -> float:
        """Jaccard of the two tracks' TSC sets."""
        A, B = self._track_tscs[a], self._track_tscs[b]
        union = A | B
        return len(A & B) / len(union) if union else 0.0

    def similarity_matrix(self):
        """(track ids sorted, 247 x 247 numpy array of track_similarity), computed once."""
        if self._sim is None:
            import numpy as np
            ids = sorted(self.tracks)
            col = {t: j for j, t in enumerate(sorted(self.tscs))}
            M = np.zeros((len(ids), len(col)))
            for i, tid in enumerate(ids):
                for t in self._track_tscs[tid]:
                    M[i, col[t]] = 1.0
            inter = M @ M.T
            size = M.sum(axis=1)
            union = size[:, None] + size[None, :] - inter
            self._sim = (ids, np.divide(inter, union, out=np.zeros_like(inter), where=union > 0))
        return self._sim


def load_taxonomy(directory: Path | str = DEFAULT_DIR, verify: bool = True) -> Taxonomy:
    d = Path(directory)
    sectors = {int(r["sector_id"]): r["name"].strip() for r in _read(d / "sector.csv")}
    tracks = {int(r["track_id"]): {"name": r["name"].strip(), "sector_id": int(r["sector_id"])}
              for r in _read(d / "track.csv")}
    tscs = {int(r["tsc_id"]): r["title"].strip() for r in _read(d / "tsc.csv")}
    roles = {int(r["role_id"]): {"name": r["name"].strip(), "track_id": int(r["track_id"])}
             for r in _read(d / "job_role.csv")}
    links = _read(d / "job_role_tsc.csv")
    by_role: dict[int, set] = defaultdict(set)
    for r in links:
        by_role[int(r["role_id"])].add(int(r["tsc_id"]))
    role_tscs = {rid: frozenset(by_role.get(rid, ())) for rid in roles}

    index: dict[tuple[str, str], int] = {}
    for tid, t in tracks.items():
        key = (norm(t["name"]), norm(sectors[t["sector_id"]]))
        if key in index:
            raise ValueError(f"track and sector pair {key} is not unique")
        index[key] = tid
    per_track: dict[int, set] = defaultdict(set)
    for rid, r in roles.items():
        per_track[r["track_id"]] |= role_tscs[rid]
    track_tscs = {tid: frozenset(per_track.get(tid, ())) for tid in tracks}

    tax = Taxonomy(sectors, tracks, tscs, roles, role_tscs, index, track_tscs)
    if verify:
        got = {"sector": len(sectors), "track": len(tracks), "tsc": len(tscs), "job_role": len(roles),
               "job_role_tsc": len(links)}
        if got != EXPECTED_COUNTS:
            raise ValueError(f"taxonomy export changed: expected {EXPECTED_COUNTS}, got {got}")
        if len(set(tscs.values())) != len(tscs):
            raise ValueError("TSC titles are not unique; the title bridge to taxonomy_greygigz would be ambiguous")
    return tax


def bridge_to_greygigz(tax: Taxonomy, old=None) -> dict[str, dict]:
    """Translate between this ID space and taxonomy_greygigz/'s, by name.

    Returns {"track": {new_id: old_category_id}, "tsc": {new_id: old_tag_id}, "track_old": {old: new},
    "tsc_old": {old: new}}. TSCs are matched by title, tracks by (track name, sector name). `old` is a
    greygigz.Taxonomy (loaded from taxonomy_greygigz/ when omitted); anything that does not match raises, so a
    changed export cannot be bridged silently wrong."""
    if old is None:
        import greygigz
        old = greygigz.load_taxonomy(verify=False)
    old_tsc = {title: tid for tid, title in old.tags.items()}
    tsc_map = {}
    for new_id, title in tax.tscs.items():
        if title not in old_tsc:
            raise ValueError(f"TSC {new_id} {title!r} has no title match in taxonomy_greygigz")
        tsc_map[new_id] = old_tsc[title]
    old_track = {(norm(old.track_display_name(cid)), norm(old.sectors[cid])): cid for cid in old.tracks}
    track_map = {}
    for new_id in tax.tracks:
        key = (norm(tax.tracks[new_id]["name"]), norm(tax.sector_of(new_id)))
        if key not in old_track:
            raise ValueError(f"track {new_id} {key} has no match in taxonomy_greygigz")
        track_map[new_id] = old_track[key]
    return {"track": track_map, "tsc": tsc_map,
            "track_old": {o: n for n, o in track_map.items()}, "tsc_old": {o: n for n, o in tsc_map.items()}}
