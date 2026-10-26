"""
Taxonomy recall channel (Card 2) and taxonomy match features (Card 3).

The third recall channel: match a gig's SkillsFuture tags against a provider's
showcase tags, on the shared `sf_v1` vocabulary that both sides of the v2 corpus
now carry (categories -> specialisations -> search_tag + skills).

Two consumers, mirroring the other two channels:
  * `TaxonomyRetriever.rank()`  -> a ranked list for RRF fusion (Card 2)
  * `taxonomy_scores()`         -> per-pair match scores for the LambdaMART
                                   ranker (Card 3, "Recall Channel Matching Scores")

Scoring is COVERAGE OF THE GIG'S NEED, not symmetric similarity: a gig asking
for one track is fully covered by a provider holding it, however many other
tracks that provider also holds. Asymmetric by design -- a provider cannot
dilute their own match by listing more specialisations.

All weights below are FIXED and UNTUNED: the v2 corpus carries no relevance
labels, so there is nothing to fit them against. They are a documented choice,
not a fitted result, and are exposed individually as features so a trained
ranker can learn its own combination when labels exist.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

# Composite weights for the single recall-ranking score. search_tag already
# encodes "<specialisation> (<category>)", so the category and specialisation
# terms are deliberately small -- they only break ties between tag-less records.
TAX_WEIGHTS = {"search_tag": 0.5, "skill": 0.3, "specialisation": 0.1, "category": 0.1}


@dataclass(frozen=True)
class TagSet:
    categories: frozenset[str] = frozenset()
    specialisations: frozenset[str] = frozenset()
    search_tags: frozenset[str] = frozenset()
    skills: frozenset[str] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.categories or self.specialisations or self.search_tags or self.skills)


def tags_from_nested(tags_obj) -> TagSet:
    """Build a TagSet from the v2 nested `tags` object.

    Accepts either the full `{"taxonomy_version", "categories": [...]}` object or
    the bare categories list that the SWE-side CSV stores in its `tags_json`
    column -- the two are the same data at different nesting depths.
    """
    cats_list = tags_obj if isinstance(tags_obj, list) else (tags_obj or {}).get("categories") or []
    cats, specs, stags, skills = set(), set(), set(), set()
    for c in cats_list:
        if c.get("name"):
            cats.add(c["name"])
        for s in c.get("specialisations") or []:
            if s.get("name"):
                specs.add(s["name"])
            if s.get("search_tag"):
                stags.add(s["search_tag"])
            skills.update(s.get("skills") or [])
    return TagSet(frozenset(cats), frozenset(specs), frozenset(stags), frozenset(skills))


def tags_from_flattened(category, specialisation, skills) -> TagSet:
    """Build a TagSet from the SWE-side flattened columns.

    `category`/`specialisation` are ' | '-joined; search_tags are the
    specialisation strings themselves, which is exactly the platform's
    "<track> (<sector>)" tag format.
    """
    split = lambda v: {x.strip() for x in str(v or "").split("|") if x.strip()}  # noqa: E731
    specs = split(specialisation)
    return TagSet(frozenset(split(category)), frozenset(specs),
                  frozenset(specs), frozenset(split(skills)))


def record_tags(record: dict) -> TagSet:
    """TagSet for a pipeline record, nested `tags`/`tags_json` first, then the
    flattened SWE-side columns. Returns an empty TagSet when there are none."""
    nested = record.get("tags") or record.get("tags_json")
    if isinstance(nested, str):
        try:
            nested = json.loads(nested) if nested.strip() else None
        except json.JSONDecodeError:
            nested = None
    if nested:
        return tags_from_nested(nested)
    if record.get("category") or record.get("specialisation") or record.get("skills"):
        return tags_from_flattened(record.get("category"), record.get("specialisation"),
                                   record.get("skills"))
    # v5 fallback: that corpus carries a single `industry` sector label instead of
    # the SkillsFuture tree. Map it onto the CATEGORY axis so the third recall
    # channel and the match features still have an axis to work with. Populated on
    # 92% of both sides; measured, same-industry pairs are grade>=2 78.7% of the
    # time against 44.4% for cross-industry pairs, so it is a real signal.
    if record.get("industry"):
        return TagSet(categories=frozenset({str(record["industry"])}))
    return TagSet()


def _coverage(need: frozenset[str], have: frozenset[str]) -> float:
    """Fraction of the gig's items the provider also holds. 0.0 when the gig asks
    for nothing on this axis -- an empty need is not a satisfied need."""
    if not need:
        return 0.0
    return len(need & have) / len(need)


def taxonomy_scores(gig: TagSet, prov: TagSet) -> dict[str, float]:
    """Per-pair match components. All in [0, 1] except the shared-skill count."""
    st = _coverage(gig.search_tags, prov.search_tags)
    sp = _coverage(gig.specialisations, prov.specialisations)
    ca = _coverage(gig.categories, prov.categories)
    sk = _coverage(gig.skills, prov.skills)
    return {
        "tax_search_tag": st,
        "tax_specialisation": sp,
        "tax_category": ca,
        "tax_skill": sk,
        "tax_n_shared_skills": float(len(gig.skills & prov.skills)),
        "taxonomy_score": (TAX_WEIGHTS["search_tag"] * st + TAX_WEIGHTS["skill"] * sk
                           + TAX_WEIGHTS["specialisation"] * sp + TAX_WEIGHTS["category"] * ca),
    }


TAXONOMY_FEATURES = ["tax_search_tag", "tax_specialisation", "tax_category", "tax_skill",
                     "tax_n_shared_skills", "taxonomy_score"]


class TaxonomyRetriever:
    """Third recall channel: rank providers by tag match against the gig.

    Providers with no overlapping tag are excluded rather than tied at zero --
    a zero-match provider carries no taxonomy signal and would otherwise inject
    thousands of arbitrary ranks into the RRF fusion.
    """

    def __init__(self, providers: list[dict]):
        self._ids: list[int] = []
        self._tags: list[TagSet] = []
        self._by_tag: dict[str, list[int]] = {}
        for p in providers:
            ts = record_tags(p)
            if not ts:
                continue
            idx = len(self._ids)
            self._ids.append(int(p["provider_id"]))
            self._tags.append(ts)
            for t in ts.search_tags | ts.categories | ts.specialisations | ts.skills:
                self._by_tag.setdefault(t, []).append(idx)

    def candidates(self, gig: TagSet) -> list[int]:
        """Indices of providers sharing at least one tag with the gig."""
        hit: set[int] = set()
        for t in gig.search_tags | gig.categories | gig.specialisations | gig.skills:
            hit.update(self._by_tag.get(t, ()))
        return sorted(hit)

    def rank(self, gig_tags: TagSet, top_n: int | None = None) -> list[tuple[int, float]]:
        """[(provider_id, taxonomy_score), ...] sorted descending, zeros dropped."""
        scored = [(self._ids[i], taxonomy_scores(gig_tags, self._tags[i])["taxonomy_score"])
                  for i in self.candidates(gig_tags)]
        scored = [(pid, s) for pid, s in scored if s > 0.0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_n] if top_n else scored
