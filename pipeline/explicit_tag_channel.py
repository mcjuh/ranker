"""
Third recall channel over the tags users actually selected, not tags predicted from text.

The front end stores `search_tags` ("Specialisation (Category)") picked from the back-end taxonomy, so the tagger in
tag_corpus.py / tag_channel.py is not needed for records that carry them. Records are matched on the composite tag
string, never on the bare specialisation ("Operations" exists under both Financial Services and Healthcare).

    channel = ExplicitTagChannel(hirers, providers)          # lists of frontend_schema.adapt_hirer / adapt_provider records
    channel.rank("H001")                                     # [(provider_id, score), ...] best first, zero scores dropped
    channel.rank("H001", tiebreak=dense_scores)              # ties inside a score block broken by a second signal
    channel.features("H001", "P003")                         # per-pair overlap features for the Stage-2 ranker

The output has the same shape as TagChannel.rank, BM25Retriever.rank and rank_from_raw, so it fuses with
retrieval_rrf.rrf_fuse_n.

Scoring (two levels, because a record has only 1 to 3 tags and a hirer often shares none with any provider):

    score(d) = idf_overlap_tags(d) + category_weight * idf_overlap_categories(d)

- exact level: TagBM25 over composite tag strings. Providers carry 1-3 tags, so the BM25 length term varies (it was a
  constant when every provider had 30 predicted tags); b defaults to 0, which is the plain IDF-weighted overlap.
- category level: the same index over category names (Accountancy, Financial Services, ...). A hirer whose exact tags
  nobody carries still reaches providers in the same Function or Industry, at a lower weight. The credit is added on top
  of an exact match too, so an exact match always outranks a category-only match with the same categories.

Ties are common (binary tags, few distinct values). `tiebreak` orders them by a second per-provider signal (the dense
cosine is the obvious one); without it ties keep provider order, which is arbitrary for RRF.

Hirers with no tags fall back to `fallback.rank(hire_id)` (the predicted-tag TagChannel) when one is given, else get an
empty list. Providers without tags are never scored here; they still reach the fusion through BM25 and dense.
"""
import re

from retrieval_tagbm25 import TagBM25

_TAG = re.compile(r"^(?P<spec>.+?)\s*\((?P<cat>[^()]+)\)\s*$")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def parse_tag(tag: str) -> tuple[str, str | None]:
    """'Financial Accounting (Accountancy)' -> ('Financial Accounting', 'Accountancy'); no brackets -> (tag, None)."""
    m = _TAG.match(tag.strip())
    return (m.group("spec").strip(), m.group("cat").strip()) if m else (tag.strip(), None)


def tag_keys(record: dict) -> frozenset[str]:
    """Normalised composite tags of a record, read from `search_tags` only."""
    return frozenset(norm(t) for t in record.get("search_tags") or [] if t and t.strip())


def category_groups(record: dict) -> dict[str, str]:
    """{normalised category name: 'Function' | 'Industry' | ''} from `category` plus the brackets of `search_tags`."""
    groups = {norm(c["name"]): c.get("group", "") for c in record.get("category") or [] if c.get("name")}
    for t in record.get("search_tags") or []:
        cat = parse_tag(t)[1]
        if cat:
            groups.setdefault(norm(cat), "")
    return groups


def _id(record: dict, *keys: str) -> str:
    for k in keys:
        if k in record:
            return str(record[k])
    raise KeyError(f"record has none of {keys}")


class ExplicitTagChannel:
    def __init__(self, hirers: list[dict], providers: list[dict], k1: float = 1.2, b: float = 0.0,
                 category_weight: float = 0.25, fallback=None):
        if category_weight < 0:
            raise ValueError("category_weight must be >= 0")
        self.category_weight, self.fallback = category_weight, fallback
        self.provider_ids = [_id(p, "provider_id") for p in providers]
        self.hirer_ids = [_id(h, "hire_id", "hirer_id") for h in hirers]
        self._h_tags = {i: tag_keys(h) for i, h in zip(self.hirer_ids, hirers)}
        self._p_tags = {i: tag_keys(p) for i, p in zip(self.provider_ids, providers)}
        self._h_cats = {i: category_groups(h) for i, h in zip(self.hirer_ids, hirers)}
        self._p_cats = {i: category_groups(p) for i, p in zip(self.provider_ids, providers)}
        self.tag_index = TagBM25(self.provider_ids, [self._p_tags[i] for i in self.provider_ids], k1=k1, b=b)
        self.cat_index = TagBM25(self.provider_ids, [set(self._p_cats[i]) for i in self.provider_ids], k1=k1, b=b)

    def has_tags(self, hire_id: str) -> bool:
        return bool(self._h_tags.get(hire_id))

    def scores(self, hire_id: str):
        """numpy vector over `provider_ids`: exact IDF overlap + category_weight * category IDF overlap."""
        s = self.tag_index.scores(self._h_tags[hire_id])
        c = self.cat_index.scores(set(self._h_cats[hire_id]))
        return s + self.category_weight * c

    def rank(self, hire_id: str, tiebreak: dict | None = None, top_k: int | None = None) -> list[tuple[str, float]]:
        if not self.has_tags(hire_id):
            return self.fallback.rank(hire_id, top_k) if self.fallback is not None else []
        s = self.scores(hire_id)
        tb = tiebreak or {}
        order = sorted((i for i in range(len(s)) if s[i] > 0),
                       key=lambda i: (-round(float(s[i]), 9), -float(tb.get(self.provider_ids[i], 0.0)), i))
        if top_k is not None:
            order = order[:top_k]
        return [(self.provider_ids[i], float(s[i])) for i in order]

    def features(self, hire_id: str, provider_id: str) -> dict:
        """Per-pair overlap features. function_match / industry_match are 1 when the two sides share at least one
        category of that group; tag_overlap counts shared composite tags; tag_idf_overlap uses the provider-pool IDF."""
        ht, pt = self._h_tags[hire_id], self._p_tags[provider_id]
        hc, pc = self._h_cats[hire_id], self._p_cats[provider_id]
        shared_tags = ht & pt
        shared_cats = set(hc) & set(pc)
        idf = {t: float(self.tag_index.idf[self.tag_index._col[t]]) for t in shared_tags if t in self.tag_index._col}
        union = ht | pt
        return {
            "tag_overlap": len(shared_tags),
            "tag_idf_overlap": round(sum(idf.values()), 6),
            "tag_jaccard": round(len(shared_tags) / len(union), 6) if union else 0.0,
            "category_overlap": len(shared_cats),
            "function_match": int(any(hc[c] == "Function" for c in shared_cats)),
            "industry_match": int(any(hc[c] == "Industry" for c in shared_cats)),
        }
