"""
Provider recall channel built from tag_corpus.py output: providers are documents made of their
predicted tag IDs, and a gig's query is its predicted tag IDs, scored by retrieval_tagbm25.TagBM25.

    channel = TagChannel(DATA_DIR)              # reads tags_providers.json / tags_hirers.json
    channel.rank(hire_id)                       # [(provider_id, score), ...] best first, zero scores dropped
    channel.rank_text(gig_vec, tag_vecs, tag_ids)   # the same for a gig that is not in tags_hirers*.json (no API call)

The output has the same shape as BM25Retriever.rank and rank_from_raw, so it fuses with
retrieval_rrf.rrf_fuse_n. This channel is off unless a script is run with --tag-channel.

Hubness-corrected variants (tag_corpus.py --hubness center writes tags_*_hc.json, --hubness z writes tags_*_hz.json):

    TagChannel(DATA_DIR, variant="hc")                  # same BM25 over the corrected top-30 tags
    TagChannel(DATA_DIR, variant="hc", scorer="wcos")   # cosine of relu(score - tau) tag weights (TagWeightedCosine)
"""
import json
from pathlib import Path

import numpy as np

from retrieval_tagbm25 import TagBM25, TagWeightedCosine

# How many of the tagger's top tags are kept per text. 30 is the most the tagger stores and the value
# eval_tag_channel.py sat selected on the dev gigs (results_tag/sat.json, "selected"); fewer tags scored
# lower. Pools and labels must be built with the same value the evaluation ranks with.
DEFAULT_TOP_M_PROVIDER = 30
DEFAULT_TOP_M_HIRER = 30
# weighted scorer: tags whose corrected score is <= tau carry no weight. Centred scores are cosine differences
# (per-tag sd is about 0.05), z-scores are in sd units. Flat in a sweep over 0.03-0.07 (centred) and 1.0-1.5 (z).
VARIANT_TAU = {"hc": 0.05, "hz": 1.0}
SCORERS = ("bm25", "wcos")


def tag_file(data_dir: Path, side: str, variant: str = "") -> Path:
    """tags_hirers.json / tags_providers.json, or tags_hirers_<variant>.json for a variant (e.g. 'hz')."""
    return Path(data_dir) / f"tags_{side}{'_' + variant if variant else ''}.json"


def load_tags(path: Path, top_m: int) -> dict[int, list[int]]:
    """{id: [tag_id, ...]} keeping each text's `top_m` best tags (the file stores them best first)."""
    tagged = json.loads(Path(path).read_text(encoding="utf-8"))["tags"]
    return {int(i): [t for t, _cos in tags[:top_m]] for i, tags in tagged.items()}


def load_tag_scores(path: Path, top_m: int) -> dict[int, dict[int, float]]:
    """{id: {tag_id: score}} for each text's `top_m` best tags (the score is the cosine, or the corrected score in
    *_hc / *_hz files)."""
    tagged = json.loads(Path(path).read_text(encoding="utf-8"))["tags"]
    return {int(i): {t: s for t, s in tags[:top_m]} for i, tags in tagged.items()}


def load_tag_stats(path: Path):
    """(tag_ids, mean, sd) stored by tag_corpus.py --hubness in a tags_*_<variant>.json file, or None for a raw file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "mean" not in payload:
        return None
    return list(payload["tag_ids"]), np.array(payload["mean"], dtype=float), np.array(payload["sd"], dtype=float)


class TagChannel:
    def __init__(self, data_dir: Path, top_m_provider: int = DEFAULT_TOP_M_PROVIDER,
                 top_m_hirer: int = DEFAULT_TOP_M_HIRER, k1: float = 1.2, b: float = 0.75,
                 variant: str = "", scorer: str = "bm25", tau: float | None = None,
                 max_returned: int | None = None):
        """`variant` picks the tag files ('' = raw cosine, 'hc' = centred, 'hz' = z-scored). `scorer='wcos'` needs
        corrected files; `tau` defaults per variant (VARIANT_TAU). `max_returned` caps the list handed to RRF
        (None = every provider with evidence)."""
        data_dir = Path(data_dir)
        if scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}, got {scorer!r}")
        if scorer == "wcos" and variant not in VARIANT_TAU:
            raise ValueError("scorer='wcos' needs hubness-corrected tags: run tag_corpus.py --hubness center "
                             "and pass variant='hc'")
        for side in ("providers", "hirers"):
            path = tag_file(data_dir, side, variant)
            if not path.exists():
                hint = {"hc": " --hubness center", "hz": " --hubness z"}.get(variant, "")
                raise FileNotFoundError(f"{path} not found; run pipeline/tag_corpus.py{hint} first")
        self.scorer, self.max_returned, self.variant, self.top_m_hirer = scorer, max_returned, variant, top_m_hirer
        self._hirer_stats = load_tag_stats(tag_file(data_dir, "hirers", variant)) if variant else None
        if scorer == "wcos":
            providers = load_tag_scores(tag_file(data_dir, "providers", variant), top_m_provider)
            self.hirer_tags = load_tag_scores(tag_file(data_dir, "hirers", variant), top_m_hirer)
            self.index = TagWeightedCosine(list(providers), list(providers.values()),
                                          tau=VARIANT_TAU[variant] if tau is None else tau)
        else:
            providers = load_tags(tag_file(data_dir, "providers", variant), top_m_provider)
            self.hirer_tags = load_tags(tag_file(data_dir, "hirers", variant), top_m_hirer)
            self.index = TagBM25(list(providers), list(providers.values()), k1=k1, b=b)

    def rank(self, hire_id: int, top_k: int | None = None) -> list[tuple[int, float]]:
        top_k = self.max_returned if top_k is None else top_k
        return self.index.rank(self.hirer_tags[hire_id], top_k=top_k)

    def rank_text(self, text_vec, tag_vecs, tag_ids, top_k: int | None = None) -> list[tuple[int, float]]:
        """`rank()` for a gig that is not in the tag files, so a new gig needs no precomputed entry and no API call.
        `text_vec` is the gig embedded as a query (as tag_corpus.py embeds gigs), `tag_vecs` the taxonomy titles
        embedded as passages and `tag_ids` their ids in the order the tag files were built with (sorted). The
        per-tag correction that built a variant's file is applied from the statistics stored in it, so for a gig
        that is in the file the result equals rank(), up to the 4-decimal rounding of the stored scores."""
        from tag_corpus import top_tags

        standardise = None
        if self.variant:
            if self._hirer_stats is None:
                raise ValueError(f"tags_hirers_{self.variant}.json stores no per-tag statistics; "
                                 f"rebuild it with pipeline/tag_corpus.py --hubness")
            stored_ids, mean, sd = self._hirer_stats
            if stored_ids != list(tag_ids):
                raise ValueError("tag_ids differ from the tag list the hirer tag file was built with")
            standardise = (mean, sd)
        (tags,) = top_tags(np.asarray(text_vec, dtype=float)[None, :], tag_vecs, list(tag_ids), self.top_m_hirer,
                           standardise)
        query = {t: s for t, s in tags} if self.scorer == "wcos" else [t for t, _ in tags]
        top_k = self.max_returned if top_k is None else top_k
        return self.index.rank(query, top_k=top_k)
