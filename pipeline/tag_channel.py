"""
Provider recall channel built from tag_corpus.py output: providers are documents made of their
predicted tag IDs, and a gig's query is its predicted tag IDs, scored by retrieval_tagbm25.TagBM25.

    channel = TagChannel(DATA_DIR)              # reads tags_providers.json / tags_hirers.json
    channel.rank(hire_id)                       # [(provider_id, score), ...] best first, zero scores dropped

The output has the same shape as BM25Retriever.rank and rank_from_raw, so it fuses with
retrieval_rrf.rrf_fuse_n. This channel is off unless a script is run with --tag-channel.
"""
import json
from pathlib import Path

from retrieval_tagbm25 import TagBM25

# How many of the tagger's top tags are kept per text. These are starting points, to be tuned on a
# dev split (see the evaluation), not settled values.
DEFAULT_TOP_M_PROVIDER = 10
DEFAULT_TOP_M_HIRER = 10


def load_tags(path: Path, top_m: int) -> dict[int, list[int]]:
    """{id: [tag_id, ...]} keeping each text's `top_m` best tags (the file stores them best first)."""
    tagged = json.loads(Path(path).read_text(encoding="utf-8"))["tags"]
    return {int(i): [t for t, _cos in tags[:top_m]] for i, tags in tagged.items()}


class TagChannel:
    def __init__(self, data_dir: Path, top_m_provider: int = DEFAULT_TOP_M_PROVIDER,
                 top_m_hirer: int = DEFAULT_TOP_M_HIRER, k1: float = 1.2, b: float = 0.75):
        data_dir = Path(data_dir)
        for name in ("tags_providers.json", "tags_hirers.json"):
            if not (data_dir / name).exists():
                raise FileNotFoundError(f"{data_dir / name} not found; run pipeline/tag_corpus.py first")
        providers = load_tags(data_dir / "tags_providers.json", top_m_provider)
        self.hirer_tags = load_tags(data_dir / "tags_hirers.json", top_m_hirer)
        self.index = TagBM25(list(providers), list(providers.values()), k1=k1, b=b)

    def rank(self, hire_id: int, top_k: int | None = None) -> list[tuple[int, float]]:
        return self.index.rank(self.hirer_tags[hire_id], top_k=top_k)
