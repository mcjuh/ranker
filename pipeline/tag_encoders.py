"""
Encoder registry for the tag channel's tagger (tag_corpus.py and eval_tag_encoder.py).

The tagger is a matmul of text vectors against tag-title vectors, so its quality is the quality of the encoder.
An `EncoderSpec` says which sentence-transformers model to load and how that model wants its inputs prefixed;
`embed(spec, texts, role)` returns raw (un-normalised) vectors and caches them under pipeline/cache/tagenc/<name>/
(git-ignored), keyed by the sha1 of each text, so an interrupted or repeated run only encodes what is missing.

Roles follow tag_corpus.py: "query" gets the model's query prefix, "doc" its passage prefix (empty for most models).
The model ids and prefixes are the ones the model cards give; check them against the card before trusting a new entry.
Vectors from different devices or encoder versions differ by about 1e-3 in cosine, so keep a baseline and a candidate
in one comparison on the same device and cache (the specs' cache names include the encoder name for that reason).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
CACHE_ROOT = "tagenc"
BATCH_SIZE = 32

MXBAI_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class EncoderSpec:
    name: str                     # short id: cache folder, result keys
    hf_id: str
    query_prefix: str = ""
    doc_prefix: str = ""
    max_seq_length: int | None = None   # None = the model's own default
    trust_remote_code: bool = False


SPECS: dict[str, EncoderSpec] = {s.name: s for s in [
    # the encoder the dense channel and the current tagger use: the baseline every candidate is compared with
    EncoderSpec("mxbai", "mixedbread-ai/mxbai-embed-large-v1", query_prefix=MXBAI_QUERY_PREFIX),
    EncoderSpec("bge-large", "BAAI/bge-large-en-v1.5",
                query_prefix="Represent this sentence for searching relevant passages: "),
    EncoderSpec("e5-large", "intfloat/e5-large-v2", query_prefix="query: ", doc_prefix="passage: "),
    EncoderSpec("gte-large", "thenlper/gte-large"),
    EncoderSpec("arctic-l", "Snowflake/snowflake-arctic-embed-l",
                query_prefix="Represent this sentence for searching relevant passages: "),
    EncoderSpec("mpnet", "sentence-transformers/all-mpnet-base-v2"),
    EncoderSpec("minilm-l12", "sentence-transformers/all-MiniLM-L12-v2"),
    EncoderSpec("all-roberta", "sentence-transformers/all-roberta-large-v1"),
    EncoderSpec("nomic", "nomic-ai/nomic-embed-text-v1.5", query_prefix="search_query: ",
                doc_prefix="search_document: ", trust_remote_code=True),
    EncoderSpec("qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B",
                query_prefix="Instruct: Given a text, retrieve the skills it describes\nQuery: "),
]}

_models: dict[str, object] = {}


def get_spec(name: str) -> EncoderSpec:
    if name not in SPECS:
        raise KeyError(f"unknown encoder {name!r}; known: {', '.join(SPECS)}")
    return SPECS[name]


def _model(spec: EncoderSpec):
    if spec.name not in _models:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(spec.hf_id, trust_remote_code=spec.trust_remote_code)
        if spec.max_seq_length:
            m.max_seq_length = spec.max_seq_length
        _models[spec.name] = m
    return _models[spec.name]


def prefixed(spec: EncoderSpec, texts: list[str], role: str) -> list[str]:
    """`texts` with the spec's prefix for `role` ("query" or "doc")."""
    if role not in ("query", "doc"):
        raise ValueError(f"role must be 'query' or 'doc', got {role!r}")
    prefix = spec.query_prefix if role == "query" else spec.doc_prefix
    return [prefix + t for t in texts]


def encode(spec: EncoderSpec, texts: list[str], role: str) -> np.ndarray:
    return _model(spec).encode(prefixed(spec, texts, role), batch_size=BATCH_SIZE, convert_to_numpy=True,
                               show_progress_bar=False)


def embed(spec: EncoderSpec, texts: list[str], role: str, tag: str = "") -> np.ndarray:
    """Raw vectors for `texts` in `role`, cached per (encoder, role, tag). `tag` separates text families that should
    not share a file (it is only a file name; the cache is keyed by text, so a repeated text is never encoded twice)."""
    from build_judging_pools_sat import CACHE_DIR, embed_cached

    folder = CACHE_DIR / CACHE_ROOT / spec.name
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{CACHE_ROOT}/{spec.name}/{role}{'_' + tag if tag else ''}"
    out = None
    chunk = 256                                  # save after every chunk so a long run that dies loses at most one
    for end in list(range(chunk, len(texts), chunk)) + [len(texts)]:
        out = embed_cached(name, texts[:end], lambda ts: encode(spec, ts, role))
    return out
