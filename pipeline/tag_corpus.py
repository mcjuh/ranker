"""
Tags gigs and providers with taxonomy tag IDs by embedding nearest-tag, so the tag-ID BM25 channel
(retrieval_tagbm25.py) has tag sets to match on. data_sat has no tags of its own, only a coarse
`industry` field, so these tags are *predicted*; the channel inherits every error made here.

For each text the m most similar tag titles (cosine, mxbai-embed-large-v1, full 1024 dims) are kept:

    gigs       hirer_text(h) embedded as a QUERY (instruction prefix), tag titles as passages
    providers  provider_text(p) embedded as a PASSAGE, tag titles as queries (instruction prefix)

That is the asymmetric direction the model was trained for, and it reuses exactly the gig and
provider embeddings the dense channel already needs (same texts, same cache names as
build_judging_pools_sat.py), so tagging costs one pass over the tag titles on top of them.
Tags are ranked per text and the top --top-m (default 30) are stored with their cosines; the
channel picks how many to use, so m can be tuned without re-embedding.

Writes to pipeline/<data-dir>/:
    tags_hirers.json     {"model", "top_m", "tags": {hire_id: [[tag_id, cosine], ...]}}
    tags_providers.json  {"model", "top_m", "tags": {provider_id: [[tag_id, cosine], ...]}}

Run (from the repo root, in the .venv):
    python pipeline/tag_corpus.py --data-dir data_sat
    python pipeline/tag_corpus.py --data-dir data_sat --limit 24     # smoke test: prints, writes nothing

Embedding is cached in chunks under pipeline/cache/ (git-ignored), so an interrupted run resumes.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from corpus import hirer_text, provider_text
from greygigz import DEFAULT_EXPORT_DIR, load_taxonomy

BASE = Path(__file__).parent
DEFAULT_TOP_M = 30
EMBED_CHUNK = 128


def _unit(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def top_tags(text_vecs: np.ndarray, tag_vecs: np.ndarray, tag_ids: list[int], m: int) -> list[list[tuple[int, float]]]:
    """For each row of `text_vecs`, its `m` most similar tags by cosine: [[(tag_id, cosine), ...], ...],
    best first. Equal cosines keep tag order, so the output is deterministic."""
    sims = _unit(np.asarray(text_vecs, dtype=float)) @ _unit(np.asarray(tag_vecs, dtype=float)).T
    m = min(m, sims.shape[1])
    out = []
    for row in sims:
        best = np.argsort(-row, kind="stable")[:m]
        out.append([(tag_ids[j], float(row[j])) for j in best])
    return out


def embed_in_chunks(name: str, texts: list[str], encode, chunk: int = EMBED_CHUNK) -> np.ndarray:
    """Embeddings for `texts` via build_judging_pools_sat.embed_cached, saved after every chunk so a
    long CPU run that is interrupted loses at most one chunk. Each call only encodes texts the cache lacks."""
    from build_judging_pools_sat import embed_cached

    out = None
    for end in list(range(chunk, len(texts), chunk)) + [len(texts)]:
        out = embed_cached(name, texts[:end], encode)
    return out


def _write(path: Path, model: str, top_m: int, tagged: dict):
    payload = {"model": model, "top_m": top_m,
               "tags": {str(i): [[t, round(c, 4)] for t, c in tags] for i, tags in tagged.items()}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default="data_sat", help="dataset folder under pipeline/")
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    ap.add_argument("--top-m", type=int, default=DEFAULT_TOP_M)
    ap.add_argument("--limit", type=int, help="tag only the first N gigs and providers; print, write nothing")
    args = ap.parse_args()

    from retrieval_dense import BASE_MODEL_NAME, encode_docs, encode_queries

    data_dir = BASE / args.data_dir
    prefix = args.data_dir.removeprefix("data_")  # data_sat -> "sat": shares build_judging_pools_sat's cache names
    hirers = json.loads((data_dir / "hirers.json").read_text(encoding="utf-8"))
    providers = json.loads((data_dir / "providers.json").read_text(encoding="utf-8"))
    if args.limit:
        hirers, providers = hirers[: args.limit], providers[: args.limit]

    tax = load_taxonomy(args.export_dir)
    tag_ids = sorted(tax.tags)
    titles = [tax.tags[t] for t in tag_ids]
    print(f"{len(hirers)} gigs, {len(providers)} providers, {len(tag_ids)} tags; model {BASE_MODEL_NAME}", flush=True)

    t0 = time.time()
    tag_as_passage = embed_in_chunks("taxonomy_tags_mxbai_docs_by_text", titles, encode_docs)
    tag_as_query = embed_in_chunks("taxonomy_tags_mxbai_queries_by_text", titles, encode_queries)
    print(f"tag titles embedded in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    gig_vecs = embed_in_chunks(f"{prefix}_queries_mxbai_by_text", [hirer_text(h) for h in hirers], encode_queries)
    print(f"gigs embedded in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    prov_vecs = embed_in_chunks(f"{prefix}_docs_mxbai_by_text", [provider_text(p) for p in providers], encode_docs)
    print(f"providers embedded in {time.time() - t0:.0f}s", flush=True)

    gig_tags = dict(zip((h["hire_id"] for h in hirers), top_tags(gig_vecs, tag_as_passage, tag_ids, args.top_m)))
    prov_tags = dict(zip((p["provider_id"] for p in providers), top_tags(prov_vecs, tag_as_query, tag_ids, args.top_m)))

    for label, rows, tagged, key, title in [
        ("gig", hirers, gig_tags, "hire_id", "hire_title"),
        ("provider", providers, prov_tags, "provider_id", "about_title"),
    ]:
        for r in rows[:3]:
            picks = ", ".join(f"{tax.tags[t]} ({c:.2f})" for t, c in tagged[r[key]][:6])
            print(f"  {label} {r[key]} '{r[title][:70]}' -> {picks}")

    if args.limit:
        print("--limit given: nothing written")
        return
    _write(data_dir / "tags_hirers.json", BASE_MODEL_NAME, args.top_m, gig_tags)
    _write(data_dir / "tags_providers.json", BASE_MODEL_NAME, args.top_m, prov_tags)
    print(f"wrote tags_hirers.json and tags_providers.json to {data_dir}")


if __name__ == "__main__":
    main()
