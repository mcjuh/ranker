"""RRF (Reciprocal Rank Fusion).

Card 2's three recall channels -- taxonomy (match scoring), lexical (BM25) and
semantic (dense embeddings) -- are fused here on RANKS rather than raw scores,
which live on incomparable scales. With two channels this is exactly the weekly
update's pseudocode:

    score = 1/(k + rank_sparse) + 1/(k + rank_dense), k=60 default

With the taxonomy channel present it is the natural three-term extension:

    score = 1/(k + rank_taxonomy) + 1/(k + rank_sparse) + 1/(k + rank_dense)

A channel that does not contain a provider simply contributes nothing for it,
so a two-channel run is unaffected by the third being available.
"""


def _ranks_from_scored_list(scored: list[tuple[int, float]]) -> dict[int, int]:
    """scored is already sorted descending by score; rank 1 = best."""
    return {pid: i + 1 for i, (pid, _score) in enumerate(scored)}


def rrf_fuse(
    sparse_ranked: list[tuple[int, float]],
    dense_ranked: list[tuple[int, float]],
    k: int = 60,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
    taxonomy_ranked: list[tuple[int, float]] | None = None,
    taxonomy_weight: float = 1.0,
) -> list[tuple[int, float]]:
    """Returns [(provider_id, rrf_score), ...] sorted descending.

    `taxonomy_ranked` is the third (Card 2) channel; pass None to reproduce the
    original two-term formula exactly.
    """
    channels = [
        (_ranks_from_scored_list(sparse_ranked), sparse_weight),
        (_ranks_from_scored_list(dense_ranked), dense_weight),
    ]
    if taxonomy_ranked is not None:
        channels.append((_ranks_from_scored_list(taxonomy_ranked), taxonomy_weight))

    all_ids = set().union(*(set(ranks) for ranks, _ in channels))

    fused = []
    for pid in all_ids:
        score = 0.0
        for ranks, weight in channels:
            if pid in ranks:
                score += weight * (1.0 / (k + ranks[pid]))
        fused.append((pid, score))

    return sorted(fused, key=lambda x: x[1], reverse=True)
