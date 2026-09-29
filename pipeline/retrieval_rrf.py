"""
RRF (Reciprocal Rank Fusion) -- combines a sparse (BM25) ranking and a dense
(embedding) ranking into one fused ranking, using ranks rather than raw
scores (which live on incomparable scales). Matches the weekly update's
pseudocode: score = 1/(k + rank_sparse) + 1/(k + rank_dense), k=60 default.
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
) -> list[tuple[int, float]]:
    """Returns [(provider_id, rrf_score), ...] sorted descending."""
    sparse_ranks = _ranks_from_scored_list(sparse_ranked)
    dense_ranks = _ranks_from_scored_list(dense_ranked)
    all_ids = set(sparse_ranks) | set(dense_ranks)

    fused = []
    for pid in all_ids:
        score = 0.0
        if pid in sparse_ranks:
            score += sparse_weight * (1.0 / (k + sparse_ranks[pid]))
        if pid in dense_ranks:
            score += dense_weight * (1.0 / (k + dense_ranks[pid]))
        fused.append((pid, score))

    return sorted(fused, key=lambda x: x[1], reverse=True)


def rrf_fuse_n(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[int, float]]:
    """N-way RRF: score = sum_i weights[i] / (k + rank_i), over the lists an id appears in.

    Same arithmetic as `rrf_fuse` (which is left untouched and still used by the two-channel
    path), so for two lists the scores are identical. Ties are broken by first appearance across
    the lists, in list order, so the output is deterministic. Returns
    [(id, rrf_score), ...] sorted descending."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(f"{len(ranked_lists)} ranked lists but {len(weights)} weights")

    fused: dict[int, float] = {}
    for weight, scored in zip(weights, ranked_lists):
        for pid, rank in _ranks_from_scored_list(scored).items():
            fused[pid] = fused.get(pid, 0.0) + weight * (1.0 / (k + rank))

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
