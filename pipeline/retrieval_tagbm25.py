"""
Tag-ID BM25: a third recall channel that matches on taxonomy tag IDs, not text.

Documents are sets of tag IDs (providers, or job roles when the channel is evaluated on the
taxonomy itself); a query is the set of tag IDs a user picked from the same vocabulary. Tags are
matched by ID, never by name. With binary term frequency Okapi BM25 collapses to

    B     N x T binary doc-tag matrix (scipy CSR)
    df_t  = sum_d B[d, t]
    idf_t = ln(1 + (N - df_t + 0.5) / (df_t + 0.5))     (never negative, even when df_t == N)
    c_d   = (k1 + 1) / (1 + k1 * (1 - b + b * |d| / avgdl))
    s     = c * (B @ (idf * q))                          (q is the 0/1 query vector over T)

`|d|` is the number of tags on document d. With tf fixed at 1, k1 only rescales every score by a
constant per document length, so `b` is the parameter worth tuning (b = 0 gives c = 1 and the score
is exactly the IDF-weighted overlap).

`rank()` returns `[(doc_id, score), ...]` sorted descending, the same shape `BM25Retriever.rank`
and `retrieval_dense.rank_from_raw` return, so it plugs straight into `retrieval_rrf`. Documents
with a zero score are dropped: a channel that has no evidence for a document must not give it an
(arbitrary, tie-broken) rank that RRF would then reward.

The same index also scores the ablation baselines (`method=`): plain `coverage`, un-normalised
`idf_overlap`, and `sum_levels` (needs proficiency levels per doc-tag pair).

numpy + scipy only; nothing here imports torch, sentence-transformers or rank_bm25.
"""
import numpy as np
import scipy.sparse as sp

METHODS = ("bm25", "coverage", "idf_overlap", "sum_levels")


class TagBM25:
    def __init__(self, doc_ids, doc_tags, k1: float = 1.2, b: float = 0.75, levels: dict | None = None):
        """doc_ids[i] is the identifier returned for the document whose tag IDs are doc_tags[i].
        Repeated tag IDs on one document collapse (binary tf). `levels`, if given, maps
        (doc_id, tag_id) -> proficiency level and is only used by method="sum_levels"."""
        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError(f"need k1 >= 0 and 0 <= b <= 1, got k1={k1}, b={b}")
        doc_ids = list(doc_ids)
        doc_tags = [frozenset(t) for t in doc_tags]
        if len(doc_ids) != len(doc_tags):
            raise ValueError(f"{len(doc_ids)} doc_ids but {len(doc_tags)} tag sets")
        self.doc_ids = doc_ids
        self.k1, self.b = k1, b

        self.tag_ids = sorted(set().union(*doc_tags))
        self._col = {t: j for j, t in enumerate(self.tag_ids)}
        rows, cols = [], []
        for i, tags in enumerate(doc_tags):
            for t in tags:
                rows.append(i)
                cols.append(self._col[t])
        n, m = len(doc_ids), len(self.tag_ids)
        self.B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, m))

        self.df = np.asarray(self.B.sum(axis=0)).ravel()
        self.dl = np.asarray(self.B.sum(axis=1)).ravel()
        mean_len = float(self.dl.mean()) if n else 0.0
        self.avgdl = mean_len if mean_len > 0 else 1.0
        self.idf = np.log1p((n - self.df + 0.5) / (self.df + 0.5))
        self.c = (k1 + 1.0) / (1.0 + k1 * (1.0 - b + b * self.dl / self.avgdl))

        self.L = None
        if levels is not None:
            coo = self.B.tocoo()
            vals = [levels[(doc_ids[i], self.tag_ids[j])] for i, j in zip(coo.row, coo.col)]
            self.L = sp.csr_matrix((np.asarray(vals, dtype=float), (coo.row, coo.col)), shape=(n, m))

    @classmethod
    def from_pairs(cls, pairs, doc_ids=None, **kwargs):
        """Build from (doc_id, tag_id) pairs, e.g. the speciality_tags bridge table. Documents are
        ordered by `doc_ids` if given (a doc with no pairs is kept, with zero tags), else sorted."""
        grouped: dict = {}
        for d, t in pairs:
            grouped.setdefault(d, set()).add(t)
        order = list(doc_ids) if doc_ids is not None else sorted(grouped)
        return cls(order, [grouped.get(d, ()) for d in order], **kwargs)

    def query_matrix(self, queries) -> sp.csr_matrix:
        """0/1 matrix, one row per query. Tag IDs outside the vocabulary and repeated IDs are ignored."""
        queries = list(queries)
        rows, cols = [], []
        for i, q in enumerate(queries):
            for j in {self._col[t] for t in q if t in self._col}:
                rows.append(i)
                cols.append(j)
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(queries), len(self.tag_ids)))

    def score_matrix(self, queries, method: str = "bm25") -> np.ndarray:
        """Dense (n_queries x n_docs) scores. `queries` is an iterable of tag-ID iterables."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        Q = self.query_matrix(queries)
        if method == "bm25":
            return (Q @ sp.diags(self.idf) @ self.B.T).toarray() * self.c
        if method == "idf_overlap":
            return (Q @ sp.diags(self.idf) @ self.B.T).toarray()
        if method == "coverage":
            overlap = (Q @ self.B.T).toarray()
            n_known = np.asarray(Q.sum(axis=1)).reshape(-1, 1)
            return np.divide(overlap, n_known, out=np.zeros_like(overlap), where=n_known > 0)
        if self.L is None:
            raise ValueError('method="sum_levels" needs levels= at construction')
        return (Q @ self.L.T).toarray()

    def scores(self, tag_ids, method: str = "bm25") -> np.ndarray:
        return self.score_matrix([tag_ids], method)[0]

    def rank(self, tag_ids, method: str = "bm25", top_k: int | None = None) -> list[tuple]:
        """[(doc_id, score), ...] descending, zero scores dropped. Ties keep document order."""
        s = self.scores(tag_ids, method)
        order = np.argsort(-s, kind="stable")
        order = order[s[order] > 0]
        if top_k is not None:
            order = order[:top_k]
        return [(self.doc_ids[i], float(s[i])) for i in order]
