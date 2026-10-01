"""
Bench for tag-channel encoders: which embedding model (or ensemble) makes the best tagger?

The tagger is a matmul, text vectors against the 2,088 tag-title vectors, so its quality is the encoder's. Every
candidate here is turned into per-tag-centred scores (cosine minus the tag's mean over the corpus of texts, the
current best recipe, TAG_CHANNEL.md section 11) divided by their pooled standard deviation, so encoders whose cosines
sit on different scales are comparable and the weighted-cosine threshold (tau = 1.0 sd, which is 0.05 for mxbai) means
the same thing for all. An ensemble "a+b" averages those unit-free matrices.

Three benches, cheapest first; none calls the LLM grader.
  A  gold tags      each role description (name + description, a passage; tag titles are queries, the provider
                    direction) is tagged and compared with the role's gold tags: precision / recall at m, hit-any,
                    and role@k / track@k when the role is retrieved from its predicted tags (TagBM25, b 0.75). Dev and
                    test halves of the equivalence-class split of eval_tag_channel.RoleWorld (803 classes each).
                    Role text is cleaner than gig and provider text, so this is an upper bound.
  B  real gigs      the dev gigs' ORIGINAL-graded pairs (llm_judgments_merged.json; the audit confirmed original
                    positives) ranked by the cosine of the gig's and provider's weighted tag vectors
                    (relu(score - tau) over the top 30 tags, as TagChannel scorer 'wcos'): per-gig AUC, NDCG@5 and
                    NDCG@10. Same fixed pairs for every candidate, so no unjudged-pair bias. The pool is small
                    (about 21 pairs and 2 positives per gig) and was built from BM25 and dense, so this measures
                    reranking inside a short list, not the recall the channel adds: AUC is its sensitive metric,
                    and its CIs are wide.
  C  diagnostics    hubness (share of all top-30 tag slots held by the 50 most frequent tags, tags never picked, most
                    frequent tag) and complementarity: mean per-gig Spearman of the candidate's pair scores with
                    dense's over all providers, and the share of the candidate's top-50 that the RRF(bm25, dense)
                    top-50 lacks (novelty: neither good nor bad alone, read it next to B). The original labels
                    cannot measure marginal recall, because their pool was built from the RRF top 20.

Differences to the baseline encoder are paired, with 95% bootstrap CIs over classes (A) or gigs (B, C). Selection is
meant to use the dev half only; `--split test` is for confirming a finalist.

Run from the repo root, in the GPU venv for the first embedding pass (see tag_encoders.py), then anywhere:
    python pipeline/eval_tag_encoder.py --encoders mxbai bge-large e5-large mxbai+bge-large
    python pipeline/eval_tag_encoder.py --encoders mxbai --benches A          # roles only
Results are written to pipeline/results_tag/encoder_bench_<split>.json (or --out-name).
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

import eval_tag_channel as etc
import eval_tag_downstream as ds
import eval_tag_variants as ev
from corpus import hirer_text, provider_text
from greygigz import DEFAULT_EXPORT_DIR, load_taxonomy
from tag_corpus import tag_similarities
from tag_encoders import embed, get_spec

BASE = Path(__file__).parent
SEED = ds.SEED
M_GRID = (5, 10, 20, 30)
TOP_M = 30            # tags kept per text for the real-data benches (the channel's default)
TAU_SD = 1.0          # weighted-cosine threshold in pooled-sd units (0.05 for mxbai, the tuned VARIANT_TAU['hc'])
SLOT_TOP = 50         # hubness: share of all tag slots held by this many most frequent tags
MARGINAL_K = 50
SPLITS = ("dev", "test")


# ---------------------------------------------------------------------------
# Scores: centred, unit-free matrices
# ---------------------------------------------------------------------------

def centred_unit(sims: np.ndarray) -> np.ndarray:
    """Per-tag centred scores over the rows of `sims`, divided by their pooled standard deviation.

    Subtracting the tag's mean over the corpus is TAG_CHANNEL.md section 11's fix for hubness; the pooled scale only
    makes encoders comparable (it does not change the order of any text's tags)."""
    c = sims - sims.mean(axis=0, keepdims=True)
    sd = c.std()
    return c / sd if sd > 0 else c


def ensemble(mats: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted mean of unit-free score matrices (same rows and columns), re-scaled to unit pooled sd."""
    m = np.average(mats, axis=0, weights=weights)
    sd = m.std()
    return m / sd if sd > 0 else m


def top_indices(scores: np.ndarray, m: int) -> np.ndarray:
    """Column indices of each row's `m` best scores, best first; equal scores keep tag order (deterministic)."""
    return np.argsort(-scores, axis=1, kind="stable")[:, :m]


def gold_tag_metrics(scores: np.ndarray, gold: np.ndarray, ms=M_GRID) -> dict[str, np.ndarray]:
    """Per-text precision@m, recall@m and hit-any@m of the top-m tags against a boolean gold matrix (texts x tags)."""
    hits = np.take_along_axis(gold, top_indices(scores, max(ms)), axis=1).cumsum(axis=1)
    n_gold = np.maximum(gold.sum(axis=1), 1)
    out = {}
    for m in ms:
        out[f"precision@{m}"] = hits[:, m - 1] / m
        out[f"recall@{m}"] = hits[:, m - 1] / n_gold
        out[f"hit@{m}"] = (hits[:, m - 1] > 0).astype(float)
    return out


def weighted_vectors(scores: np.ndarray, m: int = TOP_M, tau: float = TAU_SD) -> np.ndarray:
    """Each text's tag weights relu(score - tau) over its top-m tags, zero elsewhere (TagWeightedCosine's vectors)."""
    out = np.zeros_like(scores)
    idx = top_indices(scores, m)
    rows = np.arange(len(scores))[:, None]
    out[rows, idx] = np.maximum(scores[rows, idx] - tau, 0.0)
    return out


def unit_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n == 0, 1.0, n)


def slot_stats(scores: np.ndarray, m: int = TOP_M, top: int = SLOT_TOP) -> dict[str, float]:
    """Hubness of the top-m tags: share of all slots held by the `top` most frequent tags, tags never picked, and the
    number of texts carrying the most frequent tag."""
    counts = np.bincount(top_indices(scores, m).ravel(), minlength=scores.shape[1])
    ordered = np.sort(counts)[::-1]
    return {"slot_share_top50": float(ordered[:top].sum() / counts.sum()), "tags_used": int((counts > 0).sum()),
            "tags_never_picked": int((counts == 0).sum()), "max_tag_freq": int(ordered[0])}


def auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Probability a random positive outscores a random negative (ties count half); nan if either side is empty."""
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def rank_pool(scores: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Indices of a gig's pooled providers, best first; ties broken by a fixed random key so no candidate benefits from
    provider order (most weighted tag vectors tie at 0)."""
    return np.lexsort((tiebreak, -scores))


# ---------------------------------------------------------------------------
# Candidates: one centred score matrix per text family
# ---------------------------------------------------------------------------

class Candidate:
    """Centred unit-free score matrices for the role texts (A) and the gig and provider texts (B, C) of one encoder, recipe
    or ensemble. `roles` maps a split ("dev", "test") to the role-text matrix to use when evaluating the classes of that
    split: a recipe that learns from the taxonomy's roles (prototypes, kNN) builds it only from the *other* half's
    classes, so an evaluated role never informs its own tags; for a title-only recipe the two are the same array.
    `raw_roles` keeps the uncorrected cosines of a title-only recipe for the raw-tagger reference row."""
    def __init__(self, name: str, roles: dict, gigs, provs, raw_roles=None):
        self.name, self.roles, self.gigs, self.provs, self.raw_roles = name, roles, gigs, provs, raw_roles


class Corpora:
    """The texts every candidate tags, built once: role descriptions of every equivalence class, tag titles, gigs and
    providers of the data folder."""
    def __init__(self, export_dir: Path, data_dir: Path, seed: int):
        self.tax = load_taxonomy(export_dir)
        self.world = etc.RoleWorld(self.tax, seed)
        self.tag_ids = sorted(self.tax.tags)
        self.titles = [self.tax.tags[t] for t in self.tag_ids]
        self.role_texts = [f"{self.tax.roles[r]['name']}. {self.tax.roles[r]['description']}" for r in self.world.reps]
        load = lambda n: json.loads((data_dir / n).read_text(encoding="utf-8"))
        self.hirers, self.providers = load("hirers.json"), load("providers.json")
        self.gig_texts = [hirer_text(h) for h in self.hirers]
        self.prov_texts = [provider_text(p) for p in self.providers]
        self.gold = np.zeros((len(self.world.reps), len(self.tag_ids)), dtype=bool)
        col = {t: j for j, t in enumerate(self.tag_ids)}
        for i, rep in enumerate(self.world.reps):
            for t in self.tax.role_tags[rep]:
                self.gold[i, col[t]] = True
        self.classes = {"all": np.arange(len(self.world.reps)),
                        **{sp: np.flatnonzero(self.world.split_of_class[sp]) for sp in SPLITS}}

    def train_classes(self, split: str) -> np.ndarray:
        """Classes whose roles a taxonomy-learning recipe may use while tagging the classes of `split`."""
        return self.classes["test" if split == "dev" else "dev"]


class Embeds:
    """Raw (cached) embeddings of every text family under one encoder. Tag titles are embedded twice: as queries, to be
    matched with role and provider texts embedded as passages, and as passages, to be matched with gigs embedded as
    queries (the two directions tag_corpus.py uses)."""
    def __init__(self, name: str, corp: Corpora):
        spec = get_spec(name)
        t0 = time.time()
        self.tags_q, self.tags_p = embed(spec, corp.titles, "query", "tags"), embed(spec, corp.titles, "doc", "tags")
        self.roles = embed(spec, corp.role_texts, "doc", "roles")
        self.gigs = embed(spec, corp.gig_texts, "query", "gigs")
        self.provs = embed(spec, corp.prov_texts, "doc", "provs")
        print(f"  {name}: embeddings ready in {time.time() - t0:.0f}s", flush=True)


def tag_prototypes(title_vecs: np.ndarray, role_vecs: np.ndarray, gold: np.ndarray, a: float) -> np.ndarray:
    """Tag vectors that blend the title with the roles carrying the tag: a * unit(title) + (1 - a) * unit(mean of the
    unit role vectors whose gold tags include it). A tag no role carries keeps its title vector."""
    titles, roles = unit_rows(title_vecs), unit_rows(role_vecs)
    proto = unit_rows(gold.T.astype(float) @ roles)        # (tags x d)
    has = gold.sum(axis=0) > 0
    return np.where(has[:, None], a * titles + (1 - a) * proto, titles)


def role_knn_scores(text_vecs: np.ndarray, role_vecs: np.ndarray, gold: np.ndarray, tau: float) -> np.ndarray:
    """Soft map each text to the roles it resembles and inherit their curated tag sets:
    softmax(cos(text, roles) / tau) @ gold. Rows are weights over tags, in [0, 1]."""
    sims = unit_rows(text_vecs) @ unit_rows(role_vecs).T / tau
    sims -= sims.max(axis=1, keepdims=True)
    w = np.exp(sims)
    w /= w.sum(axis=1, keepdims=True)
    return w @ gold.astype(float)


HEAD_SCALE = 20.0     # softmax inverse temperature of the head's listwise loss (cosines live in a narrow band)
HEAD_STEPS = 150
HEAD_LR = 2e-3


def fit_tag_head(title_vecs: np.ndarray, role_vecs: np.ndarray, gold: np.ndarray, lam: float,
                 steps: int = HEAD_STEPS, lr: float = HEAD_LR, scale: float = HEAD_SCALE) -> np.ndarray:
    """Stage 3: a residual on the tag vectors, learnt from the gold tags of `role_vecs` and held near the titles.

    Returns the residual R (tags x d, add it to the unit titles with `head_tags`) that minimises the listwise loss
        -mean_roles mean_{gold tags g} log softmax_j(scale * cos(role, unit(title_j + R_j)))_g  +  lam * mean ||R_j||^2
    from R = 0 (full-batch Adam, `steps` steps). The penalty keeps R small in the title scale, so lam -> infinity is the
    title recipe. Every tag takes part in the softmax: a tag some role carries is pulled toward it, any tag is pushed
    away from the roles that wrongly attract it (a tag no role carries is only ever pushed). Roles with no gold tag are
    skipped. Needs torch (CPU is enough)."""
    import torch

    titles = torch.tensor(unit_rows(title_vecs), dtype=torch.float32)
    roles = torch.tensor(unit_rows(role_vecs), dtype=torch.float32)
    keep = gold.sum(axis=1) > 0
    target = torch.tensor(gold[keep].astype(np.float32))
    target = target / target.sum(dim=1, keepdim=True)
    roles = roles[torch.tensor(keep)]
    R = torch.zeros_like(titles, requires_grad=True)
    opt = torch.optim.Adam([R], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        tags = torch.nn.functional.normalize(titles + R, dim=1)
        logp = torch.log_softmax(scale * roles @ tags.T, dim=1)
        loss = -(target * logp).sum(dim=1).mean() + lam * (R ** 2).sum(dim=1).mean()
        loss.backward()
        opt.step()
    return R.detach().numpy().astype(np.float64)


def head_tags(title_vecs: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Unit tag vectors unit(unit(title) + residual). The residual is learnt on the query-side title vectors and added
    to either side's titles (for models with one tower the two sides are the same vectors)."""
    return unit_rows(unit_rows(title_vecs) + residual)


HUB = re.compile(r"^(?P<kind>center|csls|dsm)(?P<arg>[0-9.]*)$")
CSLS_K = 10


def hub_correct(sims: np.ndarray, kind: str, arg: float | None = None) -> np.ndarray:
    """Hubness correction of a (texts x tags) cosine matrix: a per-tag shift that REPLACES centring (a later centring
    would cancel it), with statistics taken over the rows of `sims`, as centring's are. csls and dsm penalise a tag by
    how strongly it attracts the texts, which the mean misses when a hub tag has an ordinary mean but a fat upper tail.
      center  s - the tag's mean over the texts (the current recipe, here so it can be compared through the same path).
      csls  s - r_tag / 2, r_tag = the tag's mean cosine over its `arg` (default 10) closest texts (cross-domain
            similarity local scaling; for ranking tags within a text the text-side term is a constant).
      dsm   s - t * log mean_texts exp(s / t), t = `arg` pooled sds of the centred matrix (default 1): the dual
            softmax / inverted softmax. t -> infinity is centring; a small t penalises a tag by its maximum, so the tag
            slots are shared more evenly."""
    if kind == "center":
        return sims - sims.mean(axis=0, keepdims=True)
    if kind == "csls":
        k = max(int(arg or CSLS_K), 1)
        r = np.sort(sims, axis=0)[-k:].mean(axis=0)
        return sims - 0.5 * r
    if kind == "dsm":
        t = (arg or 1.0) * float((sims - sims.mean(axis=0, keepdims=True)).std())
        if t <= 0:
            return sims
        from scipy.special import logsumexp
        return sims - t * (logsumexp(sims / t, axis=0) - np.log(len(sims)))
    raise ValueError(f"unknown hubness correction {kind!r}")


def global_unit(x: np.ndarray) -> np.ndarray:
    """Subtract the overall mean and divide by the overall sd: the unit-free scale of `centred_unit` without touching the
    per-tag shifts a hubness correction applied. For a centred matrix it is the identity up to the sd."""
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x - x.mean()


def split_hub(name: str) -> tuple[str, tuple[str, float | None] | None]:
    """"qwen3-0.6b:head10~dsm0.5" -> ("qwen3-0.6b:head10", ("dsm", 0.5)); no "~" -> (name, None)."""
    base, _, hub = name.partition("~")
    if not hub:
        return base, None
    m = HUB.match(hub)
    if not m:
        raise ValueError(f"cannot parse hubness correction {hub!r} in {name!r}")
    return base, (m["kind"], float(m["arg"]) if m["arg"] else None)


RECIPE = re.compile(r"^(?P<enc>[^:]+)(?::(?P<kind>title|proto|knn|mix|head)(?P<args>[0-9._]*))?$")


def parse_name(name: str) -> tuple[str, str, list[float]]:
    """"bge-large" -> title recipe; "mxbai:proto0.5" (blend weight a on the title); "mxbai:knn0.05" (softmax
    temperature); "mxbai:mix1.0_0.05" (title + 1.0 * knn at tau 0.05); "mxbai:head0.1" (supervised residual head with
    penalty 0.1, fit_tag_head)."""
    m = RECIPE.match(name)
    if not m:
        raise ValueError(f"cannot parse candidate {name!r}")
    args = [float(x) for x in m["args"].split("_") if x] if m["args"] else []
    return m["enc"], m["kind"] or "title", args


def build_candidate(name: str, corp: Corpora, embeds: dict | None = None) -> Candidate:
    """Score one encoder with one recipe, or "a+b+..." for the unit-free mean of several candidates. `embeds` caches the
    per-encoder embeddings across candidates."""
    embeds = {} if embeds is None else embeds
    if "+" in name:                              # each part may carry its own "~correction"
        parts = [build_candidate(n, corp, embeds) for n in name.split("+")]
        roles = {sp: ensemble([p.roles[sp] for p in parts]) for sp in SPLITS}
        return Candidate(name, roles, *(ensemble([getattr(p, f) for p in parts]) for f in ("gigs", "provs")))
    return _build_single(name, *split_hub(name), corp, embeds)


def _build_single(name: str, base: str, hub: tuple[str, float | None] | None, corp: Corpora, embeds: dict) -> Candidate:
    """One encoder with one recipe and, if `hub` is set, a hubness correction in place of centring."""
    enc, kind, args = parse_name(base)
    if hub and kind not in ("title", "proto", "head"):
        raise ValueError(f"{name!r}: a hubness correction needs cosine scores (title, proto or head), not {kind}")
    norm = (lambda sims: global_unit(hub_correct(sims, *hub))) if hub else centred_unit
    if enc not in embeds:
        embeds[enc] = Embeds(enc, corp)
    e = embeds[enc]
    families = {"roles": (e.roles, e.tags_q), "gigs": (e.gigs, e.tags_p), "provs": (e.provs, e.tags_q)}

    residuals: dict[bytes, np.ndarray] = {}          # one fit per set of training classes, shared by the three families
    if kind == "head":
        for train in {corp.classes["all"].tobytes(): corp.classes["all"],
                      **{corp.train_classes(sp).tobytes(): corp.train_classes(sp) for sp in SPLITS}}.values():
            residuals[train.tobytes()] = fit_tag_head(e.tags_q, e.roles[train], corp.gold[train], args[0])

    def scores(train: np.ndarray, family: str) -> np.ndarray:
        text, title = families[family]
        role_vecs, gold = e.roles[train], corp.gold[train]
        if kind == "title":
            return norm(tag_similarities(text, title))
        if kind == "proto":
            return norm(tag_similarities(text, tag_prototypes(title, role_vecs, gold, args[0])))
        if kind == "head":
            return norm(tag_similarities(text, head_tags(title, residuals[train.tobytes()])))
        if kind == "knn":
            return centred_unit(role_knn_scores(text, role_vecs, gold, args[0]))
        lam, tau = args
        return ensemble([centred_unit(tag_similarities(text, title)),
                         centred_unit(role_knn_scores(text, role_vecs, gold, tau))], [1.0, lam])

    everything = corp.classes["all"]
    if kind == "title":
        roles, raw = dict.fromkeys(SPLITS, scores(everything, "roles")), tag_similarities(e.roles, e.tags_q)
    else:
        roles, raw = {sp: scores(corp.train_classes(sp), "roles") for sp in SPLITS}, None
    return Candidate(name, roles, scores(everything, "gigs"), scores(everything, "provs"), raw_roles=raw)


# ---------------------------------------------------------------------------
# Bench A: gold tags
# ---------------------------------------------------------------------------

A_METRICS = [f"{k}@{m}" for m in M_GRID for k in ("precision", "recall", "hit")] + \
            ["role@1", "role@5", "role@10", "track@1", "track@3"]


def bench_a(cand: Candidate, corp: Corpora, split: str, retrieval_m: int = 5) -> dict[str, dict[str, np.ndarray]]:
    """{'raw'|'hc': {metric: per-class array}} on the classes of `split`. Role retrieval uses the top `retrieval_m`
    predicted tags (m = 5 was the dev-selected value in TAG_CHANNEL.md section 3)."""
    world = corp.world
    classes = np.flatnonzero(world.split_of_class[split])
    idx = world.index()
    out = {}
    for label, S in (("raw", cand.raw_roles), ("hc", cand.roles[split])):
        if S is None:
            continue
        S = S[classes]
        ms = gold_tag_metrics(S, corp.gold[classes])
        top = top_indices(S, retrieval_m)
        queries = [[corp.tag_ids[j] for j in row] for row in top]
        hm = etc.hit_metrics(world, idx.score_matrix(queries), classes)
        ms.update({k: hm[k] for k in ("role@1", "role@5", "role@10", "track@1", "track@3")})
        out[label] = ms
    return out


# ---------------------------------------------------------------------------
# Benches B and C: real gigs
# ---------------------------------------------------------------------------

class RealData:
    """Everything the real-data benches share: dev gigs, their original-graded pool, dense and RRF2 references."""
    def __init__(self, corp: Corpora, data_dir: Path, split: str, seed: int):
        self.gig_ids = ev.split_gigs(data_dir, seed, split)
        gig_row = {str(h["hire_id"]): i for i, h in enumerate(corp.hirers)}
        self.rows = np.array([gig_row[str(g)] for g in self.gig_ids])
        self.prov_ids = [int(p["provider_id"]) for p in corp.providers]
        load = lambda n: json.loads((data_dir / n).read_text(encoding="utf-8"))
        grades, self.gt = load("llm_judgments_merged.json"), load("ground_truth_llm.json")
        col = {p: j for j, p in enumerate(self.prov_ids)}
        self.pool = {g: (np.array([col[int(p)] for p in grades[str(g)]]),
                         np.array([v >= 2 for v in grades[str(g)].values()])) for g in self.gig_ids}
        self.tiebreak = ds.rng_for(seed, "enc/tiebreak").random(len(self.prov_ids))
        self._references(corp)

    def _references(self, corp: Corpora):
        """Dense cosine (the cached mxbai vectors the dense channel uses) and the RRF2 lists for the dev gigs."""
        doc_raw = etc.cached_embeddings("sat_docs_mxbai_by_text", corp.prov_texts)
        q_raw = etc.cached_embeddings("sat_queries_mxbai_by_text", [corp.gig_texts[i] for i in self.rows])
        # the cached vectors are float16; cosines in float16 overflow and lose the order of near ties
        self.dense = unit_rows(q_raw.astype(np.float64)) @ unit_rows(doc_raw.astype(np.float64)).T
        hirers = [corp.hirers[i] for i in self.rows]
        ex = etc.existing_channels(hirers, corp.providers, doc_raw, q_raw)
        col = {p: j for j, p in enumerate(self.prov_ids)}
        self.rrf2_top = {g: {col[int(p)] for p in ex["rrf2"][str(g)].ids[:MARGINAL_K]} for g in self.gig_ids}


def pair_scores(cand: Candidate, real: RealData, tau: float = TAU_SD) -> np.ndarray:
    """(dev gigs x providers) cosine of the gigs' and providers' weighted tag vectors."""
    G = unit_rows(weighted_vectors(cand.gigs[real.rows], TOP_M, tau))
    P = unit_rows(weighted_vectors(cand.provs, TOP_M, tau))
    return G @ P.T


def pool_metrics(score_matrix: np.ndarray, real: RealData) -> dict[str, np.ndarray]:
    """Per-gig NDCG@5, NDCG@10 and AUC over the gig's original-graded pool; gigs without a positive or a negative are
    nan (dropped from means by nanmean)."""
    from evaluate import ndcg_at_k
    out = {"NDCG@5": [], "NDCG@10": [], "AUC": []}
    for i, g in enumerate(real.gig_ids):
        cols, positive = real.pool[g]
        s = score_matrix[i, cols]
        order = rank_pool(s, real.tiebreak[cols])
        ranked = [real.prov_ids[cols[j]] for j in order]
        row = real.gt.get(str(g), {})
        valid = positive.any() and not positive.all()
        out["NDCG@5"].append(ndcg_at_k(ranked, row, 5) if valid else np.nan)
        out["NDCG@10"].append(ndcg_at_k(ranked, row, 10) if valid else np.nan)
        out["AUC"].append(auc(s, positive))
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def complementarity(score_matrix: np.ndarray, real: RealData) -> dict[str, np.ndarray]:
    """Per-gig Spearman of the candidate's pair scores with dense's over all providers, and the share of the
    candidate's top-50 providers that are not in the gig's RRF(bm25, dense) top-50."""
    ranks = lambda M: np.apply_along_axis(rankdata, 1, M)
    a, b = ranks(score_matrix), ranks(real.dense)
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    spearman = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    novelty = [len({int(j) for j in rank_pool(score_matrix[i], real.tiebreak)[:MARGINAL_K]} - real.rrf2_top[g])
               / MARGINAL_K for i, g in enumerate(real.gig_ids)]
    return {"spearman_vs_dense": spearman, "novelty@50": np.array(novelty)}


def bench_real(cand: Candidate, real: RealData) -> tuple[dict, dict]:
    """(per-gig metrics of benches B and C, scalar hubness stats per side)."""
    S = pair_scores(cand, real)
    per_gig = {**pool_metrics(S, real), **complementarity(S, real)}
    hub = {"gigs": slot_stats(cand.gigs), "providers": slot_stats(cand.provs)}
    return per_gig, hub


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def paired(a: np.ndarray, b: np.ndarray, boot: np.ndarray) -> list[float]:
    """Mean difference a - b and its 95% bootstrap CI (nan-aware); `boot` holds resamples of positions."""
    d = a - b
    return list(ds.boot_ci(d, boot))


def fmt_diff(t: list[float]) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]{'*' if t[1] > 0 or t[2] < 0 else ' '}"


def report_a(results: dict, baseline: str, split: str, n_boot: int) -> dict:
    n = len(next(iter(next(iter(results.values())).values()))["precision@5"])
    boot = ds.rng_for(SEED, f"enc/boot/A/{split}").integers(0, n, size=(n_boot, n))
    shown = ["precision@5", "precision@10", "precision@30", "recall@30", "role@1", "role@5", "role@10", "track@3"]
    print(f"\n=== A. gold tags, {split} classes ({n}); baseline = {baseline} (hc); * = paired 95% CI excludes 0 ===")
    print(f"{'candidate':22}" + "".join(f"{m:>14}" for m in shown))
    out = {}
    for name, by_label in results.items():
        for label, ms in by_label.items():
            key = f"{name}/{label}"
            out[key] = {"means": {m: float(ms[m].mean()) for m in A_METRICS}}
            if baseline in results and not (name == baseline and label == "hc"):
                out[key]["vs_baseline"] = {m: paired(ms[m], results[baseline]["hc"][m], boot) for m in A_METRICS}
            print(f"{key:22}" + "".join(f"{out[key]['means'][m]:14.3f}" for m in shown))
            if "vs_baseline" in out[key]:
                print(f"{'  minus ' + baseline + '/hc':22}" + "".join(f"{fmt_diff(out[key]['vs_baseline'][m]):>14}" for m in shown))
    return out


def report_real(per_gig: dict, hubs: dict, baseline: str, split: str, real: RealData, n_boot: int) -> dict:
    n = len(real.gig_ids)
    boot = ds.rng_for(SEED, f"enc/boot/B/{split}").integers(0, n, size=(n_boot, n))
    cols = ["AUC", "NDCG@5", "NDCG@10", "spearman_vs_dense", "novelty@50"]
    print(f"\n=== B + C. real gigs, {split} ({n} gigs), original grades; baseline = {baseline}; * = CI excludes 0 ===")
    print(f"{'candidate':22}" + "".join(f"{c:>20}" for c in cols) + f"{'slots top50 g/p':>18}{'never p':>9}{'max p':>7}")
    out = {}
    for name, ms in per_gig.items():
        out[name] = {"means": {c: float(np.nanmean(ms[c])) for c in cols}, "hubness": hubs[name]}
        if baseline in per_gig and name != baseline:
            out[name]["vs_baseline"] = {c: paired(ms[c], per_gig[baseline][c], boot) for c in cols}
        h = hubs[name]
        print(f"{name:22}" + "".join(f"{out[name]['means'][c]:20.3f}" for c in cols)
              + f"{h['gigs']['slot_share_top50']:9.3f}/{h['providers']['slot_share_top50']:.3f}"
              f"{h['providers']['tags_never_picked']:9d}{h['providers']['max_tag_freq']:7d}")
        if "vs_baseline" in out[name]:
            print(f"{'  minus ' + baseline:22}" + "".join(f"{fmt_diff(out[name]['vs_baseline'][c]):>20}" for c in cols))
    dense = pool_metrics(real.dense, real)
    out["reference/dense"] = {"means": {c: float(np.nanmean(dense[c])) for c in ("AUC", "NDCG@5", "NDCG@10")}}
    print("reference: dense cosine on the same pool  " + "  ".join(f"{c} {v:.3f}" for c, v in out["reference/dense"]["means"].items()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", default=["mxbai"],
                    help="encoder names from tag_encoders.SPECS; 'a+b' averages two or more. The first is the baseline "
                         "unless --baseline is given")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--benches", nargs="+", choices=["A", "real"], default=["A", "real"])
    ap.add_argument("--split", choices=["dev", "test"], default="dev",
                    help="role classes (A) and gigs (real) to evaluate; choose on dev, confirm a finalist on test")
    ap.add_argument("--data-dir", default="data_sat")
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-name", default=None, help="result file name under results_tag/ "
                    "(default encoder_bench_<split>.json)")
    args = ap.parse_args()
    baseline = args.baseline or args.encoders[0]
    names = list(dict.fromkeys([baseline, *args.encoders]))

    t0 = time.time()
    data_dir = BASE / args.data_dir
    corp = Corpora(args.export_dir, data_dir, SEED)
    embeds = {}
    cands = {n: build_candidate(n, corp, embeds) for n in names}
    out = {"split": args.split, "baseline": baseline, "encoders": names}
    if "A" in args.benches:
        out["A"] = report_a({n: bench_a(c, corp, args.split) for n, c in cands.items()}, baseline, args.split, args.n_boot)
    if "real" in args.benches:
        real = RealData(corp, data_dir, args.split, SEED)
        per_gig, hubs = {}, {}
        for n, c in cands.items():
            per_gig[n], hubs[n] = bench_real(c, real)
        out["real"] = report_real(per_gig, hubs, baseline, args.split, real, args.n_boot)
    out["seconds"] = round(time.time() - t0, 1)
    path = BASE / "results_tag" / (args.out_name or f"encoder_bench_{args.split}.json")
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {path} ({out['seconds']}s)")


if __name__ == "__main__":
    main()
