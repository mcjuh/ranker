"""
Evaluation of the tag-ID BM25 recall channel (retrieval_tagbm25.TagBM25).

    python pipeline/eval_tag_channel.py roles          # Part A: role-level, on the taxonomy alone
    python pipeline/eval_tag_channel.py tagger         # tagger error, measured on role descriptions (gold tags exist)
    python pipeline/eval_tag_channel.py sat            # Part B: provider-level, on tagged data_sat (needs tag_corpus.py)

Every random draw comes from a numpy Generator seeded from (--seed, a purpose string), so a rerun
reproduces the numbers exactly, and within a condition every method scores the same queries.

Part A. Documents are the 2,001 job roles; a query is a set of tag IDs sampled from one role's own
tags (optimistic by construction: the query always contains real evidence). The truth is that role
scored at tag-set-equivalence level, because 619 roles share an identical tag set with another role
and cannot be told apart from tags: rankings are collapsed to one entry per equivalence class, and
exact ties are broken in expectation (no method is helped by role-ID order). Tracks are keyed by
category ID. The track distribution is the score mass of the top-R roles per track.

Dev/test split: equivalence classes are shuffled with a fixed seed and halved, so twin roles never
straddle the split. Tuning (b, LSA rank) uses dev queries only; everything reported as a result is
on test. The one exception, marked, is the b sweep table, which is a sensitivity display on test and
not used to select anything. 95% intervals come from a bootstrap over classes (queries drawn from
one class are not independent).

Selection metric for tuning: mean of role@1, role@5, role@10 on dev, default condition.
"""
import argparse
import csv
import hashlib
import json
import time
import zlib
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from scipy.stats import rankdata

from greygigz import DEFAULT_EXPORT_DIR, load_taxonomy, track_distribution
from retrieval_tagbm25 import TagBM25

BASE = Path(__file__).parent
RESULTS_DIR = BASE / "results_tag"

SEED = 7
KS = (1, 5, 10)                      # role@k
TRACK_KS = (1, 3)                    # track@k
TOP_R = 10                           # roles aggregated into the track distribution
B_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
K1_GRID = (0.5, 1.2, 2.0, 3.0)
LSA_DIMS = (32, 64, 128, 256, 512, 1024, 2001)   # 2001 = no truncation (cosine on idf-weighted vectors)
RRF_K = 60
TIE_TOL = 1e-9
K1 = 1.2
SHOWN = ("role@1", "role@5", "role@10", "track@1")

# name, total query size, tag choice, wrong tags among them, draws per class, where wrong tags come from.
# A wrong tag is one the source role does not have: drawn uniformly from the whole vocabulary ("vocab"),
# or from other roles of the same track ("track": a plausible mistake, far harsher than a random tag).
# Each noisy condition has a clean twin with the same real evidence (wrong*_on_4 -> clean_4, *_on_5 ->
# default_5, *_on_3 -> few_3), so the effect of the wrong tags is not confounded with fewer real ones.
CONDITIONS = [
    ("default_5", 5, "random", 0, 5, "vocab"),
    ("few_3", 3, "random", 0, 5, "vocab"),
    ("clean_4", 4, "random", 0, 5, "vocab"),
    ("many_10", 10, "random", 0, 5, "vocab"),
    ("generic_5", 5, "generic", 0, 1, "vocab"),    # the role's 5 most widespread tags (deterministic)
    ("specific_5", 5, "specific", 0, 1, "vocab"),  # the role's 5 rarest tags (deterministic)
    ("wrong1_on_4", 5, "random", 1, 5, "vocab"),   # 4 real + 1 wrong
    ("wrong1_on_5", 6, "random", 1, 5, "vocab"),   # 5 real + 1 wrong: the default query with one wrong tag added
    ("wrong2_on_3", 5, "random", 2, 5, "vocab"),   # 3 real + 2 wrong
    ("near1_on_5", 6, "random", 1, 5, "track"),    # as wrong1_on_5, but the wrong tag belongs to a sibling role
    ("near2_on_3", 5, "random", 2, 5, "track"),    # as wrong2_on_3, with sibling-role wrong tags
]
PRIMARY = "default_5"


def rng_for(seed: int, purpose: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(purpose.encode())])


# ---------------------------------------------------------------------------
# The role world: index, equivalence classes, tracks, split
# ---------------------------------------------------------------------------

class RoleWorld:
    def __init__(self, tax, seed: int, draws_scale: float = 1.0):
        self.tax, self.seed, self.draws_scale = tax, seed, draws_scale
        self.role_ids = sorted(tax.role_tags)
        col = {r: i for i, r in enumerate(self.role_ids)}
        cls_of_role = tax.equivalence_classes()
        self.reps = sorted(set(cls_of_role.values()))              # smallest role id of each class
        cls_index = {rep: i for i, rep in enumerate(self.reps)}
        self.n_classes = len(self.reps)
        self.class_of_col = np.array([cls_index[cls_of_role[r]] for r in self.role_ids])
        self.col_order = np.argsort(self.class_of_col, kind="stable")   # columns grouped by class
        self.class_starts = np.searchsorted(self.class_of_col[self.col_order], np.arange(self.n_classes))
        self.role_track_map = tax.role_track()
        self.role_track = np.array([self.role_track_map[r] for r in self.role_ids])
        rep_cols = np.array([col[r] for r in self.reps])
        self.class_track = self.role_track[rep_cols]                    # strict truth: the source role's own track
        self.class_tracks = [set() for _ in range(self.n_classes)]
        for c, t in zip(self.class_of_col, self.role_track):
            self.class_tracks[c].add(int(t))
        self.track_ids = sorted(tax.tracks)
        self.track_name = {t: tax.tracks[t]["name"] for t in self.track_ids}
        self.track_tags = {t: set() for t in self.track_ids}            # every tag used by some role of the track
        for r, t in zip(self.role_ids, self.role_track):
            self.track_tags[int(t)] |= set(tax.role_tags[r])

        perm = rng_for(seed, "split").permutation(self.n_classes)
        half = self.n_classes // 2
        self.split_of_class = {"dev": np.zeros(self.n_classes, bool), "test": np.zeros(self.n_classes, bool)}
        self.split_of_class["dev"][perm[:half]] = True
        self.split_of_class["test"][perm[half:]] = True

    def index(self, k1=K1, b=0.75) -> TagBM25:
        return TagBM25(self.role_ids, [self.tax.role_tags[r] for r in self.role_ids], k1=k1, b=b,
                       levels=self.tax.levels)

    def class_scores(self, S: np.ndarray) -> np.ndarray:
        """Per-class score = max over the class's members (identical tag sets score identically)."""
        return np.maximum.reduceat(S[:, self.col_order], self.class_starts, axis=1)

    def make_queries(self, cond, df_by_tag: dict, vocab: np.ndarray):
        name, size, mode, n_wrong, draws, wrong_from = cond
        draws = max(1, int(round(draws * self.draws_scale))) if mode == "random" else 1
        rng = rng_for(self.seed, f"queries/{name}")
        queries, qcls = [], []
        for c, rep in enumerate(self.reps):
            tags = np.array(sorted(self.tax.role_tags[rep]))
            outside = None
            if n_wrong:
                outside = np.setdiff1d(vocab, tags, assume_unique=True)
                if wrong_from == "track":
                    siblings = np.setdiff1d(np.array(sorted(self.track_tags[int(self.class_track[c])])), tags,
                                            assume_unique=True)
                    outside = siblings if len(siblings) >= n_wrong else outside
            for _ in range(draws):
                n_right = min(size - n_wrong, len(tags))
                if mode == "random":
                    picked = rng.choice(tags, size=n_right, replace=False)
                else:
                    order = sorted(tags, key=lambda t: (df_by_tag[t], t))   # rarest first
                    picked = order[:n_right] if mode == "specific" else order[::-1][:n_right]
                wrong = rng.choice(outside, size=n_wrong, replace=False) if n_wrong else []
                queries.append([int(t) for t in picked] + [int(t) for t in wrong])
                qcls.append(c)
        return queries, np.array(qcls)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hit_metrics(world: RoleWorld, S: np.ndarray, qcls: np.ndarray, drop_nonpositive: bool = True,
                class_scores: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-query metrics from an (n_queries x n_roles) score matrix.

    role@k: the truth class is in the top k *classes* (identical roles collapsed). Exact ties with the
    truth are broken uniformly at random in expectation, so twins and tied supersets cost the method
    what they would cost on average. A truth scoring <= 0 is not retrieved (the channel drops it).
    track@k: the truth track is among the k best tracks of the top-R roles' score mass; strict uses the
    source role's own track, lenient accepts any track of its equivalence class."""
    n = len(qcls)
    Sc = world.class_scores(S) if class_scores is None else class_scores
    s_true = Sc[np.arange(n), qcls]
    greater = (Sc > (s_true + TIE_TOL)[:, None]).sum(1)
    ties = (np.abs(Sc - s_true[:, None]) <= TIE_TOL).sum(1)
    valid = s_true > TIE_TOL if drop_nonpositive else np.ones(n, bool)
    out = {f"role@{k}": np.where(valid, np.clip((k - greater) / ties, 0.0, 1.0), 0.0) for k in KS}

    top = np.argsort(-S, axis=1, kind="stable")[:, :TOP_R]
    strict = {k: np.zeros(n) for k in TRACK_KS}
    lenient1, samename1 = np.zeros(n), np.zeros(n)
    for i in range(n):
        ranked = [(world.role_ids[j], S[i, j]) for j in top[i] if S[i, j] > 0]
        dist = [c for c, _ in track_distribution(ranked, world.role_track_map, top_r=TOP_R)]
        truth = int(world.class_track[qcls[i]])
        for k in TRACK_KS:
            strict[k][i] = float(truth in dist[:k])
        lenient1[i] = float(bool(dist) and dist[0] in world.class_tracks[qcls[i]])
        samename1[i] = float(bool(dist) and world.track_name[dist[0]] == world.track_name[truth])
    for k in TRACK_KS:
        out[f"track@{k}"] = strict[k]
    out["track@1_lenient"] = lenient1        # any track of the truth's equivalence class
    out["track@1_samename"] = samename1      # or any track sharing the truth track's name
    return out


def class_means(metrics: dict, qcls: np.ndarray):
    """Average each metric per class. Returns (class ids, {metric: per-class means})."""
    ids, inv = np.unique(qcls, return_inverse=True)
    counts = np.bincount(inv)
    return ids, {m: np.bincount(inv, weights=v) / counts for m, v in metrics.items()}


def ci(cm: np.ndarray, boot_idx: np.ndarray):
    means = cm[boot_idx].mean(axis=1)
    return float(cm.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(t) -> str:
    return f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"


# ---------------------------------------------------------------------------
# Scorers beyond TagBM25's own baselines
# ---------------------------------------------------------------------------

def _unit(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(norms == 0, 1.0, norms)


def lsa_scores(idx: TagBM25, queries, V: np.ndarray) -> np.ndarray:
    """Cosine in a latent space: docs = W V, queries = (idf * q) V, with W the idf-weighted binary matrix."""
    D = (idx.B @ sp.diags(idx.idf)) @ V
    Q = (idx.query_matrix(queries) @ sp.diags(idx.idf)) @ V
    return _unit(Q) @ _unit(D).T


def rrf_matrix(mats, drop_nonpositive, k: int = RRF_K) -> np.ndarray:
    """Vectorised RRF over score matrices. Ties share the best rank ('min'), so identical roles get
    identical fused scores; lists that drop non-positive scores contribute nothing for them."""
    total = 0.0
    for S, drop in zip(mats, drop_nonpositive):
        contrib = 1.0 / (k + rankdata(-S, method="min", axis=1))
        total = total + (np.where(S > 0, contrib, 0.0) if drop else contrib)
    return total


def lsa_basis(idx: TagBM25, k_max: int) -> np.ndarray:
    """Right singular vectors of the idf-weighted matrix (T x k), columns by descending singular value.
    An exact dense SVD (the matrix is ~2,000 x 2,000), so it is deterministic and every rank up to
    k_max is available without an iterative solver."""
    W = (idx.B @ sp.diags(idx.idf)).toarray()
    _, _, Vt = svd(W, full_matrices=False)
    return Vt[:k_max].T


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------

def run_roles(args):
    t_start = time.time()
    tax = load_taxonomy(args.export_dir)
    world = RoleWorld(tax, args.seed, draws_scale=args.draws_scale)
    idx0 = world.index()                                        # default b: used by parameter-free scorers
    df_by_tag = dict(zip(idx0.tag_ids, idx0.df))
    vocab = np.array(idx0.tag_ids)
    span = sum(len(t) > 1 for t in world.class_tracks)
    print(f"{len(world.role_ids)} roles, {world.n_classes} equivalence classes "
          f"({span} span more than one track), {len(vocab)} tags; "
          f"dev {world.split_of_class['dev'].sum()} / test {world.split_of_class['test'].sum()} classes", flush=True)

    data, views, boots = {}, {}, {}
    for cond in CONDITIONS:                                      # generated once, shared by every method
        data[cond[0]] = world.make_queries(cond, df_by_tag, vocab)
        qs = data[cond[0]][0]
        print(f"  {cond[0]:>11}: {len(qs)} queries, mean {np.mean([len(q) for q in qs]):.1f} tags", flush=True)

    def view(cond, split):
        if (cond, split) not in views:
            queries, qcls = data[cond]
            keep = np.flatnonzero(world.split_of_class[split][qcls])
            views[(cond, split)] = ([queries[i] for i in keep], qcls[keep])
        return views[(cond, split)]

    def evaluate(cond, split, S, drop=True, class_scores=None):
        _, qcls = view(cond, split)
        ids, cm = class_means(hit_metrics(world, S, qcls, drop, class_scores), qcls)
        if split not in boots:   # one resample per split, so methods AND conditions can be compared pairwise
            boots[split] = rng_for(args.seed, f"boot/{split}").integers(0, len(ids), size=(args.n_boot, len(ids)))
        return cm, boots[split]

    def score(idx, cond, split, method="bm25"):
        return idx.score_matrix(view(cond, split)[0], method)

    def selection_value(cm):  # the tuning target: mean of role@1/5/10
        return float(np.mean([cm[f"role@{k}"].mean() for k in KS]))

    result = {"seed": args.seed, "n_roles": len(world.role_ids), "n_classes": world.n_classes,
              "classes_spanning_tracks": span, "n_boot": args.n_boot, "draws_scale": args.draws_scale,
              "conditions": {c[0]: list(c[1:]) for c in CONDITIONS}}

    # ---- tune b on dev (default condition, k1 fixed) -----------------------
    dev_sweep = {b: selection_value(evaluate(PRIMARY, "dev", score(world.index(b=b), PRIMARY, "dev"))[0])
                 for b in B_GRID}
    best_b = max(B_GRID, key=lambda b: (round(dev_sweep[b], 12), -abs(b - 0.75)))   # ties -> nearest the 0.75 default
    print("\ndev selection metric (mean of role@1/5/10) by b:", {b: round(v, 4) for b, v in dev_sweep.items()})
    print(f"selected b = {best_b} (k1 = {K1})")
    result["dev_b_sweep"] = {str(b): v for b, v in dev_sweep.items()}
    result["selected_b"] = best_b
    idx = world.index(b=best_b)

    # ---- LSA rank tuned on dev -------------------------------------------------
    V = lsa_basis(idx0, max(LSA_DIMS))
    dev_lsa = {d: selection_value(evaluate(
        PRIMARY, "dev", lsa_scores(idx0, view(PRIMARY, "dev")[0], V[:, :d]), drop=False)[0]) for d in LSA_DIMS}
    best_d = max(LSA_DIMS, key=lambda d: dev_lsa[d])
    print("dev selection metric for LSA by rank:", {d: round(v, 4) for d, v in dev_lsa.items()}, f"-> rank {best_d}")
    result["dev_lsa_sweep"] = {str(d): v for d, v in dev_lsa.items()}
    result["selected_lsa_rank"] = best_d

    # ---- main table on test -----------------------------------------------------
    def lsa(cond, split):
        return lsa_scores(idx0, view(cond, split)[0], V[:, :best_d])

    bm25_name = f"bm25 (k1={K1}, b={best_b})"
    methods = {
        bm25_name: (lambda c, s: score(idx, c, s), True),
        "idf_overlap": (lambda c, s: score(idx0, c, s, "idf_overlap"), True),
        "coverage": (lambda c, s: score(idx0, c, s, "coverage"), True),
        "sum_levels": (lambda c, s: score(idx0, c, s, "sum_levels"), True),
        f"lsa (rank {best_d})": (lambda c, s: lsa(c, s), False),
        f"bm25+lsa RRF (rank {best_d})": (
            lambda c, s: rrf_matrix([score(idx, c, s), lsa(c, s)], [True, False]), True),
    }
    metric_names = [f"role@{k}" for k in KS] + [f"track@{k}" for k in TRACK_KS] + ["track@1_lenient", "track@1_samename"]
    main, main_cm = {}, {}
    print(f"\n=== TEST, condition {PRIMARY} (95% CI over classes) ===")
    print(f"{'method':32}" + "".join(f"{m:>22}" for m in metric_names[:5]))
    for name, (fn, drop) in methods.items():
        cm, bidx = evaluate(PRIMARY, "test", fn(PRIMARY, "test"), drop=drop)
        main_cm[name] = cm
        main[name] = {m: ci(cm[m], bidx) for m in metric_names}
        print(f"{name:32}" + "".join(f"{fmt(main[name][m]):>22}" for m in metric_names[:5]))
    print("track@1 credited to a same-named or equivalence-class track (diagnostic of the track error): "
          + "; ".join(f"{n.split(' (')[0]} {main[n]['track@1_lenient'][0]:.3f} / {main[n]['track@1_samename'][0]:.3f}"
                      for n in main))
    result["test_main"] = main

    paired = {}
    for other in methods:
        if other != bm25_name:
            paired[other] = {m: ci(main_cm[bm25_name][m] - main_cm[other][m], bidx) for m in metric_names}
    result["test_bm25_minus_other"] = paired
    print("\npaired difference, bm25 minus other (role@1 / role@5 / role@10 / track@1):")
    for other, d in paired.items():
        print(f"  vs {other:30} " + "  ".join(fmt(d[m]) for m in SHOWN))

    # ---- b sweep, k1 sensitivity (test; sensitivity display, nothing is selected from it) -------------
    print(f"\n=== b sweep on TEST ({PRIMARY}, k1={K1}); sensitivity only, selection was made on dev ===")
    result["test_b_sweep"], b_cm = {}, {}
    for b in B_GRID:
        cm, bidx = evaluate(PRIMARY, "test", score(world.index(b=b), PRIMARY, "test"))
        b_cm[b] = cm
        result["test_b_sweep"][str(b)] = {m: ci(cm[m], bidx) for m in SHOWN}
        print(f"  b={b:<5}" + "  ".join(f"{m}={fmt(result['test_b_sweep'][str(b)][m])}" for m in SHOWN))
    print("  paired difference vs the b=0.75 default (role@1 / role@5 / role@10):")
    result["test_b_vs_default"] = {}
    for b in B_GRID:
        if b != 0.75:
            d = {m: ci(b_cm[b][m] - b_cm[0.75][m], bidx) for m in ("role@1", "role@5", "role@10")}
            result["test_b_vs_default"][str(b)] = d
            print(f"    b={b:<5}" + "  ".join(fmt(d[m]) for m in d))
    print(f"\n=== k1 sensitivity on TEST ({PRIMARY}, b={best_b}) ===")
    result["test_k1_sensitivity"] = {}
    for k1 in K1_GRID:
        cm, bidx = evaluate(PRIMARY, "test", score(world.index(k1=k1, b=best_b), PRIMARY, "test"))
        result["test_k1_sensitivity"][str(k1)] = {m: ci(cm[m], bidx) for m in SHOWN}
        print(f"  k1={k1:<4}" + "  ".join(f"{m}={fmt(result['test_k1_sensitivity'][str(k1)][m])}" for m in SHOWN))

    # ---- query conditions on test -------------------------------------------------
    show = {"bm25": lambda c: score(idx, c, "test"), "idf_overlap": lambda c: score(idx0, c, "test", "idf_overlap"),
            "coverage": lambda c: score(idx0, c, "test", "coverage")}
    print("\n=== query conditions on TEST (role@1 / role@5 / track@1) ===")
    result["test_conditions"], cond_cm = {}, {}
    for cond in CONDITIONS:
        c = cond[0]
        result["test_conditions"][c] = {}
        for name, fn in show.items():
            cm, bidx = evaluate(c, "test", fn(c))
            cond_cm[(c, name)] = cm
            result["test_conditions"][c][name] = {m: ci(cm[m], bidx) for m in SHOWN}
        print(f"  {c:>11} " + "   ".join(
            f"{n}: " + "/".join(f"{result['test_conditions'][c][n][m][0]:.3f}" for m in ("role@1", "role@5", "track@1"))
            for n in show))

    # what a wrong tag costs: each noisy condition minus its clean twin (same real evidence), paired over classes
    print("\nbm25, effect of wrong tags (noisy minus clean twin), paired 95% CI: role@1 / role@5 / track@1")
    result["test_wrong_tag_effect"] = {}
    for noisy, clean in (("wrong1_on_5", "default_5"), ("wrong1_on_4", "clean_4"), ("wrong2_on_3", "few_3"),
                         ("near1_on_5", "default_5"), ("near2_on_3", "few_3")):
        d = {m: ci(cond_cm[(noisy, "bm25")][m] - cond_cm[(clean, "bm25")][m], bidx) for m in ("role@1", "role@5", "track@1")}
        result["test_wrong_tag_effect"][f"{noisy} - {clean}"] = d
        print(f"  {noisy} - {clean}: " + "  ".join(fmt(d[m]) for m in d))

    result["funnel"] = run_funnel(world, args, view, evaluate, score, idx, best_b)

    result["seconds"] = round(time.time() - t_start, 1)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "roles.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {out} ({result['seconds']}s)")


def run_funnel(world: RoleWorld, args, view, evaluate, score, idx: TagBM25, best_b: float):
    """Track-first funnel: predict one track with BM25 over tracks (a track's document = the union of its
    roles' tags), then rank only that track's roles. Compared with flat retrieval on the same test queries,
    and with an oracle funnel that is told the true track."""
    track_col = {t: i for i, t in enumerate(world.track_ids)}
    role_track_col = np.array([track_col[int(t)] for t in world.role_track])
    union = {t: set() for t in world.track_ids}
    for r, t in zip(world.role_ids, world.role_track):
        union[int(t)] |= set(world.tax.role_tags[r])
    track_index = TagBM25(world.track_ids, [union[t] for t in world.track_ids], k1=K1, b=best_b)

    queries, qcls = view(PRIMARY, "test")
    st = track_index.score_matrix(queries)
    rng = rng_for(args.seed, "funnel/ties")
    pred = np.argmax(st + 1e-12 * rng.random(st.shape), axis=1)          # random tie-break
    truth = np.array([track_col[int(world.class_track[c])] for c in qcls])
    track_acc = float((pred == truth).mean())

    S = score(idx, PRIMARY, "test")
    rows = {}
    for label, tq in (("funnel (predicted track)", pred), ("oracle funnel (true track)", truth)):
        masked = world.class_scores(S * (role_track_col[None, :] == tq[:, None]))
        cm, bidx = evaluate(PRIMARY, "test", S, class_scores=masked)
        rows[label] = {m: ci(cm[m], bidx) for m in ("role@1", "role@5", "role@10")}
    cm, bidx = evaluate(PRIMARY, "test", S)
    rows["flat"] = {m: ci(cm[m], bidx) for m in SHOWN}
    # A funnel that routes to the wrong track finds (almost) nothing, so funnel ~ accuracy x oracle. It beats
    # flat retrieval only when accuracy exceeds flat / oracle: the break-even track accuracy at each k.
    breakeven = {m: rows["flat"][m][0] / rows["oracle funnel (true track)"][m][0] for m in ("role@1", "role@5", "role@10")}
    print(f"\n=== track-first funnel vs flat, TEST {PRIMARY} ===")
    print(f"track accuracy of the track-level scorer (top-1): {track_acc:.3f}; "
          f"flat retrieval's own track@1: {fmt(rows['flat']['track@1'])}")
    for label, r in rows.items():
        print(f"  {label:28}" + "".join(f"{m}={fmt(r[m])}  " for m in ("role@1", "role@5", "role@10")))
    print("  break-even track accuracy (flat / oracle): " + ", ".join(f"{m} {v:.3f}" for m, v in breakeven.items()))
    return {"track_scorer_accuracy_top1": track_acc, "rows": rows, "breakeven_track_accuracy": breakeven}


# ---------------------------------------------------------------------------
# Part B: provider-level, on tagged data_sat
# ---------------------------------------------------------------------------

GIG_METRICS = ("P@5", "P@10", "P@20", "R@10", "R@20", "R@50", "R@100", "NDCG@5", "NDCG@10", "NDCG@20", "MRR")
SAT_M_GRID = (5, 10, 20, 30)         # tags kept per text (the tagger stores 30)
SAT_MARGINAL_KS = (10, 20, 50)


class Ranked:
    """A ranking as two arrays (best first): provider ids and scores. Compact enough to hold every gig's
    full ranking of all 2,165 providers for several channels."""
    def __init__(self, ids, scores):
        self.ids, self.scores = np.asarray(ids), np.asarray(scores)

    def pairs(self):
        return list(zip(self.ids.tolist(), self.scores.tolist()))

    def top(self, k: int) -> set:
        return set(self.ids[:k].tolist())


def cached_embeddings(name: str, texts: list[str], cache_dir: Path | None = None) -> np.ndarray:
    """Vectors from the embed_cached files under pipeline/cache/ (keyed by sha1 of the text). Never
    encodes: a missing text means tag_corpus.py has not finished, and that is an error, not a reason
    to spend hours re-embedding."""
    cache = cache_dir or BASE / "cache"
    vecs = np.load(cache / f"{name}.npy")
    by_key = dict(zip(json.loads((cache / f"{name}.keys.json").read_text()), vecs))
    hashes = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    missing = sum(h not in by_key for h in hashes)
    if missing:
        raise SystemExit(f"{missing} of {len(texts)} texts are not in the '{name}' cache; run pipeline/tag_corpus.py first")
    return np.stack([by_key[h] for h in hashes])


def existing_channels(hirers, providers, doc_raw, q_raw) -> dict[str, dict[str, Ranked]]:
    """The two existing channels and their RRF, recomputed exactly as features.py / build_judging_pools_sat.py do."""
    from corpus import hirer_text
    from retrieval_bm25 import BM25Retriever
    from retrieval_dense import rank_from_raw
    from retrieval_rrf import rrf_fuse

    pids = [p["provider_id"] for p in providers]
    bm25 = BM25Retriever(providers, refined=True)
    out = {"bm25": {}, "dense": {}, "rrf2": {}}
    for i, h in enumerate(hirers):
        hid = str(h["hire_id"])
        sparse = bm25.rank(hirer_text(h), query_title=h["hire_title"])
        dense = rank_from_raw(doc_raw, q_raw[i], pids)
        for name, scored in (("bm25", sparse), ("dense", dense), ("rrf2", rrf_fuse(sparse, dense, k=60))):
            out[name][hid] = Ranked([p for p, _ in scored], [s for _, s in scored])
    return out


def gig_metrics(rankings: dict[str, Ranked], gigs, gt, judged=None) -> dict[str, np.ndarray]:
    """Per-gig P@5/10/20, R@10/20/50/100, NDCG@5/10/20, MRR via evaluate.py (relevant = grade >= 2). With `judged`,
    unjudged providers are dropped from each list first ("condensed list"), so a channel that surfaces
    providers nobody graded is not penalised for them."""
    from evaluate import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
    out = {m: [] for m in GIG_METRICS}
    for hid in gigs:
        ids = rankings[hid].ids.tolist()
        if judged is not None:
            ids = [p for p in ids if str(p) in judged[hid]]
        row = gt.get(hid, {})
        for k in (10, 20, 50, 100):
            out[f"R@{k}"].append(recall_at_k(ids, row, k))
        for k in (5, 10, 20):
            out[f"NDCG@{k}"].append(ndcg_at_k(ids, row, k))
            out[f"P@{k}"].append(precision_at_k(ids, row, k))
        out["MRR"].append(reciprocal_rank(ids, row))
    return {m: np.array(v, dtype=float) for m, v in out.items()}


def judged_share(rankings: dict[str, Ranked], gigs, judged, k: int) -> float:
    """Mean over gigs of the fraction of a channel's top-k providers that have a grade. Below 1.0 the channel's
    unjudged providers count as irrelevant in every metric, so its numbers at k are lower bounds."""
    return float(np.mean([np.mean([str(p) in judged[h] for p in rankings[h].ids[:k].tolist()] or [1.0])
                          for h in gigs]))


def unique_positives(target: Ranked, others: list[Ranked], positives: set, k: int) -> set:
    """Relevant providers that `target` has in its top k and none of `others` has in theirs."""
    seen = set().union(*(o.top(k) for o in others)) if others else set()
    return (target.top(k) - seen) & positives


def run_sat(args):
    from corpus import hirer_text, provider_text
    from evaluate import RELEVANCE_THRESHOLD
    from retrieval_rrf import rrf_fuse_n

    t_start = time.time()
    data_dir = BASE / args.data_dir
    load = lambda name: json.loads((data_dir / name).read_text(encoding="utf-8"))
    hirers, providers = load("hirers.json"), load("providers.json")
    judged, gt = load("llm_judgments_merged.json"), load("ground_truth_llm.json")
    raw_p, raw_h = load("tags_providers.json")["tags"], load("tags_hirers.json")["tags"]
    pids = [p["provider_id"] for p in providers]
    hids = [str(h["hire_id"]) for h in hirers]
    positives = {h: {int(p) for p, s in gt.get(h, {}).items() if s >= RELEVANCE_THRESHOLD} for h in hids}
    gigs = [h for h in hids if positives[h]]
    perm = rng_for(args.seed, "sat/split").permutation(len(gigs))
    dev, test = [gigs[i] for i in sorted(perm[: len(gigs) // 2])], [gigs[i] for i in sorted(perm[len(gigs) // 2:])]
    print(f"{len(hids)} gigs, {len(pids)} providers; {len(gigs)} gigs have a grade>=2 provider "
          f"({sum(len(positives[h]) for h in gigs)} positives); dev {len(dev)} / test {len(test)} gigs", flush=True)
    result = {"seed": args.seed, "n_gigs": len(hids), "n_gigs_with_positive": len(gigs),
              "n_positives": sum(len(positives[h]) for h in gigs), "n_dev": len(dev), "n_test": len(test)}

    # ---- existing channels, recomputed, and checked against the committed pool ------------------------
    ex = existing_channels(hirers, providers,
                           cached_embeddings("sat_docs_mxbai_by_text", [provider_text(p) for p in providers]),
                           cached_embeddings("sat_queries_mxbai_by_text", [hirer_text(h) for h in hirers]))
    committed: dict[str, set] = {}
    with (BASE / "features_data_sat" / "candidates_top50.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            committed.setdefault(row["hire_id"], set()).add(int(row["provider_id"]))
    overlap = np.array([len(ex["rrf2"][h].top(50) & committed[h]) / 50 for h in hids])
    print(f"recomputed RRF top-50 vs committed candidates_top50.csv: mean overlap {overlap.mean():.4f}, "
          f"identical for {(overlap == 1).mean():.1%} of gigs, worst {overlap.min():.2f}", flush=True)
    result["reproduction"] = {"mean_overlap": float(overlap.mean()), "identical_gigs": float((overlap == 1).mean()),
                              "worst": float(overlap.min())}

    # ---- tag channel: tuned on dev gigs -----------------------------------------------------------------
    def tag_lists(raw, m):
        return {int(i): [t for t, _ in tags[:m]] for i, tags in raw.items()}

    pid_arr = np.array(pids)

    def tag_rankings(m_p, m_h, b, which, hirer_tags=None):
        prov, hq = tag_lists(raw_p, m_p), hirer_tags or tag_lists(raw_h, m_h)
        idx = TagBM25(pids, [prov[p] for p in pids], k1=K1, b=b)
        S = idx.score_matrix([hq[int(h)] for h in which])
        out = {}
        for h, row in zip(which, S):
            order = np.argsort(-row, kind="stable")
            order = order[row[order] > 0]
            out[h] = Ranked(pid_arr[order], row[order])
        return out

    def r_at(rankings, gigs_, k):
        from evaluate import recall_at_k
        return float(np.mean([recall_at_k(rankings[h].ids[:k].tolist(), gt[h], k) for h in gigs_]))

    grid = {}
    for m_p in SAT_M_GRID:
        for m_h in SAT_M_GRID:
            for b in B_GRID:
                grid[(m_p, m_h, b)] = r_at(tag_rankings(m_p, m_h, b, dev), dev, 50)
    best = max(grid, key=lambda c: (round(grid[c], 9), -c[0], -c[1], -abs(c[2] - 0.75)))
    m_p, m_h, b = best
    print(f"\ndev tuning of the tag channel (objective: channel-alone R@50 on dev): best m_provider={m_p}, "
          f"m_gig={m_h}, b={b} -> {grid[best]:.4f}")
    print("  top 5 configs: " + "; ".join(f"{c}: {v:.4f}" for c, v in sorted(grid.items(), key=lambda kv: -kv[1])[:5]))
    result["dev_grid_R@50"] = {f"m_p={c[0]},m_h={c[1]},b={c[2]}": v for c, v in grid.items()}
    result["selected"] = {"m_provider": m_p, "m_gig": m_h, "b": b, "k1": K1, "dev_R@50": grid[best]}

    tag = tag_rankings(m_p, m_h, b, hids)
    ex["tag"] = tag

    def report(res, judged, gt, positives, title):
        """Everything from 'channel alone' to 'fused', against one label set. Fills `res`."""
        print(f"\n########## labels: {title} ##########")
        # ---- channel alone, test gigs ---------------------------------------------------------------------
        boot = rng_for(args.seed, "sat/boot").integers(0, len(test), size=(args.n_boot, len(test)))
        metric_names = ("R@10", "R@50", "R@100", "NDCG@10", "MRR")      # fused section
        alone = {}
        print(f"\n=== TEST ({len(test)} gigs with a grade>=2 provider), channels alone; 95% CI over gigs ===")
        print(f"{'channel':10}" + "".join(f"{m:>21}" for m in GIG_METRICS))   # wide: one CI per cell
        per_gig = {}
        for name in ("bm25", "dense", "rrf2", "tag"):
            per_gig[name] = gig_metrics(ex[name], test, gt)
            alone[name] = {m: ci(per_gig[name][m], boot) for m in GIG_METRICS}
            print(f"{name:10}" + "".join(f"{fmt(alone[name][m]):>21}" for m in GIG_METRICS))
        res["test_channels_alone"] = alone
        shares = {name: {str(k): judged_share(ex[name], test, judged, k) for k in (10, 20, 50)}
                  for name in ("bm25", "dense", "rrf2", "tag")}
        res["test_judged_share_of_top_k"] = shares
        print("share of each channel's top-K that has a grade (1.00 = fully judged, so its metrics at K are exact; "
              "less = lower bound):")
        for name, row in shares.items():
            print(f"  {name:6}" + "".join(f"  top-{k}: {v:.3f}" for k, v in row.items()))

        # control: the same channel with every gig's tags swapped for another gig's
        shuffled = rng_for(args.seed, "sat/permute").permutation(len(hids))
        tags_h = tag_lists(raw_h, m_h)
        swapped = {int(hids[i]): tags_h[int(hids[j])] for i, j in enumerate(shuffled)}
        control = tag_rankings(m_p, m_h, b, test, hirer_tags=swapped)
        ctl = gig_metrics(control, test, gt)
        res["control_permuted_gig_tags"] = {m: ci(ctl[m], boot) for m in GIG_METRICS}
        print(f"control, gig tags shuffled across gigs: R@50 {fmt(res['control_permuted_gig_tags']['R@50'])} "
              f"(a random ranking gives about {50 / len(pids):.3f})")

        # ---- what the tag channel adds ---------------------------------------------------------------------
        print("\n=== marginal recall on TEST: relevant providers in a channel's top-K that neither of the other two "
              "channels has in theirs ===")
        print("(labels were pooled from the two existing channels, so this is biased against the tag channel; "
              "see the unjudged counts below)")
        marginal = {}
        for k in SAT_MARGINAL_KS:
            marginal[k] = {}
            for target, others in (("tag", ("bm25", "dense")), ("bm25", ("dense", "tag")), ("dense", ("bm25", "tag"))):
                found = [unique_positives(ex[target][h], [ex[o][h] for o in others], positives[h], k) for h in test]
                marginal[k][target] = {
                    "gigs_with_unique_hit": float(np.mean([len(f) > 0 for f in found])),
                    "share_of_positives": float(sum(len(f) for f in found) / sum(len(positives[h]) for h in test)),
                    "n_unique_hits": int(sum(len(f) for f in found))}
            print(f"  K={k:<3}" + "   ".join(
                f"{t}: {v['gigs_with_unique_hit']:.3f} of gigs ({v['n_unique_hits']} hits, {v['share_of_positives']:.3f} of positives)"
                for t, v in marginal[k].items()))
        res["test_marginal_recall"] = {str(k): v for k, v in marginal.items()}

        outside = [(h, p) for h in gigs for p in positives[h] if p not in ex["rrf2"][h].top(50)]
        got = [(h, p) for h, p in outside if p in ex["tag"][h].top(50)]
        print(f"positives outside the existing RRF top-50 pool: {len(outside)} (on {len({h for h, _ in outside})} gigs); "
              f"the tag channel has {len(got)} of them in its top-50")
        res["outside_pool"] = {"n": len(outside), "tag_top50": len(got)}

        print("\nunjudged tag-only candidates (nobody graded them, so they count as irrelevant above), all gigs:")
        res["ungraded"] = {}
        for k in SAT_MARGINAL_KS:
            only = {h: ex["tag"][h].top(k) - ex["bm25"][h].top(k) - ex["dense"][h].top(k) for h in hids}
            n_only = sum(len(v) for v in only.values())
            n_unjudged = sum(len(v - {int(p) for p in judged[h]}) for h, v in only.items())
            n_top_unjudged = sum(len(ex["tag"][h].top(k) - {int(p) for p in judged[h]}) for h in hids)
            res["ungraded"][str(k)] = {"tag_only_pairs": n_only, "tag_only_unjudged": n_unjudged,
                                          "tag_top_k_unjudged": n_top_unjudged}
            print(f"  K={k:<3} tag-only pairs {n_only}, of which unjudged {n_unjudged} ({n_unjudged / max(n_only, 1):.1%}); "
                  f"pairs to grade to cover the whole tag top-{k}: {n_top_unjudged}")

        # ---- fused: refined BM25 + dense + tag, against the existing two-channel RRF -----------------------------
        print(f"\n=== TEST, fused ranking: RRF(bm25, dense) vs RRF(bm25, dense, tag), paired 95% CI over gigs ===")
        fused_rows = {}
        for w in (1.0, 0.5):
            fused = {h: (lambda ids: Ranked([p for p, _ in ids], [s for _, s in ids]))(
                rrf_fuse_n([ex["bm25"][h].pairs(), ex["dense"][h].pairs(), ex["tag"][h].pairs()], weights=[1.0, 1.0, w]))
                for h in test}
            for label, jd in (("standard", None), ("condensed (unjudged dropped)", judged)):
                base = gig_metrics(ex["rrf2"], test, gt, jd)
                new = gig_metrics(fused, test, gt, jd)
                fused_rows[f"w={w} {label}"] = {
                    m: {"rrf2": float(base[m].mean()), "rrf3": float(new[m].mean()), "diff": ci(new[m] - base[m], boot)}
                    for m in metric_names}
                print(f"  tag weight {w}, {label}:")
                for m in metric_names:
                    r = fused_rows[f"w={w} {label}"][m]
                    print(f"    {m:8} rrf2 {r['rrf2']:.4f}  rrf3 {r['rrf3']:.4f}  diff {fmt(r['diff'])}")
        res["test_fused"] = fused_rows

    report(result, judged, gt, positives, "original (pooled from bm25 + dense + random)")
    if args.extra_grades:
        judged_x, gt_x = load("llm_judgments_merged_tag.json"), load("ground_truth_llm_tag.json")
        positives_x = {h: {int(p) for p, sc in gt_x.get(h, {}).items() if sc >= RELEVANCE_THRESHOLD} for h in hids}
        added = sum(len(judged_x[h]) - len(judged.get(h, {})) for h in judged_x)
        new_pos = sum(len(positives_x[h]) - len(positives[h]) for h in hids)
        print(f"\nextended labels: +{added} graded pairs, +{new_pos} positives (dev/test split unchanged)")
        result["extended_labels"] = {"extra_graded_pairs": added, "extra_positives": new_pos}
        report(result["extended_labels"], judged_x, gt_x, positives_x,
               "extended (+ tag-only pairs graded by labeller.py)")

    result["seconds"] = round(time.time() - t_start, 1)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "sat.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {out} ({result['seconds']}s)")


# ---------------------------------------------------------------------------
# Tagger check: role descriptions have gold tags, so the tagger's own error can be measured
# ---------------------------------------------------------------------------

TAGGER_M_GRID = (5, 10, 20, 30)


def run_tagger(args):
    """Tag role descriptions (name + description, embedded as passages, tags as queries: the direction used
    for providers) and measure (a) how many of a role's gold tags the tagger recovers and (b) how well the
    role is then retrieved from the *predicted* tags, the gap to Part A's gold-tag queries being the price of
    tagging. A seeded sample of equivalence classes per split; (m, b) is tuned on the dev sample only."""
    from retrieval_dense import encode_docs, encode_queries
    from tag_corpus import embed_in_chunks, top_tags

    t_start = time.time()
    tax = load_taxonomy(args.export_dir)
    world = RoleWorld(tax, args.seed)
    tag_ids = sorted(tax.tags)
    tag_vecs = embed_in_chunks("taxonomy_tags_mxbai_queries_by_text", [tax.tags[t] for t in tag_ids], encode_queries)

    rng = rng_for(args.seed, "tagger/sample")
    sample = {s: np.sort(rng.choice(np.flatnonzero(world.split_of_class[s]), size=args.n_roles // 2, replace=False))
              for s in ("dev", "test")}
    order = np.concatenate([sample["dev"], sample["test"]])
    reps = [world.reps[c] for c in order]
    texts = [f"{tax.roles[r]['name']}. {tax.roles[r]['description']}" for r in reps]
    print(f"tagging {len(texts)} role descriptions ({args.n_roles // 2} dev / {args.n_roles // 2} test classes)", flush=True)
    vecs = embed_in_chunks("taxonomy_roles_mxbai_docs_by_text", texts, encode_docs)
    predicted = top_tags(vecs, tag_vecs, tag_ids, max(TAGGER_M_GRID))
    where = {c: i for i, c in enumerate(order)}

    result = {"seed": args.seed, "n_roles_per_split": args.n_roles // 2}
    boot = rng_for(args.seed, "tagger/boot").integers(0, args.n_roles // 2, size=(args.n_boot, args.n_roles // 2))
    chance = float(np.mean([len(tax.role_tags[r]) for r in reps]) / len(tag_ids))

    # (a) tag quality against the gold tag sets, on the test sample
    print(f"\n=== tagger vs gold tags, TEST sample (chance precision {chance:.3f}); 95% CI over roles ===")
    quality = {}
    test_rows = [where[c] for c in sample["test"]]
    for m in TAGGER_M_GRID:
        prec, rec, hit = [], [], []
        for i in test_rows:
            gold = tax.role_tags[reps[i]]
            got = {t for t, _ in predicted[i][:m]} & gold
            prec.append(len(got) / m)
            rec.append(len(got) / len(gold))
            hit.append(float(bool(got)))
        quality[m] = {"precision": ci(np.array(prec), boot), "recall": ci(np.array(rec), boot),
                      "hit_any": ci(np.array(hit), boot)}
        print(f"  m={m:<3} precision {fmt(quality[m]['precision'])}  recall {fmt(quality[m]['recall'])}  "
              f"at least one gold tag {fmt(quality[m]['hit_any'])}")
    result["test_tag_quality"] = {str(m): v for m, v in quality.items()}

    # (b) role retrieval from predicted tags
    def evaluate_split(split, m, b):
        rows = [where[c] for c in sample[split]]
        idx = world.index(b=b)
        S = idx.score_matrix([[t for t, _ in predicted[i][:m]] for i in rows])
        hm = hit_metrics(world, S, sample[split])
        _, cm = class_means(hm, sample[split])
        return cm

    sel = lambda cm: float(np.mean([cm[f"role@{k}"].mean() for k in KS]))
    dev_grid = {(m, b): sel(evaluate_split("dev", m, b)) for m in TAGGER_M_GRID for b in B_GRID}
    best_m, best_b = max(dev_grid, key=lambda c: (round(dev_grid[c], 9), -c[0], -abs(c[1] - 0.75)))
    print(f"\ndev tuning (mean of role@1/5/10 from predicted tags): best m={best_m}, b={best_b} -> {dev_grid[(best_m, best_b)]:.4f}")
    result["dev_grid"] = {f"m={m},b={b}": v for (m, b), v in dev_grid.items()}
    result["selected"] = {"m": best_m, "b": best_b}
    print("=== role retrieval from PREDICTED tags, TEST sample (Part A used gold-tag queries) ===")
    e2e = {}
    for m in TAGGER_M_GRID:
        cm = evaluate_split("test", m, best_b)
        e2e[m] = {k: ci(cm[k], boot) for k in ("role@1", "role@5", "role@10", "track@1", "track@3")}
        print(f"  m={m:<3}{' (tuned)' if m == best_m else '         '}" + "  ".join(f"{k} {fmt(v)}" for k, v in e2e[m].items()))
    result["test_role_retrieval_from_predicted_tags"] = {str(m): v for m, v in e2e.items()}

    result["seconds"] = round(time.time() - t_start, 1)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "tagger.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {out} ({result['seconds']}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("part", choices=["roles", "sat", "tagger"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    ap.add_argument("--data-dir", default="data_sat", help="(sat) dataset folder under pipeline/")
    ap.add_argument("--extra-grades", action="store_true",
                    help="(sat) also report on the extended labels (llm_judgments_merged_tag.json and "
                         "ground_truth_llm_tag.json in --data-dir, made by labeller.py merge); the "
                         "original-label numbers are unchanged")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples over classes / gigs")
    ap.add_argument("--n-roles", type=int, default=400, help="(tagger) role descriptions to tag, half dev, half test")
    ap.add_argument("--draws-scale", type=float, default=1.0,
                    help="(roles) scale the number of random query draws per class (0.2 for a quick smoke run)")
    args = ap.parse_args()
    {"roles": run_roles, "sat": run_sat, "tagger": run_tagger}[args.part](args)


if __name__ == "__main__":
    main()
