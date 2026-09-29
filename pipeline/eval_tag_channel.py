"""
Evaluation of the tag-ID BM25 recall channel (retrieval_tagbm25.TagBM25).

    python pipeline/eval_tag_channel.py roles          # Part A: role-level, on the taxonomy alone
    python pipeline/eval_tag_channel.py sat            # Part B: provider-level, on tagged data_sat

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("part", choices=["roles", "sat"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples over classes")
    ap.add_argument("--draws-scale", type=float, default=1.0,
                    help="scale the number of random query draws per class (0.2 for a quick smoke run)")
    args = ap.parse_args()
    if args.part == "roles":
        run_roles(args)
    else:
        raise SystemExit("Part B (sat) is added once data_sat has been tagged")


if __name__ == "__main__":
    main()
