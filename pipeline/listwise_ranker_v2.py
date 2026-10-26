"""
Set-wise ranker v2: adds the OOF cross-encoder score and a fold-aware provider
prior as INPUT features, and supports a second (ListNet) loss.

Rationale: the v1 model attends across the 50 candidates but only sees the
retrieval/tag features. The cross-encoder reads the full (gig, provider) text and
is the strongest *independent* signal we have, but rank-level RRF is a coarse way
to combine it. Letting the set-wise model consume `ce_score` directly lets it
learn per-query how much to trust the CE. The provider prior is rebuilt per fold
from training queries only (leave-own-query-out), so no label leaks.

    python pipeline/listwise_ranker_v2.py --ce-tag v5ce_minilm --loss pairwise --tag pair
    python pipeline/listwise_ranker_v2.py --ce-tag v5ce_minilm --loss listnet  --tag lnet
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

import evaluate as E
import rerank_ltr as R
import tier1_rank as T
import hpo_ltr as H
from listwise_ranker import SetRanker

GAIN = {0: 0.0, 1: 33.0, 2: 67.0, 3: 100.0}


def rangk_desc(vals):
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    out = [0] * len(vals)
    prev, prev_rank = None, 0
    for pos, i in enumerate(order, 1):
        if vals[i] != prev:
            prev, prev_rank = vals[i], pos
        out[i] = prev_rank
    return out


def add_ce_rank_features(grp, cols):
    """Per-query rank of each CE score + consensus votes. Ranks are relative to
    the shortlist, which is how a reranker is actually consumed."""
    names = [f"rank_{c}" for c in cols] + ["ce_top10_votes", "ce_top20_votes", "ce_mean_rank"]
    for q, rs in grp.items():
        ranks = {c: rangk_desc([r[c] for r in rs]) for c in cols}
        for c in cols:
            for r, v in zip(rs, ranks[c]):
                r[f"rank_{c}"] = float(v)
        for i, r in enumerate(rs):
            rk = [ranks[c][i] for c in cols]
            r["ce_top10_votes"] = float(sum(x <= 10 for x in rk))
            r["ce_top20_votes"] = float(sum(x <= 20 for x in rk))
            r["ce_mean_rank"] = float(np.mean(rk))
    return names


def prior_by_fold(rows, tr_q, alpha=5.0):
    tot_s, tot_c = defaultdict(float), defaultdict(int)
    pq_s, pq_c = defaultdict(float), defaultdict(int)
    gsum = gcnt = 0
    for r in rows:
        if r["label"] is None or r["hire_id"] not in tr_q:
            continue
        p, q, y = r["provider_id"], r["hire_id"], r["label"]
        tot_s[p] += y; tot_c[p] += 1; pq_s[(p, q)] += y; pq_c[(p, q)] += 1
        gsum += y; gcnt += 1
    gm = gsum / gcnt if gcnt else 0.0
    for r in rows:
        p, q = r["provider_id"], r["hire_id"]
        s, c = tot_s[p], tot_c[p]
        if q in tr_q:
            s -= pq_s[(p, q)]; c -= pq_c[(p, q)]
        r["prov_prior"] = (s + alpha * gm) / (c + alpha)
        r["prov_ct"] = float(np.log1p(c))


def build_x(by_query, feats, queries):
    out = []
    for q in queries:
        x = np.asarray([[r[f] for f in feats] for r in by_query[q]], dtype=np.float32)
        mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        out.append(np.clip((x - mu) / sd, -5, 5))
    return np.stack(out)


def pairwise_loss(scores, gains, mask):
    g = gains * mask.float()
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    w = torch.relu(g.unsqueeze(2) - g.unsqueeze(1))
    return -(torch.nn.functional.logsigmoid(diff) * w).sum() / (w.sum() + 1e-8)


def listnet_loss(scores, gains, mask):
    g = (gains * mask.float()).clamp_min(0)
    p = g / g.sum(dim=1, keepdim=True).clamp_min(1e-8)
    logq = torch.log_softmax(scores.masked_fill(~mask, -1e9), dim=1)
    return -(p * logq).sum(dim=1).mean()


def lambdarank_loss(scores, gains, mask):
    """LambdaRank: pair weight = |gain_i - gain_j| * |disc(rank_i) - disc(rank_j)|.

    The discount uses each candidate's CURRENT predicted rank (detached), exactly
    the standard lambda gradient, so the loss focuses on swaps near the top of the
    list -- the only swaps NDCG@10 can see.
    """
    g = gains * mask.float()
    pos = torch.argsort(torch.argsort(scores.detach(), dim=1, descending=True), dim=1).float() + 1.0
    disc = 1.0 / torch.log2(pos + 1.0)
    delta = g.unsqueeze(2) - g.unsqueeze(1)
    w = torch.relu(delta) * torch.abs(disc.unsqueeze(2) - disc.unsqueeze(1))
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    return -(torch.nn.functional.logsigmoid(diff) * w).sum() / (w.sum() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax")
    ap.add_argument("--ce-tag", default="v5ce_minilm")
    ap.add_argument("--ce-tag-extra", default=None,
                    help="a second cross-encoder score to merge as ce_score2 (e.g. v5ce_bge)")
    ap.add_argument("--ce-tag-third", default=None,
                    help="a third cross-encoder score to merge as ce_score3 (e.g. v5ce_gte)")
    ap.add_argument("--ce-tag-fourth", default=None,
                    help="a fourth cross-encoder score to merge as ce_score4")
    ap.add_argument("--ce-data-dir", default="data_set_v5")
    ap.add_argument("--ltr-scores", action="store_true",
                    help="merge features_data/<data-dir>/ltr_oof_scores.csv (the tuned LambdaMART's OOF score) ")
    ap.add_argument("--soft-labels", action="store_true",
                    help="weight the pairwise loss by the grader's continuous score (0-1) instead "
                         "of the coarse 0/1/2/3 gains")
    ap.add_argument("--loss", default="pairwise", choices=["pairwise", "listnet", "lambdarank"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="0 = GroupKFold by query; >0 = shuffled KFold over queries with this seed")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch-queries", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = json.loads((R.BASE / args.data_dir / "ground_truth_llm.json").read_text())
    T.gt_global = gt
    rows, by_query, all_feats = H.build_rows(args.data_dir)
    feats = list(all_feats)

    if args.ce_tag:
        ce_path = R.BASE / "features_data" / args.ce_data_dir / f"ce_scores_{args.ce_tag}.csv"
        ce = {(r["hire_id"], r["provider_id"]): float(r["ce_score"]) for r in csv.DictReader(ce_path.open())}
        miss = 0
        for r in rows:
            v = ce.get((str(r["hire_id"]), str(r["provider_id"])))
            if v is None:
                v = float("nan"); miss += 1
            r["ce_score"] = v
        # NaN would poison the transformer; standardisation below turns it into 0
        for r in rows:
            if r["ce_score"] != r["ce_score"]:
                r["ce_score"] = 0.0
        feats.append("ce_score")
        print(f"ce_score merged ({miss} missing -> 0.0)")

    if args.ce_tag_extra:
        ce_path = R.BASE / "features_data" / args.ce_data_dir / f"ce_scores_{args.ce_tag_extra}.csv"
        ce = {(r["hire_id"], r["provider_id"]): float(r["ce_score"]) for r in csv.DictReader(ce_path.open())}
        miss = 0
        for r in rows:
            v = ce.get((str(r["hire_id"]), str(r["provider_id"])))
            if v is None:
                v = float("nan"); miss += 1
            r["ce_score2"] = 0.0 if v != v else v
        feats.append("ce_score2")
        print(f"ce_score2 merged ({miss} missing -> 0.0)")

    if args.ce_tag_third:
        ce_path = R.BASE / "features_data" / args.ce_data_dir / f"ce_scores_{args.ce_tag_third}.csv"
        ce = {(r["hire_id"], r["provider_id"]): float(r["ce_score"]) for r in csv.DictReader(ce_path.open())}
        miss = 0
        for r in rows:
            v = ce.get((str(r["hire_id"]), str(r["provider_id"])))
            if v is None:
                v = float("nan"); miss += 1
            r["ce_score3"] = 0.0 if v != v else v
        feats.append("ce_score3")
        print(f"ce_score3 merged ({miss} missing -> 0.0)")

    if args.ce_tag_fourth:
        ce_path = R.BASE / "features_data" / args.ce_data_dir / f"ce_scores_{args.ce_tag_fourth}.csv"
        ce = {(r["hire_id"], r["provider_id"]): float(r["ce_score"]) for r in csv.DictReader(ce_path.open())}
        miss = 0
        for r in rows:
            v = ce.get((str(r["hire_id"]), str(r["provider_id"])))
            if v is None:
                v = float("nan"); miss += 1
            r["ce_score4"] = 0.0 if v != v else v
        feats.append("ce_score4")
        print(f"ce_score4 merged ({miss} missing -> 0.0)")

    grp = defaultdict(list)
    for r in rows:
        grp[r["hire_id"]].append(r)
    ce_cols = [c for c in ("ce_score", "ce_score2", "ce_score3", "ce_score4") if c in feats]
    if ce_cols:
        extra = add_ce_rank_features(grp, ce_cols)
        feats.extend(extra)
        print(f"CE rank/vote features added: {extra}")

    if args.ltr_scores:
        p = R.FEAT_DIR / "ltr_oof_scores.csv"
        sc = {(r["hire_id"], r["provider_id"]): float(r["ltr_score"]) for r in csv.DictReader(p.open())}
        miss = 0
        for r in rows:
            v = sc.get((str(r["hire_id"]), str(r["provider_id"])))
            if v is None:
                v = float("nan"); miss += 1
            r["ltr_score"] = 0.0 if v != v else v
        feats.append("ltr_score")
        print(f"ltr_score merged ({miss} missing -> 0.0)")

    queries = sorted(by_query)
    q_index = {q: i for i, q in enumerate(queries)}
    soft = None
    if args.soft_labels:
        from soft_label_rank import load_soft
        soft = load_soft(R.BASE / "data" / "data_set_v5")
        print(f"soft labels loaded: {len(soft)} pairs")
    G = np.stack([[
        100.0 * soft[(r["hire_id"], r["provider_id"])] if (soft is not None
        and (r["hire_id"], r["provider_id"]) in soft and r["label"] is not None)
        else (GAIN.get(r["label"], 0.0) if r["label"] is not None else 0.0)
        for r in by_query[q]] for q in queries]).astype(np.float32)
    M = np.stack([[r["label"] is not None for r in by_query[q]] for q in queries])
    print(f"queries={len(queries)} features={len(feats)} device={device} loss={args.loss}")

    loss_fn = {"pairwise": pairwise_loss, "listnet": listnet_loss,
               "lambdarank": lambdarank_loss}[args.loss]
    oof = {}
    if args.split_seed == 0:
        folds_iter = GroupKFold(n_splits=args.folds).split(queries, groups=queries)
    else:
        from sklearn.model_selection import KFold
        folds_iter = KFold(n_splits=args.folds, shuffle=True,
                           random_state=args.split_seed).split(queries)
    for fold, (tr, te) in enumerate(folds_iter, 1):
        tr_q = [queries[i] for i in tr]
        te_q = [queries[i] for i in te]
        prior_by_fold(rows, set(tr_q))

        Xtr = torch.tensor(build_x(by_query, feats, tr_q), device=device)
        Xte = torch.tensor(build_x(by_query, feats, te_q), device=device)
        Gtr = torch.tensor(G[[q_index[q] for q in tr_q]], device=device)
        Mtr = torch.tensor(M[[q_index[q] for q in tr_q]], device=device)

        fold_scores = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = SetRanker(len(feats), d=args.d_model, layers=args.layers,
                              dropout=args.dropout).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            n = len(tr_q)
            for _ in range(args.epochs):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, args.batch_queries):
                    idx = perm[i:i + args.batch_queries]
                    opt.zero_grad()
                    loss = loss_fn(model(Xtr[idx]), Gtr[idx], Mtr[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                fold_scores.append(model(Xte).cpu().numpy())
        sc = np.mean(fold_scores, axis=0)
        for k, q in enumerate(te_q):
            for r, s in zip(by_query[q], sc[k]):
                oof[(q, r["provider_id"])] = float(s)

        lists = {q: [r["provider_id"] for r in sorted(by_query[q], key=lambda r: -oof[(q, r["provider_id"])])]
                 for q in te_q}
        m = E.evaluate_all_hirers({str(q): v for q, v in lists.items()},
                                  {k: v for k, v in gt.items() if int(k) in set(te_q)}, ks=(10,))
        print(f"  fold {fold}: NDCG@10={m['ndcg@10']:.4f}")

    lists = T.ranked_from_oof(rows, by_query, oof)
    m = E.evaluate_all_hirers({str(q): v for q, v in lists.items()}, gt, ks=(5, 10))
    print(f"OOF: P@5={m['precision@5']:.4f} NDCG@5={m['ndcg@5']:.4f} "
          f"P@10={m['precision@10']:.4f} NDCG@10={m['ndcg@10']:.4f} MRR={m['mrr']:.4f}")
    out = Path(R.RESULTS_DIR) / f"listwise_{args.tag}.json"
    out.write_text(json.dumps({str(q): [int(p) for p in v] for q, v in lists.items()}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
