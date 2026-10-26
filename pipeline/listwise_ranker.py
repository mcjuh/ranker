"""
Set-wise neural ranker: a small transformer over each query's 50 candidates.

The GBDT scores every candidate independently from its own features; it cannot
see the rest of the shortlist. NDCG@10 is a *list* metric, so a model that
attends across the candidate set can in principle learn relative ordering
("this provider is the best fit for this gig") that a per-row tree cannot.

Architecture: per-candidate MLP -> 2 transformer layers (self-attention over the
50 candidates) -> scalar score. Loss: pairwise logistic weighted by |gain_i -
gain_j| (a LambdaRank surrogate), computed over *labelled* candidates only.

Everything is out-of-fold by query (GroupKFold), evaluated against the same
graded ground truth. Runs on GPU.

    python pipeline/listwise_ranker.py --data-dir data/data_set_v5_tax
"""
import argparse
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

GAIN = {0: 0.0, 1: 33.0, 2: 67.0, 3: 100.0}


class SetRanker(nn.Module):
    def __init__(self, n_feat, d=64, layers=2, heads=4, dropout=0.15):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_feat, d), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=2 * d,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):                       # [B, N, F]
        h = self.enc(self.inp(x))
        return self.head(h).squeeze(-1)         # [B, N]


def rank_loss(scores, gains, mask):
    """Pairwise logistic, weighted by |gain_i - gain_j|.

    Fully vectorised: [B,N] -> [B,N,N]. Labelledness is encoded by zeroing the
    gains of unlabelled candidates, so they never contribute weight.
    """
    g = gains * mask.float()
    diff = scores.unsqueeze(2) - scores.unsqueeze(1)      # s_i - s_j
    w = torch.relu(g.unsqueeze(2) - g.unsqueeze(1))       # >0 only when g_i > g_j
    denom = w.sum() + 1e-8
    return -(torch.nn.functional.logsigmoid(diff) * w).sum() / denom


def build_tensors(by_query, feats, queries):
    """[Q, N, F] standardized per query, plus gains and label masks."""
    X, G, M = [], [], []
    for q in queries:
        rs = by_query[q]
        x = np.asarray([[r[f] for f in feats] for r in rs], dtype=np.float32)
        mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        x = np.clip((x - mu) / sd, -5, 5)
        g = np.asarray([GAIN.get(r["label"], 0.0) if r["label"] is not None else 0.0 for r in rs],
                       dtype=np.float32)
        m = np.asarray([r["label"] is not None for r in rs])
        X.append(x); G.append(g); M.append(m)
    return np.stack(X), np.stack(G), np.stack(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-queries", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3, help="average this many seeds per fold")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    gt = json.loads((R.BASE / args.data_dir / "ground_truth_llm.json").read_text())
    T.gt_global = gt
    rows, by_query, all_feats = H.build_rows(args.data_dir)
    feats = [f for f in all_feats if f not in T.NEW_PRIOR]   # no fold-dependent prior
    queries = sorted(by_query)
    print(f"queries={len(queries)} features={len(feats)} device={device}")

    X, G, M = build_tensors(by_query, feats, queries)
    q_index = {q: i for i, q in enumerate(queries)}
    rng = np.random.default_rng(0)
    oof = {}

    for fold, (tr, te) in enumerate(GroupKFold(n_splits=args.folds).split(queries, groups=queries), 1):
        tr_q = [queries[i] for i in tr]
        te_q = [queries[i] for i in te]
        Xtr = torch.tensor(X[[q_index[q] for q in tr_q]], device=device)
        Gtr = torch.tensor(G[[q_index[q] for q in tr_q]], device=device)
        Mtr = torch.tensor(M[[q_index[q] for q in tr_q]], device=device)

        fold_scores = []
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = SetRanker(len(feats)).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            model.train()
            n = len(tr_q)
            for ep in range(args.epochs):
                perm = torch.randperm(n, device=device)
                for i in range(0, n, args.batch_queries):
                    idx = perm[i:i + args.batch_queries]
                    opt.zero_grad()
                    loss = rank_loss(model(Xtr[idx]), Gtr[idx], Mtr[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
            model.eval()
            with torch.no_grad():
                te_idx = [q_index[q] for q in te_q]
                sc = model(torch.tensor(X[te_idx], device=device)).cpu().numpy()
            fold_scores.append(sc)

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

    out = Path(R.RESULTS_DIR) / "listwise.json"
    out.write_text(json.dumps({str(q): [int(p) for p in v] for q, v in lists.items()}))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
