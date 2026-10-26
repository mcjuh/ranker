"""
Fine-tune the dense retriever on data_set_v5's LLM grades.

`finetune_embeddings.py` targets the retired synthetic corpus (formula GT,
`_*_with_taxonomy.json`, `category_id`). This adapts the same recipe to v5:

  anchor   = gig text (with the model's query prefix)
  positive = provider with grade 3 (fallback: grade >= 2)
  negative = HARD negative: a grade-0/1 provider that shares the gig's taxonomy
             category (same neighbourhood, wrong fit) -- the boundary the ranker
             is weakest on. Falls back to any grade-0/1 provider.

MultipleNegativesRankingLoss with `hardness_mode="hard_negatives"` uses the
explicit negatives plus in-batch negatives. MatryoshkaLoss keeps the 1024->64
truncation ladder valid, matching retrieval_dense.

Saves a SentenceTransformer checkpoint that `features.py --dense-model` can load.

    python pipeline/finetune_dense_v5.py --epochs 3 --batch-size 32
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses,
)

from corpus import hirer_text, provider_text
from retrieval_dense import BASE_MODEL_NAME, QUERY_PREFIX_GENERIC

BASE = Path(__file__).parent
MATRYOSHKA_DIMS = [1024, 512, 256, 128, 64]


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def record_categories(rec):
    return {c["name"] for c in (rec.get("tags") or {}).get("categories") or []}


def build_rows(hirers, providers, judgments, rng, max_pos=3, max_neg_per_pos=2):
    prov = {p["provider_id"]: p for p in providers}
    pcats = {p["provider_id"]: record_categories(p) for p in providers}
    rows = []
    for h in hirers:
        row = judgments.get(str(h["hire_id"]), {})
        if not row:
            continue
        gcats = record_categories(h)
        pos = [int(p) for p, g in row.items() if g >= 3]
        if not pos:
            pos = [int(p) for p, g in row.items() if g >= 2]
        if not pos:
            continue
        pos = pos[:max_pos]
        negs = [int(p) for p, g in row.items() if g <= 1 and (pcats[int(p)] & gcats)]
        if not negs:
            negs = [int(p) for p, g in row.items() if g <= 1]
        if not negs:
            continue
        anchor = QUERY_PREFIX_GENERIC + hirer_text(h)
        for pid in pos:
            chosen = rng.sample(negs, min(max_neg_per_pos, len(negs)))
            for nid in chosen:
                rows.append(dict(anchor=anchor,
                                 positive=provider_text(prov[pid]),
                                 negative=provider_text(prov[nid])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax")
    ap.add_argument("--source-data", default="data/data_set_v5")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-name", default="mxbai-ft-v5")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    hirers = load_json(BASE / args.data_dir / "hirers.json")
    providers = load_json(BASE / args.data_dir / "providers.json")
    judgments = load_json(BASE / args.source_data / "llm_judgments_merged.json")

    rows = build_rows(hirers, providers, judgments, rng)
    rng.shuffle(rows)
    print(f"{len(rows)} hard-negative triplets over {len(hirers)} gigs; device={device}")
    if not rows:
        raise SystemExit("no triplets built")

    model = SentenceTransformer(BASE_MODEL_NAME, device=device)
    base_loss = losses.MultipleNegativesRankingLoss(model, hardness_mode="hard_negatives")
    train_loss = losses.MatryoshkaLoss(model, base_loss, matryoshka_dims=MATRYOSHKA_DIMS,
                                       n_dims_per_step=1)

    out_dir = BASE / "models_data" / "data_set_v5" / args.out_name
    trainer_dir = BASE / "models_data" / "data_set_v5" / f"_{args.out_name}-trainer"
    st_args = SentenceTransformerTrainingArguments(
        output_dir=str(trainer_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, warmup_steps=0.1, max_grad_norm=args.max_grad_norm,
        fp16=False, bf16=False, report_to=[], logging_steps=10, save_strategy="no",
        seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(model=model, args=st_args,
                                         train_dataset=Dataset.from_list(rows), loss=train_loss)
    trainer.train()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
