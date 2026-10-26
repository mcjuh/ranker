"""
Merge a grading top-up into a corpus's label files.

The v5 judging pool came from the text retrievers (RRF/BM25/dense top-k plus a few
randoms), so a ranker that promotes providers those retrievers ranked low surfaces
pairs nobody graded -- and evaluate.py scores an ungraded pair as grade 0. The
SkillsFuture tag features do exactly that: before the top-up, 13% of their top-10
was ungraded, which read as a -0.04 NDCG@10 "regression" (see ablation.md §5).

`data/data_set_v5/judgments_topup_tags.jsonl` holds those pairs, graded by the same
model and prompt as the original 21,696 (qwen3.8:27b, rubric_01.v5; a 60-pair
re-grade of already-graded pairs reproduced 60/60 grades exactly). This script folds
them into:

  data/<corpus>/llm_judgments_merged.json      raw 0-3 grades (what features.py reads)
  data/<corpus>/ground_truth_llm.json          0/33/67/100, grade > 0 only (what evaluate.py reads)
  features_data/<corpus>/candidates_top50.csv  label / judged columns
  features_data/<corpus>/train_pairs.csv       the judged subset of the above

Idempotent: a pair that already carries a grade is never overwritten, so rerunning is a
no-op. The top-up file stays the record of which grades were added, so the original
label set is always `merged minus top-up` (tag_ablation.py relies on this).

    python pipeline/merge_topup_grades.py --data-dir data/data_set_v5
    python pipeline/merge_topup_grades.py --data-dir data/data_set_v5_tax
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent
TOPUP = BASE / "data/data_set_v5/judgments_topup_tags.jsonl"


def load_topup(path: Path = TOPUP) -> dict:
    """{(hire_id, provider_id) as str: grade} for every top-up record."""
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(str(r["hire_id"]), str(r["provider_id"]))] = int(r["grade"])
    return out


def rewrite_csv(path: Path, merged: dict, judged_only: bool) -> int:
    """Fill label/judged for newly graded rows; returns how many rows changed."""
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
    changed = 0
    for r in rows:
        g = merged.get(r["hire_id"], {}).get(r["provider_id"])
        if g is not None and r["judged"] != "1":
            r["label"], r["judged"] = str(g), "1"
            changed += 1
    if judged_only:
        # train_pairs.csv is the judged subset of candidates_top50.csv, in its row order
        cand = path.with_name("candidates_top50.csv")
        with cand.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["judged"] == "1"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5")
    ap.add_argument("--topup", type=Path, default=TOPUP)
    args = ap.parse_args()

    data_dir = BASE / args.data_dir
    feat_dir = BASE / args.data_dir.replace("data/", "features_data/", 1)
    topup = load_topup(args.topup)

    merged_path = data_dir / "llm_judgments_merged.json"
    merged = json.loads(merged_path.read_text())
    added = Counter()
    for (h, p), g in topup.items():
        if p not in merged.setdefault(h, {}):
            merged[h][p] = g
            added[g] += 1
    merged_path.write_text(json.dumps(merged, indent=1))
    gt = {h: {p: round(g * 100 / 3) for p, g in d.items() if g > 0} for h, d in merged.items()}
    (data_dir / "ground_truth_llm.json").write_text(json.dumps(gt, indent=1))

    n_cand = rewrite_csv(feat_dir / "candidates_top50.csv", merged, judged_only=False)
    rewrite_csv(feat_dir / "train_pairs.csv", merged, judged_only=True)
    print(f"{args.data_dir}: {sum(added.values())} new grades added "
          f"(by grade {dict(sorted(added.items()))}); {n_cand} candidate rows newly labelled; "
          f"{sum(map(len, merged.values()))} graded pairs in total")


if __name__ == "__main__":
    main()
