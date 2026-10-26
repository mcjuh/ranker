"""
Per-section BM25 features for the Stage-2 ranker.

`rerank_ltr.py --bm25-fields` (merged in f0b7fcc) expects this script's output,
but the implementation file and its CSV were never committed -- this restores
them. Instead of one blended provider document (what `bm25_score` scores), each
provider section gets its own Okapi index, so the model can tell *where* a query
term matched (title vs about vs services vs experience) rather than only how much.

Columns (consumed by `rerank_ltr.BM25_FIELD_FEATURES`):
  bm25_title_title    about_title + services_offered_title (title-weighted 3x)
  bm25_desc_about     about_description
  bm25_desc_services  services_offered_description
  bm25_desc_exp       relevant_experience
  bm25_field_max      max of the four
  bm25_field_gap      max - second-highest (how concentrated the match is in one section)

Query tokenization is copied verbatim from `BM25Retriever._query_tokens` with a
title, including the repo's pre-existing 4x-title quirk, so these columns are on
the same scale/behaviour as `bm25_score`.

    python pipeline/bm25_fields.py --data-dir data/data_set_v5_tax
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from corpus import hirer_text, hirer_bm25_tokens, tokenize, simple_stem
from retrieval_bm25 import _load_stemmed_synonyms

BASE = Path(__file__).parent
BM25_FIELD_FEATURES = ["bm25_title_title", "bm25_desc_about", "bm25_desc_services",
                       "bm25_desc_exp", "bm25_field_max", "bm25_field_gap"]
SECTIONS = ("title", "about", "services", "exp")
SECTION_KEY = {"about": "about_description",
               "services": "services_offered_description",
               "exp": "relevant_experience"}


def section_tokens(p: dict, section: str, title_weight: int = 3) -> list[str]:
    if section == "title":
        text = " ".join(x for x in [p.get("about_title"), p.get("services_offered_title")] if x)
        return tokenize(text, stem=True) * title_weight
    return tokenize(p.get(SECTION_KEY[section]) or "", stem=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax")
    ap.add_argument("--tag", default=None,
                    help="output namespace under features_data/ (default: basename of --data-dir)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--title-weight", type=int, default=3)
    args = ap.parse_args()

    data_dir = BASE / args.data_dir
    tag = args.tag or data_dir.name
    feat_dir = BASE / "features_data" / tag

    providers = json.loads((data_dir / "providers.json").read_text(encoding="utf-8"))
    hirers = {h["hire_id"]: h for h in json.loads((data_dir / "hirers.json").read_text(encoding="utf-8"))}
    pids = [p["provider_id"] for p in providers]
    pid_index = {pid: i for i, pid in enumerate(pids)}

    print(f"building {len(SECTIONS)} field indexes over {len(providers)} providers ...")
    indexes = {s: BM25Okapi([section_tokens(p, s, args.title_weight) for p in providers])
               for s in SECTIONS}

    synonyms = _load_stemmed_synonyms()

    def query_tokens(h: dict) -> list[str]:
        # Mirror BM25Retriever._query_tokens(query_text=hirer_text(h), query_title=...)
        h_like = {"hire_title": h.get("hire_title"),
                  "hire_description": hirer_text(h),
                  "hire_description_additional_notes": None}
        toks = hirer_bm25_tokens(h_like, args.title_weight)
        out = list(toks)
        for t in toks:
            out.extend(synonyms.get(simple_stem(t), []))
        return out

    cand = defaultdict(list)
    with (feat_dir / f"candidates_top{args.top_k}.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            cand[int(r["hire_id"])].append(int(r["provider_id"]))

    out_path = feat_dir / "bm25_fields.csv"
    n = 0
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hire_id", "provider_id"] + BM25_FIELD_FEATURES)
        for hid in sorted(cand):
            qt = query_tokens(hirers[hid])
            sc = {s: indexes[s].get_scores(qt) for s in SECTIONS}
            for pid in cand[hid]:
                i = pid_index[pid]
                vals = [float(sc[s][i]) for s in SECTIONS]
                ordered = sorted(vals, reverse=True)
                mx = ordered[0]
                gap = ordered[0] - ordered[1] if len(ordered) > 1 else 0.0
                w.writerow([hid, pid] + [round(v, 6) for v in vals] + [round(mx, 6), round(gap, 6)])
                n += 1

    print(f"wrote {n} rows x {len(BM25_FIELD_FEATURES)} field-BM25 features to {out_path}")


if __name__ == "__main__":
    main()
