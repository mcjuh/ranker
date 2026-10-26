"""
Build the pipeline's canonical corpus from the schema-v2 data in `data/new/`,
and run the third (taxonomy) recall channel over it.

The SWE side already ships `data/new/*.csv` with the pipeline's own column names
(about_title, hire_description, ...) plus the flattened taxonomy, so this reads
those directly -- no field re-interpretation -- and only adds what the pipeline
requires and the v2 files do not carry: a `provider_id`, and `hire_id` taken
from `hirer_ref`.

Outputs (namespaced like every other dataset):
  data_v2/providers.json, data_v2/hirers.json        canonical records
  results_data_v2/taxonomy.json                      Card 2 third recall channel
  features_data_v2/taxonomy_scores.csv               Card 3 taxonomy features

Usage:
  python build_v2_corpus.py --data-dir data/new --tag data_v2
"""
import argparse
import csv
import json
from pathlib import Path

from taxonomy import TAXONOMY_FEATURES, TaxonomyRetriever, record_tags, taxonomy_scores

BASE = Path(__file__).parent

PROVIDER_COLS = ["about_title", "about_description", "services_offered_title",
                 "services_offered_description", "relevant_experience",
                 "years_experience", "categories_first", "specialisations_first"]
HIRER_COLS = ["hire_title", "hire_description", "hire_description_additional_notes",
              "duration_weeks_min", "duration_weeks_max"]


def _int_or_none(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def load_providers(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8-sig") as f:
        for i, r in enumerate(csv.DictReader(f), start=1):
            rec = {"provider_id": i}
            for c in PROVIDER_COLS:
                if c in r:
                    rec[c] = r[c]
            rec["years_experience"] = _int_or_none(r.get("years_experience"))
            # v2 taxonomy: nested object straight from the SWE side, kept verbatim.
            rec["tags"] = json.loads(r["tags_json"]) if r.get("tags_json") else None
            # Flattened fallbacks so record_tags() works even without tags_json.
            rec["category"] = r.get("category", "")
            rec["specialisation"] = r.get("specialisation", "")
            rec["skills"] = r.get("skills", "")
            rec["source_file"] = r.get("source_file", "")
            out.append(rec)
    return out


def load_hirers(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rec = {"hire_id": r["hirer_ref"]}
            for c in HIRER_COLS:
                if c in r:
                    rec[c] = r[c]
            rec["duration_weeks_min"] = _int_or_none(r.get("duration_weeks_min"))
            rec["duration_weeks_max"] = _int_or_none(r.get("duration_weeks_max"))
            rec["tags"] = json.loads(r["tags_json"]) if r.get("tags_json") else None
            rec["category"] = r.get("category", "")
            rec["specialisation"] = r.get("specialisation", "")
            rec["skills"] = r.get("skills", "")
            rec["source_file"] = r.get("source_file", "")
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/new")
    ap.add_argument("--tag", default="data_v2", help="output namespace (data_<tag>/, results_<tag>/)")
    args = ap.parse_args()

    src = BASE / args.data_dir
    data_dir = BASE / args.tag
    results_dir = BASE / f"results_{args.tag}"
    feat_dir = BASE / f"features_{args.tag}"
    for d in (data_dir, results_dir, feat_dir):
        d.mkdir(exist_ok=True)

    providers = load_providers(src / "providers.csv")
    hirers = load_hirers(src / "hirers.csv")
    (data_dir / "providers.json").write_text(json.dumps(providers, indent=2), encoding="utf-8")
    (data_dir / "hirers.json").write_text(json.dumps(hirers, indent=2), encoding="utf-8")
    print(f"corpus: {len(providers)} providers, {len(hirers)} hirers -> {data_dir}/")

    n_ptags = sum(1 for p in providers if record_tags(p))
    n_htags = sum(1 for h in hirers if record_tags(h))
    print(f"  with taxonomy tags: providers {n_ptags}/{len(providers)} "
          f"({n_ptags/len(providers):.1%}), hirers {n_htags}/{len(hirers)} ({n_htags/len(hirers):.1%})")

    # ---- third recall channel: rank providers by taxonomy match per gig ----
    retriever = TaxonomyRetriever(providers)
    print(f"  taxonomy index: {len(retriever._ids)} tagged providers, "
          f"{len(retriever._by_tag)} distinct tags")

    recall, feature_rows, hits = {}, [], []
    for h in hirers:
        g = record_tags(h)
        ranked = retriever.rank(g)
        recall[h["hire_id"]] = [pid for pid, _ in ranked]
        hits.append(len(ranked))
        for pid, score in ranked:
            p = providers[pid - 1]                      # provider_id is 1-based row order
            s = taxonomy_scores(g, record_tags(p))
            feature_rows.append((h["hire_id"], pid, s))

    (results_dir / "taxonomy.json").write_text(json.dumps(recall, indent=2), encoding="utf-8")
    hits_sorted = sorted(hits)
    print(f"\ntaxonomy recall -> {results_dir}/taxonomy.json")
    print(f"  gigs with >=1 tagged provider: {sum(1 for x in hits if x)}/{len(hirers)}")
    print(f"  candidates per gig: median {hits_sorted[len(hits_sorted)//2]}, "
          f"min {hits_sorted[0]}, max {hits_sorted[-1]}")

    with (feat_dir / "taxonomy_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hire_id", "provider_id"] + TAXONOMY_FEATURES)
        for hid, pid, s in feature_rows:
            w.writerow([hid, pid] + [f"{s[k]:.6f}" for k in TAXONOMY_FEATURES])
    print(f"taxonomy features -> {feat_dir}/taxonomy_scores.csv ({len(feature_rows)} rows)")

    import statistics as st
    print("\n  mean feature value over matched pairs (untuned components, Card 3):")
    for k in TAXONOMY_FEATURES:
        vals = [s[k] for _, _, s in feature_rows]
        print(f"    {k:<22} mean {st.fmean(vals):.4f}   max {max(vals):.4f}")


if __name__ == "__main__":
    main()
