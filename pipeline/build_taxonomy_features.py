"""
Build the taxonomy recall channel + match features for a corpus that already exists.

build_v2_corpus.py does this inline while rebuilding from the raw CSVs in data/new,
which would clobber a corpus built from any other source (e.g. data_pilot, whose
records come from platform_sample and carry different wording). This runs the same
logic against data_<tag>/hirers.json + providers.json.

Usage: python build_taxonomy_features.py --tag data_pilot
Writes: results_<tag>/taxonomy.json, features_<tag>/taxonomy_scores.csv
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

from taxonomy import TAXONOMY_FEATURES, TaxonomyRetriever, record_tags, taxonomy_scores

BASE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="data_pilot", help="corpus dir, e.g. data_pilot")
    args = ap.parse_args()

    tag = args.tag
    # `--tag` may be a bare corpus name (data_sat -> data_sat/) or a path under
    # data/ (data/data_set_v5). Pick whichever actually holds the corpus.
    data_dir = next((p for p in (BASE / tag, BASE / f"data_{tag}")
                     if (p / "hirers.json").exists()), None)
    if data_dir is None:
        raise SystemExit(f"no hirers.json under {BASE / tag} or {BASE / f'data_{tag}'}")
    results_dir = BASE / f"results_{tag}"
    feat_dir = BASE / f"features_{tag}"
    results_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    hirers = json.loads((data_dir / "hirers.json").read_text(encoding="utf-8"))
    providers = json.loads((data_dir / "providers.json").read_text(encoding="utf-8"))
    print(f"{data_dir.name}: {len(providers)} providers, {len(hirers)} hirers")

    retriever = TaxonomyRetriever(providers)
    print(f"  taxonomy index: {len(retriever._ids)} tagged providers, "
          f"{len(retriever._by_tag)} distinct tags")

    recall, feature_rows, hits = {}, [], []
    for h in hirers:
        g = record_tags(h)
        ranked = retriever.rank(g)
        recall[h["hire_id"]] = [pid for pid, _ in ranked]
        hits.append(len(ranked))
        for pid, _ in ranked:
            p = providers[pid - 1]                       # provider_id is 1-based row order
            feature_rows.append((h["hire_id"], pid, taxonomy_scores(g, record_tags(p))))

    (results_dir / "taxonomy.json").write_text(json.dumps(recall, indent=2), encoding="utf-8")
    hs = sorted(hits)
    print(f"  gigs with >=1 tagged provider: {sum(1 for x in hits if x)}/{len(hirers)}")
    print(f"  candidates per gig: median {hs[len(hs)//2]}, min {hs[0]}, max {hs[-1]}")

    with (feat_dir / "taxonomy_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hire_id", "provider_id"] + TAXONOMY_FEATURES)
        for hid, pid, s in feature_rows:
            w.writerow([hid, pid] + [f"{s[k]:.6f}" for k in TAXONOMY_FEATURES])
    print(f"taxonomy features -> {feat_dir}/taxonomy_scores.csv ({len(feature_rows)} rows)")

    print("\n  mean feature value over matched pairs:")
    for k in TAXONOMY_FEATURES:
        vals = [s[k] for _, _, s in feature_rows]
        print(f"    {k:<22} mean {st.fmean(vals):.4f}   max {max(vals):.4f}")


if __name__ == "__main__":
    main()
