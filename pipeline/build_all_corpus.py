"""
data_all: one corpus using everything usable in data/new.

IN (all of it):
  * 1,106 real hirers + 3,024 real providers (2,229 extracted + 795 manufactured)
  * 30 crafted gigs   -- hand-written schema-v2 queries, no id collisions
  * 900 real grades   -- gig_grades.csv, re-pointed at corpus ids via source_file

OUT (and why):
  * _hirers_with_taxonomy.json / _providers_with_taxonomy.json / ground_truth*.json /
    llm_judgments_merged.json -- the 130-gig synthetic set. Superseded by the v2
    corpus; its ids (1..130) join to nothing here. Including it would add stale
    queries with no shared id space.
  * platform_sample providers/gigs as separate records -- they are anonymised
    re-renderings of corpus entities (27/30 gigs, 28/30 providers share a
    source_file). Adding them would put two copies of the same entity in the
    retrieval pool. Instead the GRADES are carried over (below).
  * availability / rate -- 0% populated in the raw schema, so no feature can use them.

LABEL CAVEAT, stated plainly: the grades were assigned against the platform_sample
TEXT, not the corpus text for the same entity. Attaching them by source_file assumes
an anonymised rewrite preserves relevance. That is an assumption, not a fact, and it
is the price of having one corpus instead of two. The alternative -- swapping those
27 gigs / 28 providers for their sample wording -- trades label fidelity against
corpus consistency; say the word and I will build that variant instead.

Usage: python build_all_corpus.py --tag data_all
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from build_pilot_corpus import nest_tags

BASE = Path(__file__).parent
N = BASE / "data" / "new"


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="data_all")
    ap.add_argument("--base-corpus", default="data_v2", help="already-built corpus to extend")
    args = ap.parse_args()
    out = BASE / args.tag
    out.mkdir(parents=True, exist_ok=True)

    hirers = jload(BASE / args.base_corpus / "hirers.json")
    providers = jload(BASE / args.base_corpus / "providers.json")
    print(f"base corpus {args.base_corpus}: {len(hirers)} hirers, {len(providers)} providers")

    # ---- overlay the graded wording onto the records that carry labels -------
    # The grades were assigned against the platform_sample TEXT, not the corpus text
    # for the same entity. Without this overlay the labels attach to documents whose
    # wording was never judged, and the labelled pairs become unrecoverable by
    # ranking: measured on the plain merge, the six relevant providers land at ranks
    # 222-2058 of 3,024 and every metric reads 0.0000.
    ps = jload(N / "platform_sample/providers_sample.json")
    hs = jload(N / "platform_sample/hirers_sample.json")
    gig_by_sf = {x["source_file"]: x for x in hs}
    prov_by_sf = {x["source_file"]: x for x in ps}
    n_g = n_p = 0
    for h in hirers:
        s = gig_by_sf.get(h["source_file"])
        if not s:
            continue
        h["hire_title"] = s.get("gig_title") or h["hire_title"]
        h["hire_description"] = s.get("short_gig_description") or h["hire_description"]
        if s.get("search_tags"):
            h["tags"] = nest_tags(s)
        h["text_source"] = "platform_sample (graded)"
        n_g += 1
    for p in providers:
        s = prov_by_sf.get(p["source_file"])
        if not s:
            continue
        p["about_title"] = s.get("about_headline") or s.get("title") or p["about_title"]
        p["about_description"] = s.get("about_bio") or p["about_description"]
        svc = [x for x in (s.get("services_i_offer") or []) if isinstance(x, dict)]
        if svc:
            p["services_offered_title"] = " | ".join(x.get("service_title", "") for x in svc)
            p["services_offered_description"] = " ".join(x.get("service_detail", "") for x in svc)
        ach = s.get("relevant_achievements") or []
        if ach:
            p["relevant_experience"] = " ".join(str(a) for a in ach)
        if s.get("search_tags"):
            p["tags"] = nest_tags(s)
        p["text_source"] = "platform_sample (graded)"
        n_p += 1
    print(f"  overlaid graded wording: {n_g} gigs, {n_p} providers")

    # ---- + 30 crafted gigs -------------------------------------------------
    crafted = jload(N / "crafted_gigs/hirers_crafted.json")
    have_hsf = {h["source_file"] for h in hirers}
    have_ref = {h["hire_id"] for h in hirers}
    added = 0
    for c in crafted:
        if c["source_file"] in have_hsf or c["hirer_ref"] in have_ref:
            print(f"  skip (collision): {c['source_file']}")
            continue
        hirers.append({
            "hire_id": c["hirer_ref"],
            "hire_title": c.get("gig_title") or "",
            "hire_description": c.get("short_description") or "",
            "hire_description_additional_notes": c.get("additional_notes") or "",
            "duration_weeks_min": c.get("duration_weeks_min"),
            "duration_weeks_max": c.get("duration_weeks_max"),
            "tags": c.get("tags") or [],
            "source_file": c["source_file"],
            "provenance": "crafted",
        })
        added += 1
    print(f"  + {added} crafted gigs -> {len(hirers)} queries")

    # ---- + 900 real grades, re-pointed at corpus ids via source_file --------
    prov_id_by_sf = {p["source_file"]: int(p["provider_id"]) for p in providers}
    hire_id_by_sf = {h["source_file"]: h["hire_id"] for h in hirers}
    sf_by_pid = {p["provider_id"]: p.get("source_file") for p in ps}
    sf_by_gid = {h["gig_id"]: h.get("source_file") for h in hs}

    grades, drop = {}, Counter()
    for r in csv.DictReader((N / "gig_grades.csv").open(encoding="utf-8-sig")):
        hid = hire_id_by_sf.get(sf_by_gid.get(r["gig_id"]))
        pid = prov_id_by_sf.get(sf_by_pid.get(r["provider_id"]))
        if hid is None:
            drop["gig not in corpus"] += 1
            continue
        if pid is None:
            drop["provider not in corpus"] += 1
            continue
        grades.setdefault(hid, {})[str(pid)] = int(r["grade"])
    gt = {h: {p: round(v * 100 / 3) for p, v in row.items()} for h, row in grades.items()}

    (out / "hirers.json").write_text(json.dumps(hirers, indent=1), encoding="utf-8")
    (out / "providers.json").write_text(json.dumps(providers, indent=1), encoding="utf-8")
    (out / "llm_judgments_merged.json").write_text(json.dumps(grades, indent=1), encoding="utf-8")
    (out / "ground_truth_llm.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")

    man = sum(1 for p in providers if str(p.get("source_file", "")).startswith("synthetic-provider__"))
    print(f"\n-> {out}/")
    print(f"  queries   {len(hirers)}  ({added} crafted + {len(hirers) - added} extracted)")
    print(f"  providers {len(providers)}  ({len(providers) - man} extracted + {man} manufactured)")
    print(f"  labelled  {sum(len(v) for v in grades.values())} pairs over {len(grades)} queries"
          f"  grades {dict(sorted(Counter(v for r in grades.values() for v in r.values()).items()))}")
    print(f"  dropped from gig_grades.csv: {dict(drop) or 'none'}")


if __name__ == "__main__":
    main()
