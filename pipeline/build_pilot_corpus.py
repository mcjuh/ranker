"""
Build a pilot corpus from data/new/platform_sample + gig_grades.csv.

Why a separate corpus rather than the main one: the grades in gig_grades.csv were
assigned against *these* (anonymised) texts -- the corpus rows they trace back to
by source_file carry different wording. Features must come from the same text the
grader read, so the pilot gets its own data dir.

The upside: the grader covered all 30 providers for all 30 gigs, so every candidate
in the pool is judged. No unjudged-as-negative contamination, which the main corpus
cannot avoid.

30 queries is a smoke-test scale, not a result-generating scale.

Usage: python build_pilot_corpus.py --tag data_pilot
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent
SRC = BASE / "data" / "new" / "platform_sample"
GRADES = BASE / "data" / "new" / "gig_grades.csv"


def split_tag(t: str):
    """'Financial Forensics (Accountancy)' -> ('Financial Forensics', 'Accountancy')."""
    t = str(t).strip()
    if t.endswith(")") and " (" in t:
        spec, cat = t.rsplit(" (", 1)
        return spec.strip(), cat[:-1].strip()
    return t, None


def nest_tags(rec: dict) -> list:
    """Rebuild the v2 nested taxonomy shape from the flattened sample columns."""
    cats: dict[str, set] = {}
    for t in rec.get("search_tags") or []:
        spec, cat = split_tag(t)
        if cat:
            cats.setdefault(cat, set()).add(spec)
    if not cats:
        for c in rec.get("category") or []:
            cats.setdefault(c.get("name") or "Uncategorised", set())
        for s in rec.get("specialisation") or []:
            next(iter(cats.values())).add(s)

    skills_by_spec: dict[str, list] = {}
    for tp in rec.get("technical_proficiency") or []:
        if isinstance(tp, dict):
            for s in tp.get("skills") or []:
                skills_by_spec.setdefault(str(tp.get("category") or ""), []).append(s)

    out = []
    for cat, specs in cats.items():
        entries = []
        for spec in sorted(specs):
            sk = []
            for k, v in skills_by_spec.items():
                if k and (k.lower() in spec.lower() or spec.lower() in k.lower()):
                    sk = v
                    break
            entries.append({"name": spec, "search_tag": f"{spec} ({cat})", "skills": sk})
        out.append({"name": cat, "group": None, "specialisations": entries})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="data_pilot")
    args = ap.parse_args()
    out = BASE / args.tag
    out.mkdir(parents=True, exist_ok=True)

    hs = json.loads((SRC / "hirers_sample.json").read_text(encoding="utf-8"))
    ps = json.loads((SRC / "providers_sample.json").read_text(encoding="utf-8"))
    grades = list(csv.DictReader(GRADES.open(encoding="utf-8-sig")))

    hirers = [{
        "hire_id": h["gig_id"],
        "hire_title": h.get("gig_title") or "",
        "hire_description": h.get("short_gig_description") or "",
        "hire_description_additional_notes": "",
        "duration_weeks_min": None, "duration_weeks_max": None,
        "tags": nest_tags(h),
        "source_file": h.get("source_file"), "hirer_ref": h.get("hirer_id"),
    } for h in hs]

    providers = [{
        "provider_id": int(str(p["provider_id"]).lstrip("P")),
        "about_title": p.get("about_headline") or p.get("title") or "",
        "about_description": p.get("about_bio") or "",
        "services_offered_title": " | ".join(
            s.get("service_title", "") for s in (p.get("services_i_offer") or []) if isinstance(s, dict)),
        "services_offered_description": " ".join(
            s.get("service_detail", "") for s in (p.get("services_i_offer") or []) if isinstance(s, dict)),
        "relevant_experience": " ".join(
            str(a) for a in (p.get("relevant_achievements") or [])),
        "years_experience": None,
        "tags": nest_tags(p),
        "source_file": p.get("source_file"),
    } for p in ps]

    valid = {h["hire_id"] for h in hirers}
    pids = {p["provider_id"] for p in providers}
    judged, dropped = {}, Counter()
    for r in grades:
        g, p = r["gig_id"], int(str(r["provider_id"]).lstrip("P"))
        if g not in valid or p not in pids:
            dropped["pair outside the 30x30 sample"] += 1
            continue
        judged.setdefault(g, {})[str(p)] = int(r["grade"])

    # ground_truth_llm.json uses the 0/33/67/100 scale evaluate.py thresholds at 40
    gt = {g: {p: round(v * 100 / 3) for p, v in row.items()} for g, row in judged.items()}

    (out / "hirers.json").write_text(json.dumps(hirers, indent=1), encoding="utf-8")
    (out / "providers.json").write_text(json.dumps(providers, indent=1), encoding="utf-8")
    (out / "llm_judgments_merged.json").write_text(json.dumps(judged, indent=1), encoding="utf-8")
    (out / "ground_truth_llm.json").write_text(json.dumps(gt, indent=1), encoding="utf-8")

    tagged_h = sum(1 for h in hirers if h["tags"])
    tagged_p = sum(1 for p in providers if p["tags"])
    print(f"pilot corpus -> {out}/")
    print(f"  hirers {len(hirers)} (tagged {tagged_h})   providers {len(providers)} (tagged {tagged_p})")
    print(f"  graded gigs {len(judged)}   graded pairs {sum(len(v) for v in judged.values())}   dropped {dict(dropped) or 'none'}")
    print(f"  grades: {dict(sorted(Counter(v for r in judged.values() for v in r.values()).items()))}")
    print(f"  pairs per gig: {dict(Counter(len(v) for v in judged.values()))}")
    print("\n*** 30 queries -> smoke-test scale. Not a result-generating corpus. ***")


if __name__ == "__main__":
    main()
