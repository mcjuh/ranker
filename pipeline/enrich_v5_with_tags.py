"""
Graft the SkillsFuture taxonomy onto data_set_v5.

v5 (kohtw's data_sat_v5) dropped the taxonomy, keeping only a single `industry`
label -- which measured as a dead end: a 36-value, massively-tied channel that cost
-0.09 NDCG@10 when fused into recall.

The mcjuh drop still carries the full sf_v1 tree at 100% coverage and joins to v5
by source_file with zero loss (1106/1106 gigs, 3024/3024 providers). So the real
third channel is recoverable without asking anyone for new data.

Only `tags` is copied. v5's TEXT fields are left untouched, because the v5 grades
were assigned against v5's wording -- swapping the text would invalidate them.

Writes data/data_set_v5_tax/ so the original v5 stays pristine.
"""
import json
from collections import Counter
from pathlib import Path

V5 = Path("data/data_set_v5")
MC = Path("/tmp/mcjuh/data/output")
OUT = Path("data/data_set_v5_tax")

im = json.loads((V5 / "id_map.json").read_text())
sf_of_hid = im["hirers"]                       # v5 id -> source_file
sf_of_pid = im["providers"]

v5_h = json.loads((V5 / "hirers.json").read_text())
v5_p = json.loads((V5 / "providers.json").read_text())
mc_h = json.loads((MC / "hirers.json").read_text())
mc_p = json.loads((MC / "providers.json").read_text())

tags_by_sf_h = {r["source_file"]: r.get("tags") for r in mc_h if r.get("tags")}
tags_by_sf_p = {r["source_file"]: r.get("tags") for r in mc_p if r.get("tags")}

n_h = n_p = 0
for r in v5_h:
    t = tags_by_sf_h.get(sf_of_hid.get(str(r["hire_id"])))
    if t:
        r["tags"] = t
        n_h += 1
for r in v5_p:
    t = tags_by_sf_p.get(sf_of_pid.get(str(r["provider_id"])))
    if t:
        r["tags"] = t
        n_p += 1

OUT.mkdir(parents=True, exist_ok=True)
# text + labels: copy v5 unchanged; only the two record files gain `tags`
for f in ("llm_judgments_merged.json", "ground_truth_llm.json", "id_map.json"):
    (OUT / f).write_text((V5 / f).read_text(), encoding="utf-8")
(OUT / "hirers.json").write_text(json.dumps(v5_h, indent=1), encoding="utf-8")
(OUT / "providers.json").write_text(json.dumps(v5_p, indent=1), encoding="utf-8")


def axis_counts(recs):
    cats, sps, sts, sks = set(), set(), set(), set()
    for r in recs:
        for c in (r.get("tags") or {}).get("categories") or []:
            cats.add(c.get("name"))
            for s in c.get("specialisations") or []:
                sps.add(s.get("name"))
                if s.get("search_tag"):
                    sts.add(s["search_tag"])
                sks.update(s.get("skills") or [])
    return len(cats), len(sps), len(sts), len(sks)


ch, sh, sth, skh = axis_counts(v5_h)
cp, sp, stp, skp = axis_counts(v5_p)
print(f"enriched -> {OUT}/")
print(f"  gigs with tags      : {n_h}/{len(v5_h)}")
print(f"  providers with tags : {n_p}/{len(v5_p)}")
print()
print("  distinct vocabulary on each axis (gig / provider):")
print(f"    categories      {ch:>5} / {cp}")
print(f"    specialisations {sh:>5} / {sp}")
print(f"    search_tags     {sth:>5} / {stp}")
print(f"    skills          {skh:>5} / {skp}")
