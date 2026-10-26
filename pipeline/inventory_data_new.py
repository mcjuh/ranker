"""
Inventory data/new: what is usable, what overlaps, what conflicts.

The goal is one corpus that uses everything. The blockers are duplicate entities
(platform_sample is an anonymised re-rendering of corpus records, so the same
source_file appears twice with different text) and stale files (the 130-gig
synthetic set was superseded by the v2 corpus).
"""
import csv
import json
from collections import Counter
from pathlib import Path

N = Path("data/new")


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def cload(p):
    return list(csv.DictReader(Path(p).open(encoding="utf-8-sig")))


print("=" * 72)
print("FILES IN data/new")
print("=" * 72)
for p in sorted(N.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(N)!s:<52} {p.stat().st_size:>10,}")

print("\n" + "=" * 72)
print("1. MAIN CORPUS")
print("=" * 72)
hirers, providers = jload(N / "hirers.json"), jload(N / "providers.json")
print(f"  hirers.json    {len(hirers):>5}   providers.json  {len(providers):>5}")
print(f"  hirer ids      {hirers[0]['hirer_ref']}, {hirers[1]['hirer_ref']} ...")
man = [p for p in providers if str(p.get("source_file", "")).startswith("synthetic-provider__")]
print(f"  of which manufactured: {len(man)}   real: {len(providers) - len(man)}")

print("\n" + "=" * 72)
print("2. CRAFTED GIGS  (30 hand-written, schema v2)")
print("=" * 72)
crafted = jload(N / "crafted_gigs/hirers_crafted.json")
csf = {c["source_file"] for c in crafted}
hrefs = {c["hirer_ref"] for c in crafted}
hsf = {h["source_file"] for h in hirers}
hhrefs = {h["hirer_ref"] for h in hirers}
print(f"  records        {len(crafted)}")
print(f"  source_file collisions with corpus : {len(csf & hsf)}")
print(f"  hirer_ref collisions with corpus   : {len(hrefs & hhrefs)}")
print(f"  -> {'NEW queries, no collision' if not (csf & hsf) else 'COLLIDES'}")
print(f"  tagged: {sum(1 for c in crafted if c.get('tags'))}/{len(crafted)}")

print("\n" + "=" * 72)
print("3. PLATFORM SAMPLE  (30x30, the ONLY graded records)")
print("=" * 72)
hs, ps = jload(N / "platform_sample/hirers_sample.json"), jload(N / "platform_sample/providers_sample.json")
hssf = {h["source_file"] for h in hs}
pssf = {p["source_file"] for p in ps}
psf = {p["source_file"] for p in providers}
print(f"  sample gigs {len(hs)}   sample providers {len(ps)}")
print(f"  sample gig source_file also in corpus     : {len(hssf & hsf)}/{len(hs)}")
print(f"  sample prov source_file also in corpus    : {len(pssf & psf)}/{len(ps)}")
print("  -> SAME ENTITIES as corpus, different text (anonymised rewrite)")
print(f"  sample provs NOT in corpus (addable new)  : {len(pssf - psf)}")
print(f"  sample gigs  NOT in corpus (addable new)  : {len(hssf - hsf)}")

print("\n" + "=" * 72)
print("4. GRADES")
print("=" * 72)
g = cload(N / "gig_grades.csv")
print(f"  gig_grades.csv  {len(g)} pairs, {len({r['gig_id'] for r in g})} gigs, "
      f"grades {dict(sorted(Counter(int(r['grade']) for r in g).items()))}")
# can they be re-pointed at corpus records via source_file?
psf_by_pid = {p["provider_id"]: p["source_file"] for p in ps}
hsf_by_gid = {h["gig_id"]: h["source_file"] for h in hs}
prov_id_by_sf = {p["source_file"]: p["provider_id"] for p in providers}
hire_id_by_sf = {h["source_file"]: h["hire_id"] for h in hirers}
mappable = sum(1 for r in g
               if psf_by_pid[r["provider_id"]] in prov_id_by_sf
               and hsf_by_gid[r["gig_id"]] in hire_id_by_sf)
print(f"  pairs re-pointable onto corpus ids via source_file : {mappable}/{len(g)}")
print(f"  -> {len({r['gig_id'] for r in g if hsf_by_gid[r['gig_id']] in hire_id_by_sf})} corpus gigs would gain labels")

print("\n" + "=" * 72)
print("5. OLD SYNTHETIC (superseded)")
print("=" * 72)
for f in ("llm_judgments_merged.json", "ground_truth_llm.json", "ground_truth.json"):
    d = jload(N / f)
    print(f"  {f:<32} {len(d)} queries, keys {list(d)[:3]}")
old_h = jload(N / "_hirers_with_taxonomy.json")
print(f"  _hirers_with_taxonomy.json       {len(old_h)} records")
print(f"  -> 130-gig synthetic set, replaced by the v2 corpus; ids do not join")

print("\n" + "=" * 72)
print("6. WHAT A UNION CORPUS WOULD LOOK LIKE")
print("=" * 72)
print(f"  queries   : {len(hirers)} corpus + {len(crafted)} crafted = {len(hirers) + len(crafted)}")
print(f"  providers : {len(providers)} corpus (incl. {len(man)} manufactured)"
      f" + {len(pssf - psf)} sample-only = {len(providers) + len(pssf - psf)}")
print(f"  labels    : {mappable} pairs over "
      f"{len({r['gig_id'] for r in g if hsf_by_gid[r['gig_id']] in hire_id_by_sf})} queries")
