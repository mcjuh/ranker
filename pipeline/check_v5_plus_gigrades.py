"""
Should gig_grades.csv (mcjuh data/output) be combined with data_set_v5?

The tempting reason to: +900 labelled pairs. The reason to hesitate: the two files
come from DIFFERENT labellers -- the old rubric weights budget/seniority/
availability heavily (only 0.67% of pairs reach grade>=2), v5's weights them down
(61.5% reach grade>=2). If the same pair scores differently under the two, merging
them teaches the model two contradictory definitions of relevance.

This measures it: map both onto v5 ids via source_file, then compare the grades for
pairs graded by BOTH.
"""
import csv
import json
from collections import Counter
from pathlib import Path

V5 = Path("data/data_set_v5")
MC = Path("/tmp/mcjuh/data/output")

im = json.loads((V5 / "id_map.json").read_text())
sf_to_hid = {sf: hid for hid, sf in im["hirers"].items()}
sf_to_pid = {sf: pid for pid, sf in im["providers"].items()}

hs = json.loads((MC / "platform_sample/hirers_sample.json").read_text())
ps = json.loads((MC / "platform_sample/providers_sample.json").read_text())
grades = list(csv.DictReader((MC / "gig_grades.csv").open(encoding="utf-8-sig")))

print("=== 1. can the pilot records be mapped onto v5 ids? ===")
gig_map = {h["gig_id"]: sf_to_hid.get(h.get("source_file")) for h in hs}
prov_map = {p["provider_id"]: sf_to_pid.get(p.get("source_file")) for p in ps}
print(f"  sample gigs      mapped to a v5 hire_id : {sum(1 for v in gig_map.values() if v)}/{len(gig_map)}")
print(f"  sample providers mapped to a v5 provider: {sum(1 for v in prov_map.values() if v)}/{len(prov_map)}")

pairs = {}
for r in grades:
    g, p = gig_map.get(r["gig_id"]), prov_map.get(r["provider_id"])
    if g and p:
        pairs[(str(g), str(p))] = int(r["grade"])
print(f"  gig_grades pairs mappable to v5        : {len(pairs)}/{len(grades)}")

print("\n=== 2. are those pairs ALSO graded in v5? ===")
v5 = json.loads((V5 / "llm_judgments_merged.json").read_text())
both = {k: (old, v5[k[0]][k[1]]) for k, old in pairs.items() if k[0] in v5 and k[1] in v5[k[0]]}
print(f"  graded in BOTH files: {len(both)}/{len(pairs)}")

if both:
    print("\n=== 3. do the two labellers AGREE? ===")
    old_g = [o for o, _ in both.values()]
    new_g = [n for _, n in both.values()]
    print(f"  old rubric: {dict(sorted(Counter(old_g).items()))}")
    print(f"  v5  rubric: {dict(sorted(Counter(new_g).items()))}")
    same = sum(1 for o, n in both.values() if o == n)
    print(f"  exact agreement  : {same}/{len(both)} = {same/len(both):.1%}")
    # the binary decision the metrics actually use: grade >= 2
    both_rel = sum(1 for o, n in both.values() if o >= 2 and n >= 2)
    old_only = sum(1 for o, n in both.values() if o >= 2 and n < 2)
    new_only = sum(1 for o, n in both.values() if o < 2 and n >= 2)
    neither = sum(1 for o, n in both.values() if o < 2 and n < 2)
    print(f"  relevance (grade>=2) agreement: both {both_rel}, old-only {old_only}, v5-only {new_only}, neither {neither}")
    if len(both) > 1:
        import numpy as np
        r = np.corrcoef(old_g, new_g)[0, 1]
        print(f"  Pearson correlation of grades: {r:+.3f}")
    print("\n  confusion (rows = old grade, cols = v5 grade):")
    M = [[0] * 4 for _ in range(4)]
    for o, n in both.values():
        M[o][n] += 1
    print("        v5:   0     1     2     3")
    for i in range(4):
        print(f"   old {i}: " + "".join(f"{M[i][j]:>6}" for j in range(4)))

print("\n=== 4. are the manufactured providers already in v5? ===")
lines = [l.rstrip("\n") for l in (MC / "manufactured_providers.txt").open(encoding="utf-8")]
body = [l for l in lines if l and not l.startswith("#")]
hdr = body[0].split("\t")
msf = {dict(zip(hdr, r.split("\t")))["source_file"] for r in body[1:]}
print(f"  manufactured providers in v5: {len(msf & set(sf_to_pid))}/{len(msf)}")
