"""
Can the data/output material be trained on?

Answers two separate questions, which the first test conflated:
  A. does gig_grades.csv map to anything?              (id join)
  B. do its records live in the v2 retrieval corpus?   (so a ranker could rank them)
  C. is the grading real signal or noise?              (permutation test)
"""
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

N = Path("data/new")
S, C = N / "platform_sample", N / "crafted_gigs"


def tags_of(x):
    """search_tags arrives as a list in .json but a JSON string in .csv."""
    t = x.get("search_tags")
    if isinstance(t, str):
        try:
            t = json.loads(t)
        except Exception:
            t = [p for p in t.strip("[]").replace("'", "").split(",") if p.strip()]
    return {str(v).split(" (")[0].strip().lower() for v in (t or [])}


hs = {x["gig_id"]: x for x in json.loads((S / "hirers_sample.json").read_text())}
ps = {x["provider_id"]: x for x in json.loads((S / "providers_sample.json").read_text())}
grades = list(csv.DictReader((N / "gig_grades.csv").open(encoding="utf-8-sig")))

print("=== A. does gig_grades.csv map? ===")
hirer_ids = {x["hirer_id"] for x in hs.values()}
print(f"  gig_id       : {sum(r['gig_id'] in hs for r in grades)}/{len(grades)}")
print(f"  hirer_id     : {sum(r['hirer_id'] in hirer_ids for r in grades)}/{len(grades)}")
print(f"  provider_id  : {sum(r['provider_id'] in ps for r in grades)}/{len(grades)}")

print("\n=== B. do the sample records live in the v2 retrieval corpus? ===")
v2h = json.loads(Path("data_v2/hirers.json").read_text()) if Path("data_v2/hirers.json").exists() else []
v2p = json.loads(Path("data_v2/providers.json").read_text()) if Path("data_v2/providers.json").exists() else []
if v2h:
    vh = {x.get("source_file") for x in v2h}
    vp = {x.get("source_file") for x in v2p}
    print(f"  sample hirers  in corpus: {sum(x.get('source_file') in vh for x in hs.values())}/{len(hs)}")
    print(f"  sample provs   in corpus: {sum(x.get('source_file') in vp for x in ps.values())}/{len(ps)}")
    print(f"  corpus hirers vs sample distinct: {len(vh)} vs {len({x.get('source_file') for x in hs.values()})}")
else:
    print("  data_v2/ absent (being rebuilt) - skipped")

crafted = json.loads((C / "hirers_crafted.json").read_text())
print(f"  crafted gigs            : {len(crafted)}  (hirer_ref e.g. {crafted[0]['hirer_ref']})")
print(f"  crafted in corpus       : {sum(x['source_file'] in {y.get('source_file') for y in v2h} for x in crafted)}"
      if v2h else "  crafted in corpus       : n/a")

print("\n=== C. is the grading real signal? ===")
H = lambda x: " ".join(str(x.get(k) or "") for k in ("gig_title", "short_gig_description"))
P = lambda x: " ".join(str(x.get(k) or "") for k in ("title", "about_headline", "about_bio"))
gl = [hs[f"G{i:03d}"] for i in range(1, 31)]
pl = [ps[f"P{i:03d}"] for i in range(1, 31)]
tf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform(
    [H(x) for x in gl] + [P(x) for x in pl])
M = cosine_similarity(tf[:30], tf[30:])
gi = {f"G{i:03d}": i - 1 for i in range(1, 31)}
pi = {f"P{i:03d}": i - 1 for i in range(1, 31)}

# tag-overlap matrix, built from the JSON records (authoritative)
T = np.zeros((30, 30))
for i in range(30):
    for j in range(30):
        T[i, j] = len(tags_of(gl[i]) & tags_of(pl[j]))

pairs = [(gi[r["gig_id"]], pi[r["provider_id"]], int(r["grade"])) for r in grades]
print(f"  grade distribution: {dict(sorted(Counter(g for *_, g in pairs).items()))}")
print(f"  rows with grade>=1 : {sum(1 for *_, g in pairs if g >= 1)}/{len(pairs)}")


def diff(sim, idx=None):
    idx = idx or list(range(30))
    rel = [sim[idx[i], j] for i, j, g in pairs if g >= 1]
    irr = [sim[idx[i], j] for i, j, g in pairs if g == 0]
    return float(np.mean(rel) - np.mean(irr))


for name, sim in (("TF-IDF cosine", M), ("tag overlap", T)):
    obs = diff(sim)
    rng = random.Random(11)
    null = []
    for _ in range(1000):
        p = list(range(30))
        rng.shuffle(p)
        null.append(diff(sim, p))
    null = np.array(null)
    p = float((null >= obs).mean())
    print(f"\n  {name}: grade>=1 minus grade-0 = {obs:+.4f}")
    print(f"    permutation null: {null.mean():+.4f} +/- {null.std():.4f}   p = {p:.3f}")

print("\n=== D. what a model would actually have to learn from ===")
pos_gigs = {i for i, j, g in pairs if g >= 1}
print(f"  queries with any positive : {len(pos_gigs)}/30")
print(f"  positives total           : {sum(1 for *_, g in pairs if g >= 1)}")
print(f"  grade-3                   : {sum(1 for *_, g in pairs if g == 3)}")
print(f"  candidates per query      : 30, every query")
