"""Is gig_grades.csv positionally mappable onto the v2 corpus?

Hypothesis H1: G0NN/H0NN -> row NN of data/new/hirers.csv, P0NN -> row NN of
data/new/providers.csv.

If H1 holds, the grades must carry CONTENT signal: a graded-relevant pair should
be more similar (taxonomy overlap, TF-IDF) than a grade-0 pair. If the ids are
arbitrary placeholders, grades are noise w.r.t. content and the difference
collapses to zero.

Test: observed statistic = mean sim(grade>=1) - mean sim(grade==0).
Null: permute the gig<->hirer assignment (keeping grades fixed), 500 draws.
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from taxonomy import record_tags, taxonomy_scores

N = Path("data/new")
grades = list(csv.DictReader((N / "gig_grades.csv").open(encoding="utf-8-sig")))
hirers = list(csv.DictReader((N / "hirers.csv").open(encoding="utf-8-sig")))
providers = list(csv.DictReader((N / "providers.csv").open(encoding="utf-8-sig")))

print(f"grades {len(grades)} rows | hirers {len(hirers)} | providers {len(providers)}")

gi = {f"G{i:03d}": i - 1 for i in range(1, 31)}
pi = {f"P{i:03d}": i - 1 for i in range(1, 31)}

rows = []
for r in grades:
    i, j = gi[r["gig_id"]], pi[r["provider_id"]]
    rows.append({"i": i, "j": j, "grade": int(r["grade"])})

H_TEXT = lambda h: " ".join(str(h.get(k) or "") for k in ("hire_title", "hire_description"))
P_TEXT = lambda p: " ".join(str(p.get(k) or "") for k in
                            ("about_title", "about_description", "services_offered_title",
                             "services_offered_description", "relevant_experience"))

# --- similarity matrices, computed once for every (hirer, provider) pair used ---
h_vec = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit_transform(
    [H_TEXT(h) for h in hirers] + [P_TEXT(p) for p in providers])
n_h = len(hirers)
tfidf = cosine_similarity(h_vec[:n_h], h_vec[n_h:])

htags = [record_tags({"tags_json": h.get("tags_json"), "category": h.get("category"),
                      "specialisation": h.get("specialisation"), "skills": h.get("skills")})
         for h in hirers]
ptags = [record_tags({"tags_json": p.get("tags_json"), "category": p.get("category"),
                      "specialisation": p.get("specialisation"), "skills": p.get("skills")})
         for p in providers]
tax = np.zeros((n_h, len(providers)))
for i in range(30):
    for j in range(30):
        tax[i, j] = taxonomy_scores(htags[i], ptags[j])["tax_search_tag"]


def stat(mapping, sim):
    """mean sim over grade>=1 minus mean over grade==0, under a gig->hirer mapping."""
    rel = [sim[mapping[r["i"]], r["j"]] for r in rows if r["grade"] >= 1]
    irr = [sim[mapping[r["i"]], r["j"]] for r in rows if r["grade"] == 0]
    return float(np.mean(rel) - np.mean(irr)), float(np.mean(rel)), float(np.mean(irr))


identity = list(range(30))
for name, sim in (("taxonomy search_tag", tax), ("TF-IDF cosine", tfidf)):
    obs, mr, mi = stat(identity, sim)
    rng = random.Random(7)
    null = []
    for _ in range(500):
        perm = identity[:]
        rng.shuffle(perm)
        null.append(stat(perm, sim)[0])
    null = np.array(null)
    p = float((null >= obs).mean())
    print(f"\n{name}")
    print(f"  graded-relevant mean {mr:.4f}  vs  grade-0 mean {mi:.4f}   diff {obs:+.4f}")
    print(f"  permutation null: mean {null.mean():+.4f}  sd {null.std():.4f}  "
          f"p(null >= observed) = {p:.3f}")

print("\n--- how many rows even carry signal? ---")
print(f"  grade>=1 rows: {sum(1 for r in rows if r['grade'] >= 1)} of {len(rows)}")
print(f"  distinct (hirer,provider) pairs under H1: "
      f"{len({(r['i'], r['j']) for r in rows})}")
