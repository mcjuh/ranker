"""
SkillsFuture tag-match features for the Stage-2 ranker.

Both sides of data_set_v5_tax carry the sf_v1 tree (category -> specialisation ->
skills), grafted on by enrich_v5_with_tags.py. These features say how far a
provider's tags reach into the gig's: a ranker feature, NOT a recall channel (a third
RRF channel was measured and loses; ablation.md §3 finding 3).

  tag_level      0 none, 1 shared category, 2 shared specialisation,
                 3 shared specialisation AND a shared skill. The strongest single
                 feature: P(grade>=2) climbs 27% -> 37% -> 52% -> 74% across the levels
                 even at RRF ranks 31-50, i.e. it carries signal text retrieval misses.
  has_tags_gig   1 if the gig / provider has any tags. Without these an untagged
  has_tags_prov  record (87 gigs, 252 providers) is indistinguishable from a mismatch.
  cat_any        any shared category
  spec_cov       share of the gig's (category, specialisation) pairs the provider holds
  skill_cov      share of the gig's skills the provider holds (exact string match)
  skill_idf_cov  skill_cov weighted by skill rarity over the provider catalogue, so a
                 shared niche skill counts for more than "Stakeholder Management"
  skill_soft     mean over the gig's skills of the best cosine to any provider skill
                 (mxbai-embed-large on the skill NAMES). Exact match misses near-synonyms
                 ("Data and Statistical Analytics" vs "Data Analytics"): pairs with only a
                 near-match grade between exact-match and no-match pairs.

Category and specialisation are matched exactly: they are a closed vocabulary
(37 / 144 values) and the name-level fuzzy variants were measured to add nothing.

Industry is deliberately absent: a provider's `industry` is always one of its tag
categories, so it duplicates cat_any (they agree on 89% of candidate pairs).

Writes features_data/<corpus>/tag_features.csv for every row of candidates_top50.csv.
Run it as its own process -- it loads torch, and torch + xgboost in one process
segfaults on macOS (OpenMP runtime clash); rerank_ltr.py only reads the CSV.

    python pipeline/tag_features.py --data-dir data/data_set_v5_tax
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
TAG_FEATURES = ["tag_level", "has_tags_gig", "has_tags_prov", "cat_any", "spec_cov",
                "skill_cov", "skill_idf_cov", "skill_soft"]
SKILL_MODEL = "mixedbread-ai/mxbai-embed-large-v1"


def tag_axes(record: dict):
    """(categories, (category, specialisation) pairs, skills) from a record's sf_v1 tags.
    Specialisations are keyed with their category: 7 names recur under several
    categories (e.g. 'Project Management') and mean different work in each."""
    cats = (record.get("tags") or {}).get("categories") or []
    return ({c["name"] for c in cats},
            {(c["name"], s["name"]) for c in cats for s in c.get("specialisations") or []},
            {k for c in cats for s in c.get("specialisations") or [] for k in s.get("skills") or []})


class TagMatcher:
    def __init__(self, hirers: list, providers: list, skill_vectors: dict | None = None):
        self.gig = {r["hire_id"]: tag_axes(r) for r in hirers}
        self.prov = {r["provider_id"]: tag_axes(r) for r in providers}
        n = len(providers)
        df = Counter(k for _, _, skills in self.prov.values() for k in skills)
        self.idf = {k: math.log(n / (1 + v)) for k, v in df.items()}
        self.idf_unseen = math.log(n)          # a gig skill no provider holds: maximally rare
        self.vec = skill_vectors or {}

    def features(self, hire_id: int, provider_id: int) -> dict:
        gc, gs, gk = self.gig[hire_id]
        pc, ps, pk = self.prov[provider_id]
        shared_skills = gk & pk
        if shared_skills and gs & ps:
            level = 3
        elif gs & ps:
            level = 2
        elif gc & pc:
            level = 1
        else:
            level = 0
        soft = 0.0
        if gk and pk and self.vec:
            G = np.stack([self.vec[k] for k in gk])
            P = np.stack([self.vec[k] for k in pk])
            soft = float((G @ P.T).max(axis=1).mean())
        return {
            "tag_level": float(level),
            "has_tags_gig": float(bool(gc)),
            "has_tags_prov": float(bool(pc)),
            "cat_any": float(bool(gc & pc)),
            "spec_cov": len(gs & ps) / len(gs) if gs else 0.0,
            "skill_cov": len(shared_skills) / len(gk) if gk else 0.0,
            "skill_idf_cov": (sum(self.idf.get(k, 0.0) for k in shared_skills)
                              / sum(self.idf.get(k, self.idf_unseen) for k in gk)) if gk else 0.0,
            "skill_soft": soft,
        }


def embed_skills(skills: list, cache: Path) -> dict:
    """Unit-normalised mxbai vectors for the skill names, cached per corpus."""
    if cache.exists():
        data = json.loads(cache.read_text())
        if set(data) >= set(skills):
            return {k: np.asarray(data[k], dtype=np.float32) for k in skills}
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(SKILL_MODEL, device="cpu")
    vecs = model.encode(skills, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    # mxbai loads in float16; rounding a float16 to 6 places overflows it to inf.
    vecs = np.asarray(vecs, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({k: v.round(6).tolist() for k, v in zip(skills, vecs)}))
    return {k: v.astype(np.float32) for k, v in zip(skills, vecs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/data_set_v5_tax",
                    help="a corpus whose hirers.json / providers.json carry sf_v1 `tags`")
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    data_dir = BASE / args.data_dir
    feat_dir = BASE / args.data_dir.replace("data/", "features_data/", 1)
    cache_dir = BASE / args.data_dir.replace("data/", "cache_data/", 1)
    hirers = json.loads((data_dir / "hirers.json").read_text(encoding="utf-8"))
    providers = json.loads((data_dir / "providers.json").read_text(encoding="utf-8"))
    skills = sorted({k for r in hirers + providers for k in tag_axes(r)[2]})
    if not skills:
        raise SystemExit(f"{data_dir} has no sf_v1 tags -- build it with enrich_v5_with_tags.py")

    matcher = TagMatcher(hirers, providers, embed_skills(skills, cache_dir / "skill_vectors.json"))
    with (feat_dir / f"candidates_top{args.top_k}.csv").open(newline="") as f:
        pairs = [(int(r["hire_id"]), int(r["provider_id"])) for r in csv.DictReader(f)]

    out = feat_dir / "tag_features.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hire_id", "provider_id"] + TAG_FEATURES)
        for h, p in pairs:
            fx = matcher.features(h, p)
            w.writerow([h, p] + [round(fx[k], 6) for k in TAG_FEATURES])
    print(f"wrote {len(pairs)} rows x {len(TAG_FEATURES)} tag features to {out} "
          f"({len(skills)} distinct skills embedded)")


if __name__ == "__main__":
    main()
