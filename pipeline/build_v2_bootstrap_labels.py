"""
BOOTSTRAP labels for the v2 corpus, derived from the manufactured-provider manifest.

====================  THESE ARE NOT JUDGED RELEVANCE LABELS  ====================

`aimed_grade` in data/new/manufactured_providers.txt is the INTENT a profile was
generated with -- 'direct' aimed at grade 3, 'adjacent' at grade 2 -- not a grade
produced by reading the (gig, provider) pair. Each profile was written *against its
gig's own wording*, so it is a strong match by construction.

Consequence: a model trained on these labels learns "find the profile crafted to fit
this gig", which it would score highly on anyway. Any metric computed against them
measures that, NOT "find the provider a hirer would actually hire". Treat every
number from this model as a HARNESS CHECK and never quote it as a result.

They exist only because the corpus otherwise contains zero labelled pairs, so
without them nothing downstream can run at all. Replace both files the moment the
real rubric_0_3.v2 grading pass over the v2 pools lands.

Also note the second, quieter problem: only the 423 target gigs have any positive.
Everything else judged-irrelevant means every unjudged real provider is scored as a
negative, which is the label noise features.py warns about for `--label-policy all`.

Produces the two files features.py expects:
  data_v2/llm_judgments_merged.json   {hire_id: {provider_id: grade 0-3}}    (training)
  data_v2/ground_truth_llm.json       {hire_id: {provider_id: 0/33/67/100}}  (evaluation)

Usage: python build_v2_bootstrap_labels.py --data-dir data/new --tag data_v2
"""
import argparse
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent


def read_manifest(path: Path) -> list[dict]:
    """The manifest is TSV with '#' comment lines and one header row."""
    lines = [l.rstrip("\n") for l in path.open(encoding="utf-8")]
    body = [l for l in lines if l and not l.startswith("#")]
    if not body:
        raise SystemExit(f"{path} has no data rows")
    header = body[0].split("\t")
    return [dict(zip(header, row.split("\t"))) for row in body[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/new")
    ap.add_argument("--tag", default="data_v2")
    args = ap.parse_args()

    src, out = BASE / args.data_dir, BASE / args.tag
    records = read_manifest(src / "manufactured_providers.txt")

    providers = json.loads((out / "providers.json").read_text(encoding="utf-8"))
    hirers = json.loads((out / "hirers.json").read_text(encoding="utf-8"))
    pid_by_file = {p["source_file"]: int(p["provider_id"]) for p in providers}
    valid_hires = {h["hire_id"] for h in hirers}

    grades, skipped = {}, Counter()
    for r in records:
        hire_id, src_file = r["target_hirer_ref"], r["source_file"]
        if hire_id not in valid_hires:
            skipped["target gig not in corpus"] += 1
            continue
        if src_file not in pid_by_file:
            skipped["manufactured provider not in corpus"] += 1
            continue
        grade = int(r["aimed_grade"])
        grades.setdefault(hire_id, {})[str(pid_by_file[src_file])] = grade

    ground_truth = {h: {pid: round(g * 100 / 3) for pid, g in row.items()}
                    for h, row in grades.items()}

    (out / "llm_judgments_merged.json").write_text(json.dumps(grades, indent=2), encoding="utf-8")
    (out / "ground_truth_llm.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    dist = Counter(g for row in grades.values() for g in row.values())
    print(f"manifest rows     : {len(records)}")
    print(f"target gigs       : {len(grades)} of {len(hirers)} in the corpus")
    print(f"labelled pairs    : {sum(len(v) for v in grades.values())}   grades: {dict(sorted(dist.items()))}")
    print(f"skipped           : {dict(skipped) or 'none'}")
    print(f"gigs with no label: {len(hirers) - len(grades)}  (all candidates will be 0 -> no gradient, harmless)")
    print(f"\nwrote {out}/llm_judgments_merged.json and {out}/ground_truth_llm.json")
    print("\n*** BOOTSTRAP LABELS FROM `aimed_grade` -- NOT JUDGED RELEVANCE. ***")
    print("*** Any metric from this model is a harness check, not a result. ***")


if __name__ == "__main__":
    main()
