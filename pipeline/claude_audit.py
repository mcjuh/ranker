"""
Blind audit of the qwen grader: a second, independent grader (Claude, in-session) regrades a small
stratified sample of (gig, provider) pairs, and the two sets of grades are compared.

Why: every label in this repo comes from one model (qwen3.8:27b), and the tag-channel pairs came from a
reconstruction of its prompt that is more lenient (labeller.py calibration). This asks whether the
verdicts on the channel survive a grader from a different model family.

Modes (run from the repo root):
    python pipeline/claude_audit.py sample            # draws the sample; writes audit/sample_key.json
    python pipeline/claude_audit.py show --part 1     # prints items WITHOUT any qwen information
    python pipeline/claude_audit.py score             # joins audit/claude_grades.jsonl to the key and compares

Blinding: `show` prints only the gig, the provider and their terms, in shuffled order, under neutral ids.
The key (stratum, qwen grade, qwen probabilities) is written to a separate file that the grader must not
read before its grades are saved to audit/claude_grades.jsonl.
Grades are one JSON object per line: {"item": "P01", "grade": 0-3, "note": "..."}.
"""
import argparse
import json
import random
import sys
import types
from pathlib import Path

# labeller imports features, which imports the dense retriever and so torch. This script only needs the
# prompt-formatting helpers, so the encoder module is stubbed rather than installed.
sys.modules.setdefault("sentence_transformers", types.SimpleNamespace(SentenceTransformer=None))

from corpus import hirer_text, provider_text  # noqa: E402
from labeller import hirer_terms, provider_terms  # noqa: E402

BASE = Path(__file__).parent
DATA = BASE / "data_sat"
AUDIT = BASE / "audit"
SEED = 7
PER_PAGE = 20

# stratum -> (source file, how many, predicate on the qwen probabilities / grade)
STRATA = {
    "borderline": ("judgments.jsonl", 16, lambda r: 0.35 < sum(r["probs"][2:]) < 0.65),
    "confident_pos": ("judgments.jsonl", 6, lambda r: sum(r["probs"][2:]) > 0.90),
    "confident_neg": ("judgments.jsonl", 6, lambda r: sum(r["probs"][2:]) < 0.05),
    "tag_only_pos": ("judgments_tag.jsonl", 8, lambda r: r["grade"] >= 2),
    "tag_only_neg": ("judgments_tag.jsonl", 4, lambda r: r["grade"] < 2),
}


def _records(name: str) -> list[dict]:
    path = DATA / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def draw_sample() -> list[dict]:
    rng = random.Random(SEED)
    original = _records("judgments.jsonl")
    seen = {(r["hire_id"], r["provider_id"]) for r in original}
    pools = {"judgments.jsonl": [r for r in original if r["status"] == "ok"],
             "judgments_tag.jsonl": [r for r in _records("judgments_tag.jsonl")
                                     if r["status"] == "ok" and (r["hire_id"], r["provider_id"]) not in seen]}
    chosen, used = [], set()
    for stratum, (source, n, keep) in STRATA.items():
        candidates = [r for r in pools[source] if keep(r) and (r["hire_id"], r["provider_id"]) not in used]
        # at most one pair per gig per sample, so 40 items span 40 different gigs
        rng.shuffle(candidates)
        taken_gigs = {c["hire_id"] for c in chosen}
        picked = []
        for r in candidates:
            if r["hire_id"] in taken_gigs:
                continue
            picked.append(r)
            taken_gigs.add(r["hire_id"])
            if len(picked) == n:
                break
        if len(picked) < n:
            raise SystemExit(f"stratum {stratum}: only {len(picked)} of {n} available")
        for r in picked:
            used.add((r["hire_id"], r["provider_id"]))
            chosen.append({"hire_id": r["hire_id"], "provider_id": r["provider_id"], "stratum": stratum,
                           "qwen_grade": r["grade"], "qwen_probs": r["probs"],
                           "qwen_prompt": r["prompt_version"]})
    rng.shuffle(chosen)
    for i, item in enumerate(chosen, start=1):
        item["item"] = f"P{i:02d}"
    return chosen


def _load_entities():
    hirers = {str(h["hire_id"]): h for h in json.loads((DATA / "hirers.json").read_text(encoding="utf-8"))}
    providers = {str(p["provider_id"]): p
                 for p in json.loads((DATA / "providers.json").read_text(encoding="utf-8"))}
    return hirers, providers


def cmd_sample(_args):
    sample = draw_sample()
    AUDIT.mkdir(exist_ok=True)
    (AUDIT / "sample_key.json").write_text(json.dumps(sample, indent=1), encoding="utf-8")
    counts = {}
    for s in sample:
        counts[s["stratum"]] = counts.get(s["stratum"], 0) + 1
    print(f"wrote audit/sample_key.json: {len(sample)} items, seed {SEED} (strata counts withheld from `show`)")


def cmd_show(args):
    sample = json.loads((AUDIT / "sample_key.json").read_text(encoding="utf-8"))
    hirers, providers = _load_entities()
    page = sample[(args.part - 1) * PER_PAGE: args.part * PER_PAGE]
    for it in page:
        h, p = hirers[str(it["hire_id"])], providers[str(it["provider_id"])]
        print(f"##### {it['item']}")
        print("=== GIG ===")
        print("\n".join([hirer_text(h), *hirer_terms(h)]))
        print("=== PROVIDER PROFILE ===")
        print("\n".join([provider_text(p), *provider_terms(p)]))
        print()


def _kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def cmd_score(_args):
    key = {k["item"]: k for k in json.loads((AUDIT / "sample_key.json").read_text(encoding="utf-8"))}
    grades = {}
    for line in (AUDIT / "claude_grades.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            g = json.loads(line)
            grades[g["item"]] = g["grade"]
    missing = sorted(set(key) - set(grades))
    if missing:
        raise SystemExit(f"ungraded items: {missing}")
    rows = [(key[i], grades[i]) for i in sorted(key)]

    def report(name, subset):
        if not subset:
            return
        q = [r["qwen_grade"] for r, _ in subset]
        c = [g for _, g in subset]
        n = len(subset)
        exact = sum(x == y for x, y in zip(q, c)) / n
        within1 = sum(abs(x - y) <= 1 for x, y in zip(q, c)) / n
        qpos, cpos = [int(x >= 2) for x in q], [int(y >= 2) for y in c]
        agree = sum(x == y for x, y in zip(qpos, cpos)) / n
        k = _kappa(qpos, cpos)
        print(f"{name:<16} n={n:<3} exact {exact:.2f}  within1 {within1:.2f}  >=2 agree {agree:.2f}  "
              f"kappa(>=2) {'n/a' if k is None else f'{k:.2f}'}  qwen>=2 {sum(qpos)}  claude>=2 {sum(cpos)}")

    report("ALL", rows)
    for stratum in STRATA:
        report(stratum, [r for r in rows if r[0]["stratum"] == stratum])
    print("\nconfusion (rows qwen, cols claude):")
    m = [[0] * 4 for _ in range(4)]
    for r, g in rows:
        m[r["qwen_grade"]][g] += 1
    for i, row in enumerate(m):
        print(f"  {i}: {row}")
    print("\ndisagreements on the >=2 line:")
    for r, g in rows:
        if (r["qwen_grade"] >= 2) != (g >= 2):
            print(f"  {r['item']} {r['stratum']:<14} qwen {r['qwen_grade']} (P>=2 {sum(r['qwen_probs'][2:]):.2f}) "
                  f"claude {g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample").set_defaults(fn=cmd_sample)
    sh = sub.add_parser("show")
    sh.add_argument("--part", type=int, choices=[1, 2], required=True)
    sh.set_defaults(fn=cmd_show)
    sub.add_parser("score").set_defaults(fn=cmd_score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
