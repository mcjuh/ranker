"""
Blind audit of the qwen grader: a second, independent grader (Claude, in-session) regrades a stratified
sample of (gig, provider) pairs, and the two sets of grades are compared.

Why: every label in this repo comes from one model (qwen3.8:27b), and the tag-channel pairs came from a
reconstruction of its prompt that is more lenient (labeller.py calibration). This asks whether the
verdicts on the channel survive a grader from a different model family.

Modes (run from the repo root; --round 1 is the 40-pair pilot, --round 2 the 160-pair follow-up, --round 3 the
adjudication of the pairs the hubness result rests on, see ROUNDS):
    python pipeline/claude_audit.py sample   [--round N]            # draws the sample, writes the key
    python pipeline/claude_audit.py show     [--round N] --part K   # prints items WITHOUT any qwen information
    python pipeline/claude_audit.py score    [--round N]            # joins the grades to the key and compares
    python pipeline/claude_audit.py termcheck                       # rule-based check of qwen's term caps

Blinding: `show` prints only the gig, the provider and their terms, in shuffled order, under neutral ids.
The key (stratum, qwen grade, qwen probabilities) is written to a separate file that the grader must not
read before its grades are saved to audit/claude_grades*.jsonl, one JSON object per line:
{"item": "P01", "grade": 0-3, "note": "..."}.

`termcheck` needs no second grader: it applies the rubric's own definition of a serious term mismatch
(rate > 1.4x the top of the budget, mid vs expert, a start > 30 days late or >= 2 days a week short) and
counts how often a grade >= 2 was given anyway, which the rubric forbids ("1 = ... relevant expertise with
a serious mismatch").
"""
import argparse
import json
import math
import random
import sys
import types
from datetime import timedelta
from pathlib import Path

# labeller imports features, which imports the dense retriever and so torch. This script only needs the
# prompt-formatting helpers, so the encoder module is stubbed rather than installed.
sys.modules.setdefault("sentence_transformers", types.SimpleNamespace(SentenceTransformer=None))

from corpus import hirer_text, provider_text  # noqa: E402
from features import _parse_avail_date  # noqa: E402
from labeller import ANCHOR_DATE, hirer_terms, provider_terms  # noqa: E402

BASE = Path(__file__).parent
DATA = BASE / "data_sat"
AUDIT = BASE / "audit"

# stratum -> (source file, how many, predicate on the qwen record). Round 1 over-samples borderline pairs;
# round 2 draws at random within each qwen grade so rates are unbiased within a stratum, with the original
# grader's pairs as a control for the reproduced prompt's.
ROUNDS = {
    1: {
        "seed": 7, "prefix": "P", "per_page": 20, "key": "sample_key.json", "grades": "claude_grades.jsonl",
        "strata": {
            "borderline": ("judgments.jsonl", 16, lambda r: 0.35 < sum(r["probs"][2:]) < 0.65),
            "confident_pos": ("judgments.jsonl", 6, lambda r: sum(r["probs"][2:]) > 0.90),
            "confident_neg": ("judgments.jsonl", 6, lambda r: sum(r["probs"][2:]) < 0.05),
            "tag_only_pos": ("judgments_tag.jsonl", 8, lambda r: r["grade"] >= 2),
            "tag_only_neg": ("judgments_tag.jsonl", 4, lambda r: r["grade"] < 2),
        },
    },
    2: {
        "seed": 8, "prefix": "Q", "per_page": 40, "key": "sample2_key.json", "grades": "claude_grades2.jsonl",
        "strata": {
            "tag_only_pos": ("judgments_tag.jsonl", 80, lambda r: r["grade"] >= 2),
            "tag_only_neg": ("judgments_tag.jsonl", 40, lambda r: r["grade"] < 2),
            "orig_pos": ("judgments.jsonl", 30, lambda r: r["grade"] >= 2),
            "orig_neg": ("judgments.jsonl", 10, lambda r: r["grade"] < 2),
        },
    },
    3: {
        # Adjudication of what the hubness-corrected channel's gain rests on (TAG_CHANNEL.md section 11). Only repro
        # positives are drawn from the two diff strata: a gain that is pure grader leniency shows up as the entering
        # positives being confirmed less often than the leaving ones. The discordant stratum (original < 2, repro >= 2,
        # the very same pairs) separates the prompt's leniency from the pairs' population; concordant is its control.
        "seed": 9, "prefix": "R", "per_page": 40, "key": "sample3_key.json", "grades": "claude_grades3.jsonl",
        "strata": {
            "diff_in_pos": ("diff_in", 70, lambda r: r["grade"] >= 2),
            "diff_out_pos": ("diff_out", 70, lambda r: r["grade"] >= 2),
            "discordant": ("doubly_graded", 40, lambda r: r["orig_grade"] < 2 <= r["grade"]),
            "concordant_pos": ("doubly_graded", 15, lambda r: r["orig_grade"] >= 2 and r["grade"] >= 2),
        },
    },
}


def _records(name: str) -> list[dict]:
    path = DATA / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _round3_pools(original: list[dict]) -> dict[str, list[dict]]:
    """Reproduction-graded records for the round 3 strata, each with the original grade attached when there is one."""
    repro = {}
    for name in ("judgments_variants.jsonl", "judgments_regrade.jsonl", "judgments_tag.jsonl", "calibration.jsonl"):
        for r in _records(name):
            if r["status"] == "ok":
                repro.setdefault((r["hire_id"], r["provider_id"]), r)
    orig = {(r["hire_id"], r["provider_id"]): r["grade"] for r in original if r["status"] == "ok"}
    with_orig = lambda key, r: {**r, "orig_grade": orig.get(key)}
    diff = json.loads((AUDIT / "diff_pairs_test.json").read_text(encoding="utf-8"))
    side = lambda pairs: [with_orig(tuple(k), repro[tuple(k)]) for k in pairs if tuple(k) in repro]
    return {"diff_in": side(diff["entering"]), "diff_out": side(diff["leaving"]),
            "doubly_graded": [with_orig(k, r) for k, r in repro.items() if k in orig]}


def draw_sample(rnd: int) -> list[dict]:
    cfg = ROUNDS[rnd]
    rng = random.Random(cfg["seed"])
    original = _records("judgments.jsonl")
    in_original = {(r["hire_id"], r["provider_id"]) for r in original}
    pools = {"judgments.jsonl": [r for r in original if r["status"] == "ok"],
             "judgments_tag.jsonl": [r for r in _records("judgments_tag.jsonl")
                                     if r["status"] == "ok" and (r["hire_id"], r["provider_id"]) not in in_original]}
    if rnd == 3:
        pools.update(_round3_pools(original))
    used = set()  # pairs already audited in an earlier round are never drawn again
    for earlier in range(1, rnd):
        key_path = AUDIT / ROUNDS[earlier]["key"]
        if key_path.exists():
            used |= {(k["hire_id"], k["provider_id"]) for k in json.loads(key_path.read_text(encoding="utf-8"))}
    chosen, taken_gigs = [], set()
    for stratum, (source, n, keep) in cfg["strata"].items():
        candidates = [r for r in pools[source] if keep(r) and (r["hire_id"], r["provider_id"]) not in used]
        rng.shuffle(candidates)
        picked = []
        for r in candidates:   # at most one pair per gig within a round
            if r["hire_id"] in taken_gigs:
                continue
            picked.append(r)
            taken_gigs.add(r["hire_id"])
            if len(picked) == n:
                break
        if len(picked) < n:
            raise SystemExit(f"stratum {stratum}: only {len(picked)} of {n} available")
        for r in picked:
            chosen.append({"hire_id": r["hire_id"], "provider_id": r["provider_id"], "stratum": stratum,
                           "qwen_grade": r["grade"], "qwen_probs": r["probs"], "qwen_prompt": r["prompt_version"],
                           "orig_grade": r.get("orig_grade")})
    rng.shuffle(chosen)
    width = 2 if rnd == 1 else 3
    for i, item in enumerate(chosen, start=1):
        item["item"] = f"{cfg['prefix']}{i:0{width}d}"
    return chosen


def _load_entities():
    hirers = {str(h["hire_id"]): h for h in json.loads((DATA / "hirers.json").read_text(encoding="utf-8"))}
    providers = {str(p["provider_id"]): p
                 for p in json.loads((DATA / "providers.json").read_text(encoding="utf-8"))}
    return hirers, providers


def cmd_sample(args):
    cfg = ROUNDS[args.round]
    sample = draw_sample(args.round)
    AUDIT.mkdir(exist_ok=True)
    (AUDIT / cfg["key"]).write_text(json.dumps(sample, indent=1), encoding="utf-8")
    print(f"wrote audit/{cfg['key']}: {len(sample)} items, seed {cfg['seed']} (strata withheld from `show`)")


def cmd_show(args):
    cfg = ROUNDS[args.round]
    sample = json.loads((AUDIT / cfg["key"]).read_text(encoding="utf-8"))
    hirers, providers = _load_entities()
    n = cfg["per_page"]
    for it in sample[(args.part - 1) * n: args.part * n]:
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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def _fmt_ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.2f} [{lo:.2f}, {hi:.2f}]" if n else "n/a"


def newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """p1 - p2 with Newcombe's hybrid-score 95% interval (Wilson bounds of each proportion)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def score_round3(rows: list) -> None:
    """The two questions round 3 was drawn for. The verdict rule was fixed before any grade was read:
    entering positives confirmed LOWER than leaving ones (interval excludes 0) = the gain is grader leniency;
    otherwise the audit does not contradict the gain (it is underpowered for a small gap, which is stated)."""
    cnt = {}
    for stratum in ("diff_in_pos", "diff_out_pos", "discordant", "concordant_pos"):
        sub = [g for r, g in rows if r["stratum"] == stratum]
        cnt[stratum] = (sum(g >= 2 for g in sub), len(sub))
    print("\nROUND 3 (Claude >= 2 rates, Wilson 95%)")
    for stratum, (k, n) in cnt.items():
        print(f"  {stratum:<15} {_fmt_ci(k, n)}")
    (ka, na), (kb, nb) = cnt["discordant"], cnt["concordant_pos"]
    print(f"  discordant (original < 2, repro >= 2): Claude sides with the reproduction on {ka}/{na}, with the original "
          f"on {na - ka}/{na}; concordant control confirmed {kb}/{nb}")
    (k1, n1), (k2, n2) = cnt["diff_in_pos"], cnt["diff_out_pos"]
    d, lo, hi = newcombe_diff(k1, n1, k2, n2)
    verdict = ("ENTERING LOWER: the gain looks like grader leniency" if hi < 0 else
               "no evidence the entering positives are confirmed less often" if d >= 0 or hi >= 0 and lo < 0 else "")
    print(f"  entering minus leaving positives, confirmed by Claude: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")
    try:
        from scipy.stats import fisher_exact
        print(f"  Fisher exact p = {fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]:.4f}")
    except ImportError:
        pass
    print(f"  power note: with n = {n1} and {n2}, a true gap of 0.08 (what the Stage-1 P@10 gain implies at about 3 "
          f"swapped pairs per gig) is not reliably detectable; a wide interval is not evidence of no gap.")


def cmd_score(args):
    cfg = ROUNDS[args.round]
    key = {k["item"]: k for k in json.loads((AUDIT / cfg["key"]).read_text(encoding="utf-8"))}
    grades = {}
    for line in (AUDIT / cfg["grades"]).read_text(encoding="utf-8").splitlines():
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
        qpos, cpos = [int(x >= 2) for x in q], [int(y >= 2) for y in c]
        agree = sum(x == y for x, y in zip(qpos, cpos))
        k = _kappa(qpos, cpos)
        down = sum(x > y for x, y in zip(q, c))
        up = sum(x < y for x, y in zip(q, c))
        print(f"{name:<14} n={n:<3} exact {exact:.2f}  >=2 agree {_fmt_ci(agree, n)}  "
              f"kappa {'n/a' if k is None else f'{k:.2f}'}  claude lower {down} / higher {up}  "
              f"qwen>=2 {sum(qpos)}  claude>=2 {sum(cpos)}")

    report("ALL", rows)
    for stratum in cfg["strata"]:
        report(stratum, [r for r in rows if r[0]["stratum"] == stratum])

    print("\nconfusion (rows qwen, cols claude):")
    m = [[0] * 4 for _ in range(4)]
    for r, g in rows:
        m[r["qwen_grade"]][g] += 1
    for i, row in enumerate(m):
        print(f"  {i}: {row}")

    # Among pairs qwen graded >= 2, how many does Claude also grade >= 2? (the precision of qwen's positives)
    print("\nshare of qwen's >=2 grades that Claude also grades >=2 (Wilson 95% CI):")
    shares = {}
    for stratum in cfg["strata"]:
        if stratum.endswith("_pos"):
            sub = [g for r, g in rows if r["stratum"] == stratum]
            shares[stratum] = (sum(g >= 2 for g in sub), len(sub))
            print(f"  {stratum:<14} {_fmt_ci(*shares[stratum])}")
    # Same table split by qwen's own P(grade >= 2): if the gap between strata survived only at low confidence it
    # would be a calibration artefact; it is reported so the reader can see whether it does.
    bins = ((0.0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01))
    for stratum in cfg["strata"]:
        if not stratum.endswith("_pos"):
            continue
        cells = []
        for lo, hi in bins:
            sub = [g for r, g in rows if r["stratum"] == stratum and lo <= sum(r["qwen_probs"][2:]) < hi]
            if sub:
                cells.append(f"P[{lo:.2f},{hi:.2f}) {sum(g >= 2 for g in sub)}/{len(sub)}")
        print(f"  {stratum:<14} Claude >=2 by qwen P(>=2): " + "   ".join(cells))
    if "tag_only_pos" in shares and "orig_pos" in shares:
        try:
            from scipy.stats import fisher_exact
            (a, na), (b, nb) = shares["tag_only_pos"], shares["orig_pos"]
            _, p = fisher_exact([[a, na - a], [b, nb - b]])
            print(f"  tag-only vs original positives, Fisher exact p = {p:.4f}")
        except ImportError:
            pass
    if args.round == 3:
        score_round3(rows)


def _serious_mismatch(h: dict, p: dict) -> list[str]:
    """The rubric's definition of a serious term mismatch, applied to the structured fields."""
    out = []
    try:
        hi, rate = float(h["budget_hi"]), float(p["rate_per_hour"])
        if rate > 1.4 * hi:
            out.append("budget")
    except (KeyError, TypeError, ValueError):
        pass
    levels = {"mid": 0, "senior": 1, "expert": 2}
    need, have = levels.get(h.get("seniority_needed")), levels.get(p.get("seniority"))
    if need is not None and have is not None and abs(need - have) == 2:
        out.append("seniority")
    start, avail = _parse_avail_date(h.get("start_by"), ANCHOR_DATE), _parse_avail_date(p.get("available_from"), ANCHOR_DATE)
    if start and avail and avail - start > timedelta(days=30):
        out.append("start")
    try:
        if float(h["commitment"]) - float(p["capacity"]) >= 2:
            out.append("days")
    except (KeyError, TypeError, ValueError):
        pass
    return out


def cmd_termcheck_paired(hirers, providers):
    """Same rule, but on pairs graded under BOTH prompts, so the population is identical: the conditional rate
    P(grade >= 2 | serious mismatch) for each prompt (the share-of-positives above depends on the population)."""
    orig = {(r["hire_id"], r["provider_id"]): r["grade"] for r in _records("judgments.jsonl") if r["status"] == "ok"}
    repro = {}
    for name in ("judgments_variants.jsonl", "judgments_regrade.jsonl", "judgments_tag.jsonl", "calibration.jsonl"):
        for r in _records(name):
            if r["status"] == "ok":
                repro.setdefault((r["hire_id"], r["provider_id"]), r["grade"])
    both = [k for k in repro if k in orig]
    flagged = {k for k in both if _serious_mismatch(hirers[str(k[0])], providers[str(k[1])])}
    print(f"\npairs graded under both prompts: {len(both)}, of which with a serious term mismatch: {len(flagged)}")
    for label, keys in (("serious mismatch", flagged), ("no mismatch", set(both) - flagged)):
        o, r = sum(orig[k] >= 2 for k in keys), sum(repro[k] >= 2 for k in keys)
        print(f"  {label:<17} n={len(keys):<5} original >= 2: {_fmt_ci(o, len(keys))}   repro >= 2: {_fmt_ci(r, len(keys))}")


def cmd_termcheck(_args):
    hirers, providers = _load_entities()
    in_original = set()
    for name, label in (("judgments.jsonl", "original (rubric_0_3.v2)"), ("judgments_tag.jsonl", "tag-only (v2-repro)")):
        recs = [r for r in _records(name) if r["status"] == "ok"]
        if name == "judgments.jsonl":
            in_original = {(r["hire_id"], r["provider_id"]) for r in recs}
        else:
            recs = [r for r in recs if (r["hire_id"], r["provider_id"]) not in in_original]
        pos = [r for r in recs if r["grade"] >= 2]
        flagged = [r for r in pos if _serious_mismatch(hirers[str(r["hire_id"])], providers[str(r["provider_id"])])]
        by_kind = {}
        for r in flagged:
            for kind in _serious_mismatch(hirers[str(r["hire_id"])], providers[str(r["provider_id"])]):
                by_kind[kind] = by_kind.get(kind, 0) + 1
        allflag = sum(bool(_serious_mismatch(hirers[str(r["hire_id"])], providers[str(r["provider_id"])])) for r in recs)
        print(f"{label}: {len(recs)} pairs, {len(pos)} graded >= 2")
        print(f"  graded >= 2 despite a serious term mismatch: {_fmt_ci(len(flagged), len(pos))}   by kind {by_kind}")
        print(f"  share of ALL pairs with a serious mismatch: {allflag / len(recs):.3f}")
    cmd_termcheck_paired(hirers, providers)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sample", cmd_sample), ("show", cmd_show), ("score", cmd_score)):
        sp = sub.add_parser(name)
        sp.add_argument("--round", type=int, choices=sorted(ROUNDS), default=1)
        if name == "show":
            sp.add_argument("--part", type=int, required=True)
        sp.set_defaults(fn=fn)
    sub.add_parser("termcheck").set_defaults(fn=cmd_termcheck)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
