"""
How well do the grader prompts agree with the blind Claude audit grades, on the pairs the tag channel surfaced?

The calibration pairs (labeller.py calibrate) come from the original pools, so they cannot show whether a prompt still
over-credits the pairs only the tag channel surfaces, which is where the reconstruction was lenient (audit/RESULTS.md).
This regrades the 355 pairs of audit rounds 2 and 3 (`labeller.py pairs --prompt cont|orig --pools
judging_pools_audit.json`) and compares each prompt's binary relevance (grade >= 2, or score >= a threshold) with Claude's
blind grade >= 2. The reconstruction's own grade is in the keys (`qwen_grade`, `qwen_probs`), so all three are scored on the
same pairs. One rater (Claude), no human labels; the strata were drawn by the reconstruction's grade, so the rates are
not population rates, only comparisons between prompts on the same pairs.

Run from the repo root, in the .venv:
    python pipeline/labeller.py pairs --prompt cont --pools judging_pools_audit.json --out judgments_audit_cont.jsonl
    python pipeline/labeller.py pairs --prompt orig --pools judging_pools_audit.json --out judgments_audit_orig.jsonl
    python pipeline/audit/prompt_vs_claude.py
Writes pipeline/results_tag/prompt_vs_claude.json.
"""
import json
import math
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE))
from labeller import _auc, cohen_kappa, read_records  # noqa: E402

AUDIT = PIPELINE / "audit"
DATA = PIPELINE / "data_sat"
THRESHOLDS = (0.5, 0.6, 0.7)           # continuous score counted relevant at or above these
# strata whose pairs the tag channel surfaced and the reconstruction graded; the rest come from the original pools
TAG_STRATA = ("tag_only_pos", "tag_only_neg", "diff_in_pos", "diff_out_pos")
STRATUM_ORDER = ("tag_only_pos", "tag_only_neg", "diff_in_pos", "diff_out_pos", "discordant", "concordant_pos",
                 "orig_pos", "orig_neg")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def load_rows() -> list[dict]:
    """One row per audited pair: stratum, Claude's grade, the reconstruction's grade and expected grade, and the
    continuous / original-text records when they exist."""
    claude = {}
    for name in ("claude_grades2.jsonl", "claude_grades3.jsonl"):
        for r in read_records(AUDIT / name):
            claude[(name[-7], r["item"])] = r["grade"]
    cont = {(r["hire_id"], r["provider_id"]): r for r in read_records(DATA / "judgments_audit_cont.jsonl") if r.get("status") == "ok"}
    orig = {(r["hire_id"], r["provider_id"]): r for r in read_records(DATA / "judgments_audit_orig.jsonl") if r.get("status") == "ok"}
    rows = []
    for rnd, key_file in (("2", "sample2_key.json"), ("3", "sample3_key.json")):
        for it in json.loads((AUDIT / key_file).read_text(encoding="utf-8")):
            pair = (it["hire_id"], it["provider_id"])
            if pair not in cont or pair not in orig:
                continue
            rows.append({
                "stratum": it["stratum"], "claude": claude[(rnd, it["item"])], "qwen_prompt": it["qwen_prompt"],
                "repro_or_original_grade": it["qwen_grade"],
                "repro_expected": sum(g * p for g, p in enumerate(it["qwen_probs"])),
                "cont_score": cont[pair]["score"], "orig_grade": orig[pair]["grade"], "orig_expected": orig[pair]["expected_grade"],
            })
    return rows


def raters(rows: list[dict]) -> dict[str, list[bool]]:
    """Each prompt's binary relevance on every row."""
    out = {"reconstruction (grade>=2)": [r["repro_or_original_grade"] >= 2 for r in rows],
           "original text (grade>=2)": [r["orig_grade"] >= 2 for r in rows]}
    for t in THRESHOLDS:
        out[f"continuous (score>={t})"] = [r["cont_score"] >= t for r in rows]
    return out


def summarise(rows: list[dict]) -> dict:
    claude = [r["claude"] >= 2 for r in rows]
    result = {"n": len(rows), "claude_ge2_rate": sum(claude) / len(rows), "raters": {}}
    for name, rel in raters(rows).items():
        both = sum(a and b for a, b in zip(rel, claude))
        new_only = sum(a and not b for a, b in zip(rel, claude))       # the prompt says relevant, Claude says not
        claude_only = sum(b and not a for a, b in zip(rel, claude))
        result["raters"][name] = {
            "rated_ge2": sum(rel), "agree": sum(a == b for a, b in zip(rel, claude)) / len(rows),
            "kappa": cohen_kappa([int(x) for x in rel], [int(x) for x in claude], labels=(0, 1)),
            "both_ge2": both, "prompt_only_ge2": new_only, "claude_only_ge2": claude_only,
        }
    scores = {"reconstruction (expected grade)": [r["repro_expected"] for r in rows],
              "original text (expected grade)": [r["orig_expected"] for r in rows],
              "continuous (score)": [r["cont_score"] for r in rows]}
    result["auc_vs_claude_ge2"] = {k: _auc([s for s, c in zip(v, claude) if c], [s for s, c in zip(v, claude) if not c])
                                   for k, v in scores.items()}
    return result


def main():
    rows = load_rows()
    if not rows:
        raise SystemExit("no regraded audit pairs found; run the two labeller.py pairs commands in the header first")
    groups = {"all audited pairs": rows,
              "tag-surfaced pairs (tag-only, diff in/out)": [r for r in rows if r["stratum"] in TAG_STRATA],
              **{s: [r for r in rows if r["stratum"] == s] for s in STRATUM_ORDER}}
    report = {name: summarise(g) for name, g in groups.items() if g}
    for name, rep in report.items():
        print(f"\n=== {name}: n={rep['n']}, Claude grades >= 2 on {rep['claude_ge2_rate']:.2f} ===")
        print(f"{'rater':30}{'rated>=2':>9}{'agree':>7}{'kappa':>7}{'both':>6}{'prompt only':>13}{'claude only':>13}")
        for rater, v in rep["raters"].items():
            print(f"{rater:30}{v['rated_ge2']:9d}{v['agree']:7.2f}{v['kappa']:7.2f}{v['both_ge2']:6d}"
                  f"{v['prompt_only_ge2']:13d}{v['claude_only_ge2']:13d}")
        if name.startswith(("all", "tag-surfaced")):
            print("AUC vs Claude >= 2: " + "; ".join(f"{k} {v:.3f}" for k, v in rep["auc_vs_claude_ge2"].items()))
    # the pointed question: of the reconstruction's positives that Claude confirmed or rejected, what does each new prompt do?
    repro_pos = [r for r in rows if r["repro_or_original_grade"] >= 2 and r["qwen_prompt"].endswith("repro")]
    print(f"\n=== reconstruction positives (n={len(repro_pos)}): confirmed by Claude vs kept by each prompt ===")
    for name, key in (("original text", lambda r: r["orig_grade"] >= 2), ("continuous >=0.5", lambda r: r["cont_score"] >= 0.5),
                      ("continuous >=0.6", lambda r: r["cont_score"] >= 0.6), ("continuous >=0.7", lambda r: r["cont_score"] >= 0.7)):
        conf = [r for r in repro_pos if r["claude"] >= 2]
        rej = [r for r in repro_pos if r["claude"] < 2]
        kept_rej = sum(key(r) for r in rej)
        kept_conf = sum(key(r) for r in conf)
        lo, hi = wilson(kept_rej, len(rej))
        print(f"{name:18} keeps {kept_conf}/{len(conf)} Claude-confirmed positives; still credits {kept_rej}/{len(rej)} "
              f"Claude-rejected ones ({kept_rej / len(rej):.2f} [{lo:.2f}, {hi:.2f}])")
        report.setdefault("repro_positives", {})[name] = {"kept_confirmed": kept_conf, "n_confirmed": len(conf),
                                                          "kept_rejected": kept_rej, "n_rejected": len(rej)}
    path = PIPELINE / "results_tag" / "prompt_vs_claude.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
