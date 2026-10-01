"""
Zero-shot 0-3 relevance labeller for (gig, provider) pairs, reproducing the grading of
`data_sat/judgments.jsonl` (see label.md) so the pairs only the tag channel surfaces can be graded.

The original prompt file (`rubric_0_3_v2.md`) lives in the BT4103-Scrape-and-Tag repo, which is not
available here. The prompt below is a reconstruction from the rubric in label.md, so it is versioned
`rubric_0_3.v2-repro` and the `calibrate` mode must be run before any new grade is trusted: it
regrades already-graded pairs and reports agreement with the original labels.

Scoring follows the original: temperature 0, one generated token, the grade is the argmax of the
model's log-probabilities over the tokens "0".."3", `expected_grade` their probability-weighted mean.
The endpoint serves a reasoning model, which puts its answer in a `reasoning` field unless it is sent
`reasoning_effort: "none"`; an empty answer is retried once with thinking switched off another way.

Endpoint settings come from the environment or an untracked `.env` (SOCLAAS_BASE_URL, SOCLAAS_API_KEY,
SOCLAAS_MODEL); the key is never printed or written to any output.

Modes (run from the repo root, in the .venv):
    python pipeline/labeller.py pairs     --data-dir data_sat --pools judging_pools_tag.json --out judgments_tag.jsonl
    python pipeline/labeller.py calibrate --data-dir data_sat
    python pipeline/labeller.py merge     --data-dir data_sat --judgments judgments_tag.jsonl
Every mode is resumable: finished pairs (status "ok") in the output file are skipped.
"""
import argparse
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path

from corpus import hirer_text, provider_text
from features import _parse_avail_date

BASE = Path(__file__).parent
RUBRIC_VERSION = "rubric_0_3.v2-repro"
GRADES = (0, 1, 2, 3)
# label.md: "Dates count from 2026-10-01". Sentinels ("now", "asap") resolve to this fixed anchor rather
# than date.today(), so a prompt (and so a grade) does not depend on the day it is sent.
ANCHOR_DATE = date(2026, 10, 1)
ORIGINAL_EXPECTED = {0: 0.20, 1: 0.83, 2: 1.47, 3: 2.26}   # label.md: mean expected_grade per assigned grade
DEFAULT_RPS = 1.0
MAX_ATTEMPTS = 4
ABORT_AFTER_CONSECUTIVE_ERRORS = 20


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_env(path: Path) -> dict:
    """Minimal .env reader (python-dotenv is not installed): KEY=VALUE per line, split on the first '=',
    surrounding quotes stripped, blank lines and # comments skipped."""
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def endpoint_config(env_path: Path = BASE.parent / ".env") -> dict:
    """{base_url, api_key, model}: real environment variables win over the .env file."""
    file_env = load_env(env_path)
    get = lambda k: os.environ.get(k) or file_env.get(k)
    cfg = {"base_url": get("SOCLAAS_BASE_URL"), "api_key": get("SOCLAAS_API_KEY"), "model": get("SOCLAAS_MODEL")}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise SystemExit(f"labeller: missing {missing}; set SOCLAAS_* in the environment or .env (see .env.example)")
    return cfg


def http_complete(cfg: dict, timeout: float = 60.0):
    """A `complete(payload) -> response dict` that POSTs to {base_url}/chat/completions."""
    url = cfg["base_url"].rstrip("/") + "/chat/completions"

    def complete(payload: dict) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    return complete


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

RUBRIC = """You are grading how well a service provider fits a client's gig posting.

Step 1. Judge CONTENT fit: compare what the provider has done and offers with what the gig actually needs.
Step 2. Check the TERMS (budget, seniority, availability). Terms can only lower a grade, never raise it.

How to read the terms:
- Budget: the provider's hourly rate against the gig's budget range. A rate below the range, or within it, is fine.
  A rate up to about 15% over the top of the range is a minor mismatch; more than about 40% over is a serious mismatch.
  Between those is worse than minor but not yet serious.
- Seniority (mid < senior < expert): the same level is fine; one level apart is a minor mismatch; mid against expert
  is a serious mismatch. The gap counts the same in either direction.
- Availability: the provider starting up to 2 weeks after the gig's start date, or offering 1 day a week fewer than
  the gig needs, is a minor mismatch. Starting more than 1 month late, or offering 2 or more days a week fewer than
  needed, is a serious mismatch.

Grades:
3 = the provider's expertise directly addresses the specific need, with at most one minor mismatch.
2 = relevant but not perfect content, with at most minor mismatches; or excellent content with several minor mismatches.
1 = only surface relevance; or relevant expertise with a serious mismatch.
0 = no genuine content relevance, whatever the terms.

Answer with a single digit, 0, 1, 2 or 3, and nothing else."""


# The text of the original prompt as supplied by its owner (BT4103-Scrape-and-Tag `rubric_0_3_v2.md`, pasted into a
# session, not read from that repo, so it is unverified against the file). Selected with `--prompt orig`. It carries the
# anchors the reconstruction above lacks: grade 2 is "an adjacent sub-focus", grade 1 is "an adjacent area or a
# transferable skill". The pasted text has no output instruction, so OUTPUT_INSTRUCTION is appended; scoring reads the
# first token's log-probabilities, so that instruction is what makes the first token a grade digit.
RUBRIC_ORIG = """You are matching a gig (a piece of work a client wants done) to a provider profile (an experienced professional's showcase). Score how well the provider fits this gig on an ordinal scale from 0 to 3: both what they have done and whether the practical terms work.

First judge the content fit: does the provider's expertise address what the gig needs? Then check the terms:

Budget: the provider's hourly rate against the gig's budget range. Up to about 15% over the top of the range is a minor mismatch; more than about 40% over is serious. A rate below the range is fine.
Seniority: the provider's level against the level the gig asks for (mid < senior < expert). One level apart is a minor mismatch; mid against expert is serious.
Availability: whether the provider can start by the gig's start date and offers the days per week the gig needs. Starting up to 2 weeks late, or offering 1 day a week fewer, is minor; starting more than a month late, or 2 or more days a week fewer, is serious.
Judgement: 3 = excellent match: the provider's specific expertise directly addresses this gig's specific need, and the terms work (at most one minor mismatch). 2 = good match: genuinely relevant domain and skill set, but not a perfect fit (for example an adjacent sub-focus), with at most minor mismatches in the terms; OR an excellent content fit with several minor mismatches. 1 = weak match: some surface relevance (an adjacent area or a transferable skill), but not a real fit for this need; OR relevant expertise with a serious mismatch in the terms. 0 = not relevant: no genuine content-level relevance, whatever the terms."""
OUTPUT_INSTRUCTION = "Output only a single integer between 0 and 3 inclusive."
RUBRIC_VERSION_ORIG = "rubric_0_3.v2-orig-text"

# A continuous 0-1 variant (`--prompt cont`): the same content-then-terms rubric with anchor bands instead of integer
# grades. The model writes a decimal, so the score is the greedy-decoded number, not a first-token log-probability read.
# "Today is ..." is rendered from the anchor date, which is the 1 October 2026 the pasted text gives.
CONT_RUBRIC = """You are matching a gig (a piece of work a client wants done) to a provider profile (an experienced professional's showcase). Score how well the provider fits this gig on a continuous scale from 0 to 1: both what they have done and whether the practical terms work. Today is {today}.

First judge the content fit: does the provider's expertise address what the gig needs? Then check the terms:

Budget: the provider's hourly rate against the gig's budget range. Up to about 15% over the top of the range is a minor mismatch; more than about 40% over is serious. A rate below the range is fine.
Seniority: the provider's level against the level the gig asks for (mid < senior < expert). One level apart is a minor mismatch; mid against expert is serious.
Availability: whether the provider can start by the gig's start date and offers the days per week the gig needs. Starting up to 2 weeks late, or offering 1 day a week fewer, is minor; starting more than a month late, or 2 or more days a week fewer, is serious.
Use these anchor bands: 0.85-1.00 = excellent match: the provider's specific expertise directly addresses this gig's specific need, and the terms work (at most one minor mismatch). 0.55-0.80 = good match: genuinely relevant domain and skill set, but not a perfect fit (for example an adjacent sub-focus), with at most minor mismatches in the terms; OR an excellent content fit with several minor mismatches. 0.20-0.45 = weak match: some surface relevance (an adjacent area or a transferable skill), but not a real fit for this need; OR relevant expertise with a serious mismatch in the terms. 0.00-0.15 = not relevant: no genuine content-level relevance, whatever the terms.

Pick the band first, then place the score within it: higher when the content fit is closer to the band above and the terms are clean, lower when it is closer to the band below or the terms have mismatches. Use the full range of each band rather than defaulting to its edges or to round numbers. The terms can only lower a score, never raise it above what the content fit supports."""
CONT_OUTPUT_INSTRUCTION = ("Output only the score, as a decimal number between 0 and 1 with two decimal places "
                           "(for example 0.07, 0.38, 0.64 or 0.91).")
RUBRIC_VERSION_CONT = "rubric_0_1.v2-cont"
# The anchor bands leave gaps (0.15-0.20, 0.45-0.55, 0.80-0.85); a score maps to the 0-3 grade of the band it falls in,
# cutting each gap in the middle. Only for comparison with the integer grades, the score itself is the label.
BAND_CUTS = (0.175, 0.50, 0.825)
PROMPT_VERSIONS = {"repro": RUBRIC_VERSION, "orig": RUBRIC_VERSION_ORIG, "cont": RUBRIC_VERSION_CONT}


def _fmt_date(value: str | None, today: date) -> str | None:
    d = _parse_avail_date(value, today)
    return f"{d.day} {d.strftime('%b %Y')}" if d else None


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _days(n) -> str | None:
    n = _num(n)
    if n is None:
        return None
    n = int(n)
    return f"{n} day{'s' if n != 1 else ''} a week"


def hirer_terms(h: dict, today: date = ANCHOR_DATE) -> list[str]:
    lo, hi = _num(h.get("budget_lo")), _num(h.get("budget_hi"))
    lines = []
    if lo is not None and hi is not None:
        lines.append(f"Budget: S${lo:g}-{hi:g} per hour")
    if h.get("seniority_needed"):
        lines.append(f"Seniority needed: {h['seniority_needed']}")
    start, days = _fmt_date(h.get("start_by"), today), _days(h.get("commitment"))
    if start:
        lines.append(f"Start date: {start}")
    if days:
        lines.append(f"Commitment: {days}")
    return lines


def provider_terms(p: dict, today: date = ANCHOR_DATE) -> list[str]:
    rate = _num(p.get("rate_per_hour"))
    lines = []
    if rate is not None:
        lines.append(f"Rate: S${rate:g} per hour")
    if p.get("seniority"):
        lines.append(f"Seniority: {p['seniority']}")
    start, days = _fmt_date(p.get("available_from"), today), _days(p.get("capacity"))
    if start and days:
        lines.append(f"Availability: from {start}, {days}")
    elif p.get("availability"):
        lines.append(f"Availability: {p['availability']}")
    return lines


def build_prompt(hirer: dict, provider: dict, today: date = ANCHOR_DATE, variant: str = "repro") -> str:
    """The grading prompt for one pair. `variant` "repro" is the reconstruction (the default every earlier grade used);
    "orig" is the rubric text supplied by the original's owner, with the gig and profile followed by their terms."""
    if variant == "cont":
        return "\n".join([
            CONT_RUBRIC.format(today=f"{today.day} {today.strftime('%B %Y')}"), "",
            "Gig: " + hirer_text(hirer), *hirer_terms(hirer, today), "",
            "Provider profile: " + provider_text(provider), *provider_terms(provider, today), "",
            CONT_OUTPUT_INSTRUCTION,
        ])
    if variant == "orig":
        return "\n".join([
            RUBRIC_ORIG, "",
            "Gig: " + hirer_text(hirer), *hirer_terms(hirer, today), "",
            "Provider profile: " + provider_text(provider), *provider_terms(provider, today), "",
            OUTPUT_INSTRUCTION,
        ])
    if variant != "repro":
        raise ValueError(f"variant must be one of {sorted(PROMPT_VERSIONS)}, got {variant!r}")
    return "\n".join([
        RUBRIC, "",
        "=== GIG ===", hirer_text(hirer), *hirer_terms(hirer, today), "",
        "=== PROVIDER PROFILE ===", provider_text(provider), *provider_terms(provider, today), "",
        "Grade (0, 1, 2 or 3):",
    ])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def grade_distribution(top_logprobs: list[dict]) -> list[float] | None:
    """Probabilities of the grades 0..3 from the first generated token's top log-probs, renormalised over
    those four tokens (other tokens are dropped, digits that are missing count as 0). Tokens that differ only
    in surrounding whitespace ("3", " 3") are the same grade and their probabilities add. None if no grade
    digit appears at all."""
    mass = [0.0] * len(GRADES)
    for entry in top_logprobs or []:
        token = str(entry.get("token", "")).strip()
        if token in {str(g) for g in GRADES}:
            mass[int(token)] += math.exp(entry["logprob"])
    total = sum(mass)
    return [m / total for m in mass] if total > 0 else None


def _first_token_top_logprobs(response: dict) -> list[dict]:
    try:
        return response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return []


def _content(response: dict) -> str:
    try:
        return (response["choices"][0]["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


SCORE_RE = re.compile(r"(?<![\d.])(?:[01](?:\.\d+)?|\.\d+)(?![\d.])")


def parse_score(text: str) -> float | None:
    """The first decimal in [0, 1] in a model's answer ("0.64", "1", ".5"), else None. A number outside the range, or
    one glued to other digits or dots, is not a score."""
    m = SCORE_RE.search(text or "")
    if m is None:
        return None
    value = float(m.group())
    return value if 0.0 <= value <= 1.0 else None


def band_grade(score: float) -> int:
    """The 0-3 grade of the anchor band a 0-1 score falls in (BAND_CUTS cut the gaps between bands)."""
    return sum(score >= cut for cut in BAND_CUTS)


class Labeller:
    def __init__(self, complete, model: str, today: date = ANCHOR_DATE, clock=time.time, variant: str = "repro"):
        self.complete, self.model, self.today, self.clock = complete, model, today, clock
        self.variant, self.prompt_version = variant, PROMPT_VERSIONS[variant]

    def payload(self, prompt: str, *, alt_thinking_off: bool = False) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 8 if self.variant == "cont" else 2, "logprobs": True, "top_logprobs": 20,
        }
        if alt_thinking_off:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            body["reasoning_effort"] = "none"
        return body

    def grade_pair(self, hirer: dict, provider: dict) -> dict:
        """One graded record, schema of a judgments.jsonl line. status is "ok" or "error: <why>"."""
        prompt = build_prompt(hirer, provider, self.today, self.variant)
        started = self.clock()
        response, error = None, None
        for alt in (False, True):                       # retry once, thinking switched off the other way
            response = self.complete(self.payload(prompt, alt_thinking_off=alt))
            if _content(response):
                error = None
                break
            error = "error: empty content (thinking leaked)"
        record = {
            "hire_id": str(hirer["hire_id"]), "provider_id": str(provider["provider_id"]),
            "model": self.model, "prompt_version": self.prompt_version,
            "timestamp": datetime.fromtimestamp(self.clock()).isoformat(timespec="seconds"),
            "status": "ok", "grade": None, "expected_grade": None, "probs": None,
            "generated": _content(response), "scoring": "logprobs",
            "usage": {k: (response.get("usage") or {}).get(k) for k in ("prompt_tokens", "completion_tokens")},
            "elapsed": round(self.clock() - started, 2),
        }
        if self.variant == "cont":                      # the score is the decoded number; no grade distribution exists
            score = None if error else parse_score(_content(response))
            if error is None and score is None:
                error = "error: no score in output"
            record["scoring"] = "generated"
            if error:
                record["status"] = error
                return record
            record["score"], record["grade"] = score, band_grade(score)
            return record
        probs = None if error else grade_distribution(_first_token_top_logprobs(response))
        if error is None and probs is None:
            error = "error: no grade token in top_logprobs"
        if error:
            record["status"] = error
            return record
        record["probs"] = [round(p, 4) for p in probs]
        record["grade"] = max(GRADES, key=lambda g: probs[g])   # argmax, first maximum wins ties
        record["expected_grade"] = round(sum(g * p for g, p in zip(GRADES, probs)), 4)
        return record


# ---------------------------------------------------------------------------
# Running a list of pairs
# ---------------------------------------------------------------------------

def read_records(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finished_pairs(records: list[dict]) -> set[tuple[str, str]]:
    return {(r["hire_id"], r["provider_id"]) for r in records if r.get("status") == "ok"}


def seeded_order(pairs: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    ordered = sorted(set(pairs))
    random.Random(seed).shuffle(ordered)
    return ordered


def run_pairs(pairs, hirers: dict, providers: dict, labeller: Labeller, out_path: Path, *,
              rps: float = DEFAULT_RPS, seed: int = 7, limit: int | None = None,
              sleep=time.sleep, clock=time.monotonic, log=print) -> dict:
    """Grade `pairs` (hire_id, provider_id) into the append-only JSONL `out_path`, skipping pairs already
    graded "ok" there (errors are retried). Seeded random order so an interrupted run leaves a random
    sample; calls are spaced 1/rps apart; a failing call is retried with backoff and, if it keeps failing,
    recorded as an error and skipped. Stops if errors run on back to back (endpoint down)."""
    out_path = Path(out_path)
    done = finished_pairs(read_records(out_path))
    todo = [p for p in seeded_order(list(pairs), seed) if p not in done]
    if limit is not None:
        todo = todo[:limit]
    stats = {"requested": len(set(pairs)), "already_done": len(done & set(pairs)), "graded": 0, "errors": 0}
    interval = 1.0 / rps if rps else 0.0
    last_call, consecutive = None, 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for n, (hid, pid) in enumerate(todo, 1):
            if last_call is not None and interval:
                wait = interval - (clock() - last_call)
                if wait > 0:
                    sleep(wait)
            last_call = clock()
            record = None
            for attempt in range(MAX_ATTEMPTS):
                try:
                    record = labeller.grade_pair(hirers[hid], providers[pid])
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError) as exc:
                    record = {"hire_id": hid, "provider_id": pid, "model": labeller.model,
                              "prompt_version": labeller.prompt_version, "status": f"error: {type(exc).__name__}",
                              "grade": None, "expected_grade": None, "probs": None}
                    if attempt < MAX_ATTEMPTS - 1:
                        sleep(2.0 ** (attempt + 1))
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            if record["status"] == "ok":
                stats["graded"] += 1
                consecutive = 0
            else:
                stats["errors"] += 1
                consecutive += 1
                if consecutive >= ABORT_AFTER_CONSECUTIVE_ERRORS:
                    raise RuntimeError(f"{consecutive} consecutive errors (last: {record['status']}); stopping")
            if n % 100 == 0:
                log(f"  {n}/{len(todo)} graded this run ({stats['errors']} errors)")
    return stats


# ---------------------------------------------------------------------------
# Merging into label files
# ---------------------------------------------------------------------------

def merge_labels(base_merged: dict, records: list[dict]) -> tuple[dict, dict]:
    """New {hire_id: {provider_id: grade}} = `base_merged` plus the "ok" records. A pair that already has a
    label keeps it (never overwritten, even when the new grade differs). Returns (merged, counts)."""
    merged = {h: dict(ps) for h, ps in base_merged.items()}
    counts = {"added": 0, "kept_existing": 0, "skipped_errors": 0}
    for r in records:
        if r.get("status") != "ok":
            counts["skipped_errors"] += 1
            continue
        slot = merged.setdefault(r["hire_id"], {})
        if r["provider_id"] in slot:
            counts["kept_existing"] += 1
        else:
            slot[r["provider_id"]] = r["grade"]
            counts["added"] += 1
    return merged, counts


def ground_truth_from(merged: dict) -> dict:
    """ground_truth_llm.json format: grade > 0 only, scores 0/33/67/100."""
    score = {1: 33, 2: 67, 3: 100}
    return {h: {p: score[g] for p, g in ps.items() if g > 0} for h, ps in merged.items()
            if any(g > 0 for g in ps.values())}


# ---------------------------------------------------------------------------
# Calibration statistics
# ---------------------------------------------------------------------------

def confusion_matrix(original: list[int], regraded: list[int]) -> list[list[int]]:
    """rows = original grade, columns = new grade."""
    m = [[0] * len(GRADES) for _ in GRADES]
    for o, r in zip(original, regraded):
        m[o][r] += 1
    return m


def cohen_kappa(a: list[int], b: list[int], labels=GRADES, weights: str | None = None) -> float:
    """Cohen's kappa between two raters; weights None (unweighted) or "quadratic"."""
    k = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(a)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1.0 / n
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    def w(i, j):
        if weights == "quadratic":
            return ((i - j) / (k - 1)) ** 2 if k > 1 else 0.0
        return 0.0 if i == j else 1.0
    observed = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    expected = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k))
    return 1.0 - observed / expected if expected > 0 else 1.0


def _auc(pos: list[float], neg: list[float]) -> float | None:
    """P(a random positive scores above a random negative); ties count half. None if either side is empty."""
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    var = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return cov / var if var else 0.0


def continuous_calibration_report(original: list[int], scores: list[float]) -> dict:
    """How a continuous 0-1 regrade relates to the integer grades it replaces: rank agreement, how well the score
    separates each grade boundary (AUC), the score distribution per original grade, and the banded comparison
    (score -> the grade of its anchor band) that the integer calibration report uses."""
    n = len(original)
    banded = [band_grade(s) for s in scores]
    per_grade = {}
    for g in GRADES:
        vals = sorted(s for o, s in zip(original, scores) if o == g)
        per_grade[g] = {"n": len(vals), "mean": sum(vals) / len(vals) if vals else None,
                        "median": vals[len(vals) // 2] if vals else None,
                        "min": vals[0] if vals else None, "max": vals[-1] if vals else None}
    in_gap = sum(0.15 < s < 0.20 or 0.45 < s < 0.55 or 0.80 < s < 0.85 for s in scores)
    hundredths = [round(s * 100) for s in scores]
    return {
        "n": n,
        "spearman_with_original_grade": spearman([float(o) for o in original], scores),
        "auc_original_ge1": _auc([s for o, s in zip(original, scores) if o >= 1], [s for o, s in zip(original, scores) if o < 1]),
        "auc_original_ge2": _auc([s for o, s in zip(original, scores) if o >= 2], [s for o, s in zip(original, scores) if o < 2]),
        "auc_original_ge3": _auc([s for o, s in zip(original, scores) if o >= 3], [s for o, s in zip(original, scores) if o < 3]),
        "score_by_original_grade": per_grade,
        "banded_exact_agreement": sum(o == b for o, b in zip(original, banded)) / n,
        "banded_within_one": sum(abs(o - b) <= 1 for o, b in zip(original, banded)) / n,
        "banded_quadratic_weighted_kappa": cohen_kappa(original, banded, weights="quadratic"),
        "banded_kappa_grade_ge2": cohen_kappa([int(o >= 2) for o in original], [int(b >= 2) for b in banded], labels=(0, 1)),
        "banded_confusion_matrix_rows_original_cols_banded": confusion_matrix(original, banded),
        "share_of_scores_in_band_gaps": in_gap / n,
        "distinct_scores": len(set(scores)),
        "share_ending_in_0_or_5_hundredths": sum(h % 5 == 0 for h in hundredths) / n,
    }


def calibration_report(original: list[int], regraded: list[int], regraded_expected: list[float]) -> dict:
    n = len(original)
    by_assigned = {}
    for g in GRADES:
        vals = [e for r, e in zip(regraded, regraded_expected) if r == g]
        by_assigned[g] = {"n": len(vals), "mean_expected_grade": (sum(vals) / len(vals)) if vals else None,
                          "original_mean_expected_grade": ORIGINAL_EXPECTED[g]}
    hi_o, hi_r = [int(o >= 2) for o in original], [int(r >= 2) for r in regraded]
    return {
        "n": n,
        "exact_agreement": sum(o == r for o, r in zip(original, regraded)) / n,
        "within_one": sum(abs(o - r) <= 1 for o, r in zip(original, regraded)) / n,
        "quadratic_weighted_kappa": cohen_kappa(original, regraded, weights="quadratic"),
        "kappa_grade_ge2": cohen_kappa(hi_o, hi_r, labels=(0, 1)),
        "confusion_matrix_rows_original_cols_new": confusion_matrix(original, regraded),
        "by_assigned_grade": by_assigned,
    }


def calibration_sample(merged: dict, per_grade: int, seed: int) -> list[tuple[str, str, int]]:
    """per_grade already-graded pairs for each grade 0..3 (fewer if a grade has fewer), seeded."""
    by_grade = {g: [] for g in GRADES}
    for h in sorted(merged, key=str):
        for p in sorted(merged[h], key=str):
            by_grade[merged[h][p]].append((h, p))
    rng = random.Random(seed)
    sample = []
    for g in GRADES:
        pool = by_grade[g]
        sample += [(h, p, g) for h, p in rng.sample(pool, min(per_grade, len(pool)))]
    return sample


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_data(data_dir: Path):
    hirers = {str(h["hire_id"]): h for h in _load_json(data_dir / "hirers.json")}
    providers = {str(p["provider_id"]): p for p in _load_json(data_dir / "providers.json")}
    return hirers, providers


def _pairs_from_pools(pools: dict) -> list[tuple[str, str]]:
    return [(str(h), str(p)) for h, ps in pools.items() for p in ps]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["pairs", "calibrate", "merge"])
    ap.add_argument("--data-dir", type=Path, default=BASE / "data_sat")
    ap.add_argument("--pools", default="judging_pools_tag.json", help="pairs mode: pool file in --data-dir")
    ap.add_argument("--out", default=None, help="JSONL output in --data-dir (pairs: judgments_tag.jsonl, calibrate: calibration.jsonl)")
    ap.add_argument("--judgments", default="judgments_tag.jsonl", help="merge mode: JSONL in --data-dir")
    ap.add_argument("--rps", type=float, default=DEFAULT_RPS)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None, help="grade at most this many pairs this run")
    ap.add_argument("--per-grade", type=int, default=75, help="calibrate mode: pairs per original grade")
    ap.add_argument("--prompt", choices=sorted(PROMPT_VERSIONS), default="repro",
                    help="repro: the reconstruction every earlier grade used (default); orig: the rubric text supplied "
                         "by the original's owner plus an explicit output instruction; cont: the same rubric as a "
                         "continuous 0-1 score with anchor bands (records carry `score`; `grade` is its band). "
                         "With orig or cont the default output files get an _orig / _cont suffix, so no earlier file "
                         "is overwritten")
    args = ap.parse_args(argv)
    suffix = {"repro": "", "orig": "_orig", "cont": "_cont"}[args.prompt]
    version = PROMPT_VERSIONS[args.prompt]
    data_dir = args.data_dir

    if args.mode == "merge":
        records = read_records(data_dir / args.judgments)
        merged, counts = merge_labels(_load_json(data_dir / "llm_judgments_merged.json"), records)
        (data_dir / "llm_judgments_merged_tag.json").write_text(json.dumps(merged), encoding="utf-8")
        (data_dir / "ground_truth_llm_tag.json").write_text(json.dumps(ground_truth_from(merged)), encoding="utf-8")
        print(f"merged {args.judgments}: {counts} -> llm_judgments_merged_tag.json, ground_truth_llm_tag.json")
        return

    cfg = endpoint_config()
    hirers, providers = _load_data(data_dir)
    labeller = Labeller(http_complete(cfg), cfg["model"], variant=args.prompt)

    if args.mode == "pairs":
        pairs = _pairs_from_pools(_load_json(data_dir / args.pools))
        out = data_dir / (args.out or f"judgments_tag{suffix}.jsonl")
        print(f"{len(set(pairs))} pairs -> {out.name} (model {cfg['model']}, prompt {version}, {args.rps} calls/s)")
        print(run_pairs(pairs, hirers, providers, labeller, out, rps=args.rps, seed=args.seed, limit=args.limit))
        return

    merged = _load_json(data_dir / "llm_judgments_merged.json")
    sample = calibration_sample(merged, args.per_grade, args.seed)
    out = data_dir / (args.out or f"calibration{suffix}.jsonl")
    print(f"regrading {len(sample)} already-graded pairs with prompt {version} -> {out.name}")
    run_pairs([(h, p) for h, p, _ in sample], hirers, providers, labeller, out, rps=args.rps, seed=args.seed, limit=args.limit)
    latest = {(r["hire_id"], r["provider_id"]): r for r in read_records(out) if r.get("status") == "ok"}
    rows = [(g, latest[(h, p)]) for h, p, g in sample if (h, p) in latest]
    if not rows:
        raise SystemExit("no regraded pairs to report")
    if args.prompt == "cont":
        report = continuous_calibration_report([g for g, _ in rows], [r["score"] for _, r in rows])
    else:
        report = calibration_report([g for g, _ in rows], [r["grade"] for _, r in rows], [r["expected_grade"] for _, r in rows])
    report["rubric_version"] = version
    print(json.dumps(report, indent=1))
    (BASE / "results_tag").mkdir(exist_ok=True)
    (BASE / "results_tag" / f"calibration{suffix}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
