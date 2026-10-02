"""
Adapter from the front-end-aligned sample schema (hirers_sample.json / providers_sample.json) to the record shape
the rest of the pipeline reads (`corpus.hirer_text`, `corpus.provider_text`, `features.py`).

    hirer = adapt_hirer(raw)          # {"hire_id": "H001", "hire_title", "hire_description", "search_tags", ...}
    provider = adapt_provider(raw)    # {"provider_id": "P001", "about_title", ..., "rate_per_hour", "available_from", ...}

What the adapter does, and why:

- IDs stay strings ("H001", "P001"); nothing here converts them to int.
- `search_tags` ("Specialisation (Category)", user-selected from the back-end taxonomy) is copied through untouched. It is
  the authoritative pairing: `category` and `specialisation` are de-duplicated projections and are NOT parallel lists
  (a record can have one category and two specialisations), so they are never zipped.
- Generation metadata (`review_flag`, `localisation_changes`, `source_row`, `source_file`) and `credentials` are dropped:
  none of them may reach retrieval text or a feature (`source_file` names a real firm).
- Provider text is rebuilt from the new fields. "Not focused on ..." / "Does not include ..." sentences are removed from
  the service text: BM25 and dense both read them as positive evidence for the very thing they exclude.
  `how_i_work` is generic process prose and is left out unless `include_how_i_work=True`.
- `rate` and `availability` stay free text in the payload. They are parsed into `rate_per_hour`, `available_from` (ISO
  date or "now") and `capacity` (days a week) with the conventions below, so `features.py` can read them; anything that
  does not parse becomes None and the feature falls back to its neutral value.
"""
import re
from datetime import date

HOURS_PER_DAY = 8        # a day rate is turned into an hourly rate with this
HOURS_PER_MONTH = 160    # a monthly retainer, with this (20 working days x 8 h)

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
     "december"], start=1)}

EXCLUSION_SENTENCE = re.compile(r"^\s*(not focused on|not focussed on|does not include|do not include|excludes?|"
                                r"not suitable for|not intended for)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_RATE = re.compile(r"S\$\s*([\d,]+(?:\.\d+)?)\s*(?:per|/)\s*(hour|day|month)", re.IGNORECASE)
_MONTH_YEAR = re.compile(r"\b(mid-)?(" + "|".join(MONTHS) + r")\s+(\d{4})\b", re.IGNORECASE)
_DAYS_PER_WEEK = re.compile(r"(?:(\d)\s*-\s*)?(\d)\s+days?\s+(?:a|per)\s+week", re.IGNORECASE)
_DURATION_WEEKS = re.compile(r"engagement duration:\s*(\d+)\s*(?:-\s*(\d+))?\s*weeks?", re.IGNORECASE)


def strip_exclusions(text: str) -> str:
    """Drop sentences that say what the provider does NOT do."""
    if not text:
        return ""
    kept = [s for s in _SENTENCE_SPLIT.split(text.strip()) if not EXCLUSION_SENTENCE.match(s)]
    return " ".join(kept)


def parse_rate_per_hour(rate: str | None) -> float | None:
    """First 'S$<amount> per hour|day|month' in the text, as an hourly rate; None when there is no such phrase
    (e.g. a fixed fee only). A day is HOURS_PER_DAY hours and a month HOURS_PER_MONTH."""
    m = _RATE.search(rate or "")
    if not m:
        return None
    amount = float(m.group(1).replace(",", ""))
    per = {"hour": 1, "day": HOURS_PER_DAY, "month": HOURS_PER_MONTH}[m.group(2).lower()]
    return round(amount / per, 2)


def parse_available_from(availability: str | None) -> str | None:
    """'now' for 'available immediately', else the ISO date of '<month> <year>' (the 1st; 'mid-' gives the 15th).
    None when neither is present."""
    t = availability or ""
    if re.search(r"\bimmediately\b", t, re.IGNORECASE):
        return "now"
    m = _MONTH_YEAR.search(t)
    if not m:
        return None
    day = 15 if m.group(1) else 1
    return date(int(m.group(3)), MONTHS[m.group(2).lower()], day).isoformat()


def parse_capacity_days(availability: str | None) -> int | None:
    """Days a week on offer ('up to 3 days a week' -> 3, '1-2 days a week' -> 2); None when not stated."""
    m = _DAYS_PER_WEEK.search(availability or "")
    return int(m.group(2)) if m else None


def parse_duration_weeks(description: str | None) -> tuple[int, int] | None:
    """(low, high) from 'Engagement duration: 4-6 weeks' in a gig description."""
    m = _DURATION_WEEKS.search(description or "")
    if not m:
        return None
    lo = int(m.group(1))
    return lo, int(m.group(2) or lo)


def _tags(raw: dict) -> list[str]:
    return [t for t in (raw.get("search_tags") or []) if isinstance(t, str) and t.strip()]


def adapt_hirer(raw: dict) -> dict:
    out = {
        "hire_id": str(raw["hirer_id"]),
        "gig_id": raw.get("gig_id"),
        "hire_title": raw.get("gig_title") or "",
        "hire_description": raw.get("short_gig_description") or "",
        "hire_description_additional_notes": "",
        "category": raw.get("category") or [],
        "specialisation": raw.get("specialisation") or [],
        "search_tags": _tags(raw),
    }
    duration = parse_duration_weeks(out["hire_description"])
    if duration:
        out["duration_weeks_lo"], out["duration_weeks_hi"] = duration
    return out


def _join_service_text(services: list[dict], key: str, clean: bool) -> str:
    parts = [(s.get(key) or "").strip() for s in services or []]
    return " ".join(strip_exclusions(p) if clean else p for p in parts if p)


def adapt_provider(raw: dict, include_how_i_work: bool = False, drop_exclusions: bool = True) -> dict:
    services = raw.get("services_i_offer") or []
    skills = [f"{g.get('category', '')}: {', '.join(g.get('skills') or [])}"
              for g in raw.get("technical_proficiency") or []]
    experience = " ".join(list(raw.get("relevant_achievements") or []) + skills
                          + ([raw["how_i_work"]] if include_how_i_work and raw.get("how_i_work") else []))
    availability = raw.get("availability")
    return {
        "provider_id": str(raw["provider_id"]),
        "about_title": raw.get("about_headline") or raw.get("title") or "",
        "about_description": raw.get("about_bio") or "",
        "services_offered_title": " ".join((s.get("service_title") or "").strip() for s in services).strip(),
        "services_offered_description": _join_service_text(services, "service_detail", drop_exclusions),
        "relevant_experience": experience,
        "category": raw.get("category") or [],
        "specialisation": raw.get("specialisation") or [],
        "search_tags": _tags(raw),
        "rate": raw.get("rate"),
        "availability": availability,
        "rate_per_hour": parse_rate_per_hour(raw.get("rate")),
        "available_from": parse_available_from(availability),
        "capacity": parse_capacity_days(availability),
    }
