"""Turn a scraped company record into structured, model-ready signals.

A company record (produced by ``scrape.py`` or ``synth.py``) looks like::

    {
      "domain": "acme.com",
      "name": "Acme",
      "pages": [{"url": ..., "kind": "home|about|careers|pricing|...", "title": ..., "text": ...}],
      "jobs": {"ats": "greenhouse", "titles": ["Account Executive", ...]} | None,
    }

Everything here is heuristic text mining over public web pages. Each signal is
NaN when the site does not disclose it, and the models handle missing values.
"""
from __future__ import annotations

import math
import re
from datetime import date

CURRENT_YEAR = date.today().year
NAN = float("nan")

ENTERPRISE_TERMS = ["soc 2", "soc2", "iso 27001", "hipaa", "gdpr", "single sign-on", "sso",
                    "saml", "scim", "audit log", "role-based access", "data residency",
                    "99.9% uptime", "uptime sla"]
PLG_TERMS = ["start free", "free trial", "sign up free", "get started for free", "try it free",
             "free forever", "no credit card", "start for free"]
SALES_TERMS = ["request a demo", "book a demo", "schedule a demo", "contact sales",
               "talk to sales", "get a demo", "talk to an expert"]
API_TERMS = ["api reference", "api docs", "developer docs", "rest api", "graphql", "sdk",
             "webhooks", "open api"]
B2C_TERMS = ["add to cart", "shop now", "free shipping", "download on the app store",
             "get it on google play", "for you and your family", "subscribe and save",
             "your personal"]
AWARD_TERMS = ["inc. 5000", "inc 5000", "deloitte fast 500", "g2 leader", "gartner",
               "forrester wave", "best places to work", "fastest-growing"]
INTL_TERMS = ["london", "berlin", "paris", "amsterdam", "dublin", "singapore", "sydney",
              "toronto", "tokyo", "sao paulo", "são paulo", "emea", "apac", "latam",
              "bangalore", "madrid", "stockholm", "mexico city"]
BOOTSTRAP_TERMS = ["bootstrapped", "self-funded", "profitable since", "customer-funded",
                   "never raised", "profitable and growing"]

NUM = r"(\d{1,3}(?:,\d{3})+|\d+)"
FOUNDED_RE = re.compile(r"(?:founded|established|started|launched|since)\s+(?:in\s+)?((?:19|20)\d{2})")
SERIES_RE = re.compile(r"series\s+([a-h])\b")
RAISED_RE = re.compile(r"(?:raised|raises|secured|closed|closes|announces)\s+(?:a\s+|an\s+)?"
                       r"\$\s?(\d+(?:\.\d+)?)\s?(m|mm|million|b|bn|billion)\b")
EMP_RE = re.compile(NUM + r"\+?\s+(?:employees|people|team members|teammates|staff|person team)")
TEAM_OF_RE = re.compile(r"team of\s+(?:over\s+|more than\s+)?" + NUM)
CUST_RE = re.compile(NUM + r"\+?\s+(?:customers|companies|businesses|organizations|clients|"
                     r"practices|firms|teams|locations|agencies|contractors|clinics)")
COPYRIGHT_RE = re.compile(r"(?:©|\(c\)|copyright)\s*(?:(?:19|20)\d{2}\s*[-–]\s*)?((?:19|20)\d{2})")

ROLE_RE = re.compile(
    r"\b(?:(?:senior|sr\.?|staff|principal|lead|junior|head of|vp of|director of)\s+)?"
    r"(?:[a-z&/-]+\s){0,3}?(?:engineer|developer|designer|manager|executive|representative|"
    r"analyst|scientist|specialist|director|architect|recruiter|accountant|consultant|"
    r"associate|coordinator)\b")
GTM_ROLE = re.compile(r"account executive|sales|sdr|bdr|business development|customer success|"
                      r"account manager|partnership|revenue|marketing|solutions? consultant|"
                      r"implementation|onboarding")
ENG_ROLE = re.compile(r"engineer|developer|sre|devops|data scientist|architect|machine learning")

SERIES_LETTERS = "abcdefgh"

# Signals used as model features, in a fixed order.
SIGNAL_COLUMNS = [
    "n_pages", "log_words", "has_pricing_page", "has_careers_page", "has_customers_page",
    "has_security_page", "has_integrations_page", "plg_score", "sales_led_score",
    "enterprise_score", "api_score", "b2c_score", "award_score", "intl_score", "bootstrapped",
    "company_age", "max_series", "log_raised_usd_m", "log_employees", "log_customers",
    "log_open_jobs", "jobs_per_employee", "gtm_job_share", "eng_job_share", "has_ats",
    "site_staleness_years",
]


def _count(text: str, terms: list[str]) -> int:
    return sum(1 for t in terms if t in text)


def _to_int(s: str) -> int:
    return int(s.replace(",", ""))


def _max_match(regex: re.Pattern, text: str, lo: int, hi: int) -> float:
    vals = [_to_int(m) for m in regex.findall(text)]
    vals = [v for v in vals if lo <= v <= hi]
    return float(max(vals)) if vals else NAN


def _log1p(x: float) -> float:
    return math.log1p(x) if x == x else NAN  # NaN-safe


def job_titles_from_text(careers_text: str) -> list[str]:
    """Fallback when no ATS API is available: pull role-like phrases from careers copy."""
    titles = {m.group(0).strip() for m in ROLE_RE.finditer(careers_text.lower())}
    return sorted(t for t in titles if len(t.split()) >= 2)


def extract_signals(record: dict) -> dict:
    pages = record.get("pages") or []
    kinds = {p.get("kind", "other") for p in pages}
    by_kind: dict[str, str] = {}
    for p in pages:
        by_kind[p.get("kind", "other")] = by_kind.get(p.get("kind", "other"), "") + " " + (p.get("text") or "")
    full = " ".join(((p.get("title") or "") + " " + (p.get("text") or "")) for p in pages).lower()
    words = len(full.split())

    # --- funding / stage --------------------------------------------------------------
    series = [SERIES_LETTERS.index(s) + 1 for s in SERIES_RE.findall(full)]
    max_series = float(max(series)) if series else NAN
    raised = []
    for amt, unit in RAISED_RE.findall(full):
        v = float(amt) * (1000.0 if unit in ("b", "bn", "billion") else 1.0)
        if 0.1 <= v <= 20000:
            raised.append(v)
    raised_m = max(raised) if raised else NAN

    founded = [int(y) for y in FOUNDED_RE.findall(full) if 1950 <= int(y) <= CURRENT_YEAR]
    founded_year = float(min(founded)) if founded else NAN

    employees = _max_match(EMP_RE, full, 2, 300_000)
    team_of = _max_match(TEAM_OF_RE, full, 2, 300_000)
    if employees != employees or (team_of == team_of and team_of > employees):
        employees = team_of
    customers = _max_match(CUST_RE, full, 5, 50_000_000)

    copyright_years = [int(y) for y in COPYRIGHT_RE.findall(full) if int(y) <= CURRENT_YEAR]
    staleness = float(CURRENT_YEAR - max(copyright_years)) if copyright_years else NAN

    # --- hiring -------------------------------------------------------------------------
    jobs = record.get("jobs") or {}
    titles = jobs.get("titles")
    has_ats = 1.0 if jobs.get("ats") else 0.0
    if titles is None:
        titles = job_titles_from_text(by_kind.get("careers", "")) if "careers" in kinds else []
    n_jobs = float(len(titles)) if ("careers" in kinds or has_ats) else NAN
    tl = [t.lower() for t in titles]
    gtm_share = sum(bool(GTM_ROLE.search(t)) for t in tl) / len(tl) if tl else NAN
    eng_share = sum(bool(ENG_ROLE.search(t)) for t in tl) / len(tl) if tl else NAN
    jobs_per_emp = n_jobs / employees if (n_jobs == n_jobs and employees == employees and employees > 0) else NAN

    pricing_text = by_kind.get("pricing", "").lower()
    plg = _count(full, PLG_TERMS) + (1 if re.search(r"\$\d+\s*(?:/|per)\s*(?:mo|month|user|seat)", pricing_text) else 0)
    sales_led = _count(full, SALES_TERMS)

    return {
        "n_pages": float(len(pages)),
        "log_words": _log1p(float(words)),
        "has_pricing_page": float("pricing" in kinds),
        "has_careers_page": float("careers" in kinds),
        "has_customers_page": float("customers" in kinds),
        "has_security_page": float("security" in kinds),
        "has_integrations_page": float("integrations" in kinds),
        "plg_score": float(plg),
        "sales_led_score": float(sales_led),
        "enterprise_score": float(_count(full, ENTERPRISE_TERMS)),
        "api_score": float(_count(full, API_TERMS)),
        "b2c_score": float(_count(full, B2C_TERMS)),
        "award_score": float(_count(full, AWARD_TERMS)),
        "intl_score": float(_count(full, INTL_TERMS)),
        "bootstrapped": float(_count(full, BOOTSTRAP_TERMS) > 0),
        "company_age": float(CURRENT_YEAR - founded_year) if founded_year == founded_year else NAN,
        "max_series": max_series,
        "log_raised_usd_m": _log1p(raised_m),
        "log_employees": _log1p(employees),
        "log_customers": _log1p(customers),
        "log_open_jobs": _log1p(n_jobs),
        "jobs_per_employee": jobs_per_emp,
        "gtm_job_share": gtm_share,
        "eng_job_share": eng_share,
        "has_ats": has_ats,
        "site_staleness_years": staleness,
        # Human-readable fields (reported, not used as features directly)
        "_founded_year": founded_year,
        "_employees": employees,
        "_open_jobs": n_jobs,
        "_series_letter": SERIES_LETTERS[int(max_series) - 1].upper() if max_series == max_series else "",
        "_pricing_model": ("sales-led" if sales_led > plg else "self-serve/PLG" if plg > 0 else "unknown"),
    }


def series_letter_to_int(letter: str | None) -> float:
    if not letter:
        return NAN
    return float(SERIES_LETTERS.index(letter.strip().lower()) + 1)
