"""Feature construction: website text embedding + thesis similarity + structured signals."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .signals import SIGNAL_COLUMNS, extract_signals

MAX_CHARS_PER_PAGE = 20_000

# Human-readable names for explanation output. Text components are grouped.
FEATURE_LABELS = {
    "text": "Business description resembles your seed companies",
    "thesis_similarity": "Website language matches thesis",
    "thesis_kw_hits": "Thesis keywords on site",
    "exclude_kw_hits": "Off-thesis keywords on site",
    "has_pricing_page": "Has pricing page",
    "has_careers_page": "Has careers page",
    "has_customers_page": "Has customers / case-study page",
    "has_security_page": "Has security / trust page",
    "has_integrations_page": "Has integrations ecosystem",
    "plg_score": "Self-serve / free-trial motion",
    "sales_led_score": "Sales-led motion (demo / contact sales)",
    "enterprise_score": "Enterprise readiness (SOC 2, SSO, HIPAA…)",
    "api_score": "Developer / API surface",
    "b2c_score": "Consumer signals",
    "award_score": "Growth awards (Inc. 5000, G2…)",
    "intl_score": "International footprint",
    "bootstrapped": "Bootstrapped / profitable",
    "company_age": "Company age",
    "max_series": "Latest disclosed funding round",
    "log_raised_usd_m": "Disclosed capital raised",
    "log_employees": "Headcount",
    "log_customers": "Customer count claimed",
    "log_open_jobs": "Open roles",
    "jobs_per_employee": "Hiring intensity (open roles / headcount)",
    "gtm_job_share": "Share of hiring in sales / CS / marketing",
    "eng_job_share": "Share of hiring in engineering",
    "has_ats": "Uses a modern ATS",
    "site_staleness_years": "Website staleness",
    "n_pages": "Site depth",
    "log_words": "Site content volume",
}


# How to phrase a driver when the company's site simply doesn't disclose it.
# E.g. "no disclosed funding" helps a bootstrapped thesis; that is not "capital raised".
MISSING_LABELS = {
    "log_raised_usd_m": "No VC funding disclosed",
    "max_series": "No funding round disclosed",
    "company_age": "Founding year not disclosed",
    "log_employees": "Headcount not disclosed",
    "log_customers": "No customer count claimed",
    "log_open_jobs": "No careers page found",
    "jobs_per_employee": "Hiring intensity unknown",
    "gtm_job_share": "No open roles listed",
    "eng_job_share": "No open roles listed",
    "site_staleness_years": "No copyright year on site",
}


def keyword_regex(keywords: list[str]) -> list[tuple[str, re.Pattern]]:
    """Whole-word matching: 'crypto' must not match 'cryptographic'."""
    return [(k, re.compile(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])")) for k in keywords]


def display_name(name: str | None, domain: str) -> str:
    """Scraped names often come from <title> ("Home", "Golf Software by Teesnap",
    "Alkmist: External Collaboration Portal"). Reduce to the company name."""
    stem = domain.split(".")[0]
    fallback = stem[:1].upper() + stem[1:]
    if not name:
        return fallback
    n = re.sub(r"[®™©]", "", str(name)).strip()
    key = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
    segs = [x.strip() for x in re.split(r"\s[|\-–—:•]\s|:\s|\s\|", n) if x.strip()]
    for seg in segs:  # a segment that matches the domain is the name
        k = key(seg)
        if len(k) >= 3 and (k in stem or stem in k) and len(seg.split()) <= 4:
            words = seg.split()  # "Govenda Board Success Platform" -> "Govenda"
            for i in range(1, len(words) + 1):
                if stem in key(" ".join(words[:i])):
                    return " ".join(words[:i])
            return seg
    m = re.search(r"\bby\s+([A-Z][\w.&'-]+(?:\s[A-Z][\w.&'-]+)?)\s*$", n)
    if m:
        return m.group(1)
    first = segs[0] if segs else n
    generic = key(first) in {"home", "homenew", "welcome", "homepage", "index", "untitled"}
    if generic or len(first) > 30 or len(first.split()) > 4:
        return fallback
    return first


def company_text(record: dict) -> str:
    parts = [record.get("name") or ""]
    for p in record.get("pages") or []:
        parts.append(p.get("title") or "")
        parts.append((p.get("text") or "")[:MAX_CHARS_PER_PAGE])
    return " ".join(parts)


class Featurizer:
    """Fit on the whole universe (unsupervised, no labels used), then transform."""

    def __init__(self, thesis: dict, svd_components: int = 48, random_state: int = 0):
        self.thesis_text = thesis.get("description", "")
        self.include_kw = keyword_regex([k.lower() for k in thesis.get("include_keywords", [])])
        self.exclude_kw = keyword_regex([k.lower() for k in thesis.get("exclude_keywords", [])])
        self.svd_components = svd_components
        self.random_state = random_state

    def fit(self, records: list[dict]) -> "Featurizer":
        texts = [company_text(r) for r in records]
        self.tfidf = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), sublinear_tf=True,
                                     min_df=2, max_df=0.6, stop_words="english")
        tf = self.tfidf.fit_transform(texts)
        k = max(2, min(self.svd_components, tf.shape[1] - 1, tf.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.random_state).fit(tf)
        self.thesis_vec = self.tfidf.transform([self.thesis_text])
        return self

    def text_matrix(self, records: list[dict]):
        """L2-normalised TF-IDF rows (used for nearest-comparable lookups)."""
        return self.tfidf.transform([company_text(r) for r in records])

    def transform(self, records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
        texts = [company_text(r) for r in records]
        tf = self.tfidf.transform(texts)
        emb = normalize(self.svd.transform(tf))
        text_df = pd.DataFrame(emb, columns=[f"text_{i:02d}" for i in range(emb.shape[1])])

        lowered = [t.lower() for t in texts]
        sig = pd.DataFrame([extract_signals(r) for r in records])
        feats = sig[SIGNAL_COLUMNS].copy()
        feats["thesis_similarity"] = np.asarray((tf @ self.thesis_vec.T).todense()).ravel()
        feats["thesis_kw_hits"] = [sum(bool(rx.search(t)) for _, rx in self.include_kw) for t in lowered]
        found = [[k for k, rx in self.exclude_kw if rx.search(t)] for t in lowered]
        feats["exclude_kw_hits"] = [len(f) for f in found]
        sig["_exclude_kw_found"] = [", ".join(f) for f in found]

        X = pd.concat([text_df, feats.reset_index(drop=True)], axis=1).astype(float)
        return X, sig

    @staticmethod
    def feature_groups(columns: list[str]) -> dict[str, list[int]]:
        """Map readable group -> column indices (all text_* columns form one group)."""
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(columns):
            key = "text" if c.startswith("text_") else c
            groups.setdefault(key, []).append(i)
        return groups
