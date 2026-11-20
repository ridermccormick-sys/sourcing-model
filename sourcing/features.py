"""Feature construction: website text embedding + thesis similarity + structured signals."""
from __future__ import annotations

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
        self.include_kw = [k.lower() for k in thesis.get("include_keywords", [])]
        self.exclude_kw = [k.lower() for k in thesis.get("exclude_keywords", [])]
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
        feats["thesis_kw_hits"] = [sum(k in t for k in self.include_kw) for t in lowered]
        feats["exclude_kw_hits"] = [sum(k in t for k in self.exclude_kw) for t in lowered]
        sig["_exclude_kw_found"] = [", ".join(k for k in self.exclude_kw if k in t) for t in lowered]

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
