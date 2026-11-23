"""Train and score: ties features, PU model, filters and explanations together."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .features import FEATURE_LABELS, MISSING_LABELS, Featurizer, display_name
from .model import PUBagger, cross_validate_pu, explain
from .signals import series_letter_to_int

log = logging.getLogger(__name__)


def load_records(path: str | Path) -> list[dict]:
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    seen, out = set(), []
    for r in recs:  # keep the latest record per domain
        r["domain"] = normalize_domain(r["domain"])
        if r["domain"] in seen:
            out = [x for x in out if x["domain"] != r["domain"]]
        seen.add(r["domain"])
        out.append(r)
    return out


def normalize_domain(d: str) -> str:
    d = str(d).strip().lower()
    for p in ("https://", "http://"):
        d = d.removeprefix(p)
    return d.removeprefix("www.").split("/")[0]


def load_seeds(path: str | Path) -> pd.DataFrame:
    seeds = pd.read_csv(path, comment="#", skip_blank_lines=True)
    seeds["domain"] = seeds["domain"].map(normalize_domain)
    seeds["label"] = seeds["label"].astype(int)
    bad = set(seeds["label"]) - {0, 1}
    if bad:
        raise ValueError(f"seed labels must be 0 or 1, found {bad}")
    return seeds.drop_duplicates("domain", keep="last")


def _labels(records: list[dict], seeds: pd.DataFrame) -> np.ndarray:
    lab = dict(zip(seeds["domain"], seeds["label"]))
    return np.array([lab.get(r["domain"], -1) for r in records])


def _model_params(cfg: dict) -> dict:
    m = cfg["model"]
    return dict(n_bags=m["n_bags"], unlabeled_ratio=m["unlabeled_ratio"],
                negative_weight=m["negative_weight"], classifier=m["classifier"])


def train(records: list[dict], seeds: pd.DataFrame, cfg: dict, run_cv: bool = True) -> dict:
    y = _labels(records, seeds)
    missing = set(seeds["domain"]) - {r["domain"] for r in records}
    if missing:
        log.warning("%d seed domains have no scraped record (scrape them too): %s",
                    len(missing), sorted(missing)[:10])
    log.info("positives=%d explicit negatives=%d unlabeled=%d",
             (y == 1).sum(), (y == 0).sum(), (y == -1).sum())

    rs = cfg["model"]["random_state"]
    feat = Featurizer(cfg["thesis"], cfg["model"]["svd_components"], rs).fit(records)
    X, _ = feat.transform(records)

    cv = None
    if run_cv:
        log.info("cross-validating (hiding positives in the unlabeled pool)...")
        cv = cross_validate_pu(X.values, y, n_folds=cfg["model"]["cv_folds"], random_state=rs,
                               **_model_params(cfg))
    model = PUBagger(random_state=rs, **_model_params(cfg)).fit(X.values, y)

    pos = [r for r, l in zip(records, y) if l == 1]
    return {
        "featurizer": feat,
        "model": model,
        "columns": list(X.columns),
        "background": np.nanmedian(X.values[y == -1], axis=0),
        "seed_labels": dict(zip(seeds["domain"], seeds["label"])),
        "seed_pos_names": [display_name(r.get("name"), r["domain"]) for r in pos],
        "seed_pos_tfidf": feat.text_matrix(pos),
        "cv_metrics": cv,
        "config": cfg,
    }


def _flags(sig: pd.Series, filters: dict) -> list[str]:
    f = []
    fy, emp, ser = sig["_founded_year"], sig["_employees"], sig["max_series"]
    if fy == fy and not (filters.get("founded_year_min", 0) <= fy <= filters.get("founded_year_max", 9999)):
        f.append(f"founded {int(fy)}")
    if emp == emp and not (filters.get("employees_min", 0) <= emp <= filters.get("employees_max", 10**9)):
        f.append(f"~{int(emp)} employees")
    max_ser = series_letter_to_int(filters.get("max_series"))
    if ser == ser and max_ser == max_ser and ser > max_ser:
        f.append(f"Series {sig['_series_letter']} (beyond {filters['max_series']})")
    if sig["_exclude_kw_found"]:
        f.append(f"off-thesis keywords: {sig['_exclude_kw_found']}")
    return f


def score(records: list[dict], artifact: dict, include_seeds: bool = False) -> pd.DataFrame:
    cfg = artifact["config"]
    feat: Featurizer = artifact["featurizer"]
    model: PUBagger = artifact["model"]
    X, sig = feat.transform(records)
    X = X.reindex(columns=artifact["columns"])
    p = model.predict(X.values)

    seed_labels = artifact["seed_labels"]
    in_seed = np.array([r["domain"] in seed_labels for r in records])
    # percentile among non-seed companies -> "fit score" 0-100
    ref = np.sort(p[~in_seed]) if (~in_seed).any() else np.sort(p)
    fit_score = 100.0 * np.searchsorted(ref, p, side="right") / len(ref)

    df = pd.DataFrame({
        "domain": [r["domain"] for r in records],
        "name": [display_name(r.get("name"), r["domain"]) for r in records],
        "fit_score": fit_score.round(1),
        "model_probability": p.round(4),
        "seed_label": [seed_labels.get(r["domain"], "") for r in records],
        "founded_year": sig["_founded_year"],
        "employees": sig["_employees"],
        "open_jobs": sig["_open_jobs"],
        "gtm_job_share": sig["gtm_job_share"].round(2),
        "latest_series": sig["_series_letter"],
        "pricing_model": sig["_pricing_model"],
        "enterprise_score": sig["enterprise_score"],
    })
    flags = [_flags(sig.iloc[i], cfg.get("filters", {})) for i in range(len(records))]
    df["eligible"] = [not f for f in flags]
    df["flags"] = ["; ".join(f) for f in flags]

    # nearest seed comparable (by website text)
    sims = cosine_similarity(feat.text_matrix(records), artifact["seed_pos_tfidf"])
    sims[sims > 0.999] = -1  # a seed company is not its own comparable
    best = sims.argmax(axis=1)
    df["closest_seed_company"] = [artifact["seed_pos_names"][j] for j in best]
    df["closest_seed_similarity"] = sims[np.arange(len(records)), best].round(3)

    if not include_seeds:
        df = df[~in_seed]
    df = df.sort_values(["eligible", "model_probability"], ascending=[False, False])

    # explanations for the top N
    n = min(cfg["model"]["explain_top_n"], len(df))
    top_idx = df.index[:n].to_numpy()
    groups = feat.feature_groups(artifact["columns"])
    expl = explain(model, X.values[top_idx], groups, artifact["background"])
    col = {c: i for i, c in enumerate(artifact["columns"])}

    def label(g: str, row: int) -> str:
        if g in col and X.values[row, col[g]] != X.values[row, col[g]]:  # NaN = not on the site
            return MISSING_LABELS.get(g, f"{FEATURE_LABELS.get(g, g)}: not disclosed")
        return FEATURE_LABELS.get(g, g)

    why, drag = [""] * len(df), [""] * len(df)
    for k, (ups, dn) in enumerate(expl):
        row = top_idx[k]
        why[k] = " | ".join(f"{label(g, row)} (+{d:.2f})" for g, d in ups)
        drag[k] = f"{label(dn[0], row)} ({dn[1]:.2f})" if dn else ""
    df["why"] = why
    df["biggest_drag"] = drag
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df.reset_index(drop=True)
