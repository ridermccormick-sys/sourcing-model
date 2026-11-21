"""Command line entry point.

    python -m sourcing.cli demo                       # end-to-end on synthetic data
    python -m sourcing.cli scrape --universe universe.csv --out data/companies.jsonl
    python -m sourcing.cli train  --companies data/companies.jsonl --seeds seeds.csv
    python -m sourcing.cli score  --companies data/companies.jsonl --out output/ranked.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from . import pipeline

log = logging.getLogger("sourcing")


def _cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _print_cv(cv: dict | None):
    if not cv:
        return
    print("\nCross-validation (seed positives hidden among unlabeled companies, then re-ranked):")
    print(f"  positives={cv['n_positives']}  explicit negatives={cv['n_negatives']}  unlabeled={cv['n_unlabeled']}")
    print(f"  median percentile of hidden positives : {cv['heldout_positive_median_percentile']:.1%}")
    print(f"  recall in top 5% / 10% / 20%          : {cv['recall_at_top_5pct']:.0%} / "
          f"{cv['recall_at_top_10pct']:.0%} / {cv['recall_at_top_20pct']:.0%}"
          "   (random ranking = 5% / 10% / 20%)")
    if "heldout_negative_median_percentile" in cv:
        print(f"  median percentile of hidden negatives : {cv['heldout_negative_median_percentile']:.1%}  (lower is better)")


def cmd_scrape(a):
    from .scrape import crawl_universe
    cfg = _cfg(a.config)
    def rows(df):
        col = lambda c: df[c] if c in df else [None] * len(df)
        txt = lambda v: v.strip() if isinstance(v, str) and v.strip() else None
        return [(pipeline.normalize_domain(d), txt(n), txt(url)) for d, n, url in zip(df["domain"], col("name"), col("url"))]

    merged: dict[str, tuple] = {}
    for d, n, url in rows(pd.read_csv(a.universe, comment="#")):
        merged.setdefault(d, (d, n, url))
    if a.seeds:  # seeds must be scraped too; a seed's name/url wins over the universe row
        for d, n, url in rows(pipeline.load_seeds(a.seeds)):
            _, n0, u0 = merged.get(d, (d, None, None))
            merged[d] = (d, n or n0, url or u0)
    universe = list(merged.values())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ok = crawl_universe(universe, cfg, a.out)
    print(f"done: {ok} companies crawled successfully -> {a.out}")


def cmd_train(a):
    cfg = _cfg(a.config)
    recs = [r for r in pipeline.load_records(a.companies) if r.get("pages")]
    art = pipeline.train(recs, pipeline.load_seeds(a.seeds), cfg, run_cv=not a.no_cv)
    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, a.model)
    _print_cv(art["cv_metrics"])
    print(f"\nmodel saved -> {a.model}")


def cmd_score(a):
    art = joblib.load(a.model)
    recs = [r for r in pipeline.load_records(a.companies) if r.get("pages")]
    df = pipeline.score(recs, art, include_seeds=a.include_seeds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    cols = ["rank", "name", "domain", "fit_score", "employees", "open_jobs", "closest_seed_company", "why"]
    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        print(df[cols].head(a.show).to_string(index=False))
    print(f"\n{len(df)} companies ranked -> {a.out}")


def cmd_demo(a):
    from scipy.stats import spearmanr
    from .synth import generate

    cfg = _cfg(a.config)
    d = Path(a.workdir)
    d.mkdir(parents=True, exist_ok=True)
    print(f"generating synthetic universe of {a.n} companies...")
    recs, seeds, truth = generate(n=a.n, seed=a.seed)
    with open(d / "companies.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    seeds.to_csv(d / "seeds.csv", index=False)
    truth.to_csv(d / "truth.csv", index=False)

    art = pipeline.train(recs, seeds, cfg)
    joblib.dump(art, d / "model.joblib")
    _print_cv(art["cv_metrics"])

    ranked = pipeline.score(recs, art)
    ranked.to_csv(d / "ranked.csv", index=False)

    # Compare against the hidden ground truth the model never saw.
    m = ranked.merge(truth, on="domain")
    top_true = set(truth.loc[truth["true_fit"] >= truth["true_fit"].quantile(0.95), "domain"]) - set(seeds["domain"])
    rho = spearmanr(m["model_probability"], m["true_fit"]).statistic
    thesis_only = pipeline.Featurizer(cfg["thesis"]).fit(recs)
    X, _ = thesis_only.transform(recs)
    base = pd.DataFrame({"domain": [r["domain"] for r in recs], "s": X["thesis_similarity"]}).merge(m[["domain", "true_fit"]])
    rho_base = spearmanr(base["s"], base["true_fit"]).statistic
    p50 = len(set(ranked["domain"].head(50)) & top_true) / 50
    p50_base = len(set(base.sort_values("s", ascending=False)["domain"].head(50)) & top_true) / 50
    p50_rand = len(top_true) / len(ranked)
    print("\nAgainst hidden ground truth (synthetic data only):")
    print(f"  Spearman(score, true fit)              : model {rho:.2f}   thesis-keyword baseline {rho_base:.2f}")
    print(f"  precision@50 for true top-5% companies : model {p50:.0%}   thesis-keyword baseline {p50_base:.0%}   random {p50_rand:.0%}")
    print("  archetype mix of model top 50          :",
          {str(k): int(v) for k, v in m.sort_values("rank").head(50)["archetype"].value_counts().items()})

    cols = ["rank", "name", "fit_score", "employees", "open_jobs", "latest_series", "eligible", "why"]
    with pd.option_context("display.max_colwidth", 110, "display.width", 250):
        print("\nTop 15:")
        print(ranked[cols].head(15).to_string(index=False))
    print(f"\nfiles written to {d}/ (companies.jsonl, seeds.csv, model.joblib, ranked.csv)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sourcing", description="Growth-equity company sourcing model")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scrape", help="crawl company websites")
    s.add_argument("--universe", required=True, help="CSV with a 'domain' column (optional 'name')")
    s.add_argument("--seeds", help="seeds CSV; seed domains are crawled too")
    s.add_argument("--out", default="data/companies.jsonl")
    s.set_defaults(func=cmd_scrape)

    t = sub.add_parser("train", help="train the thesis-fit model")
    t.add_argument("--companies", default="data/companies.jsonl")
    t.add_argument("--seeds", required=True)
    t.add_argument("--model", default="models/model.joblib")
    t.add_argument("--no-cv", action="store_true")
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("score", help="rank companies")
    c.add_argument("--companies", default="data/companies.jsonl")
    c.add_argument("--model", default="models/model.joblib")
    c.add_argument("--out", default="output/ranked.csv")
    c.add_argument("--include-seeds", action="store_true")
    c.add_argument("--show", type=int, default=20)
    c.set_defaults(func=cmd_score)

    d = sub.add_parser("demo", help="run end-to-end on synthetic data")
    d.add_argument("--n", type=int, default=3000)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--workdir", default="output/demo")
    d.set_defaults(func=cmd_demo)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")
    a.func(a)


if __name__ == "__main__":
    main()
