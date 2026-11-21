import yaml

from sourcing import pipeline
from sourcing.synth import generate


def test_end_to_end_beats_random():
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["model"].update(n_bags=8, cv_folds=3, explain_top_n=20, svd_components=24)
    recs, seeds, truth = generate(n=600, n_pos=25, n_neg=15, seed=1)
    art = pipeline.train(recs, seeds, cfg)
    cv = art["cv_metrics"]
    assert cv["recall_at_top_20pct"] > 0.5          # random would be 0.2
    ranked = pipeline.score(recs, art)
    assert len(ranked) == len(recs) - len(seeds)     # seeds excluded
    assert ranked["fit_score"].between(0, 100).all()
    assert ranked["why"].iloc[0] != ""
    m = ranked.merge(truth, on="domain")
    assert m.head(30)["true_fit"].mean() > m["true_fit"].mean()


def test_scrape_cli_seed_url_overrides_universe(tmp_path, monkeypatch):
    import sourcing.scrape as sc
    from sourcing import cli
    (tmp_path / "u.csv").write_text("domain,name,url,source\nacme.com,,,x\nbeta.com,Beta,,x\n")
    (tmp_path / "s.csv").write_text("domain,label,url,note\nacme.com,1,https://web.archive.org/web/2022/https://acme.com/,n\n"
                                    "gamma.com,0,,n\n")
    cap = {}
    monkeypatch.setattr(sc, "crawl_universe", lambda u, cfg, out: cap.setdefault("u", u) and 0)
    cli.main(["scrape", "--universe", str(tmp_path / "u.csv"), "--seeds", str(tmp_path / "s.csv"),
              "--out", str(tmp_path / "o.jsonl")])
    got = {d: (n, u) for d, n, u in cap["u"]}
    assert got == {"acme.com": (None, "https://web.archive.org/web/2022/https://acme.com/"),
                   "beta.com": ("Beta", None), "gamma.com": (None, None)}
