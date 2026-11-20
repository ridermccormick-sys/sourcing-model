import json

from sourcing.scrape import Crawler

CFG = {"scrape": {"user_agent": "test", "request_timeout_s": 1, "per_host_delay_s": 0,
                  "max_pages_per_company": 8, "fetch_ats_jobs": True, "cache_dir": None}}

SITE = {
    "https://acme.com": """<html><head><title>Acme | Construction software</title>
        <meta name="description" content="Operations platform for contractors"></head><body>
        <nav><a href="/about">About</a><a href="/careers">Careers</a><a href="/pricing">Pricing</a>
        <a href="https://twitter.com/acme">tw</a><a href="/blog/2024/some-post">post</a></nav>
        <p>Request a demo</p><script>var x=1</script></body></html>""",
    "https://acme.com/about": "<html><body>Founded in 2015. Team of 120 people.</body></html>",
    "https://acme.com/careers": '<html><body><iframe src="https://boards.greenhouse.io/embed/job_board?for=acme"></iframe></body></html>',
    "https://acme.com/pricing": "<html><body>Contact sales</body></html>",
    "https://boards-api.greenhouse.io/v1/boards/acme/jobs": json.dumps({"jobs": [{"title": "Account Executive"}, {"title": "Engineer"}]}),
}


def test_crawl_with_mocked_http(tmp_path, monkeypatch):
    cfg = {"scrape": dict(CFG["scrape"], cache_dir=str(tmp_path))}
    c = Crawler(cfg)
    monkeypatch.setattr(c, "get", lambda url, check_robots=True: (SITE.get(url), url))
    r = c.crawl("acme.com")
    kinds = [p["kind"] for p in r["pages"]]
    assert r["error"] is None
    assert kinds[0] == "home" and {"about", "careers", "pricing"} <= set(kinds)
    assert "var x" not in r["pages"][0]["text"]            # scripts stripped
    assert "Operations platform" in r["pages"][0]["text"]  # meta description kept
    assert r["name"] == "Acme"
    assert r["jobs"] == {"ats": "greenhouse", "board": "acme", "titles": ["Account Executive", "Engineer"]}


def test_classify():
    assert Crawler.classify("https://a.com/careers") == "careers"
    assert Crawler.classify("https://a.com/company/about-us") == "about"
    assert Crawler.classify("https://a.com/blog/2024/a-long-post") is None
    assert Crawler.classify("https://a.com/") is None


WB = "https://web.archive.org/web/20230601000000/https://visuallease.com"
ARCHIVE = {
    WB: """<html><title>Visual Lease | Lease accounting software</title><body>
        <a href="/web/20230601000000/https://visuallease.com/about">About</a>
        <a href="/web/20230601000000/https://visuallease.com/pricing">Pricing</a>
        <a href="/web/20230601000000/https://twitter.com/vl">tw</a>
        Lease administration software. Request a demo.</body></html>""",
    WB + "/about": "<html><body>Founded in 2001.</body></html>",
    WB + "/pricing": "<html><body>Contact sales</body></html>",
}


def test_wayback_snapshot_for_acquired_company(tmp_path, monkeypatch):
    c = Crawler({"scrape": dict(CFG["scrape"], cache_dir=str(tmp_path))})
    monkeypatch.setattr(c, "get", lambda url, check_robots=True: (ARCHIVE.get(url), url))
    r = c.crawl("visuallease.com", start_url=WB)
    assert r["redirected_to"] is None
    assert {p["kind"] for p in r["pages"]} == {"home", "about", "pricing"}


def test_redirect_to_acquirer_is_flagged(tmp_path, monkeypatch):
    c = Crawler({"scrape": dict(CFG["scrape"], cache_dir=str(tmp_path))})
    monkeypatch.setattr(c, "get", lambda url, check_robots=True:
                        ("<html><title>CoStar</title><body>hi</body></html>", "https://www.costar.com/"))
    r = c.crawl("visuallease.com")
    assert r["redirected_to"] == "costar.com"
