"""Polite public-website crawler for company sourcing.

For each domain it fetches the homepage and a handful of high-signal pages
(about, careers, pricing, customers, security, integrations, product, news). If
the careers page embeds a Greenhouse, Lever, Ashby or Workable board, it pulls
open roles from that vendor's public job-board API.

Politeness: obeys robots.txt, waits between requests to the same host, caches
every response on disk (re-runs don't hit sites again), and never logs in or
bypasses paywalls. Check each site's terms of service and your counsel's
guidance on scraping before running this at scale.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

PAGE_KINDS = {  # order = priority when the page budget is tight
    "careers": ["careers", "jobs", "join-us", "join", "work-with-us", "hiring", "open-positions"],
    "about": ["about", "company", "our-story", "who-we-are", "team"],
    "pricing": ["pricing", "plans"],
    "customers": ["customers", "case-studies", "case-study", "success-stories", "testimonials"],
    "security": ["security", "trust", "compliance"],
    "integrations": ["integrations", "partners", "marketplace", "app-directory"],
    "product": ["product", "platform", "features", "solutions"],
    "news": ["newsroom", "news", "press", "blog"],
}
ATS = {
    "greenhouse": (re.compile(r"(?:boards|job-boards)(?:-api)?\.greenhouse\.io/(?:embed/job_board(?:/js)?\?for=)?([A-Za-z0-9_-]+)"),
                   "https://boards-api.greenhouse.io/v1/boards/{}/jobs", lambda j: [x["title"] for x in j.get("jobs", [])]),
    "lever": (re.compile(r"jobs\.lever\.co/([A-Za-z0-9_-]+)"),
              "https://api.lever.co/v0/postings/{}?mode=json", lambda j: [x.get("text", "") for x in j]),
    "ashby": (re.compile(r"jobs\.ashbyhq\.com/([A-Za-z0-9_.%-]+)"),
              "https://api.ashbyhq.com/posting-api/job-board/{}", lambda j: [x["title"] for x in j.get("jobs", [])]),
    "workable": (re.compile(r"apply\.workable\.com/([A-Za-z0-9_-]+)"),
                 "https://apply.workable.com/api/v1/widget/accounts/{}", lambda j: [x["title"] for x in j.get("jobs", [])]),
}
ATS_IGNORE = {"embed", "js", "api", "v1", "jobs", "j"}


class Crawler:
    def __init__(self, cfg: dict):
        s = cfg["scrape"]
        self.ua = s["user_agent"]
        self.timeout = s["request_timeout_s"]
        self.delay = s["per_host_delay_s"]
        self.max_pages = s["max_pages_per_company"]
        self.fetch_ats = s.get("fetch_ats_jobs", True)
        self.cache = Path(s["cache_dir"])
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.ua, "Accept-Language": "en-US,en;q=0.8"})
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._last_hit: dict[str, float] = {}
        self._lock = threading.Lock()

    # -- low level ------------------------------------------------------------------
    def _allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        with self._lock:
            rp = self._robots.get(host, "missing")
        if rp == "missing":
            rp = robotparser.RobotFileParser()
            try:
                r = self.session.get(f"{urlparse(url).scheme}://{host}/robots.txt", timeout=self.timeout)
                if r.status_code >= 400:
                    rp = None  # no robots.txt -> allowed
                else:
                    rp.parse(r.text.splitlines())
            except requests.RequestException:
                rp = None
            with self._lock:
                self._robots[host] = rp
        return True if rp is None else rp.can_fetch(self.ua, url)

    def _throttle(self, host: str):
        with self._lock:
            wait = self._last_hit.get(host, 0) + self.delay - time.time()
            self._last_hit[host] = max(time.time(), self._last_hit.get(host, 0) + self.delay)
        if wait > 0:
            time.sleep(wait)

    def get(self, url: str, check_robots: bool = True) -> tuple[str | None, str]:
        """Return (body, final_url). Cached on disk."""
        key = self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".json")
        if key.exists():
            c = json.loads(key.read_text())
            return c["body"], c["final_url"]
        if check_robots and not self._allowed(url):
            log.info("robots.txt disallows %s", url)
            return None, url
        self._throttle(urlparse(url).netloc)
        try:
            r = self.session.get(url, timeout=self.timeout, allow_redirects=True)
        except requests.RequestException as e:
            log.debug("fetch failed %s: %s", url, e)
            return None, url
        ctype = r.headers.get("content-type", "")
        body = r.text if r.ok and ("html" in ctype or "json" in ctype) else None
        key.write_text(json.dumps({"body": body, "final_url": r.url, "status": r.status_code}))
        return body, r.url

    # -- parsing --------------------------------------------------------------------
    @staticmethod
    def parse(html: str, base_url: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        links = [urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True)]
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        meta = soup.find("meta", attrs={"name": "description"})
        desc = meta.get("content", "") if meta else ""
        for t in soup(["script", "style", "noscript", "svg", "iframe"]):
            t.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return {"title": title, "text": (desc + " " + text)[:60_000], "links": links}

    @staticmethod
    def classify(url: str) -> str | None:
        path = urlparse(url).path.lower().strip("/")
        if not path or path.count("/") > 1:
            return None
        segs = path.split("/")
        for kind, keys in PAGE_KINDS.items():
            if any(seg == k or seg.startswith(k + "-") or seg.endswith("-" + k) for seg in segs for k in keys):
                return kind
        return None

    def _jobs(self, html_blobs: list[str], links: list[str]) -> dict | None:
        hay = " ".join(html_blobs) + " " + " ".join(links)
        for vendor, (rx, api, extract) in ATS.items():
            for token in rx.findall(hay):
                if token.lower() in ATS_IGNORE:
                    continue
                body, _ = self.get(api.format(token), check_robots=False)
                if not body:
                    continue
                try:
                    return {"ats": vendor, "board": token, "titles": extract(json.loads(body))}
                except (ValueError, KeyError, TypeError, AttributeError):
                    continue
        return None

    # -- company --------------------------------------------------------------------
    def crawl(self, domain: str, name: str | None = None) -> dict:
        rec = {"domain": domain, "name": name or domain.split(".")[0].title(), "pages": [], "jobs": None,
               "fetched_at": datetime.now(timezone.utc).isoformat(), "error": None}
        home_html, home_url = None, None
        for candidate in (f"https://{domain}", f"https://www.{domain}", f"http://{domain}"):
            home_html, home_url = self.get(candidate)
            if home_html:
                break
        if not home_html:
            rec["error"] = "homepage unreachable or disallowed"
            return rec

        home = self.parse(home_html, home_url)
        rec["pages"].append({"url": home_url, "kind": "home", "title": home["title"], "text": home["text"]})
        if not name and home["title"]:
            rec["name"] = re.split(r"\s[|\-–—:]\s", home["title"])[0].strip()[:80] or rec["name"]

        site = urlparse(home_url).netloc.removeprefix("www.")
        chosen: dict[str, str] = {}
        for link in home["links"]:
            u = urlparse(link)
            if u.netloc.removeprefix("www.") != site:
                continue
            kind = self.classify(link)
            if kind and kind not in chosen:
                chosen[kind] = link.split("#")[0]
        ordered = [k for k in PAGE_KINDS if k in chosen][: self.max_pages - 1]

        html_blobs, all_links = [home_html], list(home["links"])
        for kind in ordered:
            html, final = self.get(chosen[kind])
            if not html:
                continue
            p = self.parse(html, final)
            rec["pages"].append({"url": final, "kind": kind, "title": p["title"], "text": p["text"]})
            if kind == "careers":
                html_blobs.append(html)
                all_links += p["links"]
        if self.fetch_ats:
            rec["jobs"] = self._jobs(html_blobs, all_links)
        return rec


def crawl_universe(universe: list[tuple[str, str | None]], cfg: dict, out_path: str | Path,
                   skip_existing: bool = True) -> int:
    out_path = Path(out_path)
    done = set()
    if skip_existing and out_path.exists():
        done = {json.loads(l)["domain"] for l in out_path.read_text().splitlines() if l.strip()}
    todo = [(d, n) for d, n in universe if d not in done]
    log.info("crawling %d companies (%d already done)", len(todo), len(done))
    crawler = Crawler(cfg)
    n_ok = 0
    with open(out_path, "a") as f, ThreadPoolExecutor(cfg["scrape"]["workers"]) as ex:
        futs = {ex.submit(crawler.crawl, d, n): d for d, n in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:  # never let one site kill the run
                rec = {"domain": futs[fut], "pages": [], "jobs": None, "error": repr(e)}
            n_ok += rec["error"] is None
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                log.info("  %d/%d crawled (%d ok)", i, len(todo), n_ok)
    return n_ok
