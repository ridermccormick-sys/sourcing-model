# Company sourcing model (growth equity)

Ranks a universe of companies by **fit to your investment thesis** using only
public website data. It learns from a short list of companies your firm has
invested in or pursued.

```
universe.csv ──scrape──▶ companies.jsonl ──features──▶ PU model ──▶ ranked.csv
seeds.csv ───────────────────────────────────────────▲              (score, flags, "why", closest comp)
```

## Quick start

```bash
pip install -r requirements.txt
python -m sourcing.cli demo          # full pipeline on 3,000 synthetic companies (~1 min)
python -m pytest -q
```

## Using real data

1. **Edit `config.yaml`**: the thesis description, include/exclude keywords and hard filters (founding year, headcount, latest round).
2. **Build `seeds.csv`** (`domain,label,note`):
   * `1`: portfolio companies, companies you term-sheeted or ran deep diligence on, and companies you lost but would have backed. Aim for **30–100**. More, and more varied, positives are the biggest lever on quality.
   * `0`: companies you passed on **because they were off-thesis**. Leave out passes for price, competition or timing. Those were good fits, so labeling them 0 would teach the model the wrong thing.
   * Optional `url` column: where to start crawling, when that differs from `https://<domain>`. Use it for a subdomain (`https://site.acme.com/`) or, for an **acquired company whose domain now redirects to the buyer**, a Wayback Machine snapshot from before the deal: `https://web.archive.org/web/20220101/https://acme.com/`. The scraper logs a warning for every domain that redirects to a different site. Check that list after each scrape. The same column works in `universe.csv`.
   * `seeds*.csv` and `universe*.csv` are git-ignored, because they hold confidential deal data.
3. **Build `universe.csv`** (`domain,name`) from any lists of candidate companies: G2/Capterra category pages, conference exhibitor and sponsor lists, industry association member directories, Inc. 5000 / Deloitte Fast 500, state SaaS lists, YC and accelerator directories, and portfolio pages of seed and Series A funds. More breadth is better; the model does the filtering.
4. Run:
   ```bash
   python -m sourcing.cli scrape --universe universe.csv --seeds seeds.csv --out data/companies.jsonl
   python -m sourcing.cli tune   --seeds seeds.csv          # optional: pick model settings for your data
   python -m sourcing.cli train  --companies data/companies.jsonl --seeds seeds.csv
   python -m sourcing.cli score  --companies data/companies.jsonl --out output/ranked.csv
   ```

## How it works

**Scraping** (`sourcing/scrape.py`). Fetches the homepage plus up to 7 high-signal pages (careers, about, pricing, customers, security, integrations, product, news). Open roles come from the public Greenhouse / Lever / Ashby / Workable job-board APIs when a careers page embeds one. The crawler obeys robots.txt, rate-limits each host, caches everything on disk (re-runs are free and resumable), and runs 8 workers in parallel.

**Signals** (`sourcing/signals.py`) are extracted from the page text:

| Area | Signals |
|---|---|
| Stage | founding year, stated headcount, latest disclosed series, capital raised, bootstrapped/profitable |
| Momentum | open roles, roles per employee, % of hiring in sales/CS/marketing vs engineering, growth awards, site freshness |
| Business model | sales-led vs self-serve, enterprise readiness (SOC 2/SSO/HIPAA…), API surface, consumer signals, customer count, international offices |
| What they do | TF-IDF → SVD embedding of all site text, similarity to the thesis description, thesis / off-thesis keyword hits |

A missing signal stays missing (NaN), not zero. "Doesn't say how many employees it has" is different from "has zero employees."

**Model** (`sourcing/model.py`). Uses positive-unlabeled bagging. Unlabeled companies aren't negatives: many great fits just haven't been looked at yet. Each of 40 bags trains positives against a random sample of the unlabeled universe, with explicit negatives added at extra weight. The classifier in each bag blends gradient boosting with regularized logistic regression, which stays stable when the training set is small. Scores are averaged across bags.

**Nearest-seed similarity**. The classifier learns one "typical portfolio company". When a portfolio spans many unrelated niches, a second scorer helps: similarity to the *single closest* seed company. `knn_weight` in `config.yaml` blends the two (0 = classifier only, 1 = nearest seed only).

**Tuning**. `python -m sourcing.cli tune --seeds seeds.csv` runs the self-test for several settings (classifier type, text-embedding size, `knn_weight`) on your own data and prints the best one to put in `config.yaml`. Re-run it whenever the seed list changes a lot.

**Evaluation**. Cross-validation hides a fold of your positives *inside* the unlabeled pool, retrains, and checks where they land. This measures what matters: would the model have surfaced a company you liked if you hadn't already found it? Reported as recall in the top 5/10/20% (random = 5/10/20%).

**Output** (`ranked.csv`), one row per company:
* `fit_score`: percentile within the universe (0–100)
* `eligible` / `flags`: hard-filter violations, such as Series D+, outside the headcount band, or off-thesis keywords. Flagged companies are ranked below eligible ones but kept in the file.
* `why` / `biggest_drag`: per-company drivers, computed by occlusion (each feature group is reset to the universe median and the score change is measured)
* `closest_seed_company`: the most similar portfolio/pursued company. Useful as a talking point in outreach.

Seed companies are left out of the ranking. Add `--include-seeds` to see them.

## Demo results (synthetic data)

```
recall in top 5% / 10% / 20% : 69% / 84% / 93%   (random = 5% / 10% / 20%)
precision@50 vs hidden truth : model 52%   thesis-keyword baseline 22%   random 4%
```
Synthetic data is easier than real data. Treat these numbers as a check that the pipeline works, not as a performance estimate.

## Limitations and next steps

* **Snapshot only.** The strongest growth signal is *change over time*. Re-scrape monthly into dated files and add deltas (open roles Δ, new enterprise features, new offices) as features.
* **Self-reported data.** Websites over- and under-state things. Headcount from LinkedIn and traffic from SimilarWeb, if you license them, would add a lot. Add them as extra columns in `signals.py`.
* **JavaScript-heavy sites** return little text to `requests`. For those, swap in a Playwright fetch inside `Crawler.get`.
* **Feedback loop.** Every week, have the deal team mark top-ranked companies as 1 (worth a call) or 0 (off-thesis), append them to `seeds.csv`, and retrain. This is the fastest way to improve the model.
* **Legal.** Respect each site's terms of service. The crawler identifies itself, honors robots.txt and throttles itself, but check your scraping policy with counsel.
