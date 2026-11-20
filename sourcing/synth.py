"""Synthetic company universe for demos and tests.

Generates records in exactly the format ``scrape.py`` writes (page text + job
titles), plus a hidden ``true_fit`` score. Every downstream step (signal
extraction, features, model) runs unchanged, and the demo can check how well
the model recovers fit that it never sees directly.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

ARCHETYPES = {  # name: (share of universe, thesis base fit)
    "vertical_saas": (0.15, 1.0), "horizontal_saas": (0.20, 0.55), "devtools": (0.12, 0.35),
    "fintech": (0.08, 0.2), "consumer_app": (0.15, 0.05), "dtc_brand": (0.10, 0.0),
    "services": (0.12, 0.1), "hardware": (0.08, 0.15),
}
VERTICALS = ["dental practices", "construction contractors", "property managers", "veterinary clinics",
             "independent insurance agencies", "law firms", "restaurant groups", "home services businesses",
             "freight brokers", "credit unions", "senior living operators", "auto repair shops",
             "field service teams", "school districts", "behavioral health providers", "manufacturers",
             "physical therapy clinics", "trucking fleets", "hotel operators", "pharmacies"]
VERT_PRODUCTS = ["practice management software", "operations platform", "scheduling and billing software",
                 "compliance automation platform", "payments and invoicing platform", "cloud ERP",
                 "workforce management software", "system of record"]
HORIZ = [("HR and payroll platform", "HR teams"), ("marketing automation software", "marketing teams"),
         ("spend management platform", "finance teams"), ("sales engagement platform", "sales teams"),
         ("project management tool", "teams of every size"), ("customer support software", "support teams"),
         ("contract management software", "legal teams"), ("analytics platform", "data teams")]
DEV = ["observability platform", "CI/CD pipeline", "serverless database", "API gateway",
       "feature flag platform", "security scanning tool", "vector database", "developer portal"]
FIN = ["small business lending platform", "corporate card for startups", "buy now pay later service",
       "crypto exchange", "neobank for freelancers", "embedded insurance API"]
CONS = ["meditation app", "dating app", "fitness tracking app", "photo editing app",
        "language learning app", "budgeting app for families", "recipe and meal planning app"]
DTC = ["sustainable sneakers", "clean skincare", "fresh pet food", "hybrid mattresses",
       "premium coffee subscriptions", "athleisure apparel"]
SERV = ["digital marketing agency", "IT consulting firm", "staffing agency", "software development agency",
        "managed IT services provider"]
HW = ["industrial IoT sensors", "commercial drones", "desktop 3D printers", "smart home cameras",
      "warehouse robotics"]
GENERIC = ["We help you save time and focus on what matters.", "Powered by AI and built for scale.",
           "Streamline workflows and boost productivity.", "Our mission is to make work better.",
           "Loved by customers around the world.", "Built by a passionate team.",
           "See why thousands choose us.", "Seamless onboarding and world-class support."]
CITIES_US = ["Austin", "Denver", "Chicago", "Atlanta", "Nashville", "Salt Lake City", "Columbus",
             "Raleigh", "San Francisco", "New York", "Boston", "Seattle", "Indianapolis", "Phoenix"]
CITIES_INTL = ["London", "Berlin", "Toronto", "Sydney", "Dublin", "Amsterdam", "Singapore"]
ROLES_GTM = ["Account Executive", "Senior Account Executive", "Sales Development Representative",
             "Customer Success Manager", "Implementation Specialist", "Product Marketing Manager",
             "Solutions Consultant", "Partnerships Manager", "Enterprise Account Executive"]
ROLES_ENG = ["Senior Software Engineer", "Staff Engineer", "Data Engineer", "Frontend Engineer",
             "Site Reliability Engineer", "Machine Learning Engineer"]
ROLES_OTHER = ["Product Designer", "Financial Analyst", "Recruiter", "Product Manager",
               "Operations Associate", "Store Associate", "Warehouse Coordinator"]
SYL = ["ar", "bo", "ca", "den", "el", "fa", "gri", "hal", "io", "ja", "ka", "lu", "mo", "nex", "or",
       "pa", "qui", "ro", "sa", "tal", "ve", "wi", "xa", "yo", "zen", "tri", "vor", "lin", "mar", "sol"]
SUFFIX = ["", "", "ly", "io", "ify", " Labs", " HQ", " Health", " Works", " Cloud"]


def _name(rng, used):
    while True:
        n = "".join(rng.choice(SYL, size=rng.integers(2, 4))).capitalize() + rng.choice(SUFFIX)
        d = n.lower().replace(" ", "") + ".com"
        if d not in used:
            used.add(d)
            return n, d


def _company(rng, arche, used):
    name, domain = _name(rng, used)
    founded = int(np.clip(rng.normal(2015, 6), 1995, 2024))
    age = max(2026 - founded, 1)
    employees = int(np.clip(math.exp(rng.normal(math.log(8 + 12 * age), 0.9)), 2, 20000))
    growth = rng.normal(0.0, 1.0)  # latent momentum (z-score)
    b2b = arche in ("vertical_saas", "horizontal_saas", "devtools", "fintech")
    boot = rng.random() < (0.3 if arche in ("vertical_saas", "services") else 0.15)
    if boot:
        series = None
    else:
        series = "seed" if employees < 20 else "A" if employees < 80 else "B" if employees < 250 else \
            "C" if employees < 600 else rng.choice(["D", "E"])
    raised = {None: 0, "seed": 3, "A": 15, "B": 45, "C": 110, "D": 250, "E": 500}[series] * rng.uniform(0.6, 1.6)
    ent = int(np.clip(rng.poisson(1 + 2 * (employees > 60) + (arche == "vertical_saas")) if b2b else rng.poisson(0.2), 0, 6))
    sales_led = (rng.random() < {"vertical_saas": 0.85, "horizontal_saas": 0.5, "devtools": 0.3,
                                  "fintech": 0.5}.get(arche, 0.1))
    n_jobs = int(rng.poisson(max(employees * 0.06 * math.exp(0.6 * growth), 0.2)))
    gtm_p = 0.45 if (b2b and sales_led) else 0.2 if b2b else 0.15
    intl = rng.random() < (0.1 + 0.3 * (employees > 200))
    award = rng.random() < 1 / (1 + math.exp(-(growth - 1.2) * 2))

    # --- hidden thesis fit ------------------------------------------------------------
    base = ARCHETYPES[arche][1]
    size_fit = math.exp(-((math.log(employees) - math.log(180)) ** 2) / (2 * 0.8 ** 2))
    cap = 0.5 if (boot or series in ("A", "B")) else -0.8 if series in ("D", "E") else -0.3 if series == "seed" else 0.0
    true_fit = 2.0 * base + 1.2 * size_fit + cap + 0.6 * growth + 0.15 * ent + (0.3 if sales_led and b2b else 0) \
        + (-0.6 if not 2008 <= founded <= 2022 else 0) + rng.normal(0, 0.35)

    # --- website copy ---------------------------------------------------------------
    if arche == "vertical_saas":
        cust = rng.choice(VERTICALS); prod = rng.choice(VERT_PRODUCTS)
        what = f"the {prod} built for {cust}"
        extra = (f"Scheduling, billing, payments, reporting and compliance in one place, designed for the "
                 f"day-to-day workflow of {cust}. Replace spreadsheets and legacy on-premise tools.")
    elif arche == "horizontal_saas":
        prod, cust = HORIZ[rng.integers(len(HORIZ))]
        what = f"the {prod} for {cust}"
        extra = f"Automate workflows, integrate with the tools you already use, and get real-time reporting. Used by {rng.choice(VERTICALS)} and more."
    elif arche == "devtools":
        prod = rng.choice(DEV); cust = "engineering teams"
        what = f"the {prod} for modern {cust}"
        extra = "Ship faster with our open source core, REST API, SDKs for every language and webhooks. Read the API reference and developer docs."
    elif arche == "fintech":
        prod = rng.choice(FIN); cust = "businesses"
        what = f"a {prod}"
        extra = "Fast approvals, transparent fees, and payments that just work. " + ("Trade bitcoin and other crypto assets on the blockchain." if "crypto" in prod else "")
    elif arche == "consumer_app":
        prod = rng.choice(CONS); cust = "people"
        what = f"the {prod} millions love"
        extra = "Your personal coach in your pocket, for you and your family. Download on the App Store or get it on Google Play."
    elif arche == "dtc_brand":
        prod = rng.choice(DTC); cust = "customers"
        what = f"{prod} made better"
        extra = "Shop now and enjoy free shipping on every order. Add to cart and subscribe and save."
    elif arche == "services":
        prod = rng.choice(SERV); cust = "clients"
        what = f"a {prod}"
        extra = "Our experts deliver strategy, implementation and managed services for clients across industries."
    else:
        prod = rng.choice(HW); cust = "operators"
        what = f"{prod} for industry"
        extra = "Rugged hardware, cloud dashboard, and installation support. Request a quote today."

    gen = " ".join(rng.choice(GENERIC, size=3, replace=False))
    hq = rng.choice(CITIES_US)
    customers = int(employees * rng.uniform(3, 30)) if b2b else int(employees * rng.uniform(500, 5000))
    cta = "Request a demo or talk to sales." if sales_led else "Start free trial, no credit card required."
    home = (f"{name} is {what}. {extra} {gen} {cta} "
            + (f"Trusted by {customers:,}+ {cust if b2b else 'customers'}. " if rng.random() < 0.6 else "")
            + (f"Named to the Inc. 5000 list of fastest-growing companies. " if award else "")
            + f"© {2026 if rng.random() < 0.9 else rng.integers(2017, 2025)} {name}")
    about = f"About {name}. Headquartered in {hq}"
    about += (f" with offices in {', '.join(rng.choice(CITIES_INTL, size=2, replace=False))}." if intl else ".")
    if rng.random() < 0.65:
        about += f" Founded in {founded}, "
    else:
        about += " "
    if rng.random() < 0.6:
        about += f"we are a team of {employees} people "
    about += f"on a mission to transform how {cust} work. "
    if boot and rng.random() < 0.7:
        about += "We are bootstrapped and profitable, funded by our customers. "
    elif series and rng.random() < 0.6:
        about += f"In {min(2025, founded + rng.integers(1, 6))} we raised a ${raised:.0f} million Series {series.upper() if series != 'seed' else 'A'} " \
                 f"led by leading investors. " if series != "seed" else f"Backed by top seed investors. "

    pages = [{"url": f"https://{domain}/", "kind": "home", "title": f"{name} | {what}", "text": home},
             {"url": f"https://{domain}/about", "kind": "about", "title": f"About {name}", "text": about}]
    jobs = None
    if n_jobs > 0 or rng.random() < 0.5:
        k = n_jobs
        titles = list(rng.choice(ROLES_GTM, size=k)) if k else []
        for i in range(k):
            u = rng.random()
            if u > gtm_p:
                titles[i] = rng.choice(ROLES_ENG if (b2b and u < gtm_p + 0.4) else ROLES_OTHER)
        careers = f"Careers at {name}. Join our team in {hq}. " + (
            "We're hiring! Open positions: " + "; ".join(titles) if titles else "No open positions right now.")
        pages.append({"url": f"https://{domain}/careers", "kind": "careers", "title": "Careers", "text": careers})
        if rng.random() < (0.7 if b2b else 0.35):
            jobs = {"ats": rng.choice(["greenhouse", "lever", "ashby"]), "titles": [str(t) for t in titles]}
    if b2b or rng.random() < 0.3:
        price = ("Pricing tailored to your organization. Contact sales for a custom quote." if sales_led
                 else f"Starter ${rng.integers(9, 60)}/month per user. Pro ${rng.integers(60, 200)}/month per user. Free trial included.")
        pages.append({"url": f"https://{domain}/pricing", "kind": "pricing", "title": "Pricing", "text": price})
    if ent > 0:
        terms = ["SOC 2 Type II", "HIPAA", "SSO and SAML", "SCIM provisioning", "audit logs", "GDPR", "99.9% uptime SLA"]
        sec = "Security and trust. " + ", ".join(rng.choice(terms, size=min(ent, len(terms)), replace=False)) + "."
        pages.append({"url": f"https://{domain}/security", "kind": "security", "title": "Security", "text": sec})
    if b2b and rng.random() < 0.6:
        pages.append({"url": f"https://{domain}/customers", "kind": "customers", "title": "Customers",
                      "text": f"Customer stories. Read the case study on how leading {cust} grew with {name}."})
    rec = {"domain": domain, "name": name, "pages": pages, "jobs": jobs, "error": None}
    truth = {"domain": domain, "archetype": arche, "true_fit": true_fit, "employees": employees,
             "founded": founded, "series": series or "bootstrapped", "growth_z": growth}
    return rec, truth


def generate(n: int = 3000, n_pos: int = 45, n_neg: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    names, shares = zip(*[(k, v[0]) for k, v in ARCHETYPES.items()])
    arches = rng.choice(names, size=n, p=np.array(shares) / sum(shares))
    used: set[str] = set()
    recs, truth = zip(*[_company(rng, a, used) for a in arches])
    truth = pd.DataFrame(truth)

    # Deal-team history: positives are mostly (not only) high true-fit companies the
    # firm happened to find; negatives are companies it looked at and passed on.
    fit = truth["true_fit"].to_numpy()
    w = np.exp(3.0 * (fit - fit.max()))
    pos = rng.choice(n, size=n_pos, replace=False, p=w / w.sum())
    rest = np.setdiff1d(np.arange(n), pos)
    mid = rest[(fit[rest] > np.quantile(fit, 0.3)) & (fit[rest] < np.quantile(fit, 0.75))]
    low = rest[fit[rest] <= np.quantile(fit, 0.3)]
    neg = np.concatenate([rng.choice(mid, size=int(n_neg * 0.6), replace=False),
                          rng.choice(low, size=n_neg - int(n_neg * 0.6), replace=False)])
    seeds = pd.DataFrame({"domain": truth["domain"].iloc[np.r_[pos, neg]].to_numpy(),
                          "label": [1] * len(pos) + [0] * len(neg),
                          "note": ["portfolio/pursued"] * len(pos) + ["passed - off thesis"] * len(neg)})
    return list(recs), seeds, truth
