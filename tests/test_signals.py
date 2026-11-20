from sourcing.signals import extract_signals


def rec(pages, jobs=None):
    return {"domain": "x.com", "name": "X", "pages": pages, "jobs": jobs}


def test_extracts_stage_and_go_to_market():
    r = rec([
        {"kind": "home", "text": "Acme is the practice management software for dental practices. "
                                 "Request a demo. Trusted by 2,500+ practices. © 2026 Acme"},
        {"kind": "about", "text": "Founded in 2016, we are a team of 180 people. "
                                  "In 2022 we raised a $40 million Series B."},
        {"kind": "security", "text": "SOC 2 Type II, HIPAA, SSO and SAML."},
        {"kind": "pricing", "text": "Contact sales for a custom quote."},
    ], jobs={"ats": "greenhouse", "titles": ["Account Executive", "Customer Success Manager",
                                            "Senior Software Engineer", "Product Designer"]})
    s = extract_signals(r)
    assert s["_founded_year"] == 2016
    assert s["_employees"] == 180
    assert s["_series_letter"] == "B"
    assert round(s["log_raised_usd_m"], 3) == round(__import__("math").log1p(40), 3)
    assert s["enterprise_score"] >= 4
    assert s["_pricing_model"] == "sales-led"
    assert s["_open_jobs"] == 4 and s["gtm_job_share"] == 0.5 and s["eng_job_share"] == 0.25
    assert s["site_staleness_years"] <= 1


def test_missing_info_is_nan_not_zero():
    s = extract_signals(rec([{"kind": "home", "text": "We make great things."}]))
    for k in ("company_age", "max_series", "log_employees", "log_open_jobs", "gtm_job_share"):
        assert s[k] != s[k], k  # NaN


def test_careers_text_fallback_without_ats():
    s = extract_signals(rec([{"kind": "careers", "text": "Open roles: Senior Account Executive; "
                                                         "Staff Software Engineer; Customer Success Manager"}]))
    assert s["_open_jobs"] >= 3
    assert s["gtm_job_share"] > 0
