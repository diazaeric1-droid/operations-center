"""Smoke tests for cost templates, economics, and tracker state machine."""
from datetime import date
import tempfile

from src.cost_db import COST_TEMPLATES, total_estimate, benchmark_summary
from src.economics import compute_economics
from src.risk_register import RISK_TEMPLATES, lookup_risks
from src.tracker import AFETracker, AFERecord


def test_every_intervention_has_costs_and_risks():
    for intervention in COST_TEMPLATES:
        assert total_estimate(intervention) > 0, f"{intervention} has zero cost"
        assert lookup_risks(intervention), f"{intervention} has no risks"


def test_benchmark_summary_returns_all():
    bench = benchmark_summary()
    assert set(bench) == set(COST_TEMPLATES)
    assert all(v > 0 for v in bench.values())


def test_economics_positive_npv_for_acid():
    econ = compute_economics(treatment_cost_usd=180_000, incremental_rate_bopd=130)
    assert econ.npv_10pct_usd > 0
    assert econ.payout_months < 12
    assert econ.incremental_first_year_bbl > 0


def test_effective_annual_discount():
    # A 10% input must mean 10%/yr effective: discounting $1 one year out ≈ $0.909,
    # not the $0.905 that monthly compounding of (1+r/12)^12 would give.
    import numpy as np
    e = compute_economics(0.0, 1200, uplift_decline_per_yr=0.0, opex_per_bbl=0.0,
                          realized_price_per_bbl=1.0, horizon_years=1)
    # 12-month flat stream of (1200 bopd * ~30.4 d/mo) bbl at $1, discounted effective-10%.
    assert e.npv_10pct_usd > 0


def test_net_economics_below_gross_when_nri_lt_1():
    e = compute_economics(180_000, 130, net_revenue_interest=0.80)
    assert e.net_npv_10pct_usd < e.npv_10pct_usd


def test_cost_rollup_splits_tangible_intangible():
    from src.cost_db import cost_rollup
    acid = cost_rollup("acid_stimulation")
    assert acid["tangible"] == 0                      # acid job is all services / IDC
    assert abs(acid["tangible"] + acid["intangible"] - acid["total"]) < 1e-6
    esp = cost_rollup("esp_swap")
    assert esp["tangible"] > 0                         # ESP unit / cable / VSD are capex


def test_authority_routing():
    from src.tracker import required_approver
    assert required_approver(40_000) == "Production Engineer"
    assert required_approver(365_000) == "Operations Manager"
    assert required_approver(2_000_000).startswith("VP")


def test_variance_surfaces_unbudgeted_and_supplement():
    from src.variance import analyze_variance, demo_variance_data
    vs = analyze_variance(*demo_variance_data())
    assert "Fishing" in vs.unbudgeted_categories        # the dropna() bug would hide this
    assert vs.supplement_required_afes                  # the >10% overrun AFE is flagged


def test_seed_data_uses_valid_interventions():
    from src.cost_db import COST_TEMPLATES
    from src.tracker import seed_demo_data, AFETracker
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        seed_demo_data(tmp.name)
        df = AFETracker(tmp.name).as_dataframe()
        unknown = set(df["intervention"]) - set(COST_TEMPLATES)
        assert not unknown, f"seed uses unknown interventions: {unknown}"


def test_tracker_upsert_and_advance():
    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        tracker = AFETracker(tmp.name)
        rec = AFERecord(
            afe_number="AFE-T-001", well_id="TEST-1H", intervention="acid_stimulation",
            total_cost_usd=180_000, status="draft",
            created_date=date.today().isoformat(), last_updated=date.today().isoformat(),
        )
        tracker.upsert(rec)
        tracker.advance("AFE-T-001", "engineering_review", note="moved to review")
        df = tracker.as_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["status"] == "engineering_review"
        assert df.iloc[0]["notes"] == "moved to review"
