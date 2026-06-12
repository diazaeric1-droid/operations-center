"""Pipeline stage-3 handoff: WellDiagnosis → deterministic AFE markdown."""
from src.handoff import render_afe_markdown


DIAG = {
    "schema": "pe-pipeline/well-diagnosis/v1",
    "well_id": "well_013",
    "api_number": "TBD-ASSIGN",
    "field": "Synthetic Delaware Basin",
    "operator": "Synthetic Operator LLC",
    "intervention": "gas_lift_optimization",
    "primary_diagnosis": "Gas interference. ESP 30-day failure risk 61%.",
    "incremental_rate_bopd": 33.0,
    "expected_uplift_decline_per_yr": 0.6,
    "requested_by": "ESP Failure-Risk Agent (auto)",
    "esp_risk_score": 0.61,
    "suspected_mode": "Gas interference — intake pressure collapse",
}


def test_render_afe_markdown_has_all_sections():
    md = render_afe_markdown(DIAG, working_interest=1.0, net_revenue_interest=0.80)
    for section in ("# AUTHORIZATION FOR EXPENDITURE", "## Cost Breakdown",
                    "## Economics", "## Risk Register", "## Approvals",
                    "Net NPV to operator", "Price-deck sensitivity"):
        assert section in md, f"missing: {section}"
    # authority routing for a small workover should land on the PE
    assert "Production Engineer" in md


def test_render_is_deterministic_and_keyfree():
    # No ANTHROPIC_API_KEY needed; two renders are identical.
    assert render_afe_markdown(DIAG) == render_afe_markdown(DIAG)


def test_no_duplicate_risk_sentence():
    md = render_afe_markdown(DIAG)
    # primary_diagnosis already carries the risk; the summary must not repeat it.
    assert md.count("ESP 30-day failure risk 61%") == 1
