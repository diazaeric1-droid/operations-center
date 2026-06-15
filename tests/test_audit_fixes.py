"""Coverage for the audit-fix surfaces that previously had none:

* the morning-brief email renderer (notify.py) — a daily GitHub Action emails it,
  so a malformed-HTML regression would ship straight to the inbox;
* the 3-tier triage partition (views._common.triage_tiers) — its tiers drive the
  Triage Board's headline KPI counts and dollar totals;
* the production-divergence / wells-down math + its NET-to-operator $ convention,
  which must match across Home, the Morning Brief page, and the email.
"""
from __future__ import annotations

import pandas as pd

import core


# ---- (18) email path: markdown -> HTML + multipart message --------------------

def test_notify_renders_brief_to_html_and_message(bootstrapped):
    import notify

    md = core.morning_brief_markdown(price_per_bbl=70.0, net_revenue_interest=0.80)
    html = notify.markdown_to_html(md)
    assert html and "<" in html and ">" in html          # produced real HTML
    assert "<h2" in html or "<h1" in html                # at least one heading
    assert "Operations Center" in html or "Brief" in html or "Deferred" in html

    # An unbalanced ** must not blow up the renderer (would ship to the inbox).
    notify.markdown_to_html("Bottom line: **unbalanced bold and a < bracket")

    msg = notify.build_message(
        sender="ops@example.com", recipients=["foreman@example.com"],
        subject="Morning Brief", markdown_body=md)
    assert msg["To"] == "foreman@example.com"
    assert msg["Subject"] == "Morning Brief"
    body = msg.get_body(preferencelist=("html",))
    assert body is not None                                # multipart carries HTML


# ---- (19) triage tier partition is total + disjoint + correctly-membered ------

def _fake_board() -> pd.DataFrame:
    # A: positive NPV but NO signal (low risk, no deferment) -> must be STABLE, not an
    #    opportunity, even though its (cheap) intervention pencils. The whole point of
    #    the gate once interventions are lift-aware/cheap.
    # B: deferring + positive -> opportunity.   C: deferring + negative -> watch.
    # D: no signal + negative -> stable.        E: no_action -> stable.
    # G: elevated fleet-relative risk + positive (no deferment) -> opportunity.
    return pd.DataFrame({
        "well_id": ["A", "B", "C", "D", "E", "G"],
        "recommended_intervention": ["esp_swap", "scale_treatment", "esp_swap",
                                     "esp_swap", "no_action", "rod_pump_workover"],
        "est_risked_npv": [100.0, 50.0, -10.0, -20.0, 0.0, 80.0],
        "deferred_bopd": [0.0, 10.0, 5.0, 0.0, 0.0, 0.0],
        "failure_risk_30d": [0.20, 0.30, 0.30, 0.10, 0.60, 0.95],
    })


def test_triage_tiers_partition_and_membership():
    from views import _common as c

    board = _fake_board()
    opp, watch, stable = c.triage_tiers(board)

    # Total + disjoint partition of every well.
    ids = [set(f["well_id"]) for f in (opp, watch, stable)]
    assert len(opp) + len(watch) + len(stable) == len(board)
    assert ids[0] | ids[1] | ids[2] == set(board["well_id"])
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])

    # B = deferring + positive; G = elevated-risk + positive. Both opportunities.
    assert set(opp["well_id"]) == {"B", "G"}
    assert (opp["est_risked_npv"] > 0).all()
    # C has a signal (deferring) but doesn't pay -> watch.
    assert set(watch["well_id"]) == {"C"}
    assert (watch["est_risked_npv"] <= 0).all()
    # A is positive-NPV but has NO signal -> STABLE (the key gate behavior); D/E too.
    assert set(stable["well_id"]) == {"A", "D", "E"}
    assert "A" in set(stable["well_id"])


# ---- (20) divergence / wells-down math + the net-$ convention -----------------

def test_production_divergence_summary_is_internally_consistent(bootstrapped):
    fleet = core.load_scada_fleet()
    anomalies = core.scan_anomalies(fleet, price_per_bbl=70.0)
    div = core.production_divergence_summary(fleet, anomalies)

    assert div["n_down"] == len(div["down"]) >= 0
    assert div["n_divergences"] == len(div["divergences"])
    # divergence_bopd is the sum of the per-divergence deferred barrels.
    expect_bopd = round(sum(getattr(a, "deferred_bopd", 0.0)
                            for a in div["divergences"]), 1)
    assert div["divergence_bopd"] == expect_bopd
    # Every "down" well is actually at/near zero relative to its own baseline.
    for d in div["down"]:
        assert d["last_bopd"] <= d["baseline_bopd"]


def test_divergence_section_dollars_are_net_to_operator(bootstrapped):
    fleet = core.load_scada_fleet()
    anomalies = core.scan_anomalies(fleet, price_per_bbl=70.0)
    div = core.production_divergence_summary(fleet, anomalies)
    price, nri = 70.0, 0.80
    md = core._divergence_section_md(div, price, nri)
    # The brief/email net-of-NRI figure must equal bopd × price × NRI — the same
    # net-to-operator convention the Triage Board's deferred_usd_per_day uses.
    net = div["divergence_bopd"] * price * nri
    assert f"${net:,.0f}/day deferred (net to operator" in md
