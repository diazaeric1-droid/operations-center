"""Coverage for the v0.4.0 levers: Monte-Carlo AFE economics, the economic-limit
calc, and the triage-ranking backtest scorecard."""
from __future__ import annotations

import core

PRICE = 70.0
NRI = 0.80


def _top_diag():
    board = core.rank_fleet(price_per_bbl=PRICE, net_revenue_interest=NRI)
    wid = str(board["well_id"].iloc[0])
    alert = core.alert_for(wid, price_per_bbl=PRICE)
    return wid, core.diagnose(alert), core.well_scada(alert)


# ---- Monte-Carlo NPV reconciles with the deterministic AFE + is ordered --------

def test_afe_monte_carlo_reconciles_and_is_ordered(bootstrapped):
    _wid, diag, _scada = _top_diag()
    mc = core.afe_monte_carlo(diag, realized_price=PRICE, net_revenue_interest=NRI)
    assert mc is not None
    assert mc["p10"] <= mc["p50"] <= mc["p90"]            # percentiles ordered
    assert 0.0 <= mc["prob_payout"] <= 1.0
    assert set(mc["tornado"]) == {"incremental_rate_bopd", "uplift_decline_per_yr",
                                  "realized_price_per_bbl"}
    # The net base NPV must equal the AFE's deterministic Net NPV (exact: cost certain).
    econ = core.afe_economics.compute_economics(
        mc["cost"], float(diag["incremental_rate_bopd"]),
        uplift_decline_per_yr=float(diag["expected_uplift_decline_per_yr"]),
        realized_price_per_bbl=PRICE, working_interest=1.0, net_revenue_interest=NRI)
    assert abs(mc["base"] - econ.net_npv_10pct_usd) < 1.0


# ---- economic limit is sane --------------------------------------------------

def test_economic_limit_is_sane(bootstrapped):
    _wid, _diag, scada = _top_diag()
    el = core.economic_limit(scada, realized_price=PRICE, net_revenue_interest=NRI)
    assert el is not None
    assert el["q_limit_bopd"] > 0
    assert el["net_margin_per_bbl"] == PRICE * NRI - 12.0     # price*NRI - opex
    assert el["months_remaining"] >= 0
    # A producing well sits above its own economic-limit rate.
    assert el["q_now_bopd"] > el["q_limit_bopd"]


# ---- triage scorecard is honest (better than random, not a trivial 1.0) -------

def test_interventions_are_lift_appropriate(bootstrapped):
    """The lift-aware intervention engine: every recommended intervention must be one
    that physically applies to the well's artificial-lift type (no ESP swap on a
    rod-pumped well, no gas-lift optimization on a well with no injection)."""
    import fleet_registry as fr

    board = core.rank_fleet(price_per_bbl=PRICE, net_revenue_interest=NRI)
    acting = board[board["recommended_intervention"] != "no_action"]
    valid = {
        "ESP": {"esp_swap", "scale_treatment"},
        "Rod pump": {"rod_pump_workover", "scale_treatment"},
        "Gas lift": {"gas_lift_optimization", "scale_treatment"},
        "Flowing": {"acid_stimulation", "scale_treatment"},
    }
    bad = []
    for _, r in acting.iterrows():
        lift = fr.get(str(r["well_id"])).lift
        if r["recommended_intervention"] not in valid.get(lift, set()):
            bad.append((str(r["well_id"]), lift, r["recommended_intervention"]))
    assert not bad, f"lift-inappropriate interventions: {bad[:10]}"
    # And the specific impossibilities the audit flagged never occur:
    lift_of = {str(r["well_id"]): fr.get(str(r["well_id"])).lift
               for _, r in acting.iterrows()}
    for _, r in acting.iterrows():
        interv, lift = r["recommended_intervention"], lift_of[str(r["well_id"])]
        if interv == "gas_lift_optimization":
            assert lift == "Gas lift"
        if interv == "rod_pump_workover":
            assert lift == "Rod pump"
        if interv == "esp_swap":
            assert lift == "ESP"


def test_triage_scorecard_is_honest(bootstrapped):
    board = core.rank_fleet(price_per_bbl=PRICE, net_revenue_interest=NRI)
    sc = core.triage_scorecard(board)
    assert sc is not None
    assert 0 < sc["n_impaired"] < sc["n_wells"]
    for k in (5, 10, 20):
        p = sc["at_k"][k]["precision"]
        assert 0.0 <= p <= 1.0
    # The ranking beats a random draw at the top (lift > 1) but isn't a perfect oracle.
    assert sc["at_k"][10]["lift"] > 1.0
    assert sc["recall_at_n_impaired"] < 1.0
