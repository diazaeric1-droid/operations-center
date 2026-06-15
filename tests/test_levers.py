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

def test_economic_limit_guards_down_wells(bootstrapped):
    """A reserves read on a currently-shut-in well is meaningless — economic_limit must
    return status='down', not an absurd remaining-life off the pre-outage rate."""
    import pandas as pd
    gt = pd.read_csv(core.DIGEST_FLEET.parent / "ground_truth.csv")
    shut = str(gt[gt["seeded_mode"] == "shut_in"]["well_id"].iloc[0])
    el = core.economic_limit(core.well_scada(core.alert_for(shut, price_per_bbl=PRICE)),
                             realized_price=PRICE, net_revenue_interest=NRI)
    assert el is not None and el.get("status") == "down"


def test_gaslift_fault_shows_on_displayed_channels(bootstrapped):
    """A gas-lift well with a gas-interference/lock fault must show the symptom on its
    DISPLAYED channels (injection falls) so the shown evidence matches the diagnosis —
    not only on the hidden intake channel (the flagship-well contradiction)."""
    import numpy as np
    import pandas as pd

    import fleet_registry as fr
    gt = pd.read_csv(core.DIGEST_FLEET.parent / "ground_truth.csv")
    gas = [str(w) for w in gt[gt["seeded_mode"].isin(["gas_interference", "gas_lock"])]["well_id"]]
    checked = 0
    for w in gas:
        if fr.get(w).lift != "Gas lift":
            continue
        d = pd.read_csv(core.DIGEST_FLEET / f"{w}.csv")
        oil = d["bopd"].to_numpy(dtype=float)
        inj = d["gas_inj_mcfd"].to_numpy(dtype=float)
        assert oil[-1] < oil[-31] * 0.92            # oil fell over the window
        assert inj[-1] < inj[-31] * 0.92            # AND the DISPLAYED injection fell
        checked += 1
    assert checked >= 5


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


def test_fit_well_decline_aligns_and_is_sane(bootstrapped):
    _wid, _diag, scada = _top_diag()
    fit = core.fit_well_decline(scada)
    assert fit is not None
    assert len(fit["expected"]) == len(scada)          # aligned to the full history
    assert fit["implied_deferment_bopd"] >= 0.0


def test_well_tiers_cover_the_fleet(bootstrapped):
    fleet = core.load_scada_fleet()
    board = core.rank_fleet(price_per_bbl=PRICE, net_revenue_interest=NRI)
    tiers = core.well_tiers(fleet, board)
    assert len(tiers) == len(fleet)
    assert set(tiers.values()) <= {"down", "watch", "healthy"}


def test_esp_model_ships_calibrated(bootstrapped):
    # Audit #25: a silent calibration fall-through (to raw, uncalibrated probabilities)
    # must be catchable — the shipped model must carry a fitted Platt calibrator.
    m = core.esp_model.ESPRiskModel.load(str(core.ESP_MODEL))
    assert m.calibrator is not None


def test_esp_model_recalibrated_on_digest_fleet(bootstrapped):
    """The ESP model is trained ON the digest fleet (the fleet the console scores) with
    its ground-truth labels, so the score is calibrated and separates impaired from
    healthy — not a uniform out-of-distribution blob."""
    import numpy as np
    import pandas as pd

    ev = core.esp_model_eval()
    assert ev is not None
    assert ev["trained_on"] == "digest_fleet_ground_truth"
    assert ev["calibrated"] is True
    assert ev["auroc_cv_mean"] > 0.7                  # honest OOF AUROC, well above chance
    assert 0 < ev["n_positives"] < ev["n_wells"]

    fleet = core.load_scada_fleet()
    risk = core._score_fleet_risk(fleet, core.ESP_MODEL)
    gt = pd.read_csv(core.DIGEST_FLEET.parent / "ground_truth.csv").set_index("well_id")
    imp = np.median([risk[w] for w in risk if int(gt.loc[w, "impaired"]) == 1])
    heal = np.median([risk[w] for w in risk if int(gt.loc[w, "impaired"]) == 0])
    assert imp > heal + 0.3                           # impaired clearly above healthy


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
