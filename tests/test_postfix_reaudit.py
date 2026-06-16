"""Coverage for the v0.7.2 post-fix re-audit surface — each test pins the BEHAVIOR a
fix changed, so a regression that reopens one of the closed self-contradictions fails CI:

* #1 gas-lift gas-interference evidence cites the DISPLAYED channels, never the hidden
  ESP intake pressure;
* #2 the persisted eval is end-to-end calibrated out-of-fold (the Brier describes the
  shipped calibrated probabilities), self-healed via the eval-method marker;
* #3 economic_limit reports the recent producing rate and flags below-established-trend;
* #4 fleet_health_summary's elevated-risk count is absolute + fully accounted for
  (no "0 amber while 37 wells are high-risk");
* #5 currently-down wells route to a Restore queue, out of the priced opportunities;
* #6 the opportunity signal includes an absolute calibrated-risk floor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import core


PRICE, NRI = 70.0, 0.80


# ---- #1: gas-lift evidence never cites the hidden ESP intake -------------------

def test_gaslift_gas_interference_evidence_uses_displayed_channels():
    from esp import explainer

    v = {"imb_max": 4.0, "imb_days": 0, "bfpd_cv": 0.05, "downtime": 0,
         "intake_mean": 56.0, "intake_slope": -2.6, "amps_slope": 0.0,
         "temp_slope": 0.0, "bfpd_slope": -5.0, "runtime_mean": 99.0, "freq_slope": 0.0}
    _label, evidence = explainer._phrase_mode("gas_interference", "Gas lift", v)
    low = evidence.lower()
    # The flagship contradiction: a gas-lift well has no intake gauge on screen, so its
    # evidence must not quote the intake pressure (56 psi) as "flowing pressure".
    assert "intake" not in low
    assert "56" not in evidence and "psi" not in low
    # It must point at the channels a gas-lift reviewer actually sees.
    assert "injection" in low and "casing" in low

    # A FLOWING well DOES display downhole (intake) pressure, so its branch may still
    # reference pressure — sanity that the fix didn't blank the flowing path.
    _l2, ev_flowing = explainer._phrase_mode("gas_interference", "Flowing", v)
    assert "pressure" in ev_flowing.lower()


# ---- #2: end-to-end calibrated OOF, self-healing marker ------------------------

def test_esp_eval_is_calibrated_oof(bootstrapped):
    ev = core.esp_model_eval()
    assert ev is not None
    assert ev["eval_method"] == core.ESP_EVAL_METHOD == "calibrated_oof_v2"
    assert ev["calibrated"] is True
    # Calibrated-OOF AUROC is honest (a touch below the raw booster) and the Brier
    # reflects the calibrated probabilities, not a near-zero raw-booster artifact.
    assert 0.70 < ev["auroc_cv_mean"] < 0.999
    assert 0.0 < ev["brier"] < 0.20
    assert "recall_at_top10pct" in ev          # recall is persisted (model card renders it)


def test_stale_eval_method_triggers_retrain(monkeypatch, bootstrapped):
    """A model whose eval marker predates the current method must self-heal (retrain),
    so a warm container never serves stale model-card numbers."""
    import json
    ev = json.loads(core.ESP_EVAL.read_text())
    ev["eval_method"] = "old_raw_v1"
    core.ESP_EVAL.write_text(json.dumps(ev))
    core.ensure_esp_model(log=lambda *a, **k: None)
    assert core.esp_model_eval()["eval_method"] == core.ESP_EVAL_METHOD


# ---- #3: economic limit reads the recent rate + flags below-trend --------------

def _series(plateau: float, days: int, tail: list[float]) -> pd.DataFrame:
    oil = [plateau] * days + tail
    dates = pd.date_range("2025-01-01", periods=len(oil), freq="D")
    return pd.DataFrame({"date": dates, "bopd": oil})


def test_economic_limit_flags_below_trend_and_uses_recent_rate():
    # A well on a long ~300 BOPD plateau that has recently fallen to ~140.
    scada = _series(300.0, 175, list(np.linspace(290, 140, 25)))
    el = core.economic_limit(scada, realized_price=PRICE, net_revenue_interest=NRI)
    assert el is not None and el["status"] == "ok"
    assert el["below_established_trend"] is True
    # Current rate reflects the recent collapse, not the pre-collapse plateau.
    assert el["q_now_bopd"] < el["q_trend_bopd"]
    assert el["q_now_bopd"] < 220.0            # nowhere near the 300 plateau
    assert el["q_now_bopd"] > el["q_limit_bopd"]


def test_below_trend_remaining_life_is_computed_from_depressed_rate():
    """Pins the Well 360 below-trend warning copy: remaining life is computed from the
    CURRENT (depressed) rate — so it already reflects today's reduced deliverability, and
    using the higher plateau rate would give a LONGER life, not a shorter one. (The earlier
    warning claimed the opposite; this guards the corrected direction.)"""
    scada = _series(300.0, 175, list(np.linspace(290, 140, 25)))
    el = core.economic_limit(scada, realized_price=PRICE, net_revenue_interest=NRI)
    assert el["below_established_trend"] is True
    if el["months_remaining"] == float("inf"):
        return  # flat established trend → no finite horizon to compare; copy still holds
    # Reconstruct the monthly decline from the reported annual rate and recompute months
    # from BOTH the current rate and the plateau, holding the (established-trend) slope.
    Di = np.log(1.0 - el["annual_decline_pct"] / 100.0) / 365.0
    d_monthly = -Di * 30.4
    months_from_now = np.log(el["q_now_bopd"] / el["q_limit_bopd"]) / d_monthly
    months_from_trend = np.log(el["q_trend_bopd"] / el["q_limit_bopd"]) / d_monthly
    # The displayed figure tracks the depressed current rate…
    assert abs(months_from_now - el["months_remaining"]) < max(1.0, 0.02 * el["months_remaining"])
    # …and the plateau rate would give MORE months (the current-rate figure is conservative).
    assert months_from_trend > el["months_remaining"]


def test_economic_limit_healthy_well_not_flagged_below_trend():
    scada = _series(300.0, 200, [300.0] * 10)
    el = core.economic_limit(scada, realized_price=PRICE, net_revenue_interest=NRI)
    assert el is not None and el["status"] == "ok"
    assert el["below_established_trend"] is False


# ---- #4: elevated-risk count is absolute and fully accounted for ---------------

def test_fleet_health_elevated_risk_is_accounted_for(bootstrapped):
    fleet = core.load_scada_fleet()
    anomalies = core.scan_anomalies(fleet, price_per_bbl=PRICE)
    risk = core._score_fleet_risk(fleet, core.ESP_MODEL)
    h = core.fleet_health_summary(fleet, anomalies, risk_by_well=risk)
    # Every well at elevated ABSOLUTE risk is either already Impaired or on the amber
    # watch list — so a low amber count never contradicts a large high-risk population
    # (the "Elevated Risk: 0 while 37 score >=0.8" finding).
    assert h["elevated_abs"] >= 1
    assert h["elevated_abs"] <= h["elevated_abs_impaired"] + h["watch"]
    assert h["healthy"] + h["watch"] + h["impaired"] == h["total"]


# ---- #5: shut-in wells route to a Restore queue, not priced opportunities ------

def test_down_wells_route_to_restore_tier(bootstrapped):
    from views import _common as vc

    fleet = core.load_scada_fleet()
    down = core.down_wells(fleet)
    assert down, "expected at least one shut-in well in the seeded fleet"

    board = vc.board_with_deferred(PRICE, NRI)
    restore, remaining = vc.restore_tier(board, down)
    # Down wells are pulled OUT of the ranked board into the restore queue…
    assert set(restore["well_id"].astype(str)) == set(down)
    # …and none of them survive into the opportunity/watch/stable partition.
    assert not (set(remaining["well_id"].astype(str)) & set(down))
    opp, _watch, _stable = vc.triage_tiers(remaining)
    assert not (set(opp["well_id"].astype(str)) & set(down))


# ---- #6: the opportunity signal includes an absolute calibrated-risk floor -----

def test_opportunity_signal_includes_absolute_risk_floor():
    from views import _common as vc

    frame = pd.DataFrame({
        "deferred_bopd": [0.0, 0.0, 0.0],
        # below the abs floor / at the abs floor / well above it
        "failure_risk_30d": [0.20, core.ELEVATED_RISK_ABS_30D, 0.90],
        "est_risked_npv": [10.0, 10.0, 10.0],
    })
    sig = vc.opportunity_signal(frame).tolist()
    assert sig[0] is False or sig[0] == False          # 0.20: no signal
    assert sig[1] and sig[2]                             # >= absolute floor: signalled
