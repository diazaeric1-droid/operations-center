"""Tests for the prioritized recovery work-queue (Quantify → Authorize chain).

The queue must only surface RECOVERABLE causes (not planned / reservoir / unclassified),
carry positive recoverable $, be sorted by priority_score desc, never exceed total
deferred $, and tolerate empty input.
"""
import numpy as np
import pandas as pd

from src.analytics import recovery_queue
from src.deferment import compute_deferment
from src.data_loader import EVENT_COLUMNS
from src.reason_codes import is_recoverable


def _prod(bopd, runtime):
    n = len(bopd)
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n),
        "bopd": bopd, "bfpd": np.array(bopd) * 2.0,
        "gas_mcfd": np.array(bopd) * 1.5, "runtime_pct": runtime,
    })


def _fleet_and_events():
    """Two wells: one ESP failure (recoverable artificial_lift), one planned workover
    (not recoverable) and one reservoir/watering-out stretch (not recoverable)."""
    # well_001: 4-day ESP outage -> artificial_lift (recoverable)
    b1 = np.full(40, 300.0); r1 = np.full(40, 100.0)
    b1[20:24] = 0.0; r1[20:24] = 0.0
    # well_002: planned workover outage (not recoverable) + a curtailment we'll code reservoir
    b2 = np.full(40, 250.0); r2 = np.full(40, 100.0)
    b2[10:14] = 0.0; r2[10:14] = 0.0          # planned
    b2[30:34] = 80.0                          # underperformance -> reservoir (loading up)
    fleet = {"well_001": _prod(b1, r1), "well_002": _prod(b2, r2)}
    events = pd.DataFrame([
        {"well_id": "well_001", "start_date": pd.Timestamp("2026-01-21"),
         "end_date": pd.Timestamp("2026-01-24"), "note": "ESP tripped on underload, VSD fault"},
        {"well_id": "well_002", "start_date": pd.Timestamp("2026-01-11"),
         "end_date": pd.Timestamp("2026-01-14"), "note": "planned workover, scheduled maintenance"},
        {"well_id": "well_002", "start_date": pd.Timestamp("2026-01-31"),
         "end_date": pd.Timestamp("2026-02-03"), "note": "well loading up, watering out, liquid loading"},
    ], columns=EVENT_COLUMNS)
    return fleet, events


def test_excludes_planned_and_reservoir_causes():
    fleet, events = _fleet_and_events()
    daily = compute_deferment(fleet, events, price_per_bbl=70.0)
    q = recovery_queue(daily, events, oil_price=70.0)
    assert len(q) > 0
    # every queued reason_key is recoverable; no planned / reservoir / unclassified
    assert q["reason_key"].map(is_recoverable).all()
    assert not q["reason_key"].isin({"planned", "reservoir", "unclassified"}).any()


def test_recoverable_usd_positive_and_sorted_by_priority():
    fleet, events = _fleet_and_events()
    daily = compute_deferment(fleet, events, price_per_bbl=70.0)
    q = recovery_queue(daily, events, oil_price=70.0)
    assert (q["recoverable_usd"] > 0).all()
    scores = q["priority_score"].to_numpy()
    assert np.all(np.diff(scores) <= 1e-9)   # non-increasing == sorted desc


def test_sum_recoverable_le_total_deferred():
    fleet, events = _fleet_and_events()
    daily = compute_deferment(fleet, events, price_per_bbl=70.0)
    q = recovery_queue(daily, events, oil_price=70.0)
    total_deferred_usd = float(daily["deferred_usd"].sum())
    assert q["recoverable_usd"].sum() <= total_deferred_usd + 1e-6


def test_empty_input_yields_empty_frame():
    empty = pd.DataFrame()
    q = recovery_queue(empty, None, oil_price=70.0)
    assert q.empty
    assert list(q.columns)  # has the documented column schema, no error


def test_no_recoverable_loss_yields_empty_frame():
    # a perfectly healthy well -> no deferment -> empty queue
    b = np.full(30, 400.0); r = np.full(30, 100.0)
    daily = compute_deferment({"w": _prod(b, r)}, pd.DataFrame(columns=EVENT_COLUMNS),
                              price_per_bbl=70.0)
    q = recovery_queue(daily, None, oil_price=70.0)
    assert q.empty
