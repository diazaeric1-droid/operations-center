"""Tolerance-gated tests for the deferment ENGINE's quantity accounting (Phase 2).

Builds a synthetic fleet with KNOWN injected downtime + underperformance per well — so
true deferred / recoverable barrels are known — runs the engine, and asserts the error
vs ground truth stays within a sane, cadence-appropriate bound. CI fails if the engine's
deferred-bbl error exceeds the bound. Runs on BOTH a daily-cadence and a monthly-cadence
representation of the IDENTICAL fleet, proving the engine is cadence-aware (the core fix:
time-based calendar-day volumes, not fixed row-count windows).

Honest expectation: daily data → ~exact on everything; monthly data (real Colorado/NDIC
grain) → downtime exact (days-produced is explicit) but short sub-month rate dips are
smeared by the producing-day average, so underperformance is under-counted. The eval
gates each cadence on what it can actually resolve.
"""
import tempfile

import pytest

from evals.quantity_fleet import build_events, build_truth_fleet
from evals.quantity_recovery import (
    DAILY_RECOVER_TOL,
    DAILY_SPLIT_TOL,
    DAILY_TOTAL_TOL,
    MONTHLY_DOWNTIME_TOL,
    MONTHLY_TOTAL_TOL,
    evaluate,
    gate,
)
from src.analytics import recovery_opportunity
from src.deferment import classify_events, compute_deferment
from src.ndic import load_ndic_fleet

SEEDS = [0, 7, 13]


@pytest.fixture(scope="module")
def report():
    return evaluate(seed=7)


def test_gate_passes_default(report):
    """The committed eval gate passes at the default seed (the CI quantity gate)."""
    fails = gate(report)
    assert not fails, "quantity eval gate violations:\n" + "\n".join(fails)


def test_daily_is_essentially_exact(report):
    """At daily cadence the engine recovers ground truth almost exactly on every metric."""
    e = report["daily"]["err_pct"]
    assert abs(e["total_deferred"]) <= DAILY_TOTAL_TOL
    assert abs(e["downtime"]) <= DAILY_SPLIT_TOL
    assert abs(e["underperf"]) <= DAILY_SPLIT_TOL
    assert abs(e["recoverable"]) <= DAILY_RECOVER_TOL


def test_monthly_downtime_exact_and_total_bounded(report):
    """Monthly data resolves downtime exactly (days-produced) and the total deferred-bbl
    error stays within the documented (smeared-underperformance) bound — NOT the orders-
    of-magnitude error the old row-count engine produced on monthly data."""
    e = report["monthly"]["err_pct"]
    assert abs(e["downtime"]) <= MONTHLY_DOWNTIME_TOL
    assert abs(e["total_deferred"]) <= MONTHLY_TOTAL_TOL


def test_cadence_awareness_downtime_matches_across_cadences(report):
    """The same physical fleet yields the SAME downtime barrels at daily and monthly
    cadence — the cadence-aware, time-based volume accounting (would be ~impossible under
    the old fixed-row-window engine, which treated a month-row like a day-row)."""
    assert abs(report["downtime_cadence_gap_pct"]) <= 1.0


@pytest.mark.parametrize("seed", SEEDS)
def test_deferred_bbl_within_bound_multi_seed(seed):
    """Deferred-bbl error gate holds across several seeds (not a single lucky draw)."""
    res = evaluate(seed=seed)
    assert abs(res["daily"]["err_pct"]["total_deferred"]) <= DAILY_TOTAL_TOL
    assert abs(res["monthly"]["err_pct"]["total_deferred"]) <= MONTHLY_TOTAL_TOL
    assert abs(res["monthly"]["err_pct"]["downtime"]) <= MONTHLY_DOWNTIME_TOL


def test_recoverable_le_total_deferred_both_cadences(report):
    """Recoverable opportunity can never exceed total deferred (a basic accounting bound)."""
    for cad in ("daily", "monthly"):
        eng = report[cad]["engine"]
        assert eng["recoverable_bbl"] <= eng["total_deferred_bbl"] + 1e-6


def test_monthly_oil_volume_is_exact():
    """Sanity on the calendar-day volume accounting: engine actual barrels at monthly
    cadence equal the raw injected oil exactly (proves volumes, not rates, are summed)."""
    daily_fleet, monthly_df, truth = build_truth_fleet(seed=7)
    events = build_events(truth, daily_fleet)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        monthly_df.to_csv(f.name, index=False)
        path = f.name
    fleet = load_ndic_fleet(path)
    evc = classify_events(events)
    daily = compute_deferment(fleet, evc, price_per_bbl=70.0)
    engine_actual = float(daily["actual_vol"].sum())
    raw_oil = float(monthly_df["oil_bbl"].sum())
    assert engine_actual == pytest.approx(raw_oil, rel=1e-6)
    # and recovery_opportunity stays within the deferred total
    rec = recovery_opportunity(daily)
    assert rec["recoverable_bbl"] <= float(daily["total_def"].sum()) + 1e-6
