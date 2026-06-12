"""Tests for the deferment engine: healthy wells read ~0, outages are caught, and the
downtime/rate split sums to the total."""
import numpy as np
import pandas as pd

from src.deferment import DEADBAND_FRAC, _well_deferment, compute_deferment
from src.data_loader import EVENT_COLUMNS


def _prod(bopd, runtime):
    n = len(bopd)
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n),
        "bopd": bopd, "bfpd": np.array(bopd) * 2.0,
        "gas_mcfd": np.array(bopd) * 1.5, "runtime_pct": runtime,
    })


def test_healthy_well_reads_near_zero():
    rng = np.random.default_rng(0)
    bopd = np.clip(500 * np.exp(-0.3 * np.arange(60) / 365) * (1 + rng.normal(0, 0.03, 60)), 1, None)
    df = _well_deferment("w", _prod(bopd, np.full(60, 100.0)))
    # within-noise variation must not register as deferment
    assert df["total_def"].sum() < 0.01 * df["potential"].sum()


def test_full_outage_is_deferred_and_is_downtime():
    bopd = np.full(60, 400.0)
    runtime = np.full(60, 100.0)
    bopd[40:45] = 0.0          # 5-day full outage
    runtime[40:45] = 0.0
    df = _well_deferment("w", _prod(bopd, runtime))
    outage = df.iloc[40:45]
    assert outage["total_def"].sum() > 1500          # ~5 * ~400
    # a full outage is attributed to downtime, not underperformance
    assert outage["rate_def"].sum() < 1.0
    assert outage["downtime_def"].sum() > 1500


def test_decomposition_sums_to_total():
    bopd = np.full(40, 300.0); runtime = np.full(40, 100.0)
    bopd[20:25] = 120.0          # curtailment while up
    runtime[10:13] = 30.0; bopd[10:13] = 90.0   # partial downtime
    df = _well_deferment("w", _prod(bopd, runtime))
    assert np.allclose((df["downtime_def"] + df["rate_def"]).values, df["total_def"].values, atol=1e-6)


def test_deadband_ignores_small_gaps():
    bopd = np.full(40, 500.0); runtime = np.full(40, 100.0)
    bopd[20] = 500.0 * (1 - DEADBAND_FRAC / 2)   # tiny dip, below deadband
    df = _well_deferment("w", _prod(bopd, runtime))
    assert df.iloc[20]["total_def"] == 0.0


def test_compute_deferment_attributes_and_prices():
    bopd = np.full(40, 300.0); runtime = np.full(40, 100.0)
    bopd[20:24] = 0.0; runtime[20:24] = 0.0
    fleet = {"well_001": _prod(bopd, runtime)}
    events = pd.DataFrame([{
        "well_id": "well_001",
        "start_date": pd.Timestamp("2026-01-21"), "end_date": pd.Timestamp("2026-01-24"),
        "note": "ESP tripped on underload, VSD fault"}], columns=EVENT_COLUMNS)
    daily = compute_deferment(fleet, events, price_per_bbl=70.0)
    loss = daily[daily["total_def"] > 1]
    assert (loss["reason_key"] == "artificial_lift").all()
    assert np.isclose((daily["total_def"] * 70.0).sum(), daily["deferred_usd"].sum())
