"""Tests for the NDIC real-data adapter (src/ndic.py).

The adapter maps a tidy monthly NDIC extract into the SAME fleet structure
``compute_deferment`` consumes. These tests pin the monthly→rate + downtime math
and confirm the result is a drop-in for the deferment engine — and that there are
no fabricated reason codes on real data (cause attribution is N/A).
"""
import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from src.ndic import load_ndic_fleet, ndic_well_meta, parse_ndic_csv
from src.deferment import compute_deferment
from src.data_loader import EVENT_COLUMNS, PROD_COLUMNS

TEMPLATE = Path(__file__).resolve().parents[1] / "data" / "real" / "ndic" / "_TEMPLATE.csv"


def test_parses_template_into_fleet():
    """The shipped _TEMPLATE.csv parses into a one-well fleet with the engine schema."""
    fleet = load_ndic_fleet(TEMPLATE)
    assert set(fleet) == {"DEMO_0001"}
    df = fleet["DEMO_0001"]
    # produces every column the deferment engine reads (load_fleet's PROD_COLUMNS).
    for col in PROD_COLUMNS:
        assert col in df.columns, col
    assert len(df) == 2
    assert df["date"].is_monotonic_increasing


def test_monthly_rate_and_downtime_math():
    """rate = oil_bbl/days, fluid = (oil+water)/days, gas = gas_mcf/days,
    uptime = days/days_in_month*100 — pinned against the template's row 1
    (Jan-2024: oil 1000, gas 500, water 400, days 28; Jan has 31 days)."""
    df = load_ndic_fleet(TEMPLATE)["DEMO_0001"]
    r0 = df.iloc[0]
    assert np.isclose(r0["bopd"], 1000 / 28)
    assert np.isclose(r0["bfpd"], (1000 + 400) / 28)
    assert np.isclose(r0["gas_mcfd"], 500 / 28)
    assert np.isclose(r0["runtime_pct"], 28 / 31 * 100.0)
    # days-produced preserved (the real downtime input) + correct days-in-month.
    assert r0["days"] == 28
    assert r0["days_in_month"] == calendar.monthrange(2024, 1)[1] == 31


def test_uptime_capped_and_leap_year():
    """Uptime is days/days_in_month, capped at 100%; Feb-2024 is a leap month (29)."""
    df = load_ndic_fleet(TEMPLATE)["DEMO_0001"]
    feb = df.iloc[1]
    assert feb["days_in_month"] == 29           # 2024 leap year
    assert np.isclose(feb["runtime_pct"], 20 / 29 * 100.0)
    assert (df["runtime_pct"] <= 100.0 + 1e-9).all()


def test_full_uptime_month_clamps_to_100():
    """A filing reporting days == days_in_month yields exactly 100% uptime (no >100)."""
    raw = pd.DataFrame([
        {"well_id": "W1", "well_name": "n", "operator": "o", "field": "f",
         "formation": "Bakken", "date": "2023-03", "oil_bbl": 3100, "gas_mcf": 1000,
         "water_bbl": 500, "days": 31},  # March = 31 days, fully up
    ])
    p = Path(_write_tmp(raw))
    df = load_ndic_fleet(p)["W1"]
    assert np.isclose(df.iloc[0]["runtime_pct"], 100.0)


def test_drop_in_for_compute_deferment_no_fake_causes():
    """Adapter output flows through compute_deferment producing real deferment — and
    with NO reason codes (empty events), nothing is fabricated: every loss is
    'unclassified' / non-recoverable (cause attribution N/A).

    The monthly avg rate (oil_bbl/days) drops well below the established potential in
    one month — exactly what a low-uptime / curtailed month looks like in NDIC data."""
    months = pd.period_range("2023-01", periods=14, freq="M")
    rows = []
    for i, m in enumerate(months):
        dim = m.days_in_month
        if i == 10:                            # bad month: low uptime AND a rate drop
            days, oil = 6, 18.0                # bopd = 3.0 vs ~100 potential
        else:
            days, oil = dim, 100.0 * dim       # ~100 bopd, fully up
        rows.append({"well_id": "WX", "well_name": "Test 1H", "operator": "Op",
                     "field": "Fld", "formation": "Bakken", "date": str(m),
                     "oil_bbl": oil, "gas_mcf": oil * 0.5, "water_bbl": oil * 0.4,
                     "days": days})
    p = Path(_write_tmp(pd.DataFrame(rows)))
    fleet = load_ndic_fleet(p)

    evc = pd.DataFrame(columns=[*EVENT_COLUMNS, "reason_key"])  # no public reason codes
    daily = compute_deferment(fleet, evc, price_per_bbl=70.0)
    assert len(daily) == 14
    # the bad month registers real deferment...
    assert float(daily["total_def"].sum()) > 0
    loss = daily[daily["total_def"] > 1e-6]
    # ...and it's uncoded: no recoverable causes were invented from thin air.
    assert (loss["reason_key"] == "unclassified").all()
    assert not daily["recoverable"].any()
    # the low-uptime month routes loss to the downtime bucket (days-produced signal).
    assert float(daily["downtime_def"].sum()) > 0
    assert np.isclose((daily["total_def"] * 70.0).sum(), daily["deferred_usd"].sum())


def test_well_meta_one_row_per_well():
    meta = ndic_well_meta(TEMPLATE)
    assert list(meta["well_id"]) == ["DEMO_0001"]
    assert meta.iloc[0]["formation"] == "Bakken"


def test_missing_column_raises():
    bad = pd.DataFrame([{"well_id": "W1", "date": "2023-01", "oil_bbl": 100, "days": 30}])
    p = Path(_write_tmp(bad))
    try:
        parse_ndic_csv(p)
    except ValueError as e:
        assert "missing" in str(e).lower()
    else:
        raise AssertionError("expected ValueError on missing columns")


def test_drops_nonpositive_days_rows():
    """Rows with days <= 0 (shut-in/no-report months) are dropped, not divided by zero."""
    raw = pd.DataFrame([
        {"well_id": "W1", "well_name": "n", "operator": "o", "field": "f",
         "formation": "Bakken", "date": "2023-01", "oil_bbl": 0, "gas_mcf": 0,
         "water_bbl": 0, "days": 0},
        {"well_id": "W1", "well_name": "n", "operator": "o", "field": "f",
         "formation": "Bakken", "date": "2023-02", "oil_bbl": 2800, "gas_mcf": 100,
         "water_bbl": 50, "days": 28},
    ])
    p = Path(_write_tmp(raw))
    df = load_ndic_fleet(p)["W1"]
    assert len(df) == 1                        # the days==0 row was dropped
    assert df.iloc[0]["days"] == 28
    assert np.isfinite(df["bopd"]).all()


# --- helper ----------------------------------------------------------------

def _write_tmp(df: pd.DataFrame) -> str:
    import tempfile
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    df.to_csv(f.name, index=False)
    f.close()
    return f.name
