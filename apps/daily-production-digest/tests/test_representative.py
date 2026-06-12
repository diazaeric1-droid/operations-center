"""Tests for representative-vs-anomalous production-data classification.

Data quality FOR trending: a shut-in / zero point and a gross outlier are flagged
non-representative (excluded before a decline fit), a clean point is representative,
and an all-clean series flags none.
"""
import numpy as np
import pandas as pd

from src.representative import (
    R_DROPOUT,
    R_OUTLIER,
    R_ZERO,
    classify_representative,
    representative_pct,
)


def _decline_series(days: int = 40, qi: float = 300.0, d: float = 0.01) -> np.ndarray:
    """A clean exponential decline (no noise) — every point representative."""
    t = np.arange(days)
    return qi * np.exp(-d * t)


def test_clean_decline_flags_none():
    rates = _decline_series()
    out = classify_representative(rates)
    assert out["representative"].all()
    assert out.summary.n_excluded == 0
    assert out.summary.representative_pct == 100.0


def test_zero_rate_point_is_non_representative():
    rates = _decline_series().copy()
    rates[20] = 0.0  # a shut-in / zero-rate day
    out = classify_representative(rates)
    assert not out.loc[20, "representative"]
    assert R_ZERO in out.loc[20, "reason"]
    # the surrounding clean points stay representative
    assert out.loc[19, "representative"] and out.loc[21, "representative"]


def test_gross_outlier_is_non_representative():
    rates = _decline_series().copy()
    rates[30] = rates[30] * 6.0  # gross spike well off the decline trend
    out = classify_representative(rates)
    assert not out.loc[30, "representative"]
    assert R_OUTLIER in out.loc[30, "reason"]


def test_clean_point_stays_representative_amid_anomalies():
    rates = _decline_series().copy()
    rates[10] = 0.0          # shut-in
    rates[25] = rates[25] * 5  # outlier
    out = classify_representative(rates)
    # A normal interior point (index 15) is untouched.
    assert out.loc[15, "representative"]
    assert out.loc[15, "reason"] == ""
    assert out.summary.n_excluded >= 2


def test_shutin_via_low_runtime_flagged():
    days = 30
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=days),
        "bopd": _decline_series(days),
        "runtime_pct": np.full(days, 99.0),
        "motor_amps": np.full(days, 60.0),
    })
    df.loc[12, "runtime_pct"] = 10.0  # heavily-cycled / shut-in day
    out = classify_representative(df)
    assert not out.loc[12, "representative"]
    assert R_ZERO in out.loc[12, "reason"]


def test_metering_dropout_vs_real_shutin():
    days = 30
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=days),
        "bopd": _decline_series(days),
        "runtime_pct": np.full(days, 99.0),
        "motor_amps": np.full(days, 60.0),
    })
    # Rate blank while the pump is plainly running → metering dropout (not a shut-in).
    df.loc[15, "bopd"] = np.nan
    out = classify_representative(df)
    assert not out.loc[15, "representative"]
    assert R_DROPOUT in out.loc[15, "reason"]


def test_dataframe_keeps_date_column():
    days = 20
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=days),
        "bopd": _decline_series(days),
    })
    out = classify_representative(df)
    assert "date" in out.columns
    assert len(out) == days


def test_representative_pct_helper():
    rates = _decline_series(40).copy()
    rates[5] = 0.0
    pct = representative_pct(rates)
    assert 0.0 < pct < 100.0
    # matches the full-classification summary
    assert pct == classify_representative(rates).summary.representative_pct
