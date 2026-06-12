"""Tests for the rolling lost-production ledger (src/ledger.py).

Covers: a tidy non-empty frame on the demo fleet, monotonic cumulative $,
per-cause reconciliation to the period total, and a clean empty-fleet result.
Also uses a hand-built fleet with a *sustained* multi-day rate loss so the
cumulative/accrual behaviour is exercised beyond the demo's single-day event.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_fleet
from src.ledger import LEDGER_COLUMNS, build_ledger

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "synthetic" / "fleet"


def _sustained_loss_fleet(days: int = 20) -> dict:
    """One healthy well + one well that steps down ~45% for the last 5 days, so
    the scan flags a rate loss on multiple consecutive as-of days."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-05-01", periods=days)

    def well(bopd):
        return pd.DataFrame({
            "date": dates,
            "bopd": bopd,
            "bfpd": rng.normal(1800, 40, days),
            "intake_pressure_psi": rng.normal(120, 3, days),
            "motor_temp_f": rng.normal(290, 2, days),
            "motor_amps": rng.normal(60, 0.5, days),
            "runtime_pct": np.clip(rng.normal(99, 0.2, days), 0, 100),
        })

    healthy = well(rng.normal(200, 6, days))
    dropped_bopd = rng.normal(200, 6, days)
    dropped_bopd[-5:] = dropped_bopd[-5:] * 0.55  # sustained ~45% step-down
    return {"well_001": healthy, "well_002": well(dropped_bopd)}


def test_demo_fleet_returns_tidy_frame():
    fleet = load_fleet(DATA_DIR)
    ledger, summary = build_ledger(fleet, window_days=30)
    assert list(ledger.columns) == LEDGER_COLUMNS
    assert not ledger.empty
    # Every dollar column is non-negative and the summary reconciles to the rows.
    assert (ledger["deferred_usd"] >= 0).all()
    assert summary["period_deferred_usd"] == round(float(ledger["deferred_usd"].sum()), 0)
    assert summary["recoverable_usd"] <= summary["period_deferred_usd"]
    assert summary["top_cause"] in set(ledger["cause"])


def test_cumulative_usd_monotonic_non_decreasing():
    ledger, _ = build_ledger(_sustained_loss_fleet(), window_days=30)
    assert not ledger.empty
    assert ledger["cumulative_usd"].is_monotonic_increasing
    # cumulative is exactly the running sum of the per-row deferred $.
    assert np.allclose(ledger["cumulative_usd"].values,
                       ledger["deferred_usd"].cumsum().values)


def test_sustained_loss_accrues_over_multiple_days():
    """A multi-day step-down should accrue on more than one as-of day."""
    ledger, summary = build_ledger(_sustained_loss_fleet(), window_days=30)
    assert summary["days_scanned"] >= 1
    assert ledger["date"].nunique() >= 2  # accrues across consecutive days
    assert summary["period_deferred_usd"] > 0
    assert summary["period_deferred_bbl"] > 0


def test_per_cause_sums_reconcile_to_total():
    ledger, summary = build_ledger(_sustained_loss_fleet(), window_days=30)
    by_cause = ledger.groupby("cause")["deferred_usd"].sum()
    assert round(float(by_cause.sum()), 0) == summary["period_deferred_usd"]
    assert summary["top_cause_usd"] == round(float(by_cause.max()), 0)


def test_empty_fleet_yields_empty_zero_ledger():
    ledger, summary = build_ledger({}, window_days=30)
    assert ledger.empty
    assert list(ledger.columns) == LEDGER_COLUMNS
    assert summary["period_deferred_usd"] == 0.0
    assert summary["recoverable_usd"] == 0.0
    assert summary["top_cause"] is None
    assert summary["days_scanned"] == 0


def test_fleet_of_empty_frames_no_error():
    """A fleet of shut-in / empty frames must not raise and yields a zero ledger."""
    fleet = {"well_001": pd.DataFrame(columns=["date", "bopd"]),
             "well_002": pd.DataFrame(columns=["date", "bopd"])}
    ledger, summary = build_ledger(fleet, window_days=30)
    assert ledger.empty
    assert summary["period_deferred_usd"] == 0.0
