"""Smoke tests for anomaly detection rules."""
import numpy as np
import pandas as pd

from src.anomaly_detector import (
    detect_amps_creep,
    detect_intake_collapse,
    detect_motor_temp_spike,
    detect_rate_drop,
    detect_runtime_degradation,
    scan_fleet,
)


def base(days: int = 14, **overrides) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    base = {
        "date": pd.date_range("2026-05-01", periods=days),
        "bopd": rng.normal(200, 10, days),
        "bfpd": rng.normal(1800, 80, days),
        "intake_pressure_psi": rng.normal(120, 5, days),
        "motor_temp_f": rng.normal(290, 3, days),
        "motor_amps": rng.normal(60, 1, days),
        "runtime_pct": rng.normal(99, 0.3, days),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_rate_drop_flags_45_pct_drop():
    df = base()
    df.loc[df.index[-1], "bopd"] = df["bopd"].iloc[-1] * 0.55
    a = detect_rate_drop("w1", df)
    assert a is not None
    assert a.severity == "HIGH"


def test_rate_drop_ignores_normal_noise():
    assert detect_rate_drop("w1", base()) is None


def test_intake_collapse_flags_falling_trend():
    df = base()
    p = df["intake_pressure_psi"].to_numpy(copy=True)
    p[-5:] = np.linspace(p[-5], 20, 5)
    df["intake_pressure_psi"] = p
    a = detect_intake_collapse("w1", df)
    assert a is not None
    assert a.severity in ("HIGH", "MEDIUM")


def test_motor_temp_spike_above_340():
    df = base()
    df.loc[df.index[-1], "motor_temp_f"] = 348
    a = detect_motor_temp_spike("w1", df)
    assert a is not None
    assert a.severity == "HIGH"


def test_runtime_degradation_below_70():
    df = base()
    df.loc[df.index[-1], "runtime_pct"] = 65
    a = detect_runtime_degradation("w1", df)
    assert a is not None
    assert a.severity == "HIGH"


def test_amps_creep_over_7_days():
    df = base(motor_amps=np.linspace(60, 72, 14))
    a = detect_amps_creep("w1", df)
    assert a is not None
    assert a.severity == "MEDIUM"


def test_scan_fleet_sorts_high_first():
    fleet = {}
    df_high = base()
    df_high.loc[df_high.index[-1], "motor_temp_f"] = 348
    fleet["well_b"] = df_high

    df_med = base(motor_amps=np.linspace(60, 72, 14))
    fleet["well_a"] = df_med

    anomalies = scan_fleet(fleet)
    assert anomalies[0].severity == "HIGH"


def test_rate_drop_reports_deferred_economics():
    df = base()
    df.loc[df.index[-1], "bopd"] = df["bopd"].iloc[-1] * 0.55
    a = detect_rate_drop("w1", df)
    assert a.deferred_bopd > 0
    fleet = {"w1": df}
    scanned = scan_fleet(fleet, price_per_bbl=70.0)
    assert scanned[0].deferred_usd_per_day == round(scanned[0].deferred_bopd * 70.0, 0)


def test_metering_dropout_vs_silent_swallow():
    # Oil rate blank while the pump runs → a metering-dropout flag, not a silent miss.
    from src.anomaly_detector import detect_data_quality
    df = base()
    df.loc[df.index[-1], "bopd"] = np.nan
    df.loc[df.index[-1], "motor_amps"] = 60
    df.loc[df.index[-1], "runtime_pct"] = 99
    a = detect_data_quality("w1", df)
    assert a is not None and a.category == "meter_dropout"
    # And the flat-mean rate rule must NOT silently fire on the NaN.
    assert detect_rate_drop("w1", df) is None


def test_acknowledged_events_are_flagged_and_deprioritized():
    df = base()
    df.loc[df.index[-1], "runtime_pct"] = 60   # HIGH runtime degradation
    fleet = {"well_x": df}
    anoms = scan_fleet(fleet, acknowledged=[{"well": "well_x", "category": "runtime_degradation"}])
    assert anoms and all(a.acknowledged for a in anoms)


def test_decline_aware_suppresses_steep_decliner_false_positive():
    from src.anomaly_detector import detect_rate_drop_decline_aware
    t = np.arange(30)
    df = base(days=30, bopd=350 * 0.94 ** t)
    # Flat-mean over-flags the steep but healthy decline; decline-aware does not.
    assert detect_rate_drop("w1", df) is not None
    assert detect_rate_drop_decline_aware("w1", df) is None
