"""Deterministic anomaly detection rules over fleet SCADA.

Each rule returns an Anomaly dataclass with severity (HIGH/MEDIUM/LOW),
category, the well affected, and the specific evidence triggering the flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


def _slope_per_step(values) -> float:
    """Least-squares slope per step over a window (robust to endpoint noise,
    unlike a 2-point first/last difference)."""
    y = np.asarray(values, dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def robust_z(values) -> float:
    """Robust z-score of the last point vs the rest of the window, using the
    median + MAD (median absolute deviation) instead of mean/std.

    robust_z = 0.6745 * (x - median) / MAD  (0.6745 ≈ Φ⁻¹(0.75) makes MAD a
    consistent estimator of σ for normal data). Median/MAD are unaffected by the
    very outlier we're trying to detect, so a single bad day can't inflate the
    baseline the way mean/std would.

    Returns the signed robust z of the final value relative to the *preceding*
    points. NaNs are dropped first. Guards MAD==0 (constant baseline) — returns
    0.0 rather than dividing by zero — so a flat well never produces a spurious
    infinite z.
    """
    y = np.asarray(values, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 3:
        return 0.0
    point = float(y[-1])
    baseline = y[:-1]
    med = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - med)))
    if mad <= 1e-9:
        # Degenerate (flat) baseline: fall back to std; if that's also ~0 the
        # series is constant and there is, by definition, no anomaly.
        std = float(np.std(baseline))
        if std <= 1e-9:
            return 0.0
        return (point - med) / std
    return 0.6745 * (point - med) / mad


def _expected_decline_rate(values, extrapolate: int = 0) -> float | None:
    """Fit a simple exponential (Arps-style) decline by log-linear regression and
    return the expected value at ``last_index + extrapolate``.

    Production naturally declines, so a flat 7-day mean over-flags a healthy
    well. We fit log(rate) = a + b·day via np.polyfit (no scipy), then the
    decline-expected rate is exp(a·t + b). Pass ``extrapolate=1`` with a history
    window that EXCLUDES today to predict today's expected rate from the trend
    alone — so a single step-down day can't contaminate (flatten) its own
    baseline. Non-positive rates are dropped before the log; if too few positive
    points remain, returns None and the caller falls back to the flat-mean rule.
    """
    y = np.asarray(values, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 4:
        return None
    x = np.arange(n)
    mask = y > 0
    if mask.sum() < 3:
        return None
    xf, yf = x[mask], y[mask]
    a, b = np.polyfit(xf, np.log(yf), 1)  # slope a (per day), intercept b
    expected = float(np.exp(a * (x[-1] + extrapolate) + b))
    if not np.isfinite(expected) or expected <= 0:
        return None
    return expected


Severity = Literal["HIGH", "MEDIUM", "LOW"]

# Default realized oil price for deferred-production economics. A flag is only
# actionable if you know what it's costing — this converts a rate drop into $/day.
DEFAULT_OIL_PRICE = 70.0


@dataclass
class Anomaly:
    well_id: str
    severity: Severity
    category: str        # e.g., "rate_drop", "intake_collapse", "amps_creep"
    headline: str        # Short human-readable summary
    evidence: dict       # The specific numbers backing the call
    recommended_action: str
    # Deferred-production economics (set for rate-loss anomalies; 0 for pure-risk
    # flags like amps creep where nothing is being lost *yet*).
    deferred_bopd: float = 0.0
    deferred_usd_per_day: float = 0.0
    acknowledged: bool = False   # suppressed via the ack list (known/planned event)


def _water_cut(row) -> float | None:
    """Water cut % from a SCADA row (needs bopd + bfpd). None if unavailable."""
    try:
        bopd, bfpd = float(row["bopd"]), float(row["bfpd"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(bopd) or not np.isfinite(bfpd) or bfpd <= 0:
        return None
    return (bfpd - bopd) / bfpd * 100.0


def detect_data_quality(well_id: str, scada: pd.DataFrame) -> Anomaly | None:
    """Flag a stale / dropped final reading and classify it: comms loss vs. metering
    dropout vs. a real trip. A blank or zero tag is itself an event a control-room PE
    chases — silently swallowing it (NaN < threshold == False) is exactly what erodes
    trust in an auto-brief.
    """
    if len(scada) < 1:
        return None
    last = scada.iloc[-1]
    key_channels = [c for c in ("bopd", "intake_pressure_psi", "motor_amps", "runtime_pct")
                    if c in scada.columns]
    vals = {c: float(last[c]) if pd.notna(last[c]) else np.nan for c in key_channels}
    n_nan = sum(np.isnan(v) for v in vals.values())

    # All key channels blank → comms loss (RTU / poll failure), not a well problem.
    if key_channels and n_nan == len(key_channels):
        return Anomaly(
            well_id=well_id, severity="MEDIUM", category="comms_loss",
            headline="No SCADA on last poll — all key tags blank (comms / RTU loss)",
            evidence={"missing_channels": key_channels},
            recommended_action="Check RTU / SCADA poll + cell/radio link before dispatching a truck.",
        )
    # Rate tag blank/zero but the pump is clearly running → metering dropout, not a
    # dead well (don't roll a truck for a flow transmitter glitch).
    bopd = vals.get("bopd", np.nan)
    amps = vals.get("motor_amps", np.nan)
    runtime = vals.get("runtime_pct", np.nan)
    rate_dead = np.isnan(bopd) or bopd <= 0
    pump_running = (np.isfinite(amps) and amps > 5) or (np.isfinite(runtime) and runtime > 50)
    if rate_dead and pump_running:
        return Anomaly(
            well_id=well_id, severity="MEDIUM", category="meter_dropout",
            headline="Oil rate reads 0/blank while the pump is running — likely a metering dropout",
            evidence={"bopd": None if np.isnan(bopd) else round(bopd, 1),
                      "motor_amps": None if np.isnan(amps) else round(amps, 1),
                      "runtime_pct": None if np.isnan(runtime) else round(runtime, 1)},
            recommended_action="Verify the flow/Coriolis meter + LACT; reconcile against tank gauge before flagging a deferral.",
        )
    return None


# ---- detection rules --------------------------------------------------------

def detect_rate_drop(well_id: str, scada: pd.DataFrame) -> Anomaly | None:
    """Flag if last 24h BOPD is >15% below 7-day rolling average."""
    if len(scada) < 8:
        return None
    last_day = scada.iloc[-1]["bopd"]
    if pd.isna(last_day):
        return None  # dropped reading → handled by detect_data_quality, not silently dropped
    baseline = scada.iloc[-8:-1]["bopd"].mean()
    if baseline <= 0:
        return None
    drop_pct = (last_day - baseline) / baseline * 100
    if drop_pct < -25:
        severity = "HIGH"
        action = "Field check within 2 hours; check pump status, separator levels, and ESDV positions"
    elif drop_pct < -15:
        severity = "MEDIUM"
        action = "Review next-day; pull dyno card or ESP readings before end of day"
    else:
        return None
    # Robust z of today's rate vs this well's own recent baseline (median + MAD).
    rz = robust_z(scada.iloc[-8:]["bopd"].values)
    deferred = max(baseline - last_day, 0.0)
    ev = {"last_24h_bopd": round(last_day, 1), "baseline_bopd": round(baseline, 1),
          "drop_pct": round(drop_pct, 1), "robust_z": round(rz, 2),
          "deferred_bopd": round(deferred, 1)}
    # Water-cut context: a rising water cut alongside the oil drop points at watering
    # out (reservoir) rather than a pump issue — different action entirely.
    wc_now = _water_cut(scada.iloc[-1])
    wc_base = _water_cut(scada.iloc[-8:-1].mean(numeric_only=True))
    wc_note = ""
    if wc_now is not None and wc_base is not None:
        ev["water_cut_pct"] = round(wc_now, 1)
        ev["water_cut_delta_pts"] = round(wc_now - wc_base, 1)
        if wc_now - wc_base >= 4:
            wc_note = f"; water cut up {wc_now - wc_base:.0f} pts → likely watering out, not a pump issue"
    return Anomaly(
        well_id=well_id, severity=severity, category="rate_drop",
        headline=f"BOPD dropped {abs(drop_pct):.0f}% vs 7-day baseline "
                 f"({abs(rz):.1f}σ off own baseline){wc_note}",
        evidence=ev,
        recommended_action=action,
        deferred_bopd=deferred,
    )


def detect_rate_drop_decline_aware(well_id: str, scada: pd.DataFrame) -> Anomaly | None:
    """Decline-aware rate drop. A 15% drop on a well declining 1%/day is normal —
    flat-mean rules over-flag it. Fit an exponential (Arps) decline by log-linear
    regression over the window, compute the decline-EXPECTED rate today, and flag
    only when today's rate is materially below what the decline trend predicts.

    Refinement to detect_rate_drop (both stay in RULES); this one catches the
    *excess* drop after accounting for natural decline, so it suppresses false
    positives on steep-but-healthy decliners and still catches a real step-down.
    """
    if "bopd" not in scada.columns or len(scada) < 8:
        return None
    window = scada.iloc[-14:]["bopd"].values if len(scada) >= 14 else scada["bopd"].values
    last_day = float(window[-1])
    if np.isnan(last_day):
        return None  # dropped reading → detect_data_quality
    # Fit the decline on history EXCLUDING today, then extrapolate one step, so a
    # one-day collapse doesn't flatten the very trend it's measured against.
    expected = _expected_decline_rate(window[:-1], extrapolate=1)
    if expected is None or expected <= 0:
        return None  # fall back to flat-mean detect_rate_drop
    resid_pct = (last_day - expected) / expected * 100
    if resid_pct >= -15:  # within decline-expected band → not an anomaly
        return None
    if resid_pct < -25:
        severity = "HIGH"
        action = "Field check within 2 hours; drop exceeds natural decline — check pump status, separator levels, ESDV positions"
    else:
        severity = "MEDIUM"
        action = "Review next-day; drop is beyond the decline trend — pull dyno card or ESP readings before end of day"
    rz = robust_z(window[-8:])
    deferred = max(expected - last_day, 0.0)
    return Anomaly(
        well_id=well_id, severity=severity, category="rate_drop_decline_aware",
        headline=f"BOPD {abs(resid_pct):.0f}% below decline-expected ({abs(rz):.1f}σ off own baseline)",
        evidence={"last_24h_bopd": round(last_day, 1),
                  "decline_expected_bopd": round(expected, 1),
                  "residual_pct": round(resid_pct, 1),
                  "robust_z": round(rz, 2),
                  "deferred_bopd": round(deferred, 1)},
        recommended_action=action,
        deferred_bopd=deferred,
    )


def detect_intake_collapse(well_id: str, scada: pd.DataFrame) -> Anomaly | None:
    """ESP intake pressure trending toward zero — gas interference / pump-off risk."""
    if "intake_pressure_psi" not in scada.columns or len(scada) < 5:
        return None
    last5 = scada.iloc[-5:]["intake_pressure_psi"].values
    if np.isnan(last5[-1]) or last5[-1] >= 40:
        return None
    # Falling trend (least-squares slope, not a noisy 2-point difference)
    slope = _slope_per_step(last5)
    if slope >= 0:
        return None
    severity = "HIGH" if last5[-1] < 25 else "MEDIUM"
    return Anomaly(
        well_id=well_id, severity=severity, category="intake_collapse",
        headline=f"Intake pressure {last5[-1]:.0f} psi, declining {slope:.1f} psi/day",
        evidence={"current_intake_psi": round(float(last5[-1]), 1),
                  "5d_slope_psi_per_day": round(float(slope), 2)},
        recommended_action="VSD frequency check + gas separator inspection; if no recovery in 48h, escalate to workover queue",
    )


def detect_motor_temp_spike(well_id: str, scada: pd.DataFrame) -> Anomaly | None:
    if "motor_temp_f" not in scada.columns or len(scada) < 8:
        return None
    last_day = scada.iloc[-1]["motor_temp_f"]
    if pd.isna(last_day):
        return None  # dropped reading → detect_data_quality
    baseline = scada.iloc[-8:-1]["motor_temp_f"].mean()
    rz = robust_z(scada.iloc[-8:]["motor_temp_f"].values)
    if last_day > 340:
        severity = "HIGH"
    # MEDIUM requires BOTH a real absolute rise AND statistical significance vs the
    # well's own noise — a noisy well's one warm day shouldn't trip a flag.
    elif last_day > baseline + 15 and rz >= 3:
        severity = "MEDIUM"
    else:
        return None
    return Anomaly(
        well_id=well_id, severity=severity, category="motor_temp_spike",
        headline=f"Motor temp {last_day:.0f}°F (+{last_day - baseline:.0f}°F vs 7-day avg, {abs(rz):.1f}σ off own baseline)",
        evidence={"current_temp_f": round(float(last_day), 1),
                  "baseline_temp_f": round(float(baseline), 1),
                  "robust_z": round(rz, 2)},
        recommended_action="Reduce VSD frequency; if temp not falling within 4h, plan controlled shutdown",
    )


def detect_runtime_degradation(well_id: str, scada: pd.DataFrame) -> Anomaly | None:
    if "runtime_pct" not in scada.columns or len(scada) < 1:
        return None
    last_day = scada.iloc[-1]["runtime_pct"]
    if pd.isna(last_day) or last_day >= 90:
        return None
    severity = "HIGH" if last_day < 70 else "MEDIUM"
    return Anomaly(
        well_id=well_id, severity=severity, category="runtime_degradation",
        headline=f"Runtime only {last_day:.0f}% in last 24h",
        evidence={"runtime_pct": round(float(last_day), 1)},
        recommended_action="Pull cycle log; identify trip reason (gas lock, overload, surface power)",
    )


def detect_amps_creep(well_id: str, scada: pd.DataFrame) -> Anomaly | None:
    """Slow amps creep over 7 days — early scale / mechanical wear signal."""
    if "motor_amps" not in scada.columns or len(scada) < 8:
        return None
    window = scada.iloc[-8:]["motor_amps"].values
    # Least-squares slope over the 8-day window — the old first/last difference
    # was dominated by daily noise and missed real creep (and vice-versa).
    slope_per_day = _slope_per_step(window)
    if np.isnan(slope_per_day) or slope_per_day < 0.3:
        return None
    return Anomaly(
        well_id=well_id, severity="MEDIUM", category="amps_creep",
        headline=f"Motor amps creeping +{slope_per_day:.2f} A/day over 8-day window",
        evidence={"current_amps": round(float(window[-1]), 1),
                  "8d_slope_amps_per_day": round(float(slope_per_day), 2)},
        recommended_action="Trend casing pressure and intake pressure; if both stable, schedule scale-treatment workover within 30 days",
    )


RULES = [detect_data_quality, detect_rate_drop, detect_rate_drop_decline_aware,
         detect_intake_collapse, detect_motor_temp_spike, detect_runtime_degradation,
         detect_amps_creep]


def _ack_match(anomaly: Anomaly, acknowledged) -> bool:
    """True if this anomaly is on the acknowledged/suppressed list (known/planned
    event). ``acknowledged`` is an iterable of dicts {well, category?} — a missing
    or '*' category matches any category for that well."""
    if not acknowledged:
        return False
    for entry in acknowledged:
        if entry.get("well") != anomaly.well_id:
            continue
        cat = entry.get("category")
        if cat in (None, "*", anomaly.category):
            return True
    return False


def scan_fleet(fleet: dict[str, pd.DataFrame],
               price_per_bbl: float = DEFAULT_OIL_PRICE,
               acknowledged=None) -> list[Anomaly]:
    anomalies = []
    for well_id, scada in fleet.items():
        for rule in RULES:
            result = rule(well_id, scada)
            if result is not None:
                anomalies.append(result)

    # The decline-aware rate rule is AUTHORITATIVE: whenever a decline fit is
    # feasible for a well, it owns the rate-drop call — so we drop the flat-mean
    # rate_drop for that well whether decline-aware fired (a real excess drop) or
    # stayed silent (verified on-trend). This suppresses the flat-mean rule's
    # false positive on a steep-but-healthy decliner. Flat-mean survives only as a
    # fallback on series too short to fit a decline. A metering dropout supersedes
    # both rate rules (a flat-lined transmitter reads as a -100% drop).
    decline_capable_wells = {
        wid for wid, scada in fleet.items()
        if "bopd" in scada.columns and len(scada) >= 8
        and _expected_decline_rate(scada["bopd"].values[-14:][:-1], extrapolate=1) is not None
    }
    dropout_wells = {a.well_id for a in anomalies if a.category == "meter_dropout"}
    anomalies = [
        a for a in anomalies
        if not (a.category == "rate_drop" and a.well_id in decline_capable_wells)
        and not (a.category in ("rate_drop", "rate_drop_decline_aware") and a.well_id in dropout_wells)
    ]

    # Attach deferred-production $ and acknowledged flags.
    for a in anomalies:
        a.deferred_usd_per_day = round(a.deferred_bopd * price_per_bbl, 0)
        a.acknowledged = _ack_match(a, acknowledged)

    # Sort: unacknowledged first, then HIGH→MEDIUM→LOW, then by DEFERRED $ (money
    # first — a foreman works the biggest leak, not the alphabetically-first well),
    # then well_id for stability.
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    anomalies.sort(key=lambda a: (a.acknowledged, severity_order[a.severity],
                                  -a.deferred_usd_per_day, a.well_id))
    return anomalies


def load_acknowledgements(path) -> list[dict]:
    """Load an acknowledged/suppression list (YAML) of known or planned events so a
    well on a scheduled workover doesn't re-fire HIGH every morning and train the
    team to ignore the brief. Returns [] if the file is missing/empty.

    Format (acknowledged.yml):
        - well: well_028
          category: runtime_degradation   # optional; omit or '*' = any category
          note: planned workover 2026-06-05
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return []
    import yaml
    data = yaml.safe_load(p.read_text()) or []
    return [e for e in data if isinstance(e, dict) and e.get("well")]
