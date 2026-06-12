"""Representative-vs-anomalous production-data classification (data quality FOR
trending) — distinct from the surveillance alerting in ``anomaly_detector``.

Before you fit a decline / type curve you must first decide which historical points
are *representative* of the well's real producing behavior and which should be
EXCLUDED so they don't bias the trend: shut-in / zero-rate days, metering dropouts
(rate reads 0/blank while the pump is plainly running), and gross outliers versus a
robust, decline-aware trend. (This is the data-cleaning step WellProductivity.jl
describes as "anomaly detection to filter non-representative data points, e.g.
production shutdowns" — reimplemented here in Python.)

That is a different job from ``anomaly_detector.scan_fleet``, which raises an
operational *alert* on the latest day (a real rate drop is a HIGH flag you act on,
not a point you silently delete). Here every point in the history is labeled, and a
flagged point is one you'd drop from a curve fit — a shut-in is perfectly healthy,
it just isn't on-trend data.

Deterministic and dependency-light (numpy/pandas). Reuses the robust statistics
(median/MAD ``robust_z``) and the Arps ``_expected_decline_rate`` from
``anomaly_detector`` rather than re-deriving them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .anomaly_detector import _expected_decline_rate, robust_z

# Tunables (deterministic). A point is non-representative for trending if ANY fires.
ZERO_RATE_EPS = 1e-6      # rate at/below this is a zero / shut-in day
LOW_RUNTIME_PCT = 50.0    # runtime at/below this → shut-in / heavily-cycled day
PUMP_ON_AMPS = 5.0        # motor amps above this means the pump is actually running
PUMP_ON_RUNTIME = 50.0    # runtime above this means the pump is actually running
ROBUST_Z_OUTLIER = 4.0    # |robust z| beyond this vs the local trend → gross outlier

# Reason codes (stable strings; one point may carry several, joined by "; ").
R_ZERO = "zero_or_shutin"        # zero / near-zero rate or shut-in via runtime/days
R_DROPOUT = "meter_dropout"      # rate 0/blank while the pump is clearly running
R_OUTLIER = "robust_outlier"     # gross outlier vs a decline-aware robust trend
R_MISSING = "missing_rate"       # rate is NaN and we can't tell it's a dropout


@dataclass
class RepresentativeSummary:
    """Per-well roll-up of the representative classification."""
    n_points: int
    n_representative: int
    n_excluded: int
    representative_pct: float          # share of points usable for trending (0–100)
    reason_counts: dict[str, int]      # reason code -> count of excluded points


def _rate_series(series_or_scada, rate_col: str = "bopd") -> tuple[np.ndarray, pd.DataFrame | None]:
    """Coerce the input into a 1-D rate array, and (if a DataFrame) keep the frame so
    runtime/amps context columns are available. Accepts a bare list / np.ndarray /
    pd.Series of rates, or a SCADA DataFrame with a ``rate_col`` (default ``bopd``)."""
    if isinstance(series_or_scada, pd.DataFrame):
        if rate_col not in series_or_scada.columns:
            raise KeyError(f"rate column {rate_col!r} not in DataFrame columns {list(series_or_scada.columns)}")
        return series_or_scada[rate_col].to_numpy(dtype=float), series_or_scada
    if isinstance(series_or_scada, pd.Series):
        return series_or_scada.to_numpy(dtype=float), None
    return np.asarray(series_or_scada, dtype=float), None


def _decline_aware_trend(rates: np.ndarray) -> np.ndarray | None:
    """Per-point expected rate from an Arps (log-linear) decline fit over the positive
    history, so the outlier test is taken against the DECLINE-expected level rather than
    a flat mean (a steep-but-healthy decliner shouldn't read late points as outliers).

    Reuses ``anomaly_detector._expected_decline_rate``: we fit on the full positive
    series and read its expected value at each index via successive extrapolations.
    Returns ``None`` if a decline can't be fit (caller falls back to a flat robust z)."""
    n = len(rates)
    if n < 4:
        return None
    base = rates[rates > 0]
    if len(base) < 3:
        return None
    # _expected_decline_rate(window, extrapolate=k) gives exp-fit value at index
    # len(window)-1+k. Fit once on the positive baseline, then read each original
    # index i as an offset from that baseline's last index.
    last_idx = len(base) - 1
    expected = np.empty(n, dtype=float)
    ok = False
    for i in range(n):
        val = _expected_decline_rate(base, extrapolate=int(i - last_idx))
        if val is None:
            return None
        expected[i] = val
        ok = True
    return expected if ok else None


def classify_representative(
    series_or_scada,
    rate_col: str = "bopd",
) -> pd.DataFrame:
    """Classify each production point as representative vs. non-representative FOR
    decline / type-curve trending.

    Parameters
    ----------
    series_or_scada
        A per-well SCADA ``DataFrame`` (uses ``rate_col`` plus, when present,
        ``runtime_pct`` and ``motor_amps`` for shut-in / dropout context), or a bare
        rate series (list / np.ndarray / pd.Series).
    rate_col
        Rate column name when a DataFrame is passed (default ``"bopd"``).

    Returns
    -------
    pandas.DataFrame
        One row per input point, in input order, with columns:
        ``rate`` (float, NaN preserved), ``representative`` (bool — True = keep for
        trending), and ``reason`` (str — "" when representative, else "; "-joined
        reason codes). Carries a ``.summary`` attribute (``RepresentativeSummary``) and,
        when a DataFrame was passed, its ``date`` column for plotting/joining.

    Deterministic: same input → same output, no randomness, no API key.
    """
    rates, frame = _rate_series(series_or_scada, rate_col=rate_col)
    n = len(rates)

    runtime = amps = None
    if frame is not None:
        if "runtime_pct" in frame.columns:
            runtime = frame["runtime_pct"].to_numpy(dtype=float)
        if "motor_amps" in frame.columns:
            amps = frame["motor_amps"].to_numpy(dtype=float)

    # Decline-aware expected level per point (falls back to flat robust z if a decline
    # can't be fit). The robust z of each point is taken against the residual from the
    # trend so that a healthy decline doesn't read its own tail as anomalous.
    expected = _decline_aware_trend(rates)
    if expected is not None:
        # Residual ratio (actual/expected) over positive points; an outlier is a point
        # whose residual is a gross MAD-outlier vs the other residuals. robust_z scores
        # the LAST element of an array vs the rest, so we evaluate each index by moving
        # it to the end of the residual vector.
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.where((expected > 0) & np.isfinite(rates), rates / expected, np.nan)
    else:
        resid = rates.astype(float)

    reasons: list[str] = []
    representative: list[bool] = []
    for i in range(n):
        r = rates[i]
        pt_reasons: list[str] = []

        rt = float(runtime[i]) if runtime is not None and np.isfinite(runtime[i]) else None
        am = float(amps[i]) if amps is not None and np.isfinite(amps[i]) else None
        pump_running = (am is not None and am > PUMP_ON_AMPS) or (rt is not None and rt > PUMP_ON_RUNTIME)

        if np.isnan(r):
            # Blank rate while the pump is clearly running → metering dropout; else a
            # plain missing reading. Either way it's not representative for trending.
            pt_reasons.append(R_DROPOUT if pump_running else R_MISSING)
        elif r <= ZERO_RATE_EPS:
            # Zero rate with the pump running is a dropout; otherwise a shut-in / zero day.
            pt_reasons.append(R_DROPOUT if pump_running else R_ZERO)
        else:
            # Producing, positive rate. A low runtime still means a shut-in / heavily
            # cycled day that doesn't represent steady-state deliverability.
            if rt is not None and rt <= LOW_RUNTIME_PCT:
                pt_reasons.append(R_ZERO)
            # Gross outlier vs the (decline-aware) robust trend.
            series_for_z = resid[: i + 1]
            series_for_z = series_for_z[np.isfinite(series_for_z)]
            if len(series_for_z) >= 3:
                # robust_z scores the last element; build [others..., this point].
                this_val = resid[i]
                if np.isfinite(this_val):
                    others = resid[:i][np.isfinite(resid[:i])]
                    if len(others) >= 2:
                        rz = robust_z(np.append(others, this_val))
                        if abs(rz) >= ROBUST_Z_OUTLIER:
                            pt_reasons.append(R_OUTLIER)

        representative.append(not pt_reasons)
        reasons.append("; ".join(pt_reasons))

    out = pd.DataFrame({
        "rate": rates,
        "representative": representative,
        "reason": reasons,
    })
    if frame is not None and "date" in frame.columns:
        out.insert(0, "date", frame["date"].to_numpy())

    n_excl = int((~out["representative"]).sum())
    reason_counts: dict[str, int] = {}
    for rs in reasons:
        for code in (c for c in rs.split("; ") if c):
            reason_counts[code] = reason_counts.get(code, 0) + 1
    out.summary = RepresentativeSummary(
        n_points=n,
        n_representative=n - n_excl,
        n_excluded=n_excl,
        representative_pct=round((n - n_excl) / n * 100.0, 1) if n else 100.0,
        reason_counts=reason_counts,
    )
    return out


def representative_pct(series_or_scada, rate_col: str = "bopd") -> float:
    """Convenience: just the representative-share percentage (0–100) for a well —
    handy for a fleet-table column. Empty series → 100.0 (nothing to exclude)."""
    return classify_representative(series_or_scada, rate_col=rate_col).summary.representative_pct
