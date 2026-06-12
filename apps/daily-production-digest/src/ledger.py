"""Rolling lost-production ledger — Monitor → Quantify chain into Deferment IQ.

The Daily Digest scan is a *point-in-time* monitor: it tells you what's leaking
*today*. A base-management review needs the *cumulative* picture — how much
deferred oil ($) has piled up over a trailing period, broken down by cause.

This module reconstructs that period ledger by **replaying the existing
deterministic scan day-by-day** over a trailing window. The fleet CSVs hold a
multi-day date range, so for each "as-of" day we slice every well's history up
to that day, run the SAME `scan_fleet` the morning brief uses, and record the
deferred bbl/$ by category. No new economics are invented — every dollar comes
from `Anomaly.deferred_bopd * price` exactly as the live scan computes it.

Why a day-by-day replay (and its honest limits):
  - It's a faithful daily monitor: each day's number is what the brief *would*
    have reported that morning, so the cumulative total is a real run-rate sum.
  - It does NOT de-duplicate a multi-day event: a well that defers for five
    straight days contributes five daily deferrals (a run-rate accrual), which
    is the correct accounting for a "$/day deferred, summed over the period"
    ledger — not a single discrete loss volume. Deferment IQ does the full
    potential-vs-actual base-management accounting (planned/reservoir splits,
    MTTR, capture rate); this ledger is the lightweight upstream accrual.
  - A day only enters the ledger if the as-of slice has enough history for the
    rules to fire (>= MIN_HISTORY days); earlier days are skipped, not errored.
"""
from __future__ import annotations

import pandas as pd

from .anomaly_detector import DEFAULT_OIL_PRICE, scan_fleet

# The scan's rate-drop rules need >= 8 days of history to fit a baseline; below
# that the day yields no rate-loss economics, so we don't bother scanning it.
MIN_HISTORY = 8

# Fraction of period deferred $ realistically recoverable. Mirrors the framing
# used downstream (planned + reservoir-driven loss is NOT recoverable); a flat,
# clearly-labelled estimate here, with the full split living in Deferment IQ.
RECOVERABLE_FRACTION = 0.65

LEDGER_COLUMNS = ["date", "cause", "deferred_bbl", "deferred_usd", "cumulative_usd"]


def _all_dates(fleet: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    """Sorted union of every date appearing in any well's history."""
    dates: set[pd.Timestamp] = set()
    for df in fleet.values():
        if len(df) and "date" in df.columns:
            dates.update(df["date"].tolist())
    return sorted(dates)


def _slice_asof(fleet: dict[str, pd.DataFrame], asof) -> dict[str, pd.DataFrame]:
    """Fleet as it would have looked on `asof` — each well truncated to rows
    with date <= asof. Empty frames stay empty (scan_fleet tolerates them)."""
    out = {}
    for well_id, df in fleet.items():
        if len(df) and "date" in df.columns:
            out[well_id] = df[df["date"] <= asof].reset_index(drop=True)
        else:
            out[well_id] = df
    return out


def build_ledger(
    fleet: dict[str, pd.DataFrame],
    window_days: int = 30,
    price_per_bbl: float = DEFAULT_OIL_PRICE,
    acknowledged=None,
) -> tuple[pd.DataFrame, dict]:
    """Replay the deterministic scan day-by-day and aggregate deferred bbl/$ by
    cause into a cumulative period ledger.

    Parameters
    ----------
    fleet : dict[well_id -> SCADA DataFrame] (as returned by ``load_fleet``).
    window_days : trailing window length, counted in *calendar days present in
        the data* ending at the latest date. Days without enough history to
        scan are skipped silently.
    price_per_bbl, acknowledged : passed straight through to ``scan_fleet`` so
        the economics and suppressions match the live brief exactly.

    Returns
    -------
    (ledger, summary)
      ledger : tidy DataFrame [date, cause, deferred_bbl, deferred_usd,
        cumulative_usd] — one row per (day, cause) that had deferred production.
        Sorted by date then cause; ``cumulative_usd`` is the running period
        total of ``deferred_usd`` and is monotonically non-decreasing.
      summary : dict {period_deferred_usd, recoverable_usd, top_cause,
        top_cause_usd, period_deferred_bbl, days_scanned, window_start,
        window_end}. Zeros / None on an empty or all-quiet fleet.
    """
    all_dates = _all_dates(fleet)
    empty_summary = {
        "period_deferred_usd": 0.0,
        "recoverable_usd": 0.0,
        "top_cause": None,
        "top_cause_usd": 0.0,
        "period_deferred_bbl": 0.0,
        "days_scanned": 0,
        "window_start": None,
        "window_end": None,
    }
    if not all_dates:
        return pd.DataFrame(columns=LEDGER_COLUMNS), empty_summary

    window_dates = all_dates[-window_days:]
    rows: list[dict] = []
    for asof in window_dates:
        # Skip days too early to have a scannable baseline for any well.
        max_hist = max(
            (int((df["date"] <= asof).sum()) for df in fleet.values()
             if len(df) and "date" in df.columns),
            default=0,
        )
        if max_hist < MIN_HISTORY:
            continue
        asof_fleet = _slice_asof(fleet, asof)
        anomalies = scan_fleet(asof_fleet, price_per_bbl=price_per_bbl,
                               acknowledged=acknowledged)
        # Run-rate accrual: only unacknowledged, actively-deferring anomalies.
        per_cause_bbl: dict[str, float] = {}
        per_cause_usd: dict[str, float] = {}
        for a in anomalies:
            if a.acknowledged or a.deferred_usd_per_day <= 0:
                continue
            per_cause_bbl[a.category] = per_cause_bbl.get(a.category, 0.0) + a.deferred_bopd
            per_cause_usd[a.category] = per_cause_usd.get(a.category, 0.0) + a.deferred_usd_per_day
        for cause in sorted(per_cause_usd):
            rows.append({
                "date": asof,
                "cause": cause,
                "deferred_bbl": round(per_cause_bbl[cause], 1),
                "deferred_usd": round(per_cause_usd[cause], 0),
            })

    days_scanned = len({r["date"] for r in rows}) if rows else sum(
        1 for asof in window_dates
        if max((int((df["date"] <= asof).sum()) for df in fleet.values()
                if len(df) and "date" in df.columns), default=0) >= MIN_HISTORY
    )

    if not rows:
        empty_summary["days_scanned"] = days_scanned
        empty_summary["window_start"] = window_dates[0]
        empty_summary["window_end"] = window_dates[-1]
        return pd.DataFrame(columns=LEDGER_COLUMNS), empty_summary

    ledger = pd.DataFrame(rows).sort_values(["date", "cause"]).reset_index(drop=True)
    ledger["cumulative_usd"] = ledger["deferred_usd"].cumsum()
    ledger = ledger[LEDGER_COLUMNS]

    period_usd = float(ledger["deferred_usd"].sum())
    period_bbl = float(ledger["deferred_bbl"].sum())
    by_cause = ledger.groupby("cause")["deferred_usd"].sum().sort_values(ascending=False)
    top_cause = by_cause.index[0]
    summary = {
        "period_deferred_usd": round(period_usd, 0),
        "recoverable_usd": round(period_usd * RECOVERABLE_FRACTION, 0),
        "top_cause": top_cause,
        "top_cause_usd": round(float(by_cause.iloc[0]), 0),
        "period_deferred_bbl": round(period_bbl, 1),
        "days_scanned": days_scanned,
        "window_start": window_dates[0],
        "window_end": window_dates[-1],
    }
    return ledger, summary
