"""Deferment engine: per-record lost-oil decomposition + reason-code attribution.

CADENCE-AWARE, TIME-BASED accounting
------------------------------------
The engine works in **calendar-day volume** terms (barrels over each record's real
calendar span), NOT in fixed row-count windows. This is the core correctness fix: a
1-row *monthly* record and a 1-row *daily* record cover very different amounts of
calendar time, so treating "one row == one day" mis-counts potential and deferment on
monthly data (real Colorado ECMC / NDIC filings are monthly). Every record carries an
explicit calendar span (``span_days``) and producing-time (``producing_days``);
downtime lives in the gap between them.

For each record we split the gap between potential and actual into two **volume** (bbl)
buckets over that record's calendar span:

  - downtime deferment      : pot_rate × (calendar_days − producing_days)
                              (well was OFF for part of the span — days-produced gap)
  - underperformance ("rate"): (pot_rate − up_rate) × producing_days
                              (UNDERPERFORMED while up — choked, high line pressure,
                               watering out)

where ``pot_rate`` is the capability (producing-day) rate from ``potential.py`` and
``up_rate`` is the record's own producing-day rate. The two buckets sum exactly to
``max(0, potential_volume − actual_volume)``. An ~8%-of-potential deadband zeroes
within-noise gaps so a healthy well reads ~0 deferred.

Each record's loss is attributed to the cause of the downtime/curtailment EVENT
covering it (classified from its note); records with a loss but no event become
'unclassified' — uncaptured deferment, itself a finding the asset team should chase.

OUTPUT CONTRACT
---------------
``total_def``, ``downtime_def``, ``rate_def`` and ``deferred_usd`` are **volumes**
(bbl / $) over each record's span — so summing them across records gives correct fleet
barrels/$ for any cadence (for daily data a span is 1 day, so a volume equals its
rate numerically and behavior is unchanged). ``bopd`` and ``potential`` remain
per-record **rates** (BOPD) for charting; ``actual_vol`` / ``potential_vol`` /
``span_days`` / ``producing_days`` are provided for transparency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .potential import producing_day_rate_filled, record_spans, well_potential
from .reason_codes import classify, is_planned, is_recoverable, label_for

# Losses smaller than this fraction of potential are measurement/normal-variation noise,
# not deferment — so a healthy well reads ~0 deferred (avoids phantom background loss).
DEADBAND_FRAC = 0.08


def _well_deferment(well_id: str, prod: pd.DataFrame) -> pd.DataFrame:
    """Per-record deferment for one well, in calendar-day **volume** (bbl) terms."""
    cal, prod_days = record_spans(prod)
    cal = cal.to_numpy(dtype=float)
    prod_days = prod_days.to_numpy(dtype=float)

    pot_rate = well_potential(prod).to_numpy(dtype=float)          # producing-day capability, BOPD
    up_rate = producing_day_rate_filled(prod).to_numpy(dtype=float)  # this record's producing-day rate
    bopd = prod["bopd"].to_numpy(dtype=float)                      # as-reported rate (for display)

    # Calendar-day volumes (bbl) over each record's real span.
    potential_vol = pot_rate * cal
    actual_vol = up_rate * prod_days
    gap = np.maximum(potential_vol - actual_vol, 0.0)

    # Deadband: ignore gaps within measurement/normal-variation noise of potential.
    counts = gap > (DEADBAND_FRAC * potential_vol)
    total = np.where(counts, gap, 0.0)

    # Structural, exact split (both terms ≥ 0): downtime = lost calendar time at
    # capability; underperformance = rate shortfall over the producing time.
    downtime = np.maximum(pot_rate * (cal - prod_days), 0.0)
    rate = np.maximum((pot_rate - up_rate) * prod_days, 0.0)
    # Renormalize to the deadbanded total so the two buckets always sum to `total`
    # (and both vanish together when the gap is within the deadband).
    split_sum = downtime + rate
    scale = np.divide(total, split_sum, out=np.zeros_like(total), where=split_sum > 0)
    downtime = downtime * scale
    rate = rate * scale

    out = pd.DataFrame({
        "well_id": well_id,
        "date": prod["date"].values,
        "bopd": bopd,
        "runtime_pct": prod["runtime_pct"].to_numpy(dtype=float),
        "potential": pot_rate,                 # producing-day capability RATE (BOPD) — for charts
        "span_days": cal,                      # calendar days this record covers
        "producing_days": prod_days,           # producing days within the span (downtime = span − this)
        "potential_vol": potential_vol,        # bbl the well could make over the calendar span
        "actual_vol": actual_vol,              # bbl actually made over the span
        "downtime_def": downtime,              # bbl lost to downtime (span − producing days)
        "rate_def": rate,                      # bbl lost to underperformance while up
        "total_def": total,                    # bbl deferred (downtime + rate), deadbanded
    })
    return out


def _attribution_lookup(events: pd.DataFrame, use_llm: bool, client, model: str) -> pd.DataFrame:
    """Classify each event's note once; return events with a reason_key column."""
    if events.empty:
        return events.assign(reason_key=pd.Series(dtype=str))
    ev = events.copy()
    ev["reason_key"] = [classify(n, use_llm=use_llm, client=client, model=model)
                        for n in ev["note"].fillna("")]
    return ev


def classify_events(events: pd.DataFrame, use_llm: bool = False, client=None,
                    model: str = "claude-sonnet-4-6") -> pd.DataFrame:
    """Public: classify each event's note once (adds a ``reason_key`` column).
    Pass the result to ``compute_deferment`` + ``mttr_by_cause`` to avoid re-classifying."""
    return _attribution_lookup(events, use_llm, client, model)


def _reason_for_day(well_id: str, date, ev_by_well: dict) -> str:
    for start, end, key in ev_by_well.get(well_id, ()):  # small per-well interval list
        if start <= date <= end:
            return key
    return "unclassified"


def compute_deferment(fleet: dict[str, pd.DataFrame], events: pd.DataFrame,
                      price_per_bbl: float = 70.0, use_llm: bool = False,
                      client=None, model: str = "claude-sonnet-4-6") -> pd.DataFrame:
    """Per-record deferment table for the whole fleet, attributed + priced.

    Returns one row per well-record with the potential rate, the calendar-day
    downtime/rate volume split (bbl), the assigned reason code (+ label, recoverable,
    planned flags), and deferred $. Cadence-aware — correct for daily and monthly inputs.
    """
    ev = events if "reason_key" in events.columns else _attribution_lookup(events, use_llm, client, model)
    ev_by_well: dict[str, list] = {}
    for _, row in ev.iterrows():
        ev_by_well.setdefault(row["well_id"], []).append(
            (row["start_date"], row["end_date"], row["reason_key"]))

    frames = [_well_deferment(wid, prod) for wid, prod in fleet.items()]
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if daily.empty:
        return daily

    has_loss = daily["total_def"] > 1e-6
    keys = np.where(
        has_loss.to_numpy(),
        [_reason_for_day(w, d, ev_by_well) for w, d in zip(daily["well_id"], daily["date"])],
        "",  # no loss -> no reason
    )
    daily["reason_key"] = keys
    daily["reason_label"] = [label_for(k) if k else "" for k in keys]
    daily["recoverable"] = [bool(k) and is_recoverable(k) for k in keys]
    daily["planned"] = [bool(k) and is_planned(k) for k in keys]
    daily["deferred_usd"] = daily["total_def"] * float(price_per_bbl)
    return daily
