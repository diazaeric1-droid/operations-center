"""Well potential (entitlement) model — the deterministic backbone of deferment.

Deferment only means something against a *potential*: the rate a well WOULD make if
fully up and unconstrained. Estimating it without circular reasoning:

  1. Use only **full-uptime records** (runtime ≥ 90%) — records where the well was
     essentially up, so their *producing-day* rate reflects capability, not downtime.
     (Down/partial records are excluded so they can't drag the estimate down.)
  2. Take a trailing **upper-ish quantile (P75)** of those up-record producing-day
     rates over a rolling **calendar-day** window. P75 (not the median) biases toward
     the well's better days so a stretch of *curtailed-but-up* records can't quietly
     redefine capability; the rolling window lets capability decline naturally over time.

CADENCE-AWARENESS
-----------------
The model is **time-based**, not row-count-based. ``window`` is a span in *calendar
days* (default 28). A daily record covers ~1 calendar day, so ~28 rows fall in the
window; a monthly record covers ~30 calendar days, so only the most recent ~1 row
does. The capability estimate therefore spans the same real time horizon whether the
input is daily (synthetic) or monthly (real Colorado/NDIC), instead of silently
becoming a 28-*month* (~2.3 yr) window on monthly data.

Capability is expressed as a **producing-day rate** (BOPD while up): for a monthly
record that is ``oil_bbl / days_produced``; for a daily record it is the day's rate
grossed up for partial runtime. The potential *volume* over a record's calendar span
is then ``capability_rate × calendar_days`` — see ``deferment.py``.

Pair this with the deadband in ``deferment.py`` (losses under ~8% of potential are
treated as measurement noise, not deferment) so a healthy well reads ~0 deferred.
Transparent and decline-aware — the kind of model an ops/reserves engineer will accept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WINDOW = 28      # trailing **calendar days** for the capability estimate
DEFAULT_Q = 0.75         # quantile of up-record producing-day rates representing capability
UP_RUNTIME = 90.0        # a record at/above this runtime % counts toward capability


def record_spans(prod: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(calendar_days, producing_days) per record — the cadence-aware time basis.

    Both are returned as float Series aligned to ``prod``'s index, in *days*.

    - If the loader supplied explicit spans (monthly NDIC/ECMC carries ``days`` and
      ``days_in_month``), they are used verbatim: ``calendar_days = days_in_month``,
      ``producing_days = days``. Downtime lives in the gap ``calendar_days −
      producing_days``.
    - Otherwise the cadence is inferred from the ``date`` gaps (a daily series → a
      ~1-day span/record). ``producing_days`` then comes from ``runtime_pct`` so a partial
      day counts as partially up. This makes a daily record's calendar span explicit
      instead of assuming "one row == one day" everywhere downstream.
    """
    n = len(prod)
    runtime = (prod["runtime_pct"].clip(lower=0, upper=100) / 100.0).to_numpy(dtype=float)

    if {"days_in_month", "days"}.issubset(prod.columns):
        cal = prod["days_in_month"].to_numpy(dtype=float)
        prod_days = prod["days"].to_numpy(dtype=float)
        cal = np.where(np.isfinite(cal) & (cal > 0), cal, 1.0)
        prod_days = np.clip(np.where(np.isfinite(prod_days), prod_days, 0.0), 0.0, cal)
    else:
        # Infer the calendar span of each record from the spacing of the dates. The span
        # of a record is the gap to the NEXT record (how long this rate stood); the last
        # record inherits the median gap. Daily data → ~1.0; weekly → ~7.0; etc.
        dates = pd.to_datetime(prod["date"]).to_numpy()
        if n >= 2:
            deltas = np.diff(dates).astype("timedelta64[s]").astype(float) / 86400.0
            deltas = deltas[deltas > 0]
            step = float(np.median(deltas)) if deltas.size else 1.0
            gaps = np.diff(dates).astype("timedelta64[s]").astype(float) / 86400.0
            cal = np.empty(n, dtype=float)
            cal[:-1] = np.where(gaps > 0, gaps, step)
            cal[-1] = step
        else:
            cal = np.ones(n, dtype=float)
        cal = np.where(np.isfinite(cal) & (cal > 0), cal, 1.0)
        # Producing-time within the span scales with runtime (downtime = the rest).
        prod_days = np.clip(runtime, 0.0, 1.0) * cal

    idx = prod.index
    return (pd.Series(cal, index=idx), pd.Series(prod_days, index=idx))


def producing_day_rate(prod: pd.DataFrame) -> pd.Series:
    """Per-record capability rate, BOPD **while producing** (NaN where down).

    This is the rate that reflects what the well can do when it is up — the input to
    the capability quantile. For a monthly record it is ``oil_bbl / days_produced``
    (already what ``bopd`` is in the NDIC/ECMC loader); for a daily record it is the
    day's rate grossed up for partial runtime (``bopd / runtime_fraction``). Records
    below ``UP_RUNTIME`` are masked so downtime can't depress the estimate.
    """
    cal, prod_days = record_spans(prod)
    bopd = prod["bopd"].to_numpy(dtype=float)
    runtime_pct = prod["runtime_pct"].clip(lower=0, upper=100).to_numpy(dtype=float)

    if {"days_in_month", "days"}.issubset(prod.columns):
        # bopd is already a producing-day rate (oil_bbl / days). Use it directly.
        up_rate = bopd.copy()
    else:
        # bopd is a calendar-day average; gross up by runtime to a producing-day rate.
        r = np.clip(runtime_pct / 100.0, 0.9, 1.0)
        up_rate = bopd / r

    up = runtime_pct >= UP_RUNTIME
    return pd.Series(np.where(up, up_rate, np.nan), index=prod.index)


def well_potential(prod: pd.DataFrame, window: int = DEFAULT_WINDOW,
                   q: float = DEFAULT_Q) -> pd.Series:
    """Per-record potential (entitlement) **producing-day** rate, BOPD.

    Index aligns with ``prod``. Cadence-aware: the trailing quantile is taken over a
    ``window``-**calendar-day** span (time-based), so the same real horizon is used for
    daily and monthly inputs. The result is a producing-day capability rate; multiply
    by a record's calendar days for its potential *volume* (done in ``deferment.py``).

    Potential is floored at the record's own producing-day rate, so it can never be
    below what the well actually achieved while up.
    """
    cal, prod_days = record_spans(prod)
    rate_up = producing_day_rate(prod)

    # Time-indexed rolling quantile: a calendar-day window, not a fixed row count. Index
    # by the cumulative calendar day of each record so ``window`` means days for any
    # cadence. min_periods=1 keeps early records defined; ffill/bfill cover all-NaN heads.
    cum_days = cal.cumsum()
    tmp = pd.DataFrame({"rate_up": rate_up.to_numpy()},
                       index=pd.to_timedelta(cum_days.to_numpy(), unit="D"))
    win = f"{int(max(window, 1))}D"
    cap = tmp["rate_up"].rolling(win, min_periods=1).quantile(q).to_numpy()

    own = producing_day_rate_filled(prod).to_numpy()
    cap = pd.Series(cap, index=prod.index).ffill().bfill()
    cap = cap.fillna(pd.Series(own, index=prod.index))
    return pd.Series(np.maximum(cap.to_numpy(), own), index=prod.index)


def producing_day_rate_filled(prod: pd.DataFrame) -> pd.Series:
    """Producing-day rate with down-record gaps filled by the observed average rate.

    Used only as the floor / fallback for ``well_potential`` so the potential is never
    NaN and never below the well's own up-rate, even on a record where it was down.
    """
    cal, prod_days = record_spans(prod)
    bopd = prod["bopd"].to_numpy(dtype=float)
    if {"days_in_month", "days"}.issubset(prod.columns):
        own = bopd.copy()
    else:
        r = np.clip(prod["runtime_pct"].clip(lower=0, upper=100).to_numpy(dtype=float) / 100.0,
                    0.9, 1.0)
        own = bopd / r
    s = pd.Series(own, index=prod.index)
    return s.fillna(s.median()).fillna(0.0)
