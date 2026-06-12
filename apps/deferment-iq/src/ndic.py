"""North Dakota (NDIC) real-data adapter — public monthly Bakken production.

The North Dakota Industrial Commission (NDIC / Department of Mineral Resources)
publishes **per-well MONTHLY** production filings for every well in the state:
oil (bbl), gas (mcf), water (bbl), and the number of **days produced** in the
month. That ``days`` field is a genuine downtime signal — downtime for the month
is ``days_in_month − days``, so the same potential-vs-actual + downtime/rate
decomposition the synthetic engine runs applies directly to real Bakken wells.

What the public data does NOT carry: **reason codes / operator cause notes**. The
deferment QUANTITY (how many barrels were lost, and how much was downtime vs.
underperformance) is fully real; the *cause attribution* has no public input, so
on real data the cause is reported as "uncoded / unknown" — N/A, not fabricated.

This adapter maps a tidy monthly NDIC extract into the EXACT fleet structure the
deferment engine already consumes (``dict[well_id -> DataFrame]`` with columns
``date, bopd, bfpd, gas_mcfd, runtime_pct``), at a monthly cadence:

    bopd      = oil_bbl   / max(days, 1)        # avg oil rate over producing days
    bfpd      = (oil + water) / max(days, 1)    # total fluid (oil + water)
    gas_mcfd  = gas_mcf   / max(days, 1)
    runtime_pct = days / days_in_month * 100    # REAL uptime from days-produced

So ``well_potential`` (P75 of full-uptime months) and ``compute_deferment``
(potential − actual, split into downtime vs. rate) run unchanged on it. No API
key, no network — the app reads a local extract a user drops in.

Input CSV schema (tidy, one row per well-month)::

    well_id, well_name, operator, field, formation, date, oil_bbl, gas_mcf, water_bbl, days

``date`` is the production month as ``YYYY-MM`` (a full ``YYYY-MM-DD`` is also
accepted and normalized to the month). All other columns are required; numeric
columns coerce and bad/blank rows are dropped defensively.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import pandas as pd

# Tidy monthly schema the public NDIC export is reshaped into (see data/real/ndic/README.md).
NDIC_COLUMNS = [
    "well_id", "well_name", "operator", "field", "formation",
    "date", "oil_bbl", "gas_mcf", "water_bbl", "days",
]

# On real data there is no operator cause note / reason code in the public filing.
# The deferment quantity is real; the cause is uncoded. The app degrades gracefully
# on this rather than inventing a cause.
REAL_DATA_CAUSE_NOTE = (
    "cause attribution N/A — NDIC public monthly filings carry no reason codes"
)


def _days_in_month(ts: pd.Timestamp) -> int:
    return calendar.monthrange(int(ts.year), int(ts.month))[1]


def parse_ndic_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read + validate a tidy monthly NDIC CSV into a long DataFrame.

    Returns one row per well-month with the source columns plus a parsed
    month-start ``date`` (``Timestamp``). Raises ``ValueError`` if required
    columns are missing or no usable rows remain.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"NDIC extract not found: {p}")

    df = pd.read_csv(p)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in NDIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"NDIC CSV missing columns {missing}; expected {NDIC_COLUMNS}")

    df = df[NDIC_COLUMNS].copy()
    # Parse the production month (accept YYYY-MM or full dates), normalize to month start.
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    for col in ("oil_bbl", "gas_mcf", "water_bbl", "days"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["well_id"] = df["well_id"].astype(str).str.strip()

    # Drop rows we can't use: no id, no month, or non-positive producing days.
    df = df.dropna(subset=["date", "oil_bbl", "days"])
    df = df[(df["well_id"] != "") & (df["days"] > 0)]
    if df.empty:
        raise ValueError(f"NDIC CSV {p.name} has no usable production rows after parsing")

    return df.sort_values(["well_id", "date"]).reset_index(drop=True)


def _well_frame(g: pd.DataFrame) -> pd.DataFrame:
    """One well's monthly rows → the daily-engine schema (monthly cadence).

    Rates are per *producing* day (oil_bbl / days); runtime_pct comes from the
    real days-produced vs. days-in-month, which is the downtime signal.
    """
    g = g.sort_values("date").reset_index(drop=True)
    days = g["days"].clip(lower=0)
    safe_days = days.where(days > 0, 1.0)  # guard /0; rows with days<=0 already dropped
    dim = g["date"].map(_days_in_month).astype(float)

    oil = g["oil_bbl"].fillna(0.0).clip(lower=0.0)
    water = g["water_bbl"].fillna(0.0).clip(lower=0.0)
    gas = g["gas_mcf"].fillna(0.0).clip(lower=0.0)

    bopd = oil / safe_days
    bfpd = (oil + water) / safe_days
    gas_mcfd = gas / safe_days
    # Uptime = producing days / days-in-month, capped at 100% (a filing can report
    # days == days_in_month). This is the real downtime fraction the engine uses.
    runtime_pct = (days / dim * 100.0).clip(lower=0.0, upper=100.0)

    return pd.DataFrame({
        "date": g["date"].values,
        "bopd": bopd.to_numpy(dtype=float),
        "bfpd": bfpd.to_numpy(dtype=float),
        "gas_mcfd": gas_mcfd.to_numpy(dtype=float),
        "runtime_pct": runtime_pct.to_numpy(dtype=float),
        "days": days.to_numpy(dtype=float),               # producing days (real downtime input)
        "days_in_month": dim.to_numpy(dtype=float),
    }).sort_values("date").reset_index(drop=True)


def load_ndic_fleet(csv_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load a tidy monthly NDIC extract into the shared fleet structure.

    Returns ``dict[well_id -> DataFrame]`` with the same columns ``load_fleet``
    produces (``date, bopd, bfpd, gas_mcfd, runtime_pct``) plus ``days`` /
    ``days_in_month`` for transparency, at a monthly cadence. The result is a
    drop-in for ``compute_deferment`` — potential, downtime, and rate deferment
    all compute on it. Cause attribution is N/A on real data (no public reason
    codes); see ``REAL_DATA_CAUSE_NOTE``.
    """
    df = parse_ndic_csv(csv_path)
    fleet: dict[str, pd.DataFrame] = {}
    for well_id, g in df.groupby("well_id", sort=True):
        fleet[str(well_id)] = _well_frame(g)
    return fleet


def ndic_well_meta(csv_path: str | Path) -> pd.DataFrame:
    """Per-well identity from the extract (well_id, name, operator, field, formation).

    Used only for display labels on the real-data pages; one row per well_id.
    """
    df = parse_ndic_csv(csv_path)
    meta = (df.sort_values("date")
            .groupby("well_id", as_index=False)
            .agg(well_name=("well_name", "last"), operator=("operator", "last"),
                 field=("field", "last"), formation=("formation", "last")))
    return meta.sort_values("well_id").reset_index(drop=True)
