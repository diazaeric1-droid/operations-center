"""Ground-truth synthetic fleet for the quantity-recovery eval.

Unlike ``data/synthetic/generate.py`` (which targets the reason-code CLASSIFIER), this
module builds a fleet where the **deferred barrels are known analytically**, so the
deferment ENGINE's quantity accounting can be checked against ground truth — and it
emits the SAME physical fleet at two cadences (daily and monthly) to prove the engine
is cadence-aware (same true barrels either way).

Construction (per well, over a whole number of months so daily↔monthly line up):
  - Flat baseline up-rate ``q`` BOPD. Flat (no decline) so the true potential is exactly
    ``q`` and the engine's decline-aware P75 estimate lands on it — isolating the
    quantity math from potential-estimation noise.
  - **Downtime injection**: ``K`` full-down CALENDAR days (well produces 0, runtime 0).
    True downtime-deferred = ``q × K`` bbl, attributable to a recoverable cause.
  - **Underperformance injection**: ``M`` fully-up days (runtime 100%) at a reduced rate
    ``f·q`` (``f`` well below 1, and below the 8% deadband). True underperformance-
    deferred = ``q × (1 − f) × M`` bbl.
  - All other days: fully up at ``q``.

The two representations of the identical physical history:
  * DAILY  — one row per calendar day with that day's bopd + runtime.
  * MONTHLY (NDIC/ECMC schema) — per month: ``oil_bbl`` = Σ daily bbl, ``days`` =
    producing-day count (down days excluded), so ``bopd = oil_bbl/days`` is the
    producing-day rate and ``runtime = days/days_in_month`` carries the downtime. An
    underperformance day stays a producing day (in ``days``) but pulls the monthly
    producing-day rate below ``q`` → the engine books it as underperformance; a downtime
    day drops out of ``days`` → booked as downtime. Same ground truth, both cadences.

``build_truth_fleet`` returns ``(daily_fleet, monthly_df, truth)`` where ``truth`` lists
the per-well injected barrels (downtime / underperformance / recoverable / total), so the
eval can compare engine output to exact numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Recoverable vs non-recoverable downtime causes (drives the true recoverable barrels).
# Recoverable downtime can be recovered by operator action; reservoir/planned cannot.
RECOVERABLE_NOTE = "ESP tripped on underload, VSD fault — artificial lift"   # artificial_lift
NONRECOVER_NOTE = "planned workover, scheduled maintenance"                  # planned (not recov.)


@dataclass
class WellTruth:
    well_id: str
    q: float                       # baseline up-rate, BOPD
    downtime_bbl: float            # true barrels lost to downtime
    underperf_bbl: float           # true barrels lost to underperformance
    recoverable_bbl: float         # true recoverable barrels (recoverable-cause downtime)
    note: str                      # event note (cause) for the downtime window
    recoverable_cause: bool

    @property
    def total_bbl(self) -> float:
        return self.downtime_bbl + self.underperf_bbl


@dataclass
class FleetTruth:
    wells: list[WellTruth] = field(default_factory=list)

    @property
    def total_deferred_bbl(self) -> float:
        return float(sum(w.total_bbl for w in self.wells))

    @property
    def total_downtime_bbl(self) -> float:
        return float(sum(w.downtime_bbl for w in self.wells))

    @property
    def total_underperf_bbl(self) -> float:
        return float(sum(w.underperf_bbl for w in self.wells))

    @property
    def total_recoverable_bbl(self) -> float:
        return float(sum(w.recoverable_bbl for w in self.wells))


def _month_starts(start: str, n_months: int) -> list[pd.Timestamp]:
    return list(pd.period_range(start=start, periods=n_months, freq="M")
                .to_timestamp())


def build_truth_fleet(n_wells: int = 24, n_months: int = 8, start: str = "2023-01",
                      seed: int = 7):
    """Build the ground-truth fleet at both cadences.

    Returns ``(daily_fleet, monthly_df, truth)``:
      - ``daily_fleet``: ``dict[well_id -> daily DataFrame]`` (engine/``load_fleet`` schema).
      - ``monthly_df``: tidy monthly NDIC/ECMC DataFrame for ``load_ndic_fleet`` (one row
        per well-month).
      - ``truth``: :class:`FleetTruth` with the exact injected barrels per well.
    """
    rng = np.random.default_rng(seed)
    months = _month_starts(start, n_months)
    # Full daily calendar spanning those months.
    cal_start = months[0]
    cal_end = (months[-1] + pd.offsets.MonthEnd(0))
    dates = pd.date_range(cal_start, cal_end, freq="D")
    n_days = len(dates)

    truth = FleetTruth()
    daily_fleet: dict[str, pd.DataFrame] = {}
    monthly_rows: list[dict] = []

    # Day-index slices for each calendar month (so injections can be month-aligned and
    # therefore resolvable at BOTH cadences — a sub-month rate dip is smeared by monthly
    # averaging, so underperformance is injected as a WHOLE month at reduced rate).
    ym_series = pd.Series(dates).dt.to_period("M")
    month_slices: list[tuple[int, int]] = []
    for ym in ym_series.unique():
        idx = np.where(ym_series.to_numpy() == ym)[0]
        month_slices.append((int(idx[0]), int(idx[-1]) + 1))
    if len(month_slices) < 3:
        raise ValueError("build_truth_fleet needs n_months >= 3 for clean injections")

    for i in range(n_wells):
        wid = f"well_{i + 1:03d}"
        q = float(rng.uniform(150, 800))            # baseline up-rate, BOPD
        bopd = np.full(n_days, q)
        runtime = np.full(n_days, 100.0)

        # Choose two DISTINCT non-edge months: one for downtime, one for underperformance.
        # (Month 0 is left clean so the potential estimate locks onto q before any loss.)
        mid = list(range(1, len(month_slices)))
        dt_month, up_month = rng.choice(mid, size=2, replace=False)
        dt_lo, dt_hi = month_slices[int(dt_month)]
        up_lo, up_hi = month_slices[int(up_month)]

        # ---- inject DOWNTIME: a contiguous full-down run of K days WITHIN one month ----
        # Confined to one month so that month's days-produced drops by exactly K while the
        # rate of its remaining producing days stays q (downtime, not underperformance).
        month_len = dt_hi - dt_lo
        K = int(rng.integers(3, min(9, month_len - 1)))
        d_start = dt_lo + int(rng.integers(0, month_len - K))
        bopd[d_start:d_start + K] = 0.0
        runtime[d_start:d_start + K] = 0.0
        downtime_bbl = q * K
        recoverable = bool(rng.random() < 0.65)     # most downtime is recoverable
        note = RECOVERABLE_NOTE if recoverable else NONRECOVER_NOTE

        # ---- inject UNDERPERFORMANCE: a short fully-up run at reduced rate f·q ----
        # Kept SHORT (a few days) and confined to a month with no downtime, so the
        # decline-aware P75 potential stays anchored to the surrounding full-rate days
        # (a long curtailment would look like genuine decline to any trailing model and
        # is intentionally out of scope). At DAILY cadence this is recovered exactly; at
        # MONTHLY cadence a sub-month rate dip is partly smeared by the producing-day
        # average — an inherent, documented limit of monthly data the eval reports openly.
        up_len = up_hi - up_lo
        M = int(rng.integers(4, min(9, up_len)))
        u_start = up_lo + int(rng.integers(0, up_len - M))
        f = float(rng.uniform(0.45, 0.80))          # rate fraction while up
        bopd[u_start:u_start + M] = f * q
        runtime[u_start:u_start + M] = 100.0         # UP, just choked → underperformance
        underperf_bbl = q * (1.0 - f) * M

        # The event note codes ALL of this well's loss to one cause (the engine attributes
        # by covering interval). So for a recoverable-cause well, its full deferred volume
        # (downtime + underperformance) is recoverable opportunity; for a planned-cause
        # well, none of it is. This matches what ``recovery_opportunity`` sums.
        truth.wells.append(WellTruth(
            well_id=wid, q=q, downtime_bbl=downtime_bbl, underperf_bbl=underperf_bbl,
            recoverable_bbl=((downtime_bbl + underperf_bbl) if recoverable else 0.0),
            note=note, recoverable_cause=recoverable))

        # ---- DAILY representation ----
        daily_fleet[wid] = pd.DataFrame({
            "date": dates,
            "bopd": np.round(bopd, 4),
            "bfpd": np.round(bopd * 1.8, 4),
            "gas_mcfd": np.round(bopd * 1.2, 4),
            "runtime_pct": np.round(runtime, 4),
        })

        # ---- MONTHLY representation (aggregate the identical daily history) ----
        df_d = pd.DataFrame({"date": dates, "bopd": bopd, "runtime": runtime})
        df_d["ym"] = df_d["date"].dt.to_period("M")
        for ym, g in df_d.groupby("ym"):
            producing = int((g["runtime"] > 0).sum())          # days the well produced
            oil = float(g["bopd"].sum())                       # Σ daily bbl == month oil
            monthly_rows.append({
                "well_id": wid, "well_name": f"Truth {i+1}H", "operator": "EvalCo",
                "field": "EvalField", "formation": "Niobrara",
                "date": str(ym), "oil_bbl": round(oil, 4),
                "gas_mcf": round(oil * 0.6, 4), "water_bbl": round(oil * 0.4, 4),
                "days": producing,
            })

    monthly_df = pd.DataFrame(monthly_rows)
    return daily_fleet, monthly_df, truth


def build_events(truth: FleetTruth, daily_fleet: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Event log covering each well's downtime window with its (recoverable/planned) note.

    The engine attributes a record's loss to the event whose [start, end] interval
    contains the record's date. We span each well's whole calendar so the loss is coded
    on both cadences (daily rows and the month-start rows of monthly data both fall
    inside the window).
    """
    rows = []
    for w in truth.wells:
        d = daily_fleet[w.well_id]
        rows.append({
            "well_id": w.well_id,
            "start_date": pd.Timestamp(d["date"].iloc[0]),
            "end_date": pd.Timestamp(d["date"].iloc[-1]),
            "note": w.note,
        })
    return pd.DataFrame(rows, columns=["well_id", "start_date", "end_date", "note"])
