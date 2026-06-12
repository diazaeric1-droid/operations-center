"""Persistent event state machine for the morning brief.

WHY THIS EXISTS
---------------
``anomaly_detector.scan_fleet`` is *stateless*: it only looks at the recent
trailing window (the last 1/5/7/8/14 days, per rule). That is correct for *first*
detection, but it has no memory. A real outage that lasts ten days fires NEW for a
couple of days and then **silently vanishes** from the brief once the dropped
production level has aged into the rolling baseline (the baseline absorbs the new,
lower level, so "today vs baseline" looks normal again). A control-room PE would
never accept a still-down well disappearing from the brief on day 4.

This module adds the missing memory: a small persisted **event store** plus a
lifecycle that runs each day on top of the existing scan.

LIFECYCLE
---------
Each event is keyed by ``(well_id, event_type, start_date)`` and moves through:

    NEW       first day we detected it
    ONGOING   still abnormal on a subsequent day (whether the raw detector still
              fires OR production is simply still depressed vs the pre-event
              baseline — this is the bug fix: the event persists past the
              detector's lookback window)
    RESOLVED  production recovered back within the normal band (or the raw
              signal cleared for a non-rate event); shown once more, then dropped
    (dropped) after ``POST_RESOLUTION_DAYS`` of post-resolution mention

The crucial transition is ONGOING-without-a-fresh-detection. For a rate-loss
event we capture the **pre-event baseline** at NEW time; on later days, even when
``scan_fleet`` has gone quiet, if today's production is still materially below that
baseline the event stays ONGOING and we keep accruing deferred bbl/$. Recovery
back inside the band is what RESOLVES it. For non-rate events (intake collapse,
amps creep, runtime, temp, data-quality) we keep them open while the raw detector
fires and resolve them after a grace period of clean polls.

PERSISTENCE
-----------
Stored in SQLite via the stdlib ``sqlite3`` module — the same lightweight,
dependency-free pattern the repo already uses for the historian adapter
(``src.sources.SQLiteFleetSource``). One ``events`` table; the store is keyed so a
re-run of the same day is **idempotent** (re-processing 2026-05-29 twice does not
double-count). The morning brief reads open + just-resolved events from here, so
ongoing events show a running **duration** and **cumulative deferred barrels**
instead of being re-detected as brand-new each morning or dropping off the edge of
the window.

This layer does NOT change detection thresholds, the acknowledge/suppress
behavior, or the deferred-bbl/$ economics — every dollar still comes from
``Anomaly.deferred_bopd * price`` exactly as the live scan computes it. It only
remembers what the stateless scan forgets.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .anomaly_detector import Anomaly, DEFAULT_OIL_PRICE, scan_fleet

# Default on-disk location for the event store. Lives next to the briefs so the
# whole "what the agent remembers across runs" surface is in one place. Tests pass
# an explicit ``:memory:`` or temp path instead.
DEFAULT_EVENT_DB = "briefs/events.db"

# Lifecycle states.
NEW = "NEW"
ONGOING = "ONGOING"
RESOLVED = "RESOLVED"

# Categories that carry rate-loss economics. For these we can keep an event open
# purely from the production level (baseline memory) even after the stateless
# detector goes quiet — that is the core fix. Non-rate categories rely on the raw
# detector firing plus a clean-poll grace period to resolve.
RATE_CATEGORIES = {"rate_drop", "rate_drop_decline_aware"}

# An open rate event is considered RECOVERED (and so RESOLVED) once today's rate
# climbs back to within this fraction of its pre-event baseline. 0.90 == "within
# 10% of normal". This is intentionally a hair looser than the 15% *detection*
# threshold so an event doesn't flap RESOLVED→reopened on a single noisy day that
# is still ~12% down; recovery should be a genuine return to band.
RECOVERY_BAND = 0.90

# Days a RESOLVED event remains in the brief (the "back to normal — closing out"
# mention) before it drops off entirely.
POST_RESOLUTION_DAYS = 1

# Grace days a non-rate event may go without a fresh detection before we treat the
# signal as cleared and RESOLVE it. Tolerates a one-day comms gap / poll miss.
NONRATE_GRACE_DAYS = 1


@dataclass
class Event:
    """One tracked event across its lifetime. Mirrors a row in the ``events`` table.

    ``deferred_bopd``/``deferred_usd`` are *cumulative* over the event's life (the
    running total a base-management review wants), distinct from the per-day
    ``last_deferred_bopd`` snapshot used to render today's rate. ``baseline_bopd``
    is the pre-event production level captured at open time, which is what lets a
    rate event stay ONGOING after the stateless detector's window has moved on.
    """
    well_id: str
    event_type: str
    start_date: str                 # ISO date (YYYY-MM-DD) — part of the key
    state: str = NEW
    last_seen_date: str = ""        # most recent day this event was processed
    last_detected_date: str = ""    # most recent day the raw detector actually fired
    resolved_date: str = ""         # ISO date production returned to band ("" until RESOLVED)
    duration_days: int = 1          # inclusive span start_date..last_seen_date
    deferred_bopd: float = 0.0      # cumulative deferred barrels over the event
    deferred_usd: float = 0.0       # cumulative deferred $ over the event
    last_deferred_bopd: float = 0.0 # today's deferred barrels (for the current-rate line)
    last_deferred_usd: float = 0.0  # today's deferred $
    baseline_bopd: float = 0.0      # pre-event production baseline (rate events)
    severity: str = "MEDIUM"        # latest severity seen
    headline: str = ""              # latest raw headline (for narration)
    recommended_action: str = ""
    acknowledged: bool = False      # mirrors the latest scan's ack flag
    post_resolution_days: int = 0   # how many days we've shown it since RESOLVED

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.well_id, self.event_type, self.start_date)


# ---- SQLite store -----------------------------------------------------------

_COLUMNS = [
    "well_id", "event_type", "start_date", "state", "last_seen_date",
    "last_detected_date", "resolved_date", "duration_days", "deferred_bopd",
    "deferred_usd", "last_deferred_bopd", "last_deferred_usd", "baseline_bopd",
    "severity", "headline", "recommended_action", "acknowledged",
    "post_resolution_days",
]


class EventStore:
    """Thin persistence layer over a single ``events`` SQLite table.

    Keyed by ``(well_id, event_type, start_date)`` (a composite PRIMARY KEY) so the
    daily lifecycle is idempotent: ``upsert`` replaces the row for a key rather
    than appending, and re-running a given as-of day twice yields the same store.
    stdlib-only, matching ``src.sources.SQLiteFleetSource``.
    """

    def __init__(self, db_path: str | Path = DEFAULT_EVENT_DB):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = None
        else:
            # An in-memory DB only persists while its connection is open, so hold one.
            self._conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return self._conn if self._conn is not None else sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    well_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    state TEXT NOT NULL,
                    last_seen_date TEXT,
                    last_detected_date TEXT,
                    resolved_date TEXT,
                    duration_days INTEGER,
                    deferred_bopd REAL,
                    deferred_usd REAL,
                    last_deferred_bopd REAL,
                    last_deferred_usd REAL,
                    baseline_bopd REAL,
                    severity TEXT,
                    headline TEXT,
                    recommended_action TEXT,
                    acknowledged INTEGER,
                    post_resolution_days INTEGER,
                    PRIMARY KEY (well_id, event_type, start_date)
                )
                """
            )
            conn.commit()
        finally:
            if self._conn is None:
                conn.close()

    def _row_to_event(self, row: tuple) -> Event:
        d = dict(zip(_COLUMNS, row))
        d["acknowledged"] = bool(d["acknowledged"])
        return Event(**d)

    def all_events(self) -> list[Event]:
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM events"
            ).fetchall()
        finally:
            if self._conn is None:
                conn.close()
        return [self._row_to_event(r) for r in rows]

    def open_events(self) -> list[Event]:
        """Events not yet dropped — NEW/ONGOING, plus RESOLVED still in their
        post-resolution mention window."""
        return [e for e in self.all_events() if e.state in (NEW, ONGOING) or
                (e.state == RESOLVED and e.post_resolution_days < POST_RESOLUTION_DAYS)]

    def upsert(self, event: Event) -> None:
        vals = [getattr(event, c) for c in _COLUMNS]
        vals[_COLUMNS.index("acknowledged")] = int(event.acknowledged)
        placeholders = ", ".join(["?"] * len(_COLUMNS))
        conn = self._connect()
        try:
            conn.execute(
                f"INSERT OR REPLACE INTO events ({', '.join(_COLUMNS)}) "
                f"VALUES ({placeholders})",
                vals,
            )
            conn.commit()
        finally:
            if self._conn is None:
                conn.close()

    def delete(self, event: Event) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "DELETE FROM events WHERE well_id=? AND event_type=? AND start_date=?",
                event.key,
            )
            conn.commit()
        finally:
            if self._conn is None:
                conn.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# ---- lifecycle --------------------------------------------------------------

def _current_bopd(scada: pd.DataFrame) -> float | None:
    """Latest non-NaN oil rate from a well frame, or None."""
    if scada is None or not len(scada) or "bopd" not in scada.columns:
        return None
    val = scada.iloc[-1]["bopd"]
    return float(val) if pd.notna(val) else None


def _baseline_bopd(scada: pd.DataFrame, lookback: int = 7) -> float:
    """Pre-event production baseline: mean of the rows *before* the last day, which
    is the level the rate event represents a deferral from. Mirrors how
    ``detect_rate_drop`` forms its baseline (the 7 days preceding today)."""
    if scada is None or len(scada) < 2 or "bopd" not in scada.columns:
        return 0.0
    prior = scada.iloc[-(lookback + 1):-1]["bopd"]
    prior = prior[prior.notna()]
    return float(prior.mean()) if len(prior) else 0.0


def _days_between(a: str, b: str) -> int:
    """Inclusive day span from ISO date ``a`` to ISO date ``b`` (>=1)."""
    da = datetime.fromisoformat(a).date()
    db = datetime.fromisoformat(b).date()
    return (db - da).days + 1


def update_events(
    store: EventStore,
    fleet: dict[str, pd.DataFrame],
    as_of: str,
    price_per_bbl: float = DEFAULT_OIL_PRICE,
    acknowledged=None,
) -> list[Event]:
    """Advance the event state machine by one day and persist the result.

    Runs the existing stateless ``scan_fleet`` for ``as_of``, then reconciles the
    raw detections against the persisted open events:

      * a detection that matches an open event  -> ONGOING (bump duration + accrue)
      * a detection with no open event          -> open a NEW event
      * an open RATE event with no detection     -> stays ONGOING while production is
        still below ``RECOVERY_BAND`` of its pre-event baseline; RESOLVED once it
        recovers into band  (**the fix** — survives the detector's lookback window)
      * an open NON-RATE event with no detection -> RESOLVED after ``NONRATE_GRACE_DAYS``
      * a RESOLVED event                         -> ages out after ``POST_RESOLUTION_DAYS``

    Idempotent for a given ``as_of``: a row is only advanced if ``as_of`` is newer
    than its ``last_seen_date``, so re-processing the same day twice is a no-op.

    Returns the list of events that are "live for the brief" today (NEW + ONGOING +
    just-RESOLVED within their post-resolution mention), money-first sorted to match
    ``scan_fleet``.
    """
    anomalies = scan_fleet(fleet, price_per_bbl=price_per_bbl, acknowledged=acknowledged)
    # Index today's raw detections by (well_id, category). At most one per pair.
    detected: dict[tuple[str, str], Anomaly] = {}
    for a in anomalies:
        detected[(a.well_id, a.category)] = a

    existing = {e.key[:2]: e for e in store.open_events()}  # (well, type) -> open event

    touched: list[Event] = []
    handled_keys: set[tuple[str, str]] = set()

    # 1) Walk today's detections: continue a matching open event or open a NEW one.
    for (well_id, category), anom in detected.items():
        handled_keys.add((well_id, category))
        scada = fleet.get(well_id)
        ev = existing.get((well_id, category))
        if ev is None:
            # Brand-new event.
            base = _baseline_bopd(scada) if category in RATE_CATEGORIES else 0.0
            ev = Event(
                well_id=well_id, event_type=category, start_date=as_of, state=NEW,
                last_seen_date=as_of, last_detected_date=as_of, duration_days=1,
                deferred_bopd=round(anom.deferred_bopd, 1),
                deferred_usd=round(anom.deferred_usd_per_day, 0),
                last_deferred_bopd=round(anom.deferred_bopd, 1),
                last_deferred_usd=round(anom.deferred_usd_per_day, 0),
                baseline_bopd=round(base, 1),
                severity=anom.severity, headline=anom.headline,
                recommended_action=anom.recommended_action,
                acknowledged=anom.acknowledged,
            )
            store.upsert(ev)
            touched.append(ev)
            continue
        # Continue an existing open event (idempotency guard on as_of).
        if ev.last_seen_date and ev.last_seen_date >= as_of:
            touched.append(ev)
            continue
        ev.state = ONGOING
        ev.duration_days = _days_between(ev.start_date, as_of)
        ev.last_seen_date = as_of
        ev.last_detected_date = as_of
        ev.deferred_bopd = round(ev.deferred_bopd + anom.deferred_bopd, 1)
        ev.deferred_usd = round(ev.deferred_usd + anom.deferred_usd_per_day, 0)
        ev.last_deferred_bopd = round(anom.deferred_bopd, 1)
        ev.last_deferred_usd = round(anom.deferred_usd_per_day, 0)
        ev.severity = anom.severity
        ev.headline = anom.headline
        ev.recommended_action = anom.recommended_action
        ev.acknowledged = anom.acknowledged
        if ev.baseline_bopd <= 0 and category in RATE_CATEGORIES:
            ev.baseline_bopd = round(_baseline_bopd(scada), 1)
        store.upsert(ev)
        touched.append(ev)

    # 2) Walk open events with NO detection today — the core of the fix.
    for (well_id, category), ev in existing.items():
        if (well_id, category) in handled_keys:
            continue
        if ev.last_seen_date and ev.last_seen_date >= as_of:
            touched.append(ev)
            continue
        scada = fleet.get(well_id)

        if ev.state == RESOLVED:
            # Already resolved on a prior day; age the post-resolution mention.
            ev.post_resolution_days += 1
            ev.last_seen_date = as_of
            if ev.post_resolution_days >= POST_RESOLUTION_DAYS:
                store.delete(ev)              # drop off the brief entirely
            else:
                store.upsert(ev)
                touched.append(ev)
            continue

        if category in RATE_CATEGORIES:
            cur = _current_bopd(scada)
            base = ev.baseline_bopd or _baseline_bopd(scada)
            still_down = (
                cur is not None and base > 0 and cur < base * RECOVERY_BAND
            )
            if still_down:
                # STILL ABNORMAL though the stateless detector went quiet → ONGOING.
                # Keep accruing the standing deferral (baseline − current rate).
                day_defer = max(base - cur, 0.0)
                day_usd = round(day_defer * price_per_bbl, 0)
                ev.state = ONGOING
                ev.duration_days = _days_between(ev.start_date, as_of)
                ev.last_seen_date = as_of
                ev.deferred_bopd = round(ev.deferred_bopd + day_defer, 1)
                ev.deferred_usd = round(ev.deferred_usd + day_usd, 0)
                ev.last_deferred_bopd = round(day_defer, 1)
                ev.last_deferred_usd = day_usd
                ev.headline = (f"Still down — {day_defer:.0f} BOPD below pre-event "
                               f"baseline ({base:.0f} → {cur:.0f}); detector window "
                               f"elapsed but production has not recovered")
                store.upsert(ev)
                touched.append(ev)
            else:
                _resolve(ev, as_of, cur)
                store.upsert(ev)
                touched.append(ev)
        else:
            # Non-rate event: resolve after the clean-poll grace period.
            gap = _days_between(ev.last_detected_date or ev.start_date, as_of) - 1
            if gap > NONRATE_GRACE_DAYS:
                _resolve(ev, as_of, _current_bopd(scada))
                store.upsert(ev)
                touched.append(ev)
            else:
                # Within grace: hold ONGOING (a one-day poll gap shouldn't close it).
                ev.state = ONGOING
                ev.duration_days = _days_between(ev.start_date, as_of)
                ev.last_seen_date = as_of
                ev.last_deferred_bopd = 0.0
                ev.last_deferred_usd = 0.0
                store.upsert(ev)
                touched.append(ev)

    return _sort_for_brief(touched)


def _resolve(ev: Event, as_of: str, cur_bopd: float | None) -> None:
    """Mark an event RESOLVED as of ``as_of`` (production back in band / signal clear)."""
    ev.state = RESOLVED
    ev.resolved_date = as_of
    ev.last_seen_date = as_of
    ev.duration_days = _days_between(ev.start_date, as_of)
    ev.last_deferred_bopd = 0.0
    ev.last_deferred_usd = 0.0
    ev.post_resolution_days = 0
    recovered = (f"{cur_bopd:.0f} BOPD" if cur_bopd is not None else "signal cleared")
    ev.headline = f"Recovered — back within normal band ({recovered}); closing out"


def _sort_for_brief(events: list[Event]) -> list[Event]:
    """Money-first ordering matching ``scan_fleet``: unacknowledged before
    acknowledged, then HIGH→MEDIUM→LOW, then by *cumulative* deferred $ desc, then
    well_id. RESOLVED events sink below still-open ones of the same severity."""
    sev = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    state_rank = {NEW: 0, ONGOING: 0, RESOLVED: 1}
    return sorted(
        events,
        key=lambda e: (e.acknowledged, state_rank.get(e.state, 0),
                       sev.get(e.severity, 3), -e.deferred_usd, e.well_id),
    )
