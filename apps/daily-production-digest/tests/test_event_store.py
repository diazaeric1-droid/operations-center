"""Tests for the persistent event state machine (src/event_store.py).

The headline test is the regression for the bug this module fixes: a CONFIRMED,
still-ongoing outage USED TO vanish from the brief on ~day 4 because it aged out
of the stateless detector's lookback window. With the state machine an injected
10-day outage must remain ONGOING on day 5 (and every day until production
recovers). The rest cover the NEW→ONGOING→RESOLVED→drop lifecycle, SQLite
persistence across "process restarts", idempotent re-runs of the same as-of day,
cumulative deferral accrual, and that the acknowledge/suppress flag is preserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.anomaly_detector import scan_fleet
from src.event_store import (
    NEW,
    ONGOING,
    RESOLVED,
    POST_RESOLUTION_DAYS,
    EventStore,
    update_events,
)


# ---- fixtures ---------------------------------------------------------------

def _well(bopd, days, seed=0, **overrides):
    rng = np.random.default_rng(seed)
    base = {
        "date": pd.date_range("2026-05-01", periods=days),
        "bopd": bopd,
        "bfpd": rng.normal(1800, 30, days),
        "intake_pressure_psi": rng.normal(120, 3, days),
        "motor_temp_f": rng.normal(290, 2, days),
        "motor_amps": rng.normal(60, 0.5, days),
        "runtime_pct": np.clip(rng.normal(99, 0.2, days), 0, 100),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _outage_well(total_days, outage_start_idx, outage_len, drop_to=110, normal=200,
                 seed=0, recover=True):
    """A well at ~``normal`` BOPD that steps down to ~``drop_to`` for ``outage_len``
    days starting at ``outage_start_idx`` (0-based). If ``recover`` and there are
    days after the outage, production returns to ``normal``."""
    rng = np.random.default_rng(seed)
    bopd = rng.normal(normal, 5, total_days)
    end = outage_start_idx + outage_len
    bopd[outage_start_idx:end] = rng.normal(drop_to, 4, outage_len)
    if recover:
        bopd[end:] = rng.normal(normal, 5, max(total_days - end, 0))
    return _well(bopd, total_days, seed=seed), pd.date_range("2026-05-01", periods=total_days)


def _replay(store, well_df, dates, start_idx, n_days, well_id="well_X"):
    """Advance the state machine one as-of day at a time across [start_idx, start_idx+n_days)."""
    out = []
    for d in range(start_idx, start_idx + n_days):
        asof = dates[d].date().isoformat()
        live = update_events(store, {well_id: well_df.iloc[: d + 1]}, as_of=asof)
        out.append((asof, live))
    return out


# ---- the regression: a 10-day outage must stay ONGOING on day 5 -------------

def test_ten_day_outage_stays_ongoing_on_day5_regression():
    """REGRESSION: before the state machine a sustained outage vanished from the
    brief on ~day 4 (the dropped level aged into the rolling baseline and the
    stateless detector went quiet). It must now remain ONGOING through day 5 — and
    in fact every day of the 10-day outage."""
    outage_start = 30
    well_df, dates = _outage_well(total_days=45, outage_start_idx=outage_start,
                                  outage_len=10, recover=False)
    store = EventStore(":memory:")

    # Confirm the stateless detector ALONE goes quiet by day 4 (the bug being fixed):
    quiet_by_day4 = False
    for d in range(outage_start, outage_start + 5):
        raw = scan_fleet({"well_X": well_df.iloc[: d + 1]})
        fired = any("rate" in a.category for a in raw)
        if (d - outage_start + 1) >= 4 and not fired:
            quiet_by_day4 = True
    assert quiet_by_day4, "precondition: the stateless detector should fall silent by day 4"

    # Now replay all 10 days through the state machine.
    _replay(store, well_df, dates, outage_start, 10)

    rate_events = [e for e in store.open_events() if "rate" in e.event_type]
    assert rate_events, "the outage must still be an open event, not dropped"
    ev = rate_events[0]
    # Day 5 is the prompt's explicit assertion; by the end of the replay it is day 10.
    assert ev.state == ONGOING
    assert ev.duration_days == 10, "duration must track the full outage span"
    assert ev.deferred_bopd > 0 and ev.deferred_usd > 0, "cumulative deferral accrues"


def test_outage_ongoing_every_day_until_recovery():
    """Day-by-day: NEW on day 1, ONGOING days 2..10, never absent while still down."""
    outage_start = 30
    well_df, dates = _outage_well(total_days=45, outage_start_idx=outage_start,
                                  outage_len=10, recover=False)
    store = EventStore(":memory:")
    states_by_day = []
    for asof, live in _replay(store, well_df, dates, outage_start, 10):
        rate_ev = [e for e in live if "rate" in e.event_type]
        states_by_day.append(rate_ev[0].state if rate_ev else None)

    assert states_by_day[0] == NEW
    assert all(s == ONGOING for s in states_by_day[1:]), states_by_day
    assert None not in states_by_day, "the outage never disappears from the brief"


# ---- full lifecycle: recover then drop off ---------------------------------

def test_lifecycle_new_ongoing_resolved_then_dropped():
    outage_start = 30
    well_df, dates = _outage_well(total_days=45, outage_start_idx=outage_start,
                                  outage_len=8, recover=True)
    store = EventStore(":memory:")
    # Replay the 8-day outage + 3 recovery days.
    seq = _replay(store, well_df, dates, outage_start, 11)

    # Find the day the event becomes RESOLVED (first recovery day after the outage).
    states = []
    for asof, live in seq:
        rate_ev = [e for e in live if "rate" in e.event_type]
        states.append(rate_ev[0].state if rate_ev else "ABSENT")

    assert NEW in states and ONGOING in states and RESOLVED in states
    # After the post-resolution mention window, the event drops off the store.
    remaining = [e for e in store.open_events() if "rate" in e.event_type]
    assert not remaining, "a recovered event drops off after its post-resolution mention"


def test_resolved_event_shown_once_then_gone():
    """A RESOLVED event appears in exactly POST_RESOLUTION_DAYS briefs, then drops."""
    outage_start = 30
    well_df, dates = _outage_well(total_days=45, outage_start_idx=outage_start,
                                  outage_len=6, recover=True)
    store = EventStore(":memory:")
    resolved_appearances = 0
    for asof, live in _replay(store, well_df, dates, outage_start, 10):
        if any(e.state == RESOLVED and "rate" in e.event_type for e in live):
            resolved_appearances += 1
    assert resolved_appearances == POST_RESOLUTION_DAYS


# ---- persistence + idempotency ----------------------------------------------

def test_state_persists_across_store_reopen(tmp_path):
    """The event survives a 'process restart' — reopen the SQLite store and the
    open outage is still there with its accrued duration/deferral."""
    db = tmp_path / "events.db"
    outage_start = 30
    well_df, dates = _outage_well(total_days=45, outage_start_idx=outage_start,
                                  outage_len=10, recover=False)

    store = EventStore(db)
    _replay(store, well_df, dates, outage_start, 5)
    store.close()
    dur_before = [e.duration_days for e in EventStore(db).open_events() if "rate" in e.event_type][0]

    # Reopen (simulating tomorrow's cron run) and advance further.
    store2 = EventStore(db)
    _replay(store2, well_df, dates, outage_start + 5, 5)
    open_events = [e for e in store2.open_events() if "rate" in e.event_type]
    store2.close()
    assert open_events and open_events[0].state == ONGOING
    assert open_events[0].duration_days == 10 > dur_before


def test_rerunning_same_day_is_idempotent():
    """Re-processing the same as-of day must not double-count the cumulative deferral."""
    outage_start = 30
    well_df, dates = _outage_well(total_days=45, outage_start_idx=outage_start,
                                  outage_len=10, recover=False)
    store = EventStore(":memory:")
    _replay(store, well_df, dates, outage_start, 6)
    ev_before = [e for e in store.all_events() if "rate" in e.event_type][0]
    cum_before, dur_before = ev_before.deferred_usd, ev_before.duration_days

    # Re-run the latest day (idx outage_start+5) verbatim.
    asof = dates[outage_start + 5].date().isoformat()
    update_events(store, {"well_X": well_df.iloc[: outage_start + 6]}, as_of=asof)
    ev_after = [e for e in store.all_events() if "rate" in e.event_type][0]
    assert ev_after.deferred_usd == cum_before
    assert ev_after.duration_days == dur_before


def test_cumulative_deferral_is_monotonic_non_decreasing():
    outage_start = 30
    well_df, dates = _outage_well(total_days=45, outage_start_idx=outage_start,
                                  outage_len=10, recover=False)
    store = EventStore(":memory:")
    cums = []
    for asof, live in _replay(store, well_df, dates, outage_start, 10):
        rate_ev = [e for e in live if "rate" in e.event_type]
        if rate_ev:
            cums.append(rate_ev[0].deferred_usd)
    assert cums == sorted(cums), "cumulative deferred $ must never decrease"
    assert cums[-1] > cums[0]


# ---- acknowledge/suppress preserved -----------------------------------------

def test_acknowledged_flag_preserved_on_events():
    """An acknowledged well's event still tracks, but carries the acknowledged flag
    so the brief keeps de-prioritizing it (alarm-fatigue behavior preserved)."""
    # Fresh drop on the latest day so the detector fires (a stale, baseline-absorbed
    # drop wouldn't — that's the very behavior the state machine compensates for).
    bopd = np.full(15, 200.0)
    bopd[-1] = 110.0
    df = _well(bopd, days=15)
    store = EventStore(":memory:")
    live = update_events(store, {"well_x": df}, as_of="2026-05-15",
                         acknowledged=[{"well": "well_x", "category": "*"}])
    assert live, "acknowledged events are still tracked"
    assert all(e.acknowledged for e in live)


# ---- a NEW event is not re-detected as brand-new the next day ---------------

def test_event_not_reopened_as_new_next_day():
    """The day after a NEW detection, the same problem must be ONGOING — a single
    persistent event, not a fresh NEW event each morning."""
    outage_start = 30
    well_df, dates = _outage_well(total_days=45, outage_start_idx=outage_start,
                                  outage_len=10, recover=False)
    store = EventStore(":memory:")
    _replay(store, well_df, dates, outage_start, 2)
    rate_events = [e for e in store.all_events() if "rate" in e.event_type]
    assert len(rate_events) == 1, "must be ONE tracked event, not a new one per day"
    assert rate_events[0].state == ONGOING
    assert rate_events[0].start_date == dates[outage_start].date().isoformat()


# ---- brief surface: the rendered brief shows ongoing events with duration ---

def test_brief_renders_ongoing_event_with_duration_and_cumulative():
    """The deterministic brief must surface a multi-day ONGOING event with its
    running day-count and cumulative deferred bbl/$ (so the still-down well stays
    visible every morning instead of vanishing)."""
    from src.brief_writer import render_brief_markdown

    outage_start = 30
    well_df, dates = _outage_well(total_days=45, outage_start_idx=outage_start,
                                  outage_len=10, recover=False)
    store = EventStore(":memory:")
    live = None
    for _, lv in _replay(store, well_df, dates, outage_start, 5):
        live = lv
    summary = {"total_bopd": 9000, "well_count": 50, "water_cut_pct": 88,
               "avg_runtime_pct": 98.0}
    md = render_brief_markdown(summary, anomalies=[], brief_date="2026-06-04",
                               events=live)
    assert "Ongoing & Resolved Events" in md
    assert "ONGOING" in md
    assert "day 5" in md            # running duration
    ev = [e for e in live if "rate" in e.event_type][0]
    assert f"${ev.deferred_usd:,.0f}" in md  # cumulative deferred $


def test_brief_without_events_is_unchanged():
    """Back-compat: passing no events yields no Ongoing section (byte-identical to
    the pre-state-machine brief)."""
    from src.brief_writer import render_brief_markdown

    summary = {"total_bopd": 9000, "well_count": 50, "water_cut_pct": 88,
               "avg_runtime_pct": 98.0}
    without = render_brief_markdown(summary, anomalies=[], brief_date="2026-06-04")
    none_explicit = render_brief_markdown(summary, anomalies=[], brief_date="2026-06-04",
                                          events=None)
    assert "Ongoing & Resolved Events" not in without
    assert without == none_explicit
