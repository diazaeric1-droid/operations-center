"""Tests for the demo's event-lifecycle surfacing (Task B).

The Streamlit demo replays the fleet's recent history through the persistent event
state machine and renders the *Ongoing & Resolved Events* the morning brief shows.
The committed synthetic fleet injects faults only on the FINAL day, so the demo
offers an "inject a demo outage" toggle that splices a sustained multi-day rate
outage into one healthy well to make the NEW -> ONGOING lifecycle demonstrable.

These tests cover the part unique to the demo: that the injector mutates only an
in-memory copy (committed CSVs untouched) and that replaying its history through
the SAME ``update_events`` path the scheduler/brief use yields a multi-day ONGOING
rate event with a growing cumulative deferral. The state machine itself is covered
exhaustively in ``test_event_store.py``.
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "synthetic" / "fleet"
# The demo app + its vendored modules (theme / fleet_registry) live in demo/.
sys.path.insert(0, str(REPO_ROOT / "demo"))

import app  # noqa: E402  (demo/app.py — importable bare; the nav .run() is a no-op)
from src.data_loader import load_fleet  # noqa: E402
from src.event_store import (  # noqa: E402
    NEW,
    ONGOING,
    EventStore,
    update_events,
)


def _replay(fleet, replay_days=app.REPLAY_DAYS):
    """Drive the event store over the trailing window exactly as the demo does —
    each as-of day feeds every well its history-to-date through update_events."""
    all_dates = sorted({d for df in fleet.values() if df is not None and len(df)
                        for d in df["date"]})
    spine = all_dates[-replay_days:]
    store = EventStore(":memory:")
    live = []
    try:
        for ts in spine:
            as_of = pd.Timestamp(ts).date().isoformat()
            sliced = {wid: df[df["date"] <= ts] for wid, df in fleet.items()}
            sliced = {wid: d for wid, d in sliced.items() if len(d)}
            live = update_events(store, sliced, as_of=as_of)
    finally:
        store.close()
    return live


def test_inject_demo_outage_is_in_memory_only():
    """The injector returns a new fleet with the target well held down — and must
    NOT mutate the original frames (the committed CSVs stay untouched)."""
    fleet = load_fleet(DATA_DIR)
    original_tail = fleet[app.DEMO_OUTAGE_WELL]["bopd"].tail(app.DEMO_OUTAGE_LEN).tolist()

    injected = app._inject_demo_outage(fleet)

    # Original frame is unchanged.
    assert fleet[app.DEMO_OUTAGE_WELL]["bopd"].tail(app.DEMO_OUTAGE_LEN).tolist() == original_tail
    # Injected copy holds the last LEN days at the reduced level (a real drop).
    inj_tail = injected[app.DEMO_OUTAGE_WELL]["bopd"].tail(app.DEMO_OUTAGE_LEN)
    assert (inj_tail < min(original_tail)).all()
    # Other wells are passed through by reference (untouched).
    other = next(w for w in fleet if w != app.DEMO_OUTAGE_WELL)
    assert injected[other] is fleet[other]


def test_inject_demo_outage_noop_when_target_absent():
    """A fleet without the demo well (e.g. a BYOD upload) is returned as-is."""
    fleet = {"ACME_01": load_fleet(DATA_DIR)[app.DEMO_OUTAGE_WELL]}
    assert app.DEMO_OUTAGE_WELL not in fleet  # precondition
    assert app._inject_demo_outage(fleet) is fleet  # unchanged, same object


def test_replay_with_demo_outage_yields_multiday_ongoing_rate_event():
    """The whole point of the toggle: with the demo outage injected, replaying the
    history through the state machine leaves the target well ONGOING past day 1 with
    a positive cumulative deferral — the lifecycle a point-in-time scan can't show."""
    fleet = app._inject_demo_outage(load_fleet(DATA_DIR))
    live = _replay(fleet)

    demo_events = [e for e in live if e.well_id == app.DEMO_OUTAGE_WELL]
    ongoing_rate = [e for e in demo_events
                    if e.state == ONGOING and "rate" in e.event_type and e.duration_days > 1]
    assert ongoing_rate, f"expected a multi-day ONGOING rate event; got {demo_events}"
    ev = ongoing_rate[0]
    # Duration tracks the injected span and cumulative deferral has accrued.
    assert ev.duration_days >= app.DEMO_OUTAGE_LEN - 1
    assert ev.deferred_bopd > 0 and ev.deferred_usd > 0


def test_replay_returns_brief_shaped_event_list():
    """The replay returns the same live-event objects the brief consumes — every
    entry is an Event in a brief-relevant state (NEW / ONGOING / RESOLVED)."""
    from src.event_store import RESOLVED
    live = _replay(load_fleet(DATA_DIR))
    assert live, "the committed fleet should surface at least one live event"
    assert all(e.state in (NEW, ONGOING, RESOLVED) for e in live)
    # Without the demo outage the committed fleet still has natural multi-day ONGOING
    # events (e.g. the steep-decliner decoy the flat-mean rate rule keeps firing).
    assert any(e.state == ONGOING and e.duration_days > 1 for e in live)
