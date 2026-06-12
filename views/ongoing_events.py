"""Today · Ongoing Events — the persistent event state machine (NEW → ONGOING →
RESOLVED), ported from daily-production-digest v0.6.3's events panel.

The Anomalies view is a point-in-time scan of the latest day; this page adds
memory: the fleet's recent history is replayed through the same EventStore the
morning brief and scheduler drive, so a confirmed multi-day outage stays ONGOING
with a running duration and cumulative deferred bbl/$.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import product_theme as pt
import theme

from views import _common as c


def render() -> None:
    c.ensure_state()
    price, _nri, _disc = c.deck()

    pt.masthead("ops", "Ongoing Events",
                "Event lifecycle with memory: confirmed outages stay ONGOING with "
                "running duration and cumulative deferred bbl/$.")

    token = c.scada_token()
    fleet = c.fleet_for_token(token)
    is_upload = token != c.DISK_TOKEN
    pt.context_bar([
        ("Surveillance fleet", c.scada_source_label(token)),
        ("As of", c.fleet_as_of(fleet)),
        ("Deck", c.deck_label()),
        ("Replay", f"trailing {_replay_days()} as-of days, in-memory store"),
    ])

    import core
    inject = False
    if not is_upload:
        inject = st.toggle(
            "Inject a demo outage (multi-day ONGOING rate event)", value=True,
            key="inject_demo_outage",
            help=f"Synthetic only: holds the last {core.DEMO_OUTAGE_LEN} days of "
                 f"{core.DEMO_OUTAGE_WELL} at ~{core.DEMO_OUTAGE_FRACTION:.0%} of its "
                 "pre-event baseline so the NEW→ONGOING lifecycle is demonstrable. "
                 "The committed fleet injects faults only on the final day; this "
                 "mutates an in-memory copy — the CSVs are untouched.")
        if inject:
            st.caption(f"Demo outage active on **{core.DEMO_OUTAGE_WELL}** — a "
                       f"sustained ~{1 - core.DEMO_OUTAGE_FRACTION:.0%} rate loss held "
                       f"over the last {core.DEMO_OUTAGE_LEN} days (no recovery), so "
                       "it reads ONGOING with a growing cumulative deferral.")
    else:
        st.caption("Replaying your uploaded fleet's history — any multi-day outage "
                   "in your data surfaces here as an ONGOING event.")

    events = c.replay_events(token, price, inject)
    NEW, ONGOING, RESOLVED = (core.digest_events.NEW, core.digest_events.ONGOING,
                              core.digest_events.RESOLVED)
    open_evts = [e for e in events if e.state in (NEW, ONGOING)]
    resolved = [e for e in events if e.state == RESOLVED]
    multi_day = [e for e in open_evts if e.duration_days > 1]

    pt.kpi_row([
        {"label": "Open Events (NEW/ONGOING)", "value": f"{len(open_evts)}"},
        {"label": "Multi-Day ONGOING", "value": f"{len(multi_day)}",
         "help": "Open events past their first day — the lifecycle a point-in-time "
                 "scan can't keep visible."},
        {"label": "Cumulative Deferred (open)",
         "value": f"${sum(e.deferred_usd for e in open_evts):,.0f}",
         "help": "Sum of cumulative deferred $ across open events over their life."},
    ])

    pt.section("Open Events")
    if open_evts:
        rows = []
        for e in sorted(open_evts, key=lambda e: (-e.deferred_usd, e.well_id)):
            rows.append({
                "Well": e.well_id,
                "Event Type": e.event_type,
                "State": e.state,
                "Start Date": e.start_date,
                "Duration (days)": e.duration_days,
                "Cumulative Deferred bbl": round(e.deferred_bopd, 0),
                "Cumulative Deferred $": round(e.deferred_usd, 0),
                "Today's Deferral $": round(e.last_deferred_usd, 0),
                "Ack": "suppressed" if e.acknowledged else "",
            })
        ev_df = pd.DataFrame(rows)
        st.dataframe(
            ev_df, width="stretch", hide_index=True,
            column_config={
                "Cumulative Deferred $": st.column_config.NumberColumn(format="$%d"),
                "Cumulative Deferred bbl": st.column_config.NumberColumn(format="%d"),
                "Today's Deferral $": st.column_config.NumberColumn(format="$%d"),
                "Duration (days)": st.column_config.NumberColumn(format="%d d"),
            })
        st.download_button("Download open events (CSV)",
                           data=ev_df.to_csv(index=False),
                           file_name="ops_open_events.csv", mime="text/csv")
    else:
        pt.empty_state("No open (NEW/ONGOING) events on the replayed history.")

    if resolved:
        pt.section("Recently Resolved (closing out)")
        for e in resolved:
            span = f"{e.duration_days}-day" if e.duration_days > 1 else "1-day"
            cum = (f" — ~{e.deferred_bopd:,.0f} bbl (${e.deferred_usd:,.0f}) deferred "
                   "over the event" if e.deferred_bopd > 0 else "")
            st.markdown(f"- **{e.well_id}** ({e.event_type}) — {span} event "
                        f"RESOLVED{cum}.")

    theme.source_note(
        "Events are replayed through the digest's persistent state machine "
        "(EventStore, in-memory here — nothing persisted): a rate event opened from "
        "a confirmed drop stays ONGOING while production holds below its pre-event "
        "baseline (accruing cumulative deferred bbl/$ at the deck oil price) and "
        "RESOLVES on recovery into band — the same lifecycle the morning brief "
        "reports.")
    _backtest_caption()
    theme.references(["arps", "deferment"])


def _replay_days() -> int:
    import core
    return core.REPLAY_DAYS


def _backtest_caption() -> None:
    """Honest-eval framing: the component's committed event-lifecycle backtest."""
    import core
    metrics = core.APP_DIRS["digest"] / "data" / "backtest_v2_metrics.json"
    try:
        m = json.loads(metrics.read_text())
        st.caption(
            f"Component backtest (committed snapshot, seeded outages + decoy wells): "
            f"event precision {m['event_precision']:.2f} · recall "
            f"{m['event_recall']:.2f} · duration MAE {m['duration_mae_days']:.1f} d "
            f"· detection latency {m['mean_latency_days']:.1f} d. One decoy "
            "(re-calibration step) still opens a spurious event — that's the honest "
            "miss, not hidden.")
    except Exception:  # noqa: BLE001 — snapshot absent: skip, never invent numbers
        pass
