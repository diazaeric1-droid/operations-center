"""Well File · Well 360 — one well, everything the console knows about it.

A conversation-starting one-pager for a thorough well review: identity + how long
it's been online + work history, a status verdict, lift-aware production and
diagnostic trends (with today's alert marked), the ESP 30-day risk and recommended
intervention, and the event history.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fleet_registry
import product_theme as pt
import theme

from views import _common as c

# Lift-specific diagnostic channels shown under the production streams — the SINGLE
# shared definition (Surveillance uses the same one) so the two pages can't diverge.
_LIFT_DIAG = c.LIFT_CHANNELS


def _sync_well() -> None:
    st.session_state["well_id"] = st.session_state["w360_pick"]


def render() -> None:
    c.ensure_state()
    price, nri, _disc = c.deck()
    well_id = st.session_state["well_id"]
    ids = c.scada_well_ids()
    if not well_id and not ids:
        pt.masthead("ops", "Well 360", "Select a well.")
        pt.empty_state("No fleet loaded — run bootstrap (first app start).")
        return

    meta = fleet_registry.get(well_id)
    hist = fleet_registry.well_history(well_id)
    pt.masthead("ops", "Well 360",
                f"{well_id} · {meta.name} — {meta.lift} · {meta.basin} "
                f"{meta.formation} · online {hist['years_online']} yr. A one-page "
                "well review.")

    # ---- drill-down bar: pick any well + jump to its Action Chain ---------------
    if st.session_state.get("w360_pick") != well_id and well_id in ids:
        st.session_state["w360_pick"] = well_id
    dd = st.columns([3, 2, 2])
    with dd[0]:
        st.selectbox("Drill into well", ids, key="w360_pick", on_change=_sync_well,
                     help="Jump to any well — keeps the sidebar selection in lockstep.")
    with dd[1]:
        st.metric("API-14", meta.api14)
    with dd[2]:
        import views
        ac = views.PAGE_OBJECTS.get("Action Chain")
        st.caption("Next step")
        if ac is not None:
            try:
                st.page_link(ac, label="→ Run the Action Chain",
                             icon=":material/account_tree:")
            except Exception:  # noqa: BLE001
                pass

    import core
    alert = c.alert_for_selected(price)
    flagged = alert.get("category") != "fleet_scan"
    pt.context_bar([
        ("Well", f"{well_id} · {meta.name}"),
        ("Online since", f"{hist['online_since']} ({hist['years_online']} yr)"),
        ("Deck", c.deck_label()),
        ("Today's digest", "flagged" if flagged else "not flagged (fleet scan)"),
    ])

    # ---- status verdict (from the triage tiers) --------------------------------
    board = c.board_with_deferred(price, nri)
    brow = board[board["well_id"] == well_id]
    _status_banner(brow, hist, price, nri)

    # ---- identity --------------------------------------------------------------
    mh = st.columns(4)
    mh[0].metric("Lift", meta.lift)
    mh[1].metric("Lateral (ft)", f"{meta.lateral_length_ft:,}")
    mh[2].metric("Basin · Formation", f"{meta.basin} · {meta.formation}")
    mh[3].metric("Peer Group", meta.peer_group)
    mh2 = st.columns(4)
    mh2[0].metric("Area", meta.area)
    mh2[1].metric("Online Since", hist["online_since"],
                  f"{hist['years_online']} yr", delta_color="off")
    mh2[2].metric("Prior Interventions", f"{hist['n_workovers']}",
                  f"last {hist['last_worked']}" if hist["last_worked"] else "none on record",
                  delta_color="off")
    mh2[3].metric("Digest Status", "Flagged" if flagged else "Fleet scan")
    theme.well_cross_links("pipeline", well_id)
    if meta.hero:
        st.caption(f"Hero well: {meta.storyline}")

    # ---- production + lift-aware diagnostics -----------------------------------
    pt.section("Production & Diagnostics — Trend",
               "Oil / water / gas plus the diagnostics that matter for this lift "
               "type; today's digest alert (if any) is marked on the timeline.")
    _trend_chart(core.well_scada(alert), meta, alert if flagged else None)
    theme.source_note(
        f"Daily channels for a {meta.lift} well. Water = gross fluid − oil. The "
        "dashed marker is the date of today's digest alert for this well. The chart "
        f"shows the most recent SCADA window, not the full history since "
        f"{hist['online_since']} ({hist['years_online']} yr online).")
    if flagged:
        st.markdown(f"**{alert['category']}** · severity **{alert['severity']}** — "
                    f"{alert['headline']}")

    # ---- risk + recommended intervention ---------------------------------------
    pt.section("Failure Risk & Recommended Intervention",
               "The ESP agent scores the well's SCADA signature; the mode classifier "
               "maps it to a priced intervention.")
    diag = c.diagnosis(well_id, price)
    r1, r2, r3 = st.columns(3)
    r1.metric("30-Day Failure Signal", f"{diag['esp_risk_score']:.0%}",
              help="A fleet-relative ESP ranking on this synthetic fleet, not a "
                   "calibrated absolute probability.")
    r2.metric("Suspected Mode", diag["suspected_mode"].split("—")[0].strip() or "—",
              help=diag["suspected_mode"])
    r3.metric("Recommended Intervention", diag["intervention"].replace("_", " "))
    st.caption(diag["primary_diagnosis"])

    # ---- economic limit & remaining producing life -----------------------------
    pt.section("Economic Limit & Remaining Life",
               "The rate at which net revenue equals lease operating expense — the "
               "rate you'd plug & abandon at — and how long this well's own decline "
               "leaves before it gets there.")
    el = core.economic_limit(core.well_scada(alert), realized_price=price,
                             net_revenue_interest=nri)
    if el is None:
        st.caption("Not enough producing history to estimate an economic limit.")
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric("Economic-limit rate", f"{el['q_limit_bopd']:,.0f} BOPD",
                  help="Net revenue = lease operating expense at this rate.")
        months = el["months_remaining"]
        e2.metric("Remaining life (est.)",
                  "—" if months == float("inf") else f"{months / 12:,.1f} yr",
                  f"{el['q_now_bopd']:,.0f} BOPD now", delta_color="off")
        e3.metric("Net margin", f"${el['net_margin_per_bbl']:,.0f}/bbl",
                  help="Realized price × NRI − variable opex.")
        st.caption(f"Assumes ${el['loe_per_month']:,.0f}/well-month fixed LOE and the "
                   f"well's fitted decline ({el['annual_decline_pct']:,.0f}%/yr). "
                   "Illustrative carrying cost — set per asset in a real deployment.")

    # ---- well work history ------------------------------------------------------
    pt.section("Well History — Interventions",
               "How many times this well has been worked, and what was done — the "
               "context for whether another job is warranted.")
    if hist["records"]:
        wh = pd.DataFrame(hist["records"])
        wh["cost_usd"] = wh["cost_usd"].map(lambda v: f"${v:,.0f}")
        wh["uplift_bopd"] = wh["uplift_bopd"].map(lambda v: f"+{v:,.0f}")
        wh = wh[["date", "type", "cost_usd", "uplift_bopd", "result"]]
        wh.columns = ["Date", "Intervention", "Cost", "Uplift (bopd)", "Result"]
        st.dataframe(wh, width="stretch", hide_index=True)
        spend = sum(r["cost_usd"] for r in hist["records"])
        st.caption(f"{hist['n_workovers']} prior intervention(s) since "
                   f"{hist['online_since']} · ~${spend:,.0f} lifetime workover spend. "
                   "Illustrative synthetic history (deterministic per well).")
    else:
        st.caption("No prior interventions on record — this well has run since "
                   f"{hist['online_since']} without a workover.")

    # ---- event history ----------------------------------------------------------
    pt.section("Event History",
               "This well's events from the state-machine replay of the recent "
               "history (raw committed fleet, no demo injection).")
    events = [e for e in c.replay_events(c.DISK_TOKEN, price, False)
              if e.well_id == well_id]
    if events:
        ev_df = pd.DataFrame([{
            "Event Type": e.event_type, "State": e.state, "Start Date": e.start_date,
            "Duration (days)": e.duration_days,
            "Cumulative Deferred bbl": round(e.deferred_bopd, 0),
            "Cumulative Deferred $": round(e.deferred_usd, 0),
        } for e in events])
        st.dataframe(ev_df, width="stretch", hide_index=True)
    else:
        st.caption("No open or recently-resolved events for this well on the "
                   "replayed window.")

    st.caption("Next step: run the full detect → predict → authorize flow for this "
               "well on the **Action Chain** page.")
    theme.references(["arps", "shap", "npv"])


def _status_banner(brow, hist, price: float, nri: float) -> None:
    """A one-line verdict: opportunity / watch / stable, with the economics."""
    if not len(brow):
        return
    r = brow.iloc[0]
    npv = float(r["est_risked_npv"])
    deferred = float(r["deferred_usd_per_day"])
    intervention = str(r["recommended_intervention"]).replace("_", " ")
    if r["recommended_intervention"] == "no_action":
        pill = pt.pill("STABLE — no action", "ok")
        msg = "Producing on trend with no deferment; nothing to do today."
    elif npv > 0:
        pill = pt.pill("OPPORTUNITY — value-accretive", "ok")
        msg = (f"Recommended {intervention} clears its cost — risked NPV "
               f"**${npv:,.0f}**. Authorize on the Action Chain.")
    elif deferred > 0:
        pill = pt.pill("AT-RISK WATCH — monitor", "warn")
        msg = (f"Losing **${deferred:,.0f}/day** (net), but {intervention} doesn't "
               f"clear its cost yet (risked NPV −${abs(npv):,.0f}). Monitor and "
               "re-rank as the signal strengthens.")
    else:
        pill = pt.pill("STABLE — no action", "muted")
        msg = "No deferment and no value-accretive intervention today."
    chips = (pill + " "
             + pt.pill(f"deferred ${deferred:,.0f}/day", "warn" if deferred else "muted")
             + " " + pt.pill(f"{hist['n_workovers']} prior jobs", "info"))
    st.markdown(chips, unsafe_allow_html=True)
    st.caption(msg)


def _trend_chart(scada, meta, alert) -> None:
    """Oil/water/gas streams + lift-aware diagnostics on stacked auto-scaled axes."""
    d = scada.copy()
    if "bfpd" in d.columns and "bopd" in d.columns:
        d["water"] = (d["bfpd"] - d["bopd"]).clip(lower=0)
    streams = [("bopd", "Oil (BOPD)", theme.GREEN), ("water", "Water (BWPD)", theme.BLUE),
               ("gas_mcfd", "Gas (MCF/d)", theme.AMBER)]
    rows = [(col, lbl, clr) for col, lbl, clr in streams if col in d.columns]
    rows += [(col, lbl, clr) for col, lbl, clr in _LIFT_DIAG.get(meta.lift, [])
             if col in d.columns]
    if not rows:
        pt.empty_state("No SCADA channels for this well.")
        return
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=[r[1] for r in rows])
    for i, (col, _lbl, clr) in enumerate(rows, start=1):
        fig.add_trace(go.Scatter(x=d["date"], y=d[col], line=dict(color=clr, width=1.4)),
                      row=i, col=1)
        # Overlay the well's own expected exponential decline on the oil panel — the
        # "on the type curve, or deferring?" read (the gap below = implied deferment).
        if col == "bopd":
            import core
            fit = core.fit_well_decline(d)
            if fit is not None:
                fig.add_trace(go.Scatter(
                    x=fit["dates"], y=fit["expected"], name="Expected decline",
                    line=dict(color=theme.RED, width=1.2, dash="dash")), row=i, col=1)
    if alert is not None and alert.get("date"):
        ts = pd.Timestamp(alert["date"])
        for i in range(1, len(rows) + 1):
            fig.add_vline(x=ts, line_color=theme.RED, line_dash="dash",
                          line_width=1.1, row=i, col=1)
    fig.update_layout(showlegend=False)
    for ann in fig["layout"]["annotations"]:
        ann["x"], ann["xanchor"], ann["font"] = 0.0, "left", dict(size=11)
    st.plotly_chart(theme.style_fig(fig, height=140 * len(rows) + 40, legend=False),
                    width="stretch")
