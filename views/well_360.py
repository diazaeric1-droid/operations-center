"""Well File · Well 360 — one well, everything the console knows about it.

Registry identity, SCADA trends with alert overlays, event history, the ESP
30-day risk score, and the recommended intervention. Ported from pe-pipeline's
per-well drill-down + the digest's well page.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fleet_registry
import product_theme as pt
import theme

from views import _common as c


def render() -> None:
    c.ensure_state()
    price, nri, _disc = c.deck()
    well_id = st.session_state["well_id"]
    if not well_id:
        pt.masthead("ops", "Well 360", "Select a well in the sidebar.")
        pt.empty_state("No well selected — pick one from the sidebar selector.")
        return

    meta = fleet_registry.get(well_id)
    pt.masthead("ops", "Well 360",
                f"{well_id} · {meta.name} — registry identity, SCADA trends, "
                "events, and 30-day failure risk.")

    import core
    alert = c.alert_for_selected(price)
    flagged = alert.get("category") != "fleet_scan"
    pt.context_bar([
        ("Well", f"{well_id} · {meta.name}"),
        ("Surveillance fleet", c.scada_source_label(c.DISK_TOKEN)),
        ("Deck", c.deck_label()),
        ("Today's digest", "flagged" if flagged else "not flagged (fleet scan)"),
    ])
    theme.data_badge("synthetic", "Modeled daily SCADA fleet with known ground truth "
                                  "— public production is monthly, not daily.")

    mh = st.columns(4)
    mh[0].metric("Lift", meta.lift)
    mh[1].metric("Lateral (ft)", f"{meta.lateral_length_ft:,}")
    mh[2].metric("Basin · Formation", f"{meta.basin} · {meta.formation}")
    mh[3].metric("Peer Group", meta.peer_group)
    theme.well_cross_links("pipeline", well_id)
    if meta.hero:
        st.caption(f"Hero well: {meta.storyline}")

    # ---- risk + intervention (ESP scoring on this well) ----------------------
    pt.section("Failure Risk & Recommended Intervention",
               "The ESP agent scores the well's SCADA signature; the mode "
               "classifier maps it to a priced intervention.")
    diag = c.diagnosis(well_id, price)
    r1, r2, r3 = st.columns(3)
    r1.metric("30-Day Failure Risk", f"{diag['esp_risk_score']:.0%}")
    r2.metric("Suspected Mode", diag["suspected_mode"].split("—")[0].strip() or "—")
    r3.metric("Recommended Intervention", diag["intervention"].replace("_", " "))
    st.caption(diag["primary_diagnosis"])
    board = c.board(price, nri)
    row = board[board["well_id"] == well_id]
    if len(row):
        rr = row.iloc[0]
        st.markdown(
            pt.pill(f"Triage rank #{row.index[0] + 1} of {len(board)}", "info") + " " +
            pt.pill(f"risked NPV ${float(rr['est_risked_npv']):,.0f} "
                    f"({str(rr['npv_basis']).replace('_', ' ')})",
                    "warn" if float(rr["est_risked_npv"]) > 0 else "muted"),
            unsafe_allow_html=True)

    # ---- SCADA trends with alert overlay --------------------------------------
    pt.section("SCADA Trend — Alert Overlay",
               "Production + ESP diagnostics; today's digest alert (if any) is "
               "marked on the timeline.")
    scada = core.well_scada(alert)
    fig = go.Figure()
    for col, color in [("bopd", theme.BLUE), ("intake_pressure_psi", theme.PURPLE),
                       ("motor_temp_f", theme.RED), ("motor_amps", theme.GREEN)]:
        if col in scada.columns:
            fig.add_trace(go.Scatter(x=scada["date"], y=scada[col], name=col,
                                     line=dict(color=color, width=1.6)))
    if flagged and alert.get("date"):
        ts = pd.Timestamp(alert["date"])
        fig.add_vline(x=ts, line_color=theme.RED, line_dash="dash", line_width=1.5)
        fig.add_annotation(x=ts, y=1.0, yref="paper", showarrow=False,
                           text=f"alert · {alert['category']}",
                           font=dict(color=theme.RED, size=11), yanchor="bottom")
    st.plotly_chart(theme.style_fig(fig, height=360), width="stretch")
    theme.source_note(
        "Daily SCADA channels (BOPD, intake psi, motor °F, amps) from the digest "
        "fleet; the dashed marker is the date of today's digest alert for this well.")
    if flagged:
        st.markdown(f"**{alert['category']}** · severity **{alert['severity']}** — "
                    f"{alert['headline']}")
        deferred_usd = float(alert.get("deferred_bopd", 0.0)) * price * nri
        if deferred_usd > 0:
            st.metric("Deferred $/day (net)", f"${deferred_usd:,.0f}",
                      delta=f"{alert.get('deferred_bopd', 0):,.0f} bopd",
                      delta_color="inverse")
    else:
        st.caption("Not flagged by today's digest — this page scores it via the "
                   "fleet scan anyway.")

    # ---- event history ---------------------------------------------------------
    pt.section("Event History",
               "This well's events from the state-machine replay of the recent "
               "history (no demo injection here — raw committed fleet).")
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
