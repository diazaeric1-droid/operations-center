"""Well File · Well 360 — one well, everything the console knows about it.

Registry identity, SCADA trends with alert overlays, event history, the ESP
30-day risk score, and the recommended intervention. Ported from pe-pipeline's
per-well drill-down + the digest's well page.
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
    pt.masthead("ops", "Well 360",
                f"{well_id} · {meta.name} — registry identity, SCADA trends, "
                "events, and 30-day failure risk.")

    # ---- drill-down bar: pick any well + jump to its Action Chain ---------------
    if st.session_state.get("w360_pick") != well_id and well_id in ids:
        st.session_state["w360_pick"] = well_id
    dd = st.columns([3, 2, 2])
    with dd[0]:
        st.selectbox("Drill into well", ids, key="w360_pick", on_change=_sync_well,
                     help="Jump to any well in the surveillance fleet — keeps the "
                          "sidebar selection in lockstep.")
    with dd[1]:
        st.metric("API-14", meta.api14)
    with dd[2]:
        import views
        ac = views.PAGE_OBJECTS.get("Action Chain")
        st.caption("Next step")
        if ac is not None:
            try:
                st.page_link(ac, label="→ Run the Action Chain", icon=":material/account_tree:")
            except Exception:  # noqa: BLE001 — outside navigation (AppTest)
                pass

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
    mh2 = st.columns(4)
    mh2[0].metric("Area", meta.area)
    mh2[1].metric("First Production", meta.first_prod)
    mh2[2].metric("Operator Case", "WI 1.00")
    mh2[3].metric("Digest Status", "Flagged" if flagged else "Fleet scan")
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
        npv = float(rr["est_risked_npv"])
        basis = str(rr["npv_basis"]).replace("_", " ")
        if npv > 0:
            npv_pill = pt.pill(f"opportunity · risked NPV ${npv:,.0f} ({basis})", "ok")
        elif rr["recommended_intervention"] == "no_action":
            npv_pill = pt.pill("no action — below thresholds", "muted")
        else:
            npv_pill = pt.pill(f"watch · risked NPV −${abs(npv):,.0f} — intervening "
                               f"now loses value ({basis})", "warn")
        st.markdown(
            pt.pill(f"Triage rank #{row.index[0] + 1} of {len(board)}", "info")
            + " " + npv_pill, unsafe_allow_html=True)

    # ---- SCADA trends with alert overlay --------------------------------------
    pt.section("SCADA Trend — Alert Overlay",
               "Each channel on its own axis (so the ESP leading indicators aren't "
               "flattened by the temperature scale); today's digest alert is marked.")
    scada = core.well_scada(alert)
    channels = [("bopd", "Oil rate (BOPD)", theme.BLUE),
                ("intake_pressure_psi", "Intake pressure (psi)", theme.PURPLE),
                ("motor_temp_f", "Motor temp (°F)", theme.RED),
                ("motor_amps", "Motor amps (A)", theme.GREEN)]
    present = [(col, lbl, clr) for col, lbl, clr in channels if col in scada.columns]
    if present:
        fig = make_subplots(rows=len(present), cols=1, shared_xaxes=True,
                            vertical_spacing=0.045,
                            subplot_titles=[lbl for _, lbl, _ in present])
        for i, (col, lbl, clr) in enumerate(present, start=1):
            fig.add_trace(go.Scatter(x=scada["date"], y=scada[col], name=lbl,
                                     line=dict(color=clr, width=1.6)), row=i, col=1)
        if flagged and alert.get("date"):
            ts = pd.Timestamp(alert["date"])
            for i in range(1, len(present) + 1):
                fig.add_vline(x=ts, line_color=theme.RED, line_dash="dash",
                              line_width=1.2, row=i, col=1)
        fig.update_layout(showlegend=False)
        for ann in fig["layout"]["annotations"]:  # left-align subplot titles, smaller
            ann["x"], ann["xanchor"], ann["font"] = 0.0, "left", dict(size=12)
        st.plotly_chart(theme.style_fig(fig, height=110 * len(present) + 60,
                                        legend=False), width="stretch")
    theme.source_note(
        "Daily SCADA channels from the digest fleet, each on its own auto-scaled "
        "axis — a rising intake-pressure or amp trend stays legible instead of being "
        "compressed under motor temperature. The dashed marker is today's digest "
        "alert date for this well.")
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
