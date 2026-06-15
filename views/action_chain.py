"""Well File · Action Chain — detect → predict → authorize for any well.

Pick (or jump to a flagged) well and the console runs the whole machine end to end:
the digest alert → the ESP diagnosis → a decision-ready AFE with cost, risked
economics, and approval routing. Deterministic at every hop; no API key.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import fleet_registry
import product_theme as pt
import theme

from views import _common as c


def _sync_ac() -> None:
    st.session_state["well_id"] = st.session_state["ac_pick"]


def _sync_ac_flagged() -> None:
    v = st.session_state.get("ac_flagged")
    if v and v != "—":
        st.session_state["well_id"] = v


def _approver(cost: float) -> str:
    if cost < 100_000:
        return "Field Superintendent"
    if cost < 500_000:
        return "Asset Manager"
    return "Asset VP / Partner AFE"


def render() -> None:
    c.ensure_state()
    price, nri, _disc = c.deck()
    ids = c.scada_well_ids()
    well_id = st.session_state["well_id"] or (ids[0] if ids else None)
    if not well_id:
        pt.masthead("ops", "Action Chain", "Select a well.")
        pt.empty_state("No fleet loaded — run bootstrap (first app start).")
        return

    meta = fleet_registry.get(well_id)
    pt.masthead("ops", "Action Chain",
                f"{well_id} · {meta.name} — one well through the whole machine: "
                "digest alert → ESP diagnosis → decision-ready AFE.")

    import core

    # ---- well picker + flagged-today quick-pick --------------------------------
    anoms = c.scan(c.DISK_TOKEN, price)
    active = [a for a in anoms if not a.acknowledged]
    flagged_ids = [a.well_id for a in active]
    dd = st.columns([3, 3])
    with dd[0]:
        if st.session_state.get("ac_pick") != well_id and well_id in ids:
            st.session_state["ac_pick"] = well_id
        st.selectbox("Build an AFE for well", ids, key="ac_pick", on_change=_sync_ac,
                     help="Any well in the fleet — the chain runs end to end and "
                          "produces a decision-ready AFE.")
    with dd[1]:
        opts = ["—"] + flagged_ids
        st.selectbox(f"Jump to a flagged well ({len(flagged_ids)} today)", opts,
                     key="ac_flagged", on_change=_sync_ac_flagged,
                     help="Wells the digest flagged on the latest scan.")

    pt.context_bar([
        ("Well", f"{well_id} · {meta.name}"),
        ("Deck", c.deck_label()),
        ("Working interest", "1.00 (operator case)"),
        ("Chain", "deterministic · no API key"),
    ])
    theme.data_badge("synthetic", "Modeled daily SCADA fleet with known ground truth "
                                  "— public production is monthly, not daily.")

    # ---- flagged-today panel (so the Detect stage is never an empty 'none') -----
    with st.expander(f"Flagged by today's digest — {len(active)} wells "
                     "(pick one above to build its AFE)", expanded=False):
        if active:
            ft = pd.DataFrame([
                {"Well": a.well_id, "Severity": a.severity, "Category": a.category,
                 "Deferred $/day (net)": (f"${float(a.deferred_bopd) * price * nri:,.0f}"
                                          if a.deferred_bopd else "—"),
                 "Headline": a.headline}
                for a in sorted(active, key=lambda a: -float(a.deferred_bopd or 0))])
            st.dataframe(ft, width="stretch", hide_index=True)
        else:
            st.caption("No active anomalies on the latest scan.")

    # ---- economic verdict ------------------------------------------------------
    board = c.board(price, nri)
    brow = board[board["well_id"] == well_id]
    npv = float(brow["est_risked_npv"].iloc[0]) if len(brow) else 0.0
    rec_int = (str(brow["recommended_intervention"].iloc[0]) if len(brow)
               else "no_action")
    top_opp = _top_opportunity(board)
    if rec_int == "no_action":
        st.info(f"**Economic verdict: NO ACTION.** {well_id} is below the action "
                "thresholds — the chain still runs to show the mechanics, but there "
                "is nothing to authorize.")
    elif npv > 0:
        st.success(f"**Economic verdict: AUTHORIZE.** {well_id}'s recommended "
                   f"intervention is value-accretive — risked NPV **${npv:,.0f}**.")
    else:
        msg = (f"**Economic verdict: MONITOR — do not authorize yet.** At today's "
               f"failure risk and cost, {well_id}'s intervention does NOT clear its "
               f"cost (risked NPV **−${abs(npv):,.0f}**).")
        if top_opp:
            msg += f" For a value-accretive example, jump to **{top_opp}** above."
        st.warning(msg)

    # ---- Stage 1 · Detect ------------------------------------------------------
    alert = c.alert_for_selected(price)
    flagged = alert.get("category") != "fleet_scan"
    scan_deferred = c.deferred_by_well(c.DISK_TOKEN, price).get(str(well_id), 0.0)
    deferred_bopd = max(float(alert.get("deferred_bopd", 0.0) or 0.0), scan_deferred)
    pt.section("1 · Detect — Daily Production Digest",
               "The stage-1 artifact: a WellAlert from the morning scan.")
    if flagged:
        st.markdown(f"**{alert['category']}** · severity **{alert['severity']}** — "
                    f"{alert['headline']}")
    else:
        st.info("This well was **not** flagged by today's digest — it surfaced via "
                "fleet risk scoring, so a fleet-scan alert is synthesized and the "
                "chain still runs. (Use *Jump to a flagged well* above to start from "
                "a digest alert instead.)")
    s1 = st.columns(3)
    s1[0].metric("Deferred BOPD", f"{deferred_bopd:,.1f}")
    s1[1].metric("Deferred $/day (net)", f"${deferred_bopd * price * nri:,.0f}")
    s1[2].metric("Baseline BOPD", f"{float(alert.get('baseline_bopd', 0.0)):,.0f}"
                 if alert.get("baseline_bopd") else "—")
    alert_json = json.dumps(alert, indent=2, default=str)
    with st.expander("WellAlert artifact (JSON)"):
        st.code(alert_json, language="json")
    st.download_button("Download alert (JSON)", data=alert_json,
                       file_name=f"alert_{well_id}.json", mime="application/json")

    st.divider()

    # ---- Stage 2 · Predict -----------------------------------------------------
    pt.section("2 · Predict — ESP Failure-Risk Agent",
               "The stage-2 artifact: an AFE-ready WellDiagnosis (risk score + "
               "failure mode + priced intervention).")
    diag = c.diagnosis(well_id, price)
    mode_full = str(diag["suspected_mode"]).strip()
    mode_short = mode_full.split("—")[0].strip() or "—"
    m1, m2, m3 = st.columns(3)
    m1.metric("30-Day Failure Signal", f"{diag['esp_risk_score']:.0%}",
              help="Platt-calibrated probability from the ESP model trained on this "
                   "fleet's labeled faults (out-of-fold AUROC ≈0.99; model card on "
                   "Methods & Limitations).")
    m2.metric("Suspected Mode", mode_short, help=mode_full)
    m3.metric("Intervention", diag["intervention"].replace("_", " "))
    st.caption(diag["primary_diagnosis"])
    diag_json = json.dumps(diag, indent=2, default=str)
    with st.expander("WellDiagnosis artifact (JSON)"):
        st.code(diag_json, language="json")
    st.download_button("Download diagnosis (JSON)", data=diag_json,
                       file_name=f"diagnosis_{well_id}.json", mime="application/json")

    st.divider()

    # ---- Stage 3 · Authorize ----------------------------------------------------
    pt.section("3 · Authorize — AFE Copilot",
               "The stage-3 artifact: a deterministic Authorization for Expenditure "
               "(cost rollup + PV10 economics + risk register + approval routing).")
    cost = c.intervention_cost(diag["intervention"])
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Intervention", diag["intervention"].replace("_", " "))
    a2.metric("Est. AFE Cost", "—" if pd.isna(cost) else f"${cost:,.0f}")
    a3.metric("Risked NPV (risk-weighted)",
              f"${npv:,.0f}" if npv >= 0 else f"−${abs(npv):,.0f}",
              help="The board's risk-weighted NPV: risk × PV(net revenue) − cost. "
                   "This is BELOW the AFE's deterministic Net NPV below (which is not "
                   "risk-weighted) — that gap is the failure-risk discount, not a "
                   "discrepancy.")
    a4.metric("Routes To", "—" if pd.isna(cost) else _approver(cost),
              help="Authority-limit approval routing by AFE size.")
    if npv <= 0 and rec_int != "no_action":
        st.warning("Per the verdict above, this AFE is generated to show what the "
                   "chain WOULD authorize — it is **not** recommended for approval "
                   "today (risked NPV is non-positive).")
    afe_md = core.render_afe(diag, working_interest=1.0, net_revenue_interest=nri,
                             realized_price=price)
    st.download_button("⬇ Download decision-ready AFE (markdown)", data=afe_md,
                       file_name=f"AFE_{well_id}_{diag['intervention']}.md",
                       mime="text/markdown", type="primary")
    with st.expander("Full AFE document", expanded=npv > 0):
        st.markdown(afe_md)
    st.caption("The AFE's **Net NPV** is deterministic (not risk-weighted); the "
               "**Risked NPV** metric above multiplies the upside by the 30-day "
               "failure signal and nets the certain cost, so Risked NPV ≤ Net NPV "
               "on the same well.")
    c.pinned_pv10_caption()

    # ---- Monte-Carlo economics: the distributional view a capital review expects ---
    pt.section("Monte-Carlo Economics — P10 / P50 / P90",
               "A single-point NPV reads junior at sign-off. This runs 10,000 trials "
               "over the three biggest uncertainties (incremental rate, uplift decline, "
               "realized price) to show the band and the chance the job pays out.")
    mc = core.afe_monte_carlo(diag, realized_price=price, net_revenue_interest=nri)
    if mc is None:
        st.caption("Monte-Carlo economics need a priced intervention — not available "
                   "for this well's recommendation.")
    else:
        mcs = st.columns(4)
        mcs[0].metric("P10 (downside)", f"${mc['p10']:,.0f}")
        mcs[1].metric("P50 (median)", f"${mc['p50']:,.0f}")
        mcs[2].metric("P90 (upside)", f"${mc['p90']:,.0f}")
        mcs[3].metric("P(payout < 24 mo)", f"{mc['prob_payout']:.0%}")
        _tornado_chart(mc)
        theme.source_note(
            f"{mc['n_trials']:,} trials, net-to-operator at the deck price/NRI (PV10). "
            "The P50 reconciles with the AFE's deterministic Net NPV above; the tornado "
            "shows each variable's NPV swing when moved to its P10/P90 with the others "
            "held at base — where the risk to this AFE actually lives.")

    theme.source_note(
        "Engineering math is deterministic at every hop; economics use the deck oil "
        "price and NRI with working interest 1.0 (operator case) and the AFE "
        "component's PV10 convention. The LLM is optional and confined to narration.")
    theme.references(["npv", "shap"])


_TORNADO_LABELS = {
    "incremental_rate_bopd": "Incremental rate (±30%)",
    "uplift_decline_per_yr": "Uplift decline (±0.15/yr)",
    "realized_price_per_bbl": "Realized price (±1σ)",
}


def _tornado_chart(mc: dict) -> None:
    """Horizontal tornado: each variable's NPV span (low→high) around the base NPV."""
    import plotly.graph_objects as go

    base = mc["base"]
    items = sorted(mc["tornado"].items(), key=lambda kv: kv[1]["swing"])
    fig = go.Figure()
    for name, t in items:
        lo, hi = sorted((t["low"], t["high"]))
        fig.add_trace(go.Bar(
            y=[_TORNADO_LABELS.get(name, name)], x=[hi - lo], base=lo,
            orientation="h", marker_color=theme.BLUE,
            hovertemplate=f"{_TORNADO_LABELS.get(name, name)}<br>"
                          f"P10 NPV: ${lo:,.0f}<br>P90 NPV: ${hi:,.0f}"
                          f"<br>swing: ${t['swing']:,.0f}<extra></extra>",
            showlegend=False))
    fig.add_vline(x=base, line_color=theme.NAVY, line_dash="dash", line_width=1.2)
    fig.update_layout(xaxis_title="Net NPV ($, net to operator) — dashed line = base case",
                      yaxis_title="", bargap=0.45)
    st.plotly_chart(theme.style_fig(fig, height=70 * len(items) + 110, legend=False),
                    width="stretch")


def _top_opportunity(board) -> str | None:
    """The well_id of the top value-accretive opportunity (positive risked NPV)."""
    if board is None or board.empty:
        return None
    pos = board[board["est_risked_npv"] > 0]
    return str(pos["well_id"].iloc[0]) if len(pos) else None
