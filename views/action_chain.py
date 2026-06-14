"""Well File · Action Chain — detect → predict → authorize for the selected well.

Ported from pe-pipeline's per-well drill-down: each stage's artifact is shown and
downloadable (alert JSON → diagnosis dict → AFE markdown). Deterministic at every
hop; no API key.
"""
from __future__ import annotations

import json

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
        pt.masthead("ops", "Action Chain", "Select a well in the sidebar.")
        pt.empty_state("No well selected — pick one from the sidebar selector.")
        return

    meta = fleet_registry.get(well_id)
    pt.masthead("ops", "Action Chain",
                f"{well_id} · {meta.name} — one well through the whole machine: "
                "digest alert → ESP diagnosis → decision-ready AFE.")
    pt.context_bar([
        ("Well", f"{well_id} · {meta.name}"),
        ("Deck", c.deck_label()),
        ("Working interest", "1.00 (operator case)"),
        ("Chain", "deterministic · no API key"),
    ])
    theme.data_badge("synthetic", "Modeled daily SCADA fleet with known ground truth "
                                  "— public production is monthly, not daily.")

    import core

    # ---- economic verdict banner (honest framing before the mechanics) ----------
    board = c.board(price, nri)
    brow = board[board["well_id"] == well_id]
    npv = float(brow["est_risked_npv"].iloc[0]) if len(brow) else 0.0
    rec_int = (str(brow["recommended_intervention"].iloc[0]) if len(brow)
               else "no_action")
    top_opp = _top_opportunity(board)
    if rec_int == "no_action":
        st.info(f"**Economic verdict: NO ACTION.** {well_id} is below the action "
                "thresholds — the chain below still runs end-to-end to show the "
                "mechanics, but there is nothing to authorize.")
    elif npv > 0:
        st.success(f"**Economic verdict: AUTHORIZE.** {well_id}'s recommended "
                   f"intervention is value-accretive — risked NPV **${npv:,.0f}** at "
                   "the deck price/NRI. The chain below builds the AFE.")
    else:
        msg = (f"**Economic verdict: MONITOR — do not authorize yet.** At today's "
               f"failure risk and cost, {well_id}'s intervention does NOT clear its "
               f"cost (risked NPV **−${abs(npv):,.0f}**). The chain below shows the "
               "full detect → predict → authorize mechanics, but the call is to watch "
               "and re-rank as the signal strengthens.")
        if top_opp:
            msg += (f" For a value-accretive example, select **{top_opp}** in the "
                    "sidebar.")
        st.warning(msg)

    # ---- Stage 1 · Detect ------------------------------------------------------
    alert = c.alert_for_selected(price)
    flagged = alert.get("category") != "fleet_scan"
    # The ESP alert feed carries no deferred barrels by design; join the digest's
    # rate-loss scan so a flagged well shows its real deferred production.
    scan_deferred = c.deferred_by_well(c.DISK_TOKEN, price).get(str(well_id), 0.0)
    deferred_bopd = max(float(alert.get("deferred_bopd", 0.0) or 0.0), scan_deferred)
    pt.section("1 · Detect — Daily Production Digest",
               "The stage-1 artifact: a WellAlert from the morning scan.")
    if flagged:
        st.markdown(f"**{alert['category']}** · severity **{alert['severity']}** — "
                    f"{alert['headline']}")
        deferred_usd = deferred_bopd * price * nri
        s1a, s1b = st.columns(2)
        s1a.metric("Deferred BOPD", f"{deferred_bopd:,.1f}")
        s1b.metric("Deferred $/day (net)", f"${deferred_usd:,.0f}")
    else:
        st.info("This well was **not** flagged by today's digest — it surfaced via "
                "fleet risk scoring, so a fleet-scan alert is synthesized and the "
                "chain still runs.")
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
    m1.metric("30-Day Failure Risk", f"{diag['esp_risk_score']:.0%}")
    m2.metric("Suspected Mode", mode_short,
              help=mode_full if mode_full != mode_short else None)
    m3.metric("Intervention", diag["intervention"].replace("_", " "))
    if mode_full != mode_short:
        st.caption(f"Full mode read: {mode_full}")
    st.caption(diag["primary_diagnosis"])
    diag_json = json.dumps(diag, indent=2, default=str)
    with st.expander("WellDiagnosis artifact (JSON)"):
        st.code(diag_json, language="json")
    st.download_button("Download diagnosis (JSON)", data=diag_json,
                       file_name=f"diagnosis_{well_id}.json",
                       mime="application/json")

    st.divider()

    # ---- Stage 3 · Authorize ----------------------------------------------------
    pt.section("3 · Authorize — AFE Copilot",
               "The stage-3 artifact: a deterministic Authorization for "
               "Expenditure (cost rollup + PV10 economics + risk register + "
               "approval routing).")
    if npv <= 0 and rec_int != "no_action":
        st.warning("Per the economic verdict above, this AFE is **not** recommended "
                   "for authorization today — risked NPV is non-positive. It is "
                   "generated to show the deterministic AFE the chain WOULD produce "
                   "if the well deteriorated into the money.")
    afe_md = core.render_afe(diag, working_interest=1.0, net_revenue_interest=nri,
                             realized_price=price)
    st.download_button("Download AFE (markdown)", data=afe_md,
                       file_name=f"AFE_{well_id}_{diag['intervention']}.md",
                       mime="text/markdown")
    with st.container(border=True):
        st.markdown(afe_md)
    c.pinned_pv10_caption()
    theme.source_note(
        "Engineering math is deterministic at every hop; economics use the deck "
        "oil price and NRI with working interest 1.0 (operator case) and the AFE "
        "component's PV10 convention. The LLM is optional everywhere and confined to "
        "narration.")
    theme.references(["npv", "shap"])


def _top_opportunity(board) -> str | None:
    """The well_id of the top value-accretive opportunity (positive risked NPV), or
    None when the fleet has none today."""
    if board is None or board.empty:
        return None
    pos = board[board["est_risked_npv"] > 0]
    return str(pos["well_id"].iloc[0]) if len(pos) else None
