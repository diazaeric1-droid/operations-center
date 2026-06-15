"""Today · Triage Board — the whole fleet ranked by risked-NPV opportunity.

Ported from pe-pipeline's Fleet Triage overview (same ranking engine — the
product tests pin numeric equality against pipeline_core.rank_fleet).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fleet_registry as fr
import product_theme as pt
import theme

from views import _common as c


def render() -> None:
    c.ensure_state()
    price, nri, _disc = c.deck()

    pt.masthead("ops", "Triage Board",
                "Every well ranked by the risk-weighted dollars an intervention "
                "could protect — where to look first.")
    pt.context_bar([
        ("Surveillance fleet", c.scada_source_label(c.DISK_TOKEN)),
        ("Deck", c.deck_label()),
        ("Ranking", "risked NPV = risk × PV(net revenue) − intervention cost"),
    ])
    theme.data_badge("synthetic", "Modeled daily SCADA fleet with known ground truth "
                                  "— public production is monthly, not daily.")

    board = fr.enrich(c.board_with_deferred(price, nri))
    if board.empty:
        pt.empty_state("No wells in the fleet — nothing to triage.",
                       "Run bootstrap (first app start) to generate the fleet.")
        return
    import core
    if core.risk_scoring_degraded():
        st.warning("⚠️ **ESP risk model unavailable** — every well is showing the "
                   f"baseline {core.BASELINE_RISK_30D:.0%} failure risk, so this "
                   "ranking reflects deferred production only, not the failure signal. "
                   "Re-run bootstrap or check the model artifact / dependencies.")
    opportunities, watch, stable = c.triage_tiers(board)

    pt.kpi_row([
        {"label": "Fleet Size", "value": f"{len(board)} wells"},
        {"label": "Opportunities", "value": f"{len(opportunities)}",
         "help": "Wells whose recommended intervention clears its own cost today "
                 "(positive risk-weighted NPV) — value-accretive now."},
        {"label": "At-Risk Watch", "value": f"{len(watch)}",
         "delta_color": "off",
         "help": "Wells actively deferring production where an intervention is not "
                 "yet economic — monitor and re-rank, don't spend capital."},
        {"label": "Addressable Risked NPV",
         "value": f"${float(opportunities['est_risked_npv'].sum()):,.0f}",
         "help": "Σ of the positive risk-weighted net-to-operator NPV across the "
                 "value-accretive interventions. Total fleet deferment runs "
                 f"${float(board['deferred_usd_per_day'].sum()):,.0f}/day (net)."},
    ])

    pt.section("Top Opportunities — Value-Accretive Interventions",
               "Only wells whose intervention clears its cost today (positive "
               "risk-weighted NPV). A well off this list isn't necessarily healthy — "
               "it may be on the At-Risk Watch List below, where intervening now "
               "would lose money.")
    if opportunities.empty:
        pt.empty_state(
            "No value-accretive interventions on the fleet right now.",
            "Every flagged well is on the At-Risk Watch List below: the failure "
            "signal is present, but at today's risk and intervention cost the "
            "economics don't clear. The play is to hold and re-rank as risk climbs.")
    else:
        top = opportunities.head(12).iloc[::-1]
        bar = go.Figure(go.Bar(
            x=top["est_risked_npv"], y=top["well_id"], orientation="h",
            marker_color=theme.GREEN,
            customdata=top[["failure_risk_30d", "recommended_intervention",
                            "deferred_usd_per_day"]],
            hovertemplate="<b>%{y}</b><br>Risked NPV: $%{x:,.0f}"
                          "<br>Intervention: %{customdata[1]}"
                          "<br>30-day risk signal: %{customdata[0]:.0%}"
                          "<br>Deferred: $%{customdata[2]:,.0f}/day<extra></extra>",
            text=[f"${v:,.0f} · {i.replace('_', ' ')}"
                  for v, i in zip(top["est_risked_npv"],
                                  top["recommended_intervention"])],
            textposition="auto"))
        bar.update_layout(xaxis_title="Risk-weighted NPV ($, net to operator)",
                          yaxis_title="")
        st.plotly_chart(
            theme.style_fig(bar, height=max(280, 30 * len(top) + 90), legend=False),
            width="stretch")
        theme.source_note(
            "Risked NPV = the ESP 30-day failure signal × PV(net revenue the "
            "intervention protects) − the intervention cost (AFE cost rollup + PV10 "
            "economics at the deck price/NRI). The cost is certain, so only the upside "
            "is chance-weighted. The bar is labeled with the intervention to run.")
        c.pinned_pv10_caption()
        hero = fr.get(str(opportunities["well_id"].iloc[0]))
        if hero.hero:
            st.info(f"**{hero.well_id} — {hero.name}** · {hero.basin} Basin · "
                    f"{hero.formation} · {hero.lift} lift. {hero.storyline}")

    pt.section("Recommended Interventions",
               "What to run, on which well, what it costs, and what it protects — "
               "ranked by risk-weighted NPV. Open any well on Well 360 / Action "
               "Chain via the sidebar selector.")
    if opportunities.empty:
        st.caption("No value-accretive interventions right now — see the At-Risk "
                   "Watch List below.")
    else:
        st.dataframe(_intervention_table(opportunities), width="stretch",
                     hide_index=True)
        theme.source_note(
            "Intervention + cost come from the AFE component's cost database; "
            "'NPV Basis' flags wells where the full chain economics weren't reachable "
            "and a transparent proxy was used. ★ marks a fleet-registry hero well.")

    pt.section("At-Risk Watch List",
               "Failure signature present, but intervening now destroys value "
               "(non-positive risk-weighted NPV at today's risk and cost). The action "
               "is to MONITOR and re-rank as the signal strengthens — not to spend "
               "capital yet.")
    if watch.empty:
        st.caption("No wells on the watch list — every flagged well is either "
                   "value-accretive above or below the no-action thresholds.")
    else:
        w = watch.head(15)
        wt = pd.DataFrame({
            "Well": [f"★ {x}" if h else x for x, h in zip(w["well_id"], w["hero"])],
            "Field": w["basin"] + " · " + w["formation"],
            "Lift": w["lift"],
            "Risk Rank": w["failure_risk_30d"].rank(ascending=False).astype(int),
            "30-Day Risk Signal": w["failure_risk_30d"].map(lambda x: f"{x:.0%}"),
            "Deferred $/day": w["deferred_usd_per_day"].map(lambda x: f"${x:,.0f}"),
            "Indicated If It Fails": w["recommended_intervention"].str.replace("_", " "),
            "Risked NPV (now)": w["est_risked_npv"].map(lambda x: f"−${abs(x):,.0f}"),
        })
        st.dataframe(wt, width="stretch", hide_index=True)
        st.caption("'30-Day Risk Signal' is a fleet-relative ESP ranking on this "
                   "synthetic fleet, not a calibrated absolute probability. "
                   "'Indicated If It Fails' is the intervention that would be run if "
                   "the well deteriorates — it is NOT a recommendation to act today.")

    pt.section("No-Action Tier — Stable Wells",
               f"{len(stable)} wells producing on trend with no deferment and no "
               "value-accretive intervention — nothing to do today. Listed for "
               "completeness (full fleet coverage, not just the exceptions).")
    if stable.empty:
        st.caption("No wells in the stable tier on this run.")
    else:
        sd = pd.DataFrame({
            "Well": [f"★ {w}" if h else w
                     for w, h in zip(stable["well_id"], stable["hero"])],
            "Field": stable["basin"] + " · " + stable["formation"],
            "Lift": stable["lift"],
            "Lateral (ft)": stable["lateral_length_ft"].map(lambda x: f"{int(x):,}"),
            "Deferred $/day": stable["deferred_usd_per_day"].map(lambda x: f"${x:,.0f}"),
            "Status": "stable — no action",
        })
        st.dataframe(sd, width="stretch", hide_index=True, height=360)
        st.caption("These clear the action thresholds: no deferment and no positive "
                   "intervention NPV. Their ESP score is a low relative signal on "
                   "this fleet, not an absolute failure probability, so it is not "
                   "shown here to avoid implying a healthy well is about to fail.")

    raw = c.board_with_deferred(price, nri)  # display frame (real deferred joined in)
    st.download_button("Download triage board (CSV)", data=raw.to_csv(index=False),
                       file_name="ops_triage_board.csv", mime="text/csv",
                       help="Full ranked board, all tiers — no-action wells carry "
                            "intervention 'no_action' and opportunity 0.")

    theme.references(["npv", "shap"])


def _intervention_table(opps: pd.DataFrame) -> pd.DataFrame:
    """The value-accretive interventions, intervention-and-cost forward."""
    cost = opps["recommended_intervention"].map(c.intervention_cost)
    return pd.DataFrame({
        "Well": [f"★ {w}" if h else w
                 for w, h in zip(opps["well_id"], opps["hero"])],
        "Recommended Intervention": opps["recommended_intervention"].str.replace("_", " "),
        "Est. Cost": cost.map(lambda x: "—" if pd.isna(x) else f"${x:,.0f}"),
        "Risked NPV": opps["est_risked_npv"].map(lambda x: f"${x:,.0f}"),
        "Addressable BOPD": opps["incremental_bopd"].map(lambda x: f"{x:,.0f}"),
        "Deferred $/day": opps["deferred_usd_per_day"].map(lambda x: f"${x:,.0f}"),
        "30-Day Risk": opps["failure_risk_30d"].map(lambda x: f"{x:.0%}"),
        "Field": opps["basin"] + " · " + opps["formation"],
        "Lift": opps["lift"],
        "NPV Basis": opps["npv_basis"].str.replace("_", " "),
    })
