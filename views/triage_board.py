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
        ("Ranking", "risked NPV = intervention net NPV × 30-day ESP failure risk"),
    ])
    theme.data_badge("synthetic", "Modeled daily SCADA fleet with known ground truth "
                                  "— public production is monthly, not daily.")

    board = fr.enrich(c.board(price, nri))
    if board.empty:
        pt.empty_state("No wells in the fleet — nothing to triage.",
                       "Run bootstrap (first app start) to generate the fleet.")
        return
    action, no_action = c.split_board(board)

    at_risk = int((board["failure_risk_30d"] >= 0.5).sum())
    pt.kpi_row([
        {"label": "Fleet Size", "value": f"{len(board)} wells"},
        {"label": "At-Risk (≥50% 30-day)", "value": f"{at_risk}",
         "help": "ESP 30-day failure probability ≥ 50%."},
        {"label": "Total Deferred $/day",
         "value": f"${float(board['deferred_usd_per_day'].sum()):,.0f}",
         "help": "Σ deferred bopd × price × NRI across the fleet."},
        {"label": "Addressable Risked NPV",
         "value": f"${float(board['est_risked_npv'].clip(lower=0).sum()):,.0f}",
         "help": "Σ of each well's positive risk-weighted net-to-operator NPV "
                 "from its recommended intervention."},
    ])

    pt.section("Top Opportunities — Risked NPV by Well")
    top = action.head(12).iloc[::-1]
    colors = [theme.RED if r >= 0.5 else theme.AMBER for r in top["failure_risk_30d"]]
    bar = go.Figure(go.Bar(
        x=top["est_risked_npv"], y=top["well_id"], orientation="h",
        marker_color=colors,
        customdata=top[["failure_risk_30d", "recommended_intervention"]],
        hovertemplate="<b>%{y}</b><br>Risked NPV: $%{x:,.0f}"
                      "<br>30-day risk: %{customdata[0]:.0%}"
                      "<br>Intervention: %{customdata[1]}<extra></extra>",
        text=[f"${v:,.0f}" for v in top["est_risked_npv"]], textposition="auto",
    ))
    bar.update_layout(xaxis_title="Risked NPV ($)", yaxis_title="")
    st.plotly_chart(theme.style_fig(bar, height=380, legend=False), width="stretch")
    theme.source_note(
        "Risked NPV = net-to-operator NPV of the recommended intervention (AFE cost "
        "rollup + PV10 economics at the deck price/NRI) × the ESP 30-day failure "
        "probability. Red = risk ≥ 50%, amber below.")
    c.pinned_pv10_caption()

    if not action.empty:
        hero = fr.get(str(action["well_id"].iloc[0]))
        if hero.hero:
            st.info(f"**{hero.well_id} — {hero.name}** · {hero.basin} Basin · "
                    f"{hero.formation} · {hero.lift} lift. {hero.storyline}")

    pt.section("Action Wells",
               "Sorted descending by opportunity score; open any well on the "
               "Well 360 / Action Chain pages via the sidebar selector.")
    if action.empty:
        pt.empty_state("All wells are below the no-action thresholds — nothing "
                       "requires attention right now.")
    else:
        disp = pd.DataFrame({
            "Well": [f"★ {w}" if h else w
                     for w, h in zip(action["well_id"], action["hero"])],
            "Field": action["basin"] + " · " + action["formation"],
            "Lift": action["lift"],
            "Lateral (ft)": action["lateral_length_ft"].map(lambda x: f"{int(x):,}"),
            "30-Day Risk": action["failure_risk_30d"].map(lambda x: f"{x:.0%}"),
            "Deferred $/day": action["deferred_usd_per_day"].map(lambda x: f"${x:,.0f}"),
            "Addressable BOPD": action["incremental_bopd"].map(lambda x: f"{x:,.0f}"),
            "Intervention": action["recommended_intervention"].str.replace("_", " "),
            "Risked NPV": action["est_risked_npv"].map(lambda x: f"${x:,.0f}"),
            "NPV Basis": action["npv_basis"].str.replace("_", " "),
        })
        st.dataframe(disp, width="stretch", hide_index=True)
    theme.source_note(
        "Where the chain's AFE intervention economics aren't reachable, a transparent "
        "fallback is used — deferred $/day × 365 × failure risk — flagged in the "
        "'NPV Basis' column. ★ marks a fleet-registry hero well with an end-to-end "
        "story; field / lift / lateral come from the shared registry.")

    pt.section("No-Action Tier",
               "Below the thresholds: risked NPV < $10k AND 30-day risk < 15%, "
               "or zero deferment with risk < 10%.")
    if no_action.empty:
        st.caption("No wells in the no-action tier on this run — every well clears "
                   "the action thresholds.")
    else:
        with st.expander(f"{len(no_action)} well(s) require no action"):
            show = no_action[["well_id", "failure_risk_30d", "deferred_bopd",
                              "deferred_usd_per_day"]].copy()
            show["failure_risk_30d"] = show["failure_risk_30d"].map(lambda x: f"{x:.0%}")
            show["deferred_usd_per_day"] = show["deferred_usd_per_day"].map(
                lambda x: f"${x:,.0f}")
            show.columns = ["Well", "30-Day Risk", "Deferred BOPD", "Deferred $/day"]
            st.dataframe(show, width="stretch", hide_index=True)

    raw = c.board(price, nri)  # un-enriched frame: the certified ranking columns only
    st.download_button("Download triage board (CSV)", data=raw.to_csv(index=False),
                       file_name="ops_triage_board.csv", mime="text/csv",
                       help="Full ranked board, both tiers — no-action wells carry "
                            "intervention 'no_action' and opportunity 0.")

    theme.references(["npv", "shap"])
