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
         "help": "Wells with a real trigger (deferring production OR elevated "
                 "fleet-relative risk) whose recommended, LIFT-APPROPRIATE intervention "
                 "clears its own cost today (positive risk-weighted NPV). A cheap "
                 "intervention that merely pencils on a no-signal well is not enough."},
        {"label": "At-Risk Watch", "value": f"{len(watch)}",
         "delta_color": "off",
         "help": "Wells with a trigger (deferring production or elevated risk) where "
                 "the intervention is not yet economic — monitor and re-rank, don't "
                 "spend capital."},
        {"label": "Addressable Risked NPV",
         "value": f"${float(opportunities['est_risked_npv'].sum()):,.0f}",
         "help": "Σ of the positive risk-weighted net-to-operator NPV across the "
                 "value-accretive interventions. Total fleet deferment runs "
                 f"${float(board['deferred_usd_per_day'].sum()):,.0f}/day (net)."},
    ])

    _ranking_scorecard(board)

    pt.section("Top Opportunities — Value-Accretive Interventions",
               "Wells with a real trigger (deferring production or elevated "
               "fleet-relative risk) whose LIFT-APPROPRIATE intervention clears its "
               "cost today (positive risk-weighted NPV). A well off this list isn't "
               "necessarily healthy — it may be on the At-Risk Watch List below, where "
               "intervening now would lose money.")
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
               f"{len(stable)} wells with no trigger to act — not deferring production "
               "and not in the fleet's elevated-risk quartile — so there's nothing to "
               "do today even where a cheap intervention would technically pencil. "
               "Listed for completeness (full fleet coverage, not just the exceptions).")
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
        st.caption("No trigger to act: not deferring production and not in the fleet's "
                   "elevated-risk quartile. Their ESP score is a low relative signal on "
                   "this fleet, not an absolute failure probability, so it is not shown "
                   "here to avoid implying a healthy well is about to fail.")

    raw = c.board_with_deferred(price, nri)  # display frame (real deferred joined in)
    st.download_button("Download triage board (CSV)", data=raw.to_csv(index=False),
                       file_name="ops_triage_board.csv", mime="text/csv",
                       help="Full ranked board, all tiers — no-action wells carry "
                            "intervention 'no_action' and opportunity 0.")

    theme.references(["npv", "shap"])


def _ranking_scorecard(board: pd.DataFrame) -> None:
    """Does the ranking actually surface the impaired wells? precision@k + lift vs
    random, scored against the fleet's known seeded faults — the same honest-backtest
    treatment the digest's event detector and the deferment classifier already get."""
    import core
    sc = core.triage_scorecard(board)
    if not sc:
        return
    with st.expander("Ranking scorecard — does this ranking catch the failures? "
                     f"(P@10 {sc['at_k'][10]['precision']:.0%}, "
                     f"{sc['at_k'][10]['lift']:.1f}× lift)", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Precision @5", f"{sc['at_k'][5]['precision']:.0%}",
                       f"{sc['at_k'][5]['lift']:.1f}× vs random", delta_color="off")
        cols[1].metric("Precision @10", f"{sc['at_k'][10]['precision']:.0%}",
                       f"{sc['at_k'][10]['lift']:.1f}× vs random", delta_color="off")
        cols[2].metric("Precision @20", f"{sc['at_k'][20]['precision']:.0%}",
                       f"{sc['at_k'][20]['lift']:.1f}× vs random", delta_color="off")
        cols[3].metric("Recall @impaired", f"{sc['recall_at_n_impaired']:.0%}",
                       f"{sc['n_impaired']}/{sc['n_wells']} seeded", delta_color="off")
        st.caption(
            f"Ground truth: {sc['n_impaired']} of {sc['n_wells']} wells carry a real "
            f"seeded fault ({sc['base_rate']:.0%} base rate). Ranking by risked NPV, the "
            f"top 10 are {sc['at_k'][10]['lift']:.1f}× more likely to be truly impaired "
            "than a random draw — honest (not a trivial 100%): low-rate failure modes "
            "(e.g. early electrical) defer few barrels, so they rank lower. Scored on "
            "the generator's signature labels for THIS fleet (the ESP model's own "
            "labels.csv is a different fleet and doesn't join here).")


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
