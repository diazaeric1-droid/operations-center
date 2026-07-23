"""Today · Optimization Board — the whole fleet ranked by risked-NPV opportunity.

Ported from pe-pipeline's fleet-triage overview (same ranking engine — the
product tests pin numeric equality against pipeline_core.rank_fleet). Renamed
from the earlier "triage board" title per PE field feedback (module filename kept
to avoid import churn; every user-visible string says Optimization Board).
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

    pt.masthead("ops", "Optimization Board",
                "Every well ranked by the risk-weighted dollars an intervention "
                "could protect — where to look first.")
    pt.context_bar([
        ("Surveillance fleet", c.scada_source_label(c.DISK_TOKEN)),
        ("Deck", c.deck_label()),
        ("Ranking", "risked NPV = risk × PV(net revenue) − intervention cost"),
    ])
    c.page_purpose(
        "**The question this page answers: which economic interventions do I "
        "authorize first, ranked by risked NPV?**\n\n"
        "- **When:** fourth stop of the 6:30am loop — after the Brief tells you "
        "what's wrong, this ranks what's worth capital.\n"
        "- **Headline read:** *Addressable Risked NPV* ($) — Σ of positive "
        "risk-weighted NPV (risked NPV = 30-day failure signal × PV of the net "
        "revenue the intervention protects − intervention cost; the cost is "
        "certain, so only the upside is chance-weighted). Four tiers: Restore "
        "(down — fix first), Opportunities (act now), Watch (signal present, "
        "doesn't pay yet), Stable (nothing to do).\n"
        "- **Selecting a row** opens the well on **Surveillance** to confirm the "
        "signal — the confirm-before-authorize discipline; the **Action Chain** "
        "then picks the same well up automatically (the selection is "
        "console-wide).\n"
        "- **Next:** build the AFE on the **Action Chain**.")
    theme.data_badge("synthetic", "Modeled daily SCADA fleet with known ground truth "
                                  "— public production is monthly, not daily.")

    full_board = fr.enrich(c.board_with_deferred(price, nri))
    if full_board.empty:
        pt.empty_state("No wells in the fleet — nothing to rank.",
                       "Run bootstrap (first app start) to generate the fleet.")
        return
    with st.expander("Filters — CTB · lift type · basin · county", expanded=False):
        keep = set(c.fleet_filter_controls("ob", list(full_board["well_id"].astype(str))))
    board = full_board[full_board["well_id"].astype(str).isin(keep)] \
        .reset_index(drop=True)
    net_view = c.gross_net_toggle()
    if len(board) < len(full_board):
        st.caption(f"Filtered: **{len(board)} of {len(full_board)}** wells — the KPIs "
                   "and tiers below reflect the filtered selection (the CSV export "
                   "stays full-fleet).")
        if board.empty:
            pt.empty_state("No wells match the active filters.",
                           "Clear a filter above to bring wells back.")
            return
    nmap = c.nri_map(board["well_id"].astype(str))
    ev_days = c.ongoing_event_days(c.DISK_TOKEN, price)
    import core
    if core.risk_scoring_degraded():
        st.warning("⚠️ **ESP risk model unavailable** — every well is showing the "
                   f"baseline {core.BASELINE_RISK_30D:.0%} failure risk, so this "
                   "ranking reflects deferred production only, not the failure signal. "
                   "Re-run bootstrap or check the model artifact / dependencies.")
    # Pull currently-down / shut-in wells into a Restore queue BEFORE the triage
    # partition — a shut-in well is a restore-first job, not a priced opportunity (its
    # Well 360 says as much). View-layer only; the certified ranking is untouched.
    restore, board_live = c.restore_tier(board, c.down_well_set(c.DISK_TOKEN))
    opportunities, watch, stable = c.triage_tiers(board_live)

    _sel_def_label, _sel_def = _deferred_usd_display(board, nmap, net_view, price)
    pt.kpi_row([
        {"label": "Selection", "value": f"{len(board)} wells",
         "delta": (f"of {len(full_board)} fleet"
                   if len(board) < len(full_board) else None),
         "delta_color": "off"},
        {"label": "Opportunities", "value": f"{len(opportunities)}",
         "help": "Wells with a real trigger (deferring production, the fleet's top-quartile "
                 "risk, OR ≥50% calibrated failure probability) whose recommended, "
                 "LIFT-APPROPRIATE intervention clears its own cost today (positive "
                 "risk-weighted NPV). A cheap intervention that merely pencils on a "
                 "no-signal well is not enough."},
        {"label": "At-Risk Watch", "value": f"{len(watch)}",
         "delta_color": "off",
         "help": "Wells with a trigger (deferring production or elevated risk) where "
                 "the intervention is not yet economic — monitor and re-rank, don't "
                 "spend capital."},
        {"label": "Addressable Risked NPV",
         "value": f"${float(opportunities['est_risked_npv'].sum()):,.0f}",
         "help": "Σ of the positive risk-weighted net-to-operator NPV across the "
                 "value-accretive interventions (certified economics at the deck "
                 "NRI). The selection's deferment runs "
                 f"${float(_sel_def.sum()):,.0f}/day "
                 f"({'net, per-well NRI' if net_view else 'gross 8/8'})."},
    ])
    st.caption(
        f"Selection deferment: **{float(board['deferred_bopd'].sum()):,.0f} bbl/day "
        f"gross** · **${float(_sel_def.sum()):,.0f}/day "
        f"{'net (per-well NRI)' if net_view else 'gross'}**. The certified ranking "
        "economics (risked NPV) use the sidebar deck NRI; the toggle above re-states "
        "displayed deferred-$ columns only.")

    _ranking_scorecard(full_board)

    if not restore.empty:
        pt.section("Restore First — Wells Down",
                   f"{len(restore)} well(s) producing ≈0 right now. A shut-in well is a "
                   "restore-production job, not a priced intervention — these are held "
                   "OUT of the opportunity ranking below (you can't 'optimize gas lift' "
                   "on a well that's offline). Bring them back, then they re-rank.")
        rt = pd.DataFrame({
            "Well": [f"★ {w}" if h else w
                     for w, h in zip(restore["well_id"], restore["hero"])],
            "CTB": restore["ctb"],
            "Field": restore["basin"] + " · " + restore["formation"],
            "Lift": restore["lift"],
            "Down (days)": [f"{ev_days[str(w)]}d" if str(w) in ev_days else "—"
                            for w in restore["well_id"]],
            "30-Day Risk Signal": restore["failure_risk_30d"].map(lambda x: f"{x:.0%}"),
            "Status": "down — restore production first",
        })
        ev_r = st.dataframe(rt, width="stretch", hide_index=True,
                            on_select="rerun", selection_mode="single-row",
                            key="ob_restore_sel")
        c.handle_row_jump(ev_r, restore, "_ob_restore_jump")
        st.caption("Currently down (≈0 production vs the well's own baseline). "
                   "'Down (days)' is the open event's running duration from the event "
                   "state machine. Select a row to open the well on Surveillance. The "
                   "economic limit and any priced intervention recompute off the "
                   "recovered rate once the well is back online — see Well 360.")

    pt.section("Top Opportunities — Value-Accretive Interventions",
               "Wells with a real trigger (deferring production, the fleet's top-quartile "
               "risk, or ≥50% calibrated failure probability) whose LIFT-APPROPRIATE "
               "intervention clears its cost today (positive risk-weighted NPV). A well "
               "off this list isn't necessarily healthy — it may be on the At-Risk Watch "
               "List below, where intervening now would lose money.")
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
        top_wid = str(opportunities["well_id"].iloc[0])
        if top_wid in ev_days:
            st.warning(f"⚠ **{top_wid} is in an ongoing downtime event "
                       f"({ev_days[top_wid]}d)** — verify the post-restart rate "
                       "before acting on its recommendation.")
        hero = fr.get(top_wid)
        if hero.hero:
            st.info(f"**{hero.well_id} — {hero.name}** · {hero.basin} Basin · "
                    f"{hero.formation} · {hero.lift} lift. {hero.storyline}")

    pt.section("Recommended Interventions",
               "What to run, on which well, what it costs, and what it protects — "
               "ranked by risk-weighted NPV. Select a row to open the well on "
               "Surveillance and confirm the recommendation against its production.")
    if opportunities.empty:
        st.caption("No value-accretive interventions right now — see the At-Risk "
                   "Watch List below.")
    else:
        ev_i = st.dataframe(
            _intervention_table(opportunities, ev_days, nmap, net_view, price),
            width="stretch", hide_index=True,
            on_select="rerun", selection_mode="single-row", key="ob_interv_sel",
            column_config={
                "Addressable BOPD": st.column_config.TextColumn(
                    help="Incremental oil the intervention is modeled to protect "
                         "or add."),
                "NPV Basis": st.column_config.TextColumn(
                    help="Full chain economics vs transparent proxy — flags wells "
                         "where the certified chain economics weren't reachable."),
                "Downtime Context": st.column_config.TextColumn(
                    help="Open state-machine event on this well — verify the "
                         "post-restart rate before acting."),
                "30-Day Risk": st.column_config.TextColumn(
                    help="Platt-calibrated 30-day failure probability from the "
                         "failure-risk model (model card on Methods & "
                         "Limitations)."),
            })
        c.handle_row_jump(ev_i, opportunities, "_ob_interv_jump")
        c.next_step("Action Chain",
                    "→ Build the AFE for the selected well on the Action Chain")
        st.caption("Selecting a row sets the console-wide well — Surveillance "
                   "opens to confirm the signal; the Action Chain picks the same "
                   "well up automatically.")
        theme.source_note(
            "Intervention + cost come from the AFE component's cost database; "
            "'NPV Basis' flags wells where the full chain economics weren't reachable "
            "and a transparent proxy was used. ★ marks a fleet-registry hero well. "
            "'Downtime Context' flags wells in an OPEN event from the state machine — "
            "verify the post-restart rate before acting on those recommendations.")
    st.caption(c.NO_CHOKE_CAPTION)

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
        w_lbl, w_usd = _deferred_usd_display(w, nmap, net_view, price)
        wt = pd.DataFrame({
            "Well": [f"★ {x}" if h else x for x, h in zip(w["well_id"], w["hero"])],
            "CTB": w["ctb"],
            "Field": w["basin"] + " · " + w["formation"],
            "Lift": w["lift"],
            "Risk Rank": w["failure_risk_30d"].rank(ascending=False).astype(int),
            "30-Day Risk Signal": w["failure_risk_30d"].map(lambda x: f"{x:.0%}"),
            w_lbl: w_usd.map(lambda x: f"${x:,.0f}"),
            "Indicated If It Fails": w["recommended_intervention"].str.replace("_", " "),
            "Risked NPV (now)": w["est_risked_npv"].map(lambda x: f"−${abs(x):,.0f}"),
        })
        ev_w = st.dataframe(wt, width="stretch", hide_index=True,
                            on_select="rerun", selection_mode="single-row",
                            key="ob_watch_sel",
                            column_config={
                                "Risk Rank": st.column_config.NumberColumn(
                                    help="Rank within this watch list by 30-day "
                                         "signal (1 = highest risk)."),
                                "Indicated If It Fails": st.column_config.TextColumn(
                                    help="The intervention that would run if the "
                                         "well deteriorates — NOT a recommendation "
                                         "to act today."),
                                "Risked NPV (now)": st.column_config.TextColumn(
                                    help="Risk × PV(net revenue) − cost at TODAY's "
                                         "signal — negative here, which is why "
                                         "this well is watch, not authorize."),
                            })
        c.handle_row_jump(ev_w, w, "_ob_watch_jump")
        st.caption("'30-Day Risk Signal' is a Platt-calibrated probability from the "
                   "ESP model trained on this fleet's labeled faults (calibrated "
                   "out-of-fold AUROC ≈0.98 on clean synthetic signatures — see Methods). "
                   "'Indicated If It Fails' is the intervention that would be run if "
                   "the well deteriorates — it is NOT a recommendation to act today.")

    pt.section("No-Action Tier — Stable Wells",
               f"{len(stable)} wells with no trigger to act — not deferring production, "
               "not in the fleet's top-quartile risk, and below the "
               f"{int(core.ELEVATED_RISK_ABS_30D * 100)}% calibrated-risk floor — so "
               "there's nothing to do today even where a cheap intervention would "
               "technically pencil. Listed for completeness (full fleet coverage).")
    if stable.empty:
        st.caption("No wells in the stable tier on this run.")
    else:
        s_lbl, s_usd = _deferred_usd_display(stable, nmap, net_view, price)
        sd = pd.DataFrame({
            "Well": [f"★ {w}" if h else w
                     for w, h in zip(stable["well_id"], stable["hero"])],
            "CTB": stable["ctb"],
            "Field": stable["basin"] + " · " + stable["formation"],
            "Lift": stable["lift"],
            "Lateral (ft)": stable["lateral_length_ft"].map(lambda x: f"{int(x):,}"),
            s_lbl: s_usd.map(lambda x: f"${x:,.0f}"),
            "Status": "stable — no action",
        })
        st.dataframe(sd, width="stretch", hide_index=True, height=360,
                     column_config={
                         "Lateral (ft)": st.column_config.TextColumn(
                             help="Completion lateral length — context for peer "
                                  "comparison, not a ranking input."),
                         "Status": st.column_config.TextColumn(
                             help="No trigger to act today — see the caption below "
                                  "for the exact gate."),
                     })
        st.caption("No trigger to act: not deferring production, below the fleet's "
                   "top-quartile risk cut, AND below the "
                   f"{int(core.ELEVATED_RISK_ABS_30D * 100)}% calibrated-risk floor. This "
                   "is a 'nothing to do today' tier, not a clean bill of health — open any "
                   "well on Well 360 for its specific 30-day signal.")

    raw = c.board_with_deferred(price, nri)  # display frame (real deferred joined in)
    st.download_button("Download optimization board (CSV)", data=raw.to_csv(index=False),
                       file_name="ops_optimization_board.csv", mime="text/csv",
                       help="Full ranked board (unfiltered), all tiers — no-action "
                            "wells carry intervention 'no_action' and opportunity 0.")

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


def _deferred_usd_display(frame: pd.DataFrame, nmap: dict, net_view: bool,
                          price: float) -> tuple[str, pd.Series]:
    """(column label, $/day series) for the displayed deferred-$ convention:
    gross (bopd × price) or net entitlement (bopd × price × the WELL's OWN NRI —
    registry default or the Data-page session override). Display-layer only; the
    certified ranking columns are untouched."""
    if net_view:
        f = frame["well_id"].astype(str).map(lambda w: nmap.get(w, 0.8))
        return ("Deferred $/day (net, well NRI)",
                (frame["deferred_bopd"] * price * f).round(0))
    return "Deferred $/day (gross)", (frame["deferred_bopd"] * price).round(0)


def _downtime_context(well_id, ev_days: dict) -> str:
    n = ev_days.get(str(well_id))
    return f"⚠ in ongoing event {n}d — verify post-restart" if n else "—"


def _intervention_table(opps: pd.DataFrame, ev_days: dict, nmap: dict,
                        net_view: bool, price: float) -> pd.DataFrame:
    """The value-accretive interventions, intervention-and-cost forward, with the
    open-event downtime context beside each recommendation."""
    cost = opps["recommended_intervention"].map(c.intervention_cost)
    lbl, usd = _deferred_usd_display(opps, nmap, net_view, price)
    return pd.DataFrame({
        "Well": [f"★ {w}" if h else w
                 for w, h in zip(opps["well_id"], opps["hero"])],
        "Recommended Intervention": opps["recommended_intervention"].str.replace("_", " "),
        "Downtime Context": [_downtime_context(w, ev_days) for w in opps["well_id"]],
        "Est. Cost": cost.map(lambda x: "—" if pd.isna(x) else f"${x:,.0f}"),
        "Risked NPV": opps["est_risked_npv"].map(lambda x: f"${x:,.0f}"),
        "Addressable BOPD": opps["incremental_bopd"].map(lambda x: f"{x:,.0f}"),
        lbl: usd.map(lambda x: f"${x:,.0f}"),
        "30-Day Risk": opps["failure_risk_30d"].map(lambda x: f"{x:.0%}"),
        "CTB": opps["ctb"],
        "Field": opps["basin"] + " · " + opps["formation"],
        "Lift": opps["lift"],
        "NPV Basis": opps["npv_basis"].str.replace("_", " "),
    })
