"""Loss Accounting · Recovery Work Queue — ranked recoverable opportunities.

Ported from deferment-iq's demo (v0.5.1): one actionable item per (well,
recoverable cause), ranked by recoverable $ ÷ MTTR. Real public sources have no
reason codes, so the queue is honestly N/A there.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import product_theme as pt
import theme

from views import _common as c


def render() -> None:
    c.ensure_state()
    price, _nri, _disc = c.deck()

    pt.masthead("ops", "Recovery Work Queue",
                "From where the barrels are lost to what to do next, what it is "
                "worth, and who acts — the Quantify → Authorize handoff.")

    source = c.loss_source_control()
    is_real = c.loss_is_real(source)
    pt.context_bar([
        ("Loss-accounting source", c.loss_context(source)),
        ("Deck", c.deck_label()),
        ("Ranking", "priority = recoverable $ ÷ MTTR (days)"),
    ])
    theme.data_badge(*c.loss_badge(source))

    if is_real:
        st.info("**Cause attribution N/A — no public reason codes.** The recovery "
                "queue ranks actionable items per (well, **recoverable cause**), "
                "which requires the operator's coded downtime log. Public monthly "
                "filings give real deferment **quantity** (from days-produced) but "
                "no cause, so there is nothing to attribute or authorize here. "
                "Switch the source above to **Synthetic (reason-coded demo)** for "
                "the full Quantify → Authorize work queue.")
        theme.references(["npv"])
        return

    import core
    fleet, evc, daily = c.deferment_data(c.loss_source_token(source), price)
    queue = core.deferment_analytics.recovery_queue(daily, evc, price)

    if not len(queue):
        pt.empty_state("No recoverable deferment in the current period — nothing "
                       "to queue.")
        theme.references(["npv"])
        return

    toprow = queue.iloc[0]
    pt.kpi_row([
        {"label": "Total Recoverable",
         "value": f"${float(queue['recoverable_usd'].sum()):,.0f}",
         "help": "Sum of recoverable $ across every queued item; planned + "
                 "reservoir losses excluded (you can't get those barrels back)."},
        {"label": "Actionable Items", "value": f"{len(queue)}",
         "help": "Distinct (well, recoverable cause) interventions."},
        {"label": "Fastest High-Value Win",
         "value": f"{toprow['well_id']} · {toprow['cause']}",
         "delta": f"${toprow['recoverable_usd']:,.0f} · {toprow['mttr_days']:.1f} d",
         "delta_color": "off",
         "help": "Highest value-per-day-to-restore item — do this first."},
    ])

    st.caption("Recoverable $ is **gross** deferred revenue (lost barrels × deck "
               "price) over the period — the base-management recovery target, not a "
               "net-of-cost NPV. Per-well intervention cost, NRI, and PV10 economics "
               "are applied on the **Action Chain** page when an item is authorized. "
               "Priority = recoverable $ ÷ MTTR ranks value per day-to-restore.")

    pt.section("Top Recovery Opportunities by $",
               "A quick high-value win outranks a slow one of similar size.")
    bar = queue.head(12).iloc[::-1]
    causes = list(dict.fromkeys(queue["cause"]))
    cmap = {cause: theme.COLORWAY[i % len(theme.COLORWAY)]
            for i, cause in enumerate(causes)}
    bf = go.Figure()
    for cause in causes:
        sub = bar[bar["cause"] == cause]
        if not len(sub):
            continue
        bf.add_bar(
            y=[f"{w} · {cause}" for w in sub["well_id"]], x=sub["recoverable_usd"],
            name=cause, orientation="h", marker_color=cmap[cause],
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>")
    bf.update_layout(barmode="stack", xaxis_title="Recoverable $")
    st.plotly_chart(theme.style_fig(bf, height=420), width="stretch")
    theme.source_note(
        "Recoverable $ = recoverable bbl × deck oil price, per (well, recoverable "
        "cause); planned + reservoir excluded. Ranked by priority = recoverable $ ÷ "
        "MTTR (days) — value per day-to-restore.")

    pt.section("The Queue")
    disp = queue.copy()
    disp.insert(0, "#", range(1, len(disp) + 1))
    disp["recoverable_usd"] = disp["recoverable_usd"].map(lambda v: f"${v:,.0f}")
    disp["recoverable_bbl"] = disp["recoverable_bbl"].map(lambda v: f"{v:,.0f}")
    disp["mttr_days"] = disp["mttr_days"].map(lambda v: f"{v:.1f}")
    disp["priority_score"] = disp["priority_score"].map(lambda v: f"{v:,.0f}")
    disp = disp[["#", "well_id", "cause", "suggested_action",
                 "recoverable_bbl", "recoverable_usd", "mttr_days", "priority_score"]]
    disp.columns = ["#", "Well", "Cause", "Suggested Action",
                    "Recoverable bbl", "Recoverable $", "MTTR (d)", "Priority ($/day)"]
    st.dataframe(disp, width="stretch", hide_index=True)
    st.download_button("Download work queue (CSV)", data=queue.to_csv(index=False),
                       file_name="ops_recovery_work_queue.csv", mime="text/csv")

    pt.section("Authorize the Top Interventions",
               "Each item is sized and ready to hand to capital authorization.")
    for _, r in queue.head(5).iterrows():
        st.markdown(
            f"**{r['well_id']} — {r['cause']}** · {r['suggested_action']} · recover "
            f"**{r['recoverable_bbl']:,.0f} bbl (${r['recoverable_usd']:,.0f})**, "
            f"~{r['mttr_days']:.1f}-day restore")
    st.caption("Surveillance-fleet wells route through this console's **Action "
               "Chain** page (detect → predict → authorize). Program-level capital "
               "allocation lives in the **Capital Desk** product (sidebar switcher).")

    _proactive_drift_lane(queue)

    theme.references(["npv", "pareto"])


def _proactive_drift_lane(queue) -> None:
    """Forward-looking complement to the reactive recovery queue: wells the deep
    autoencoder flags as drifting that aren't deferring yet. Address drift before it
    becomes downtime. Silent no-op when the optional DL extras/model are absent."""
    ew = c.early_warning_flags(c.DISK_TOKEN)
    if ew.empty:
        return
    deep = ew[ew["deep_only"]]
    queued = (set(queue["well_id"].astype(str))
              if queue is not None and len(queue) else set())
    proactive = deep[~deep["well"].isin(queued)]
    if proactive.empty:
        return
    pt.section("Proactive — Deep-Drift Candidates (Pre-Deferment)",
               "Wells the deep autoencoder flags as drifting from normal that aren't "
               "deferring yet — not in the reactive queue above. Catching drift before "
               "it becomes downtime is the cheapest barrel you'll ever save.")
    rows = proactive.head(10)
    tbl = pd.DataFrame({
        "Well": list(rows["well"]),
        "Top Drifting Channel": list(rows["driver"]),
        "Drift Score": [f"{s:.2f}" for s in rows["score"]],
        "Suggested Action": "inspect / pre-empt before deferment",
    })
    st.dataframe(tbl, width="stretch", hide_index=True)
    theme.source_note(
        "Forward-looking complement to the reactive queue: deep-drift flags from the "
        "LSTM autoencoder ('deep-only' = the rate-drop alarm has not fired). Surfaced "
        "so slow degraders are addressed before they start deferring barrels. Full "
        "leaderboard on Surveillance → Early Warning · Deep AI.")
