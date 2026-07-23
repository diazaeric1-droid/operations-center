"""Loss Accounting · Recovery Work Queue — ranked recoverable opportunities.

Ported from deferment-iq's demo (v0.5.1): one actionable item per (well,
recoverable cause), ranked by recoverable $ ÷ MTTR. Real public sources have no
reason codes, so the queue is honestly N/A there.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import product_theme as pt
import theme

from views import _common as c


def _loss_well(wid) -> str:
    """Display label for a LOSS-BOOK well. The monthly loss-accounting fleet
    reuses ids like 'well_013' that ALSO exist in the daily surveillance fleet —
    but they are DIFFERENT wells from a different dataset ('no fake join' is the
    product's own invariant). The '(loss book)' suffix keeps a user from carrying
    a queue id into the Action Chain and unknowingly building an AFE for an
    unrelated surveillance well."""
    return f"{wid} (loss book)"


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
    c.page_purpose(
        "**The question this page answers: of the barrels we CAN get back, "
        "which jobs do I run first?**\n\n"
        "- **When:** last stop of the loss-accounting leg — after Causes & "
        "Pareto names the causes, this ranks one actionable item per (well, "
        "recoverable cause).\n"
        "- **Headline read:** *Priority ($/day)* = recoverable $ ÷ MTTR (mean "
        "time to restore, days) — value per day-to-restore, so a quick "
        "high-value win outranks a slow one of similar size. Recoverable $ is "
        "GROSS deferred revenue (lost barrels × deck price), not a net-of-cost "
        "NPV.\n"
        "- **Careful:** these well ids belong to the monthly loss-accounting "
        "fleet, NOT the surveillance wells that share the same id — the queue "
        "sizes the job; it cannot hand a well to the Action Chain.\n"
        "- **Next:** authorization for surveillance-fleet wells runs on the "
        "**Action Chain**; program-level capital lives in **Capital Desk**.")
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
         "value": f"{_loss_well(toprow['well_id'])} · {toprow['cause']}",
         "delta": f"${toprow['recoverable_usd']:,.0f} · {toprow['mttr_days']:.1f} d",
         "delta_color": "off",
         "help": "Highest value-per-day-to-restore item — do this first. "
                 "'(loss book)' = the monthly loss-accounting fleet, not the "
                 "surveillance well with the same id."},
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
            y=[f"{_loss_well(w)} · {cause}" for w in sub["well_id"]],
            x=sub["recoverable_usd"],
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
    disp["well_id"] = disp["well_id"].map(_loss_well)
    disp["recoverable_usd"] = disp["recoverable_usd"].map(lambda v: f"${v:,.0f}")
    disp["recoverable_bbl"] = disp["recoverable_bbl"].map(lambda v: f"{v:,.0f}")
    disp["mttr_days"] = disp["mttr_days"].map(lambda v: f"{v:.1f}")
    disp["priority_score"] = disp["priority_score"].map(lambda v: f"{v:,.0f}")
    disp = disp[["#", "well_id", "cause", "suggested_action",
                 "recoverable_bbl", "recoverable_usd", "mttr_days", "priority_score"]]
    disp.columns = ["#", "Well", "Cause", "Suggested Action",
                    "Recoverable bbl", "Recoverable $", "MTTR (d)", "Priority ($/day)"]
    st.dataframe(disp, width="stretch", hide_index=True,
                 column_config={
                     "Well": st.column_config.TextColumn(
                         help="Monthly loss-accounting fleet — NOT the "
                              "surveillance well sharing the same id (different "
                              "dataset; no join is faked)."),
                     "MTTR (d)": st.column_config.TextColumn(
                         help="Mean time to restore for this cause, days — the "
                              "denominator of the priority score."),
                     "Priority ($/day)": st.column_config.TextColumn(
                         help="Recoverable $ ÷ MTTR — value per day-to-restore; "
                              "the queue's ranking key."),
                 })
    st.download_button("Download work queue (CSV)", data=queue.to_csv(index=False),
                       file_name="ops_recovery_work_queue.csv", mime="text/csv")

    pt.section("Authorize the Top Interventions",
               "Each item is sized and ready to hand to capital authorization.")
    for _, r in queue.head(5).iterrows():
        st.markdown(
            f"**{_loss_well(r['well_id'])} — {r['cause']}** · "
            f"{r['suggested_action']} · recover "
            f"**{r['recoverable_bbl']:,.0f} bbl (${r['recoverable_usd']:,.0f})**, "
            f"~{r['mttr_days']:.1f}-day restore")
    st.caption("These well ids belong to the monthly loss-accounting fleet and are "
               "NOT the surveillance wells that share the same id — the queue "
               "quantifies and sizes the job; authorization on the Action Chain "
               "applies to surveillance-fleet wells only.")
    # The promised Quantify → Authorize handoff, one click — a LINK only, never a
    # well jump: carrying a loss-book id into the chain would BE the fake join.
    c.next_step("Action Chain",
                "→ Authorize a surveillance-fleet well (Action Chain)")
    st.caption("Program-level capital allocation lives in the **Capital Desk** "
               "product (sidebar switcher).")

    theme.references(["npv", "pareto"])
