"""Loss Accounting · Deferment Overview — potential vs actual, downtime vs
underperformance. Ported from deferment-iq's demo (v0.5.1); defaults to REAL
Colorado ECMC monthly records.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

import product_theme as pt
import theme

from views import _common as c


def render() -> None:
    c.ensure_state()
    price, _nri, _disc = c.deck()

    pt.masthead("ops", "Deferment Overview",
                "Where the barrels are going on the monthly book: potential vs "
                "actual, split into downtime vs underperformance.")

    source = c.loss_source_control()
    is_real = c.loss_is_real(source)
    pt.context_bar([
        ("Loss-accounting source", c.loss_context(source)),
        ("Deck", c.deck_label()),
        ("Cadence", "monthly (real public records)" if is_real
         else "daily (modeled fleet)"),
    ])
    theme.data_badge(*c.loss_badge(source))
    st.caption("This page runs on its **own** dataset — the monthly loss-accounting "
               "book — not the daily SCADA fleet the Today/Well File pages use. "
               "No join is faked between them (see Sources & BYOD).")

    import core
    fleet, evc, daily = c.deferment_data(c.loss_source_token(source), price)
    A = core.deferment_analytics
    k = A.fleet_kpis(daily, price)
    if not k:
        pt.empty_state("No production records in the active source.")
        return
    rec = A.recovery_opportunity(daily)

    kpis = [
        {"label": "Production Efficiency", "value": f"{k['uptime_pct']:.1f}%",
         "help": "Actual ÷ potential over the period."},
        {"label": "Deferred", "value": f"${k['deferred_usd']:,.0f}",
         "delta": f"{k['pct_deferred']:.1f}% of potential",
         "delta_color": "inverse"},
        {"label": "Deferred Rate", "value": f"{k['deferred_bopd_avg']:,.0f} BOPD",
         "help": "Deferred volume ÷ calendar days in the period."},
    ]
    if is_real:
        kpis += [
            {"label": "Recoverable Opportunity", "value": "N/A",
             "help": "Needs reason codes — not in public data."},
            {"label": "Reason-Code Capture", "value": "N/A",
             "help": "Public monthly filings carry no reason codes — cause "
                     "attribution is N/A. The deferment QUANTITY is real."},
        ]
    else:
        kpis += [
            {"label": "Recoverable Opportunity",
             "value": f"${rec['recoverable_usd']:,.0f}",
             "help": "Excludes planned + reservoir losses (you can't get those "
                     "barrels back)."},
            {"label": "Reason-Code Capture",
             "value": f"{k['capture_rate_pct']:.0f}%",
             "delta": "coding gap" if k["capture_rate_pct"] < 90 else "good",
             "delta_color": "inverse" if k["capture_rate_pct"] < 90 else "off"},
        ]
    pt.kpi_row(kpis)
    if is_real and source == core.DEF_SRC_REAL_CO:
        st.caption(f"Real-fleet anchor: {k['n_wells']} DJ Basin wells, "
                   f"**{k['pct_deferred']:.1f}% of potential deferred** over the "
                   "period — computed from public days-produced records, not modeled.")

    left, right = st.columns(2)
    with left:
        pt.section("Deferment Waterfall (bbl)",
                   "Potential → downtime → underperformance → actual.")
        wf = A.waterfall(daily)
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(wf) - 2) + ["total"],
            x=[s["label"] for s in wf], y=[s["value"] for s in wf],
            connector={"line": {"color": theme.GREY}},
            decreasing={"marker": {"color": theme.RED}},
            increasing={"marker": {"color": theme.BLUE}},
            totals={"marker": {"color": theme.NAVY}}))
        st.plotly_chart(theme.style_fig(fig, height=380), width="stretch")
        theme.source_note(
            "Potential from full-uptime months (P75, decline-aware); deferred = "
            "potential − actual, bridged potential → downtime → underperformance → "
            "actual, in bbl.")
        if is_real:
            st.caption("Real data: downtime comes from days-produced; the per-cause "
                       "split needs an operator's coded event log (see Causes & "
                       "Pareto).")
    with right:
        pt.section("Deferment Trend (Weekly bbl)")
        tr = A.deferment_trend(daily, "W")
        tf = go.Figure(go.Scatter(x=tr["date"], y=tr["deferred_bbl"], fill="tozeroy",
                                  line=dict(color=theme.RED)))
        st.plotly_chart(theme.style_fig(tf, height=380, legend=False),
                        width="stretch")
        theme.source_note("Deferred barrels per week = Σ (potential − actual) over "
                          "the wells in the period.")

    pt.section("Worst-Offender Wells")
    top = A.top_wells(daily, 10)
    disp = top.copy()
    disp["deferred_usd"] = disp["deferred_usd"].map(lambda v: f"${v:,.0f}")
    disp["deferred_bbl"] = disp["deferred_bbl"].map(lambda v: f"{v:,.0f}")
    disp["uptime_pct"] = disp["uptime_pct"].map(lambda v: f"{v:.0f}%")
    if is_real:
        disp["top_cause"] = "N/A (uncoded)"
    disp.columns = ["Well", "Deferred bbl", "Deferred $", "Dominant Cause", "Uptime"]
    st.dataframe(disp, width="stretch", hide_index=True)
    st.download_button("Download deferment summary (CSV)",
                       data=top.to_csv(index=False),
                       file_name="ops_deferment_summary.csv", mime="text/csv")
    if is_real:
        st.caption("Ranked by **real** deferred barrels/$ (potential vs. actual). "
                   "Dominant cause is N/A — no public reason codes.")

    theme.references(["deferment", "arps"])
