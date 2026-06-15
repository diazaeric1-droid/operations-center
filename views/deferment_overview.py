"""Loss Accounting · Deferment Overview — potential vs actual, downtime vs
underperformance. Ported from deferment-iq's demo (v0.5.1); defaults to the
synthetic reason-coded monthly book (the real Colorado ECMC extract is a
bring-your-own reference on Sources & BYOD, not the default).
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
    period = _period_label(daily)
    st.caption(f"Period: **{period}** · {k['n_wells']} wells · all deferment figures "
               "below are over this period. Deferment $ is gross (8/8 working "
               "interest) at the deck oil price — the standard base-management book; "
               "net-to-operator economics live on the Triage Board / Action Chain.")

    if not is_real:
        _deferment_buckets(A, daily, price)

    left, right = st.columns(2)
    with left:
        pt.section("Deferment Waterfall (bbl)",
                   "Gross potential → barrels lost to each cause → actual produced.")
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

    pt.section("Worst-Offender Wells",
               "The wells bleeding the most barrels — each tagged with the dominant "
               "cause carrying most of its deferred dollars.")
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
    else:
        st.caption("Ranked by deferred barrels/$ (potential vs. actual). The "
                   "**Dominant Cause** is the reason code carrying the most of each "
                   "well's deferred dollars — the single thing to fix first on that "
                   "well.")

    theme.references(["deferment", "arps"])


def _period_label(daily) -> str:
    """Human period span of the loss-accounting book (so KPIs aren't an unlabeled
    multi-period aggregate)."""
    try:
        import pandas as pd
        d = pd.to_datetime(daily["date"])
        lo, hi = d.min(), d.max()
        months = max(1, round((hi - lo).days / 30.44))
        return f"{lo.date():%b %Y} – {hi.date():%b %Y} ({months} mo)"
    except Exception:  # noqa: BLE001
        return "full available history"


def _deferment_buckets(A, daily, price: float) -> None:
    """Deferred $ bucketed into operational categories (artificial lift, surface
    facility, power, gathering, wellbore, planned, weather, reservoir) — where the
    downtime concentrates and how much is genuinely recoverable."""
    pareto = A.pareto_by_cause(daily)
    pt.section("Deferment Buckets by Category",
               "Every lost barrel bucketed into its operational category — where the "
               "downtime concentrates, and how much you can actually get back.")
    if not len(pareto):
        st.caption("No classified deferment in the period.")
        return
    colors = [theme.BLUE if r else theme.GREY for r in pareto["recoverable"]]
    bf = go.Figure(go.Bar(
        x=pareto["deferred_usd"], y=pareto["label"], orientation="h",
        marker_color=colors,
        text=[f"${v:,.0f} · {p:.0f}%"
              for v, p in zip(pareto["deferred_usd"], pareto["pct_of_total"])],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>"))
    bf.update_layout(xaxis_title="Deferred $ (gross, at deck price)", yaxis_title="")
    st.plotly_chart(
        theme.style_fig(bf, height=max(260, 34 * len(pareto) + 70), legend=False),
        width="stretch")
    st.caption("Blue = recoverable by operator action · grey = planned / reservoir / "
               "uncaptured (you can't get those barrels back).")

    recoverable = float(pareto.loc[pareto["recoverable"], "deferred_usd"].sum())
    planned_res = float(pareto.loc[~pareto["recoverable"]
                                   & (pareto["reason_key"] != "unclassified"),
                                   "deferred_usd"].sum())
    uncaptured = float(pareto.loc[pareto["reason_key"] == "unclassified",
                                  "deferred_usd"].sum())
    m = st.columns(3)
    m[0].metric("Recoverable", f"${recoverable:,.0f}",
                help="Operator-addressable categories — the real recovery target "
                     "(excludes planned, reservoir, and uncaptured).")
    m[1].metric("Planned / Reservoir", f"${planned_res:,.0f}",
                help="Expected work or physics-driven decline — not an opportunity.")
    m[2].metric("Uncaptured (coding gap)", f"${uncaptured:,.0f}",
                help="Deferment with no operator reason code — a data-quality gap to "
                     "close, NOT a root cause. It is excluded from the recovery "
                     "target rather than counted as a category to 'fix'.")
