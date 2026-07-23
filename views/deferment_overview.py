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

    # ---- display conventions (PE feedback OC8 + OC9): bbl-first + gross/net ----
    uc1, uc2 = st.columns([2, 2])
    with uc1:
        units = st.radio("Show deferment in", ["Barrels", "Dollars"],
                         key="def_units", horizontal=True,
                         help="Barrels: the operational base-management read "
                              "(default). Dollars: the same figures at the deck "
                              "oil price.")
    with uc2:
        net_view = c.gross_net_toggle()
    in_bbl = units == "Barrels"
    if net_view:
        daily = _net_daily(daily)
        st.caption("**NET view:** every volume and dollar below is net entitlement — "
                   "each well's barrels × its OWN NRI (registry default, editable on "
                   "Sources & BYOD). Ratios (efficiency, % deferred) are unchanged."
                   + (" Per-well NRI on a non-synthetic source is illustrative "
                      "registry data." if is_real else ""))

    k = A.fleet_kpis(daily, price)
    if not k:
        pt.empty_state("No production records in the active source.")
        return
    rec = A.recovery_opportunity(daily)
    pareto = A.pareto_by_cause(daily)

    if in_bbl:
        deferred_kpi = {
            "label": "Deferred", "value": f"{k['deferred_bbl']:,.0f} bbl",
            "delta": f"{k['pct_deferred']:.1f}% of potential",
            "delta_color": "inverse",
            "help": f"≈ ${k['deferred_usd']:,.0f} at the deck price."}
        rec_kpi = {
            "label": "Recoverable Opportunity",
            "value": f"{float(pareto.loc[pareto['recoverable'], 'deferred_bbl'].sum()) if len(pareto) else 0.0:,.0f} bbl",
            "help": "Recoverable-cause barrels — excludes planned + reservoir "
                    f"losses. ≈ ${rec['recoverable_usd']:,.0f} at the deck price."}
    else:
        deferred_kpi = {
            "label": "Deferred", "value": f"${k['deferred_usd']:,.0f}",
            "delta": f"{k['pct_deferred']:.1f}% of potential",
            "delta_color": "inverse",
            "help": f"{k['deferred_bbl']:,.0f} bbl at the deck price."}
        rec_kpi = {
            "label": "Recoverable Opportunity",
            "value": f"${rec['recoverable_usd']:,.0f}",
            "help": "Excludes planned + reservoir losses (you can't get those "
                    "barrels back)."}
    kpis = [
        {"label": "Production Efficiency", "value": f"{k['uptime_pct']:.1f}%",
         "help": "Actual ÷ potential over the period."},
        deferred_kpi,
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
            rec_kpi,
            {"label": "Reason-Code Capture",
             "value": f"{k['capture_rate_pct']:.0f}%",
             "delta": "coding gap" if k["capture_rate_pct"] < 90 else "good",
             "delta_color": "inverse" if k["capture_rate_pct"] < 90 else "off"},
        ]
    pt.kpi_row(kpis)
    period = _period_label(daily)
    conv = ("net entitlement (per-well NRI)" if net_view
            else "gross (8/8 working interest)")
    st.caption(f"Period: **{period}** · {k['n_wells']} wells · all deferment figures "
               f"below are over this period, shown {conv} at the deck oil price — "
               "the base-management book; certified net-to-operator economics live "
               "on the Optimization Board / Action Chain.")

    if not is_real:
        _deferment_buckets(A, daily, price, in_bbl)

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
    if not in_bbl:  # dollars mode leads with the $ column
        disp = disp[["Well", "Deferred $", "Deferred bbl", "Dominant Cause", "Uptime"]]
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


def _net_daily(daily):
    """View-layer NET entitlement copy of the daily deferment table: every volume /
    dollar column scaled by each row's well NRI (session override → registry
    default). Ratios (uptime, % deferred) are unchanged. The vendored analytics are
    untouched — they just receive net-entitlement inputs."""
    out = daily.copy()
    factor = out["well_id"].astype(str).map(
        c.nri_map(out["well_id"].astype(str).unique()))
    for col in ("total_def", "deferred_usd", "potential", "potential_vol",
                "bopd", "actual_vol"):
        if col in out.columns:
            out[col] = out[col] * factor
    return out


def _deferment_buckets(A, daily, price: float, in_bbl: bool) -> None:
    """Deferred bbl/$ bucketed into operational categories (artificial lift, surface
    facility, power, gathering, wellbore, planned, weather, reservoir) — where the
    downtime concentrates and how much is genuinely recoverable. ``in_bbl`` switches
    the whole panel between barrels (default, per PE feedback) and dollars."""
    pareto = A.pareto_by_cause(daily)
    pt.section("Deferment Buckets by Category",
               "Every lost barrel bucketed into its operational category — where the "
               "downtime concentrates, and how much you can actually get back.")
    if not len(pareto):
        st.caption("No classified deferment in the period.")
        return
    val_col = "deferred_bbl" if in_bbl else "deferred_usd"
    fmt = (lambda v: f"{v:,.0f} bbl") if in_bbl else (lambda v: f"${v:,.0f}")
    colors = [theme.BLUE if r else theme.GREY for r in pareto["recoverable"]]
    bf = go.Figure(go.Bar(
        x=pareto[val_col], y=pareto["label"], orientation="h",
        marker_color=colors,
        text=[f"{fmt(v)} · {p:.0f}%"
              for v, p in zip(pareto[val_col], pareto["pct_of_total"])],
        textposition="auto",
        hovertemplate=("<b>%{y}</b><br>%{x:,.0f} bbl<extra></extra>" if in_bbl
                       else "<b>%{y}</b><br>$%{x:,.0f}<extra></extra>")))
    bf.update_layout(
        xaxis_title=("Deferred bbl (at the displayed interest convention)" if in_bbl
                     else "Deferred $ (at deck price, displayed interest convention)"),
        yaxis_title="")
    st.plotly_chart(
        theme.style_fig(bf, height=max(260, 34 * len(pareto) + 70), legend=False),
        width="stretch")
    st.caption("Blue = recoverable by operator action · grey = planned / reservoir / "
               "uncaptured (you can't get those barrels back).")

    recoverable = float(pareto.loc[pareto["recoverable"], val_col].sum())
    planned_res = float(pareto.loc[~pareto["recoverable"]
                                   & (pareto["reason_key"] != "unclassified"),
                                   val_col].sum())
    uncaptured = float(pareto.loc[pareto["reason_key"] == "unclassified",
                                  val_col].sum())
    m = st.columns(3)
    m[0].metric("Recoverable", fmt(recoverable),
                help="Operator-addressable categories — the real recovery target "
                     "(excludes planned, reservoir, and uncaptured).")
    m[1].metric("Planned / Reservoir", fmt(planned_res),
                help="Expected work or physics-driven decline — not an opportunity.")
    m[2].metric("Uncaptured (coding gap)", fmt(uncaptured),
                help="Deferment with no operator reason code — a data-quality gap to "
                     "close, NOT a root cause. It is excluded from the recovery "
                     "target rather than counted as a category to 'fix'.")
