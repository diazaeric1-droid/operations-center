"""Loss Accounting · Causes & Pareto — reason-code attribution and the $-Pareto.

Ported from deferment-iq's demo (v0.5.1). On real public monthly data the cause
is honestly N/A (no public reason codes); the synthetic reason-coded fleet powers
the full attribution + the classifier eval.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import product_theme as pt
import theme

from views import _common as c


def render() -> None:
    c.ensure_state()
    price, _nri, _disc = c.deck()

    pt.masthead("ops", "Causes & Pareto",
                "Every lost barrel tagged to a cause — the vital-few causes that "
                "drive most of the deferred dollars.")

    source = c.loss_source_control()
    is_real = c.loss_is_real(source)
    pt.context_bar([
        ("Loss-accounting source", c.loss_context(source)),
        ("Deck", c.deck_label()),
        ("Classifier", "deterministic rules (~92% on the eval set)"
         if not is_real else "N/A on public data — no reason codes"),
    ])
    theme.data_badge(*c.loss_badge(source))

    import core
    fleet, evc, daily = c.deferment_data(c.loss_source_token(source), price)
    A = core.deferment_analytics
    k = A.fleet_kpis(daily, price)
    if not k:
        pt.empty_state("No production records in the active source.")
        return
    pareto = A.pareto_by_cause(daily)

    pt.kpi_row([
        {"label": "Deferred $", "value": f"${k['deferred_usd']:,.0f}",
         "delta": f"{k['pct_deferred']:.1f}% of potential", "delta_color": "inverse"},
        {"label": "Reason-Code Capture",
         "value": "N/A" if is_real else f"{k['capture_rate_pct']:.0f}%",
         "help": ("Public monthly filings carry no reason codes." if is_real else
                  "Share of deferred barrels carrying a classified cause — "
                  "uncaptured deferment is a data-quality gap to close.")},
        {"label": "Causes in Play",
         "value": "N/A" if is_real else f"{len(pareto)}"},
    ])

    pt.section("Where the Barrels Go — $ by Cause")
    if is_real:
        st.info("**Cause attribution N/A** — public monthly filings carry no reason "
                "codes or operator cause notes. The deferment **quantity** is real "
                "(from days-produced); the per-cause $-Pareto needs an operator's "
                "coded event log. Switch the source above to **Synthetic "
                "(reason-coded demo)** to see the full attribution.")
    elif len(pareto):
        pf = go.Figure()
        pf.add_bar(x=pareto["label"], y=pareto["deferred_usd"], name="Deferred $",
                   marker_color=[theme.BLUE if r else theme.GREY
                                 for r in pareto["recoverable"]])
        pf.add_scatter(x=pareto["label"], y=pareto["cum_pct"], name="Cumulative %",
                       yaxis="y2", line=dict(color=theme.RED))
        pf.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100],
                                     title="cum %"))
        st.plotly_chart(theme.style_fig(pf, height=380), width="stretch")
        st.caption("Blue = recoverable · grey = planned/reservoir (not recoverable).")
        theme.source_note(
            "Deferred $ = deferred bbl × deck oil price, ranked Pareto by cause "
            "(vital-few first); cumulative % overlaid. Cause from the deterministic "
            "reason-code classifier over the operator's free-text notes.")
    else:
        pt.empty_state("No classified deferment in the current period.")

    col_l, col_r = st.columns(2)
    with col_l:
        pt.section("MTTR by Cause (days)",
                   "Mean time to restore — slow causes hold barrels hostage longer.")
        if is_real:
            st.info("MTTR needs a coded event log (start/end + cause) — N/A on "
                    "public monthly data.")
        else:
            m = A.mttr_by_cause(evc)
            if len(m):
                mm = m.copy()
                mm["mttr_days"] = mm["mttr_days"].map(lambda v: f"{v:.1f}")
                st.dataframe(
                    mm[["label", "n_events", "mttr_days", "total_event_days"]]
                    .rename(columns={"label": "Cause", "n_events": "Events",
                                     "mttr_days": "MTTR (d)",
                                     "total_event_days": "Down-Days"}),
                    width="stretch", hide_index=True)
            else:
                st.caption("No classified events in the period.")
    with col_r:
        pt.section("Classified Events",
                   "The operator notes the classifier read, with its call.")
        if is_real or not len(evc):
            st.caption("No coded event log on this source.")
        else:
            label_for = core.deferment_reasons.label_for
            ev = evc[["well_id", "start_date", "end_date", "note", "reason_key"]].copy()
            ev["reason_key"] = ev["reason_key"].map(label_for)
            ev.columns = ["Well", "Start", "End", "Operator Note", "Classified Cause"]
            st.dataframe(ev, width="stretch", hide_index=True, height=300)

    _eval_section(is_real)
    theme.references(["pareto", "deferment"])


def _eval_section(is_real: bool) -> None:
    """The classifier's honest eval vs ground-truth causes (committed summary)."""
    import core
    pt.section("Classifier Eval — vs Ground-Truth Causes",
               "The synthetic event log carries a ground-truth cause the classifier "
               "never sees; it is scored against it, and a CI gate fails under 80%.")
    if is_real:
        st.caption("The eval is measured on the **synthetic** reason-coded set (its "
                   "ground-truth labels) — there is no ground truth to score against "
                   "on public data.")
    eval_path = core.APP_DIRS["deferment"] / "evals" / "results" / "summary.json"
    try:
        res = json.loads(eval_path.read_text())
    except Exception:  # noqa: BLE001 — eval snapshot absent: never invent numbers
        st.caption("No committed eval summary found in the vendored component.")
        return
    e1, e2 = st.columns(2)
    e1.metric("Overall Accuracy", f"{res['accuracy'] * 100:.0f}%",
              f"{res['n']} events", delta_color="off")
    e2.metric("Classes", len(res["per_class"]))
    rows = [{"Cause": cause, "Precision": m["precision"], "Recall": m["recall"],
             "F1": m["f1"], "n": m["support"]}
            for cause, m in res["per_class"].items()]
    pc = pd.DataFrame(rows)
    for col in ("Precision", "Recall", "F1"):
        pc[col] = pc[col].map(lambda v: f"{v:.2f}" if isinstance(v, (int, float))
                              else "—")
    st.dataframe(pc, width="stretch", hide_index=True)
    st.caption("Residual misses are the deliberately vague notes (e.g. \"well down, "
               "see foreman\") — exactly where the optional LLM classifier earns "
               "its keep.")
