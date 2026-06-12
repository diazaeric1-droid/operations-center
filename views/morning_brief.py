"""Today · Morning Brief — overnight scan, ranked anomalies, deterministic brief.

Ported from daily-production-digest's demo (v0.6.3): the same deterministic
detectors and brief writer, run on the active SCADA fleet (synthetic disk fleet,
or the session upload from the Data page).
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

    pt.masthead("ops", "Morning Brief",
                "The overnight scan: deterministic detectors, money-first ranking, "
                "and the brief a Staff PE would write at 6:30am.")

    token = c.scada_token()
    fleet = c.fleet_for_token(token)
    is_upload = token != c.DISK_TOKEN
    pt.context_bar([
        ("Surveillance fleet", c.scada_source_label(token)),
        ("As of", c.fleet_as_of(fleet)),
        ("Deck", c.deck_label()),
    ])
    if is_upload:
        theme.data_badge("real", "Your uploaded fleet SCADA — parsed in memory for "
                                 "this session only, nothing stored server-side.")
    else:
        theme.data_badge("synthetic", "Modeled daily SCADA fleet with known ground "
                                      "truth — public production is monthly, not daily.")

    import core
    anomalies = c.scan(token, price)
    active = [a for a in anomalies if not a.acknowledged]
    acked = [a for a in anomalies if a.acknowledged]
    summary = core.digest_loader.fleet_summary(fleet)

    sev = pd.Series([a.severity for a in active]).value_counts() if active else {}
    pt.kpi_row([
        {"label": "Wells", "value": f"{summary['well_count']}"},
        {"label": "Fleet BOPD", "value": f"{summary['total_bopd']:,.0f}"},
        {"label": "Active Anomalies", "value": f"{len(active)}",
         "delta": f"{int(sev.get('HIGH', 0))} HIGH" if active else None,
         "delta_color": "inverse" if active else "off"},
        {"label": "Deferred at Risk",
         "value": f"${sum(a.deferred_usd_per_day for a in active):,.0f}/day",
         "help": "Σ of active anomalies' deferred $/day at the deck oil price "
                 "(gross; the Triage Board nets revenue by NRI)."},
    ])

    pt.section("Ranked Anomalies",
               "Money-first: deferred $/day first, then severity. Pure-risk flags "
               "(nothing lost yet) rank after quantified losses.")
    if not anomalies:
        pt.empty_state("No anomalies on the latest scan.")
    else:
        offenders = [a for a in active if a.deferred_usd_per_day > 0][:10]
        if offenders:
            offenders = list(reversed(offenders))
            fig = go.Figure(go.Bar(
                x=[a.deferred_usd_per_day for a in offenders],
                y=[a.well_id for a in offenders],
                orientation="h", marker_color=theme.RED))
            fig.update_layout(title="Top Deferred-$ Offenders ($/day)",
                              xaxis_title="Deferred $/day")
            st.plotly_chart(theme.style_fig(fig, height=300, legend=False),
                            width="stretch")
            theme.source_note(
                "Deferred $/day = (decline-expected rate − actual) × deck oil price, "
                "per the digest's deterministic detectors.")

        table = pd.DataFrame([
            {"Well": a.well_id, "Severity": a.severity, "Category": a.category,
             "Deferred $/day": (f"${a.deferred_usd_per_day:,.0f}"
                                if a.deferred_usd_per_day else "—"),
             "Headline": a.headline,
             "Recommended Action": a.recommended_action}
            for a in active
        ])
        if table.empty:
            st.caption("Every anomaly on the latest scan is acknowledged/suppressed.")
        else:
            st.dataframe(table, width="stretch", hide_index=True)
            st.download_button("Download anomaly list (CSV)",
                               data=table.to_csv(index=False),
                               file_name="ops_anomalies.csv", mime="text/csv")
        theme.source_note(
            "Anomalies flagged by deterministic robust statistics on each well's own "
            "recent baseline: a median/MAD robust z-score plus a decline-aware "
            "rate-drop vs. the expected Arps rate — no fixed thresholds, so one bad "
            "day can't inflate the baseline.")

    pt.section("Acknowledged / Suppressed",
               "Known or planned events (acknowledged.yml) — kept out of the active "
               "list so a planned workover doesn't re-fire HIGH every morning.")
    if acked:
        ack_df = pd.DataFrame([
            {"Well": a.well_id, "Severity": a.severity, "Category": a.category,
             "Headline": a.headline} for a in acked])
        with st.expander(f"{len(acked)} acknowledged anomaly(ies)"):
            st.dataframe(ack_df, width="stretch", hide_index=True)
    else:
        st.caption("Nothing is acknowledged/suppressed on the current scan.")

    pt.section("The Brief",
               "Deterministic by default — same data an LLM would narrate, just "
               "templated. Add a key in the sidebar for the narrated version.")
    events = c.replay_events(token, price, False)
    brief_md = core.digest_brief.render_brief_markdown(summary, anomalies,
                                                       events=events)
    llm_key = st.session_state.get("anthropic_key", "")
    col_a, col_b = st.columns([1, 3])
    with col_a:
        want_llm = st.button("Write narrated brief", type="primary",
                             disabled=not llm_key,
                             help="Uses your session-only Anthropic key (sidebar). "
                                  "Without a key, the deterministic brief below is "
                                  "the product.")
    if want_llm and llm_key:
        try:
            from anthropic import Anthropic
            with st.spinner("Writing the narrated brief…"):
                st.session_state["brief_md_llm"] = core.digest_brief.write_brief(
                    summary, anomalies, client=Anthropic(api_key=llm_key),
                    events=events)
        except Exception as exc:  # noqa: BLE001 — bad key / network: stay deterministic
            st.warning(f"Narrated brief unavailable ({type(exc).__name__}); "
                       "showing the deterministic brief.")
    narrated = st.session_state.get("brief_md_llm")
    with st.container(border=True):
        st.markdown(narrated or brief_md)
    if narrated:
        st.caption("LLM-narrated (session only). The deterministic detectors and "
                   "numbers underneath are unchanged.")
    st.download_button("Download brief (markdown)", data=narrated or brief_md,
                       file_name="ops_morning_brief.md", mime="text/markdown")

    theme.references(["arps", "deferment"])
