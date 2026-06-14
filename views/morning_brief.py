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
    price, nri, _disc = c.deck()

    pt.masthead("ops", "Morning Brief",
                "The overnight scan: deterministic detectors, money-first ranking, "
                "wells down, production divergences, and the 6:30am brief.")

    token = c.scada_token()
    fleet = c.fleet_for_token(token)
    is_upload = token != c.DISK_TOKEN
    as_of = c.fleet_as_of(fleet)
    pt.context_bar([
        ("Surveillance fleet", c.scada_source_label(token)),
        ("As of", as_of),
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
    div = core.production_divergence_summary(fleet, anomalies)
    net_deferred = sum(float(getattr(a, "deferred_bopd", 0.0) or 0.0)
                       for a in active) * price * nri

    sev = pd.Series([a.severity for a in active]).value_counts() if active else {}
    pt.kpi_row([
        {"label": "Wells Scanned", "value": f"{summary['well_count']}"},
        {"label": "Active Anomalies", "value": f"{len(active)}",
         "delta": f"{int(sev.get('HIGH', 0))} HIGH" if active else None,
         "delta_color": "inverse" if active else "off"},
        {"label": "Wells Down", "value": f"{div['n_down']}",
         "delta_color": "off",
         "help": "Wells at ≈0 oil production on the latest day (full shut-in / "
                 "outage vs the well's own recent baseline)."},
        {"label": "Production Divergences", "value": f"{div['n_divergences']}",
         "delta": (f"{div['divergence_bopd']:,.0f} bopd"
                   if div["n_divergences"] else None),
         "delta_color": "inverse" if div["n_divergences"] else "off",
         "help": "Wells producing materially below their decline-expected rate "
                 "(the digest's decline-aware rate-loss detector)."},
        {"label": "Deferred at Risk (net)", "value": f"${net_deferred:,.0f}/day",
         "delta_color": "inverse",
         "help": "Active anomalies' deferred barrels × deck oil price × NRI "
                 "(net-to-operator)."},
    ])
    fs = st.columns(4)
    fs[0].metric("Fleet Oil Rate", f"{summary['total_bopd']:,.0f} BOPD")
    fs[1].metric("Fleet Gross Fluid", f"{summary['total_bfpd']:,.0f} BFPD")
    fs[2].metric("Water Cut", f"{summary['water_cut_pct']:.0f}%")
    fs[3].metric("Avg Runtime", f"{summary['avg_runtime_pct']:.0f}%")

    pt.section("Wells Down & Production Divergences",
               "Who is OFF, and who is producing below decline — the first thing a "
               "foreman triages at 6:30am.")
    dcol, vcol = st.columns(2)
    with dcol:
        if div["down"]:
            dd = pd.DataFrame(div["down"])[["well_id", "last_bopd", "baseline_bopd"]]
            dd.columns = ["Well", "Latest BOPD", "Baseline BOPD"]
            st.dataframe(dd, width="stretch", hide_index=True, height=240)
        else:
            st.success("No wells down — every well is producing on the latest day.")
    with vcol:
        if div["divergences"]:
            vv = pd.DataFrame([
                {"Well": a.well_id, "Category": a.category,
                 "Deferred bopd": round(getattr(a, "deferred_bopd", 0.0), 1),
                 "Deferred $/day (net)": round(
                     float(getattr(a, "deferred_bopd", 0.0) or 0.0) * price * nri, 0)}
                for a in div["divergences"]])
            st.dataframe(vv, width="stretch", hide_index=True, height=240,
                         column_config={
                             "Deferred $/day (net)":
                                 st.column_config.NumberColumn(format="$%d")})
        else:
            st.success("No production divergences — every well is on its decline curve.")

    pt.section("Ranked Anomalies",
               "Money-first: deferred $/day first, then severity. Pure-risk flags "
               "(nothing lost yet) rank after quantified losses.")
    if not anomalies:
        pt.empty_state("No anomalies on the latest scan.")
    else:
        offenders = [a for a in active if a.deferred_usd_per_day > 0][:15]
        if offenders:
            offenders = list(reversed(offenders))
            fig = go.Figure(go.Bar(
                x=[float(a.deferred_bopd) * price * nri for a in offenders],
                y=[a.well_id for a in offenders],
                orientation="h", marker_color=theme.RED,
                text=[f"{a.category}" for a in offenders], textposition="auto"))
            fig.update_layout(title="Top Deferred-$ Offenders (net $/day)",
                              xaxis_title="Deferred $/day (net to operator)")
            st.plotly_chart(
                theme.style_fig(fig, height=max(300, 26 * len(offenders) + 90),
                                legend=False), width="stretch")
            theme.source_note(
                "Deferred $/day (net) = deferred barrels × deck oil price × NRI. "
                "Deferred barrels come from the digest's decline-aware detector "
                "(decline-expected rate − actual), money-first.")

        table = pd.DataFrame([
            {"Well": a.well_id, "Severity": a.severity, "Category": a.category,
             "Deferred $/day (net)": (f"${float(a.deferred_bopd) * price * nri:,.0f}"
                                      if a.deferred_bopd else "—"),
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
    # Date the brief to the data's as-of day (not today's wall clock), then append
    # the Production Divergences & Wells Down section so the page and the emailed/
    # downloaded report carry the same content.
    brief_md = core.digest_brief.render_brief_markdown(
        summary, anomalies, brief_date=as_of, events=events)
    brief_md = brief_md + "\n\n" + core._divergence_section_md(div, price, nri)
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
                narr = core.digest_brief.write_brief(
                    summary, anomalies, client=Anthropic(api_key=llm_key),
                    events=events)
                st.session_state["brief_md_llm"] = (
                    narr + "\n\n" + core._divergence_section_md(div, price, nri))
        except Exception as exc:  # noqa: BLE001 — bad key / network: stay deterministic
            st.warning(f"Narrated brief unavailable ({type(exc).__name__}); "
                       "showing the deterministic brief.")
    narrated = st.session_state.get("brief_md_llm")
    final_brief = narrated or brief_md
    with st.container(border=True):
        st.markdown(final_brief)
    if narrated:
        st.caption("LLM-narrated (session only). The deterministic detectors and "
                   "numbers underneath are unchanged.")
    st.download_button("Download brief (markdown)", data=final_brief,
                       file_name=f"ops_morning_brief_{as_of}.md",
                       mime="text/markdown")

    _email_brief_ui(final_brief, f"Operations Center — Morning Brief {as_of}")

    theme.references(["arps", "deferment"])


def _email_brief_ui(brief_md: str, default_subject: str) -> None:
    """Send the brief now via SMTP (session-only creds), and point at the daily
    GitHub Action for automated 6:30am sends."""
    with st.expander("📧 Email this brief / set up the daily morning send"):
        st.caption("Send it now, or wire the bundled GitHub Action to email it every "
                   "morning automatically. SMTP credentials are used only for this "
                   "send and are never stored server-side.")
        col1, col2 = st.columns(2)
        host = col1.text_input("SMTP host", key="smtp_host",
                               placeholder="smtp.gmail.com")
        port = col2.number_input("Port", value=587, step=1, key="smtp_port",
                                 help="587 = STARTTLS (typical), 465 = SSL.")
        user = col1.text_input("SMTP username", key="smtp_user")
        pw = col2.text_input("SMTP password / app-password", type="password",
                             key="smtp_pass")
        sender = col1.text_input("From", key="smtp_from",
                                 help="Defaults to the SMTP username if blank.")
        to = col2.text_input("To (comma-separated)", key="smtp_to")
        subject = st.text_input("Subject", value=default_subject, key="smtp_subject")
        if st.button("Send brief now", type="primary"):
            missing = [n for n, v in (("SMTP host", host), ("username", user),
                                      ("password", pw), ("recipient(s)", to))
                       if not str(v).strip()]
            if missing:
                st.warning("Fill in: " + ", ".join(missing) + ".")
            else:
                try:
                    import notify
                    notify.send_email(
                        host=host.strip(), port=int(port), username=user.strip(),
                        password=pw, sender=(sender.strip() or user.strip()),
                        recipients=to.split(","), subject=subject,
                        markdown_body=brief_md, use_tls=int(port) != 465)
                    st.success(f"Brief sent to {to.strip()}.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Send failed — {type(exc).__name__}: {exc}")
        st.markdown(
            "**Automated daily send:** the repo ships "
            "`.github/workflows/daily-brief.yml`, which runs "
            "`scripts/daily_brief_email.py` on a cron (12:30 UTC ≈ 6:30am US "
            "Central). Add `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, "
            "`BRIEF_FROM`, and `BRIEF_TO` as GitHub Action secrets to turn it on — "
            "no code changes needed.")
