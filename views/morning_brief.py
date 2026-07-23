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
    ew_md, ew_deep = _early_warning(token)   # deep-drift catches (optional torch)

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

    # ---- one list vs the classic panels (PE feedback OC4) ----------------------
    events = c.replay_events(token, price, False)
    vc1, vc2 = st.columns([2, 2])
    with vc1:
        st.radio("View", ["Unified list", "Detailed panels"], key="brief_view",
                 horizontal=True,
                 help="Unified list (default): ONE ranked list — new, ongoing, and "
                      "just-resolved events plus scan-only anomalies — ordered by "
                      "BO/day impact. Detailed panels: the classic three-panel "
                      "layout (wells down / divergences, ranked anomalies, "
                      "acknowledged).")
    with vc2:
        net_view = c.gross_net_toggle()
    if st.session_state.get("brief_view", "Unified list") == "Detailed panels":
        _detailed_panels(div, active, acked, anomalies, ew_deep, price, nri)
    else:
        _unified_list(events, active, net_view, price)

    _the_brief(token, price, nri, as_of, summary, anomalies, div, events, ew_md)

    theme.references(["arps", "deferment"])


def _unified_list(events, active, net_view: bool, price: float) -> None:
    """OC4: one ranked list — events (NEW / ONGOING / RESOLVED) + scan-only
    anomalies, ordered by today's BO/day impact (net of per-well NRI when the NET
    toggle is on). Select a row to open the well on Surveillance."""
    pt.section("Unified Priority List",
               "Everything that needs eyes, in ONE list: new + ongoing + "
               "just-resolved events and scan-only anomalies, ordered by today's "
               "BO/day impact.")
    nmap = c.nri_map({str(e.well_id) for e in (events or [])}
                     | {str(a.well_id) for a in (active or [])})
    df = c.unified_brief_frame(events, active, nmap, net_view, price)
    if df.empty:
        st.success("Nothing to chase — no open events or active anomalies on the "
                   "latest scan.")
        return
    counts = df["status"].value_counts()
    st.markdown(
        pt.pill(f"{int(counts.get('NEW', 0))} new",
                "bad" if counts.get("NEW", 0) else "ok") + " "
        + pt.pill(f"{int(counts.get('ONGOING', 0))} ongoing",
                  "warn" if counts.get("ONGOING", 0) else "ok") + " "
        + pt.pill(f"{int(counts.get('RESOLVED', 0))} resolved", "ok"),
        unsafe_allow_html=True)
    badge = {"NEW": "🔴 NEW", "ONGOING": "🟠 ONGOING", "RESOLVED": "🟢 RESOLVED"}
    usd_lbl = "$ /day (net, well NRI)" if net_view else "$ /day (gross)"
    show = pd.DataFrame({
        "Well": df["well_id"],
        "Status": df["status"].map(badge),
        "Event / Category": df["kind"].str.replace("_", " "),
        "Days": df["days"].map(lambda v: "—" if pd.isna(v) else f"{int(v)}d"),
        "BO/day (gross)": df["gross_bopd"].map(lambda v: f"{v:,.1f}"),
        "BO/day (net, well NRI)": df["net_bopd"].map(lambda v: f"{v:,.1f}"),
        usd_lbl: df["usd_per_day"].map(lambda v: f"${v:,.0f}"),
        "Cum. deferred bbl": df["cum_bbl"].map(
            lambda v: "—" if pd.isna(v) else f"{v:,.0f}"),
    })
    ev = st.dataframe(show, width="stretch", hide_index=True,
                      on_select="rerun", selection_mode="single-row",
                      key="mb_unified")
    c.handle_row_jump(ev, df, "_mb_jump")
    theme.source_note(
        f"Ordered by {'NET' if net_view else 'GROSS'} BO/day (today's deferral; "
        "resolved events close out at 0 and fall to the bottom), status as the "
        "tiebreaker. NET = × each well's OWN NRI (registry default, editable on "
        "Sources & BYOD); GROSS is the digest's native 8/8 convention. Where a well "
        "has both an open event and a scan anomaly, the event's deferral is shown "
        "(the two windows differ slightly). 'Days' is the event's running duration — "
        "scan-only anomalies have no event yet ('—'). Select a row to open the well "
        "on Surveillance. The classic panels are under **View → Detailed panels**.")


def _detailed_panels(div, active, acked, anomalies, ew_deep, price: float,
                     nri: float) -> None:
    """The original three-panel layout (wells down / divergences, ranked anomalies,
    deep drift, acknowledged) — unchanged, behind the view toggle."""
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

    if ew_deep is not None and not ew_deep.empty:
        pt.section("Early Warning — Deep Drift",
                   "Wells the deep autoencoder flags as drifting that the rate-drop "
                   "alarm hasn't caught yet — slow degraders to review before they "
                   "trip a hard alarm. (Full leaderboard on Surveillance → Early "
                   "Warning · Deep AI.)")
        ewt = ew_deep.head(10)[["well", "driver", "score"]].copy()
        ewt["score"] = ewt["score"].map(lambda v: f"{v:.2f}")
        ewt.columns = ["Well", "Top drifting channel", "Drift score"]
        st.dataframe(ewt, width="stretch", hide_index=True)
        theme.source_note(
            "Deep early-warning flags from the LSTM autoencoder (trained on healthy "
            "wells only); 'deep-only' = the rate-drop alarm did not fire. Surfaced "
            "here so the daily brief catches slow drift automatically.")

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


def _the_brief(token: str, price: float, nri: float, as_of: str, summary,
               anomalies, div, events, ew_md: str) -> None:
    """'The Brief' + download + email — the product's output, rendered in BOTH
    views (outside the unified/detailed toggle). Body unchanged."""
    import core
    pt.section("The Brief",
               "Deterministic by default — same data an LLM would narrate, just "
               "templated. Add a key in the sidebar for the narrated version.")
    # Date the brief to the data's as-of day (not today's wall clock), then append
    # the Production Divergences & Wells Down section so the page and the emailed/
    # downloaded report carry the same content.
    brief_md = core.digest_brief.render_brief_markdown(
        summary, anomalies, brief_date=as_of, events=events)
    brief_md = brief_md + "\n\n" + core._divergence_section_md(div, price, nri)
    if ew_md:
        brief_md = brief_md + "\n\n" + ew_md
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
                    narr + "\n\n" + core._divergence_section_md(div, price, nri)
                    + (("\n\n" + ew_md) if ew_md else ""))
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


# ---- Early Warning · deep drift (optional torch) -----------------------------
# Surface the autoencoder's deep-only catches (drift the rate-drop alarm misses)
# in the daily brief, so slow degraders show up automatically. Scoring lives in
# dl/score.py (shared with Surveillance); torch is optional, so this is a silent
# no-op when the extras / trained model are absent — the brief stays clean.
def _early_warning_md(deep: pd.DataFrame) -> str:
    """Brief section listing the deep-only catches (empty string if none)."""
    if deep is None or deep.empty:
        return ""
    lines = [
        "## Early Warning — Deep Drift Detector",
        f"{len(deep)} well(s) are drifting from normal that the rate-drop alarm has "
        "**not** flagged yet — early degraders to review before they trip a hard alarm:",
    ]
    for _, r in deep.head(8).iterrows():
        lines.append(f"- **{r['well']}** — {r['driver']} drifting "
                     f"(drift score {r['score']:.2f})")
    return "\n".join(lines)


def _early_warning(token: str) -> tuple[str, "pd.DataFrame | None"]:
    """(brief markdown, deep-only frame) for the active fleet — ("", None) when the
    deep detector isn't available."""
    ew = c.early_warning_flags(token)
    if ew is None or ew.empty:
        return "", None
    deep = ew[ew["deep_only"]]
    return _early_warning_md(deep), deep


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
