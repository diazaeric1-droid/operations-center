"""Today · Home — the 6:30am landing page for a production foreman."""
from __future__ import annotations

import streamlit as st

import product_theme as pt
import theme

from views import _common as c


def render() -> None:
    c.ensure_state()
    price, nri, _disc = c.deck()

    pt.masthead("ops", "Home",
                "What broke overnight, what it is costing, and what to do first.")

    token = c.scada_token()
    fleet = c.fleet_for_token(token)
    anomalies = c.scan(token, price)
    active = [a for a in anomalies if not a.acknowledged]
    # Net deferred (× NRI) so the headline matches the Triage Board / chain economics.
    net_deferred_usd = sum(float(getattr(a, "deferred_bopd", 0.0) or 0.0)
                           for a in active) * price * nri

    # Same gated tiers the Triage Board uses (real deferred + signal-gated), so Home's
    # "Top Opportunity" is exactly the Triage Board's #1.
    b = c.board_with_deferred(price, nri)
    opportunities, _watch, _stable = c.triage_tiers(b)

    pt.context_bar([
        ("Surveillance fleet", c.scada_source_label(token)),
        ("As of", c.fleet_as_of(fleet)),
        ("Deck", c.deck_label()),
        ("Loss accounting", c.loss_context(st.session_state["data_source"])),
    ])

    top_label, top_value = "none today", "—"
    if not opportunities.empty:
        top = opportunities.iloc[0]
        top_label = str(top["well_id"])
        top_value = f"${float(top['est_risked_npv']):,.0f}"
    pt.kpi_row([
        {"label": "Wells Scanned", "value": f"{core_fleet_size()}",
         "help": "Wells the digest scanned on the latest day."},
        {"label": "Open Alerts", "value": f"{len(active)}",
         "help": "Active anomalies on the latest scan (acknowledged events excluded)."},
        {"label": "Deferred $/day (net)", "value": f"${net_deferred_usd:,.0f}",
         "delta_color": "inverse",
         "help": "Active anomalies' deferred barrels × deck oil price × NRI "
                 "(net-to-operator — same convention as the Triage Board)."},
        {"label": f"Top Opportunity · {top_label}", "value": top_value,
         "help": "Largest VALUE-ACCRETIVE intervention on the Triage Board "
                 "(positive risk-weighted NPV). 'none today' means no intervention "
                 "currently clears its cost — the fleet is being held, not worked."},
    ])

    import core
    if core.risk_scoring_degraded():
        st.warning("⚠️ **ESP risk model unavailable** — fleet risk is showing the "
                   f"baseline {core.BASELINE_RISK_30D:.0%}; the Top Opportunity / "
                   "triage figures reflect deferred production only until the model "
                   "is restored (re-run bootstrap).")
    if token != c.DISK_TOKEN:
        st.caption("Note: Open Alerts and Deferred $/day reflect **your uploaded "
                   "fleet**, but Top Opportunity and the Triage Board still run on the "
                   "synthetic demo fleet (the triage ranking isn't wired to BYOD yet) "
                   "— so the 'authorize' step below may name a synthetic well.")

    _fleet_health(fleet, anomalies, b)

    events = c.replay_events(token, price, False)
    _what_broke_and_next(anomalies, fleet, opportunities, price, nri, events)

    pt.section("The Morning Loop",
               "Jump straight to the work — surveillance → brief → triage → "
               "loss accounting → action chain.")
    _loop_cards()

    with st.expander("Methods — two datasets, stated plainly"):
        st.markdown(
            "**Today + Well File** run on a synthetic daily SCADA fleet (modeled "
            "Permian, known ground truth — public production data is monthly, not "
            "daily). **Loss Accounting** runs on a synthetic, reason-coded monthly "
            "book with ground-truth causes (so cause attribution, MTTR, and the "
            "recovery queue all work). They are different datasets at different "
            "cadences; this console does not fake a join between them. Bring your own "
            "daily SCADA or monthly book on the **Sources & BYOD** page.")

    theme.references(["arps", "deferment", "npv"])


def _what_broke_and_next(anomalies, fleet, opportunities, price, nri, events) -> None:
    """The two questions a foreman opens the console for: what broke overnight, and
    what to do first. 'What Broke Overnight' is a true day-over-day DIFF off the event
    state machine — what's NEW, what RESOLVED, what's STILL down — not a stateless
    re-list of today's scan."""
    import core

    active = [a for a in anomalies if not a.acknowledged]
    div = core.production_divergence_summary(fleet, anomalies)
    evs = [e for e in events if not getattr(e, "acknowledged", False)]
    new = [e for e in evs if e.state == "NEW"]
    resolved = [e for e in evs if e.state == "RESOLVED"]
    ongoing = [e for e in evs if e.state == "ONGOING"]
    left, right = st.columns(2)
    with left:
        pt.section("What Broke Overnight",
                   "The day-over-day delta from the event state machine.")
        st.markdown(
            pt.pill(f"{len(new)} new", "bad" if new else "ok") + " "
            + pt.pill(f"{len(ongoing)} still ongoing", "warn" if ongoing else "ok")
            + " " + pt.pill(f"{len(resolved)} resolved", "ok"),
            unsafe_allow_html=True)
        if not new and not resolved and not ongoing:
            st.success("Nothing changed overnight — no new, ongoing, or just-resolved "
                       "events on the latest scan.")
        if new:
            st.markdown("**🔴 New today**")
            for e in sorted(new, key=lambda e: -float(getattr(e, "deferred_bopd", 0) or 0))[:4]:
                st.markdown(f"- **{e.well_id}** — {e.event_type}")
        if resolved:
            st.markdown("**🟢 Resolved**")
            st.caption(", ".join(f"{e.well_id} ({e.event_type})" for e in resolved[:6]))
        if ongoing:
            st.caption(f"Still ongoing: " + ", ".join(
                f"{e.well_id} · {e.duration_days}d" for e in
                sorted(ongoing, key=lambda e: -e.duration_days)[:5]))
        if div["n_down"]:
            st.caption("Down right now: "
                       + ", ".join(d["well_id"] for d in div["down"][:6]))
    with right:
        pt.section("What To Do First")
        steps = []
        if not opportunities.empty:
            t = opportunities.iloc[0]
            steps.append(
                f"**Authorize {t['well_id']}** — "
                f"{str(t['recommended_intervention']).replace('_', ' ')} "
                f"(risked NPV ${float(t['est_risked_npv']):,.0f}); build the AFE on "
                "the **Action Chain**.")
        if div["divergences"]:
            a = div["divergences"][0]
            net = float(getattr(a, "deferred_bopd", 0.0) or 0.0) * price * nri
            steps.append(f"**Chase {a.well_id}** — the biggest live leak "
                         f"(~${net:,.0f}/day net); details on the **Morning Brief**.")
        if div["n_down"]:
            steps.append(f"**Restore {div['down'][0]['well_id']}**"
                         + (f" + {div['n_down'] - 1} more" if div["n_down"] > 1 else "")
                         + " — zero production right now.")
        if not steps:
            steps.append("Hold — the fleet is on trend. Review the watch list on the "
                         "**Triage Board**.")
        for i, s in enumerate(steps[:3], 1):
            st.markdown(f"{i}. {s}")


def _loop_cards() -> None:
    """Boxed quick-links into the loop (cards pop more than a plain link row)."""
    import views
    cards = [
        ("Surveillance", "Fleet & per-well production · type-curve check"),
        ("Morning Brief", "Overnight scan — what broke, money-first"),
        ("Triage Board", "Fleet ranked by risked-NPV opportunity"),
        ("Deferment Overview", "Where the barrels go, by cause"),
        ("Action Chain", "Detect → predict → authorize an AFE"),
    ]
    cols = st.columns(len(cards))
    for col, (title, desc) in zip(cols, cards):
        page = views.PAGE_OBJECTS.get(title)
        with col, st.container(border=True):
            linked = False
            if page is not None:
                try:
                    st.page_link(page)
                    linked = True
                except Exception:  # noqa: BLE001 — outside a navigation context
                    linked = False
            if not linked:
                st.markdown(f"**{title}**")
            st.caption(desc)


def _fleet_health(fleet: dict, anomalies: list, board) -> None:
    """Quick-glance fleet status — a green/amber/red bar + counts + positive pills."""
    import core

    risk = ({} if board is None or board.empty
            else dict(zip(board["well_id"].astype(str),
                          board["failure_risk_30d"].astype(float))))
    h = core.fleet_health_summary(fleet, anomalies, risk_by_well=risk)
    total = max(h["total"], 1)

    pt.section("Fleet Health at a Glance",
               "Every well classified green / amber / red on live production and "
               "relative failure-signature risk — the 5-second read.")
    g = 100.0 * h["healthy"] / total
    a = 100.0 * h["watch"] / total
    r = 100.0 * h["impaired"] / total
    st.markdown(
        '<div style="display:flex;height:14px;border-radius:7px;overflow:hidden;'
        'border:1px solid #e5e7eb;margin:0.1rem 0 0.55rem 0">'
        f'<div style="width:{g:.1f}%;background:#1b7a3d" title="healthy"></div>'
        f'<div style="width:{a:.1f}%;background:#d9a015" title="watch"></div>'
        f'<div style="width:{r:.1f}%;background:#b42318" title="impaired"></div>'
        '</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Healthy", f"{h['healthy']}", f"{h['pct_nominal']:.0f}% nominal")
    cols[1].metric("Elevated Risk", f"{h['watch']}",
                   "highest-risk quartile / DQ flag" if h["watch"] else None,
                   delta_color="off")
    cols[2].metric("Impaired", f"{h['impaired']}",
                   f"{h['down']} down · {h['losing']} losing", delta_color="inverse")
    cols[3].metric("Fleet Oil Rate", f"{h['fleet_bopd']:,.0f} BOPD",
                   help="Latest-day total oil across the surveillance fleet.")
    n_high = sum(1 for x in anomalies
                 if not x.acknowledged and x.severity == "HIGH")
    st.markdown(
        pt.pill(f"{h['healthy']} healthy ({h['pct_nominal']:.0f}% nominal)", "ok")
        + " " + pt.pill(f"{h['watch']} elevated risk", "warn" if h["watch"] else "ok")
        + " " + pt.pill(f"{h['impaired']} impaired", "bad" if h["impaired"] else "ok")
        + " " + pt.pill(f"{n_high} HIGH-severity alert{'s' if n_high != 1 else ''}",
                        "bad" if n_high else "ok"),
        unsafe_allow_html=True)
    st.caption("Elevated Risk = the fleet's own highest-risk quartile by the failure "
               "signature (a Platt-calibrated probability from a model trained on this "
               "fleet's labeled faults; model card on Methods & Limitations) plus any "
               "non-$ data-quality flag. This is a HEALTH read — distinct from the "
               "Triage Board's economic 'Watch' tier (a losing well an intervention "
               "wouldn't pay for yet). All figures deterministic — no API key required.")


def core_fleet_size() -> int:
    import core
    return core.fleet_size()
