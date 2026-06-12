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
    deferred_usd = sum(a.deferred_usd_per_day for a in active)

    b = c.board(price, nri)
    action, _no_action = c.split_board(b)

    pt.context_bar([
        ("Surveillance fleet", c.scada_source_label(token)),
        ("As of", c.fleet_as_of(fleet)),
        ("Deck", c.deck_label()),
        ("Loss accounting", c.loss_context(st.session_state["data_source"])),
    ])

    top_label, top_value = "—", "—"
    if not action.empty:
        top = action.iloc[0]
        top_label = str(top["well_id"])
        top_value = f"${float(top['est_risked_npv']):,.0f}"
    pt.kpi_row([
        {"label": "Wells Scanned", "value": f"{core_fleet_size()}",
         "help": "Wells the digest scanned on the latest day."},
        {"label": "Open Alerts", "value": f"{len(active)}",
         "help": "Active anomalies on the latest scan (acknowledged events excluded)."},
        {"label": "Est. Deferred $/day", "value": f"${deferred_usd:,.0f}",
         "delta_color": "inverse",
         "help": "Sum of active anomalies' deferred $/day at the deck oil price "
                 "(gross; the Triage Board nets revenue by NRI)."},
        {"label": f"Top Opportunity · {top_label}", "value": top_value,
         "help": "Largest risked-NPV intervention on the Triage Board."},
    ])

    n_high = sum(1 for a in active if a.severity == "HIGH")
    at_risk = int((b["failure_risk_30d"] >= 0.5).sum()) if not b.empty else 0
    st.markdown(
        pt.pill(f"{n_high} HIGH severity", "bad" if n_high else "ok") + " " +
        pt.pill(f"{at_risk} wells ≥50% 30-day ESP risk",
                "warn" if at_risk else "ok") + " " +
        pt.pill("deterministic — no API key needed", "muted"),
        unsafe_allow_html=True)

    pt.section("The Morning Loop",
               "Surveillance → loss accounting → fleet triage → action chain.")
    st.markdown(
        "1. **Morning Brief** — the overnight scan: every anomaly the deterministic "
        "detectors raised, ranked money-first, with the deterministic brief.\n"
        "2. **Ongoing Events** — the event state machine: multi-day outages that stay "
        "ONGOING with running duration and cumulative deferred bbl/$.\n"
        "3. **Loss Accounting** — where the barrels are going on the monthly book "
        "(real Colorado ECMC by default): waterfall, $-Pareto by cause, recovery queue.\n"
        "4. **Triage Board** — the whole fleet ranked by risked-NPV opportunity, so "
        "the first work order goes to the biggest defensible number.\n"
        "5. **Action Chain** — for the selected well, run detect → predict → "
        "authorize and walk away with a decision-ready AFE.")
    _page_links()

    pt.section("Two Datasets, Stated Plainly")
    st.markdown(
        "**Today + Well File** run on a synthetic daily SCADA fleet (modeled Permian, "
        "known ground truth — public production data is monthly, not daily). "
        "**Loss Accounting** runs on real Colorado ECMC monthly records by default. "
        "They are different datasets at different cadences; this console does not "
        "fake a join between them. Full provenance on the **Sources & BYOD** page.")

    theme.references(["arps", "deferment", "npv"])


def _page_links() -> None:
    """Quick links into the loop (uses the Page registry app.py fills; quietly
    degrades to nothing when a view runs outside st.navigation, e.g. AppTest)."""
    import views
    wanted = ["Morning Brief", "Ongoing Events", "Deferment Overview",
              "Triage Board", "Action Chain"]
    pages = [views.PAGE_OBJECTS[t] for t in wanted if t in views.PAGE_OBJECTS]
    if not pages:
        return
    cols = st.columns(len(pages))
    for col, page in zip(cols, pages):
        with col:
            try:
                st.page_link(page)
            except Exception:  # noqa: BLE001 — outside a navigation context
                return


def core_fleet_size() -> int:
    import core
    return core.fleet_size()
