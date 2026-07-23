"""Well File · Well 360 — one well, everything the console knows about it.

A conversation-starting one-pager for a thorough well review: identity + how long
it's been online + work history, a status verdict, lift-aware production and
diagnostic trends (with today's alert marked), the ESP 30-day risk and recommended
intervention, and the event history.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fleet_registry
import product_theme as pt
import theme

from views import _common as c

# Lift-specific diagnostic channels shown under the production streams — the SINGLE
# shared definition (Surveillance uses the same one) so the two pages can't diverge.
_LIFT_DIAG = c.LIFT_CHANNELS


def _sync_well() -> None:
    st.session_state["well_id"] = st.session_state["w360_pick"]


def render() -> None:
    c.ensure_state()
    price, nri, _disc = c.deck()
    well_id = st.session_state["well_id"]
    ids = c.scada_well_ids()
    if not well_id and not ids:
        pt.masthead("ops", "Well 360", "Select a well.")
        pt.empty_state("No fleet loaded — run bootstrap (first app start).")
        return

    meta = fleet_registry.get(well_id)
    hist = fleet_registry.well_history(well_id)
    pt.masthead("ops", "Well 360",
                f"{well_id} · {meta.name} — {meta.lift} · {meta.basin} "
                f"{meta.formation} · online {hist['years_online']} yr. A one-page "
                "well review.")

    # ---- drill-down bar: pick any well + jump to its Action Chain ---------------
    if st.session_state.get("w360_pick") != well_id and well_id in ids:
        st.session_state["w360_pick"] = well_id
    dd = st.columns([3, 2, 2])
    with dd[0]:
        st.selectbox("Drill into well", ids, key="w360_pick", on_change=_sync_well,
                     format_func=c.well_label,
                     help="Jump to any well — keeps the sidebar selection in "
                          "lockstep. The id is portable to Engineering Workbench "
                          "and Capital Desk.")
    with dd[1]:
        st.metric("API-14", meta.api14)
    with dd[2]:
        st.caption("Next step")
        c.next_step("Action Chain", "→ Run the Action Chain",
                    icon=":material/account_tree:")
    st.caption("Deeper engineering lenses for this well — nodal, PVT, gas-lift, "
               "run-life — live in **Engineering Workbench → Well Case File** "
               "(engineering-workbench.streamlit.app), same well id.")

    import core
    alert = c.alert_for_selected(price)
    flagged = alert.get("category") != "fleet_scan"
    pt.context_bar([
        ("Well", f"{well_id} · {meta.name}"),
        ("Online since", f"{hist['online_since']} ({hist['years_online']} yr)"),
        ("Deck", c.deck_label()),
        ("Today's digest", "flagged" if flagged else "not flagged (fleet scan)"),
    ])
    c.page_purpose(
        "**The question this page answers: what is the full story on THIS well — "
        "status, trends, risk, remaining life, and history — on one page?**\n\n"
        "- **When:** whenever a well surfaces anywhere in the loop (map click, "
        "Brief row, board row) and you want the whole picture before acting.\n"
        "- **Headline read:** the status verdict banner — OPPORTUNITY (act, "
        "risked NPV $) / AT-RISK WATCH (losing $/day but the fix doesn't pay "
        "yet) / STABLE (nothing to do) — plus the *30-Day Failure Signal* (a "
        "Platt-calibrated probability, i.e. the model's raw scores rescaled so "
        "'60%' really means ~60%).\n"
        "- **Also here:** the economic limit (the BOPD rate where net revenue = "
        "lease operating expense — the rate you'd plug & abandon at) and the "
        "well's work history.\n"
        "- **Next:** run the **Action Chain** to price and authorize the "
        "recommended job for this exact well.")

    # ---- status verdict (from the board tiers) ---------------------------------
    board = c.board_with_deferred(price, nri)
    brow = board[board["well_id"] == well_id]
    ev_days = c.ongoing_event_days(c.DISK_TOKEN, price)
    _status_banner(brow, hist, price, nri, ev_days.get(str(well_id)))

    # ---- identity --------------------------------------------------------------
    mh = st.columns(4)
    mh[0].metric("Lift", meta.lift)
    mh[1].metric("Lateral (ft)", f"{meta.lateral_length_ft:,}")
    mh[2].metric("Basin · Formation", f"{meta.basin} · {meta.formation}")
    mh[3].metric("Peer Group", meta.peer_group)
    mh2 = st.columns(4)
    mh2[0].metric("Area", meta.area)
    mh2[1].metric("Online Since", hist["online_since"],
                  f"{hist['years_online']} yr", delta_color="off")
    mh2[2].metric("Prior Interventions", f"{hist['n_workovers']}",
                  f"last {hist['last_worked']}" if hist["last_worked"] else "none on record",
                  delta_color="off")
    mh2[3].metric("Digest Status", "Flagged" if flagged else "Fleet scan")
    theme.well_cross_links("pipeline", well_id)
    if meta.hero:
        st.caption(f"Hero well: {meta.storyline}")

    # ---- production + lift-aware diagnostics -----------------------------------
    pt.section("Production & Diagnostics — Trend",
               "Oil / water / gas plus the diagnostics that matter for this lift "
               "type; today's digest alert (if any) is marked on the timeline.")
    _trend_chart(core.well_scada(alert), meta, alert if flagged else None)
    theme.source_note(
        f"Daily channels for a {meta.lift} well. Water = gross fluid − oil. The "
        "dashed marker is the date of today's digest alert for this well. The chart "
        f"shows the most recent SCADA window, not the full history since "
        f"{hist['online_since']} ({hist['years_online']} yr online). Dashed "
        "'Expected decline' on the oil panel = this well's own fitted exponential "
        "decline — a self-referential trend check, not an offset-well type curve.")
    if flagged:
        st.markdown(f"**{alert['category']}** · severity **{alert['severity']}** — "
                    f"{alert['headline']}")

    # ---- risk + recommended intervention ---------------------------------------
    # The risk model is generic (ground-truth-trained), but the wording is named for
    # the ESP demo; scope the "ESP" label to actual ESP wells so a gas-lift, rod-pump,
    # or flowing well isn't told it has an ESP it doesn't have.
    is_esp = meta.lift == "ESP"
    scorer = "ESP agent" if is_esp else "failure-risk agent"
    model_name = "ESP model" if is_esp else "failure-risk model"
    pt.section("Failure Risk & Recommended Intervention",
               f"The {scorer} scores the well's SCADA signature; the mode classifier "
               "maps it to a lift-appropriate priced intervention.")
    diag = c.diagnosis(well_id, price)
    r1, r2, r3 = st.columns(3)
    r1.metric("30-Day Failure Signal", f"{diag['esp_risk_score']:.0%}",
              help=f"A Platt-calibrated probability from the {model_name} trained on this "
                   "fleet's labeled faults (calibrated out-of-fold AUROC ≈0.98 — high "
                   "because the synthetic signatures are cleanly separable, not a "
                   "real-world claim). Full model card on Methods & Limitations.")
    r2.metric("Suspected Mode", diag["suspected_mode"].split("—")[0].strip() or "—",
              help=diag["suspected_mode"])
    r3.metric("Recommended Intervention", diag["intervention"].replace("_", " "))
    st.caption(diag["primary_diagnosis"])

    # ---- economic limit & remaining producing life -----------------------------
    pt.section("Economic Limit & Remaining Life",
               "The rate at which net revenue equals lease operating expense — the "
               "rate you'd plug & abandon at — and how long this well's own decline "
               "leaves before it gets there.")
    el = core.economic_limit(core.well_scada(alert), realized_price=price,
                             net_revenue_interest=nri)
    if el is None:
        st.caption("Not enough producing history to estimate an economic limit.")
    elif el.get("status") == "down":
        st.warning("**Well is currently down / collapsed.** A reserves read is "
                   "meaningless while it's offline — restore production first, then "
                   "the economic limit recomputes off the recovered rate.")
        st.caption(f"Net margin ${el['net_margin_per_bbl']:,.0f}/bbl · "
                   f"recent producing rate ~{el['q_now_bopd']:,.0f} BOPD.")
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric("Economic-limit rate", f"{el['q_limit_bopd']:,.0f} BOPD",
                  help="Net revenue = lease operating expense at this rate.")
        months = el["months_remaining"]
        e2.metric("Remaining life (est.)",
                  "—" if months == float("inf") else f"{months / 12:,.1f} yr",
                  f"{el['q_now_bopd']:,.0f} BOPD now", delta_color="off")
        e3.metric("Net margin", f"${el['net_margin_per_bbl']:,.0f}/bbl",
                  help="Realized price × NRI − variable opex.")
        if el.get("below_established_trend"):
            st.warning(
                f"**Producing below its established trend** — ~{el['q_now_bopd']:,.0f} "
                f"BOPD now vs ~{el.get('q_trend_bopd', el['q_now_bopd']):,.0f} on the "
                "prior 30-day plateau. Remaining life is computed from this **current "
                "(depressed) rate**, so it already reflects today's reduced deliverability "
                "— but it projects forward on the gentler **established-trend** decline, "
                "not the steeper recent drop. If the acute decline continues instead of "
                "reverting to trend, the true remaining life is **shorter** than shown. "
                "Read it alongside the failure signal above — a long-run reserves "
                "baseline, not a verdict on the active problem.")
        st.caption(f"Current rate is the recent trailing producing rate; assumes "
                   f"${el['loe_per_month']:,.0f}/well-month fixed LOE and the well's "
                   f"established-trend decline ({el['annual_decline_pct']:,.0f}%/yr) — the "
                   "same decline the type-curve overlay above uses. Illustrative carrying "
                   "cost — set per asset in a real deployment.")

    # ---- well work history ------------------------------------------------------
    pt.section("Well History — Interventions",
               "How many times this well has been worked, and what was done — the "
               "context for whether another job is warranted.")
    if hist["records"]:
        wh = pd.DataFrame(hist["records"])
        wh["cost_usd"] = wh["cost_usd"].map(lambda v: f"${v:,.0f}")
        wh["uplift_bopd"] = wh["uplift_bopd"].map(lambda v: f"+{v:,.0f}")
        wh = wh[["date", "type", "cost_usd", "uplift_bopd", "result"]]
        wh.columns = ["Date", "Intervention", "Cost", "Uplift (bopd)", "Result"]
        st.dataframe(wh, width="stretch", hide_index=True)
        spend = sum(r["cost_usd"] for r in hist["records"])
        st.caption(f"{hist['n_workovers']} prior intervention(s) since "
                   f"{hist['online_since']} · ~${spend:,.0f} lifetime workover spend. "
                   "Illustrative synthetic history (deterministic per well).")
    else:
        st.caption("No prior interventions on record — this well has run since "
                   f"{hist['online_since']} without a workover.")

    # ---- event history ----------------------------------------------------------
    pt.section("Event History",
               "This well's events from the state-machine replay of the recent "
               "history (raw committed fleet, no demo injection).")
    events = [e for e in c.replay_events(c.DISK_TOKEN, price, False)
              if e.well_id == well_id]
    if events:
        ev_df = pd.DataFrame([{
            "Event Type": e.event_type, "State": e.state, "Start Date": e.start_date,
            "Duration (days)": e.duration_days,
            "Cumulative Deferred bbl": round(e.deferred_bopd, 0),
            "Cumulative Deferred $": round(e.deferred_usd, 0),
        } for e in events])
        st.dataframe(ev_df, width="stretch", hide_index=True)
    else:
        st.caption("No open or recently-resolved events for this well on the "
                   "replayed window.")

    c.next_step("Action Chain",
                "→ Run the full detect → predict → authorize flow for this well "
                "(Action Chain)", icon=":material/account_tree:")
    theme.references(["arps", "shap", "npv"])


def _status_banner(brow, hist, price: float, nri: float,
                   event_days: int | None = None) -> None:
    """A one-line verdict: opportunity / watch / stable, with the economics — plus
    the open-event downtime context (verify post-restart before acting)."""
    if not len(brow):
        return
    r = brow.iloc[0]
    npv = float(r["est_risked_npv"])
    deferred = float(r["deferred_usd_per_day"])
    intervention = str(r["recommended_intervention"]).replace("_", " ")
    if r["recommended_intervention"] == "no_action":
        pill = pt.pill("STABLE — no action", "ok")
        msg = "Producing on trend with no deferment; nothing to do today."
    elif npv > 0:
        pill = pt.pill("OPPORTUNITY — value-accretive", "ok")
        msg = (f"Recommended {intervention} clears its cost — risked NPV "
               f"**${npv:,.0f}**. Authorize on the Action Chain.")
    elif deferred > 0:
        pill = pt.pill("AT-RISK WATCH — monitor", "warn")
        msg = (f"Losing **${deferred:,.0f}/day** (net), but {intervention} doesn't "
               f"clear its cost yet (risked NPV −${abs(npv):,.0f}). Monitor and "
               "re-rank as the signal strengthens.")
    else:
        pill = pt.pill("STABLE — no action", "muted")
        msg = "No deferment and no value-accretive intervention today."
    chips = (pill + " "
             + pt.pill(f"deferred ${deferred:,.0f}/day", "warn" if deferred else "muted")
             + " " + pt.pill(f"{hist['n_workovers']} prior jobs", "info"))
    if event_days:
        chips += " " + pt.pill(f"ongoing event {event_days}d — verify post-restart",
                               "warn")
        msg += (f" ⚠ This well is in an OPEN downtime event ({event_days}d running) — "
                "verify the post-restart rate before acting on any recommendation.")
    st.markdown(chips, unsafe_allow_html=True)
    st.caption(msg)


def _trend_chart(scada, meta, alert) -> None:
    """Oil/water/gas streams + lift-aware diagnostics on stacked auto-scaled axes."""
    d = scada.copy()
    if "bfpd" in d.columns and "bopd" in d.columns:
        d["water"] = (d["bfpd"] - d["bopd"]).clip(lower=0)
    streams = [("bopd", "Oil (BOPD)", theme.GREEN), ("water", "Water (BWPD)", theme.BLUE),
               ("gas_mcfd", "Gas (MCF/d)", theme.AMBER)]
    rows = [(col, lbl, clr) for col, lbl, clr in streams if col in d.columns]
    rows += [(col, lbl, clr) for col, lbl, clr in _LIFT_DIAG.get(meta.lift, [])
             if col in d.columns]
    if not rows:
        pt.empty_state("No SCADA channels for this well.")
        return
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=[r[1] for r in rows])
    for i, (col, _lbl, clr) in enumerate(rows, start=1):
        fig.add_trace(go.Scatter(x=d["date"], y=d[col], line=dict(color=clr, width=1.4)),
                      row=i, col=1)
        # Overlay the well's own expected exponential decline on the oil panel — the
        # "on the type curve, or deferring?" read (the gap below = implied deferment).
        if col == "bopd":
            import core
            fit = core.fit_well_decline(d)
            if fit is not None:
                fig.add_trace(go.Scatter(
                    x=fit["dates"], y=fit["expected"], name="Expected decline",
                    line=dict(color=theme.RED, width=1.2, dash="dash")), row=i, col=1)
    if alert is not None and alert.get("date"):
        ts = pd.Timestamp(alert["date"])
        for i in range(1, len(rows) + 1):
            fig.add_vline(x=ts, line_color=theme.RED, line_dash="dash",
                          line_width=1.1, row=i, col=1)
    fig.update_layout(showlegend=False)
    for ann in fig["layout"]["annotations"]:
        ann["x"], ann["xanchor"], ann["font"] = 0.0, "left", dict(size=11)
    st.plotly_chart(theme.style_fig(fig, height=140 * len(rows) + 40, legend=False),
                    width="stretch")
