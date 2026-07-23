"""Today · Surveillance — Spotfire-style production surveillance for the fleet.

Fleet oil / water / gas rate-time with moving averages and a decline / type-curve
check (are we holding the expected decline, or deferring?), plus a per-well
drill-down whose diagnostics adapt to the well's artificial-lift type (gas-lift
injection + casing/tubing pressure for gas-lift, ESP intake/amps/temp for ESP, …).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fleet_registry
import product_theme as pt
import theme

from views import _common as c

# Per-lift diagnostic channels for the per-well drill-down — the SINGLE shared
# definition (Well 360 imports the same one) so the two pages can't diverge on which
# channels / colors a lift type shows.
_LIFT_CHANNELS = c.LIFT_CHANNELS

_RANGES = {"90 days": 90, "180 days": 180, "1 year": 365, "Lifetime": 10_000}


def _sync_surv_well() -> None:
    st.session_state["well_id"] = st.session_state["surv_well"]


_TIER_COLOR = {"down": "#b42318", "watch": "#d9a015", "healthy": "#1b7a3d"}
_TIER_SIZE = {"down": 16, "watch": 12, "healthy": 9}


def _fleet_map(fleet: dict, price: float) -> None:
    """Tier-colored, CLICKABLE map of the fleet (Spotfire/OFM/Avocet all open on a
    map). Coloured green / amber / red by live health tier; clicking a well loads it
    into the per-well drill-down below (and the global Well File selection)."""
    import core
    board = c.board_with_deferred(price, st.session_state.get("nri", 0.80))
    tiers = core.well_tiers(fleet, board)
    rows = []
    for wid in fleet:
        wid = str(wid)
        meta = fleet_registry.get(wid)
        lat, lon = fleet_registry.surface_latlon(wid)
        tier = tiers.get(wid, "healthy")
        rows.append({"well_id": wid, "lat": lat, "lon": lon, "tier": tier,
                     "color": _TIER_COLOR[tier], "size": _TIER_SIZE[tier],
                     "hover": (f"{wid} · {meta.name}<br>{meta.lift} · {meta.ctb}"
                               f"<br>{meta.basin} · {meta.formation} · {tier}")})
    mdf = pd.DataFrame(rows)
    n = {"down": 0, "watch": 0, "healthy": 0}
    for w in fleet:
        n[tiers.get(str(w), "healthy")] += 1
    st.markdown(
        pt.pill(f"{n['healthy']} healthy", "ok") + " "
        + pt.pill(f"{n['watch']} watch", "warn" if n["watch"] else "ok") + " "
        + pt.pill(f"{n['down']} down", "bad" if n["down"] else "ok"),
        unsafe_allow_html=True)
    fig = go.Figure(go.Scattermap(
        lat=mdf["lat"], lon=mdf["lon"], mode="markers",
        marker=dict(size=mdf["size"], color=mdf["color"], opacity=0.9),
        customdata=mdf["well_id"], hovertext=mdf["hover"], hoverinfo="text"))
    fig.update_layout(
        map=dict(style="open-street-map",
                 center=dict(lat=float(mdf["lat"].mean()),
                             lon=float(mdf["lon"].mean())),
                 zoom=6.1),
        height=430, margin=dict(l=0, r=0, t=6, b=0), showlegend=False)
    event = st.plotly_chart(fig, key="surv_map", on_select="rerun",
                            selection_mode="points", width="stretch")
    _apply_map_selection(event)
    theme.source_note(
        "Click a well to load it in the per-well drill-down below (also sets the "
        "global Well File selection). Well locations are SYNTHETIC — each well's real "
        "Permian county (from the fleet registry) placed at the county centroid with a "
        "deterministic within-county jitter. Colour is the live health tier (green "
        "healthy / amber watch / red down).")


def _apply_map_selection(event) -> None:
    """Route a map click into the global selected well. None-safe (the AppTest
    harness renders with no selection) and deduped with a session sentinel —
    Plotly selection state persists across reruns, so without the sentinel the page
    would re-jump to the clicked well on every rerun."""
    try:
        pts = list(event.selection.points) if event is not None else []
    except Exception:  # noqa: BLE001
        pts = []
    if not pts:
        # Deselect (empty selection event) — clear the sentinel so the user can
        # re-select the same well from the map later (click A → dropdown B →
        # click A again must work).
        st.session_state.pop("_surv_map_handled", None)
        return
    cd = pts[0].get("customdata")
    wid = str(cd[0] if isinstance(cd, (list, tuple)) else cd) if cd is not None else ""
    if not wid or st.session_state.get("_surv_map_handled") == wid:
        return
    st.session_state["_surv_map_handled"] = wid
    # Writing well_id here would raise StreamlitAPIException — the sidebar selectbox
    # owns that key and has already rendered this run. Park the target for app.py's
    # top-of-run handoff and rerun; the sentinel above stops the rerun from looping,
    # and the next run's drill-down pre-sync picks the well up before its selectbox.
    st.session_state["_well_jump"] = wid
    st.rerun()


def render() -> None:
    c.ensure_state()
    price, _nri, _disc = c.deck()

    pt.masthead("ops", "Surveillance",
                "Fleet production surveillance — oil, water and gas rate-time, the "
                "decline/type-curve check, and a lift-aware per-well drill-down.")
    token = c.scada_token()
    fleet = c.fleet_for_token(token)
    pt.context_bar([
        ("Surveillance fleet", c.scada_source_label(token)),
        ("As of", c.fleet_as_of(fleet)),
        ("Deck", c.deck_label()),
    ])
    theme.data_badge("synthetic", "Modeled daily SCADA fleet with known ground truth "
                                  "— public production is monthly, not daily.")

    _live_body(fleet, price)


def _live_body(fleet: dict, price: float) -> None:
    """Spotfire-style live surveillance — fleet rate-time, type-curve check, fleet
    map, and the lift-aware per-well drill-down."""
    all_ids = [str(w) for w in fleet]
    with st.expander("Filters — CTB · lift type · basin · county", expanded=False):
        keep = c.fleet_filter_controls("surv", all_ids)
    keepset = set(keep)
    if len(keep) < len(all_ids):
        fleet = {w: df for w, df in fleet.items() if str(w) in keepset}
        st.caption(f"Filtered: **{len(fleet)} of {len(all_ids)}** wells — the KPIs, "
                   "charts, map, and drill-down list below reflect the filtered "
                   "selection only.")
        if not fleet:
            pt.empty_state("No wells match the active filters.",
                           "Clear a filter above to bring wells back.")
            return
    ff, n = _fleet_frame(fleet)
    if ff.empty:
        pt.empty_state("No production in the fleet.")
        return
    latest = ff.iloc[-1]
    wc = 100.0 * latest["water"] / (latest["oil"] + latest["water"]) \
        if (latest["oil"] + latest["water"]) > 0 else 0.0
    online = int((pd.DataFrame({w: (df["bopd"].iloc[-1] if df is not None and len(df)
                                    else 0) for w, df in fleet.items()},
                               index=[0]).iloc[0] >= 1.0).sum())
    pt.kpi_row([
        {"label": "Fleet Oil", "value": f"{latest['oil']:,.0f} BOPD"},
        {"label": "Fleet Water", "value": f"{latest['water']:,.0f} BWPD"},
        {"label": "Fleet Gas", "value": f"{latest['gas']:,.0f} MCF/d"},
        {"label": "Avg Oil / Well", "value": f"{latest['oil'] / max(n, 1):,.0f} BOPD"},
    ])
    pt.kpi_row([
        {"label": "Water Cut", "value": f"{wc:.0f}%",
         "help": "Fleet water ÷ (oil + water) on the latest day."},
        {"label": "Wells Online", "value": f"{online} / {n}",
         "help": "Wells producing ≥1 BOPD on the latest day (the rest are down)."},
        {"label": "Fleet GOR", "value": f"{1000 * latest['gas'] / max(latest['oil'], 1):,.0f} scf/bbl",
         "help": "Gas-oil ratio across the fleet on the latest day."},
        {"label": "Days of History", "value": f"{len(ff):,}"},
    ])

    rng_label = st.radio("Time range", list(_RANGES), index=2, horizontal=True,
                         key="surv_range")
    win = _RANGES[rng_label]
    fwin = ff.tail(win)

    pt.section("Fleet Production — Rate / Time",
               "Daily fleet rate by stream with a 30-day moving average — the "
               "first thing a surveillance engineer scans every morning.")
    _streams_chart(fwin)
    theme.source_note(
        "Fleet daily totals: oil = Σ bopd, water = Σ (bfpd − bopd), gas = Σ gas_mcfd "
        "across producing wells. Dashed line is the trailing 30-day moving average.")

    pt.section("On the Type Curve?",
               "Fleet oil against its own exponential-decline fit — are we holding "
               "the expected decline, or losing barrels to downtime / underperformance?")
    _typecurve_chart(ff)

    st.divider()
    pt.section("Fleet Map — Health by Location",
               "Where the fleet is and how it's doing — the spatial view every "
               "surveillance tool opens on, coloured by live health tier.")
    _fleet_map(fleet, price)

    st.divider()
    pt.section("Per-Well Surveillance",
               "Drill into any well — production streams plus the diagnostics that "
               "matter for its artificial-lift type.")
    ids = [w for w in c.scada_well_ids() if w in keepset] or c.scada_well_ids()
    cur = st.session_state.get("well_id") or (ids[0] if ids else None)
    if st.session_state.get("surv_well") != cur and cur in ids:
        st.session_state["surv_well"] = cur
    if st.session_state.get("surv_well") not in ids and ids:
        # active filters excluded the previously-picked well — keep the widget valid
        st.session_state["surv_well"] = ids[0]
    wsel = st.selectbox("Well", ids, key="surv_well", on_change=_sync_surv_well,
                        help="Also sets the globally-selected well for Well 360 / "
                             "Action Chain.")
    meta = fleet_registry.get(wsel)
    st.markdown(
        pt.pill(f"{meta.lift} lift", "info") + " " +
        pt.pill(f"{meta.ctb}", "muted") + " " +
        pt.pill(f"{meta.basin} · {meta.formation}", "muted") + " " +
        pt.pill(f"{meta.lateral_length_ft:,} ft lateral", "muted"),
        unsafe_allow_html=True)
    _well_charts(fleet.get(wsel), meta, win)

    theme.references(["arps", "deferment"])


# ---- fleet helpers -----------------------------------------------------------

def _fleet_frame(fleet: dict) -> tuple[pd.DataFrame, int]:
    """Daily fleet totals (oil/water/gas) summed across producing wells."""
    frames = []
    for _wid, df in fleet.items():
        if df is None or not len(df) or "bopd" not in df.columns:
            continue
        d = df[["date", "bopd", "bfpd", "gas_mcfd"]].copy()
        d["water"] = (d["bfpd"] - d["bopd"]).clip(lower=0)
        frames.append(d.set_index("date"))
    if not frames:
        return pd.DataFrame(), 0
    oil = sum(f["bopd"] for f in frames)
    water = sum(f["water"] for f in frames)
    gas = sum(f["gas_mcfd"] for f in frames)
    out = pd.DataFrame({"date": oil.index, "oil": oil.to_numpy(),
                        "water": water.to_numpy(), "gas": gas.to_numpy()})
    return out.reset_index(drop=True), len(frames)


def _streams_chart(ff: pd.DataFrame) -> None:
    streams = [("oil", "Oil (BOPD)", theme.GREEN), ("water", "Water (BWPD)", theme.BLUE),
               ("gas", "Gas (MCF/d)", theme.AMBER)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[s[1] for s in streams])
    for i, (col, lbl, clr) in enumerate(streams, start=1):
        fig.add_trace(go.Scatter(x=ff["date"], y=ff[col], name=lbl,
                                 line=dict(color=clr, width=1.4)), row=i, col=1)
        ma = ff[col].rolling(30, min_periods=5).mean()
        fig.add_trace(go.Scatter(x=ff["date"], y=ma, name=f"{lbl} 30-d MA",
                                 line=dict(color=theme.NAVY, width=1.1, dash="dash")),
                      row=i, col=1)
    fig.update_layout(showlegend=False)
    for ann in fig["layout"]["annotations"]:
        ann["x"], ann["xanchor"], ann["font"] = 0.0, "left", dict(size=12)
    st.plotly_chart(theme.style_fig(fig, height=520, legend=False), width="stretch")


def _typecurve_chart(ff: pd.DataFrame) -> None:
    t = np.arange(len(ff), dtype=float)
    oil = ff["oil"].to_numpy(dtype=float)
    # Fit exponential decline qi·exp(−Di·t) on the first 80% (the established trend),
    # then extrapolate so recent under-performance shows as a gap below the curve.
    fit_n = max(30, int(len(ff) * 0.8))
    pos = oil[:fit_n] > 0
    if pos.sum() >= 10:
        Di, lnqi = np.polyfit(t[:fit_n][pos], np.log(oil[:fit_n][pos]), 1)
        expected = np.exp(lnqi) * np.exp(Di * t)
    else:
        expected = np.full_like(oil, oil.mean())
        Di = 0.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ff["date"], y=oil, name="Actual fleet oil",
                             line=dict(color=theme.GREEN, width=1.6)))
    fig.add_trace(go.Scatter(x=ff["date"], y=expected, name="Expected decline (type curve)",
                             line=dict(color=theme.RED, width=1.4, dash="dash")))
    fig.update_layout(xaxis_title="", yaxis_title="Fleet oil (BOPD)")
    st.plotly_chart(theme.style_fig(fig, height=340), width="stretch")
    recent_act = float(oil[-7:].mean())
    recent_exp = float(expected[-7:].mean())
    var_pct = 100.0 * (recent_act - recent_exp) / recent_exp if recent_exp else 0.0
    # Effective annual decline = 1 − exp(Di·365) (Di is per-day, negative for a
    # decliner). The earlier exp(|Di|·365)−1 form overstated it (~12% vs ~10%).
    eff_annual_decline = (1.0 - np.exp(Di * 365.0)) * 100.0
    cols = st.columns(3)
    cols[0].metric("Annual decline (fit)", f"{eff_annual_decline:,.0f}%",
                   help="Effective annual decline from the exponential fit "
                        "(1 − exp(Di·365), Di in 1/day).")
    cols[1].metric("Actual vs type curve", f"{var_pct:+.1f}%",
                   delta="below curve" if var_pct < -1 else "on/above curve",
                   delta_color="inverse" if var_pct < -1 else "off",
                   help="Trailing-7-day fleet oil vs the extrapolated decline.")
    cols[2].metric("Implied deferment", f"{max(recent_exp - recent_act, 0):,.0f} BOPD",
                   help="How far below the expected decline the fleet is producing "
                        "(barrels the Loss Accounting book quantifies by cause).")
    if var_pct < -1:
        st.caption("The fleet is producing **below** its decline trend — downtime "
                   "and underperformance are deferring barrels. The Morning Brief "
                   "and Optimization Board rank the wells driving the gap.")
    else:
        st.caption("The fleet is tracking its expected decline — no material "
                   "fleet-wide deferment beyond normal decline.")


def _well_charts(df, meta, win: int) -> None:
    if df is None or not len(df):
        pt.empty_state("No SCADA for this well.")
        return
    import core
    d = df.tail(win).copy()
    d["water"] = (d["bfpd"] - d["bopd"]).clip(lower=0)

    # production streams
    streams = [("bopd", "Oil (BOPD)", theme.GREEN), ("water", "Water (BWPD)", theme.BLUE),
               ("gas_mcfd", "Gas (MCF/d)", theme.AMBER)]
    present = [(col, lbl, clr) for col, lbl, clr in streams if col in d.columns]
    diag = [(col, lbl, clr) for col, lbl, clr in _LIFT_CHANNELS.get(meta.lift, [])
            if col in d.columns]
    rows = present + diag
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, subplot_titles=[r[1] for r in rows])
    for i, (col, _lbl, clr) in enumerate(rows, start=1):
        fig.add_trace(go.Scatter(x=d["date"], y=d[col], line=dict(color=clr, width=1.4)),
                      row=i, col=1)
        # Per-well expected decline on the oil panel (fit on the FULL history, drawn
        # over the visible window) — is this well on its type curve, or deferring?
        if col == "bopd":
            fit = core.fit_well_decline(df)
            if fit is not None:
                exp = list(fit["expected"])[-len(d):]
                fig.add_trace(go.Scatter(x=d["date"], y=exp, name="Expected decline",
                              line=dict(color=theme.RED, width=1.2, dash="dash")),
                              row=i, col=1)
    fig.update_layout(showlegend=False)
    for ann in fig["layout"]["annotations"]:
        ann["x"], ann["xanchor"], ann["font"] = 0.0, "left", dict(size=11)
    st.plotly_chart(theme.style_fig(fig, height=150 * len(rows) + 40, legend=False),
                    width="stretch")
    note = {"Gas lift": "Gas-lift wells: watch injection rate vs casing pressure — "
                        "falling injection with RISING casing is a downhole "
                        "valve/orifice (or annulus) problem; falling injection with "
                        "FALLING casing points upstream to the compressor / gas "
                        "supply. Either way, restore injection to recover rate.",
            "ESP": "ESP wells: intake pressure, motor amps and current imbalance are "
                   "the failure-signature channels the ESP risk model scores.",
            "Rod pump": "Rod-pump wells: runtime and motor load proxy pump fillage / "
                        "pound-off.",
            "Flowing": "Flowing wells: no downhole pump — the oil/water/gas streams "
                       "and downhole pressure carry the review."}.get(meta.lift, "")
    if note:
        theme.source_note(note)
