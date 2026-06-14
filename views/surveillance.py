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

# Per-lift diagnostic channels for the per-well drill-down: (column, label, color).
_LIFT_CHANNELS = {
    "Gas lift": [("gas_inj_mcfd", "Lift-gas injection (MCF/d)", theme.TEAL),
                 ("casing_pressure_psi", "Casing pressure (psi)", theme.PURPLE),
                 ("tubing_pressure_psi", "Tubing pressure (psi)", theme.BLUE)],
    "ESP": [("intake_pressure_psi", "Intake pressure (psi)", theme.PURPLE),
            ("motor_amps", "Motor amps (A)", theme.GREEN),
            ("current_imbalance_pct", "Current imbalance (%)", theme.RED)],
    "Rod pump": [("runtime_pct", "Runtime (%)", theme.GREEN),
                 ("motor_amps", "Motor load (A)", theme.AMBER)],
    "Flowing": [("tubing_pressure_psi", "Tubing pressure (psi)", theme.BLUE),
                ("gas_mcfd", "Gas (MCF/d)", theme.AMBER)],
}

_RANGES = {"90 days": 90, "180 days": 180, "1 year": 365, "Lifetime": 10_000}


def _sync_surv_well() -> None:
    st.session_state["well_id"] = st.session_state["surv_well"]


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
    pt.section("Per-Well Surveillance",
               "Drill into any well — production streams plus the diagnostics that "
               "matter for its artificial-lift type.")
    ids = c.scada_well_ids()
    cur = st.session_state.get("well_id") or (ids[0] if ids else None)
    if st.session_state.get("surv_well") != cur and cur in ids:
        st.session_state["surv_well"] = cur
    wsel = st.selectbox("Well", ids, key="surv_well", on_change=_sync_surv_well,
                        help="Also sets the globally-selected well for Well 360 / "
                             "Action Chain.")
    meta = fleet_registry.get(wsel)
    st.markdown(
        pt.pill(f"{meta.lift} lift", "info") + " " +
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
    cols = st.columns(3)
    cols[0].metric("Annual decline (fit)", f"{(1 - np.exp(-Di * 365)) * -100:,.0f}%"
                   if Di < 0 else f"{(np.exp(-Di * 365) - 1) * -100:,.0f}%",
                   help="Effective annual decline from the exponential fit.")
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
                   "and Triage Board rank the wells driving the gap.")
    else:
        st.caption("The fleet is tracking its expected decline — no material "
                   "fleet-wide deferment beyond normal decline.")


def _well_charts(df, meta, win: int) -> None:
    if df is None or not len(df):
        pt.empty_state("No SCADA for this well.")
        return
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
    fig.update_layout(showlegend=False)
    for ann in fig["layout"]["annotations"]:
        ann["x"], ann["xanchor"], ann["font"] = 0.0, "left", dict(size=11)
    st.plotly_chart(theme.style_fig(fig, height=150 * len(rows) + 40, legend=False),
                    width="stretch")
    note = {"Gas lift": "Gas-lift wells: watch injection rate vs casing pressure — "
                        "falling injection with rising casing is a valve/compressor "
                        "issue (restore injection to recover rate).",
            "ESP": "ESP wells: intake pressure, motor amps and current imbalance are "
                   "the failure-signature channels the ESP risk model scores.",
            "Rod pump": "Rod-pump wells: runtime and motor load proxy pump fillage / "
                        "pound-off.",
            "Flowing": "Flowing wells: tubing pressure and gas trend the natural "
                       "drive."}.get(meta.lift, "")
    if note:
        theme.source_note(note)
