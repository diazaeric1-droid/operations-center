"""Shared view-layer helpers for Operations Center.

Owns the Streamlit caching wrappers over ``core`` (core itself stays
streamlit-free), the global session-state guard, the deck/context labels, and
the Loss Accounting source control. Views import this as ``from views import
_common as c``.
"""
from __future__ import annotations

import hashlib
import os
import tempfile

import pandas as pd
import streamlit as st

import core

# ---- global session-state contract ------------------------------------------
# app.py seeds these before navigation runs; views call ensure_state() too so
# each page is independently executable (AppTest per-view coverage).
STATE_DEFAULTS: dict = {
    "oil_price": 70.0,
    "nri": 0.80,
    "discount": 0.10,
    "well_id": None,          # resolved to the first fleet well on first use
    "data_source": core.DEF_SRC_SYNTHETIC,  # Loss Accounting source (synthetic default)
    "anthropic_key": "",
    # session-only uploads (Data page writes these; nothing touches disk)
    "scada_upload": None,         # bytes | None — fleet SCADA CSV (digest schema)
    "scada_upload_name": "",
    "scada_source": "disk",       # "disk" | "upload" — which fleet Brief/Events use
    "deferment_upload": None,     # bytes | None — tidy monthly production CSV
    "deferment_upload_name": "",
}


def ensure_state() -> None:
    for k, v in STATE_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    if st.session_state.get("well_id") is None:
        ids = scada_well_ids()
        if ids:
            st.session_state["well_id"] = ids[0]


@st.cache_data(show_spinner=False)
def scada_well_ids() -> list[str]:
    """Sorted well ids of the bootstrapped digest fleet (cheap glob, no parse)."""
    return sorted(p.stem for p in core.DIGEST_FLEET.glob("well_*.csv"))


def deck() -> tuple[float, float, float]:
    """(oil_price, nri, discount) from session state."""
    ss = st.session_state
    return float(ss.get("oil_price", 70.0)), float(ss.get("nri", 0.80)), \
        float(ss.get("discount", 0.10))


def deck_label() -> str:
    price, nri, disc = deck()
    return f"${price:,.2f}/bbl · NRI {nri:.2f} · {disc:.0%} disc."


# ---- SCADA fleet (Today + Well File pages) -----------------------------------

_BYOD_SCADA: dict[str, bytes] = {}  # token -> uploaded CSV bytes (process-local)

DISK_TOKEN = "disk"


@st.cache_resource(show_spinner=False)
def _disk_fleet() -> dict:
    return core.load_scada_fleet()


@st.cache_resource(show_spinner=False)
def _byod_fleet(token: str) -> dict:
    data = _BYOD_SCADA.get(token)
    if data is None:
        raise KeyError("uploaded fleet is no longer in memory — please re-upload")
    return core.load_scada_fleet_from_bytes(data)


def scada_token() -> str:
    """Resolve the active SCADA source to a cache token (and register bytes)."""
    ss = st.session_state
    data = ss.get("scada_upload")
    if ss.get("scada_source") == "upload" and data:
        token = f"byod::{hashlib.sha1(data).hexdigest()}"
        _BYOD_SCADA[token] = data
        return token
    return DISK_TOKEN


def fleet_for_token(token: str) -> dict:
    return _byod_fleet(token) if token.startswith("byod::") else _disk_fleet()


def scada_source_label(token: str | None = None) -> str:
    token = token or scada_token()
    if token.startswith("byod::"):
        name = st.session_state.get("scada_upload_name") or "uploaded CSV"
        return f"Your fleet — {name} (session only)"
    return f"Synthetic daily SCADA · {len(scada_well_ids())} wells"


def fleet_as_of(fleet: dict) -> str:
    """Latest date across the fleet (ISO) for the context bar."""
    dates = [df["date"].max() for df in fleet.values() if df is not None and len(df)]
    return pd.Timestamp(max(dates)).date().isoformat() if dates else "—"


@st.cache_resource(show_spinner=False)
def scan(token: str, price: float) -> list:
    """Full deterministic fleet scan (Anomaly dataclasses → cache_resource)."""
    return core.scan_anomalies(fleet_for_token(token), price_per_bbl=price)


@st.cache_resource(show_spinner=False)
def replay_events(token: str, price: float, inject_demo: bool) -> list:
    """Event state-machine replay (Event dataclasses → cache_resource)."""
    return core.replay_events(fleet_for_token(token), price_per_bbl=price,
                              inject_demo=inject_demo)


@st.cache_data(show_spinner=False)
def alerts(price: float) -> list[dict]:
    """ESP-related chain alerts (stage-1 artifacts) on the bootstrapped fleet."""
    return core.get_alerts(price_per_bbl=price)


@st.cache_data(show_spinner="Ranking the fleet by risked-NPV opportunity…")
def board(price: float, nri: float) -> pd.DataFrame:
    return core.rank_fleet(price_per_bbl=price, net_revenue_interest=nri)


@st.cache_data(show_spinner=False)
def deferred_by_well(token: str, price: float) -> dict:
    """Per-well deferred bopd from the digest's rate-loss scan — the REAL money
    signal. The Triage Board ranks on the ESP alert feed, which carries no deferred
    barrels by design (an ESP failure signature ≠ a quantified rate loss), so the
    board's own deferred column is zero; this joins the digest's decline-aware
    rate-loss detector back in for an honest deferred-$ column."""
    out: dict[str, float] = {}
    for a in scan(token, price):
        if getattr(a, "acknowledged", False):
            continue
        d = float(getattr(a, "deferred_bopd", 0.0) or 0.0)
        if d > 0:
            out[str(a.well_id)] = max(out.get(str(a.well_id), 0.0), d)
    return out


def board_with_deferred(price: float, nri: float, token: str | None = None
                        ) -> pd.DataFrame:
    """The certified ranked board with a DISPLAY ``deferred_bopd`` / ``deferred_usd_
    per_day`` sourced from the digest rate-loss scan (real values), leaving the
    certified ranking columns (risk, intervention, risked NPV) untouched."""
    b = board(price, nri).copy()
    dbw = deferred_by_well(token or DISK_TOKEN, price)
    b["deferred_bopd"] = b["well_id"].map(lambda w: dbw.get(str(w), 0.0)).round(1)
    b["deferred_usd_per_day"] = (b["deferred_bopd"] * price * nri).round(0)
    return b


def split_board(b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(action wells, no-action tier) — same split pe-pipeline's board uses."""
    mask = b["recommended_intervention"] == "no_action"
    return b[~mask].reset_index(drop=True), b[mask].reset_index(drop=True)


# ---- Loss Accounting (deferment) source --------------------------------------

LOSS_SOURCE_LABELS = {
    core.DEF_SRC_REAL_CO: "Real — Colorado DJ Basin (ECMC)",
    core.DEF_SRC_SYNTHETIC: "Synthetic (reason-coded demo)",
    core.DEF_SRC_UPLOAD: "Your upload (monthly CSV)",
}

_BYOD_DEF: dict[str, bytes] = {}


def loss_source_control() -> str:
    """Render the Loss Accounting source selector and return the active source key.

    The built-in source is the synthetic reason-coded fleet (full cause attribution
    + a ground-truth classifier eval); a validated monthly CSV from the Data page
    adds a "Your upload" option. The real Colorado ECMC monthly extract is retained
    only as a bring-your-own reference dataset (Sources & BYOD), not a default —
    public monthly filings carry no reason codes, so they can't drive cause
    attribution, MTTR, or the recovery queue."""
    ss = st.session_state
    options = [core.DEF_SRC_SYNTHETIC]
    if ss.get("deferment_upload"):
        options.append(core.DEF_SRC_UPLOAD)
    if ss.get("data_source") not in options:
        ss["data_source"] = core.DEF_SRC_SYNTHETIC
    if len(options) == 1:
        # Only the synthetic source is available — no radio needed; state is set.
        return ss["data_source"]
    st.radio(
        "Loss-accounting source", options, key="data_source", horizontal=True,
        format_func=lambda k: LOSS_SOURCE_LABELS.get(k, k),
        help="Synthetic: modeled fleet with reason-coded events + ground truth "
             "(powers cause attribution, MTTR, the recovery queue, and the "
             "classifier eval). Your upload: a tidy monthly production CSV from the "
             "Data page (real quantity, cause N/A — public monthly data has no "
             "reason codes).")
    return ss["data_source"]


@st.cache_data(show_spinner=False)
def intervention_cost(intervention: str) -> float:
    """All-in cost of an intervention from the AFE component's cost database
    (NaN if the intervention isn't priced — rendered as '—')."""
    try:
        return float(core.afe_cost_db.cost_rollup(intervention)["total"])
    except Exception:  # noqa: BLE001
        return float("nan")


def split_opportunities(action: "pd.DataFrame") -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Split the action tier into value-accretive opportunities vs an at-risk watch
    list. An "opportunity" is a well whose recommended intervention has a POSITIVE
    risk-weighted NPV (worth doing now); a well with a non-positive risked NPV is on
    watch — the failure signature is there but intervening now destroys value, so
    the action is to monitor and re-rank as risk climbs, not to spend capital."""
    pos = action["est_risked_npv"] > 0
    return action[pos].reset_index(drop=True), action[~pos].reset_index(drop=True)


def loss_badge(source: str) -> tuple[str, str]:
    """(kind, detail) for theme.data_badge on the active Loss Accounting source."""
    if source == core.DEF_SRC_SYNTHETIC:
        return ("synthetic", "Modeled fleet with reason-coded events + known ground "
                             "truth (~92% classifier eval).")
    if source == core.DEF_SRC_UPLOAD:
        return ("real", "Your uploaded monthly production CSV — parsed in memory "
                        "for this session only, nothing stored server-side.")
    return ("real", "Colorado ECMC (COGCC) public monthly records — DJ Basin "
                    "Niobrara/Codell horizontals (Weld County). Downtime from "
                    "days-produced; cause attribution N/A (no public reason codes).")


@st.cache_data(show_spinner="Computing deferment…")
def deferment_data(source_token: str, price: float):
    """(fleet, events_classified, daily) for a Loss Accounting source token:
    ``real_co`` | ``synthetic`` | ``upload::<sha1>``."""
    if source_token == core.DEF_SRC_SYNTHETIC:
        return core.load_deferment_synthetic(price_per_bbl=price)
    if source_token.startswith("upload::"):
        data = _BYOD_DEF.get(source_token)
        if data is None:
            raise KeyError("uploaded monthly CSV is no longer in memory — please re-upload")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        try:
            tmp.write(data)
            tmp.close()
            return core.load_deferment_real(tmp.name, price_per_bbl=price)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
    return core.load_deferment_real(price_per_bbl=price)


def loss_source_token(source: str) -> str:
    """Map the active source key to a deferment_data cache token."""
    if source == core.DEF_SRC_UPLOAD:
        data = st.session_state.get("deferment_upload")
        if data:
            token = f"upload::{hashlib.sha1(data).hexdigest()}"
            _BYOD_DEF[token] = data
            return token
        return core.DEF_SRC_SYNTHETIC  # bytes gone → fall back to the built-in source
    return source


def loss_is_real(source: str) -> bool:
    """Real monthly sources have real quantities but NO reason codes (cause N/A)."""
    return source != core.DEF_SRC_SYNTHETIC


def loss_context(source: str) -> str:
    return LOSS_SOURCE_LABELS.get(source, source)


# ---- per-well helpers (Well File pages) ---------------------------------------

def alert_for_selected(price: float) -> dict:
    """WellAlert-shaped dict for the globally selected well (real digest alert if
    one fired today, else a synthesized fleet-scan alert)."""
    ensure_state()
    return core.alert_for(st.session_state["well_id"], price_per_bbl=price,
                          alerts=alerts(price))


@st.cache_data(show_spinner="Scoring 30-day ESP failure risk…")
def diagnosis(well_id: str, price: float) -> dict:
    a = core.alert_for(well_id, price_per_bbl=price, alerts=alerts(price))
    return core.diagnose(a)


def pinned_pv10_caption() -> None:
    """Honest note where chain NPV figures surface: the AFE component's certified
    economics discount at 10% (PV10); the deck's discount slider does not re-rate
    the chain math (it would silently diverge from the components)."""
    disc = float(st.session_state.get("discount", 0.10))
    if abs(disc - 0.10) > 1e-9:
        st.caption(f"Deck discount is set to {disc:.0%}, but chain economics are "
                   "computed by the AFE component's certified PV10 kernel — NPV "
                   "figures on this page discount at 10%.")
