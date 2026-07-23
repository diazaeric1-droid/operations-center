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
import fleet_registry
import theme

# Canonical per-lift diagnostic channels for the per-well drill-downs, shared by
# Surveillance and Well 360 so the two pages can NEVER show a different channel set
# (or color) for the same lift type: (column, label, color).
LIFT_CHANNELS: dict[str, list[tuple[str, str, str]]] = {
    "Gas lift": [("gas_inj_mcfd", "Lift-gas injection (MCF/d)", theme.TEAL),
                 ("casing_pressure_psi", "Casing pressure (psi)", theme.PURPLE),
                 ("tubing_pressure_psi", "Tubing pressure (psi)", theme.BLUE)],
    "ESP": [("intake_pressure_psi", "Intake pressure (psi)", theme.PURPLE),
            ("motor_temp_f", "Motor temp (°F)", theme.RED),
            ("motor_amps", "Motor amps (A)", theme.GREEN),
            ("current_imbalance_pct", "Current imbalance (%)", theme.AMBER)],
    "Rod pump": [("runtime_pct", "Runtime (%)", theme.GREEN),
                 ("motor_amps", "Motor load (A)", theme.AMBER)],
    # Flowing wells have no downhole pump; downhole pressure is the one diagnostic
    # the synthetic SCADA carries (the production streams do the rest of the review).
    "Flowing": [("intake_pressure_psi", "Downhole pressure (psi)", theme.PURPLE)],
}

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
    # per-well NRI session overrides (Data page editor) + the GROSS/NET display toggle
    "nri_overrides": {},          # {well_id: nri} — session-only, wins over the registry
    "net_view": False,            # False = gross (8/8); True = net (per-well NRI)
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


@st.cache_data(show_spinner=False)
def early_warning_flags(token: str) -> pd.DataFrame:
    """Deep-drift flags for the active SCADA fleet (the LSTM autoencoder in dl/).

    The single Streamlit-cached entry point shared by Surveillance, the Morning
    Brief, the Optimization Board, and the Recovery Queue. Returns an EMPTY frame when
    the optional torch extras / trained model are absent, so every caller can use
    it unconditionally and simply skip its section when the result is empty —
    keeping the deployed app fully decoupled from the DL stack. Columns:
    well, score, driver, maxz, flagged, alarm, deep_only.
    """
    try:
        from dl import score as dl_score
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    if not dl_score.model_ready():
        return pd.DataFrame()
    return dl_score.flag_table(dl_score.score_fleet_latest(fleet_for_token(token)))


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
    signal. The Optimization Board ranks on the ESP alert feed, which carries no deferred
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


def opportunity_signal(frame: "pd.DataFrame") -> "pd.Series":
    """Boolean mask of wells with a REAL trigger to act: actively deferring production
    OR in the fleet's OWN elevated-risk quartile (a robust fleet-relative cut on the
    now-calibrated ESP score, trained on this fleet's labeled faults). A
    healthy well whose now-cheaper, lift-correct intervention merely pencils is NOT an
    opportunity — it needs a signal first. This gate keeps the opportunity count honest
    once interventions are lift-aware (rod-pump workovers, gas-lift jobs) and therefore
    cheaper than a default ESP swap."""
    deferred = (frame["deferred_bopd"] > 0) if "deferred_bopd" in frame else False
    if "failure_risk_30d" in frame and len(frame):
        thresh = float(frame["failure_risk_30d"].quantile(0.75))
        # Fleet-relative quartile OR the absolute calibrated band. The absolute floor
        # matters on a heavily-impaired fleet, where q75 floats up to ~0.9 and would
        # otherwise leave 0.5–0.9 wells in the "stable / reads healthy" tier purely for
        # sitting below the quartile line — now that the score is calibrated, a ≥50%
        # failure probability is a signal regardless of the fleet's shape.
        elevated = ((frame["failure_risk_30d"] >= thresh)
                    | (frame["failure_risk_30d"] >= core.ELEVATED_RISK_ABS_30D))
    else:
        elevated = False
    return deferred | elevated


def split_opportunities(action: "pd.DataFrame") -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Split the action tier into value-accretive opportunities vs an at-risk watch
    list. An "opportunity" is a well that has a real trigger (deferring production or
    elevated fleet-relative risk) AND a positive risk-weighted NPV; everything else
    with a signal is on watch (monitor and re-rank, don't spend capital yet)."""
    signal = opportunity_signal(action)
    pos = action["est_risked_npv"] > 0
    return (action[signal & pos].reset_index(drop=True),
            action[~(signal & pos)].reset_index(drop=True))


def triage_tiers(board: "pd.DataFrame") -> tuple["pd.DataFrame", "pd.DataFrame",
                                                 "pd.DataFrame"]:
    """Three operating tiers off the enriched board (needs the real deferred column
    from ``board_with_deferred``):

    * **opportunities** — a real trigger (deferring production OR elevated
      fleet-relative risk) AND a positive risk-weighted NPV: value-accretive now.
    * **watch** — has a trigger but the intervention doesn't pay yet (non-positive
      NPV): monitor and re-rank, don't spend capital.
    * **stable** — no trigger (or explicitly no-action): nothing to do — the bulk of a
      healthy fleet. A cheap intervention that happens to pencil is NOT enough to make
      a no-signal well an opportunity; these wells read low on the calibrated ESP score.
    """
    no_action = board["recommended_intervention"] == "no_action"
    signal = opportunity_signal(board)
    pos = board["est_risked_npv"] > 0
    opportunities = board[~no_action & signal & pos].reset_index(drop=True)
    watch = board[~no_action & signal & ~pos].reset_index(drop=True)
    stable = board[no_action | ~signal].reset_index(drop=True)
    return opportunities, watch, stable


@st.cache_data(show_spinner=False)
def down_well_set(token: str) -> set:
    """Wells currently down / shut-in on the active SCADA fleet (Restore-tier routing)."""
    return core.down_wells(fleet_for_token(token))


def restore_tier(board: "pd.DataFrame",
                 down_wells: set) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Split currently-down / shut-in wells out of the ranked board into a RESTORE queue
    BEFORE the opportunity/watch/stable partition. A shut-in well is a restore-first job,
    not a priced intervention — leaving it in the board makes it show as a "$X gas-lift
    opportunity" while its Well 360 says "restore production first" (the two disagreeing).
    Returns (restore, remaining). Pure display routing — the certified ranking columns are
    untouched, so rank_fleet parity with pe-pipeline is unaffected."""
    if board is None or getattr(board, "empty", True) or not down_wells:
        empty = board.iloc[0:0] if board is not None else board
        return empty, board
    mask = board["well_id"].astype(str).isin({str(w) for w in down_wells})
    return board[mask].reset_index(drop=True), board[~mask].reset_index(drop=True)


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


# ---- per-well NRI + GROSS/NET convention (PE feedback round 1) ----------------

def well_nri(well_id: str, overrides: dict | None = None) -> float:
    """Per-well NRI: the session override from the Data-page editor if one exists,
    else the registry's deterministic default. ``overrides`` is injectable for
    headless tests; views let it default to session state."""
    if overrides is None:
        overrides = st.session_state.get("nri_overrides") or {}
    try:
        v = overrides.get(str(well_id))
        return float(v) if v is not None else float(fleet_registry.nri_for(str(well_id)))
    except Exception:  # noqa: BLE001
        return float(st.session_state.get("nri", 0.80))


def nri_map(well_ids, overrides: dict | None = None) -> dict[str, float]:
    """{well_id: per-well NRI} for a set of wells (override → registry default)."""
    return {str(w): well_nri(str(w), overrides) for w in well_ids}


def gross_net_toggle() -> bool:
    """The shared GROSS (8/8) vs NET (per-well NRI) display toggle — one session key
    (``net_view``) so the choice follows the user across roll-up pages. Returns True
    when NET is active."""
    st.toggle(
        "Net to interest (× per-well NRI)", key="net_view",
        help="OFF = gross (8/8 working interest), the standard base-management "
             "convention. ON = net entitlement: each well's volumes/dollars × its "
             "OWN NRI (registry default, editable per well on Sources & BYOD). "
             "The certified ranking economics still use the sidebar deck NRI.")
    return bool(st.session_state.get("net_view"))


# ---- CTB / lift / basin / county fleet filters (PE feedback round 1) -----------

def filter_ids_by_meta(well_ids, ctb=(), lift=(), basin=(), area=()) -> list[str]:
    """Pure filter: the well ids whose registry metadata passes every ACTIVE filter
    (an empty selection means 'no filter' on that dimension). Streamlit-free."""
    out = []
    for w in well_ids:
        m = fleet_registry.get(str(w))
        if ctb and m.ctb not in ctb:
            continue
        if lift and m.lift not in lift:
            continue
        if basin and m.basin not in basin:
            continue
        if area and m.area not in area:
            continue
        out.append(str(w))
    return out


def fleet_filter_controls(key_prefix: str, well_ids) -> list[str]:
    """Render the shared CTB / lift / basin / county multiselects (empty = all) and
    return the well ids passing every active filter. Defaults reproduce the
    unfiltered fleet exactly."""
    metas = [fleet_registry.get(str(w)) for w in well_ids]
    cols = st.columns(4)
    ctb = cols[0].multiselect("CTB (tank battery)", sorted({m.ctb for m in metas}),
                              key=f"{key_prefix}_f_ctb")
    lift = cols[1].multiselect("Lift type", sorted({m.lift for m in metas}),
                               key=f"{key_prefix}_f_lift")
    basin = cols[2].multiselect("Basin", sorted({m.basin for m in metas}),
                                key=f"{key_prefix}_f_basin")
    area = cols[3].multiselect("County", sorted({m.area for m in metas}),
                               key=f"{key_prefix}_f_area")
    return filter_ids_by_meta(well_ids, ctb=set(ctb), lift=set(lift),
                              basin=set(basin), area=set(area))


# ---- cross-page drill-through (PE feedback round 1) ----------------------------

def jump_to_well(page_title: str, well_id: str) -> bool:
    """Set the global selected well and switch to ``page_title`` (e.g. Surveillance,
    whose drill-down pre-syncs from ``well_id``). Returns False when the page object
    isn't registered (per-view AppTest harness) so callers can render a fallback
    caption instead. NOTE: ``st.switch_page`` ends the current run by raising
    Streamlit's rerun control-flow exception — never wrap this call in a bare
    ``except Exception``."""
    import views
    page = views.PAGE_OBJECTS.get(page_title)
    if page is None:
        return False
    # Park the target for app.py's top-of-run handoff — writing well_id here would
    # raise StreamlitAPIException (the sidebar selectbox owns that key and has
    # already rendered this run).
    st.session_state["_well_jump"] = str(well_id)
    st.switch_page(page)
    return True


def handle_row_jump(event, source_frame, sentinel_key: str,
                    page_title: str = "Surveillance") -> None:
    """Shared single-row-selection → drill-through handler for st.dataframe tables.
    Reads the selected row POSITION from the selection event, maps it to
    ``source_frame['well_id']`` (never the formatted display frame), dedupes with a
    session sentinel (dataframe selection state persists across reruns), and jumps.
    None-safe for the AppTest harness."""
    try:
        rows = list(event.selection.rows) if event is not None else []
    except Exception:  # noqa: BLE001
        rows = []
    if not rows:
        # Deselect (empty selection event) — clear the sentinel so the same row
        # can be re-selected later.
        st.session_state.pop(sentinel_key, None)
        return
    if source_frame is None or not len(source_frame):
        return
    pos = rows[0]
    if pos >= len(source_frame):
        return
    wid = str(source_frame.iloc[pos]["well_id"])
    if st.session_state.get(sentinel_key) == wid:
        return  # stale selection re-fired on return to this page — ignore
    st.session_state[sentinel_key] = wid
    if not jump_to_well(page_title, wid):
        st.caption(f"Selected **{wid}** — open **{page_title}** to review it "
                   "(the well is now the global selection).")


# ---- downtime context for recommendations (PE feedback round 1) ----------------

@st.cache_data(show_spinner=False)
def ongoing_event_days(token: str, price: float) -> dict[str, int]:
    """{well_id: running duration (days)} of OPEN downtime events (NEW/ONGOING, not
    acknowledged) from the event state machine — replayed WITHOUT the demo outage
    injection so recommendations are never gated on a demo artifact."""
    return {str(e.well_id): int(e.duration_days)
            for e in replay_events(token, price, False)
            if e.state in ("NEW", "ONGOING")
            and not getattr(e, "acknowledged", False)}


NO_CHOKE_CAPTION = (
    "The SCADA set carries no choke-position channel, so choke moves can't be "
    "separated from reservoir/lift losses here — recommendations are gated on "
    "downtime state (ongoing-event context) instead of choke data.")


# ---- unified morning-brief list (PE feedback round 1) --------------------------

BRIEF_STATUS_ORDER = {"NEW": 0, "ONGOING": 1, "RESOLVED": 2}


def unified_brief_frame(events, anomalies, nri_by_well: dict,
                        net_view: bool, price: float) -> pd.DataFrame:
    """ONE ranked list for the Morning Brief: state-machine events (NEW / ONGOING /
    RESOLVED) plus active scan anomalies on wells with no open event, ordered by
    today's BO/day impact (net of per-well NRI when ``net_view``), status as the
    tiebreaker. When a well has both an open event and a scan anomaly, the EVENT's
    deferral is used (the two windows differ slightly). Pure / streamlit-free."""
    rows: list[dict] = []
    open_wells: set[str] = set()
    for e in events or []:
        if getattr(e, "acknowledged", False):
            continue
        state = str(e.state)
        if state not in BRIEF_STATUS_ORDER:
            continue
        wid = str(e.well_id)
        gross = (float(getattr(e, "last_deferred_bopd", 0.0) or 0.0)
                 if state in ("NEW", "ONGOING") else 0.0)
        rows.append({"well_id": wid, "status": state,
                     "kind": str(e.event_type), "days": int(e.duration_days),
                     "gross_bopd": round(gross, 1),
                     "cum_bbl": round(float(getattr(e, "deferred_bopd", 0.0) or 0.0), 0)})
        if state in ("NEW", "ONGOING"):
            open_wells.add(wid)
    for a in anomalies or []:
        wid = str(a.well_id)
        if getattr(a, "acknowledged", False) or wid in open_wells:
            continue
        rows.append({"well_id": wid, "status": "NEW", "kind": str(a.category),
                     "days": None,
                     "gross_bopd": round(float(getattr(a, "deferred_bopd", 0.0) or 0.0), 1),
                     "cum_bbl": None})
    if not rows:
        return pd.DataFrame(columns=["well_id", "status", "kind", "days",
                                     "gross_bopd", "net_bopd", "rank_bopd",
                                     "usd_per_day", "cum_bbl"])
    df = pd.DataFrame(rows)
    df["net_bopd"] = (df["well_id"].map(lambda w: nri_by_well.get(str(w), 0.8))
                      * df["gross_bopd"]).round(1)
    df["rank_bopd"] = df["net_bopd"] if net_view else df["gross_bopd"]
    df["usd_per_day"] = (df["rank_bopd"] * price).round(0)
    df["_ord"] = df["status"].map(BRIEF_STATUS_ORDER)
    df = df.sort_values(["rank_bopd", "_ord", "well_id"],
                        ascending=[False, True, True], kind="mergesort")
    return df.drop(columns="_ord").reset_index(drop=True)


def pinned_pv10_caption() -> None:
    """Honest note where chain NPV figures surface: the AFE component's certified
    economics discount at 10% (PV10); the deck's discount slider does not re-rate
    the chain math (it would silently diverge from the components)."""
    disc = float(st.session_state.get("discount", 0.10))
    if abs(disc - 0.10) > 1e-9:
        st.caption(f"Deck discount is set to {disc:.0%}, but chain economics are "
                   "computed by the AFE component's certified PV10 kernel — NPV "
                   "figures on this page discount at 10%.")
