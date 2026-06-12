"""Streamlit fleet explorer for the Daily Production Digest.

Multipage (``st.navigation`` + ``st.Page``): a Fleet Overview page (trends,
snapshot KPIs, the morning brief, the deferred-$ offender bar, the lost-production
ledger, and a sortable fleet table) plus one drill-down page per well (its own
oil/gas/water + SCADA diagnostics, a health note, and the anomaly economics).

Detection stays deterministic; the brief is BYOK-optional (a deterministic brief
renders with no API key). Heavy loads are cached on string args.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import date
from functools import partial
from pathlib import Path

# Ensure repo root is on sys.path so `src.*` imports work on Streamlit Cloud, and
# the demo dir so the vendored `theme` / `fleet_registry` resolve regardless of cwd
# (Streamlit adds the entrypoint dir at runtime; AppTest / other contexts may not).
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
for _p in (str(REPO_ROOT), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- Self-heal stale bytecode / module cache (Streamlit Cloud) --------------
# Streamlit reuses the container across redeploys; a cached .pyc or already-imported
# OLD module can lack symbols added in a newer commit, surfacing as a startup
# ImportError for a name that exists in the source. Purge src/ bytecode + evict
# cached src modules so every submodule reloads from CURRENT source (no-op when clean).
import shutil as _shutil
for _pycache in (REPO_ROOT / "src").rglob("__pycache__"):
    _shutil.rmtree(_pycache, ignore_errors=True)
for _name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_name]

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import fleet_registry
# --- warm-container module self-heal (vendored top-level modules) -----------
# Streamlit Cloud reuses the container across redeploys; a cached OLD `theme` /
# `fleet_registry` in sys.modules (or a stale .pyc) lacks symbols added in a newer
# commit -> AttributeError (e.g. theme.how_to). Drop their bytecode + evict the cached
# modules so the imports below reload from the CURRENT commit's source.
import shutil as _sh_heal
_sh_heal.rmtree(Path(__file__).resolve().parent / "__pycache__", ignore_errors=True)
for _stale in ("theme", "fleet_registry"):
    sys.modules.pop(_stale, None)

import theme
from src import __version__
from src.anomaly_detector import (
    _expected_decline_rate,
    load_acknowledgements,
    scan_fleet,
)
from src.data_loader import (
    BYOD_REQUIRED_COLUMNS,
    build_fleet_table,
    fleet_summary,
    fleet_template_csv,
    load_fleet,
    load_fleet_from_csv,
    production_variance_pct,
    slice_window,
    validate_scada_columns,
)
from src.event_store import (
    NEW,
    ONGOING,
    RESOLVED,
    EventStore,
    update_events,
)
from src.ledger import build_ledger
from src.representative import classify_representative


DATA_DIR = REPO_ROOT / "data" / "synthetic" / "fleet"
BRIEFS_DIR = REPO_ROOT / "briefs"
ACK_PATH = REPO_ROOT / "acknowledged.yml"

# Data-source selector (sidebar). Synthetic (the committed demo fleet) is default;
# the BYOD option lets a user drop their own fleet SCADA CSV — same scan + brief.
SRC_SYNTHETIC = "Synthetic demo fleet"
SRC_UPLOAD = "Upload your own fleet SCADA CSV"

# Token used to key the cached heavy loads for the synthetic source (the on-disk
# fleet dir). The BYOD source keys on a content hash instead (see _resolve_fleet).
SYNTHETIC_TOKEN = f"synthetic::{DATA_DIR}"

# Time-range control: label -> trailing-window length in days (None = Lifetime).
RANGE_OPTIONS: dict[str, int | None] = {
    "7D": 7, "30D": 30, "3mo": 90, "6mo": 180, "1Y": 365, "Lifetime": None,
}
DEFAULT_RANGE = "30D"

# Event-lifecycle replay: how many trailing as-of days to push through the event
# state machine (each day re-scans the whole fleet, ~18 ms/day, so this is bounded
# for responsiveness; the recent window is where every multi-day signature lives).
REPLAY_DAYS = 60

# Demo-outage injection (Task B). The committed synthetic fleet injects faults only
# on the FINAL day, so a replay alone won't show a clean multi-day ONGOING *rate*
# event with a growing cumulative deferral. This optional toggle splices a sustained
# rate outage into one otherwise-healthy well — held down (no recovery) over the tail
# of the window — so the NEW→ONGOING lifecycle + cumulative deferred bbl/$ is visibly
# demonstrable. It mutates only an in-memory copy; the committed CSVs are untouched.
DEMO_OUTAGE_WELL = "well_001"     # a healthy (non-seeded) well in the demo fleet
DEMO_OUTAGE_LEN = 12              # consecutive down days ending on the latest day
DEMO_OUTAGE_FRACTION = 0.55       # held at ~55% of pre-event baseline (a ~45% loss)


# ---- cached heavy loads (string tokens so they hash/cache cleanly) ----------
# Every heavy load is keyed on a string ``token`` that identifies the data source:
# the synthetic on-disk fleet (``SYNTHETIC_TOKEN``) or a bring-your-own-data upload
# (``byod::<sha1>``). The uploaded CSV bytes are stashed in session_state under that
# token so the cached parse below is a pure function of the token — nothing about
# the upload is written into the repo, and each distinct upload caches separately.

_BYOD_BYTES: dict[str, bytes] = {}  # token -> uploaded CSV bytes (process-local)


@st.cache_data(show_spinner=False)
def _byod_fleet_cached(token: str) -> dict:
    """Parse an uploaded fleet CSV (looked up by ``token``) via the EXISTING loader.

    Reuses ``load_fleet_from_csv`` — same schema check + per-well date parse/sort as
    the on-disk synthetic loader — so detection / scan / brief see identical frames.
    Cached on the token (a content hash) so a re-run doesn't re-parse."""
    import io
    data = _BYOD_BYTES.get(token)
    if data is None:  # cache survived but the bytes did not (rare warm-container case)
        raise KeyError("uploaded data is no longer in memory — please re-upload")
    return load_fleet_from_csv(io.BytesIO(data))


def _fleet_for_token(token: str) -> dict:
    """Resolve a source token to a fleet dict (synthetic dir-load or BYOD parse)."""
    if token.startswith("byod::"):
        return _byod_fleet_cached(token)
    return _load_fleet_cached(str(DATA_DIR))


@st.cache_data(show_spinner=False)
def _load_fleet_cached(data_dir: str) -> dict:
    """Cache the expensive per-well CSV load. Returns dict[str, DataFrame]."""
    return load_fleet(data_dir)


@st.cache_resource(show_spinner=False)
def _scan_fleet_cached(token: str, ack_path: str) -> list:
    """Cache the deterministic fleet scan over the latest day per well.

    Uses cache_resource (not cache_data) because it returns a list of `Anomaly`
    dataclass objects — Streamlit's cache_data serializer rejects custom classes on
    Python 3.14 / newer Streamlit. The scan result is read-only here, so sharing the
    cached object across sessions is safe."""
    fleet = _fleet_for_token(token)
    acknowledged = load_acknowledgements(ack_path)
    return scan_fleet(fleet, acknowledged=acknowledged)


@st.cache_data(show_spinner=False)
def _build_ledger_cached(token: str, ack_path: str, window_days: int = 30):
    """Cache the day-by-day ledger replay over a trailing window."""
    fleet = _fleet_for_token(token)
    acknowledged = load_acknowledgements(ack_path)
    return build_ledger(fleet, window_days=window_days, acknowledged=acknowledged)


@st.cache_data(show_spinner=False)
def _representative_fleet_cached(token: str, window_days: int | None) -> pd.DataFrame:
    """Per-well representative-vs-anomalous data-quality summary over the window.

    For each well, classify which oil-rate points are representative for decline /
    type-curve trending (vs shut-ins / zero days, metering dropouts, gross outliers)
    and report the representative share + excluded count. Deterministic, no API key."""
    fleet = _fleet_for_token(token)
    rows = []
    for well_id in sorted(fleet):
        df = fleet[well_id]
        if df is None or not len(df):
            continue
        win = slice_window(df, window_days)
        try:
            res = classify_representative(win, rate_col="bopd")
        except Exception:
            continue
        s = res.summary
        top_reason = max(s.reason_counts, key=s.reason_counts.get) if s.reason_counts else "—"
        rows.append({
            "Well": well_id,
            "Representative %": s.representative_pct,
            "Points": s.n_points,
            "Excluded": s.n_excluded,
            "Top exclusion reason": top_reason,
        })
    return pd.DataFrame(rows)


# ---- event state-machine replay (Ongoing & Resolved lifecycle) -------------

def _inject_demo_outage(fleet: dict) -> dict:
    """Return a shallow fleet copy with a sustained multi-day rate outage spliced
    into ``DEMO_OUTAGE_WELL`` so the NEW→ONGOING lifecycle is demonstrable.

    The committed synthetic fleet only injects faults on the FINAL day, so a raw
    replay produces no clean multi-day ONGOING *rate* event. Here we hold the last
    ``DEMO_OUTAGE_LEN`` days of one healthy well at ``DEMO_OUTAGE_FRACTION`` of its
    pre-event baseline (and scale gas with it for physical consistency) — held down,
    not recovered, so on the latest as-of day it reads ONGOING at ~day N with a
    growing cumulative deferral. Only the in-memory copy is mutated; the committed
    CSVs are never touched. A no-op if the target well isn't present (e.g. BYOD)."""
    df = fleet.get(DEMO_OUTAGE_WELL)
    if df is None or len(df) < DEMO_OUTAGE_LEN + 8:
        return fleet
    out = dict(fleet)
    df2 = df.copy()
    n = len(df2)
    start = n - DEMO_OUTAGE_LEN
    baseline = float(df2["bopd"].iloc[start - 7:start].mean())  # 7 days pre-event
    target = baseline * DEMO_OUTAGE_FRACTION
    idx = df2.index[start:]
    df2.loc[idx, "bopd"] = target
    if "gas_mcfd" in df2.columns and baseline > 0:
        # Keep GOR roughly constant: drop gas in the same proportion as oil.
        df2.loc[idx, "gas_mcfd"] = df2.loc[idx, "gas_mcfd"] * (target / baseline)
    out[DEMO_OUTAGE_WELL] = df2
    return out


@st.cache_resource(show_spinner=False)
def _replay_events_cached(token: str, ack_path: str, inject_demo: bool,
                          replay_days: int = REPLAY_DAYS) -> list:
    """Replay the fleet's recent history through the persistent event state machine
    and return the live events on the latest as-of day (NEW / ONGOING / RESOLVED).

    This is the SAME code path ``scheduler.run`` / the brief writer drive: each
    trailing as-of day, in order, is pushed through ``update_events`` against an
    in-memory ``EventStore`` so events open (NEW), persist (ONGOING) even after the
    stateless scan goes quiet, and resolve (RESOLVED) on recovery. The store is
    ``:memory:`` and per-call, so the demo is stateless across users and writes
    nothing into the repo. Cached on (token, inject_demo) so it doesn't recompute
    every rerun.

    Returns the same money-first event list ``render_brief_markdown`` consumes, so
    the table matches the brief's *Ongoing & Resolved Events* section. Uses
    cache_resource (not cache_data) because the value is a list of ``Event``
    dataclasses, read-only here — the same rationale as ``_scan_fleet_cached``."""
    fleet = _fleet_for_token(token)
    if inject_demo:
        fleet = _inject_demo_outage(fleet)
    acknowledged = load_acknowledgements(ack_path)

    # A shared calendar spine: the union of well dates, ascending. The synthetic
    # fleet shares one spine; for ragged BYOD we still iterate the global tail.
    all_dates = sorted({d for df in fleet.values() if df is not None and len(df)
                        for d in df["date"]})
    if not all_dates:
        return []
    spine = all_dates[-replay_days:] if replay_days and replay_days > 0 else all_dates

    store = EventStore(":memory:")
    live: list = []
    try:
        for as_of_ts in spine:
            as_of = pd.Timestamp(as_of_ts).date().isoformat()
            # Feed each well its history UP TO as_of (history-to-date), exactly like
            # backtest_v2 / scheduler across consecutive mornings.
            sliced = {wid: df[df["date"] <= as_of_ts]
                      for wid, df in fleet.items() if df is not None and len(df)}
            sliced = {wid: d for wid, d in sliced.items() if len(d)}
            live = update_events(store, sliced, as_of=as_of, acknowledged=acknowledged)
    finally:
        store.close()
    return live


def _bootstrap_fleet() -> None:
    """Generate synthetic fleet data on first run — or regenerate if the on-disk data
    predates the current schema (a redeploy reusing an old container without the
    `gas_mcfd` channel), so the loader never trips on a stale column set."""
    existing = sorted(DATA_DIR.glob("well_*.csv"))
    stale = False
    if existing:
        try:
            stale = "gas_mcfd" not in pd.read_csv(existing[0], nrows=1).columns
        except Exception:
            stale = True
    if not existing or stale:
        with st.status("First-time setup: generating synthetic fleet…", expanded=False):
            subprocess.run(
                [sys.executable, str(REPO_ROOT / "data" / "synthetic" / "generate_fleet.py")],
                check=True,
            )


# ---- data-source selection (synthetic vs bring-your-own SCADA) -------------

def _byod_caption() -> None:
    """Document the BYOD schema + the privacy guarantee, with a template download."""
    st.caption(
        "**Required columns** (one row per well per day): "
        f"`{'`, `'.join(BYOD_REQUIRED_COLUMNS)}`. "
        "`date` is YYYY-MM-DD; `well_id` groups rows into wells; rates are daily "
        "(`bopd` oil, `bfpd` gross fluid, `gas_mcfd` gas), plus ESP/well diagnostics "
        "(`intake_pressure_psi`, `motor_temp_f`, `motor_amps`, `runtime_pct`). "
        "Extra columns are ignored. **Nothing is stored server-side** — the file is "
        "parsed in memory for this session only and never written to disk or logged.")
    st.download_button(
        "⬇ Download a template CSV", data=fleet_template_csv(),
        file_name="fleet_scada_template.csv", mime="text/csv",
        help="A header row in the required schema + two example daily rows for one well.")


def _resolve_fleet() -> tuple[str, bool]:
    """Render the sidebar data-source control and return ``(token, is_byod)``.

    Synthetic → the committed demo fleet (``SYNTHETIC_TOKEN``). BYOD → the user's
    uploaded fleet SCADA CSV, validated up front against ``BYOD_REQUIRED_COLUMNS``;
    on a missing/invalid file we show a clear ``st.error`` and ``st.stop()`` (never
    crash). The valid upload's bytes are cached under a content-hash token so every
    cached load (scan / ledger / events / data-quality) runs the SAME path as the
    synthetic source — only the source token differs.
    """
    with st.sidebar:
        st.header("Data source")
        source = st.radio(
            "Fleet SCADA source", [SRC_SYNTHETIC, SRC_UPLOAD], index=0,
            key="data_source",
            help="Synthetic = the committed modeled Permian fleet (known ground "
                 "truth). Upload your own = drop a fleet SCADA CSV in the schema "
                 "below and get the same scan, brief, ledger, and event lifecycle "
                 "on your data. Nothing is stored server-side.")
        if source != SRC_UPLOAD:
            return SYNTHETIC_TOKEN, False

        uploaded = st.file_uploader("Fleet SCADA CSV", type=["csv"], key="byod_csv")
        _byod_caption()

    if uploaded is None:
        st.info("Upload a fleet SCADA CSV to analyze your own wells, or switch the "
                "**Data source** back to the synthetic demo fleet in the sidebar.")
        st.stop()

    # Validate columns up front against the raw header — a clear error beats a parse
    # traceback. Read once; reuse the bytes for both the check and the cached parse.
    import hashlib
    import io
    data = uploaded.getvalue()
    try:
        head = pd.read_csv(io.BytesIO(data), nrows=0)
    except Exception as exc:
        st.error(f"Could not read that file as CSV: {exc}")
        st.stop()
    missing = validate_scada_columns(head)
    if missing:
        st.error(
            "Uploaded CSV is missing required column(s): "
            f"**{', '.join(missing)}**.\n\nRequired columns are: "
            f"`{'`, `'.join(BYOD_REQUIRED_COLUMNS)}`. "
            "Download the template above for the exact schema.")
        st.stop()

    token = f"byod::{hashlib.sha1(data).hexdigest()}"
    _BYOD_BYTES[token] = data
    # Surface any per-well parse problem (e.g. an unparseable date) as a clean error.
    try:
        fleet = _byod_fleet_cached(token)
    except Exception as exc:
        st.error(f"Could not load the uploaded fleet: {exc}")
        st.stop()
    if not fleet:
        st.error("No wells found in the uploaded CSV — check the `well_id` column.")
        st.stop()
    return token, True


# ---- shared helpers --------------------------------------------------------

def _time_range_control(context: str) -> int | None:
    """Render the shared trailing-window segmented control and return its window
    length in days (None = Lifetime). State is namespaced per ``context`` so the
    overview and each well page remember their own selection across reruns."""
    key = f"range_{context}"
    # Seed the selection once in session_state instead of passing default=. Passing
    # BOTH default= and key= to a selection widget is a known cross-version footgun:
    # on some Streamlit builds the default re-asserts on rerun and snaps the choice
    # back (the "stuck on 30D" symptom). With the value owned by session_state, every
    # user click persists and the charts/KPIs re-slice immediately.
    if key not in st.session_state:
        st.session_state[key] = DEFAULT_RANGE
    label = st.segmented_control(
        "Time range", options=list(RANGE_OPTIONS), key=key,
        help="Slices the trailing window for every chart + KPI on this page.")
    if not label:  # single-mode control can be cleared -> fall back to default
        label = DEFAULT_RANGE
    return RANGE_OPTIONS[label]


def _fleet_daily_totals(fleet: dict, window_days: int | None) -> pd.DataFrame:
    """Per-day fleet totals (oil / gas / water) over the trailing window, indexed
    by date. Water = Σ(bfpd − bopd)."""
    oil: dict = {}
    gas: dict = {}
    water: dict = {}
    for df in fleet.values():
        if not len(df):
            continue
        win = slice_window(df, window_days)
        for _, row in win.iterrows():
            d = row["date"]
            if pd.notna(row.get("bopd")):
                oil[d] = oil.get(d, 0.0) + float(row["bopd"])
                if pd.notna(row.get("bfpd")):
                    water[d] = water.get(d, 0.0) + float(row["bfpd"]) - float(row["bopd"])
            if pd.notna(row.get("gas_mcfd")):
                gas[d] = gas.get(d, 0.0) + float(row["gas_mcfd"])
    idx = sorted(set(oil) | set(gas) | set(water))
    return pd.DataFrame({
        "date": idx,
        "oil": [oil.get(d, float("nan")) for d in idx],
        "gas": [gas.get(d, float("nan")) for d in idx],
        "water": [water.get(d, float("nan")) for d in idx],
    })


def _line(x, y, name, color, y_title, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                             line=dict(color=color, width=2)))
    fig.update_layout(title=title, yaxis_title=y_title)
    return theme.style_fig(fig, height=300, legend=False)


def _oil_with_representative(win: pd.DataFrame) -> None:
    """Oil-rate trend with non-representative points (excluded from trending) marked.

    Reuses ``classify_representative`` to flag shut-ins / zero days, metering dropouts,
    and gross outliers vs a robust decline-aware trend — the points a decline / type
    curve should be fit WITHOUT. Healthy points are the blue line; excluded points get
    a distinct red ✕. Guarded so a malformed window never breaks the page."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win["date"], y=win["bopd"], mode="lines",
                             name="Oil (BOPD)", line=dict(color=theme.BLUE, width=2)))
    excl_caption = None
    try:
        res = classify_representative(win, rate_col="bopd")
        mask = ~res["representative"].to_numpy()
        if mask.any():
            ex_dates = win["date"].to_numpy()[mask]
            ex_rates = win["bopd"].to_numpy()[mask]
            fig.add_trace(go.Scatter(
                x=ex_dates, y=ex_rates, mode="markers",
                name="Excluded from trending",
                marker=dict(color=theme.RED, size=10, symbol="x",
                            line=dict(width=1.5, color=theme.RED))))
            s = res.summary
            reasons = ", ".join(sorted(s.reason_counts)) if s.reason_counts else "—"
            excl_caption = (f"**{s.n_excluded}** of {s.n_points} points "
                            f"({s.representative_pct:.0f}% representative) excluded from "
                            f"trending — {reasons}.")
    except Exception:
        excl_caption = None  # never break the chart on a data-quality hiccup
    fig.update_layout(title="Oil Rate (BOPD) — Representative vs Excluded Points",
                      yaxis_title="BOPD")
    st.plotly_chart(theme.style_fig(fig, height=300, legend=True), width="stretch")
    if excl_caption:
        st.caption(excl_caption)
    else:
        st.caption("All points are representative for decline / type-curve trending.")
    theme.source_note(
        "Red ✕ marks points excluded from a decline / type-curve fit — shut-in / "
        "zero-rate days, metering dropouts, or gross outliers (|robust z| > 4 vs. a "
        "decline-aware robust trend); the blue line is the on-trend rate.")
    theme.references(["arps", "deferment"])


def _anomaly_for(well_id: str, anomalies: list):
    for a in anomalies:
        if a.well_id == well_id:
            return a
    return None


def _back_to_overview():
    # Link to the registered overview page object when available (set during
    # navigation wiring); fall back to the entrypoint path otherwise.
    target = globals().get("overview")
    try:
        st.page_link(target if target is not None else "app.py",
                     label="← Back to Fleet Overview", icon="📊")
    except Exception:
        pass


# =====================================================================
# PAGE: Fleet overview
# =====================================================================

def render_overview() -> None:
    token, is_byod = _resolve_fleet()
    src_chip = ("your fleet · uploaded", "info") if is_byod else ("synthetic", "info")
    theme.header(
        "Daily Production Digest",
        subtitle="Scheduled AI agent that writes a morning brief for asset teams. "
                 "Built by an ex-OXY / ex-Shell Staff PE.",
        chips=[(f"v{__version__}", "ver"), src_chip, ("scheduled agent", "info")],
    )
    if is_byod:
        theme.data_badge("real", "Your uploaded fleet SCADA — parsed in memory for this "
                                 "session only, nothing stored server-side.")
    else:
        theme.data_badge("synthetic", "Modeled daily SCADA fleet with known ground truth — public production is monthly, not daily.")

    theme.how_to(
        "- **What this is** — a daily production digest over a Permian SCADA fleet: "
        "fleet-wide rate trends, snapshot KPIs, an AI morning brief, the deferred-$ "
        "offenders, the lost-production ledger, and a per-well drill-down.\n"
        "- **Time range** — the segmented control (**7D · 30D · 3mo · 6mo · 1Y · "
        "Lifetime**) re-slices every chart and KPI on the page to that trailing window.\n"
        "- **Oil / Gas / Water tabs** — switch the fleet trend between total oil "
        "(BOPD), gas (MCFD), and water (BWPD).\n"
        "- **Fleet table + per-well pages** — the **📋 Fleet Table** tab is one sortable "
        "row per well; open any well from the **Wells** section in the sidebar to drill "
        "into its own production + SCADA-diagnostic charts and health note.\n"
        "- **Data quality** — the **🧹 Data Quality** view flags the points a decline / "
        "type-curve fit should exclude (shut-ins / zero-rate days, metering dropouts, "
        "gross outliers) so they don't bias the trend — separate from the operational "
        "**🚨 Anomalies** alerts."
    )

    with st.expander(f"🆕 What's New in v{__version__}"):
        st.markdown(
            "- **Upload your own fleet SCADA** — the **Data source** control in the "
            "sidebar now takes a fleet SCADA CSV (`well_id`, `date`, oil/gas/water "
            "rates + ESP diagnostics) and runs the **same** scan, brief, ledger, and "
            "event lifecycle on your data. Columns are validated up front; nothing is "
            "stored server-side (parsed in memory only) and a template is downloadable.\n"
            "- **Ongoing Events tab** — replays the fleet's recent history through the "
            "persistent event state machine (`NEW → ONGOING → RESOLVED`, the same path "
            "the morning brief + scheduler drive) so a confirmed multi-day outage stays "
            "**ONGOING** with a running **duration** + **cumulative deferred bbl/$** "
            "instead of vanishing once it ages out of the detector's window.\n"
            "- **Representative-vs-anomalous data quality** — the **🧹 Data quality** tab "
            "classifies which oil-rate points are usable for decline / type-curve trending "
            "vs which to exclude (shut-ins / zero days, metering dropouts, gross outliers); "
            "per-well oil charts mark the **excluded** points.\n"
            "- **Fleet explorer (multipage)** — a Fleet Overview plus a **drill-down page "
            "per well**, each with its own production + SCADA-diagnostic charts; **Oil / "
            "Gas / Water** fleet trends, a **production-variance** KPI, a **sortable fleet "
            "table**, and the **lost-production ledger** — all retained."
        )

    fleet = _fleet_for_token(token)
    anomalies = _scan_fleet_cached(token, str(ACK_PATH))
    active = [a for a in anomalies if not a.acknowledged]
    anomaly_map = {a.well_id: f"{a.severity} · {a.category}" for a in active}

    window_days = _time_range_control("overview")
    totals = _fleet_daily_totals(fleet, window_days)

    # --- three fleet trend charts (Oil | Gas | Water) -----------------------
    st.subheader("Fleet Production Trend")
    t_oil, t_gas, t_water = st.tabs(["Oil (BOPD)", "Gas (MCFD)", "Water (BWPD)"])
    with t_oil:
        st.plotly_chart(_line(totals["date"], totals["oil"], "Oil", theme.BLUE,
                              "Total BOPD", "Total Fleet Oil Rate (BOPD)"), width="stretch")
    with t_gas:
        st.plotly_chart(_line(totals["date"], totals["gas"], "Gas", theme.AMBER,
                              "Total MCFD", "Total Fleet Gas Rate (MCFD)"), width="stretch")
    with t_water:
        st.plotly_chart(_line(totals["date"], totals["water"], "Water", theme.TEAL,
                              "Total BWPD", "Total Fleet Water Rate (BWPD)"), width="stretch")

    # --- fleet snapshot KPIs incl. production variance ----------------------
    summary = fleet_summary(fleet)
    var_oil = production_variance_pct(totals["oil"].values)
    var_gas = production_variance_pct(totals["gas"].values)
    var_water = production_variance_pct(totals["water"].values)

    st.subheader("Fleet Snapshot")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Wells", summary["well_count"])
    k2.metric("Total BOPD", f"{summary['total_bopd']:,.0f}",
              delta=f"{var_oil:+.1f}% over window")
    k3.metric("Total MCFD", f"{summary['total_gas_mcfd']:,.0f}",
              delta=f"{var_gas:+.1f}% over window")
    bwpd = summary["total_bfpd"] - summary["total_bopd"]
    k4.metric("Total BWPD", f"{bwpd:,.0f}", delta=f"{var_water:+.1f}% over window",
              delta_color="inverse")
    k5, k6, k7 = st.columns(3)
    k5.metric("Water cut", f"{summary['water_cut_pct']:.0f}%")
    k6.metric("Avg runtime", f"{summary['avg_runtime_pct']:.1f}%")
    total_deferred = sum(a.deferred_usd_per_day for a in active)
    k7.metric("Deferred at risk", f"${total_deferred:,.0f}/day")
    theme.source_note(
        "Production variance % = (recent-edge avg − start-edge avg) / start-edge avg "
        "over the selected window (7-day average at each end; positive = rising). "
        "Deferred at risk sums each active anomaly's deferred-$ /day.")

    # --- brief + offenders + ongoing events + fleet table + data quality ----
    BRIEFS_DIR.mkdir(exist_ok=True)
    tab_brief, tab_anom, tab_events, tab_table, tab_quality = st.tabs(
        ["📝 Morning Brief", "🚨 Anomalies", "🔁 Ongoing Events", "📋 Fleet Table",
         "🧹 Data Quality"])

    with tab_brief:
        _brief_panel(fleet, anomalies)

    with tab_anom:
        _anomaly_panel(anomalies, active)

    with tab_events:
        _events_panel(token, is_byod)

    with tab_table:
        st.caption("One row per well over the selected window — sort any column. "
                   "Open a well from the **Wells** section in the sidebar to drill in.")
        table = build_fleet_table(fleet, window_days=window_days,
                                  anomaly_by_well=anomaly_map)
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button("⬇ Download fleet table (CSV)", data=table.to_csv(index=False),
                           file_name="digest_fleet.csv", mime="text/csv")

    with tab_quality:
        _data_quality_panel(token, window_days)

    # --- lost-production ledger ---------------------------------------------
    _ledger_section(token)


def _brief_panel(fleet: dict, anomalies: list) -> None:
    """Brief controls (BYOK + generate) + brief-history selector, in the body."""
    summary = fleet_summary(fleet)
    c1, c2 = st.columns([1, 1])
    with c1:
        byok_key = st.text_input(
            "🔑 Anthropic API key (optional)", type="password",
            help="Bring your own key — used only for this session, never stored. Powers the "
                 "Senior-PE narrated brief. Without it, a deterministic brief is rendered.")
        if st.button("Run morning brief now", type="primary"):
            with st.spinner("Scanning fleet + writing brief…"):
                from src.brief_writer import (
                    MissingAPIKey, render_brief_markdown, write_brief,
                )
                client = None
                if byok_key:
                    from anthropic import Anthropic
                    client = Anthropic(api_key=byok_key)
                try:
                    brief_md = write_brief(summary, anomalies, client=client)
                except MissingAPIKey:
                    brief_md = render_brief_markdown(summary, anomalies)
                    st.info("No API key entered — generated a deterministic brief.")
                today = date.today().isoformat()
                (BRIEFS_DIR / f"{today}.md").write_text(brief_md)
            st.success("Brief generated.")
            st.rerun()
    with c2:
        briefs = sorted(BRIEFS_DIR.glob("*.md"), reverse=True)
        if briefs:
            selected = st.selectbox("Brief history", briefs,
                                    format_func=lambda p: p.stem)
        else:
            selected = None

    st.divider()
    if selected:
        st.markdown(selected.read_text())
    elif (BRIEFS_DIR / "sample.md").exists():
        st.info("Showing committed sample brief. Click \"Run morning brief now\" for a fresh one.")
        st.markdown((BRIEFS_DIR / "sample.md").read_text())
    else:
        st.info("No briefs yet. Click \"Run morning brief now\".")


def _anomaly_panel(anomalies: list, active: list) -> None:
    if not anomalies:
        st.success("No anomalies on the latest scan.")
        return
    sev_counts = pd.Series([a.severity for a in active]).value_counts()
    cols = st.columns(max(len(sev_counts), 1))
    for c, (sev, count) in zip(cols, sev_counts.items()):
        c.metric(sev, int(count))

    offenders = [a for a in active if a.deferred_usd_per_day > 0][:10]
    if offenders:
        offenders = list(reversed(offenders))  # largest at top
        fig_def = go.Figure()
        fig_def.add_trace(go.Bar(
            x=[a.deferred_usd_per_day for a in offenders],
            y=[a.well_id for a in offenders],
            orientation="h", marker_color=theme.RED,
        ))
        fig_def.update_layout(title="Top Deferred-$ Offenders ($/day)",
                              xaxis_title="Deferred $/day")
        st.plotly_chart(theme.style_fig(fig_def, height=300, legend=False),
                        width="stretch")

    df = pd.DataFrame([
        {"Well": a.well_id, "Sev": a.severity, "Category": a.category,
         "Deferred $/day": f"${a.deferred_usd_per_day:,.0f}" if a.deferred_usd_per_day else "—",
         "Headline": a.headline, "Ack": "🔕" if a.acknowledged else ""}
        for a in anomalies
    ])
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("⬇ Download anomaly list (CSV)", data=df.to_csv(index=False),
                       file_name="digest_anomalies.csv", mime="text/csv")
    theme.source_note(
        "Anomalies flagged by deterministic robust statistics on each well's own "
        "recent baseline: a median/MAD robust z-score (robust_z = 0.6745·(x − median)"
        "/MAD) plus a decline-aware rate-drop vs. the expected Arps rate — no fixed "
        "thresholds, so a single bad day can't inflate the baseline.")
    theme.references(["arps"])


def _data_quality_panel(token: str, window_days: int | None) -> None:
    """Representative-vs-anomalous data-quality view: which points each well's
    decline/type-curve trending should be fit on, and which to EXCLUDE (shut-ins /
    zero days, metering dropouts, gross outliers vs a robust decline-aware trend).

    Distinct from the Anomalies tab: that raises an *operational* alert on the latest
    day; this is the pre-trending data-cleaning step (a shut-in is a healthy well, just
    not on-trend data) — the same filtering WellProductivity.jl applies before a fit."""
    st.caption(
        "Before decline / type-curve trending, each oil-rate point is classified "
        "**representative** (usable for a fit) vs **non-representative** (shut-in / "
        "zero-rate day, metering dropout, or a gross outlier vs a robust decline-aware "
        "trend). This filters non-representative points so they don't bias the trend — "
        "separate from the operational alerts in the **Anomalies** tab.")

    rep = _representative_fleet_cached(token, window_days)
    if rep.empty:
        st.info("No wells with a scannable rate history in the selected window.")
        return

    q1, q2, q3 = st.columns(3)
    q1.metric("Fleet representative %", f"{rep['Representative %'].mean():.1f}%",
              help="Mean over wells of the share of points usable for trending.")
    q2.metric("Points excluded (window)", f"{int(rep['Excluded'].sum()):,}")
    q3.metric("Wells with exclusions", int((rep["Excluded"] > 0).sum()))

    # Lowest representative % first — those wells most need data cleaning before a fit.
    worst = rep.sort_values("Representative %").head(15)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=worst["Representative %"], y=worst["Well"], orientation="h",
        marker_color=theme.BLUE,
        text=[f"{v:.0f}%" for v in worst["Representative %"]], textposition="auto"))
    fig.update_layout(title="Representative Data % by Well (lowest first)",
                      xaxis_title="Representative %", xaxis_range=[0, 100])
    st.plotly_chart(theme.style_fig(fig, height=340, legend=False), width="stretch")
    theme.source_note(
        "Representative % = share of a well's oil-rate points usable for a decline / "
        "type-curve fit; the rest are excluded as shut-in / zero-rate days, metering "
        "dropouts, or gross outliers (|robust z| > 4 vs. a decline-aware robust trend). "
        "Fleet figure is the mean of that share across wells.")

    st.dataframe(rep.sort_values("Representative %"), width="stretch", hide_index=True)
    st.caption("Open a well in the **Wells** sidebar to see exactly which points are "
               "marked excluded on its decline chart.")
    theme.references(["arps", "deferment"])


def _events_panel(token: str, is_byod: bool) -> None:
    """Ongoing & Resolved Events — the persistent state machine surfaced in the UI.

    Replays the fleet's recent history through the SAME event store the morning
    brief + scheduler drive (NEW→ONGOING→RESOLVED), then renders the open and
    just-resolved events with running duration + cumulative deferred bbl/$ — the
    table that matches the brief's *Ongoing & Resolved Events* section. This is the
    lifecycle the stateless Anomalies tab can't show: a confirmed outage that
    persists as ONGOING every day instead of vanishing once it ages out of the
    detector's lookback window."""
    st.caption(
        "The **Anomalies** tab is a point-in-time scan of the latest day. This view "
        "adds **memory**: the fleet's recent history is replayed through the "
        "persistent event state machine (`NEW → ONGOING → RESOLVED`) — the same path "
        "the morning brief and the scheduler run — so a confirmed multi-day outage "
        "stays **ONGOING** with a running duration and **cumulative** deferred bbl/$ "
        "instead of dropping off once it ages into the rolling baseline. The replay "
        "runs in an in-memory store (nothing persisted).")

    # The committed synthetic fleet injects faults only on the final day, so there's
    # no clean multi-day ONGOING *rate* event to show. This toggle splices a
    # sustained outage into one healthy well to make the lifecycle demonstrable.
    inject_demo = False
    if not is_byod:
        inject_demo = st.toggle(
            "Inject a demo outage (multi-day ONGOING rate event)", value=True,
            key="inject_demo_outage",
            help=f"Holds the last {DEMO_OUTAGE_LEN} days of {DEMO_OUTAGE_WELL} at "
                 f"~{DEMO_OUTAGE_FRACTION:.0%} of its pre-event baseline so it reads "
                 "ONGOING with a growing cumulative deferral. The committed fleet "
                 "only injects faults on the final day; this mutates an in-memory "
                 "copy only — the committed CSVs and fixtures are untouched.")
        if inject_demo:
            st.caption(f"Demo outage active on **{DEMO_OUTAGE_WELL}** — a sustained "
                       f"~{1 - DEMO_OUTAGE_FRACTION:.0%} rate loss held over the last "
                       f"{DEMO_OUTAGE_LEN} days (no recovery), so it stays ONGOING.")
    else:
        st.caption("Replaying your uploaded fleet's history — any multi-day outage in "
                   "your data will surface here as an ONGOING event.")

    events = _replay_events_cached(token, str(ACK_PATH), inject_demo)
    open_evts = [e for e in events if e.state in (NEW, ONGOING)]
    resolved = [e for e in events if e.state == RESOLVED]
    multi_day = [e for e in open_evts if e.duration_days > 1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Open events (NEW/ONGOING)", len(open_evts))
    c2.metric("Multi-day ONGOING", len(multi_day),
              help="Open events past their first day — the lifecycle a point-in-time "
                   "scan can't keep visible.")
    c3.metric("Cumulative deferred (open)",
              f"${sum(e.deferred_usd for e in open_evts):,.0f}",
              help="Sum of cumulative deferred $ across open events over their life.")

    if not open_evts and not resolved:
        st.success("No open or recently-resolved events on the replayed history.")
        theme.references(["arps", "deferment"])
        return

    if open_evts:
        rows = []
        for e in sorted(open_evts, key=lambda e: (-e.deferred_usd, e.well_id)):
            rows.append({
                "Well": e.well_id,
                "Event type": e.event_type,
                "State": e.state,
                "Start date": e.start_date,
                "Duration (days)": e.duration_days,
                "Cumulative deferred bbl": round(e.deferred_bopd, 0) if e.deferred_bopd else 0,
                "Cumulative deferred $": round(e.deferred_usd, 0) if e.deferred_usd else 0,
                "Today's deferral $": round(e.last_deferred_usd, 0) if e.last_deferred_usd else 0,
                "Ack": "🔕" if e.acknowledged else "",
            })
        ev_df = pd.DataFrame(rows)
        st.dataframe(
            ev_df, width="stretch", hide_index=True,
            column_config={
                "Cumulative deferred $": st.column_config.NumberColumn(format="$%d"),
                "Cumulative deferred bbl": st.column_config.NumberColumn(format="%d"),
                "Today's deferral $": st.column_config.NumberColumn(format="$%d"),
                "Duration (days)": st.column_config.NumberColumn(format="%d d"),
            })
        st.download_button(
            "⬇ Download open events (CSV)", data=ev_df.to_csv(index=False),
            file_name="digest_open_events.csv", mime="text/csv")
    else:
        st.info("No open (NEW/ONGOING) events on the replayed history.")

    if resolved:
        st.markdown("**Recently resolved (closing out):**")
        for e in resolved:
            span = f"{e.duration_days}-day" if e.duration_days > 1 else "1-day"
            cum = (f" — ~{e.deferred_bopd:,.0f} bbl (${e.deferred_usd:,.0f}) deferred "
                   "over the event" if e.deferred_bopd > 0 else "")
            st.markdown(f"- ✅ **{e.well_id}** ({e.event_type}) — {span} event RESOLVED{cum}.")

    theme.source_note(
        "Events are replayed through the persistent state machine "
        "(`src.event_store`): a rate event opened from a confirmed drop stays "
        "ONGOING while production holds below its pre-event baseline (accruing "
        "cumulative deferred bbl/$ = baseline − current × oil price) and RESOLVES on "
        "recovery into band — the same lifecycle the morning brief reports.")
    theme.references(["arps", "deferment"])


def _ledger_section(token: str) -> None:
    st.divider()
    st.subheader("📉 Lost-Production Ledger")
    ledger, led_summary = _build_ledger_cached(token, str(ACK_PATH), 30)

    win_start = led_summary.get("window_start")
    win_end = led_summary.get("window_end")
    win_label = (f"{win_start.date()} → {win_end.date()}"
                 if win_start is not None and win_end is not None else "trailing window")
    st.caption(
        f"Cumulative deferred production accrued over the {win_label} window "
        f"({led_summary['days_scanned']} day(s) with a scannable baseline) — the same "
        "deterministic scan + deferred-$ economics as the morning brief, summed by cause.")

    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("Period deferred $", f"${led_summary['period_deferred_usd']:,.0f}")
    lc2.metric("Recoverable $ (est.)", f"${led_summary['recoverable_usd']:,.0f}",
               help="~65% of period deferred — excludes the typically planned/reservoir-driven "
                    "share. Full base-management split lives in Deferment IQ.")
    lc3.metric("Top cause", str(led_summary["top_cause"] or "—"),
               delta=f"${led_summary['top_cause_usd']:,.0f}" if led_summary["top_cause"] else None,
               delta_color="off")

    if not ledger.empty:
        daily = ledger.groupby("date", as_index=False)["deferred_usd"].sum()
        daily["cumulative_usd"] = daily["deferred_usd"].cumsum()
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=daily["date"], y=daily["cumulative_usd"], mode="lines+markers",
            name="Cumulative deferred $", fill="tozeroy",
            line=dict(color=theme.RED, width=2), marker=dict(size=5),
            fillcolor="rgba(192,80,77,0.20)"))
        fig_cum.update_layout(title="Cumulative Deferred Production ($) Over Window",
                              yaxis_title="Cumulative deferred $")
        st.plotly_chart(theme.style_fig(fig_cum, height=300, legend=False), width="stretch")

        fig_split = go.Figure()
        for i, cause in enumerate(sorted(ledger["cause"].unique())):
            sub = ledger[ledger["cause"] == cause]
            fig_split.add_trace(go.Bar(
                x=sub["date"], y=sub["deferred_usd"], name=cause,
                marker_color=theme.COLORWAY[i % len(theme.COLORWAY)]))
        fig_split.update_layout(barmode="stack", title="Deferred $ by Cause (period split)",
                                yaxis_title="Deferred $/day")
        st.plotly_chart(theme.style_fig(fig_split, height=300), width="stretch")

        with st.expander("Ledger Detail (tidy: date · cause · bbl · $ · cumulative)"):
            st.dataframe(ledger, width="stretch", hide_index=True)
    else:
        st.info("No deferred-production events accrued in the trailing window on this fleet.")

    st.markdown(
        "📊 **Full base-management accounting in [Deferment IQ]"
        "(https://deferment-iq.streamlit.app)** — potential/entitlement modeling, "
        "downtime-vs-underperformance waterfall, $-Pareto by cause, MTTR, capture-rate, and "
        "the recoverable-opportunity split. This ledger is the lightweight Monitor→Quantify "
        "upstream of that weekly VP review.")
    theme.source_note(
        "Deferred $ = (decline-aware potential − actual) × realized oil price, summed by "
        "cause over the window; recoverable ≈ 65% (excludes the typically planned / "
        "reservoir-driven share).")
    theme.references(["arps", "deferment"])


# =====================================================================
# PAGE: per-well drill-down
# =====================================================================

def render_well(well_id: str) -> None:
    token, is_byod = _resolve_fleet()
    fleet = _fleet_for_token(token)
    anomalies = _scan_fleet_cached(token, str(ACK_PATH))
    meta = fleet_registry.get(well_id)
    df = fleet.get(well_id)

    src_chip = ("your fleet · uploaded", "info") if is_byod else (meta.peer_group, "info")
    theme.header(
        f"{well_id} · {meta.name}",
        subtitle=f"{meta.lift} · {meta.basin} · {meta.formation} · {meta.area}",
        chips=[(f"v{__version__}", "ver"), src_chip],
    )
    if is_byod:
        theme.data_badge("real", "Your uploaded fleet SCADA — parsed in memory for this "
                                 "session only, nothing stored server-side.")
    else:
        theme.data_badge("synthetic", "Modeled daily SCADA fleet with known ground truth — public production is monthly, not daily.")
    theme.well_cross_links("pe-digest", well_id)
    _back_to_overview()

    if df is None or not len(df):
        if is_byod:
            st.info(
                f"**{well_id}** isn't in your uploaded fleet. The per-well menu in the "
                "sidebar is keyed to the synthetic demo wells (`well_NNN`); your wells "
                "are listed in the **Fleet Table** on the Fleet Overview. Open the "
                "**Fleet Overview** to see them, or switch the **Data source** back to "
                "the synthetic demo fleet to drill into this well.")
        else:
            st.warning("No SCADA history for this well.")
        _back_to_overview()
        return

    window_days = _time_range_control(well_id)
    win = slice_window(df, window_days)
    last = win.iloc[-1]

    bopd = float(last["bopd"]) if pd.notna(last["bopd"]) else float("nan")
    bfpd = float(last["bfpd"]) if pd.notna(last["bfpd"]) else float("nan")
    gas = float(last["gas_mcfd"]) if pd.notna(last.get("gas_mcfd")) else float("nan")
    bwpd = bfpd - bopd
    water_cut = (bwpd / bfpd * 100.0) if bfpd > 0 else float("nan")
    gor = (gas * 1000.0 / bopd) if bopd > 0 else float("nan")
    var_oil = production_variance_pct(win["bopd"].values)

    # metrics row
    m = st.columns(5)
    m[0].metric("BOPD", f"{bopd:,.0f}", delta=f"{var_oil:+.1f}%")
    m[1].metric("BWPD", f"{bwpd:,.0f}")
    m[2].metric("MCFD", f"{gas:,.0f}")
    m[3].metric("Water cut %", f"{water_cut:.1f}%")
    m[4].metric("GOR (scf/bbl)", f"{gor:,.0f}")
    m2 = st.columns(5)
    m2[0].metric("Lateral (ft)", f"{meta.lateral_length_ft:,}")
    m2[1].metric("Days on prod", f"{len(df)}")
    m2[2].metric("Intake psi", f"{float(last['intake_pressure_psi']):.0f}")
    m2[3].metric("Runtime %", f"{float(last['runtime_pct']):.1f}")
    m2[4].metric("Prod variance %", f"{var_oil:+.1f}%")

    # production graphs
    st.subheader("Production")
    p_oil, p_gas, p_water, p_wc = st.tabs(
        ["Oil (BOPD)", "Gas (MCFD)", "Water (BWPD)", "Water Cut %"])
    with p_oil:
        _oil_with_representative(win)
    with p_gas:
        st.plotly_chart(_line(win["date"], win["gas_mcfd"], "Gas", theme.AMBER,
                              "MCFD", "Gas Rate (MCFD)"), width="stretch")
    with p_water:
        st.plotly_chart(_line(win["date"], win["bfpd"] - win["bopd"], "Water", theme.TEAL,
                              "BWPD", "Water Rate (BWPD)"), width="stretch")
    with p_wc:
        wc = (win["bfpd"] - win["bopd"]) / win["bfpd"] * 100.0
        st.plotly_chart(_line(win["date"], wc, "Water cut", theme.GREY,
                              "Water cut %", "Water Cut Trend (%)"), width="stretch")

    # SCADA diagnostics
    st.subheader("SCADA Diagnostics")
    d_int, d_temp, d_amps, d_rt = st.tabs(
        ["Intake Psi", "Motor Temp °F", "Motor Amps", "Runtime %"])
    with d_int:
        st.plotly_chart(_line(win["date"], win["intake_pressure_psi"], "Intake",
                              theme.PURPLE, "psi", "Intake Pressure (psi)"), width="stretch")
    with d_temp:
        st.plotly_chart(_line(win["date"], win["motor_temp_f"], "Temp", theme.RED,
                              "°F", "Motor Temperature (°F)"), width="stretch")
    with d_amps:
        st.plotly_chart(_line(win["date"], win["motor_amps"], "Amps", theme.GREEN,
                              "A", "Motor Amps (A)"), width="stretch")
    with d_rt:
        st.plotly_chart(_line(win["date"], win["runtime_pct"], "Runtime", theme.BLUE,
                              "%", "Runtime (%)"), width="stretch")

    # health note: expected-vs-actual + any anomaly
    st.subheader("Health Note")
    _well_health_note(df, well_id, anomalies)
    _back_to_overview()


def _well_health_note(df: pd.DataFrame, well_id: str, anomalies: list) -> None:
    # Decline-expected rate today from history excluding today (matches the scan).
    window = df.iloc[-14:]["bopd"].values if len(df) >= 14 else df["bopd"].values
    expected = _expected_decline_rate(window[:-1], extrapolate=1) if len(window) >= 5 else None
    last = float(df.iloc[-1]["bopd"])
    if expected:
        resid = (last - expected) / expected * 100.0
        kind = "ok" if resid >= -15 else ("high" if resid < -25 else "warn")
        theme.flag(
            f"Latest BOPD {last:,.0f} vs decline-expected {expected:,.0f} "
            f"({resid:+.0f}% residual)", kind)
    else:
        st.caption("Not enough positive history to fit a decline-expected rate.")

    a = _anomaly_for(well_id, anomalies)
    if a is None:
        theme.flag("No active anomaly on the latest scan.", "ok")
        return
    kind = {"HIGH": "high", "MEDIUM": "warn"}.get(a.severity, "warn")
    suffix = " (acknowledged / planned)" if a.acknowledged else ""
    theme.flag(f"{a.severity} · {a.category}: {a.headline}{suffix}", kind)
    if a.deferred_usd_per_day:
        st.metric("Deferred production", f"{a.deferred_bopd:,.0f} bbl/day",
                  delta=f"${a.deferred_usd_per_day:,.0f}/day", delta_color="inverse")
    st.caption(f"Recommended action: {a.recommended_action}")


# =====================================================================
# Shared setup (runs every rerun) + navigation
# =====================================================================

theme.setup_page("Daily Production Digest", icon="📅")
theme.suite_nav("pe-digest")
_bootstrap_fleet()

_fleet = _load_fleet_cached(str(DATA_DIR))

overview = st.Page(render_overview, title="Fleet Overview", icon="📊", default=True)
wells = [
    st.Page(partial(render_well, wid), title=wid, url_path=wid)
    for wid in sorted(_fleet)
]
st.navigation({"Fleet": [overview], "Wells": wells}).run()
