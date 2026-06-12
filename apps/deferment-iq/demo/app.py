"""Deferment IQ — base-management / lost-oil dashboard (multipage).

Deterministic deferment accounting (potential vs actual, by reason code) with an
optional LLM reason-code classifier and an LLM-narrated VP review. Bring-your-own-key.

Multipage (``st.navigation`` + ``st.Page``): a **Fleet Overview** page (KPIs,
base-management review, recovery work-queue, classifier eval, and a sortable
fleet table) plus one **drill-down page per well** (its potential-vs-actual +
deferred bars, the well's events, KPIs, and its recovery items). Detection /
deferment / reason-code logic is unchanged; the LLM stays BYOK-optional.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from functools import partial
from pathlib import Path

# Ensure repo root + demo dir are importable so ``src.*`` and the vendored
# ``theme`` / ``fleet_registry`` resolve regardless of cwd / Streamlit context.
DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent
for _p in (str(REPO_ROOT), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Self-heal stale bytecode / module cache (Streamlit / HF container reuse).
import shutil as _shutil
for _pyc in (REPO_ROOT / "src").rglob("__pycache__"):
    _shutil.rmtree(_pyc, ignore_errors=True)
for _m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

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
from src import analytics as A
from src.data_loader import load_events, load_fleet
from src.deferment import classify_events, compute_deferment
from src.narrator import MissingAPIKey, render_review_markdown, write_review

DATA = REPO_ROOT / "data" / "synthetic"
WELLS = DATA / "wells"
REAL_COLORADO = REPO_ROOT / "data" / "real" / "colorado" / "production.csv"
REAL_NDIC = REPO_ROOT / "data" / "real" / "ndic" / "production.csv"
EVAL = REPO_ROOT / "evals" / "results" / "summary.json"
AFE_COPILOT_URL = "https://afe-copilot.streamlit.app"

# Data-source toggle values (sidebar radio). Synthetic (demo) is the default.
SRC_REAL_CO = "Real — Colorado DJ Basin (ECMC)"
SRC_SYNTHETIC = "Synthetic (demo)"
SRC_REAL_NDIC = "Real — North Dakota (NDIC, your export)"
SRC_UPLOAD = "Upload my data (CSV)"

_BADGE_SYNTHETIC = ("synthetic",
                    "Modeled fleet with reason-coded events + known ground truth "
                    "(~92% classifier eval).")
_BADGE_REAL_CO = ("real",
                  "Colorado ECMC (COGCC) public monthly records — DJ Basin Niobrara/Codell "
                  "horizontals (Weld County). Downtime from days-produced; cause attribution "
                  "N/A (no public reason codes).")
_BADGE_REAL = ("real",
               "North Dakota (NDIC) public monthly filings — Bakken. Downtime from "
               "days-produced; cause attribution N/A (no public reason codes).")
_BADGE_UPLOAD = ("real", "User-uploaded monthly production data.")


# ---- bootstrap + cached loads ----------------------------------------------

def _bootstrap():
    if not any(WELLS.glob("well_*.csv")):
        with st.status("First-time setup: generating synthetic fleet…", expanded=False):
            subprocess.run([sys.executable, str(DATA / "generate.py")], check=True)


@st.cache_data(show_spinner=False)
def _load(price_per_bbl, use_llm_flag, has_key, byok_key):
    """Cache the SYNTHETIC fleet load + classification + deferment compute. Keyed on
    the LLM toggle + key presence so a deterministic run (no key) caches cleanly."""
    fleet = load_fleet(WELLS)
    events = load_events(DATA / "events.csv")
    client = None
    if use_llm_flag and has_key:
        from anthropic import Anthropic
        client = Anthropic(api_key=byok_key)
    evc = classify_events(events, use_llm=use_llm_flag and has_key, client=client)
    daily = compute_deferment(fleet, evc, price_per_bbl=price_per_bbl)
    return fleet, evc, daily


@st.cache_data(show_spinner=False)
def _load_real(price_per_bbl, csv_path):
    """Cache a REAL monthly extract load + deferment compute (keyed on the CSV path).

    Works for either real source — the FREE Colorado ECMC default or a user-supplied
    NDIC export — because both share the tidy monthly schema and the same transform
    (rate from oil_bbl/days, downtime from days-produced). There are NO public reason
    codes, so events is empty and every lost barrel is 'uncoded' — the deferment
    QUANTITY is real, the cause is N/A. Returns (fleet, empty_events, daily)."""
    from src.ndic import load_ndic_fleet
    from src.data_loader import EVENT_COLUMNS
    fleet = load_ndic_fleet(csv_path)
    evc = pd.DataFrame(columns=[*EVENT_COLUMNS, "reason_key"])  # no real reason codes
    daily = compute_deferment(fleet, evc, price_per_bbl=price_per_bbl)
    return fleet, evc, daily


def _resolve_source(data_source: str, uploaded=None):
    """Map the sidebar choice → (is_real, real_csv_path_or_None, badge args).

    Default is real **Colorado (ECMC)** — free public monthly records, committed to the
    repo. A selected real source is honored only when its extract exists; otherwise we
    warn and fall back to synthetic so the app always renders.

    For SRC_UPLOAD, returns the sentinel string "UPLOAD" as the csv path — callers must
    check for this and handle the uploaded file object from session state / the return
    value of _sidebar_controls() themselves."""
    if data_source == SRC_REAL_CO:
        if REAL_COLORADO.exists():
            return True, str(REAL_COLORADO), _BADGE_REAL_CO
        st.warning(
            f"Colorado extract missing at `{REAL_COLORADO.relative_to(REPO_ROOT)}` — "
            "falling back to the synthetic demo fleet.")
    elif data_source == SRC_REAL_NDIC:
        if REAL_NDIC.exists():
            return True, str(REAL_NDIC), _BADGE_REAL
        st.warning(
            "Real — North Dakota (NDIC) selected, but no extract found at "
            f"`{REAL_NDIC.relative_to(REPO_ROOT)}`. NDIC bulk data is a **paid "
            "subscription** (see `data/real/ndic/README.md`); the default **Colorado** "
            "source is free real data. Falling back to synthetic.")
    elif data_source == SRC_UPLOAD:
        return True, "UPLOAD", _BADGE_UPLOAD
    return False, None, _BADGE_SYNTHETIC


@st.cache_data(show_spinner=False)
def _fleet_well_ids() -> list[str]:
    """Sorted well ids for navigation wiring (cheap glob, no CSV parse)."""
    return sorted(p.stem for p in WELLS.glob("well_*.csv"))


# ---- shared helpers --------------------------------------------------------

def _sidebar_controls() -> tuple[float, str, bool, str, object]:
    """Render the shared sidebar settings and return (price, byok_key, use_llm, data_source, uploaded)."""
    with st.sidebar:
        st.header("Settings")
        data_source = st.radio(
            "Data source", [SRC_SYNTHETIC, SRC_REAL_CO, SRC_REAL_NDIC, SRC_UPLOAD], index=0,
            key="data_source",
            help="Synthetic = modeled fleet with reason-coded events + ground truth "
                 "(powers the classifier eval); the default. Real — Colorado = FREE ECMC "
                 "public monthly records (DJ Basin Niobrara/Codell horizontals). Real — "
                 "North Dakota = drop your own NDIC monthly export (NDIC bulk data is a "
                 "paid subscription). Upload my data = drop any monthly production CSV in "
                 "the tidy schema. Real monthly data has real downtime (days-produced) "
                 "but no public reason codes, so cause attribution is N/A.")
        if data_source == SRC_UPLOAD:
            uploaded = st.file_uploader(
                "Monthly production CSV", type=["csv"],
                help="Columns: well_id, date (YYYY-MM), oil_bbl, gas_mcf, water_bbl, "
                     "days_on (or days_produced)")
        else:
            uploaded = None
        price = st.number_input("Realized oil price ($/bbl)", 20.0, 150.0, 70.0, 1.0,
                                key="oil_price")
        byok_key = st.text_input(
            "🔑 Anthropic API key (optional)", type="password", key="byok_key",
            help="Bring your own key — used only for this session, never stored. Powers the LLM "
                 "reason-code classifier and the narrated VP review. Everything else works without it.")
        use_llm = st.checkbox("🤖 Use LLM for reason-code classification", value=False,
                              key="use_llm",
                              help="Re-classify event notes with Claude (needs key). Default is the "
                                   "deterministic rules classifier. Synthetic data only.")
    return price, byok_key, use_llm, data_source, uploaded


def _back_to_overview():
    target = globals().get("overview")
    try:
        st.page_link(target if target is not None else "app.py",
                     label="← Back to Fleet overview", icon="📊")
    except Exception:
        pass


def _build_fleet_table(daily: pd.DataFrame, price: float) -> pd.DataFrame:
    """One row per well joined with registry metadata + recovery $ / capture %.

    Reuses ``analytics.top_wells`` (deferment + dominant cause + uptime) and
    ``analytics.recovery_queue`` (recoverable $) so the numbers match the rest of
    the app; capture % = recoverable $ / deferred $ per well."""
    well_ids = sorted(daily["well_id"].unique()) if len(daily) else []
    if not well_ids:
        return pd.DataFrame()

    # top_wells over the whole fleet (n = all) → per-well deferred bbl/$, cause, uptime.
    tw = A.top_wells(daily, n=len(well_ids)).set_index("well_id")
    queue = A.recovery_queue(daily, oil_price=price)
    rec_by_well = (queue.groupby("well_id")["recoverable_usd"].sum().to_dict()
                   if len(queue) else {})

    rows = []
    for wid in well_ids:
        meta = fleet_registry.get(wid)
        deferred_bbl = float(tw.loc[wid, "deferred_bbl"]) if wid in tw.index else 0.0
        deferred_usd = float(tw.loc[wid, "deferred_usd"]) if wid in tw.index else 0.0
        cause = str(tw.loc[wid, "top_cause"]) if wid in tw.index else "—"
        uptime = float(tw.loc[wid, "uptime_pct"]) if wid in tw.index else 100.0
        rec_usd = float(rec_by_well.get(wid, 0.0))
        capture = (rec_usd / deferred_usd * 100.0) if deferred_usd > 0 else 0.0
        rows.append({
            "Well": wid,
            "Lift": meta.lift,
            "Lateral (ft)": meta.lateral_length_ft,
            "Basin · Formation": f"{meta.basin} · {meta.formation}",
            "Deferred bbl": round(deferred_bbl, 0),
            "Deferred $": round(deferred_usd, 0),
            "Dominant cause": cause,
            "Uptime %": round(uptime, 1),
            "Recoverable $": round(rec_usd, 0),
            "Capture %": round(capture, 0),
        })
    out = pd.DataFrame(rows).sort_values("Deferred $", ascending=False).reset_index(drop=True)
    return out


# =====================================================================
# PAGE: Fleet overview
# =====================================================================

def render_overview() -> None:
    price, byok_key, use_llm, data_source, uploaded = _sidebar_controls()
    is_real, real_csv, badge = _resolve_source(data_source, uploaded)

    src_chip = (("Colorado DJ Basin · real" if data_source == SRC_REAL_CO
                 else ("User-uploaded data · real" if data_source == SRC_UPLOAD
                       else "North Dakota (NDIC) · real")), "info") if is_real \
        else ("~92% reason-code acc", "eval")
    theme.header(
        "Deferment IQ",
        subtitle="Base management / lost-oil accounting — where are the barrels going, what's it costing, "
                 "and what's recoverable. Built by an ex-OXY / ex-Shell Staff Production Engineer.",
        chips=[(f"v{__version__}", "ver"), src_chip, ("fleet explorer", "info")],
    )
    theme.data_badge(*badge)

    theme.how_to(
        "- **Deferment = potential − actual.** Each well's *potential* (entitlement) is "
        "modeled from its full-uptime months (P75, decline-aware); the deferred volume is "
        "what's left after subtracting the actual produced volume.\n"
        "- **Two kinds of loss.** The gap is split into **downtime** (well off / curtailed — "
        "from days-produced / runtime) vs. **underperformance** (well on but making less than "
        "potential rate). An 8% deadband keeps healthy wells reading ~0.\n"
        "- **Data source toggle (sidebar).** Defaults to **real Colorado ECMC** public monthly "
        "records (DJ Basin); switch to **Synthetic (demo)** for the full reason-coded fleet, or "
        "drop your own North Dakota (NDIC) export.\n"
        "- **On REAL public data the QUANTITY is real, the cause is N/A.** Public monthly filings "
        "carry no operator reason codes, so the deferred barrels/$ are real (from days-produced) "
        "but per-cause attribution, the recovery queue, and MTTR are N/A — use **Synthetic (demo)** "
        "to see those.")

    with st.expander(f"🆕 What Is This / v{__version__}"):
        st.markdown(
            "- **Deferment vs. potential** — each well's entitlement is modeled from its full-uptime "
            "days (P75, decline-aware); the gap to actual is split into **downtime** vs. **underperformance**.\n"
            "- **Reason-code attribution** — every lost barrel is tagged to a cause from the operator's "
            "free-text note (deterministic rules classifier, ~92% on the eval; optional LLM for the long tail).\n"
            "- **The VP views** — deferment waterfall, Pareto of $ by cause, worst-offender wells, MTTR, and "
            "the **recoverable** opportunity (excludes planned + reservoir, which you can't get back).\n"
            "- **Capture rate** flags uncaptured (un-coded) deferment — a real data-quality gap to close.\n"
            "- Deterministic engine; the LLM only classifies the messy tail and narrates. Bring your own key.\n"
            "\n"
            "**New — fleet explorer (multipage):**\n"
            "- **Fleet Overview** + a **drill-down page per well** (`st.navigation`) — open any well from the "
            "**Wells** section in the sidebar for its potential-vs-actual + deferred-bar chart, its events, "
            "and its recovery items.\n"
            "- **Sortable fleet table** — one row per well with lift, lateral, basin·formation, deferred "
            "bbl/$, dominant cause, uptime %, recoverable $, and capture %.\n"
            "- Prioritized recovery work-queue (recoverable $ ÷ MTTR), MTTR-by-cause, shared fleet registry, "
            "unified suite theme + cross-app navigator."
        )

    if is_real and real_csv == "UPLOAD":
        if uploaded is None:
            st.info("Upload a CSV to analyze your own fleet.")
            st.stop()
        import tempfile, os
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            fleet, evc, daily = _load_real(price, tmp_path)
        except Exception as e:
            st.exception(e)
            st.stop()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    elif is_real:
        fleet, evc, daily = _load_real(price, real_csv)
    else:
        fleet, evc, daily = _load(price, use_llm, bool(byok_key), byok_key)
    k = A.fleet_kpis(daily, price)
    pareto = A.pareto_by_cause(daily)
    top = A.top_wells(daily, 10)
    rec = A.recovery_opportunity(daily)
    queue = A.recovery_queue(daily, evc, price)

    tab_review, tab_queue, tab_table, tab_eval = st.tabs(
        ["📋 Base-Management Review", "🔧 Recovery Queue", "📋 Fleet Table", "🎯 Classifier Eval"])

    with tab_review:
        _review_section(k, pareto, top, rec, daily, evc, price, byok_key, is_real)
    with tab_queue:
        _queue_section(queue, is_real)
    with tab_table:
        _fleet_table_section(daily, price, is_real, real_csv)
    with tab_eval:
        _eval_section(is_real)


def _review_section(k, pareto, top, rec, daily, evc, price, byok_key, is_real=False) -> None:
    c = st.columns(5)
    c[0].metric("Production efficiency", f"{k['uptime_pct']:.1f}%", help="Actual ÷ potential")
    c[1].metric("Deferred", f"${k['deferred_usd']:,.0f}", f"{k['pct_deferred']:.1f}% of potential",
                delta_color="inverse")
    c[2].metric("Deferred rate", f"{k['deferred_bopd_avg']:,.0f} BOPD")
    if is_real:
        c[3].metric("Recoverable opportunity", "N/A", help="Needs reason codes — not in public data")
        c[4].metric("Reason-code capture", "N/A",
                    help="Public monthly filings carry no reason codes — cause attribution is N/A. "
                         "The deferment QUANTITY (above) is real.")
    else:
        c[3].metric("Recoverable opportunity", f"${rec['recoverable_usd']:,.0f}")
        c[4].metric("Reason-code capture", f"{k['capture_rate_pct']:.0f}%",
                    delta=("coding gap" if k['capture_rate_pct'] < 90 else "good"),
                    delta_color=("inverse" if k['capture_rate_pct'] < 90 else "off"))

    left, right = st.columns(2)
    with left:
        st.subheader("Deferment Waterfall (bbl)")
        wf = A.waterfall(daily)
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(wf) - 2) + ["total"],
            x=[s["label"] for s in wf], y=[s["value"] for s in wf],
            connector={"line": {"color": theme.GREY}},
            decreasing={"marker": {"color": theme.RED}},
            increasing={"marker": {"color": theme.BLUE}},
            totals={"marker": {"color": theme.NAVY}}))
        st.plotly_chart(theme.style_fig(fig, height=380), width="stretch")
        theme.source_note(
            "Potential from full-uptime months (P75, decline-aware); deferred = potential − "
            "actual, bridged potential → downtime → underperformance → actual, in bbl.")
        if is_real:
            st.caption("Real data: the bridge is gross potential → **uncoded** deferment → actual "
                       "(no per-cause split — public filings have no reason codes).")
    with right:
        st.subheader("Where the Barrels Go — $ by Cause")
        if is_real:
            st.info("**Cause attribution N/A** — public monthly filings carry no reason "
                    "codes or operator cause notes. The deferment **quantity** is real (from "
                    "days-produced); the per-cause $-Pareto needs an operator's coded event log.")
        elif len(pareto):
            pf = go.Figure()
            pf.add_bar(x=pareto["label"], y=pareto["deferred_usd"], name="Deferred $",
                       marker_color=[theme.BLUE if r else theme.GREY for r in pareto["recoverable"]])
            pf.add_scatter(x=pareto["label"], y=pareto["cum_pct"], name="Cumulative %",
                           yaxis="y2", line=dict(color=theme.RED))
            pf.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], title="cum %"))
            st.plotly_chart(theme.style_fig(pf, height=380), width="stretch")
            st.caption("Blue = recoverable · grey = planned/reservoir (not recoverable).")
            theme.source_note(
                "Deferred $ = deferred bbl × realized oil price, ranked Pareto by cause "
                "(vital-few first); cumulative % overlaid. Cause from the reason-code classifier.")

    st.subheader("Worst-Offender Wells")
    disp = top.copy()
    disp["deferred_usd"] = disp["deferred_usd"].map(lambda v: f"${v:,.0f}")
    disp["deferred_bbl"] = disp["deferred_bbl"].map(lambda v: f"{v:,.0f}")
    disp["uptime_pct"] = disp["uptime_pct"].map(lambda v: f"{v:.0f}%")
    if is_real:
        disp["top_cause"] = "N/A (uncoded)"
    disp.columns = ["Well", "Deferred bbl", "Deferred $", "Dominant cause", "Uptime"]
    st.dataframe(disp, width="stretch", hide_index=True)
    st.download_button("⬇ Download deferment summary (CSV)", data=top.to_csv(index=False),
                       file_name="deferment_fleet.csv", mime="text/csv")
    if is_real:
        st.caption("Ranked by **real** deferred barrels/$ (potential vs. actual). Dominant "
                   "cause is N/A — no public reason codes.")

    mc1, mc2 = st.columns(2)
    with mc1:
        st.subheader("MTTR by Cause (days)")
        m = A.mttr_by_cause(evc)
        if is_real:
            st.info("MTTR needs a coded event log (start/end + cause) — N/A on public monthly data.")
        elif len(m):
            mm = m.copy(); mm["mttr_days"] = mm["mttr_days"].map(lambda v: f"{v:.1f}")
            st.dataframe(mm[["label", "n_events", "mttr_days", "total_event_days"]]
                         .rename(columns={"label": "Cause", "n_events": "Events",
                                          "mttr_days": "MTTR (d)", "total_event_days": "Down-days"}),
                         width="stretch", hide_index=True)
            ms = m.sort_values("mttr_days")
            mf = go.Figure(go.Bar(x=ms["mttr_days"], y=ms["label"], orientation="h",
                                  marker_color=theme.AMBER))
            mf.update_layout(xaxis_title="MTTR (days)")
            st.plotly_chart(theme.style_fig(mf, height=260, legend=False), width="stretch")
    with mc2:
        st.subheader("Deferment Trend (Weekly bbl)")
        tr = A.deferment_trend(daily, "W")
        tf = go.Figure(go.Scatter(x=tr["date"], y=tr["deferred_bbl"], fill="tozeroy",
                                  line=dict(color=theme.RED)))
        st.plotly_chart(theme.style_fig(tf, height=260, legend=False), width="stretch")

    st.divider()
    st.subheader("📝 Senior-PE Base-Management Review")
    if is_real:
        st.info("The narrated review summarizes deferment **by cause** and the **recoverable** "
                "opportunity — both N/A on public monthly data (no reason codes). The real numbers "
                "are the production-efficiency and deferred-barrel/$ KPIs above. Switch to the "
                "**Synthetic (demo)** data source for the full reason-coded VP review.")
    elif st.button("Generate review", type="primary"):
        try:
            client = None
            if byok_key:
                from anthropic import Anthropic
                client = Anthropic(api_key=byok_key)
            with st.spinner("Writing the VP review…"):
                md = write_review(k, pareto, top, rec, brief_date=date.today().isoformat(), client=client)
            st.markdown(md)
        except MissingAPIKey:
            st.info("No API key — showing the deterministic review. Add your Anthropic key in the "
                    "sidebar for the Senior-PE narrated version.")
            st.markdown(render_review_markdown(k, pareto, top, rec))

    theme.references(["deferment", "pareto", "npv"])


def _queue_section(queue, is_real=False) -> None:
    if is_real:
        st.subheader("Prioritized Recovery Work-Queue")
        st.info("**Cause attribution N/A — no public reason codes.** The recovery queue ranks "
                "actionable items per (well, **recoverable cause**), which requires the operator's "
                "coded downtime log. Public monthly filings give real deferment **quantity** (from "
                "days-produced) but no cause, so there's nothing to attribute or authorize here. "
                "Switch to **Synthetic (demo)** for the full Quantify → Authorize work-queue.")
        theme.references(["npv"])
        return
    st.subheader("Prioritized Recovery Work-Queue")
    st.caption(
        "From *where are the barrels lost* to *what to do next, what it's worth, who acts* — "
        "the **Quantify → Authorize** handoff. One actionable item per (well, recoverable cause); "
        "planned work and reservoir/watering-out are excluded (you can't get those barrels back). "
        "Ranked by **priority_score = recoverable $ ÷ MTTR (days)** — value per day-to-restore, so "
        "a quick high-value win outranks a slow one of similar size.")

    if not len(queue):
        st.info("No recoverable deferment in the current period — nothing to queue.")
        theme.references(["npv"])
        return

    total_rec_usd = float(queue["recoverable_usd"].sum())
    n_items = int(len(queue))
    toprow = queue.iloc[0]

    kc = st.columns(3)
    kc[0].metric("Total recoverable", f"${total_rec_usd:,.0f}",
                 help="Sum of recoverable $ across every queued item.")
    kc[1].metric("Actionable items", f"{n_items}",
                 help="Distinct (well, recoverable cause) interventions.")
    kc[2].metric("Fastest high-value win",
                 f"{toprow['well_id']} · {toprow['cause']}",
                 f"${toprow['recoverable_usd']:,.0f} · {toprow['mttr_days']:.1f}d",
                 help="Highest value-per-day-to-restore item — do this first.")

    bar = queue.head(12).iloc[::-1]
    causes = list(dict.fromkeys(queue["cause"]))
    cmap = {c: theme.COLORWAY[i % len(theme.COLORWAY)] for i, c in enumerate(causes)}
    bf = go.Figure()
    for c in causes:
        sub = bar[bar["cause"] == c]
        if not len(sub):
            continue
        bf.add_bar(
            y=[f"{w} · {c}" for w in sub["well_id"]], x=sub["recoverable_usd"],
            name=c, orientation="h", marker_color=cmap[c],
            hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>")
    bf.update_layout(barmode="stack", xaxis_title="Recoverable $",
                     title="Top Recovery Opportunities by $ (Colored by Cause)")
    st.plotly_chart(theme.style_fig(bf, height=420), width="stretch")
    theme.source_note(
        "Recoverable $ = recoverable bbl × realized oil price, per (well, recoverable cause); "
        "planned + reservoir excluded. Ranked by priority = recoverable $ ÷ MTTR (days).")

    disp = queue.copy()
    disp.insert(0, "#", range(1, len(disp) + 1))
    disp["recoverable_usd"] = disp["recoverable_usd"].map(lambda v: f"${v:,.0f}")
    disp["recoverable_bbl"] = disp["recoverable_bbl"].map(lambda v: f"{v:,.0f}")
    disp["mttr_days"] = disp["mttr_days"].map(lambda v: f"{v:.1f}")
    disp["priority_score"] = disp["priority_score"].map(lambda v: f"{v:,.0f}")
    disp = disp[["#", "well_id", "cause", "suggested_action",
                 "recoverable_bbl", "recoverable_usd", "mttr_days", "priority_score"]]
    disp.columns = ["#", "Well", "Cause", "Suggested action",
                    "Recoverable bbl", "Recoverable $", "MTTR (d)", "Priority ($/day)"]
    st.dataframe(disp, width="stretch", hide_index=True)
    st.download_button(
        "⬇ Download CSV",
        data=queue.to_csv(index=False),
        file_name="recovery_work_queue.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Authorize the Top Interventions")
    st.caption("Each item is sized and ready to hand to capital authorization.")
    for _, r in queue.head(5).iterrows():
        st.markdown(
            f"**{r['well_id']} — {r['cause']}** · {r['suggested_action']} · "
            f"recover **{r['recoverable_bbl']:,.0f} bbl (${r['recoverable_usd']:,.0f})**, "
            f"~{r['mttr_days']:.1f}-day restore — "
            f"[authorize the intervention in AFE Copilot ↗]({AFE_COPILOT_URL})")
    st.caption("Deep-links open AFE Copilot in a new tab to draft the Authorization for Expenditure.")

    theme.references(["npv"])


def _build_real_fleet_table(daily: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """One row per well from the active REAL extract — real volumes/uptime, real
    identity (operator/field/formation from the public record), cause N/A.

    Built directly off ``daily`` + the extract's own well meta (NOT the Permian
    registry, which would stamp placeholder Midland/Delaware metadata onto these wells)."""
    if daily is None or not len(daily):
        return pd.DataFrame()
    meta = {}
    try:
        from src.ndic import ndic_well_meta
        m = ndic_well_meta(csv_path)
        meta = m.set_index("well_id").to_dict("index")
    except Exception:
        meta = {}
    d = daily.copy()
    # Calendar-day VOLUME columns (cadence-aware); fall back to per-record rate for
    # daily data / older frames where a span is one day (volume == rate).
    d["_pot_vol"] = d["potential_vol"] if "potential_vol" in d.columns else d["potential"]
    d["_act_vol"] = d["actual_vol"] if "actual_vol" in d.columns else d["bopd"]
    g = d.groupby("well_id").agg(
        deferred_bbl=("total_def", "sum"), deferred_usd=("deferred_usd", "sum"),
        potential=("_pot_vol", "sum"), actual=("_act_vol", "sum")).reset_index()
    g["uptime_pct"] = (g["actual"] / g["potential"] * 100.0).where(g["potential"] > 0, 100.0)
    rows = []
    for _, r in g.iterrows():
        info = meta.get(r["well_id"], {})
        rows.append({
            "Well": r["well_id"],
            "Operator": info.get("operator", "—"),
            "Field": info.get("field", "—"),
            "Formation": info.get("formation", "—"),
            "Deferred bbl": round(float(r["deferred_bbl"]), 0),
            "Deferred $": round(float(r["deferred_usd"]), 0),
            "Dominant cause": "N/A (uncoded)",
            "Uptime %": round(float(r["uptime_pct"]), 1),
        })
    return pd.DataFrame(rows).sort_values("Deferred $", ascending=False).reset_index(drop=True)


def _fleet_table_section(daily, price, is_real=False, csv_path=None) -> None:
    st.caption("One row per well — sort any column. Open a well from the **Wells** section in "
               "the sidebar to drill in (potential-vs-actual, deferred bars, events, recovery items).")
    if is_real:
        table = _build_real_fleet_table(daily, csv_path)
        if table.empty:
            st.info("No fleet data available.")
            return
        st.dataframe(
            table, width="stretch", hide_index=True,
            column_config={
                "Deferred $": st.column_config.NumberColumn(format="$%d"),
                "Deferred bbl": st.column_config.NumberColumn(format="%d"),
                "Uptime %": st.column_config.NumberColumn(format="%.1f%%"),
            })
        st.caption("Real public monthly data: volumes + uptime are real; **Dominant cause is "
                   "N/A** (no public reason codes), so there's no recoverable-$/capture-% column.")
        st.download_button(
            "⬇ Download CSV",
            data=table.to_csv(index=False),
            file_name="deferment_fleet_table.csv",
            mime="text/csv",
        )
        return
    table = _build_fleet_table(daily, price)
    if table.empty:
        st.info("No fleet data available.")
        return
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Deferred $": st.column_config.NumberColumn(format="$%d"),
            "Recoverable $": st.column_config.NumberColumn(format="$%d"),
            "Deferred bbl": st.column_config.NumberColumn(format="%d"),
            "Lateral (ft)": st.column_config.NumberColumn(format="%d"),
            "Uptime %": st.column_config.NumberColumn(format="%.1f%%"),
            "Capture %": st.column_config.NumberColumn(format="%d%%"),
        })
    st.download_button(
        "⬇ Download CSV",
        data=table.to_csv(index=False),
        file_name="deferment_fleet_table.csv",
        mime="text/csv",
    )


def _eval_section(is_real=False) -> None:
    st.subheader("Reason-Code Classifier — Eval vs. Ground-Truth Causes")
    if is_real:
        st.info("**Cause attribution N/A — no public reason codes.** Public monthly filings carry no "
                "operator cause notes, so there's no ground truth to score a classifier against on "
                "real data. The eval below is measured on the **synthetic** reason-coded set (its "
                "ground-truth labels) — the credential for the classifier; it does not run on the "
                "real Bakken extract.")
    st.caption("The event log carries a ground-truth cause the classifier never sees. The deterministic "
               "rules classifier is scored on it (precision/recall/F1 + accuracy). A CI gate fails the "
               "build under 80%. Run `python -m evals.run_evals` to refresh.")
    if EVAL.exists():
        res = json.loads(EVAL.read_text())
        e1, e2 = st.columns(2)
        e1.metric("Overall accuracy", f"{res['accuracy']*100:.0f}%", f"{res['n']} events")
        e2.metric("Classes", len(res["per_class"]))
        rows = [{"Cause": c, "Precision": m["precision"], "Recall": m["recall"],
                 "F1": m["f1"], "n": m["support"]} for c, m in res["per_class"].items()]
        pc = pd.DataFrame(rows)
        for col in ("Precision", "Recall", "F1"):
            pc[col] = pc[col].map(lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "—")
        st.dataframe(pc, width="stretch", hide_index=True)
        st.caption("Residual misses are the deliberately vague notes (e.g. \"well down, see foreman\") — "
                   "exactly where the optional LLM classifier earns its keep.")
    else:
        st.info("No eval summary yet — run `python -m evals.run_evals`.")

    theme.references(["pareto"])


# =====================================================================
# PAGE: per-well drill-down
# =====================================================================

def _render_well_real(well_id: str, price: float, csv_path: str,
                      data_source: str, badge) -> None:
    """Per-well drill-down on the active REAL extract (Colorado ECMC default, or a user's
    NDIC export): real potential-vs-actual + deferred bars from monthly records, real
    identity (operator/field/formation), cause attribution N/A (no public reason codes)."""
    if data_source == SRC_REAL_CO:
        src_label = "Colorado ECMC"
        src_chip = "Colorado · real"
    elif data_source == SRC_UPLOAD:
        src_label = "User-uploaded data"
        src_chip = "User-uploaded · real"
    else:
        src_label = "North Dakota (NDIC)"
        src_chip = "NDIC · real"
    fleet, evc, daily = _load_real(price, csv_path)
    info = {}
    try:
        from src.ndic import ndic_well_meta
        info = ndic_well_meta(csv_path).set_index("well_id").to_dict("index").get(well_id, {})
    except Exception:
        info = {}

    name = info.get("well_name", well_id)
    sub = " · ".join(str(v) for v in (info.get("operator"), info.get("field"),
                                      info.get("formation")) if v)
    theme.header(f"{well_id} · {name}",
                 subtitle=sub or f"{src_label} public monthly record",
                 chips=[(f"v{__version__}", "ver"), (src_chip, "info")])
    theme.data_badge(*badge)
    _back_to_overview()

    wd = daily[daily["well_id"] == well_id] if len(daily) else daily.iloc[0:0]
    if not len(wd):
        st.info(
            f"**{well_id}** isn't in the {src_label} extract. The per-well menu in the sidebar is "
            "keyed to the synthetic demo fleet (`well_0NN`); the real wells live in the **Fleet "
            "table** on the overview. Open the **Fleet overview** to browse the real wells, or "
            "switch the **Data source** back to *Synthetic (demo)* to drill into this well.")
        _back_to_overview()
        return

    deferred_bbl = float(wd["total_def"].sum())
    # Calendar-day volumes (cadence-aware) for the uptime ratio; fall back to rate sums.
    potential = float((wd["potential_vol"] if "potential_vol" in wd else wd["potential"]).sum())
    actual = float((wd["actual_vol"] if "actual_vol" in wd else wd["bopd"]).sum())
    uptime = (actual / potential * 100.0) if potential > 0 else 100.0

    m = st.columns(4)
    m[0].metric("Deferred bbl", f"{deferred_bbl:,.0f}", help="Real — potential vs. actual")
    m[1].metric("Deferred $", f"${deferred_bbl * price:,.0f}", delta_color="inverse")
    m[2].metric("Uptime %", f"{uptime:.1f}%", help="Actual ÷ potential (downtime from days-produced)")
    m[3].metric("Dominant cause", "N/A", help="No public reason codes — cause attribution N/A")

    st.subheader("Potential vs. Actual — Deferred Barrels (Monthly)")
    fig = go.Figure()
    fig.add_scatter(x=wd["date"], y=wd["potential"], name="Potential",
                    line=dict(color=theme.BLUE, dash="dash"))
    fig.add_scatter(x=wd["date"], y=wd["bopd"], name="Actual BOPD",
                    line=dict(color=theme.NAVY))
    # Bars share the BOPD axis: deferred VOLUME averaged over each month's calendar days.
    _def_rate = (wd["total_def"] / wd["span_days"]) if "span_days" in wd else wd["total_def"]
    fig.add_bar(x=wd["date"], y=_def_rate, name="Deferred (avg BOPD)",
                marker_color=theme.RED, opacity=0.5)
    st.plotly_chart(theme.style_fig(fig, height=380), width="stretch")
    theme.source_note(
        "Monthly: rate = oil_bbl ÷ days-produced; potential is decline-aware (P75 of full-uptime "
        "months); deferred = potential_calendar_volume − actual, split into downtime "
        "(days_in_month − days-produced) vs. underperformance. Bars show deferred volume as an "
        "avg BOPD over the month; the Deferred-bbl metric above is the true monthly volume. "
        "Volumes/downtime real; cause N/A.")
    st.caption("Monthly cadence: rate = oil_bbl ÷ days-produced; downtime = days_in_month − "
               "days-produced. Volumes and downtime are real; **cause is N/A** (no public reason codes).")

    st.subheader("Recovery Items for This Well")
    st.info("**Cause attribution N/A — no public reason codes.** Recovery items require a coded "
            "cause to attribute and authorize; public monthly data has none. The deferment quantity "
            "above is real.")
    theme.references(["deferment", "npv"])
    _back_to_overview()


def render_well(well_id: str) -> None:
    price, byok_key, use_llm, data_source, uploaded = _sidebar_controls()
    is_real, real_csv, badge = _resolve_source(data_source, uploaded)
    if is_real and real_csv == "UPLOAD":
        if uploaded is None:
            st.info("Upload a CSV to analyze your own fleet.")
            st.stop()
        import tempfile, os
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            _render_well_real(well_id, price, tmp_path, data_source, badge)
        except Exception as e:
            st.exception(e)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return
    if is_real:
        _render_well_real(well_id, price, real_csv, data_source, badge)
        return

    fleet, evc, daily = _load(price, use_llm, bool(byok_key), byok_key)
    meta = fleet_registry.get(well_id)

    theme.header(
        f"{well_id} · {meta.name}",
        subtitle=f"{meta.lift} · {meta.basin} · {meta.formation} · {meta.area}",
        chips=[(f"v{__version__}", "ver"), (meta.peer_group, "info")],
    )
    theme.data_badge(*_BADGE_SYNTHETIC)
    theme.well_cross_links("deferment", well_id)
    _back_to_overview()

    wd = daily[daily["well_id"] == well_id]
    if not len(wd):
        st.warning("No production history for this well.")
        _back_to_overview()
        return

    deferred_bbl = float(wd["total_def"].sum())
    deferred_usd = deferred_bbl * price
    # Calendar-day volumes (cadence-aware) for the uptime ratio; fall back to rate sums.
    potential = float((wd["potential_vol"] if "potential_vol" in wd else wd["potential"]).sum())
    actual = float((wd["actual_vol"] if "actual_vol" in wd else wd["bopd"]).sum())
    uptime = (actual / potential * 100.0) if potential > 0 else 100.0

    # dominant cause + recovery items for this well (reuse the same analytics).
    loss = wd[wd["total_def"] > 1e-6]
    if len(loss):
        cause_key = (loss.groupby("reason_key")["deferred_usd"].sum()
                     .sort_values(ascending=False).index[0])
        from src.reason_codes import label_for
        dominant_cause = label_for(cause_key)
    else:
        dominant_cause = "—"
    well_queue = A.recovery_queue(wd, evc[evc["well_id"] == well_id] if "well_id" in evc.columns
                                  else None, price)

    m = st.columns(5)
    m[0].metric("Deferred bbl", f"{deferred_bbl:,.0f}")
    m[1].metric("Deferred $", f"${deferred_usd:,.0f}", delta_color="inverse")
    m[2].metric("Uptime %", f"{uptime:.1f}%", help="Actual ÷ potential over the period")
    m[3].metric("Dominant cause", dominant_cause)
    m[4].metric("Lateral (ft)", f"{meta.lateral_length_ft:,}")

    # potential (dashed) vs actual BOPD + deferred bars overlay
    st.subheader("Potential vs. Actual — Deferred Barrels")
    fig = go.Figure()
    fig.add_scatter(x=wd["date"], y=wd["potential"], name="Potential",
                    line=dict(color=theme.BLUE, dash="dash"))
    fig.add_scatter(x=wd["date"], y=wd["bopd"], name="Actual BOPD",
                    line=dict(color=theme.NAVY))
    # Bars share the BOPD axis: deferred volume averaged over each record's calendar span
    # (daily data → one day, so this equals the day's deferred bbl).
    _def_rate = (wd["total_def"] / wd["span_days"]) if "span_days" in wd else wd["total_def"]
    fig.add_bar(x=wd["date"], y=_def_rate, name="Deferred (avg BOPD)",
                marker_color=theme.RED, opacity=0.5)
    st.plotly_chart(theme.style_fig(fig, height=380), width="stretch")
    theme.source_note(
        "Potential from full-uptime days (P75, decline-aware); deferred = potential − "
        "actual, split into downtime vs. underperformance; bars are deferred volume as avg BOPD.")

    # events table for this well
    ev = evc[evc["well_id"] == well_id] if "well_id" in evc.columns else pd.DataFrame()
    if len(ev):
        st.subheader("Events for This Well")
        show = ev[["start_date", "end_date", "note", "reason_key"]].copy()
        from src.reason_codes import label_for
        show["reason_key"] = show["reason_key"].map(label_for)
        show.columns = ["Start", "End", "Operator note", "Classified cause"]
        st.dataframe(show, width="stretch", hide_index=True)
    else:
        st.caption("No downtime/curtailment events logged for this well.")

    # this well's recovery items
    st.subheader("Recovery Items for This Well")
    if len(well_queue):
        wq = well_queue.copy()
        wq["recoverable_usd"] = wq["recoverable_usd"].map(lambda v: f"${v:,.0f}")
        wq["recoverable_bbl"] = wq["recoverable_bbl"].map(lambda v: f"{v:,.0f}")
        wq["mttr_days"] = wq["mttr_days"].map(lambda v: f"{v:.1f}")
        wq = wq[["cause", "suggested_action", "recoverable_bbl", "recoverable_usd", "mttr_days"]]
        wq.columns = ["Cause", "Suggested action", "Recoverable bbl", "Recoverable $", "MTTR (d)"]
        st.dataframe(wq, width="stretch", hide_index=True)
        st.caption(f"[Authorize an intervention in AFE Copilot ↗]({AFE_COPILOT_URL})")
    else:
        st.info("No recoverable deferment for this well — nothing to queue.")

    theme.references(["deferment", "npv"])
    _back_to_overview()


# =====================================================================
# Shared setup (runs every rerun) + navigation
# =====================================================================

theme.setup_page("Deferment IQ", icon="🛢️")
theme.suite_nav("deferment")
_bootstrap()

overview = st.Page(render_overview, title="Fleet Overview", icon="📊", default=True)
wells = [
    st.Page(partial(render_well, wid), title=wid, url_path=wid)
    for wid in _fleet_well_ids()
]
st.navigation({"Fleet": [overview], "Wells": wells}).run()
