"""Streamlit AFE Copilot — multipage fleet-overview + per-AFE drill-down.

Multipage (``st.navigation`` + ``st.Page``): an **Overview** page (in-flight KPIs,
a sortable AFE pipeline table, plus Draft / Variance / Cost-Benchmark tabs) and one
**drill-down page per AFE** in the SQLite tracker (cost waterfall, net economics +
tornado, risk register, authority routing, immutable audit trail, and actual-vs-AFE
variance for that AFE).

Deterministic end-to-end: every cost / economics / variance / docx feature runs with
ZERO API key. LLM AFE-narrative drafting is BYOK-optional (key entered in the Draft
tab body). Charts use ``theme.style_fig``.
"""
from __future__ import annotations

import json
import sys
import tempfile
from functools import partial
from pathlib import Path

# Ensure repo root is on sys.path so `src.*` imports work on Streamlit Cloud, and the
# demo dir so the vendored `theme` / `fleet_registry` resolve regardless of cwd
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
import plotly.express as px
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
from src.cost_db import (
    COST_TEMPLATES, cost_rollup, lookup_cost_template, total_estimate)
from src.drafter import AFEDiagnosis, MissingAPIKey, run_drafter
try:
    from src.economics import jib_split, price_sensitivity, simulate_economics
    _MC_AVAILABLE = True
except Exception as _mc_err:  # never let an optional analytics import take down the app
    simulate_economics = None
    price_sensitivity = None
    jib_split = None
    _MC_AVAILABLE = False
    _MC_IMPORT_ERROR = repr(_mc_err)
from src.models import AFEDiagnosis as AFEDiagnosisModel
from src.risk_register import lookup_risks
from src.tracker import (
    AFETracker, IN_FLIGHT_STATUSES, required_approver, seed_demo_data)
from src.variance import (
    SUPPLEMENT_THRESHOLD_PCT, analyze_variance, demo_variance_data)


DB_PATH = REPO_ROOT / "pipeline.sqlite"


# ---- cached loads ----------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_tracker(db_path: str) -> AFETracker:
    """Cached tracker handle.

    Uses cache_resource (NOT cache_data) because it returns a custom ``AFETracker``
    instance — Streamlit's cache_data serializer rejects custom classes on
    Python 3.14 / newer Streamlit. The tracker only wraps a path + opens a short-lived
    SQLite connection per call, so sharing it across sessions is safe."""
    return AFETracker(db_path)


@st.cache_data(show_spinner=False)
def _pipeline_df(db_path: str, _cache_token: int) -> pd.DataFrame:
    """The AFE pipeline as a plain DataFrame (cache_data-safe).

    ``_cache_token`` lets the caller bust this cache after seeding/inserts without
    touching the cached tracker resource."""
    return _get_tracker(db_path).as_dataframe()


@st.cache_data(show_spinner=False)
def _events_df(db_path: str, afe_number: str, _cache_token: int) -> pd.DataFrame:
    """Audit-trail events for one AFE as a plain DataFrame (cache_data-safe)."""
    return _get_tracker(db_path).events(afe_number)


@st.cache_data(show_spinner=False)
def _variance_for(afe_number: str) -> pd.DataFrame | None:
    """Per-AFE actual-vs-AFE line detail (plain DataFrame), or None if no actuals.

    Reuses the deterministic demo variance dataset; returns the merged line-level
    frame restricted to this AFE so the drill-down can render its own variance."""
    afe_df, actuals_df = demo_variance_data()
    if afe_number not in set(afe_df["afe_number"]) | set(actuals_df["afe_number"]):
        return None
    sub_afe = afe_df[afe_df["afe_number"] == afe_number]
    sub_act = actuals_df[actuals_df["afe_number"] == afe_number]
    merged = sub_afe.merge(sub_act, on=["afe_number", "category"], how="outer").fillna(0)
    merged["variance_usd"] = merged["actual_usd"] - merged["line_total_usd"]
    return merged.sort_values("variance_usd", ascending=False)


def _variance_pct_for(afe_number: str) -> float | None:
    """AFE-level actual-vs-budget % overrun for the pipeline table, or None."""
    m = _variance_for(afe_number)
    if m is None:
        return None
    budget = float(m["line_total_usd"].sum())
    actual = float(m["actual_usd"].sum())
    return ((actual - budget) / budget * 100.0) if budget else None


def _bump_cache_token() -> int:
    st.session_state["_afe_cache_token"] = st.session_state.get("_afe_cache_token", 0) + 1
    return st.session_state["_afe_cache_token"]


# ---- shared helpers --------------------------------------------------------

def _back_to_overview() -> None:
    target = globals().get("overview")
    try:
        st.page_link(target if target is not None else "app.py",
                     label="← Back to AFE overview", icon="📋")
    except Exception:
        pass


def _registry_meta(well_id: str):
    """Return fleet_registry metadata ONLY for shared ``well_0NN`` ids; else None.

    The tracker's demo wells use the ``ED-NNH`` convention, which is not part of the
    shared fleet registry, so enrichment is conditional (the registry never raises,
    but we only show it when the id actually follows the suite convention)."""
    if isinstance(well_id, str) and well_id.startswith("well_"):
        return fleet_registry.get(well_id)
    return None


def _cost_waterfall(intervention: str, total: float) -> go.Figure:
    """Cost waterfall — direct line items → contingency → total AFE. (Logic preserved
    byte-for-byte from the original Draft tab, just factored into a helper.)"""
    _wf_items = lookup_cost_template(intervention)
    _direct = [li for li in _wf_items if li.category != "Contingency"]
    _contingency = sum(li.total_usd for li in _wf_items if li.category == "Contingency")
    _wf_labels = [li.category for li in _direct] + ["Contingency", "Total AFE cost"]
    _wf_measures = ["relative"] * (len(_direct) + 1) + ["total"]
    _wf_values = [li.total_usd for li in _direct] + [_contingency, 0]
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=_wf_measures,
        x=_wf_labels,
        y=_wf_values,
        text=[f"${v:,.0f}" for v in _wf_values[:-1]] + [f"${total:,.0f}"],
        textposition="outside",
        connector={"line": {"color": theme.GRID}},
        increasing={"marker": {"color": theme.BLUE}},
        decreasing={"marker": {"color": theme.RED}},
        totals={"marker": {"color": theme.NAVY}},
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
    ))
    fig_wf.update_layout(title="Cost Waterfall — Line Items → Contingency → Total AFE",
                         yaxis_title="USD")
    return theme.style_fig(fig_wf, height=360, legend=False)


def _tornado_fig(mc) -> go.Figure:
    """Tornado — NPV swing per variable (logic preserved from the original Draft tab)."""
    items = sorted(mc.tornado.items(), key=lambda kv: kv[1]["swing"])
    labels = [k.replace("_", " ") for k, _ in items]
    lows = [v["low"] for _, v in items]
    highs = [v["high"] for _, v in items]
    base = mc.base_npv_usd
    fig_t = go.Figure()
    fig_t.add_trace(go.Bar(
        y=labels, x=[base - lo for lo in lows], base=lows,
        orientation="h", name="downside", marker_color=theme.RED,
        hovertemplate="low NPV: $%{base:,.0f}<extra></extra>",
    ))
    fig_t.add_trace(go.Bar(
        y=labels, x=[hi - base for hi in highs], base=base,
        orientation="h", name="upside", marker_color=theme.BLUE,
        hovertemplate="high NPV: $%{x:,.0f}<extra></extra>",
    ))
    fig_t.add_vline(x=base, line_dash="dash", line_color=theme.NAVY,
                    annotation_text=f"base ${base/1e6:,.2f}M")
    fig_t.update_layout(barmode="overlay", showlegend=True,
                        xaxis_title="NPV @ 10% (USD)",
                        title="Tornado — NPV Swing Per Variable")
    return theme.style_fig(fig_t, height=320)


# =====================================================================
# PAGE: Overview
# =====================================================================

def render_overview() -> None:
    theme.header(
        "AFE Copilot",
        subtitle="Draft, track, and analyze AFEs — built for multi-rig E&P operators. "
                 "Deterministic cost / economics / variance; LLM narrative is BYOK-optional.",
        chips=[(f"v{__version__}", "ver"), ("AFE pipeline", "info"),
               ("document agent", "info")],
    )
    theme.data_badge("synthetic", "Illustrative AFE cost templates + pipeline tracker — cost/authority data is never public.")

    theme.how_to(
        "- An **AFE (Authorization For Expenditure)** is the capital-approval document an "
        "operator signs before spending on a well job — it states the scope, the cost "
        "breakdown, and the expected economics so the right authority level can approve it.\n"
        "- **Draft New AFE** turns a well diagnosis (typed in, loaded from an example, or "
        "chained from the Production Engineer Copilot) into a costed AFE: a benchmark cost "
        "table, tangible/intangible (IDC) split, **net NPV** to the operator (working-"
        "interest cost share, NRI revenue share), a price-deck sensitivity strip, and a "
        "Monte-Carlo P10/P50/P90 — all deterministic, no API key. Add your own Anthropic "
        "key only to draft the AI-written narrative.\n"
        "- **Routing** maps each AFE's dollar value to the required sign-off level "
        "(delegation-of-authority limits: PE < $50k · Eng Mgr < $250k · Ops Mgr < $1MM · "
        "VP above).\n"
        "- The **AFE pipeline** tracks every in-flight AFE in a local SQLite store — gross "
        "cost, net NPV, status, days-in-status, and an immutable audit trail. Open any AFE "
        "from the **AFEs** section in the sidebar to drill in, and review closed-out jobs "
        "in **Actual-vs-AFE variance** (a supplemental AFE is flagged when actuals run "
        ">10% over)."
    )

    with st.expander(f"🆕 What's New in v{__version__}"):
        st.markdown(
            "- **Multipage explorer** — an Overview plus a **drill-down page per AFE** "
            "(`st.navigation`): each AFE's cost waterfall, net economics + tornado, risk "
            "register, authority routing, immutable audit trail, and actual-vs-AFE variance.\n"
            "- **Sortable AFE pipeline table** — one row per AFE (cost, net NPV, status, "
            "required approver, days-in-status, supplement flag, variance) — sort any column, "
            "then open the AFE from the **AFEs** section in the sidebar.\n"
            "- **Unified dark + navy suite theme** + cross-app sidebar suite navigator.\n"
            "- One-click AFE export to Word (.docx) and the cost waterfall retained in **Draft New AFE**.\n"
            "- Shared **fleet registry** enrichment when an AFE references a suite `well_0NN` id."
        )

    token = st.session_state.get("_afe_cache_token", 0)
    df = _pipeline_df(str(DB_PATH), token)

    if df.empty:
        st.info("No AFEs yet. Use the **Draft New AFE** tab below to create one.")
    else:
        _overview_kpis(df)
        _overview_table(df)

    st.divider()
    tab_drafter, tab_variance, tab_benchmarks = st.tabs(
        ["📝 Draft New AFE", "📊 Variance", "🏷️ Cost Benchmarks"])
    with tab_drafter:
        _drafter_panel()
    with tab_variance:
        _variance_panel()
    with tab_benchmarks:
        _benchmarks_panel()


def _overview_kpis(df: pd.DataFrame) -> None:
    in_flight_mask = df.status.isin(list(IN_FLIGHT_STATUSES))
    # Count AFEs whose cost lands above the Production-Engineer authority limit
    # (i.e. anything needing an Eng Mgr or higher sign-off).
    over_pe_threshold = int((df["required_approver"] != "Production Engineer").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("In-flight $", f"${df.loc[in_flight_mask, 'total_cost_usd'].sum() / 1e6:.1f}M",
              help="Total cost of draft / engineering-review / finance-review AFEs "
                   "(excludes executed + rejected).")
    c2.metric("In-flight AFEs", int(in_flight_mask.sum()))
    c3.metric("Approved (not executed)", int((df.status == "approved").sum()))
    c4.metric("Above PE authority", over_pe_threshold,
              help="AFEs whose $ value needs an Engineering Manager or higher sign-off.")

    st.caption("**By status** · " + " · ".join(
        f"{s.replace('_', ' ')}: {int((df.status == s).sum())}"
        for s in ["draft", "engineering_review", "finance_review",
                  "approved", "executed", "rejected"]))
    st.caption("**By required approver** · " + " · ".join(
        f"{role}: {int((df.required_approver == role).sum())}"
        for role in df.required_approver.unique()))


def _overview_table(df: pd.DataFrame) -> None:
    st.subheader("AFE Pipeline")
    st.caption(
        "One row per AFE — sort any column. **Supplement?** flags an actual >10% over "
        "the AFE (a supplemental AFE is policy-required). Open an AFE from the **AFEs** "
        "section in the sidebar to drill into its waterfall, economics, risks, and audit trail.")

    rows = []
    for _, r in df.iterrows():
        afe_no = r["afe_number"]
        try:
            net_npv = _net_npv_for(r)
        except Exception:
            net_npv = None
        var_pct = _variance_pct_for(afe_no)
        supplement = (var_pct is not None and var_pct > SUPPLEMENT_THRESHOLD_PCT)
        rows.append({
            "AFE #": afe_no,
            "Well/Project": r["well_id"],
            "Intervention": r["intervention"],
            "Gross cost $": float(r["total_cost_usd"]),
            "Net NPV $": net_npv,
            "Status": r["status"],
            "Required approver": r["required_approver"],
            "Days in status": int(r["days_in_status"]),
            "Supplement?": "⚠️" if supplement else "",
            "Variance vs actual": (f"{var_pct:+.0f}%" if var_pct is not None else "—"),
        })
    table = pd.DataFrame(rows)
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "Gross cost $": st.column_config.NumberColumn(format="$%,.0f"),
            "Net NPV $": st.column_config.NumberColumn(format="$%,.0f"),
        },
    )
    st.download_button("⬇ Download pipeline (CSV)", data=table.to_csv(index=False),
                       file_name="afe_pipeline.csv", mime="text/csv")
    st.caption("`Required approver` is the delegation-of-authority level the AFE's $ value "
               "needs (PE < $50k · Eng Mgr < $250k · Ops Mgr < $1MM · VP above).")
    theme.source_note(
        "Net NPV (USD) = deterministic DCF at +100 BOPD base uplift, for cross-AFE ranking "
        "only; gross cost (USD) from the AFE's benchmark cost template; variance = "
        "actual − AFE budget (%).")


def _net_npv_for(row) -> float | None:
    """Deterministic net-NPV estimate for a tracker row using its intervention's
    benchmark cost. Uses a nominal +100 BOPD uplift at base assumptions (the same
    deterministic economics engine the Draft tab uses) — for ranking only, so the
    table has a comparable economics column. None when economics is unavailable or
    the intervention is cost-only (P&A)."""
    if not _MC_AVAILABLE or row["intervention"] not in COST_TEMPLATES:
        return None
    if row["intervention"] == "p_and_a":
        return None
    from src.economics import compute_economics as _ce
    net = _ce(float(row["total_cost_usd"]), 100.0)
    return float(net.net_npv_10pct_usd)


# =====================================================================
# Overview tabs (Draft / Variance / Benchmarks) — logic preserved
# =====================================================================

def _drafter_panel() -> None:
    st.subheader("Generate a New AFE")
    st.caption(
        "Cost tables, tangible/intangible split, net economics, price deck, and "
        "Monte-Carlo all work without a key. Enter your **own** Anthropic key below "
        "only to draft the AI-written AFE narrative (used for this session, never stored).")
    byok_key = st.text_input(
        "🔑 Anthropic API key (optional)", type="password",
        help="Bring your own key — used only for this session, never stored. Powers the "
             "AI-written AFE narrative.")

    # ---- One-click chain from Production Engineer Copilot -------------------
    with st.expander("🔗 Chain From Production Engineer Copilot (Paste Diagnosis JSON)"):
        st.caption(
            "Paste a diagnosis exported by the Production Engineer Copilot (Project 1). "
            "It is validated before it can become an AFE — invalid fields are reported "
            "in plain English instead of a stack trace.")
        pe_upload = st.file_uploader("Upload PE-Copilot diagnosis .json", type=["json"],
                                     key="pe_copilot_upload")
        pe_text = st.text_area("…or paste the diagnosis JSON here", height=160,
                               key="pe_copilot_text")
        if st.button("Validate & load into drafter", key="pe_copilot_load"):
            raw = None
            if pe_upload is not None:
                raw = pe_upload.getvalue().decode("utf-8")
            elif pe_text.strip():
                raw = pe_text
            if not raw:
                st.warning("Paste JSON or upload a file first.")
            else:
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as e:
                    st.error(f"That isn't valid JSON: {e}")
                else:
                    try:
                        diag = AFEDiagnosisModel.from_pe_copilot(payload)
                    except ValueError as e:
                        st.error("Diagnosis rejected:")
                        for line in str(e).splitlines():
                            st.markdown(line)
                    else:
                        st.session_state["pe_preset"] = {
                            "well_id": diag.well_id,
                            "api_number": diag.api_number,
                            "field": diag.field,
                            "operator": diag.operator,
                            "intervention": diag.intervention,
                            "primary_diagnosis": diag.primary_diagnosis,
                            "incremental_rate_bopd": diag.incremental_rate_bopd,
                            "expected_uplift_decline_per_yr": diag.expected_uplift_decline_per_yr,
                            "requested_by": diag.requested_by,
                        }
                        st.success(
                            f"Validated diagnosis for {diag.well_id} "
                            f"({diag.intervention}). Fields loaded below.")

    examples_dir = REPO_ROOT / "examples"
    sample_files = sorted(examples_dir.glob("well_diagnosis*.json")) if examples_dir.exists() else []
    if sample_files:
        chosen = st.selectbox("Or load an example", ["(custom)"] + [str(p) for p in sample_files])
    else:
        chosen = "(custom)"

    if chosen != "(custom)":
        with open(chosen) as f:
            preset = json.load(f)
    elif "pe_preset" in st.session_state:
        preset = st.session_state["pe_preset"]
    else:
        preset = {}

    well_id = st.text_input("Well ID", value=preset.get("well_id", ""))
    api = st.text_input("API #", value=preset.get("api_number", ""))
    field = st.text_input("Field", value=preset.get("field", ""))
    operator = st.text_input("Operator", value=preset.get("operator", ""))
    intervention = st.selectbox(
        "Intervention type", list(COST_TEMPLATES),
        index=list(COST_TEMPLATES).index(preset.get("intervention", "acid_stimulation"))
        if preset.get("intervention") in COST_TEMPLATES else 0)
    diagnosis_text = st.text_area("Primary diagnosis (free-form)",
                                  value=preset.get("primary_diagnosis", ""), height=120)
    incremental_rate = st.number_input("Incremental uplift (BOPD)",
                                       value=float(preset.get("incremental_rate_bopd", 100)))
    decline = st.number_input("Uplift decline (per year)",
                              value=float(preset.get("expected_uplift_decline_per_yr", 0.6)))
    requested_by = st.text_input("Requested by",
                                 value=preset.get("requested_by", "Eric Diaz, Staff PE"))

    # ---- Net economics & price deck (deterministic — no API key needed) -----
    st.markdown("---")
    st.subheader("Net Economics, Price Deck & Partner Split")
    rollup = cost_rollup(intervention)
    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("AFE total (gross)", f"${rollup['total']:,.0f}")
    gc2.metric("Tangible (capitalized)", f"${rollup['tangible']:,.0f}")
    gc3.metric("Intangible (IDC)", f"${rollup['intangible']:,.0f}")

    st.plotly_chart(_cost_waterfall(intervention, rollup["total"]), width="stretch")
    theme.source_note(
        "Benchmark cost template for the selected intervention; bars in USD, building "
        "direct line items → contingency → total AFE.")

    wc1, wc2, wc3 = st.columns(3)
    working_interest = wc1.number_input("Working interest (WI)", 0.0, 1.0, 1.0, 0.05,
                                        help="Operator's share of COST.")
    net_revenue_interest = wc2.number_input("Net revenue interest (NRI)", 0.0, 1.0, 0.80, 0.01,
                                            help="Operator's share of REVENUE after royalty.")
    realized_price = wc3.number_input("Realized price ($/bbl)", 20.0, 150.0, 65.0, 1.0)

    if _MC_AVAILABLE and incremental_rate > 0:
        from src.economics import compute_economics as _ce
        net = _ce(rollup["total"], incremental_rate, uplift_decline_per_yr=decline,
                  realized_price_per_bbl=realized_price,
                  working_interest=working_interest, net_revenue_interest=net_revenue_interest)
        nc1, nc2, nc3 = st.columns(3)
        nc1.metric("Gross NPV @ 10%", f"${net.npv_10pct_usd/1e6:,.2f}M")
        nc2.metric("Net NPV to operator", f"${net.net_npv_10pct_usd/1e6:,.2f}M",
                   help="WI% of cost, NRI% of revenue — what the operator actually books.")
        nc3.metric("Payout", f"{net.payout_months:.0f} mo"
                   if net.payout_months != float('inf') else "—")

        deck = price_sensitivity(rollup["total"], incremental_rate,
                                 uplift_decline_per_yr=decline,
                                 working_interest=working_interest,
                                 net_revenue_interest=net_revenue_interest)
        deck_df = pd.DataFrame(deck)
        deck_df = deck_df.assign(
            **{"Realized $/bbl": deck_df["realized_price"].map(lambda v: f"${v:,.0f}"),
               "Gross NPV": deck_df["npv_usd"].map(lambda v: f"${v/1e6:,.2f}M"),
               "Net NPV": deck_df["net_npv_usd"].map(lambda v: f"${v/1e6:,.2f}M"),
               "Payout (mo)": deck_df["payout_months"].map(
                   lambda v: f"{v:.0f}" if v != float('inf') else "—")})
        st.caption("Price-deck sensitivity (NPV at a fixed rate across a realized-price strip):")
        st.dataframe(deck_df[["Realized $/bbl", "Gross NPV", "Net NPV", "Payout (mo)"]],
                     width="stretch", hide_index=True)

        if working_interest < 1.0:
            partners = {"Operator (you)": working_interest,
                        "Non-op partner(s)": round(1.0 - working_interest, 4)}
            jib = pd.DataFrame(jib_split(rollup["total"], partners))
            jib["net_cost_usd"] = jib["net_cost_usd"].map(lambda v: f"${v:,.0f}")
            jib["working_interest"] = jib["working_interest"].map(lambda v: f"{v:.0%}")
            st.caption("JIB cost allocation (gross AFE billed by working interest):")
            st.dataframe(jib[["partner", "working_interest", "net_cost_usd"]],
                         width="stretch", hide_index=True)

    # ---- Monte-Carlo economics (pure numpy — no API key needed) -------------
    st.markdown("---")
    st.subheader("Probabilistic Economics (Monte-Carlo, Gross)")
    st.caption(
        "10,000 trials over incremental rate (±30%), uplift decline (±0.15 abs), "
        "and realized price (~$12 sd). Treatment cost is the benchmark estimate for "
        "the selected intervention.")
    if not _MC_AVAILABLE:
        st.info("Probabilistic economics is temporarily unavailable in this build; "
                "the rest of the app is unaffected.")
    elif st.button("Run Monte-Carlo NPV"):
        if incremental_rate <= 0:
            st.error("Incremental uplift must be greater than 0 to run economics.")
        else:
            treatment_cost = total_estimate(intervention)
            mc = simulate_economics(
                treatment_cost_usd=treatment_cost,
                incremental_rate_bopd=incremental_rate,
                uplift_decline_per_yr=decline,
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P10 NPV (downside)", f"${mc.npv_p10_usd/1e6:,.2f}M")
            m2.metric("P50 NPV (median)", f"${mc.npv_p50_usd/1e6:,.2f}M")
            m3.metric("P90 NPV (upside)", f"${mc.npv_p90_usd/1e6:,.2f}M")
            m4.metric("P(payout < 24 mo)", f"{mc.probability_of_payout*100:.0f}%")
            st.plotly_chart(_tornado_fig(mc), width="stretch")
            theme.source_note(
                "NPV @ 10% (USD) swing as each variable moves over its sampled range; "
                "dashed line is the base-case NPV.")

    theme.references(["npv"])

    if st.button("Draft AFE", type="primary"):
        if not well_id or not diagnosis_text:
            st.error("Well ID and primary diagnosis are required.")
        else:
            diagnosis = AFEDiagnosis(
                well_id=well_id, api_number=api, field=field, operator=operator,
                intervention=intervention, primary_diagnosis=diagnosis_text,
                incremental_rate_bopd=incremental_rate,
                expected_uplift_decline_per_yr=decline,
                requested_by=requested_by,
            )
            try:
                with st.spinner("Drafting AFE..."):
                    markdown = run_drafter(diagnosis, api_key=byok_key or None)
                st.markdown(markdown)
                dl_md, dl_docx = st.columns(2)
                dl_md.download_button(
                    "Download .md", markdown,
                    file_name=f"AFE_{well_id}_{intervention}.md")
                from src.docx_builder import build_docx
                with tempfile.TemporaryDirectory() as _td:
                    _docx_path = build_docx(
                        markdown, Path(_td) / f"AFE_{well_id}_{intervention}.docx", diagnosis)
                    _docx_bytes = _docx_path.read_bytes()
                dl_docx.download_button(
                    "Download AFE (.docx)", _docx_bytes,
                    file_name=f"AFE_{well_id}_{intervention}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except MissingAPIKey:
                st.warning("Enter your **Anthropic API key** above to draft the AFE narrative. "
                           "Everything else on this page — cost tables, tangible/intangible split, "
                           "net economics, price deck, and Monte-Carlo — works without a key.")
            except Exception as _draft_err:  # bad/out-of-credit key, rate limit, network, docx
                st.error(
                    "The AFE narrative couldn't be drafted — the Anthropic API call failed "
                    "(commonly an invalid, rate-limited, or out-of-credit key). Check the key "
                    "you entered above and try again. Everything else on this page — cost "
                    "tables, tangible/intangible split, net economics, price deck, and "
                    "Monte-Carlo — works without a key.")
                st.caption(f"Details: {type(_draft_err).__name__}: {_draft_err}")
            else:
                # Stash the current form values so the submit button below can read them
                # even after Streamlit reruns this panel (button click → rerun).
                st.session_state["_pending_draft"] = {
                    "well_id": well_id,
                    "intervention": intervention,
                    "total_cost_usd": total_estimate(intervention),
                    "requested_by": requested_by,
                }

    # ---- Submit drafted AFE to the pipeline tracker -------------------------
    pending = st.session_state.get("_pending_draft")
    if pending:
        st.divider()
        st.caption(
            f"Draft ready: **{pending['well_id']}** · {pending['intervention'].replace('_', ' ')} "
            f"· gross ${pending['total_cost_usd']:,.0f}"
        )
        if st.button("Submit to Pipeline", type="primary", key="submit_to_pipeline"):
            import datetime as _dt
            tracker = _get_tracker(str(DB_PATH))
            # Generate a sequential AFE number that won't collide with seed rows.
            _tok = st.session_state.get("_afe_cache_token", 0)
            _existing = _pipeline_df(str(DB_PATH), _tok)
            _next_num = (
                max(
                    (int(n.split("-")[-1]) for n in _existing["afe_number"]
                     if n.startswith("AFE-")),
                    default=53,
                ) + 1
            )
            new_afe_id = f"AFE-{_dt.date.today().year}-{_next_num:04d}"
            today_iso = _dt.date.today().isoformat()
            from src.tracker import AFERecord as _AFERec
            tracker.upsert(_AFERec(
                afe_number=new_afe_id,
                well_id=pending["well_id"],
                intervention=pending["intervention"],
                total_cost_usd=pending["total_cost_usd"],
                status="draft",
                created_date=today_iso,
                last_updated=today_iso,
                requested_by=pending["requested_by"],
                notes="Submitted via Draft New AFE",
            ))
            # Clear the pending draft and bust the pipeline cache so the table refreshes.
            del st.session_state["_pending_draft"]
            _bump_cache_token()
            st.success(f"AFE **{new_afe_id}** added to the pipeline as *draft*. "
                       "Refresh the Overview to see it in the table.")
            st.rerun()


def _variance_panel() -> None:
    st.subheader("Actual-vs-AFE Variance (Closed-Out AFEs)")
    st.caption("Demo actuals for two closed AFEs — including a 100%-unbudgeted 'Fishing' line "
               "and a rig overrun that trips the supplemental-AFE policy (>10%).")
    afe_df, actuals_df = demo_variance_data()
    vs = analyze_variance(afe_df, actuals_df)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("AFEs analyzed", vs.n_afes)
    v2.metric("Portfolio variance", f"{vs.overall_variance_pct:+.1f}%")
    v3.metric("Over budget", vs.over_budget_count)
    v4.metric("Total actual", f"${vs.total_actual_usd/1e6:,.2f}M")

    if vs.worst_offender_category:
        pct = f" ({vs.worst_offender_pct:+.0f}%)" if vs.worst_offender_pct is not None else " (unbudgeted)"
        st.markdown(f"**Worst-offender category:** {vs.worst_offender_category} — "
                    f"**${vs.worst_offender_overrun_usd:,.0f}** overrun{pct}")
    if vs.unbudgeted_categories:
        st.warning("Unbudgeted actuals (no AFE line existed): "
                   + ", ".join(vs.unbudgeted_categories))
    if vs.supplement_required_afes:
        st.error("⚠️ Supplemental AFE required (actuals exceed AFE by >10%): "
                 + ", ".join(vs.supplement_required_afes))

    merged = afe_df.merge(actuals_df, on=["afe_number", "category"], how="outer").fillna(0)
    merged["variance_usd"] = merged["actual_usd"] - merged["line_total_usd"]
    merged = merged.sort_values("variance_usd", ascending=False)
    disp = merged.copy()
    for c in ("line_total_usd", "actual_usd", "variance_usd"):
        disp[c] = disp[c].apply(lambda v: f"${v:,.0f}")
    disp.columns = ["AFE", "Category", "AFE budget", "Actual", "Variance"]
    st.dataframe(disp, width="stretch", hide_index=True)
    st.download_button("⬇ Download variance (CSV)", data=merged.to_csv(index=False),
                       file_name="afe_variance.csv", mime="text/csv")
    theme.source_note(
        "Per-category variance (USD) = actual − AFE budget; rows sorted by largest "
        "overrun. Supplemental AFE flags an overrun above the policy threshold "
        f"(>{SUPPLEMENT_THRESHOLD_PCT:.0f}%).")


def _benchmarks_panel() -> None:
    st.subheader("Reference Cost Per Intervention (Synthetic Permian Benchmarks)")
    rows = []
    for interv in COST_TEMPLATES:
        r = cost_rollup(interv)
        rows.append({"intervention": interv, "total_usd": r["total"],
                     "tangible_usd": r["tangible"], "intangible_usd": r["intangible"]})
    bench_df = pd.DataFrame(rows)
    for c in ("total_usd", "tangible_usd", "intangible_usd"):
        bench_df[c] = bench_df[c].apply(lambda v: f"${v:,.0f}")
    bench_df.columns = ["Intervention", "Total", "Tangible (capex)", "Intangible (IDC)"]
    st.dataframe(bench_df, width="stretch", hide_index=True)
    st.caption("Tangible = capitalized equipment (depreciated); Intangible = IDC "
               "(rig, services, labor, chemicals — currently expensed).")


# =====================================================================
# PAGE: per-AFE drill-down
# =====================================================================

def render_afe(afe_id: str) -> None:
    token = st.session_state.get("_afe_cache_token", 0)
    df = _pipeline_df(str(DB_PATH), token)
    match = df[df["afe_number"] == afe_id]
    if match.empty:
        theme.header(afe_id, subtitle="AFE not found in the tracker.",
                     chips=[(f"v{__version__}", "ver")])
        _back_to_overview()
        st.warning("This AFE is no longer in the pipeline.")
        return
    row = match.iloc[0]
    intervention = row["intervention"]
    well_id = row["well_id"]
    total = float(row["total_cost_usd"])

    meta = _registry_meta(well_id)
    subtitle = f"{well_id} · {intervention.replace('_', ' ')}"
    if meta is not None:
        subtitle += f" · {meta.lift} · {meta.basin} · {meta.formation}"
    theme.header(
        f"{afe_id}", subtitle=subtitle,
        chips=[(f"v{__version__}", "ver"), (row["status"].replace("_", " "), "info"),
               (row["required_approver"], "warn")],
    )
    _back_to_overview()

    # ---- header metrics -----------------------------------------------------
    rollup = cost_rollup(intervention) if intervention in COST_TEMPLATES else None
    h = st.columns(5)
    h[0].metric("Gross cost", f"${total:,.0f}")
    h[1].metric("Status", row["status"].replace("_", " "))
    h[2].metric("Days in status", int(row["days_in_status"]))
    h[3].metric("Bottleneck risk", row["bottleneck_risk"])
    h[4].metric("Rig", row["rig_name"] or "—")

    if intervention not in COST_TEMPLATES:
        st.info(f"Intervention `{intervention}` has no cost template — showing tracker "
                "metadata, routing, and audit trail only.")
        _afe_routing(row, total)
        _afe_audit(afe_id)
        _afe_variance(afe_id)
        _back_to_overview()
        return

    # ---- cost waterfall -----------------------------------------------------
    st.subheader("Cost Breakdown")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("AFE total (gross)", f"${rollup['total']:,.0f}")
    cc2.metric("Tangible (capitalized)", f"${rollup['tangible']:,.0f}")
    cc3.metric("Intangible (IDC)", f"${rollup['intangible']:,.0f}")
    st.plotly_chart(_cost_waterfall(intervention, rollup["total"]), width="stretch")
    theme.source_note(
        "Benchmark cost template for this intervention; bars in USD, building direct "
        "line items → contingency → total AFE.")

    with st.expander("Line-Item Detail"):
        li = lookup_cost_template(intervention)
        li_df = pd.DataFrame([
            {"Category": x.category, "Description": x.description, "Qty": x.qty,
             "Unit": x.unit, "Unit cost $": x.unit_cost_usd, "Total $": x.total_usd,
             "Vendor": x.vendor or "TBD", "Class": x.cost_class}
            for x in li])
        st.dataframe(li_df, width="stretch", hide_index=True,
                     column_config={
                         "Unit cost $": st.column_config.NumberColumn(format="$%,.0f"),
                         "Total $": st.column_config.NumberColumn(format="$%,.0f"),
                     })

    # ---- net economics + tornado -------------------------------------------
    _afe_economics(intervention, total)

    # ---- risk register ------------------------------------------------------
    st.subheader("Risk Register")
    risks = lookup_risks(intervention)
    if risks:
        risk_df = pd.DataFrame([
            {"Category": r.category, "Risk": r.description, "Likelihood": r.likelihood,
             "Consequence": r.consequence, "Mitigation": r.mitigation}
            for r in risks])
        st.dataframe(risk_df, width="stretch", hide_index=True)
    else:
        st.caption("No standard risk register for this intervention.")

    # ---- authority routing --------------------------------------------------
    _afe_routing(row, total)

    # ---- audit trail --------------------------------------------------------
    _afe_audit(afe_id)

    # ---- variance for this AFE ---------------------------------------------
    _afe_variance(afe_id)

    _back_to_overview()


def _afe_economics(intervention: str, total: float) -> None:
    st.subheader("Net Economics")
    if not _MC_AVAILABLE:
        st.info("Economics module unavailable in this build.")
        return
    if intervention == "p_and_a":
        st.caption("P&A / cost-only job — no production uplift, so NPV / payout do not apply. "
                   "Justified against remaining liability, plugging-bond release, and avoided "
                   "idle-well carrying cost.")
        return
    from src.economics import compute_economics as _ce
    # Nominal +100 BOPD at base assumptions — deterministic, for the drill-down view.
    rate = 100.0
    net = _ce(total, rate)
    e1, e2, e3 = st.columns(3)
    e1.metric("Gross NPV @ 10%", f"${net.npv_10pct_usd/1e6:,.2f}M")
    e2.metric("Net NPV to operator", f"${net.net_npv_10pct_usd/1e6:,.2f}M")
    e3.metric("Payout", f"{net.payout_months:.0f} mo"
              if net.payout_months != float('inf') else "—")
    st.caption(f"Illustrative at +{rate:,.0f} BOPD uplift, base price/decline deck "
               "(this AFE's actual uplift lives in the source diagnosis).")
    mc = simulate_economics(treatment_cost_usd=total, incremental_rate_bopd=rate)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("P10 NPV", f"${mc.npv_p10_usd/1e6:,.2f}M")
    t2.metric("P50 NPV", f"${mc.npv_p50_usd/1e6:,.2f}M")
    t3.metric("P90 NPV", f"${mc.npv_p90_usd/1e6:,.2f}M")
    t4.metric("P(payout < 24 mo)", f"{mc.probability_of_payout*100:.0f}%")
    st.plotly_chart(_tornado_fig(mc), width="stretch")
    theme.source_note(
        "NPV @ 10% (USD) swing as each variable moves over its sampled range; dashed "
        "line is the base-case NPV.")
    theme.references(["npv"])


def _afe_routing(row, total: float) -> None:
    st.subheader("Authority Routing")
    approver = row["required_approver"]
    st.markdown(
        f"This **${total:,.0f}** AFE requires sign-off at the **{approver}** authority level.")
    st.caption("Delegation-of-authority limits: Production Engineer < $50k · "
               "Engineering Manager < $250k · Operations Manager < $1MM · VP / Asset Manager above.")
    # Visual ladder of the approval chain up to the required level.
    chain = ["Production Engineer", "Engineering Manager", "Operations Manager", "VP / Asset Manager"]
    try:
        idx = chain.index(approver)
    except ValueError:
        idx = len(chain) - 1
    for i, role in enumerate(chain):
        theme.flag(role + (" ✔ required" if i == idx else ""),
                   "ok" if i <= idx else "warn")


def _afe_audit(afe_id: str) -> None:
    st.subheader("Status / Audit Trail")
    token = st.session_state.get("_afe_cache_token", 0)
    ev = _events_df(str(DB_PATH), afe_id, token)
    if ev.empty:
        st.caption("No status-change events recorded for this AFE.")
        return
    st.caption("Immutable status-change log (newest first) — every transition is appended, "
               "never overwritten (what an internal-audit / SOX reviewer expects).")
    st.dataframe(ev[["ts", "from_status", "to_status", "actor", "note"]],
                 width="stretch", hide_index=True)


def _afe_variance(afe_id: str) -> None:
    st.subheader("Actual-vs-AFE Variance")
    m = _variance_for(afe_id)
    if m is None:
        st.caption("No closed-out actuals recorded for this AFE yet.")
        return
    budget = float(m["line_total_usd"].sum())
    actual = float(m["actual_usd"].sum())
    var_pct = ((actual - budget) / budget * 100.0) if budget else 0.0
    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("AFE budget", f"${budget:,.0f}")
    vc2.metric("Actual", f"${actual:,.0f}")
    vc3.metric("Variance", f"{var_pct:+.1f}%",
               delta=f"${actual - budget:,.0f}", delta_color="inverse")
    if var_pct > SUPPLEMENT_THRESHOLD_PCT:
        st.error(f"⚠️ Supplemental AFE required — actuals exceed the AFE by >{SUPPLEMENT_THRESHOLD_PCT:.0f}%.")
    disp = m.copy()
    for c in ("line_total_usd", "actual_usd", "variance_usd"):
        disp[c] = disp[c].apply(lambda v: f"${v:,.0f}")
    disp = disp[["category", "line_total_usd", "actual_usd", "variance_usd"]]
    disp.columns = ["Category", "AFE budget", "Actual", "Variance"]
    st.dataframe(disp, width="stretch", hide_index=True)


# =====================================================================
# Shared setup (runs every rerun) + navigation
# =====================================================================

theme.setup_page("AFE Copilot", icon="📝")
theme.suite_nav("afe")

# Seed demo AFEs on first run so the per-AFE pages exist (reuses existing seed logic).
if not DB_PATH.exists():
    seed_demo_data(DB_PATH)
    _bump_cache_token()

_token = st.session_state.get("_afe_cache_token", 0)
_df = _pipeline_df(str(DB_PATH), _token)

overview = st.Page(render_overview, title="Overview", icon="📋", default=True)
afe_pages = [
    st.Page(partial(render_afe, afe_no), title=afe_no, url_path=afe_no)
    for afe_no in (sorted(_df["afe_number"]) if not _df.empty else [])
]
nav_spec = {"Overview": [overview]}
if afe_pages:
    nav_spec["AFEs"] = afe_pages
st.navigation(nav_spec).run()
