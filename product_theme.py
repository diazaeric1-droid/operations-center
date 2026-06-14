"""product_theme — enterprise presentation layer for the three consolidated
operator products (Operations Center · Engineering Workbench · Capital Desk).

Sits ON TOP of the suite's vendored ``theme.py`` (same directory): theme.py keeps
the brand tokens, Plotly styling, citations, and provenance helpers; this module
adds the product chrome — masthead, context bar, KPI rows, status pills, the
cross-product switcher, and a denser "enterprise console" CSS layer. Views import
ONLY this module:

    import product_theme as pt
    pt.masthead("ops", "Triage Board", "Fleet ranked by risked-NPV opportunity")
    pt.context_bar([("Asset", "Permian synthetic fleet"), ("Deck", "$70 · 80% NRI")])
    pt.kpi_row([{"label": "Open Alerts", "value": "7", "delta": "+2 vs yesterday"}])

Pure presentation: no side effects beyond Streamlit calls. Vendored byte-identical
into each product repo (the same pattern as theme.py / fleet_registry.py).
"""
from __future__ import annotations

from html import escape

import streamlit as st

import theme
from theme import (  # re-exported so views need a single import
    style_fig, data_badge, references, source_note, how_to, flag,
    NAVY, BLUE, RED, GREEN, AMBER, PURPLE, TEAL, GREY, COLORWAY, CITATIONS,
)

PRODUCT_VERSION = "0.3.0"

# The three consolidated operator products. Each entry:
#   (key, display name, tagline, live url)
PRODUCTS = [
    ("ops", "Operations Center",
     "Surveillance · loss accounting · triage",
     "https://operations-center.streamlit.app"),
    ("workbench", "Engineering Workbench",
     "Design · diagnose · predict · optimize",
     "https://engineering-workbench.streamlit.app"),
    ("capital", "Capital Desk",
     "Authorize · program · screen",
     "https://capital-desk.streamlit.app"),
]

COMPONENTS_URL = "https://github.com/diazaeric1-droid"

ENTERPRISE_CSS = """
<style>
    /* denser console layout on top of theme.CSS */
    .block-container,
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"] {padding-top: 4.2rem; max-width: 1500px;}
    [data-testid="stMetric"] {padding: 0.45rem 0.7rem; border-radius: 8px;}
    [data-testid="stMetricValue"] {font-size: 1.12rem;}
    [data-testid="stMetricLabel"] {font-size: 0.68rem; letter-spacing: 0.02em;
                                   text-transform: uppercase;}
    [data-testid="stDataFrame"] {font-size: 0.82rem;}
    h3 {font-size: 1.02rem !important;}
    h4 {font-size: 0.95rem !important;}

    /* masthead */
    .pt-eyebrow {font-size: 0.66rem; font-weight: 800; letter-spacing: 0.1em;
                 text-transform: uppercase; color: #4F81BD; margin-bottom: 0.1rem;}
    .pt-mast {display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
              border-bottom: 1px solid #e5e7eb; padding-bottom: 0.55rem;
              margin-bottom: 0.75rem;}
    .pt-title {font-size: 1.42rem; font-weight: 750; color: #1f2937; line-height: 1.15;}
    .pt-desc {font-size: 0.82rem; color: #6b7280; margin-top: 0.1rem;}
    .pt-chips {margin-left: auto; display: flex; gap: 0.4rem; flex-wrap: wrap;}

    /* global context bar */
    .pt-ctx {display: flex; gap: 1.4rem; flex-wrap: wrap; background: #f6f8fa;
             border: 1px solid #e5e7eb; border-radius: 8px; padding: 0.42rem 0.9rem;
             margin: 0 0 0.9rem 0; font-size: 0.78rem; color: #475467;}
    .pt-ctx b {color: #1f2937; font-weight: 650;}
    .pt-ctx .ctx-l {color: #98a2b3; font-weight: 600; text-transform: uppercase;
                    font-size: 0.64rem; letter-spacing: 0.05em; margin-right: 0.25rem;}

    /* status pills (inline, e.g. inside markdown) */
    .pt-pill {display: inline-block; padding: 0.14rem 0.55rem; border-radius: 999px;
              font-size: 0.7rem; font-weight: 700;}
    .pt-pill.ok    {background:#e7f6ec; color:#1b7a3d; border:1px solid #b7e0c4;}
    .pt-pill.warn  {background:#fdf3e2; color:#9a6a16; border:1px solid #f0d9a8;}
    .pt-pill.bad   {background:#fdeaea; color:#b42318; border:1px solid #f4c7c2;}
    .pt-pill.info  {background:#e8f0fb; color:#1c4f8a; border:1px solid #c7dcf5;}
    .pt-pill.muted {background:#f2f4f7; color:#475467; border:1px solid #e5e7eb;}

    /* in-page section headers */
    .pt-sec {font-size: 0.98rem; font-weight: 700; color: #1f2937;
             margin: 0.9rem 0 0.1rem 0;}
    .pt-sec-d {font-size: 0.78rem; color: #6b7280; margin-bottom: 0.5rem;}

    /* product switcher (sidebar) */
    .pt-sw {border-bottom: 1px solid #e5e7eb; padding-bottom: 0.7rem;
            margin-bottom: 0.8rem;}
    .pt-sw-t {font-size: 0.66rem; font-weight: 800; letter-spacing: 0.1em;
              text-transform: uppercase; color: #98a2b3; margin-bottom: 0.4rem;}
    .pt-sw a {display: block; font-size: 0.84rem; font-weight: 600; color: #1F3A5F;
              text-decoration: none; padding: 0.16rem 0;}
    .pt-sw a:hover {color: #4F81BD;}
    .pt-sw .cur {display: block; font-size: 0.84rem; font-weight: 750; color: #4F81BD;
                 padding: 0.16rem 0 0.16rem 0.45rem; border-left: 2px solid #4F81BD;
                 margin-left: -0.45rem;}
    .pt-sw .d {font-size: 0.68rem; color: #98a2b3; margin: -0.1rem 0 0.25rem 0;}
    .pt-sw .all {font-size: 0.74rem; color: #6b7280;}
    .pt-sw .all a {display: inline; font-size: 0.74rem; font-weight: 600;}

    /* st.navigation section labels */
    [data-testid="stSidebarNavSectionHeader"],
    [data-testid="stNavSectionHeader"] {font-size: 0.66rem; letter-spacing: 0.09em;
        text-transform: uppercase; color: #98a2b3; font-weight: 800;}

    /* empty / unavailable-lens state */
    .pt-empty {background:#f9fafb; border:1px dashed #d0d5dd; border-radius:10px;
               padding:1.1rem 1.2rem; color:#667085; font-size:0.85rem; margin:0.6rem 0;}
    .pt-empty b {color:#475467;}
</style>
"""


def _product(key: str):
    for p in PRODUCTS:
        if p[0] == key:
            return p
    raise KeyError(f"unknown product key: {key!r} (known: {[p[0] for p in PRODUCTS]})")


def setup_product(product_key: str, icon: str = "🛢️") -> None:
    """``st.set_page_config`` + inject theme.CSS and the enterprise layer.

    Call ONCE, first, from app.py — views must never call this (nor
    ``theme.setup_page``)."""
    _key, name, tagline, _url = _product(product_key)
    st.set_page_config(page_title=name, page_icon=icon, layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(theme.CSS, unsafe_allow_html=True)
    st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)


def masthead(product_key: str, module_title: str, module_desc: str = "",
             chips=None) -> None:
    """Standard page top: product eyebrow, module title, right-aligned chips.

    chips: list of (text, kind) with kind ∈ {ver, eval, info, warn};
    defaults to the product version chip."""
    _key, name, _tagline, _url = _product(product_key)
    if chips is None:
        chips = [(f"v{PRODUCT_VERSION}", "ver")]
    chips_html = ""
    if chips:
        chips_html = ('<div class="pt-chips">'
                      + "".join(theme._chip_html(t, k) for t, k in chips)
                      + "</div>")
    desc = f'<div class="pt-desc">{escape(module_desc)}</div>' if module_desc else ""
    st.markdown(
        f'<div class="pt-eyebrow">{escape(name)}</div>'
        f'<div class="pt-mast"><div>'
        f'<div class="pt-title">{escape(module_title)}</div>{desc}'
        f'</div>{chips_html}</div>',
        unsafe_allow_html=True,
    )


def context_bar(pairs) -> None:
    """Persistent global-context strip: list of (label, value) pairs.

    Render right under the masthead on every page so the user always sees the
    asset / price deck / data source the page is computed against."""
    cells = "".join(
        f'<span><span class="ctx-l">{escape(str(label))}</span>'
        f'<b>{escape(str(value))}</b></span>'
        for label, value in pairs
    )
    st.markdown(f'<div class="pt-ctx">{cells}</div>', unsafe_allow_html=True)


def kpi_row(items) -> None:
    """One row of compact KPI metrics. items: list of dicts with keys
    label, value, and optional delta, delta_color ('normal'|'inverse'|'off'), help."""
    if not items:
        return
    cols = st.columns(len(items))
    for col, m in zip(cols, items):
        col.metric(
            m["label"], m["value"],
            delta=m.get("delta"),
            delta_color=m.get("delta_color", "normal"),
            help=m.get("help"),
        )


def pill(text: str, kind: str = "ok") -> str:
    """Return inline pill HTML (kind ∈ ok|warn|bad|info|muted) for embedding
    in st.markdown(..., unsafe_allow_html=True) content."""
    if kind not in ("ok", "warn", "bad", "info", "muted"):
        kind = "muted"
    return f'<span class="pt-pill {kind}">{escape(str(text))}</span>'


def section(title: str, desc: str = "") -> None:
    """Consistent in-page section header with optional one-line description."""
    st.markdown(f'<div class="pt-sec">{escape(title)}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="pt-sec-d">{escape(desc)}</div>',
                    unsafe_allow_html=True)


def empty_state(message: str, hint: str = "") -> None:
    """Quiet placeholder for an unavailable lens / empty dataset (no fake numbers)."""
    h = f"<br><b>{escape(hint)}</b>" if hint else ""
    st.markdown(f'<div class="pt-empty">{escape(message)}{h}</div>',
                unsafe_allow_html=True)


def product_switcher(current_key: str) -> None:
    """Sidebar cross-product navigation (the M365-style switcher for the trio).

    Call once from app.py's sidebar block, after the product-specific controls."""
    rows = []
    for key, name, tagline, url in PRODUCTS:
        if key == current_key:
            rows.append(f'<span class="cur">● {escape(name)}</span>'
                        f'<div class="d">{escape(tagline)}</div>')
        else:
            rows.append(f'<a href="{escape(url)}" target="_blank" rel="noopener">'
                        f'{escape(name)}</a>'
                        f'<div class="d">{escape(tagline)}</div>')
    rows.append(f'<span class="all">Component apps: '
                f'<a href="{escape(COMPONENTS_URL)}" target="_blank" '
                f'rel="noopener">github/diazaeric1-droid</a></span>')
    st.sidebar.markdown(
        '<div class="pt-sw"><div class="pt-sw-t">Operator Products</div>'
        + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )
