"""Operations Center — the morning-triage console for a production operation.

Surveillance → loss accounting → fleet triage → action chain, in one process:
the vendored component apps (daily-production-digest, deferment-iq,
esp-failure-risk-agent, afe-copilot) are loaded under import aliases by core.py
(the pattern proven in pe-pipeline, which this product absorbs), and a new
enterprise presentation layer (product_theme) renders the pages.

Run: streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# --- warm-container module self-heal (vendored top-level modules) -------------
# Streamlit Cloud reuses the container across redeploys; a cached OLD theme /
# fleet_registry / product_theme in sys.modules (or stale .pyc) can lack symbols
# added in a newer commit. Drop bytecode + evict so imports reload from source.
import shutil as _sh_heal

HERE = Path(__file__).resolve().parent
_sh_heal.rmtree(HERE / "__pycache__", ignore_errors=True)
for _stale in ("theme", "fleet_registry", "product_theme"):
    sys.modules.pop(_stale, None)
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import product_theme as pt  # noqa: E402

pt.setup_product("ops")  # st.set_page_config + theme CSS — FIRST, once, only here

# Guarded import: the component apps are vendored under apps/; fail readable.
try:
    import core
    from views import PAGES, PAGE_OBJECTS
    from views import _common as c
except Exception as e:  # noqa: BLE001
    st.title("Operations Center")
    st.error("Couldn't load the bundled component apps.\n\n"
             f"```\n{type(e).__name__}: {e}\n```\n\n"
             "The four components are vendored under `apps/` in this repo. Run "
             "the app from the repo root, or set `OPS_APPS_ROOT` to where the "
             "apps live.")
    st.stop()


# --- first-run bootstrap: synthetic data + trained ESP model (cached) ----------
@st.cache_resource(show_spinner=False)
def _bootstrap() -> bool:
    with st.status("First-time setup — generating fleets + training the ESP "
                   "risk model (~30s, one time)…", expanded=True) as status:
        core.bootstrap(log=status.write)
        status.update(label="Setup complete.", state="complete", expanded=False)
    return True


_bootstrap()

# --- global session-state contract (seed BEFORE widgets + navigation) ----------
for _k, _v in c.STATE_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

_well_ids = c.scada_well_ids()
if st.session_state["well_id"] not in _well_ids:
    st.session_state["well_id"] = _well_ids[0] if _well_ids else None

# --- global sidebar -------------------------------------------------------------
with st.sidebar:
    st.caption("**Console data** — Today + Well File: synthetic daily SCADA "
               f"({len(_well_ids)} wells, regenerated at bootstrap). "
               "Loss Accounting: real Colorado ECMC monthly records by default. "
               "Two datasets, two cadences — never joined. Details: Sources & BYOD.")
    if _well_ids:
        st.selectbox("Selected well (Well File pages)", _well_ids, key="well_id",
                     help="Drives Well 360 and Action Chain. Pick from the "
                          "surveillance fleet; the digest, ESP agent, and AFE "
                          "chain all key on this id.")
    st.subheader("Price deck")
    st.slider("Oil price ($/bbl)", min_value=20.0, max_value=150.0, step=1.0,
              key="oil_price",
              help="Realized price — drives deferred-$, triage ranking, and AFE "
                   "economics.")
    st.slider("Net revenue interest (NRI)", min_value=0.0, max_value=1.0,
              step=0.01, key="nri",
              help="Share of revenue after royalty; nets the revenue side of "
                   "triage + AFE economics.")
    st.slider("Discount rate", min_value=0.0, max_value=0.30, step=0.01,
              key="discount",
              help="Deck context. Chain economics use the AFE component's "
                   "certified PV10 kernel; a non-10% deck discount is flagged on "
                   "NPV pages rather than silently re-rating component math.")
    st.text_input("Anthropic API key (optional)", type="password",
                  key="anthropic_key",
                  help="Session-only, never stored. Powers narrated briefs; every "
                       "number works without it.")
    pt.product_switcher("ops")

# --- navigation ------------------------------------------------------------------
_nav: dict[str, list] = {}
for _section, _specs in PAGES.items():
    _pages = []
    for _title, _icon, _slug, _render, _default in _specs:
        _page = st.Page(_render, title=_title, icon=_icon, url_path=_slug,
                        default=_default)
        _pages.append(_page)
        PAGE_OBJECTS[_title] = _page
    _nav[_section] = _pages

st.navigation(_nav).run()
