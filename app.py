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


# --- first-run bootstrap: synthetic data + trained ESP model -------------------
# The data + ESP model are .gitignore'd, so a cold container regenerates them
# (~30s, one time). Show a COMPACT, collapsible status bar only when there is real
# first-run work to do; on a warm container the artifacts already exist and this is
# silent. Live log streams into the drill-down, and the bar always reaches a
# terminal complete/error state — so the spinner can never sit forever.
def _artifacts_ready() -> bool:
    try:
        return (core.ESP_MODEL.exists()
                and any(core.DIGEST_FLEET.glob("well_*.csv"))
                and any(core.DEFERMENT_WELLS.glob("well_*.csv")))
    except Exception:  # noqa: BLE001
        return False


if not _artifacts_ready():
    with st.status("First-time setup — generating the synthetic fleet and training "
                   "the ESP risk model (~30s, one time). Expand for the log.",
                   expanded=False) as _status:
        try:
            core.bootstrap(log=_status.write)
            _status.update(
                label="Setup complete — fleet generated, ESP model trained.",
                state="complete")
        except Exception as _e:  # noqa: BLE001
            _status.update(label="First-time setup failed — see details.",
                           state="error")
            st.error(
                "Couldn't complete first-time setup (synthetic data generation / "
                f"ESP model training).\n\n```\n{type(_e).__name__}: {_e}\n```\n\n"
                "Reload to retry; if it persists, check the app logs.")
            st.stop()

# --- global session-state contract (seed BEFORE widgets + navigation) ----------
for _k, _v in c.STATE_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

_well_ids = c.scada_well_ids()
if st.session_state["well_id"] not in _well_ids:
    st.session_state["well_id"] = _well_ids[0] if _well_ids else None

# --- global sidebar -------------------------------------------------------------
# Lead with the Operator Products switcher (the portfolio's three products), then
# the controls that drive this console. Data provenance lives on Sources & BYOD.
pt.product_switcher("ops")
with st.sidebar:
    st.subheader("Well file")
    if _well_ids:
        st.selectbox("Selected well (Well File pages)", _well_ids, key="well_id",
                     help="Drives Well 360 and Action Chain. Pick from the "
                          "surveillance fleet; the digest, ESP agent, and AFE "
                          "chain all key on this id.")
        st.caption(f"Synthetic Permian demo fleet · {len(_well_ids)} wells · "
                   "full provenance on **Sources & BYOD**.")
    st.subheader("Price deck")
    st.slider("Oil price ($/bbl)", min_value=20.0, max_value=150.0, step=1.0,
              key="oil_price",
              help="Realized price — drives deferred-$, triage ranking, and AFE "
                   "economics.")
    st.slider("Net revenue interest (NRI)", min_value=0.0, max_value=1.0,
              step=0.01, key="nri",
              help="Share of revenue after royalty; nets the revenue side of "
                   "triage + AFE economics.")
    st.caption("Discounting is fixed at **PV10** (10%) — the AFE component's "
               "certified economics kernel; every NPV on the console uses it.")
    st.text_input("Anthropic API key (optional)", type="password",
                  key="anthropic_key",
                  help="Session-only, never stored. Powers narrated briefs; every "
                       "number works without it.")

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
