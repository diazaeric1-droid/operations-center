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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# --- warm-container module self-heal (ALL product-owned modules) --------------
# Streamlit Cloud reuses the Python process across redeploys; a cached OLD copy of
# one of OUR modules in sys.modules (or a stale .pyc) can lack symbols added in a
# newer commit -> AttributeError at run (e.g. a views helper missing a new function).
# Drop our bytecode + evict every product-owned module so the imports below + the
# view pages reload from the CURRENT commit's source. Gated ONCE per session
# (Streamlit re-runs this whole script on every interaction). Skipped under pytest,
# where modules are already fresh and evicting would break module-identity invariants.
if "pytest" not in sys.modules and not st.session_state.get("_ops_healed"):
    import shutil as _sh_heal
    for _pyc in HERE.rglob("__pycache__"):
        _sh_heal.rmtree(_pyc, ignore_errors=True)
    _OWN = ("core", "product_theme", "theme", "fleet_registry",
            "digest", "deferment", "esp", "afe", "views", "src")
    for _m in list(sys.modules):
        if any(_m == p or _m.startswith(p + ".") for p in _OWN):
            sys.modules.pop(_m, None)
    # Re-pin the repo root to sys.path[0] BEFORE the re-imports below. The vendored
    # digest data_loader lazily inserts its demo/ dir (which carries an OLD
    # fleet_registry copy) at sys.path[0] during a prior session's render, so after
    # this eviction a bare `import fleet_registry` would otherwise resolve to that
    # stale demo copy (live AttributeError: WellMeta has no `ctb`). The top-of-file
    # insert can't help — it is guarded by `not in sys.path` and never re-pins.
    if str(HERE) in sys.path:
        sys.path.remove(str(HERE))
    sys.path.insert(0, str(HERE))
    st.session_state["_ops_healed"] = True

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
                and not core.digest_fleet_stale()      # 50→100 self-heal must fire
                and any(core.DEFERMENT_WELLS.glob("well_*.csv")))
    except Exception:  # noqa: BLE001
        return False


if not _artifacts_ready():
    with st.status("First-time setup — generating the synthetic fleet and training "
                   "the ESP risk model (~30s, one time). Expand for the log.",
                   expanded=False) as _status:
        try:
            core.bootstrap(log=_status.write)
            # The fleet may have just been regenerated (e.g. a warm container that
            # still held the old 50-well fleet). Drop cached well-id lists / fleet
            # frames / rankings so every page serves the fresh fleet, not the cache.
            st.cache_data.clear()
            st.cache_resource.clear()
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
    st.subheader("Well File")
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
              help="Realized price — drives deferred-$, the board ranking, and AFE "
                   "economics.")
    st.slider("Deck NRI (chain economics)", min_value=0.0, max_value=1.0,
              step=0.01, key="nri",
              help="Share of revenue after royalty; nets the revenue side of the "
                   "board ranking + AFE economics (one auditable number for capital "
                   "decisions). Per-WELL NRI for the roll-up pages' NET views is "
                   "edited on Sources & BYOD.")
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
