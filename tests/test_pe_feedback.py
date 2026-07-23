"""Coverage for PE field-feedback round 1 (v0.8.0):

* registry additive fields (surface coordinates, CTB, per-well NRI) —
  deterministic, bounded, varied, and truly additive (existing values untouched);
* the per-well NRI override / GROSS-NET helpers;
* the unified morning-brief list builder (ordering + NRI-aware ranking);
* downtime context (ongoing_event_days excludes the demo injection);
* the shared fleet filters (pure logic);
* rename completeness: no user-visible "Triage Board" string survives.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import fleet_registry as fr

ROOT = Path(__file__).resolve().parent.parent
PRICE = 70.0


# ---- registry: additive, deterministic, bounded --------------------------------

def test_registry_existing_values_untouched():
    """The extension must be purely additive — curated + derived values unchanged."""
    m = fr.get("well_007")
    assert (m.name, m.lift, m.api14) == ("Garza 7H", "Rod pump", "42-227-30007")
    assert m.lateral_length_ft == 7800
    d = fr.get("well_002")
    assert d.basin == "Midland" and d.well_id == "well_002"
    # META_COLUMNS is a strict superset of the original tuple, in order.
    assert fr.META_COLUMNS[:7] == ("basin", "area", "formation", "lift",
                                   "lateral_length_ft", "peer_group", "hero")
    assert "ctb" in fr.META_COLUMNS and "nri" in fr.META_COLUMNS


def test_surface_latlon_deterministic_and_within_county():
    ids = [f"well_{n:03d}" for n in range(1, 101)]
    for wid in ids:
        a = fr.surface_latlon(wid)
        b = fr.surface_latlon(wid)
        assert a == b, "surface_latlon must be stable across calls"
    # Every well sits within its county centroid ± the documented jitter.
    for wid in ids:
        county = fr.get(wid).area.split(" Co.")[0].strip()
        lat0, lon0 = fr._COUNTY_LATLON[county]
        lat, lon = fr.surface_latlon(wid)
        assert abs(lat - lat0) <= 0.13 + 1e-9
        assert abs(lon - lon0) <= 0.16 + 1e-9
    # WellMeta properties delegate to the same function.
    m = fr.get("well_013")
    assert (m.lat, m.lon) == fr.surface_latlon("well_013")


def test_ctb_deterministic_and_clustered_by_county():
    ids = [f"well_{n:03d}" for n in range(1, 101)]
    ctbs = {wid: fr.ctb_for(wid) for wid in ids}
    assert ctbs == {wid: fr.ctb_for(wid) for wid in ids}  # stable
    for wid, ctb in ctbs.items():
        county = fr.get(wid).area.split(" Co.")[0].strip()
        assert ctb.startswith(county + " CTB-"), \
            f"{wid}: CTB {ctb!r} not clustered under county {county!r}"
    # A CTB is a real grouping dimension: far fewer batteries than wells.
    n_batteries = len(set(ctbs.values()))
    assert 4 <= n_batteries <= 24
    assert fr.get("well_022").ctb == fr.ctb_for("well_022")


def test_nri_varied_bounded_and_deterministic():
    ids = [f"well_{n:03d}" for n in range(1, 101)]
    vals = [fr.nri_for(w) for w in ids]
    assert vals == [fr.nri_for(w) for w in ids]           # stable
    assert all(0.65 <= v <= 0.90 for v in vals)            # plausible NRI band
    assert len(set(vals)) >= 20, "per-well NRI must be varied, not one flat number"
    assert fr.get("well_041").nri == fr.nri_for("well_041")


def test_enrich_joins_ctb_and_nri():
    df = pd.DataFrame({"well_id": ["well_001", "well_013"]})
    out = fr.enrich(df)
    assert "ctb" in out.columns and "nri" in out.columns
    assert out.loc[0, "ctb"] == fr.ctb_for("well_001")
    assert out.loc[1, "nri"] == fr.nri_for("well_013")


# ---- per-well NRI helpers ------------------------------------------------------

def test_well_nri_override_wins_over_registry_default():
    from views import _common as c

    assert c.well_nri("well_001", overrides={}) == fr.nri_for("well_001")
    assert c.well_nri("well_001", overrides={"well_001": 0.5}) == 0.5
    m = c.nri_map(["well_001", "well_002"], overrides={"well_002": 0.6})
    assert m["well_001"] == fr.nri_for("well_001")
    assert m["well_002"] == 0.6


# ---- fleet filters (pure logic) ------------------------------------------------

def test_filter_ids_by_meta_dimensions_and_empty_means_all():
    from views import _common as c

    ids = [f"well_{n:03d}" for n in range(1, 101)]
    assert c.filter_ids_by_meta(ids) == ids                # no filter = identity
    esp = c.filter_ids_by_meta(ids, lift={"ESP"})
    assert esp and all(fr.get(w).lift == "ESP" for w in esp)
    one_ctb = fr.ctb_for("well_010")
    grp = c.filter_ids_by_meta(ids, ctb={one_ctb})
    assert grp and all(fr.ctb_for(w) == one_ctb for w in grp)
    both = c.filter_ids_by_meta(ids, basin={"Midland"}, lift={"Gas lift"})
    assert all(fr.get(w).basin == "Midland" and fr.get(w).lift == "Gas lift"
               for w in both)


# ---- unified morning-brief list ------------------------------------------------

def _event(wid, state, bopd, days, cum=100.0, kind="rate_loss", ack=False):
    return SimpleNamespace(well_id=wid, state=state, event_type=kind,
                           duration_days=days, last_deferred_bopd=bopd,
                           deferred_bopd=cum, acknowledged=ack)


def _anom(wid, bopd, cat="rate_drop", ack=False):
    return SimpleNamespace(well_id=wid, category=cat, deferred_bopd=bopd,
                           acknowledged=ack)


def test_unified_brief_frame_orders_by_impact_and_badges_status():
    from views import _common as c

    events = [
        _event("w_ong", "ONGOING", 50.0, 6),
        _event("w_new", "NEW", 80.0, 1),
        _event("w_res", "RESOLVED", 40.0, 9),   # resolved → 0 today, bottom
        _event("w_ack", "ONGOING", 99.0, 3, ack=True),  # acknowledged → excluded
    ]
    anoms = [_anom("w_scan", 20.0), _anom("w_ong", 15.0)]  # w_ong has an open event
    nri = {"w_ong": 0.8, "w_new": 0.8, "w_res": 0.8, "w_scan": 0.8}
    df = c.unified_brief_frame(events, anoms, nri, net_view=False, price=PRICE)
    assert list(df["well_id"]) == ["w_new", "w_ong", "w_scan", "w_res"]
    assert "w_ack" not in set(df["well_id"])
    # The open event's number wins over the same well's scan anomaly (one row).
    assert (df["well_id"] == "w_ong").sum() == 1
    assert df.loc[df["well_id"] == "w_ong", "gross_bopd"].iloc[0] == 50.0
    assert df.loc[df["well_id"] == "w_res", "rank_bopd"].iloc[0] == 0.0
    assert set(df["status"]) == {"NEW", "ONGOING", "RESOLVED"}


def test_unified_brief_frame_net_ranking_respects_per_well_nri():
    from views import _common as c

    events = [_event("w_a", "ONGOING", 100.0, 2), _event("w_b", "ONGOING", 90.0, 2)]
    # Gross: A(100) > B(90). Net: A×0.30=30 < B×0.85=76.5 → order flips.
    nri = {"w_a": 0.30, "w_b": 0.85}
    gross = c.unified_brief_frame(events, [], nri, net_view=False, price=PRICE)
    net = c.unified_brief_frame(events, [], nri, net_view=True, price=PRICE)
    assert list(gross["well_id"]) == ["w_a", "w_b"]
    assert list(net["well_id"]) == ["w_b", "w_a"]
    assert net.loc[net["well_id"] == "w_b", "usd_per_day"].iloc[0] == \
        round(90.0 * 0.85 * PRICE, 0)


def test_unified_brief_frame_empty_inputs():
    from views import _common as c

    df = c.unified_brief_frame([], [], {}, net_view=True, price=PRICE)
    assert df.empty and "well_id" in df.columns


# ---- downtime context ----------------------------------------------------------

def test_ongoing_event_days_excludes_demo_injection(bootstrapped):
    import core
    from views import _common as c

    days = c.ongoing_event_days(c.DISK_TOKEN, PRICE)
    assert isinstance(days, dict)
    assert all(isinstance(k, str) and int(v) >= 1 for k, v in days.items())
    # The replay runs WITHOUT the demo injection: the injected 12-day outage on the
    # demo well must never leak into recommendation gating (the raw fleet may carry
    # its own short real event for that well — that one is legitimate).
    assert days.get(core.DEMO_OUTAGE_WELL, 0) < core.DEMO_OUTAGE_LEN
    injected = {str(e.well_id): int(e.duration_days)
                for e in core.replay_events(core.load_scada_fleet(),
                                            price_per_bbl=PRICE, inject_demo=True)
                if e.state in ("NEW", "ONGOING")}
    assert injected.get(core.DEMO_OUTAGE_WELL) == core.DEMO_OUTAGE_LEN  # sanity


# ---- AppTest coverage of the new display branches ------------------------------

def _run_view(title: str, monkeypatch, extra_state: dict):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("OPS_VIEW", title)
    at = AppTest.from_file(str(ROOT / "tests" / "view_harness.py"),
                           default_timeout=300)
    at.session_state["oil_price"] = PRICE
    at.session_state["nri"] = 0.80
    at.session_state["discount"] = 0.10
    at.session_state["well_id"] = "well_001"
    at.session_state["data_source"] = "synthetic"
    at.session_state["anthropic_key"] = ""
    for k, v in extra_state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, f"{title}: {[str(e.value) for e in at.exception]}"
    return at


def test_deferment_overview_renders_dollars_and_net(bootstrapped, monkeypatch):
    """The OC8 Dollars mode + the OC9 per-well-NRI NET view render exception-free."""
    _run_view("Deferment Overview", monkeypatch,
              {"def_units": "Dollars", "net_view": True})


def test_optimization_board_renders_net_view(bootstrapped, monkeypatch):
    _run_view("Optimization Board", monkeypatch,
              {"net_view": True, "nri_overrides": {"well_001": 0.5}})


def test_morning_brief_renders_detailed_panels_view(bootstrapped, monkeypatch):
    """The classic three-panel layout survives behind the OC4 view toggle."""
    _run_view("Morning Brief", monkeypatch,
              {"brief_view": "Detailed panels", "net_view": True})


# ---- rename completeness (OC5) -------------------------------------------------

def test_no_user_visible_triage_board_strings_remain():
    """'Triage Board' must not survive anywhere a user (or the README) sees it.
    Internal identifiers (triage_tiers, module filename) are intentionally kept."""
    targets = sorted((ROOT / "views").glob("*.py")) + [ROOT / "app.py",
                                                       ROOT / "README.md",
                                                       ROOT / "product_theme.py"]
    offenders = [p.name for p in targets if "Triage Board" in p.read_text()]
    assert not offenders, f"'Triage Board' still present in: {offenders}"


def test_board_referential_triage_wording_reworded():
    """Rendered strings must not call the board or its ranking 'triage' either.
    Pins the specific strings flagged in review (internal identifiers such as
    triage_tiers / triage_scorecard / the module filename are intentionally kept)."""
    checks = [
        ("views/methods.py", "**Triage ranking:**"),
        ("views/home.py", "triage figures"),
        ("views/home.py", "triage ranking"),
        ("views/home.py", "brief → triage"),
        ("app.py", "triage ranking"),
        ("product_theme.py", "loss accounting · triage"),
        ("README.md", "morning-triage console"),
        ("README.md", "fleet triage →"),
    ]
    for rel, needle in checks:
        assert needle not in (ROOT / rel).read_text(), f"{needle!r} still in {rel}"


def test_optimization_board_is_the_nav_title_and_csv_name():
    import views

    titles = [t for specs in views.PAGES.values() for (t, *_rest) in specs]
    assert "Optimization Board" in titles and "Triage Board" not in titles
    src = (ROOT / "views" / "triage_board.py").read_text()
    assert "ops_optimization_board.csv" in src
    assert "ops_triage_board.csv" not in src


def test_heal_repins_repo_root_ahead_of_vendored_demo_dirs():
    """The warm-container self-heal must re-pin the repo root to sys.path[0] when it
    evicts modules: the vendored digest data_loader lazily inserts its demo/ dir
    (carrying an OLD fleet_registry copy) at sys.path[0], and the top-of-file insert
    is guarded by `not in sys.path` so it never re-pins. Without the re-pin, every
    session after the first Home render re-imports the STALE demo registry
    (live AttributeError: WellMeta has no `ctb`)."""
    src = (ROOT / "app.py").read_text()
    heal = src[src.index("_OWN"):src.index('_ops_healed"] = True')]
    assert "sys.path.remove(str(HERE))" in heal
    assert "sys.path.insert(0, str(HERE))" in heal


def test_root_fleet_registry_wins_when_root_is_pinned_first():
    """Resolution-order invariant behind the heal re-pin: with the digest demo dir
    at sys.path[0] and the repo root re-pinned ahead of it, a fresh import must
    resolve the ROOT registry (which has the PE-feedback fields), not the vendored
    demo copy. Restores sys.path and the original module object afterwards."""
    import importlib
    import fleet_registry as _orig

    demo = ROOT / "apps" / "daily-production-digest" / "demo"
    assert (demo / "fleet_registry.py").exists()
    saved_path = list(sys.path)
    try:
        sys.path.insert(0, str(demo))          # what data_loader does at runtime
        if str(ROOT) in sys.path:
            sys.path.remove(str(ROOT))
        sys.path.insert(0, str(ROOT))          # what the heal now guarantees
        sys.modules.pop("fleet_registry", None)
        fresh = importlib.import_module("fleet_registry")
        assert Path(fresh.__file__).resolve() == (ROOT / "fleet_registry.py").resolve()
        assert hasattr(fresh, "ctb_for") and hasattr(fresh, "surface_latlon")
    finally:
        sys.path[:] = saved_path
        sys.modules["fleet_registry"] = _orig  # preserve module identity for the suite


def test_well_jump_handoff_is_consumed_before_the_sidebar_widget():
    """Cross-page jumps (map click, board/brief row-select) park the target well in
    _well_jump; app.py must consume it into well_id BEFORE the sidebar selectbox
    (which OWNS the well_id key) instantiates. A direct mid-page write raises
    StreamlitAPIException on the live app — AppTest can't fire plotly/dataframe
    selection events, so this pins the handoff half of the mechanism."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.session_state["_well_jump"] = "well_042"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["well_id"] == "well_042"
    assert "_well_jump" not in at.session_state


def test_no_mid_script_writes_to_the_widget_owned_well_id_key():
    """The sidebar selectbox owns key 'well_id', so a page-body write raises live.
    Legal writers are on_change callbacks (run pre-script) and ensure_state's
    None-only fallback (reachable only in the per-view harness, where the sidebar
    widget never rendered). The jump paths must go through _well_jump."""
    assert 'st.session_state["well_id"]' not in _fn_source("views/_common.py", "jump_to_well")
    assert 'st.session_state["_well_jump"]' in _fn_source("views/_common.py", "jump_to_well")
    assert 'st.session_state["well_id"]' not in _fn_source("views/surveillance.py",
                                                           "_apply_map_selection")
    assert 'st.session_state["_well_jump"]' in _fn_source("views/surveillance.py",
                                                          "_apply_map_selection")


def _fn_source(rel: str, fn: str) -> str:
    import ast
    import textwrap

    tree = ast.parse((ROOT / rel).read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == fn)
    return textwrap.dedent(ast.get_source_segment((ROOT / rel).read_text(), node))


# =================================================================================
# UX / guidance round 2 (v0.9.0): page-purpose affordance, next-step pointers,
# exceedance convention, loss-book disambiguation, typed deck inputs, terminology.
# =================================================================================

import pytest  # noqa: E402

ALL_TITLES = ["Home", "Surveillance", "Morning Brief", "Optimization Board",
              "Ongoing Events", "Deferment Overview", "Causes & Pareto",
              "Recovery Work Queue", "Well 360", "Action Chain",
              "Sources & BYOD", "Methods & Limitations"]

# Every page_purpose body leads with this phrase — the AppTest render check keys
# on it (the popover's markdown body lands in the element tree).
PURPOSE_MARKER = "The question this page answers"


def _view_sources() -> dict:
    """{page title: source Path} straight from the navigation spec, so a page
    added to PAGES without the convention fails these tests automatically."""
    import views
    out = {}
    for specs in views.PAGES.values():
        for (title, _icon, _slug, render, _default) in specs:
            out[title] = Path(sys.modules[render.__module__].__file__)
    return out


def test_every_view_calls_page_purpose_exactly_once():
    """The CD-ported convention: ONE 'ℹ️ What is this page for?' popover per view,
    placed by a single helper call (source pin — a dropped or doubled call on any
    registered page fails loudly)."""
    for title, path in _view_sources().items():
        src = path.read_text()
        n = src.count("c.page_purpose(")
        assert n == 1, f"{title}: expected exactly 1 page_purpose call, found {n}"


@pytest.mark.parametrize("title", ALL_TITLES)
def test_each_view_renders_exactly_one_purpose_popover(title, bootstrapped,
                                                       monkeypatch):
    """Render check: the popover BODY (its lead phrase) reaches the element tree
    exactly once on every registered page."""
    at = _run_view(title, monkeypatch, {})
    hits = [m for m in at.markdown if PURPOSE_MARKER in str(m.value)]
    assert len(hits) == 1, (f"{title}: expected exactly 1 rendered page-purpose "
                            f"body, found {len(hits)}")


def test_page_purpose_helper_is_product_local_popover():
    src = _fn_source("views/_common.py", "page_purpose")
    assert 'st.popover("ℹ️ What is this page for?")' in src
    # NOT in the vendored chrome (product_theme must not drift across products).
    assert "page_purpose" not in (ROOT / "product_theme.py").read_text()


# ---- OC-WO-4: SPE exceedance convention -----------------------------------------

def test_action_chain_monte_carlo_uses_the_exceedance_convention():
    """Display relabel only: the 90th-percentile (upside) NPV is labeled P10, the
    10th-percentile (downside) P90 — P10 ≥ P50 ≥ P90 for NPV, matching EW's
    read-only vendored core. The old percentile-index labels must be gone."""
    src = (ROOT / "views" / "action_chain.py").read_text()
    assert "P10 (downside)" not in src and "P90 (upside)" not in src
    # the mapping itself: the LARGER percentile output carries the P10 label
    assert 'metric("P10 (upside)", f"${mc[\'p90\']:,.0f}"' in src
    assert 'metric("P90 (downside)", f"${mc[\'p10\']:,.0f}"' in src


def test_methods_carries_the_verbatim_exceedance_sentence():
    from views import methods
    assert methods.EXCEEDANCE_SENTENCE == (
        "Suite convention: Pxx = probability of exceedance — P10 is the high "
        "case, P90 the low case.")
    assert "EXCEEDANCE_SENTENCE" in (ROOT / "views" / "methods.py").read_text()


# ---- OC-WO-10: stage-2 agent name scoped by lift --------------------------------

def test_action_chain_stage2_header_scopes_agent_name_by_lift(bootstrapped,
                                                              monkeypatch):
    """A gas-lift / rod-pump / flowing well's Action Chain must not title stage 2
    'ESP Failure-Risk Agent' (Well 360 fixed this scoping in round 1; the chain
    header now mirrors it)."""
    non_esp = next(w for w in (f"well_{n:03d}" for n in range(1, 101))
                   if fr.get(w).lift != "ESP")
    at = _run_view("Action Chain", monkeypatch, {"well_id": non_esp})
    md = " ".join(str(m.value) for m in at.markdown)
    # Pin the SECTION TITLE (the page-owned string). The vendored AFE document
    # (apps/afe-copilot, read-only) legitimately says 'Prepared By (auto — ESP
    # Failure-Risk Agent)' inside the artifact, so the assertion targets the
    # stage header, not every string on the page.
    assert "2 · Predict — ESP Failure-Risk Agent" not in md, \
        f"{non_esp} ({fr.get(non_esp).lift}) still titled the ESP agent"
    assert "2 · Predict — Failure-Risk Agent" in md


# ---- OC-WO-2: Recovery Work Queue — dead end + id collision ---------------------

def test_recovery_queue_disambiguates_loss_book_wells_and_links_the_chain():
    src = (ROOT / "views" / "recovery_queue.py").read_text()
    assert "(loss book)" in src                       # display disambiguation
    assert "NOT the surveillance wells that share the same id" in src
    assert 'c.next_step("Action Chain"' in src        # the one-click handoff
    # …but NEVER a well jump from this page — a loss-book id carried into the
    # chain would BE the fake join the console's own invariant forbids.
    assert "jump_to_well" not in src
    assert "handle_row_jump" not in src
    assert "_well_jump" not in src


# ---- OC-WO-3: Home — clickable next steps ---------------------------------------

def test_home_what_to_do_first_steps_are_clickable():
    src = _fn_source("views/home.py", "_what_broke_and_next")
    assert '.button("Go →"' in src
    assert "c.jump_to_well(" in src
    # jumps go through the sanctioned handoff — never a direct write to the
    # widget-owned key from the page body (crashed OC in production).
    assert 'st.session_state["well_id"]' not in src
    assert "Counts differ by design" in (ROOT / "views" / "home.py").read_text()


# ---- OC-WO-6 / OC-WO-7: board path to authorize, surveillance warm ending -------

def test_board_has_visible_path_to_authorize_and_column_help():
    src = (ROOT / "views" / "triage_board.py").read_text()
    assert "Build the AFE for the selected well on the Action Chain" in src
    assert "the Action Chain picks the same" in src   # …well up automatically
    for col in ("Addressable BOPD", "NPV Basis", "Risk Rank"):
        assert f'"{col}": st.column_config' in src, f"missing help on {col!r}"


def test_surveillance_decline_check_retitle_and_drilldown_next_steps():
    src = (ROOT / "views" / "surveillance.py").read_text()
    assert "On Trend? — Fleet Decline Check" in src
    assert "On the Type Curve?" not in src
    assert "not an offset-well type curve" in src
    assert 'c.next_step("Well 360"' in src
    assert 'c.next_step("Action Chain"' in src
    # the map's amber definition names the HEALTH read vs the board's economic watch
    assert "distinct from the " in src and "At-Risk Watch" in src
    # …and Methods carries the canonical three-tier mapping
    assert "One word, three tiers" in (ROOT / "views" / "methods.py").read_text()


# ---- OC-WO-8: row-jump parity + demo-injection disclosure -----------------------

def test_ongoing_events_row_jump_and_demo_disclosure():
    src = (ROOT / "views" / "ongoing_events.py").read_text()
    assert 'on_select="rerun"' in src and 'selection_mode="single-row"' in src
    assert 'c.handle_row_jump(ev, src_df, "_oe_jump")' in src
    assert "appears ONLY on this page" in src
    assert "Select a row to open the well on Surveillance" in src


def test_brief_detailed_panels_have_row_jump_parity():
    src = (ROOT / "views" / "morning_brief.py").read_text()
    for sentinel in ("_mb_down_jump", "_mb_div_jump", "_mb_anom_jump"):
        assert sentinel in src, f"detailed panels lost the {sentinel} row-jump"


# ---- OC-WO-9: human-readable well labels ----------------------------------------

def test_well_label_formats_and_preserves_raw_values():
    from views import _common as c

    m = fr.get("well_007")
    assert c.well_label("well_007") == f"well_007 · {m.name} ({m.lift})"
    # formatter is display-only: every picker passes it as format_func, values raw
    for rel in ("app.py", "views/surveillance.py", "views/well_360.py",
                "views/action_chain.py"):
        assert "format_func=c.well_label" in (ROOT / rel).read_text(), rel


# ---- OC-WO-11: sidebar convergence ----------------------------------------------

def test_sidebar_deck_is_typed_inputs_with_portfolio_labels():
    src = (ROOT / "app.py").read_text()
    assert "st.slider" not in src                      # exact typed inputs (round-1 ask)
    assert 'st.number_input("Oil price ($/bbl)"' in src
    assert 'st.number_input("NRI (net revenue interest)"' in src
    assert "help=c.NRI_HELP" in src
    assert "DF(m) = (1+r)^(m/12)" in src
    assert "Capital Desk additionally nets severance + ad valorem taxes" in src
    assert 'st.selectbox("Well", _well_ids, key="well_id"' in src


# ---- OC-WO-12: terminology / units / export labels ------------------------------

def test_display_terminology_and_export_labels_are_aligned():
    for p in sorted((ROOT / "views").glob("*.py")) + [ROOT / "app.py"]:
        src = p.read_text()
        assert "BO/day" not in src, f"'BO/day' unit token survives in {p.name}"
        assert "⬇" not in src, f"glyph in a download label in {p.name}"
        assert "(markdown)" not in src, f"lowercase format token in {p.name}"
    mb = (ROOT / "views" / "morning_brief.py").read_text()
    assert "Deferred at Risk" not in mb               # unified money label
    assert "Deferred $/day (net)" in mb
    # portfolio deck cell format string
    assert '"${price:.0f}/bbl · NRI {nri:.0%} · {disc:.1%} disc"' \
        in (ROOT / "views" / "_common.py").read_text()


# ---- OC-WO-5: cross-product pointers are honest captions ------------------------

def test_cross_product_pointers_name_product_page_and_url():
    ac = (ROOT / "views" / "action_chain.py").read_text()
    assert "capital-desk.streamlit.app" in ac
    assert "Draft AFE" in ac
    w3 = (ROOT / "views" / "well_360.py").read_text()
    assert "engineering-workbench.streamlit.app" in w3
    assert "same well id" in w3
