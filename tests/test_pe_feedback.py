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
