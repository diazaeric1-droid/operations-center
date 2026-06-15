"""Core engine tests: alias loading, bootstrap artifacts, and the NUMERIC
INVARIANTS that pin Operations Center's numbers to the absorbed components:

(a) rank_fleet ≡ pe-pipeline's pipeline_core.rank_fleet on the same apps/ dir
    (subprocess with PE_APPS_ROOT pointed at ours) — identical DataFrame.
(b) get_alerts ≡ digest.handoff.export_alerts called directly — same count,
    same top well, same deferred_bopd.
(c) Loss-accounting %-deferred on the REAL Colorado fleet through the view-layer
    code path ≡ the component's own analytics — and equals the published 6.0%.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import core

PRICE = 70.0
NRI = 0.80

PE_PIPELINE = Path("/Users/ericbot/code/upstream-suite/pe-pipeline")


# ---- alias imports -----------------------------------------------------------

def test_alias_packages_resolve_from_vendored_apps():
    for alias in ("digest", "esp", "afe", "deferment"):
        mod = sys.modules.get(alias)
        assert mod is not None, f"alias {alias!r} not registered"
        assert str(core.APP_DIRS[alias]) in str(Path(mod.__file__).resolve()), \
            f"{alias} did not load from the vendored copy"


def test_alias_versions_match_absorbed_components():
    assert sys.modules["digest"].__version__ == "0.6.3"
    assert sys.modules["deferment"].__version__ == "0.5.1"
    assert sys.modules["esp"].__version__ == "0.7.3"


def test_entry_points_importable_via_core():
    # One callable per absorbed component, reachable through core's bindings.
    assert callable(core.digest_handoff.export_alerts)
    assert callable(core.deferment_engine.compute_deferment)
    assert callable(core.esp_handoff.diagnose)
    assert callable(core.afe_handoff.render_afe_markdown)
    assert callable(core.econ_core.risked_npv)


# ---- bootstrap artifacts -----------------------------------------------------

def test_bootstrap_produces_expected_files(bootstrapped):
    assert len(list(core.DIGEST_FLEET.glob("well_*.csv"))) == 100
    assert core.ESP_MODEL.exists()
    assert len(list(core.DEFERMENT_WELLS.glob("well_*.csv"))) == 40
    assert core.DEFERMENT_EVENTS.exists()
    assert core.DEFERMENT_REAL_CO.exists(), "committed real Colorado extract missing"


# ---- triage board contract -----------------------------------------------------

@pytest.fixture(scope="module")
def board(bootstrapped) -> pd.DataFrame:
    return core.rank_fleet(price_per_bbl=PRICE, net_revenue_interest=NRI)


def test_board_schema_and_sort(board: pd.DataFrame):
    assert list(board.columns) == list(core.RANK_COLUMNS)
    assert not board.empty
    assert board["opportunity_score"].is_monotonic_decreasing
    assert (board["opportunity_score"] == board["est_risked_npv"]).all()


def test_board_deferred_usd_is_net(board: pd.DataFrame):
    expected = board["deferred_bopd"] * PRICE * NRI
    assert ((board["deferred_usd_per_day"] - expected).abs() < 0.01).all()


def test_board_no_action_tier_semantics(board: pd.DataFrame):
    tier = board[board["recommended_intervention"] == "no_action"]
    # Tier wells carry zero opportunity by construction (may be empty on this data).
    assert (tier["est_risked_npv"] == 0.0).all()
    assert (tier["opportunity_score"] == 0.0).all()


def test_empty_fleet_returns_typed_empty_frame(monkeypatch):
    monkeypatch.setattr(core, "DIGEST_FLEET",
                        Path(__file__).resolve().parent / "_no_such_fleet")
    df = core.rank_fleet(price_per_bbl=PRICE, net_revenue_interest=NRI)
    assert df.empty
    assert list(df.columns) == list(core.RANK_COLUMNS)


# ---- NUMERIC INVARIANT (a): parity with pe-pipeline's orchestrator -------------

@pytest.mark.skipif(not PE_PIPELINE.exists(),
                    reason="pe-pipeline checkout not present (CI runs without it)")
def test_rank_fleet_identical_to_pipeline_core(board: pd.DataFrame):
    """pe-pipeline's pipeline_core, pointed at OUR vendored apps via PE_APPS_ROOT,
    must produce an identical ranked frame — proof the absorbed orchestrator's
    math survived the port bit-for-bit."""
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(PE_PIPELINE)!r})\n"
        "import pipeline_core as pc\n"
        f"df = pc.rank_fleet(price_per_bbl={PRICE}, net_revenue_interest={NRI})\n"
        "df.to_json(sys.stdout, orient='split')\n"
    )
    env = dict(os.environ, PE_APPS_ROOT=str(core.APPS_ROOT))
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    ref = pd.read_json(io.StringIO(out.stdout), orient="split")
    ref = ref.astype({c: t for c, t in core.RANK_COLUMNS.items()})
    pd.testing.assert_frame_equal(board.reset_index(drop=True),
                                  ref.reset_index(drop=True))
    # Spelled out for the report: same top well, same risked-NPV value.
    assert board["well_id"].iloc[0] == ref["well_id"].iloc[0]
    assert board["est_risked_npv"].iloc[0] == ref["est_risked_npv"].iloc[0]


# ---- NUMERIC INVARIANT (b): digest alert feed parity ----------------------------

def test_get_alerts_matches_digest_handoff_directly(bootstrapped):
    mine = core.get_alerts(price_per_bbl=PRICE)
    direct = core.digest_handoff.export_alerts(
        core.DIGEST_FLEET, price_per_bbl=PRICE, ack_path=str(core.DIGEST_ACK))
    assert len(mine) == len(direct)
    assert mine[0]["well_id"] == direct[0]["well_id"]
    assert mine[0]["deferred_bopd"] == direct[0]["deferred_bopd"]
    # Known component values on the seeded 100-well fleet (generator is deterministic).
    assert len(mine) == 9
    assert mine[0]["well_id"] == "well_029"
    assert mine[0]["deferred_bopd"] == 0.0


# ---- NUMERIC INVARIANT (c): real-Colorado %-deferred parity ----------------------

def test_real_colorado_pct_deferred_matches_component(bootstrapped):
    """The Loss Accounting pages' code path (views/_common.deferment_data →
    fleet_kpis) must equal the component computed directly from its own modules,
    and reproduce the published real-fleet ~6.0% deferred."""
    from views import _common as vc

    # View-layer path (exactly what Deferment Overview renders).
    _f, _e, daily_view = vc.deferment_data(core.DEF_SRC_REAL_CO, PRICE)
    k_view = core.deferment_analytics.fleet_kpis(daily_view, PRICE)

    # Component path, called directly on its own modules.
    fleet = core.deferment_ndic.load_ndic_fleet(str(core.DEFERMENT_REAL_CO))
    evc = pd.DataFrame(columns=[*core.deferment_loader.EVENT_COLUMNS, "reason_key"])
    daily_direct = core.deferment_engine.compute_deferment(
        fleet, evc, price_per_bbl=PRICE)
    k_direct = core.deferment_analytics.fleet_kpis(daily_direct, PRICE)

    assert k_view["pct_deferred"] == k_direct["pct_deferred"]
    assert k_view["deferred_usd"] == k_direct["deferred_usd"]
    assert k_view["n_wells"] == k_direct["n_wells"] == 28
    assert round(k_view["pct_deferred"], 1) == 6.0  # the honest real-fleet anchor


# ---- NUMERIC INVARIANT (d): the ONE import-rewritten file behaves identically ----

DIGEST_REPO = Path("/Users/ericbot/code/upstream-suite/daily-production-digest")


@pytest.mark.skipif(not DIGEST_REPO.exists(),
                    reason="digest component checkout not present (CI runs without it)")
def test_ledger_rewrite_behavior_matches_original_component(bootstrapped):
    """src/ledger.py is the ONLY vendored file we transformed (absolute
    ``from src.anomaly_detector`` → package-relative, required by the alias
    loader). Prove the transform changed nothing: the ORIGINAL component, run
    from its own repo (where ``src.*`` imports resolve) on OUR bootstrapped
    fleet, must produce the same ledger numbers as the vendored module."""
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(DIGEST_REPO)!r})\n"
        "from src.data_loader import load_fleet\n"
        "from src.ledger import build_ledger\n"
        f"fleet = load_fleet({str(core.DIGEST_FLEET)!r})\n"
        "ledger, summary = build_ledger(fleet, window_days=30)\n"
        "print(json.dumps({'period_deferred_usd': summary['period_deferred_usd'],\n"
        "                  'days_scanned': summary['days_scanned'],\n"
        "                  'top_cause': summary['top_cause'],\n"
        "                  'rows': len(ledger)}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    ref = json.loads(out.stdout.strip().splitlines()[-1])

    ledger, summary = core.digest_ledger.build_ledger(
        core.load_scada_fleet(), window_days=30)
    assert summary["period_deferred_usd"] == ref["period_deferred_usd"]
    assert summary["days_scanned"] == ref["days_scanned"]
    assert summary["top_cause"] == ref["top_cause"]
    assert len(ledger) == ref["rows"]


# ---- chain + events -------------------------------------------------------------

def test_run_chain_for_selected_well(bootstrapped):
    res = core.run_chain(price_per_bbl=PRICE, net_revenue_interest=NRI,
                         well_id="well_001", log=lambda *_a, **_k: None)
    assert res["top"]["well_id"] == "well_001"
    assert 0.0 <= res["diagnosis"]["esp_risk_score"] <= 1.0
    assert "well_001" in res["afe_md"]
    assert "Authorization for Expenditure" in res["afe_md"] or "AFE" in res["afe_md"]


def test_replay_events_demo_outage_is_ongoing_12_days(bootstrapped):
    fleet = core.load_scada_fleet()
    events = core.replay_events(fleet, price_per_bbl=PRICE, inject_demo=True)
    by_well = {e.well_id: e for e in events}
    evt = by_well.get(core.DEMO_OUTAGE_WELL)
    assert evt is not None, "injected demo outage did not open an event"
    assert evt.state == "ONGOING"
    assert evt.duration_days == core.DEMO_OUTAGE_LEN
    assert evt.deferred_usd > 0


def test_core_importable_without_streamlit():
    """core must stay streamlit-free (CI bootstrap + tests drive it headless)."""
    code = ("import sys\n"
            "sys.modules['streamlit'] = None  # poison: any import attempt fails\n"
            f"sys.path.insert(0, {str(Path(core.__file__).parent)!r})\n"
            "import importlib\n"
            "import core as c2\n"
            "assert c2.fleet_size() >= 0\n"
            "print('OK')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK" in out.stdout
