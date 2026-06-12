"""Operations Center core — in-process engine over the vendored component apps.

Adapted from pe-pipeline's ``pipeline_core.py`` (the proven alias loader). The
component apps are each packaged as a top-level ``src`` package, so they can't all
be imported normally (the name collides). This module loads each app's ``src``
under a distinct alias via importlib:

    digest     -> apps/daily-production-digest/src    (anomaly scan, brief, events)
    deferment  -> apps/deferment-iq/src               (lost-oil accounting)
    esp        -> apps/esp-failure-risk-agent/src     (30-day failure-risk scoring)
    afe        -> apps/afe-copilot/src                 (cost rollup + economics + AFE)

so the whole surveillance console — triage board, morning brief, event lifecycle,
loss accounting, and the detect → predict → authorize chain — runs in ONE Python
process with no subprocesses and no per-app virtualenvs.

``afe.econ_core`` is the suite-wide economics kernel; the triage ranking risks
cash flows with the exact same convention the AFE Copilot authorizes against
(same pattern as pipeline_core).

IMPORTANT: this module stays importable WITHOUT streamlit (no streamlit import at
module top) so product tests and CI bootstrap can drive it headless. Views add
their own ``st.cache_*`` wrappers on top.
"""
from __future__ import annotations

import importlib
import importlib.util
import io
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Apps are vendored plain directories under apps/; override for unusual layouts.
APPS_ROOT = Path(os.environ.get("OPS_APPS_ROOT", HERE / "apps"))
APP_DIRS = {
    "digest": APPS_ROOT / "daily-production-digest",
    "deferment": APPS_ROOT / "deferment-iq",
    "esp": APPS_ROOT / "esp-failure-risk-agent",
    "afe": APPS_ROOT / "afe-copilot",
}


def _load_pkg(app_dir: Path, alias: str):
    """Load ``app_dir/src`` as a top-level package named ``alias`` so its internal
    relative imports (``from .features import ...``) resolve under that alias."""
    if alias in sys.modules:
        return sys.modules[alias]
    src = app_dir / "src"
    if not (src / "__init__.py").exists():
        raise FileNotFoundError(
            f"{alias}: missing {src}/__init__.py — the apps are vendored under apps/; "
            f"run from the repo root (or set OPS_APPS_ROOT).")
    spec = importlib.util.spec_from_file_location(
        alias, src / "__init__.py", submodule_search_locations=[str(src)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# Register the four packages under aliases, then import the entry points.
for _alias, _dir in APP_DIRS.items():
    _load_pkg(_dir, _alias)

digest_handoff = importlib.import_module("digest.handoff")
digest_loader = importlib.import_module("digest.data_loader")
digest_detector = importlib.import_module("digest.anomaly_detector")
digest_brief = importlib.import_module("digest.brief_writer")
digest_events = importlib.import_module("digest.event_store")
digest_ledger = importlib.import_module("digest.ledger")
esp_handoff = importlib.import_module("esp.handoff")
esp_loader = importlib.import_module("esp.data_loader")
esp_features = importlib.import_module("esp.features")
esp_model = importlib.import_module("esp.model")
esp_explainer = importlib.import_module("esp.explainer")
afe_handoff = importlib.import_module("afe.handoff")
afe_cost_db = importlib.import_module("afe.cost_db")
afe_economics = importlib.import_module("afe.economics")
# Suite-wide economics kernel, vendored inside the AFE app and reused here so the
# triage board risks cash flows with the exact same convention as the AFE it chains.
econ_core = importlib.import_module("afe.econ_core")
deferment_loader = importlib.import_module("deferment.data_loader")
deferment_engine = importlib.import_module("deferment.deferment")
deferment_analytics = importlib.import_module("deferment.analytics")
deferment_ndic = importlib.import_module("deferment.ndic")
deferment_reasons = importlib.import_module("deferment.reason_codes")
deferment_narrator = importlib.import_module("deferment.narrator")

# ---- canonical data locations (inside the vendored apps) ---------------------
DIGEST_FLEET = APP_DIRS["digest"] / "data" / "synthetic" / "fleet"
DIGEST_ACK = APP_DIRS["digest"] / "acknowledged.yml"
ESP_DATA = APP_DIRS["esp"] / "data" / "synthetic"
ESP_MODEL = APP_DIRS["esp"] / "artifacts" / "esp_risk_model.joblib"
DEFERMENT_DATA = APP_DIRS["deferment"] / "data" / "synthetic"
DEFERMENT_WELLS = DEFERMENT_DATA / "wells"
DEFERMENT_EVENTS = DEFERMENT_DATA / "events.csv"
DEFERMENT_REAL_CO = APP_DIRS["deferment"] / "data" / "real" / "colorado" / "production.csv"

# Deferment data-source keys (Loss Accounting pages + the Data page share these).
DEF_SRC_REAL_CO = "real_co"      # Colorado ECMC public monthly records (committed) — DEFAULT
DEF_SRC_SYNTHETIC = "synthetic"  # modeled reason-coded fleet (regenerated by bootstrap)
DEF_SRC_UPLOAD = "upload"        # user-uploaded tidy monthly CSV (session only)


# ---- bootstrap (data + model are .gitignore'd, so regenerate on first run) ----

def ensure_digest_data(log=print) -> None:
    if not any(DIGEST_FLEET.glob("well_*.csv")):
        log("Generating synthetic SCADA fleet (digest)…")
        runpy.run_path(str(APP_DIRS["digest"] / "data" / "synthetic" / "generate_fleet.py"),
                       run_name="__main__")


def ensure_deferment_data(log=print) -> None:
    if not any(DEFERMENT_WELLS.glob("well_*.csv")) or not DEFERMENT_EVENTS.exists():
        log("Generating synthetic reason-coded fleet (deferment)…")
        runpy.run_path(str(DEFERMENT_DATA / "generate.py"), run_name="__main__")


def ensure_esp_model(log=print) -> Path:
    if ESP_MODEL.exists():
        return ESP_MODEL
    if not any(ESP_DATA.glob("well_*.csv")):
        log("Generating synthetic SCADA (ESP)…")
        runpy.run_path(str(ESP_DATA / "generate.py"), run_name="__main__")
    log("Training the ESP failure-risk model (~30s, one time)…")
    fleet = esp_loader.load_fleet(ESP_DATA)
    X = esp_features.featurize_fleet(fleet)
    labels = esp_loader.load_labels(ESP_DATA / "labels.csv").set_index("well_id")["failed_within_30d"]
    aligned = X.join(labels, how="inner")
    m = esp_model.ESPRiskModel()
    m.fit(aligned[X.columns], aligned["failed_within_30d"])
    m.save(ESP_MODEL)
    return ESP_MODEL


def bootstrap(log=print) -> None:
    """Regenerate every gitignored artifact the console needs (idempotent)."""
    ensure_digest_data(log)
    ensure_deferment_data(log)
    ensure_esp_model(log)


# ---- SCADA fleet (Today / Well File pages) -----------------------------------

def load_scada_fleet() -> dict:
    """The bootstrapped synthetic SCADA fleet: dict[well_id -> DataFrame]."""
    return digest_loader.load_fleet(DIGEST_FLEET)


def load_scada_fleet_from_bytes(data: bytes) -> dict:
    """BYOD path: parse one uploaded fleet SCADA CSV through the digest's EXISTING
    loader (same schema validation + per-well date parse/sort as on-disk)."""
    return digest_loader.load_fleet_from_csv(io.BytesIO(data))


def scan_anomalies(fleet: dict, price_per_bbl: float = 70.0) -> list:
    """Full deterministic fleet scan (all categories), acknowledged events flagged.
    Returns the digest's money-first sorted list of ``Anomaly`` dataclasses."""
    acknowledged = digest_detector.load_acknowledgements(str(DIGEST_ACK))
    return digest_detector.scan_fleet(fleet, price_per_bbl=price_per_bbl,
                                      acknowledged=acknowledged)


def get_alerts(price_per_bbl: float = 70.0) -> list[dict]:
    """ESP-related WellAlerts (the chain's stage-1 artifact), ranked money-first."""
    return digest_handoff.export_alerts(
        DIGEST_FLEET, price_per_bbl=price_per_bbl, ack_path=str(DIGEST_ACK))


def fleet_size() -> int:
    """Number of wells the digest scanned this run (for the funnel caption)."""
    try:
        return len(digest_loader.load_fleet(DIGEST_FLEET))
    except Exception:  # noqa: BLE001
        return 0


def alert_for(well_id: str, price_per_bbl: float = 70.0,
              alerts: list[dict] | None = None) -> dict:
    """A WellAlert-shaped dict for ANY fleet well — the real alert if the digest
    flagged it, else a synthesized one pointing at the well's SCADA CSV so the
    detect → predict → authorize chain works for non-alerted wells too."""
    alerts = get_alerts(price_per_bbl=price_per_bbl) if alerts is None else alerts
    for a in alerts:
        if a["well_id"] == well_id:
            return a
    import datetime as _dt
    return {
        "well_id": well_id,
        "category": "fleet_scan",
        "severity": "—",
        "headline": "Not flagged by today's digest — surfaced by fleet risk scoring.",
        "deferred_bopd": 0.0,
        "baseline_bopd": 0.0,
        "scada_csv": str((DIGEST_FLEET / f"{well_id}.csv").resolve()),
        "date": _dt.date.today().isoformat(),
    }


def diagnose(alert: dict, model_path: Path | None = None) -> dict:
    """Stage 2: ESP scores the alert's well → AFE-ready WellDiagnosis."""
    return esp_handoff.diagnose(
        alert["scada_csv"], well_id=alert.get("well_id"),
        deferred_bopd=alert.get("deferred_bopd", 0.0),
        baseline_bopd=alert.get("baseline_bopd", 0.0),
        model_path=str(model_path or ESP_MODEL))


def render_afe(diag: dict, working_interest: float = 1.0,
               net_revenue_interest: float = 0.80, realized_price: float = 70.0) -> str:
    """Stage 3: WellDiagnosis → deterministic AFE markdown."""
    return afe_handoff.render_afe_markdown(
        diag, working_interest=working_interest,
        net_revenue_interest=net_revenue_interest, realized_price=realized_price)


def well_scada(alert_or_csv) -> "object":
    """Load the well's SCADA (with the digest's bopd column) for plotting."""
    csv = alert_or_csv["scada_csv"] if isinstance(alert_or_csv, dict) else alert_or_csv
    return digest_loader.load_well(csv)


# ---- fleet triage board (deterministic, no LLM) ------------------------------
# Ported from pipeline_core.rank_fleet — same schema, same thresholds, same
# economics — so the board reproduces the pe-pipeline orchestrator's values.

RANK_COLUMNS: dict[str, str] = {
    "well_id": "string",
    "deferred_bopd": "float64",
    "deferred_usd_per_day": "float64",
    "failure_risk_30d": "float64",
    "recommended_intervention": "string",
    "incremental_bopd": "float64",       # addressable rate the intervention protects/restores
    "est_risked_npv": "float64",         # risk-weighted net NPV (chain economics or proxy)
    "npv_basis": "string",               # "chain_economics" or "proxy"
    "opportunity_score": "float64",      # the sort key (== est_risked_npv)
}


def _empty_rank_frame() -> "object":
    import pandas as pd
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in RANK_COLUMNS.items()})


def _score_fleet_risk(fleet: dict, model_path: Path) -> dict[str, float]:
    """Score every well in ``fleet`` with the ESP model → {well_id: risk}.
    Degrades to an empty map (baseline risk fallback) if the model is unavailable."""
    try:
        X = esp_features.featurize_fleet(fleet)
        model = esp_model.ESPRiskModel.load(str(model_path))
        probs = model.predict_proba(X)
        return {str(wid): float(p) for wid, p in zip(X.index, probs)}
    except Exception:  # noqa: BLE001  (missing model/deps → baseline fallback)
        return {}


# When the ESP model can't score a well, assign this baseline 30-day risk so the
# board is still fully populated and deterministic. Conservative (low) by design.
BASELINE_RISK_30D = 0.05

# No-action tier thresholds (identical to pipeline_core — tune here only).
_NO_ACTION_NPV_THRESHOLD = 10_000    # $ — below this, not worth flagging
_NO_ACTION_RISK_THRESHOLD = 0.15     # 30-day failure probability


def rank_fleet(price_per_bbl: float = 70.0, net_revenue_interest: float = 0.80,
               model_path: Path | None = None) -> "object":
    """Rank the WHOLE bootstrapped fleet by risked-NPV opportunity (deterministic).

    Pure and deterministic given ``(fleet, price_per_bbl, net_revenue_interest)`` —
    no LLM, no API key. ``bootstrap()`` must have run. Same columns, thresholds, and
    economics as pe-pipeline's ``pipeline_core.rank_fleet`` (the product tests pin
    numeric equality against it). Returns a DataFrame sorted descending by
    ``opportunity_score`` (ties broken by ``failure_risk_30d``)."""
    import pandas as pd

    model_path = Path(model_path or ESP_MODEL)

    fleet = digest_loader.load_fleet(DIGEST_FLEET)
    if not fleet:
        return _empty_rank_frame()

    deferred_by_well: dict[str, float] = {}
    for a in get_alerts(price_per_bbl=price_per_bbl):
        deferred_by_well[a["well_id"]] = float(a.get("deferred_bopd", 0.0))

    risk_by_well = _score_fleet_risk(fleet, model_path)

    rows: list[dict] = []
    for well_id, scada in fleet.items():
        well_id = str(well_id)
        baseline_bopd = (float(scada["bopd"].tail(7).mean())
                         if "bopd" in scada.columns and len(scada) else 0.0)
        deferred_bopd = deferred_by_well.get(well_id, 0.0)
        deferred_usd_per_day = deferred_bopd * price_per_bbl * net_revenue_interest

        risk = risk_by_well.get(well_id, BASELINE_RISK_30D)

        try:
            feats = esp_features.featurize_well(scada)
            mode, _evidence = esp_explainer.classify_failure_mode(feats)
        except Exception:  # noqa: BLE001
            mode = ""
        intervention, frac = esp_handoff._map_mode(mode)

        incremental_bopd = round(max(deferred_bopd, frac * baseline_bopd, 20.0), 1)

        est_risked_npv = deferred_usd_per_day * 365.0 * risk
        npv_basis = "proxy"
        try:
            total_cost = afe_cost_db.cost_rollup(intervention)["total"]
            econ = afe_economics.compute_economics(
                total_cost, incremental_bopd,
                realized_price_per_bbl=price_per_bbl,
                net_revenue_interest=net_revenue_interest, working_interest=1.0)
            # risked NPV = risk · PV(net revenue) − cost (cost is certain; only the
            # upside is risk-weighted — econ_core.risked_npv).
            est_risked_npv = econ_core.risked_npv(
                econ.net_npv_10pct_usd + total_cost, total_cost, risk)
            npv_basis = "chain_economics"
        except Exception:  # noqa: BLE001
            pass

        no_action = (
            (est_risked_npv < _NO_ACTION_NPV_THRESHOLD and risk < _NO_ACTION_RISK_THRESHOLD)
            or (deferred_bopd == 0.0 and risk < 0.10)
        )
        if no_action:
            intervention = "no_action"
            est_risked_npv = 0.0

        rows.append({
            "well_id": well_id,
            "deferred_bopd": round(deferred_bopd, 1),
            "deferred_usd_per_day": round(deferred_usd_per_day, 2),
            "failure_risk_30d": round(risk, 4),
            "recommended_intervention": intervention,
            "incremental_bopd": incremental_bopd,
            "est_risked_npv": round(est_risked_npv, 2),
            "npv_basis": npv_basis,
            "opportunity_score": round(est_risked_npv, 2),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["opportunity_score", "failure_risk_30d"],
                        ascending=[False, False], kind="mergesort").reset_index(drop=True)
    for col, dtype in RANK_COLUMNS.items():
        df[col] = df[col].astype(dtype)
    return df[list(RANK_COLUMNS)]


def run_chain(price_per_bbl: float = 70.0, working_interest: float = 1.0,
              net_revenue_interest: float = 0.80, well_id: str | None = None,
              log=print) -> dict:
    """Detect → predict → authorize, end to end; returns every stage's artifact.

    With ``well_id=None`` the chain runs for the top money-first alert (pe-pipeline
    behavior). With an explicit ``well_id`` it runs for THAT well — the real digest
    alert if one fired, else a synthesized fleet-scan alert — so the Action Chain
    page works for any selected well. Zero-key deterministic."""
    bootstrap(log)
    alerts = get_alerts(price_per_bbl)
    if well_id is None:
        if not alerts:
            return {"alerts": [], "top": None, "diagnosis": None, "afe_md": None}
        top = alerts[0]
    else:
        top = alert_for(well_id, price_per_bbl=price_per_bbl, alerts=alerts)
    diag = diagnose(top)
    afe_md = render_afe(diag, working_interest, net_revenue_interest, price_per_bbl)
    return {"alerts": alerts, "top": top, "diagnosis": diag, "afe_md": afe_md}


# ---- event state-machine replay (Ongoing Events page) ------------------------
# Ported from the digest demo's _replay_events_cached / _inject_demo_outage, made
# streamlit-free (views cache on top). Same EventStore(":memory:") replay the
# morning brief + scheduler drive.

REPLAY_DAYS = 60

DEMO_OUTAGE_WELL = "well_001"     # a healthy (non-seeded) well in the demo fleet
DEMO_OUTAGE_LEN = 12              # consecutive down days ending on the latest day
DEMO_OUTAGE_FRACTION = 0.55       # held at ~55% of pre-event baseline (~45% loss)


def inject_demo_outage(fleet: dict) -> dict:
    """Shallow fleet copy with a sustained multi-day rate outage spliced into
    ``DEMO_OUTAGE_WELL`` (synthetic demo only; in-memory, CSVs untouched)."""
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
        df2.loc[idx, "gas_mcfd"] = df2.loc[idx, "gas_mcfd"] * (target / baseline)
    out[DEMO_OUTAGE_WELL] = df2
    return out


def replay_events(fleet: dict, price_per_bbl: float = 70.0,
                  inject_demo: bool = False, replay_days: int = REPLAY_DAYS) -> list:
    """Replay the fleet's recent history through the persistent event state machine
    (``EventStore(":memory:")``) and return the live events on the latest as-of day
    (NEW / ONGOING / RESOLVED) — the SAME path the morning brief and the digest's
    scheduler drive. Stateless across calls; writes nothing to disk."""
    import pandas as pd

    if inject_demo:
        fleet = inject_demo_outage(fleet)
    acknowledged = digest_detector.load_acknowledgements(str(DIGEST_ACK))

    all_dates = sorted({d for df in fleet.values() if df is not None and len(df)
                        for d in df["date"]})
    if not all_dates:
        return []
    spine = all_dates[-replay_days:] if replay_days and replay_days > 0 else all_dates

    store = digest_events.EventStore(":memory:")
    live: list = []
    try:
        for as_of_ts in spine:
            as_of = pd.Timestamp(as_of_ts).date().isoformat()
            sliced = {wid: df[df["date"] <= as_of_ts]
                      for wid, df in fleet.items() if df is not None and len(df)}
            sliced = {wid: d for wid, d in sliced.items() if len(d)}
            live = digest_events.update_events(store, sliced, as_of=as_of,
                                               price_per_bbl=price_per_bbl,
                                               acknowledged=acknowledged)
    finally:
        store.close()
    return live


# ---- deferment loading (Loss Accounting pages) --------------------------------
# The deferment dataset is a DIFFERENT dataset at a different cadence than the
# SCADA fleet above (real Colorado ECMC is monthly; the SCADA fleet is daily
# synthetic). No join is faked between them — see the Data page.

def load_deferment_synthetic(price_per_bbl: float = 70.0):
    """Synthetic reason-coded fleet → (fleet, events_classified, daily). The
    deterministic rules classifier runs over the operator notes (no LLM here;
    views may re-run with an LLM client when a key is supplied)."""
    fleet = deferment_loader.load_fleet(DEFERMENT_WELLS)
    events = deferment_loader.load_events(DEFERMENT_EVENTS)
    evc = deferment_engine.classify_events(events, use_llm=False)
    daily = deferment_engine.compute_deferment(fleet, evc, price_per_bbl=price_per_bbl)
    return fleet, evc, daily


def load_deferment_real(csv_path: str | Path | None = None,
                        price_per_bbl: float = 70.0):
    """Real monthly extract (default: the committed Colorado ECMC file) →
    (fleet, empty_events, daily). No public reason codes exist, so events is
    empty and cause attribution is honestly N/A — the deferment QUANTITY is real."""
    import pandas as pd
    csv_path = str(csv_path or DEFERMENT_REAL_CO)
    fleet = deferment_ndic.load_ndic_fleet(csv_path)
    evc = pd.DataFrame(columns=[*deferment_loader.EVENT_COLUMNS, "reason_key"])
    daily = deferment_engine.compute_deferment(fleet, evc, price_per_bbl=price_per_bbl)
    return fleet, evc, daily


def deferment_well_meta(csv_path: str | Path | None = None):
    """Per-well identity (name/operator/field/formation) from a real extract."""
    return deferment_ndic.ndic_well_meta(str(csv_path or DEFERMENT_REAL_CO))


def monthly_template_csv() -> str:
    """Tidy monthly production template for the deferment upload (the exact schema
    ``deferment.ndic.parse_ndic_csv`` validates — derived from NDIC_COLUMNS so the
    template can never drift from the loader)."""
    header = ",".join(deferment_ndic.NDIC_COLUMNS)
    rows = [
        "WELL_A,EXAMPLE 1-2H,Example Operator LLC,Example Field,Niobrara,"
        "2026-01,9300,14100,5200,31",
        "WELL_A,EXAMPLE 1-2H,Example Operator LLC,Example Field,Niobrara,"
        "2026-02,8100,12400,4900,24",
    ]
    return "\n".join([header, *rows]) + "\n"
