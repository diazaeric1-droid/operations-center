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
import logging
import os
import re
import runpy
import sys
from pathlib import Path

_log = logging.getLogger("operations_center")

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
DEF_SRC_REAL_CO = "real_co"      # Colorado ECMC public monthly records (BYO reference, not default)
DEF_SRC_SYNTHETIC = "synthetic"  # modeled reason-coded fleet (regenerated by bootstrap) — DEFAULT
DEF_SRC_UPLOAD = "upload"        # user-uploaded tidy monthly CSV (session only)


# ---- bootstrap (data + model are .gitignore'd, so regenerate on first run) ----

def _expected_fleet_size() -> int:
    """Well count the digest generator declares. Lets bootstrap detect a STALE
    on-disk fleet after the generator changes — the fleet CSVs are gitignored and a
    warm Streamlit container (or an existing checkout) keeps the OLD fleet otherwise,
    so a 50→100 change would never take effect without this check."""
    try:
        txt = (APP_DIRS["digest"] / "data" / "synthetic"
               / "generate_fleet.py").read_text()
        m = re.search(r"^N_WELLS\s*=\s*(\d+)", txt, re.M)
        return int(m.group(1)) if m else 0
    except Exception:  # noqa: BLE001
        return 0


def digest_fleet_stale() -> bool:
    """True when the on-disk SCADA fleet count differs from the generator's N_WELLS.
    A warm Streamlit container (or an old checkout) keeps the previous gitignored
    fleet after a 50→100 change; app.py folds this into its readiness check so the
    bootstrap self-heal actually fires instead of the old fleet being served forever."""
    expected = _expected_fleet_size()
    if not expected:
        return False
    return len(list(DIGEST_FLEET.glob("well_*.csv"))) != expected


def ensure_digest_data(log=print) -> None:
    existing = sorted(DIGEST_FLEET.glob("well_*.csv"))
    expected = _expected_fleet_size()
    stale = bool(expected) and len(existing) != expected
    # Also regenerate if the ranking-scorecard ground truth is missing (an existing
    # checkout / warm container can have the right well count but predate it).
    gt_missing = bool(existing) and not (DIGEST_FLEET.parent / "ground_truth.csv").exists()
    if not existing or stale or gt_missing:
        if stale:
            log(f"SCADA fleet is stale ({len(existing)} wells on disk; generator "
                f"declares {expected}) — regenerating…")
            for p in existing:
                p.unlink()
        elif gt_missing:
            log("SCADA fleet present but ground_truth.csv missing — regenerating…")
        else:
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


# ---- fleet health + production divergence (streamlit-free) -------------------
# Shared by the Home fleet-health glance, the Morning Brief metrics, and the
# headless email script — one definition so every surface reports the same number.

# A well is "down" if its latest daily oil rate is effectively zero relative to its
# own recent baseline (a real shut-in, not a naturally low-rate well).
DOWN_ABS_BOPD = 1.0          # below this absolute rate = down regardless of baseline
DOWN_FRAC_OF_BASELINE = 0.10  # OR below 10% of a meaningful (>5 bopd) baseline
_DOWN_BASELINE_MIN = 5.0      # baseline must exceed this to call a collapse "down"


def _latest_and_baseline(df) -> tuple[float, float]:
    """(latest bopd, 7-day pre-latest baseline bopd) for one well's SCADA frame."""
    if df is None or "bopd" not in getattr(df, "columns", []) or not len(df):
        return 0.0, 0.0
    last = float(df["bopd"].iloc[-1])
    base = float(df["bopd"].iloc[-8:-1].mean()) if len(df) >= 8 else last
    return last, base


def _is_down(last: float, base: float) -> bool:
    return last < DOWN_ABS_BOPD or (base > _DOWN_BASELINE_MIN
                                    and last < DOWN_FRAC_OF_BASELINE * base)


def elevated_risk_wells(risk_by_well: dict | None,
                        top_frac: float = 0.25) -> set[str]:
    """The fleet's OWN highest-risk wells (top ``top_frac`` by 30-day ESP score).

    The ESP model is trained on the ESP component's synthetic SCADA, so its score
    on the surveillance fleet is a *relative* failure-signature ranking, not a
    calibrated absolute probability — an absolute cutoff (e.g. ≥50%) would flag most
    of the fleet. A fleet-relative top quantile is the honest, credible reading:
    "the wells whose signature looks worst right now." A well must also sit above
    the fleet median to be flagged, so a uniformly-healthy fleet flags nobody."""
    if not risk_by_well:
        return set()
    vals = sorted(float(r) for r in risk_by_well.values() if r is not None)
    if not vals:
        return set()
    k = max(1, int(round(len(vals) * top_frac)))
    cutoff = vals[-k]
    median = vals[len(vals) // 2]
    cutoff = max(cutoff, median + 1e-9)
    return {str(w) for w, r in risk_by_well.items()
            if r is not None and float(r) >= cutoff}


def fleet_health_summary(fleet: dict, anomalies: list,
                         risk_by_well: dict | None = None,
                         risk_top_frac: float = 0.25) -> dict:
    """Quick-glance fleet status, classifying every well green/amber/red:

    * **impaired** (red): producing a real loss right now — an active anomaly with
      deferred $/day, or a well that is down (≈0 production).
    * **watch** (amber): a non-$ active flag (e.g. comms/metering) or one of the
      fleet's own highest-risk wells (top ``risk_top_frac`` by ESP score).
    * **healthy** (green): everything else — flowing on/around baseline, low relative
      risk.

    Pure/deterministic. ESP risk is folded in *relatively* (see
    ``elevated_risk_wells``) so the glance never claims most of the fleet is failing."""
    total = len(fleet)
    active = [a for a in anomalies if not getattr(a, "acknowledged", False)]
    losing = {str(a.well_id) for a in active if getattr(a, "deferred_bopd", 0.0) > 0}
    flagged = {str(a.well_id) for a in active}

    down, fleet_bopd = set(), 0.0
    for wid, df in fleet.items():
        last, base = _latest_and_baseline(df)
        fleet_bopd += max(last, 0.0)
        if _is_down(last, base):
            down.add(str(wid))

    at_risk = elevated_risk_wells(risk_by_well, top_frac=risk_top_frac)

    impaired = losing | down
    watch = (flagged | at_risk) - impaired
    healthy = max(total - len(impaired) - len(watch), 0)
    return {
        "total": total,
        "healthy": healthy,
        "watch": len(watch),
        "impaired": len(impaired),
        "down": len(down),
        "losing": len(losing),
        "at_risk": len(at_risk),
        "fleet_bopd": round(fleet_bopd, 0),
        "pct_nominal": round(100.0 * healthy / total, 0) if total else 0.0,
    }


# Categories the digest raises for a well diverging from its own decline-expected
# rate (production divergence) vs a data-quality flag (comms/metering).
DIVERGENCE_CATEGORIES = ("rate_drop", "rate_drop_decline_aware")


def production_divergence_summary(fleet: dict, anomalies: list) -> dict:
    """Wells down + wells diverging from decline-expected production.

    * ``down`` — wells at ≈0 production (full shut-in / outage).
    * ``divergences`` — active rate-loss anomalies (off the well's own decline-
      expected rate), money-first, with deferred bopd/$ already attached.

    The metric a foreman wants at 6:30am: who is OFF, and who is BELEAGUERED."""
    active = [a for a in anomalies if not getattr(a, "acknowledged", False)]
    div = [a for a in active if getattr(a, "category", "") in DIVERGENCE_CATEGORIES]
    div = sorted(div, key=lambda a: -getattr(a, "deferred_usd_per_day", 0.0))

    down = []
    for wid, df in fleet.items():
        last, base = _latest_and_baseline(df)
        if _is_down(last, base):
            down.append({"well_id": str(wid), "last_bopd": round(last, 1),
                         "baseline_bopd": round(base, 1)})
    down.sort(key=lambda d: -d["baseline_bopd"])

    return {
        "n_down": len(down),
        "down": down,
        "n_divergences": len(div),
        "divergence_bopd": round(sum(getattr(a, "deferred_bopd", 0.0) for a in div), 1),
        "divergence_usd_day": round(
            sum(getattr(a, "deferred_usd_per_day", 0.0) for a in div), 0),
        "divergences": div,
    }


def _divergence_section_md(div: dict, price_per_bbl: float,
                           net_revenue_interest: float = 0.80) -> str:
    """Markdown block appended to the brief (and emailed) — wells down + divergences.

    Deferred $ is NET-to-operator (× NRI) to match the Morning Brief page KPIs and
    the Triage Board convention (so the inbox and the page never disagree)."""
    nri = net_revenue_interest

    def _net(bopd: float) -> float:
        return float(bopd or 0.0) * price_per_bbl * nri

    L = ["## Production Divergences & Wells Down", ""]
    L.append(f"- **Wells down (≈0 production):** {div['n_down']}")
    L.append(f"- **Production divergences (off decline-expected rate):** "
             f"{div['n_divergences']} — {div['divergence_bopd']:,.0f} bopd / "
             f"${_net(div['divergence_bopd']):,.0f}/day deferred (net to operator at "
             f"${price_per_bbl:,.0f}/bbl × NRI {nri:.2f})")
    L.append("")
    if div["down"]:
        L.append("**Wells down**")
        L.append("")
        L.append("| Well | Latest bopd | Baseline bopd |")
        L.append("|---|---|---|")
        for d in div["down"][:15]:
            L.append(f"| {d['well_id']} | {d['last_bopd']:,.1f} | "
                     f"{d['baseline_bopd']:,.1f} |")
        L.append("")
    if div["divergences"]:
        L.append("**Top production divergences**")
        L.append("")
        L.append("| Well | Category | Deferred bopd | Deferred $/day (net) |")
        L.append("|---|---|---|---|")
        for a in div["divergences"][:15]:
            L.append(f"| {a.well_id} | {a.category} | "
                     f"{getattr(a, 'deferred_bopd', 0.0):,.1f} | "
                     f"${_net(getattr(a, 'deferred_bopd', 0.0)):,.0f} |")
        L.append("")
    if not div["down"] and not div["divergences"]:
        L.append("_No wells down and no production divergences on the latest scan._")
    return "\n".join(L)


def fleet_as_of(fleet: dict) -> str:
    """Latest production date across a SCADA fleet (ISO), or today if empty —
    streamlit-free so the headless brief and the page date identically."""
    import datetime as _dt

    import pandas as pd
    dates = [df["date"].max() for df in fleet.values()
             if df is not None and len(df)]
    return (pd.Timestamp(max(dates)).date().isoformat() if dates
            else _dt.date.today().isoformat())


def scada_as_of() -> str:
    """As-of day of the bootstrapped SCADA fleet (for the email subject line)."""
    return fleet_as_of(load_scada_fleet())


def morning_brief_markdown(price_per_bbl: float = 70.0, inject_demo: bool = False,
                           net_revenue_interest: float = 0.80) -> str:
    """The full morning brief as markdown (deterministic, streamlit-free).

    The vendored digest brief (anomalies + ongoing events) plus the Production
    Divergences & Wells Down section — the SAME composition the Morning Brief page
    renders and the daily email script sends, so the page and the inbox match:
    dated to the data's as-of day and net-of-NRI throughout."""
    fleet = load_scada_fleet()
    anomalies = scan_anomalies(fleet, price_per_bbl=price_per_bbl)
    summary = digest_loader.fleet_summary(fleet)
    events = replay_events(fleet, price_per_bbl=price_per_bbl, inject_demo=inject_demo)
    as_of = fleet_as_of(fleet)
    # The vendored digest reports deferred $ GROSS (barrels × price); the rest of the
    # Operations Center is net-to-operator (× NRI). Net the anomalies' deferred-$
    # before rendering so the brief body — and the daily email built from it — match
    # the Morning Brief page's net "Deferred at Risk" KPI for the same barrels.
    base = digest_brief.render_brief_markdown(
        summary, _net_anomalies(anomalies, net_revenue_interest),
        brief_date=as_of, events=events)
    div = production_divergence_summary(fleet, anomalies)
    return base + "\n\n" + _divergence_section_md(div, price_per_bbl,
                                                  net_revenue_interest)


def _net_anomalies(anomalies: list, net_revenue_interest: float) -> list:
    """Copies of the digest anomalies with deferred_usd_per_day netted to operator
    (× NRI). Copies, not mutations — the scan list is cached and shared."""
    import copy
    out = []
    for a in anomalies:
        b = copy.copy(a)
        try:
            b.deferred_usd_per_day = round(
                float(getattr(a, "deferred_usd_per_day", 0.0) or 0.0)
                * net_revenue_interest, 0)
        except Exception:  # noqa: BLE001
            pass
        out.append(b)
    return out


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
    csv = (DIGEST_FLEET / f"{well_id}.csv").resolve()
    # Carry the well's real recent baseline rate (same trailing-7-day mean the triage
    # board uses) so the downstream ESP→AFE diagnosis sizes its incremental uplift on
    # the SAME number as the board — otherwise a non-flagged well's AFE collapses to
    # the 20-bopd floor while the board shows ~baseline, and the risked NPV (board)
    # can exceed the un-risked AFE NPV (an economic impossibility on the same well).
    baseline_bopd = _recent_baseline_bopd(csv)
    return {
        "well_id": well_id,
        "category": "fleet_scan",
        "severity": "—",
        "headline": "Not flagged by today's digest — surfaced by fleet risk scoring.",
        "deferred_bopd": 0.0,
        "baseline_bopd": round(baseline_bopd, 1),
        "scada_csv": str(csv),
        "date": _dt.date.today().isoformat(),
    }


def _recent_baseline_bopd(scada_csv) -> float:
    """Trailing-7-day mean oil rate for a well CSV — identical basis to rank_fleet's
    ``baseline_bopd`` so the chain and the board agree. 0.0 if unreadable."""
    try:
        import pandas as pd
        s = pd.read_csv(scada_csv, usecols=["bopd"])["bopd"]
        return float(s.tail(7).mean()) if len(s) else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def diagnose(alert: dict, model_path: Path | None = None) -> dict:
    """Stage 2: ESP scores the alert's well → AFE-ready WellDiagnosis. The alert's
    ``baseline_bopd`` (the well's real trailing rate, even for non-flagged wells) sizes
    the uplift on the SAME basis the Triage Board uses, so the Action Chain AFE and the
    board agree on the incremental rate."""
    return esp_handoff.diagnose(
        alert["scada_csv"], well_id=alert.get("well_id"),
        deferred_bopd=alert.get("deferred_bopd", 0.0),
        baseline_bopd=alert.get("baseline_bopd", 0.0),
        model_path=str(model_path or ESP_MODEL),
        lift=_lift_of(alert.get("well_id")))


def render_afe(diag: dict, working_interest: float = 1.0,
               net_revenue_interest: float = 0.80, realized_price: float = 70.0) -> str:
    """Stage 3: WellDiagnosis → deterministic AFE markdown."""
    return afe_handoff.render_afe_markdown(
        diag, working_interest=working_interest,
        net_revenue_interest=net_revenue_interest, realized_price=realized_price)


def afe_monte_carlo(diag: dict, realized_price: float = 70.0,
                    net_revenue_interest: float = 0.80,
                    n_trials: int = 10_000) -> dict | None:
    """Monte-Carlo NPV for the AFE's recommended intervention — the distributional
    view (P10/P50/P90 + probability-of-payout + a tornado) a capital review expects at
    sign-off instead of a single point. Reuses the AFE component's ``simulate_economics``
    engine; the gross results are netted to the operator (cost is certain at WI 1.0, so
    ``net(x) = NRI·(x+cost) − cost`` is exact on every percentile) which makes the
    net base NPV equal the AFE's deterministic Net NPV. Returns None if unpriced."""
    try:
        cost = float(afe_cost_db.cost_rollup(diag["intervention"])["total"])
    except Exception:  # noqa: BLE001
        return None
    mc = afe_economics.simulate_economics(
        cost, float(diag.get("incremental_rate_bopd", 0.0) or 0.0),
        uplift_decline_per_yr=float(diag.get("expected_uplift_decline_per_yr", 0.6)),
        realized_price_per_bbl=realized_price, n_trials=n_trials)
    nri = net_revenue_interest

    def _net(x: float) -> float:
        return nri * (x + cost) - cost

    return {
        "cost": cost, "n_trials": int(mc.n_trials),
        "p10": _net(mc.npv_p10_usd), "p50": _net(mc.npv_p50_usd),
        "p90": _net(mc.npv_p90_usd), "mean": _net(mc.npv_mean_usd),
        "base": _net(mc.base_npv_usd), "prob_payout": float(mc.probability_of_payout),
        "tornado": {k: {"low": _net(v["low"]), "high": _net(v["high"]),
                        "swing": nri * v["swing"]} for k, v in mc.tornado.items()},
    }


# Illustrative fixed lease operating expense per producing well-month (the carrying
# cost the economic-limit calc compares net revenue against). Stated, not hidden.
ECONOMIC_LIMIT_LOE_PER_MONTH = 9_000.0


def economic_limit(scada, realized_price: float = 70.0,
                   net_revenue_interest: float = 0.80, opex_per_bbl: float = 12.0,
                   loe_per_month: float = ECONOMIC_LIMIT_LOE_PER_MONTH) -> dict | None:
    """Economic limit + remaining producing life for a well: the oil rate at which net
    revenue equals fixed lease operating expense (the rate you'd P&A at), and the months
    from today's rate — declining at the well's own fitted exponential rate — to reach
    it. The number a PE defends in a reserves/abandonment review. None if not estimable."""
    import numpy as np
    try:
        oil = np.asarray(scada["bopd"], dtype=float)
    except Exception:  # noqa: BLE001
        return None
    oil = oil[np.isfinite(oil) & (oil > 0)]
    if len(oil) < 30:
        return None
    margin = realized_price * net_revenue_interest - opex_per_bbl   # net $/bbl
    if margin <= 0:
        return None
    q_limit = loe_per_month / (margin * 30.4)                       # bopd at the limit
    q_now = float(np.mean(oil[-30:]))
    t = np.arange(len(oil), dtype=float)
    fit_n = min(len(oil), 180)
    Di = float(np.polyfit(t[-fit_n:], np.log(oil[-fit_n:]), 1)[0])  # 1/day, <0 for a decliner
    d_monthly = -Di * 30.4
    if q_now <= q_limit:
        months = 0.0
    elif d_monthly <= 0:
        months = float("inf")                                       # flat/rising: no limit in sight
    else:
        months = float(np.log(q_now / q_limit) / d_monthly)
    return {"q_limit_bopd": q_limit, "q_now_bopd": q_now, "months_remaining": months,
            "net_margin_per_bbl": margin, "loe_per_month": loe_per_month,
            "annual_decline_pct": float((1 - np.exp(Di * 365)) * 100)}


def triage_scorecard(board) -> dict | None:
    """Score the Triage Board's RANKING against the fleet's known seeded faults —
    precision@k and lift-over-random — so the headline ranking carries a backtest like
    the digest's event detector and the deferment classifier already do. Ground truth is
    the generator's per-well signature assignment (``ground_truth.csv``); the ESP
    component's ``labels.csv`` is for a different fleet and does not join here."""
    import pandas as pd
    gt_path = APP_DIRS["digest"] / "data" / "synthetic" / "ground_truth.csv"
    if board is None or getattr(board, "empty", True) or not gt_path.exists():
        return None
    gt = pd.read_csv(gt_path)
    truth = {str(w): int(i) for w, i in zip(gt["well_id"], gt["impaired"])}
    n, n_pos = len(board), int(sum(truth.values()))
    if not n or not n_pos:
        return None
    ranked = (board.sort_values("opportunity_score", ascending=False)["well_id"]
              .astype(str).tolist())
    base_rate = n_pos / n
    at_k = {}
    for k in (5, 10, 20):
        kk = min(k, n)
        hits = sum(truth.get(w, 0) for w in ranked[:kk])
        prec = hits / kk
        at_k[k] = {"precision": prec, "hits": hits,
                   "lift": (prec / base_rate) if base_rate else 0.0}
    recall = sum(truth.get(w, 0) for w in ranked[:n_pos]) / n_pos
    return {"n_wells": n, "n_impaired": n_pos, "base_rate": base_rate,
            "at_k": at_k, "recall_at_n_impaired": recall}


def fit_well_decline(scada, fit_frac: float = 0.8) -> dict | None:
    """Fit an exponential decline to a well's oil on its ESTABLISHED trend (the first
    ``fit_frac`` of history) and extrapolate, so recent under-performance shows as a gap
    BELOW the curve — the per-well 'is this well on its type curve, or deferring?' read
    that the fleet-level chart only answers in aggregate. Returns the expected series
    aligned to the full history plus the trailing variance vs that curve. None if the
    history is too short / too gappy to fit."""
    import numpy as np
    try:
        oil = np.asarray(scada["bopd"], dtype=float)
        dates = scada["date"]
    except Exception:  # noqa: BLE001
        return None
    n = len(oil)
    if n < 30:
        return None
    t = np.arange(n, dtype=float)
    fit_n = max(30, int(n * fit_frac))
    pos = oil[:fit_n] > 0
    if int(pos.sum()) < 10:
        return None
    Di, lnqi = np.polyfit(t[:fit_n][pos], np.log(oil[:fit_n][pos]), 1)
    expected = np.exp(lnqi) * np.exp(Di * t)
    recent_act = float(np.nanmean(oil[-7:]))
    recent_exp = float(np.nanmean(expected[-7:]))
    var_pct = 100.0 * (recent_act - recent_exp) / recent_exp if recent_exp else 0.0
    return {"dates": dates, "expected": expected, "var_pct": var_pct,
            "implied_deferment_bopd": max(recent_exp - recent_act, 0.0),
            "annual_decline_pct": float((1 - np.exp(Di * 365)) * 100)}


def well_tiers(fleet: dict, board) -> dict:
    """Per-well health tier for the fleet map: 'down' | 'watch' | 'healthy'. A well is
    DOWN if at/near zero vs its baseline, WATCH if deferring production or in the
    fleet's own elevated-risk quartile (fleet-relative — the ESP score is OOD here),
    else HEALTHY."""
    import numpy as np
    has_board = board is not None and not getattr(board, "empty", True)
    risk = (dict(zip(board["well_id"].astype(str), board["failure_risk_30d"].astype(float)))
            if has_board else {})
    deferred = (dict(zip(board["well_id"].astype(str), board["deferred_bopd"].astype(float)))
                if has_board and "deferred_bopd" in board else {})
    thr = float(np.quantile(list(risk.values()), 0.75)) if risk else 1.0
    out: dict[str, str] = {}
    for wid, df in fleet.items():
        wid = str(wid)
        last, base = _latest_and_baseline(df)
        if _is_down(last, base):
            out[wid] = "down"
        elif deferred.get(wid, 0.0) > 0 or risk.get(wid, 0.0) >= thr:
            out[wid] = "watch"
        else:
            out[wid] = "healthy"
    return out


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


# Set by _score_fleet_risk: True when the ESP model couldn't score the fleet and
# every well fell back to BASELINE_RISK_30D. Views read risk_scoring_degraded() to
# surface a banner instead of silently showing a uniform 5% fleet (which a sharp PE
# would either be misled by or distrust).
_RISK_SCORING_DEGRADED = False


def risk_scoring_degraded() -> bool:
    """True if the most recent fleet risk scoring fell back to baseline (model
    unavailable / failed to load). Reflects the last actual scoring pass."""
    return _RISK_SCORING_DEGRADED


def _score_fleet_risk(fleet: dict, model_path: Path) -> dict[str, float]:
    """Score every well in ``fleet`` with the ESP model → {well_id: risk}.
    Degrades to an empty map (baseline risk fallback) if the model is unavailable."""
    global _RISK_SCORING_DEGRADED
    try:
        X = esp_features.featurize_fleet(fleet)
        model = esp_model.ESPRiskModel.load(str(model_path))
        probs = model.predict_proba(X)
        _RISK_SCORING_DEGRADED = False
        return {str(wid): float(p) for wid, p in zip(X.index, probs)}
    except Exception as e:  # noqa: BLE001  (missing model/deps → baseline fallback)
        _RISK_SCORING_DEGRADED = True
        _log.warning("ESP fleet risk scoring unavailable (%s: %s) — every well "
                     "falls back to baseline %.0f%% risk.",
                     type(e).__name__, e, BASELINE_RISK_30D * 100)
        return {}


# When the ESP model can't score a well, assign this baseline 30-day risk so the
# board is still fully populated and deterministic. Conservative (low) by design.
BASELINE_RISK_30D = 0.05

# No-action tier thresholds (identical to pipeline_core — tune here only).
_NO_ACTION_NPV_THRESHOLD = 10_000    # $ — below this, not worth flagging
_NO_ACTION_RISK_THRESHOLD = 0.15     # 30-day failure probability


def _lift_of(well_id: str) -> str | None:
    """The well's artificial-lift type from the shared fleet registry (None if the
    registry is unavailable — callers then fall back to the lift-agnostic mapping)."""
    try:
        import fleet_registry
        return fleet_registry.get(str(well_id)).lift
    except Exception:  # noqa: BLE001
        return None


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
        # Gate the priced intervention to one valid for the well's artificial-lift
        # type (no ESP swap on a rod-pumped well, no gas-lift optimization on a well
        # with no injection) — the recommendation a PE reads must be physical.
        intervention, frac = esp_handoff._map_mode(mode, _lift_of(well_id))

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
