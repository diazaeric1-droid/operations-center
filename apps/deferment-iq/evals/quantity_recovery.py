"""Quantity-recovery eval — validate the recoverable-barrels accounting vs ground truth.

The reason-code eval (``run_evals.py``) checks the CLASSIFIER. This one checks the
deferment ENGINE's *quantity* math: given a synthetic fleet with KNOWN injected
downtime + underperformance per well (so true deferred / recoverable barrels are known),
run the engine and report:

  - error on TOTAL deferred bbl,
  - error on the DOWNTIME-vs-UNDERPERFORMANCE split,
  - recovery-quantity accuracy (engine recoverable opportunity vs the true recoverable).

It runs on BOTH a daily-cadence and a monthly-cadence representation of the IDENTICAL
physical fleet, to prove the engine is cadence-aware. Honest expectation, baked into the
tolerances:

  * DAILY data carries full sub-day resolution → the engine recovers ground truth almost
    exactly (total, the downtime/underperformance split, and recoverable all ~0% error).
  * MONTHLY data (real Colorado ECMC / NDIC grain) resolves DOWNTIME exactly (days-
    produced is an explicit field) but SMEARS short sub-month rate dips into the
    producing-day average, so underperformance is under-counted — an inherent limit of
    public monthly data, reported openly rather than hidden. The cadence-aware *volume*
    accounting is still proven correct: monthly downtime barrels match daily exactly.

Run: ``python -m evals.quantity_recovery`` (writes evals/results/quantity_summary.json).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from evals.quantity_fleet import build_events, build_truth_fleet
from src.analytics import recovery_opportunity
from src.deferment import classify_events, compute_deferment
from src.ndic import load_ndic_fleet

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evals" / "results"


def _pct_err(got: float, true: float) -> float:
    """Signed percent error; 0 when both ~0, inf-safe when only truth is ~0."""
    if abs(true) < 1e-9:
        return 0.0 if abs(got) < 1e-9 else float("inf")
    return (got - true) / true * 100.0


def _run_cadence(fleet: dict[str, pd.DataFrame], events: pd.DataFrame, truth) -> dict:
    evc = classify_events(events)
    daily = compute_deferment(fleet, evc, price_per_bbl=70.0)
    downtime = float(daily["downtime_def"].sum())
    underperf = float(daily["rate_def"].sum())
    total = float(daily["total_def"].sum())
    rec = recovery_opportunity(daily)
    recoverable = float(rec["recoverable_bbl"])
    return {
        "n_records": int(len(daily)),
        "engine": {
            "total_deferred_bbl": total,
            "downtime_bbl": downtime,
            "underperf_bbl": underperf,
            "recoverable_bbl": recoverable,
        },
        "truth": {
            "total_deferred_bbl": truth.total_deferred_bbl,
            "downtime_bbl": truth.total_downtime_bbl,
            "underperf_bbl": truth.total_underperf_bbl,
            "recoverable_bbl": truth.total_recoverable_bbl,
        },
        "err_pct": {
            "total_deferred": _pct_err(total, truth.total_deferred_bbl),
            "downtime": _pct_err(downtime, truth.total_downtime_bbl),
            "underperf": _pct_err(underperf, truth.total_underperf_bbl),
            "recoverable": _pct_err(recoverable, truth.total_recoverable_bbl),
        },
    }


def evaluate(n_wells: int = 24, n_months: int = 8, seed: int = 7) -> dict:
    """Run the engine on the ground-truth fleet at both cadences; return the report dict."""
    daily_fleet, monthly_df, truth = build_truth_fleet(
        n_wells=n_wells, n_months=n_months, seed=seed)
    events = build_events(truth, daily_fleet)

    daily_res = _run_cadence(daily_fleet, events, truth)

    # Monthly: materialize the tidy NDIC/ECMC CSV and load via the real adapter.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        monthly_df.to_csv(f.name, index=False)
        monthly_path = f.name
    monthly_fleet = load_ndic_fleet(monthly_path)
    monthly_res = _run_cadence(monthly_fleet, events, truth)

    return {
        "n_wells": n_wells,
        "n_months": n_months,
        "seed": seed,
        "daily": daily_res,
        "monthly": monthly_res,
        # Cadence consistency: monthly downtime barrels should equal daily's (the part
        # monthly data can resolve), proving the time-based volume accounting.
        "downtime_cadence_gap_pct": _pct_err(
            monthly_res["engine"]["downtime_bbl"], daily_res["engine"]["downtime_bbl"]),
    }


# Tolerance gates (CI fails if exceeded). Cadence-appropriate and honest:
#   DAILY   — full resolution, accounting must be ~exact on everything.
#   MONTHLY — downtime is explicit (tight), total is bounded but loose because short
#             sub-month rate dips are smeared by the producing-day average (documented).
DAILY_TOTAL_TOL = 1.0          # % — daily total deferred bbl (engine is exact at daily)
DAILY_SPLIT_TOL = 2.0          # % — daily downtime & underperformance
DAILY_RECOVER_TOL = 1.0        # % — daily recoverable opportunity
MONTHLY_DOWNTIME_TOL = 2.0     # % — monthly downtime bbl (days-produced is exact)
# Monthly total under-counts the smeared sub-month underperformance. Across 20 seeds the
# observed error is ~24–33%; 40% leaves a sane margin while still catching a real
# regression (e.g. the old row-count bug, which made monthly off by orders of magnitude).
MONTHLY_TOTAL_TOL = 40.0       # %
CADENCE_DOWNTIME_TOL = 1.0     # % — monthly vs daily downtime barrels must agree (proves cadence-awareness)


def gate(res: dict) -> list[str]:
    """Return a list of tolerance violations (empty == pass)."""
    fails: list[str] = []
    d, m = res["daily"]["err_pct"], res["monthly"]["err_pct"]

    def chk(name, val, tol):
        if abs(val) > tol:
            fails.append(f"{name}: |{val:.1f}%| > {tol:.1f}%")

    chk("daily.total_deferred", d["total_deferred"], DAILY_TOTAL_TOL)
    chk("daily.downtime", d["downtime"], DAILY_SPLIT_TOL)
    chk("daily.underperf", d["underperf"], DAILY_SPLIT_TOL)
    chk("daily.recoverable", d["recoverable"], DAILY_RECOVER_TOL)
    chk("monthly.downtime", m["downtime"], MONTHLY_DOWNTIME_TOL)
    chk("monthly.total_deferred", m["total_deferred"], MONTHLY_TOTAL_TOL)
    chk("downtime_cadence_gap", res["downtime_cadence_gap_pct"], CADENCE_DOWNTIME_TOL)
    return fails


def _fmt(res: dict) -> str:
    lines = []
    for cad in ("daily", "monthly"):
        r = res[cad]
        e, t = r["engine"], r["truth"]
        ep = r["err_pct"]
        lines.append(f"  {cad.upper():<8} ({r['n_records']} records)")
        lines.append(f"    total deferred : {e['total_deferred_bbl']:>10,.0f} bbl  "
                     f"(truth {t['total_deferred_bbl']:>10,.0f}, err {ep['total_deferred']:+.1f}%)")
        lines.append(f"    downtime       : {e['downtime_bbl']:>10,.0f} bbl  "
                     f"(truth {t['downtime_bbl']:>10,.0f}, err {ep['downtime']:+.1f}%)")
        lines.append(f"    underperf      : {e['underperf_bbl']:>10,.0f} bbl  "
                     f"(truth {t['underperf_bbl']:>10,.0f}, err {ep['underperf']:+.1f}%)")
        lines.append(f"    recoverable    : {e['recoverable_bbl']:>10,.0f} bbl  "
                     f"(truth {t['recoverable_bbl']:>10,.0f}, err {ep['recoverable']:+.1f}%)")
    lines.append(f"  downtime cadence gap (monthly vs daily): "
                 f"{res['downtime_cadence_gap_pct']:+.2f}%")
    return "\n".join(lines)


def main():
    res = evaluate()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "quantity_summary.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"Quantity-recovery eval — {res['n_wells']} wells × {res['n_months']} months, "
          f"both cadences (seed {res['seed']})")
    print(_fmt(res))
    print(f"\nWrote {OUT / 'quantity_summary.json'}")
    fails = gate(res)
    if fails:
        raise SystemExit("Quantity eval gate FAILED:\n  - " + "\n  - ".join(fails))
    print("Gate PASSED (all errors within cadence-appropriate tolerance).")


if __name__ == "__main__":
    main()
