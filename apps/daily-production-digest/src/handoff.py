"""Pipeline handoff (stage 1 of 3): export detected anomalies as ``WellAlert``
artifacts for the ESP Failure-Risk Agent.

This is the first hop of the detect → predict → authorize pipeline:

    Daily Production Digest  ──WellAlert──▶  ESP Failure-Risk Agent  ──WellDiagnosis──▶  AFE Copilot

Only anomalies that implicate the artificial-lift system (ESP) are worth a 30-day
failure-risk score — a pure reservoir rate drop is not. The alert carries the
absolute path to the well's SCADA CSV so the next stage can score the SAME well.

WellAlert schema (``pe-pipeline/well-alert/v1``):
    well_id, category, severity, headline, deferred_bopd, baseline_bopd, scada_csv, date
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .anomaly_detector import DEFAULT_OIL_PRICE, load_acknowledgements, scan_fleet
from .data_loader import load_fleet

ALERT_SCHEMA = "pe-pipeline/well-alert/v1"

# Anomaly categories that are genuine ESP / artificial-lift mechanical signatures
# (worth a 30-day failure-risk score). Rate drops are deliberately excluded — a
# rate drop is reservoir-ambiguous and routes to a reservoir/surface review, not
# to ESP failure prediction; data-quality flags aren't well problems at all.
ESP_RELATED = {
    "intake_collapse", "amps_creep", "motor_temp_spike", "runtime_degradation",
}


def export_alerts(data_dir, price_per_bbl: float = DEFAULT_OIL_PRICE,
                  ack_path="acknowledged.yml", brief_date: str | None = None) -> list[dict]:
    """Scan the fleet and return ESP-related WellAlert dicts, ranked by the same
    money-first order scan_fleet uses (acknowledged events excluded)."""
    data_dir = Path(data_dir)
    fleet = load_fleet(data_dir)
    acknowledged = load_acknowledgements(ack_path)
    anomalies = scan_fleet(fleet, price_per_bbl=price_per_bbl, acknowledged=acknowledged)
    brief_date = brief_date or date.today().isoformat()

    alerts: list[dict] = []
    for a in anomalies:
        if a.acknowledged or a.category not in ESP_RELATED:
            continue
        scada = fleet.get(a.well_id)
        baseline_bopd = (float(scada["bopd"].tail(7).mean())
                         if scada is not None and "bopd" in scada.columns and len(scada) else 0.0)
        alerts.append({
            "schema": ALERT_SCHEMA,
            "well_id": a.well_id,
            "category": a.category,
            "severity": a.severity,
            "headline": a.headline,
            "deferred_bopd": round(a.deferred_bopd, 1),
            "baseline_bopd": round(baseline_bopd, 1),
            "scada_csv": str((data_dir / f"{a.well_id}.csv").resolve()),
            "date": brief_date,
        })
    return alerts


def main():
    parser = argparse.ArgumentParser(description="Export ESP-related WellAlerts for the pipeline.")
    parser.add_argument("--data-dir", default="data/synthetic/fleet")
    parser.add_argument("--ack", default="acknowledged.yml")
    parser.add_argument("--price", type=float, default=DEFAULT_OIL_PRICE)
    parser.add_argument("--out", default=None, help="Write JSON list here (default: stdout)")
    args = parser.parse_args()

    alerts = export_alerts(args.data_dir, price_per_bbl=args.price, ack_path=args.ack)
    payload = json.dumps(alerts, indent=2)
    if args.out:
        Path(args.out).write_text(payload)
        print(f"Wrote {len(alerts)} ESP-related alert(s) to {args.out}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
