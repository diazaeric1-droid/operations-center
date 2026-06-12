"""Pipeline stage-1 handoff: digest → ESP WellAlerts."""
from pathlib import Path

from src.handoff import ALERT_SCHEMA, ESP_RELATED, export_alerts

FLEET = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "fleet"


def test_export_alerts_only_forwards_esp_signatures():
    alerts = export_alerts(FLEET, ack_path="does_not_exist.yml")
    assert alerts, "seeded fleet should yield ESP-related alerts (e.g. intake collapse)"
    assert all(a["schema"] == ALERT_SCHEMA for a in alerts)
    assert all(a["category"] in ESP_RELATED for a in alerts)
    # rate drops are reservoir-ambiguous and must NOT be forwarded to ESP scoring
    assert all("rate_drop" not in a["category"] for a in alerts)


def test_alert_points_at_the_right_scada_file():
    alerts = export_alerts(FLEET, ack_path="does_not_exist.yml")
    a = alerts[0]
    assert Path(a["scada_csv"]).name == f"{a['well_id']}.csv"
    assert Path(a["scada_csv"]).exists()
