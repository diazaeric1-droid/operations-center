"""AFE pipeline tracker — minimal SQLite-backed state machine for in-flight AFEs.

Designed to demonstrate pipeline visibility without a real ERP integration.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd


Status = Literal["draft", "engineering_review", "finance_review", "approved", "executed", "rejected"]
STATUS_ORDER = ["draft", "engineering_review", "finance_review", "approved", "executed"]
IN_FLIGHT_STATUSES = ("draft", "engineering_review", "finance_review")

# Typical days per stage (used for bottleneck prediction)
STAGE_SLA_DAYS = {
    "draft": 2,
    "engineering_review": 5,
    "finance_review": 8,
    "approved": 3,
    "executed": None,
}

# Delegation-of-authority limits — the approver whose sign-off the AFE's $ value
# requires. This is the "I've actually shepherded capital through approval" signal:
# a $58k workover stops at the Engineering Manager; a $365k ESP swap needs the
# Ops Manager; anything over $1MM goes to the VP. (Ordered low→high.)
AUTHORITY_LIMITS = [
    (50_000, "Production Engineer"),
    (250_000, "Engineering Manager"),
    (1_000_000, "Operations Manager"),
    (float("inf"), "VP / Asset Manager"),
]


def required_approver(total_cost_usd: float) -> str:
    """Lowest authority level whose limit covers this AFE's cost."""
    for limit, role in AUTHORITY_LIMITS:
        if total_cost_usd <= limit:
            return role
    return AUTHORITY_LIMITS[-1][1]


@dataclass
class AFERecord:
    afe_number: str
    well_id: str
    intervention: str
    total_cost_usd: float
    status: Status
    created_date: str
    last_updated: str
    rig_name: str | None = None
    requested_by: str | None = None
    notes: str | None = None


class AFETracker:
    def __init__(self, db_path: str | Path = "pipeline.sqlite"):
        self.db_path = Path(db_path)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS afes (
                    afe_number TEXT PRIMARY KEY,
                    well_id TEXT NOT NULL,
                    intervention TEXT NOT NULL,
                    total_cost_usd REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_date TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    rig_name TEXT,
                    requested_by TEXT,
                    notes TEXT
                )
            """)
            # Immutable audit log — every status change is appended, never overwritten.
            # This is what an internal-audit / SOX reviewer expects of a capital tracker.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS afe_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    afe_number TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    actor TEXT,
                    note TEXT
                )
            """)

    def _log_event(self, conn, afe_number, from_status, to_status, actor=None, note=None):
        conn.execute(
            "INSERT INTO afe_events (afe_number, ts, from_status, to_status, actor, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (afe_number, datetime.now().isoformat(timespec="seconds"),
             from_status, to_status, actor, note),
        )

    def upsert(self, rec: AFERecord) -> None:
        with self._conn() as conn:
            prior = conn.execute(
                "SELECT status FROM afes WHERE afe_number = ?", (rec.afe_number,)
            ).fetchone()
            conn.execute("""
                INSERT INTO afes (afe_number, well_id, intervention, total_cost_usd,
                                  status, created_date, last_updated, rig_name, requested_by, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(afe_number) DO UPDATE SET
                    well_id=excluded.well_id,
                    intervention=excluded.intervention,
                    status=excluded.status,
                    total_cost_usd=excluded.total_cost_usd,
                    last_updated=excluded.last_updated,
                    rig_name=excluded.rig_name,
                    requested_by=excluded.requested_by,
                    notes=excluded.notes
            """, (rec.afe_number, rec.well_id, rec.intervention, rec.total_cost_usd,
                  rec.status, rec.created_date, rec.last_updated,
                  rec.rig_name, rec.requested_by, rec.notes))
            if prior is None:
                self._log_event(conn, rec.afe_number, None, rec.status,
                                actor=rec.requested_by, note="created")
            elif prior["status"] != rec.status:
                self._log_event(conn, rec.afe_number, prior["status"], rec.status,
                                actor=rec.requested_by, note=rec.notes)

    def advance(self, afe_number: str, to_status: Status, note: str | None = None,
                actor: str | None = None) -> None:
        with self._conn() as conn:
            prior = conn.execute(
                "SELECT status FROM afes WHERE afe_number = ?", (afe_number,)
            ).fetchone()
            conn.execute("""
                UPDATE afes SET status = ?, last_updated = ?, notes = COALESCE(?, notes)
                WHERE afe_number = ?
            """, (to_status, date.today().isoformat(), note, afe_number))
            self._log_event(conn, afe_number,
                            prior["status"] if prior else None, to_status, actor, note)

    def events(self, afe_number: str | None = None) -> pd.DataFrame:
        """Return the audit trail (optionally for one AFE), newest first."""
        with self._conn() as conn:
            if afe_number:
                return pd.read_sql(
                    "SELECT * FROM afe_events WHERE afe_number = ? ORDER BY id DESC",
                    conn, params=(afe_number,))
            return pd.read_sql("SELECT * FROM afe_events ORDER BY id DESC", conn)

    def as_dataframe(self) -> pd.DataFrame:
        with self._conn() as conn:
            df = pd.read_sql("SELECT * FROM afes ORDER BY created_date DESC", conn)
        if df.empty:
            return df
        df["last_updated"] = pd.to_datetime(df["last_updated"])
        df["created_date"] = pd.to_datetime(df["created_date"])
        df["days_in_status"] = (pd.Timestamp.now().normalize() - df["last_updated"]).dt.days
        df["days_open"] = (pd.Timestamp.now().normalize() - df["created_date"]).dt.days
        df["bottleneck_risk"] = df.apply(self._risk, axis=1)
        df["required_approver"] = df["total_cost_usd"].apply(required_approver)
        return df

    @staticmethod
    def _risk(row) -> str:
        sla = STAGE_SLA_DAYS.get(row["status"])
        if sla is None or row["status"] in ("executed", "approved"):
            return "—"
        if row["days_in_status"] > sla * 1.5:
            return "HIGH"
        if row["days_in_status"] > sla:
            return "MEDIUM"
        return "LOW"


def seed_demo_data(db_path: str | Path = "pipeline.sqlite") -> None:
    """Populate the tracker with 12 fake AFEs spanning the status pipeline."""
    tracker = AFETracker(db_path)
    today = date.today()

    rows = [
        ("AFE-2026-0042", "ED-001H", "acid_stimulation",       210_000, "engineering_review",  9, "Rig 03"),
        ("AFE-2026-0043", "ED-002H", "esp_swap",               340_000, "finance_review",     15, "Rig 07"),
        ("AFE-2026-0044", "ED-005H", "gas_lift_optimization",  135_000, "draft",                1, "Rig 02"),
        ("AFE-2026-0045", "ED-008H", "scale_treatment",         92_000, "approved",             3, "Rig 03"),
        ("AFE-2026-0046", "ED-012H", "esp_to_beam_conversion", 305_000, "engineering_review",  12, "Rig 09"),
        ("AFE-2026-0047", "ED-014H", "rod_pump_workover",       58_000, "executed",            21, "Rig 11"),
        ("AFE-2026-0048", "ED-017H", "gas_lift_optimization",   22_000, "finance_review",       4, "Rig 06"),
        ("AFE-2026-0049", "ED-019H", "paraffin_treatment",      17_000, "approved",             1, "Rig 04"),
        ("AFE-2026-0050", "ED-020H", "p_and_a",                238_000, "draft",                5, "Rig 11"),
        ("AFE-2026-0051", "ED-022H", "acid_stimulation",       195_000, "executed",            14, "Rig 03"),
        ("AFE-2026-0052", "ED-024H", "esp_swap",               365_000, "rejected",            10, "Rig 09"),
        ("AFE-2026-0053", "ED-025H", "scale_treatment",         88_000, "engineering_review",   2, "Rig 02"),
    ]
    for afe_no, well, interv, cost, status, days_old, rig in rows:
        created = (today - timedelta(days=days_old + 2)).isoformat()
        updated = (today - timedelta(days=days_old)).isoformat()
        tracker.upsert(AFERecord(
            afe_number=afe_no, well_id=well, intervention=interv,
            total_cost_usd=cost, status=status,
            created_date=created, last_updated=updated,
            rig_name=rig, requested_by="Senior PE",
        ))


if __name__ == "__main__":
    seed_demo_data()
    print("Seeded 12 AFEs into pipeline.sqlite")
    df = AFETracker().as_dataframe()
    print(df.to_string(index=False))
