"""Pluggable historian adapters.

Production deployments don't read CSVs — they pull from PI / Ignition / OSIsoft /
SQL historians. This module defines a narrow `FleetSource` protocol (the contract
the rest of the pipeline depends on) and ships three concrete adapters that all
honor `SCADA_COLUMNS`:

- ``CsvFleetSource``        — the original per-well-CSV loader, refactored to the protocol
- ``CsvTimeRangeFleetSource`` — same CSVs, filtered to an inclusive [start, end] date range
- ``SQLiteFleetSource``     — a stdlib-only sqlite3 adapter (one ``readings`` table)

Swapping historians is a one-line change at the call site; everything downstream
(anomaly_detector, brief_writer) only sees ``dict[str, pd.DataFrame]``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from .data_loader import SCADA_COLUMNS, load_well


@runtime_checkable
class FleetSource(Protocol):
    """Any object that can hand the pipeline a fleet of per-well SCADA frames.

    Implementations must return ``{well_id: DataFrame}`` where every DataFrame
    has at least ``SCADA_COLUMNS``, is sorted ascending by ``date``, and has a
    reset index. The rest of the system depends only on this contract.
    """

    def load_fleet(self) -> dict[str, pd.DataFrame]:  # pragma: no cover - protocol
        ...


def _validate_frame(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Enforce the column contract and canonical ordering for one well frame."""
    missing = set(SCADA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")
    return df.sort_values("date").reset_index(drop=True)


class CsvFleetSource:
    """Original adapter: one ``well_*.csv`` per well in ``data_dir``."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load_fleet(self) -> dict[str, pd.DataFrame]:
        fleet: dict[str, pd.DataFrame] = {}
        for csv in sorted(self.data_dir.glob("well_*.csv")):
            fleet[csv.stem] = load_well(csv)
        return fleet


class CsvTimeRangeFleetSource:
    """Same per-well CSVs, but each frame is clipped to an inclusive date range.

    Mirrors how a historian query is bounded by a time window — e.g. "give me
    the last 14 days for the morning brief" — instead of loading full history.
    ``start``/``end`` may be ``None`` (open-ended) and are parsed with
    ``pd.Timestamp``.
    """

    def __init__(self, data_dir: str | Path, start=None, end=None):
        self.data_dir = Path(data_dir)
        self.start = pd.Timestamp(start) if start is not None else None
        self.end = pd.Timestamp(end) if end is not None else None

    def load_fleet(self) -> dict[str, pd.DataFrame]:
        fleet: dict[str, pd.DataFrame] = {}
        for csv in sorted(self.data_dir.glob("well_*.csv")):
            df = load_well(csv)
            if self.start is not None:
                df = df[df["date"] >= self.start]
            if self.end is not None:
                df = df[df["date"] <= self.end]
            fleet[csv.stem] = df.reset_index(drop=True)
        return fleet


class SQLiteFleetSource:
    """stdlib-only adapter over a SQLite historian.

    Expects a single table (default ``readings``) with a ``well_id`` column plus
    the ``SCADA_COLUMNS``. ``date`` is stored as ISO text and parsed back on read.
    Useful for demos/tests where a real PI/Ignition tag server isn't available,
    and as a worked example of the protocol against a SQL backend.

    ``SQLiteFleetSource.from_fleet(...)`` materializes a fleet dict into a
    throwaway DB so the round-trip is easy to exercise.
    """

    TABLE = "readings"

    def __init__(self, db_path: str | Path, table: str | None = None):
        self.db_path = str(db_path)
        table = table or self.TABLE
        # Guard the table identifier (it's interpolated into SQL, not bindable).
        if not table.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {table!r}")
        self.table = table

    def load_fleet(self) -> dict[str, pd.DataFrame]:
        cols = ", ".join(["well_id", *SCADA_COLUMNS])
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                f"SELECT {cols} FROM {self.table} ORDER BY well_id, date",
                conn,
                parse_dates=["date"],
            )
        finally:
            conn.close()
        fleet: dict[str, pd.DataFrame] = {}
        for well_id, grp in df.groupby("well_id"):
            frame = grp.drop(columns=["well_id"])
            fleet[str(well_id)] = _validate_frame(frame, f"{self.table}:{well_id}")
        return fleet

    @classmethod
    def from_fleet(cls, fleet: dict[str, pd.DataFrame], db_path: str | Path,
                   table: str | None = None) -> "SQLiteFleetSource":
        """Write a fleet dict into a SQLite DB and return a source pointing at it."""
        table = table or cls.TABLE
        rows = []
        for well_id, df in fleet.items():
            sub = df[SCADA_COLUMNS].copy()
            sub.insert(0, "well_id", well_id)
            rows.append(sub)
        all_rows = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["well_id", *SCADA_COLUMNS]
        )
        # Store full ISO datetime, not date-only — real historians (PI/Ignition) are
        # sub-daily, and truncating to midnight collapses intraday readings to one key.
        # ISO-8601 text remains lexicographically sortable for the ORDER BY date.
        all_rows["date"] = pd.to_datetime(all_rows["date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        conn = sqlite3.connect(str(db_path))
        try:
            all_rows.to_sql(table, conn, if_exists="replace", index=False)
        finally:
            conn.close()
        return cls(db_path, table=table)
