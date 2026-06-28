"""Watermark + cooldown state — the memory that lets the watcher 'decide'.

We persist a small JSON blob: which note signatures we've already seen, when we
last ran, and when we last *alerted*. A note's signature is a hash of its
identity + mutable fields, so an edited note counts as new. The delta (current
signatures minus seen) is what the gate reasons over.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def sig(record) -> str:
    """Stable per-note signature: changes if any load-bearing field changes."""
    raw = "|".join(str(x) for x in (
        record.doc_id, record.well_id, record.start_date, record.end_date,
        record.duration_days, record.deferred_bbl, record.cause, record.note))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class WatchState:
    seen: set[str] = field(default_factory=set)
    last_run: str | None = None
    last_alert: str | None = None
    count: int = 0
    runs: int = 0

    # --- persistence ---------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "WatchState":
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return cls()
        return cls(
            seen=set(d.get("seen", [])),
            last_run=d.get("last_run"),
            last_alert=d.get("last_alert"),
            count=int(d.get("count", 0)),
            runs=int(d.get("runs", 0)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "seen": sorted(self.seen),
            "last_run": self.last_run,
            "last_alert": self.last_alert,
            "count": self.count,
            "runs": self.runs,
        }, indent=2))

    # --- the 'decide' helpers ------------------------------------------------
    @property
    def is_cold_start(self) -> bool:
        return self.last_run is None and not self.seen

    def delta(self, records) -> list:
        """Records whose signature we have not seen before."""
        return [r for r in records if sig(r) not in self.seen]

    def mark_seen(self, records) -> None:
        self.seen.update(sig(r) for r in records)

    def cooldown_active(self, hours: int, ref: datetime | None = None) -> bool:
        if not self.last_alert:
            return False
        ref = ref or now_utc()
        try:
            last = datetime.fromisoformat(self.last_alert)
        except ValueError:
            return False
        return (ref - last).total_seconds() < hours * 3600
