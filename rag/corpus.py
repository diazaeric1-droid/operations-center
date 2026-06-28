"""Operator-note corpus: the documents the RAG index is built over.

Two sources, unioned:

  1. The real synthetic events file the deferment app already ships
     (``apps/deferment-iq/data/synthetic/events.csv``) — carries genuine
     well_ids, notes, and a ``true_cause`` label.
  2. A larger generated corpus of realistic field notes so semantic search has
     enough volume to be meaningful (24 notes don't show off retrieval; a few
     hundred do). Generation is seeded -> reproducible -> testable.

Output is plain Python (``NoteRecord``) so this module is unit-testable with
**no** heavy dependencies. ``records_to_documents`` converts to LlamaIndex
``Document`` objects lazily, only when llama-index is installed.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EVENTS_CSV = (_HERE.parent / "apps" / "deferment-iq" / "data" / "synthetic"
               / "events.csv")

# Reason keys mirror apps/deferment-iq/src/reason_codes.py (kept as plain
# strings here so corpus generation has zero import coupling to the vendored
# app). The RAG layer never *classifies* — it retrieves — but carrying the
# cause as metadata lets the engine answer "show me only power events", etc.
RECOVERABLE = {
    "artificial_lift", "surface_facility", "power", "gathering_thirdparty",
    "wellbore", "weather",
}


@dataclass(frozen=True)
class NoteRecord:
    """One operator note + the structured fields a pumper logs alongside it."""
    doc_id: str
    well_id: str
    start_date: str          # ISO date
    end_date: str            # ISO date
    duration_days: int
    deferred_bbl: int
    cause: str               # reason key (ground-truth for synthetic rows)
    note: str
    source: str              # "events_csv" | "synthetic"

    @property
    def recoverable(self) -> bool:
        return self.cause in RECOVERABLE

    def to_text(self) -> str:
        """The text that gets embedded + retrieved.

        Front-loads the free-text note (what a query is usually *about*) and
        appends a compact structured tail so retrieval can also lock onto
        well/cause/duration phrasing in a question.
        """
        rec = "recoverable" if self.recoverable else "non-recoverable"
        return (
            f"{self.note}. "
            f"Well {self.well_id}; cause {self.cause} ({rec}); "
            f"{self.duration_days}-day outage starting {self.start_date}; "
            f"~{self.deferred_bbl} bbl deferred."
        )

    def metadata(self) -> dict:
        d = asdict(self)
        d["recoverable"] = self.recoverable
        return d


# --- synthetic generator ------------------------------------------------------
# Realistic note templates per cause. Variety in equipment, fault detail, and
# field response is deliberate: it forces retrieval to lean on meaning, not just
# a shared keyword (the whole reason RAG beats the keyword classifier here).
_TEMPLATES: dict[str, tuple[str, ...]] = {
    "artificial_lift": (
        "ESP tripped on underload, VSD fault {fault}, waiting on electrician",
        "rod string parted near {depth} ft, pulling unit scheduled",
        "pump off, low fillage, tuning the POC setpoints",
        "downhole pump worn, no fluid over pump, candidate for workover",
        "gas lift unstable, well heading, adjusting injection rate",
        "ESP overload trips repeating after restart, suspect gas interference",
        "plunger not cycling, dropped and stuck, slickline called out",
        "gearbox noise on the pumping unit, derated until inspection",
    ),
    "surface_facility": (
        "compressor down, no gas takeaway off the pad",
        "heater treater tripped on high level, emulsion upset",
        "separator dump valve hung, high level shut-in",
        "LACT unit fault, can't move oil, tank battery filling",
        "VRU down, flaring, curtailed to stay under permit",
        "treater burner flameout overnight, cold and slugging",
    ),
    "power": (
        "transformer failure, utility en route",
        "lost power to the pad, substation breaker open",
        "lightning strike took out the genset, running on backup",
        "grid outage area-wide, no ETA from co-op",
        "recurring breaker trips at the VSD, electrician investigating",
    ),
    "gathering_thirdparty": (
        "gas plant down, midstream curtailment",
        "high line pressure, backed out, sales line restricted",
        "third party pipeline maintenance, nomination cut",
        "gathering system slugging, intermittent high backpressure",
        "plant upset downstream, takeaway curtailed to 60 percent",
    ),
    "wellbore": (
        "scale buildup restricting flow, scheduling a cleanout",
        "paraffin/wax cut in tubing, hot oil truck ordered",
        "sand production, choked back to protect equipment",
        "hydrate forming in the flowline, methanol injection started",
        "hole in tubing suspected, fluid level not building",
    ),
    "planned": (
        "planned workover, rig moving in",
        "scheduled well test, off line for the day",
        "routine PM on the unit, locked out per procedure",
        "wireline run for the completion, shut in",
        "turnaround on the facility, planned outage",
    ),
    "weather": (
        "freeze off overnight, lines frozen, thawing rigs out",
        "winter storm, froze the location, crews unable to access",
        "cold front, instrument air froze, controls down",
        "ice on the pad, deferred access for safety",
    ),
    "reservoir": (
        "pressure depletion, declining inflow",
        "watering out, water cut climbing, lift struggling",
        "liquid loading, well loaded up and died",
        "GOR rising, reservoir energy fading",
    ),
}

_FAULTS = ("F012", "F031", "F007", "F045", "F101", "underload", "overload")
_DEPTHS = ("5200", "6400", "7100", "8300", "4800")


def _well_pool(seed_records: list[NoteRecord]) -> list[str]:
    """Wells to attach synthetic notes to — reuse the real fleet ids so the
    corpus stays coherent with the rest of the product, padded if sparse."""
    real = sorted({r.well_id for r in seed_records})
    pool = list(real)
    i = 1
    while len(pool) < 12:
        cand = f"PERM-{i:04d}"
        if cand not in pool:
            pool.append(cand)
        i += 1
    return pool


def _load_events_csv() -> list[NoteRecord]:
    if not _EVENTS_CSV.exists():
        return []
    out: list[NoteRecord] = []
    with _EVENTS_CSV.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            note = (row.get("note") or "").strip()
            if not note:
                continue
            sd = (row.get("start_date") or "2024-01-01").strip()
            ed = (row.get("end_date") or sd).strip()
            dur = _days_between(sd, ed)
            out.append(NoteRecord(
                doc_id=f"evt-{i:03d}",
                well_id=(row.get("well_id") or "UNKNOWN").strip(),
                start_date=sd, end_date=ed, duration_days=dur,
                deferred_bbl=max(dur, 1) * 95,
                cause=(row.get("true_cause") or "unclassified").strip(),
                note=note, source="events_csv"))
    return out


def _days_between(a: str, b: str) -> int:
    try:
        da = date.fromisoformat(a)
        db = date.fromisoformat(b)
        return max((db - da).days, 1)
    except Exception:  # noqa: BLE001
        return 1


def build_note_records(n_synthetic: int = 360, seed: int = 7) -> list[NoteRecord]:
    """Real events.csv rows + ``n_synthetic`` generated notes (seeded)."""
    seed_records = _load_events_csv()
    rng = random.Random(seed)
    wells = _well_pool(seed_records)
    causes = list(_TEMPLATES)
    base = date(2024, 1, 1)

    synth: list[NoteRecord] = []
    for i in range(n_synthetic):
        cause = rng.choice(causes)
        tmpl = rng.choice(_TEMPLATES[cause])
        note = tmpl.format(fault=rng.choice(_FAULTS), depth=rng.choice(_DEPTHS))
        start = base + timedelta(days=rng.randint(0, 540))
        # weather skews short, planned/AL skew longer — gives duration queries teeth
        dur = {
            "weather": rng.randint(1, 3),
            "planned": rng.randint(1, 6),
            "artificial_lift": rng.randint(1, 14),
        }.get(cause, rng.randint(1, 9))
        end = start + timedelta(days=dur)
        synth.append(NoteRecord(
            doc_id=f"syn-{i:04d}",
            well_id=rng.choice(wells),
            start_date=start.isoformat(), end_date=end.isoformat(),
            duration_days=dur, deferred_bbl=dur * rng.randint(60, 180),
            cause=cause, note=note, source="synthetic"))
    return seed_records + synth


def records_to_documents(records: list[NoteRecord]):
    """Convert to LlamaIndex ``Document`` objects (lazy import).

    Raises a clear ImportError if the RAG extras aren't installed.
    """
    try:
        from llama_index.core import Document
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "llama-index is not installed. Install the RAG extras:\n"
            "    pip install -r requirements-rag.txt") from e
    return [
        Document(text=r.to_text(), doc_id=r.doc_id, metadata=r.metadata(),
                 excluded_embed_metadata_keys=["doc_id", "deferred_bbl"],
                 excluded_llm_metadata_keys=["doc_id"])
        for r in records
    ]


if __name__ == "__main__":  # quick smoke
    recs = build_note_records()
    print(f"{len(recs)} note records "
          f"({sum(r.source == 'events_csv' for r in recs)} real, "
          f"{sum(r.source == 'synthetic' for r in recs)} synthetic)")
    for r in recs[:3]:
        print(" •", r.to_text())
