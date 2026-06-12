"""Load fleet production time series + the downtime/curtailment event log.

Production deployments would replace this with a pull from the production database
/ allocation system (Quorum, P2/Enverus, OFM, PHDWin) + the downtime reason-code
log; the CSV contract here mirrors that shape.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROD_COLUMNS = ["date", "bopd", "bfpd", "gas_mcfd", "runtime_pct"]
EVENT_COLUMNS = ["well_id", "start_date", "end_date", "note"]


def load_well(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(PROD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{Path(path).name}: missing production columns {missing}")
    return df.sort_values("date").reset_index(drop=True)


def load_fleet(wells_dir: str | Path) -> dict[str, pd.DataFrame]:
    wells_dir = Path(wells_dir)
    fleet = {}
    for csv in sorted(wells_dir.glob("well_*.csv")):
        fleet[csv.stem] = load_well(csv)
    return fleet


def load_events(path: str | Path) -> pd.DataFrame:
    """Downtime/curtailment events. Columns: well_id, start_date, end_date, note.
    An optional ``true_cause`` column (ground truth) is preserved if present but is
    never used by the classifier — only by the eval harness."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    df = pd.read_csv(p, parse_dates=["start_date", "end_date"])
    missing = set(EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"events file missing columns {missing}")
    return df
