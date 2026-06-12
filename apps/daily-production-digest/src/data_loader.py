"""Load fleet SCADA from per-well CSV files. Production deployments would replace
this with a connector to PI / Ignition / OSIsoft / SQL data historians."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


SCADA_COLUMNS = ["date", "bopd", "bfpd", "gas_mcfd", "intake_pressure_psi", "motor_temp_f", "motor_amps", "runtime_pct"]

# Bring-your-own-data: a single uploaded CSV carries the WHOLE fleet, so it needs a
# ``well_id`` column to split rows into per-well frames (the on-disk synthetic fleet
# encodes the id in the filename instead). Everything else is the standard SCADA
# channel set the detectors + brief already consume.
BYOD_REQUIRED_COLUMNS = ["well_id", *SCADA_COLUMNS]


def validate_scada_columns(df: pd.DataFrame) -> list[str]:
    """Return the BYOD-required columns missing from ``df`` (empty list == valid).

    Small, Streamlit-free helper so an uploaded CSV can be checked up front and a
    clear error shown before anything tries to parse it. Required columns are
    ``well_id`` + the SCADA channels in :data:`BYOD_REQUIRED_COLUMNS`; order does
    not matter (set difference), extra columns are ignored. Missing columns are
    returned in canonical order so the error message reads consistently."""
    if df is None:
        return list(BYOD_REQUIRED_COLUMNS)
    have = set(df.columns)
    return [c for c in BYOD_REQUIRED_COLUMNS if c not in have]


def load_well(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(SCADA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name if isinstance(path, Path) else path}: missing columns {missing}")
    return df.sort_values("date").reset_index(drop=True)


def load_fleet(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    fleet = {}
    for csv in sorted(data_dir.glob("well_*.csv")):
        fleet[csv.stem] = load_well(csv)
    return fleet


def load_fleet_from_csv(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load a fleet from a SINGLE uploaded CSV that carries a ``well_id`` column.

    This is the bring-your-own-data path: one file holds every well's daily SCADA.
    Validates the columns up front (raises ``ValueError`` listing what's missing),
    then splits by ``well_id`` and runs each well's rows through the EXACT same
    on-disk loader the synthetic source uses (``load_well`` via a per-well temp
    CSV) — same date parsing, sort, and schema check — so detection / scan / brief
    all see identically-shaped frames regardless of source. Well ids are coerced to
    ``str`` to match the synthetic ``well_NNN`` keys.
    """
    df = pd.read_csv(path)
    missing = validate_scada_columns(df)
    if missing:
        raise ValueError(f"uploaded CSV missing required columns: {missing}")

    fleet: dict[str, pd.DataFrame] = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for raw_id, group in df.groupby("well_id", sort=True):
            well_id = str(raw_id)
            tmp = Path(tmpdir) / "well.csv"
            # Drop the id column so the per-well frame matches the on-disk schema
            # (SCADA_COLUMNS only), then reuse load_well for parse + sort + validate.
            group.drop(columns=["well_id"]).to_csv(tmp, index=False)
            fleet[well_id] = load_well(tmp)
    return fleet


def fleet_template_csv() -> str:
    """A ready-to-fill BYOD template (header + two example daily rows for one well).

    Reuses :data:`BYOD_REQUIRED_COLUMNS` so the template can never drift from what
    :func:`load_fleet_from_csv` requires — the download users get is exactly the
    schema the loader validates."""
    header = ",".join(BYOD_REQUIRED_COLUMNS)
    rows = [
        "WELL_A,2026-01-01,220,1800,270,120,290,60,99",
        "WELL_A,2026-01-02,218,1810,268,121,290,60,99",
    ]
    return "\n".join([header, *rows]) + "\n"


def fleet_summary(fleet: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Aggregate fleet-wide stats from the most recent day per well."""
    # Skip empty frames (e.g. a shut-in / brand-new well a historian returns with
    # no rows) instead of raising IndexError on .iloc[-1].
    latest_rows = [df.iloc[-1] for df in fleet.values() if len(df)]
    total_bopd = sum(r["bopd"] for r in latest_rows)
    total_bfpd = sum(r["bfpd"] for r in latest_rows)
    avg_runtime = sum(r["runtime_pct"] for r in latest_rows) / max(len(latest_rows), 1)
    total_gas = sum(r.get("gas_mcfd", 0.0) for r in latest_rows) if latest_rows else 0.0
    return {
        "well_count": len(fleet),
        "total_bopd": float(total_bopd),
        "total_bfpd": float(total_bfpd),
        "total_gas_mcfd": float(total_gas),
        "avg_runtime_pct": float(avg_runtime),
        "water_cut_pct": float((total_bfpd - total_bopd) / total_bfpd * 100) if total_bfpd > 0 else 0.0,
    }


def slice_window(df: pd.DataFrame, window_days: int | None) -> pd.DataFrame:
    """Trailing ``window_days`` rows of a per-well SCADA frame (already date-sorted
    by ``load_well``). ``None`` / non-positive → the full history (Lifetime)."""
    if df is None or not len(df):
        return df
    if not window_days or window_days <= 0 or window_days >= len(df):
        return df
    return df.iloc[-int(window_days):]


def production_variance_pct(values, edge_days: int = 7) -> float:
    """Percent change in a production series over a window: recent ``edge_days``
    average vs. the first ``edge_days`` average.

    A positive number means production rose over the window, negative means it
    fell. NaNs are dropped; returns 0.0 when there isn't enough data or the
    baseline edge is non-positive (so a brand-new / dead well never divides by
    zero). Sign convention: ``(recent_avg - start_avg) / start_avg * 100``.
    """
    y = np.asarray(values, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 2:
        return 0.0
    k = max(1, min(int(edge_days), len(y) // 2 if len(y) >= 2 else 1))
    k = max(1, min(k, len(y)))
    start_avg = float(np.mean(y[:k]))
    recent_avg = float(np.mean(y[-k:]))
    if start_avg <= 0:
        return 0.0
    return (recent_avg - start_avg) / start_avg * 100.0


def build_fleet_table(
    fleet: dict[str, pd.DataFrame],
    window_days: int | None = 30,
    anomaly_by_well: dict[str, str] | None = None,
) -> pd.DataFrame:
    """One row per well summarizing the latest day + over-window dynamics, joined
    with registry metadata (lift / lateral / basin / formation).

    Columns: Well, Lift, Lateral (ft), Basin·Formation, BOPD, BWPD, MCFD,
    Water cut %, GOR (scf/bbl), Production variance %, Days on prod, Runtime %,
    Anomaly. ``anomaly_by_well`` maps well_id -> a short flag string (e.g. from
    ``scan_fleet``); wells absent from it show "—".

    Pure pandas (no Streamlit). Registry enrichment is done per-well via the
    vendored ``fleet_registry`` so the builder is importable from tests."""
    import sys
    from pathlib import Path as _Path
    _demo = _Path(__file__).resolve().parent.parent / "demo"
    if str(_demo) not in sys.path:
        sys.path.insert(0, str(_demo))
    import fleet_registry  # type: ignore

    anomaly_by_well = anomaly_by_well or {}
    rows = []
    for well_id in sorted(fleet):
        df = fleet[well_id]
        if df is None or not len(df):
            continue
        win = slice_window(df, window_days)
        last = win.iloc[-1]
        bopd = float(last["bopd"]) if pd.notna(last["bopd"]) else float("nan")
        bfpd = float(last["bfpd"]) if pd.notna(last["bfpd"]) else float("nan")
        gas = float(last["gas_mcfd"]) if "gas_mcfd" in win.columns and pd.notna(last["gas_mcfd"]) else float("nan")
        bwpd = bfpd - bopd if np.isfinite(bopd) and np.isfinite(bfpd) else float("nan")
        water_cut = (bwpd / bfpd * 100.0) if np.isfinite(bwpd) and bfpd > 0 else float("nan")
        gor = (gas * 1000.0 / bopd) if np.isfinite(gas) and np.isfinite(bopd) and bopd > 0 else float("nan")
        meta = fleet_registry.get(well_id)
        rows.append({
            "Well": well_id,
            "Lift": meta.lift,
            "Lateral (ft)": int(meta.lateral_length_ft),
            "Basin·Formation": f"{meta.basin} · {meta.formation}",
            "BOPD": round(bopd, 1) if np.isfinite(bopd) else None,
            "BWPD": round(bwpd, 1) if np.isfinite(bwpd) else None,
            "MCFD": round(gas, 1) if np.isfinite(gas) else None,
            "Water cut %": round(water_cut, 1) if np.isfinite(water_cut) else None,
            "GOR (scf/bbl)": round(gor) if np.isfinite(gor) else None,
            "Production variance %": round(production_variance_pct(win["bopd"].values), 1),
            "Days on prod": int(len(df)),
            "Runtime %": round(float(last["runtime_pct"]), 1) if pd.notna(last["runtime_pct"]) else None,
            "Anomaly": anomaly_by_well.get(well_id, "—"),
        })
    return pd.DataFrame(rows)
