"""Generate a synthetic 50-well fleet with ~400 days of daily SCADA per well.

A long history (400 days) makes the app's time-range toggles (7D / 30D / 3mo /
6mo / 1Y / Lifetime) meaningful, while the seeded anomalies + near-threshold
decoys keep their signatures in the RECENT window (last ~5 days) so the daily
digest's trailing-window detection still flags them on the latest data.

Every well also carries a ``gas_mcfd`` channel: gas is correlated to oil via a
per-well GOR (gas-oil ratio, scf/bbl) so MCFD = bopd * GOR / 1000, with the same
decline trend as oil plus realistic noise. Gassier lift types (Gas lift / Flowing)
get a higher GOR.

Deterministic: every well is seeded off its number, so reruns are byte-identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the vendored fleet_registry importable so gas GOR can key off lift type.
_DEMO = Path(__file__).resolve().parent.parent.parent / "demo"
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))
try:
    import fleet_registry  # type: ignore
except Exception:  # pragma: no cover - registry is always vendored, but stay robust
    fleet_registry = None


OUT = Path(__file__).parent / "fleet"
OUT.mkdir(exist_ok=True)
N_WELLS = 50
N_DAYS = 400
RNG = np.random.default_rng(11)

# Keep the same END_DATE as the original 30-day fleet so date-sensitive consumers
# (ledger trailing window, briefs) still land on the same "latest day". Only the
# history reaches further back now.
END_DATE = pd.Timestamp("2026-05-29")
DATES = pd.date_range(end=END_DATE, periods=N_DAYS)

# Number of trailing days an injected anomaly/decoy signature occupies. Detection
# rules look at the last 1, 5, 7, 8 or 14 days, so placing signatures here keeps
# them inside every rule's window regardless of total history length.
RECENT = 5


def _gor_for(seed: int) -> float:
    """Per-well gas-oil ratio (scf/bbl), deterministic from the well number and
    biased higher for gassy lift types. Range ~500–3000 scf/bbl."""
    lift = None
    if fleet_registry is not None:
        lift = fleet_registry.get(f"well_{seed:03d}").lift
    rng = np.random.default_rng(seed + 7000)
    base = float(rng.uniform(600, 1400))  # baseline dissolved-gas GOR
    if lift in ("Gas lift", "Flowing"):
        base += float(rng.uniform(700, 1400))  # gassier wells
    elif lift == "Rod pump":
        base *= 0.85  # shallower / less gassy
    return float(np.clip(base, 500, 3000))


def _add_gas(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add a ``gas_mcfd`` channel correlated to oil via the well's GOR.

    MCFD = bopd * GOR / 1000, so gas inherits oil's decline + step-downs, plus a
    small multiplicative noise that is independent of oil's noise (so the two
    series aren't perfectly collinear)."""
    rng = np.random.default_rng(seed + 9000)
    gor = _gor_for(seed)
    oil = df["bopd"].to_numpy(dtype=float)
    noise = rng.normal(1.0, 0.04, len(df))  # 4% multiplicative noise
    gas = np.clip(oil, 0, None) * gor / 1000.0 * noise
    df = df.copy()
    df["gas_mcfd"] = np.round(np.clip(gas, 0, None), 1)
    return df


def healthy_well(seed: int) -> pd.DataFrame:
    """Synthetic daily SCADA for a stable well over N_DAYS.

    Oil/fluid carry a gentle exponential decline (~0.03%/day) over the long
    history with realistic daily-average noise (~3-5% CoV), so a Lifetime view
    shows a believable decline while short windows look essentially flat. Diagnostic
    channels (intake, temp, amps, runtime) stay stationary. Gas is added via GOR.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N_DAYS)
    decline = 0.9997 ** t  # ~11% over 400 days — gentle base decline
    df = pd.DataFrame({
        "date": DATES,
        "bopd": np.clip(rng.normal(220, 8, N_DAYS) * decline, 80, 600),          # ~3.6% CoV
        "bfpd": np.clip(rng.normal(1800, 60, N_DAYS), 1200, 2800),               # ~3.3% CoV
        "intake_pressure_psi": np.clip(rng.normal(120, 4, N_DAYS), 70, 200),     # ~3.3% CoV
        "motor_temp_f": np.clip(rng.normal(290, 2, N_DAYS), 270, 320),           # ~0.7% CoV
        "motor_amps": _flat_recent_amps(rng),                                    # ~1.3% CoV, flat tail
        "runtime_pct": np.clip(rng.normal(99, 0.4, N_DAYS), 92, 100),
    })
    return _add_gas(df, seed)


def _flat_recent_amps(rng) -> np.ndarray:
    """Stationary motor-amps with a de-trended recent window. Over a long history
    the trailing 8-day least-squares slope of pure noise can drift past the
    amps_creep threshold by chance and spuriously flag a healthy well; we remove
    the recent-window slope so only the deliberately-injected creep wells fire."""
    amps = np.clip(rng.normal(60, 0.8, N_DAYS), 50, 72)
    win = 8
    tail = amps[-win:]
    x = np.arange(win)
    slope = float(np.polyfit(x, tail, 1)[0])
    amps[-win:] = tail - slope * x  # remove the trailing slope, keep the noise level
    return np.clip(amps, 50, 72)


# ---- inject specific anomalies in named wells so the brief has signal -------
# Each places its signature in the RECENT trailing window so the latest-day scan
# still fires regardless of the longer history. Gas is recomputed AFTER the oil
# edit so a rate drop also shows in gas (physically consistent).

def well_with_rate_drop(seed: int) -> pd.DataFrame:
    df = healthy_well(seed)
    df.loc[df.index[-1], "bopd"] = df["bopd"].iloc[-1] * 0.55  # 45% drop in last 24h
    return _add_gas(df, seed)


def well_with_intake_collapse(seed: int) -> pd.DataFrame:
    df = healthy_well(seed)
    p = df["intake_pressure_psi"].to_numpy(copy=True)
    p[-RECENT:] = np.linspace(p[-RECENT], 18, RECENT)  # collapsing to 18 psi over 5 days
    df["intake_pressure_psi"] = p
    return df


def well_with_motor_temp_spike(seed: int) -> pd.DataFrame:
    df = healthy_well(seed)
    df.loc[df.index[-1], "motor_temp_f"] = 348  # HIGH threshold = 340
    return df


def well_with_runtime_degradation(seed: int) -> pd.DataFrame:
    df = healthy_well(seed)
    df.loc[df.index[-1], "runtime_pct"] = 62  # HIGH threshold = 70
    return df


def well_with_amps_creep(seed: int) -> pd.DataFrame:
    """Amps creep over the recent ~8-day window (the rule's lookback), not the
    whole 400-day history — a creep diluted over 400 days would never trip."""
    df = healthy_well(seed)
    amps = df["motor_amps"].to_numpy(copy=True)
    win = 8
    amps[-win:] = amps[-win:] + np.linspace(0, 6, win)  # ~0.75 A/day over 8 days
    df["motor_amps"] = np.clip(amps, 50, 80)
    return df


# ---- decoys: look anomalous but should NOT fire (so P/R aren't trivially 1.0) ---

def decoy_subthreshold_dip(seed: int) -> pd.DataFrame:
    """Last day dips ~12% — below the 15% flag threshold. Should NOT fire."""
    df = healthy_well(seed)
    df.loc[df.index[-1], "bopd"] = df["bopd"].iloc[-1] * 0.88
    return _add_gas(df, seed)


def decoy_steep_decliner(seed: int) -> pd.DataFrame:
    """A healthy but fast natural decliner over the recent window. The flat-mean
    rate_drop rule over-flags this (today is well below the trailing mean) — a
    FALSE POSITIVE — but the decline-aware rule correctly sees it's on-trend and
    stays quiet. This is the pair that justifies the decline-aware refinement.

    The steep decline is confined to the recent 14-day window (the decline-aware
    rule's lookback) so the long history doesn't wash it out."""
    df = healthy_well(seed)
    rng = np.random.default_rng(seed)
    win = 14
    tt = np.arange(win)
    bopd = df["bopd"].to_numpy(copy=True)
    bopd[-win:] = np.clip(350 * 0.94 ** tt + rng.normal(0, 2, win), 1, None)
    df["bopd"] = bopd
    return _add_gas(df, seed)


def decoy_noisy_amps(seed: int) -> pd.DataFrame:
    """High day-to-day amps noise but zero trend — should NOT trip amps_creep."""
    df = healthy_well(seed)
    amps = np.clip(
        df["motor_amps"].to_numpy() + np.random.default_rng(seed).normal(0, 4, N_DAYS), 50, 72)
    # De-trend the recent 8-day window: keep the high day-to-day noise but remove
    # any chance slope so the (correct) negative isn't flipped by the long history.
    win = 8
    x = np.arange(win)
    slope = float(np.polyfit(x, amps[-win:], 1)[0])
    amps[-win:] = np.clip(amps[-win:] - slope * x, 50, 72)
    df["motor_amps"] = amps
    return df


def decoy_borderline_intake(seed: int) -> pd.DataFrame:
    """Intake dips toward ~45 psi (above the 40 psi threshold) — should NOT fire."""
    df = healthy_well(seed)
    p = df["intake_pressure_psi"].to_numpy(copy=True)
    p[-RECENT:] = np.linspace(p[-RECENT], 45, RECENT)
    df["intake_pressure_psi"] = p
    return df


# ---- driver -----------------------------------------------------------------

SEEDED_ANOMALIES = [
    ("well_007", well_with_rate_drop),           # HIGH rate drop
    ("well_013", well_with_intake_collapse),     # HIGH intake collapse
    ("well_022", well_with_motor_temp_spike),    # HIGH motor temp
    ("well_028", well_with_runtime_degradation), # HIGH runtime
    ("well_034", well_with_amps_creep),          # MEDIUM amps creep
    ("well_041", well_with_amps_creep),          # MEDIUM amps creep
]

# Negatives that sit near a threshold — they should NOT fire. They make the
# backtest produce real false positives / true negatives instead of a perfect score.
DECOY_WELLS = [
    ("well_045", decoy_subthreshold_dip),
    ("well_046", decoy_steep_decliner),
    ("well_047", decoy_noisy_amps),
    ("well_048", decoy_borderline_intake),
]


def main():
    special = {**dict(SEEDED_ANOMALIES), **dict(DECOY_WELLS)}
    for name, builder in special.items():
        idx = int(name.split("_")[1])
        builder(seed=idx).to_csv(OUT / f"{name}.csv", index=False)

    for i in range(1, N_WELLS + 1):
        name = f"well_{i:03d}"
        if name in special:
            continue
        healthy_well(seed=i).to_csv(OUT / f"{name}.csv", index=False)

    print(f"Wrote {N_WELLS} wells × {N_DAYS} days to {OUT} "
          f"({len(SEEDED_ANOMALIES)} seeded anomalies, {len(DECOY_WELLS)} near-threshold decoys)")


if __name__ == "__main__":
    main()
