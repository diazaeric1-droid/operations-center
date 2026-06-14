"""Generate a synthetic 100-well fleet with ~400 days of daily SCADA per well.

A long history (400 days) makes the app's time-range toggles meaningful, while the
seeded signatures persist over the RECENT ~30-day window so BOTH the daily digest's
trailing-window detectors AND the ESP model's 30-day-slope features see them.

Channels (every well):
    bopd, bfpd, gas_mcfd                       — production
    intake_pressure_psi, motor_temp_f,
    motor_amps, runtime_pct                    — ESP diagnostics
    current_imbalance_pct, drive_freq_hz       — electrical / VSD (the two channels
                                                 the ESP model keys on; without them
                                                 every well scored a uniform ~0.55)
Gas-lift wells additionally carry:
    gas_inj_mcfd, casing_pressure_psi, tubing_pressure_psi

Gas is correlated to oil via a per-well GOR; gassier lift types get a higher GOR.

The seeded signatures are designed so the suspected-failure-mode classifier and the
AFE economics produce a REALISTIC spread, not "one opportunity, 49 negatives":
    gas interference / gas lock  -> gas_lift_optimization (cheap)  -> opportunities
    scale / abrasive buildup     -> scale_treatment (medium)        -> opportunities
    downthrust / electrical      -> esp_swap (expensive)            -> at-risk watch
    multi-day shut-ins           -> wells down
    sustained under-performance  -> production divergences

Deterministic: every well is seeded off its number, so reruns are byte-identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the vendored fleet_registry importable so GOR + lift-specific channels can
# key off the well's lift type.
_DEMO = Path(__file__).resolve().parent.parent.parent / "demo"
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))
try:
    import fleet_registry  # type: ignore
except Exception:  # pragma: no cover - registry is always vendored, but stay robust
    fleet_registry = None


OUT = Path(__file__).parent / "fleet"
OUT.mkdir(exist_ok=True)
N_WELLS = 100
N_DAYS = 400
RNG = np.random.default_rng(11)

END_DATE = pd.Timestamp("2026-05-29")
DATES = pd.date_range(end=END_DATE, periods=N_DAYS)

RECENT = 5    # trailing days a sharp single-event signature occupies
WIN30 = 30    # trailing window the ESP 30-day-slope features + sustained events use


def _lift(seed: int) -> str:
    if fleet_registry is not None:
        return fleet_registry.get(f"well_{seed:03d}").lift
    return "ESP"


def _gor_for(seed: int) -> float:
    """Per-well gas-oil ratio (scf/bbl), deterministic, biased high for gassy lifts."""
    lift = _lift(seed)
    rng = np.random.default_rng(seed + 7000)
    base = float(rng.uniform(600, 1400))
    if lift in ("Gas lift", "Flowing"):
        base += float(rng.uniform(700, 1400))
    elif lift == "Rod pump":
        base *= 0.85
    return float(np.clip(base, 500, 3000))


def _add_gas(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """``gas_mcfd`` correlated to oil via the well's GOR (inherits oil's decline)."""
    rng = np.random.default_rng(seed + 9000)
    gor = _gor_for(seed)
    oil = df["bopd"].to_numpy(dtype=float)
    noise = rng.normal(1.0, 0.04, len(df))
    gas = np.clip(oil, 0, None) * gor / 1000.0 * noise
    df = df.copy()
    df["gas_mcfd"] = np.round(np.clip(gas, 0, None), 1)
    return df


def _add_gaslift(df: pd.DataFrame, seed: int, underinject: bool = False) -> pd.DataFrame:
    """Gas-lift wells carry an injection rate + casing/tubing pressures. An
    ``underinject`` well loses ~45% of its injection over the recent window (a
    compressor/valve issue) and produces below potential as a result."""
    if _lift(seed) != "Gas lift":
        return df
    rng = np.random.default_rng(seed + 12000)
    n = len(df)
    inj = np.clip(rng.normal(420, 14, n), 320, 560)         # MCF/d injection
    casing = np.clip(rng.normal(950, 22, n), 800, 1150)     # psi
    tubing = np.clip(rng.normal(310, 12, n), 220, 430)      # psi
    df = df.copy()
    if underinject:
        k = WIN30
        inj[-k:] = inj[-k:] * np.linspace(1.0, 0.55, k)     # injection falls off
        casing[-k:] = casing[-k:] + np.linspace(0, 80, k)   # casing builds (valve)
    df["gas_inj_mcfd"] = np.round(inj, 0)
    df["casing_pressure_psi"] = np.round(casing, 0)
    df["tubing_pressure_psi"] = np.round(tubing, 0)
    return df


def _flat_recent(arr: np.ndarray, win: int = WIN30) -> np.ndarray:
    """Remove any chance trailing-window slope from stationary noise so a healthy
    well never trips a slope-based detector by accident."""
    x = np.arange(win)
    slope = float(np.polyfit(x, arr[-win:], 1)[0])
    arr[-win:] = arr[-win:] - slope * x
    return arr


def healthy_well(seed: int) -> pd.DataFrame:
    """Stable well on a gentle base decline; diagnostics stationary + de-trended."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_DAYS)
    decline = 0.9997 ** t  # ~11% over 400 days
    bopd = np.clip(rng.normal(220, 8, N_DAYS) * decline, 80, 600)
    amps = np.clip(rng.normal(60, 0.8, N_DAYS), 50, 72)
    imb = np.clip(rng.normal(2.2, 0.5, N_DAYS), 0.5, 5.0)
    df = pd.DataFrame({
        "date": DATES,
        "bopd": bopd,
        "bfpd": np.clip(rng.normal(1800, 60, N_DAYS), 1200, 2800),
        "intake_pressure_psi": np.clip(rng.normal(120, 4, N_DAYS), 70, 200),
        "motor_temp_f": np.clip(rng.normal(290, 2, N_DAYS), 270, 320),
        "motor_amps": np.clip(_flat_recent(amps), 50, 72),
        "runtime_pct": np.clip(rng.normal(99, 0.4, N_DAYS), 92, 100),
        "current_imbalance_pct": np.clip(_flat_recent(imb), 0.3, 6.0),
        "drive_freq_hz": np.clip(rng.normal(59.5, 0.15, N_DAYS), 56, 62),
    })
    return df


# ---- failure signatures (persist over the recent ~30-day window) -------------

def sig_gas_interference(seed: int) -> pd.DataFrame:
    """Smooth intake collapse over 30 days + oil sagging — gas interference →
    gas-lift optimization (cheap fix → a clear opportunity)."""
    df = healthy_well(seed)
    k = WIN30
    p = df["intake_pressure_psi"].to_numpy(copy=True)
    p[-k:] = np.linspace(p[-k], 48, k)
    df["intake_pressure_psi"] = p
    oil = df["bopd"].to_numpy(copy=True)
    oil[-k:] = oil[-k:] * np.linspace(1.0, 0.74, k)   # ~26% gas-interference loss
    df["bopd"] = oil
    return df


def sig_gas_lock(seed: int) -> pd.DataFrame:
    """Volatile flow + low-runtime cycling + rising frequency — gas lock →
    gas-lift optimization (opportunity)."""
    df = healthy_well(seed)
    k = WIN30
    rng = np.random.default_rng(seed + 222)
    bfpd = df["bfpd"].to_numpy(copy=True)
    bfpd[-k:] = bfpd[-k:] * (1.0 + rng.normal(0, 0.22, k))    # CV well above 0.15
    df["bfpd"] = np.clip(bfpd, 200, 3200)
    rt = df["runtime_pct"].to_numpy(copy=True)
    low = rng.choice(np.arange(k), size=6, replace=False)
    rt[-k:][low] = rng.uniform(55, 70, len(low))             # several pump-off days
    df["runtime_pct"] = rt
    fr = df["drive_freq_hz"].to_numpy(copy=True)
    fr[-k:] = fr[-k:] + np.linspace(0, 2.4, k)               # VSD chasing pump-off
    df["drive_freq_hz"] = np.clip(fr, 56, 64)
    oil = df["bopd"].to_numpy(copy=True)
    oil[-k:] = oil[-k:] * np.linspace(1.0, 0.82, k)
    df["bopd"] = oil
    return df


def sig_scale(seed: int) -> pd.DataFrame:
    """Amps + temp creeping together with stable intake — scale/abrasive buildup →
    scale treatment (medium-cost opportunity)."""
    df = healthy_well(seed)
    k = WIN30
    amps = df["motor_amps"].to_numpy(copy=True)
    amps[-k:] = amps[-k:] + np.linspace(0, 6.5, k)           # +0.21 A/d
    df["motor_amps"] = np.clip(amps, 50, 82)
    temp = df["motor_temp_f"].to_numpy(copy=True)
    temp[-k:] = temp[-k:] + np.linspace(0, 4.2, k)           # +0.14 °F/d
    df["motor_temp_f"] = np.clip(temp, 270, 345)
    oil = df["bopd"].to_numpy(copy=True)
    oil[-k:] = oil[-k:] * np.linspace(1.0, 0.86, k)
    df["bopd"] = oil
    return df


def sig_downthrust(seed: int) -> pd.DataFrame:
    """Rate sliding, runtime down, amps flat — downthrust/declining inflow →
    esp_swap (expensive → at-risk watch, not yet economic)."""
    df = healthy_well(seed)
    k = WIN30
    bfpd = df["bfpd"].to_numpy(copy=True)
    bfpd[-k:] = bfpd[-k:] - np.linspace(0, 12 * k, k)        # ~ -12 bbl/d/d
    df["bfpd"] = np.clip(bfpd, 200, 3200)
    rt = df["runtime_pct"].to_numpy(copy=True)
    rt[-k:] = np.linspace(rt[-k], 93, k)
    df["runtime_pct"] = np.clip(rt, 60, 100)
    oil = df["bopd"].to_numpy(copy=True)
    oil[-k:] = oil[-k:] * np.linspace(1.0, 0.80, k)
    df["bopd"] = oil
    return df


def sig_electrical(seed: int) -> pd.DataFrame:
    """Current imbalance ramping past 9% on multiple days — electrical/incipient
    short → esp_swap (at-risk watch)."""
    df = healthy_well(seed)
    k = WIN30
    imb = df["current_imbalance_pct"].to_numpy(copy=True)
    imb[-k:] = imb[-k:] + np.linspace(0, 11, k)             # peaks ~13%, several >8%
    df["current_imbalance_pct"] = np.clip(imb, 0.3, 18)
    oil = df["bopd"].to_numpy(copy=True)
    oil[-k:] = oil[-k:] * np.linspace(1.0, 0.93, k)
    df["bopd"] = oil
    return df


def sig_shut_in(seed: int) -> pd.DataFrame:
    """Multi-day full shut-in ending on the latest day — a WELL DOWN."""
    df = healthy_well(seed)
    d = 6
    for col, val in (("bopd", 0.3), ("bfpd", 2.0), ("runtime_pct", 0.0)):
        a = df[col].to_numpy(copy=True)
        a[-d:] = val
        df[col] = a
    return df


def sig_rate_loss(seed: int) -> pd.DataFrame:
    """Sustained ~35% under-performance vs the decline-expected rate over ~20 days
    — a production divergence (deferred barrels) without a single hard fault."""
    df = healthy_well(seed)
    k = 20
    oil = df["bopd"].to_numpy(copy=True)
    oil[-k:] = oil[-k:] * 0.65
    df["bopd"] = oil
    bfpd = df["bfpd"].to_numpy(copy=True)
    bfpd[-k:] = bfpd[-k:] * 0.9
    df["bfpd"] = bfpd
    return df


# ---- decoys: look anomalous but should NOT fire (keeps P/R off a trivial 1.0) ---

def decoy_subthreshold_dip(seed: int) -> pd.DataFrame:
    df = healthy_well(seed)
    df.loc[df.index[-1], "bopd"] = df["bopd"].iloc[-1] * 0.88   # ~12%, below 15% flag
    return df


def decoy_steep_decliner(seed: int) -> pd.DataFrame:
    df = healthy_well(seed)
    rng = np.random.default_rng(seed)
    win = 14
    tt = np.arange(win)
    bopd = df["bopd"].to_numpy(copy=True)
    bopd[-win:] = np.clip(350 * 0.94 ** tt + rng.normal(0, 2, win), 1, None)
    df["bopd"] = bopd
    return df


def decoy_borderline_intake(seed: int) -> pd.DataFrame:
    df = healthy_well(seed)
    p = df["intake_pressure_psi"].to_numpy(copy=True)
    p[-RECENT:] = np.linspace(p[-RECENT], 45, RECENT)          # 45 psi > 40 threshold
    df["intake_pressure_psi"] = p
    return df


# ---- well → signature assignment ---------------------------------------------
# Hero wells keep storylines consistent with the registry; the rest spread a
# realistic mix across the 100-well fleet.

SIGNATURES = {
    "gas_interference": ([13, 15, 27, 33, 52, 61, 74, 88], sig_gas_interference),
    "gas_lock":         ([19, 44, 66, 91], sig_gas_lock),
    "scale":            ([11, 25, 38, 57, 72, 95], sig_scale),
    "downthrust":       ([8, 41, 48, 53, 69, 83], sig_downthrust),
    "electrical":       ([22, 36, 60, 77], sig_electrical),
    "shut_in":          ([14, 31, 58, 84], sig_shut_in),
    "rate_loss":        ([7, 29, 46, 63, 79, 92], sig_rate_loss),
}
UNDERINJECT = {21, 70}   # gas-lift wells that lose injection (gas-lift opportunity)
DECOYS = {
    96: decoy_subthreshold_dip,
    97: decoy_steep_decliner,
    98: decoy_borderline_intake,
}

# Rate-affected modes also get a RECENT acute oil drop so the digest's trailing-day
# detector quantifies a real deferred rate (the 30-day diagnostic signature drives
# the ESP mode; this gives the daily scan + loss accounting a number to book). The
# drop is on the last day vs a clean 7-day baseline (the proven detector pattern).
# Shut-ins are already at ~0; electrical wells barely move on rate, so neither here.
RATE_DROP_FRAC = {
    "gas_interference": 0.58, "gas_lock": 0.62, "scale": 0.66,
    "downthrust": 0.55, "rate_loss": 0.5,
}


def _recent_oil_drop(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Drop the latest day's oil to ``frac`` of the prior 7-day baseline (and pull
    gross fluid down proportionally) so the rate-drop detector fires with a real
    deferral."""
    df = df.copy()
    base = float(df["bopd"].iloc[-8:-1].mean())
    df.loc[df.index[-1], "bopd"] = base * frac
    bf_base = float(df["bfpd"].iloc[-8:-1].mean())
    df.loc[df.index[-1], "bfpd"] = bf_base * (0.5 + 0.5 * frac)
    return df


_RATE_DROP_WELLS = {n: RATE_DROP_FRAC[name]
                    for name, (nums, _fn) in SIGNATURES.items()
                    if name in RATE_DROP_FRAC for n in nums}


def _builder_for(n: int):
    for _name, (nums, fn) in SIGNATURES.items():
        if n in nums:
            return fn
    if n in DECOYS:
        return DECOYS[n]
    return healthy_well


def main():
    for i in range(1, N_WELLS + 1):
        name = f"well_{i:03d}"
        builder = _builder_for(i)
        df = builder(seed=i)
        if i in _RATE_DROP_WELLS:
            df = _recent_oil_drop(df, _RATE_DROP_WELLS[i])
        df = _add_gas(df, i)                                   # gas AFTER oil edits
        df = _add_gaslift(df, i, underinject=(i in UNDERINJECT))
        for c in ("bopd", "bfpd", "intake_pressure_psi", "motor_temp_f",
                  "motor_amps", "runtime_pct", "current_imbalance_pct",
                  "drive_freq_hz"):
            df[c] = df[c].round(1)
        df.to_csv(OUT / f"{name}.csv", index=False)

    seeded = sum(len(nums) for nums, _ in SIGNATURES.values())
    print(f"Wrote {N_WELLS} wells × {N_DAYS} days to {OUT} "
          f"({seeded} seeded signatures across {len(SIGNATURES)} modes, "
          f"{len(UNDERINJECT)} gas-lift under-injection, {len(DECOYS)} decoys)")


if __name__ == "__main__":
    main()
