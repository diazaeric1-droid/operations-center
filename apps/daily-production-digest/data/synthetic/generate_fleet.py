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
    """Stable well on a gentle base decline; diagnostics stationary + de-trended.

    Each well gets its OWN initial rate, decline, and water cut (a real fleet spans
    an order of magnitude in rate and a wide water-cut range — not one 220-bopd /
    88%-WC clone ×100). Gross fluid is DERIVED from oil + water cut, so the two
    curves move together (oil declines → gross declines) instead of an independent
    draw that left gross fluid flat while oil fell."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_DAYS)
    # Per-well initial oil rate (lognormal): ~35–950 bopd spread across the fleet.
    qi = float(np.clip(np.exp(rng.normal(np.log(170.0), 0.62)), 35.0, 950.0))
    # Per-well exponential decline (~6–20% effective annual).
    d_daily = float(rng.uniform(0.00018, 0.00060))
    decline = np.exp(-d_daily * t)
    bopd = np.clip(rng.normal(qi, qi * 0.035, N_DAYS) * decline, 5.0, 1600.0)
    # Per-well water cut, drifting gently upward over life; gross fluid follows it.
    wc0 = float(np.clip(rng.uniform(0.55, 0.92), 0.40, 0.96))
    wc = np.clip(wc0 + np.linspace(0.0, float(rng.uniform(0.0, 0.05)), N_DAYS),
                 0.30, 0.985)
    bfpd = np.clip(bopd / (1.0 - wc), bopd + 5.0, 9000.0)
    amps = np.clip(rng.normal(60, 0.8, N_DAYS), 50, 72)
    imb = np.clip(rng.normal(2.2, 0.5, N_DAYS), 0.5, 5.0)
    df = pd.DataFrame({
        "date": DATES,
        "bopd": bopd,
        "bfpd": bfpd,
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
    df["bfpd"] = np.clip(bfpd, 20, None)
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
    bfpd[-k:] = bfpd[-k:] * np.linspace(1.0, 0.82, k)        # ~18% gross-fluid fade
    df["bfpd"] = np.clip(bfpd, 20, None)
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
    """Production diverging GRADUALLY below the decline-expected rate over ~20 days
    (down to ~35% below by the end) — a deferred-barrels divergence without a single
    hard fault. A smooth ramp, not a rectangular step, because a real organic loss
    accrues continuously (the rectangular 0.65 step read as a metering/allocation
    artifact, not a decline divergence)."""
    df = healthy_well(seed)
    k = 20
    ramp = np.linspace(1.0, 0.62, k)                 # widening gap below the curve
    oil = df["bopd"].to_numpy(copy=True)
    oil[-k:] = oil[-k:] * ramp
    df["bopd"] = oil
    bfpd = df["bfpd"].to_numpy(copy=True)
    bfpd[-k:] = bfpd[-k:] * np.linspace(1.0, 0.88, k)
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


# ---- well → signature assignment (LIFT-AWARE) --------------------------------
# A failure mode is assigned only to wells whose ARTIFICIAL-LIFT TYPE it can
# physically occur on, so the recommended intervention is never nonsensical (no
# "gas-lift optimization" on an ESP well, no "ESP swap" on a flowing well):
#   gas interference / gas lock  -> Gas-lift wells  -> gas_lift_optimization (cheap)
#   scale / downthrust / electrical -> ESP wells    -> scale_tx / esp_swap
#   shut-in                       -> any lift        -> well down
#   rate-loss divergence          -> rod-pump/flowing -> lift-agnostic deferral
# Hero wells are anchored to the signature their registry storyline describes.
import random as _random

_HERO_SIG = {7: "rate_loss", 8: "downthrust", 13: "gas_interference",
             22: "electrical", 41: "downthrust", 48: "downthrust"}

# (name, builder, lift pool the fault is valid on, target well count)
_SIG_PLAN = [
    ("gas_interference", sig_gas_interference, "Gas lift", 8),
    ("gas_lock",         sig_gas_lock,         "Gas lift", 4),
    ("scale",            sig_scale,            "ESP",      6),
    ("downthrust",       sig_downthrust,       "ESP",      6),
    ("electrical",       sig_electrical,       "ESP",      4),
    ("shut_in",          sig_shut_in,          "any",      4),
    ("rate_loss",        sig_rate_loss,        "base",     6),
]
UNDERINJECT: set = set()   # retired: the gas-lift fault now shows on the displayed
# injection/casing channels of the gas_interference / gas_lock wells themselves
# (see _GASLIFT_SYMPTOM below), so no separate "loses injection but no rate loss"
# well exists to read as a textbook problem the console fails to flag.
DECOYS = {
    96: decoy_subthreshold_dip,
    97: decoy_steep_decliner,
    98: decoy_borderline_intake,
}


def _assign_signatures() -> dict:
    """Deterministically map each signature to lift-appropriate wells (heroes first),
    returning the same ``{name: (well_numbers, builder_fn)}`` shape the rest of the
    module consumes. Stable across runs (fixed seed, sorted pools)."""
    pools: dict = {}
    for n in range(1, N_WELLS + 1):
        pools.setdefault(_lift(n), []).append(n)
    reserved = set(DECOYS) | set(UNDERINJECT)
    nums: dict = {name: set() for name, _f, _l, _c in _SIG_PLAN}
    for well, name in _HERO_SIG.items():
        nums[name].add(well)
        reserved.add(well)
    rng = _random.Random(4242)
    for name, _fn, lift_req, count in _SIG_PLAN:
        if lift_req == "any":
            pool = list(range(1, N_WELLS + 1))
        elif lift_req == "base":         # lift-agnostic divergence: keep off gas-lift
            pool = pools.get("Rod pump", []) + pools.get("Flowing", [])
        else:
            pool = list(pools.get(lift_req, []))
        pool = sorted(n for n in pool if n not in reserved)
        need = max(0, count - len(nums[name]))
        chosen = rng.sample(pool, min(need, len(pool)))
        reserved.update(chosen)
        nums[name].update(chosen)
    return {name: (sorted(nums[name]), fn) for name, fn, _l, _c in _SIG_PLAN}


SIGNATURES = _assign_signatures()

# Gas-lift wells whose fault must show on the DISPLAYED gas-lift channels (injection
# falls, casing builds) so the shown evidence matches the gas-interference / gas-lock
# diagnosis and the "restore injection to recover rate" recommendation. Without this,
# the well's oil collapses while its visible lift channels sit still (the real driver,
# an intake-pressure collapse, is an ESP channel not shown on a gas-lift well) — a
# self-contradiction a gas-lift PE catches instantly.
_GASLIFT_SYMPTOM = set(SIGNATURES["gas_interference"][0]) | set(SIGNATURES["gas_lock"][0])

# Rate-affected modes also get a RECENT acute oil drop so the digest's trailing-day
# detector quantifies a real deferred rate (the 30-day diagnostic signature drives
# the ESP mode; this gives the daily scan + loss accounting a number to book). The
# drop is on the last day vs a clean 7-day baseline (the proven detector pattern).
# Shut-ins are already at ~0; electrical wells barely move on rate, so neither here.
RATE_DROP_FRAC = {
    "gas_interference": 0.58, "gas_lock": 0.62, "scale": 0.66,
    "downthrust": 0.55, "rate_loss": 0.5,
}


def _recent_oil_drop(df: pd.DataFrame, frac: float, seed: int = 0) -> pd.DataFrame:
    """Pull the last ~3 days of oil down toward ``frac`` of the prior 7-day baseline
    (gross fluid proportionally) so the trailing-day rate-drop detector books a real
    deferral. Spans a few noisy days, not one isolated point — a single half-rate day
    on an otherwise-flat plateau reads as a bad allocation day, not a real event."""
    df = df.copy()
    rng = np.random.default_rng(seed + 31000)
    d = 3
    base = float(df["bopd"].iloc[-(d + 7):-d].mean())
    bf_base = float(df["bfpd"].iloc[-(d + 7):-d].mean())
    # Step toward the impaired rate over the last d days, with mild day-to-day noise.
    steps = np.linspace((1.0 + frac) / 2.0, frac, d) * rng.normal(1.0, 0.03, d)
    idx = df.index[-d:]
    df.loc[idx, "bopd"] = base * steps
    df.loc[idx, "bfpd"] = bf_base * (0.55 + 0.45 * steps)
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
            df = _recent_oil_drop(df, _RATE_DROP_WELLS[i], seed=i)
        df = _add_gas(df, i)                                   # gas AFTER oil edits
        df = _add_gaslift(df, i, underinject=(i in _GASLIFT_SYMPTOM))
        for c in ("bopd", "bfpd", "intake_pressure_psi", "motor_temp_f",
                  "motor_amps", "runtime_pct", "current_imbalance_pct",
                  "drive_freq_hz"):
            df[c] = df[c].round(1)
        df.to_csv(OUT / f"{name}.csv", index=False)

    # Persist the GROUND TRUTH (which wells carry a real seeded fault) so the Triage
    # Board's ranking can be scored honestly — precision@k / lift — on THIS fleet. The
    # ESP component's labels.csv is for a different fleet and doesn't join here.
    sig_of = {n: name for name, (nums, _fn) in SIGNATURES.items() for n in nums}
    gt = ["well_id,seeded_mode,impaired"]
    for i in range(1, N_WELLS + 1):
        if i in sig_of:
            mode, impaired = sig_of[i], 1
        elif i in DECOYS:
            mode, impaired = "decoy", 0
        elif i in UNDERINJECT:
            mode, impaired = "under_injection", 0
        else:
            mode, impaired = "healthy", 0
        gt.append(f"well_{i:03d},{mode},{impaired}")
    (OUT.parent / "ground_truth.csv").write_text("\n".join(gt) + "\n")

    seeded = sum(len(nums) for nums, _ in SIGNATURES.values())
    print(f"Wrote {N_WELLS} wells × {N_DAYS} days to {OUT} "
          f"({seeded} seeded signatures across {len(SIGNATURES)} modes, "
          f"{len(_GASLIFT_SYMPTOM)} gas-lift wells show the injection symptom, "
          f"{len(DECOYS)} decoys; ground_truth.csv written)")


if __name__ == "__main__":
    main()
