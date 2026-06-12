"""Generate a synthetic fleet for Deferment IQ: 40 wells x 90 days of production with
injected downtime/curtailment, plus an event log of free-text operator notes tagged
with a ground-truth cause (so the reason-code classifier can be honestly evaluated).

Realism choices:
- Most notes clearly name the cause ("ESP tripped…", "compressor down…") — operators
  really do — but ~15% are vague ("well down, see foreman") so the rules classifier
  does NOT score a trivial 100%.
- A few wells have downtime with NO event row -> uncaptured (unclassified) deferment,
  which makes the capture-rate KPI < 100% (a real finding, not a perfect demo).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
WELLS = OUT / "wells"
WELLS.mkdir(parents=True, exist_ok=True)
N_WELLS, N_DAYS, SEED = 40, 90, 11
DATES = pd.date_range(end=pd.Timestamp("2026-05-31"), periods=N_DAYS)

# cause -> (is_curtailment, note variants).  Last variant of each is deliberately vague.
NOTES = {
    "artificial_lift": (False, [
        "ESP tripped on underload, VSD fault F012, waiting on electrician",
        "rod string parted, pulling unit scheduled Tuesday",
        "gas lift unstable, well heading, adjusting injection",
        "pump off, low fillage, tuning the POC", "downhole pump worn, no fluid over pump",
        "well down, see foreman"]),
    "surface_facility": (False, [
        "separator dump valve stuck, well shut in", "compressor down, no gas takeaway off the pad",
        "tank battery full, LACT meter fault", "heater treater tripped on high level, emulsion upset",
        "shut in on facility upset"]),
    "power": (False, [
        "lost power to the pad, main breaker tripped", "substation outage, entire pad offline",
        "transformer failure, utility en route", "no power at location"]),
    "gathering_thirdparty": (True, [
        "high line pressure from gathering, curtailed", "gas plant down, midstream curtailment",
        "sales line pressure spike, backpressure on wells", "third party takeaway constraint"]),
    "wellbore": (False, [
        "scale buildup restricting flow", "paraffin plugging tubing, hot oil scheduled",
        "sand production, choked back", "fill over perforations, cleanout needed"]),
    "planned": (False, [
        "scheduled ESP workover", "planned well test in progress", "routine maintenance on the unit",
        "wireline run for survey", "rig move on location"]),
    "weather": (False, [
        "winter storm freeze-off across the pad", "froze up overnight, thawing out",
        "lightning strike took out the controller"]),
    "reservoir": (True, [
        "water cut climbing, well watering out", "liquid loading, won't unload",
        "pressure depletion, declining inflow"]),
}
CAUSES = list(NOTES)
WEIGHTS = np.array([0.30, 0.18, 0.10, 0.15, 0.12, 0.08, 0.04, 0.03])


def _well(i: int):
    rng = np.random.default_rng(i + 100)
    b0 = rng.uniform(150, 900)
    decl = rng.uniform(0.10, 0.55)
    t = np.arange(N_DAYS)
    trend = b0 * np.exp(-decl * t / 365.0)
    bopd = np.clip(trend * (1 + rng.normal(0, 0.03, N_DAYS)), 1, None)
    runtime = np.clip(rng.normal(99.5, 0.4, N_DAYS), 92, 100)
    wc = rng.uniform(0.40, 0.92)
    gor = rng.uniform(0.5, 3.0)
    return rng, trend, bopd, runtime, wc, gor


def main():
    master = np.random.default_rng(SEED)
    events = []
    # which wells get an injected event
    evt_wells = set(master.choice(N_WELLS, size=26, replace=False).tolist())
    # a couple wells get downtime with NO event row -> uncaptured deferment
    uncaptured_wells = set(master.choice(sorted(evt_wells), size=2, replace=False).tolist())

    for i in range(N_WELLS):
        wid = f"well_{i + 1:03d}"
        rng, trend, bopd, runtime, wc, gor = _well(i)

        if i in evt_wells:
            cause = CAUSES[master.choice(len(CAUSES), p=WEIGHTS / WEIGHTS.sum())]
            is_curtail, variants = NOTES[cause]
            # last variant of each cause is the vaguest; pick it only ~12% of the time
            vi = int(master.integers(0, len(variants) - 1)) if master.random() < 0.88 else len(variants) - 1
            note = variants[vi]
            dur = int(master.integers(4, 12) if is_curtail else master.integers(1, 7))
            start = int(master.integers(10, N_DAYS - dur - 1))
            sl = slice(start, start + dur)
            if is_curtail:                       # up but reduced rate
                bopd[sl] = trend[sl] * master.uniform(0.40, 0.70)
            else:                                # down (full or partial)
                rt = master.choice([0.0, 0.0, 0.0, float(master.uniform(5, 25))])
                runtime[sl] = rt
                bopd[sl] = trend[sl] * rt / 100.0
            if i not in uncaptured_wells:        # captured events get an event-log row
                events.append({"well_id": wid,
                               "start_date": DATES[start].date().isoformat(),
                               "end_date": DATES[start + dur - 1].date().isoformat(),
                               "note": note, "true_cause": cause})

        bfpd = np.clip(bopd / max(1 - wc, 0.08), bopd, None)
        gas = np.clip(bopd * gor * rng.uniform(0.8, 1.2), 0, None)
        pd.DataFrame({
            "date": DATES, "bopd": np.round(bopd, 1), "bfpd": np.round(bfpd, 1),
            "gas_mcfd": np.round(gas, 1), "runtime_pct": np.round(runtime, 1),
        }).to_csv(WELLS / f"{wid}.csv", index=False)

    pd.DataFrame(events).to_csv(OUT / "events.csv", index=False)
    print(f"Wrote {N_WELLS} wells to {WELLS} and {len(events)} events to events.csv "
          f"({len(uncaptured_wells)} wells have uncaptured downtime).")


if __name__ == "__main__":
    main()
