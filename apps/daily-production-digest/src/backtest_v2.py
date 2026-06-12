"""Backtest v2 — validate the EVENT STATE MACHINE, not just per-day detection.

The original ``src.backtest`` scores each detector's per-day precision/recall/lead
against single-day seeded faults. That cannot catch the bug the state machine
fixes: a *multi-day* outage that the stateless detector flags for a day or two and
then forgets. Backtest v2 closes that gap.

It builds a synthetic fleet with **injected multi-day outages of known
start/end**, replays the persistent state machine day-by-day over the full
history (exactly as ``scheduler.run`` would across consecutive mornings), and
measures lifecycle-level metrics against the injected ground truth:

  * EVENT precision / recall — did we open exactly one event per real outage and
    no spurious ones? (near-threshold decoys are included so this isn't a trivial
    1.0; a healthy steep decliner that the flat-mean rule false-positives on, a
    sub-threshold dip, etc.)
  * DURATION accuracy — detected open→resolved span vs the injected span (mean
    absolute error in days).
  * Detection LATENCY — lead time from the real onset to the day we first opened
    the event (NEW).
  * PERSISTENCE — the bug's regression: an injected 10-day outage must still be
    ONGOING on day 5 (it must not vanish on day 4), and must stay open every day
    until production recovers.

Run it::

    python -m src.backtest_v2            # prints the metrics summary
    python -m src.backtest_v2 --json     # machine-readable summary

The committed metrics snapshot lives at ``data/backtest_v2_metrics.json``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .anomaly_detector import DEFAULT_OIL_PRICE
from .event_store import NEW, ONGOING, RESOLVED, EventStore, update_events

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = REPO_ROOT / "data" / "backtest_v2_metrics.json"

# Calendar spine for the injected fleet (independent of the demo fleet so this
# backtest never depends on, or perturbs, the committed CSVs).
N_DAYS = 60
START = pd.Timestamp("2026-03-01")
DATES = pd.date_range(START, periods=N_DAYS)
NORMAL_BOPD = 200.0


@dataclass
class InjectedOutage:
    """Ground truth for one injected multi-day rate outage."""
    well_id: str
    onset_idx: int          # 0-based index of the first down day
    length: int             # number of consecutive down days
    drop_to: float          # the held-down production level

    @property
    def end_idx(self) -> int:
        return self.onset_idx + self.length - 1   # last down day (inclusive)

    @property
    def true_duration(self) -> int:
        return self.length


def _scada(bopd: np.ndarray, seed: int) -> pd.DataFrame:
    """Wrap an oil-rate array in a full SCADA frame with quiet diagnostic channels
    (so ONLY the rate outage fires — no confounding intake/temp/amps anomalies)."""
    rng = np.random.default_rng(seed + 100)
    n = len(bopd)
    return pd.DataFrame({
        "date": DATES[:n],
        "bopd": bopd,
        "bfpd": rng.normal(1800, 25, n),
        "intake_pressure_psi": rng.normal(120, 3, n),
        "motor_temp_f": rng.normal(290, 2, n),
        "motor_amps": _flat_amps(rng, n),
        "runtime_pct": np.clip(rng.normal(99, 0.2, n), 0, 100),
    })


def _flat_amps(rng, n: int) -> np.ndarray:
    """Stationary amps with the trailing 8-day slope removed, so a chance noise
    drift never trips amps_creep and pollutes the rate-event precision."""
    amps = np.clip(rng.normal(60, 0.8, n), 50, 72)
    win = min(8, n)
    if win >= 2:
        x = np.arange(win)
        slope = float(np.polyfit(x, amps[-win:], 1)[0])
        amps[-win:] = np.clip(amps[-win:] - slope * x, 50, 72)
    return amps


def _outage_well(o: InjectedOutage, seed: int) -> pd.DataFrame:
    """Healthy at NORMAL_BOPD, stepping down to ``o.drop_to`` for the injected
    window, then recovering to normal (if there are days left after the outage)."""
    rng = np.random.default_rng(seed)
    bopd = rng.normal(NORMAL_BOPD, 5, N_DAYS)
    end = o.onset_idx + o.length
    bopd[o.onset_idx:end] = rng.normal(o.drop_to, 4, o.length)
    bopd[end:] = rng.normal(NORMAL_BOPD, 5, max(N_DAYS - end, 0))
    return _scada(bopd, seed)


# ---- decoys: near-threshold negatives so precision/recall aren't trivially 1 ----
# Two kinds, both deterministic (noise-free where determinism matters):
#   * CLEAN NEGATIVES that look anomalous but the detector correctly rejects
#     (sub-threshold dip, smooth steep decliner) — they validate the threshold and
#     the decline-aware suppression. They must open NO event.
#   * One realistic SPURIOUS POSITIVE (a metering recalibration step) the detector
#     legitimately flags but which is a *data artifact*, not real lost production —
#     a genuine event-level false positive that keeps precision honestly < 1.0.

def _decoy_subthreshold_dip(seed: int) -> pd.DataFrame:
    """A shallow multi-day dip (~12% down) that stays under the 15% flag — must
    NOT open an event."""
    rng = np.random.default_rng(seed)
    bopd = rng.normal(NORMAL_BOPD, 5, N_DAYS)
    bopd[-6:] = rng.normal(NORMAL_BOPD * 0.88, 3, 6)
    return _scada(bopd, seed)


def _decoy_steep_decliner(seed: int) -> pd.DataFrame:
    """A healthy but fast natural decliner — smooth exponential decline (~6%/day)
    from the well's own baseline, no discontinuity. The flat-mean rate rule would
    over-flag the steepness, but the authoritative decline-aware rule correctly
    sees it's perfectly on-trend and stays quiet, so NO event opens. Validates the
    decline-aware suppression that the original generator's decoy targets."""
    # Noise-free so the replay is deterministic and the rule's on-trend judgment is
    # not perturbed into a spurious step-down by a single noisy day.
    t = np.arange(N_DAYS)
    bopd = NORMAL_BOPD * 0.94 ** t          # steep but smooth — on-trend everywhere
    return _scada(np.clip(bopd, 1, None), seed)


def _decoy_meter_recal_step(seed: int) -> pd.DataFrame:
    """A metering recalibration: the measured oil rate steps down ~17% on one day
    and HOLDS there (the meter factor changed, the well did not). The detector
    legitimately flags this as a rate drop — it has no way to know the physical
    rate is unchanged — so it opens an event. That makes it a genuine event-level
    FALSE POSITIVE for *lost-production* tracking, which keeps precision honestly
    below 1.0 (the operator would reconcile against the tank gauge and dismiss it).
    Deterministic (noise-free step) so the FP is stable, not flaky."""
    bopd = np.full(N_DAYS, NORMAL_BOPD)
    bopd[-5:] = NORMAL_BOPD * 0.83          # held 17% step-down (a recal, not real loss)
    return _scada(bopd, seed)


# Injected ground truth: multi-day outages of varied span (incl. the 10-day case
# the regression hinges on), placed so they end a few days before the window end
# (leaving recovery days so RESOLVED is observable).
INJECTED: list[InjectedOutage] = [
    InjectedOutage("well_out10", onset_idx=40, length=10, drop_to=110.0),  # the 10-day regression case
    InjectedOutage("well_out07", onset_idx=44, length=7, drop_to=120.0),
    InjectedOutage("well_out05", onset_idx=48, length=5, drop_to=100.0),
    InjectedOutage("well_out03", onset_idx=50, length=3, drop_to=115.0),
]

DECOYS = {
    "well_dec_dip": _decoy_subthreshold_dip,
    "well_dec_decline": _decoy_steep_decliner,
    "well_dec_recal": _decoy_meter_recal_step,
}

# Decoys that must open NO event (clean negatives the detector should reject).
DECOY_CLEAN_NEGATIVE_WELLS = {"well_dec_dip", "well_dec_decline"}
# Decoys that legitimately trip a detector but are NOT real lost production — each
# tracked event there is counted as a false positive for outage tracking.
DECOY_SPURIOUS_WELLS = {"well_dec_recal"}
# Every decoy must NOT be a real multi-day outage.
DECOY_NEGATIVE_WELLS = DECOY_CLEAN_NEGATIVE_WELLS | DECOY_SPURIOUS_WELLS


def build_injected_fleet() -> dict[str, pd.DataFrame]:
    """Deterministic fleet: the injected multi-day outages + a few healthy wells +
    the near-threshold decoys."""
    fleet: dict[str, pd.DataFrame] = {}
    for o in INJECTED:
        fleet[o.well_id] = _outage_well(o, seed=hash(o.well_id) % 10_000)
    for name, builder in DECOYS.items():
        fleet[name] = builder(seed=hash(name) % 10_000)
    # A handful of plainly-healthy wells so the fleet isn't all-anomalous.
    for i in range(3):
        wid = f"well_ok{i:02d}"
        fleet[wid] = _scada(np.random.default_rng(i).normal(NORMAL_BOPD, 5, N_DAYS), seed=i)
    return fleet


# ---- replay + scoring -------------------------------------------------------

@dataclass
class EventLifecycleMetrics:
    # event-level confusion (one tracked rate-event per real outage)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    # duration accuracy (days), latency (days from onset to NEW)
    duration_abs_errors: list[float] = field(default_factory=list)
    latencies: list[int] = field(default_factory=list)
    # persistence regression
    outage10_ongoing_on_day5: bool = False
    outage10_open_every_day: bool = False
    per_outage: dict = field(default_factory=dict)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def duration_mae(self) -> float:
        return (sum(self.duration_abs_errors) / len(self.duration_abs_errors)
                if self.duration_abs_errors else 0.0)

    @property
    def mean_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0


def run_backtest(price_per_bbl: float = DEFAULT_OIL_PRICE) -> EventLifecycleMetrics:
    """Replay the state machine day-by-day over the injected fleet and score it.

    For each as-of day we feed every well's history-to-date into ``update_events``
    (the same call ``scheduler.run`` makes), recording per-well the first day a
    rate event opened, its observed open→resolved duration, and — for the 10-day
    well — whether it was ONGOING on day 5 and open every outage day.
    """
    fleet = build_injected_fleet()
    store = EventStore(":memory:")

    onset_first_seen: dict[str, int] = {}     # well -> as-of idx the rate event first appeared
    resolved_at: dict[str, int] = {}          # well -> as-of idx it became RESOLVED
    ever_opened: set[str] = set()             # wells that opened any rate event
    out10_states_during: list[str | None] = []  # well_out10 state on each outage day

    out10 = next(o for o in INJECTED if o.well_id == "well_out10")

    for idx in range(N_DAYS):
        asof = DATES[idx].date().isoformat()
        sliced = {wid: df.iloc[: idx + 1] for wid, df in fleet.items()}
        update_events(store, sliced, as_of=asof, price_per_bbl=price_per_bbl)

        for ev in store.all_events():
            if "rate" not in ev.event_type:
                continue
            wid = ev.well_id
            ever_opened.add(wid)
            onset_first_seen.setdefault(wid, idx)
            if ev.state == RESOLVED and wid not in resolved_at:
                resolved_at[wid] = idx

        # Track the 10-day well's state across its injected outage window.
        if out10.onset_idx <= idx <= out10.end_idx:
            r = [e for e in store.open_events()
                 if e.well_id == out10.well_id and "rate" in e.event_type]
            out10_states_during.append(r[0].state if r else None)

    m = EventLifecycleMetrics()

    # Event precision/recall against the real multi-day outages.
    for o in INJECTED:
        opened = o.well_id in ever_opened
        m.per_outage[o.well_id] = {"injected_duration": o.true_duration, "opened": opened}
        if opened:
            m.tp += 1
            # latency: days from injected onset to first day the event appeared.
            lat = max(onset_first_seen[o.well_id] - o.onset_idx, 0)
            m.latencies.append(lat)
            # duration: observed ABNORMAL span = first-seen day .. last down day.
            # The event RESOLVES the day production recovers (one past the last down
            # day), so the down span is [first, resolved_idx) i.e. resolved-first.
            # If it never resolves in-window, it's still down through the last day.
            first = onset_first_seen[o.well_id]
            res = resolved_at.get(o.well_id)
            observed = (res - first) if res is not None else (N_DAYS - first)
            m.duration_abs_errors.append(abs(observed - o.true_duration))
            m.per_outage[o.well_id].update(
                {"observed_duration": observed, "latency_days": lat})
        else:
            m.fn += 1
            m.per_outage[o.well_id]["observed_duration"] = 0

    # False positives: decoys that should NOT be a real outage but opened an event.
    # Iterate sorted so the committed metrics JSON has a stable key order (set
    # iteration order is otherwise nondeterministic across processes).
    for wid in sorted(DECOY_NEGATIVE_WELLS):
        if wid in ever_opened:
            m.fp += 1
        m.per_outage[wid] = {"injected_duration": 0, "opened": wid in ever_opened}

    # Persistence regression (the bug): well_out10 ONGOING on day 5 + open all days.
    if len(out10_states_during) >= 5:
        m.outage10_ongoing_on_day5 = out10_states_during[4] == ONGOING
    m.outage10_open_every_day = all(
        s in (NEW, ONGOING) for s in out10_states_during) and bool(out10_states_during)
    m.per_outage["well_out10"]["states_during_outage"] = out10_states_during

    store.close()
    return m


def metrics_to_dict(m: EventLifecycleMetrics) -> dict:
    return {
        "event_precision": round(m.precision, 3),
        "event_recall": round(m.recall, 3),
        "event_f1": round(m.f1, 3),
        "tp": m.tp, "fp": m.fp, "fn": m.fn,
        "duration_mae_days": round(m.duration_mae, 2),
        "mean_latency_days": round(m.mean_latency, 2),
        "outage10_ongoing_on_day5": m.outage10_ongoing_on_day5,
        "outage10_open_every_day": m.outage10_open_every_day,
        "n_injected_outages": len(INJECTED),
        "n_decoy_negatives": len(DECOY_NEGATIVE_WELLS),
        "n_decoy_clean_negatives": len(DECOY_CLEAN_NEGATIVE_WELLS),
        "n_decoy_spurious_positives": len(DECOY_SPURIOUS_WELLS),
        "per_outage": m.per_outage,
    }


def print_report(m: EventLifecycleMetrics) -> None:
    print("Backtest v2 — event-lifecycle metrics (injected multi-day outages)\n")
    print(f"  injected outages : {len(INJECTED)}  "
          f"(durations: {', '.join(str(o.true_duration)+'d' for o in INJECTED)})")
    print(f"  decoy negatives  : {len(DECOY_NEGATIVE_WELLS)} "
          f"({len(DECOY_CLEAN_NEGATIVE_WELLS)} clean: sub-threshold dip + smooth steep "
          f"decliner; {len(DECOY_SPURIOUS_WELLS)} spurious: metering-recal step → 1 FP)\n")
    print(f"  EVENT precision  : {m.precision:.2f}   "
          f"(TP={m.tp}  FP={m.fp}  FN={m.fn})")
    print(f"  EVENT recall     : {m.recall:.2f}")
    print(f"  EVENT F1         : {m.f1:.2f}")
    print(f"  duration MAE     : {m.duration_mae:.2f} days")
    print(f"  mean latency     : {m.mean_latency:.2f} days (onset → NEW)\n")
    flag = "PASS" if m.outage10_ongoing_on_day5 else "FAIL"
    print(f"  [{flag}] 10-day outage ONGOING on day 5 (regression: before-fix it vanished day 4)")
    flag2 = "PASS" if m.outage10_open_every_day else "FAIL"
    print(f"  [{flag2}] 10-day outage open (NEW/ONGOING) every day of the outage")
    print("\n  per-outage:")
    for wid, d in m.per_outage.items():
        if d.get("injected_duration", 0) > 0:
            print(f"    {wid:<14} injected={d['injected_duration']}d "
                  f"observed={d.get('observed_duration', 0)}d "
                  f"latency={d.get('latency_days', '-')}d opened={d['opened']}")


def main():
    parser = argparse.ArgumentParser(description="Backtest v2: event-lifecycle metrics.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--write", action="store_true",
                        help=f"Write the metrics summary to {METRICS_PATH}.")
    args = parser.parse_args()

    m = run_backtest()
    if args.json:
        print(json.dumps(metrics_to_dict(m), indent=2))
    else:
        print_report(m)
    if args.write:
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(metrics_to_dict(m), indent=2) + "\n")
        print(f"\nWrote metrics summary to {METRICS_PATH}")


if __name__ == "__main__":
    main()
