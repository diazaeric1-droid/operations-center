"""Tests for backtest v2 — the EVENT-LIFECYCLE backtest (src/backtest_v2.py).

These validate the state machine against injected multi-day outages of known
start/end, not just per-day detection. The headline assertion is the bug's
regression: an injected 10-day outage must remain ONGOING on day 5 (it must NOT
vanish on day 4). We also check event-level precision/recall stay meaningful (the
near-threshold decoys keep precision honestly < 1.0), DURATION accuracy, and
detection LATENCY.
"""
from __future__ import annotations

from src.backtest_v2 import (
    DECOY_CLEAN_NEGATIVE_WELLS,
    DECOY_SPURIOUS_WELLS,
    INJECTED,
    build_injected_fleet,
    metrics_to_dict,
    run_backtest,
)


def test_regression_ten_day_outage_ongoing_on_day5():
    """THE bug regression, at the backtest level: the injected 10-day outage is
    ONGOING on day 5 and open (NEW/ONGOING) every day of the outage — before the
    state-machine fix it vanished on day 4."""
    m = run_backtest()
    assert m.outage10_ongoing_on_day5, "10-day outage must be ONGOING on day 5, not gone"
    assert m.outage10_open_every_day, "10-day outage must stay open every outage day"


def test_event_recall_is_perfect_on_injected_outages():
    """Every injected multi-day outage opens an event (no missed outages)."""
    m = run_backtest()
    assert m.fn == 0
    assert m.recall == 1.0
    assert m.tp == len(INJECTED)


def test_event_precision_is_non_trivial():
    """The decoys must keep precision honestly below 1.0 — a metering-recal step is
    a spurious positive the tracker should (and does) flag, so precision < 1."""
    m = run_backtest()
    assert m.fp >= 1, "the spurious-positive decoy should produce at least one FP"
    assert 0.5 <= m.precision < 1.0, f"precision should be informative, got {m.precision}"


def test_clean_negative_decoys_open_no_event():
    """The sub-threshold dip and the smooth steep decliner must NOT open an event
    (validates the 15% threshold and the decline-aware suppression)."""
    from src.backtest_v2 import DATES, N_DAYS
    from src.event_store import EventStore, update_events

    fleet = build_injected_fleet()
    store = EventStore(":memory:")
    opened: set[str] = set()
    for idx in range(N_DAYS):
        sliced = {w: df.iloc[: idx + 1] for w, df in fleet.items()}
        update_events(store, sliced, as_of=DATES[idx].date().isoformat())
        for e in store.all_events():
            if "rate" in e.event_type:
                opened.add(e.well_id)
    store.close()
    assert not (DECOY_CLEAN_NEGATIVE_WELLS & opened), \
        f"clean-negative decoys wrongly opened events: {DECOY_CLEAN_NEGATIVE_WELLS & opened}"
    assert DECOY_SPURIOUS_WELLS & opened, "the spurious decoy should have opened (it's the FP)"


def test_duration_accuracy_within_one_day():
    """Detected open→resolved abnormal span matches the injected span (MAE small)."""
    m = run_backtest()
    assert m.duration_mae <= 1.0, f"duration MAE too high: {m.duration_mae}"
    # Each individual observed duration is within 1 day of its injected length.
    for o in INJECTED:
        obs = m.per_outage[o.well_id]["observed_duration"]
        assert abs(obs - o.true_duration) <= 1, (o.well_id, obs, o.true_duration)


def test_detection_latency_is_small():
    """Step outages are caught on/near their onset day — low latency."""
    m = run_backtest()
    assert m.mean_latency <= 1.0, f"mean latency too high: {m.mean_latency}"


def test_metrics_dict_is_serializable_and_complete():
    """The committed metrics summary has the expected keys (it's what we snapshot)."""
    d = metrics_to_dict(run_backtest())
    for key in ("event_precision", "event_recall", "event_f1", "tp", "fp", "fn",
                "duration_mae_days", "mean_latency_days", "outage10_ongoing_on_day5",
                "outage10_open_every_day", "n_injected_outages"):
        assert key in d
    import json
    json.dumps(d)  # must be JSON-serializable


def test_backtest_is_deterministic():
    """Fixed seeds → identical metrics on every run (no flaky precision)."""
    a = metrics_to_dict(run_backtest())
    b = metrics_to_dict(run_backtest())
    for k in ("event_precision", "event_recall", "tp", "fp", "fn",
              "duration_mae_days", "mean_latency_days"):
        assert a[k] == b[k], f"non-deterministic metric: {k}"
