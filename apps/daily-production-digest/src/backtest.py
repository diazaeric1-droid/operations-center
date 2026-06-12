"""Backtest harness: score the deterministic detectors against the known seeded
anomalies in the synthetic fleet, and sweep thresholds to show the precision /
recall / lead-time tradeoff per rule.

Ground truth comes straight from ``data/synthetic/generate_fleet.py``'s
``SEEDED_ANOMALIES`` list — the wells that generator deliberately corrupts, and
the category it injects. Any other well firing is a false positive; any seeded
well that doesn't fire is a false negative.

Lead time: each seeded fault is injected at a known day offset from the end of
the 30-day window (a step-change on the last day, or a multi-day ramp). We
truncate the well's history day-by-day and find the *first* day the rule fires;
lead time = (days before the fault fully manifests) that we'd have caught it.

Run it::

    python -m src.backtest
    python -m src.backtest --data-dir data/synthetic/fleet --sweep
"""
from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .anomaly_detector import (
    detect_amps_creep,
    detect_intake_collapse,
    detect_motor_temp_spike,
    detect_rate_drop,
    detect_rate_drop_decline_aware,
    detect_runtime_degradation,
)
from .data_loader import load_fleet

REPO_ROOT = Path(__file__).resolve().parent.parent
GEN_PATH = REPO_ROOT / "data" / "synthetic" / "generate_fleet.py"

# Map the generator's builder-function names → the detector category they target.
# (Decline-aware rate drop is a refinement of the same rate-drop fault.)
_BUILDER_TO_CATEGORY = {
    "well_with_rate_drop": {"rate_drop", "rate_drop_decline_aware"},
    "well_with_intake_collapse": {"intake_collapse"},
    "well_with_motor_temp_spike": {"motor_temp_spike"},
    "well_with_runtime_degradation": {"runtime_degradation"},
    "well_with_amps_creep": {"amps_creep"},
}

# How many days before the end of the window the fault begins to manifest. Used
# for lead-time scoring. Step-changes land on the final day (1 day of signal);
# ramps build over their span.
_FAULT_MANIFEST_DAYS = {
    "rate_drop": 1,
    "intake_collapse": 5,
    "motor_temp_spike": 1,
    "runtime_degradation": 1,
    "amps_creep": 30,
}

_DETECTORS = {
    "rate_drop": detect_rate_drop,
    "rate_drop_decline_aware": detect_rate_drop_decline_aware,
    "intake_collapse": detect_intake_collapse,
    "motor_temp_spike": detect_motor_temp_spike,
    "runtime_degradation": detect_runtime_degradation,
    "amps_creep": detect_amps_creep,
}


def load_ground_truth() -> dict[str, set[str]]:
    """Load ``SEEDED_ANOMALIES`` from generate_fleet.py → {well_id: {categories}}."""
    spec = importlib.util.spec_from_file_location("generate_fleet", GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    truth: dict[str, set[str]] = {}
    for well_id, builder in mod.SEEDED_ANOMALIES:
        cats = _BUILDER_TO_CATEGORY.get(builder.__name__, set())
        truth.setdefault(well_id, set()).update(cats)
    return truth


@dataclass
class RuleScore:
    rule: str
    tp: int
    fp: int
    fn: int
    latencies: list[int]        # days from fault ONSET to first detection (ramp faults)
    early_warnings: list[int]   # days BEFORE full manifestation we first fired

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else float("nan")

    @property
    def mean_lead(self) -> float:
        return sum(self.early_warnings) / len(self.early_warnings) if self.early_warnings else 0.0


def _first_fire(detector, scada: pd.DataFrame, manifest_days: int) -> tuple[int, int] | None:
    """Walk history forward one day at a time; on the first day the rule fires,
    return (detection_latency, early_warning):

    - ``detection_latency`` = days from fault ONSET (n - manifest_days) to detection.
      For a single-day step fault (manifest_days=1) this is 0 by construction — you
      cannot pre-warn a same-day step. For a multi-day ramp it's how deep into the
      ramp we were when we caught it (lower = faster).
    - ``early_warning`` = days before the fault FULLY manifests (the final day) that
      we first fired (n - end).

    Returns None if the rule never fires over the truncated histories.
    """
    n = len(scada)
    onset_idx = n - manifest_days          # 0-based index where the fault begins
    for end in range(1, n + 1):
        if detector("w", scada.iloc[:end]) is not None:
            first_idx = end - 1            # last row included in this truncation
            latency = max(first_idx - onset_idx, 0)
            early_warning = max((n - 1) - first_idx, 0)
            return latency, early_warning
    return None


def score_rules(fleet: dict[str, pd.DataFrame], truth: dict[str, set[str]]) -> list[RuleScore]:
    scores: list[RuleScore] = []
    for cat, detector in _DETECTORS.items():
        tp = fp = fn = 0
        latencies: list[int] = []
        early: list[int] = []
        for well_id, scada in fleet.items():
            fired = detector(well_id, scada) is not None
            is_truth = cat in truth.get(well_id, set())
            if fired and is_truth:
                tp += 1
                manifest = _FAULT_MANIFEST_DAYS.get(cat, 1)
                res = _first_fire(detector, scada, manifest)
                if res is not None:
                    # Only ramp faults (manifest_days > 1) yield an informative latency.
                    if manifest > 1:
                        latencies.append(res[0])
                    early.append(res[1])
            elif fired and not is_truth:
                fp += 1
            elif not fired and is_truth:
                fn += 1
        scores.append(RuleScore(cat, tp, fp, fn, latencies, early))
    return scores


def print_report(scores: list[RuleScore]) -> None:
    print(f"{'rule':<28}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>8}{'rec':>8}{'F1':>8}"
          f"{'latency':>9}{'lead(d)':>9}")
    print("-" * 86)
    for s in scores:
        lat = f"{s.mean_latency:.1f}" if s.latencies else "—"
        print(f"{s.rule:<28}{s.tp:>4}{s.fp:>4}{s.fn:>4}"
              f"{s.precision:>8.2f}{s.recall:>8.2f}{s.f1:>8.2f}{lat:>9}{s.mean_lead:>9.1f}")
    tot_tp = sum(s.tp for s in scores)
    tot_fp = sum(s.fp for s in scores)
    tot_fn = sum(s.fn for s in scores)
    p = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    r = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0
    print("-" * 77)
    print(f"{'OVERALL':<28}{tot_tp:>4}{tot_fp:>4}{tot_fn:>4}{p:>8.2f}{r:>8.2f}"
          f"{(2*p*r/(p+r) if (p+r) else 0):>8.2f}")


def sweep_amps_creep(fleet: dict[str, pd.DataFrame], truth: dict[str, set[str]],
                     slopes=(0.1, 0.2, 0.3, 0.4, 0.5)) -> None:
    """Threshold sweep for amps-creep slope — shows the precision/recall tradeoff
    as you tighten the A/day flag. (The detector's own threshold is 0.3.)"""
    from .anomaly_detector import _slope_per_step
    print("\namps_creep slope-threshold sweep (A/day):")
    print(f"{'thresh':>8}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>8}{'rec':>8}")
    for thr in slopes:
        tp = fp = fn = 0
        for well_id, scada in fleet.items():
            if "motor_amps" not in scada.columns or len(scada) < 8:
                continue
            slope = _slope_per_step(scada.iloc[-8:]["motor_amps"].values)
            fired = slope >= thr
            is_truth = "amps_creep" in truth.get(well_id, set())
            if fired and is_truth:
                tp += 1
            elif fired and not is_truth:
                fp += 1
            elif not fired and is_truth:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"{thr:>8.2f}{tp:>4}{fp:>4}{fn:>4}{prec:>8.2f}{rec:>8.2f}")


def main():
    parser = argparse.ArgumentParser(description="Backtest detectors vs seeded anomalies.")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "synthetic" / "fleet"))
    parser.add_argument("--sweep", action="store_true", help="Also run threshold sweeps.")
    args = parser.parse_args()

    fleet = load_fleet(args.data_dir)
    truth = load_ground_truth()
    print(f"Fleet: {len(fleet)} wells · ground-truth seeded wells: {len(truth)}")
    seeded_desc = ", ".join(
        "{}({})".format(w, "/".join(sorted(c))) for w, c in sorted(truth.items())
    )
    print(f"Seeded: {seeded_desc}\n")

    scores = score_rules(fleet, truth)
    print_report(scores)
    if args.sweep:
        sweep_amps_creep(fleet, truth)


if __name__ == "__main__":
    main()
