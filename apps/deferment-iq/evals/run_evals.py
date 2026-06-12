"""Eval harness for the reason-code classifier.

The event log carries a ground-truth ``true_cause`` the classifier never sees. We run
the deterministic rules classifier over each note and score it: overall accuracy +
per-class precision/recall/F1 + a confusion matrix. Writes evals/results/summary.json
(the demo's Eval tab and the CI gate read this). Run: ``python -m evals.run_evals``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.data_loader import load_events
from src.reason_codes import REASON_CODES, classify_rules

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "data" / "synthetic" / "events.csv"
OUT = ROOT / "evals" / "results"
CLASSES = [rc.key for rc in REASON_CODES] + ["unclassified"]


def evaluate(events_path=EVENTS) -> dict:
    ev = load_events(events_path)
    if ev.empty or "true_cause" not in ev.columns:
        raise SystemExit("No labeled events (need a true_cause column).")

    rows = []
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))
    hits = 0
    for _, e in ev.iterrows():
        true = str(e["true_cause"])
        pred, _score = classify_rules(str(e["note"]))
        ok = (pred == true)
        hits += ok
        confusion[true][pred] += 1
        if ok:
            tp[true] += 1
        else:
            fp[pred] += 1; fn[true] += 1
        rows.append({"note": e["note"], "true": true, "pred": pred, "match": ok})

    n = len(ev)
    per_class = {}
    for c in CLASSES:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else None
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else None
        f1 = (2 * p * r / (p + r)) if (p and r) else None
        support = tp[c] + fn[c]
        if support or tp[c] + fp[c]:
            per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": support}

    return {
        "n": n, "accuracy": hits / n if n else 0.0,
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "cases": rows,
    }


def main():
    res = evaluate()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"Reason-code classifier — {res['n']} events · accuracy {res['accuracy']*100:.1f}%")
    print(f"{'class':<22}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>5}")
    for c, m in sorted(res["per_class"].items(), key=lambda kv: -(kv[1]['support'] or 0)):
        fmt = lambda x: f"{x:.2f}" if isinstance(x, float) else "—"
        print(f"{c:<22}{fmt(m['precision']):>7}{fmt(m['recall']):>7}{fmt(m['f1']):>7}{m['support']:>5}")
    print(f"\nWrote {OUT / 'summary.json'}")
    # CI gate: fail if accuracy drops below threshold
    THRESH = 0.80
    if res["accuracy"] < THRESH:
        raise SystemExit(f"Eval gate FAILED: accuracy {res['accuracy']:.2f} < {THRESH}")


if __name__ == "__main__":
    main()
