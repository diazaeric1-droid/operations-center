---
title: Deferment IQ
emoji: 🛢️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# Deferment IQ — base management / lost-oil accounting

> Quantifies the barrels (and dollars) a fleet is *losing* against each well's potential,
> tags every loss to a cause, and tells the asset team what's **recoverable** — the number a
> production VP reviews every single week.

Built by a Staff Production Engineer (ex-OXY, ex-Shell) who ran base-management reviews on
Permian and Gulf of Mexico assets.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)

---

## The problem

"Base management" — arresting decline and recovering unplanned downtime — is the #1 weekly
asset review, because the cheapest barrel is the one you already had. Every operator tracks
**deferred (lost) production by reason code** against each well's potential, and PEs burn hours
reconciling theoretical vs. actual and tagging causes from messy operator notes. Get it wrong and
you chase the wrong wells.

## What it does

1. **Models each well's potential** (entitlement) from its full-uptime days only — a decline-aware
   P75 of the downtime-normalized rate — so a healthy well reads ~0 deferred (no phantom loss).
2. **Computes deferment** = potential − actual, split into **downtime** (well was off) vs.
   **underperformance** (choked / high line pressure / watering out while up), with an 8% deadband
   so normal noise isn't counted.
3. **Attributes every lost barrel to a cause** from the operator's free-text note — a deterministic
   keyword classifier (~92% on the eval), with an optional **LLM classifier** (bring-your-own-key)
   for the messy long tail.
4. **Surfaces the VP views** — deferment waterfall, a Pareto of $ by cause, worst-offender wells,
   MTTR by cause, and the **recoverable opportunity** (excludes planned work and reservoir
   watering-out, which you can't get back).
5. **Flags the capture gap** — how much deferment has *no* reason code (uncaptured), a real
   data-quality finding — and writes a Senior-PE base-management review.

Detection and accounting are deterministic; the LLM only classifies the ambiguous tail and
narrates. **Everything works with no API key.**

## How it's evaluated

Two committed, CI-gated evals:

**1. Reason-code classifier.** The synthetic event log carries a ground-truth cause the
classifier never sees. The rules classifier is scored against it — **accuracy + per-class
precision/recall/F1 + confusion** — and a **CI gate fails the build under 80%** (current ~92%).
Residual misses are the deliberately vague notes ("well down, see foreman") — exactly where the
optional LLM classifier earns its keep.

**2. Quantity-recovery (engine accounting).** A separate eval builds a synthetic fleet with
**known injected downtime + underperformance per well**, so the true deferred / recoverable
barrels are known, then checks the engine's *quantity* math against them — error on total
deferred bbl, on the downtime-vs-underperformance split, and on recovery opportunity. It runs on
**both a daily-cadence and a monthly-cadence** representation of the identical fleet to prove the
engine is **cadence-aware**: daily recovers ground truth ~exactly, and monthly resolves downtime
**exactly** (days-produced is explicit) with downtime barrels matching daily to the bit. Short
sub-month rate dips are smeared by the monthly producing-day average, so monthly under-counts
underperformance — an inherent limit of public monthly data, reported openly and gated per
cadence. A **CI gate fails the build** if the deferred-bbl error exceeds a sane bound.

```bash
python -m evals.run_evals            # classifier eval
python -m evals.quantity_recovery    # engine quantity / cadence-awareness eval
```

## Quick start

```bash
pip install -e ".[demo,dev]"
python data/synthetic/generate.py     # 40 wells x 90 days + an event log of operator notes
python -m evals.run_evals             # classifier eval
python -m evals.quantity_recovery     # engine quantity / cadence-awareness eval
streamlit run demo/app.py
```

## The new capability this demonstrates

Beyond the rest of the suite (surveillance, ESP failure, well review, AFEs), this adds
**reason-code NLP classification, capacity/entitlement modeling, and Pareto/waterfall loss
accounting** — the base-management discipline every operator runs and the metric VPs live on.

## Part of a multi-agent suite

Chains off the **Daily Production Digest** (which flags *today's* anomalies): Deferment IQ rolls
those events up into the *period* loss accounting and the $-recoverable opportunity that drives the
weekly review and the workover backlog.

## License

MIT.

## Contact

Eric Diaz II — [LinkedIn](https://www.linkedin.com/in/eric-a-diaz2) — diaz.a.eric1@gmail.com
