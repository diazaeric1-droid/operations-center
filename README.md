# Operations Center

**What broke overnight, what it's costing, and what to do first.**

The morning console for a production operation: surveillance → loss
accounting → optimization board → action chain, in one deterministic Streamlit app.
A foreman opens it at 6:30am and leaves with a ranked board and a
decision-ready AFE — no API key required for any number on any page.

One of three consolidated operator products in the Upstream Copilot Suite
(Operations Center · Engineering Workbench · Capital Desk).

## Module Map

| Section | Page | What it answers |
|---|---|---|
| Today | Home | The 6:30am landing: wells scanned, open alerts, deferred $/day, top opportunity |
| Today | Surveillance | Fleet rate-time + clickable fleet map (click a well to drill down) + CTB/lift/basin/county filters |
| Today | Optimization Board | The whole fleet ranked by risked-NPV opportunity (tiers, downtime context, drill-through to Surveillance, CSV export) |
| Today | Morning Brief | Overnight scan: unified ranked list (new + ongoing + resolved, by BOPD) or the classic detailed panels; the brief itself (LLM narration optional) |
| Today | Ongoing Events | The event state machine — multi-day outages stay ONGOING with running duration + cumulative deferred bbl/$ |
| Loss Accounting | Deferment Overview | Potential vs actual on the monthly book — downtime vs underperformance waterfall, % deferred |
| Loss Accounting | Causes & Pareto | Reason-code attribution, $-Pareto by cause, capture rate, classifier eval |
| Loss Accounting | Recovery Work Queue | Ranked recoverable opportunities (recoverable $ ÷ MTTR), CSV export |
| Well File | Well 360 | One well: registry identity, SCADA trends with alert overlays, events, 30-day ESP risk |
| Well File | Action Chain | Detect → predict → authorize for the selected well; every stage artifact downloadable |
| Data | Sources & BYOD | Provenance for both datasets + session-only uploads (SCADA fleet, monthly production) |

## In-App Guidance (v0.9.0)

- **Every page opens with an "ℹ️ What is this page for?" popover** — the question
  the page answers, where it sits in the 6:30am loop, how to read the headline
  output, and the next page in the spine (plain-PE language; no number removed).
- **Next steps are links/buttons, not bold text**: Home's "What To Do First"
  carries one-click *Go →* buttons (the well selection follows automatically),
  the Optimization Board links straight to the Action Chain, and the
  Surveillance drill-down ends with Well 360 / Action Chain pointers.
- **Typed deck inputs**: oil price and NRI are exact `number_input`s (portfolio
  labels/ranges); well pickers show `well_013 · <name> (<lift>)` labels while
  values stay raw ids.
- **SPE exceedance convention**: P10 = high case, P90 = low case on the Action
  Chain Monte-Carlo (display relabel only; the math is unchanged).
- **Recovery Work Queue disambiguation**: loss-book wells render as
  `well_013 (loss book)` — they are NOT the surveillance wells sharing the same
  id, and the page links (never well-jumps) to the Action Chain.

## Built On

| Component | Version | Contribution |
|---|---|---|
| daily-production-digest | v0.6.3 | Anomaly detectors, morning brief, event state machine, SCADA loaders/BYOD |
| deferment-iq | v0.5.1 | Deferment engine (potential vs actual), reason codes, Pareto/MTTR/recovery analytics, real Colorado ECMC extract |
| esp-failure-risk-agent | v0.7.3 | 30-day failure-risk scoring + failure-mode classification (chain scoring only) |
| afe-copilot | v0.6.2 | Cost rollup + PV10 intervention economics + deterministic AFE markdown (`afe.econ_core` is the suite economics kernel) |
| pe-pipeline | absorbed | `core.py` adapts its `pipeline_core.py` (alias loader, bootstrap, `rank_fleet` triage math); its triage-board + per-well chain UI patterns ported into `views/` |

## Architecture

- **Vendored components + alias loader.** The four component apps live under
  `apps/` as plain directories (mirrored from their repos). Each ships a
  top-level `src` package — the names collide — so `core.py` loads each `src`
  under a distinct importlib alias (`digest`, `deferment`, `esp`, `afe`) and
  the whole console runs in ONE process: no subprocesses, no per-app venvs,
  single self-contained clone for Streamlit Cloud.
- **Certified math cores unchanged.** Vendored copies are byte-identical to the
  components (one mechanical import rewrite, documented + behavior-proven in
  `VENDORING.md`); product tests pin numeric equality against the components
  and against pe-pipeline's orchestrator.
- **New presentation layer.** Views never touch component internals beyond the
  public modules `core.py` binds; pages render through `product_theme.py`
  (masthead/context bar/KPI rows) over the suite's shared `theme.py`.
- **`core.py` is streamlit-free** so CI bootstraps and tests headless.

## Run Locally

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

First run bootstraps the gitignored artifacts (~30s): the 50-well synthetic
SCADA fleet, the deferment demo fleet, and the trained ESP risk model.

Tests: `.venv/bin/pip install pytest && .venv/bin/python -m pytest -q`

## Honest Data Notes

- **Two datasets, never joined.** Today + Well File run on a **synthetic daily
  SCADA fleet** (50 modeled Permian wells, known ground truth — public
  production data is monthly; daily SCADA with ESP diagnostics does not exist
  publicly). Loss Accounting defaults to **real Colorado ECMC monthly records**
  (DJ Basin Niobrara/Codell horizontals). Different datasets, different
  cadences; the console states the split instead of faking a join.
- **Real quantity, honest N/A.** On the real monthly book the deferment
  quantity is real (~6.0% of potential on the committed extract, from
  days-produced); cause attribution, MTTR, and the recovery queue honestly read
  N/A — public filings carry no reason codes.
- **Evals are shown, including the misses.** Digest event-lifecycle backtest:
  precision 0.80 / recall 1.00 on seeded outages + decoys (one decoy still
  opens a spurious event — reported, not hidden). Deferment reason-code
  classifier: ~92% on its ground-truth eval set, synthetic only.
- **The LLM is optional everywhere** (session-only BYOK) and confined to
  narration; every number is deterministic.
