# Changelog

All notable changes to Operations Center are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-06-14

Lift-aware intervention engine (the deferred audit lever, completed).

### Changed
- **Recommended interventions are now lift-appropriate.** The mode→intervention map
  is gated by the well's artificial-lift type, so the board never recommends a
  physically-impossible job: ESP swaps only on ESP wells, rod-pump workovers on
  rod-pumped wells, gas-lift optimization only on gas-lift wells, stimulation on
  flowing wells. (Previously a non-ESP well could default to `esp_swap` — e.g. a
  rod-pump well in the top opportunities.) Mirrored bit-for-bit into pe-pipeline's
  `pipeline_core` so the certified `rank_fleet ≡ pipeline_core` parity invariant
  still holds; the Action Chain AFE uses the same lift-aware mapping, so the AFE and
  the board agree.
- **Opportunity gating keeps the count honest.** Because lift-correct interventions
  are cheaper than a blanket ESP swap, they pencil on far more wells against the ESP
  model's out-of-distribution risk — which would have inflated "opportunities" from
  ~26 to ~63. An opportunity now requires a real trigger — actively deferring
  production OR the fleet's own elevated-risk quartile — AND a positive risk-weighted
  NPV. A cheap intervention that merely pencils on a no-signal well is now correctly
  "stable", not an opportunity. Result on the demo fleet: ~30 opportunities / ~11
  watch / ~59 stable, with diverse, physical interventions (esp_swap / rod_pump_workover
  / gas_lift_optimization / acid / scale). Home and the Triage Board use the same
  gated tiers, so Home's "Top Opportunity" is exactly the board's #1.

### Tests
- Regression guard: every recommended intervention is valid for its lift type. Triage
  partition test exercises both the deferred and elevated-risk signal paths. 40 tests
  pass; parity holds. (pe-pipeline: lift map + `well_013` lift updated, 13 pass.)

## [0.4.0] — 2026-06-14

Three top-recommended levers from the audit, wired in.

### Added
- **Monte-Carlo AFE economics** on the Action Chain. The recommended intervention now
  shows a P10/P50/P90 NPV band, a P(payout < 24 mo), and a tornado of the three biggest
  uncertainties (incremental rate, uplift decline, realized price) — 10,000 trials over
  the AFE component's already-built `simulate_economics` engine, which had zero callers.
  Results are netted to the operator so the **P50 reconciles exactly with the AFE's
  deterministic Net NPV** (verified). A single-point NPV reads junior at sign-off; this
  is the distributional view a capital review runs on.
- **Economic limit & remaining life** on Well 360. The rate at which net revenue equals
  fixed lease operating expense (the P&A rate) and the months from today's rate —
  declining at the well's own fitted exponential — to reach it. The number a PE defends
  in a reserves/abandonment review; it was absent entirely.
- **Ranking scorecard** on the Triage Board. precision@5/10/20 + lift-over-random +
  recall, scored against the fleet's known seeded faults — the same honest-backtest
  treatment the digest's event detector and the deferment classifier already carry.
  Honest by construction (≈80% precision@10, ~2× lift, recall < 1.0 — low-rate modes
  defer few barrels and rank lower), not a trivial 100%. The generator now persists a
  `ground_truth.csv` for this (the ESP model's `labels.csv` is a different fleet).

### Notes
- Bootstrap regenerates the fleet if `ground_truth.csv` is missing (so the scorecard
  appears even on a warm container that predates it).
- Deferred (documented in v0.3.1): the lift-aware *intervention* engine — it needs a
  coordinated `pipeline_core` change + opportunity-gating re-tune to keep the count
  credible, so it stays a tracked follow-up rather than a regression risk.

## [0.3.1] — 2026-06-14

PE-credibility audit fixes (multi-agent review of the whole console).

### Fixed — correctness
- **Action Chain rate/NPV reconciliation (blocker).** A non-flagged well's
  synthesized alert carried `baseline_bopd = 0`, so its AFE uplift collapsed to the
  20-bopd floor while the Triage Board used the real rate — the metric strip and the
  AFE disagreed, and the risk-weighted NPV could exceed the un-risked AFE NPV (an
  economic impossibility). `alert_for` now carries the well's real trailing-7-day
  baseline, so the board and the AFE size the incremental rate identically.
- **Risked-NPV labels corrected.** The Triage Board described the metric as
  "net NPV × failure signal" but computes `risk × PV(net revenue) − cost`; the
  context-bar and source-note now state the actual convention. The Action Chain's
  "Risked NPV" metric is relabeled risk-weighted with a caption explaining why it is
  below the AFE's deterministic Net NPV.
- **Morning brief deferred-$ is net.** The vendored digest reports deferred $ gross;
  the brief body (and the daily email) now net it by NRI to match the page KPI.
- **ESP model-load failure is now visible.** A failed model load no longer silently
  flattens the fleet to a uniform baseline risk — `risk_scoring_degraded()` drives a
  banner on Home and the Triage Board, and the caught error is logged.

### Fixed — domain realism (what a production engineer would catch)
- **Workover history is lift-aware.** Well 360's intervention history no longer shows
  physically-impossible jobs (an ESP swap on a rod-pumped well); each job is drawn
  only from interventions valid for the well's lift type, and uplift is scaled to the
  job size (defensible $/bopd instead of RNG).
- **Gas-interference signatures only on gas-lift wells.** The generator seeds gas
  interference / gas lock onto gas-lift wells (the flagship well_013 is now a gas-lift
  well), so "gas-lift optimization" is never recommended for a well with no injection.
  Each failure signature is seeded only on lift types it can physically occur on.
- **Heterogeneous fleet.** Wells now span a realistic rate range (~35–950 bopd) with
  per-well decline and water cut; gross fluid is derived from oil + water cut (was a
  flat 220-bopd / 88%-WC clone ×100). Rate-loss divergences ramp gradually instead of
  a rectangular step.
- **Hero storylines match their data** (well_008 downthrust, well_013 gas-lift).

### Fixed — consistency & infra
- **Warm-container fleet self-heal actually fires.** `_artifacts_ready()` was satisfied
  by the old 50-well CSVs, so the 50→100 regeneration never ran on a warm container;
  it is now fleet-count-aware and clears caches after a regen.
- Shared per-lift diagnostic-channel map (Surveillance and Well 360 can no longer
  diverge); Surveillance type-curve annual-decline formula corrected; gas-lift
  valve-vs-compressor note corrected; "Sources & BYOD" well count rendered
  dynamically (was a stale "50"); stale "Colorado default" docstrings updated; Home
  discloses when triage figures still reflect the synthetic fleet under a BYOD upload.
- Sidebar header capitalized to **Well File**.
- Dependencies bounded (`scikit-learn>=1.6,<1.10`, etc.) + `runtime.txt` pins Python
  3.12 to match CI so the deployed env equals the tested one; daily-brief Action
  passes `BRIEF_NRI` and gains `timeout-minutes` + a concurrency guard.

### Tests
- New coverage for the email renderer (`notify`), the 3-tier triage partition
  invariant, and the production-divergence / net-$ convention. 36 tests pass; the
  `rank_fleet ≡ pipeline_core` parity invariant still holds on the new fleet.

### Known follow-up
- The intervention *recommendation* on the board is still lift-agnostic for the
  unclassified/default case (an OOD ESP score can still default a non-ESP well to
  `esp_swap`). A lift-aware intervention engine is the right fix but requires a
  coordinated `pipeline_core` change + opportunity-gating re-tune to keep the
  opportunity count credible — tracked as a follow-up, not shipped here.

## [0.3.0] — 2026-06-14

Realistic 100-well fleet + a new Surveillance page + Home/triage rework.

### Data
- **Fleet doubled to 100 wells** with a realistic health distribution. The
  generator now seeds signatures that persist over the ESP model's 30-day feature
  window (not just a 5-day blip) and adds the two channels the model keys on
  (`current_imbalance_pct`, `drive_freq_hz`) plus gas-lift channels
  (`gas_inj_mcfd`, `casing_pressure_psi`, `tubing_pressure_psi`). Result: the
  Triage Board shows **~22 genuine opportunities** (gas-lift / scale fixes that
  clear their cost) instead of one, plus real wells-down and production
  divergences. Pinned tests updated for the new fleet.

### Added
- **Surveillance page** (Spotfire-style) — fleet oil/water/gas rate-time with
  30-day moving averages, an exponential decline / type-curve check (actual vs
  expected, implied deferment), and a per-well drill-down whose diagnostics adapt
  to the artificial-lift type (gas-lift injection + casing/tubing pressure for
  gas-lift wells, intake/amps/imbalance for ESP, runtime/load for rod pump).
- **Triage Board — three tiers**: value-accretive **Opportunities**, an **At-Risk
  Watch List** (losing production but not yet economic to fix), and a visible
  **No-Action / Stable** table (full-fleet coverage, not just exceptions).
- **Home** — **What Broke Overnight** + **What To Do First** (prioritized actions,
  not just cost); boxed quick-link cards into the loop; the two-datasets note moved
  into a Methods expander.
- **Navigation** — Morning Brief now sits above the Triage Board; Surveillance
  leads the Today section after Home.
- **Well 360 revamp** — a conversation-starting one-pager: a status verdict
  (opportunity / watch / stable), **online-since** + a synthetic **work history**
  (how many times the well has been worked, what was done, lifetime spend), and
  lift-aware production + diagnostic trends (oil/water/gas plus the channels that
  matter for the lift type) with today's alert marked.
- **Action Chain revamp** — drill into **any** well (or jump to one of the wells
  flagged today, so the Detect stage is never an empty "none"); a richer Detect
  stage; and an automated AFE summary card (intervention, cost, risked NPV, and
  authority-limit approval routing) with a one-click decision-ready AFE download.

## [0.2.0] — 2026-06-13

PE-credibility pass (full senior-PE + UX audit, then fixes) — presentation-layer
only; every certified number and all 31 invariant tests are unchanged.

### Fixed (credibility)
- **Triage Board no longer presents negative-NPV interventions as "opportunities."**
  The action tier is split into **value-accretive opportunities** (positive
  risk-weighted NPV) and an **At-Risk Watch List** (failure signal present but
  intervening now destroys value → monitor, don't authorize). The Top
  Opportunities chart shows positives only and labels each bar with the
  intervention to run.
- **Deferred $/day is now real across the console.** The board ranks on the ESP
  alert feed, which carries no deferred barrels by design, so the money columns
  read $0; they now join the digest's decline-aware rate-loss scan for honest,
  net-of-NRI deferred dollars (display only — the certified frame is untouched).
- **ESP 30-day risk is presented as a fleet-relative ranking**, not a calibrated
  absolute probability (the model is trained on the ESP component's fleet), so the
  console no longer implies most wells will fail within 30 days.
- **Well 360 SCADA chart** now plots each channel (oil rate / intake pressure /
  motor temp / amps) on its own auto-scaled axis with units — the ESP leading
  indicators are legible instead of flattened under motor temperature.
- **Loss Accounting** defaults to the synthetic reason-coded fleet (full cause
  attribution, MTTR, recovery queue, classifier eval); real Colorado ECMC is now a
  bring-your-own reference, not the default. "Unclassified" is treated as a
  data-quality gap, not a root cause; MTTR is shown for recoverable causes only.
- **Morning brief** is dated to the data's as-of day (not wall-clock); deferred-$
  is net-of-NRI consistently on Home / Morning Brief / Triage Board.
- **Dead discount slider removed** — discounting is fixed at the certified PV10.

### Added
- **Home — Fleet Health at a Glance**: green/amber/red status bar + healthy / on
  watch / impaired counts, % nominal, fleet oil rate, and HIGH-severity pill.
- **Morning Brief — Wells Down & Production Divergences** metrics + tables on the
  page and appended to the downloadable/emailed brief; field-status row (oil/fluid/
  water-cut/runtime); Top Deferred-$ Offenders expanded.
- **Email the morning brief** — in-app SMTP send (session-only credentials) plus a
  headless `scripts/daily_brief_email.py` and a `daily-brief.yml` GitHub Action for
  an automated every-morning send (`notify.py` renders clean text+HTML).
- **Deferment Buckets by Category** (artificial lift, surface facility, power,
  gathering, wellbore, planned, weather, reservoir) with recoverable / planned /
  uncaptured split; dominant cause surfaced on worst-offender wells; period label.
- **Well 360 drill-down** — in-page well picker (synced to the sidebar) + identity
  fields (API-14, area, first production) + a jump to the Action Chain.
- **Action Chain economic verdict banner** — AUTHORIZE / MONITOR / NO ACTION up
  front, so a non-economic AFE is framed honestly instead of as a recommendation.
- **Open Events** ranked by duration (days); sidebar leads with Operator Products.

## [0.1.0] — 2026-06-11

### Added
- **Operations Center v0.1.0** — the consolidated morning-triage console:
  surveillance → loss accounting → fleet triage → action chain in one
  deterministic Streamlit app (`st.navigation`, four sections, ten pages).
- **Absorbed components** (vendored under `apps/`, loaded via the importlib
  alias loader in `core.py`):
  - daily-production-digest **v0.6.3** — detectors, brief, event state machine
  - deferment-iq **v0.5.1** — deferment engine + analytics, real Colorado ECMC
    monthly extract (the Loss Accounting default)
  - esp-failure-risk-agent **v0.7.3** — 30-day failure-risk scoring (chain only)
  - afe-copilot **v0.6.2** — cost rollup, PV10 economics, AFE markdown
  - pe-pipeline — absorbed (not vendored): `core.py` adapts `pipeline_core.py`;
    triage-board + per-well chain UI ported into `views/`
- **Enterprise presentation layer** — `product_theme.py` masthead / context bar
  / KPI rows / pills / product switcher over the suite `theme.py`; Material
  icons in navigation, no emoji.
- **Global session contract** — oil price / NRI / discount deck, selected well,
  loss-accounting source, optional session-only Anthropic key.
- **BYOD** — one consolidated Sources & BYOD page: fleet SCADA upload (drives
  Morning Brief + Ongoing Events) and monthly production upload (drives Loss
  Accounting), validated through the components' own loaders; session-only.
- **Product tests (31)** — alias/bootstrap checks, AppTest render smoke +
  per-view coverage, and four numeric invariants pinning the console to its
  components: `rank_fleet` ≡ pe-pipeline's `pipeline_core` (frame-identical),
  `get_alerts` ≡ `digest.handoff.export_alerts`, real-Colorado %-deferred via
  the view path ≡ the component's analytics (6.0%), and the one import-rewritten
  file (`digest/src/ledger.py`) behavior-identical to the original component.
