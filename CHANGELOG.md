# Changelog

All notable changes to Operations Center are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] — 2026-07-23

PE field-feedback round 1 — a practicing petroleum engineer reviewed the live console;
every feasible item is implemented (the one data gap — no choke channel in the SCADA
schema — is disclosed in-page rather than faked). The certified cores are untouched:
`rank_fleet` parity with pe-pipeline stays bit-for-bit, all registry changes are
purely additive, and per-well NRI lives only in view-layer display columns.

### Added
- **Clickable fleet map → per-well drill-down (OC1).** The Surveillance map is now a
  Plotly scatter map (token-free OSM tiles): clicking a well loads it in the per-well
  drill-down below and sets the global Well File selection (deselecting clears the
  selection latch, so the same well can be re-picked from the map after switching
  wells in the dropdown). Coordinates moved into
  `fleet_registry.surface_latlon()` (same deterministic county-centroid + jitter
  formula — no well moved) so every product surface shares one source.
- **CTB / lift / basin / county filters (OC2).** A deterministic CTB (central tank
  battery) assignment joined the registry (`ctb_for`, ~2 batteries per county);
  shared filter controls on Surveillance AND the Optimization Board, with honest
  "Filtered: N of M wells" captions (the board CSV export stays full-fleet).
- **Downtime context on recommendations (OC3, honest version).** Recommendations for
  wells in an OPEN state-machine event carry "⚠ in ongoing event Nd — verify
  post-restart before acting" on the Optimization Board, Well 360, and the Action
  Chain. The SCADA schema has no choke-position channel — disclosed in-page where
  recommendations render (and on Methods) instead of inventing choke data.
- **Per-well NRI + GROSS/NET views (OC9).** Deterministic, varied per-well NRI in the
  registry (`nri_for`, ≈0.73–0.85 by basin/well) with a session-only `st.data_editor`
  override table on Sources & BYOD; GROSS (8/8) vs NET (× per-well NRI) toggles on
  the Morning Brief unified list, Optimization Board deferred-$ columns, and the
  Deferment Overview. Certified chain/ranking economics keep the sidebar deck NRI
  (relabeled "Deck NRI — chain economics"); the convention split is stated on Methods.
- **Cross-page drill-through (OC6).** Shared `jump_to_well` mechanism (global
  `well_id` + `st.switch_page`): selecting a row on the Optimization Board's
  intervention / restore / watch tables or the Morning Brief unified list opens the
  well on Surveillance with it preselected.
- **Deferment in barrels (OC8).** A Barrels (default) / Dollars toggle threads
  through the Deferment Overview KPIs, category buckets, and worst-offender table.

### Changed
- **Triage Board → Optimization Board (OC5).** Renamed everywhere user-visible (nav,
  masthead, home cards, captions, README, tests; CSV now
  `ops_optimization_board.csv`), including board-referential lowercase "triage"
  wording in rendered captions, slider help, the product-switcher tagline, and the
  README lede. The URL slug changed `triage-board` →
  `optimization-board`, so old deep links 404. Internal module/function names kept.
- **Morning Brief defaults to ONE unified list (OC4).** New + ongoing + resolved
  events and scan-only anomalies in a single ranked list ordered by BO/day impact
  (respecting the NRI toggle), status-badged; the classic three-panel layout is
  intact behind a "Detailed panels" view toggle. "The Brief" + email are unchanged.

### Registry (additive only — documented in the module docstring)
- `surface_latlon` / `ctb_for` / `nri_for` + `lat`/`lon`/`ctb`/`nri` WellMeta
  properties; META_COLUMNS gained `ctb` and `nri`. No existing field or value
  changed; this copy diverges from sibling repos until they take the same block.

## [0.7.3] — 2026-06-16

Scoped adversarial re-audit of the v0.7.0/v0.7.1 post-fix surface (a second 6-persona
pass, every finding reproduced against the code and 3-lens verified) closed the residual
self-contradictions a PE would still hit on the first wells they open — plus the
model-card honesty gap. Parity with pe-pipeline's `pipeline_core.rank_fleet` is preserved
bit-for-bit (the certified ranking and `_map_mode` are untouched; #5/#6/#9 are view-layer).
55 tests (was 47).

### Fixed — model card honesty (the ML-reviewer's question)
- **Out-of-fold metrics are now END-TO-END calibrated.** `_cross_validate` previously
  scored each fold with the *raw* booster, so the persisted AUROC/Brier described the
  uncalibrated model while a "Calibrated: yes" tile sat beside them. Each fold now trains
  *and* Platt-calibrates exactly as the shipped artifact does, so the Brier describes the
  calibrated probabilities the console shows (AUROC 0.994→**0.979**, Brier 0.036→**0.055** —
  the honest numbers). An `eval_method` marker self-heals a stale eval on warm containers.
- **Recall is now on the model card** beside precision, with the honest framing that
  precision@10% is structurally near-100% on a heavily-impaired fleet and recall@10% is
  capped by flagging only the top 10% — read them as a pair.
- **No fabricated real-world AUROC.** The "~0.85 on a real historian" figure is reframed
  as an engineering expectation (we have no labeled operator historian to measure one),
  not a quoted result. Stale "AUROC ≈0.99" captions updated to ≈0.98.

### Fixed — on-screen self-contradictions (a PE catches these first)
- **Gas-lift gas-interference evidence cites the displayed channels.** It quoted the
  hidden ESP intake pressure ("flowing pressure at 56 psi") on a well whose shown channels
  are injection/casing/tubing; it now describes the falling injection / building casing the
  reviewer actually sees. (Flowing wells, which *do* display downhole pressure, unchanged.)
- **Economic Limit reads the current rate, not the pre-collapse plateau.** The displayed
  rate is now the recent trailing producing rate (matching the chart), a `below_established_
  trend` flag warns when a well is producing under its own trend, and the down-guard keys
  off a sustained recent rate — so a well at ~half its plateau no longer reads "374 BOPD,
  19 yr left" next to a high failure signal.
- **Home "Elevated Risk" can't read 0 while wells are high-risk.** The amber bucket is now
  an absolute calibrated band (≥50%) of wells not already losing, and surfaces how many
  high-risk wells are already counted in Impaired — so a low amber count reads as "the
  high-risk wells are already in the red," not "nothing is at risk."
- **Shut-in wells route to a Restore queue.** Currently-down wells are pulled out of the
  Triage Board's priced opportunities into a "Restore First" tier (and excluded from Home's
  Top Opportunity) — no more $395K "gas-lift optimization" on a well whose own page says
  "restore production first." View-layer; the certified ranking is untouched.
- **The Stable tier no longer calls high-risk wells "healthy."** The opportunity signal
  gained an absolute calibrated-risk floor (≥50%), so 0.5–0.9 wells leave the no-action
  tier; the "these wells read healthy" caption is gone.
- **Action Chain degrades to "diagnose first" on an ambiguous mode.** When the classifier
  finds no dominant signature, the verdict and the diagnosis no longer present the priced
  intervention as a confident recommendation — it's flagged as contingent on confirming
  the mode (dyno card / pressure survey) first.
- **Uplift-horizon assumption is stated where the NPV appears.** The 5-year/0.6-decline
  uplift tail is generous for a short-scope job (a 1-day gas-lift optimization); the Monte-
  Carlo note now discloses the horizon instead of just endorsing the figure, with an
  explicit upper-bound caveat on optimization jobs and a centralized note on Methods.

## [0.7.2] — 2026-06-15

### Fixed
- **Warm-container module self-heal hardened.** Streamlit Cloud reuses the Python
  process across redeploys, so a cached OLD copy of one of our modules can lack
  symbols added in a newer commit → `AttributeError` at run. The self-heal now runs
  once per session and evicts **every** product-owned module — the component aliases
  (`digest`, `deferment`, `esp`, `afe`) and `src.*` in addition to `core`,
  `product_theme`, `theme`, `fleet_registry`, and `views.*` (previously it ran every
  rerun and only evicted the three shared modules). Skipped under pytest to preserve
  the cross-test module-identity invariants.

## [0.7.1] — 2026-06-15

PE-scrutiny readiness pass — fixes the on-screen self-contradictions a 6-persona
adversarial review (senior PE / reservoir / asset-manager / ML / skeptic / honesty)
reproduced in the live app, the day before peer review.

### Fixed — credibility (a sharp PE catches these in the first five minutes)
- **Flagship gas-lift well's diagnosis now matches its displayed evidence.** A
  gas-interference/gas-lock well's oil collapsed while its *shown* gas-lift channels
  (injection, casing) sat still — the real driver (intake-pressure collapse) is an ESP
  channel not displayed on a gas-lift well. The generator now drives the fault through
  the **displayed** channels (injection falls ~40%, casing builds), so "falling
  injection with rising casing → restore injection" reads true. (Retired the separate
  under-injection wells that read as an unflagged textbook problem.)
- **"ESP 30-day failure risk" + ESP-only field steps no longer shown on 61 non-ESP
  wells.** The diagnosis wording and recommended field steps are now lift-aware (a
  gas-lift / rod-pump / flowing well isn't told to "megger the motor" or that it has an
  ESP). *(classify_failure_mode gains an optional `lift`; detection stays deterministic
  and lift-agnostic, so the board / parity are unchanged.)*
- **Economic-limit "remaining life" no longer absurd on down wells.** A shut-in well
  used to read "296 BOPD now / 1.1 yr" (the 180-day fit read a 6-day outage as terminal
  decline). It now returns a **"well currently down — restore first"** state, uses a
  robust producing-day rate, and fits decline on the **same established-trend basis as
  the type-curve overlay** — so the two decline numbers on Well 360 no longer disagree.
- **One approval-routing schedule.** The Action Chain "Routes To" metric and the AFE
  document used two different authority tables (Field Superintendent vs Production
  Engineer for the same cost); both now call the AFE component's `required_approver`.

### Fixed — wording / consistency
- The Monte-Carlo caption now correctly says the **base case (mean)** reconciles with
  the deterministic Net NPV, with the P50 slightly below it (right-skew) — it isn't the
  P50 that reconciles.
- Home's amber health bucket renamed **"Elevated Risk"** (with a clarifying caption) so
  the word "Watch" doesn't read 0 on Home and 8 on the Triage Board's economic tier.
- Removed a stale "ESP score is OOD here" code comment left over from before v0.7.0.

### Tests
- Regression guards: gas-lift faults show on displayed channels; economic-limit guards
  down wells. 47 tests pass; `rank_fleet ≡ pipeline_core` parity holds (pe-pipeline
  updated for the diagnosis wording).

## [0.7.0] — 2026-06-14

ESP model recalibrated on the digest fleet.

### Changed
- **The ESP failure-risk model is now trained ON the digest fleet** — the fleet the
  console actually scores — using the generator's ground-truth fault labels, instead
  of being trained on the ESP component's *different* fleet and scoring this one
  out-of-distribution. The 30-day failure score is now a **Platt-calibrated
  probability** that separates impaired wells (~0.82) from healthy ones (~0.04),
  rather than a uniform ~0.5 blob the console could only rank relatively. Honest
  **out-of-fold** eval (stratified CV) is persisted and shown on a new **model card**
  (Methods & Limitations): AUROC, Brier, precision@10%, calibrated flag, n-impaired.
  *The high AUROC (≈0.99) is disclosed as an upper bound on clean separable synthetic
  signatures — not a real-world claim; ~0.85 is what real-well data would show.*
- **Board reads cleaner as a result:** healthy wells now correctly fall into the
  no-action tier (their calibrated risk is low) instead of everything looking
  "active" under OOD scores — ~60 stable / ~26 opportunities / ~8 watch.
- All "fleet-relative, not a calibrated probability — ignore the absolute number"
  disclaimers across Home / Triage / Well 360 / Action Chain are replaced with the
  honest calibrated framing pointing at the model card.

### Notes
- Bootstrap **self-heals an old OOD-trained artifact**: a model without the new
  `esp_eval.json` marker is retrained on the digest fleet (so warm containers and old
  checkouts upgrade automatically).
- **Parity preserved**: both `core` and `pipeline_core` load the same retrained
  artifact, so the `rank_fleet ≡ pipeline_core` invariant still holds — no
  certified-math change. 45 tests pass.

## [0.6.0] — 2026-06-14

The remaining audit levers + the minor leftovers.

### Added
- **Per-well type-curve overlay.** Well 360 and the Surveillance per-well drill-down
  now draw each well's own fitted exponential decline over its oil — the "is this well
  on its type curve, or deferring?" read that previously existed only at the fleet
  level (`core.fit_well_decline`).
- **"What changed since yesterday" on Home.** "What Broke Overnight" is now a true
  day-over-day **diff** off the event state machine — NEW / STILL-ONGOING / RESOLVED
  groups (with what's down right now) — instead of a stateless re-list of today's scan.
- **Fleet map on Surveillance.** A geospatial view coloured green/amber/red by live
  health tier (`core.well_tiers`). Each well is placed at its real Permian county's
  centroid with a deterministic within-county jitter (synthetic coordinates, disclosed).
- **Methods & Limitations page** (Data section). A model card consolidating the
  economics conventions, the ESP fleet-relative-not-calibrated disclosure, the
  lift-aware rule, the two-datasets provenance, and the committed backtest numbers —
  the page a sharp reviewer looks for, instead of scattered fine-print captions.

### Changed
- Dependencies: `scipy` now explicitly pinned (`>=1.13,<2`) instead of floating as a
  transitive dep, so the whole numeric stack is bounded for reproducible deploys.

### Tests
- New coverage: per-well decline fit alignment, the map's per-well health tiers, and a
  guard that the shipped ESP model carries a fitted Platt calibrator (audit #25 — a
  silent fall-through to uncalibrated probabilities now fails CI). 44 tests pass.

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
