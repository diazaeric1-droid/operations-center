# Changelog

All notable changes to Operations Center are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
