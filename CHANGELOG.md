# Changelog

All notable changes to Operations Center are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
