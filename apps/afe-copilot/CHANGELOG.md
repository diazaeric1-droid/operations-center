# Changelog

All notable changes to AFE Copilot are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] — 2026-06-11
### Changed
- Economics now backed by shared `econ_core` kernel; numbers unchanged.

## [0.6.1] — 2026-06-07
### Changed
- **Light theme** — suite-wide migration from dark/navy to a professional light palette (white surfaces, `plotly_white` charts, navy/blue accents retained); transparent fixed header so the title never clips. `runtime.txt` pinned to Python 3.11.

## [0.6.0] — 2026-06-06

### Added
- **Multipage explorer** — an **Overview** (in-flight $ KPIs, by-status / by-approver breakdown,
  and a **sortable AFE pipeline table**: AFE #, well/project, intervention, gross $, net NPV,
  status, required approver, days-in-status, supplement flag, variance) alongside Draft / Variance /
  Cost-Benchmark tabs, plus a **drill-down page per AFE** (`st.navigation`) with its cost waterfall,
  net economics + tornado, risk register, authority routing, immutable audit trail, and actual-vs-AFE
  variance.

### Fixed
- Cache the SQLite tracker via `st.cache_resource` (not `cache_data`) so it doesn't trip
  `UnserializableReturnValueError` on Python 3.14 / newer Streamlit.

## [0.5.0] — 2026-06-06

### Added
- **Unified dark + navy suite theme** with a **cross-app sidebar suite navigator** — consistent
  look and one-click hopping across the upstream-PE app suite.
- **One-click AFE export to Word (.docx)** wired into the Draft tab, alongside the existing
  Markdown (.md) download.
- **Cost waterfall chart** visualizing the build-up from direct line items → contingency → total.
- **Shared fleet registry**: Permian field / formation identity is now consistent across the
  whole suite.

### Changed
- Swept the deprecated `use_container_width` argument (→ `width="stretch"`); requires
  `streamlit>=1.50`.

## [0.4.0] — 2026-06-03

### Added
- **Working-interest / NRI net economics**: `compute_economics` now takes `working_interest`
  and `net_revenue_interest`, returning the operator's net cost and **net NPV** alongside the
  gross figures (gross NPV implicitly assumed 100% WI/NRI, overstating the operator's value).
- **JIB partner-allocation preview** (`jib_split`) and a **price-deck sensitivity** strip
  (`price_sensitivity`) — NPV/payout across a realized-price deck.
- **Tangible vs. intangible (IDC) cost split** on every estimate (`cost_rollup`, `cost_class`
  on `LineItem`) — the tax view finance asks for; shown in the Benchmarks tab.
- **Authority-limit approval routing** (`required_approver`): the delegation-of-authority level
  an AFE's $ value requires (PE < $50k · Eng Mgr < $250k · Ops Mgr < $1MM · VP above).
- **AFE-supplement flagging**: variance flags any AFE whose actuals exceed the AFE by the policy
  threshold (>10%) as requiring a supplemental AFE.
- **Variance Analyzer wired into the app** as its own tab (it was implemented but never imported)
  — with demo actuals incl. an unbudgeted line and a >10% overrun.
- **Immutable audit trail** (`afe_events` table): every status change is appended with from→to,
  timestamp, actor, and note; surfaced in the Pipeline tab.

### Fixed
- **Discounting was 10.47% effective, not 10%** (monthly compounding of `(1+r/12)^m`). Now uses
  true effective-annual `(1+r)^(m/12)`, so "NPV @ 10%" means 10%/yr. (Both deterministic and
  Monte-Carlo paths.)
- **Sample AFE arithmetic didn't close**: the flagship `sample_afe_acid_stimulation.md` showed a
  $18,500 contingency labelled "10%" (10% of $249,100 is $24,910) and a $267,600 total. All
  figures reconciled to the current cost & economics modules ($274,010 total).
- **Variance hid 100%-unbudgeted overruns**: a `dropna()` dropped any category with no AFE
  budget (e.g. an unplanned "Fishing" cost) — the single most important overrun case. Worst
  offender is now ranked by $ overrun and unbudgeted categories are surfaced explicitly.
- **`run_drafter` crashed with a bare `KeyError`** when `ANTHROPIC_API_KEY` was unset (the live
  demo's headline button). Now raises a typed `MissingAPIKey`; the app degrades gracefully and
  the cost/economics/variance features all work without a key.
- **P&A / cost-only jobs no longer emit `$inf/bbl`**: the `compute_economics` tool returns a
  "not applicable" note so the drafter frames the spend against liability / plugging-bond.
- Tracker seed used a non-existent `gas_separator` intervention → now a valid type; `upsert`
  `ON CONFLICT` refreshes all editable fields; Monte-Carlo payout guards `months_cap <= 0`;
  "$ in pipeline" metric counts in-flight only (excludes executed/rejected).
- Version strings aligned to 0.4.0 (`pyproject.toml` was still 0.1.0, `__init__` 0.3.3).

## [0.3.3] — 2026-06-02

- Self-heal stale Streamlit bytecode cache at startup: purge `src/` `__pycache__`
  and evict cached `src` modules so newly-added functions reload from current source
  after a redeploy. Fixes the startup ImportError cascade seen after adding new
  symbols to existing modules (the app no longer needs a manual Reboot to pick them up).

## [0.3.2] — 2026-06-02

- Resilience: the optional Monte-Carlo economics import is now guarded, so if a
  build/runtime hiccup makes it unavailable the rest of the app still loads (the
  section shows a notice instead of white-screening). Root cause of the v0.3.0/0.3.1
  outage was a sticky Streamlit bytecode cache serving a pre-`simulate_economics`
  `economics.pyc`; a full app Reboot clears it.

## [0.3.1] — 2026-06-02

- Republish to force a clean Streamlit Cloud rebuild (the v0.3.0 deploy served a
  stale build that failed importing `simulate_economics`). No functional change —
  the source was already correct.

## [0.3.0] — 2026-06-02

- Monte-Carlo AFE economics (P10/P50/P90 + tornado sensitivity)
- Validated one-click chain from Production Engineer Copilot (schema validation, friendly errors)
- Contingency now computed from its stated % (cost table can't drift from the math)
- docx generation decoupled from the Anthropic SDK
- Fixed payout off-by-one; variance no longer crashes on empty input; Word tables render bold (no literal **)

## [0.2.0]

- Initial public demo.
