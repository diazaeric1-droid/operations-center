# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/);
this project follows [Semantic Versioning](https://semver.org/).

## [0.5.1] — 2026-06-11

### Fixed
- **Cadence-aware deferment engine (correctness)** — the engine now works in **calendar-day volume** terms instead of fixed row-count windows. Each record carries an explicit calendar span (`span_days`) and producing-time (`producing_days`); potential is a decline-aware P75 over a **time-based** (calendar-day) window, and deferment = `potential_calendar_volume − actual`, split into **downtime** (`days_in_month − days-produced`) vs. **underperformance** (rate shortfall while up), with the 8% deadband preserved. The previous row-count logic treated a monthly record like a daily one, badly mis-counting potential and deferment on real **monthly** Colorado ECMC / NDIC data (e.g. a month with 21 lost producing days read **0** deferred barrels; fleet KPIs showed `pct_deferred` > 100% with deferred > potential). Now monthly `actual_bbl` equals the raw `oil_bbl` exactly and the waterfall/Pareto/KPIs are coherent. Daily synthetic results are unchanged. Reason-code classifier, waterfall, Pareto, MTTR, and public signatures preserved (`compute_deferment` adds `*_vol` / `span_days` columns).

### Added
- **Quantity-recovery eval (Phase 2)** — a ground-truth fleet with **known injected** downtime + underperformance per well validates the engine's barrel accounting: error on total deferred bbl, on the downtime/underperformance split, and on recovery opportunity vs. the true recoverable. Runs on **both daily and monthly** cadence of the identical fleet to prove cadence-awareness (daily ~exact; monthly downtime exact and matching daily to the bit; monthly under-counts the smeared sub-month underperformance — documented). **CI-gated** (`evals/quantity_recovery.py`, `tests/test_quantity_recovery.py`) — the build fails if the deferred-bbl error exceeds a sane, cadence-appropriate bound.

## [0.5.0] — 2026-06-07
### Added
- **Real public data is now the DEFAULT** — Colorado ECMC **DJ Basin** per-well monthly production (committed 28-well slice). The real loader is now CSV-path driven (Colorado default; NDIC as a bring-your-own-export path). Honest provenance: deferment **quantity** is real, cause attribution N/A (no public reason codes).
### Changed
- **Light theme** — suite-wide migration from dark/navy to a professional light palette (white surfaces, `plotly_white` charts, navy/blue accents retained); transparent fixed header so the title never clips. `runtime.txt` pinned to Python 3.11.

## [0.4.0] — 2026-06-06

### Added
- **Real-data option (North Dakota / NDIC)** — a "Data source" toggle (Synthetic default | Real —
  NDIC) + adapter (`src/ndic.py`) ingesting tidy per-well **monthly** Bakken filings; downtime comes
  from **days-produced**, and cause attribution shows **N/A on real data** (no public reason codes).
  Drops in at `data/real/ndic/production.csv` (README + template; the CSV is gitignored).
- **Data-provenance badge** under the header (green "REAL DATA — NDIC/Bakken" vs amber "SYNTHETIC").

## [0.3.0] — 2026-06-06

### Added
- **Fleet explorer (multipage)** — the base-management review, recovery work-queue, and
  classifier eval now live on a Fleet Overview alongside a **sortable fleet table** (lift,
  lateral, basin·formation, deferred bbl/$, dominant cause, uptime %, recoverable $, capture %),
  plus a **drill-down page per well** (`st.navigation`) with its potential-vs-actual chart,
  events, KPIs, and recovery items.

## [0.2.0] — 2026-06-06

### Added
- **Unified suite theme** — dark + navy styling shared across the suite, plus a cross-app sidebar
  **suite navigator** to jump between the apps.
- **Prioritized recovery work-queue** — actionable (well × recoverable cause) items ranked by
  **recoverable $ ÷ MTTR**, each with a suggested action and a **deep-link to AFE Copilot**.
- **MTTR-by-cause bar chart**.
- **Shared fleet registry** — Permian field/formation identity is now consistent across the suite.

### Changed
- **Robustness:** empty-frame guard in `recovery_opportunity`; swept the deprecated
  `use_container_width` (→ `width="stretch"`); requires `streamlit>=1.50`.

## [0.1.0] — 2026-06-04

Initial release — base-management / lost-oil accounting.

### Added
- **Potential (entitlement) model** (`src/potential.py`): per-day capability from full-uptime
  days only (P75, decline-aware, rolling), so a healthy well reads ~0 deferred.
- **Deferment engine** (`src/deferment.py`): splits the potential-vs-actual gap into
  **downtime** vs. **underperformance**, with an 8%-of-potential deadband to ignore measurement
  noise; attributes each lost barrel to the cause of the covering event.
- **Reason-code classifier** (`src/reason_codes.py`): canonical 8-cause taxonomy + a deterministic
  keyword classifier over operator free-text notes, with an optional LLM classifier (BYOK) for the
  long tail that always falls back to the rules. `recoverable`/`planned` flags drive the opportunity.
- **Analytics** (`src/analytics.py`): KPIs (production efficiency, deferred $, capture rate),
  deferment waterfall, Pareto of $ by cause, worst-offender wells, MTTR by cause, recovery
  opportunity, and weekly trend.
- **Narrated review** (`src/narrator.py`): Senior-PE / VP base-management review — LLM-narrated
  (BYOK) with a deterministic templated fallback, so it runs with no key.
- **Eval harness** (`evals/run_evals.py`): scores the rules classifier vs. ground-truth causes
  (accuracy + per-class precision/recall/F1 + confusion); **CI gate fails under 80%**. Current
  accuracy ~92% on the synthetic event log.
- **Streamlit app** + Docker/HF deploy config + bring-your-own-key.
- Synthetic 40-well × 90-day fleet with injected downtime/curtailment, realistic operator notes
  (incl. deliberately vague ones), and a couple of *uncaptured* wells so capture rate is < 100%.
