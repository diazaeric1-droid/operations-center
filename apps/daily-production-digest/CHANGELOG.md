# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.3] — 2026-06-11

### Added
- **BYOD — upload your own fleet SCADA CSV** — a new demo data-source option runs
  the same scan + brief + ledger + data-quality path on a user's own fleet. One
  uploaded CSV (schema: `well_id`, `date`, `bopd`, `bfpd`, `gas_mcfd`,
  `intake_pressure_psi`, `motor_temp_f`, `motor_amps`, `runtime_pct`) is split by
  `well_id` and run through the existing `load_well` loader (no parallel parser).
  Columns are validated up front (`validate_scada_columns`); a missing/invalid file
  shows a clear error and stops instead of crashing. Nothing is stored server-side
  (parsed in memory only); a template CSV is downloadable.
- **Event lifecycle surfaced in the UI** — a new *Ongoing Events* tab replays the
  fleet's recent history through the persistent event state machine
  (`NEW → ONGOING → RESOLVED`, the same path `scheduler.run` / the brief writer use,
  in a per-session in-memory SQLite store) and renders the open + just-resolved
  events with running **duration** and **cumulative deferred bbl/$** — matching the
  brief. An "inject a demo outage" toggle splices a sustained multi-day rate outage
  into one healthy well (in-memory only; committed fixtures untouched) so the
  ONGOING lifecycle is visibly demonstrable on the single-day-fault demo fleet.

## [0.6.2] — 2026-06-11

### Fixed
- **Confirmed outages no longer vanish after ~3 days** — the detectors are
  stateless (they only see the recent trailing window), so a sustained outage fired
  `NEW` for a day or two and then silently disappeared once the dropped production
  level aged into the rolling baseline (the baseline absorbed the new, lower level
  and "today vs baseline" looked normal again). A real 10-day outage vanished on
  ~day 4. Added a **persistent event state machine** (`src/event_store.py`):
  `NEW → ONGOING → RESOLVED → (dropped after a short post-resolution mention)`,
  keyed by `(well_id, event_type, start_date)` in a stdlib-`sqlite3` `events` table
  (same pattern as `src.sources.SQLiteFleetSource`). For rate-loss events the
  **pre-event baseline** is captured at open time; on later days, even when
  `scan_fleet` goes quiet, if today's rate is still below the recovery band of that
  baseline the event stays **ONGOING** and keeps accruing **cumulative deferred
  bbl/$**. Recovery into band RESOLVES it; non-rate events resolve after a
  clean-poll grace period. Processing a given as-of day is **idempotent** (re-runs
  never double-count). The morning brief now shows ongoing events with their
  running **duration** + cumulative deferral (`render_brief_markdown` / `write_brief`
  gain an optional `events` arg — no events means a byte-identical brief);
  `scheduler.run` drives the machine each day. **Acknowledge/suppress** and the
  money-first deferred-$ ranking are preserved.

### Added
- **Backtest v2 — event-lifecycle metrics** (`src/backtest_v2.py`,
  `python -m src.backtest_v2`). Replays the state machine day-by-day over a fleet
  with **injected multi-day outages of known start/end** and scores **event-level
  precision/recall**, **duration accuracy** (open→resolved span vs injected span),
  **detection latency** (onset→`NEW`), and the **persistence regression** (an
  injected 10-day outage must be `ONGOING` on day 5, not gone on day 4). Keeps
  near-threshold decoys — two clean negatives the detector rejects (sub-threshold
  dip; smooth steep decliner the decline-aware rule passes) and one spurious
  positive (a metering-recal step) — so event precision is a real **0.80**, not a
  trivial 1.0. On the injected fleet: precision **0.80** (TP=4/FP=1/FN=0), recall
  **1.00**, F1 **0.89**, duration MAE **0.00 d**, mean latency **0.00 d**; the
  committed metrics snapshot is `data/backtest_v2_metrics.json`.

## [0.6.1] — 2026-06-07
### Fixed
- **Time-range toggle stuck on 30D** — `st.segmented_control` was created with both `default=` and `key=` (the default re-asserted on rerun and snapped the selection back). It now owns its selection in `session_state`, so **7D/30D/3mo/6mo/1Y/Lifetime** re-slice every chart + KPI (verified via AppTest).
### Changed
- **Light theme** — suite-wide migration from dark/navy to a professional light palette (white surfaces, `plotly_white` charts, navy/blue accents retained); transparent fixed header so the title never clips. `runtime.txt` pinned to Python 3.11.

## [0.6.0] — 2026-06-07

### Added
- **Representative-vs-anomalous data-quality classification** (`src/representative.py`,
  `classify_representative`) — flags which oil-rate points are **representative** for
  decline / type-curve trending vs which to **EXCLUDE** (shut-in / zero-rate days,
  metering dropouts, gross outliers vs a robust decline-aware trend). This is the
  pre-trending data-cleaning step — distinct from the surveillance alerting in
  `anomaly_detector` (a shut-in is a healthy well, just not on-trend data). Reuses the
  existing robust statistics (`robust_z` median/MAD + Arps `_expected_decline_rate`)
  rather than duplicating them. Deterministic, no API key.
- **"Data quality — representative vs anomalous" view** — a new overview tab with a
  fleet representative-% metric, a lowest-first per-well bar, and a sortable table
  (representative %, points, excluded, top exclusion reason). On each per-well page the
  oil-rate chart now **marks the excluded points** (distinct red ✕) with a caption of
  how many were dropped and why.

## [0.5.0] — 2026-06-06

### Added
- **Fleet explorer (multipage)** — a Fleet Overview plus a **drill-down page per well**
  (`st.navigation`), each with its own oil/gas/water + SCADA-diagnostic charts and a health note.
- **Gas channel** — every well now carries **`gas_mcfd`** (GOR-correlated to oil); ~**400 days** of
  daily history so the new **time-range toggles (7D / 30D / 3mo / 6mo / 1Y / Lifetime)** are meaningful.
- **Three fleet trend streams** (Total Oil / Gas / Water) over the selected window, plus **production
  variance** deltas (recent-7d vs window-start) in the snapshot.
- **Sortable Fleet table** — per-well BOPD, BWPD, MCFD, water-cut, GOR, **lift + lateral length**
  (from the shared fleet registry), basin·formation, production variance, days-on-prod, anomaly flag.
- Bootstrap regenerates data if the on-disk schema is stale (missing `gas_mcfd`).

## [0.4.0] — 2026-06-06

### Added
- **Unified dark + navy suite theme** and a **cross-app sidebar suite navigator** so the
  digest looks and links like one product alongside the rest of the PE suite.
- **First visualizations** (previously text-only): a **fleet oil-rate trend** and a
  **top deferred-$ offender bar** chart surface the leak at a glance.
- **Rolling lost-production ledger**: cumulative deferred **$/bbl by cause** over a trailing
  window (MTD-style), with a **deep-link to Deferment IQ** for full period loss accounting.
- **Shared fleet registry**: Permian field/formation identity is now consistent across the suite.

### Changed
- **Performance**: cached fleet load/scan, and the app **auto-selects the new brief** after
  generation so the freshest output is shown without a manual pick.

### Fixed
- Swept the deprecated `use_container_width` (→ `width="stretch"`); requires `streamlit>=1.50`.

## [0.3.0] — 2026-06-03

### Added
- **Deferred-production economics**: rate-loss anomalies carry `deferred_bopd` and
  `deferred_usd_per_day`, and the brief is **ranked by money at risk**, not z-score —
  the foreman works the biggest leak first, not the alphabetically-first well.
- **Data-quality detection** (`detect_data_quality`): a blank/zero rate while the pump
  runs is flagged as a **metering dropout**, all-tags-blank as **comms loss** — instead
  of being silently swallowed (`NaN < threshold == False`) or mistaken for a real trip.
- **Acknowledge / suppress** known events via `acknowledged.yml` so a planned workover
  doesn't re-fire HIGH every morning (alarm-fatigue control); suppressed items move to a
  "Data Quality / Acknowledged" section.
- **Water-cut context** on rate drops — a rising water cut alongside the oil drop points
  at watering out (reservoir), not a pump issue.
- **No-API-key operation**: `render_brief_markdown` produces a full deterministic brief
  when `ANTHROPIC_API_KEY` is unset, so cron/CI/the demo never crash with a bare `KeyError`.
- **Honest backtest**: near-threshold **decoy wells** (sub-threshold dip, steep-but-healthy
  decliner, noisy amps, borderline intake) so precision/recall aren't a trivial 1.00 — the
  flat-mean rate rule now visibly false-positives (precision 0.50) where decline-aware does
  not (1.00). **Lead-time** is now a real metric: detection latency from fault onset +
  early-warning days before full manifestation (the `manifest_days` parameter is actually used).
- Optional **Slack notification** step in the GitHub Action (runs only if `SLACK_WEBHOOK_URL`
  is set) — the README claim is now backed by a real step.

### Fixed
- **Decline-aware rule is now authoritative**: when a decline fit is feasible it owns the
  rate-drop call, suppressing the flat-mean rule's false positive on a steep healthy decliner;
  flat-mean survives only as a fallback for series too short to fit. It also fits the trend on
  history **excluding today** (extrapolating one step) so a one-day step-down can't flatten its
  own baseline.
- **Motor-temp MEDIUM** now requires statistical significance (robust-z ≥ 3) *and* the +15°F
  rise, so a noisy well's single warm day no longer trips a flag (robust-z was decorative).
- **GitHub Action time was wrong half the year**: the comment claimed 6:30am Central for
  `30 12 UTC`, true only in winter (CST); `30 11 UTC` is 6:30am during CDT. Documented the
  fixed-UTC/no-DST behavior.
- **SQLite adapter truncated timestamps to date-only** (`%Y-%m-%d`), collapsing sub-daily
  historian readings to one key — now stores full ISO datetime; table identifier is validated.
- `robust_z` dropped its confusing dead `x=` override parameter and now ignores NaNs.
- `write_brief` raises a typed `MissingAPIKey`; version strings aligned to 0.3.0.

## [0.2.1] — 2026-06-02

- Self-heal stale Streamlit bytecode cache at startup: purge `src/` `__pycache__`
  and evict cached `src` modules so newly-added functions reload from current source
  after a redeploy. Fixes the startup ImportError cascade seen after adding new
  symbols to existing modules (the app no longer needs a manual Reboot to pick them up).

## [0.2.0] — 2026-06-02

### Added
- **Robust anomaly detection** — per-well rolling median + MAD robust z-scores
  (`robust_z = 0.6745·(x − median)/MAD`) so each flag can report "N sigma off
  this well's own baseline" instead of a fleet-wide rule of thumb. MAD==0 is
  guarded (no div-by-zero on a flat baseline).
- **Decline-aware rate-drop flagging** — fits an exponential (Arps) decline via
  numpy log-linear regression and flags drops relative to the decline-EXPECTED
  rate today, not a flat 7-day mean, so a healthy steep decliner stops
  over-flagging. Added as a refinement alongside the original rule.
- **Least-squares trend slopes** — amps-creep and intake-collapse now use a
  least-squares slope over the window instead of a noisy 2-point first/last
  estimate, which recovers an amps-creep well the endpoint estimator missed.
- **Pluggable historian adapter protocol** (`src/sources.py`) — a `FleetSource`
  `typing.Protocol` plus a refactored CSV adapter and two more adapters:
  `CsvTimeRangeFleetSource` (date-range filtered) and a stdlib-only
  `SQLiteFleetSource`. All honor the `SCADA_COLUMNS` contract.
- **Backtest harness** (`src/backtest.py`, `python -m src.backtest`) — scores
  every detector against the generator's seeded anomalies as ground truth and
  reports precision / recall / lead-time per rule, with an optional threshold
  sweep.

### Changed
- Empty/short-frame guards in `fleet_summary` and the detectors.
- `brief_writer` honors the `MODEL` environment variable documented in
  `.env.example`.

## [0.1.0] — Initial public demo

- Deterministic anomaly detector (rate drop, intake collapse, motor temp spike,
  runtime degradation, amps creep), Claude-powered Senior-PE brief writer,
  Streamlit history viewer, and a GitHub Actions workflow for daily cloud runs.
