# Real data — Colorado ECMC (COGCC) DJ Basin monthly production

This is Deferment IQ's **default REAL data source**. `production.csv` here is **genuine
public-record production** — loaded via `src/ndic.py` (the monthly transform is
source-agnostic) when the sidebar **Data source** is **"Real — Colorado DJ Basin (ECMC)"**
(the default). The badge shows green **REAL DATA**.

## Why Colorado

NDIC (North Dakota / Bakken) per-well monthly production is **paywalled** ($100/yr "Basic
Services" subscription), so it cannot be a free committable default. **Colorado ECMC**
(formerly COGCC) publishes the **same grain — per-well, per-month oil/gas/water + producing
days — for free** as public records. The wells are **DJ Basin Niobrara/Codell horizontals
(Weld County)**: onshore unconventional, directly analogous to a Bakken/Permian horizontal
program. NDIC stays available as a bring-your-own-export option (`../ndic/README.md`).

## What it powers here — and the honest gap

The **`days` (days-produced)** field is a real downtime signal: monthly downtime is
`days_in_month − days`, so the full **potential-vs-actual** decomposition (downtime vs.
underperformance) runs on real wells. What public monthly data does **not** carry:

- **No operator reason codes** → on real data the deferment **QUANTITY is real** but
  **cause attribution is N/A** (not fabricated). The reason-code classifier / recovery
  authorization is a synthetic-only feature.
- **Monthly cadence**, no daily detail; no ESP telemetry; no labeled events.

## Schema — `production.csv`

`well_id, well_name, operator, field, formation, date (YYYY-MM), oil_bbl, gas_mcf, water_bbl, days`

(28 wells, ~2,000 well-months, 2016–2026, 17 operators.) Built from the free ECMC public
endpoints — see `../../../../production-engineer-copilot/data/real/colorado/fetch_colorado.py`
for the reproducible harvester (Colorado public records are redistributable, so the CSV is
committed). Both PE Copilot and Deferment IQ consume this identical file.
