# Real data — North Dakota (NDIC) monthly Bakken production

> **Note:** The suite now defaults to **free Colorado ECMC** real data (`../colorado/`). NDIC per-well monthly production is a **paid subscription** ($100/yr Basic Services), so this remains a *bring-your-own-export* path — drop your own `production.csv` here to run on real North Dakota Bakken wells.

Deferment IQ defaults to a **synthetic** reason-coded fleet (so the classifier eval
has ground truth). It also runs on **real public data**: per-well **monthly**
production from the **North Dakota Industrial Commission (NDIC) / Department of
Mineral Resources** — the Bakken/Three Forks play, onshore.

Pick **"Real — North Dakota (NDIC)"** in the app sidebar. If
`data/real/ndic/production.csv` exists it loads via the adapter (`src/ndic.py`);
otherwise the app shows a warning and falls back to synthetic. **No API key, no
network call** — the app only reads a local file you drop in here.

## What's real vs. N/A on this data

| Signal | Real on NDIC data? |
| --- | --- |
| Oil / gas / water volumes | ✅ from the public filing |
| **Days produced** → uptime / **downtime** | ✅ `downtime = days_in_month − days` is the real downtime signal |
| Potential vs. actual, deferment quantity | ✅ computed (same engine as synthetic) |
| **Reason codes / cause attribution** | ❌ **N/A** — public filings carry *no* operator cause notes |

So the **deferment QUANTITY is real**; the **cause is uncoded/unknown**. On real
data the Recovery queue, reason-code Pareto, and Classifier-eval sections show
"cause attribution N/A — no public reason codes" rather than inventing a cause.

## File the app reads

```
data/real/ndic/production.csv
```

A tidy, **one-row-per-well-month** CSV with this header (see `_TEMPLATE.csv` for a
clearly-fake 2-row example — `well_id` `DEMO_0001`, placeholder values, NOT real
Bakken numbers):

```
well_id,well_name,operator,field,formation,date,oil_bbl,gas_mcf,water_bbl,days
```

| Column | Meaning |
| --- | --- |
| `well_id` | stable id (NDIC file number or API-10/14) |
| `well_name` | well name / label |
| `operator` | operator of record |
| `field` | NDIC field name |
| `formation` | producing pool (e.g. `Bakken`, `Three Forks`) |
| `date` | production **month**, `YYYY-MM` (a full `YYYY-MM-DD` is also accepted) |
| `oil_bbl` | oil produced that month (bbl) |
| `gas_mcf` | gas produced that month (mcf) |
| `water_bbl` | water produced that month (bbl) |
| `days` | **days produced** in the month (the downtime input) |

The adapter maps each well-month into the engine's daily schema at a monthly
cadence: `bopd = oil_bbl / max(days,1)`, `bfpd = (oil_bbl + water_bbl) / max(days,1)`,
`gas_mcfd = gas_mcf / max(days,1)`, and `runtime_pct = days / days_in_month * 100`.
Rows with non-positive `days` or an unparseable month are dropped defensively.

## How to produce `production.csv` from an NDIC export

The data is public; no key required.

1. **Source.** NDIC Department of Mineral Resources, Oil & Gas Division —
   <https://www.dmr.nd.gov/oilgas/>. Production is available through the
   monthly production reports / the "Producing Wells" and well-search downloads.
   (Some bulk/historical downloads sit behind the inexpensive public
   subscription; the current monthly statistical reports are free.)
2. **Pull** the per-well monthly oil/gas/water and **days-produced** columns for
   the wells (or field/operator) you want, for the months you want.
3. **Reshape to tidy** — one row per (well, month). Rename the columns to the
   header above (`oil_bbl, gas_mcf, water_bbl, days`, month → `date` as `YYYY-MM`).
   Add `well_name / operator / field / formation` from the well header.
4. **Save** the result as `data/real/ndic/production.csv` (this folder).
5. Launch the app and choose **"Real — North Dakota (NDIC)"** in the sidebar.

`production.csv` is git-ignored by default (you bring your own extract); the
`_TEMPLATE.csv` and this README are the only files tracked here.
