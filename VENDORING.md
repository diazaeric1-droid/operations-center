# Vendoring Record

The four component apps are vendored as plain directories under `apps/`
(mirrored from their own repos — no submodules), then loaded under import
aliases by `core.py` (`digest`, `deferment`, `esp`, `afe`). Copies exclude
`.git`, `.venv`, `__pycache__`, `.pytest_cache`, `*.egg-info`, `htmlcov`,
`.ruff_cache`.

## Verification (against the component repos at the recorded versions)

Method: `diff -r <component> apps/<component>` with the exclusions above, plus a
per-tracked-file `cmp` sweep over each component's `git ls-files`.

| Component | Version | Tracked files | Byte-identical | Transformed |
|---|---|---|---|---|
| daily-production-digest | 0.6.3 | 42 | 41 | **1** (see below) |
| deferment-iq | 0.5.1 | 40 | 40 | 0 |
| esp-failure-risk-agent | 0.7.3 | 40 | 40 | 0 |
| afe-copilot | 0.6.2 | 35 | 35 | 0 |

Remaining `diff -r` noise is exclusively the components' **gitignored,
regenerated** data (digest `data/synthetic/fleet/`, deferment
`data/synthetic/wells|events.csv`, ESP `data/synthetic/*.csv` +
`artifacts/`) — `core.bootstrap()` recreates all of it on first run.

## Transformed files

### `apps/daily-production-digest/src/ledger.py` (line 30)

```diff
-from src.anomaly_detector import DEFAULT_OIL_PRICE, scan_fleet
+from .anomaly_detector import DEFAULT_OIL_PRICE, scan_fleet
```

**Why:** the alias loader imports the package as `digest`, so the absolute
`src.` import cannot resolve (there is no top-level `src` in this process; the
name would collide across four components). Mechanical rewrite to the
package-relative form every other digest module already uses.

**Behavior proof:** `tests/test_core.py::
test_ledger_rewrite_behavior_matches_original_component` runs the ORIGINAL
component from its own repo (where `src.*` resolves) on the same bootstrapped
fleet in a subprocess and asserts the vendored module reproduces its ledger
numbers exactly (`period_deferred_usd`, `days_scanned`, `top_cause`, row
count).

### deferment-iq import audit

`grep -rn '^from src\|^import src' deferment-iq/src/` → **no matches**; the
component's `src/` package is already fully package-relative, so it vendors
byte-identical. (Its `demo/app.py` does use `from src.x import y`, but demo
apps are not loaded by the alias loader — Operations Center re-implements those
pages in `views/`.)

## Not vendored

**pe-pipeline is absorbed, not vendored:** `core.py` is an adaptation of its
`pipeline_core.py` (same alias loader, same bootstrap, same `rank_fleet`
math/thresholds, plus the `deferment` alias and the event-replay helpers).
`tests/test_core.py::test_rank_fleet_identical_to_pipeline_core` pins the
absorbed ranking to the original orchestrator bit-for-bit by running
pe-pipeline's `pipeline_core` against this repo's `apps/` (via `PE_APPS_ROOT`)
in a subprocess and asserting frame equality.

`theme.py`, `fleet_registry.py` (from well-gas-lift-advisor/demo) and
`product_theme.py` (from `_shared/`) are byte-identical copies at the repo
root, per the suite's vendoring convention.
