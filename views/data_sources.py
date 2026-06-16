"""Data · Sources & BYOD — provenance for both datasets and one upload point.

The console runs on TWO datasets at different cadences. They are not joined:

* Surveillance fleet (Today + Well File) — synthetic DAILY SCADA (100 wells),
  known ground truth. Public production data is monthly, so daily SCADA must be
  modeled; the digest's detectors are backtested against seeded faults + decoys.
  (The well count is rendered dynamically in-page from the bootstrapped fleet.)
* Loss-accounting book (Loss Accounting) — SYNTHETIC reason-coded monthly fleet by
  default (40 wells, ground-truth causes), so cause attribution / MTTR / recovery
  queue all run. Bring your own monthly book (or a real public extract) below.

Uploads are session-only: parsed in memory, never written to disk or logged.
"""
from __future__ import annotations

import hashlib
import io

import pandas as pd
import streamlit as st

import product_theme as pt
import theme

from views import _common as c


def render() -> None:
    c.ensure_state()
    price, _nri, _disc = c.deck()

    pt.masthead("ops", "Sources & BYOD",
                "What every page is computed from — and where to drop your own "
                "data. Nothing is stored server-side.")
    pt.context_bar([
        ("Surveillance fleet", c.scada_source_label()),
        ("Loss accounting", c.loss_context(st.session_state["data_source"])),
        ("Deck", c.deck_label()),
    ])

    pt.section("Two Datasets, No Fake Join")
    n_scada = len(c.scada_well_ids())  # rendered from the bootstrapped fleet, never stale
    st.markdown(
        "| Console area | Dataset | Cadence | Provenance |\n"
        "|---|---|---|---|\n"
        "| Today + Well File | Surveillance fleet (SCADA) | Daily | **Synthetic** "
        f"modeled Permian fleet, {n_scada} wells, known ground truth |\n"
        "| Loss Accounting | Production book | Monthly | **Synthetic** reason-coded "
        "fleet, 40 wells, ground-truth causes — bring your own monthly book below |\n")
    st.caption(
        "Both lenses run on modeled fleets with known ground truth — the honest demo "
        "posture: daily SCADA with ESP diagnostics and a reason-coded downtime log "
        "don't exist together in any public dataset. They are different datasets at "
        "different cadences; this console keeps them side by side and does **not** "
        "fabricate a join. Bring your own daily SCADA or monthly production book "
        "below to run the same engines on your wells.")

    st.divider()

    # ---- Upload 1: fleet SCADA (digest schema) --------------------------------
    pt.section("Upload — Fleet SCADA (daily)",
               "Runs the SAME deterministic scan, brief, and event lifecycle on "
               "your wells (Morning Brief + Ongoing Events pages).")
    import core
    digest_loader = core.digest_loader
    st.caption(
        "**Required columns** (one row per well per day): "
        f"`{'`, `'.join(digest_loader.BYOD_REQUIRED_COLUMNS)}`. "
        "`date` is YYYY-MM-DD; `well_id` groups rows into wells; rates are daily "
        "(`bopd` oil, `bfpd` gross fluid, `gas_mcfd` gas) plus ESP diagnostics. "
        "Extra columns are ignored. **Nothing is stored server-side** — the file "
        "is parsed in memory for this session only and never written to disk or "
        "logged.")
    st.download_button("Download SCADA template CSV",
                       data=digest_loader.fleet_template_csv(),
                       file_name="fleet_scada_template.csv", mime="text/csv")
    up = st.file_uploader("Fleet SCADA CSV", type=["csv"], key="scada_uploader")
    if up is not None:
        _ingest_scada(up)
    if st.session_state.get("scada_upload"):
        n_kb = len(st.session_state["scada_upload"]) / 1024
        st.success(f"Active upload: **{st.session_state['scada_upload_name']}** "
                   f"({n_kb:,.0f} kB) — Morning Brief and Ongoing Events now run on "
                   "your fleet.")
        st.caption("Triage Board, Well 360, and Action Chain stay on the bootstrapped "
                   "synthetic fleet: the ESP risk model and the AFE chain read "
                   "per-well CSVs + a trained artifact tied to that fleet. The scan "
                   "and event lifecycle are schema-driven, so they run on yours.")
        if st.button("Clear uploaded fleet"):
            st.session_state["scada_upload"] = None
            st.session_state["scada_upload_name"] = ""
            st.session_state["scada_source"] = "disk"
            st.rerun()

    st.divider()

    # ---- Upload 2: monthly production (deferment schema) -----------------------
    pt.section("Upload — Monthly Production (loss accounting)",
               "Runs the deferment engine (potential vs actual, downtime split) on "
               "your monthly book — the Loss Accounting pages.")
    st.caption(
        "**Tidy monthly schema** (one row per well per month): "
        f"`{'`, `'.join(core.deferment_ndic.NDIC_COLUMNS)}`. "
        "`date` is YYYY-MM; rate = oil_bbl ÷ days-produced; downtime = days in "
        "month − days produced. A public monthly book carries no reason codes, so "
        "cause attribution, MTTR, and the recovery queue read N/A — the deferment "
        "**quantity** is yours and real. (The synthetic source keeps those views "
        "live because it ships ground-truth causes.) **Nothing is stored "
        "server-side.**")
    st.download_button("Download monthly template CSV",
                       data=core.monthly_template_csv(),
                       file_name="monthly_production_template.csv", mime="text/csv")
    upm = st.file_uploader("Monthly production CSV", type=["csv"],
                           key="deferment_uploader")
    if upm is not None:
        _ingest_monthly(upm)
    if st.session_state.get("deferment_upload"):
        st.success(f"Active upload: **{st.session_state['deferment_upload_name']}** — "
                   "select \"Your upload\" as the source on any Loss Accounting page.")
        if st.button("Clear uploaded monthly book"):
            st.session_state["deferment_upload"] = None
            st.session_state["deferment_upload_name"] = ""
            if st.session_state.get("data_source") == core.DEF_SRC_UPLOAD:
                st.session_state["data_source"] = core.DEF_SRC_SYNTHETIC
            st.rerun()

    st.divider()
    pt.section("Provenance Notes")
    st.markdown(
        "- **Synthetic SCADA fleet** (Today + Well File) — regenerated "
        "deterministically on first run (seeded per well); failure signatures across "
        "seven modes (gas interference/lock, scale, downthrust, electrical, shut-in, "
        "rate-loss) plus near-threshold decoys give the detectors an honest backtest "
        "(not a trivial 1.0). Each signature is seeded only on wells whose "
        "artificial-lift type it can physically occur on.\n"
        "- **Synthetic reason-coded fleet** (Loss Accounting) — modeled monthly book "
        "with a ground-truth cause on every event, so the $-Pareto, MTTR, recovery "
        "queue, and the classifier eval all run; the classifier is scored against "
        "those held-out labels (CI gate fails under 80%).\n"
        "- **ESP risk model** — XGBoost + Platt calibration, trained at bootstrap on "
        "the digest/surveillance fleet itself (the fleet the console scores) using its "
        "ground-truth fault labels; the model artifact is regenerated, never committed. "
        "Its 30-day score is a Platt-calibrated probability (calibrated out-of-fold AUROC "
        "≈0.98 on clean synthetic signatures — an upper bound on clean data, not a "
        "real-world claim). Full model card on **Methods & Limitations**.\n"
        "- **Bring your own data** — drop a daily SCADA CSV or a monthly production "
        "book above; both are parsed in memory for the session only. A real public "
        "monthly extract (e.g. Colorado ECMC / your operator export) works as the "
        "monthly upload — the deferment quantity is real, cause attribution N/A.\n"
        "- **Anthropic key (sidebar)** — optional, session-only, never stored; "
        "powers narration only. Every number on every page is deterministic.")
    theme.references(["arps", "deferment"])


def _ingest_scada(up) -> None:
    """Validate + parse a fleet SCADA upload via the digest's own loader.
    Bad columns → st.error + st.stop (clear error beats a parse traceback)."""
    import core
    data = up.getvalue()
    try:
        head = pd.read_csv(io.BytesIO(data), nrows=0)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read that file as CSV: {exc}")
        st.stop()
    missing = core.digest_loader.validate_scada_columns(head)
    if missing:
        st.error(
            "Uploaded CSV is missing required column(s): "
            f"**{', '.join(missing)}**.\n\nRequired columns are: "
            f"`{'`, `'.join(core.digest_loader.BYOD_REQUIRED_COLUMNS)}`. "
            "Download the template above for the exact schema.")
        st.stop()
    try:
        fleet = core.load_scada_fleet_from_bytes(data)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the uploaded fleet: {exc}")
        st.stop()
    if not fleet:
        st.error("No wells found in the uploaded CSV — check the `well_id` column.")
        st.stop()
    n_rows = sum(len(df) for df in fleet.values())
    st.session_state["scada_upload"] = data
    st.session_state["scada_upload_name"] = up.name
    st.session_state["scada_source"] = "upload"
    st.caption(f"Parsed {len(fleet)} well(s), {n_rows:,} daily rows "
               f"(token {hashlib.sha1(data).hexdigest()[:8]}, session-only).")


def _ingest_monthly(upm) -> None:
    """Validate + parse a monthly production upload via the deferment loader."""
    import os
    import tempfile

    import core
    data = upm.getvalue()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    try:
        tmp.write(data)
        tmp.close()
        fleet = core.deferment_ndic.load_ndic_fleet(tmp.name)
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"Could not parse the monthly CSV: {exc}\n\n"
            "Required columns: "
            f"`{'`, `'.join(core.deferment_ndic.NDIC_COLUMNS)}` — download the "
            "template above for the exact schema.")
        st.stop()
        return
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if not fleet:
        st.error("No wells found in the uploaded CSV — check the `well_id` column.")
        st.stop()
    st.session_state["deferment_upload"] = data
    st.session_state["deferment_upload_name"] = upm.name
    st.caption(f"Parsed {len(fleet)} well(s) of monthly records (session-only).")
